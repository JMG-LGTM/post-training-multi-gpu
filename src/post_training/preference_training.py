"""Critic-only IDQL experiment with same-state preference supervision.

The behavior actor is identified in provenance, never imported or instantiated.
Native sparse-reward IQL updates retain their V -> Q -> EMA order. Ranking is
an auxiliary term in that same Q optimizer step, not an extra parameter update.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import signal
import time
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.distributed as dist
import yaml

from . import training as base
from .checkpoint import load_checkpoint, save_checkpoint, rng_state, restore_rng
from .critic import (PixelIQL, augment_batch, default_input_spec, export_critic,
                     load_critic, random_shift_pixels, read_critic_metadata)
from .data import inspect_dataset
from .diagnostics import evaluate_critic
from .distributed import (GlobalBatchSampler, OptimizerGroup, active, all_rank_call,
                          initialize, primary_call, rank, reduce_metrics, unwrap,
                          world_size, wrap)
from .io import atomic_json, sha256_file, stable_hash
from .preference_config import preference_batch_plan, validate_preference_config
from .replay import CachedReplayDataset, build_replay_cache, load_normalization_file


def core_config(config):
    value = copy.deepcopy(config)
    value.pop("preference")
    return value


def frozen_members(config, info):
    """Require the same full native dataset and original root membership roster."""
    actual = {}
    for row in info["episodes"]:
        root, split = row["root_run_id"], row["split"]
        if root in actual and actual[root] != split:
            raise ValueError("Native source root crosses training and validation")
        actual[root] = split
    if config["preference"]["frozen_split_members"] is None:
        return actual, {"source": "complete_native_dataset_split", "members": actual,
                        "dataset_sha256": info["dataset_sha256"], "sha256": stable_hash(actual)}
    path = Path(config["preference"]["frozen_split_members"]).expanduser().resolve()
    snapshot = json.loads(path.read_text())
    if snapshot.get("dataset_sha256") != info["dataset_sha256"]:
        raise ValueError("Frozen native dataset identity differs")
    mapping = {}
    for row in snapshot.get("members", []):
        root, split = row["root_run_id"], row["split"]
        split = "val" if split == "validation" else split
        if split not in {"train", "val"} or root in mapping and mapping[root] != split:
            raise ValueError("Conflicting frozen root membership")
        mapping[root] = split
    if not mapping or mapping != actual:
        raise ValueError("Native root members differ from the frozen train/validation roster")
    return mapping, {"path": str(path), "sha256": sha256_file(path), "members": mapping}


def inspect_preferences(config, info):
    membership, roster = frozen_members(config, info)
    if config["preference"]["weight"] == 0:
        return {"enabled": False, "frozen_membership": roster, "package": None,
                "dataset_sha256": None, "train_count": 0, "val_count": 0}
    from .preference_data import inspect_preference_package
    result = inspect_preference_package(config["preference"]["package"],
        validation_fraction=config["data"]["validation_fraction"], seed=config["data"]["seed"],
        frozen_membership=membership, split_strategy="native_membership")
    for row in result["members"]:
        split = "val" if row["split"] == "validation" else row["split"]
        if membership.get(row["root_run_id"]) != split:
            raise ValueError("Preference pair is outside the original native root split")
    counts = {name: sum(row["source"] in config["preference"]["sources"] and
                       ("val" if row["split"] == "validation" else row["split"]) == name
                       for row in result["members"]) for name in ("train", "val")}
    if not counts["train"] or not counts["val"]:
        raise ValueError("Selected preference sources require nonempty original train and held-out roots")
    return {"enabled": True, "frozen_membership": roster,
            "package": str(Path(config["preference"]["package"]).resolve()),
            "dataset_sha256": result["dataset_sha256"],
            "train_count": counts["train"], "val_count": counts["val"],
            "sources": config["preference"]["sources"], "pair_metadata": result}


def dry_run(config, data_root, world):
    validate_preference_config(config)
    plan = preference_batch_plan(config, world)
    report = base.dry_run(core_config(config), data_root, world)
    info = inspect_dataset(data_root, core_config(config), verify=False)
    preference = inspect_preferences(config, info)
    return {**report, **plan, "algorithm": "preference_regularized_idql_critic",
            "preference": preference, "preference_settings": config["preference"],
            "behavior_policy_loaded": False, "policy_optimizer_updates": 0,
            "frozen_behavior_binding": {key: config["preference"][key] for key in
                ("behavior_policy_id", "behavior_actor_sha256")}}


def preference_tensor_batch(batch, normalization, image_size, device, *, augment_pad=0):
    """Use the native critic's train-only normalization, never BC actor stats.

    Exactly one oriented observation and one random image shift condition both
    chosen and rejected chunks. Padded motor coordinates stay exactly zero.
    """
    from .preference_data import critic_preference_batch
    result = critic_preference_batch(batch, normalization, image_size=image_size, device=device)
    if augment_pad:
        result["pixels"] = random_shift_pixels(result["pixels"], augment_pad)
    return result


def optimization_step(config, batches, critic, optimizers, device, *, preference_batches=None,
                      normalization=None):
    """One Q/V optimizer update, with an exact original implementation control."""
    pref = config["preference"]
    if pref["weight"] == 0:
        if preference_batches is not None:
            raise ValueError("Weight-zero control must not draw preference batches")
        return base.optimization_step(core_config(config), batches, None, critic, optimizers,
                                      device, actor_update=None)
    if not batches or preference_batches is None or len(preference_batches) != len(batches):
        raise ValueError("Each native microbatch requires one preference microbatch")
    accumulation = len(batches)
    augmented = [augment_batch(base._tensor_batch(batch, device), config["critic"]["random_shift_pad"])
                 for batch in batches]
    # Draw augmentation only after the exact original native augmentation draws.
    pairs = [preference_tensor_batch(batch, normalization, config["critic"]["image_size"], device,
                                    augment_pad=config["critic"]["random_shift_pad"])
             for batch in preference_batches]
    metrics = {}
    for stage, key, modules in (("value", "v_loss", ["value"]), ("q", "q_loss", ["q1", "q2"])):
        optimizer = optimizers[stage]
        optimizer.zero_grad()
        measured = {}
        for index, batch in enumerate(augmented):
            context = critic.no_sync() if hasattr(critic, "no_sync") and index + 1 < accumulation else nullcontext()
            with context:
                extra = ({"preference_batch": pairs[index], "preference_weight": pref["weight"],
                          "preference_temperature": pref["temperature"]} if stage == "q" else {})
                result = critic(batch, stage=stage, **extra)
                loss = result[key].mean()
                base._finite_loss(loss, device)
                (loss / accumulation).backward()
            for name in (key, "q_td_loss", "pref_loss", "pref_accuracy", "pref_margin"):
                if name in result:
                    measured.setdefault(name, []).append(float(result[name].detach().mean()))
        parameters = [p for name in modules for p in getattr(unwrap(critic), name).parameters()]
        metrics[f"{stage}_grad_norm"] = base._finite_step(optimizer, parameters,
                                                         config["critic"]["grad_clip"], device)
        metrics.update({name: sum(values) / len(values) for name, values in measured.items()})
    unwrap(critic).update_targets()
    from .preference_critic import preference_ranking
    with torch.no_grad():
        targets = [preference_ranking(unwrap(critic), pair, temperature=pref["temperature"], target=True)
                   for pair in pairs]
    for name in ("pref_loss", "pref_accuracy", "pref_margin", "pref_min_accuracy", "pref_min_margin"):
        metrics["target_" + name] = sum(float(result[name].mean()) for result in targets) / accumulation
    return metrics


@torch.no_grad()
def evaluate_preferences(model, loader, normalization, config, device):
    """Complete held-out pair evaluation; no augmentation, update or RNG drift."""
    if loader is None:
        return {"enabled": False, "samples": 0}
    previous = rng_state(device)
    modes = {module: module.training for module in model.modules()}
    generator = getattr(loader, "generator", None)
    loader_rng = generator.get_state() if generator is not None else None
    sums, count = {}, 0
    try:
        model.eval()
        for raw in loader:
            pair = preference_tensor_batch(raw, normalization, config["critic"]["image_size"], device)
            # PreferenceIQL exposes the same pairwise helper used by its forward.
            from .preference_critic import preference_ranking
            result = preference_ranking(model, pair, temperature=config["preference"]["temperature"])
            size = len(pair["action_mask"])
            target = preference_ranking(model, pair, temperature=config["preference"]["temperature"], target=True)
            result.update({"target_" + key: value for key, value in target.items()})
            for key in ("pref_loss", "pref_accuracy", "pref_margin", "pref_min_accuracy", "pref_min_margin",
                        "target_pref_loss", "target_pref_accuracy", "target_pref_margin",
                        "target_pref_min_accuracy", "target_pref_min_margin"):
                value = result[key]
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("Nonfinite held-out preference metric")
                sums[key] = sums.get(key, 0.) + float(value.double().sum())
            count += size
        if not count:
            raise ValueError("Held-out preference split is empty")
        return {"enabled": True, "samples": count, "split": "val", "held_out": True,
                **{key: value / count for key, value in sums.items()}}
    finally:
        for module, mode in modes.items():
            module.training = mode
        restore_rng(previous, device)
        if loader_rng is not None:
            generator.set_state(loader_rng)


def _source_identity():
    names = ["preference_training.py", "preference_config.py", "preference_critic.py",
             "critic.py", "training.py", "checkpoint.py", "distributed.py", "replay.py", "data.py",
             "archive_io.py", "diagnostics.py", "config.py", "continuation.py", "io.py"]
    return {name: sha256_file(Path(__file__).with_name(name)) for name in names}


@contextmanager
def _run_outputs(directory, config, start, resumed):
    """Own output lifecycle after setup; optional TensorBoard is rank-zero only."""
    writer = None
    def initialize_writer():
        nonlocal writer
        atomic_json(directory / "status.json", {"status": "RUNNING", "resumed_update": start,
                    "policy_optimizer_updates": 0, "behavior_policy_loaded": False})
        if config["training"]["tensorboard"]:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(directory / "tensorboard"), purge_step=start + 1 if resumed else None)
    primary_call(initialize_writer)
    try:
        yield writer
    except Exception as error:
        if rank() == 0:
            atomic_json(directory / "status.json", {"status": "FAILED", "error_type": type(error).__name__,
                        "error": str(error), "policy_optimizer_updates": 0})
        raise
    finally:
        if writer is not None:
            writer.close()


def train(config, data_root, *, resume=None):
    validate_preference_config(config)
    device = initialize(config)
    stopped = {"requested": False}
    def request_stop(*_):
        stopped["requested"] = True
    handlers = {kind: signal.signal(kind, request_stop) for kind in (signal.SIGINT, signal.SIGTERM)}
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        return _train(config, Path(data_root), device, resume, lambda: stopped["requested"])
    finally:
        for kind, handler in handlers.items():
            signal.signal(kind, handler)
        torch.use_deterministic_algorithms(deterministic)
        if active():
            dist.destroy_process_group()


def _train(config, data_root, device, resume, stopping):
    core = core_config(config)
    plan = preference_batch_plan(config, world_size())
    cfg, pref = config["training"], config["preference"]
    directory = Path(config["output_dir"]).expanduser().resolve()
    def prepare():
        if not resume and directory.exists() and any(directory.iterdir()):
            raise FileExistsError("output_dir is not empty; use a fresh directory or --resume")
        if resume and Path(resume).expanduser().absolute().parent != directory / "checkpoints":
            raise ValueError("Resume must use this run's own checkpoints directory")
        info = inspect_dataset(data_root, core, verify=True)
        preference = inspect_preferences(config, info)
        normalization, initialization = None, None
        if config["critic"]["init_checkpoint"]:
            metadata = read_critic_metadata(config["critic"]["init_checkpoint"],
                expected_model_config=base._critic_model_config(config),
                expected_input_spec=default_input_spec(base._critic_model_config(config), config["critic"]["image_size"]),
                expected_reward_definition=config["reward"],
                expected_termination_rule=config["reward"]["termination_rule"])
            normalization = metadata["normalization"]
            initialization = base._critic_initialization(config, metadata)
        if config["data"]["normalization"] and normalization is not None:
            if load_normalization_file(config["data"]["normalization"]) != normalization:
                raise ValueError("Explicit normalization differs from initialization")
        cache, reused = build_replay_cache(data_root, config["cache_dir"], core, norm_stats=normalization)
        return {"cache": str(cache), "cache_reused": reused, "dataset_sha256": info["dataset_sha256"],
                **base._split_report(info), "preference": preference, "critic_initialization": initialization}
    prepared = primary_call(prepare)
    native = CachedReplayDataset(prepared["cache"], split="train")
    validation = CachedReplayDataset(prepared["cache"], split="val")
    if not len(native) or not len(validation):
        raise ValueError("Native training and validation sets must be nonempty")
    normalization = native.metadata["normalization"]
    for seed in [cfg["seed"]]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if device.type == "cuda": torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(cfg["deterministic"])
    if pref["weight"]:
        from .preference_critic import PreferenceIQL
        critic = PreferenceIQL(**base._critic_model_config(config))
    else:
        critic = PixelIQL(**base._critic_model_config(config))
    if config["critic"]["init_checkpoint"]:
        load_critic(config["critic"]["init_checkpoint"], critic,
            expected_input_spec=default_input_spec(critic, config["critic"]["image_size"]),
            expected_normalization=normalization, expected_reward_definition=config["reward"],
            expected_termination_rule=config["reward"]["termination_rule"])
    critic = wrap(critic.to(device), device, unused=True,
                  gradient_as_bucket_view=config["distributed"]["gradient_as_bucket_view"])
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[config["critic"]["optimizer"]]
    optimizers = {"q": OptimizerGroup([*unwrap(critic).q1.parameters(), *unwrap(critic).q2.parameters()],
                    zero=cfg["zero_optimizer"], optimizer_class=optimizer_class,
                    lr=config["critic"]["learning_rate"], weight_decay=config["critic"]["weight_decay"]),
                  "value": OptimizerGroup(unwrap(critic).value.parameters(), zero=cfg["zero_optimizer"],
                    optimizer_class=optimizer_class, lr=config["critic"]["value_learning_rate"],
                    weight_decay=config["critic"]["weight_decay"])}
    effective = copy.deepcopy(config)
    effective["training"]["gradient_accumulation_steps"] = plan["gradient_accumulation_steps"]
    from .continuation import comparable_config
    contract = {"config_sha256": stable_hash(comparable_config(effective)), "world_size": world_size(),
                "dataset_sha256": prepared["dataset_sha256"], "cache_fingerprint": native.metadata["fingerprint"],
                "preference_sha256": prepared["preference"]["dataset_sha256"],
                "frozen_membership_sha256": stable_hash(prepared["preference"]["frozen_membership"]),
                "source_sha256": _source_identity(), "critic_initialization": prepared["critic_initialization"],
                "behavior_policy_id": pref["behavior_policy_id"], "behavior_actor_sha256": pref["behavior_actor_sha256"],
                **plan}
    if pref["weight"]:
        contract["source_sha256"]["preference_data.py"] = sha256_file(Path(__file__).with_name("preference_data.py"))
    algorithm = "preference_regularized_idql_critic" if pref["weight"] else "idql_critic_no_preference_control"
    metadata = {"algorithm": algorithm, "policy_optimizer_updates": 0,
                "best_metric": None, "best_update": None, "best_critic_export": None,
                "last_validation_metric": None, "last_validation_update": None,
                "last_diagnostic_metric": None, "last_diagnostic_update": None}
    start = 0
    if resume:
        start, metadata = load_checkpoint(resume, actor=None, critic=critic, optimizers=optimizers,
                                          contract=contract, device=device)
        primary_call(lambda: base._restore_run_artifacts(directory, start, metadata))
    else:
        random.seed(cfg["seed"] + rank()); np.random.seed(cfg["seed"] + rank()); torch.manual_seed(cfg["seed"] + rank())
        if device.type == "cuda": torch.cuda.manual_seed(cfg["seed"] + rank())
    if start >= plan["optimizer_updates"]:
        raise ValueError("Checkpoint already reached the configured updates")
    def setup():
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False))
        atomic_json(directory / "normalization.json", normalization)
        atomic_json(directory / "provenance.json", {"algorithm": metadata["algorithm"], "contract": contract,
            "behavior_policy_loaded": False, "policy_optimizer_updates": 0, "prepared": prepared,
            "preference_settings": pref, "normalization": normalization})
    primary_call(setup)
    sampler = GlobalBatchSampler(len(native), plan["global_micro_batch_size"], rank(), world_size(),
        cfg["seed"] + 1, start * plan["gradient_accumulation_steps"], plan["micro_batches"], cfg["sampling"])
    iterator = iter(base._loader(native, config, sampler=sampler))
    validation_loader = base._loader(validation, config, validation=True) if rank() == 0 else None
    preference_iterator, preference_validation = None, None
    if pref["weight"]:
        from .preference_data import PreferenceDataset
        args = dict(validation_fraction=config["data"]["validation_fraction"], seed=config["data"]["seed"],
                    frozen_membership=prepared["preference"]["frozen_membership"]["members"],
                    split_strategy="native_membership", sources=tuple(pref["sources"]))
        pair_train = all_rank_call(lambda: PreferenceDataset(pref["package"], split="train", **args))
        if pair_train.metadata["dataset_sha256"] != prepared["preference"]["dataset_sha256"]:
            raise ValueError("Preference dataset changed after preparation")
        pair_sampler = GlobalBatchSampler(len(pair_train), plan["preference_global_micro_batch_size"], rank(), world_size(),
            cfg["seed"] + 200003, start * plan["gradient_accumulation_steps"], plan["micro_batches"], "replacement")
        preference_iterator = iter(base._loader(pair_train, config, sampler=pair_sampler))
        def prepare_pair_validation():
            nonlocal preference_validation
            pair_val = PreferenceDataset(pref["package"], split="val", **args)
            if pair_val.metadata["dataset_sha256"] != prepared["preference"]["dataset_sha256"]:
                raise ValueError("Preference validation dataset changed after preparation")
            preference_validation = base._loader(pair_val, config, validation=True)
        primary_call(prepare_pair_validation)
    with _run_outputs(directory, config, start, resume is not None) as writer:
        started, checkpoint, final_update, stopped = time.perf_counter(), None, start, False
        def export(name, update):
            return export_critic(directory / name, unwrap(critic),
                input_spec=default_input_spec(unwrap(critic), config["critic"]["image_size"]), normalization=normalization,
                reward_definition=config["reward"], termination_rule=config["reward"]["termination_rule"],
                provenance={"algorithm": metadata["algorithm"], "update": update, "policy_optimizer_updates": 0,
                            "dataset_sha256": prepared["dataset_sha256"], "preference": prepared["preference"],
                            "preference_settings": pref, "contract": contract, **base._split_report(prepared)})
        for update in range(start, plan["optimizer_updates"]):
            begin = time.perf_counter()
            batches = [next(iterator) for _ in range(plan["gradient_accumulation_steps"])]
            pair_batches = ([next(preference_iterator) for _ in batches] if preference_iterator is not None else None)
            metrics = optimization_step(config, batches, critic, optimizers, device,
                                        preference_batches=pair_batches, normalization=normalization)
            if device.type == "cuda": torch.cuda.synchronize(device)
            metrics = reduce_metrics(metrics, device)
            final_update = update + 1
            requested = torch.tensor(int(stopping() or (directory / "stop.request").exists()), device=device)
            if active(): dist.all_reduce(requested, op=dist.ReduceOp.MAX)
            stopped = bool(requested) and final_update < plan["optimizer_updates"]
            final = stopped or final_update == plan["optimizer_updates"]
            metrics.update(optimizer_update=final_update, critic_optimizer_updates=final_update,
                q_optimizer_updates=final_update, value_optimizer_updates=final_update, policy_optimizer_updates=0,
                micro_batch_count=final_update * plan["gradient_accumulation_steps"],
                sample_draws=final_update * plan["global_batch_size"],
                preference_sample_draws=final_update * plan["preference_global_batch_size"],
                global_batch_size=plan["global_batch_size"], preference_global_batch_size=plan["preference_global_batch_size"],
                update_seconds=time.perf_counter()-begin, elapsed_seconds=time.perf_counter()-started)
            if final_update % cfg["validation_interval"] == 0 or final:
                evaluation = primary_call(lambda: evaluate_critic(unwrap(critic), validation_loader, device,
                    beta=config["critic"]["beta"], max_weight=config["critic"]["max_weight"]))
                metrics["validation"] = evaluation
                metrics["preference_validation"] = primary_call(lambda: evaluate_preferences(
                    unwrap(critic), preference_validation, normalization, config, device))
                score = evaluation["spearman_v_remaining_blocks"]
                metadata.update(last_diagnostic_metric=score, last_diagnostic_update=final_update,
                                last_validation_metric=score, last_validation_update=final_update)
                if score is not None and evaluation["finite"] and (metadata["best_metric"] is None or score < metadata["best_metric"]):
                    name = primary_call(lambda: base._best_export_name(directory, final_update))
                    metadata.update(best_metric=score, best_update=final_update, best_critic_export=name)
                    primary_call(lambda: export(name, final_update))
                    primary_call(lambda: base._atomic_copy(directory/name, directory/'critic_best.pt'))
            def write_metrics():
                with (directory / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metrics, allow_nan=False)+"\n")
                if writer is not None:
                    def add_scalars(values, prefix=""):
                        for name, value in values.items():
                            if isinstance(value, dict):
                                add_scalars(value, prefix + name + "/")
                            elif type(value) in (float, int):
                                writer.add_scalar(prefix + name, value, final_update)
                    add_scalars(metrics)
                if final_update == start + 1 or final_update % cfg["log_interval"] == 0 or final:
                    print("PREFERENCE_UPDATE " + json.dumps(metrics, allow_nan=False), flush=True)
            primary_call(write_metrics)
            if final_update % cfg["checkpoint_interval"] == 0 or final:
                checkpoint = save_checkpoint(directory/"checkpoints"/f"update_{final_update:08d}.pt", update=final_update,
                    actor=None, critic=critic, optimizers=optimizers, contract=contract, metadata=metadata,
                    device=device, checkpoint_format=cfg["checkpoint_format"])
            if stopped: break
        primary_call(lambda: export("critic_last.pt", final_update))
        summary = {"status": "STOPPED" if stopped else "COMPLETED", "algorithm": metadata["algorithm"],
            "optimizer_updates": final_update, "critic_optimizer_updates": final_update, "policy_optimizer_updates": 0,
            "native_sample_draws": final_update*plan["global_batch_size"],
            "preference_sample_draws": final_update*plan["preference_global_batch_size"],
            "checkpoint": str(checkpoint), "policy": None, "best_critic_update": metadata["best_update"],
            "best_validation_spearman": metadata["best_metric"], "elapsed_seconds": time.perf_counter()-started,
            "behavior_policy_loaded": False, "contract": contract}
        primary_call(lambda: atomic_json(directory/"summary.json", summary))
        primary_call(lambda: atomic_json(directory/"status.json", summary))
        return summary
