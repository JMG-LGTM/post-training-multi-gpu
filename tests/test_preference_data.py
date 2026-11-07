"""CPU integrity, source mixing, exact resume and pair-conditioning checks."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from post_training.io import sha256_file, stable_hash
from post_training.preference_data import (FORMAT, PreferenceBatchSampler, PreferenceDataset,
    array_sha256, conditioning_sha256, critic_preference_batch, inspect_preference_package,
    package_preferences)


def make_input(directory, *, roots=None):
    directory.mkdir(parents=True)
    roots = roots or ["root_0", "root_1", "root_2", "root_3", "root_4", "root_5", "root_6", "root_7"]
    manifest = dict(schema_version=1, format=FORMAT, image_orientation="already_vertical_flipped", files={}, records=[])
    for index, root in enumerate(roots):
        for source in ("current", "history", "sft"):
            pair_id = f"pair_{index}_{source}"
            name = f"pairs/{pair_id}.npz"
            agent = np.arange(9 * 11 * 3, dtype=np.uint8).reshape(9, 11, 3)
            wrist = 255 - agent
            state = np.arange(8, dtype=np.float32) / 8
            chosen = np.full((8, 7), .2, dtype=np.float32)
            rejected = np.full((8, 7), -.2, dtype=np.float32)
            mask = np.arange(8) < 6
            chosen[~mask] = rejected[~mask] = 0
            if source == "sft":
                rejected = chosen.copy()
            prompt = "place the bowl"
            evidence = dict(kind="sft_self_pair" if source == "sft" else "identical_sim_state",
                            conditioning_sha256=conditioning_sha256(agent, wrist, state, prompt))
            values = dict(agent_image=agent, wrist_image=wrist, raw_proprio=state,
                          chosen_actions=chosen, rejected_actions=rejected, action_mask=mask)
            if source != "sft":
                sim = np.arange(25, dtype=np.float64)
                values.update(chosen_sim_state=sim, rejected_sim_state=sim.copy())
                evidence["sim_state_sha256"] = array_sha256(sim)
            path = directory / name
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, **values)
            manifest["files"][name] = dict(sha256=sha256_file(path), bytes=path.stat().st_size)
            origin = dict(root_run_id=root, run_id=root + "_record", step=index)
            manifest["records"].append(dict(pair_id=pair_id, root_run_id=root, source=source,
                prompt=prompt, payload=name, chosen_origin=origin, rejected_origin=origin.copy(),
                same_state_evidence=evidence))
    path = directory / "input.json"
    path.write_text(json.dumps(manifest))
    return path


def publish(tmp_path, **kwargs):
    source = make_input(tmp_path / "source", **kwargs)
    target = tmp_path / "package"
    package_preferences(source, target)
    return source, target


def modify_payload(manifest_path, mutate):
    value = json.loads(manifest_path.read_text())
    record = value["records"][0]
    payload = manifest_path.parent / record["payload"]
    with np.load(payload, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    mutate(arrays, record)
    np.savez_compressed(payload, **arrays)
    value["files"][record["payload"]] = dict(sha256=sha256_file(payload), bytes=payload.stat().st_size)
    value.pop("dataset_sha256", None)
    manifest_path.write_text(json.dumps(value))


def test_portable_package_integrity_shapes_and_root_split(tmp_path):
    source, target = publish(tmp_path)
    report = inspect_preference_package(target)
    assert report["pair_count"] == 24 and report["train_count"] + report["val_count"] == 24
    expected = {f"root_{i}": ("validation" if int.from_bytes(hashlib.sha256(f"7:root_{i}".encode()).digest()[:8], "big") / 2**64 < .2 else "train") for i in range(8)}
    assert {row["root_run_id"]: row["split"] for row in report["members"]} == expected
    assert not set(report["split_members"]["train"]) & set(report["split_members"]["validation"])
    assert all(count % 3 == 0 for count in (report["train_count"], report["val_count"]))
    relocated = tmp_path / "moved"
    shutil.move(target, relocated)
    dataset = PreferenceDataset(relocated)
    item = dataset[0]
    assert item["chosen_actions"].shape == item["rejected_actions"].shape == (8, 7)
    assert item["agent_image"].dtype == torch.uint8 and item["raw_proprio"].dtype == torch.float32
    assert item["action_mask"].dtype == torch.bool
    assert set(dataset.records[0]) >= {"root_run_id", "source", "same_state_evidence"}
    assert dataset.metadata["dataset_sha256"] == report["dataset_sha256"]
    with pytest.raises(ValueError, match="output already exists"):
        package_preferences(source, relocated)


@pytest.mark.parametrize("mutate,reason", [
    (lambda a,r: a["rejected_sim_state"].__setitem__(3, 42), "states must be identical"),
    (lambda a,r: r["rejected_origin"].__setitem__("root_run_id", "unrelated"), "same root"),
    (lambda a,r: r.__setitem__("prompt", "a changed task"), "conditioning identity"),
    (lambda a,r: a["action_mask"].__setitem__(0, .5), "binary nonempty"),
    (lambda a,r: a["chosen_actions"].__setitem__((0,0), np.nan), "finite float32"),
    (lambda a,r: a["chosen_actions"].__setitem__((0,0), 1.01), "lie in"),
    (lambda a,r: a["chosen_actions"].__setitem__((7,0), .1), "padding actions"),
    (lambda a,r: a.__setitem__("chosen_actions", a["rejected_actions"].copy()), "must differ"),
    (lambda a,r: r.__setitem__("same_state_evidence", {}), "same-state evidence"),
])
def test_reject_unaligned_or_invalid_supplied_pairs(tmp_path, mutate, reason):
    source = make_input(tmp_path / "source")
    # Use a floating mask for the fractional mask test; bool indexing would coerce .5.
    if "binary" in reason:
        original = mutate
        def mutate(a, r):
            a["action_mask"] = a["action_mask"].astype(np.float32)
            original(a, r)
    modify_payload(source, mutate)
    with pytest.raises(ValueError, match=reason):
        package_preferences(source, tmp_path / "package")
    assert not (tmp_path / "package").exists()


def test_no_implicit_pairing_from_unrelated_payloads_and_source_metadata(tmp_path):
    source = make_input(tmp_path / "source")
    value = json.loads(source.read_text())
    value["records"][0]["same_state_evidence"] = {"kind": "same_timestep"}
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        package_preferences(source, tmp_path / "package")


def test_payload_tamper_and_immutable_dataset_checks(tmp_path):
    _, target = publish(tmp_path)
    dataset = PreferenceDataset(target)
    file = target / dataset.records[0]["payload"]
    file.write_bytes(file.read_bytes() + b"trailing mutation")
    with pytest.raises(ValueError, match="SHA/size"):
        inspect_preference_package(target)
    with pytest.raises(ValueError, match="immutable pair changed"):
        dataset[0]


def test_manifest_digest_and_unsafe_paths_are_rejected(tmp_path):
    _, target = publish(tmp_path)
    file = target / "manifest.json"
    value = json.loads(file.read_text())
    value["records"][0]["payload"] = "../outside.npz"
    file.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="fingerprint"):
        inspect_preference_package(target)
    value["dataset_sha256"] = stable_hash({k:v for k,v in value.items() if k != "dataset_sha256"})
    file.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="unsafe"):
        inspect_preference_package(target)


def test_frozen_full_membership_preserves_held_out_roots(tmp_path):
    _, target = publish(tmp_path)
    first = inspect_preference_package(target)
    frozen = {row["root_run_id"]: row["split"] for row in first["members"]}
    # A larger frozen source roster is allowed; subset-only rank splitting is not.
    extra = "a_root_not_present_in_pairs"
    frozen[extra] = "validation" if int.from_bytes(hashlib.sha256(f"7:{extra}".encode()).digest()[:8], "big") / 2**64 < .2 else "train"
    second = inspect_preference_package(target, frozen_membership=frozen)
    assert second["split_members"] == first["split_members"]
    changed = dict(frozen)
    name = next(iter(changed)); changed[name] = "train" if changed[name] == "validation" else "validation"
    with pytest.raises(ValueError, match="Haizhou"):
        inspect_preference_package(target, frozen_membership=changed)
    frozen.pop(name)
    with pytest.raises(ValueError, match="missing from frozen"):
        inspect_preference_package(target, frozen_membership=frozen)


def test_native_membership_uses_caller_full_source_split_without_resplitting_pairs(tmp_path):
    _, target = publish(tmp_path)
    native = {f"root_{i}": "validation" if i in (1,4) else "train" for i in range(8)}
    native["native_root_without_a_preference"] = "validation"
    info = inspect_preference_package(target, frozen_membership=native, split_strategy="native_membership")
    assert info["split_members"]["validation"] == ["root_1","root_4"]
    assert info["split_strategy"] == "native_membership"
    assert info["frozen_membership_sha256"] == stable_hash(native)
    validation = PreferenceDataset(target, split="validation", frozen_membership=native,
                                   split_strategy="native_membership", sources=("current",))
    assert {row["root_run_id"] for row in validation.records} == {"root_1","root_4"}
    assert all(row["source"]=="current" for row in validation.records)
    with pytest.raises(ValueError, match="requires the frozen"):
        inspect_preference_package(target, split_strategy="native_membership")


def test_largest_remainder_global_mixing_ddp_and_exact_resume(tmp_path):
    _, target = publish(tmp_path)
    dataset = PreferenceDataset(target)
    kw = dict(global_batch=64, world=2, seed=19, start=0, end=6,
              source_weights={"current": .8, "history": .0, "sft": .2})
    rank0 = PreferenceBatchSampler(dataset, rank=0, **kw)
    rank1 = PreferenceBatchSampler(dataset, rank=1, **kw)
    assert rank0.source_counts == {"current":51,"history":0,"sft":13}
    a, b = list(rank0), list(rank1)
    single = list(PreferenceBatchSampler(dataset, rank=0, **{**kw,"world":1}))
    assert all(len(x) == len(y) == 32 for x,y in zip(a,b))
    assert [x+y for x,y in zip(a,b)] == single
    for indices in single:
        assert Counter(dataset.records[i]["source"] for i in indices) == {"current":51,"sft":13}
    assert list(PreferenceBatchSampler(dataset, rank=0, **{**kw,"start":3})) == a[3:]
    assert any(len(set(batch)) < len(batch) for batch in single)  # intentional replacement


@pytest.mark.parametrize("weights", [{"unknown":1}, {"current":-1}, {"current":float("nan")}, {"current":0}])
def test_bad_mixture_weights_fail_closed(tmp_path, weights):
    _, target = publish(tmp_path)
    with pytest.raises(ValueError):
        PreferenceBatchSampler(PreferenceDataset(target), 64, 0, 2, 7, 0, 1, weights)


def test_mixture_requires_available_positive_sources(tmp_path):
    _, target = publish(tmp_path)
    dataset = PreferenceDataset(target)
    dataset.records = [row for row in dataset.records if row["source"] != "sft"]
    with pytest.raises(ValueError, match="no examples"):
        PreferenceBatchSampler(dataset, 64, 0, 2, 7, 0, 1)


def test_current_only_scorer_dataset_excludes_sft_selfpairs(tmp_path):
    _, target = publish(tmp_path)
    dataset = PreferenceDataset(target, sources=("current",))
    assert all(row["source"] == "current" for row in dataset.records)
    sampler = PreferenceBatchSampler(dataset, 64, 0, 2, 7, 0, 2, {"current":1,"history":0,"sft":0})
    assert sampler.source_counts == {"current":64,"history":0,"sft":0}
    assert all(len(indices)==32 for indices in sampler)


def test_critic_conversion_uses_fixed_norm_no_second_flip_and_zero_padding(tmp_path):
    _, target = publish(tmp_path)
    batch = default_collate([PreferenceDataset(target)[0], PreferenceDataset(target)[1]])
    before = {name:value.clone() for name,value in batch.items() if isinstance(value, torch.Tensor)}
    normalization = {"actions":{"q01":[-.5]*7,"q99":[.5]*7}, "state":{"q01":[0]*8,"q99":[2]*8}}
    encoded = critic_preference_batch(batch, normalization, image_size=9)
    from post_training.replay import _pixels, normalize
    assert encoded["pixels"].shape == (2,6,9,9)
    expected = _pixels(batch["agent_image"][0].numpy(), batch["wrist_image"][0].numpy(),9)
    assert np.array_equal(encoded["pixels"][0].numpy(), expected)
    valid = batch["action_mask"].bool()
    assert np.allclose(encoded["chosen_actions"][valid].numpy(), normalize(batch["chosen_actions"][valid].numpy(),normalization["actions"]))
    assert (encoded["chosen_actions"][~valid] == 0).all()
    assert (encoded["rejected_actions"][~valid] == 0).all()
    assert encoded["raw_chosen_actions"] is batch["chosen_actions"]
    assert all(torch.equal(batch[name],value) for name,value in before.items())
