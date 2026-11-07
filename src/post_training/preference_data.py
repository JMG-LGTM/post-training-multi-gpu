"""Portable, explicitly paired action preferences and deterministic mixing.

The input to :func:`package_preferences` is a JSON manifest with format
``same_state_action_preferences_v1``, schema_version=1, image_orientation set to
``already_vertical_flipped``, ``files`` mapping relative NPZ paths to
``{sha256, bytes}``, and ``records``. Each record contains pair_id, root_run_id,
source (current/history/sft), prompt, payload, chosen_origin, rejected_origin,
and same_state_evidence. Origins contain root_run_id, run_id, and step.

Each NPZ contains agent_image/wrist_image (oriented uint8 HWC RGB),
raw_proprio (float32[8]), chosen_actions/rejected_actions (float32[8,7]), and
action_mask (binary[8]). Current/history also contain chosen_sim_state and
rejected_sim_state: identical finite saved simulator vectors. Evidence is
``{kind: identical_sim_state, sim_state_sha256, conditioning_sha256}``.
The conditioning digest binds BOTH images, proprioception and prompt. SFT
records may instead use ``{kind: sft_self_pair, conditioning_sha256}``, equal
actions and identical origins, without simulator vectors.

These checks validate supplied evidence and integrity. They do not establish
that a human preference label is correct, or independently verify a claimed
source recording. No pairs are inferred from success labels or equal indices.
The initial format inherits this runner's H8/A7/P8 observation/action contract;
it contains no fixed task IDs, dataset sizes, source paths or model checkpoint.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .io import atomic_json, sha256_file, stable_hash

FORMAT = "same_state_action_preferences_v1"
SOURCES = ("current", "history", "sft")
ARRAYS = {"agent_image", "wrist_image", "raw_proprio", "chosen_actions",
          "rejected_actions", "action_mask"}
MAX_PAIR_BYTES = 32 * 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError("Preference data: " + message)


def _text(value, name):
    _require(isinstance(value, str) and bool(value.strip()), name + " must be nonempty")
    return value


def _digest(value, name):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             name + " must be a SHA-256 digest")


def _safe(root, relative):
    _text(relative, "payload path")
    part = Path(relative)
    _require(not part.is_absolute() and part.as_posix() == relative and
             all(p not in ("", ".", "..") for p in part.parts), "unsafe relative payload path")
    root = Path(root).resolve()
    target = root / part
    _require(not any(p.is_symlink() for p in (target, *target.parents) if p != root.parent),
             "symlink payload paths are forbidden")
    _require(target.resolve().is_relative_to(root), "payload escapes package")
    return target


def array_sha256(value):
    """Canonical numeric array identity, including its dtype and shape."""
    value = np.asarray(value)
    _require(value.dtype.kind in "buif" and value.dtype.hasobject is False,
             "array identity requires numeric non-object values")
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(header + b"\0")
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def conditioning_sha256(agent_image, wrist_image, raw_proprio, prompt):
    """Identity of the single observation and instruction shared by a pair."""
    return stable_hash({"agent_image": array_sha256(agent_image),
                        "wrist_image": array_sha256(wrist_image),
                        "raw_proprio": array_sha256(raw_proprio),
                        "prompt": _text(prompt, "prompt")})


def _origin(value, root):
    _require(isinstance(value, dict) and set(value) == {"root_run_id", "run_id", "step"},
             "origin requires root_run_id, run_id and step")
    _require(value["root_run_id"] == root, "chosen/rejected origins must belong to the same root")
    _text(value["run_id"], "origin.run_id")
    _require(type(value["step"]) is int and value["step"] >= 0, "origin.step must be a nonnegative integer")


def _load_pair(root, record):
    path = _safe(root, record["payload"])
    _require(path.is_file(), "missing pair payload")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        _require(len({member.filename for member in members}) == len(members), "duplicate NPZ members")
        _require(sum(member.file_size for member in members) <= MAX_PAIR_BYTES, "pair NPZ is too large")
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    evidence = record["same_state_evidence"]
    _require(isinstance(evidence, dict), "same-state evidence must be a mapping")
    _require(evidence.get("kind") in {"identical_sim_state", "sft_self_pair"}, "unsupported same-state evidence")
    self_pair = record["source"] == "sft" and evidence.get("kind") == "sft_self_pair"
    expected = ARRAYS if self_pair else ARRAYS | {"chosen_sim_state", "rejected_sim_state"}
    _require(set(values) == expected, "pair NPZ fields differ from its declared same-state evidence")
    for name in ("agent_image", "wrist_image"):
        image = values[name]
        _require(image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
                 and all(0 < size <= 1024 for size in image.shape[:2]), "images must be uint8 HWC RGB")
    _require(values["agent_image"].shape == values["wrist_image"].shape, "camera dimensions must agree")
    state = values["raw_proprio"]
    _require(state.dtype == np.float32 and state.shape == (8,) and np.isfinite(state).all(),
             "raw_proprio must be finite float32[8]")
    mask = values["action_mask"]
    _require(mask.shape == (8,) and mask.dtype.kind in "buif" and np.isfinite(mask).all()
             and np.isin(mask, [0, 1]).all() and mask.any(), "action mask must be binary nonempty [8]")
    mask = mask.astype(bool)
    _require(np.array_equal(mask, np.arange(8) < mask.sum()), "action mask must describe a contiguous prefix")
    for name in ("chosen_actions", "rejected_actions"):
        actions = values[name]
        _require(actions.dtype == np.float32 and actions.shape == (8, 7) and np.isfinite(actions).all(),
                 "actions must be finite float32[8,7]")
        _require((np.abs(actions) <= 1.0).all(), "environment actions must lie in [-1,1]")
        _require((actions[~mask] == 0).all(), "masked padding actions must be zero")
    condition = conditioning_sha256(values["agent_image"], values["wrist_image"], state, record["prompt"])
    _require(evidence.get("conditioning_sha256") == condition, "conditioning identity differs")
    if self_pair:
        _require(set(evidence) == {"kind", "conditioning_sha256"}, "unknown SFT evidence fields")
        _require(record["chosen_origin"] == record["rejected_origin"] and
                 np.array_equal(values["chosen_actions"], values["rejected_actions"]),
                 "SFT self-pairs require identical origins and actions")
    else:
        _require(set(evidence) == {"kind", "conditioning_sha256", "sim_state_sha256"}
                 and evidence["kind"] == "identical_sim_state", "unsupported same-state evidence")
        a, b = values["chosen_sim_state"], values["rejected_sim_state"]
        _require(a.ndim == 1 and a.size > 0 and a.dtype.kind == "f" and np.isfinite(a).all()
                 and b.dtype == a.dtype and b.shape == a.shape and np.isfinite(b).all()
                 and np.array_equal(a, b), "chosen/rejected saved simulator states must be identical")
        _require(evidence["sim_state_sha256"] == array_sha256(a), "saved-state identity differs")
        if record["source"] == "sft":
            _require(record["chosen_origin"] == record["rejected_origin"] and
                     np.array_equal(values["chosen_actions"], values["rejected_actions"]),
                     "SFT requires identical origins and actions")
        else:
            _require(not np.array_equal(values["chosen_actions"][mask], values["rejected_actions"][mask]),
                     "preference actions must differ on at least one valid coordinate")
    values["action_mask"] = mask
    return values


def _manifest(root, *, input_manifest=None):
    path = Path(input_manifest) if input_manifest is not None else Path(root) / "manifest.json"
    _require(path.is_file() and not path.is_symlink(), "manifest must be a regular JSON file")
    value = json.loads(path.read_text())
    required = {"schema_version", "format", "image_orientation", "files", "records"}
    allowed = required | {"dataset_sha256", "frozen_membership"}
    _require(isinstance(value, dict) and required <= set(value) <= allowed,
             "manifest fields differ from the preference schema")
    _require(type(value["schema_version"]) is int and value["schema_version"] == 1
             and value["format"] == FORMAT and value["image_orientation"] == "already_vertical_flipped",
             "unsupported preference format or image orientation")
    _require(isinstance(value["records"], list) and bool(value["records"]), "empty preference records")
    _require(isinstance(value["files"], dict) and bool(value["files"]), "missing preference payload identities")
    body = {key: item for key, item in value.items() if key != "dataset_sha256"}
    fingerprint = stable_hash(body)
    if input_manifest is None:
        _require(value.get("dataset_sha256") == fingerprint, "package manifest fingerprint differs")
    elif "dataset_sha256" in value:
        _require(value["dataset_sha256"] == fingerprint, "input manifest fingerprint differs")
    ids, paths = set(), set()
    record_keys = {"pair_id", "root_run_id", "source", "prompt", "payload", "chosen_origin", "rejected_origin", "same_state_evidence"}
    for record in value["records"]:
        _require(isinstance(record, dict) and set(record) == record_keys, "pair record fields differ")
        for field in ("pair_id", "root_run_id", "prompt"):
            _text(record[field], field)
        _require(record["pair_id"] not in ids, "duplicate pair_id")
        ids.add(record["pair_id"])
        _require(record["source"] in SOURCES, "unknown pair source")
        _origin(record["chosen_origin"], record["root_run_id"])
        _origin(record["rejected_origin"], record["root_run_id"])
        path = _safe(root, record["payload"])
        _require(path.suffix == ".npz" and path.is_file() and record["payload"] not in paths,
                 "each pair must own a unique existing NPZ payload")
        paths.add(record["payload"])
    _require(paths == set(value["files"]), "payload identity list must exactly cover pair records")
    verified_stats = {}
    for name, spec in value["files"].items():
        _require(isinstance(spec, dict) and set(spec) == {"sha256", "bytes"}, "payload identity fields differ")
        _digest(spec["sha256"], "payload sha256")
        _require(type(spec["bytes"]) is int and 0 < spec["bytes"] <= MAX_PAIR_BYTES, "invalid pair payload size")
        path = _safe(root, name)
        before = path.stat()
        _require(before.st_size == spec["bytes"] and sha256_file(path) == spec["sha256"], "pair payload SHA/size mismatch")
        after = path.stat()
        _require(all(getattr(before, key) == getattr(after, key) for key in
                     ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")), "payload changed while verifying")
        verified_stats[name] = after
    for record in value["records"]:
        _load_pair(root, record)
        after = _safe(root, record["payload"]).stat()
        _require(all(getattr(verified_stats[record["payload"]], key) == getattr(after, key) for key in
                     ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")), "payload changed while validating evidence")
    if "frozen_membership" in value:
        _membership(value["frozen_membership"])
    return value, fingerprint


def _split(roots, fraction, seed):
    _require(type(fraction) in (int, float) and math.isfinite(fraction) and 0 <= fraction < 1,
             "validation_fraction must be in [0,1)")
    _require(type(seed) is int and seed >= 0, "split seed must be a nonnegative integer")
    return {root: ("validation" if int.from_bytes(hashlib.sha256(f"{seed}:{root}".encode()).digest()[:8], "big")
                   / float(2 ** 64) < fraction else "train") for root in roots}


def _membership(value):
    _require(isinstance(value, dict) and bool(value), "frozen membership must be a nonempty root mapping")
    result = {}
    for root, split in value.items():
        _text(root, "frozen root")
        _require(split in {"train", "val", "validation"}, "invalid frozen root split")
        result[root] = "validation" if split == "val" else split
    return result


def inspect_preference_package(root, *, validation_fraction=.2, seed=7, frozen_membership=None,
                               split_strategy="haizhou_hash_threshold"):
    """Verify every small pair and return exact root-held-out metadata.

    A supplied/embedded frozen map may contain the larger original root roster;
    Default assignments must agree with the same Haizhou threshold and every
    pair root must be present. ``native_membership`` instead uses an explicitly
    supplied full-source roster as authoritative, allowing a caller to preserve
    a different native dataset's split. The caller verifies that roster against
    its native source; this reader never ranks the preference subset itself.
    """
    root = Path(root).expanduser().resolve()
    manifest, fingerprint = _manifest(root)
    roots = sorted({record["root_run_id"] for record in manifest["records"]})
    _require(split_strategy in {"haizhou_hash_threshold", "native_membership"}, "unknown split strategy")
    splits = _split(roots, validation_fraction, seed)
    embedded = manifest.get("frozen_membership")
    frozen = _membership(frozen_membership) if frozen_membership is not None else None
    if embedded is not None:
        embedded = _membership(embedded)
        _require(frozen is None or frozen == embedded, "explicit frozen membership differs from package")
        frozen = embedded
    if frozen is not None:
        _require(set(roots) <= set(frozen), "pair root is missing from frozen membership")
        if split_strategy == "haizhou_hash_threshold":
            _require(frozen == _split(frozen, validation_fraction, seed), "frozen membership differs from the Haizhou split")
            _require(all(splits[root] == frozen[root] for root in roots), "held-out root assignment changed")
        else:
            splits = {root: frozen[root] for root in roots}
    _require(split_strategy != "native_membership" or frozen is not None,
             "native_membership requires the frozen full-source root mapping")
    members = [{**record, "split": splits[record["root_run_id"]]} for record in manifest["records"]]
    source_counts = dict(Counter(row["source"] for row in members))
    split_counts = Counter(row["split"] for row in members)
    split_members = {split: sorted(root for root in roots if splits[root] == split) for split in ("train", "validation")}
    return {"schema_version": 1, "dataset_sha256": fingerprint, "fingerprint": fingerprint,
            "pair_count": len(members), "count": len(members), "train_count": split_counts["train"],
            "val_count": split_counts["validation"], "validation_count": split_counts["validation"],
            "source_counts": {source: source_counts.get(source, 0) for source in SOURCES},
            "pair_type_counts": {source: source_counts.get(source, 0) for source in SOURCES},
            "split_source_counts": {split: {source: sum(row["source"] == source and row["split"] == split for row in members)
                                            for source in SOURCES} for split in ("train", "validation")},
            "root_counts": {split: len(split_members[split]) for split in split_members},
            "split_members": split_members, "members": members, "split_strategy": split_strategy,
            "validation_fraction": validation_fraction, "split_seed": seed,
            "frozen_membership_sha256": stable_hash(frozen) if frozen is not None else None,
            "evidence_scope": "supplied_saved_states_and_conditioning; human_labels_and_recording_provenance_not_independently_verified"}


def package_preferences(input_manifest, output):
    """Verify already paired inputs and atomically publish an immutable package."""
    source_manifest = Path(input_manifest).expanduser().resolve()
    source = source_manifest.parent
    manifest, fingerprint = _manifest(source, input_manifest=source_manifest)
    manifest = {**manifest, "dataset_sha256": fingerprint}
    output = Path(output).expanduser().absolute()
    _require(not output.exists() and not output.is_symlink(), "output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
    try:
        for name in manifest["files"]:
            target = _safe(temporary, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_safe(source, name), target)
        atomic_json(temporary / "manifest.json", manifest)
        # Re-read copied bytes and evidence before publishing; source races fail.
        _manifest(temporary)
        _require(not output.exists(), "output appeared during packaging")
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"output": str(output), "dataset_sha256": fingerprint, "pair_count": len(manifest["records"])}


class PreferenceDataset(Dataset):
    """CPU-only pair reader; validation roots never enter a training instance."""
    def __init__(self, root, split="train", *, validation_fraction=.2, seed=7, frozen_membership=None,
                 sources=None, split_strategy="haizhou_hash_threshold"):
        _require(split in {"train", "val", "validation"}, "split must be train or validation")
        split = "validation" if split == "val" else split
        self.root = Path(root).expanduser().resolve()
        self.metadata = inspect_preference_package(self.root, validation_fraction=validation_fraction,
            seed=seed, frozen_membership=frozen_membership, split_strategy=split_strategy)
        selected = set(SOURCES if sources is None else sources)
        _require(bool(selected) and selected <= set(SOURCES), "unknown/empty dataset source selection")
        self.records = [record for record in self.metadata["members"] if record["split"] == split and record["source"] in selected]
        self.sources = tuple(source for source in SOURCES if source in selected)
        self.split = split
        self._stats = {record["payload"]: _safe(self.root, record["payload"]).stat() for record in self.records}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = _safe(self.root, record["payload"])
        before = self._stats[record["payload"]]
        after = path.stat()
        _require(all(getattr(before, key) == getattr(after, key) for key in
                     ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")), "immutable pair changed after inspection")
        values = _load_pair(self.root, record)
        after = path.stat()
        _require(all(getattr(before, key) == getattr(after, key) for key in
                     ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")), "immutable pair changed during reading")
        return {**{key: torch.from_numpy(np.array(values[key], copy=True)) for key in ARRAYS},
                **{key: record[key] for key in ("prompt", "root_run_id", "source", "pair_id")}}


def critic_preference_batch(batch, normalization, *, image_size=128, device="cpu"):
    """Encode one collated raw pair batch using an existing critic's statistics.

    No statistics are fitted here and images are already oriented. Both actions
    share the exact pixels/proprio/mask; padding stays zero after normalization.
    The input batch is untouched. Returned raw_* fields retain the CPU values
    for audits and future policy-side use; encoded critic tensors use ``device``.
    """
    from torch.nn import functional as F
    from .replay import normalize, validate_normalization

    _require(type(image_size) is int and 0 < image_size <= 1024, "invalid critic image size")
    stats = validate_normalization(normalization)
    tensors = {name: batch[name] for name in ARRAYS}
    _require(all(isinstance(value, torch.Tensor) and value.device.type == "cpu" for value in tensors.values()),
             "raw preference fields must remain CPU tensors")
    agent, wrist = tensors["agent_image"], tensors["wrist_image"]
    state, mask = tensors["raw_proprio"], tensors["action_mask"]
    _require(agent.dtype == wrist.dtype == torch.uint8 and agent.ndim == 4 and agent.shape == wrist.shape
             and agent.shape[-1] == 3 and agent.shape[0] > 0, "critic conversion requires batched uint8 HWC cameras")
    size = agent.shape[0]
    _require(state.dtype == torch.float32 and tuple(state.shape) == (size, 8) and torch.isfinite(state).all(),
             "critic conversion requires finite float32[B,8] proprioception")
    _require(tuple(mask.shape) == (size, 8) and torch.isfinite(mask).all()
             and ((mask == 0) | (mask == 1)).all() and mask.bool().any(1).all(), "invalid batched action mask")
    pixels = torch.cat((agent, wrist), dim=-1).permute(0, 3, 1, 2)
    pixels = F.interpolate(pixels.float(), size=(image_size, image_size), mode="bilinear", align_corners=False)
    result = {"pixels": pixels.clamp(0, 255).to(device=device, dtype=torch.uint8),
              "proprio": torch.from_numpy(normalize(state.numpy(), stats["state"])).to(device),
              "action_mask": mask.to(device=device, dtype=torch.bool), "raw_proprio": state}
    for name in ("chosen_actions", "rejected_actions"):
        actions = tensors[name]
        _require(actions.dtype == torch.float32 and tuple(actions.shape) == (size, 8, 7)
                 and torch.isfinite(actions).all() and (actions.abs() <= 1).all(), "invalid raw action batch")
        raw = actions.numpy()
        encoded = np.zeros_like(raw)
        selected = mask.bool().numpy()
        encoded[selected] = normalize(raw[selected], stats["actions"])
        result[name] = torch.from_numpy(encoded).to(device)
        result["raw_" + name] = actions
    return result


class PreferenceBatchSampler(Sampler):
    """Exact global source quotas with replacement and stateless exact resume.

    Largest remainders allocate each effective *microbatch*; configured weights
    are relative. A final permutation distributes global draws across equal
    local rank slices. Duplicate draws are intentional, including within a rank.
    With very small batches a positive source can receive quota zero; inspect
    ``source_counts`` to see the actual fixed integer allocation.
    """
    def __init__(self, dataset, global_batch, rank, world, seed, start, end,
                 source_weights=None):
        for value, name, minimum in ((global_batch, "global_batch", 1), (world, "world", 1),
                                     (rank, "rank", 0), (seed, "seed", 0), (start, "start", 0), (end, "end", 0)):
            _require(type(value) is int and value >= minimum, name + " is invalid")
        _require(global_batch % world == 0 and 0 <= rank < world and start <= end, "invalid global batch/rank/range")
        weights = {"current": .8, "history": 0., "sft": .2} if source_weights is None else dict(source_weights)
        _require(set(weights) <= set(SOURCES), "unknown sampling source")
        weights = {source: weights.get(source, 0.) for source in SOURCES}
        _require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in weights.values())
                 and math.isfinite(sum(weights.values())) and sum(weights.values()) > 0, "source weights must be finite nonnegative with a positive sum")
        self.pools = {source: [i for i, row in enumerate(dataset.records) if row["source"] == source] for source in SOURCES}
        _require(all(self.pools[source] for source in SOURCES if weights[source] > 0),
                 "positive source weight has no examples in this split")
        exact = {source: global_batch * weights[source] / sum(weights.values()) for source in SOURCES}
        counts = {source: math.floor(exact[source]) for source in SOURCES}
        order = sorted(SOURCES, key=lambda source: (-(exact[source] - counts[source]), SOURCES.index(source)))
        for source in order[:global_batch - sum(counts.values())]:
            counts[source] += 1
        self.source_counts, self.source_weights = counts, weights
        self.global_batch, self.rank, self.world = global_batch, rank, world
        self.seed, self.start, self.end = seed, start, end
        self.sampling = "replacement_by_source"

    def __len__(self):
        return self.end - self.start

    def __iter__(self):
        local = self.global_batch // self.world
        for step in range(self.start, self.end):
            key = f"preference_sampler_v1:{self.seed}:{step}".encode()
            seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") & ((1 << 63) - 1)
            generator = torch.Generator().manual_seed(seed)
            indices = []
            for source in SOURCES:
                count = self.source_counts[source]
                if count:
                    draws = torch.randint(len(self.pools[source]), (count,), generator=generator).tolist()
                    indices.extend(self.pools[source][draw] for draw in draws)
            permutation = torch.randperm(self.global_batch, generator=generator).tolist()
            mixed = [indices[index] for index in permutation]
            yield mixed[self.rank * local:(self.rank + 1) * local]
