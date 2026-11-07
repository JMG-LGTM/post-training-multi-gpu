"""Configuration for a frozen-policy, preference-regularized IQL critic.

This is an experimental objective, not FlowPRO/RPRO actor optimization.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import re

import yaml

from .config import DEFAULTS, _merge, batch_plan, validate_config


PREFERENCE_DEFAULTS = {
    "package": None,
    "weight": 0.1,
    "temperature": 1.0,
    "global_batch_size": 16,
    "per_device_batch_size": 8,
    "sources": ["current", "history"],
    "frozen_split_members": None,
    "behavior_policy_id": None,
    "behavior_actor_sha256": None,
}


def defaults():
    value = copy.deepcopy(DEFAULTS)
    value["method"] = "critic"
    value["data"]["split_strategy"] = "haizhou_hash_threshold"
    value["preference"] = copy.deepcopy(PREFERENCE_DEFAULTS)
    return value


def load_config(path, *, preferences=None, frozen_members=None):
    raw = yaml.safe_load(Path(path).read_text())
    value = _merge(defaults(), raw or {})
    if preferences is not None:
        value["preference"]["package"] = str(preferences)
    if frozen_members is not None:
        value["preference"]["frozen_split_members"] = str(frozen_members)
    validate_preference_config(value)
    return value


def validate_preference_config(config):
    validate_config(config)
    if config["method"] != "critic":
        raise ValueError("Preference IDQL trains only a critic; method must be critic")
    if config["model"]["actor_init_policy"] is not None or config["actor_bc"]["dataset_root"] is not None:
        raise ValueError("A frozen-policy scorer must not initialize or train an actor")
    if config["training"]["continuation"] is not None:
        raise ValueError("Use exact same-run --resume; IQL actor continuation is unsupported")
    if config["data"]["validation_fraction"] <= 0:
        raise ValueError("Preference training requires held-out roots")
    pref = config.get("preference")
    if not isinstance(pref, dict) or set(pref) != set(PREFERENCE_DEFAULTS):
        raise ValueError("Unknown or missing preference configuration fields")
    for key, positive in (("weight", False), ("temperature", True)):
        value = pref[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0 or positive and value == 0:
            raise ValueError(f"preference.{key} must be finite and {'positive' if positive else 'nonnegative'}")
    for key in ("global_batch_size", "per_device_batch_size"):
        if type(pref[key]) is not int or pref[key] < 1:
            raise ValueError(f"preference.{key} must be a positive integer")
    if (not isinstance(pref["sources"], list) or not pref["sources"]
            or len(set(pref["sources"])) != len(pref["sources"])
            or any(source not in {"current", "history"} for source in pref["sources"])):
        raise ValueError("preference.sources must select unique current/history preference sources")
    for key in ("package", "frozen_split_members"):
        if pref[key] is not None and (not isinstance(pref[key], str) or not pref[key].strip()):
            raise ValueError(f"preference.{key} must be a nonempty path or null")
    if pref["weight"] > 0 and pref["package"] is None:
        raise ValueError("Positive preference weight requires a verified preference package")
    identifier, digest = pref["behavior_policy_id"], pref["behavior_actor_sha256"]
    if (identifier is None) != (digest is None):
        raise ValueError("Behavior policy ID and actor SHA256 must be supplied together or both null")
    if identifier is not None:
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("preference.behavior_policy_id must be a nonempty string or null")
        if re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None:
            raise ValueError("preference.behavior_actor_sha256 must be a full SHA256 or null")


def preference_batch_plan(config, world_size):
    plan = batch_plan(config, world_size)
    pref = config["preference"]
    expected = pref["per_device_batch_size"] * world_size * plan["gradient_accumulation_steps"]
    if pref["weight"] > 0 and pref["global_batch_size"] != expected:
        raise ValueError("preference global batch != per-device batch * world * native accumulation")
    return {**plan, "preference_global_batch_size": pref["global_batch_size"] if pref["weight"] else 0,
            "preference_global_micro_batch_size": pref["per_device_batch_size"] * world_size if pref["weight"] else 0,
            "preference_per_device_batch_size": pref["per_device_batch_size"] if pref["weight"] else 0,
            "preference_expected_draws": plan["optimizer_updates"] * pref["global_batch_size"] if pref["weight"] else 0,
            "policy_optimizer_updates": 0}
