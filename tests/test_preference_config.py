import copy
from pathlib import Path

import pytest
import yaml

from post_training.preference_config import (defaults, load_config,
    preference_batch_plan, validate_preference_config)


def config():
    value = defaults()
    value["preference"]["package"] = "/not/opened/by/config"
    return value


@pytest.mark.parametrize("field,value", [("weight", -1), ("weight", True),
    ("weight", float("nan")), ("temperature", 0), ("temperature", float("inf")),
    ("global_batch_size", True), ("per_device_batch_size", 0),
    ("behavior_actor_sha256", "123")])
def test_preference_config_rejects_ambiguous_or_nonfinite_settings(field, value):
    c = config(); c["preference"][field] = value
    with pytest.raises(ValueError): validate_preference_config(c)


def test_config_is_generic_and_optional_behavior_binding_is_paired():
    c = config()
    assert c["preference"]["behavior_policy_id"] is None
    assert c["preference"]["frozen_split_members"] is None
    validate_preference_config(c)
    c["preference"]["behavior_policy_id"] = "another-robot-policy"
    with pytest.raises(ValueError, match="together"): validate_preference_config(c)
    c["preference"]["behavior_actor_sha256"] = "f" * 64
    validate_preference_config(c)


def test_weight_zero_allows_no_package_and_never_counts_pairs():
    c = defaults(); c["preference"]["weight"] = 0
    validate_preference_config(c)
    plan = preference_batch_plan(c, 1)
    assert plan["preference_expected_draws"] == plan["preference_global_batch_size"] == 0
    assert plan["policy_optimizer_updates"] == 0


def test_auxiliary_batch_plan_is_a_separate_budget_and_must_match_world():
    c = config()
    c["training"].update(global_batch_size=64, per_device_batch_size=32, gradient_accumulation_steps=1)
    p = preference_batch_plan(c, 2)
    assert p["global_batch_size"] == 64 and p["preference_global_batch_size"] == 16
    assert p["preference_global_micro_batch_size"] == 16
    c["preference"]["global_batch_size"] = 15
    with pytest.raises(ValueError, match="preference global batch"): preference_batch_plan(c, 2)


def test_strict_loader_rejects_misspelled_preference_and_actor_training(tmp_path):
    c = config(); c["preference"]["temprature"] = 1
    path = tmp_path / "bad.yaml"; path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match="Unknown config"): load_config(path)
    c = config(); c["method"] = "iql"
    with pytest.raises(ValueError, match="only a critic"): validate_preference_config(c)


@pytest.mark.parametrize("name", ["idql.yaml", "control.yaml"])
def test_shipped_configs_describe_two_card_examples(name):
    c = load_config(Path(__file__).resolve().parents[1] / "configs" / "preference" / name)
    assert preference_batch_plan(c, 2)["policy_optimizer_updates"] == 0
    assert c["critic"]["init_checkpoint"] is None and c["data"]["validation_fraction"] == .2
