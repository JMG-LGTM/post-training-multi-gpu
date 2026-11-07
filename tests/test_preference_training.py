"""Actual CPU critic updates, exact inactive control, resume and gloo coverage."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from post_training import preference_training as runner
from post_training import training as base
from post_training.data import inspect_dataset, package_dataset
from post_training.preference_config import defaults
from post_training.preference_data import package_preferences
from test_data_replay import make_run
from test_preference_data import make_input


@pytest.fixture
def inputs(tmp_path):
    roots = [f"root_{i}" for i in range(8)]
    source, native = tmp_path / "native_source", tmp_path / "native"
    for i, root in enumerate(roots):
        make_run(source, root, offset=.05*i)
    package_dataset(source, native, run_ids=roots, horizon=8)
    pairs = tmp_path / "pairs"
    package_preferences(make_input(tmp_path / "pair_source", roots=roots), pairs)
    return native, pairs


def config(tmp_path, pairs=None, *, updates=3):
    c = defaults()
    c["data"]["validation_fraction"] = .5
    c.update(output_dir=str(tmp_path / "run"), cache_dir=str(tmp_path / "cache"))
    c["training"].update(optimizer_updates=updates, global_batch_size=4,
        per_device_batch_size=2, gradient_accumulation_steps="auto", checkpoint_interval=1,
        validation_interval=1, log_interval=1, warmup_updates=0, zero_optimizer=False,
        tensorboard=False, sampling="replacement")
    c["critic"].update(image_size=16, random_shift_pad=4)
    c["distributed"].update(device="cpu", workers_per_rank=0, pin_memory=False)
    c["preference"].update(package=str(pairs) if pairs else None, weight=.1 if pairs else 0,
        global_batch_size=4, per_device_batch_size=2)
    return c


def state(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def assert_same_models(expected, actual):
    assert expected["actor"] is actual["actor"] is None
    assert expected["critic"].keys() == actual["critic"].keys()
    for name in expected["critic"]:
        torch.testing.assert_close(expected["critic"][name], actual["critic"][name], rtol=0, atol=0)


def test_weight_zero_is_exact_original_critic_run_and_never_reads_preferences(inputs, tmp_path, monkeypatch):
    native, _ = inputs
    c = config(tmp_path)
    core = runner.core_config(c); core["output_dir"] = str(tmp_path / "original")
    expected = base.train(core, native)
    c["preference"]["package"] = "/must/not/be/opened"
    def forbidden(*args, **kwargs):
        raise AssertionError("No pair inspection or sampling in the inactive control")
    monkeypatch.setattr("post_training.preference_data.inspect_preference_package", forbidden)
    monkeypatch.setattr("post_training.preference_data.PreferenceDataset", forbidden)
    actual = runner.train(c, native)
    assert_same_models(state(expected["checkpoint"]), state(actual["checkpoint"]))
    assert actual["preference_sample_draws"] == 0
    assert actual["policy_optimizer_updates"] == 0 and actual["behavior_policy_loaded"] is False


@pytest.mark.parametrize("checkpoint_format", ["consolidated", "sharded"])
def test_actual_preference_updates_stop_and_resume_exactly(inputs, tmp_path, monkeypatch, checkpoint_format):
    native, pairs = inputs
    c = config(tmp_path, pairs)
    c["training"]["checkpoint_format"] = checkpoint_format
    c["output_dir"] = str(tmp_path / "uninterrupted")
    full = runner.train(c, native)
    expected = state(full["checkpoint"])
    assert full["critic_optimizer_updates"] == 3 and full["policy_optimizer_updates"] == 0
    assert full["preference_sample_draws"] == 12
    c["output_dir"] = str(tmp_path / "resumed")
    step = runner.optimization_step
    def stop_after_update(*args, **kwargs):
        result = step(*args, **kwargs)
        (Path(c["output_dir"])/"stop.request").write_text("owned CPU test stop")
        return result
    monkeypatch.setattr(runner, "optimization_step", stop_after_update)
    paused = runner.train(c, native)
    assert paused["status"] == "STOPPED" and paused["optimizer_updates"] == 1
    monkeypatch.setattr(runner, "optimization_step", step)
    resumed = runner.train(c, native, resume=paused["checkpoint"])
    assert resumed["status"] == "COMPLETED"
    assert_same_models(expected, state(resumed["checkpoint"]))
    rows = [json.loads(line) for line in (Path(c["output_dir"])/"metrics.jsonl").read_text().splitlines()]
    assert [r["optimizer_update"] for r in rows] == [1,2,3]
    assert all("pref_loss" in r and "q_td_loss" in r and "preference_validation" in r for r in rows)
    assert all(r["preference_validation"]["held_out"] for r in rows)
    from post_training.critic import load_critic
    _, metadata = load_critic(Path(c["output_dir"])/"critic_last.pt")
    assert metadata["provenance"]["algorithm"] == "preference_regularized_idql_critic"
    assert metadata["provenance"]["policy_optimizer_updates"] == 0


def test_dry_run_checks_pair_split_without_policy_or_cache_creation(inputs, tmp_path):
    native, pairs = inputs
    c = config(tmp_path, pairs)
    before = torch.get_rng_state().clone()
    report = runner.dry_run(c, native, 2)
    assert not (tmp_path/"cache").exists()
    assert torch.equal(before, torch.get_rng_state())
    assert report["preference"]["train_count"] and report["preference"]["val_count"]
    assert report["behavior_policy_loaded"] is False and report["policy_loaded"] is False
    assert report["policy_optimizer_updates"] == 0


def test_frozen_native_membership_mismatch_fails_before_training(inputs, tmp_path):
    native, pairs = inputs
    c = config(tmp_path, pairs)
    info = inspect_dataset(native, runner.core_config(c), verify=False)
    snapshot = {"dataset_sha256":info["dataset_sha256"], "members":[
        {"root_run_id":r["root_run_id"],"run_id":r["run_id"],"split":r["split"]} for r in info["rows"]]}
    snapshot["members"][0]["split"] = "val" if snapshot["members"][0]["split"]=="train" else "train"
    path=tmp_path/"frozen.json";path.write_text(json.dumps(snapshot))
    c["preference"]["frozen_split_members"]=str(path)
    with pytest.raises(ValueError,match="root membership|root members"):
        runner.dry_run(c,native,2)


def test_pair_normalization_is_native_critic_and_padding_remains_zero(inputs):
    from post_training.preference_data import PreferenceDataset
    from torch.utils.data import default_collate
    _, pairs = inputs
    batch = default_collate([PreferenceDataset(pairs,split="train")[0]])
    stats = {"actions":{"q01":[-.8]*7,"q99":[.8]*7}, "state":{"q01":[-1.]*8,"q99":[1.]*8}}
    transformed=runner.preference_tensor_batch(batch,stats,16,torch.device("cpu"))
    mask=transformed["action_mask"]
    assert torch.all(transformed["chosen_actions"][~mask]==0)
    assert torch.all(transformed["rejected_actions"][~mask]==0)
    assert transformed["pixels"].shape==(1,6,16,16)
    assert torch.allclose(transformed["chosen_actions"][mask],torch.full_like(transformed["chosen_actions"][mask],.25),atol=2e-6)


def test_fresh_process_never_imports_actor_policy_stack(inputs,tmp_path):
    native,pairs=inputs;c=config(tmp_path,pairs,updates=1)
    path=tmp_path/"config.yaml";path.write_text(yaml.safe_dump(c))
    script="""
import importlib.abc,sys
class NoActor(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='post_training.policy' or fullname.startswith(('openpi','transformers')):
   raise AssertionError('Frozen scorer imported actor stack: '+fullname)
sys.meta_path.insert(0,NoActor())
from post_training.preference_config import load_config
from post_training.preference_training import train
r=train(load_config(sys.argv[1]),sys.argv[2])
assert r['policy_optimizer_updates']==0 and not r['behavior_policy_loaded']
"""
    result=subprocess.run([sys.executable,"-c",script,str(path),str(native)],capture_output=True,text=True,
        env={**os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"1",
             "PYTHONPATH":str(Path(__file__).resolve().parents[1]/"src")},timeout=90)
    assert result.returncode==0,result.stdout+result.stderr


def test_two_rank_gloo_full_state_resume(inputs,tmp_path):
    native,pairs=inputs;c=config(tmp_path,pairs,updates=2)
    c["training"].update(per_device_batch_size=1,gradient_accumulation_steps=2,
        zero_optimizer=True,checkpoint_format="sharded")
    c["preference"].update(per_device_batch_size=1)
    path=tmp_path/"config.yaml";path.write_text(yaml.safe_dump(c))
    project=Path(__file__).resolve().parents[1]
    command=[sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=2",
        str(project/"train_preference_idql.py"),"--config",str(path),"--data",str(native)]
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"1"}
    first=subprocess.run(command,env=env,capture_output=True,text=True,timeout=120)
    assert first.returncode==0,first.stdout+first.stderr
    checkpoint=Path(c["output_dir"])/"checkpoints/update_00000002.pt";before=state(checkpoint)
    second=subprocess.run(command+["--resume",str(Path(c["output_dir"])/"checkpoints/update_00000001.pt")],
        env=env,capture_output=True,text=True,timeout=120)
    assert second.returncode==0,second.stdout+second.stderr
    assert_same_models(before,state(checkpoint))
    assert before["contract"]["world_size"]==2


def test_cli_applies_preference_override_before_strict_validation(inputs,tmp_path):
    native,pairs=inputs;c=config(tmp_path,pairs,updates=1)
    c["preference"]["package"]=None
    path=tmp_path/"needs_cli_package.yaml";path.write_text(yaml.safe_dump(c))
    project=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(project/"train_preference_idql.py"),"--config",str(path),
        "--data",str(native),"--preferences",str(pairs),"--dry-run"],capture_output=True,text=True,
        env={**os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"1"},timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads(result.stdout)
    assert report["preference"]["dataset_sha256"] and report["preference"]["enabled"]
    assert not Path(c["cache_dir"]).exists() and not Path(c["output_dir"]).exists()


def test_replaying_old_checkpoint_preserves_immutable_best_exports(inputs,tmp_path,monkeypatch):
    native,pairs=inputs;c=config(tmp_path,pairs,updates=3)
    c["critic"]["random_shift_pad"]=0
    original_evaluate=runner.evaluate_critic
    evaluations={"count":0}
    def force_improving_selection(*args,**kwargs):
        result=original_evaluate(*args,**kwargs)
        evaluations["count"]+=1
        result["spearman_v_remaining_blocks"]=-.1*evaluations["count"]
        return result
    monkeypatch.setattr(runner,"evaluate_critic",force_improving_selection)
    runner.train(c,native)
    directory=Path(c["output_dir"])
    from post_training.io import sha256_file
    originals={str(p.relative_to(directory)):sha256_file(p) for p in (directory/"critics").glob("*.pt")}
    assert len(originals)==3
    evaluations["count"]=1
    runner.train(c,native,resume=directory/"checkpoints/update_00000001.pt")
    assert all(sha256_file(directory/name)==checksum for name,checksum in originals.items())
    assert len(list((directory/"critics").glob("*.pt")))>len(originals)


def test_heldout_preference_evaluation_preserves_models_rng_and_loader_rng(inputs,tmp_path):
    from post_training.preference_critic import PreferenceIQL
    from post_training.preference_data import PreferenceDataset
    from torch.utils.data import DataLoader
    _,pairs=inputs;c=config(tmp_path,pairs)
    # Use package's own threshold split only for this pure held-out helper check.
    data=PreferenceDataset(pairs,split="val",validation_fraction=.5,sources=("current",))
    loader=DataLoader(data,batch_size=2,generator=torch.Generator().manual_seed(71))
    model=PreferenceIQL()
    states={k:v.clone() for k,v in model.state_dict().items()}
    rng=torch.get_rng_state().clone();loader_rng=loader.generator.get_state().clone()
    stats={"actions":{"q01":[-1.]*7,"q99":[1.]*7},"state":{"q01":[-1.]*8,"q99":[1.]*8}}
    metrics=runner.evaluate_preferences(model,loader,stats,c,torch.device("cpu"))
    assert metrics["samples"]==len(data) and "target_pref_min_accuracy" in metrics
    assert torch.equal(torch.get_rng_state(),rng) and torch.equal(loader.generator.get_state(),loader_rng)
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in states.items())


def test_tensorboard_and_failure_status_are_owned_by_the_new_runner(inputs,tmp_path,monkeypatch):
    native,pairs=inputs;c=config(tmp_path,pairs,updates=1)
    c["training"]["tensorboard"]=True
    result=runner.train(c,native)
    directory=Path(c["output_dir"])
    assert list((directory/"tensorboard").glob("events.out.tfevents.*"))
    assert json.loads((directory/"status.json").read_text())["status"]=="COMPLETED"
    c["output_dir"]=str(tmp_path/"failed_owned_run")
    def broken(*args,**kwargs):
        raise FloatingPointError("owned test nonfinite boundary")
    monkeypatch.setattr(runner,"optimization_step",broken)
    with pytest.raises(FloatingPointError,match="owned test"):
        runner.train(c,native)
    status=json.loads((Path(c["output_dir"])/"status.json").read_text())
    assert status["status"]=="FAILED" and status["policy_optimizer_updates"]==0
    assert not list((Path(c["output_dir"])/"checkpoints").glob("*.pt"))
