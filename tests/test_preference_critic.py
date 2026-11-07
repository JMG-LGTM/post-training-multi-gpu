"""CPU objective, gradient, disabled-control and export compatibility tests."""
import copy
import hashlib
import math

import pytest
import torch
from torch import nn

from post_training.critic import PixelIQL, default_input_spec, export_critic, load_critic
from post_training.preference_critic import PreferenceIQL, preference_ranking


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def native(size=3):
    rng = torch.Generator().manual_seed(35)
    return {"pixels":torch.randint(0,256,(size,6,16,16),dtype=torch.uint8,generator=rng),
            "next_pixels":torch.randint(0,256,(size,6,16,16),dtype=torch.uint8,generator=rng),
            "proprio":torch.randn(size,8,generator=rng),"next_proprio":torch.randn(size,8,generator=rng),
            "actions":torch.randn(size,2,7,generator=rng),"action_mask":torch.ones(size,2),
            "reward":torch.full((size,),-1.),"bootstrap_mask":torch.ones(size)}


def pairs(size=2):
    rng = torch.Generator().manual_seed(43)
    return {"pixels":torch.randint(0,256,(size,6,16,16),dtype=torch.uint8,generator=rng),
            "proprio":torch.randn(size,8,generator=rng),
            "chosen_actions":torch.ones(size,2,7),"rejected_actions":-torch.ones(size,2,7),
            "action_mask":torch.ones(size,2)}


def gradients(model):
    return {name:None if value.grad is None else value.grad.clone() for name,value in model.named_parameters()}


@pytest.mark.parametrize('stage',['value','q','both'])
@pytest.mark.parametrize('disabled',['zero','none'])
def test_disabled_exact_base_outputs_gradients_keys_and_rng(stage,disabled):
    plain=PixelIQL(horizon=2);extended=PreferenceIQL(horizon=2)
    extended.load_state_dict(plain.state_dict());data=native()
    before=torch.get_rng_state().clone()
    expected=plain(data,stage=stage)
    actual=extended(data,stage=stage,preference_batch=object() if disabled=='zero' else None,
        preference_weight=0 if disabled=='zero' else .1,preference_temperature=float('nan'))
    assert expected.keys()==actual.keys()
    for key in expected:assert torch.equal(expected[key],actual[key])
    assert torch.equal(torch.get_rng_state(),before)
    loss_keys=[key for key in ['q_loss','v_loss'] if key in expected]
    sum(expected[key].mean() for key in loss_keys).backward()
    sum(actual[key].mean() for key in loss_keys).backward()
    eg,ag=gradients(plain),gradients(extended)
    for key in eg:
        assert (eg[key] is None)==(ag[key] is None)
        if eg[key] is not None:assert torch.equal(eg[key],ag[key])


class LinearActionQ(nn.Module):
    def __init__(self,weight):
        super().__init__();self.weight=nn.Parameter(torch.tensor(float(weight)))

    def forward(self,pixels,proprio,actions,mask):
        return self.weight*(actions*mask[...,None]).sum((1,2))


def test_exact_headwise_bt_direction_both_heads_and_batch_size_independence():
    model=PreferenceIQL(horizon=2);model.q1=LinearActionQ(0.);model.q2=LinearActionQ(0.)
    data,pref=native(3),pairs(2)
    result=model(data,stage='q',preference_batch=pref,preference_weight=.2,preference_temperature=2.)
    assert result['q_loss'].shape==(3,) and result['pref_loss'].shape==(2,)
    torch.testing.assert_close(result['pref_loss'],torch.full((2,),math.log(2)))
    torch.testing.assert_close(result['q_loss']-result['q_td_loss'],torch.full((3,),.2*math.log(2)))
    result['pref_loss'].mean().backward()
    assert model.q1.weight.grad<0 and model.q2.weight.grad<0
    assert model.q1.weight.grad==model.q2.weight.grad
    first=gradients(model)
    model.zero_grad(set_to_none=True)
    duplicate={key:value.repeat_interleave(3,0) for key,value in pref.items()}
    duplicated=model(data,stage='q',preference_batch=duplicate,preference_weight=.2,preference_temperature=2.)
    duplicated['pref_loss'].mean().backward()
    for key,value in first.items():
        if value is not None:torch.testing.assert_close(gradients(model)[key],value)


def test_individual_heads_ranked_without_min_bottleneck():
    model=PreferenceIQL(horizon=2);model.q1=LinearActionQ(.03);model.q2=LinearActionQ(-.02)
    pref=pairs();result=model(native(),stage='q',preference_batch=pref,preference_temperature=1.7)
    margin1,margin2=torch.full((2,),.03*28),torch.full((2,),-.02*28)
    expected=.5*(torch.nn.functional.softplus(-margin1/1.7)+torch.nn.functional.softplus(-margin2/1.7))
    torch.testing.assert_close(result['pref_loss'],expected)
    torch.testing.assert_close(result['pref_accuracy'],torch.full((2,),.5))
    torch.testing.assert_close(result['pref_margin'],.5*(margin1+margin2))
    assert not result['pref_margin'].requires_grad and not result['pref_accuracy'].requires_grad


def test_preference_q_stage_only_updates_online_q_targets_and_value_unchanged():
    model=PreferenceIQL(horizon=2);result=model(native(),stage='q',preference_batch=pairs())
    result['q_loss'].mean().backward()
    for network in [model.q1,model.q2]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())
        assert any(torch.count_nonzero(p.grad) for p in network.parameters())
    assert all(p.grad is None for p in model.value.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.target_q1.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.target_q2.parameters())
    assert not any('actor' in key for key in model.state_dict())


def test_value_stage_remains_original_expectile_and_ignores_preferences():
    model=PreferenceIQL(horizon=2);result=model(native(),stage='value',preference_batch=object(),
        preference_weight=-10,preference_temperature=-1)
    assert set(result)=={'v_loss','v','target_q'}
    result['v_loss'].mean().backward()
    assert all(p.grad is not None for p in model.value.parameters())
    assert all(p.grad is None for p in model.q1.parameters())


def test_shared_mask_masks_both_chunks_and_identical_pair_gradient_zero():
    model=PreferenceIQL(horizon=2);pref=pairs()
    pref['rejected_actions']=pref['chosen_actions'].clone();pref['action_mask'][:,1]=0
    pref['rejected_actions'][:,1]=999
    result=model(native(),stage='q',preference_batch=pref)
    torch.testing.assert_close(result['pref_loss'],torch.full((2,),math.log(2)))
    result['pref_loss'].sum().backward()
    assert all(torch.count_nonzero(p.grad)==0 for network in [model.q1,model.q2] for p in network.parameters())


@pytest.mark.parametrize('key,bad',[
    ('chosen_actions',torch.ones(2,3,7)),('chosen_actions',torch.full((2,2,7),float('nan'))),
    ('rejected_actions',torch.full((2,2,7),float('inf'))),('proprio',torch.full((2,8),float('nan'))),
    ('proprio',torch.ones(2,8,dtype=torch.int64)),('action_mask',torch.zeros(2,2)),
    ('action_mask',torch.full((2,2),.5)),('action_mask',torch.ones(2,1)),
    ('pixels',torch.full((2,6,16,16),256.)),('pixels',torch.ones(2,3,16,16)),
    ('pixels',torch.zeros(0,6,16,16,dtype=torch.uint8)),('pixels',torch.full((2,6,16,16),float('nan')))])
def test_malformed_pairs_fail_closed(key,bad):
    model=PreferenceIQL(horizon=2);pref=pairs();pref[key]=bad
    with pytest.raises(ValueError):model(native(),stage='q',preference_batch=pref)


@pytest.mark.parametrize('kwargs',[{'preference_weight':-1},{'preference_weight':float('nan')},
    {'preference_weight':True},{'preference_temperature':0},{'preference_temperature':float('inf')},
    {'preference_temperature':True}])
def test_invalid_objective_coefficients_fail(kwargs):
    with pytest.raises(ValueError):PreferenceIQL(horizon=2)(native(),stage='q',preference_batch=pairs(),**kwargs)


def test_nonfinite_online_q_fails_even_with_finite_inputs():
    model=PreferenceIQL(horizon=2)
    with torch.no_grad():next(model.q1.parameters()).fill_(float('nan'))
    with pytest.raises(ValueError,match='nonfinite'):model(native(),stage='q',preference_batch=pairs())


def test_existing_export_load_and_new_export_are_plain_pixel_iql_compatible(tmp_path):
    plain=PixelIQL(horizon=2);model=PreferenceIQL(horizon=2)
    assert model.model_config==plain.model_config
    assert model.state_dict().keys()==plain.state_dict().keys()
    metadata={'input_spec':default_input_spec(plain,image_size=16),
        'normalization':{'kind':'openpi_quantile','actions':{'q01':[-1.]*7,'q99':[1.]*7},
                         'proprio':{'q01':[-1.]*8,'q99':[1.]*8}},
        'reward_definition':{'kind':'sparse','block_reward':-1.,'success_reward':0.,
                             'gamma':.99,'discount_per':'block','normalized':False},
        'termination_rule':'success_aware_recording_end_v1'}
    source=tmp_path/'baseline.pt';export_critic(source,plain,**metadata)
    original_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    load_critic(source,model=model,**{'expected_'+key:value for key,value in metadata.items()})
    baseline=copy.deepcopy(plain.state_dict())
    data=native();expected=plain.scores(data)
    actual=model.scores(data)
    for key in expected:assert torch.equal(actual[key],expected[key])
    optimizer=torch.optim.SGD([*model.q1.parameters(),*model.q2.parameters()],lr=.001)
    model(data,stage='q',preference_batch=pairs())['q_loss'].mean().backward();optimizer.step()
    output=tmp_path/'preference.pt';export_critic(output,model,**metadata,
        provenance={'algorithm':'preference_regularized_idql','actor_updated':False})
    inference,saved=load_critic(output)
    assert type(inference) is PixelIQL and saved['model_config']==plain.model_config
    for key in model.scores(data):assert torch.equal(model.scores(data)[key],inference.scores(data)[key])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==original_sha
    for key,value in plain.state_dict().items():assert torch.equal(value,baseline[key])


def test_ranking_helper_matches_training_and_uses_no_td_or_rng():
    model=PreferenceIQL(horizon=2);pref=pairs()
    model.train();model.target_q1.eval()
    modes={name:module.training for name,module in model.named_modules()}
    before=torch.get_rng_state().clone()
    direct=preference_ranking(model,pref,temperature=1.4)
    forward=model(native(),stage='q',preference_batch=pref,preference_temperature=1.4)
    for key,value in direct.items():torch.testing.assert_close(value,forward[key])
    assert torch.equal(torch.get_rng_state(),before)
    assert modes=={name:module.training for name,module in model.named_modules()}
    assert not any(key in direct for key in ['q_loss','v_loss','target','weights'])
    assert direct['pref_loss'].requires_grad


def test_target_ranking_is_frozen_even_when_pair_inputs_require_grad():
    model=PreferenceIQL(horizon=2);pref=pairs()
    pref['chosen_actions'].requires_grad_(True)
    pref['rejected_actions'].requires_grad_(True)
    before={name:value.detach().clone() for name,value in model.state_dict().items()}
    with torch.no_grad():
        model.q1.head[-1].weight.add_(.2)
    target=preference_ranking(model,pref,target=True)
    plain=PixelIQL(horizon=2);plain.load_state_dict(before)
    expected=preference_ranking(plain,pref,target=True)
    for key,value in target.items():
        torch.testing.assert_close(value,expected[key]);assert not value.requires_grad
    assert all(p.grad is None for p in model.parameters())
    assert pref['chosen_actions'].grad is pref['rejected_actions'].grad is None


@pytest.mark.parametrize('five_dimensional',[False,True])
def test_generic_constructor_dimensions_and_pair_count_are_not_fixed(five_dimensional):
    model=PreferenceIQL(horizon=5,action_dim=4,proprio_dim=3,pixel_channels=3)
    data={'pixels':torch.zeros(7,3,16,16,dtype=torch.uint8),
          'proprio':torch.zeros(7,3),'chosen_actions':torch.ones(7,5,4),
          'rejected_actions':-torch.ones(7,5,4),'action_mask':torch.ones(7,5)}
    if five_dimensional:data['pixels']=data['pixels'].reshape(7,1,3,16,16)
    result=preference_ranking(model,data)
    assert all(value.shape==(7,) for value in result.values())
    assert all(torch.isfinite(value).all() for value in result.values())
    result['pref_loss'].mean().backward()
    assert all(p.grad is not None for p in model.q1.parameters())


def test_ranking_helper_rejects_non_boolean_target_switch():
    with pytest.raises(ValueError,match='boolean'):
        preference_ranking(PreferenceIQL(horizon=2),pairs(),target='online')
