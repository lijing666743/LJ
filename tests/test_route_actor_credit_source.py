"""P1a tensor/credit/optimizer gates; never create or roll an environment."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from src.config import ConfigError, load_run_config
from src.models.ca_gat_mappo import CAGATMAPPOActor, MAPPOCentralizedCritic
from src.models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution, SequentialActionMaskBatch
from src.models.ca_gat_mappo_checkpoint import (load_checkpoint_payload, restore_active_rollout,
    atomic_save_checkpoint)
from src.models.ca_gat_mappo_rollout import CAGATMAPPORolloutBuffer, CAGATMAPPORolloutChunk
from src.models.ca_gat_mappo_update import CAGATMAPPORecurrentPPOUpdater
from src.models.route_option_q import prepare_route_q, route_q_loss, epoch_q_records, write_route_q_telemetry
from src.models.route_task_credit import (KEY, prepare_task_credit, task_value_loss,
    summarize_route_credit_components, write_task_credit_telemetry)
from src.evaluation.route_credit_probe import credit_gradient_diagnostics
from tests.test_route_option_q import tiny_batch
from tests.test_route_task_credit import state, decision, step, terminal, credit
from tests.test_ca_gat_mappo_checkpoint import make_payload

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT/'tests/fixtures/configs'
P0 = CONFIGS/'e14d-p0-normalized-seed42-8k-dev.json'
P1 = CONFIGS/'e14e-p1a-task-event-credit-seed42-8k-dev.json'


@pytest.fixture(autouse=True)
def no_environment(monkeypatch):
    from src.env.environment import U2UMECEnvironment
    def forbidden(*args, **kwargs):
        raise AssertionError('P1a gates forbid environment construction and rollout')
    monkeypatch.setattr(U2UMECEnvironment, '__init__', forbidden)
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def config(path=P1):
    return load_run_config(path, cli_overrides={'training.mappo.training_device': 'cpu'})


def switched(source='task_event', beta=1.):
    critic, policy, task, old = tiny_batch(beta=beta, q_values=[0., 2., 1., 4., 0., 0.])
    batch = prepare_route_q(critic, policy, torch.zeros(1, 1, 2), task, beta,
                            actor_credit_source=source)
    return critic, policy, task, old, batch


def test_counterfactual_default_is_bitwise_previous_formula():
    _, _, _, old, explicit = switched('counterfactual_q', .7)
    for name in ('advantage', 'frozen_q', 'baseline', 'counterfactual_advantage', 'targets'):
        assert torch.equal(getattr(old, name), getattr(explicit, name))
    for n, sample in enumerate(explicit.records):
        expected = torch.tensor(.4)+.7*explicit.counterfactual_advantage[n]
        assert torch.equal(explicit.advantage[0, n], expected)


@pytest.mark.parametrize('beta', [0., 1., 99.])
def test_task_event_reuses_e14c_exact_tensor_and_ignores_cf(beta):
    _, _, task, _, batch = switched(beta=beta)
    assert torch.equal(batch.advantage, task.advantage)
    assert all(s['effective_actor_counterfactual_coefficient'] == 0 for s in batch.records)
    assert all(s['coefficient'] == 1 for s in batch.credit_samples)
    assert any(s['counterfactual_advantage'] != 0 for s in batch.records)
    assert all(s['final_route_advantage'] == s['combined_advantage'] for s in batch.records)


def test_q_inputs_targets_and_loss_identical_between_sources():
    critic, policy, task, old, batch = switched()
    for name in ('state', 'candidates', 'legal', 'chosen', 'targets', 'frozen_q', 'baseline', 'counterfactual_advantage'):
        assert torch.equal(getattr(old, name), getattr(batch, name))
    assert torch.equal(route_q_loss(critic, old)[0], route_q_loss(critic, batch)[0])
    critic.route_option_q.forward = lambda s,c,m: torch.full_like(m, 100., dtype=torch.float32)
    again = prepare_route_q(critic, policy, torch.zeros(1,1,2), task, 1., actor_credit_source='task_event')
    assert torch.equal(again.advantage, batch.advantage)


@pytest.mark.parametrize('action,status,payoff', [
    ('local','completed',1.), ('local','expired',-1.),
    ('remote','completed',1.), ('remote','expired',-1.)])
def test_original_binding_receives_terminal_event_credit(action,status,payoff):
    d=decision(action,task=19,source=0);d['formal_route_event_key']=[0,19,0]
    rows=[step(before=[state(19,value=.2)],after=[state(19,status='local' if action=='local' else 'tx')],decisions=[d]),
          step(1,before=[state(19,status='cpu')],after=[state(19,status='cpu')]),
          step(2,before=[state(19,status='cpu')],terminals=[terminal(status,19,slot=2,payoff=payoff)])]
    batch=credit(rows, shared=torch.full((3,4),.3))
    s=batch.samples[0]
    assert s['task_id']==19 and s['formal_route_event_key']==[0,19,0]
    assert s['return']==pytest.approx(.9**2*payoff)
    assert s['route_specific_advantage']==pytest.approx(.9**2*payoff-.2)
    assert (s['route_specific_advantage']>0)==(payoff>0)
    assert s['combined_advantage']==pytest.approx(.3+.9**2*payoff-.2)


def test_repeated_defer_later_bind_keeps_binding_value_not_done_payoff():
    rows=[step(i,before=[state()],after=[state()],decisions=[decision('defer',slot=i)]) for i in range(2)]
    rows += [step(2,before=[state()],after=[state(value=.4,status='tx')],decisions=[decision('remote',slot=2)]),
             step(3,before=[state(status='cpu')],terminals=[terminal(slot=3)])]
    batch=credit(rows)
    assert [s['return'] for s in batch.samples]==pytest.approx([.9**3*.4,.9**2*.4,.9])
    assert [s['credit_stop'] for s in batch.samples]==['binding_value_bootstrap']*2+['completed']
    assert len({(s['episode_id'],s['task_id'],s['route_slot']) for s in batch.samples})==3
    rows[-1]['terminals'][0]['payoff']=7.
    altered=credit(rows)
    assert [s['return'] for s in altered.samples[:2]]==[s['return'] for s in batch.samples[:2]]


def test_repeated_defer_unbound_expiry_negative_without_duplicate_terminal():
    rows=[step(i,before=[state()],after=[state()] if i<2 else [],decisions=[decision('defer',slot=i)],
               terminals=[terminal('expired',slot=2,payoff=-1.)] if i==2 else []) for i in range(3)]
    batch=credit(rows)
    assert [s['return'] for s in batch.samples]==pytest.approx([-.81,-.9,-1.])
    assert [s['credit_stop'] for s in batch.samples]==['expired']*3
    assert sum(len(r['terminals']) for r in rows)==1
    rows[-1]['terminals']*=2
    with pytest.raises(ValueError,match='duplicate task terminal'):credit(rows)


@pytest.mark.parametrize('kind,payoff', [('expired',-1.),('truncated',0.)])
def test_binding_same_slot_terminal_and_truncation_unchanged(kind,payoff):
    rows=[step(before=[state()],after=[state()],decisions=[decision('defer')]),
          step(1,before=[state()],decisions=[decision('remote',slot=1)],terminals=[terminal(kind,slot=1,payoff=payoff)])]
    b=credit(rows)
    assert [s['return'] for s in b.samples]==pytest.approx([.9*payoff,payoff])
    assert b.samples[0]['credit_stop']=='binding_same_slot_terminal'
    assert b.samples[0]['censored']==(kind=='truncated')


def test_frontier_episode_and_policy_version_boundaries():
    b=credit([step(7,before=[state(value=.1)],after=[state(value=.7)],decisions=[decision('defer',slot=7)])])
    assert b.samples[0]['return']==pytest.approx(.9*.7)
    assert b.samples[0]['credit_stop']=='rollout_bootstrap'
    rows=[step(before=[state()],decisions=[decision()],terminals=[terminal(slot=0)]),
          step(episode=1,before=[state()],decisions=[decision('defer',episode=1)],terminals=[terminal('expired',slot=0,payoff=-1.)])]
    assert [s['return'] for s in credit(rows).samples]==[1.,-1.]
    rows[-1]['policy_version']=1
    with pytest.raises(ValueError,match='policy-version'):credit(rows)


def test_configuration_diff_and_all_historical_hashes_preserved():
    a,b=load_run_config(P0),load_run_config(P1)
    raw=b.resolved_dict();raw['training']['mappo'].pop('route_actor_credit_source');raw['output']=a.resolved_dict()['output']
    assert raw==a.resolved_dict()
    assert b.training.mappo.route_task_credit_coefficient==1.
    assert b.training.mappo.route_q_input_mode=='normalized'
    assert a.config_hash=='c155c181a5bbca4d8af553694f5d5e2eaa910ff67e472ae14390c5045cacbb61'
    assert load_run_config(CONFIGS/'e14d-v1-route-counterfactual-q-seed42-8k-dev.json').config_hash.startswith('2cd92efd656e')
    for source in ['hybrid','off',None]:
        bad=replace(a,training=replace(a.training,mappo=replace(a.training.mappo,route_actor_credit_source=source)))
        with pytest.raises(ConfigError):bad.validate()
    with pytest.raises(ConfigError):
        replace(a,training=replace(a.training,mappo=replace(a.training.mappo,
            route_actor_credit_source='task_event',route_value_mode='off',route_q_input_mode='legacy'))).validate()


def test_credit_telemetry_four_components_conserve(tmp_path):
    _,_,_,_,batch=switched()
    rows=epoch_q_records(batch,batch.frozen_q,[{**s,'epoch_index':0,'ppo_ratio':1.,'surrogate':s['combined_advantage']} for s in batch.records])
    groups=summarize_route_credit_components(rows)
    for cat in ['local','remote','defer']:
        for stats in groups['epoch0/'+cat].values():
            assert stats['count']==1 and stats['std']==0
            assert stats['positive_count']+stats['negative_count']+stats['zero_count']==1
            assert stats['rms']==abs(stats['mean'])
    cfg=config();cfg=replace(cfg,output=replace(cfg.output,logs_dir=str(tmp_path)))
    buffer=SimpleNamespace(transition_at=lambda i:SimpleNamespace(rejection_or_downgrade_summary={KEY:step()}),__len__=lambda:1)
    class Buffer:
        def __len__(self):return 1
        def transition_at(self,i):return buffer.transition_at(i)
    qrow=write_route_q_telemetry(cfg,rows,1,[.5],buffer=Buffer())
    trow=write_task_credit_telemetry(cfg,rows,1,[.5])
    assert qrow['effective_actor_counterfactual_coefficient']==0
    assert trow['effective_actor_task_credit_coefficient']==1
    assert qrow['credit_components_by_epoch']==trow['credit_components_by_epoch']
    assert qrow['q_loss_coefficient']==.5 and len(qrow['samples'])==3
    json.dumps(qrow,allow_nan=False)


def test_gradient_probe_is_readonly_and_actual_gradients_not_rms():
    p=torch.nn.Parameter(torch.tensor([.1,-.2,.3]))
    new=torch.log_softmax(p,0);old=new.detach();snapshot=p.detach().clone()
    out=credit_gradient_diagnostics(new,old,torch.tensor([1.,2.,-1.]),torch.tensor([2.,-1.,1.]),torch.tensor([.01,-.01,0.]),[p])
    assert out['combined_minus_shared_task_norm']<1e-6
    assert out['task']['gradient_norm']>0 and out['CF']['gradient_norm']>0
    assert p.grad is None and torch.equal(p,snapshot) and out['optimizer_step'] is False


def test_actor_route_loss_cannot_backprop_to_q_or_other_branches():
    _,_,task,_,batch=switched()
    p=torch.nn.Parameter(torch.tensor([.1,.2,-.2]));log=torch.log_softmax(p,0)
    loss=-(log*batch.advantage[0,:3]).mean();loss.backward()
    assert p.grad is not None and all(not t.requires_grad for t in [batch.advantage,batch.frozen_q,batch.counterfactual_advantage])
    from tests.test_ca_gat_mappo_route_specific_gae import ppo_inputs,low_level_ppo
    inputs=ppo_inputs();a=low_level_ppo(inputs,route_advantage=inputs['advantage'])
    b=low_level_ppo(inputs,route_advantage=inputs['advantage']+1.)
    for k in ['branch_advantage','branch_surrogate','branch_unclipped_surrogate','branch_clipped_surrogate']:
        assert torch.equal(getattr(a,k)[...,1:],getattr(b,k)[...,1:])


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_auxiliary_q_and_task_value_optimizer_updates_only_own_parameters(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    cfg=config();actor=CAGATMAPPOActor(cfg).to(device);critic=MAPPOCentralizedCritic(cfg).to(device)
    updater=CAGATMAPPORecurrentPPOUpdater(actor,critic,cfg)
    before={k:v.clone() for k,v in critic.state_dict().items()};actor_before={k:v.clone() for k,v in actor.state_dict().items()}
    s=torch.zeros(3,critic.spec.centralized_state_dim,device=device,requires_grad=True)
    c=torch.randn(3,6,411,device=device,requires_grad=True);mask=torch.ones(3,6,dtype=torch.bool,device=device);mask[:,0]=False
    batch=SimpleNamespace(state=s,candidates=c,legal=mask,chosen=torch.tensor([1,2,3],device=device),targets=torch.tensor([1.,-.5,-1.],device=device))
    task=SimpleNamespace(state_indices=torch.arange(3),task_features=torch.zeros(3,18),value_targets=torch.tensor([1.,-.5,-1.]))
    updater.optimizers.critic_optimizer.zero_grad(set_to_none=True)
    loss=.5*route_q_loss(critic,batch)[0]+.5*task_value_loss(critic,s,task);loss.backward()
    assert s.grad is None and c.grad is None
    assert all(p.grad is None for p in actor.parameters())
    assert all(p.grad is None for p in critic.value_network.parameters())
    updater.optimizers.critic_optimizer.step()  # disposable unit model, no RL/environment
    for prefix in ['route_option_q.','route_task_value.']:
        assert any(not torch.equal(before[k],v) for k,v in critic.state_dict().items() if k.startswith(prefix))
    assert all(torch.equal(v,actor.state_dict()[k]) for k,v in actor_before.items())
    assert all(torch.equal(v,critic.state_dict()[k]) for k,v in before.items() if k.startswith('value_network.'))


def test_old_real_checkpoints_strict_load_unchanged():
    for sub,cfgpath in [('e14d_route_counterfactual_q_seed42_8k',CONFIGS/'e14d-v1-route-counterfactual-q-seed42-8k-dev.json'),
                        ('e14d_p0_normalized_seed42_8k',P0)]:
        cp=(ROOT/'tests/fixtures/historical/checkpoints'/sub/'step_4000.pt')
        sha=hashlib.sha256(cp.read_bytes()).hexdigest();payload=load_checkpoint_payload(cp);cfg=load_run_config(cfgpath)
        assert payload['config_hash']==cfg.config_hash
        CAGATMAPPOActor(cfg).load_state_dict(payload['model_state']['actor'],strict=True)
        MAPPOCentralizedCritic(cfg).load_state_dict(payload['model_state']['critic'],strict=True)
        assert hashlib.sha256(cp.read_bytes()).hexdigest()==sha


@pytest.mark.parametrize('heads,energy,critical',[
    ([],True,False), ([(0,40e6,4)],True,False),
    ([(0,40e6,2)],True,True), ([(0,40e6+1,2)],True,False),
    ([(0,40e6,2)],False,False), ([(0,40e6,8),(3,40e6,2)],True,True),
    ([(0,20e6,1)],True,True)])
def test_cpu_gate_and_actor_initialization_unchanged(heads,energy,critical):
    from src.models.integrated_scheduler import cpu_joint_statistics
    cp=(ROOT/'tests/fixtures/historical/checkpoints/e14d_p0_normalized_seed42_8k/step_4000.pt')
    payload=load_checkpoint_payload(cp)
    buf,_=restore_active_rollout(load_run_config(P0),payload['active_rollout_state'],
        checkpoint_kind=payload['checkpoint_kind'],policy_version=15)
    chunk=CAGATMAPPORolloutChunk((buf.transition_at(0),));batch=chunk.actor_batch
    contracts=[]
    for c in chunk.action_mask_batch.flattened():
        mask=np.array([True,False,False,False,False]);context=[(0.,0,1e9)]*4
        for source,cycles,slack in heads:mask[source+1]=True;context[source]=(cycles,slack,1e9)
        contracts.append(replace(c,cpu_queue_mask=mask,cpu_queue_branch_active=bool(heads),
            cpu_frequency_energy_mask=np.tile([True,energy,energy,energy],(4,1)),cpu_deadline_context=tuple(context)))
    a,b=CAGATMAPPOActor(config(P0)),CAGATMAPPOActor(config())
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    with torch.no_grad():
        outputs=[cpu_joint_statistics(actor,actor(batch,chunk.initial_hidden).recurrent_features,batch,
            tuple(contracts),None,'deterministic',None) for actor in (a,b)]
    assert torch.equal(outputs[0]['mask'],outputs[1]['mask'])
    assert bool(outputs[1]['mask'][...,0].all()) is (not critical)
    for name in ['queue_index','frequency_index']:
        assert torch.equal(outputs[0][name],outputs[1][name])


def test_new_checkpoint_roundtrip_preserves_selector_normalization_and_adam(tmp_path):
    cfg=config();actor=CAGATMAPPOActor(cfg);critic=MAPPOCentralizedCritic(cfg)
    updater=CAGATMAPPORecurrentPPOUpdater(actor,critic,cfg)
    updater.optimizers.critic_optimizer.zero_grad(set_to_none=True)
    sum(p.square().sum() for p in critic.parameters()).backward()
    updater.optimizers.critic_optimizer.step()  # disposable serialization fixture
    payload,actor,critic,updater=make_payload(cfg,'FINAL_COMPLETED',actor=actor,critic=critic,updater=updater)
    # Serialization fixture only; no claim this untrained model is an experiment result.
    path=tmp_path/'synthetic_checkpoint_fixture.pt';atomic_save_checkpoint(payload,path);loaded=load_checkpoint_payload(path)
    assert loaded['config_snapshot']['training']['mappo']['route_actor_credit_source']=='task_event'
    clone=MAPPOCentralizedCritic(cfg);clone.load_state_dict(loaded['model_state']['critic'],strict=True)
    for k,v in critic.state_dict().items():assert torch.equal(v,clone.state_dict()[k])
    new=CAGATMAPPORecurrentPPOUpdater(CAGATMAPPOActor(cfg),clone,cfg)
    new.optimizers.critic_optimizer.load_state_dict(loaded['optimizer_state']['critic_adam'])
    assert new.optimizers.critic_optimizer.state
    for left,right in zip(updater.optimizers.critic_optimizer.state.values(),new.optimizers.critic_optimizer.state.values()):
        for key in left:assert torch.equal(left[key],right[key])
    assert torch.equal(clone.route_option_q.conditioner.gain,critic.route_option_q.conditioner.gain)


def tensor_optimizer_buffer(cfg):
    """Saved observation tensors + artificial same-slot terminal fixtures."""
    oldcfg=load_run_config(P0)
    cp=(ROOT/'tests/fixtures/historical/checkpoints/e14d_p0_normalized_seed42_8k/step_4000.pt')
    payload=load_checkpoint_payload(cp)
    old,_=restore_active_rollout(oldcfg,payload['active_rollout_state'],checkpoint_kind=payload['checkpoint_kind'],policy_version=15)
    seed=next(old.transition_at(i) for i in range(len(old)) if old.transition_at(i).rejection_or_downgrade_summary[KEY]['decisions']
              and not old.transition_at(i).rejection_or_downgrade_summary['tx_paired_shadow']['samples'])
    actor=CAGATMAPPOActor(cfg);distribution=CAGATMAPPOActionDistribution(actor,cfg)
    chunk=CAGATMAPPORolloutChunk((seed,))
    with torch.no_grad():policy=distribution.evaluate_actions(chunk.actor_batch,chunk.action_mask_batch,chunk.proposals,chunk.initial_hidden)
    buffer=CAGATMAPPORolloutBuffer(cfg)
    for i in range(256):
        md=copy.deepcopy(seed.rejection_or_downgrade_summary);before=[];ds=[];ts=[]
        md['tx_paired_shadow']['rollout_offset']=i
        for original in md[KEY]['decisions']:
            u=original['source_uav'];task=i*4+u;d={**original,'task_id':task,'global_rollout_index':i,'slot':i+1,'episode_id':0,'policy_version':0,
                'actor_leaf_probability':policy.probabilities['route'][0,0,u].tolist(),'route_legal_mask':policy.action_masks['route'][0,0,u].tolist()}
            rm=policy.action_masks['route'][0,0,u,3:];rl=policy.raw_logits['route'][0,0,u,3:]
            d['actor_remote_conditional_probability']=(rl.masked_fill(~rm,-torch.inf).softmax(-1) if rm.any() else torch.zeros_like(rl)).tolist()
            before.append(state(task,u));ds.append(d);isdefer=d['category']=='defer'
            ts.append(terminal('expired' if isdefer else 'completed',task,u,i+1,-1. if isdefer else 1.))
        md[KEY]=step(i+1,before=before,decisions=ds,terminals=ts)
        # No shadow execution: pre-existing paired snapshots, if any, are just fixture labels.
        buffer.append(replace(seed,slot=i+1,episode_start=False,rejection_or_downgrade_summary=md))
    buffer.finalize();return buffer


def test_production_four_epochs_source_switch_preserves_auxiliary_losses_and_frozen_credit():
    cfg=config();buffer=tensor_optimizer_buffer(cfg);actor=CAGATMAPPOActor(cfg);critic=MAPPOCentralizedCritic(cfg)
    before={k:v.clone() for k,v in critic.state_dict().items()};updater=CAGATMAPPORecurrentPPOUpdater(actor,critic,cfg)
    result=updater.update(buffer,collected_environment_steps=256)
    assert len(result.epoch_diagnostics)==4
    assert len(updater.last_route_option_q_losses)==len(updater.last_route_task_value_losses)==4
    grouped={}
    for s in updater.last_route_option_q_records:
        assert s['final_route_advantage']==pytest.approx(s['shared_advantage']+s['task_advantage'],abs=2e-6)
        assert s['effective_actor_counterfactual_coefficient']==0
        grouped.setdefault((s['global_rollout_index'],s['source_uav']),[]).append(s)
    for rr in grouped.values():
        assert len(rr)==4
        for k in ['G_event','final_route_advantage','frozen_q','counterfactual_advantage']:
            assert all(s[k]==rr[0][k] for s in rr)
    for prefix in ['route_option_q.','route_task_value.']:
        assert any(not torch.equal(before[k],v) for k,v in critic.state_dict().items() if k.startswith(prefix))
    assert all(torch.isfinite(p).all() for p in actor.parameters())
    # Shared critic/task-value/Q fitting depend on fixed buffer, not actor source.
    other=replace(cfg,training=replace(cfg.training,mappo=replace(cfg.training.mappo,route_actor_credit_source='counterfactual_q')))
    a2,c2=CAGATMAPPOActor(other),MAPPOCentralizedCritic(other);u2=CAGATMAPPORecurrentPPOUpdater(a2,c2,other)
    u2.update(buffer,collected_environment_steps=256)
    for k,v in critic.state_dict().items():assert torch.equal(v,c2.state_dict()[k]),k
    assert updater.last_route_option_q_losses==u2.last_route_option_q_losses
    assert updater.last_route_task_value_losses==u2.last_route_task_value_losses
    # Non-Route advantages/surrogates are checked separately. Existing global
    # actor clipping couples update magnitudes; final TX/CPU weights need not
    # be bitwise equal after changing the Route gradient.

