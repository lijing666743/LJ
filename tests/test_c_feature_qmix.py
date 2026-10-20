"""Engineering fixtures and independent Q/credit oracles, never paper results."""
import copy
import itertools
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from src.config import load_run_config
from src.env.actions import ActionProposal
from src.env.environment import U2UMECEnvironment
from src.models.ca_gat_mappo import ActorObservationTensorizer,CAGATMAPPOActor,stack_actor_time
from src.models.c_feature_qmix_agent import CFeatureQNetwork
from src.models.c_feature_qmix_credit import (CREDIT_CONTRACT,collect_credit_record,
    credit_return_targets,task_credit_labels,auxiliary_losses)
from src.models.qmix_continuation import state_digest,load_continuation_checkpoint
from src.models.qmix_replay import SequenceReplay
from src.models.qmix_recurrent import reconstruct_replay_sequences
from src.models.qmix_trainer import QMIXTrainer,checkpoint_bytes,load_checkpoint
from src.evaluation.qmix_checkpoint_progress import QMIXCheckpointProgressRunner
from src.models.route_physical_relations import RoutePhysicalRelationExtractor


ROOT=Path(__file__).resolve().parents[1]
PROFILE=ROOT/'experiments/strong-baselines/configs/c-feature-credit-qmix-engineering.json'


@pytest.fixture(autouse=True,scope='module')
def single_thread():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def config(tmp_path):
    return load_run_config(PROFILE,cli_overrides={'training.mappo.training_device':'cpu',
        'evaluation.evaluation_seeds':[1042],'output.logs_dir':str(tmp_path/'logs'),
        'output.plots_dir':str(tmp_path/'plots'),'output.dashboard_logs_dir':str(tmp_path/'dash')})


@pytest.fixture
def sample(config):
    network=CFeatureQNetwork(config)
    observations=U2UMECEnvironment(config).reset().observations
    batch=ActorObservationTensorizer(config).encode_step(observations)
    route_mask=np.ones_like(observations[0].action_masks.route_mask);route_mask[0]=False
    masks=tuple(replace(obs.action_masks,route_branch_active=True,route_mask=route_mask,
        tx_branch_active=True,tx_select_mask=np.ones_like(obs.action_masks.tx_select_mask),
        cpu_queue_branch_active=True,cpu_queue_mask=np.ones_like(obs.action_masks.cpu_queue_mask),
        cpu_frequency_energy_mask=np.ones_like(obs.action_masks.cpu_frequency_energy_mask),
        power_energy_mask=np.ones_like(obs.action_masks.power_energy_mask),
        cpu_deadline_context=tuple((1.e6,100,2.e9) for _ in range(4))) for obs in observations)
    batch=replace(batch,action_masks={key:torch.ones_like(value) for key,value in batch.action_masks.items()})
    batch.action_masks['route'][:,:,:,0]=False
    batch.action_indices['route'].fill_(1)
    return network,batch,((masks,),)


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    root=tmp_path_factory.mktemp('c_feature_training')
    config=load_run_config(PROFILE,cli_overrides={'training.mappo.training_device':'cpu',
        'evaluation.evaluation_seeds':[1042],'output.logs_dir':str(root/'l'),
        'output.plots_dir':str(root/'p'),'output.dashboard_logs_dir':str(root/'d')})
    trainer=QMIXTrainer(config);trainer.train(stop_after_episodes=2)
    return trainer


def test_initial_actor_weights_and_modules_exactly_match_C(config):
    c=CAGATMAPPOActor(replace(config,method_id='ca_gat_mappo'))
    q=CFeatureQNetwork(config)
    assert state_digest(c.state_dict())==state_digest(q.backbone.state_dict())
    assert q.spec.recurrent_state_dimension==256
    assert sum(p.numel() for p in c.parameters())+129==sum(p.numel() for p in q.parameters())
    assert all(torch.count_nonzero(p)==0 for p in q.value_head.parameters())
    assert q.backbone.route_decoder.physical_projection[0].in_features==35
    assert hasattr(q.backbone,'tx_upper_private') and hasattr(q.backbone,'cpu_joint_scheduler')


def test_35_dimension_projection_is_consumed_changes_Q_and_receives_gradient(sample):
    q,batch,contracts=sample
    original=q.backbone.route_decoder.physical_extractor.extract
    calls=[]
    def changed(*args,**kwargs):
        result=original(*args,**kwargs);calls.append(result.features.shape)
        return replace(result,features=result.features+torch.linspace(-1,1,35).to(result.features))
    baseline=q.actions(batch,q.initial_hidden(),contracts)
    with patch.object(q.backbone.route_decoder.physical_extractor,'extract',side_effect=changed):
        altered=q.actions(batch,q.initial_hidden(),contracts)
    assert calls==[torch.Size((1,1,4,4,35))]
    assert not torch.equal(baseline.branch_values['route'],altered.branch_values['route'])
    baseline.utilities.sum().backward()
    params=list(q.backbone.route_decoder.physical_projection.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    assert sum(float(p.grad.abs().sum()) for p in params)>0


def test_both_hidden_lanes_equal_C_and_route_lane_is_private(sample):
    q,batch,_=sample
    state=q.initial_hidden();state[...,128:]=.1
    with torch.no_grad():
        _,out=q.encode(batch,state)
        c=q.backbone(batch,state)
    assert torch.equal(out,c.hidden_out)
    assert out.shape==(1,4,256)


@pytest.mark.parametrize('seed',[42,43,44])
def test_teacher_and_branch_sum_equal_exact_group_maximum(sample,seed):
    q,batch,contracts=sample
    generator=np.random.default_rng(seed)
    for epsilon in (0.,1.):
        out=q.actions(batch,q.initial_hidden(),contracts,epsilon=epsilon,rng=generator)
        teacher=q.actions(batch,q.initial_hidden(),contracts,proposals=out.proposals)
        assert torch.equal(out.utilities,teacher.utilities)
        expected=sum(out.branch_values[key].gather(-1,out.indices[key].unsqueeze(-1)).squeeze(-1)*out.active[...,i]
                     for i,key in enumerate(out.indices))
        assert torch.equal(out.credit_values['state_value']+expected,out.utilities)
        assert all(mask.is_legal(proposal) for mask,proposal in zip(contracts[0][0],out.proposals[0][0]))
        if epsilon==0:
            route=out.branch_values['route'].masked_fill(~out.masks['route'],-torch.inf).max(-1).values
            tx=out.credit_values['tx_atom_scores'].masked_fill(~out.credit_values['tx_support'],-torch.inf).max(-1).values
            cpu=out.credit_values['cpu_atom_scores'].masked_fill(~out.credit_values['cpu_support'],-torch.inf).max(-1).values
            assert torch.allclose(out.utilities,out.credit_values['state_value']+route+tx+cpu,atol=2e-6)


def test_TX_suffix_matches_independent_C_conditional_head_enumeration(sample):
    q,batch,contracts=sample
    # Use C's branch_logits directly for each prefix, not the Q table builder.
    with torch.no_grad():
        shared,_,_=q._features(batch,q.initial_hidden());rec=shared[0]
        out=q.actions(batch,q.initial_hidden(),contracts)
        mask=contracts[0][0][0]; best=-float('inf');best_indices=None
        for ti in range(len(mask.tx_select_domain)):
            if not mask.tx_select_mask[ti]:continue
            context={'tx_select':mask.tx_select_domain[ti]}
            for gi in np.flatnonzero(mask.mask_for('resource_group',context)):
                context['resource_group']=mask.resource_group_domain[gi]
                for wi in np.flatnonzero(mask.mask_for('resource_width',context)):
                    context['resource_width']=mask.resource_width_domain[wi]
                    indices={key:torch.full((1,1,4),int(value),dtype=torch.long) for key,value in
                             [('tx_select',ti),('resource_group',gi),('resource_width',wi)]}
                    group=q.backbone.branch_logits('resource_group',rec,indices,batch)[0,0,0,gi]
                    width=q.backbone.branch_logits('resource_width',rec,indices,batch)[0,0,0,wi]
                    power=q.backbone.branch_logits('power_level',rec,indices,batch)[0,0,0]
                    for pi in np.flatnonzero(mask.mask_for('power_level',context)):
                        base=out.branch_values['tx_select'][0,0,0,ti]
                        value=float(base+(group+width+power[pi] if ti else 0))
                        if value>best:best=value;best_indices=(ti,int(gi),int(wi),int(pi))
        chosen=tuple(int(out.indices[key][0,0,0]) for key in ('tx_select','resource_group','resource_width','power_level'))
        assert chosen==best_indices
        selected=sum(float(out.branch_values[key][0,0,0,out.indices[key][0,0,0]])*bool(out.active[0,0,0,i])
                     for i,key in enumerate(out.indices) if key in ('tx_select','resource_group','resource_width','power_level'))
        assert selected==pytest.approx(best,abs=2e-6)


def test_cpu_last_chance_gate_and_zero_frequency_match_C_support(sample):
    q,batch,contracts=sample
    masks=tuple(replace(mask,cpu_deadline_context=tuple((4.e7,1,2.e9) for _ in range(4))) for mask in contracts[0][0])
    with patch.object(q.backbone.cpu_joint_scheduler,'forward',return_value=(torch.tensor([[[[1.e6]+[0.]*12]*4]]),{})):
        out=q.actions(batch,q.initial_hidden(),((masks,),))
    assert not out.credit_values['cpu_support'][...,0].any()
    assert all(p.cpu_queue!='idle' and p.cpu_frequency>0 for p in out.proposals[0][0])
    assert not out.masks['cpu_frequency'][...,0].any()
    bad=replace(out.proposals[0][0][0],cpu_frequency=0.)
    with pytest.raises(ValueError,match='positive CPU support'):
        q.actions(batch,q.initial_hidden(),((masks,),),proposals=(((bad,*out.proposals[0][0][1:]),),))


def test_forced_TX_counterfactual_changes_only_requested_group(sample):
    q,batch,contracts=sample
    out=q.actions(batch,q.initial_hidden(),contracts,force_transmit_source=0)
    assert out.proposals[0][0][0].tx_select!='idle'
    assert not out.credit_values['tx_support'][0,0,0,0]


def test_actual_training_has_conserved_roles_live_route_targets_and_real_pairs(trained):
    assert trained.steps==64 and trained.learner.updates==15
    records=[row for rows in trained.replay.credit_records.values() for row in rows]
    assert len(records)==64
    assert all(sum(row['agent_reward'])==pytest.approx(row['team_reward']) for row in records)
    assert any(update['route_credit_samples']>0 for update in trained.updates)
    assert any(update['tx_pair_samples']>0 for update in trained.updates)
    assert trained.credit_collection_state['pair_count']==2
    pairs=[p for row in records for p in row['tx_pairs']]
    assert sum(p['shadow_steps'] for p in pairs)==trained.credit_collection_state['total_shadow_steps']
    for pair in pairs:
        assert pair['G_T']-pair['G_I']==pytest.approx(pair['d'])
        assert pair['state_isolation_status']==pair['common_randomness_status']=='PASS'
        assert 'old_p' not in pair and 'A_TX_raw' not in pair
    assert all(torch.isfinite(torch.tensor(update['loss'])) for update in trained.updates)


def test_team_target_uses_scalar_once_and_roles_are_separate_detached_targets(trained):
    trainer=trained
    sequences,context=reconstruct_replay_sequences(trainer.learner.online,trainer.learner.target,
        (trainer.replay.sequences[0],),trainer.replay)
    targets,stats=credit_return_targets(trainer.learner.online,trainer.learner.target,sequences,
        trainer.replay,'cpu',.99)
    team,roles=targets[0]
    assert team.shape==(1,4) and roles.shape==(1,4,4)
    assert not team.requires_grad and not roles.requires_grad
    assert stats['td_return_extra_slots']>0


class ConstantTaskValue(nn.Module):
    def forward(self,state,task):return state.new_full(state.shape[:-1],2.)


@pytest.mark.parametrize('outcome,payoff',[('done',1.),('expired',-1.),('truncated',0.)])
def test_Defer_bootstraps_at_binding_not_eventual_reward(trained,outcome,payoff):
    replay=copy.deepcopy(trained.replay);seq=replay.sequences[0]
    feature=[0.]*(10+2*4);task={'task_id':999,'source_uav':0,'features':feature}
    rows=[]
    for slot in range(4):
        row={'slot':slot,'before':[task],'after':[] if slot==3 else [task],
             'decisions':([{'task_id':999,'source_uav':0,'route':'defer' if slot==0 else 'local','slot':slot}] if slot<2 else []),
             'terminals':[{'task_id':999,'source_uav':0,'slot':3,'outcome':outcome,'payoff':payoff}] if slot==3 else []}
        rows.append(row)
    replay.credit_records[seq.episode]=rows
    target=copy.deepcopy(trained.learner.target);target.route_task_value=ConstantTaskValue()
    route,values=task_credit_labels(target,seq,replay,.9,'cpu')
    assert route[0][2]==pytest.approx(.9**2*2.) and route[0][3]=='binding_value_bootstrap'
    assert route[1][2]==pytest.approx(.9**2*payoff)
    assert values[0][2]==pytest.approx(.9**3*payoff)


def test_task_IDs_do_not_leak_between_episodes(trained):
    replay=copy.deepcopy(trained.replay);seq=replay.sequences[0]
    before=task_credit_labels(trained.learner.target,seq,replay,.99,'cpu')
    for row in replay.credit_records[1]:
        for event in row['terminals']:event['payoff']=9999.
    after=task_credit_labels(trained.learner.target,seq,replay,.99,'cpu')
    assert before==after


@pytest.mark.parametrize('loader',[load_checkpoint,load_continuation_checkpoint])
def test_credit_and_architecture_contracts_are_required(trained,tmp_path,loader):
    for key in ['credit','architecture']:
        payload=trained.payload('periodic');payload['implementation_contract'].pop(key)
        path=tmp_path/(key+'.pt');path.write_bytes(checkpoint_bytes(payload))
        with pytest.raises(ValueError,match=key):loader(path)


def test_exact_resume_preserves_credit_selector_and_replay(trained,tmp_path):
    source=QMIXTrainer(trained.config);source.train(stop_after_episodes=2)
    checkpoint=tmp_path/'resume.pt';checkpoint.write_bytes(checkpoint_bytes(source.payload('periodic')))
    restored=QMIXTrainer(trained.config);restored.restore(checkpoint)
    assert state_digest(restored.payload('periodic'))==state_digest(source.payload('periodic'))
    source.train(stop_after_episodes=3);restored.train(stop_after_episodes=3)
    assert state_digest(restored.payload('periodic'))==state_digest(source.payload('periodic'))


def test_new_method_checkpoint_series_is_frozen_and_has_correct_identity(config):
    trainer=QMIXTrainer(config);trainer.train(write_checkpoints=True)
    with patch('torch.optim.Adam',side_effect=AssertionError('no evaluation optimizer')):
        result=QMIXCheckpointProgressRunner(config).run()
    assert result['manifest']['method_suite'][0]=='c_feature_credit_qmix'
    assert result['manifest']['episode_count']==8 and result['manifest']['network_updates']==0
    assert len(result['rows'])==8


def test_original_C_config_and_initialization_identity_is_unchanged():
    cfg=load_run_config(ROOT/'experiments/task-allocation/configs/c-seed42-8k.json')
    assert cfg.config_hash=='9e76466588b68a6002d24d80f5e1424f370f26e97f8d4854c392f41da682ad15'
    assert state_digest(CAGATMAPPOActor(cfg).state_dict())=='c161fe274db124be9d1e45139c65ab00bdcbea9fb509b2087c6a43d5b06f0f01'


def test_effective_CPU_support_equals_C_distribution_implementation(sample):
    from src.models.integrated_scheduler import cpu_joint_statistics
    q,batch,contracts=sample
    with torch.no_grad():
        shared,_,_=q._features(batch,q.initial_hidden())
        c=cpu_joint_statistics(q.backbone,shared[0],batch,list(contracts[0][0]),None,'deterministic',None)
        out=q.actions(batch,q.initial_hidden(),contracts)
    assert torch.equal(c['mask'],out.credit_values['cpu_support'])


def test_128_slot_return_limit_uses_only_real_available_transitions(config):
    cfg=replace(config,environment=replace(config.environment,episode_horizon=160),
        training=replace(config.training,qmix=replace(config.training.qmix,max_training_episodes=1,
            max_training_environment_steps=160,checkpoint_interval_steps=160,replay_capacity_sequences=10000,
            sequence_batch_size=10000)))
    trainer=QMIXTrainer(cfg);trainer.train()
    first=trainer.replay.sequences[0]
    lengths=[]
    def zeros(online,target,sequence,device):
        lengths.append(sequence.length)
        return torch.zeros(1,sequence.length),torch.zeros(1,sequence.length,4)
    with patch('src.models.c_feature_qmix_credit.next_components',side_effect=zeros):
        labels,stats=credit_return_targets(trainer.learner.online,trainer.learner.target,(first,),trainer.replay,'cpu',.99)
    assert lengths==[128] and stats['td_return_extra_slots']==124
    assert labels[0][0].shape==(1,4) and trainer.learner.updates==0


def test_pair_return_gap_trains_private_upper_Q_without_PPO_probability(trained):
    learner=copy.deepcopy(trained.learner);replay=copy.deepcopy(trained.replay)
    sequence=next(seq for seq in replay.sequences if any(row['tx_pairs'] for row in
        replay.credit_records[seq.episode][seq.first_slot:seq.first_slot+seq.length]))
    for row in replay.credit_records[sequence.episode]:
        row['decisions']=[]
        for pair in row['tx_pairs']:pair['G_T']+=.3
    rebuilt,_=reconstruct_replay_sequences(learner.online,learner.target,(sequence,),replay);sequence=rebuilt[0]
    learner.online.zero_grad(set_to_none=True)
    out=learner.online.agent.actions(sequence.actor,sequence.hidden,sequence.contracts,proposals=sequence.proposals)
    loss,stats=auxiliary_losses(learner.online,learner.target,sequence,out,replay,.99,out.utilities.detach(),'cpu')
    loss.backward()
    parameters=list(learner.online.agent.backbone.tx_upper_private.parameters())
    assert stats['tx_pair_samples']>0 and stats['tx_pair_loss']>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in parameters)
    assert all(p.grad is None for p in learner.target.parameters())


def test_inconsistent_encoder_sizes_are_rejected(config):
    bad=replace(config,training=replace(config.training,qmix=replace(config.training.qmix,encoder_hidden_dimension=64)))
    with pytest.raises(ValueError,match='encoder dimensions'):CFeatureQNetwork(bad)


def test_state_value_models_inactive_slots_and_preserves_greedy_proposals(config,sample):
    q,batch,contracts=sample
    original=q.actions(batch,q.initial_hidden(),contracts)
    with torch.no_grad():q.value_head.bias.fill_(2.)
    shifted=q.actions(batch,q.initial_hidden(),contracts)
    assert shifted.proposals==original.proposals
    assert torch.allclose(shifted.utilities,original.utilities+2.)
    observations=U2UMECEnvironment(config).reset().observations
    idle=ActorObservationTensorizer(config).encode_step(observations)
    masks=((tuple(obs.action_masks for obs in observations),),)
    out=q.actions(idle,q.initial_hidden(),masks)
    assert not out.active.any()
    assert torch.equal(out.utilities,torch.full_like(out.utilities,2.))
    (.5*(out.utilities-3.).square().mean()).backward()
    assert q.value_head.bias.grad.item()==pytest.approx(-1.)
    assert torch.isfinite(q.value_head.weight.grad).all()


def test_target_bootstrap_includes_action_independent_state_value(config):
    cfg=replace(config,training=replace(config.training,qmix=replace(config.training.qmix,
        sequence_batch_size=10000,replay_capacity_sequences=10000)))
    trainer=QMIXTrainer(cfg);trainer.train(stop_after_episodes=1)
    from src.models.c_feature_qmix_credit import next_components
    seq=trainer.replay.sequences[0]
    with torch.no_grad():trainer.learner.target.agent.value_head.bias.fill_(3.)
    _,roles=next_components(trainer.learner.online,trainer.learner.target,seq,'cpu')
    with torch.no_grad():trainer.learner.target.agent.value_head.bias.zero_()
    _,base=next_components(trainer.learner.online,trainer.learner.target,seq,'cpu')
    assert torch.allclose(roles,base+3.,atol=2e-6)
