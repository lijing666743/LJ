"""Step05 engineering evidence only; synthetic units and fresh random actors."""
import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.config import ConfigError, RouteDecoderMode, load_run_config
from src.models.ca_gat_mappo import CAGATMAPPOActor, MAPPOCentralizedCritic, MAPPOTensorSpec
from src.models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution, ActionDistributionError
from src.models.ca_gat_mappo_checkpoint import (
    CheckpointError, atomic_save_checkpoint, load_checkpoint_payload,
    restore_model_optimizer_and_rng, validate_checkpoint_payload,
)
from src.models.ca_gat_mappo_update import CAGATMAPPORecurrentPPOUpdater
from src.models.ca_gat_mappo_rollout import CAGATMAPPORolloutBuffer, CAGATMAPPORolloutChunk
from src.models.route_candidates import build_route_candidates
from src.models.route_decision import unified_route_decision_record
from src.models.route_physical_relations import RoutePhysicalRelationExtractor, FEATURE_FIELDS
from src.models.unified_route import unified_route_checkpoint_identity
from tests.test_unified_route_learning import config, actual_fixture, ppo_loss
from tests.test_route_candidate_interface import P1A, ready_fixture
from tests.test_route_head_lr_isolation import persisted_batch
from tests.test_ca_gat_mappo_checkpoint import make_payload, assert_nested_equal
from tests.test_route_actor_credit_source import tensor_optimizer_buffer
from src.models.route_task_credit import KEY
from src.models.route_exploration import collect_route_training_records


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def physical_config(cfg=None):
    cfg = config() if cfg is None else cfg
    return replace(cfg, training=replace(cfg.training, mappo=replace(
        cfg.training.mappo, route_feature_mode="physical_relations_v1")))


def units_fixture():
    cfg = physical_config(); spec = MAPPOTensorSpec.from_config(cfg)
    ex = RoutePhysicalRelationExtractor(cfg, spec); q = ex.extractor
    batch = persisted_batch(cfg)
    s, p, e = (torch.zeros_like(x) for x in (
        batch.self_features, batch.neighbor_public_features, batch.edge_features))
    s[..., 0] = 20 / ex.energy
    s[..., 3] = 2 / ex.power
    s[..., 5] = 2e9 / ex.frequency
    s[..., 7] = 1e-28 / ex.coefficient
    s[..., q.unbound_start+4] = 2e6 / ex.bits
    s[..., q.unbound_start+5] = 6e8 / ex.cycles
    s[..., q.unbound_start+6] = 10 / (ex.slack_slots * ex.slot_s)
    s[..., q.unbound_start+7] = 1
    s[..., q.local_start] = 1 / ex.max_tasks
    s[..., q.local_start+2] = 1e9 / ex.queue_cycles
    # cpu_by_source[ego] duplicates Local; received source queues are disjoint.
    for ego in range(4):
        for source in range(4):
            value = 1e9 if source == ego else {0: 2e9, 1: 3e9, 2: 0.}[sum(j != ego for j in range(source))]
            start = q.cpu_start + source*8
            s[..., ego, start] = float(value > 0) / ex.max_tasks
            s[..., ego, start+2] = value / ex.queue_cycles
    # TX and UNBOUND aggregate cycles must not enter CPU backlog.
    s[..., q.unbound_start+2] = 90e9 / ex.queue_cycles
    for dest in range(4):
        s[..., q.tx_start+dest*8+1] = 3e6 / ex.queue_bits
        s[..., q.tx_start+dest*8+2] = 80e9 / ex.queue_cycles
    p[..., 0] = 1
    p[..., 7] = 30 / ex.energy
    p[..., 10] = 4e9 / ex.frequency
    p[..., 11] = 2e-28 / ex.coefficient
    p[..., 12] = 2 / ex.max_tasks
    p[..., 13] = 8e9 / ex.queue_cycles
    p[..., 15] = 8 / ex.horizon  # (AoI + 1)/horizon -> 7 slots
    p[..., 16] = 1
    e[..., -4] = 1e6 / ex.rate
    e[..., -3] = 1
    return cfg, ex, replace(batch, self_features=s, neighbor_public_features=p, edge_features=e)


def extract(ex, batch, ids=None):
    return ex.extract(batch.self_features, batch.neighbor_public_features, batch.edge_features, ids)


def test_si_units_cpu_once_local_zero_transmission_and_energy():
    _, ex, batch = units_fixture(); r = extract(ex, batch)
    expected_local = dict(task_bits=2e6, task_cycles=6e8, slack_s=10., frequency_hz=2e9,
        queued_cycles=6e9, cpu_task_count=3., task_cpu_s=.3, queue_cpu_s=3.,
        tx_backlog_bits=0., rate_bps=0., task_tx_s=0., queue_tx_s=0., tx_energy_j=0.,
        service_s=3.3, slack_margin_s=6.7, cpu_energy_j=.24,
        destination_energy_j=20., source_energy_j=20., destination_energy_margin_j=19.76)
    expected_remote = dict(frequency_hz=4e9, queued_cycles=8e9, task_cpu_s=.15,
        queue_cpu_s=2., tx_backlog_bits=3e6, rate_bps=1e6, task_tx_s=2., queue_tx_s=3.,
        service_s=7.15, slack_margin_s=2.85, cpu_energy_j=1.92, tx_energy_j=10.,
        source_tx_energy_margin_j=10., destination_energy_margin_j=28.08, message_aoi_slots=7.)
    for expected, option in [(expected_local, 0), (expected_remote, slice(1, None))]:
        for name, value in expected.items():
            torch.testing.assert_close(r.raw[name][..., option], torch.full_like(r.raw[name][..., option], value))
    assert not r.raw['needs_u2u'][..., 0].any()
    assert r.raw['needs_u2u'][..., 1:].all() and r.raw['service_valid'].all()
    assert r.features.shape == (1, 1, 4, 4, 35) and len(FEATURE_FIELDS) == 35
    torch.testing.assert_close(r.features[..., FEATURE_FIELDS.index('task_cpu_s')],
                              torch.log1p(r.raw['task_cpu_s']/ex.slot_s))
    # Deliberately corrupt the duplicate ego slot: canonical Local wins once.
    s = batch.self_features.clone()
    for ego in range(4): s[..., ego, ex.extractor.cpu_start+ego*8+2] = 999.
    changed = extract(ex, replace(batch, self_features=s))
    torch.testing.assert_close(changed.raw['queued_cycles'], r.raw['queued_cycles'])


@pytest.mark.parametrize('case', ['missing', 'history_zero', 'csi', 'csi_zero', 'history_preferred'])
def test_rate_missing_zero_and_stale_csi_fallback(case):
    _, ex, batch = units_fixture(); edge = batch.edge_features.clone(); r = ex.spec.ru_count
    edge[..., -3] = case in {'history_zero', 'history_preferred'}
    edge[..., -4] = 0 if case == 'history_zero' else 1e6/ex.rate
    edge[..., 2+2*r] = case in {'csi', 'csi_zero', 'history_preferred'}
    edge[..., 3+2*r] = 13/ex.horizon
    edge[..., 2:2+r] = 0 if case == 'csi_zero' else 1e-5
    edge[..., 4+2*r:4+3*r] = 3.
    edge[..., 4+3*r:4+4*r] = 1.
    x = extract(ex, replace(batch, edge_features=edge))
    if case in {'missing', 'history_zero', 'csi_zero'}:
        assert not x.raw['rate_positive'][..., 1:].any()
        assert not x.raw['service_valid'][..., 1:].any()
        assert not x.raw['service_s'][..., 1:].any()  # invalid placeholder, not 0s estimate
        assert not x.raw['task_tx_s'][..., 1:].any()
        assert x.raw['rate_observed'][..., 1:].all() == (case != 'missing')
    elif case == 'csi':
        expected = r * ex.bandwidth * torch.log2(torch.tensor(1+2/r*1e-10/(4*ex.noise_w)))
        torch.testing.assert_close(x.raw['rate_bps'][..., 1:], expected.expand(1,1,4,3))
        torch.testing.assert_close(x.raw['csi_aoi_slots'][..., 1:], torch.full((1,1,4,3),13.))
        assert x.raw['interference_all_valid'][..., 1:].all()
    else:
        torch.testing.assert_close(x.raw['rate_bps'][..., 1:], torch.full((1,1,4,3),1e6))
    assert x.raw['service_valid'][..., 0].all() and torch.isfinite(x.features).all()


@pytest.mark.parametrize('case', ['invalid_public', 'stale_public', 'no_task', 'zero_cpu', 'tiny_cpu', 'zero_energy'])
def test_boundaries_are_features_never_new_legality(case):
    _, ex, batch = units_fixture()
    s, p = batch.self_features.clone(), batch.neighbor_public_features.clone()
    if case == 'invalid_public': p[..., 0] = 0
    if case == 'stale_public': p[..., 15] = 10001/ex.horizon
    if case == 'no_task': s[..., ex.extractor.unbound_start+7] = 0
    if case == 'zero_cpu': s[..., 5] = 0; p[..., 10] = 0
    if case == 'tiny_cpu': s[..., 5] = 1e-15; p[..., 10] = 1e-15
    if case == 'zero_energy': s[..., 0] = 0; p[..., 7] = 0
    changed = replace(batch, self_features=s, neighbor_public_features=p)
    x = extract(ex, changed)
    assert torch.isfinite(x.features).all()
    if case == 'invalid_public':
        assert not x.raw['service_valid'][..., 1:].any()
        assert not x.raw['destination_energy_j'][..., 1:].any()
    if case == 'stale_public':
        assert x.raw['service_valid'].all()  # stale does not become invisible
        torch.testing.assert_close(x.raw['message_aoi_slots'][..., 1:], torch.full((1,1,4,3),10000.))
    if case in {'no_task', 'zero_cpu'}: assert not x.raw['service_valid'].any()
    if case == 'zero_energy': assert (x.raw['destination_energy_margin_j'] < 0).all()
    if case == 'tiny_cpu': assert x.raw['service_valid'].all()
    view = build_route_candidates(ex.spec, s, p, batch.edge_features, batch.action_masks['route'])
    torch.testing.assert_close(view.legal_mask,
        batch.action_masks['route'].gather(-1, view.action_indices[None,None].expand(1,1,4,4)))


def test_candidate_permutation_changes_view_not_destination_identity():
    _, ex, batch = units_fixture()
    p = batch.neighbor_public_features.clone()
    for node in range(4): p[..., node, 13] *= node+1
    batch = replace(batch, neighbor_public_features=p)
    before = extract(ex, batch)
    after = extract(ex, batch, before.node_ids[:, 1:].flip(-1))
    assert torch.equal(after.node_ids[:, 1:], before.node_ids[:, 1:].flip(-1))
    torch.testing.assert_close(after.features[..., 0, :], before.features[..., 0, :])
    torch.testing.assert_close(after.features[..., 1:, :], before.features[..., 1:, :].flip(-2))
    with pytest.raises(ValueError, match='IDs'):
        extract(ex, batch, before.node_ids[:, :3])


@pytest.mark.parametrize('count', [2,3,4,5])
def test_production_sizes_and_published_information_only(count):
    old, _, _, batch = ready_fixture(count)
    cfg = replace(old, training=replace(old.training, mappo=replace(old.training.mappo,
        route_decoder_mode=RouteDecoderMode.UNIFIED_NODE_V1, route_feature_mode='physical_relations_v1')))
    actor = CAGATMAPPOActor(cfg)
    output = actor(batch)
    assert output.raw_logits['route'].shape == (1,1,count,count+2)
    x = actor.route_decoder.physical_extractor.extract(batch.self_features,
        batch.neighbor_public_features, batch.edge_features)
    assert x.features.shape == (1,1,count,count,35)
    assert torch.equal(output.action_masks['route'], batch.action_masks['route'])
    # Remote CPU proxy is determined by this source's published message only.
    changed = batch.self_features.clone(); changed[..., 1, :] *= 2
    other = actor.route_decoder.physical_extractor.extract(changed,
        batch.neighbor_public_features, batch.edge_features)
    torch.testing.assert_close(other.features[..., 0, 1:, :], x.features[..., 0, 1:, :])


@pytest.mark.parametrize('seed', [42,43,44])
def test_b_c_common_initialization_equal_and_no_rng_consumption(seed):
    bcfg = replace(config(), seed=seed); ccfg = physical_config(bcfg)
    torch.manual_seed(919); rng = torch.random.get_rng_state().clone()
    b, c = CAGATMAPPOActor(bcfg), CAGATMAPPOActor(ccfg)
    assert torch.equal(rng, torch.random.get_rng_state())
    bs, cs = b.state_dict(), c.state_dict()
    assert set(cs)-set(bs) == {f'action_heads.route.physical_projection.{n}' for n in ['0.weight','0.bias','2.weight','2.bias']}
    for name in bs: assert torch.equal(bs[name], cs[name]), name
    bcritic, ccritic = MAPPOCentralizedCritic(bcfg).state_dict(), MAPPOCentralizedCritic(ccfg).state_dict()
    for name, value in bcritic.items():
        assert torch.equal(value, ccritic[name])
    assert bcfg.config_hash != ccfg.config_hash
    # Zero C contribution recovers B exactly on the same inputs.
    batch = persisted_batch(bcfg)
    with torch.no_grad():
        c.route_decoder.physical_projection[-1].weight.zero_()
        c.route_decoder.physical_projection[-1].bias.zero_()
        bo, co = b(batch), c(batch)
    for name in bo.raw_logits: torch.testing.assert_close(bo.raw_logits[name],co.raw_logits[name],atol=0,rtol=0)


@pytest.mark.parametrize('choice', [1,2,3,4,5])
def test_physical_features_in_actual_ppo_and_detached_q(choice):
    cfg, _, _, obs, batch, masks = actual_fixture(); cfg = physical_config(cfg)
    actor = CAGATMAPPOActor(cfg); distribution = CAGATMAPPOActionDistribution(actor, cfg)
    batch = replace(batch, self_features=batch.self_features.clone().requires_grad_())
    seen = []
    def capture(module, args):
        args[0].retain_grad(); seen.append(args[0])
    hook = actor.route_decoder.physical_projection[0].register_forward_pre_hook(capture)
    initial = distribution.deterministic_actions(batch, masks)
    proposals = ((tuple(replace(p, route=obs[u].action_masks.route_domain[choice])
                        for u,p in enumerate(initial.proposals[0][0])),),)
    policy = distribution.evaluate_actions(batch, masks, proposals, return_route_candidates=True)
    hook.remove()
    assert policy.route_candidate_features.shape[-1] == 411
    assert not policy.route_candidate_features.requires_grad
    loss = ppo_loss(policy, cfg); loss.actor_loss.backward()
    assert seen[-1].grad is not None and seen[-1].grad.abs().sum() > 0
    # Every option's physical branch participates in the same leaf softmax,
    # including when Defer is chosen; no separate supervised physical loss.
    assert (seen[-1].grad.abs().sum(-1) > 0).all()
    for parameter in actor.route_decoder.physical_projection.parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in actor.route_gru.parameters())
    assert unified_route_decision_record(policy, 0, int(obs[0].private_queues.unbound.head_task_id))['feature_mode'] == 'physical_relations_v1'


@pytest.fixture(scope='module')
def c_buffer():
    """Saved observations and synthetic outcomes; all policy tensors fresh C.

    No historical weights are loaded into this actor/critic. The fixture is
    discarded after testing and is never used by an experiment.
    """
    cfg=physical_config();actor,critic=CAGATMAPPOActor(cfg),MAPPOCentralizedCritic(cfg)
    original=tensor_optimizer_buffer(cfg);buffer=CAGATMAPPORolloutBuffer(cfg)
    hidden=actor.initial_hidden(1);distribution=CAGATMAPPOActionDistribution(actor,cfg)
    from src.models.ca_gat_mappo import ACTION_BRANCH_ORDER
    for i in range(len(original)):
        tr=original.transition_at(i);chunk=CAGATMAPPORolloutChunk((tr,))
        with torch.no_grad():
            policy=distribution.evaluate_actions(chunk.actor_batch,chunk.action_mask_batch,chunk.proposals,hidden)
            value=critic(chunk.centralized_batch)[0,0]
        metadata=copy.deepcopy(tr.rejection_or_downgrade_summary)
        tasks={d['source_uav']:d['task_id'] for d in metadata[KEY]['decisions']}
        rows=collect_route_training_records(policy,environment_step=i,episode_id=0,slot=i+1,task_ids=tasks)
        metadata['route_training_v1']=rows;by_uav={row['uav_id']:row for row in rows}
        for decision in metadata[KEY]['decisions']:
            u=decision['source_uav'];decision.update(by_uav[u])
            decision['actor_leaf_probability']=policy.probabilities['route'][0,0,u].tolist()
            decision['actor_remote_conditional_probability']=policy.route_leaf_result().conditional_remote_probabilities[0,0,u].tolist()
        metadata['policy_components_v1']={
            'names':list(policy.component_log_probs),
            'log_probs':torch.stack(list(policy.component_log_probs.values()),-1)[0,0].tolist(),
            'active':torch.stack(list(policy.component_active.values()),-1)[0,0].tolist()}
        logs=torch.stack([policy.branch_log_probs[b][0,0] for b in ACTION_BRANCH_ORDER],-1)
        buffer.append(replace(tr,hidden_in=hidden[0],old_value=value,old_branch_log_probs=logs,
            old_joint_log_prob=logs.sum(-1),rejection_or_downgrade_summary=metadata))
        hidden=policy.hidden_out
    buffer.finalize();return cfg,buffer


def test_c_actual_four_epoch_updater_uses_physical_parameters(c_buffer,tmp_path):
    cfg,buffer=c_buffer;cfg=replace(cfg,output=replace(cfg.output,logs_dir=str(tmp_path)))
    actor,critic=CAGATMAPPOActor(cfg),MAPPOCentralizedCritic(cfg)
    initial=actor.route_decoder.physical_projection[0].weight.detach().clone()
    updater=CAGATMAPPORecurrentPPOUpdater(actor,critic,cfg)
    result=updater.update(buffer,collected_environment_steps=256)
    assert len(result.epoch_diagnostics)==4
    assert not torch.equal(initial,actor.route_decoder.physical_projection[0].weight)
    for row in updater.last_route_option_q_records:
        assert row['effective_actor_counterfactual_coefficient']==0.
        assert row['effective_actor_task_credit_coefficient']==1.
        if row['epoch_index']==0:assert row['ppo_ratio']==pytest.approx(1.,abs=2e-6)
    assert len(updater.last_route_task_value_losses)==len(updater.last_route_option_q_losses)==4


def test_c_checkpoint_nonempty_adam_identity_restore_and_b_incompatible(c_buffer,tmp_path):
    cfg,full=c_buffer
    chunk=CAGATMAPPORolloutChunk((full.transition_at(0),))
    batch,masks=chunk.actor_batch,chunk.action_mask_batch
    actor, critic = CAGATMAPPOActor(cfg), MAPPOCentralizedCritic(cfg)
    updater = CAGATMAPPORecurrentPPOUpdater(actor, critic, cfg)
    distribution = CAGATMAPPOActionDistribution(actor, cfg)
    policy = distribution.deterministic_actions(batch, masks)
    updater.optimizers.actor_optimizer.zero_grad(); ppo_loss(policy,cfg).actor_loss.backward()
    updater.optimizers.actor_optimizer.step()
    payload,_,_,_ = make_payload(cfg,'FINAL_COMPLETED',actor=actor,critic=critic,updater=updater)
    from tests.test_ca_gat_mappo_checkpoint import training_state
    from src.models.ca_gat_mappo_checkpoint import serialize_active_rollout,restore_active_rollout
    pending=CAGATMAPPORolloutBuffer(cfg)
    for i in range(160):pending.append(full.transition_at(i))
    payload['checkpoint_kind']='PERIODIC_RESUME'
    payload['training_state']=training_state(cfg,'PERIODIC_RESUME',160)
    payload['active_rollout_state']=serialize_active_rollout(pending,15,'PERIODIC_RESUME')
    assert payload['route_decoder_identity'] == unified_route_checkpoint_identity(cfg.seed,'physical_relations_v1')
    path = tmp_path/'synthetic_c_fixture.pt';atomic_save_checkpoint(payload,path)
    loaded = load_checkpoint_payload(path)
    a2,c2 = CAGATMAPPOActor(cfg),MAPPOCentralizedCritic(cfg)
    u2 = CAGATMAPPORecurrentPPOUpdater(a2,c2,cfg)
    restore_model_optimizer_and_rng(config=cfg,payload=loaded,actor=a2,critic=c2,
        optimizers=u2.optimizers,dtype=torch.float32)
    restored,version=restore_active_rollout(cfg,loaded['active_rollout_state'],
        checkpoint_kind='PERIODIC_RESUME',policy_version=15)
    assert version==15 and len(restored)==160
    assert_nested_equal(unittest.TestCase(),loaded['active_rollout_state'],
        serialize_active_rollout(restored,version,'PERIODIC_RESUME'))
    assert_nested_equal(unittest.TestCase(),actor.state_dict(),a2.state_dict())
    assert_nested_equal(unittest.TestCase(),updater.optimizers.actor_optimizer.state_dict(),u2.optimizers.actor_optimizer.state_dict())
    for a,u in [(actor,updater),(a2,u2)]:
        p = CAGATMAPPOActionDistribution(a,cfg).deterministic_actions(batch,masks)
        u.optimizers.actor_optimizer.zero_grad();ppo_loss(p,cfg).actor_loss.backward();u.optimizers.actor_optimizer.step()
    assert_nested_equal(unittest.TestCase(),actor.state_dict(),a2.state_dict())
    bad = copy.deepcopy(loaded);bad['route_decoder_identity']['feature_mode']='original_option_features_v1'
    with pytest.raises(CheckpointError,match='identity'):validate_checkpoint_payload(bad)
    with pytest.raises((CheckpointError,RuntimeError)):
        restore_model_optimizer_and_rng(config=config(),payload=loaded,actor=CAGATMAPPOActor(config()),
            critic=MAPPOCentralizedCritic(config()),optimizers=CAGATMAPPORecurrentPPOUpdater(
            CAGATMAPPOActor(config()),MAPPOCentralizedCritic(config()),config()).optimizers,dtype=torch.float32)
    with pytest.raises(ActionDistributionError,match='feature'):
        CAGATMAPPOActionDistribution(actor,config())


def test_feature_configuration_guards_and_p1a_hash():
    base = load_run_config(P1A)
    assert base.config_hash == '2f143a85c0e37f605bbf38b59b3a31267a305464179593d9828c7c0a1518e3fa'
    with pytest.raises(ConfigError): physical_config(base).validate()
    with pytest.raises(ConfigError): config(route_feature_mode='unknown')
    physical_config().validate()


@pytest.mark.parametrize('support', [
    [False,True,True,False,False,False],
    [False,True,True,False,True,False],
    [True,False,False,False,False,False],
])
def test_c_degenerate_support_keeps_environment_masks_and_finite_q(support):
    from tests.test_unified_route import route_support
    cfg,_,_,obs,batch,_=actual_fixture();cfg=physical_config(cfg)
    batch,masks=route_support(batch,obs,support)
    distribution=CAGATMAPPOActionDistribution(CAGATMAPPOActor(cfg),cfg)
    policy=distribution.deterministic_actions(batch,masks)
    replay=distribution.evaluate_actions(batch,masks,policy.proposals,return_route_candidates=True)
    assert torch.equal(replay.action_masks['route'],batch.action_masks['route'])
    assert not replay.probabilities['route'][~batch.action_masks['route']].any()
    assert torch.isfinite(replay.route_candidate_features).all()
    if not any(support[3:]): assert not replay.route_candidate_features[...,204:408].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_physical_forward_ppo_gradient_and_initialization_stream():
    cfg, _, _, _, batch, masks = actual_fixture();cfg=physical_config(cfg)
    cfg=replace(cfg,training=replace(cfg.training,mappo=replace(cfg.training.mappo,training_device='cuda')))
    torch.cuda.manual_seed_all(711); before=torch.cuda.get_rng_state().clone()
    actor=CAGATMAPPOActor(cfg).cuda()
    assert torch.equal(before,torch.cuda.get_rng_state())
    policy=CAGATMAPPOActionDistribution(actor,cfg).deterministic_actions(batch.to('cuda'),masks)
    ppo_loss(policy,cfg).actor_loss.backward()
    assert actor.route_decoder.physical_projection[0].weight.grad.abs().sum()>0
