"""Step02 gates: node identity, actor visibility and read-only Route results."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.config import ConfigError, RouteDecoderMode, RunConfig, load_run_config
from src.env.environment import U2UMECEnvironment
from src.env.lifecycle import LifecycleManager
from src.env.observation import ObservationBuilder
from src.models.ca_gat_mappo import ActorObservationTensorizer, CAGATMAPPOActor
from src.models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution, SequentialActionMaskBatch
from src.models.route_candidates import (
    build_route_candidates, remote_node_ids, route_leaf_result, visible_local_cpu_burden,
)
from tests.test_route_head_lr_isolation import persisted_batch

ROOT = Path(__file__).resolve().parents[1]
P1A = ROOT / "tests/fixtures/configs/e14e-p1a-task-event-credit-seed42-8k-dev.json"


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def ready_fixture(count=4):
    base = RunConfig()
    env = replace(base.environment, uav_count=count, episode_horizon=4,
                  arrival_probabilities=(1.0,) * count,
                  profile_assignment=("Balanced",) * count,
                  profile_perturbations=((0., 0., 0., 0.),) * count,
                  building_layout=(), candidate_neighbor_radius_m=2000.,
                  velocity_std_mps=(0., 0.), shadowing_std_db=0., csi_error_std_db=0.)
    cfg = replace(base, seed=42, environment=env,
                  training=replace(base.training, mappo=replace(base.training.mappo,
                       training_device="cpu", route_decoder_mode=RouteDecoderMode.OPTION_AWARE_V1)))
    cfg.validate()
    environment = U2UMECEnvironment(cfg)
    environment.reset()
    observations = environment.step(environment.canonical_proposals()).observations
    adapter = ActorObservationTensorizer(cfg)
    return cfg, observations, adapter, adapter.encode_step(observations, episode_start=False)


def view(adapter, batch, **kwargs):
    return build_route_candidates(adapter.spec, batch.self_features,
                                  batch.neighbor_public_features, batch.edge_features,
                                  batch.action_masks["route"], **kwargs)


@pytest.mark.parametrize("count", [2, 3, 4, 5])
def test_actual_environment_node_ids_and_executor_masks(count):
    cfg, observations, adapter, batch = ready_fixture(count)
    actor = CAGATMAPPOActor(cfg)
    output = actor.forward_with_route_interface(batch)
    candidates = output.route_candidates
    assert candidates.node_ids.shape == (count, count)
    assert candidates.legal_mask.shape == (1, 1, count, count)
    assert candidates.burden_features.shape == (1, 1, count, count, 10)
    assert candidates.endpoint_features.shape[-2] == count
    for ego, observation in enumerate(observations):
        assert candidates.node_ids[ego, 0].item() == ego
        assert candidates.action_indices[ego, 0].item() == 1
        for node, action in zip(candidates.node_ids[ego, 1:], candidates.action_indices[ego, 1:]):
            assert observation.action_masks.route_domain[action.item()] == node.item()
        torch.testing.assert_close(candidates.legal_mask[0, 0, ego],
             batch.action_masks["route"][0, 0, ego, candidates.action_indices[ego]], rtol=0, atol=0)


@pytest.mark.parametrize("support", ["no_remote", "single_remote", "no_task", "invalid_public"])
def test_edge_cases_preserve_legality_separate_from_message_validity(support):
    _, _, adapter, batch = ready_fixture()
    mask = batch.action_masks["route"].clone()
    if support in {"no_remote", "single_remote"}:
        mask[..., 3:] = False
        if support == "single_remote":
            mask[..., 4] = True
    elif support == "no_task":
        mask.fill_(False); mask[..., 0] = True
    public = batch.neighbor_public_features.clone()
    if support == "invalid_public":
        public.zero_()
    batch = replace(batch, neighbor_public_features=public,
                    action_masks={**batch.action_masks, "route": mask})
    candidates = view(adapter, batch)
    if support == "invalid_public":
        assert candidates.legal_mask[..., 1:].any()
        assert not candidates.validity_features[..., 1:, 1].any()
        assert not candidates.burden_features[..., 1:, 4].any()
    else:
        assert candidates.legal_mask[..., 1:].sum().item() == (4 if support == "single_remote" else 0)
    logits = torch.zeros_like(mask, dtype=batch.self_features.dtype)
    result = route_leaf_result(logits, mask, mask.long().argmax(-1))
    assert torch.equal(result.probabilities[~mask], torch.zeros_like(result.probabilities[~mask]))
    torch.testing.assert_close(result.probabilities.sum(-1), torch.ones_like(result.chosen_log_prob))
    if support == "no_task":
        assert not candidates.legal_mask.any()
        assert (result.group_mass_argmax == -1).all()
        assert (result.chosen_node_ids == -1).all()
        assert not result.conditional_remote_probabilities.any()


def test_candidate_view_permutation_preserves_ids_masks_features_and_embeddings():
    _, _, adapter, batch = ready_fixture()
    masks = {**batch.action_masks, "route": batch.action_masks["route"].clone()}
    masks["route"][..., ::2, 3] = False
    masks["route"][..., 1::2, 4] = False
    batch = replace(batch, action_masks=masks)
    count = adapter.spec.uav_count
    width = adapter.spec.encoder_hidden_dimension
    endpoints = torch.arange(count * count * width, dtype=batch.self_features.dtype).view(1, 1, count, count, width)
    local = endpoints[..., 0, :]
    args = dict(local_endpoint_features=local, candidate_endpoint_features=endpoints)
    baseline = view(adapter, batch, **args)
    permuted = view(adapter, batch, remote_ids=remote_node_ids(count).flip(-1), **args)
    for name in ("node_ids", "action_indices"):
        torch.testing.assert_close(getattr(permuted, name)[:, 1:], getattr(baseline, name)[:, 1:].flip(-1))
    for name in ("burden_features", "validity_features", "endpoint_features"):
        torch.testing.assert_close(getattr(permuted, name)[..., 1:, :], getattr(baseline, name)[..., 1:, :].flip(-2))
    torch.testing.assert_close(permuted.legal_mask[..., 1:], baseline.legal_mask[..., 1:].flip(-1))
    torch.testing.assert_close(permuted.burden_features[..., 0, :], baseline.burden_features[..., 0, :])


def test_existing_shared_scorer_permutates_scores_with_candidate_inputs():
    cfg, _, _, batch = ready_fixture()
    actor = CAGATMAPPOActor(cfg)
    captured = []
    handle = actor.route_decoder.shared_option_scorer.register_forward_pre_hook(
        lambda module, args: captured.append(args[0]))
    actor(batch)
    handle.remove()
    fused = captured[0]
    permutation = torch.tensor([0, 3, 1, 2])
    scorer = actor.route_decoder.shared_option_scorer
    torch.testing.assert_close(scorer(fused[..., permutation, :]), scorer(fused)[..., permutation, :])


@pytest.mark.parametrize("chosen", [1, 3])
def test_actual_choice_leaf_argmax_group_mass_and_conditional_are_distinct(chosen):
    logits = torch.tensor([0., .4, .1, .2, .2, .1]).log()
    logits[0] = 0.
    logits = logits.view(1, 1, 1, 6).expand(1, 1, 4, 6).clone()
    mask = torch.ones_like(logits, dtype=torch.bool); mask[..., 0] = False
    selection = torch.full((1, 1, 4), chosen, dtype=torch.long)
    result = route_leaf_result(logits, mask, selection)
    assert (result.chosen_indices == chosen).all()
    assert (result.leaf_argmax_indices == 1).all()
    assert (result.group_mass_argmax == 1).all()  # Remote mass .5, Local leaf .4
    torch.testing.assert_close(result.conditional_remote_probabilities,
                              torch.tensor([.4, .4, .2]).expand(1, 1, 4, 3))
    torch.testing.assert_close(result.chosen_log_prob.exp(), torch.full((1, 1, 4), .4 if chosen == 1 else .2))
    if chosen == 1:
        assert result.chosen_node_ids.tolist() == [[[0, 1, 2, 3]]]
    else:
        assert result.chosen_node_ids.tolist() == [[[1, 0, 0, 0]]]


@pytest.mark.parametrize("no_remote", [False, True])
def test_extreme_logits_conditional_and_gradients_remain_finite(no_remote):
    logits = torch.tensor([0., 1000., -900., -1000., -999., -998.]).view(1, 1, 1, 6).repeat(1, 1, 4, 1).requires_grad_()
    mask = torch.ones_like(logits, dtype=torch.bool); mask[..., 0] = False
    if no_remote:
        mask[..., 3:] = False
    result = route_leaf_result(logits, mask, torch.ones((1, 1, 4), dtype=torch.long))
    loss = -result.chosen_log_prob.mean() + result.conditional_remote_probabilities[..., 0].mean()
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(result.conditional_remote_probabilities).all()
    total = result.conditional_remote_probabilities.sum(-1)
    torch.testing.assert_close(total, torch.zeros_like(total) if no_remote else torch.ones_like(total))


def test_real_queue_cpu_ownership_and_unchanged_AB_feature_semantics():
    cfg, observations, adapter, batch = ready_fixture()
    manager = LifecycleManager()
    def create(source, cycles):
        return manager.create_task(source_uav=source, data_bits=10., cpu_cycles=cycles,
                                   arrival_slot=0, deadline_slot=20)
    local = create(0, 100.); manager.route_task(local, slot=1, destination="local")
    for source, cycles in ((1, 200.), (2, 300.)):
        task = create(source, cycles); manager.route_task(task, slot=1, destination=0)
        manager.service_transmission(task, slot=2, bits=10.)
    outgoing = create(0, 400.); manager.route_task(outgoing, slot=1, destination=3)
    create(0, 500.)  # UNBOUND is not yet competing for this CPU
    queues, *_ = ObservationBuilder.from_run_config(cfg)._private_queues(manager, 0, 3)
    assert queues.local_cpu.task_count == 1
    assert queues.cpu_by_source.task_count.tolist() == [1, 1, 1, 0]
    assert queues.cpu_by_source.remaining_cycles.sum() == 600.
    updated = (replace(observations[0], private_queues=queues), *observations[1:])
    batch = adapter.encode_step(updated, episode_start=False)
    candidates = view(adapter, batch)
    total = candidates.local_cpu_total
    assert total.normalized_task_count[0, 0, 0].item() * adapter._max_tasks == pytest.approx(3.)
    assert total.normalized_remaining_cycles[0, 0, 0].item() * adapter._max_queue_cycles == pytest.approx(600.)
    # The old explicit Local feature still sees only local_cpu (100), not 600.
    assert candidates.burden_features[0, 0, 0, 0, 1].item() * adapter._max_queue_cycles == pytest.approx(100.)
    ex = CAGATMAPPOActor(cfg).route_decoder.feature_extractor
    duplicate = batch.self_features.clone()
    duplicate[0, 0, 0, ex.cpu_start + 2] = 99.  # canonical own summary replaces the alias
    corrected = visible_local_cpu_burden(duplicate, adapter.spec)
    torch.testing.assert_close(corrected.normalized_remaining_cycles, total.normalized_remaining_cycles)


def test_private_context_hidden_lanes_and_gradients_are_preserved():
    cfg = load_run_config(P1A)
    batch = persisted_batch(cfg)
    batch = replace(batch, episode_starts=torch.zeros_like(batch.episode_starts))
    actor = CAGATMAPPOActor(cfg)
    hidden = actor.initial_hidden(1)
    hidden[..., :128] = .1
    hidden[..., 128:] = -.2
    before = torch.random.get_rng_state().clone()
    old = actor(batch, hidden)
    endpoints_seen_by_decoder = []
    hook = actor.route_decoder.endpoint_projection.register_forward_pre_hook(
        lambda module, args: endpoints_seen_by_decoder.append(args[0]))
    output = actor.forward_with_route_interface(batch, hidden)
    hook.remove()
    assert torch.equal(before, torch.random.get_rng_state())
    assert old.route_candidates is None
    for key in old.raw_logits:
        torch.testing.assert_close(output.raw_logits[key], old.raw_logits[key], rtol=0, atol=0)
    torch.testing.assert_close(output.hidden_out, old.hidden_out, rtol=0, atol=0)
    candidates = output.route_candidates
    assert candidates.uses_private_backbone
    torch.testing.assert_close(candidates.endpoint_features, endpoints_seen_by_decoder[0], rtol=0, atol=0)
    torch.testing.assert_close(candidates.route_recurrent_features, output.route_recurrent_features, rtol=0, atol=0)
    torch.testing.assert_close(candidates.route_hidden_out, output.route_hidden_out, rtol=0, atol=0)
    changed_shared = hidden.clone(); changed_shared[..., :128] += 1.
    changed_private = hidden.clone(); changed_private[..., 128:] += 1.
    shared_output = actor.forward_with_route_interface(batch, changed_shared)
    private_output = actor.forward_with_route_interface(batch, changed_private)
    torch.testing.assert_close(shared_output.route_recurrent_features, output.route_recurrent_features, rtol=0, atol=0)
    torch.testing.assert_close(private_output.recurrent_features, output.recurrent_features, rtol=0, atol=0)
    assert not torch.equal(private_output.route_recurrent_features, output.route_recurrent_features)
    assert not torch.equal(shared_output.recurrent_features, output.recurrent_features)
    candidates.route_recurrent_features.square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.route_gru.parameters())
    assert all(p.grad is None for p in actor.gru.parameters())


@pytest.mark.parametrize("mode", ["deterministic", "sample", "replay"])
@pytest.mark.parametrize("decoder", [RouteDecoderMode.OPTION_AWARE_V1, RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1, RouteDecoderMode.LEGACY])
def test_production_action_result_readout_uses_the_actual_choice(mode, decoder):
    cfg, observations, _, batch = ready_fixture()
    cfg = replace(cfg, training=replace(cfg.training, mappo=replace(cfg.training.mappo, route_decoder_mode=decoder)))
    actor = CAGATMAPPOActor(cfg)
    distribution = CAGATMAPPOActionDistribution(actor, cfg)
    masks = SequentialActionMaskBatch.from_observations(observations)
    if mode == "sample":
        output = distribution.sample_actions(batch, masks)
    else:
        output = distribution.deterministic_actions(batch, masks)
        if mode == "replay":
            output = distribution.evaluate_actions(batch, masks, output.proposals)
    before = torch.random.get_rng_state().clone()
    result = output.route_leaf_result()
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(result.chosen_indices, output.action_indices["route"])
    torch.testing.assert_close(result.probabilities, output.probabilities["route"])
    torch.testing.assert_close(result.chosen_log_prob, output.branch_log_probs["route"])
    for ego in range(cfg.environment.uav_count):
        index = result.chosen_indices[0, 0, ego].item()
        value = observations[ego].action_masks.route_domain[index]
        assert output.proposals[0][0][ego].route == value


def test_batched_time_axes_and_remote_public_information_boundary():
    _, _, adapter, batch = ready_fixture()
    expanded = replace(batch,
        self_features=batch.self_features.expand(2, 3, -1, -1).clone(),
        neighbor_public_features=batch.neighbor_public_features.expand(2, 3, -1, -1, -1).clone(),
        edge_features=batch.edge_features.expand(2, 3, -1, -1, -1).clone(),
        action_masks={k: v.expand(2, 3, -1, -1).clone() for k, v in batch.action_masks.items()})
    baseline = view(adapter, expanded)
    assert baseline.legal_mask.shape == (2, 3, 4, 4)
    assert baseline.burden_features.shape == (2, 3, 4, 4, 10)
    private_change = expanded.self_features.clone(); private_change[..., 1, 20:40] += 100.
    changed = view(adapter, replace(expanded, self_features=private_change))
    torch.testing.assert_close(changed.burden_features[..., 0, :, :], baseline.burden_features[..., 0, :, :], rtol=0, atol=0)
    torch.testing.assert_close(changed.validity_features[..., 0, :, :], baseline.validity_features[..., 0, :, :], rtol=0, atol=0)
    # Only the sender-visible public history can change its helper burden.
    public = expanded.neighbor_public_features.clone(); public[..., 0, 1, 13] += 1.
    changed = view(adapter, replace(expanded, neighbor_public_features=public))
    torch.testing.assert_close(changed.burden_features[..., 0, 1, 1], baseline.burden_features[..., 0, 1, 1] + 1.)


def test_unified_mode_preserves_p1a_credit_and_old_hash():
    cfg = load_run_config(P1A)
    assert cfg.config_hash == "2f143a85c0e37f605bbf38b59b3a31267a305464179593d9828c7c0a1518e3fa"
    assert RunConfig().training.mappo.route_decoder_mode == RouteDecoderMode.LEGACY
    pending = replace(cfg, training=replace(cfg.training, mappo=replace(cfg.training.mappo,
                                      route_decoder_mode=RouteDecoderMode.UNIFIED_NODE_V1)))
    pending.validate()
    assert pending.training.mappo.route_actor_credit_source == "task_event"
    assert pending.training.mappo.route_task_credit_coefficient == 1.
    assert pending.training.mappo.route_q_value_loss_coefficient == .5


@pytest.mark.parametrize("invalid", ["self", "duplicate", "float", "empty_mask", "nan"])
def test_invalid_candidate_contracts_fail_explicitly(invalid):
    _, _, adapter, batch = ready_fixture()
    ids = remote_node_ids(4)
    if invalid == "self": ids[0, 0] = 0
    if invalid == "duplicate": ids[0, 0] = ids[0, 1]
    if invalid == "float": ids = ids.float()
    if invalid == "empty_mask":
        batch = replace(batch, action_masks={**batch.action_masks, "route": torch.zeros_like(batch.action_masks["route"])})
    if invalid == "nan":
        value = batch.self_features.clone(); value[..., 0] = float("nan")
        batch = replace(batch, self_features=value)
    with pytest.raises(ValueError):
        view(adapter, batch, remote_ids=ids)


def test_illegal_chosen_action_is_never_silently_replaced():
    logits = torch.zeros((1, 1, 4, 6))
    mask = torch.zeros_like(logits, dtype=torch.bool); mask[..., 1] = True
    with pytest.raises(ValueError, match="illegal"):
        route_leaf_result(logits, mask, torch.full((1, 1, 4), 3, dtype=torch.long))


@pytest.mark.parametrize("action", [0, 2])
def test_control_actions_remain_distinct_from_execution_nodes(action):
    logits = torch.zeros((1, 1, 4, 6))
    mask = torch.ones_like(logits, dtype=torch.bool)
    if action == 0:
        mask.fill_(False); mask[..., 0] = True
    else:
        mask[..., 0] = False
    result = route_leaf_result(logits, mask, torch.full((1, 1, 4), action, dtype=torch.long))
    assert (result.chosen_indices == action).all()
    assert (result.chosen_node_ids == -1).all()
    assert torch.isfinite(result.chosen_log_prob).all()
