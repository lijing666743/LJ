"""Step03 execution contracts only; no PPO update or experiment efficacy claim."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.config import ConfigError, RouteDecoderMode, load_run_config
from src.models.ca_gat_mappo import (
    CAGATMAPPOActor, MAPPOCentralizedCritic, MAPPONetworkError,
    OptionAwareRouteDecoderV1,
)
from src.models.ca_gat_mappo_actions import (
    ActionDistributionError, CAGATMAPPOActionDistribution, SequentialActionMaskBatch,
)
from src.models.unified_route import build_unified_route_decoder, build_unified_route_probe
from tests.test_route_candidate_interface import P1A, ready_fixture
from tests.test_route_head_lr_isolation import persisted_batch


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def fixture(count=4):
    cfg, obs, adapter, batch = ready_fixture(count)
    cfg = replace(cfg, training=replace(cfg.training, mappo=replace(
        cfg.training.mappo, route_decoder_mode=RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1)))
    return build_unified_route_probe(cfg), obs, batch


def route_support(batch, observations, support):
    mask = torch.tensor(support, dtype=torch.bool).expand_as(batch.action_masks["route"]).clone()
    batch = replace(batch, action_masks={**batch.action_masks, "route": mask},
                    action_indices={**batch.action_indices, "route": mask.long().argmax(-1)})
    contracts = tuple(replace(obs.action_masks,
                             route_mask=np.array(support, dtype=np.bool_),
                             route_branch_active=sum(support) > 1)
                      for obs in observations)
    return batch, SequentialActionMaskBatch(((contracts,),))


def fixed_logits(actor, values):
    def hook(module, args, result):
        return result.new_tensor(values).expand_as(result).clone()
    return actor.route_decoder.register_forward_hook(hook)


@pytest.mark.parametrize("values,expected", [
    ([100., 4., 0., 0., 0., 0.], 1),  # illegal Idle cannot win
    ([0., 0., 4., 0., 0., 0.], 2),
    ([0., 0., 0., 4., 0., 0.], 3),
    ([0., 0., 0., 0., 4., 0.], 4),
    ([0., 0., 0., 0., 0., 4.], 5),
    ([0., 1., 1., 1., 1., 1.], 1),  # lowest legal index on ties
    ([0., 2., -8., 1.5, 1.5, 1.5], 1),  # Remote sum wins, leaf Local wins
    ([0., -10000., 10000., -10000., -10000., -10000.], 2),
])
def test_production_selector_chooses_legal_leaf_not_group_mass(values, expected):
    probe, observations, batch = fixture()
    masks = SequentialActionMaskBatch.from_observations(observations)
    with fixed_logits(probe.actor, values):
        result = probe.distribution.deterministic_actions(batch, masks)
    leaf = result.route_leaf_result()
    assert (leaf.chosen_indices == expected).all()
    assert torch.equal(leaf.chosen_indices, leaf.leaf_argmax_indices)
    expected_probs = torch.softmax(result.raw_logits["route"].masked_fill(
        ~result.action_masks["route"], -torch.inf), -1)
    torch.testing.assert_close(result.probabilities["route"], expected_probs)
    if values[1] == 2.:
        assert (leaf.group_mass_argmax == 1).all()  # Local/Remote/Defer
        assert (leaf.chosen_indices == 1).all()
    for ego, proposal in enumerate(result.proposals[0][0]):
        assert proposal.route == observations[ego].action_masks.route_domain[expected]


@pytest.mark.parametrize("support,expected", [
    ([False, True, True, False, False, False], 1),
    ([False, True, True, False, True, False], 4),
    ([True, False, False, False, False, False], 0),
    ([False, False, True, False, False, False], 2),
])
def test_degenerate_legal_sets_use_same_distribution_in_sample_and_replay(support, expected):
    probe, observations, batch = fixture()
    batch, masks = route_support(batch, observations, support)
    with fixed_logits(probe.actor, [10000., 2., 1., 8., 7., 6.]):
        deterministic = probe.distribution.deterministic_actions(batch, masks)
        sampled = probe.distribution.sample_actions(batch, masks)
        replay = probe.distribution.evaluate_actions(batch, masks, sampled.proposals)
    assert (deterministic.action_indices["route"] == expected).all()
    assert not sampled.probabilities["route"][~sampled.action_masks["route"]].any()
    torch.testing.assert_close(sampled.branch_log_probs["route"], replay.branch_log_probs["route"])
    assert torch.isfinite(sampled.branch_log_probs["route"]).all()
    assert torch.isfinite(sampled.branch_entropies["route"]).all()
    if sum(support) == 1:
        assert not sampled.active_branches["route"].any()
        assert not sampled.branch_log_probs["route"].any()


@pytest.mark.parametrize("count", [2, 3, 4, 5])
def test_real_scorer_sampling_consumes_one_flat_categorical_and_replays(count):
    probe, observations, batch = fixture(count)
    masks = SequentialActionMaskBatch.from_observations(observations)
    generator = torch.Generator().manual_seed(881)
    clone = torch.Generator().manual_seed(881)
    logits = probe.actor(batch).raw_logits["route"]
    expected_probs = torch.softmax(logits.masked_fill(~batch.action_masks["route"], -torch.inf), -1)
    expected_indices = torch.multinomial(expected_probs.reshape(-1, count + 2),
                                         1, generator=clone).reshape(1, 1, count)
    sampled = probe.distribution.sample_actions(batch, masks, generator=generator)
    replay = probe.distribution.evaluate_actions(batch, masks, sampled.proposals)
    assert torch.equal(sampled.action_indices["route"], expected_indices)
    torch.testing.assert_close(sampled.probabilities["route"], expected_probs)
    expected_logp = torch.log_softmax(logits.masked_fill(
        ~batch.action_masks["route"], -torch.inf), -1).gather(
            -1, expected_indices.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(sampled.branch_log_probs["route"], expected_logp)
    for branch in sampled.branch_log_probs:
        torch.testing.assert_close(sampled.branch_log_probs[branch], replay.branch_log_probs[branch])
    torch.testing.assert_close(sampled.hidden_out, replay.hidden_out, rtol=0, atol=0)
    candidate_view = probe.actor.forward_with_route_interface(batch).route_candidates
    assert candidate_view.node_ids.shape == (count, count)
    assert candidate_view.endpoint_features.shape[-2] == count


def test_deterministic_and_readout_do_not_consume_policy_rng():
    probe, observations, batch = fixture()
    control, _, _ = fixture()
    masks = SequentialActionMaskBatch.from_observations(observations)
    global_before = torch.random.get_rng_state().clone()
    probe.distribution.deterministic_actions(batch, masks).route_leaf_result()
    assert not probe.distribution._generators
    assert torch.equal(global_before, torch.random.get_rng_state())
    first = probe.distribution.sample_actions(batch, masks)
    generator = next(iter(probe.distribution._generators.values()))
    before = generator.get_state().clone()
    probe.distribution.deterministic_actions(batch, masks).route_leaf_result()
    probe.distribution.evaluate_actions(batch, masks, first.proposals)
    assert torch.equal(before, generator.get_state())
    control.distribution.sample_actions(batch, masks)
    actual = probe.distribution.sample_actions(batch, masks)
    expected = control.distribution.sample_actions(batch, masks)
    for branch in actual.action_indices:
        assert torch.equal(actual.action_indices[branch], expected.action_indices[branch])


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_p1a_common_tensors_and_outputs_preserved_with_independent_route_rng(seed, monkeypatch):
    cfg = replace(load_run_config(P1A), seed=seed)
    baseline_hash = cfg.config_hash
    reference = CAGATMAPPOActor(cfg)
    reference_critic = MAPPOCentralizedCritic(cfg)
    before = torch.random.get_rng_state().clone()
    def no_load(*args, **kwargs):
        raise AssertionError("probe must never load pretrained weights")
    monkeypatch.setattr(torch, "load", no_load)
    probe = build_unified_route_probe(cfg)
    assert torch.equal(before, torch.random.get_rng_state())
    after_critic = MAPPOCentralizedCritic(cfg)
    assert probe.baseline_config is cfg and cfg.config_hash == baseline_hash
    a, b = reference.state_dict(), probe.actor.state_dict()
    common_keys = {key for key in a if not key.startswith("action_heads.route.")}
    assert common_keys == {key for key in b if not key.startswith("action_heads.route.")}
    assert any(k.startswith("route_gru.") for k in common_keys)
    assert any(k.startswith("tx_upper_private.") for k in common_keys)
    for key in common_keys:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    for key, tensor in reference_critic.state_dict().items():
        torch.testing.assert_close(tensor, after_critic.state_dict()[key], rtol=0, atol=0)
    # Same persisted observations and all recurrent lanes, without any episode evaluation.
    batch = persisted_batch(cfg)
    batch = replace(batch, episode_starts=torch.zeros_like(batch.episode_starts))
    hidden = torch.full_like(reference.initial_hidden(1), .1)
    ref_out, new_out = reference(batch, hidden), probe.actor(batch, hidden)
    torch.testing.assert_close(ref_out.hidden_out, new_out.hidden_out, rtol=0, atol=0)
    for branch in ref_out.raw_logits:
        if branch != "route":
            torch.testing.assert_close(ref_out.raw_logits[branch], new_out.raw_logits[branch], rtol=0, atol=0)
    assert not torch.equal(ref_out.raw_logits["route"], new_out.raw_logits["route"])
    repeat = build_unified_route_probe(cfg)
    for key, tensor in probe.actor.route_decoder.state_dict().items():
        torch.testing.assert_close(tensor, repeat.actor.route_decoder.state_dict()[key], rtol=0, atol=0)
    other = build_unified_route_probe(replace(cfg, seed=seed + 1))
    assert probe.initialization_seed != other.initialization_seed
    assert not torch.equal(probe.actor.route_decoder.control_head.weight,
                           other.actor.route_decoder.control_head.weight)


def test_old_flat_feature_path_reused_exactly_and_private_lane_preserved():
    cfg = load_run_config(P1A)
    probe = build_unified_route_probe(cfg)
    batch = persisted_batch(cfg)
    batch = replace(batch, episode_starts=torch.zeros_like(batch.episode_starts))
    captured = []
    hook = probe.actor.route_decoder.register_forward_pre_hook(lambda module, args: captured.append(args))
    hidden = torch.full_like(probe.actor.initial_hidden(1), .1)
    output = probe.actor.forward_with_route_interface(batch, hidden)
    hook.remove()
    old_flat = OptionAwareRouteDecoderV1(spec=probe.actor.spec)
    old_flat.load_state_dict(probe.actor.route_decoder.state_dict(), strict=True)
    torch.testing.assert_close(output.raw_logits["route"], old_flat(*captured[0]), rtol=0, atol=0)
    assert output.route_candidates.uses_private_backbone
    torch.testing.assert_close(output.route_candidates.route_recurrent_features,
                               output.route_recurrent_features, rtol=0, atol=0)
    shared_change = hidden.clone(); shared_change[..., :128] += 1.
    private_change = hidden.clone(); private_change[..., 128:] += 1.
    shared = probe.actor(batch, shared_change)
    private = probe.actor(batch, private_change)
    torch.testing.assert_close(shared.raw_logits["route"], output.raw_logits["route"], rtol=0, atol=0)
    assert not torch.equal(private.raw_logits["route"], output.raw_logits["route"])


def test_batch_and_time_axes_preserve_leaf_probabilities_and_replay():
    probe, observations, batch = fixture()
    batch = replace(batch,
        self_features=batch.self_features.expand(2, 3, -1, -1).clone(),
        neighbor_public_features=batch.neighbor_public_features.expand(2, 3, -1, -1, -1).clone(),
        edge_features=batch.edge_features.expand(2, 3, -1, -1, -1).clone(),
        neighbor_mask=batch.neighbor_mask.expand(2, 3, -1, -1).clone(),
        episode_starts=batch.episode_starts.expand(2, 3, -1).clone(),
        action_masks={k: v.expand(2, 3, -1, -1).clone() for k, v in batch.action_masks.items()},
        action_indices={k: v.expand(2, 3, -1).clone() for k, v in batch.action_indices.items()})
    contracts = tuple(obs.action_masks for obs in observations)
    masks = SequentialActionMaskBatch(((contracts,) * 3,) * 2)
    sampled = probe.distribution.sample_actions(batch, masks)
    replay = probe.distribution.evaluate_actions(batch, masks, sampled.proposals)
    assert sampled.probabilities["route"].shape == (2, 3, 4, 6)
    torch.testing.assert_close(sampled.branch_log_probs["route"], replay.branch_log_probs["route"])
    torch.testing.assert_close(sampled.hidden_out, replay.hidden_out, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_new_decoder_construction_preserves_initialized_cuda_rng():
    probe, _, _ = fixture()
    before = torch.cuda.get_rng_state_all()
    build_unified_route_decoder(probe.actor.spec, 42)
    after = torch.cuda.get_rng_state_all()
    assert len(before) == len(after)
    for a, b in zip(before, after):
        assert torch.equal(a, b)


def test_production_unified_identity_requires_matching_config():
    cfg = load_run_config(P1A)
    probe = build_unified_route_probe(cfg)
    pending = replace(cfg, training=replace(cfg.training, mappo=replace(
        cfg.training.mappo, route_decoder_mode=RouteDecoderMode.UNIFIED_NODE_V1)))
    production = CAGATMAPPOActor(pending)
    candidates = production.forward_with_route_candidates(persisted_batch(cfg)).route_candidate_features
    assert candidates.shape[-1] == 411 and not candidates.requires_grad
    for key, value in probe.actor.state_dict().items():
        torch.testing.assert_close(value, production.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ActionDistributionError, match="decoder modes differ"):
        CAGATMAPPOActionDistribution(probe.actor, cfg)
    with pytest.raises(ValueError, match="hierarchical reference"):
        build_unified_route_probe(ready_fixture()[0])
