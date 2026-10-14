"""Unified/P1a engineering fixtures; no experiment run or reusable training data."""
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.config import ConfigError, RouteDecoderMode, load_run_config
from src.env.environment import U2UMECEnvironment
from src.evaluation.route_telemetry import EvaluationRouteTelemetryCollector, build_evaluation_route_summary
from src.models.ca_gat_mappo import ACTION_BRANCH_ORDER, ActorObservationTensorizer, CAGATMAPPOActor, MAPPOCentralizedCritic, CentralizedStateTensorizer
from src.models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution, SequentialActionMaskBatch
from src.models.ca_gat_mappo_checkpoint import (
    CheckpointError, atomic_save_checkpoint, load_checkpoint_payload, restore_active_rollout,
    restore_model_optimizer_and_rng, serialize_active_rollout, validate_checkpoint_payload,
)
from src.models.ca_gat_mappo_ppo import compute_configured_ppo_objective_and_loss
from src.models.ca_gat_mappo_rollout import CAGATMAPPORolloutBuffer, CAGATMAPPORolloutChunk
from src.models.ca_gat_mappo_update import CAGATMAPPORecurrentPPOUpdater
from src.models.integrated_scheduler import policy_component_names
from src.models.route_decision import normalize_route_decision_record
from src.models.route_exploration import collect_route_training_records, summarize_route_records, route_exploration_loss
from src.models.route_option_q import prepare_route_q, route_q_loss, write_route_q_telemetry
from src.models.route_task_credit import KEY, capture_task_values, make_step_metadata, prepare_task_credit
from src.models.unified_route import unified_route_checkpoint_identity
from tests.test_route_actor_credit_source import tensor_optimizer_buffer
from tests.test_route_candidate_interface import P1A, ready_fixture
from tests.test_route_head_lr_isolation import persisted_batch
from tests.test_ca_gat_mappo_checkpoint import make_payload, assert_nested_equal
from tests.test_route_task_credit import state, step, terminal


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**overrides):
    base = load_run_config(P1A, cli_overrides={"training.mappo.training_device": "cpu"})
    cfg = replace(base, training=replace(base.training, mappo=replace(
        base.training.mappo, route_decoder_mode=RouteDecoderMode.UNIFIED_NODE_V1, **overrides)))
    cfg.validate()
    return cfg


def actual_fixture():
    simple, _, _, _ = ready_fixture()
    cfg = replace(config(), environment=simple.environment)
    cfg.validate()
    environment = U2UMECEnvironment(cfg); environment.reset()
    observations = environment.step(environment.canonical_proposals()).observations
    batch = ActorObservationTensorizer(cfg).encode_step(observations, episode_start=False)
    actor = CAGATMAPPOActor(cfg)
    distribution = CAGATMAPPOActionDistribution(actor, cfg)
    return cfg, actor, distribution, observations, batch, SequentialActionMaskBatch.from_observations(observations)


def ppo_loss(policy, cfg, *, ratio=1.):
    names = policy_component_names(cfg)
    logs = torch.stack([policy.component_log_probs[n][0] for n in names], -1)
    old = logs.detach().clone(); old[..., 0] -= torch.log(logs.new_tensor(ratio))
    active = torch.stack([policy.component_active[n][0] for n in names], -1)
    zeros = torch.zeros_like(policy.joint_log_prob[0])
    return compute_configured_ppo_objective_and_loss(
        new_joint_log_prob=logs.sum(-1), old_joint_log_prob=old.sum(-1),
        advantage=zeros, current_value=zeros, return_target=zeros,
        entropy=policy.entropy[0], sequence_valid_mask=torch.ones(1, dtype=torch.bool, device=logs.device),
        config=cfg, new_branch_log_probs=logs, old_branch_log_probs=old,
        active_branch_indicators=active, route_advantage=torch.ones_like(zeros),
        route_entropy=policy.branch_entropies["route"][0], collected_environment_steps=256,
        tx_upper_advantage=zeros, tx_upper_paired_mask=torch.zeros_like(zeros, dtype=torch.bool))


@pytest.mark.parametrize("choice", [1, 2, 3, 4, 5])
def test_each_actual_leaf_gets_production_ppo_gradient_without_credit_backprop(choice):
    cfg, actor, distribution, observations, batch, masks = actual_fixture()
    deterministic = distribution.deterministic_actions(batch, masks)
    proposals = ((tuple(replace(p, route=observations[u].action_masks.route_domain[choice])
                        for u, p in enumerate(deterministic.proposals[0][0])),),)
    policy = distribution.evaluate_actions(batch, masks, proposals, return_route_candidates=True)
    assert not policy.route_candidate_features.requires_grad
    assert policy_component_names(cfg).count("route") == 1
    torch.testing.assert_close(policy.component_log_probs["route"], policy.branch_log_probs["route"])
    loss = ppo_loss(policy, cfg)
    torch.testing.assert_close(loss.branch_ratio[..., 0], torch.ones_like(loss.branch_ratio[..., 0]))
    derivatives = torch.autograd.grad(loss.actor_loss, policy.raw_logits["route"], retain_graph=True)[0]
    assert (derivatives[..., choice] < 0).all()
    assert not derivatives[~policy.action_masks["route"]].any()
    loss.actor_loss.backward()
    assert actor.route_decoder.shared_option_scorer[-1].weight.grad.abs().sum() > 0
    assert actor.route_decoder.control_head.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.route_gru.parameters())
    assert all(p.grad is None or not p.grad.any() for p in actor.self_encoder.parameters())
    critic = MAPPOCentralizedCritic(cfg)
    assert all(p.grad is None for p in critic.parameters())


def test_chosen_leaf_ratio_clip_and_inactive_route():
    cfg, actor, distribution, _, batch, masks = actual_fixture()
    policy = distribution.deterministic_actions(batch, masks)
    loss = ppo_loss(policy, cfg, ratio=2.)
    torch.testing.assert_close(loss.branch_ratio[..., 0], torch.full_like(loss.branch_ratio[..., 0], 2.))
    torch.testing.assert_close(loss.branch_clipped_surrogate[..., 0],
                               torch.full_like(loss.branch_clipped_surrogate[..., 0], 1.2))
    from tests.test_unified_route import route_support
    _, _, _, observations, _, _ = actual_fixture()
    idle_batch, idle_masks = route_support(batch, observations, [True, False, False, False, False, False])
    # Keep three real Route-active agents: the PPO objective intentionally
    # rejects batches with no active policy component anywhere.
    mixed_mask = idle_batch.action_masks["route"].clone()
    mixed_mask[..., 1:, :] = batch.action_masks["route"][..., 1:, :]
    idle_batch = replace(idle_batch,
        action_masks={**idle_batch.action_masks, "route": mixed_mask},
        action_indices={**idle_batch.action_indices, "route": mixed_mask.long().argmax(-1)})
    mixed_contracts = (idle_masks.contracts[0][0][0], *masks.contracts[0][0][1:])
    idle_masks = SequentialActionMaskBatch(((mixed_contracts,),))
    idle = distribution.deterministic_actions(idle_batch, idle_masks)
    inactive_loss = ppo_loss(idle, cfg)
    assert inactive_loss.branch_surrogate[0, 0, 0] == 0
    assert idle.branch_log_probs["route"][0, 0, 0] == 0


def test_q_layout_detach_chosen_only_supervision_and_cf_not_in_actor():
    cfg, actor, distribution, _, batch, masks = actual_fixture()
    policy = distribution.deterministic_actions(batch, masks)
    policy = distribution.evaluate_actions(batch, masks, policy.proposals, return_route_candidates=True)
    candidates = policy.route_candidate_features
    assert candidates.shape == (1, 1, 4, 6, 411) and not candidates.requires_grad
    assert not candidates[..., 0, :].any()
    assert torch.equal(candidates[0, 0, 0, 1, -3:], torch.tensor([1., 0., 0.]))
    assert torch.equal(candidates[0, 0, 0, 2, -3:], torch.tensor([0., 1., 0.]))
    assert torch.equal(candidates[0, 0, 0, 3, -3:], torch.tensor([0., 0., 1.]))
    # Use factual leaf snapshots plus synthetic known same-slot outcomes.
    from src.models.route_task_credit import construct_task_credit
    decisions = []
    for u in range(4):
        index = int(policy.action_indices["route"][0, 0, u])
        category = "local" if index == 1 else "defer" if index == 2 else "remote"
        remote_mask = policy.action_masks["route"][0, 0, u, 3:]
        remote_logits = policy.raw_logits["route"][0, 0, u, 3:].detach()
        decisions.append(dict(task_id=u, source_uav=u, episode_id=0, global_rollout_index=0,
            category=category, route_action_index=index,
            actor_leaf_probability=policy.probabilities["route"][0, 0, u].detach().tolist(),
            route_legal_mask=policy.action_masks["route"][0, 0, u].tolist(),
            actor_remote_conditional_probability=remote_logits.masked_fill(~remote_mask, -torch.inf).softmax(-1).tolist()))
    metadata = [step(before=[state(u, u) for u in range(4)], decisions=decisions,
                     terminals=[terminal(task=u, source=u, slot=0) for u in range(4)])]
    task = construct_task_credit(metadata, torch.full((1, 4), .3), torch.ones(1, 4, dtype=torch.bool), .99, 1.)
    critic = MAPPOCentralizedCritic(cfg)
    with torch.no_grad():
        critic.route_option_q.network[-1].weight[0, 0] = 1.
    central = torch.zeros(1, 1, critic.spec.centralized_state_dim, requires_grad=True)
    q = prepare_route_q(critic, policy, central, task, 99., actor_credit_source="task_event")
    assert q.counterfactual_advantage.abs().sum() > 0
    assert torch.equal(q.advantage, task.advantage)
    assert all(r["effective_actor_counterfactual_coefficient"] == 0 for r in q.records)
    value_loss, predictions = route_q_loss(critic, q)
    derivatives = torch.autograd.grad(value_loss, predictions, retain_graph=True)[0]
    expected = torch.zeros_like(derivatives); expected.scatter_(-1, q.chosen.unsqueeze(-1), 1.)
    assert not derivatives[expected == 0].any()
    value_loss.backward()
    assert central.grad is None
    assert all(p.grad is None for p in actor.parameters())
    assert all(not getattr(q, name).requires_grad for name in ("advantage", "targets", "baseline", "frozen_q", "counterfactual_advantage"))


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_formal_constructor_common_parameters_and_original_contract(seed):
    cfg = replace(config(), seed=seed)
    oldcfg = replace(cfg, training=replace(cfg.training, mappo=replace(
        cfg.training.mappo, route_decoder_mode=RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1)))
    old, new = CAGATMAPPOActor(oldcfg), CAGATMAPPOActor(cfg)
    for key, value in old.state_dict().items():
        if not key.startswith("action_heads.route."):
            assert torch.equal(value, new.state_dict()[key]), key
    assert cfg.training.mappo.route_task_credit_coefficient == 1.
    assert cfg.training.mappo.route_q_value_loss_coefficient == .5
    assert cfg.training.mappo.route_actor_credit_source == "task_event"
    assert cfg.training.mappo.route_q_input_mode == "normalized"
    assert cfg.training.mappo.route_exploration_initial_coefficient == .05
    assert cfg.training.mappo.route_exploration_anneal_steps == 6000
    assert oldcfg.reward == cfg.reward and oldcfg.environment == cfg.environment


def test_new_training_and_eval_records_actual_task_identity_and_old_schema_na():
    cfg, actor, distribution, observations, batch, masks = actual_fixture()
    # Remote group wins mass; Local wins the deterministic leaf.
    def fixed(module, args, result):
        return result.new_tensor([0., 2., -8., 1.5, 1.5, 1.5]).expand_as(result)
    hook = actor.route_decoder.register_forward_hook(fixed)
    policy = distribution.deterministic_actions(batch, masks); hook.remove()
    ids = {o.uav_id: int(o.private_queues.unbound.head_task_id) for o in observations}
    rng = torch.random.get_rng_state().clone()
    records = collect_route_training_records(policy, environment_step=1, episode_id=0, slot=1, task_ids=ids)
    evaluation = EvaluationRouteTelemetryCollector(cfg).collect_step(
        evaluation_run_id="synthetic_engineering_fixture", episode_id=0, evaluation_seed=1042,
        observations=observations, proposals=policy.proposals[0][0], action_output=policy, step_info={})
    assert torch.equal(rng, torch.random.get_rng_state())
    for record in [*records, *evaluation]:
        assert record["decision_mode"] == "flat_leaf_v1"
        assert record["chosen_leaf_index"] == record["leaf_argmax_index"] == 1
        assert record["group_mass_argmax"] == "Remote"
        assert record["actor_top_level_argmax"] is None
        assert record["task_id"] == ids[record["uav_id"]]
        assert len(record["leaf_scores"]) == len(record["leaf_probabilities"]) == 6
        json.dumps(record, allow_nan=False)
    assert summarize_route_records(records)["all_route_active"]["actor_argmax_count"]["Local"] == 4
    summary = build_evaluation_route_summary(evaluation_run_id="fixture", decoder_mode="unified_node_v1", records=evaluation, actor_episodes=[])
    assert summary["schema_version"] == "evaluation_route_telemetry.v2"
    normalized = normalize_route_decision_record({"schema_version": "evaluation_route_telemetry.v1", "selected_top_level": "Remote"})
    assert normalized["task_id"] is None and normalized["leaf_scores"] is None and normalized["decision_mode"] is None


def test_real_lifecycle_capture_and_telemetry_are_passive():
    simple, _, _, _ = ready_fixture()
    cfg = replace(config(), environment=simple.environment)
    environment = U2UMECEnvironment(cfg); environment.reset()
    current = environment.step(environment.canonical_proposals())
    actor, critic = CAGATMAPPOActor(cfg), MAPPOCentralizedCritic(cfg)
    from src.models.ca_gat_mappo_runtime import CAGATMAPPOFactualActorRuntime
    runtime = CAGATMAPPOFactualActorRuntime(actor, cfg, device="cpu")
    central = CentralizedStateTensorizer(cfg)
    before = capture_task_values(environment, central.encode_step(current.centralized_state).features, critic, cfg)
    policy = runtime.sample_step(current.observations, actor.initial_hidden(1), episode_start=False).action_output
    result = environment.step(policy.proposals[0][0])
    after = capture_task_values(environment, central.encode_step(result.centralized_state).features, critic, cfg)
    snapshot = copy.deepcopy(environment.snapshot()); rng = runtime.policy_generator.get_state().clone()
    metadata = make_step_metadata(before=before, after=after, info=result.info, policy=policy,
                                  episode_id=0, environment_step=1, policy_version=0, capture_route_leaves=True)
    assert snapshot == environment.snapshot()
    assert torch.equal(rng, runtime.policy_generator.get_state())
    for decision in metadata["decisions"]:
        assert decision["task_id"] in {t["task_id"] for t in before}
        assert decision["decision_mode"] == "flat_leaf_v1"
        assert decision["chosen_leaf_index"] == decision["route_action_index"]


@pytest.mark.parametrize("steps", [0, 5999, 6000, 8000])
def test_group_barrier_is_regularizer_only_and_finite_at_extreme_scores(steps):
    cfg, actor, distribution, _, batch, masks = actual_fixture()
    policy = distribution.deterministic_actions(batch, masks)
    logits = policy.raw_logits["route"].detach().clone().requires_grad_()
    with torch.no_grad(): logits[..., 1] = -10000.
    altered = replace(policy, raw_logits={**policy.raw_logits, "route": logits})
    penalty = route_exploration_loss(altered, cfg.training.mappo, steps, torch.ones(1, 1, dtype=torch.bool))
    penalty.scaled_loss.backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(penalty.scaled_loss)
    if steps < 6000: assert (logits.grad[..., 1] < 0).all()
    else: assert not logits.grad.any()
    assert torch.equal(altered.action_indices["route"], policy.action_indices["route"])


@pytest.fixture(scope="module")
def frozen_buffer():
    """Repeat saved observation inputs; synthetic terminals, never experiment data.

    No historical model parameters are loaded into the fresh actor/critic.
    Replace all old log-probs and hidden lanes using the actual unified policy.
    """
    cfg = config(); actor, critic = CAGATMAPPOActor(cfg), MAPPOCentralizedCritic(cfg)
    original = tensor_optimizer_buffer(cfg)
    buffer = CAGATMAPPORolloutBuffer(cfg); hidden = actor.initial_hidden(1)
    distribution = CAGATMAPPOActionDistribution(actor, cfg)
    for i in range(len(original)):
        tr = original.transition_at(i)
        chunk = CAGATMAPPORolloutChunk((tr,))
        with torch.no_grad():
            policy = distribution.evaluate_actions(chunk.actor_batch, chunk.action_mask_batch, chunk.proposals, hidden)
            value = critic(chunk.centralized_batch)[0, 0]
        metadata = copy.deepcopy(tr.rejection_or_downgrade_summary)
        tasks = {d["source_uav"]: d["task_id"] for d in metadata[KEY]["decisions"]}
        rows = collect_route_training_records(policy, environment_step=i, episode_id=0, slot=i+1, task_ids=tasks)
        metadata["route_training_v1"] = rows
        by_uav = {row["uav_id"]: row for row in rows}
        for decision in metadata[KEY]["decisions"]:
            decision.update(by_uav[decision["source_uav"]])
            u = decision["source_uav"]
            decision["actor_leaf_probability"] = policy.probabilities["route"][0, 0, u].tolist()
            decision["actor_remote_conditional_probability"] = policy.route_leaf_result().conditional_remote_probabilities[0, 0, u].tolist()
        metadata["policy_components_v1"] = {
            "names": list(policy.component_log_probs),
            "log_probs": torch.stack(list(policy.component_log_probs.values()), -1)[0, 0].tolist(),
            "active": torch.stack(list(policy.component_active.values()), -1)[0, 0].tolist()}
        logs = torch.stack([policy.branch_log_probs[b][0, 0] for b in ACTION_BRANCH_ORDER], -1)
        buffer.append(replace(tr, hidden_in=hidden[0], old_value=value,
            old_branch_log_probs=logs, old_joint_log_prob=logs.sum(-1),
            rejection_or_downgrade_summary=metadata))
        hidden = policy.hidden_out
    buffer.finalize()
    return cfg, buffer


def test_production_four_epochs_freeze_task_q_and_cf_stays_diagnostic(frozen_buffer, tmp_path):
    cfg, buffer = frozen_buffer
    cfg = replace(cfg, output=replace(cfg.output, logs_dir=str(tmp_path)))
    actor, critic = CAGATMAPPOActor(cfg), MAPPOCentralizedCritic(cfg)
    updater = CAGATMAPPORecurrentPPOUpdater(actor, critic, cfg)
    result = updater.update(buffer, collected_environment_steps=256)
    assert len(result.epoch_diagnostics) == 4
    assert len(updater.last_route_task_value_losses) == len(updater.last_route_option_q_losses) == 4
    grouped = {}
    for sample in updater.last_route_option_q_records:
        assert sample["final_route_advantage"] == pytest.approx(sample["shared_advantage"] + sample["task_advantage"], abs=2e-6)
        assert sample["effective_actor_task_credit_coefficient"] == 1.
        assert sample["effective_actor_counterfactual_coefficient"] == 0.
        if sample["epoch_index"] == 0:
            assert sample["ppo_ratio"] == pytest.approx(1., abs=2e-6)
        grouped.setdefault((sample["global_rollout_index"], sample["source_uav"]), []).append(sample)
    assert grouped
    for rows in grouped.values():
        assert len(rows) == 4
        for name in ["G_event", "final_route_advantage", "frozen_q", "counterfactual_advantage"]:
            assert all(row[name] == rows[0][name] for row in rows)
    assert updater.optimizers.actor_optimizer.state and updater.optimizers.critic_optimizer.state
    record = write_route_q_telemetry(cfg, updater.last_route_option_q_records, 256,
                                    updater.last_route_option_q_losses, buffer=buffer)
    assert record["schema"] == "route_option_q_update.v2" and record["decision_mode"] == "flat_leaf_v1"
    assert all(r["decision_mode"] == "flat_leaf_v1" for r in record["route_decisions"])
    assert record["unselected_Q_is_prediction_only"]


def test_checkpoint_identity_adam_hidden_and_pending_rollout_resume(frozen_buffer, tmp_path):
    import unittest
    from tests.test_ca_gat_mappo_checkpoint import training_state
    cfg, full = frozen_buffer
    actor, critic = CAGATMAPPOActor(cfg), MAPPOCentralizedCritic(cfg)
    updater = CAGATMAPPORecurrentPPOUpdater(actor, critic, cfg)
    # Nonempty real Adam moments from disposable synthetic objectives only.
    for module, optimizer in [(actor, updater.optimizers.actor_optimizer), (critic, updater.optimizers.critic_optimizer)]:
        optimizer.zero_grad(set_to_none=True)
        sum(p.square().sum() for p in module.parameters()).backward(); optimizer.step()
    payload, _, _, _ = make_payload(cfg, "FINAL_COMPLETED", actor=actor, critic=critic, updater=updater)
    assert payload["route_decoder_identity"] == unified_route_checkpoint_identity(cfg.seed)
    pending = CAGATMAPPORolloutBuffer(cfg)
    for i in range(160): pending.append(full.transition_at(i))
    payload["checkpoint_kind"] = "PERIODIC_RESUME"
    payload["training_state"] = training_state(cfg, "PERIODIC_RESUME", 160)
    payload["active_rollout_state"] = serialize_active_rollout(pending, 15, "PERIODIC_RESUME")
    advanced = torch.Generator().set_state(payload["policy_rng_state"]["generator_state"])
    torch.rand(17, generator=advanced)
    payload["policy_rng_state"]["generator_state"] = advanced.get_state()
    validate_checkpoint_payload(payload, config=cfg)
    path = tmp_path / "synthetic_unified_resume_fixture.pt"
    atomic_save_checkpoint(payload, path); loaded = load_checkpoint_payload(path)
    clone_actor, clone_critic = CAGATMAPPOActor(cfg), MAPPOCentralizedCritic(cfg)
    clone_updater = CAGATMAPPORecurrentPPOUpdater(clone_actor, clone_critic, cfg)
    restored_generator = restore_model_optimizer_and_rng(config=cfg, payload=loaded, actor=clone_actor,
        critic=clone_critic, optimizers=clone_updater.optimizers, dtype=torch.float32)
    restored, version = restore_active_rollout(cfg, loaded["active_rollout_state"],
                                              checkpoint_kind="PERIODIC_RESUME", policy_version=15)
    assert version == 15 and len(restored) == 160
    assert_nested_equal(unittest.TestCase(), loaded["active_rollout_state"],
                        serialize_active_rollout(restored, version, "PERIODIC_RESUME"))
    for left, right in [(actor.state_dict(), clone_actor.state_dict()),
                        (critic.state_dict(), clone_critic.state_dict()),
                        (updater.optimizers.actor_optimizer.state_dict(), clone_updater.optimizers.actor_optimizer.state_dict()),
                        (updater.optimizers.critic_optimizer.state_dict(), clone_updater.optimizers.critic_optimizer.state_dict())]:
        assert_nested_equal(unittest.TestCase(), left, right)
    chunk = CAGATMAPPORolloutChunk((restored.transition_at(0),))
    left = CAGATMAPPOActionDistribution(actor, cfg).evaluate_actions(chunk.actor_batch, chunk.action_mask_batch, chunk.proposals, chunk.initial_hidden)
    right = CAGATMAPPOActionDistribution(clone_actor, cfg).evaluate_actions(chunk.actor_batch, chunk.action_mask_batch, chunk.proposals, chunk.initial_hidden)
    torch.testing.assert_close(left.hidden_out, right.hidden_out, rtol=0, atol=0)
    torch.testing.assert_close(left.branch_log_probs["route"], right.branch_log_probs["route"], rtol=0, atol=0)
    assert restored_generator.get_state().numel() > 0
    assert torch.equal(restored_generator.get_state(), loaded["policy_rng_state"]["generator_state"])
    # Resume the pending buffer, then compare the actual next four-epoch update
    # with the uninterrupted serialization control on identical synthetic data.
    for i in range(160, 256):
        pending.append(full.transition_at(i)); restored.append(full.transition_at(i))
    pending.finalize(); restored.finalize()
    updater.update(pending, collected_environment_steps=4096)
    clone_updater.update(restored, collected_environment_steps=4096)
    assert_nested_equal(unittest.TestCase(), actor.state_dict(), clone_actor.state_dict())
    assert_nested_equal(unittest.TestCase(), critic.state_dict(), clone_critic.state_dict())
    assert_nested_equal(unittest.TestCase(), updater.optimizers.actor_optimizer.state_dict(),
                        clone_updater.optimizers.actor_optimizer.state_dict())
    for mutate in ["missing", "feature", "seed"]:
        bad = copy.deepcopy(loaded)
        if mutate == "missing": bad.pop("route_decoder_identity")
        elif mutate == "feature": bad["route_decoder_identity"]["feature_mode"] = "unknown"
        else: bad["route_decoder_identity"]["initialization_seed"] += 1
        with pytest.raises(CheckpointError, match="identity mismatch"):
            validate_checkpoint_payload(bad)


def test_optional_telemetry_cannot_change_subsequent_sampling():
    cfg, actor, distribution, observations, batch, masks = actual_fixture()
    control = CAGATMAPPOActionDistribution(CAGATMAPPOActor(cfg), cfg)
    ids = {o.uav_id: int(o.private_queues.unbound.head_task_id) for o in observations}
    first = distribution.sample_actions(batch, masks)
    control.sample_actions(batch, masks)
    collect_route_training_records(first, environment_step=1, episode_id=0, slot=1, task_ids=ids)
    a, b = distribution.sample_actions(batch, masks), control.sample_actions(batch, masks)
    for branch in a.action_indices: assert torch.equal(a.action_indices[branch], b.action_indices[branch])


@pytest.mark.parametrize("override", [
    {"route_private_backbone": False},
    {"actor_ratio_mode": "joint"},
    {"route_training_telemetry_enabled": False},
    {"destination_supervision_mode": "offline_pretrain"},
    {"destination_oracle_rank_coefficient": 1.},
])
def test_new_combinations_do_not_bypass_p1a_or_no_pretraining_guards(override):
    with pytest.raises(ConfigError): config(**override)


@pytest.mark.parametrize("support", [
    [False, True, True, False, False, False],
    [False, True, True, False, True, False],
    [True, False, False, False, False, False],
])
def test_q_and_barrier_with_unavailable_or_single_remote(support):
    from tests.test_unified_route import route_support
    cfg, actor, distribution, observations, batch, _ = actual_fixture()
    batch, masks = route_support(batch, observations, support)
    policy = distribution.deterministic_actions(batch, masks)
    replay = distribution.evaluate_actions(batch, masks, policy.proposals, return_route_candidates=True)
    features = replay.route_candidate_features
    assert torch.isfinite(features).all() and not features.requires_grad
    if not any(support[3:]): assert not features[..., 204:408].any()
    else:
        expected = features[..., 4:5, :204].expand_as(features[..., 1:, 204:408])
        torch.testing.assert_close(features[..., 1:, 204:408], expected)
    penalty = route_exploration_loss(replay, cfg.training.mappo, 256, torch.ones(1, 1, dtype=torch.bool))
    penalty.scaled_loss.backward()
    assert torch.isfinite(penalty.scaled_loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_actual_leaf_ppo_q_gradient_and_replay():
    cfg, actor, _, _, batch, masks = actual_fixture()
    actor = actor.cuda()
    batch = replace(batch, **{name: getattr(batch, name).cuda() for name in (
        "self_features", "neighbor_public_features", "edge_features", "neighbor_mask", "episode_starts")},
        action_masks={k: v.cuda() for k, v in batch.action_masks.items()},
        action_indices={k: v.cuda() for k, v in batch.action_indices.items()})
    distribution = CAGATMAPPOActionDistribution(actor, cfg)
    sampled = distribution.sample_actions(batch, masks)
    replay = distribution.evaluate_actions(batch, masks, sampled.proposals, return_route_candidates=True)
    torch.testing.assert_close(sampled.branch_log_probs["route"], replay.branch_log_probs["route"])
    loss = ppo_loss(replay, cfg); loss.actor_loss.backward()
    assert actor.route_decoder.shared_option_scorer[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())
    assert not replay.route_candidate_features.requires_grad
    critic = MAPPOCentralizedCritic(cfg).cuda()
    legal = replay.action_masks["route"][0, 0]
    q = critic.route_option_q(torch.zeros(4, critic.spec.centralized_state_dim, device="cuda"),
                              replay.route_candidate_features[0, 0], legal)
    q.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in critic.route_option_q.parameters())
