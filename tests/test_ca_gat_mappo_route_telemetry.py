"""Fix Package 2 gate tests for behavior-neutral route training telemetry."""

from __future__ import annotations

import json
import math
from dataclasses import replace
import unittest

import numpy as np
import torch

from src.config import RunConfig
from src.env.environment import U2UMECEnvironment
from src.models.ca_gat_mappo import ActorObservationTensorizer, ActorTensorBatch, CAGATMAPPOActor, MAPPOCentralizedCritic
from src.models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution, SequentialActionMaskBatch
from src.models.ca_gat_mappo_ppo import compute_ppo_objective_and_loss
from src.models.ca_gat_mappo_route_telemetry import (
    ROUTE_TELEMETRY_SCHEMA_VERSION,
    RouteTelemetry,
    collect_route_telemetry,
    measure_route_head_gradients,
)
from src.models.ca_gat_mappo_trainer import (
    CAGATMAPPOUpdateDiagnostics,
    _restore_update_diagnostics,
    _update_checkpoint_state,
)
from src.models.ca_gat_mappo_update import (
    CAGATMAPPORecurrentPPOUpdater,
    _prepare_device_minibatch,
    RecurrentPPOEpochDiagnostics,
    RecurrentPPOUpdateOutput,
)
from src.training_artifacts import (
    _base_record,
    _csv_text,
    _json_lines,
    _ppo_epoch_records,
    _training_summary,
    read_training_metrics_csv_text,
)
from tests.test_ca_gat_mappo_training_artifacts import training_result
from tests.test_ca_gat_mappo_update import (
    align_old_policy_snapshots,
    make_seed_transition,
    make_synthetic_buffer,
    make_update_config,
)


def make_route_config() -> RunConfig:
    base = RunConfig()
    count = base.environment.uav_count
    environment = replace(
        base.environment,
        episode_horizon=8,
        arrival_probabilities=(1.0,) * count,
        profile_assignment=("Balanced",) * count,
        profile_perturbations=((0.0, 0.0, 0.0, 0.0),) * count,
        building_layout=(),
        candidate_neighbor_radius_m=2_000.0,
        velocity_std_mps=(0.0, 0.0),
        shadowing_std_db=0.0,
        csi_error_std_db=0.0,
        fixed_csi_aoi_slots=1,
    )
    config = replace(base, environment=environment)
    config.validate()
    return config


def active_route_fixture():
    config = make_route_config()
    environment = U2UMECEnvironment(config)
    reset = environment.reset()
    first = environment.step(environment.canonical_proposals())
    assert first.observations is not None
    observations = tuple(first.observations)
    assert all(item.action_masks.route_branch_active for item in observations)
    return config, environment, reset, observations


def route_value(observation, kind: str):
    contract = observation.action_masks
    if kind in {"local", "defer", "idle"}:
        return kind
    if kind != "remote":
        raise AssertionError("unknown route fixture kind")
    return next(
        value
        for value, allowed in zip(contract.route_domain, contract.route_mask)
        if bool(allowed) and isinstance(value, int) and not isinstance(value, bool)
    )


def evaluate_route_steps(
    config: RunConfig,
    environment: U2UMECEnvironment,
    observation_steps,
    route_kinds,
    *,
    actor: CAGATMAPPOActor | None = None,
):
    resolved_actor = CAGATMAPPOActor(config).eval() if actor is None else actor
    tensorizer = ActorObservationTensorizer(config)
    proposal_steps = []
    batches = []
    for observations, kinds in zip(observation_steps, route_kinds):
        proposals = tuple(
            replace(proposal, route=route_value(observation, kind))
            for proposal, observation, kind in zip(
                environment.canonical_proposals(), observations, kinds
            )
        )
        if not all(
            observation.action_masks.is_legal(proposal)
            for observation, proposal in zip(observations, proposals)
        ):
            raise AssertionError("route telemetry fixture proposal must be legal")
        proposal_steps.append(proposals)
        batches.append(
            tensorizer.encode_step(
                observations,
                proposals,
                episode_start=False,
            )
        )
    actor_batch = ActorTensorBatch(
        self_features=torch.cat([item.self_features for item in batches], dim=1),
        neighbor_public_features=torch.cat(
            [item.neighbor_public_features for item in batches], dim=1
        ),
        edge_features=torch.cat([item.edge_features for item in batches], dim=1),
        neighbor_mask=torch.cat([item.neighbor_mask for item in batches], dim=1),
        action_masks={
            branch: torch.cat([item.action_masks[branch] for item in batches], dim=1)
            for branch in batches[0].action_masks
        },
        action_indices={
            branch: torch.cat([item.action_indices[branch] for item in batches], dim=1)
            for branch in batches[0].action_indices
        },
        episode_starts=torch.cat([item.episode_starts for item in batches], dim=1),
    )
    masks = SequentialActionMaskBatch.from_time_steps(observation_steps)
    proposals = (tuple(proposal_steps),)
    distribution = CAGATMAPPOActionDistribution(resolved_actor, config)
    with torch.no_grad():
        policy = distribution.evaluate_actions(
            actor_batch,
            masks,
            proposals,
            resolved_actor.initial_hidden(1),
        )
    return resolved_actor, actor_batch, masks, proposals, policy


def collect(
    actor,
    masks,
    proposals,
    policy,
    advantage,
    return_target,
) -> RouteTelemetry:
    return collect_route_telemetry(
        policy=policy,
        action_mask_batch=masks,
        proposals=proposals,
        advantage=torch.tensor([advantage], dtype=torch.float32),
        return_target=torch.tensor([return_target], dtype=torch.float32),
        sequence_valid_mask=torch.ones((1, len(advantage)), dtype=torch.bool),
        route_head=actor.action_heads["route"],
    )


class RouteSampleTelemetryTests(unittest.TestCase):
    def test_route_inactive_does_not_create_fake_samples(self) -> None:
        config, environment, reset, _ = active_route_fixture()
        observations = tuple(reset.observations)
        kinds = tuple("idle" for _ in observations)
        actor, _, masks, proposals, policy = evaluate_route_steps(
            config, environment, (observations,), (kinds,)
        )
        telemetry = collect(actor, masks, proposals, policy, (1.0,), (2.0,))

        self.assertEqual(telemetry.route_branch_active_count, 0)
        self.assertEqual(telemetry.legal_remote_route_count, 0)
        self.assertEqual(telemetry.route_selected_local_count, 0)
        self.assertEqual(telemetry.route_selected_remote_count, 0)
        self.assertEqual(telemetry.route_selected_defer_count, 0)
        self.assertIsNone(telemetry.route_entropy_mean)
        self.assertIsNone(telemetry.route_active_advantage_mean)

    def test_active_without_legal_remote_uses_explicit_na(self) -> None:
        config, environment, _, observations = active_route_fixture()
        modified = []
        for observation in observations:
            contract = observation.action_masks
            route_mask = np.zeros(len(contract.route_domain), dtype=np.bool_)
            route_mask[contract.route_domain.index("local")] = True
            route_mask[contract.route_domain.index("defer")] = True
            modified.append(
                replace(observation, action_masks=replace(contract, route_mask=route_mask))
            )
        modified = tuple(modified)
        kinds = tuple("local" for _ in modified)
        actor, _, masks, proposals, policy = evaluate_route_steps(
            config, environment, (modified,), (kinds,)
        )
        telemetry = collect(actor, masks, proposals, policy, (1.0,), (2.0,))

        self.assertEqual(telemetry.route_branch_active_count, len(modified))
        self.assertEqual(telemetry.legal_remote_route_count, 0)
        self.assertEqual(telemetry.route_selected_local_count, len(modified))
        self.assertIsNone(telemetry.remote_selection_rate_given_legal_remote)
        self.assertIsNone(telemetry.mean_local_probability)
        self.assertIsNone(telemetry.mean_best_remote_probability)
        self.assertIsNotNone(telemetry.route_entropy_mean)

    def test_legal_remote_selected_local_remote_and_defer_counts(self) -> None:
        config, environment, _, observations = active_route_fixture()
        kinds = ("local", "remote", "defer", "remote")
        actor, _, masks, proposals, policy = evaluate_route_steps(
            config, environment, (observations,), (kinds,)
        )
        telemetry = collect(actor, masks, proposals, policy, (1.0,), (2.0,))

        self.assertEqual(telemetry.route_branch_active_count, 4)
        self.assertEqual(telemetry.legal_remote_route_count, 4)
        self.assertEqual(telemetry.route_selected_local_count, 1)
        self.assertEqual(telemetry.route_selected_remote_count, 2)
        self.assertEqual(telemetry.route_selected_defer_count, 1)
        self.assertEqual(telemetry.remote_selection_rate_given_legal_remote, 0.5)
        self.assertEqual(telemetry.legal_remote_probability_valid_sample_count, 4)

    def test_masked_illegal_remote_is_excluded_from_best_remote(self) -> None:
        config, environment, _, observations = active_route_fixture()
        modified = []
        for observation in observations:
            contract = observation.action_masks
            remote_indices = [
                index
                for index, value in enumerate(contract.route_domain)
                if isinstance(value, int) and not isinstance(value, bool)
            ]
            self.assertGreaterEqual(len(remote_indices), 2)
            route_mask = np.zeros(len(contract.route_domain), dtype=np.bool_)
            route_mask[contract.route_domain.index("local")] = True
            route_mask[contract.route_domain.index("defer")] = True
            route_mask[remote_indices[0]] = True
            modified.append(
                replace(observation, action_masks=replace(contract, route_mask=route_mask))
            )
        modified = tuple(modified)
        actor = CAGATMAPPOActor(config).eval()
        contract = modified[0].action_masks
        remote_indices = [
            index
            for index, value in enumerate(contract.route_domain)
            if isinstance(value, int) and not isinstance(value, bool)
        ]
        with torch.no_grad():
            head = actor.action_heads["route"]
            head.weight.zero_()
            head.bias.fill_(-5.0)
            head.bias[contract.route_domain.index("local")] = 5.0
            head.bias[contract.route_domain.index("defer")] = 0.0
            head.bias[remote_indices[0]] = 2.0
            head.bias[remote_indices[1]] = 100.0
        kinds = tuple("local" for _ in modified)
        actor, _, masks, proposals, policy = evaluate_route_steps(
            config, environment, (modified,), (kinds,), actor=actor
        )
        telemetry = collect(actor, masks, proposals, policy, (1.0,), (2.0,))

        self.assertAlmostEqual(
            telemetry.mean_local_minus_best_remote_logit_margin,
            3.0,
            places=6,
        )
        illegal_probability = policy.probabilities["route"][..., remote_indices[1]]
        self.assertTrue(torch.all(illegal_probability == 0.0))
        self.assertGreater(telemetry.mean_best_remote_probability, 0.0)

    def test_advantage_and_return_attribution_groups_are_exact(self) -> None:
        config, environment, _, observations = active_route_fixture()
        steps = (observations, observations, observations)
        kinds = (
            tuple("local" for _ in observations),
            tuple("remote" for _ in observations),
            tuple("defer" for _ in observations),
        )
        actor, _, masks, proposals, policy = evaluate_route_steps(
            config, environment, steps, kinds
        )
        telemetry = collect(
            actor,
            masks,
            proposals,
            policy,
            (1.0, 2.0, 3.0),
            (10.0, 20.0, 30.0),
        )

        self.assertEqual(telemetry.route_branch_active_count, 12)
        self.assertEqual(telemetry.local_route_valid_sample_count, 4)
        self.assertEqual(telemetry.remote_route_valid_sample_count, 4)
        self.assertEqual(telemetry.defer_route_valid_sample_count, 4)
        self.assertEqual(telemetry.local_route_advantage_mean, 1.0)
        self.assertEqual(telemetry.remote_route_advantage_mean, 2.0)
        self.assertEqual(telemetry.defer_route_advantage_mean, 3.0)
        self.assertEqual(telemetry.local_route_return_mean, 10.0)
        self.assertEqual(telemetry.remote_route_return_mean, 20.0)
        self.assertEqual(telemetry.defer_route_return_mean, 30.0)
        self.assertEqual(telemetry.legal_remote_local_advantage_mean, 1.0)
        self.assertEqual(telemetry.legal_remote_remote_advantage_mean, 2.0)
        self.assertAlmostEqual(telemetry.route_active_advantage_mean, 2.0)
        self.assertAlmostEqual(
            telemetry.route_active_advantage_std,
            math.sqrt(2.0 / 3.0),
            places=6,
        )


class RouteGradientAndStabilityTelemetryTests(unittest.TestCase):
    def test_route_head_gradient_none_zero_and_finite_nonzero_are_distinct(self) -> None:
        head = torch.nn.Linear(3, 4)
        missing = measure_route_head_gradients(head)
        self.assertEqual(missing.weight.status, "none")
        self.assertIsNone(missing.total.norm)

        head.weight.grad = torch.zeros_like(head.weight)
        head.bias.grad = torch.zeros_like(head.bias)
        zero = measure_route_head_gradients(head)
        self.assertEqual(zero.weight.status, "zero")
        self.assertEqual(zero.bias.status, "zero")
        self.assertEqual(zero.total.status, "zero")
        self.assertEqual(zero.total.norm, 0.0)

        head.weight.grad = torch.ones_like(head.weight)
        head.bias.grad = torch.zeros_like(head.bias)
        nonzero = measure_route_head_gradients(head)
        self.assertEqual(nonzero.weight.status, "finite_nonzero")
        self.assertEqual(nonzero.bias.status, "zero")
        self.assertEqual(nonzero.total.status, "finite_nonzero")
        self.assertAlmostEqual(nonzero.total.norm, math.sqrt(head.weight.numel()), places=6)

    def test_approx_kl_and_clip_fraction_reuse_current_ppo_statistics(self) -> None:
        ratio = torch.tensor([[1.2, 0.8]], dtype=torch.float32)
        output = compute_ppo_objective_and_loss(
            new_joint_log_prob=torch.log(ratio),
            old_joint_log_prob=torch.zeros_like(ratio),
            advantage=torch.tensor([1.0]),
            current_value=torch.tensor([0.0]),
            return_target=torch.tensor([1.0]),
            entropy=torch.zeros_like(ratio),
            sequence_valid_mask=torch.tensor([True]),
            epsilon_clip=0.1,
            value_coefficient=0.5,
            entropy_coefficient=0.01,
        )
        expected = torch.mean((ratio - 1.0) - torch.log(ratio)).item()
        self.assertAlmostEqual(output.diagnostics.approx_kl, expected, places=7)
        self.assertEqual(output.diagnostics.clipped_fraction, 1.0)


class RouteTelemetryNeutralityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = make_update_config()
        seed_transition = make_seed_transition(cls.config)
        raw = make_synthetic_buffer(cls.config, seed_transition)
        cls.buffer = align_old_policy_snapshots(cls.config, raw)

    def test_enabled_disabled_sampling_rng_and_update_parameters_are_bitwise_equal(self) -> None:
        actor_enabled = CAGATMAPPOActor(self.config)
        critic_enabled = MAPPOCentralizedCritic(self.config)
        actor_disabled = CAGATMAPPOActor(self.config)
        critic_disabled = MAPPOCentralizedCritic(self.config)
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(actor_enabled.parameters(), actor_disabled.parameters())
            )
        )
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(critic_enabled.parameters(), critic_disabled.parameters())
            )
        )
        enabled = CAGATMAPPORecurrentPPOUpdater(
            actor_enabled,
            critic_enabled,
            self.config,
            route_telemetry_enabled=True,
        )
        disabled = CAGATMAPPORecurrentPPOUpdater(
            actor_disabled,
            critic_disabled,
            self.config,
            route_telemetry_enabled=False,
        )
        from src.models.ca_gat_mappo_update import build_recurrent_ppo_minibatch

        minibatch = build_recurrent_ppo_minibatch(self.buffer, self.config)
        first_generator = torch.Generator().manual_seed(123456)
        second_generator = torch.Generator().manual_seed(123456)
        with torch.no_grad():
            first_actions = enabled.action_distribution.sample_actions(
                minibatch.actor_batch,
                minibatch.action_mask_batch,
                minibatch.initial_hidden,
                generator=first_generator,
            )
            second_actions = disabled.action_distribution.sample_actions(
                minibatch.actor_batch,
                minibatch.action_mask_batch,
                minibatch.initial_hidden,
                generator=second_generator,
            )
        self.assertEqual(first_actions.proposals, second_actions.proposals)
        for branch in first_actions.action_indices:
            self.assertTrue(
                torch.equal(
                    first_actions.action_indices[branch],
                    second_actions.action_indices[branch],
                )
            )
        self.assertTrue(torch.equal(first_generator.get_state(), second_generator.get_state()))

        original_global_rng = torch.get_rng_state().clone()
        try:
            torch.manual_seed(987654)
            common_rng = torch.get_rng_state().clone()
            enabled_output = enabled.update(self.buffer)
            enabled_rng = torch.get_rng_state().clone()
            torch.set_rng_state(common_rng)
            disabled_output = disabled.update(self.buffer)
            disabled_rng = torch.get_rng_state().clone()
        finally:
            torch.set_rng_state(original_global_rng)

        self.assertTrue(torch.equal(enabled_rng, disabled_rng))
        for left, right in zip(actor_enabled.parameters(), actor_disabled.parameters()):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(critic_enabled.parameters(), critic_disabled.parameters()):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(actor_enabled.parameters(), actor_disabled.parameters()):
            if left.grad is None or right.grad is None:
                self.assertIs(left.grad, right.grad)
            else:
                self.assertTrue(torch.equal(left.grad, right.grad))
        for left, right in zip(critic_enabled.parameters(), critic_disabled.parameters()):
            if left.grad is None or right.grad is None:
                self.assertIs(left.grad, right.grad)
            else:
                self.assertTrue(torch.equal(left.grad, right.grad))
        for optimizer_name in ("actor_optimizer", "critic_optimizer"):
            enabled_state = getattr(enabled.optimizers, optimizer_name).state_dict()
            disabled_state = getattr(disabled.optimizers, optimizer_name).state_dict()
            self.assertEqual(enabled_state["param_groups"], disabled_state["param_groups"])
            self.assertEqual(enabled_state["state"].keys(), disabled_state["state"].keys())
            for parameter_id in enabled_state["state"]:
                self.assertEqual(enabled_state["state"][parameter_id].keys(), disabled_state["state"][parameter_id].keys())
                for name, value in enabled_state["state"][parameter_id].items():
                    other = disabled_state["state"][parameter_id][name]
                    if torch.is_tensor(value):
                        self.assertTrue(torch.equal(value, other))
                    else:
                        self.assertEqual(value, other)

        fields = (
            "actor_loss",
            "critic_loss",
            "entropy_mean",
            "total_loss",
            "ratio_mean",
            "actor_grad_norm_before_clip",
            "critic_grad_norm_before_clip",
            "clip_max_norm",
            "approx_kl",
            "clip_fraction",
        )
        for left, right in zip(
            enabled_output.epoch_diagnostics,
            disabled_output.epoch_diagnostics,
        ):
            for name in fields:
                self.assertEqual(getattr(left, name), getattr(right, name))
            self.assertIsNotNone(left.route_telemetry)
            self.assertIsNone(right.route_telemetry)


    def test_enabled_disabled_full_joint_ratio_tensor_is_identical(self) -> None:
        actor_enabled = CAGATMAPPOActor(self.config)
        critic_enabled = MAPPOCentralizedCritic(self.config)
        actor_disabled = CAGATMAPPOActor(self.config)
        critic_disabled = MAPPOCentralizedCritic(self.config)
        enabled = CAGATMAPPORecurrentPPOUpdater(
            actor_enabled, critic_enabled, self.config, route_telemetry_enabled=True
        )
        disabled = CAGATMAPPORecurrentPPOUpdater(
            actor_disabled, critic_disabled, self.config, route_telemetry_enabled=False
        )
        from src.models.ca_gat_mappo_update import build_recurrent_ppo_minibatch

        minibatch = build_recurrent_ppo_minibatch(self.buffer, self.config)
        prepared_enabled = _prepare_device_minibatch(
            minibatch, enabled.device, enabled.dtype
        )
        prepared_disabled = _prepare_device_minibatch(
            minibatch, disabled.device, disabled.dtype
        )
        with torch.no_grad():
            evaluation_enabled = enabled._evaluate_prepared(prepared_enabled)
            evaluation_disabled = disabled._evaluate_prepared(prepared_disabled)
            ratio_enabled = torch.exp(
                evaluation_enabled.policy.joint_log_prob
                - prepared_enabled.old_joint_log_prob
            )
            ratio_disabled = torch.exp(
                evaluation_disabled.policy.joint_log_prob
                - prepared_disabled.old_joint_log_prob
            )
        self.assertTrue(torch.equal(ratio_enabled, ratio_disabled))

    def test_production_update_emits_all_branch_stats_and_rollout_activity(self) -> None:
        actor = CAGATMAPPOActor(self.config)
        critic = MAPPOCentralizedCritic(self.config)
        updater = CAGATMAPPORecurrentPPOUpdater(
            actor,
            critic,
            self.config,
            route_telemetry_enabled=True,
        )
        output = updater.update(self.buffer)
        telemetry = output.epoch_diagnostics[0].route_telemetry
        self.assertIsNotNone(telemetry)
        self.assertEqual(telemetry.actor_ratio_mode, "joint")
        self.assertEqual(
            tuple(telemetry.branch_ppo_dynamics.groups),
            tuple(self.config.action.sampling_order),
        )
        self.assertEqual(len(telemetry.active_branch_matrix), 256)
        self.assertTrue(
            all(
                len(time_step) == self.config.environment.uav_count
                and all(len(agent_row) == 7 for agent_row in time_step)
                for time_step in telemetry.active_branch_matrix
            )
        )
        route = telemetry.branch_ppo_dynamics.groups["route"]
        self.assertEqual(route.active_count, telemetry.route_branch_active_count)
        if route.active_count:
            self.assertIsNotNone(route.ratio_mean)
            self.assertIsNotNone(route.clip_fraction)
            self.assertIsNotNone(route.surrogate_contribution_mean)


class RouteTelemetryPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config, environment, _, observations = active_route_fixture()
        kinds = ("local", "remote", "defer", "remote")
        actor, _, masks, proposals, policy = evaluate_route_steps(
            config, environment, (observations,), (kinds,)
        )
        cls.config = config
        cls.telemetry = collect(actor, masks, proposals, policy, (1.0,), (2.0,))

    def test_serialization_is_deterministic_and_na_is_json_null_csv_blank(self) -> None:
        record = _base_record(self.config, "insufficient-horizon")
        record.update(
            {
                "record_type": "ppo_epoch",
                "telemetry_scope": "per_ppo_epoch",
                "series_index": 1,
                "route_telemetry_enabled": True,
            }
        )
        telemetry_record = self.telemetry.record()
        telemetry_record["route_telemetry_schema_version"] = telemetry_record.pop(
            "schema_version"
        )
        record.update(telemetry_record)
        self.assertEqual(_csv_text([record]), _csv_text([record]))
        self.assertEqual(_json_lines([record]), _json_lines([record]))
        decoded = json.loads(_json_lines([record]))
        self.assertEqual(
            decoded["route_telemetry_schema_version"],
            ROUTE_TELEMETRY_SCHEMA_VERSION,
        )
        self.assertIsNone(decoded["route_head_weight_grad_norm"])

    def test_legacy_csv_and_checkpoint_diagnostics_remain_readable(self) -> None:
        legacy = "run_id,record_type,series_index\nlegacy,ppo_epoch,1\n"
        row = read_training_metrics_csv_text(legacy)[0]
        self.assertEqual(row["diagnostics_schema_version"], "1")
        self.assertEqual(row["route_entropy_mean"], "")

        epochs = tuple(
            replace(
                epoch,
                approx_kl=0.01 * (index + 1),
                clip_fraction=0.10 * index,
                route_telemetry=self.telemetry,
            )
            for index, epoch in enumerate(
                training_result().updates[0].output.epoch_diagnostics
            )
        )
        base_training = training_result()
        output = replace(base_training.updates[0].output, epoch_diagnostics=epochs)
        update_with_telemetry = replace(base_training.updates[0], output=output)
        training = replace(base_training, updates=(update_with_telemetry,))
        records = _ppo_epoch_records(
            self.config,
            training,
            "insufficient-horizon",
        )
        self.assertEqual(len(records), 4)
        self.assertTrue(all(item["route_telemetry_enabled"] for item in records))
        self.assertTrue(
            all(
                item["route_telemetry_schema_version"]
                == ROUTE_TELEMETRY_SCHEMA_VERSION
                for item in records
            )
        )
        self.assertEqual(
            records[0]["route_entropy_rolling_mean"],
            records[0]["route_entropy_mean"],
        )
        summary = _training_summary(
            self.config,
            training,
            {},
            "pass",
            "insufficient-horizon",
            "metrics.csv",
        )
        route_summary = summary["route_telemetry"]
        self.assertEqual(route_summary["per_ppo_epoch"]["present_count"], 4)
        self.assertEqual(
            route_summary["final_summary"]["unique_rollout_sample_totals"]["route_branch_active_count"],
            self.telemetry.route_branch_active_count,
        )

        epochs = tuple(
            RecurrentPPOEpochDiagnostics(
                epoch_index=index,
                actor_loss=1.0,
                critic_loss=2.0,
                entropy_mean=0.5,
                total_loss=3.0,
                ratio_mean=1.0,
                actor_grad_norm_before_clip=4.0,
                critic_grad_norm_before_clip=5.0,
                clip_max_norm=0.5,
                approx_kl=0.01,
                clip_fraction=0.25,
                route_telemetry=self.telemetry,
            )
            for index in range(4)
        )
        update = CAGATMAPPOUpdateDiagnostics(
            update_index=0,
            rollout_policy_version=0,
            policy_version_after_update=1,
            output=RecurrentPPOUpdateOutput(
                epoch_diagnostics=epochs,
                chunk_count=8,
                chunk_length=32,
                valid_transition_count=256,
                old_policy_snapshot_preserved=True,
            ),
        )
        state = _update_checkpoint_state(update)
        persisted_epoch = state["output"]["epoch_diagnostics"][0]
        self.assertNotIn("route_telemetry", persisted_epoch)
        self.assertNotIn("approx_kl", persisted_epoch)
        restored = _restore_update_diagnostics(state)
        self.assertTrue(
            all(item.route_telemetry is None for item in restored.output.epoch_diagnostics)
        )


if __name__ == "__main__":
    unittest.main()
