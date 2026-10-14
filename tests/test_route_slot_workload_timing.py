"""Fix Package 3 route-slot workload timing and neutrality regressions."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from typing import Any

from src.config import (
    CHECKPOINT_KIND_PERIODIC_RESUME,
    CHECKPOINT_SCHEMA_VERSION,
    ConfigError,
    RunConfig,
    WorkloadTimingMode,
    load_run_config,
    validate_mappo_checkpoint_resume_compatibility,
)
from src.env.actions import ActionProposal
from src.env.environment import StepResult, U2UMECEnvironment
from src.env.reward import TaskWorkloadSnapshot
from src.env.tasks import Task, TaskOutcome, TaskStatus


def make_config(
    timing_mode: WorkloadTimingMode,
    *,
    deadline_slots: int = 4,
    horizon: int = 5,
) -> RunConfig:
    """Build a deterministic authentic environment with one arrival per slot."""

    base = RunConfig()
    environment = replace(
        base.environment,
        episode_horizon=horizon,
        workload_timing_mode=timing_mode,
        arrival_probabilities=(1.0, 0.0, 0.0, 0.0),
        building_layout=(),
        candidate_neighbor_radius_m=2_000.0,
        velocity_std_mps=(0.0, 0.0),
        shadowing_std_db=0.0,
        csi_error_std_db=0.0,
        fixed_csi_aoi_slots=1,
        task_data_bits_min=100.0,
        task_data_bits_max=100.0,
        task_cycles_per_bit_min=2.0,
        task_cycles_per_bit_max=2.0,
        task_deadline_factor_min=1.0,
        task_deadline_factor_max=1.0,
        minimum_task_slack_slots=deadline_slots,
        maximum_task_slack_slots=deadline_slots,
    )
    config = replace(base, seed=137, environment=environment)
    config.validate()
    return config


def ready_environment(
    timing_mode: WorkloadTimingMode,
    *,
    deadline_slots: int = 4,
) -> tuple[U2UMECEnvironment, StepResult]:
    """Advance through slot 0 so task 0 is route-eligible in slot 1."""

    environment = U2UMECEnvironment(
        make_config(timing_mode, deadline_slots=deadline_slots)
    )
    environment.reset()
    first = environment.step(environment.canonical_proposals())
    assert environment.slot == 1
    assert environment.lifecycle is not None
    assert environment.lifecycle.tasks[0].status is TaskStatus.UNBOUND
    return environment, first


def route_proposals(
    environment: U2UMECEnvironment,
    route: str | int,
) -> tuple[ActionProposal, ...]:
    proposals = list(environment.canonical_proposals())
    proposals[0] = replace(proposals[0], route=route)
    assert environment.current_observations is not None
    assert environment.current_observations[0].action_masks.is_legal(proposals[0])
    return tuple(proposals)


def local_cpu_proposals(
    environment: U2UMECEnvironment,
) -> tuple[ActionProposal, ...]:
    proposals = list(environment.canonical_proposals())
    proposals[0] = replace(
        proposals[0],
        cpu_queue="local",
        cpu_frequency=1.0,
    )
    assert environment.current_observations is not None
    assert environment.current_observations[0].action_masks.is_legal(proposals[0])
    return tuple(proposals)


def environment_rng_states(environment: U2UMECEnvironment) -> dict[str, Any]:
    assert environment.mobility_model is not None
    assert environment.channel_model is not None
    assert environment.channel_history is not None
    assert environment.traffic is not None
    return copy.deepcopy(
        {
            "mobility": environment.mobility_model.rng.bit_generator.state,
            "channel": environment.channel_model.rng.bit_generator.state,
            "csi_error": environment.channel_history.csi_error_rng.bit_generator.state,
            "arrival": environment.traffic.arrival_rng.bit_generator.state,
            "workload": environment.traffic.workload_rng.bit_generator.state,
        }
    )


def state_without_config_identity(environment: U2UMECEnvironment) -> dict[str, Any]:
    snapshot = copy.deepcopy(environment.snapshot())
    snapshot.pop("config_hash")
    return snapshot


def non_workload_reward_terms(result: StepResult) -> dict[str, int | float]:
    reward = result.info["reward"]
    excluded = {
        "urgent_workload_s",
        "normalized_workload",
        "workload_penalty",
        "reward",
    }
    return {key: value for key, value in reward.items() if key not in excluded}


class RouteSlotWorkloadTimingTests(unittest.TestCase):
    def test_config_default_enum_validation_and_legacy_canonical_identity(self) -> None:
        legacy = RunConfig()
        self.assertIs(
            legacy.environment.workload_timing_mode,
            WorkloadTimingMode.LEGACY_POST_ROUTE,
        )
        historical = asdict(legacy)
        for field_name in (
            "source_config_path",
            "cli_overrides",
            "interactive_overrides",
            "git_branch",
            "git_commit",
            "git_dirty",
        ):
            historical.pop(field_name)
        historical["environment"].pop("workload_timing_mode")
        historical["environment"].pop("channel_ablation_mode")
        # Explicit neutral defaults added by subsequent ablations/features
        # have the same identity as their historical omission.
        for field_name in (
            "channel_stochasticity_disabled", "cochannel_interference_disabled",
            "mobility_static", "workload_reference_arrival_probabilities",
        ):
            historical["environment"].pop(field_name)
        historical.pop("reward")
        historical["training"]["mappo"].pop("actor_ratio_mode")
        historical["training"]["mappo"].pop("policy_loss_aggregation_mode")
        historical["training"]["mappo"].pop("agent_credit_mode")
        historical["training"]["mappo"].pop("route_decoder_mode")
        for field_name in (
            "route_feature_mode", "destination_decoder_mode",
            "destination_oracle_rank_coefficient", "destination_supervision_mode",
            "route_actor_credit_source", "route_q_input_mode",
        ):
            historical["training"]["mappo"].pop(field_name)
        historical["training"]["mappo"].pop("tx_decoder_mode")
        historical["training"]["mappo"].pop("cpu_frequency_decoder_mode")
        historical["training"]["mappo"].pop("route_head_lr_scale")
        historical["training"]["mappo"].pop("route_private_backbone")
        historical["training"]["mappo"].pop("tx_upper_decoder_mode")
        historical["training"]["mappo"].pop("cpu_scheduler_mode")
        historical["training"]["mappo"].pop("cpu_anti_starvation_mode")
        for field_name in (
            "route_exploration_mode",
            "route_exploration_initial_coefficient",
            "route_exploration_anneal_steps",
            "route_training_telemetry_enabled",
            "route_task_credit_coefficient",
            "route_task_value_coefficient",
            "route_value_mode",
            "route_counterfactual_credit_coefficient",
            "route_q_value_loss_coefficient",
        ):
            historical["training"]["mappo"].pop(field_name)
        historical["training"]["mappo"].pop("tx_upper_optimization_mode")
        historical["training"]["mappo"].pop("tx_upper_credit_mode")
        historical["training"]["mappo"].pop("tx_q_target_normalization")
        for field_name in (
            "tx_shadow_max_pairs_per_rollout",
            "tx_shadow_step_budget_per_rollout",
            "tx_shadow_selector",
            "tx_shadow_rng_mode",
            "tx_shadow_rng_seed",
            "tx_shadow_max_conditional_draws",
            "tx_shadow_advantage_normalization",
        ):
            historical["training"]["mappo"].pop(field_name)
        historical["training"]["mappo"].pop("conditional_credit_telemetry_enabled")
        historical["training"]["mappo"].pop("conditional_credit_telemetry_max_bytes")
        historical["training"]["mappo"].pop("route_credit_mode")
        historical["training"]["mappo"].pop("route_gae_lambda")
        historical["training"]["mappo"].pop("route_only_entropy_initial_coefficient")
        historical["training"]["mappo"].pop("route_only_entropy_final_coefficient")
        historical["training"]["mappo"].pop("route_only_entropy_anneal_steps")
        historical["training"]["mappo"].pop(
            "trajectory_credit_telemetry_enabled"
        )
        historical["training"]["mappo"].pop(
            "route_diagnostic_samples_enabled"
        )
        for field_name in (
            "route_entropy_start_coefficient",
            "route_entropy_schedule_start_step",
            "route_entropy_schedule_end_step",
        ):
            historical["training"]["mappo"].pop(field_name)
        historical = json.loads(json.dumps(historical))
        self.assertEqual(legacy.resolved_dict(), historical)
        payload = json.dumps(historical, sort_keys=True, separators=(",", ":"))
        self.assertEqual(
            legacy.config_hash,
            hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        )

        parsed = load_run_config(
            cli_overrides={
                "environment.workload_timing_mode": "route_slot_pre_route"
            }
        )
        self.assertIs(
            parsed.environment.workload_timing_mode,
            WorkloadTimingMode.ROUTE_SLOT_PRE_ROUTE,
        )
        self.assertEqual(
            parsed.resolved_dict()["environment"]["workload_timing_mode"],
            "route_slot_pre_route",
        )
        self.assertNotEqual(parsed.config_hash, legacy.config_hash)
        with self.assertRaises(ConfigError):
            load_run_config(
                cli_overrides={"environment.workload_timing_mode": "unknown"}
            )

    def test_pre_route_snapshot_is_minimal_immutable_and_local_to_reward(self) -> None:
        task = Task.create(
            task_id=7,
            source_uav=0,
            data_bits=100.0,
            cpu_cycles=200.0,
            arrival_slot=0,
            deadline_slot=4,
        )
        snapshot = TaskWorkloadSnapshot.from_task(task)
        task.bind_local(binding_slot=1)

        self.assertEqual(snapshot.task_id, 7)
        self.assertEqual(snapshot.remaining_bits, 100.0)
        self.assertEqual(snapshot.remaining_cycles, 200.0)
        self.assertEqual(task.remaining_bits, 0.0)
        with self.assertRaises(FrozenInstanceError):
            snapshot.remaining_bits = 0.0  # type: ignore[misc]

    def test_same_task_route_slot_local_remote_workload_is_neutral(self) -> None:
        local, _ = ready_environment(WorkloadTimingMode.ROUTE_SLOT_PRE_ROUTE)
        remote, _ = ready_environment(WorkloadTimingMode.ROUTE_SLOT_PRE_ROUTE)
        assert local.lifecycle is not None
        assert remote.lifecycle is not None
        self.assertEqual(
            local.lifecycle.tasks[0].snapshot(),
            remote.lifecycle.tasks[0].snapshot(),
        )

        local_result = local.step(route_proposals(local, "local"))
        remote_result = remote.step(route_proposals(remote, 1))

        self.assertEqual(
            local_result.info["reward"]["urgent_workload_s"],
            remote_result.info["reward"]["urgent_workload_s"],
        )
        self.assertEqual(
            local_result.info["reward"]["workload_penalty"],
            remote_result.info["reward"]["workload_penalty"],
        )
        self.assertEqual(local_result.reward, remote_result.reward)
        self.assertEqual(
            non_workload_reward_terms(local_result),
            non_workload_reward_terms(remote_result),
        )
        self.assertEqual(local.lifecycle.tasks[0].remaining_bits, 0.0)
        self.assertEqual(remote.lifecycle.tasks[0].remaining_bits, 100.0)
        self.assertEqual(local_result.info["service"]["links"], [])
        self.assertEqual(local_result.info["service"]["cpu"], [])
        self.assertNotIn("workload_timing_mode", local_result.info)
        self.assertNotIn("pre_route_workload_component", local_result.info["reward"])

    def test_next_slot_restores_true_local_remote_physical_workload(self) -> None:
        local, _ = ready_environment(WorkloadTimingMode.ROUTE_SLOT_PRE_ROUTE)
        remote, _ = ready_environment(WorkloadTimingMode.ROUTE_SLOT_PRE_ROUTE)
        local.step(route_proposals(local, "local"))
        remote.step(route_proposals(remote, 1))

        local_next = local.step(local.canonical_proposals())
        remote_next = remote.step(remote.canonical_proposals())
        assert local.lifecycle is not None
        assert remote.lifecycle is not None
        remote_task = remote.lifecycle.tasks[0]
        denominator = remote_task.deadline_slot - 2 + 1
        expected_bit_burden = (
            remote_task.remaining_bits
            / remote.config.environment.reference_rate_bps
            / denominator
        )

        self.assertEqual(local.lifecycle.tasks[0].remaining_bits, 0.0)
        self.assertEqual(remote_task.remaining_bits, 100.0)
        self.assertAlmostEqual(
            remote_next.info["reward"]["urgent_workload_s"]
            - local_next.info["reward"]["urgent_workload_s"],
            expected_bit_burden,
        )
        self.assertGreater(
            remote_next.info["reward"]["workload_penalty"],
            local_next.info["reward"]["workload_penalty"],
        )

    def test_legacy_route_slot_values_match_post_route_formula_exactly(self) -> None:
        local, _ = ready_environment(WorkloadTimingMode.LEGACY_POST_ROUTE)
        remote, _ = ready_environment(WorkloadTimingMode.LEGACY_POST_ROUTE)
        local_result = local.step(route_proposals(local, "local"))
        remote_result = remote.step(route_proposals(remote, 1))
        assert local.lifecycle is not None
        assert remote.lifecycle is not None
        local_task = local.lifecycle.tasks[0]
        remote_task = remote.lifecycle.tasks[0]
        references = local.config.environment
        denominator = local_task.deadline_slot - 1 + 1
        expected_local = (
            local_task.remaining_cycles / references.reference_cpu_frequency_hz
        ) / denominator
        expected_remote = (
            remote_task.remaining_bits / references.reference_rate_bps
            + remote_task.remaining_cycles / references.reference_cpu_frequency_hz
        ) / denominator

        self.assertEqual(
            local_result.info["reward"]["urgent_workload_s"], expected_local
        )
        self.assertEqual(
            remote_result.info["reward"]["urgent_workload_s"], expected_remote
        )
        self.assertLess(local_result.info["reward"]["workload_penalty"], remote_result.info["reward"]["workload_penalty"])

    def test_no_new_route_binding_is_identical_between_modes(self) -> None:
        legacy, _ = ready_environment(WorkloadTimingMode.LEGACY_POST_ROUTE)
        neutral, _ = ready_environment(WorkloadTimingMode.ROUTE_SLOT_PRE_ROUTE)
        legacy.step(route_proposals(legacy, "local"))
        neutral.step(route_proposals(neutral, "local"))

        legacy_result = legacy.step(legacy.canonical_proposals())
        neutral_result = neutral.step(neutral.canonical_proposals())

        self.assertEqual(legacy_result.reward, neutral_result.reward)
        self.assertEqual(legacy_result.info["reward"], neutral_result.info["reward"])

    def test_defer_does_not_enter_special_accounting(self) -> None:
        legacy, _ = ready_environment(WorkloadTimingMode.LEGACY_POST_ROUTE)
        neutral, _ = ready_environment(WorkloadTimingMode.ROUTE_SLOT_PRE_ROUTE)

        legacy_result = legacy.step(legacy.canonical_proposals())
        neutral_result = neutral.step(neutral.canonical_proposals())
        assert legacy.lifecycle is not None
        assert neutral.lifecycle is not None

        self.assertEqual(legacy_result.reward, neutral_result.reward)
        self.assertEqual(legacy_result.info["reward"], neutral_result.info["reward"])
        self.assertIs(legacy.lifecycle.tasks[0].status, TaskStatus.UNBOUND)
        self.assertIs(neutral.lifecycle.tasks[0].status, TaskStatus.UNBOUND)
        self.assertIsNone(neutral.lifecycle.tasks[0].destination)

    def test_same_action_changes_only_workload_reward_not_state_observation_or_rng(self) -> None:
        legacy, legacy_slot0 = ready_environment(WorkloadTimingMode.LEGACY_POST_ROUTE)
        neutral, neutral_slot0 = ready_environment(WorkloadTimingMode.ROUTE_SLOT_PRE_ROUTE)
        self.assertEqual(legacy_slot0.reward, neutral_slot0.reward)
        proposals_legacy = route_proposals(legacy, "local")
        proposals_neutral = route_proposals(neutral, "local")
        self.assertEqual(proposals_legacy, proposals_neutral)

        legacy_result = legacy.step(proposals_legacy)
        neutral_result = neutral.step(proposals_neutral)

        self.assertEqual(
            non_workload_reward_terms(legacy_result),
            non_workload_reward_terms(neutral_result),
        )
        self.assertNotEqual(legacy_result.reward, neutral_result.reward)
        self.assertEqual(
            state_without_config_identity(legacy),
            state_without_config_identity(neutral),
        )
        self.assertEqual(environment_rng_states(legacy), environment_rng_states(neutral))
        self.assertEqual(
            [item.snapshot() for item in legacy_result.observations or ()],
            [item.snapshot() for item in neutral_result.observations or ()],
        )
        assert legacy_result.centralized_state is not None
        assert neutral_result.centralized_state is not None
        self.assertEqual(
            legacy_result.centralized_state.snapshot(),
            neutral_result.centralized_state.snapshot(),
        )

    def test_completion_expiration_energy_and_lifecycle_are_mode_invariant(self) -> None:
        legacy, _ = ready_environment(WorkloadTimingMode.LEGACY_POST_ROUTE)
        neutral, _ = ready_environment(WorkloadTimingMode.ROUTE_SLOT_PRE_ROUTE)
        legacy.step(route_proposals(legacy, "local"))
        neutral.step(route_proposals(neutral, "local"))
        completed_legacy = legacy.step(local_cpu_proposals(legacy))
        completed_neutral = neutral.step(local_cpu_proposals(neutral))
        assert legacy.lifecycle is not None
        assert neutral.lifecycle is not None

        self.assertEqual(completed_legacy.info["reward"], completed_neutral.info["reward"])
        self.assertEqual(completed_legacy.info["energy"], completed_neutral.info["energy"])
        self.assertGreater(completed_legacy.info["reward"]["actual_energy_j"], 0.0)
        self.assertEqual(completed_legacy.info["reward"]["completed_task_count"], 1)
        self.assertIs(legacy.lifecycle.tasks[0].outcome, TaskOutcome.DONE)
        self.assertEqual(
            legacy.lifecycle.tasks[0].snapshot(),
            neutral.lifecycle.tasks[0].snapshot(),
        )

        expired_legacy, _ = ready_environment(
            WorkloadTimingMode.LEGACY_POST_ROUTE,
            deadline_slots=2,
        )
        expired_neutral, _ = ready_environment(
            WorkloadTimingMode.ROUTE_SLOT_PRE_ROUTE,
            deadline_slots=2,
        )
        expired_legacy.step(route_proposals(expired_legacy, "local"))
        expired_neutral.step(route_proposals(expired_neutral, "local"))
        result_legacy = expired_legacy.step(expired_legacy.canonical_proposals())
        result_neutral = expired_neutral.step(expired_neutral.canonical_proposals())
        assert expired_legacy.lifecycle is not None
        assert expired_neutral.lifecycle is not None

        self.assertEqual(result_legacy.info["reward"], result_neutral.info["reward"])
        self.assertEqual(result_legacy.info["reward"]["expired_task_count"], 1)
        self.assertIs(expired_legacy.lifecycle.tasks[0].outcome, TaskOutcome.EXPIRED)
        self.assertEqual(
            expired_legacy.lifecycle.tasks[0].snapshot(),
            expired_neutral.lifecycle.tasks[0].snapshot(),
        )

    def test_checkpoint_v1_contract_accepts_legacy_default_identity(self) -> None:
        config = load_run_config(
            cli_overrides={
                "mode": "rl",
                "method_id": "ca_gat_mappo",
                "training.mappo.training_device": "cpu",
            }
        )
        self.assertNotIn(
            "workload_timing_mode",
            config.resolved_dict()["environment"],
        )
        self.assertIsNone(
            validate_mappo_checkpoint_resume_compatibility(
                config,
                schema_version=CHECKPOINT_SCHEMA_VERSION,
                checkpoint_kind=CHECKPOINT_KIND_PERIODIC_RESUME,
                method_id=config.method_id,
                git_commit=config.git_commit,
                config_hash=config.config_hash,
                training_device="cpu",
                cuda_available=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
