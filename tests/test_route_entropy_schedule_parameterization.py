"""Route-specific entropy schedule parameterization implementation gates."""

from __future__ import annotations

import io
import math
import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from src.config import (
    ActorRatioMode,
    AgentCreditMode,
    CHECKPOINT_KIND_PERIODIC_RESUME,
    CHECKPOINT_V1_TRAINING_STATE_FIELDS,
    ConfigError,
    RouteDecoderMode,
    RunConfig,
    WorkloadTimingMode,
    compute_route_entropy_schedule,
    load_run_config,
)
from src.models.ca_gat_mappo_ppo import (
    compute_configured_ppo_objective_and_loss,
    compute_ppo_objective_and_loss,
)
from src.models.ca_gat_mappo_trainer import CAGATMAPPOTrainer
from src.models.ca_gat_mappo_update import (
    RecurrentPPOEpochDiagnostics,
    RecurrentPPOUpdateOutput,
    RecurrentPPOUpdateError,
)
from src.training_artifacts import (
    TRAINING_DIAGNOSTICS_SCHEMA_VERSION,
    TRAINING_METRIC_COLUMNS,
    _csv_text,
    _json_lines,
    _ppo_epoch_records,
    read_training_metrics_csv_text,
)


EXPERIMENT_CONFIG_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "tests" / "fixtures" / "configs"
)
ENTROPY_ABLATION_CONFIG_PATHS = {
    "B0": EXPERIMENT_CONFIG_DIRECTORY
    / "cooperative-load-v2-hierarchical-optionaware-v1-entropy-b0-32k.json",
    "A": EXPERIMENT_CONFIG_DIRECTORY
    / "cooperative-load-v2-hierarchical-optionaware-v1-entropy-a-32k.json",
    "B": EXPERIMENT_CONFIG_DIRECTORY
    / "cooperative-load-v2-hierarchical-optionaware-v1-entropy-b-32k.json",
}
ENTROPY_TREATMENT_FIELDS = {
    "entropy_coefficient_schedule_enabled",
    "route_entropy_start_coefficient",
    "route_entropy_schedule_start_step",
    "route_entropy_schedule_end_step",
}


def frozen_control_config() -> RunConfig:
    """Return the frozen control without enabling the route intervention."""

    base = RunConfig()
    environment = replace(
        base.environment,
        workload_timing_mode=WorkloadTimingMode.ROUTE_SLOT_PRE_ROUTE,
        candidate_neighbor_radius_m=525.0,
    )
    mappo = replace(
        base.training.mappo,
        actor_ratio_mode=ActorRatioMode.BRANCH_SPECIFIC,
        agent_credit_mode=AgentCreditMode.ROLE_DECOMPOSED,
        training_device="cpu",
    )
    config = replace(
        base,
        mode="rl",
        method_id="ca_gat_mappo",
        environment=environment,
        training=replace(
            base.training,
            mappo=mappo,
            formal_rl_enabled=True,
        ),
    )
    config.validate()
    return config


def treatment_config(start_coefficient: float = 0.03) -> RunConfig:
    control = frozen_control_config()
    config = replace(
        control,
        training=replace(
            control.training,
            mappo=replace(
                control.training.mappo,
                entropy_coefficient_schedule_enabled=True,
                route_entropy_start_coefficient=start_coefficient,
            ),
        ),
    )
    config.validate()
    return config


def branch_specific_loss_inputs() -> dict[str, torch.Tensor]:
    time_steps = 2
    agent_count = 2
    branch_count = 7
    return {
        "new_joint_log_prob": torch.zeros((time_steps, agent_count)),
        "old_joint_log_prob": torch.zeros((time_steps, agent_count)),
        "advantage": torch.zeros((time_steps, agent_count)),
        "current_value": torch.zeros((time_steps, agent_count)),
        "return_target": torch.zeros((time_steps, agent_count)),
        "sequence_valid_mask": torch.ones(time_steps, dtype=torch.bool),
        "new_branch_log_probs": torch.zeros(
            (time_steps, agent_count, branch_count)
        ),
        "old_branch_log_probs": torch.zeros(
            (time_steps, agent_count, branch_count)
        ),
        "active_branch_indicators": torch.ones(
            (time_steps, agent_count, branch_count),
            dtype=torch.bool,
        ),
    }


def fixed_update_output() -> RecurrentPPOUpdateOutput:
    epochs = tuple(
        RecurrentPPOEpochDiagnostics(
            epoch_index=index,
            actor_loss=-0.1,
            critic_loss=0.2,
            entropy_mean=0.5,
            total_loss=-0.005,
            ratio_mean=1.0,
            actor_grad_norm_before_clip=0.3,
            critic_grad_norm_before_clip=0.4,
            clip_max_norm=0.5,
        )
        for index in range(4)
    )
    return RecurrentPPOUpdateOutput(
        epoch_diagnostics=epochs,
        chunk_count=8,
        chunk_length=32,
        valid_transition_count=256,
        old_policy_snapshot_preserved=True,
    )


class FakeFullBuffer:
    def __init__(self, length: int = 256) -> None:
        self.length = length
        self.full = length == 256
        self.finalized = False

    def __len__(self) -> int:
        return self.length

    def finalize(self) -> None:
        self.finalized = True

    def clear(self) -> None:
        self.length = 0
        self.full = False
        self.finalized = False


class RouteEntropyScheduleBoundaryTests(unittest.TestCase):
    def test_configured_boundary_values_and_discontinuity(self) -> None:
        for peak in (0.03, 0.05):
            with self.subTest(peak=peak):
                mappo = treatment_config(peak).training.mappo
                delta = peak - 0.01
                expected = {
                    0: 0.01,
                    3071: 0.01,
                    3072: peak,
                    3073: 0.01
                    + delta * (32768 - 3073) / (32768 - 3072),
                    32767: 0.01
                    + delta * (32768 - 32767) / (32768 - 3072),
                    32768: 0.01,
                    40000: 0.01,
                }
                for step, coefficient in expected.items():
                    point = compute_route_entropy_schedule(mappo, step)
                    self.assertAlmostEqual(
                        point.coefficient,
                        coefficient,
                        places=15,
                    )

                self.assertEqual(
                    compute_route_entropy_schedule(mappo, 3071).coefficient,
                    0.01,
                )
                self.assertAlmostEqual(
                    compute_route_entropy_schedule(mappo, 3072).coefficient,
                    peak,
                    places=15,
                )
                self.assertEqual(
                    compute_route_entropy_schedule(mappo, 32768).coefficient,
                    0.01,
                )
                self.assertEqual(
                    compute_route_entropy_schedule(mappo, 3072).progress,
                    0.0,
                )
                self.assertEqual(
                    compute_route_entropy_schedule(mappo, 32768).progress,
                    1.0,
                )

    def test_disabled_schedule_is_base_only_at_every_boundary(self) -> None:
        mappo = frozen_control_config().training.mappo
        for step in (0, 3071, 3072, 3073, 32767, 32768, 50000):
            with self.subTest(step=step):
                point = compute_route_entropy_schedule(mappo, step)
                self.assertFalse(point.enabled)
                self.assertEqual(point.coefficient, 0.01)
                self.assertEqual(point.progress, 0.0)


class RouteOnlyEntropyLossTests(unittest.TestCase):
    def test_route_only_coefficient_and_other_branch_isolation(self) -> None:
        for peak in (0.03, 0.05):
            with self.subTest(peak=peak):
                route_entropy = torch.full((2, 2), 0.5, requires_grad=True)
                other_entropy = torch.full((2, 2), 2.0, requires_grad=True)
                inputs = branch_specific_loss_inputs()
                output = compute_configured_ppo_objective_and_loss(
                    **inputs,
                    entropy=route_entropy + other_entropy,
                    route_entropy=route_entropy,
                    collected_environment_steps=3072,
                    config=treatment_config(peak),
                )

                self.assertAlmostEqual(
                    output.route_entropy_coefficient,
                    peak,
                    places=15,
                )
                torch.testing.assert_close(
                    output.route_entropy_loss_contribution,
                    torch.tensor(peak * 0.5),
                )
                torch.testing.assert_close(
                    output.other_branch_entropy_loss_contribution,
                    torch.tensor(0.02),
                )
                torch.testing.assert_close(
                    output.global_entropy_loss_contribution,
                    torch.tensor(peak * 0.5 + 0.02),
                )
                output.total_loss.backward()
                torch.testing.assert_close(
                    route_entropy.grad,
                    torch.full((2, 2), -peak / 4.0),
                )
                torch.testing.assert_close(
                    other_entropy.grad,
                    torch.full((2, 2), -0.01 / 4.0),
                )

    def test_intervention_changes_only_entropy_regularization(self) -> None:
        route_entropy = torch.full((2, 2), 0.5)
        other_entropy = torch.full((2, 2), 2.0)
        treatment = compute_configured_ppo_objective_and_loss(
            **branch_specific_loss_inputs(),
            entropy=route_entropy + other_entropy,
            route_entropy=route_entropy,
            collected_environment_steps=3072,
            config=treatment_config(),
        )
        control = compute_configured_ppo_objective_and_loss(
            **branch_specific_loss_inputs(),
            entropy=route_entropy + other_entropy,
            route_entropy=route_entropy,
            collected_environment_steps=3072,
            config=frozen_control_config(),
        )

        self.assertTrue(torch.equal(treatment.surrogate, control.surrogate))
        self.assertTrue(
            torch.equal(treatment.clipped_objective, control.clipped_objective)
        )
        self.assertTrue(torch.equal(treatment.actor_loss, control.actor_loss))
        self.assertTrue(torch.equal(treatment.critic_loss, control.critic_loss))
        self.assertTrue(torch.equal(treatment.entropy_mean, control.entropy_mean))
        torch.testing.assert_close(
            treatment.total_loss - control.total_loss,
            torch.tensor(-0.02 * 0.5),
        )


    def test_inactive_route_has_zero_contribution(self) -> None:
        route_source = torch.full((2, 2), 0.5, requires_grad=True)
        inactive = torch.zeros((2, 2), dtype=torch.bool)
        route_entropy = torch.where(
            inactive,
            route_source,
            torch.zeros_like(route_source),
        )
        other_entropy = torch.full((2, 2), 2.0, requires_grad=True)
        output = compute_configured_ppo_objective_and_loss(
            **branch_specific_loss_inputs(),
            entropy=route_entropy + other_entropy,
            route_entropy=route_entropy,
            collected_environment_steps=3072,
            config=treatment_config(),
        )

        self.assertEqual(
            float(output.route_entropy_loss_contribution.detach().item()),
            0.0,
        )
        torch.testing.assert_close(
            output.other_branch_entropy_loss_contribution,
            torch.tensor(0.02),
        )
        torch.testing.assert_close(
            output.global_entropy_loss_contribution,
            output.other_branch_entropy_loss_contribution,
        )
        output.total_loss.backward()
        torch.testing.assert_close(
            route_source.grad,
            torch.zeros_like(route_source),
        )
        torch.testing.assert_close(
            other_entropy.grad,
            torch.full((2, 2), -0.01 / 4.0),
        )


    def test_treatment_base_regions_are_loss_and_gradient_equivalent(self) -> None:
        config = treatment_config()
        for step in (0, 32768, 50000):
            with self.subTest(step=step):
                scheduled_entropy = torch.full(
                    (2, 2), 1.25, requires_grad=True
                )
                legacy_entropy = scheduled_entropy.detach().clone().requires_grad_()
                route_entropy = scheduled_entropy * 0.2
                scheduled_inputs = branch_specific_loss_inputs()
                legacy_inputs = {
                    name: value.detach().clone()
                    for name, value in scheduled_inputs.items()
                }
                scheduled = compute_configured_ppo_objective_and_loss(
                    **scheduled_inputs,
                    entropy=scheduled_entropy,
                    route_entropy=route_entropy,
                    collected_environment_steps=step,
                    config=config,
                )
                mappo = config.training.mappo
                legacy = compute_ppo_objective_and_loss(
                    **legacy_inputs,
                    entropy=legacy_entropy,
                    epsilon_clip=mappo.ppo_clip_epsilon,
                    value_coefficient=mappo.value_coefficient,
                    entropy_coefficient=mappo.entropy_coefficient,
                    actor_ratio_mode=mappo.actor_ratio_mode,
                    agent_credit_mode=mappo.agent_credit_mode,
                )
                self.assertTrue(torch.equal(scheduled.total_loss, legacy.total_loss))
                scheduled_gradient = torch.autograd.grad(
                    scheduled.total_loss,
                    scheduled_entropy,
                )[0]
                legacy_gradient = torch.autograd.grad(
                    legacy.total_loss,
                    legacy_entropy,
                )[0]
                self.assertTrue(torch.equal(scheduled_gradient, legacy_gradient))

    def test_disabled_path_preserves_loss_gradient_and_rng(self) -> None:
        config = frozen_control_config()
        configured_entropy = torch.full((2, 2), 1.25, requires_grad=True)
        legacy_entropy = configured_entropy.detach().clone().requires_grad_()
        configured_inputs = branch_specific_loss_inputs()
        legacy_inputs = {
            name: value.detach().clone()
            for name, value in configured_inputs.items()
        }
        before = torch.random.get_rng_state().clone()
        configured = compute_configured_ppo_objective_and_loss(
            **configured_inputs,
            entropy=configured_entropy,
            route_entropy=configured_entropy * 0.2,
            collected_environment_steps=3072,
            config=config,
        )
        mappo = config.training.mappo
        legacy = compute_ppo_objective_and_loss(
            **legacy_inputs,
            entropy=legacy_entropy,
            epsilon_clip=mappo.ppo_clip_epsilon,
            value_coefficient=mappo.value_coefficient,
            entropy_coefficient=mappo.entropy_coefficient,
            actor_ratio_mode=mappo.actor_ratio_mode,
            agent_credit_mode=mappo.agent_credit_mode,
        )
        after = torch.random.get_rng_state().clone()

        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(configured.actor_loss, legacy.actor_loss))
        self.assertTrue(torch.equal(configured.entropy_mean, legacy.entropy_mean))
        self.assertTrue(torch.equal(configured.total_loss, legacy.total_loss))
        configured_gradient = torch.autograd.grad(
            configured.total_loss,
            configured_entropy,
        )[0]
        legacy_gradient = torch.autograd.grad(
            legacy.total_loss,
            legacy_entropy,
        )[0]
        self.assertTrue(torch.equal(configured_gradient, legacy_gradient))


class RouteEntropyConfigAndResumeTests(unittest.TestCase):
    def test_canonical_identity_and_frozen_treatment(self) -> None:
        control = frozen_control_config()
        treatment = treatment_config()
        control_mappo = control.resolved_dict()["training"]["mappo"]
        treatment_mappo = treatment.resolved_dict()["training"]["mappo"]

        self.assertNotIn("route_entropy_start_coefficient", control_mappo)
        self.assertNotIn("route_entropy_schedule_start_step", control_mappo)
        self.assertNotIn("route_entropy_schedule_end_step", control_mappo)
        self.assertFalse(control_mappo["entropy_coefficient_schedule_enabled"])
        self.assertEqual(treatment_mappo["route_entropy_start_coefficient"], 0.03)
        self.assertEqual(treatment_mappo["route_entropy_schedule_start_step"], 3072)
        self.assertEqual(treatment_mappo["route_entropy_schedule_end_step"], 32768)
        self.assertNotEqual(control.config_hash, treatment.config_hash)
        self.assertNotEqual(treatment.config_hash[:12], "d4e08b4cf5ba")

        self.assertEqual(treatment.environment.candidate_neighbor_radius_m, 525.0)
        self.assertEqual(
            treatment.environment.workload_timing_mode,
            WorkloadTimingMode.ROUTE_SLOT_PRE_ROUTE,
        )
        self.assertEqual(
            treatment.training.mappo.actor_ratio_mode,
            ActorRatioMode.BRANCH_SPECIFIC,
        )
        self.assertEqual(
            treatment.training.mappo.agent_credit_mode,
            AgentCreditMode.ROLE_DECOMPOSED,
        )
        self.assertEqual(treatment.training.mappo.entropy_coefficient, 0.01)

    def test_preregistered_parameters_cannot_be_changed(self) -> None:
        config = treatment_config()
        invalid_values = {
            "entropy_coefficient": 0.02,
            "route_entropy_start_coefficient": 0.02,
            "route_entropy_schedule_start_step": 3073,
            "route_entropy_schedule_end_step": 32767,
        }
        for name, value in invalid_values.items():
            with self.subTest(name=name):
                invalid = replace(
                    config,
                    training=replace(
                        config.training,
                        mappo=replace(
                            config.training.mappo,
                            **{name: value},
                        ),
                    ),
                )
                with self.assertRaisesRegex(ConfigError, "preregistered"):
                    invalid.validate()

    def test_32k_ablation_configs_are_valid_and_matched(self) -> None:
        configs = {
            name: load_run_config(path)
            for name, path in ENTROPY_ABLATION_CONFIG_PATHS.items()
        }
        for name, config in configs.items():
            with self.subTest(group=name):
                self.assertEqual(config.scenario_id, "small")
                self.assertEqual(config.seed, 42)
                self.assertEqual(
                    config.training.mappo.max_training_environment_steps,
                    32000,
                )
                self.assertEqual(
                    config.training.mappo.route_decoder_mode,
                    RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1,
                )
                self.assertEqual(config.training.mappo.entropy_coefficient, 0.01)

        self.assertFalse(
            configs["B0"].training.mappo.entropy_coefficient_schedule_enabled
        )
        for name, peak in (("A", 0.03), ("B", 0.05)):
            with self.subTest(group=name):
                mappo = configs[name].training.mappo
                self.assertTrue(mappo.entropy_coefficient_schedule_enabled)
                self.assertEqual(mappo.route_entropy_start_coefficient, peak)
                self.assertEqual(mappo.route_entropy_schedule_start_step, 3072)
                self.assertEqual(mappo.route_entropy_schedule_end_step, 32768)
                self.assertAlmostEqual(
                    compute_route_entropy_schedule(mappo, 3072).coefficient,
                    peak,
                    places=15,
                )
                self.assertEqual(
                    compute_route_entropy_schedule(mappo, 32768).coefficient,
                    0.01,
                )

        normalized = {}
        for name, config in configs.items():
            payload = json.loads(json.dumps(config.resolved_dict()))
            mappo = payload["training"]["mappo"]
            for field in ENTROPY_TREATMENT_FIELDS:
                mappo.pop(field, None)
            normalized[name] = payload
        self.assertEqual(normalized["B0"], normalized["A"])
        self.assertEqual(normalized["B0"], normalized["B"])

    def test_checkpoint_state_reconstructs_schedule_without_extra_state(self) -> None:
        trainer = object.__new__(CAGATMAPPOTrainer)
        trainer.rollout_buffer = FakeFullBuffer(length=0)
        trainer._rollout_policy_version = None
        trainer.policy_version = 12
        trainer._episode_seeds = []
        trainer._completed_episodes = 0
        trainer._next_episode_index = 0
        trainer._transitions = 3072
        trainer._optimized = 3072
        trainer._updates = [object()] * 12
        trainer._unused_final_tail = 0
        trainer._training_complete = False

        saved = trainer._training_checkpoint_state(
            CHECKPOINT_KIND_PERIODIC_RESUME
        )
        checkpoint_stream = io.BytesIO()
        torch.save(saved, checkpoint_stream)
        checkpoint_stream.seek(0)
        restored = torch.load(checkpoint_stream, weights_only=False)
        self.assertEqual(tuple(restored), CHECKPOINT_V1_TRAINING_STATE_FIELDS)
        self.assertNotIn("route_entropy_coefficient", restored)
        before = compute_route_entropy_schedule(
            treatment_config().training.mappo,
            saved["collected_environment_transitions"],
        )
        after = compute_route_entropy_schedule(
            treatment_config().training.mappo,
            restored["collected_environment_transitions"],
        )
        self.assertEqual(before, after)
        self.assertEqual(after.coefficient, 0.03)


class RouteEntropyTrainerAndTelemetryTests(unittest.TestCase):
    def test_trainer_forwards_true_step_only_when_enabled(self) -> None:
        class TreatmentUpdater:
            def __init__(self) -> None:
                self.received = None

            def update(self, _buffer, *, collected_environment_steps):
                self.received = collected_environment_steps
                return fixed_update_output()

        updater = TreatmentUpdater()
        trainer = object.__new__(CAGATMAPPOTrainer)
        trainer.config = treatment_config()
        trainer.rollout_buffer = FakeFullBuffer()
        trainer._rollout_policy_version = 0
        trainer.policy_version = 0
        trainer.policy_generator = torch.Generator().manual_seed(123)
        trainer.updater = updater
        trainer._transitions = 3072
        updates = []
        trainer._perform_update(updates)
        self.assertEqual(updater.received, 3072)
        self.assertEqual(len(updates), 1)

        class LegacyUpdater:
            def __init__(self) -> None:
                self.called = False

            def update(self, _buffer):
                self.called = True
                return fixed_update_output()

        legacy_updater = LegacyUpdater()
        legacy_trainer = object.__new__(CAGATMAPPOTrainer)
        legacy_trainer.config = frozen_control_config()
        legacy_trainer.rollout_buffer = FakeFullBuffer()
        legacy_trainer._rollout_policy_version = 0
        legacy_trainer.policy_version = 0
        legacy_trainer.policy_generator = torch.Generator().manual_seed(123)
        legacy_trainer.updater = legacy_updater
        legacy_trainer._transitions = 3072
        legacy_trainer._perform_update([])
        self.assertTrue(legacy_updater.called)

    def test_telemetry_columns_preserve_route_plus_other_identity(self) -> None:
        epochs = tuple(
            RecurrentPPOEpochDiagnostics(
                epoch_index=index,
                actor_loss=-0.1,
                critic_loss=0.2,
                entropy_mean=2.5,
                total_loss=-0.065,
                ratio_mean=1.0,
                actor_grad_norm_before_clip=0.3,
                critic_grad_norm_before_clip=0.4,
                clip_max_norm=0.5,
                route_entropy_coefficient=0.03,
                route_entropy_schedule_progress=0.0,
                route_entropy_loss_contribution=0.015,
                other_branch_entropy_loss_contribution=0.02,
                global_entropy_loss_contribution=0.035,
                collected_environment_steps=3072,
            )
            for index in range(4)
        )
        update = SimpleNamespace(
            update_index=11,
            rollout_policy_version=11,
            policy_version_after_update=12,
            output=SimpleNamespace(epoch_diagnostics=epochs),
        )
        training = SimpleNamespace(
            updates=(update,),
            total_environment_transitions=3072,
        )
        records = _ppo_epoch_records(
            treatment_config(),
            training,
            "insufficient-horizon",
        )

        self.assertEqual(len(records), 4)
        for record in records:
            self.assertEqual(record["collected_environment_steps"], 3072)
            self.assertEqual(record["route_entropy_coefficient"], 0.03)
            self.assertEqual(record["route_entropy_schedule_progress"], 0.0)
            self.assertTrue(
                math.isclose(
                    record["global_entropy_loss_contribution"],
                    record["route_entropy_loss_contribution"]
                    + record["other_branch_entropy_loss_contribution"],
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            )

        required_fields = {
            "route_entropy_coefficient",
            "route_entropy_schedule_progress",
            "route_entropy_loss_contribution",
            "other_branch_entropy_loss_contribution",
            "global_entropy_loss_contribution",
            "collected_environment_steps",
        }
        self.assertTrue(required_fields.issubset(TRAINING_METRIC_COLUMNS))
        self.assertEqual(TRAINING_DIAGNOSTICS_SCHEMA_VERSION, 5)
        self.assertEqual(
            {record["route_entropy_coefficient"] for record in records},
            {0.03},
        )
        self.assertEqual(
            {record["collected_environment_steps"] for record in records},
            {3072},
        )

        csv_rows = read_training_metrics_csv_text(_csv_text(records))
        self.assertEqual(csv_rows[0]["route_entropy_coefficient"], "0.03")
        self.assertEqual(csv_rows[0]["collected_environment_steps"], "3072")
        json_record = json.loads(_json_lines(records).splitlines()[0])
        self.assertEqual(json_record["global_entropy_loss_contribution"], 0.035)

        legacy_csv = (
            "run_id,record_type,series_index,environment_steps\n"
            "legacy,ppo_epoch,1,256\n"
        )
        legacy_row = read_training_metrics_csv_text(legacy_csv)[0]
        self.assertEqual(legacy_row["environment_steps"], "256")
        for field in required_fields:
            self.assertEqual(legacy_row[field], "")


        with self.assertRaisesRegex(
            RecurrentPPOUpdateError,
            "global entropy contribution",
        ):
            replace(
                epochs[0],
                global_entropy_loss_contribution=0.04,
            )


if __name__ == "__main__":
    unittest.main()
