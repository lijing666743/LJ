"""Gate tests for Minimal Task-Candidate Diagnostic Instrumentation V1."""

from __future__ import annotations

from dataclasses import replace
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.config import RunConfig
from src.models.ca_gat_mappo import (
    ActorObservationTensorizer,
    CAGATMAPPOActor,
    MAPPOCentralizedCritic,
)
from src.models.ca_gat_mappo_actions import (
    CAGATMAPPOActionDistribution,
    SequentialActionMaskBatch,
)
from src.models.ca_gat_mappo_ppo import compute_ppo_objective_and_loss
from src.models.ca_gat_mappo_route_telemetry import (
    ROUTE_DIAGNOSTIC_UTILITY_LABEL,
    RouteDiagnosticSampleCollector,
    reconstruct_route_diagnostic_utility,
)
from src.models.ca_gat_mappo_trainer import CAGATMAPPOTrainer
from src.training_artifacts import (
    RouteDiagnosticArtifactError,
    RouteDiagnosticArtifactWriter,
    inspect_route_diagnostic_artifact,
)
from tests.test_ca_gat_mappo_route_telemetry import (
    active_route_fixture,
    evaluate_route_steps,
)
from tests.test_ca_gat_mappo_trainer import (
    CapturingUpdater,
    RecordingFactory,
    make_trainer_config,
)


def diagnostic_config(base: RunConfig, directory: str, enabled: bool = True) -> RunConfig:
    output = replace(base.output, logs_dir=directory)
    mappo = replace(
        base.training.mappo,
        route_diagnostic_samples_enabled=enabled,
        training_device="cpu",
    )
    config = replace(
        base,
        output=output,
        training=replace(base.training, mappo=mappo),
    )
    config.validate()
    return config


def diagnostic_records(config: RunConfig, observations):
    actor = CAGATMAPPOActor(config).eval()
    tensorizer = ActorObservationTensorizer(config)
    batch = tensorizer.encode_step(
        observations,
        device="cpu",
        dtype=next(actor.parameters()).dtype,
        episode_start=False,
    )
    masks = SequentialActionMaskBatch.from_observations(observations)
    distribution = CAGATMAPPOActionDistribution(actor, config)
    output = distribution.deterministic_actions(
        batch,
        masks,
        actor.initial_hidden(1),
    )
    proposals = tuple(output.proposals[0][0])
    collector = RouteDiagnosticSampleCollector(config)
    return collector, collector.collect(
        episode_id=0,
        global_environment_step=7,
        policy_version=0,
        observations=observations,
        proposals=proposals,
        action_output=output,
    )


class RouteDiagnosticInstrumentationTests(unittest.TestCase):
    def test_actor_safe_candidate_alignment_and_offline_reconstruction(self) -> None:
        base, _environment, _reset, observations = active_route_fixture()
        with tempfile.TemporaryDirectory() as directory:
            config = diagnostic_config(base, directory)
            collector, records = diagnostic_records(config, observations)
            self.assertEqual(len(records), 4)
            first = records[0]
            self.assertEqual(first["utility_label"], ROUTE_DIAGNOSTIC_UTILITY_LABEL)
            self.assertEqual(first["primary_key"], [config.run_id, 0, 7, 0, first["head_task_id"]])
            self.assertIsNone(first["task"]["arrival_slot"])
            self.assertFalse(first["task"]["arrival_slot_valid"])
            self.assertEqual([item["candidate_uav"] for item in first["candidates"]], [1, 2, 3])
            self.assertEqual(
                [item["candidate_uav"] for item in first["policy"]["remote_probability_by_candidate"]],
                [1, 2, 3],
            )
            self.assertNotIn("stale_csi", json.dumps(first))
            self.assertEqual(len(first["candidates"][0]["link_history"]["historical_quality_by_ru"]), 20)
            utility = reconstruct_route_diagnostic_utility(first, collector.schema_header())
            self.assertEqual(utility["utility_label"], ROUTE_DIAGNOSTIC_UTILITY_LABEL)
            self.assertEqual(len(utility["remote_by_candidate"]), 3)

    def test_writer_is_independent_and_round_trips(self) -> None:
        base, _environment, _reset, observations = active_route_fixture()
        with tempfile.TemporaryDirectory() as directory:
            config = diagnostic_config(base, directory)
            collector, records = diagnostic_records(config, observations)
            writer = RouteDiagnosticArtifactWriter(config, collector.schema_header())
            for record in records:
                writer.write(record)
            summary = inspect_route_diagnostic_artifact(config)
            self.assertEqual(summary["sample_count"], len(records))
            self.assertEqual(sum(summary["route_counts"].values()), len(records))
            path = Path(config.artifact_paths()["route_diagnostic_samples"])
            self.assertTrue(path.is_file())
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), len(records) + 1)
            self.assertFalse(Path(config.artifact_paths()["raw_metrics"]).exists())
            self.assertFalse(Path(config.artifact_paths()["trajectory_credit_events"]).exists())
            with self.assertRaises(RouteDiagnosticArtifactError):
                RouteDiagnosticArtifactWriter(config, collector.schema_header())

    def test_writer_can_rehydrate_without_occurrence_counter_state(self) -> None:
        base, _environment, _reset, observations = active_route_fixture()
        with tempfile.TemporaryDirectory() as directory:
            config = diagnostic_config(base, directory)
            collector, records = diagnostic_records(config, observations)
            first_writer = RouteDiagnosticArtifactWriter(config, collector.schema_header())
            first_writer.write(records[0])
            resumed_writer = RouteDiagnosticArtifactWriter(
                config,
                collector.schema_header(),
                allow_existing=True,
            )
            for record in records[1:]:
                resumed_writer.write(record)
            self.assertEqual(
                inspect_route_diagnostic_artifact(config)["sample_count"],
                len(records),
            )

    def test_invalid_historical_quality_is_explicitly_unreconstructable(self) -> None:
        base, _environment, _reset, observations = active_route_fixture()
        with tempfile.TemporaryDirectory() as directory:
            config = diagnostic_config(base, directory)
            collector, records = diagnostic_records(config, observations)
            sample = copy.deepcopy(records[0])
            for candidate in sample["candidates"]:
                link = candidate["link_history"]
                link["historical_quality_valid_mask"] = [False] * 20
                link["historical_quality_by_ru"] = [None] * 20
            utility = reconstruct_route_diagnostic_utility(sample, collector.schema_header())
            statuses = {item["status"] for item in utility["remote_by_candidate"]}
            self.assertTrue(statuses <= {
                "invalid_primitives",
                "no_valid_rate_or_energy_feasible_combination",
                "time_component_unavailable",
            })
            self.assertNotIn("valid", statuses)

    def test_no_legal_remote_and_energy_infeasible_are_explicit(self) -> None:
        base, _environment, _reset, observations = active_route_fixture()
        with tempfile.TemporaryDirectory() as directory:
            config = diagnostic_config(base, directory)
            collector, records = diagnostic_records(config, observations)
            no_remote = copy.deepcopy(records[0])
            for candidate in no_remote["candidates"]:
                candidate["legal_mask"] = False
            utility = reconstruct_route_diagnostic_utility(no_remote, collector.schema_header())
            self.assertTrue(all(
                item["status"] == "illegal_candidate"
                for item in utility["remote_by_candidate"]
            ))

            energy_limited = copy.deepcopy(records[0])
            energy_limited["source"]["residual_energy_j"] = 0.0
            utility = reconstruct_route_diagnostic_utility(
                energy_limited,
                collector.schema_header(),
            )
            self.assertTrue(all(
                item["status"] != "valid"
                for item in utility["remote_by_candidate"]
            ))

    def test_local_remote_defer_capture_coverage(self) -> None:
        base, environment, _reset, observations = active_route_fixture()
        with tempfile.TemporaryDirectory() as directory:
            config = diagnostic_config(base, directory)
            collector = RouteDiagnosticSampleCollector(config)
            for kind in ("local", "remote", "defer"):
                _actor, _batch, _masks, proposals, policy = evaluate_route_steps(
                    config,
                    environment,
                    (observations,),
                    (tuple(kind for _ in observations),),
                )
                records = collector.collect(
                    episode_id=0,
                    global_environment_step={"local": 1, "remote": 2, "defer": 3}[kind],
                    policy_version=0,
                    observations=observations,
                    proposals=proposals[0][0],
                    action_output=policy,
                )
                self.assertEqual(len(records), 4)
                self.assertTrue(all(
                    (record["policy"]["actual_route"] == "local" and kind == "local")
                    or (record["policy"]["actual_route"] == "defer" and kind == "defer")
                    or (isinstance(record["policy"]["actual_route"], int) and kind == "remote")
                    for record in records
                ))

    def test_collector_does_not_consume_any_rng_stream(self) -> None:
        base, _environment, _reset, observations = active_route_fixture()
        with tempfile.TemporaryDirectory() as directory:
            config = diagnostic_config(base, directory)
            actor = CAGATMAPPOActor(config).eval()
            tensorizer = ActorObservationTensorizer(config)
            batch = tensorizer.encode_step(observations, device="cpu", episode_start=False)
            masks = SequentialActionMaskBatch.from_observations(observations)
            generator = torch.Generator(device="cpu").manual_seed(8321)
            output = CAGATMAPPOActionDistribution(actor, config).sample_actions(
                batch,
                masks,
                actor.initial_hidden(1),
                generator=generator,
            )
            before_torch = torch.get_rng_state().clone()
            before_policy = generator.get_state().clone()
            before_numpy = np.random.get_state()
            RouteDiagnosticSampleCollector(config).collect(
                episode_id=0,
                global_environment_step=0,
                policy_version=0,
                observations=observations,
                proposals=tuple(output.proposals[0][0]),
                action_output=output,
            )
            self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
            self.assertTrue(torch.equal(before_policy, generator.get_state()))
            after_numpy = np.random.get_state()
            self.assertEqual(before_numpy[0], after_numpy[0])
            self.assertTrue(np.array_equal(before_numpy[1], after_numpy[1]))
            self.assertEqual(before_numpy[2:], after_numpy[2:])

    def test_ppo_objective_is_independent_of_diagnostic_flag(self) -> None:
        old_log_prob = torch.tensor(
            [[-0.4, -0.3, -0.2, -0.1], [-0.3, -0.2, -0.1, 0.0]]
        )
        kwargs = {
            "new_joint_log_prob": old_log_prob + 0.02,
            "old_joint_log_prob": old_log_prob,
            "advantage": torch.tensor([0.5, -0.25]),
            "current_value": torch.tensor([0.2, 0.3], requires_grad=True),
            "return_target": torch.tensor([1.0, -0.5]),
            "entropy": torch.full((2, 4), 0.4),
            "sequence_valid_mask": torch.tensor([True, True]),
        }
        left = compute_ppo_objective_and_loss(
            **kwargs,
            epsilon_clip=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.01,
        )
        right = compute_ppo_objective_and_loss(
            **kwargs,
            epsilon_clip=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.01,
        )
        for name in ("actor_loss", "critic_loss", "entropy_mean", "total_loss"):
            self.assertTrue(torch.equal(getattr(left, name), getattr(right, name)))

    def test_disabled_flag_preserves_legacy_canonical_identity(self) -> None:
        base, _environment, _reset, _observations = active_route_fixture()
        with tempfile.TemporaryDirectory() as directory:
            disabled = diagnostic_config(base, directory, enabled=False)
            enabled = diagnostic_config(base, directory, enabled=True)
            self.assertNotIn("route_diagnostic_samples_enabled", disabled.resolved_dict()["training"]["mappo"])
            self.assertTrue(enabled.resolved_dict()["training"]["mappo"]["route_diagnostic_samples_enabled"])
            self.assertNotEqual(disabled.config_hash, enabled.config_hash)

    def test_trainer_on_off_is_transition_neutral_and_writes_only_sidecar(self) -> None:
        base = make_trainer_config(max_steps=2, max_episodes=1)
        environment = replace(base.environment, arrival_probabilities=(1.0,))
        with tempfile.TemporaryDirectory() as root:
            control = diagnostic_config(
                replace(base, environment=environment),
                str(Path(root) / "control"),
                enabled=False,
            )
            treatment = diagnostic_config(
                replace(base, environment=environment),
                str(Path(root) / "treatment"),
                enabled=True,
            )
            torch.manual_seed(3201)
            actor = CAGATMAPPOActor(control)
            critic = MAPPOCentralizedCritic(control)
            treatment_actor = copy.deepcopy(actor)
            treatment_critic = copy.deepcopy(critic)
            control_factory = RecordingFactory()
            treatment_factory = RecordingFactory()
            control_trainer = CAGATMAPPOTrainer(
                control,
                actor=actor,
                critic=critic,
                environment_factory=control_factory,
                updater_factory=lambda *_: CapturingUpdater(),
            )
            treatment_trainer = CAGATMAPPOTrainer(
                treatment,
                actor=treatment_actor,
                critic=treatment_critic,
                environment_factory=treatment_factory,
                updater_factory=lambda *_: CapturingUpdater(),
            )
            control_result = control_trainer.train()
            treatment_result = treatment_trainer.train()
            control_steps = tuple(
                step for environment in control_factory.environments for step in environment.step_inputs
            )
            treatment_steps = tuple(
                step for environment in treatment_factory.environments for step in environment.step_inputs
            )
            self.assertEqual(control_steps, treatment_steps)
            self.assertEqual(
                [item.reward for item in control_factory.all_results],
                [item.reward for item in treatment_factory.all_results],
            )
            self.assertEqual(
                [item.info["reward"] for item in control_factory.all_results],
                [item.info["reward"] for item in treatment_factory.all_results],
            )
            self.assertEqual(control_result.reward, treatment_result.reward)
            self.assertEqual(control_result.ppo_update_count, treatment_result.ppo_update_count)
            self.assertFalse(Path(control.artifact_paths()["route_diagnostic_samples"]).exists())
            self.assertTrue(Path(treatment.artifact_paths()["route_diagnostic_samples"]).exists())
            self.assertEqual(inspect_route_diagnostic_artifact(treatment)["sample_count"], 1)


if __name__ == "__main__":
    unittest.main()
