"""Minimal Trajectory-Credit Telemetry V1 implementation gates."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from src.config import (
    CHECKPOINT_KIND_PERIODIC_RESUME,
    CHECKPOINT_SCHEMA_VERSION,
    ConfigError,
    RunConfig,
    WorkloadTimingMode,
    load_run_config,
    validate_mappo_checkpoint_resume_compatibility,
)
from src.env.environment import U2UMECEnvironment
from src.env.reward import TaskWorkloadSnapshot
from src.env.tasks import Task
from src.models.ca_gat_mappo import (
    ActorObservationTensorizer,
    CAGATMAPPOActor,
    MAPPOCentralizedCritic,
)
from src.models.ca_gat_mappo_actions import (
    CAGATMAPPOActionDistribution,
    SequentialActionMaskBatch,
)
from src.models.ca_gat_mappo_gae import compute_gae_and_returns
from src.models.ca_gat_mappo_ppo import compute_ppo_objective_and_loss
from src.models.ca_gat_mappo_route_telemetry import (
    RouteSampleTelemetry,
    RouteTelemetryError,
    TrajectoryCreditTracker,
    TrajectoryPreStepCapture,
)
from src.models.ca_gat_mappo_trainer import (
    CAGATMAPPOTrainer,
    CAGATMAPPOTrainerError,
)
from src.models.ca_gat_mappo_update import build_ca_gat_mappo_optimizers
from src.training_artifacts import (
    TrajectoryCreditArtifactError,
    TrajectoryCreditArtifactWriter,
    inspect_trajectory_credit_artifact,
    write_cagat_mappo_training_artifacts,
)
from tests.test_ca_gat_mappo_route_telemetry import make_route_config
from tests.test_ca_gat_mappo_training_artifacts import training_result
from tests.test_route_slot_workload_timing import (
    environment_rng_states,
    local_cpu_proposals,
    make_config,
    route_proposals,
    state_without_config_identity,
)


def telemetry_config(root: Path | None = None, *, horizon: int = 3) -> RunConfig:
    base = make_config(
        WorkloadTimingMode.ROUTE_SLOT_PRE_ROUTE,
        deadline_slots=4,
        horizon=horizon,
    )
    config = replace(
        base,
        training=replace(
            base.training,
            mappo=replace(
                base.training.mappo,
                trajectory_credit_telemetry_enabled=True,
            ),
        ),
        output=(
            base.output
            if root is None
            else replace(
                base.output,
                logs_dir=str(root / "logs"),
                dashboard_logs_dir=str(root / "dashboard_logs"),
                plots_dir=str(root / "plots"),
            )
        ),
    )
    config.validate()
    return config


def artifact_training_config(root: Path) -> RunConfig:
    base = telemetry_config(root, horizon=32)
    config = replace(
        base,
        mode="rl",
        method_id="ca_gat_mappo",
        training=replace(
            base.training,
            formal_rl_enabled=True,
            mappo=replace(
                base.training.mappo,
                training_device="cpu",
                max_training_episodes=8,
                max_training_environment_steps=256,
                evaluation_interval_steps=256,
                checkpoint_interval_steps=128,
            ),
        ),
    )
    config.validate()
    return config


def common_record(config: RunConfig, event_type: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "event_type": event_type,
        "run_id": config.run_id,
        "config_hash": config.config_hash,
        "git_commit": config.git_commit,
        "episode_id": 0,
        "task_id": 7,
        "route_event_key": [0, 7, 2],
    }


def route_record(config: RunConfig) -> dict[str, object]:
    record = common_record(config, "route")
    record.update({
        "route_step": 2,
        "rollout_index": 2,
        "source_uav": 0,
        "route_category": "local",
        "selected_destination_uav": 0,
        "route_action_index": 1,
        "route_status": "applied",
        "source_queue_proxy": {"head_task_id": 7},
        "legal_route_mask": [False, True, True],
        "selected_link_proxy_status": "not_applicable_local",
        "selected_link_proxy": None,
        "helper_queue_proxy_status": "not_applicable_local",
        "helper_queue_proxy": None,
    })
    return record


def terminal_record(config: RunConfig) -> dict[str, object]:
    record = common_record(config, "terminal")
    record.update({
        "route_status": "routed",
        "terminal_kind": "lifecycle",
        "outcome": "done",
        "observed_transition_step": 3,
        "terminal_step": 3,
        "source_uav": 0,
        "first_tx_step": None,
        "last_tx_step": None,
        "first_cpu_step": 3,
        "last_cpu_step": 3,
        "reward_decomposition": {
            "cumulative_workload_raw": 0.25,
            "cumulative_workload_penalty": 0.5,
        },
        "reward_ledger": {"status": "pass"},
        "workload_ledger": {"status": "pass"},
        "energy_ledger": {
            "tx_energy_attribution_method": "bits_pro_rata",
            "tx_energy_attribution_status": "diagnostic_allocation",
            "allocated_task_tx_energy": 0.0,
            "unattributed_tx_energy": 0.0,
        },
    })
    return record


def credit_record(config: RunConfig) -> dict[str, object]:
    record = common_record(config, "credit")
    record.update({
        "route_step": 2,
        "rollout_index": 2,
        "source_uav": 0,
        "route_category": "local",
        "selected_destination_uav": 0,
        "route_action_index": 1,
        "credit_status": "linked",
        "ppo_update_index": 0,
        "ppo_epoch_index": 0,
        "policy_version_before": 0,
        "policy_version_after": 1,
        "advantage": 0.75,
        "td_residual": 0.25,
        "return_target": 1.5,
        "na_reason": None,
    })
    return record


def make_task(task_id: int = 7, source_uav: int = 0) -> Task:
    return Task(
        task_id=task_id,
        source_uav=source_uav,
        data_bits=100.0,
        cpu_cycles=200.0,
        arrival_slot=0,
        deadline_slot=10,
    )


def synthetic_local_route(
    config: RunConfig,
    *,
    remote: bool = False,
) -> tuple[TrajectoryCreditTracker, list[dict[str, object]], Task]:
    events: list[dict[str, object]] = []
    tracker = TrajectoryCreditTracker(config, lambda value: events.append(dict(value)))
    item = make_task()
    destination = 1 if remote else 0
    category = "remote" if remote else "local"
    capture = TrajectoryPreStepCapture(
        episode_id=0,
        route_step=1,
        rollout_index=1,
        route_candidates={
            0: {
                "task_id": item.task_id,
                "source_uav": 0,
                "route_category": category,
                "selected_destination_uav": destination,
                "route_action_index": 3 if remote else 1,
                "task_snapshot": item.snapshot(),
                "source_queue_proxy": {"head_task_id": item.task_id},
                "legal_remote_destinations": [],
                "legal_route_mask": [False, True, True],
                "selected_link_proxy_status": "observed" if remote else "not_applicable_local",
                "selected_link_proxy": {"csi_valid": True} if remote else None,
                "helper_queue_proxy_status": "observed" if remote else "not_applicable_local",
                "helper_queue_proxy": {"valid": True} if remote else None,
            }
        },
        workload_snapshots={item.task_id: TaskWorkloadSnapshot.from_task(item)},
    )
    if remote:
        item.bind_remote(destination, binding_slot=1)
    else:
        item.bind_local(binding_slot=1)
    tracker._observe_routes(
        capture,
        {item.task_id: item},
        {"routing": [{"applied": True, "uav_id": 0, "task_id": item.task_id}]},
    )
    return tracker, events, item


def ppo_output_for_local_route() -> SimpleNamespace:
    sample = RouteSampleTelemetry(
        batch_index=0,
        time_index=1,
        agent_index=0,
        source_uav=0,
        category="local",
        legal_remote_destinations=(),
        remote_probability_by_destination={},
        local_probability=0.7,
        defer_probability=0.3,
        total_remote_probability_mass=0.0,
        best_remote_probability=None,
        number_of_legal_remote_destinations=0,
        selected_route_action_index=1,
        selected_destination_uav=0,
        old_route_log_prob=-0.5,
        new_route_log_prob=-0.4,
        route_log_ratio=0.1,
        route_ratio=math.exp(0.1),
        route_approx_kl=math.exp(0.1) - 1.1,
        route_clip_indicator=False,
        advantage=0.75,
        return_target=1.5,
        td_residual=0.25,
    )
    return SimpleNamespace(
        epoch_diagnostics=(
            SimpleNamespace(
                epoch_index=0,
                route_telemetry=SimpleNamespace(samples=(sample,)),
            ),
        )
    )


def run_real_local_trajectory(root: Path | None = None) -> tuple[
    RunConfig,
    list[dict[str, object]],
    dict[str, object],
]:
    config = telemetry_config(root, horizon=3)
    environment = U2UMECEnvironment(config)
    observations = environment.reset().observations
    events: list[dict[str, object]] = []
    tracker = TrajectoryCreditTracker(config, lambda value: events.append(dict(value)))
    for slot in range(3):
        if slot == 0:
            proposals = environment.canonical_proposals()
        elif slot == 1:
            proposals = route_proposals(environment, "local")
        else:
            proposals = local_cpu_proposals(environment)
        route_indices = {
            observation.uav_id: list(observation.action_masks.route_domain).index(
                proposal.route
            )
            for observation, proposal in zip(observations, proposals)
        }
        capture = tracker.capture_pre_step(
            episode_id=0,
            rollout_index=slot,
            observations=observations,
            proposals=proposals,
            route_action_indices=route_indices,
            environment=environment,
        )
        result = environment.step(proposals)
        tracker.observe_step(capture, environment, result.info)
        if result.terminated or result.truncated:
            break
        assert result.observations is not None
        observations = result.observations
    summary = dict(tracker.finalize_collection(
        environment=environment,
        episode_id=0,
        observed_transition_step=2,
        optimized_transitions=0,
    ))
    return config, events, summary


def run_real_expiration_trajectory() -> dict[str, object]:
    base = telemetry_config(horizon=3)
    config = replace(
        base,
        environment=replace(
            base.environment,
            minimum_task_slack_slots=1,
            maximum_task_slack_slots=1,
        ),
    )
    config.validate()
    environment = U2UMECEnvironment(config)
    observations = environment.reset().observations
    tracker = TrajectoryCreditTracker(config, lambda _value: None)
    for slot in range(3):
        proposals = environment.canonical_proposals()
        route_indices = {
            observation.uav_id: list(observation.action_masks.route_domain).index(
                proposal.route
            )
            for observation, proposal in zip(observations, proposals)
        }
        capture = tracker.capture_pre_step(
            episode_id=0,
            rollout_index=slot,
            observations=observations,
            proposals=proposals,
            route_action_indices=route_indices,
            environment=environment,
        )
        result = environment.step(proposals)
        tracker.observe_step(capture, environment, result.info)
        if result.terminated or result.truncated:
            break
        assert result.observations is not None
        observations = result.observations
    return dict(tracker.finalize_collection(
        environment=environment,
        episode_id=0,
        observed_transition_step=2,
        optimized_transitions=0,
    ))


class TrajectoryCreditConfigTests(unittest.TestCase):
    def test_01_disabled_field_is_omitted_from_legacy_identity(self) -> None:
        config = RunConfig()
        self.assertFalse(config.training.mappo.trajectory_credit_telemetry_enabled)
        self.assertNotIn(
            "trajectory_credit_telemetry_enabled",
            config.resolved_dict()["training"]["mappo"],
        )

    def test_02_enabled_field_changes_canonical_identity(self) -> None:
        control = RunConfig()
        treatment = replace(
            control,
            training=replace(
                control.training,
                mappo=replace(
                    control.training.mappo,
                    trajectory_credit_telemetry_enabled=True,
                ),
            ),
        )
        self.assertTrue(
            treatment.resolved_dict()["training"]["mappo"][
                "trajectory_credit_telemetry_enabled"
            ]
        )
        self.assertNotEqual(control.config_hash, treatment.config_hash)

    def test_03_enabled_field_requires_boolean(self) -> None:
        base = RunConfig()
        invalid = replace(
            base,
            training=replace(
                base.training,
                mappo=replace(
                    base.training.mappo,
                    trajectory_credit_telemetry_enabled=1,
                ),
            ),
        )
        with self.assertRaisesRegex(ConfigError, "must be boolean"):
            invalid.validate()

    def test_04_sidecar_path_is_run_scoped(self) -> None:
        config = telemetry_config()
        path = Path(config.artifact_paths()["trajectory_credit_events"])
        self.assertEqual(path.name, "trajectory_credit_events.jsonl")
        self.assertEqual(path.parent.name, config.run_id)

    def test_04b_cli_override_roundtrips_enabled_flag(self) -> None:
        config = load_run_config(
            cli_overrides={
                "training.mappo.trajectory_credit_telemetry_enabled": True
            }
        )
        self.assertTrue(config.training.mappo.trajectory_credit_telemetry_enabled)
        self.assertTrue(
            config.snapshot_dict()["training"]["mappo"][
                "trajectory_credit_telemetry_enabled"
            ]
        )

    def test_05_config_resume_compatibility_fails_when_enabled(self) -> None:
        config = telemetry_config()
        with self.assertRaisesRegex(ConfigError, "fresh-run-only"):
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

    def test_06_trainer_resume_fails_before_checkpoint_io(self) -> None:
        config = telemetry_config()
        with self.assertRaisesRegex(CAGATMAPPOTrainerError, "fresh-run-only"):
            CAGATMAPPOTrainer.resume_from_checkpoint(
                config, Path("definitely-absent-checkpoint.pt")
            )


class TrajectoryCreditArtifactTests(unittest.TestCase):
    def test_07_writer_requires_fresh_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = telemetry_config(Path(directory))
            TrajectoryCreditArtifactWriter(config)
            with self.assertRaisesRegex(TrajectoryCreditArtifactError, "already exists"):
                TrajectoryCreditArtifactWriter(config)

    def test_08_writer_rejects_incomplete_event_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = telemetry_config(Path(directory))
            writer = TrajectoryCreditArtifactWriter(config)
            with self.assertRaisesRegex(TrajectoryCreditArtifactError, "missing required"):
                writer.write(common_record(config, "route"))

    def test_09_writer_rejects_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = telemetry_config(Path(directory))
            writer = TrajectoryCreditArtifactWriter(config)
            record = route_record(config)
            record["run_id"] = "wrong"
            with self.assertRaisesRegex(TrajectoryCreditArtifactError, "run_id"):
                writer.write(record)

    def test_10_writer_rejects_nonfinite_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = telemetry_config(Path(directory))
            writer = TrajectoryCreditArtifactWriter(config)
            record = route_record(config)
            record["source_queue_proxy"] = {"bad": float("nan")}
            with self.assertRaises(ValueError):
                writer.write(record)

    def test_11_writer_rejects_future_channel_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = telemetry_config(Path(directory))
            writer = TrajectoryCreditArtifactWriter(config)
            record = route_record(config)
            record["selected_link_proxy"] = {"true_channel": [1.0, 0.0]}
            with self.assertRaisesRegex(TrajectoryCreditArtifactError, "non-causal"):
                writer.write(record)

    def test_12_writer_and_inspector_enforce_exact_cardinality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = telemetry_config(Path(directory))
            writer = TrajectoryCreditArtifactWriter(config)
            for record in (route_record(config), terminal_record(config), credit_record(config)):
                writer.write(record)
            summary = inspect_trajectory_credit_artifact(config)
            self.assertEqual(summary["event_counts"], {
                "route": 1,
                "terminal": 1,
                "credit": 1,
            })
            self.assertTrue(summary["route_credit_cardinality_match"])
            self.assertTrue(summary["terminal_cardinality_match"])

    def test_13_duplicate_route_and_terminal_identities_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = telemetry_config(Path(directory))
            writer = TrajectoryCreditArtifactWriter(config)
            writer.write(route_record(config))
            with self.assertRaisesRegex(TrajectoryCreditArtifactError, "duplicate route"):
                writer.write(route_record(config))
            writer.write(terminal_record(config))
            with self.assertRaisesRegex(TrajectoryCreditArtifactError, "duplicate terminal"):
                writer.write(terminal_record(config))

    def test_14_completed_artifact_group_includes_sidecar_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = artifact_training_config(Path(directory))
            writer = TrajectoryCreditArtifactWriter(config)
            for record in (route_record(config), terminal_record(config), credit_record(config)):
                writer.write(record)
            outcome = write_cagat_mappo_training_artifacts(config, training_result())
            self.assertEqual(len(outcome.artifacts), 6)
            summary = json.loads(
                Path(config.artifact_paths()["aggregate_metrics"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                summary["trajectory_credit_telemetry"]["event_counts"]["route"],
                1,
            )

    def test_14b_real_environment_events_roundtrip_through_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, events, _ = run_real_local_trajectory(Path(directory))
            writer = TrajectoryCreditArtifactWriter(config)
            for record in events:
                writer.write(record)
            summary = inspect_trajectory_credit_artifact(config)
            self.assertGreater(summary["event_counts"]["terminal"], 0)
            self.assertEqual(
                summary["event_counts"]["route"],
                summary["event_counts"]["credit"],
            )


class TrajectoryCreditTrackerTests(unittest.TestCase):
    def test_15_only_applied_routing_creates_formal_event(self) -> None:
        config = telemetry_config()
        events: list[dict[str, object]] = []
        tracker = TrajectoryCreditTracker(config, lambda value: events.append(dict(value)))
        item = make_task()
        capture = TrajectoryPreStepCapture(
            episode_id=0,
            route_step=1,
            rollout_index=1,
            route_candidates={},
            workload_snapshots={},
        )
        tracker._observe_routes(
            capture,
            {item.task_id: item},
            {"routing": [
                {"applied": False, "uav_id": 0, "task_id": item.task_id},
                {"uav_id": 0, "task_id": item.task_id, "destination": "defer"},
            ]},
        )
        self.assertEqual(events, [])

    def test_16_route_identity_is_episode_task_and_route_step(self) -> None:
        _, events, _ = synthetic_local_route(telemetry_config())
        self.assertEqual(events[0]["event_type"], "route")
        self.assertEqual(events[0]["route_event_key"], [0, 7, 1])
        self.assertEqual(events[0]["route_category"], "local")
        self.assertEqual(events[0]["route_status"], "applied")

    def test_16b_applied_remote_creates_exactly_one_route_event(self) -> None:
        _, events, _ = synthetic_local_route(telemetry_config(), remote=True)
        routes = [record for record in events if record["event_type"] == "route"]
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]["route_category"], "remote")
        self.assertEqual(routes[0]["selected_destination_uav"], 1)

    def test_17_real_route_capture_excludes_future_csi(self) -> None:
        _, events, _ = run_real_local_trajectory()
        route = next(record for record in events if record["event_type"] == "route")
        encoded = json.dumps(route, sort_keys=True).lower()
        for forbidden in (
            "true_channel",
            "future_csi",
            "channel_real",
            "channel_imag",
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(
            route["capture_source"],
            "pre_step_observation_plus_post_step_routing",
        )

    def test_18_workload_route_to_terminal_ledger_conserves(self) -> None:
        _, events, summary = run_real_local_trajectory()
        workload = summary["workload_ledger"]
        self.assertEqual(workload["status"], "pass")
        self.assertAlmostEqual(workload["raw_conservation_residual"], 0.0, places=12)
        self.assertAlmostEqual(workload["penalty_conservation_residual"], 0.0, places=12)
        terminal = next(
            record
            for record in events
            if record["event_type"] == "terminal" and record["task_id"] == 0
        )
        decomposition = terminal["reward_decomposition"]
        self.assertGreater(decomposition["cumulative_workload_raw"], 0.0)
        self.assertGreater(decomposition["cumulative_workload_penalty"], 0.0)

    def test_18b_completion_and_expiration_ledgers_conserve(self) -> None:
        _, _, completion_summary = run_real_local_trajectory()
        completion = completion_summary["reward_ledger"]
        self.assertEqual(completion["status"], "pass")
        self.assertGreater(completion["cumulative_team_completion_component"], 0.0)
        self.assertAlmostEqual(
            completion["completion_conservation_residual"], 0.0, places=12
        )
        expiration = run_real_expiration_trajectory()["reward_ledger"]
        self.assertEqual(expiration["status"], "pass")
        self.assertGreater(expiration["cumulative_team_expiration_penalty"], 0.0)
        self.assertAlmostEqual(
            expiration["expiration_conservation_residual"], 0.0, places=12
        )

    def test_19_tx_energy_is_diagnostic_bits_pro_rata_and_conserved(self) -> None:
        tracker = TrajectoryCreditTracker(telemetry_config(), lambda _value: None)
        capture = TrajectoryPreStepCapture(0, 2, 2, {}, {})
        tracker._observe_energy_and_service(
            capture,
            {"reward": {"actual_energy_j": 5.0}},
            {0: make_task(0, 0), 1: make_task(1, 0)},
            {
                "links": [{
                    "transmit_energy_j": 3.0,
                    "task_services": [
                        {"task_id": 0, "amount": 10.0},
                        {"task_id": 1, "amount": 30.0},
                    ],
                }],
                "cpu": [{
                    "task_id": 0,
                    "service_cycles": 4.0,
                    "cpu_energy_j": 2.0,
                }],
            },
        )
        energy = tracker.summary()["energy_ledger"]
        self.assertEqual(energy["tx_energy_attribution_method"], "bits_pro_rata")
        self.assertEqual(energy["tx_energy_attribution_status"], "diagnostic_allocation")
        self.assertAlmostEqual(tracker._tasks[(0, 0)].allocated_task_tx_energy, 0.75)
        self.assertAlmostEqual(tracker._tasks[(0, 1)].allocated_task_tx_energy, 2.25)
        self.assertAlmostEqual(energy["allocated_task_tx_energy"], 3.0)
        self.assertAlmostEqual(energy["unattributed_tx_energy"], 0.0)
        self.assertEqual(energy["status"], "pass")

    def test_20_cpu_energy_with_task_id_is_exact(self) -> None:
        tracker = TrajectoryCreditTracker(telemetry_config(), lambda _value: None)
        capture = TrajectoryPreStepCapture(0, 2, 2, {}, {})
        tracker._observe_energy_and_service(
            capture,
            {"reward": {"actual_energy_j": 2.0}},
            {0: make_task(0, 0)},
            {
                "links": [],
                "cpu": [{
                    "task_id": 0,
                    "service_cycles": 4.0,
                    "cpu_energy_j": 2.0,
                }],
            },
        )
        ledger = tracker._tasks[(0, 0)]
        self.assertEqual(ledger.exact_task_cpu_energy, 2.0)
        self.assertEqual(ledger.cumulative_cpu_cycles, 4.0)
        self.assertEqual(ledger.first_cpu_step, 2)
        self.assertEqual(ledger.last_cpu_step, 2)

    def test_20b_zero_cpu_cycles_do_not_create_first_service_step(self) -> None:
        tracker = TrajectoryCreditTracker(telemetry_config(), lambda _value: None)
        capture = TrajectoryPreStepCapture(0, 2, 2, {}, {})
        tracker._observe_energy_and_service(
            capture,
            {"reward": {"actual_energy_j": 0.0}},
            {0: make_task(0, 0)},
            {
                "links": [],
                "cpu": [{
                    "task_id": 0,
                    "service_cycles": 0.0,
                    "cpu_energy_j": 0.0,
                }],
            },
        )
        ledger = tracker._tasks[(0, 0)]
        self.assertIsNone(ledger.first_cpu_step)
        self.assertIsNone(ledger.last_cpu_step)

    def test_21_terminal_events_are_exactly_once_and_censor_is_distinct(self) -> None:
        config = telemetry_config()
        events: list[dict[str, object]] = []
        tracker = TrajectoryCreditTracker(config, lambda value: events.append(dict(value)))
        environment = U2UMECEnvironment(config)
        environment.reset()
        assert environment.lifecycle is not None
        item = make_task()
        environment.lifecycle.tasks[item.task_id] = item
        for _ in range(2):
            tracker.finalize_collection(
                environment=environment,
                episode_id=0,
                observed_transition_step=0,
                optimized_transitions=0,
            )
        terminals = [record for record in events if record["event_type"] == "terminal"]
        self.assertEqual(len(terminals), 1)
        self.assertEqual(terminals[0]["terminal_kind"], "collection_censored")
        self.assertIsNone(terminals[0]["outcome"])

    def test_22_epoch_zero_ppo_credit_links_one_to_one(self) -> None:
        tracker, events, _ = synthetic_local_route(telemetry_config())
        tracker.observe_ppo_update(
            ppo_output_for_local_route(),
            update_index=0,
            policy_version_before=0,
            policy_version_after=1,
            rollout_start_index=0,
            rollout_length=256,
        )
        credits = [record for record in events if record["event_type"] == "credit"]
        self.assertEqual(len(credits), 1)
        self.assertEqual(credits[0]["route_event_key"], [0, 7, 1])
        self.assertEqual(credits[0]["credit_status"], "linked")
        self.assertEqual(credits[0]["ppo_epoch_index"], 0)
        self.assertEqual(credits[0]["advantage"], 0.75)

    def test_22c_shared_gae_credit_writer_uses_formal_route_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = telemetry_config(Path(directory))
            self.assertEqual(
                config.training.mappo.route_credit_mode.value,
                "shared_gae",
            )
            tracker, events, item = synthetic_local_route(config)
            writer = TrajectoryCreditArtifactWriter(config)
            for record in events:
                writer.write(record)
            tracker._emit = writer.write
            sample = (
                ppo_output_for_local_route()
                .epoch_diagnostics[0]
                .route_telemetry.samples[0]
            )
            self.assertIsNone(sample.task_id)
            terminal = terminal_record(config)
            terminal["route_event_key"] = [0, item.task_id, 1]
            writer.write(terminal)
            tracker.observe_ppo_update(
                ppo_output_for_local_route(),
                update_index=0,
                policy_version_before=0,
                policy_version_after=1,
                rollout_start_index=0,
                rollout_length=256,
            )
            records = [
                json.loads(line)
                for line in Path(
                    config.artifact_paths()["trajectory_credit_events"]
                )
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            credit = next(
                record for record in records if record["event_type"] == "credit"
            )
            self.assertEqual(credit["task_id"], item.task_id)
            summary = inspect_trajectory_credit_artifact(config)
            self.assertTrue(summary["route_credit_cardinality_match"])

    def test_22d_shared_gae_no_head_keeps_nstep_metadata_disabled(self) -> None:
        executed, summary = CAGATMAPPOTrainer._execution_metadata(
            {
                "executed": [{"uav_id": 0}],
                "service": {
                    "routing": [
                        {
                            "uav_id": 0,
                            "task_id": None,
                            "proposal": "defer",
                            "applied": False,
                        }
                    ]
                },
            },
            episode_id=0,
            global_rollout_index=0,
            policy_version=0,
            route_credit_enabled=False,
        )
        self.assertEqual(executed, [{"uav_id": 0}])
        self.assertNotIn("route_credit_step", summary)

    def test_22b_four_ppo_epochs_still_emit_one_canonical_credit(self) -> None:
        tracker, events, _ = synthetic_local_route(telemetry_config())
        base = ppo_output_for_local_route().epoch_diagnostics[0]
        output = SimpleNamespace(
            epoch_diagnostics=tuple(
                SimpleNamespace(
                    epoch_index=index,
                    route_telemetry=base.route_telemetry,
                )
                for index in range(4)
            )
        )
        tracker.observe_ppo_update(
            output,
            update_index=0,
            policy_version_before=0,
            policy_version_after=1,
            rollout_start_index=0,
            rollout_length=256,
        )
        self.assertEqual(
            sum(record["event_type"] == "credit" for record in events),
            1,
        )

    def test_23_unoptimized_tail_credit_is_explicit_na(self) -> None:
        config = telemetry_config()
        tracker, events, item = synthetic_local_route(config)
        environment = U2UMECEnvironment(config)
        environment.reset()
        assert environment.lifecycle is not None
        environment.lifecycle.tasks[item.task_id] = item
        tracker.finalize_collection(
            environment=environment,
            episode_id=0,
            observed_transition_step=2,
            optimized_transitions=0,
        )
        credit = next(record for record in events if record["event_type"] == "credit")
        self.assertEqual(credit["credit_status"], "tail_not_optimized")
        self.assertEqual(credit["na_reason"], "tail_not_optimized")
        self.assertIsNone(credit["advantage"])

    def test_24_missing_or_duplicate_ppo_linkage_fails_fast(self) -> None:
        tracker, _, _ = synthetic_local_route(telemetry_config())
        empty = SimpleNamespace(
            epoch_diagnostics=(
                SimpleNamespace(
                    epoch_index=0,
                    route_telemetry=SimpleNamespace(samples=()),
                ),
            )
        )
        with self.assertRaisesRegex(RouteTelemetryError, "incomplete"):
            tracker.observe_ppo_update(
                empty,
                update_index=0,
                policy_version_before=0,
                policy_version_after=1,
                rollout_start_index=0,
                rollout_length=256,
            )
        tracker.observe_ppo_update(
            ppo_output_for_local_route(),
            update_index=0,
            policy_version_before=0,
            policy_version_after=1,
            rollout_start_index=0,
            rollout_length=256,
        )
        with self.assertRaisesRegex(RouteTelemetryError, "duplicate"):
            tracker.observe_ppo_update(
                ppo_output_for_local_route(),
                update_index=0,
                policy_version_before=0,
                policy_version_after=1,
                rollout_start_index=0,
                rollout_length=256,
            )


class TrajectoryCreditNeutralityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.control = make_route_config()
        self.treatment = replace(
            self.control,
            training=replace(
                self.control.training,
                mappo=replace(
                    self.control.training.mappo,
                    trajectory_credit_telemetry_enabled=True,
                ),
            ),
        )
        self.treatment.validate()

    def test_25_environment_reward_and_rng_are_flag_neutral(self) -> None:
        control = U2UMECEnvironment(self.control)
        treatment = U2UMECEnvironment(self.treatment)
        control.reset()
        treatment.reset()
        self.assertEqual(
            state_without_config_identity(control),
            state_without_config_identity(treatment),
        )
        self.assertEqual(environment_rng_states(control), environment_rng_states(treatment))
        for _ in range(3):
            self.assertEqual(control.canonical_proposals(), treatment.canonical_proposals())
            left = control.step(control.canonical_proposals())
            right = treatment.step(treatment.canonical_proposals())
            self.assertEqual(left.reward, right.reward)
            self.assertEqual(left.info["reward"], right.info["reward"])
            self.assertEqual(
                state_without_config_identity(control),
                state_without_config_identity(treatment),
            )
            self.assertEqual(
                environment_rng_states(control), environment_rng_states(treatment)
            )

    def test_26_model_parameters_actions_and_sampling_rng_are_flag_neutral(self) -> None:
        torch.manual_seed(811)
        actor_control = CAGATMAPPOActor(self.control).eval()
        critic_control = MAPPOCentralizedCritic(self.control).eval()
        torch.manual_seed(811)
        actor_treatment = CAGATMAPPOActor(self.treatment).eval()
        critic_treatment = MAPPOCentralizedCritic(self.treatment).eval()
        for left, right in zip(actor_control.parameters(), actor_treatment.parameters()):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(critic_control.parameters(), critic_treatment.parameters()):
            self.assertTrue(torch.equal(left, right))
        reset_control = U2UMECEnvironment(self.control).reset()
        reset_treatment = U2UMECEnvironment(self.treatment).reset()
        batch_control = ActorObservationTensorizer(self.control).encode_step(
            reset_control.observations, episode_start=True
        )
        batch_treatment = ActorObservationTensorizer(self.treatment).encode_step(
            reset_treatment.observations, episode_start=True
        )
        masks_control = SequentialActionMaskBatch.from_observations(
            reset_control.observations
        )
        masks_treatment = SequentialActionMaskBatch.from_observations(
            reset_treatment.observations
        )
        for name in (
            "self_features",
            "neighbor_public_features",
            "edge_features",
            "neighbor_mask",
            "episode_starts",
        ):
            self.assertTrue(torch.equal(
                getattr(batch_control, name), getattr(batch_treatment, name)
            ))
        for branch in batch_control.action_masks:
            self.assertTrue(torch.equal(
                batch_control.action_masks[branch],
                batch_treatment.action_masks[branch],
            ))
        generator_control = torch.Generator().manual_seed(912)
        generator_treatment = torch.Generator().manual_seed(912)
        output_control = CAGATMAPPOActionDistribution(
            actor_control, self.control
        ).sample_actions(
            batch_control,
            masks_control,
            actor_control.initial_hidden(1),
            generator=generator_control,
        )
        output_treatment = CAGATMAPPOActionDistribution(
            actor_treatment, self.treatment
        ).sample_actions(
            batch_treatment,
            masks_treatment,
            actor_treatment.initial_hidden(1),
            generator=generator_treatment,
        )
        self.assertEqual(output_control.proposals, output_treatment.proposals)
        for branch in output_control.action_indices:
            self.assertTrue(torch.equal(
                output_control.action_indices[branch],
                output_treatment.action_indices[branch],
            ))
        self.assertTrue(torch.equal(
            generator_control.get_state(), generator_treatment.get_state()
        ))

    def test_27_gae_targets_are_bitwise_flag_neutral(self) -> None:
        inputs = {
            "reward": torch.tensor([1.0, -0.5, 0.25, 2.0]),
            "old_value": torch.tensor([0.1, 0.2, -0.1, 0.3]),
            "bootstrap_value": torch.tensor([0.2, -0.1, 0.3, 0.0]),
            "terminated": torch.tensor([False, False, False, False]),
            "truncated": torch.tensor([False, False, False, True]),
            "episode_boundary": torch.tensor([False, False, False, True]),
            "bootstrap_allowed": torch.tensor([True, True, True, False]),
        }
        left = compute_gae_and_returns(
            **inputs,
            gamma=self.control.training.mappo.gamma,
            gae_lambda=self.control.training.mappo.gae_lambda,
        )
        right = compute_gae_and_returns(
            **inputs,
            gamma=self.treatment.training.mappo.gamma,
            gae_lambda=self.treatment.training.mappo.gae_lambda,
        )
        for name in (
            "bootstrap_mask",
            "td_residual",
            "advantage",
            "return_target",
            "sequence_mask",
        ):
            self.assertTrue(torch.equal(getattr(left, name), getattr(right, name)))

    def test_28_ppo_objective_and_loss_are_bitwise_flag_neutral(self) -> None:
        new = torch.tensor(
            [[-0.7, -0.6, -0.5, -0.4], [-0.8, -0.7, -0.6, -0.5]],
            requires_grad=True,
        )
        kwargs = {
            "new_joint_log_prob": new,
            "old_joint_log_prob": new.detach() - 0.05,
            "advantage": torch.tensor([0.5, -0.25]),
            "current_value": torch.tensor([0.2, 0.3], requires_grad=True),
            "return_target": torch.tensor([1.0, -0.5]),
            "entropy": torch.full((2, 4), 0.4),
            "sequence_valid_mask": torch.tensor([True, True]),
        }
        left = compute_ppo_objective_and_loss(
            **kwargs,
            epsilon_clip=self.control.training.mappo.ppo_clip_epsilon,
            value_coefficient=self.control.training.mappo.value_coefficient,
            entropy_coefficient=self.control.training.mappo.entropy_coefficient,
        )
        right = compute_ppo_objective_and_loss(
            **kwargs,
            epsilon_clip=self.treatment.training.mappo.ppo_clip_epsilon,
            value_coefficient=self.treatment.training.mappo.value_coefficient,
            entropy_coefficient=self.treatment.training.mappo.entropy_coefficient,
        )
        for name in ("actor_loss", "critic_loss", "entropy_mean", "total_loss"):
            self.assertTrue(torch.equal(getattr(left, name), getattr(right, name)))

    def test_29_optimizer_parameter_updates_are_bitwise_flag_neutral(self) -> None:
        torch.manual_seed(1013)
        actor_control = CAGATMAPPOActor(self.control)
        critic_control = MAPPOCentralizedCritic(self.control)
        actor_treatment = copy.deepcopy(actor_control)
        critic_treatment = copy.deepcopy(critic_control)
        optim_control = build_ca_gat_mappo_optimizers(
            actor_control, critic_control, self.control
        )
        optim_treatment = build_ca_gat_mappo_optimizers(
            actor_treatment, critic_treatment, self.treatment
        )
        for bundle in (optim_control, optim_treatment):
            bundle.actor_optimizer.zero_grad(set_to_none=True)
            bundle.critic_optimizer.zero_grad(set_to_none=True)
            actor_loss = sum(
                parameter.square().sum() for parameter in bundle.actor_parameters
            ) * 1.0e-8
            critic_loss = sum(
                parameter.square().sum() for parameter in bundle.critic_parameters
            ) * 1.0e-8
            actor_loss.backward()
            critic_loss.backward()
            bundle.actor_optimizer.step()
            bundle.critic_optimizer.step()
        for left, right in zip(actor_control.parameters(), actor_treatment.parameters()):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(critic_control.parameters(), critic_treatment.parameters()):
            self.assertTrue(torch.equal(left, right))


if __name__ == "__main__":
    unittest.main()
