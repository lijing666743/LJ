from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dataclasses import fields, is_dataclass, replace

from src.config import load_run_config
from src.models.ca_gat_mappo import ACTION_BRANCH_ORDER
from src.env.environment import U2UMECEnvironment
from src.models.ca_gat_mappo_route_telemetry import (
    RouteHeadGradientRowTelemetry,
    RouteOutcomeTracker,
    RouteSampleTelemetry,
    collect_route_telemetry,
    aggregate_route_outcomes,
    compute_route_only_ppo_dynamics,
    measure_route_head_gradient_rows,
    summarize_distribution,
)
from src.training_artifacts import (
    _base_record,
    _csv_text,
    _json_lines,
    write_cagat_mappo_training_artifacts,
)
from tests.test_ca_gat_mappo_route_telemetry import (
    active_route_fixture,
    evaluate_route_steps,
    make_route_config,
    make_update_config,
)
from tests.test_ca_gat_mappo_training_artifacts import training_result


class FineRouteCreditTelemetryTests(unittest.TestCase):
    def test_distribution_summary_is_deterministic_and_complete(self) -> None:
        summary = summarize_distribution(torch.tensor([-2.0, -1.0, 1.0, 3.0]))
        self.assertEqual(summary.count, 4)
        self.assertAlmostEqual(summary.mean, 0.25)
        self.assertAlmostEqual(summary.median, 0.0)
        self.assertAlmostEqual(summary.positive_fraction, 0.5)
        self.assertAlmostEqual(summary.negative_fraction, 0.5)
        self.assertAlmostEqual(summary.p25, -1.25)
        self.assertAlmostEqual(summary.p75, 1.5)

    def test_route_only_ppo_dynamics_is_grouped_and_never_backpropagates(self) -> None:
        old = torch.zeros(3)
        new = torch.log(torch.tensor([1.3, 0.8, 1.0]))
        active = torch.ones(3, dtype=torch.bool)
        dynamics = compute_route_only_ppo_dynamics(
            old,
            new,
            active,
            {
                "local": torch.tensor([True, False, False]),
                "remote": torch.tensor([False, True, False]),
                "defer": torch.tensor([False, False, True]),
            },
            epsilon_clip=0.1,
        )
        assert dynamics is not None
        self.assertEqual(dynamics.overall.count, 3)
        self.assertAlmostEqual(dynamics.overall.clip_fraction, 2.0 / 3.0)
        self.assertEqual(dynamics.groups["local"].count, 1)
        self.assertEqual(dynamics.groups["remote"].count, 1)
        self.assertEqual(dynamics.groups["defer"].count, 1)
        self.assertIsNone(old.grad)
        self.assertIsNone(new.grad)

    def test_route_head_rows_report_semantics_and_negative_update_direction(self) -> None:
        head = torch.nn.Linear(2, 5)
        head.weight.grad = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        head.bias.grad = torch.arange(5, dtype=torch.float32)
        rows = measure_route_head_gradient_rows(
            head,
            [("idle", "local", "defer", 1, 2)],
        )
        self.assertIsInstance(rows[1], RouteHeadGradientRowTelemetry)
        self.assertEqual(rows[1].semantic_role, "local")
        self.assertEqual(rows[3].destination_uavs, (1,))
        self.assertAlmostEqual(rows[3].parameter_update_direction_mean, -6.5)
        self.assertAlmostEqual(rows[4].bias_update_direction, -4.0)

    def test_sample_remote_mass_is_exact_legal_destination_sum(self) -> None:
        sample = RouteSampleTelemetry(
            batch_index=0,
            time_index=0,
            agent_index=0,
            source_uav=0,
            category="remote",
            legal_remote_destinations=(1, 2),
            remote_probability_by_destination={"1": 0.2, "2": 0.3},
            local_probability=0.4,
            defer_probability=0.1,
            total_remote_probability_mass=0.5,
            best_remote_probability=0.3,
            number_of_legal_remote_destinations=2,
            selected_route_action_index=4,
            selected_destination_uav=2,
            old_route_log_prob=-1.0,
            new_route_log_prob=-0.5,
            route_log_ratio=0.5,
            route_ratio=math.exp(0.5),
            route_approx_kl=math.exp(0.5) - 1.5,
            route_clip_indicator=True,
            advantage=1.0,
            return_target=2.0,
            td_residual=0.25,
        )
        record = sample.record()
        self.assertEqual(record["route_sample_number_of_legal_remote_destinations"], 2)
        self.assertEqual(record["route_sample_remote_probability_by_destination"]["2"], 0.3)

    def test_real_collector_emits_active_samples_and_probability_fields(self) -> None:
        config, environment, _, observations = active_route_fixture()
        kinds = ("local", "remote", "defer", "remote")
        actor, _, masks, proposals, policy = evaluate_route_steps(
            config, environment, (observations,), (kinds,)
        )
        old_branch_log_probs = torch.stack(
            [policy.branch_log_probs[branch] for branch in ACTION_BRANCH_ORDER],
            dim=-1,
        ).clone()
        old_branch_log_probs[..., 0] -= 0.25
        telemetry = collect_route_telemetry(
            policy=policy,
            action_mask_batch=masks,
            proposals=proposals,
            advantage=torch.tensor([[1.0]]),
            return_target=torch.tensor([[2.0]]),
            sequence_valid_mask=torch.ones((1, 1), dtype=torch.bool),
            route_head=actor.action_heads["route"],
            old_branch_log_probs=old_branch_log_probs,
        )
        self.assertEqual(telemetry.route_branch_active_count, len(observations))
        self.assertEqual(len(telemetry.samples), len(observations))
        route_probabilities = policy.probabilities["route"][0, 0]
        route_log_probs = policy.branch_log_probs["route"][0, 0]
        route_indices = policy.action_indices["route"][0, 0]
        for sample in telemetry.samples:
            contract = observations[sample.agent_index].action_masks
            legal_remote_indices = [
                index for index, value in enumerate(contract.route_domain)
                if isinstance(value, int)
                and not isinstance(value, bool)
                and bool(contract.route_mask[index])
            ]
            expected_destinations = tuple(
                contract.route_domain[index] for index in legal_remote_indices
            )
            expected_remote = {
                str(contract.route_domain[index]): float(
                    route_probabilities[sample.agent_index, index].item()
                )
                for index in legal_remote_indices
            }
            self.assertEqual(sample.legal_remote_destinations, expected_destinations)
            self.assertEqual(sample.remote_probability_by_destination, expected_remote)
            self.assertAlmostEqual(
                sample.total_remote_probability_mass,
                sum(expected_remote.values()), places=7,
            )
            self.assertAlmostEqual(
                sample.local_probability,
                float(route_probabilities[
                    sample.agent_index, contract.route_domain.index("local")
                ].item()), places=7,
            )
            self.assertAlmostEqual(
                sample.defer_probability,
                float(route_probabilities[
                    sample.agent_index, contract.route_domain.index("defer")
                ].item()), places=7,
            )
            self.assertAlmostEqual(
                sample.local_probability
                + sample.defer_probability
                + sample.total_remote_probability_mass,
                1.0, places=6,
            )
            self.assertEqual(
                sample.selected_route_action_index,
                int(route_indices[sample.agent_index].item()),
            )
            self.assertAlmostEqual(
                sample.new_route_log_prob,
                float(route_log_probs[sample.agent_index].item()), places=7,
            )
            self.assertAlmostEqual(
                sample.old_route_log_prob,
                float(old_branch_log_probs[0, 0, sample.agent_index, 0].item()),
                places=7,
            )

    def test_route_outcome_tracker_preserves_each_decision_and_terminal_slots(self) -> None:
        def reward(value: float) -> dict[str, float]:
            return {
                "reward": value,
                "workload_penalty": -0.1 * value,
                "completion_component": 0.0,
                "expiration_penalty": 0.0,
                "energy_penalty": -0.01,
            }

        completed_tracker = RouteOutcomeTracker(slot_duration_s=0.5)
        completed_tracker.observe_step(0, {
            "slot": 1,
            "reward": reward(1.0),
            "service": {"routing": [{
                "uav_id": 0, "task_id": "completed", "proposal": "defer",
            }]},
        })
        completed_tracker.observe_step(0, {
            "slot": 2,
            "reward": reward(2.0),
            "service": {"routing": [{
                "uav_id": 0, "task_id": "completed", "proposal": "local",
                "destination": "local", "selected_action_index": 1,
            }], "settled_tasks": [{
                "task_id": "completed", "arrival_slot": 0,
                "completion_slot": 2, "outcome": "done",
            }]},
        })
        completed = sorted(
            completed_tracker.finalize(), key=lambda item: item.route_decision_sequence
        )
        self.assertEqual(len(completed), 2)
        self.assertEqual([item.category for item in completed], ["defer", "local"])
        self.assertEqual([item.route_decision_slot for item in completed], [1, 2])
        self.assertEqual([item.immediate_reward for item in completed], [1.0, 2.0])
        self.assertEqual([item.outcome for item in completed], ["completed", "completed"])
        self.assertEqual([item.terminal_slot for item in completed], [2, 2])
        self.assertEqual([item.completion_slot for item in completed], [2, 2])
        self.assertEqual(completed[1].selected_route_action_index, 1)

        expired_tracker = RouteOutcomeTracker()
        expired_tracker.observe_step(0, {
            "slot": 1,
            "reward": reward(1.0),
            "service": {"routing": [{
                "uav_id": 0, "task_id": "expired", "proposal": "defer",
            }]},
        })
        expired_tracker.observe_step(0, {
            "slot": 2,
            "reward": reward(2.0),
            "service": {"routing": [{
                "uav_id": 0, "task_id": "expired", "proposal": 2,
                "destination": 2, "selected_action_index": 4,
            }]},
        })
        expired_tracker.observe_step(0, {
            "slot": 5,
            "metrics": {"expired_tasks": [{
                "task_id": "expired", "arrival_slot": 0,
                "completion_slot": None,
            }]},
        })
        expired = sorted(
            expired_tracker.finalize(), key=lambda item: item.route_decision_sequence
        )
        self.assertEqual(len(expired), 2)
        self.assertEqual([item.category for item in expired], ["defer", "remote"])
        self.assertEqual([item.outcome for item in expired], ["expired", "expired"])
        self.assertEqual([item.terminal_slot for item in expired], [5, 5])
        self.assertEqual([item.completion_slot for item in expired], [None, None])
        self.assertEqual([item.terminal_age_slots for item in expired], [5, 5])

        truncated_tracker = RouteOutcomeTracker()
        truncated_tracker.observe_step(0, {
            "slot": 3,
            "reward": reward(3.0),
            "service": {"routing": [{
                "uav_id": 1, "task_id": "truncated", "proposal": 1,
                "destination": 1,
            }]},
        })
        truncated_tracker.observe_step(0, {
            "slot": 4,
            "service": {"truncated_tasks": [{
                "task_id": "truncated", "arrival_slot": 1,
                "completion_slot": None, "outcome": "truncated",
            }]},
        })
        truncated = truncated_tracker.finalize()
        self.assertEqual(len(truncated), 1)
        self.assertEqual(truncated[0].outcome, "truncated")
        self.assertEqual(truncated[0].terminal_slot, 4)
        self.assertIsNone(truncated[0].completion_slot)
        self.assertIsNone(truncated[0].end_to_end_latency_slots)
        self.assertEqual(truncated[0].terminal_age_slots, 3)

    def test_tracker_attachment_is_environment_trajectory_neutral(self) -> None:
        config = make_route_config()
        telemetry_on = U2UMECEnvironment(config)
        telemetry_off = U2UMECEnvironment(config)
        reset_on = telemetry_on.reset()
        reset_off = telemetry_off.reset()

        def assert_value_equal(left, right) -> None:
            self.assertEqual(type(left), type(right))
            if isinstance(left, np.ndarray):
                np.testing.assert_array_equal(left, right)
            elif is_dataclass(left):
                for item in fields(left):
                    assert_value_equal(getattr(left, item.name), getattr(right, item.name))
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    assert_value_equal(left[key], right[key])
            elif isinstance(left, (tuple, list)):
                self.assertEqual(len(left), len(right))
                for left_item, right_item in zip(left, right):
                    assert_value_equal(left_item, right_item)
            else:
                self.assertEqual(left, right)

        assert_value_equal(reset_on.observations, reset_off.observations)
        assert_value_equal(reset_on.info, reset_off.info)
        assert_value_equal(reset_on.centralized_state, reset_off.centralized_state)
        tracker = RouteOutcomeTracker(config.environment.slot_duration_s)
        for _ in range(config.environment.episode_horizon):
            proposals_on = telemetry_on.canonical_proposals()
            proposals_off = telemetry_off.canonical_proposals()
            self.assertEqual(proposals_on, proposals_off)
            result_on = telemetry_on.step(proposals_on)
            result_off = telemetry_off.step(proposals_off)
            tracker.observe_step(0, result_on.info)
            assert_value_equal(result_on.observations, result_off.observations)
            self.assertEqual(result_on.reward, result_off.reward)
            self.assertEqual(result_on.terminated, result_off.terminated)
            self.assertEqual(result_on.truncated, result_off.truncated)
            assert_value_equal(result_on.info, result_off.info)
            if result_on.centralized_state is None or result_off.centralized_state is None:
                self.assertIs(result_on.centralized_state, result_off.centralized_state)
            else:
                assert_value_equal(result_on.centralized_state, result_off.centralized_state)

    def test_real_collector_writer_persists_samples_and_outcomes(self) -> None:
        config, environment, _, observations = active_route_fixture()
        kinds = ("local", "remote", "defer", "remote")
        actor, _, masks, proposals, policy = evaluate_route_steps(
            config, environment, (observations,), (kinds,)
        )
        telemetry = collect_route_telemetry(
            policy=policy,
            action_mask_batch=masks,
            proposals=proposals,
            advantage=torch.tensor([[1.0]]),
            return_target=torch.tensor([[2.0]]),
            sequence_valid_mask=torch.ones((1, 1), dtype=torch.bool),
            route_head=actor.action_heads["route"],
        )
        tracker = RouteOutcomeTracker()
        tracker.observe_step(0, {
            "slot": 1,
            "reward": {"reward": 1.0},
            "service": {"routing": [{
                "uav_id": 0, "task_id": "persisted", "proposal": "local",
                "selected_action_index": 1,
            }]},
        })
        tracker.observe_step(0, {
            "slot": 2,
            "service": {"settled_tasks": [{
                "task_id": "persisted", "arrival_slot": 0,
                "completion_slot": 2, "outcome": "completed",
            }]},
        })
        base_training = training_result()
        first_update = base_training.updates[0]
        epochs = tuple(
            replace(
                epoch,
                route_telemetry=telemetry if index == 0 else epoch.route_telemetry,
            )
            for index, epoch in enumerate(first_update.output.epoch_diagnostics)
        )
        update = replace(
            first_update,
            output=replace(first_update.output, epoch_diagnostics=epochs),
        )
        training = replace(
            base_training,
            updates=(update,) + base_training.updates[1:],
            route_outcomes=tracker.finalize(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            isolated = load_run_config(cli_overrides={
                "mode": "rl",
                "method_id": "ca_gat_mappo",
                "training.formal_rl_enabled": True,
                "training.mappo.training_device": "cpu",
                "environment.episode_horizon": 32,
                "training.mappo.max_training_episodes": 8,
                "training.mappo.max_training_environment_steps": 256,
                "training.mappo.evaluation_interval_steps": 256,
                "training.mappo.checkpoint_interval_steps": 128,
                "output.logs_dir": str(root / "logs"),
                "output.dashboard_logs_dir": str(root / "dashboard_logs"),
                "output.plots_dir": str(root / "plots"),
            })
            outcome = write_cagat_mappo_training_artifacts(isolated, training)
            paths = isolated.artifact_paths()
            raw_records = [
                json.loads(line)
                for line in Path(paths["raw_metrics"]).read_text(encoding="utf-8").splitlines()
            ]
            sample_records = [
                record for record in raw_records if record.get("record_type") == "route_sample"
            ]
            outcome_records = [
                record for record in raw_records if record.get("record_type") == "route_outcome"
            ]
            self.assertEqual(len(sample_records), len(telemetry.samples))
            self.assertGreater(len(sample_records), 0)
            self.assertEqual(len(outcome_records), 1)
            self.assertIn("route_sample_total_remote_probability_mass", sample_records[0])
            self.assertIn("route_sample_ratio", sample_records[0])
            self.assertIn("route_outcome_reward_component_scope", outcome_records[0])
            self.assertIn("route_outcome_terminal_slot", outcome_records[0])
            self.assertEqual(outcome_records[0]["route_outcome_reward_component_scope"], "team_level")
            self.assertEqual(outcome_records[0]["route_outcome_terminal_slot"], 2)
            self.assertTrue(all(str(root) in path for path in outcome.artifacts))
            self.assertEqual(_json_lines(sample_records), _json_lines(sample_records))
            self.assertEqual(_csv_text(sample_records), _csv_text(sample_records))
            with Path(paths["dashboard_csv"]).open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            csv_samples = [row for row in csv_rows if row["record_type"] == "route_sample"]
            self.assertEqual(len(csv_samples), len(sample_records))
            self.assertAlmostEqual(
                float(csv_samples[0]["route_sample_total_remote_probability_mass"]),
                sample_records[0]["route_sample_total_remote_probability_mass"],
            )
            self.assertIsNone(outcome_records[0]["route_outcome_tx_start_slot"])
            outcome_csv = next(row for row in csv_rows if row["record_type"] == "route_outcome")
            self.assertEqual(outcome_csv["route_outcome_tx_start_slot"], "")

    def test_route_outcome_tracker_joins_terminal_and_team_reward_timing(self) -> None:
        tracker = RouteOutcomeTracker(slot_duration_s=0.5)
        reward = {
            "reward": 1.0,
            "workload_penalty": -0.2,
            "completion_component": 1.5,
            "expiration_penalty": 0.0,
            "energy_penalty": -0.1,
        }
        tracker.observe_step(0, {
            "slot": 2,
            "reward": reward,
            "service": {
                "routing": [{"uav_id": 0, "task_id": "local-1", "destination": "local"}],
            },
        })
        tracker.observe_step(0, {
            "slot": 3,
            "reward": reward,
            "service": {
                "settled_tasks": [{
                    "task_id": "local-1",
                    "arrival_slot": 1,
                    "completion_slot": 3,
                    "outcome": "completed",
                }],
            },
        })
        tracker.observe_step(0, {
            "slot": 5,
            "service": {
                "routing": [{"uav_id": 0, "task_id": "remote-1", "destination": 2}],
            },
        })
        tracker.observe_step(0, {
            "slot": 6,
            "metrics": {
                "expired_tasks": [{
                    "task_id": "remote-1",
                    "arrival_slot": 4,
                    "completion_slot": 6,
                }],
            },
        })
        tracker.observe_step(0, {
            "slot": 7,
            "service": {
                "routing": [{"uav_id": 0, "task_id": "defer-1", "destination": "defer"}],
            },
        })
        outcomes = tracker.finalize()
        self.assertEqual([item.category for item in outcomes], ["defer", "local", "remote"])
        local = next(item for item in outcomes if item.task_id == "local-1")
        self.assertEqual(local.end_to_end_latency_slots, 2)
        self.assertEqual(local.reward_component_scope, "team_level")
        aggregate = aggregate_route_outcomes(outcomes)
        self.assertEqual(aggregate["local"]["completed"], 1)
        self.assertEqual(aggregate["remote"]["expired"], 1)
        self.assertEqual(aggregate["defer"]["truncated"], 1)

    def test_tracker_matches_real_environment_route_and_nested_link_snapshots(self) -> None:
        tracker = RouteOutcomeTracker(slot_duration_s=0.5)
        tracker.observe_step(0, {
            "slot": 2,
            "reward": {"reward": 0.25, "workload_penalty": -0.05, "completion_component": 0.0, "expiration_penalty": 0.0, "energy_penalty": -0.01},
            "service": {"routing": [{"uav_id": 0, "task_id": 7, "proposal": 3, "destination": 3}]},
        })
        tracker.observe_step(0, {
            "slot": 3,
            "reward": {"reward": 9.0, "workload_penalty": -9.0, "completion_component": 2.0, "expiration_penalty": 0.0, "energy_penalty": -1.0},
            "service": {
                "links": [{"task_services": [{"task_id": 7, "amount": 10.0}]}],
                "settled_tasks": [{"task_id": 7, "arrival_slot": 1, "completion_slot": 4, "cpu_entry_slot": 3, "outcome": "done"}],
            },
        })
        tracker.observe_step(0, {
            "slot": 8,
            "service": {"routing": [{"uav_id": 1, "task_id": "actual-local", "proposal": "local", "destination": 1}]},
        })
        outcomes = tracker.finalize()
        remote = next(item for item in outcomes if item.task_id == 7)
        local = next(item for item in outcomes if item.task_id == "actual-local")
        self.assertEqual(remote.category, "remote")
        self.assertEqual(remote.tx_start_slot, 3)
        self.assertEqual(remote.tx_complete_slot, 3)
        self.assertEqual(remote.outcome, "completed")
        self.assertEqual(remote.end_to_end_latency_slots, 3)
        self.assertAlmostEqual(remote.immediate_reward, 0.25)
        self.assertAlmostEqual(remote.workload_penalty_component, -0.05)
        self.assertEqual(local.category, "local")
    def test_new_artifact_fields_keep_json_null_and_csv_blank_semantics(self) -> None:
        config = make_update_config()
        row = _base_record(config, "insufficient-horizon")
        row.update({
            "record_type": "route_outcome",
            "telemetry_scope": "route_decision_to_task_outcome",
            "series_index": 1,
        })
        csv_text = _csv_text([row])
        json_text = _json_lines([row])
        self.assertIn("route_outcome_task_id", csv_text)
        self.assertIn("route_outcome_task_id", json_text)
        self.assertIn(",,", csv_text)
        self.assertIn('"route_outcome_task_id": null', json_text)


if __name__ == "__main__":
    unittest.main()
