"""Accounting and identity boundaries for the pure saved-CPU diagnostic."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.evaluation.checkpoint_energy_audit import EnergyAuditError
from src.evaluation.limited_cpu_diagnostic import (
    SERVICE_SCHEMA, analyze_cpu_episode, main, pool_cpu_summaries, validate_identity,
)


def fixture():
    identity = {"evaluation_run_id": "unit-offline", "method_id": "ca_gat_mappo",
                "evaluation_seed": 1042, "episode_id": 0}
    common = {**identity, "schema_version": SERVICE_SCHEMA}
    environment = SimpleNamespace(
        episode_horizon=5, uav_count=1, slot_duration_s=1.0, energy_tolerance_j=1e-12,
        profile_assignment=("unit",), profile_perturbations=((0, 0, 0, 0),),
        profile_ratios={"unit": (1, 1, 1, 1)}, reference_cpu_frequency_hz=4.0,
        reference_cpu_coefficient=.01, reference_initial_energy_j=10.0,
    )
    config = SimpleNamespace(environment=environment,
                             action=SimpleNamespace(cpu_frequency_levels=(0.0, .25, .5, 1.0)))
    tasks = []
    for tid, status, bind, deadline, required, remaining, first, done, expired in (
        (0, "DONE", 0, 4, 4.0, 0.0, 1, 1, None),
        (1, "EXPIRED", 1, 3, 6.0, 4.0, 2, None, 3),
        (2, "TRUNCATED", 3, 9, 4.0, 3.0, 4, None, None),
    ):
        tasks.append({**common, "task_id": tid, "source_uav": 0, "destination_uav": 0,
                      "route_bind_slot": bind, "tx_complete_slot": None,
                      "deadline_slot": deadline, "required_cycles": required,
                      "remaining_cycles": remaining, "cpu_first_service_slot": first,
                      "done_slot": done, "expired_slot": expired, "final_status": status,
                      "terminal_stage": None if status == "DONE" else "CPU_SERVICE"})
    cpu, tx = [], []
    residual = 10.0
    for slot, qlength, qcycles, tid, frequency, cycles in (
        (0, 0, 0, None, 0.0, 0.0), (1, 1, 4.0, 0, 4.0, 4.0),
        (2, 1, 6.0, 1, 2.0, 2.0), (3, 1, 4.0, None, 0.0, 0.0),
        (4, 1, 4.0, 2, 1.0, 1.0),
    ):
        energy = .01 * frequency ** 2 * cycles
        residual -= energy
        cpu.append({**common, "slot": slot, "uav_id": 0, "cpu_queue_length": qlength,
                    "cpu_queue_cycles": qcycles, "selected_task_id": tid, "served_task_id": tid,
                    "cpu_frequency": frequency, "served_cycles": cycles,
                    "cpu_energy_j": energy, "residual_energy_j": residual})
        tx.append({**common, "slot": slot, "uav_id": 0, "link_summary": True, "tx_energy_j": 0.0})
    metrics = {"generated": 3, "completed": 1, "expired": 1, "truncated": 1,
               "cpu_energy_j": .73, "tx_energy_j": 0.0, "total_energy_j": .73}
    return config, tasks, cpu, tx, metrics, identity


def analyze(parts):
    config, tasks, cpu, tx, metrics, identity = parts
    return analyze_cpu_episode(config, tasks, cpu, tx, metrics, identity=identity)


class TestLimitedCpuDiagnostic(unittest.TestCase):
    def test_actual_outcome_energy_frequency_and_cycle_conservation(self):
        parts = fixture()
        original = deepcopy(parts[1:])
        result = analyze(parts)
        self.assertEqual(parts[1:], original)
        summary = result["summary"]
        self.assertAlmostEqual(summary["cpu_energy_j"], .73)
        self.assertAlmostEqual(summary["served_cycles"], 7)
        self.assertAlmostEqual(summary["outcome_distribution"]["DONE"]["cpu_energy_j"], .64)
        self.assertAlmostEqual(summary["outcome_distribution"]["EXPIRED"]["cpu_energy_j"], .08)
        self.assertAlmostEqual(summary["outcome_distribution"]["TRUNCATED"]["cpu_energy_j"], .01)
        self.assertEqual(summary["backlogged_no_service_energy_feasible_slots"], 1)
        self.assertEqual(summary["frequency_distribution"]["1.0"]["positive_service_slots"], 1)
        self.assertEqual(result["checks"]["task_cycle_and_first_service"], 3)

    def test_pool_sums_counts_and_keeps_hardware_frequency_fixed(self):
        episode = analyze(fixture())
        summary = pool_cpu_summaries([episode, deepcopy(episode)])
        self.assertEqual(summary["generated"], 6)
        self.assertAlmostEqual(summary["cpu_energy_j"], 1.46)
        self.assertEqual(summary["uav_distribution"]["0"]["maximum_frequency_hz"], 4.0)
        self.assertEqual(summary["outcome_distribution"]["EXPIRED"]["tasks_ever_served"], 2)

    def test_rejects_mixed_method_run_scene_or_episode_and_schema(self):
        for key, value in (("method_id", "heuristic"), ("evaluation_run_id", "foreign"),
                           ("evaluation_seed", 1043), ("episode_id", 2), ("schema_version", "other")):
            with self.subTest(key=key):
                parts = fixture()
                parts[2][1][key] = value
                with self.assertRaises(EnergyAuditError):
                    analyze(parts)

    def test_missing_declared_identity_is_rejected(self):
        with self.assertRaises(EnergyAuditError):
            validate_identity([], {"method_id": "ca_gat_mappo"})

    def test_incomplete_or_duplicate_cpu_grid_is_rejected(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate):
                parts = fixture()
                parts[2].append(deepcopy(parts[2][0])) if duplicate else parts[2].pop()
                with self.assertRaises(EnergyAuditError):
                    analyze(parts)

    def test_tx_summary_is_counted_once_and_extra_task_row_has_no_energy_effect(self):
        parts = fixture()
        extra = deepcopy(parts[3][1])
        extra.update(link_summary=False, tx_energy_j=999)
        parts[3].append(extra)
        self.assertEqual(analyze(parts)["summary"]["tx_energy_j"], 0.0)
        extra["link_summary"] = True
        with self.assertRaises(EnergyAuditError):
            analyze(parts)

    def test_wrong_cycles_debit_first_service_queue_or_outcome_is_rejected(self):
        for target, key, value in ((2, "served_cycles", 3.5), (2, "cpu_energy_j", .65),
                                   (2, "cpu_queue_cycles", 5.0), (1, "cpu_first_service_slot", 2),
                                   (1, "final_status", "TRUNCATED")):
            with self.subTest(key=key):
                parts = fixture()
                parts[target][0 if target == 1 else 1][key] = value
                with self.assertRaises(EnergyAuditError):
                    analyze(parts)

    def test_negative_or_nonfinite_actual_debit_is_rejected(self):
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                parts = fixture()
                parts[2][1]["cpu_energy_j"] = value
                with self.assertRaises(EnergyAuditError):
                    analyze(parts)

    def test_source_queue_edf_head_is_required(self):
        parts = fixture()
        # Same source, overlapping task with an earlier deadline prevents task0
        # from being a legal EDF source head at slot1.
        parts[1][1].update(route_bind_slot=0, deadline_slot=2, expired_slot=2)
        parts[2][1].update(cpu_queue_length=2, cpu_queue_cycles=10)
        with self.assertRaisesRegex(EnergyAuditError, "EDF head"):
            analyze(parts)

    def test_negative_slack_uses_logged_remaining_cycles_and_inclusive_deadline(self):
        parts = fixture()
        parts[1][1].update(required_cycles=14.0, remaining_cycles=12.0)
        parts[2][2]["cpu_queue_cycles"] = 14.0
        parts[2][3]["cpu_queue_cycles"] = 12.0
        result = analyze(parts)
        self.assertEqual(result["summary"]["negative_optimistic_slack_service_slots"], 1)
        self.assertAlmostEqual(result["summary"]["cpu_energy_on_negative_optimistic_slack_j"], .08)

    def test_backlogged_idle_can_be_energy_blocked_without_cpu_debit(self):
        parts = fixture()
        config, tasks, cpu, tx, metrics, _ = parts
        config.environment.episode_horizon = 2
        config.environment.reference_initial_energy_j = .005
        tasks[:] = [tasks[0]]
        tasks[0].update(final_status="EXPIRED", done_slot=None, expired_slot=1,
                        deadline_slot=1, remaining_cycles=4.0, cpu_first_service_slot=None)
        cpu[:] = cpu[:2]
        for row in cpu:
            row.update(cpu_energy_j=0.0, residual_energy_j=.005, served_cycles=0.0,
                       cpu_frequency=0.0, served_task_id=None, selected_task_id=None)
        tx[:] = tx[:2]
        metrics.update(generated=1, completed=0, expired=1, truncated=0,
                       cpu_energy_j=0.0, total_energy_j=0.0)
        summary = analyze(parts)["summary"]
        self.assertEqual(summary["backlogged_no_service_all_heads_energy_blocked_slots"], 1)
        self.assertEqual(summary["backlogged_no_service_energy_feasible_slots"], 0)

    def test_existing_output_is_refused_before_reading_or_rebuilding_inputs(self):
        with patch("pathlib.Path.exists", return_value=True), \
             patch("src.evaluation.limited_cpu_diagnostic.build_cpu_diagnostic") as builder:
            with self.assertRaises(FileExistsError):
                main(["--output", "already-existing.json"])
            builder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
