"""Factual queue reconstruction and flat-leaf interpretation regression tests."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from src.evaluation.limited_route_diagnostic import (
    analyze_run, compare_runs, leaf_metrics, main, max_consecutive,
)


def task(task_id, *, generation=0, deadline=3, binding=None, source=0, done=False):
    return {
        "method_id": "ca_gat_mappo", "evaluation_seed": 1042,
        "task_id": task_id, "source_uav": source, "generation_slot": generation,
        "deadline_slot": deadline, "input_bits": 10.0, "required_cycles": 100.0,
        "route_bind_slot": binding, "destination_uav": source if binding is not None else None,
        "final_status": "DONE" if done else "EXPIRED",
        "terminal_stage": None if done else "UNBOUND",
    }


def decision(task_id, slot, *, defer=True, probabilities=None):
    probs = probabilities or ([0.0, 0.10, 0.35, 0.30, 0.25] if defer else [0.0, 0.70, 0.10, 0.10, 0.10])
    index = max(range(1, 5), key=lambda i: probs[i])
    label = "Local" if index == 1 else "Defer" if index == 2 else "Remote"
    mass = {"Local": probs[1], "Defer": probs[2], "Remote": sum(probs[3:])}
    return {
        "evaluation_seed": 1042, "task_id": task_id, "step": slot, "uav_id": 0,
        "leaf_scores": [0.0, *(math.log(p) for p in probs[1:])],
        "leaf_probabilities": probs, "route_legal_mask": [False, True, True, True, True],
        "route_action_domain": ["idle", "local", "defer", "remote_uav_1", "remote_uav_2"],
        "chosen_leaf_index": index, "leaf_argmax_index": index,
        "selected_top_level": label, "actor_decision_argmax": label,
        "selected_remote_id": index - 2 if index >= 3 else None,
        "group_mass": mass, "group_mass_argmax": max(mass, key=mass.get),
        "legal_route_candidates": ["local", "defer", "remote_uav_1", "remote_uav_2"],
    }


def factual_fixture():
    # Task2 expires behind task0 without ever becoming head. Task1 waits
    # behind the deferred head at slots2/3, then binds after task0 expires.
    tasks = [task(0), task(1, generation=1, deadline=5, binding=4, done=True),
             task(2, generation=1, deadline=3), task(3, generation=5, deadline=10)]
    records = [decision(0, s) for s in (1, 2, 3)] + [decision(1, 4, defer=False)]
    return tasks, records, {1042: 6}


class LimitedRouteDiagnosticTests(unittest.TestCase):
    def test_remote_mass_does_not_replace_actual_defer_leaf(self):
        row = decision(0, 1)
        result = leaf_metrics(row)
        self.assertEqual(row["group_mass_argmax"], "Remote")
        self.assertEqual(result["chosen_leaf_label"], "defer")
        self.assertTrue(result["remote_mass_argmax_differs_from_actual_leaf"])
        self.assertAlmostEqual(result["defer_minus_best_binding_probability"], 0.05)

    def test_reconstructs_edf_blocked_and_zero_opportunity_tasks_separately(self):
        tasks, records, horizons = factual_fixture()
        result = analyze_run(tasks, records, horizons)
        head = result["tasks"]["1042:0"]
        queued = result["tasks"]["1042:1"]
        invisible = result["tasks"]["1042:2"]
        tail = result["tasks"]["1042:3"]
        self.assertEqual(head["max_consecutive_defer_slots"], 3)
        self.assertEqual(queued["first_decision_slot"], 4)
        self.assertEqual(queued["first_decision_slack_slots"], 2)
        self.assertEqual(queued["first_decision_deadline_offset_slots"], 1)
        self.assertEqual(queued["post_binding_remaining_service_slots"], 1)
        self.assertEqual(queued["queued_behind_deferred_edf_head_slots"], 2)
        self.assertTrue(invisible["never_observed_as_head"])
        self.assertTrue(invisible["eligible_slots_all_behind_other_head"])
        self.assertEqual(invisible["defer_count"], 0)
        self.assertTrue(tail["no_eligible_route_slot_before_terminal"])
        self.assertFalse(tail["eligible_slots_all_behind_other_head"])
        self.assertTrue(result["summary"]["exact_edf_head_reconstruction_passed"])

    def test_missing_active_row_is_rejected(self):
        tasks, rows, horizons = factual_fixture()
        with self.assertRaisesRegex(ValueError, "EDF head differs"):
            analyze_run(tasks, rows[1:], horizons)

    def test_wrong_edf_head_is_rejected(self):
        tasks, rows, horizons = factual_fixture()
        rows[1]["task_id"] = 2
        with self.assertRaisesRegex(ValueError, "EDF head differs"):
            analyze_run(tasks, rows, horizons)

    def test_binding_inconsistent_with_lifecycle_is_rejected(self):
        tasks, rows, horizons = factual_fixture()
        tasks[1]["route_bind_slot"] = 5
        with self.assertRaisesRegex(ValueError, "binding time"):
            analyze_run(tasks, rows, horizons)

    def test_illegal_leaf_mass_and_wrong_choice_are_rejected(self):
        row = decision(0, 1)
        row["leaf_probabilities"][0] = 0.01
        row["leaf_probabilities"][1] -= 0.01
        with self.assertRaisesRegex(ValueError, "Illegal leaf"):
            leaf_metrics(row)
        row = decision(0, 1)
        row["chosen_leaf_index"] = 3
        with self.assertRaisesRegex(ValueError, "argmax"):
            leaf_metrics(row)

    def test_pairing_checks_workload_and_does_not_invent_causality(self):
        tasks, rows, horizons = factual_fixture()
        old = analyze_run(tasks, rows, horizons)
        new = copy.deepcopy(old)
        new["tasks"]["1042:1"]["lifecycle"]["final_status"] = "EXPIRED"
        new["tasks"]["1042:1"]["lifecycle"]["terminal_stage"] = "CPU_WAIT"
        paired = compare_runs(old, new)
        self.assertEqual((paired["lost"], paired["gained"]), (1, 0))
        self.assertEqual(paired["lost_task_ids"], ["1042:1"])
        self.assertFalse(paired["causal_intervention_performed"])
        new["tasks"]["1042:1"]["lifecycle"]["required_cycles"] = 101.0
        with self.assertRaisesRegex(ValueError, "workload/deadline"):
            compare_runs(old, new)

    def test_duplicate_source_slot_is_rejected(self):
        tasks, rows, horizons = factual_fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate source/slot"):
            analyze_run(tasks, [*rows, rows[0]], horizons)

    def test_consecutive_defer_excludes_interrupted_slots(self):
        self.assertEqual(max_consecutive([1, 2, 4, 5, 6, 8]), 3)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "logs") as directory:
            root = Path(directory)
            output = root / "metadata/audits/seed44_limited_diagnostic/test/route_diagnostic.json"
            output.parent.mkdir(parents=True)
            output.write_text(json.dumps({"historical": True}), encoding="utf-8")
            with self.assertRaises(FileExistsError):
                main(["--root", str(root), "--output", str(output)])
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"historical": True})

    def test_output_outside_dedicated_audit_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "logs") as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "dedicated audit"):
                main(["--root", str(root), "--output", str(root / "route_diagnostic.json")])


if __name__ == "__main__":
    unittest.main()
