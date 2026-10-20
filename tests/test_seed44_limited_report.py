"""Matched-task accounting fixtures; no experimental measurements are invented."""

from copy import deepcopy

import pytest

from src.evaluation.seed44_limited_report import paired_cpu_tasks, build_summary


def case(rows, scene=1042):
    return {"episodes": [{"identity": {"evaluation_seed": scene}, "tasks": rows}]}


def task(tid, status, energy, cycles=4):
    return {"task_id": tid, "final_status": status, "cpu_energy_j": energy,
            "required_cycles": 4, "served_cycles": cycles}


def test_same_completed_workload_is_compared_with_same_task_ids():
    left = case([task(0, "DONE", 2), task(1, "DONE", 1), task(2, "EXPIRED", .5, 2)])
    right = case([task(0, "DONE", 1), task(1, "EXPIRED", .2, 1), task(2, "DONE", 1)])
    result = paired_cpu_tasks(left, right)
    assert result["paired_tasks"] == 3
    assert result["groups"]["both_done"] == {
        "task_count": 1, "learned_cpu_energy_j": 2.0, "reference_cpu_energy_j": 1.0,
        "learned_served_cycles": 4.0, "reference_served_cycles": 4.0}
    assert result["groups"]["learned_only_done"]["task_count"] == 1
    assert result["groups"]["reference_only_done"]["task_count"] == 1


@pytest.mark.parametrize("change", ["identity", "workload", "duplicate", "done_cycles"])
def test_mismatched_paired_task_evidence_is_rejected(change):
    left = case([task(0, "DONE", 2)])
    right = deepcopy(left)
    row = right["episodes"][0]["tasks"][0]
    if change == "identity": row["task_id"] = 1
    elif change == "workload": row["required_cycles"] = 5
    elif change == "duplicate": right["episodes"][0]["tasks"].append(deepcopy(row))
    else: row["served_cycles"] = 3
    with pytest.raises(ValueError):
        paired_cpu_tasks(left, right)


def test_same_task_number_in_different_scene_is_not_the_same_identity():
    with pytest.raises(ValueError, match="identities"):
        paired_cpu_tasks(case([task(0, "DONE", 2)], 1042), case([task(0, "DONE", 2)], 1043))


def test_superseded_slack_schema_cannot_enter_final_report():
    with pytest.raises(ValueError, match="schema v2"):
        build_summary({"schema_version": "limited_route_diagnostic.v1"}, {}, {})
