"""Pooling rules for the meeting table; fixtures are arithmetic, not experiments."""

import pytest

from src.evaluation.group_meeting_figures import scene_comparison_series, summarize_episodes


def episode(generated, completed, energy, reward, delay):
    return {"metrics": {"generated": generated, "completed": completed, "expired": generated-completed,
        "truncated": 0, "total_energy_j": energy, "cpu_energy_j": energy, "tx_energy_j": 0,
        "total_reward": reward, "completed_task_e2e_delay_s": delay}}


def test_completion_pools_task_counts_instead_of_averaging_scene_ratios():
    result = summarize_episodes([episode(10, 9, 1, 10, 1), episode(100, 10, 2, 30, 3)])
    assert result["completion_pct"] == pytest.approx(100*19/110)
    assert result["mean_per_scene_completion_pct"] == pytest.approx(50)
    assert result["mean_episode_reward"] == 20


def test_energy_and_completed_delay_use_their_own_pooled_denominators():
    result = summarize_episodes([episode(10, 9, 1, 10, 1), episode(100, 10, 2, 30, 3)])
    assert result["energy_per_DONE_j"] == pytest.approx(3/19)
    assert result["completed_task_delay_s"] == pytest.approx(39/19)


def test_no_completed_tasks_leaves_completion_dependent_metrics_undefined():
    result = summarize_episodes([episode(10, 0, 1, -10, None)])
    assert result["completion_pct"] == 0
    assert result["energy_per_DONE_j"] is None
    assert result["completed_task_delay_s"] is None


def test_empty_records_cannot_be_reported_as_zero_performance():
    with pytest.raises(ValueError, match="No episode"):
        summarize_episodes([])


def test_task_outcomes_must_conserve_generated_count():
    record = episode(10, 9, 1, 10, 1)
    record["metrics"]["expired"] = 2
    with pytest.raises(ValueError, match="Task count"):
        summarize_episodes([record])


def test_total_energy_cannot_disagree_with_its_components():
    record = episode(10, 9, 1, 10, 1)
    record["metrics"]["tx_energy_j"] = 1
    with pytest.raises(ValueError, match="Energy conservation"):
        summarize_episodes([record])


def scene_records():
    # Small extract of verified count pairs; this is a chart-contract fixture.
    pairs = {"local_only": (73, 60), "heuristic": (100, 64), "C": (100, 95)}
    return {"per_scene": [{"method": method, "checkpoint_steps": 32000, "seed": seed,
        "generated": generated, "completed": done, "completion_pct": 100*done/generated}
        for method, completed in pairs.items()
        for seed, generated, done in zip((1042, 1043), (135, 122), completed)]}


def test_percentage_bar_labels_follow_scene_specific_denominators():
    result = scene_comparison_series(scene_records(), scene_ids=(1042, 1043))
    assert result["bar_height_metric"] == result["bar_label_metric"] == "completion_pct"
    assert result["series"]["C"]["completed"] == [100, 95]
    assert result["series"]["C"]["bar_labels"] == ["74.07%", "77.87%"]
    assert result["series"]["C"]["completion_pct"][1] > result["series"]["C"]["completion_pct"][0]


def test_scene_comparison_rejects_different_method_arrivals():
    data = scene_records()
    data["per_scene"][0].update(generated=136, completion_pct=100*73/136)
    with pytest.raises(ValueError, match="different arrivals"):
        scene_comparison_series(data, scene_ids=(1042, 1043))


def test_scene_comparison_rejects_count_used_as_percentage():
    data = scene_records()
    data["per_scene"][-1]["completion_pct"] = 95
    with pytest.raises(ValueError, match="percentage differs"):
        scene_comparison_series(data, scene_ids=(1042, 1043))


def test_scene_comparison_rejects_ambiguous_duplicate_record():
    data = scene_records()
    data["per_scene"].append(dict(data["per_scene"][0]))
    with pytest.raises(ValueError, match="Expected one"):
        scene_comparison_series(data, scene_ids=(1042, 1043))
