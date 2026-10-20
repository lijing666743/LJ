"""Synthetic engineering fixtures; no generated experiment measurements."""

import math

import pytest

from src.evaluation.c_seed43_report import endpoint_statistics


def row(seed, step, completion):
    return {'training_seed':seed,'checkpoint_steps':step,'completion_pct':completion,
            'total_energy_j':10,'energy_per_DONE_j':.2,'mean_episode_reward':1}


def test_primary_endpoint_does_not_pick_each_seeds_best_checkpoint():
    rows=[row(42,20000,90),row(43,20000,99),row(42,40000,60),row(43,40000,40)]
    result=endpoint_statistics(rows)
    assert result['completion_pct']['mean']==50
    assert math.isclose(result['completion_pct']['sample_std'],math.sqrt(200))
    assert result['independent_training_runs']==2
    assert result['evaluation_scenes_per_run']==5


def test_missing_primary_endpoint_cannot_be_replaced_with_intermediate():
    with pytest.raises(ValueError):
        endpoint_statistics([row(42,40000,60),row(43,36000,60)])


def test_duplicate_seed_is_not_an_additional_independent_training_run():
    with pytest.raises(ValueError):
        endpoint_statistics([row(42,40000,60),row(42,40000,40)])


def test_training_seed_sample_count_cannot_be_inflated_by_repeated_scenes():
    with pytest.raises(ValueError):
        endpoint_statistics([row(seed,40000,60) for seed in (42,43) for _ in range(5)])
