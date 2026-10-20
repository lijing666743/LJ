"""Endpoint selection fixtures only; not experimental measurements."""

import math

import pytest

from src.evaluation.c_seed44_report import endpoint_statistics


def row(seed, step=40000, completion=60, generated=605):
    return {'training_seed': seed, 'checkpoint_steps': step, 'generated': generated,
            'completion_pct': completion, 'total_energy_j': 10,
            'energy_per_DONE_j': .2, 'mean_episode_reward': 1}


def test_fixed_endpoint_counts_three_runs_without_best_point_selection():
    rows = [row(42, completion=40), row(43, completion=60), row(44, completion=80), row(44, 24000, 100)]
    result = endpoint_statistics(rows)
    assert result['completion_pct']['mean'] == 60
    assert result['completion_pct']['sample_std'] == 20
    assert result['independent_training_runs'] == 3


@pytest.mark.parametrize('rows', [
    [row(42), row(43), row(44, 36000)],
    [row(42), row(43), row(43)],
    [row(s) for s in (42, 43, 44) for _ in range(5)],
    [row(42), row(43), row(44, generated=600)],
    [row(42), row(43), row(44, completion=math.nan)],
])
def test_invalid_endpoints_are_rejected(rows):
    with pytest.raises(ValueError):
        endpoint_statistics(rows)
