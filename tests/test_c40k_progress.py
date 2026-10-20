"""Engineering fixtures for pooled development summaries, not experiment data."""

import copy

import pytest

from src.evaluation.c40k_progress import compare_baselines, summarize


def metric_record(generated, completed, energy):
    return {'metrics': {'generated': generated, 'completed': completed,
            'expired': generated - completed, 'truncated': 0,
            'total_energy_j': energy, 'cpu_energy_j': energy * 0.9,
            'tx_energy_j': energy * 0.1, 'total_reward': completed - generated}}


def test_completion_and_energy_use_pooled_denominators():
    result = summarize([metric_record(1, 1, 2), metric_record(9, 0, 8)])
    assert result['completion_pct'] == 10
    assert result['generated'] == 10 and result['completed'] == 1
    assert result['energy_per_DONE_j'] == 10
    assert result['mean_episode_reward'] == -4.5


def test_no_completions_has_no_energy_per_done():
    result = summarize([metric_record(3, 0, 6)])
    assert result['energy_per_DONE_j'] is None


def baseline_record():
    return {**metric_record(135, 5, 2), 'method_id': 'heuristic',
            'evaluation_seed': 1042, 'action_trace_sha256': 'fixture-action',
            'external_trace_sha256': 'fixture-external', 'hidden_trace_sha256': [],
            'invariants': {'conservation': True}, 'episode_horizon': 500}


@pytest.mark.parametrize('field', ['metrics', 'action_trace_sha256', 'external_trace_sha256'])
def test_baseline_drift_is_rejected(field):
    old = baseline_record()
    actual = copy.deepcopy(old)
    if field == 'metrics':
        actual[field]['completed'] += 1
    else:
        actual[field] = 'changed'
    with pytest.raises(AssertionError):
        compare_baselines([actual], [old])


def test_only_run_identity_may_change_for_fixed_baseline():
    old = baseline_record()
    actual = {**old, 'evaluation_run_id': 'different-checkpoint'}
    assert compare_baselines([actual], [old]) == 1


def test_task_comparison_pairs_identity_even_when_order_changes():
    from src.evaluation.c40k_progress_report import paired_outcomes
    old = [{'evaluation_seed': 1042, 'task_id': i, 'method_id': 'ca_gat_mappo',
            'final_status': 'DONE' if i == 0 else 'EXPIRED', 'destination_uav': 0}
           for i in range(605)]
    new = copy.deepcopy(old)
    new[0]['final_status'] = 'EXPIRED'
    new[1]['final_status'] = 'DONE'
    counts, changed = paired_outcomes(old, list(reversed(new)))
    assert counts['new_only_done'] == counts['old_only_done'] == 1
    assert counts['net_DONE_change'] == 0 and len(changed) == 2


def test_missing_task_cannot_be_treated_as_an_outcome_change():
    from src.evaluation.c40k_progress_report import paired_outcomes
    with pytest.raises(AssertionError):
        paired_outcomes([], [])


def test_existing_audit_reuses_identical_json_without_rewriting(tmp_path):
    from src.evaluation.c40k_progress_report import write_verified_json
    path = tmp_path / 'audit.json'
    write_verified_json(path, {40000: {'completed': 7}})
    before = path.read_bytes()
    write_verified_json(path, {40000: {'completed': 7}})
    assert path.read_bytes() == before


def test_changed_audit_is_rejected_without_overwriting(tmp_path):
    from src.evaluation.c40k_progress_report import write_verified_json
    path = tmp_path / 'audit.json'
    write_verified_json(path, {'completed': 7})
    before = path.read_bytes()
    with pytest.raises(AssertionError):
        write_verified_json(path, {'completed': 8})
    assert path.read_bytes() == before
