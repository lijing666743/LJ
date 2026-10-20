"""Reproducible, frozen C36k/C40k development evaluation and source audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch

from ..config import load_run_config
from ..models.ca_gat_mappo_checkpoint import load_checkpoint_payload
from ..models.ca_gat_mappo_continuation import assert_state_equal, verify_replay_boundary
from ..models.ca_gat_mappo_trainer import CAGATMAPPOTrainer
from .runner import FormalEvaluationRunner

ROOT = Path(__file__).resolve().parents[2]
OLD_AUDIT = ROOT / 'metadata/audits/c32k_progress/20261014_16k32k'
TRAIN_ROOT = ROOT / ('logs/C/seed42/40k_continuation/'
    'ca_gat_mappo__remote_stress_v2__seed-42__cfg-708b8aaa4d67__git-eb1140cefc4a')
OLD_ROOT = ROOT / ('logs/C/seed42/32k_continuation/'
    'ca_gat_mappo__remote_stress_v2__seed-42__cfg-02c7757520b8__git-eb1140cefc4a')
SEEDS = (1042, 1043, 1044, 1045, 1046)
ARRIVALS = dict(zip(SEEDS, (135, 122, 117, 102, 129)))
PROTOCOL_SHA = 'd2463e1929ea867212edec4774f54b2c5a6dbc9ef957f00dd2978402aec4849a'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def summarize(records):
    """Pool denominators and energy; never average per-scene completion ratios."""
    metrics = [r['metrics'] for r in records]
    generated = sum(m['generated'] for m in metrics)
    completed = sum(m['completed'] for m in metrics)
    energy = sum(m['total_energy_j'] for m in metrics)
    return {'generated': generated, 'completed': completed,
            'expired': sum(m['expired'] for m in metrics),
            'truncated': sum(m['truncated'] for m in metrics),
            'completion_pct': 100 * completed / generated,
            'total_energy_j': energy,
            'CPU_energy_j': sum(m['cpu_energy_j'] for m in metrics),
            'TX_energy_j': sum(m['tx_energy_j'] for m in metrics),
            'energy_per_DONE_j': energy / completed if completed else None,
            'mean_episode_reward': statistics.mean(m['total_reward'] for m in metrics)}


def compare_baselines(records, previous):
    lookup = {(r['method_id'], r['evaluation_seed']): r for r in previous}
    baseline_count = 0
    for record in records:
        old = lookup[(record['method_id'], record['evaluation_seed'])]
        assert record['external_trace_sha256'] == old['external_trace_sha256']
        assert record['metrics']['generated'] == ARRIVALS[record['evaluation_seed']]
        if record['method_id'] != 'ca_gat_mappo':
            for field in ('metrics', 'action_trace_sha256', 'external_trace_sha256',
                          'hidden_trace_sha256', 'invariants', 'episode_horizon'):
                assert record[field] == old[field], (record['method_id'], record['evaluation_seed'], field)
            baseline_count += 1
    return baseline_count


class ProgressRunner(FormalEvaluationRunner):
    def _run_episode(self, method_id, evaluation_seed, **kwargs):
        result = super()._run_episode(method_id, evaluation_seed, **kwargs)
        print(f'Evaluation seed={evaluation_seed}, method={method_id}: '
              f'DONE={result.metrics["completed"]}/{result.metrics["generated"]}, '
              f'energy={result.metrics["total_energy_j"]:.6f} J', flush=True)
        return result


def audit_training(audit):
    old = load_checkpoint_payload(OLD_ROOT / 'checkpoints/final.pt')
    boundary = load_checkpoint_payload(TRAIN_ROOT / 'checkpoints/step_32000.pt')
    final = load_checkpoint_payload(TRAIN_ROOT / 'checkpoints/final.pt')
    periodic = load_checkpoint_payload(TRAIN_ROOT / 'checkpoints/step_36000.pt')
    verify_replay_boundary(boundary, old)
    lineage = final['config_snapshot']['_metadata']['continuation']
    assert lineage['replay_reference']['status'] == 'verified_exact'
    assert lineage['replay_reference']['sha256'] == digest(OLD_ROOT / 'checkpoints/final.pt')
    state = final['training_state']
    assert (state['collected_environment_transitions'], state['optimized_transitions'],
            state['ppo_update_count'], state['completed_episodes'],
            state['unused_final_tail_transitions']) == (40000, 39936, 156, 80, 64)
    assert periodic['active_rollout_state']['rollout_length'] == 160
    for name, count in (('episode_history', 64), ('ppo_update_history', 125)):
        assert_state_equal(final['diagnostics_state'][name][:count],
                           old['diagnostics_state'][name], name + '_original32k_prefix')
    aggregate = read_json(TRAIN_ROOT / 'aggregate_metrics.json')
    assert all(aggregate['checks'].values())
    write_json(audit / 'training_integrity.json', {
        'status': 'PASS', 'original_32k_replay_boundary_exact': True,
        'original_32k_episode_and_ppo_prefix_exact': True,
        'final_state': state, 'continuation': lineage,
        'aggregate_checks': aggregate['checks'], 'training_started_by_this_audit': False})
    episodes = final['diagnostics_state']['episode_history']
    windows = []
    for start in range(0, len(episodes), 8):
        batch = episodes[start:start + 8]
        rewards = [e['reward']['reward']['total'] for e in batch]
        windows.append({'end_environment_steps': (start + len(batch)) * 500,
                        'episodes_1_based': [start + 1, start + len(batch)],
                        'mean_episode_reward': statistics.mean(rewards),
                        'sample_std_episode_reward': statistics.stdev(rewards),
                        'mean_completed_tasks': statistics.mean(e['reward']['completed_task_count']['total'] for e in batch),
                        'mean_energy_j': statistics.mean(e['reward']['actual_energy_j']['total'] for e in batch)})
    write_json(audit / 'training_reward_windows.json', {
        'window_episodes': 8, 'smoothing': 'nonoverlapping8episode arithmetic mean',
        'source': str(TRAIN_ROOT / 'checkpoints/final.pt'), 'windows': windows})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='20261014_32k40k')
    args = parser.parse_args(argv)
    if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.tag):
        parser.error('tag must contain only letters, digits, underscores and hyphens')
    audit = ROOT / 'metadata/audits/c40k_progress' / args.tag
    if audit.exists():
        raise FileExistsError('Use a new tag; existing evaluation artifacts are immutable')
    audit.mkdir(parents=True)
    original_inputs = read_json(ROOT / 'metadata/audits/continuation_launch/20261014_c40k_ready/inputs_before.json')
    actual = {rel: digest(ROOT / rel) for rel in original_inputs}
    assert actual == original_inputs, 'Original scientific inputs changed'
    for base in (TRAIN_ROOT, ROOT / 'dashboard_logs/C/seed42/40k_continuation',
                 ROOT / 'plots/C/seed42/40k_continuation'):
        for path in sorted(base.rglob('*')):
            if path.is_file():
                actual[path.relative_to(ROOT).as_posix()] = digest(path)
    write_json(audit / 'inputs_before.json', actual)
    audit_training(audit)
    print('40k integrity PASS: exact32k replay, 80 episodes, 156 PPO updates, 64 unused tail.', flush=True)
    old_run = read_json(OLD_AUDIT / 'C_32k_run.json')
    old_episode_path = next(p for p in old_run['evaluation_artifacts'] if p.endswith('episode_metrics.jsonl'))
    previous = [json.loads(line) for line in Path(old_episode_path).read_text('utf-8').splitlines()]
    write_json(audit / 'evaluation_protocol.json', {
        'training_seed': 42, 'evaluation_seeds': list(SEEDS), 'new_checkpoints': [36000, 40000],
        'expected_arrivals': ARRIVALS, 'device': 'cpu', 'dtype': 'torch.float32',
        'inference': 'masked_argmax', 'update_network': False,
        'expected_evaluation_config_sha256': PROTOCOL_SHA,
        'held_out_seeds_used': [], 'purpose': 'single-seed development stability diagnosis'})
    torch.set_num_threads(1)
    receipts = []
    for step in (36000, 40000):
        filename = 'final.pt' if step == 40000 else 'step_36000.pt'
        checkpoint = TRAIN_ROOT / 'checkpoints' / filename
        overrides = {'mode': 'evaluation', 'method_id': 'ca_gat_mappo',
                     'output.logs_dir': str(ROOT / 'logs/C/seed42/checkpoint_progress' / args.tag / f'{step//1000}k')}
        config = load_run_config(ROOT / 'experiments/task-allocation/configs/c-seed42-40k-resume.json', cli_overrides=overrides)
        write_json(audit / f'C_{step//1000}k_evaluation_config.json', config.snapshot_dict())
        runner = ProgressRunner(config, checkpoint, evaluation_device='cpu',
                                collect_route_telemetry=True, collect_service_telemetry=True,
                                development_checkpoint=True)
        assert runner.evaluation_config_sha256 == PROTOCOL_SHA
        print(f'Starting frozen C{step//1000}k evaluation: 5 development scenes, four policies.', flush=True)
        started = time.monotonic()
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Evaluation attempted optimizer update')) as adam, \
             patch.object(CAGATMAPPOTrainer, '_run', side_effect=AssertionError('Evaluation attempted training')) as training:
            result = runner.run()
        assert adam.call_count == training.call_count == 0
        records = [e.artifact_record(result.evaluation_run_id) for e in result.episodes]
        matched = compare_baselines(records, previous)
        assert matched == 15
        receipt = {'method': 'C', 'checkpoint_steps': step, 'evaluation_run_id': result.evaluation_run_id,
                   'source_checkpoint_path': str(checkpoint), 'source_checkpoint_sha256': digest(checkpoint),
                   'evaluation_config_sha256': runner.evaluation_config_sha256,
                   'duration_seconds': time.monotonic() - started,
                   'new_episode_count': len(records), 'baseline_episodes_exactly_matched': matched,
                   'external_traces_matched': len(records), 'network_updates': adam.call_count,
                   'evaluation_artifacts': list(result.artifacts),
                   'route_artifacts': list(result.diagnostic_artifacts),
                   'service_artifacts': list(result.service_diagnostic_artifacts)}
        write_json(audit / f'C_{step//1000}k_run.json', receipt)
        receipts.append(receipt)
        learned = [r for r in records if r['method_id'] == 'ca_gat_mappo']
        write_json(audit / f'C_{step//1000}k_metrics.json', {'pooled': summarize(learned), 'per_scene': learned})
        print(f'C{step//1000}k complete: {json.dumps(summarize(learned))}; all15 baselines match old32k.', flush=True)
    write_json(audit / 'evaluation_execution.json', receipts)
    after = {rel: digest(ROOT / rel) for rel in actual}
    assert actual == after, 'Training/source inputs changed during evaluation'
    write_json(audit / 'inputs_after.json', after)
    write_json(audit / 'validation.json', {'status': 'PASS', 'source_inputs_checked': len(after),
               'changed_source_inputs': 0, 'network_updates': 0, 'training_started': False,
               'new_evaluation_episodes': 40, 'baseline_episodes_exactly_matched': 30,
               'external_traces_matched': 40, 'protocol_sha256': PROTOCOL_SHA})
    print(f'Evaluation finished; audit={audit}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
