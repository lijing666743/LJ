"""Frozen C seed43 checkpoint campaign; never starts or updates training."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch

from ..config import load_run_config
from ..env.randomness import derive_training_episode_seed
from ..models.ca_gat_mappo_checkpoint import load_checkpoint_payload
from ..models.ca_gat_mappo_trainer import CAGATMAPPOTrainer
from .c40k_progress import ROOT, OLD_AUDIT, SEEDS, ARRIVALS, PROTOCOL_SHA, read_json, write_json, digest, compare_baselines
from .group_meeting_figures import summarize_episodes
from .runner import FormalEvaluationRunner

STEPS = tuple(range(4000, 40001, 4000))
CONFIG = ROOT / 'experiments/task-allocation/configs/c-seed43-40k.json'
TRAIN_ROOT = ROOT / ('logs/C/seed43/40k/'
    'ca_gat_mappo__remote_stress_v2__seed-43__cfg-ec33289ec8c8__git-eb1140cefc4a')


def audit_directory(tag):
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in tag):
        raise ValueError('Use a simple alphanumeric tag')
    return ROOT / 'metadata/audits/c_seed43_progress' / tag


def prepare(tag):
    audit = audit_directory(tag)
    if audit.exists():
        raise FileExistsError('Use --resume for completed points, or a new tag')
    audit.mkdir(parents=True)
    config = load_run_config(CONFIG)
    assert config.seed == 43 and config.training.mappo.max_environment_transitions == 40000
    expected = read_json(ROOT / 'metadata/audits/training_launch/20261014_c_seed43_40k_ready/inputs_before.json')
    frozen = {rel: digest(ROOT / rel) for rel in expected}
    assert frozen == expected, 'Original experiment inputs changed'
    for base in (TRAIN_ROOT, ROOT / 'dashboard_logs/C/seed43/40k', ROOT / 'plots/C/seed43/40k'):
        for path in sorted(base.rglob('*')):
            if path.is_file():
                frozen[path.relative_to(ROOT).as_posix()] = digest(path)
    frozen[CONFIG.relative_to(ROOT).as_posix()] = digest(CONFIG)
    write_json(audit / 'inputs_before.json', frozen)
    inventory = []
    final = None
    for step in STEPS:
        path = TRAIN_ROOT / 'checkpoints' / ('final.pt' if step == 40000 else f'step_{step}.pt')
        payload = load_checkpoint_payload(path)
        state = payload['training_state']
        assert payload['seed'] == 43 if 'seed' in payload else payload['config_snapshot']['seed'] == 43
        assert payload['config_hash'] == config.config_hash
        assert payload['config_snapshot']['_metadata'].get('continuation') is None
        expected_kind = 'FINAL_COMPLETED' if step == 40000 else 'PERIODIC_RESUME'
        assert payload['checkpoint_kind'] == expected_kind
        assert (state['collected_environment_transitions'], state['optimized_transitions'],
                state['ppo_update_count'], state['policy_version'], state['completed_episodes'],
                state['active_rollout_length'], state['unused_final_tail_transitions']) == (
                    step, step//256*256, step//256, step//256, step//500,
                    0 if step == 40000 else step%256, step%256 if step == 40000 else 0)
        diagnostics = payload['diagnostics_state']
        assert len(diagnostics['episode_history']) == step//500
        assert len(diagnostics['ppo_update_history']) == step//256
        for index, episode in enumerate(diagnostics['episode_history']):
            assert episode['episode_index'] == index and episode['completed_boundary']
            assert episode['environment_seed'] == derive_training_episode_seed(43, index)
        inventory.append({'step': step, 'path': str(path), 'sha256': digest(path),
                          'checkpoint_kind': payload['checkpoint_kind'], 'training_state': state})
        if step == 40000:
            final = payload
    assert final is not None
    aggregate = read_json(TRAIN_ROOT / 'aggregate_metrics.json')
    assert all(aggregate['checks'].values())
    windows = []
    history = final['diagnostics_state']['episode_history']
    for start in range(0, 80, 8):
        batch = history[start:start+8]
        rewards = [e['reward']['reward']['total'] for e in batch]
        windows.append({'end_environment_steps': (start+8)*500,
                        'episodes_1_based': [start+1, start+8],
                        'mean_episode_reward': statistics.mean(rewards),
                        'sample_std_episode_reward': statistics.stdev(rewards),
                        'mean_completed_tasks': statistics.mean(e['reward']['completed_task_count']['total'] for e in batch),
                        'mean_energy_j': statistics.mean(e['reward']['actual_energy_j']['total'] for e in batch)})
    write_json(audit / 'training_reward_windows.json', {'training_seed': 43, 'window_episodes': 8,
               'smoothing': 'nonoverlapping8episode arithmetic mean', 'windows': windows})
    write_json(audit / 'checkpoint_inventory.json', inventory)
    write_json(audit / 'training_integrity.json', {'status': 'PASS', 'final_state': final['training_state'],
               'fresh_training_lineage': True, 'all10_checkpoints_valid': True,
               'episode_seed_sequence_matches43': True, 'aggregate_checks': aggregate['checks'],
               'training_started_by_this_audit': False})
    write_json(audit / 'evaluation_protocol.json', {
        'training_seed': 43, 'checkpoint_steps': list(STEPS), 'primary_endpoint': 40000,
        'evaluation_seeds': list(SEEDS), 'expected_arrivals': ARRIVALS,
        'device': 'cpu', 'dtype': 'torch.float32', 'inference': 'masked_argmax',
        'update_network': False, 'expected_protocol_sha256': PROTOCOL_SHA,
        'held_out_seeds_used': [], 'expected_episodes': 200, 'expected_C_episodes': 50,
        'claim_boundary': 'Two independent training seeds; development scenes; not final held-out evidence.'})
    print('seed43 training integrity PASS: all10 checkpoints, 80 episodes, 156 updates; no continuation.', flush=True)


class ProgressRunner(FormalEvaluationRunner):
    checkpoint_step = None

    def _run_episode(self, method_id, evaluation_seed, **kwargs):
        result = super()._run_episode(method_id, evaluation_seed, **kwargs)
        print(f'C43 {self.checkpoint_step//1000}k seed={evaluation_seed} {method_id}: '
              f'DONE={result.metrics["completed"]}/{result.metrics["generated"]}, '
              f'energy={result.metrics["total_energy_j"]:.4f}J', flush=True)
        return result


def run_point(step, tag):
    torch.set_num_threads(1)
    audit = audit_directory(tag)
    point = next(row for row in read_json(audit / 'checkpoint_inventory.json') if row['step'] == step)
    assert digest(point['path']) == point['sha256']
    config = load_run_config(CONFIG, cli_overrides={
        'mode': 'evaluation', 'method_id': 'ca_gat_mappo',
        'output.logs_dir': str(ROOT / 'logs/C/seed43/checkpoint_progress' / tag / f'{step//1000}k')})
    write_json(audit / f'C43_{step//1000}k_evaluation_config.json', config.snapshot_dict())
    runner = ProgressRunner(config, point['path'], evaluation_device='cpu',
                            collect_route_telemetry=True, collect_service_telemetry=True,
                            development_checkpoint=True)
    runner.checkpoint_step = step
    assert runner.evaluation_config_sha256 == PROTOCOL_SHA
    started = time.monotonic()
    with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Frozen evaluation attempted update')) as adam, \
         patch.object(CAGATMAPPOTrainer, '_run', side_effect=AssertionError('Frozen evaluation attempted training')) as train:
        result = runner.run()
    assert adam.call_count == train.call_count == 0
    records = [e.artifact_record(result.evaluation_run_id) for e in result.episodes]
    old_run = read_json(OLD_AUDIT / 'C_32k_run.json')
    old_path = next(p for p in old_run['evaluation_artifacts'] if p.endswith('episode_metrics.jsonl'))
    previous = [json.loads(line) for line in Path(old_path).read_text('utf-8').splitlines()]
    assert compare_baselines(records, previous) == 15
    learned = [row for row in records if row['method_id'] == 'ca_gat_mappo']
    summary = summarize_episodes(learned)
    assert summary['generated'] == 605
    receipt = {'training_seed': 43, 'checkpoint_steps': step, 'evaluation_run_id': result.evaluation_run_id,
               'source_checkpoint_path': point['path'], 'source_checkpoint_sha256': point['sha256'],
               'evaluation_config_sha256': runner.evaluation_config_sha256,
               'duration_seconds': time.monotonic()-started, 'new_episode_count': len(records),
               'baseline_episodes_exactly_matched': 15, 'external_traces_matched': len(records),
               'network_updates': 0, 'training_calls': 0, 'pooled': summary,
               'evaluation_artifacts': list(result.artifacts), 'route_artifacts': list(result.diagnostic_artifacts),
               'service_artifacts': list(result.service_diagnostic_artifacts)}
    assert digest(point['path']) == point['sha256']
    write_json(audit / f'C43_{step//1000}k_run.json', receipt)
    print(f'C43 {step//1000}k COMPLETE: {summary["completed"]}/605 '
          f'({summary["completion_pct"]:.3f}%), {summary["total_energy_j"]:.4f}J; baselines exact.', flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='20261015_seed43_4k40k')
    parser.add_argument('--workers', type=int, choices=(1,2), default=1)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    audit = audit_directory(args.tag)
    if args.resume:
        frozen = read_json(audit / 'inputs_before.json')
        assert all(digest(ROOT/rel) == sha for rel,sha in frozen.items())
    else:
        prepare(args.tag)
    if args.prepare_only:
        return 0
    pending = []
    for step in STEPS:
        receipt_path = audit / f'C43_{step//1000}k_run.json'
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            assert receipt['checkpoint_steps'] == step and receipt['network_updates'] == 0
            assert all(Path(path).is_file() for path in receipt['evaluation_artifacts'])
        else:
            pending.append(step)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = {pool.submit(run_point, step, args.tag): step for step in pending}
        for job in as_completed(jobs):
            result = job.result()
            print(f'Checkpoint saved and verified: {result["checkpoint_steps"]}; '
                  f'finished={sum((audit/f"C43_{s//1000}k_run.json").exists() for s in STEPS)}/10', flush=True)
    receipts = [read_json(audit / f'C43_{s//1000}k_run.json') for s in STEPS]
    write_json(audit / 'evaluation_execution.json', receipts)
    before = read_json(audit / 'inputs_before.json')
    after = {rel: digest(ROOT/rel) for rel in before}
    assert before == after
    write_json(audit / 'inputs_after.json', after)
    write_json(audit / 'validation.json', {'status': 'PASS', 'network_updates': 0,
               'training_started': False, 'new_evaluation_episodes': 200,
               'baseline_episodes_exactly_matched': 150, 'external_traces_matched': 200,
               'checkpoint_count': 10, 'source_inputs_checked': len(before), 'source_inputs_changed': 0,
               'evaluation_protocol_sha256': PROTOCOL_SHA, 'workers': args.workers})
    print('ALL10 C43 CHECKPOINTS EVALUATED; sources unchanged; no optimizer updates.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
