"""Explicit, offline-only E16 dataset builder using the unchanged E15 oracle.

This entry point can be very expensive. It is never called by the RL trainer.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import torch

from ..config import load_run_config
from ..models.ca_gat_mappo import CAGATMAPPOActor, ActorObservationTensorizer
from ..models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution, SequentialActionMaskBatch
from ..models.ca_gat_mappo_checkpoint import load_checkpoint_payload
from ..models.destination_pretraining import (SCHEMA, FIXED_SEEDS, file_hash, stable_hash,
    feature_spec, sample_class, split_for, validate_dataset_settings, validate_records,
    dataset_hash, data_gate, initialize_from_base, validate_p1a_configuration)
from ..models.destination_ranking import PROTOCOL, masked_log_probs
from .destination_system_oracle import collect_destination_labels
from .route_oracle import FrozenOraclePolicyRunnerCache, actor_state_digest, tensor_digest, OracleReliabilityError


def build_dataset(settings):
    validate_dataset_settings(settings)
    root = Path(settings['output_dir'])
    root.mkdir(parents=True, exist_ok=False)
    base_path = settings['base_checkpoint']
    before = file_hash(base_path)
    if before != settings['base_checkpoint_sha256']:
        raise ValueError('expected mature P1a SHA mismatch')
    base_config = load_run_config(settings['base_config'])
    config = load_run_config(settings['scorer_config'])
    payload = load_checkpoint_payload(base_path)
    from .actor_loader import _validate_source_config_identity, _validate_source_compatibility
    identity = _validate_source_config_identity(payload)
    _validate_source_compatibility(base_config, identity.canonical_snapshot)
    validate_p1a_configuration(config, payload)
    if payload['training_state']['collected_environment_transitions'] != 8000:
        raise ValueError('P1a-8k mature policy is required')
    actor = CAGATMAPPOActor(base_config).eval().requires_grad_(False)
    actor.load_state_dict(payload['model_state']['actor'], strict=True)
    feature_actor, _ = initialize_from_base(config, payload)
    feature_actor.eval().requires_grad_(False)
    factual_digest = actor_state_digest(actor)
    distribution = CAGATMAPPOActionDistribution(actor, base_config)
    adapter = ActorObservationTensorizer(base_config)
    runner = FrozenOraclePolicyRunnerCache(base_config).get(actor,
        payload['training_state']['policy_version'], checkpoint_sha256=before,
        source='E16_mature_P1a_offline', config_hash=base_config.config_hash,
        git_commit=base_config.git_commit, run_id=base_config.run_id)
    spec = feature_spec(config)
    provenance = {'base_checkpoint_sha256': before, 'base_config_hash': base_config.config_hash,
                  'feature_spec': spec, 'feature_spec_hash': stable_hash(spec),
                  'protocol': PROTOCOL, 'fixed_seeds': list(FIXED_SEEDS),
                  'settings_hash': stable_hash(settings), 'factual_policy': 'frozen_P1a_masked_argmax',
                  'code_git_commit': config.git_commit}
    records, used_seeds, informative = [], [], 0
    shadow_steps, shadow_seconds = 0, 0.
    from ..env.environment import U2UMECEnvironment
    for episode, seed in enumerate(FIXED_SEEDS):
        used_seeds.append(seed)
        # Fixed dev seed is the actual environment seed, not evaluation_seed1044.
        env_config = copy.copy(base_config); object.__setattr__(env_config, 'seed', seed)
        env = U2UMECEnvironment(env_config); env.reset()
        hidden = actor.initial_hidden(1)
        while True:
            observations = env.current_observations
            batch = adapter.encode_step(observations, episode_start=env.slot == 0)
            masks = SequentialActionMaskBatch.from_observations(observations)
            with torch.inference_mode():
                policy = distribution.deterministic_actions(batch, masks, hidden)
                inputs = feature_actor(batch, hidden).destination_features
            proposals = tuple(policy.proposals[0][0])
            selected = [u for u, p in enumerate(proposals)
                        if isinstance(p.route, int) and not isinstance(p.route, bool)]
            pending = []
            if selected:
                state_before, rng_before = env.state_fingerprint(), env.rng_fingerprint()
                hidden_before, runner_before = tensor_digest(policy.hidden_out), actor_state_digest(runner.actor)
                try:
                    labels = collect_destination_labels(env, observations, proposals, policy.hidden_out,
                        runner, episode_id=episode, environment_step=episode * 500 + env.slot)
                    labels = {r['source_uav']: r for r in labels}
                    shadow_steps += sum(r['shadow_transitions'] for r in labels.values())
                    shadow_seconds += sum(r['shadow_wall_seconds'] for r in labels.values())
                    error = None
                except (ValueError, OracleReliabilityError) as exc:
                    labels, error = {}, f'{type(exc).__name__}: {exc}'
                if (env.state_fingerprint() != state_before or env.rng_fingerprint() != rng_before
                        or tensor_digest(policy.hidden_out) != hidden_before
                        or actor_state_digest(runner.actor) != runner_before):
                    raise OracleReliabilityError('failed oracle polluted factual environment')
                for source in selected:
                    ids = feature_actor.route_decoder.remote_candidate_ids[source].tolist()
                    mask = policy.action_masks['route'][0, 0, source, 3:].cpu().tolist()
                    legal = [d for d, m in zip(ids, mask) if m]
                    label = labels.get(source, {})
                    best = label.get('best_set', [])
                    scores = policy.raw_logits['route'][0, 0, source, 3:]
                    probs = masked_log_probs(scores, torch.tensor(mask)).exp().tolist()
                    pending.append({'seed': seed, 'episode': episode, 'slot': env.slot,
                        'split': split_for(seed), 'task_id': int(observations[source].private_queues.unbound.head_task_id),
                        'source_uav': source, 'candidate_ids': ids, 'candidate_mask': mask,
                        'legal_destination_ids': legal, 'scorer_input': inputs[0, 0, source].cpu().tolist(),
                        'factual_p1a_destination': proposals[source].route,
                        'p1a_destination_logits': scores.cpu().tolist(), 'p1a_destination_probabilities': probs,
                        'oracle_best_set': best, 'target': [1 / len(best) if d in best else 0 for d in ids],
                        'sample_class': sample_class(legal, best, error), 'error': error,
                        'branches': label.get('branches', []), 'oracle_provenance': label.get('provenance'),
                        'feature_spec_hash': provenance['feature_spec_hash'], 'protocol': PROTOCOL,
                        'base_checkpoint_sha256': before})
            result = env.step(proposals)
            for row in pending:
                task = env.lifecycle.tasks[row['task_id']]
                if task.destination != row['factual_p1a_destination'] or task.binding_slot != row['slot']:
                    raise OracleReliabilityError('sample is not a factual first Remote binding')
                records.append(row)
                informative += row['sample_class'] == 'informative_multi'
                with (root / 'records.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(row, allow_nan=False) + '\n')
            hidden = policy.hidden_out.detach()
            print(f'seed={seed} slot={env.slot} states={len(records)} informative={informative}', flush=True)
            reached = informative >= settings['target_informative_states'] and data_gate(records)['status'] == 'PASS'
            if reached or result.terminated or result.truncated:
                break
        if informative >= settings['target_informative_states'] and data_gate(records)['status'] == 'PASS':
            break
    if file_hash(base_path) != before or actor_state_digest(actor) != factual_digest:
        raise OracleReliabilityError('P1a base was modified')
    validate_records(records, provenance)
    dataset = {'schema': SCHEMA, 'records': records, 'provenance': provenance,
               'dataset_hash': dataset_hash(records, provenance), 'data_gate': data_gate(records),
               'collection_cost': {'destination_shadow_transitions': shadow_steps,
                                   'destination_shadow_wall_seconds': shadow_seconds},
               'used_seeds': used_seeds, 'sample_counts': dict(Counter(r['sample_class'] for r in records))}
    (root / 'dataset.json').write_text(json.dumps(dataset, allow_nan=False), encoding='utf-8')
    print(json.dumps({'OFFLINE ORACLE DATA GATE': dataset['data_gate'], 'sample_counts': dataset['sample_counts']}))
    return dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    result = build_dataset(json.loads(Path(args.settings).read_text(encoding='utf-8')))
    return 0 if result['data_gate']['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
