"""E16 offline contracts, filtering, provenance and strict warm-start loading.

No environment collection or optimizer is constructed by this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .destination_ranking import PROTOCOL, SystemAwareDestinationScorer

SCHEMA = "e16_destination_dataset.v1"
WARMSTART = "E16_PRETRAINED_WARMSTART"
SCORER_PREFIX = "action_heads.route.system_destination_scorer."
FIXED_SEEDS = (2041, 2042, 2043, 2044, 2045)


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def online_destination_enabled(config):
    """Legacy coefficient-positive E15 configs preserve their historical behavior."""
    m = config.training.mappo
    return m.destination_supervision_mode != 'offline_pretrain' and m.destination_oracle_rank_coefficient > 0


def schedule_environment_steps(new_transitions, provenance=None):
    """Keep the original absolute exploration schedule after mature warm-start."""
    return new_transitions + (provenance or {}).get('base_training_transitions', 0)


def feature_spec(config):
    from .ca_gat_mappo import MAPPOTensorSpec
    with torch.random.fork_rng(devices=[]):
        scorer = SystemAwareDestinationScorer(config, MAPPOTensorSpec.from_config(config))
    # Include the implementation, all physical scaling constants and dimensions.
    return {'version': 'e15-five-block.v1', 'implementation_sha256':
            file_hash(Path(__file__).with_name('destination_ranking.py')),
            'dimensions': list(scorer.input_dimensions),
            'scales': {k: v for k, v in vars(scorer).items()
                       if isinstance(v, (int, float))}, 'protocol': PROTOCOL}


def sample_class(legal, best, error=None):
    if error is not None or not legal or not best or not set(best) <= set(legal):
        return 'branch_invalid'
    if len(legal) == 1:
        return 'single_candidate'
    return 'all_tie_multi' if set(legal) == set(best) else 'informative_multi'


def split_for(seed):
    if seed not in FIXED_SEEDS:
        raise ValueError('only precommitted development seeds 2041..2045; eval1044 forbidden')
    return 'train' if seed <= 2043 else ('validation' if seed == 2044 else 'reserve')


def validate_dataset_settings(settings):
    if settings['seeds'] != list(FIXED_SEEDS) or settings['max_dataset_seeds'] != 5:
        raise ValueError('fixed seed order / maximum cannot be changed')
    if settings['target_informative_states'] != 150 or settings['minimum_informative_states'] != 100:
        raise ValueError('precommitted informative-state gates cannot be lowered')


def dataset_hash(records, provenance):
    return stable_hash({'schema': SCHEMA, 'records': records, 'provenance': provenance})


def validate_records(records, provenance):
    seen = set()
    for r in records:
        if r['split'] != split_for(r['seed']):
            raise ValueError('seed-group leakage')
        key = (r['seed'], r['episode'], r['task_id'])
        if key in seen:
            raise ValueError('duplicate first binding')
        seen.add(key)
        if r['base_checkpoint_sha256'] != provenance['base_checkpoint_sha256']:
            raise ValueError('mixed base checkpoints')
        if r['feature_spec_hash'] != provenance['feature_spec_hash'] or r['protocol'] != PROTOCOL:
            raise ValueError('feature/protocol mismatch')
        kind = sample_class(r['legal_destination_ids'], r['oracle_best_set'], r.get('error'))
        if kind != r['sample_class']:
            raise ValueError('incorrect informative filtering')
        if kind == 'branch_invalid':
            continue
        features = torch.tensor(r['scorer_input'], dtype=torch.float32)
        mask = torch.tensor(r['candidate_mask'], dtype=torch.bool)
        target = torch.tensor(r['target'], dtype=torch.float32)
        ids = r['candidate_ids']
        if (features.ndim != 2 or features.shape[0] != len(ids)
                or mask.shape != target.shape or len(mask) != len(ids)
                or not torch.isfinite(features).all()
                or [d for d, m in zip(ids, mask) if m] != r['legal_destination_ids']):
            raise ValueError('invalid candidate tensor/mask')
        expected = torch.tensor([1 / len(r['oracle_best_set']) if d in r['oracle_best_set'] else 0
                                 for d in ids], dtype=torch.float32)
        if not torch.equal(target, expected):
            raise ValueError('oracle-set target must be uniform and legal')


def load_dataset(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data['schema'] != SCHEMA or data['dataset_hash'] != dataset_hash(data['records'], data['provenance']):
        raise ValueError('dataset schema/hash mismatch')
    validate_records(data['records'], data['provenance'])
    if data.get('data_gate') != data_gate(data['records']):
        raise ValueError('dataset data gate must agree with actual coverage')
    return data


def data_gate(records):
    counts = {s: sum(r['sample_class'] == 'informative_multi' and r['split'] == s for r in records)
              for s in ('train', 'validation', 'reserve')}
    return {'status': 'PASS' if sum(counts.values()) >= 100 and counts['train'] >= 50
            and counts['validation'] >= 10 and counts['reserve'] >= 10 else 'FAIL',
            'informative_by_split': counts,
            'requires': '>=100 informative total; train>=50, validation>=10, reserve>=10'}


def tensor_state_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        x = v.detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode()); h.update(x.numpy().tobytes())
    return h.hexdigest()


def initialize_from_base(config, base_payload):
    from .ca_gat_mappo import CAGATMAPPOActor, MAPPOCentralizedCritic
    if base_payload['checkpoint_kind'] != 'FINAL_COMPLETED':
        raise ValueError('mature P1a final checkpoint required')
    actor = CAGATMAPPOActor(config)
    base = base_payload['model_state']['actor']
    missing, unexpected = actor.load_state_dict(base, strict=False)
    if unexpected or not missing or any(not k.startswith(SCORER_PREFIX) for k in missing):
        raise ValueError('base must be legacy P1a; only new scorer keys may be missing')
    for k, v in base.items():
        if not torch.equal(v, actor.state_dict()[k]):
            raise ValueError('base common parameter changed')
    critic = MAPPOCentralizedCritic(config)
    critic.load_state_dict(base_payload['model_state']['critic'], strict=True)
    return actor, critic


def validate_p1a_configuration(config, base_payload):
    from ..config import canonicalize_config_identity
    wanted = canonicalize_config_identity(config.snapshot_dict())
    source = canonicalize_config_identity(base_payload['config_snapshot'])
    # Only destination mode, output and factual budget differ from P1a.
    for obj in (wanted, source):
        obj.pop('_metadata', None); obj.pop('output', None)
        obj['mode'] = 'rl'; obj['launch_profile'] = 'rl-formal'
        obj['evaluation']['evaluation_seeds'] = []  # evaluator overrides only
        m = obj['training']['mappo']
        for k in ('destination_decoder_mode', 'destination_supervision_mode',
                  'destination_oracle_rank_coefficient', 'max_training_episodes',
                  'max_training_environment_steps', 'evaluation_interval_steps',
                  'checkpoint_interval_steps', 'training_device'):
            m.pop(k, None)
    if wanted != source:
        raise ValueError('E16 must preserve P1a mechanisms/hyperparameters')


def pretrain_gate(initial, final, baseline):
    # Precommitted definition of "clearly below": >=10% relative CE reduction.
    return {'status': 'PASS' if final['oracle_set_accuracy'] >= baseline + .15 - 1e-12
            and final['ce'] <= .9 * initial['ce'] else 'FAIL',
            'minimum_accuracy_gain_pp': 15, 'maximum_ce_ratio': .9,
            'initial_validation': initial, 'final_validation': final, 'p1a_validation_accuracy': baseline}


def save_warmstart(path, config, base_path, scorer_state, data, gate):
    from .ca_gat_mappo_checkpoint import load_checkpoint_payload
    if gate['status'] != 'PASS' or data['data_gate']['status'] != 'PASS':
        raise ValueError('both data and pretrain gates must pass')
    if pretrain_gate(gate['initial_validation'], gate['final_validation'],
                     gate['p1a_validation_accuracy'])['status'] != 'PASS':
        raise ValueError('pretrain gate numerical thresholds not met')
    validate_records(data['records'], data['provenance'])
    if (data_gate(data['records'])['status'] != 'PASS'
            or data['dataset_hash'] != dataset_hash(data['records'], data['provenance'])):
        raise ValueError('data gate/hash must be recomputed, not asserted')
    if file_hash(base_path) != data['provenance']['base_checkpoint_sha256']:
        raise ValueError('base checkpoint changed')
    base = load_checkpoint_payload(base_path)
    validate_p1a_configuration(config, base)
    actor, critic = initialize_from_base(config, base)
    actor.route_decoder.system_destination_scorer.load_state_dict(scorer_state, strict=True)
    provenance = {'base_checkpoint_path': str(Path(base_path).resolve()),
                  'base_checkpoint_sha256': file_hash(base_path),
                  'dataset_hash': data['dataset_hash'],
                  'feature_spec_hash': data['provenance']['feature_spec_hash'],
                  'scorer_pretrain_hash': tensor_state_hash(scorer_state),
                  'config_hash': config.config_hash, 'factual_transitions_since_warmstart': 0,
                  'base_training_transitions': base['training_state']['collected_environment_transitions'],
                  'pretrain_gate': gate, 'data_gate': data['data_gate']}
    payload = {'checkpoint_kind': WARMSTART, 'schema': 'e16-warmstart.v1',
               'method_id': 'ca_gat_mappo', 'git_commit': config.git_commit,
               'config_hash': config.config_hash, 'config_snapshot': config.snapshot_dict(),
               'runtime_provenance': {'device_type': 'cpu', 'dtype': 'torch.float32'},
               'model_state': {'actor': actor.state_dict(), 'critic': critic.state_dict()},
               'e16_provenance': provenance}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: never overwrite P1a or an earlier pretraining result.
    with path.open('xb') as f:
        torch.save(payload, f)
    return payload


def load_warmstart(path, config):
    from .ca_gat_mappo_checkpoint import load_checkpoint_payload
    from ..config import canonical_config_identity_hash
    p = torch.load(path, map_location='cpu', weights_only=True)
    expected = {'checkpoint_kind', 'schema', 'method_id', 'git_commit', 'config_hash',
                'config_snapshot', 'runtime_provenance', 'model_state', 'e16_provenance'}
    if set(p) != expected or p['schema'] != 'e16-warmstart.v1' or p['checkpoint_kind'] != WARMSTART:
        raise ValueError('not an E16 warm-start; never resume optimizer state')
    if p['method_id'] != 'ca_gat_mappo' or p['git_commit'] != p['config_snapshot']['_metadata']['git_commit']:
        raise ValueError('warm-start method/code identity mismatch')
    meta = p['e16_provenance']
    if (meta['pretrain_gate']['status'] != 'PASS' or meta['data_gate']['status'] != 'PASS'
            or meta['factual_transitions_since_warmstart'] != 0
            or p['config_hash'] != canonical_config_identity_hash(p['config_snapshot'])
            or meta['config_hash'] != p['config_hash']):
        raise ValueError('warm-start gate/config provenance invalid')
    gate = meta['pretrain_gate']
    for k in ('base_checkpoint_sha256', 'dataset_hash', 'scorer_pretrain_hash', 'feature_spec_hash'):
        if not isinstance(meta.get(k), str) or len(meta[k]) != 64:
            raise ValueError('invalid warm-start provenance digest')
        int(meta[k], 16)
    if pretrain_gate(gate['initial_validation'], gate['final_validation'],
                     gate['p1a_validation_accuracy'])['status'] != 'PASS':
        raise ValueError('pretrain gate thresholds not satisfied')
    if (config.training.mappo.destination_supervision_mode != 'offline_pretrain'
            or meta['feature_spec_hash'] != stable_hash(feature_spec(config))):
        raise ValueError('warm-start destination mode/feature spec mismatch')
    base_path = meta['base_checkpoint_path']
    if file_hash(base_path) != meta['base_checkpoint_sha256']:
        raise ValueError('read-only P1a base SHA mismatch')
    base = load_checkpoint_payload(base_path)
    validate_p1a_configuration(config, base)
    actor, critic = initialize_from_base(config, base)
    if meta.get('base_training_transitions') != base['training_state']['collected_environment_transitions']:
        raise ValueError('base exploration schedule provenance mismatch')
    for state in p['model_state'].values():
        if any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('warm-start state contains non-finite tensors')
    actor.load_state_dict(p['model_state']['actor'], strict=True)
    critic.load_state_dict(p['model_state']['critic'], strict=True)
    for name in ('actor', 'critic'):
        common = {k: p['model_state'][name][k] for k in base['model_state'][name]}
        if tensor_state_hash(common) != tensor_state_hash(base['model_state'][name]):
            raise ValueError('warm-start common P1a state is not byte-identical')
        for k, v in base['model_state'][name].items():
            if not torch.equal(v, p['model_state'][name][k]):
                raise ValueError('warm-start changed common P1a weights')
    if tensor_state_hash(actor.route_decoder.system_destination_scorer.state_dict()) != meta['scorer_pretrain_hash']:
        raise ValueError('pretrained scorer digest mismatch')
    return p
