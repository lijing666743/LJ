"""Offline Route-only checkpoint and weight-swap localization; no environment.

Gate I persisted decision snapshots omit intervening recurrent observations.
Every snapshot therefore uses a fresh zero GRU hidden state in this probe.
Unused agent positions are tensor padding, never counted as bank states.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from src.config import canonicalize_config_identity, load_run_config
from src.env.action_history import PreviousActionFeatures
from src.env.actions import ActionProposal
from src.env.observation import (
    ActorEdgeFeatures, IndexedQueueFeatures, NeighborPublicFeatures,
    PrivateQueueFeatures, QueueSummary, SelfResourceFeatures,
)
from src.models.ca_gat_mappo import ActorObservationTensorizer, CAGATMAPPOActor
from src.models.ca_gat_mappo_checkpoint import load_checkpoint_payload

STEPS = (4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000)
SHARED_PREFIXES = ('self_encoder.', 'public_encoder.', 'graph_encoder.', 'gru.')
ROUTE_PREFIX = 'action_heads.route.'
EXPECTED_CONFIGS = {
    'E3': '735fb97120ce5c733f1e558b18f0c0863e878bf27da29de67e420a1c97a31786',
    'E4': 'a216fc6b130686222b3bbd1bd19bada399d5be7043e007f0df37b8dd09e880a2',
}
EXPECTED_FINALS = {
    'E3': '3832e046d1be5064b78d0b27289d8e9379909bbb49e7143ec8d677776d630ff3',
    'E4': '0eb5cd7d8fdfd3d891af6eabb34900dac07a7b5c55dc23073ee5765f320ff5cb',
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class SnapshotDomains:
    """Preserve saved candidate ordering without rebuilding action masks."""

    def __init__(self, snapshot: dict[str, Any]):
        self.domains = snapshot['domains']

    def domain_for(self, branch: str) -> tuple[Any, ...]:
        return tuple(self.domains[branch])


def snapshot_view(snapshot: dict[str, Any]) -> SimpleNamespace:
    """Restore tensorizer-visible fields only, including exact previous action."""
    queues = snapshot['private_queues']
    private = PrivateQueueFeatures(
        QueueSummary(**queues['unbound']), QueueSummary(**queues['local_cpu']),
        IndexedQueueFeatures(**queues['tx_by_destination']),
        IndexedQueueFeatures(**queues['cpu_by_source']),
    )
    edge = dict(snapshot['edge_history'])
    edge['stale_csi'] = (np.asarray(edge.pop('stale_csi_real'))
                         + 1j * np.asarray(edge.pop('stale_csi_imag')))
    previous = snapshot['previous_action']
    comm, cpu = previous['executed_communication'], previous['executed_cpu']
    restored_previous = PreviousActionFeatures(
        uav_id=previous['uav_id'], source_slot=previous['source_slot'],
        valid=previous['valid'],
        proposal=ActionProposal(uav_id=previous['uav_id'], **previous['proposal']),
        communication_accepted=comm['accepted'],
        communication_receiver_uav=comm['receiver_uav'],
        executed_ru_count=comm['ru_count'], ru_utilization_ratio=comm['ru_utilization_ratio'],
        executed_power_ratio=comm['power_ratio'], cpu_source_uav=cpu['source_uav'],
        cpu_active=cpu['active'], executed_cpu_frequency_ratio=cpu['frequency_ratio'],
    )
    return SimpleNamespace(
        **{k: snapshot[k] for k in ('slot', 'uav_id', 'paper_uav_id',
                                  'arrival_rate_estimate', 'arrival_rate_valid_mask',
                                  'history_source_slot')},
        self_resources=SelfResourceFeatures(**snapshot['self_resources']),
        private_queues=private, neighbor_public=NeighborPublicFeatures(**snapshot['neighbor_public']),
        edge_history=ActorEdgeFeatures(**edge), previous_action=restored_previous,
        candidate_neighbor_mask=np.asarray(snapshot['candidate_neighbor_mask'], dtype=np.bool_),
        service_queue_available=np.asarray(snapshot['service_queue_available'], dtype=np.bool_),
        action_masks=SnapshotDomains(snapshot['action_masks']),
    )


def encode_bank(rows: list[dict[str, Any]], config: Any) -> dict[str, torch.Tensor]:
    tensorizer = ActorObservationTensorizer(config)
    views = [snapshot_view(row['observation']) for row in rows]
    for row, view in zip(rows, views):
        count = config.environment.uav_count
        expected = ('idle', 'local', 'defer', *(i for i in range(count) if i != view.uav_id))
        if view.action_masks.domain_for('route') != expected:
            raise ValueError('saved Route candidate ordering differs from actor architecture')
        if row['action_mask'] != row['observation']['action_masks']['masks']['route']:
            raise ValueError('bank Route mask differs from saved observation')
        if not any(row['action_mask'][3:]) or row['action_mask'][0]:
            raise ValueError('bank must contain active Remote-legal Route decisions')
    return {
        'self': torch.tensor(np.stack([tensorizer._self_vector(v) for v in views]), dtype=torch.float32),
        'public': torch.tensor(np.stack([tensorizer._public_matrix(v) for v in views]), dtype=torch.float32),
        'edge': torch.tensor(np.stack([tensorizer._edge_matrix(v) for v in views]), dtype=torch.float32),
        'neighbor': torch.tensor(np.stack([v.candidate_neighbor_mask for v in views]), dtype=torch.bool),
        'mask': torch.tensor([row['action_mask'] for row in rows], dtype=torch.bool),
        'agent': torch.tensor([row['uav_id'] for row in rows], dtype=torch.long),
    }


def route_forward(actor: CAGATMAPPOActor, bank: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Exact shared modules and Route decoder; TX/CPU/critic are never called."""
    count, agents = len(bank['agent']), actor.spec.uav_count
    spatial, local, candidates = actor.graph_encoder(
        actor.self_encoder(bank['self']), actor.public_encoder(bank['public']),
        bank['edge'], bank['neighbor'], return_candidate_values=True,
    )
    recurrent, _ = actor.gru(spatial.unsqueeze(1), torch.zeros(
        (1, count, actor.spec.gru_hidden_dimension), dtype=spatial.dtype, device=spatial.device))
    indices = torch.arange(count, device=spatial.device)
    ids = bank['agent']

    def pad(value: torch.Tensor) -> torch.Tensor:
        result = value.new_zeros((count, 1, agents, *value.shape[1:]))
        result[indices, 0, ids] = value
        return result

    masks = torch.zeros((count, 1, agents, actor.spec.action_dimensions['route']), dtype=torch.bool)
    masks[..., 0] = True
    masks[indices, 0, ids] = bank['mask']
    captured = {}

    def capture_top(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        # Actual hierarchical decoder order is Local, Defer, Remote.
        # Export Local, Remote, Defer consistently with probability columns.
        captured['top'] = output[indices, 0, ids][:, [0, 2, 1]].detach().clone()

    hook = actor.route_decoder.top_level_head.register_forward_hook(capture_top)
    try:
        logits = actor.route_logits(
            pad(recurrent[:, 0]), pad(bank['public']), pad(bank['edge']),
            cooperative_features=pad(spatial), local_endpoint_features=pad(local),
            candidate_endpoint_features=pad(candidates), self_features=pad(bank['self']),
            route_action_mask=masks,
        )[indices, 0, ids]
    finally:
        hook.remove()
    probabilities = torch.softmax(logits.masked_fill(~bank['mask'], -torch.inf), dim=-1)
    return {'logits': logits, 'top_logits': captured['top'], 'probabilities': probabilities,
            'indices': probabilities.argmax(-1), 'spatial': spatial, 'recurrent': recurrent[:, 0]}


def hybrid(base: CAGATMAPPOActor, donor: CAGATMAPPOActor, *, swap: str) -> CAGATMAPPOActor:
    result = copy.deepcopy(base)
    prefixes = SHARED_PREFIXES if swap == 'backbone' else (ROUTE_PREFIX,)
    if swap not in ('backbone', 'route'):
        raise ValueError('invalid swap target')
    source, target = donor.state_dict(), result.state_dict()
    selected = [key for key in target if key.startswith(prefixes)]
    if not selected or set(selected) != {key for key in source if key.startswith(prefixes)}:
        raise ValueError('swap module keys differ')
    for key in selected:
        if source[key].shape != target[key].shape:
            raise ValueError(f'incompatible swap tensor: {key}')
        target[key] = source[key].clone()
    result.load_state_dict(target, strict=True)
    result.eval().requires_grad_(False)
    return result


def summarize(label: str, step: int, output: dict[str, torch.Tensor],
              rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    p, chosen, logits, top = [output[k].detach().cpu() for k in
                             ('probabilities', 'indices', 'logits', 'top_logits')]
    mean = p.mean(0)
    summary = {
        'model': label, 'step': step, 'states': len(rows),
        'p_local': float(mean[1]), 'p_remote': float(p[:, 3:].sum(-1).mean()),
        'p_defer': float(mean[2]),
        'fraction_local': float((chosen == 1).double().mean()),
        'fraction_remote': float((chosen >= 3).double().mean()),
        'fraction_defer': float((chosen == 2).double().mean()),
        'top_logit_local': float(top[:, 0].mean()),
        'top_logit_remote': float(top[:, 1].mean()),
        'top_logit_defer': float(top[:, 2].mean()),
        'defer_minus_remote_mean': float((top[:, 2] - top[:, 1]).mean()),
        'defer_minus_remote_min': float((top[:, 2] - top[:, 1]).min()),
        'defer_minus_remote_max': float((top[:, 2] - top[:, 1]).max()),
        'leaf_logit_local': float(logits[:, 1].mean()),
        'leaf_logit_defer': float(logits[:, 2].mean()),
        # Legal-only logsumexp avoids averaging the -1e9 mask sentinel.
        'leaf_remote_logsumexp': float(torch.logsumexp(
            logits[:, 3:].masked_fill(~(p[:, 3:] > 0), -torch.inf), -1).mean()),
    }
    details = [
        {'model': label, 'step': step,
         **{k: row[k] for k in ('evaluation_seed', 'episode_step', 'uav_id', 'task_identity')},
         'action_mask': row['action_mask'],
         'route_domain': row['observation']['action_masks']['domains']['route'],
         'probabilities': p[i].tolist(), 'route_logits': logits[i].tolist(),
         'top_logits_local_remote_defer': top[i].tolist(),
         'selected_index': int(chosen[i])}
        for i, row in enumerate(rows)
    ]
    return summary, details


def classify(h3: dict[str, Any], h4: dict[str, Any]) -> str:
    """Require unanimous deterministic choice for the user's clear cases."""
    d3, d4 = h3['fraction_defer'] == 1, h4['fraction_defer'] == 1
    r3, r4 = h3['fraction_remote'] == 1, h4['fraction_remote'] == 1
    if d3 and r4:
        return 'A'
    if r3 and d4:
        return 'B'
    if d3 and d4:
        return 'C'
    if r3 and r4:
        return 'D'
    return 'MIXED'


def parameter_drift(e3: CAGATMAPPOActor, e4: CAGATMAPPOActor) -> list[dict[str, Any]]:
    modules = {
        'shared_GAT': ('self_encoder.', 'public_encoder.', 'graph_encoder.'),
        'shared_GRU': ('gru.',), 'Route_decoder_head': (ROUTE_PREFIX,),
        'TX_upper': ('action_heads.tx_select.top_level_head.',),
        'TX_destination': ('action_heads.tx_select.destination_head.',),
        'TX_other_context': ('action_heads.tx_select.',),
        'CPU_queue': ('action_heads.cpu_queue.',),
        'CPU_queue_embedding': ('branch_embeddings.cpu_queue.',),
        'CPU_frequency': ('action_heads.cpu_frequency.',),
    }
    left, right = dict(e3.named_parameters()), dict(e4.named_parameters())
    result = []
    for module, prefixes in modules.items():
        a = {k: v.detach().double() for k, v in left.items() if k.startswith(prefixes)}
        b = {k: v.detach().double() for k, v in right.items() if k.startswith(prefixes)}
        if module == 'TX_other_context':
            a = {k: v for k, v in a.items() if not k.startswith(modules['TX_upper'] + modules['TX_destination'])}
            b = {k: v for k, v in b.items() if not k.startswith(modules['TX_upper'] + modules['TX_destination'])}
        if not a or not b:
            raise ValueError(f'module prefix missing: {module}')
        norm_a = sum(float((v * v).sum()) for v in a.values()) ** .5
        norm_b = sum(float((v * v).sum()) for v in b.values()) ** .5
        comparable = a.keys() == b.keys() and all(a[k].shape == b[k].shape for k in a)
        delta = sum(float(((a[k] - b[k]) ** 2).sum()) for k in a) ** .5 if comparable else None
        result.append({'module': module, 'l2_delta': delta,
                       'relative_delta': delta / norm_a if delta is not None else None,
                       'e3_norm': norm_a, 'e4_norm': norm_b,
                       'e3_parameters': sum(v.numel() for v in a.values()),
                       'e4_parameters': sum(v.numel() for v in b.values()),
                       'comparable': comparable,
                       'note': '' if comparable else 'legacy Linear vs candidate scorer: no parameter correspondence'})
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(root: Path, *, output_name: str = 'e4-defer-localization-verified') -> dict[str, Any]:
    torch.set_num_threads(1)
    out = root / 'logs' / output_name
    if out.exists():
        raise FileExistsError('refusing to overwrite existing localization outputs')
    bank_path = root / 'logs/diagnostics/gate-i/source-seed43/factual_state_bank.jsonl'
    bank_hash = file_hash(bank_path)
    rows = [row for line in bank_path.read_text(encoding='utf-8').splitlines()
            if (row := json.loads(line))['branch'] == 'route']
    if not rows or len({(r['evaluation_seed'], r['episode_step'], r['uav_id']) for r in rows}) != len(rows):
        raise ValueError('empty or duplicate fixed bank')
    runs = {}
    finals, configs, inputs, identities = {}, {}, {}, []
    for arm, prefix in EXPECTED_CONFIGS.items():
        matched = list((root / 'logs/gate_m').glob(f'*seed-43__cfg-{prefix[:12]}__git-*'))
        if len(matched) != 1:
            raise ValueError(f'{arm} run identity is ambiguous')
        runs[arm] = matched[0]
        configs[arm] = load_run_config(matched[0] / 'config_snapshot.yaml')
        inputs[arm] = encode_bank(rows, configs[arm])
    if any(not torch.equal(inputs['E3'][k], inputs['E4'][k]) for k in inputs['E3']):
        raise ValueError('E3/E4 tensorized bank inputs differ')
    summaries, details, immutable = [], [], {str(bank_path): bank_hash}
    for arm in ('E3', 'E4'):
        actor = CAGATMAPPOActor(configs[arm]).eval().requires_grad_(False)
        for step in STEPS:
            path = runs[arm] / 'checkpoints' / ('final.pt' if step == 32000 else f'step_{step}.pt')
            if not path.is_file():
                identities.append({'arm': arm, 'step': step, 'missing': True})
                continue
            digest = file_hash(path)
            payload = load_checkpoint_payload(path)
            if (payload['config_hash'] != EXPECTED_CONFIGS[arm]
                    or payload['config_snapshot']['seed'] != 43
                    or payload['training_state']['collected_environment_transitions'] != step
                    or payload['checkpoint_kind'] != ('FINAL_COMPLETED' if step == 32000 else 'PERIODIC_RESUME')):
                raise ValueError(f'{arm} step {step} checkpoint identity failed')
            if step == 32000 and digest != EXPECTED_FINALS[arm]:
                raise ValueError('final checkpoint SHA-256 differs')
            immutable[str(path)] = digest
            actor.load_state_dict(payload['model_state']['actor'], strict=True)
            with torch.inference_mode():
                output = route_forward(actor, inputs[arm])
            summary, state_rows = summarize(arm, step, output, rows)
            summaries.append(summary)
            details.extend(state_rows)
            identities.append({'arm': arm, 'step': step, 'path': str(path), 'sha256': digest,
                               'config_hash': payload['config_hash'], 'git_commit': payload['git_commit']})
            print(json.dumps(summary), flush=True)
            if step == 32000:
                finals[arm] = copy.deepcopy(actor)
                configs[arm + '_snapshot'] = canonicalize_config_identity(payload['config_snapshot'])
    if set(finals) != {'E3', 'E4'}:
        raise ValueError('both finals are required')
    hybrids = [
        ('H1', finals['E3']), ('H2', finals['E4']),
        ('H3', hybrid(finals['E3'], finals['E4'], swap='backbone')),
        ('H4', hybrid(finals['E3'], finals['E4'], swap='route')),
    ]
    hybrid_rows, outputs = [], {}
    for label, actor in hybrids:
        with torch.inference_mode():
            output = route_forward(actor, inputs['E3'])
        outputs[label] = output
        summary, state_rows = summarize(label, 32000, output, rows)
        hybrid_rows.append(summary)
        details.extend(state_rows)
        print(json.dumps(summary), flush=True)
    for label, arm in (('H1', 'E3'), ('H2', 'E4')):
        original = next(r for r in summaries if r['model'] == arm and r['step'] == 32000)
        if any(original[k] != next(r for r in hybrid_rows if r['model'] == label)[k]
               for k in original if k != 'model'):
            raise ValueError('unswapped baseline repeat differs')
    drift = parameter_drift(finals['E3'], finals['E4'])
    case = classify(hybrid_rows[2], hybrid_rows[3])
    if hybrid_rows[0]['fraction_remote'] != 1 or hybrid_rows[1]['fraction_defer'] != 1:
        case = 'BASELINE_NOT_REPRODUCED'
    e4 = [r for r in summaries if r['model'] == 'E4']
    first_all = next((r['step'] for r in e4 if r['fraction_defer'] == 1), None)
    first_majority = next((r['step'] for r in e4 if r['fraction_defer'] > r['fraction_remote']), None)
    dominance_transitions = [
        {'previous_saved_step': previous['step'], 'first_saved_defer_step': current['step']}
        for previous, current in zip(e4, e4[1:])
        if previous['fraction_remote'] > previous['fraction_defer']
        and current['fraction_defer'] > current['fraction_remote']
    ]
    repr_stats = {}
    for name in ('spatial', 'recurrent'):
        a, b = outputs['H1'][name].double(), outputs['H2'][name].double()
        repr_stats[name] = {
            'relative_l2_delta': float(torch.linalg.vector_norm(b-a) / torch.linalg.vector_norm(a)),
            'mean_cosine_similarity': float(torch.nn.functional.cosine_similarity(a, b).mean()),
        }
    margins = {r['model']: r['defer_minus_remote_mean'] for r in hybrid_rows}
    interaction = margins['H2'] - margins['H3'] - margins['H4'] + margins['H1']
    for path, digest in immutable.items():
        if file_hash(Path(path)) != digest:
            raise ValueError(f'input modified during probe: {path}')
    result = {
        'schema': 'e4-defer-localization-v1', 'case': case,
        'bank': {'path': str(bank_path), 'sha256': bank_hash, 'states': len(rows),
                 'source_seed': 43, 'evaluation_seeds': sorted({r['evaluation_seed'] for r in rows}),
                 'cohort': 'Gate I persisted active Route states with at least one legal Remote candidate',
                 'hidden_protocol': 'independent zero hidden for each snapshot; no missing-history reconstruction',
                 'gate_i_recurrent_protocol_reproduced': False,
                 'per_eval_counts': {str(s): sum(r['evaluation_seed'] == s for r in rows)
                                     for s in sorted({r['evaluation_seed'] for r in rows})}},
        'device': 'cpu', 'dtype': 'float32', 'checkpoints': identities,
        'first_saved_all_defer': first_all,
        'first_saved_defer_deterministic_majority': first_majority,
        'remote_to_defer_saved_transitions': dominance_transitions,
        'evolution': summaries, 'hybrids': hybrid_rows, 'parameter_drift': drift,
        'representation': repr_stats, 'mean_top_margin_interaction': interaction,
        'shared_definition': list(SHARED_PREFIXES), 'route_definition': ROUTE_PREFIX,
        'source_inputs_unchanged_sha256': True, 'environment_instantiated': False,
        'training': False, 'evaluation': False, 'optimizer_step': False,
        'checkpoint_saved': False,
    }
    out.mkdir(parents=True)
    (out / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    with (out / 'route_state_scores.jsonl').open('w', encoding='utf-8') as stream:
        for row in details:
            stream.write(json.dumps(row, separators=(',', ':'), ensure_ascii=False) + '\n')
    dashboard = root / 'dashboard_logs' / output_name
    write_csv(dashboard / 'checkpoint_evolution.csv', summaries)
    write_csv(dashboard / 'final_weight_swap.csv', hybrid_rows)
    write_csv(dashboard / 'parameter_drift.csv', drift)
    print(f'CASE={case} first_saved_all_defer={first_all}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    run(parser.parse_args().root)
