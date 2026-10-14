"""E16 scorer-only offline pretraining, validation gate and warm-start export."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from ..config import load_run_config
from ..models.ca_gat_mappo_checkpoint import load_checkpoint_payload
from ..models.destination_pretraining import (load_dataset, feature_spec, stable_hash, file_hash,
    initialize_from_base, validate_p1a_configuration, pretrain_gate, save_warmstart, tensor_state_hash)
from ..models.destination_ranking import listwise_loss, masked_log_probs


def informative_rows(records, split):
    return [r for r in records if r['split'] == split and r['sample_class'] == 'informative_multi']


def tensors(rows, device):
    if not rows:
        raise ValueError('empty informative cohort')
    return (torch.tensor([r['scorer_input'] for r in rows], device=device, dtype=torch.float32),
            torch.tensor([r['candidate_mask'] for r in rows], device=device, dtype=torch.bool),
            torch.tensor([r['target'] for r in rows], device=device, dtype=torch.float32))


def scorer_optimizer(actor, learning_rate):
    actor.requires_grad_(False)
    scorer = actor.route_decoder.system_destination_scorer
    scorer.requires_grad_(True)
    return torch.optim.Adam(scorer.parameters(), lr=learning_rate)


def metrics(scorer, rows, device):
    if not rows:
        return {'count': 0, 'oracle_set_accuracy': None, 'ce': None, 'entropy': None,
                'top1_accuracy': None, 'confusion': {}, 'candidate_count_strata': {}}
    x, legal, target = tensors(rows, device)
    with torch.no_grad():
        scores = scorer(x)
        log = masked_log_probs(scores, legal); probability = log.exp()
        selected = probability.argmax(-1).cpu().tolist()
        ce = listwise_loss(scores, legal, target)
        entropy = -(probability * log).sum(-1)
    correct, exact, confusion, strata = [], [], {}, {}
    for i, r in zip(selected, rows):
        chosen = r['candidate_ids'][i]
        hit = chosen in r['oracle_best_set']; correct.append(hit)
        # Tie-broken top1 is descriptive only; oracle-set accuracy is the gate.
        reference = min(r['oracle_best_set']); exact.append(chosen == reference)
        key = f'{reference}->{chosen}'; confusion[key] = confusion.get(key, 0) + 1
        count = str(len(r['legal_destination_ids']))
        strata.setdefault(count, []).append(hit)
    return {'count': len(rows), 'oracle_set_accuracy': sum(correct) / len(rows),
            'top1_accuracy': sum(exact) / len(rows), 'ce': float(ce.mean()),
            'entropy': float(entropy.mean()), 'confusion': confusion,
            'candidate_count_strata': {k: {'count': len(v), 'accuracy': sum(v) / len(v)} for k, v in strata.items()}}


def distill(settings):
    data = load_dataset(settings['dataset'])
    if data['data_gate']['status'] != 'PASS':
        raise ValueError('OFFLINE ORACLE DATA GATE FAIL; no pretraining')
    config = load_run_config(settings['finetune_config'])
    if stable_hash(feature_spec(config)) != data['provenance']['feature_spec_hash']:
        raise ValueError('dataset scorer feature spec mismatch')
    base_path = settings['base_checkpoint']; before = file_hash(base_path)
    if before != data['provenance']['base_checkpoint_sha256']:
        raise ValueError('dataset base SHA mismatch')
    base = load_checkpoint_payload(base_path)
    validate_p1a_configuration(config, base)
    device = torch.device(settings['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('explicit CUDA requested but unavailable')
    root = Path(settings['output_dir']); root.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(settings['seed'])
    actor, critic = initialize_from_base(config, base)
    actor.to(device); critic.requires_grad_(False)
    optimizer = scorer_optimizer(actor, settings['learning_rate'])
    scorer = actor.route_decoder.system_destination_scorer
    cohorts = {s: informative_rows(data['records'], s) for s in ('train', 'validation', 'reserve')}
    if any(not r for r in cohorts.values()):
        raise ValueError('all three seed groups need informative coverage')
    initial = metrics(scorer, cohorts['validation'], device)
    val = cohorts['validation']
    baseline = sum(r['factual_p1a_destination'] in r['oracle_best_set'] for r in val) / len(val)
    best_ce, best, best_epoch, stale, history = initial['ce'], copy.deepcopy(scorer.state_dict()), 0, 0, []
    generator = torch.Generator().manual_seed(settings['seed'])
    for epoch in range(1, settings['max_epochs'] + 1):
        order = torch.randperm(len(cohorts['train']), generator=generator).tolist()
        for start in range(0, len(order), settings['batch_size']):
            rows = [cohorts['train'][i] for i in order[start:start + settings['batch_size']]]
            x, legal, target = tensors(rows, device)
            loss = listwise_loss(scorer(x), legal, target).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        validation = metrics(scorer, val, device)
        history.append({'epoch': epoch, 'train': metrics(scorer, cohorts['train'], device), 'validation': validation})
        # Model selection uses validation CE only, never reserve or eval1044.
        if validation['ce'] < best_ce - settings['min_delta']:
            best_ce, best, best_epoch, stale = validation['ce'], copy.deepcopy(scorer.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= settings['patience']:
            break
    scorer.load_state_dict(best)
    report = {'schema': 'e16-pretrain-report.v1', 'dataset_hash': data['dataset_hash'],
              'settings_hash': stable_hash(settings), 'config_hash': config.config_hash,
              'base_checkpoint_sha256': before, 'scorer_pretrain_hash': tensor_state_hash(best),
              'selected_epoch': best_epoch, 'initial_validation': initial, 'history': history,
              'final': {s: metrics(scorer, rows, device) for s, rows in cohorts.items()}}
    report['pretrain_gate'] = pretrain_gate(initial, report['final']['validation'], baseline)
    report['base_gradients_zero'] = all(p.grad is None for n, p in actor.named_parameters() if 'system_destination_scorer' not in n)
    if not report['base_gradients_zero'] or file_hash(base_path) != before:
        raise ValueError('pretraining modified frozen P1a')
    best = {k: v.detach().cpu() for k, v in best.items()}
    torch.save({'schema': 'e16-scorer-pretrained.v1', 'scorer_state': best, 'report': report}, root / 'scorer.pt')
    (root / 'pretrain_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    if report['pretrain_gate']['status'] == 'PASS':
        save_warmstart(root / 'warmstart.pt', config, base_path, best, data, report['pretrain_gate'])
    print(json.dumps({'E16 PRETRAIN GATE': report['pretrain_gate'], 'selected_epoch': best_epoch}))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    report = distill(json.loads(Path(args.settings).read_text(encoding='utf-8')))
    return 0 if report['pretrain_gate']['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
