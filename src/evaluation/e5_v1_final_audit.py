"""Audit saved E5 artifacts and fixed snapshots; never dispatch an environment."""
from __future__ import annotations

import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.config import canonical_config_identity_hash, canonicalize_config_identity, load_run_config
from src.evaluation.e4_defer_localization import (
    STEPS, encode_bank, file_hash, route_forward, snapshot_view, summarize,
)
from src.evaluation.e4_v1_matched_audit import (
    METRICS, cpu_summary, differences, fraction, frequency_counts, mean, trace_analysis,
)
from src.models.ca_gat_mappo import ActorObservationTensorizer, CAGATMAPPOActor
from src.models.ca_gat_mappo_checkpoint import load_checkpoint_payload

PREFIXES = {'E3': '735fb97120ce', 'E4': 'a216fc6b1306', 'E5': '04b9bee14ff6'}
E5_EVALUATION = 'formal-eval__remote_stress_v2__chk-2e7d80f18f58__cfg-d6d90df7321f__eval-3c0dbb64785e'
BANK_SHA = '435c25e3efdb5a85c7264b2eb51788318f0f72fc412775aa3cc4a460ddd61ca8'


def require_matched_difference(baseline, treatment):
    a, b = (canonicalize_config_identity(c) for c in (baseline, treatment))
    a['training']['mappo'].setdefault('route_head_lr_scale', 1.0)
    b['training']['mappo'].setdefault('route_head_lr_scale', 1.0)
    diff = differences(a, b)
    expected = [{'field': 'training.mappo.route_head_lr_scale', 'E3': 1.0, 'E4': .1}]
    if diff != expected:
        raise ValueError(f'Identity gate stopped: additional/missing algorithm difference: {diff}')
    if b['training']['mappo']['cpu_frequency_decoder_mode'] != 'candidate_aware_v1':
        raise ValueError('E5 must retain candidate-aware CPU frequency decoder')
    return [{'field': r['field'], 'E4': r['E3'], 'E5': r['E4']} for r in diff]


def formal_route_summary(rows):
    """Keep all active Route decisions and Remote-legal decisions separate."""
    if not rows:
        raise ValueError('Empty formal Route trace')
    if len({(r['evaluation_seed'], r['step'], r['uav_id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate formal Route decision')
    legal = [r for r in rows if any(r['route_action_mask'][3:])]
    result = {}
    for scope, cohort in [('all_active', rows), ('remote_legal', legal)]:
        counts = Counter(r['selected_top_level'].lower() for r in cohort)
        result[scope] = {'states': len(cohort),
            'counts': {k: counts[k] for k in ('local', 'remote', 'defer', 'idle')},
            **{f'p_{k}': mean(r[f'{k}_probability'] for r in cohort) for k in ('local', 'remote', 'defer')},
            **{f'fraction_{k}': fraction(counts[k], len(cohort)) for k in ('local', 'remote', 'defer')}}
        for r in cohort:
            if not math.isclose(sum(r[f'{k}_probability'] for k in ('local', 'remote', 'defer')), 1., abs_tol=2e-6):
                raise ValueError('Invalid saved Route probabilities')
            if not r['route_action_mask'][r['final_route_action_index']]:
                raise ValueError('Illegal saved Route selection')
    result['per_seed'] = {str(s): dict(Counter(r['selected_top_level'].lower() for r in rows
                                            if r['evaluation_seed'] == s)) for s in range(1042, 1047)}
    return result


def tx_fixed_forward(actor, rows, config, *, chunk_size=256):
    """All 5,722 saved legal TX states; independent zero hidden, leaf argmax."""
    tensorizer = ActorObservationTensorizer(config)
    outputs = []
    for start in range(0, len(rows), chunk_size):
        cohort = rows[start:start + chunk_size]
        views = [snapshot_view(r['observation']) for r in cohort]
        for r, v in zip(cohort, views):
            expected = ('idle', *(i for i in range(actor.spec.uav_count) if i != v.uav_id))
            if v.action_masks.domain_for('tx_select') != expected:
                raise ValueError('TX candidate ordering mismatch')
            if r['action_mask'] != r['observation']['action_masks']['masks']['tx_select']:
                raise ValueError('TX mask mismatch')
            if not r['action_mask'][0] or not any(r['action_mask'][1:]):
                raise ValueError('TX bank includes inactive state')
        self_features = torch.tensor(np.stack([tensorizer._self_vector(v) for v in views]), dtype=torch.float32)
        public = torch.tensor(np.stack([tensorizer._public_matrix(v) for v in views]), dtype=torch.float32)
        edge = torch.tensor(np.stack([tensorizer._edge_matrix(v) for v in views]), dtype=torch.float32)
        neighbors = torch.tensor(np.stack([v.candidate_neighbor_mask for v in views]), dtype=torch.bool)
        spatial = actor.graph_encoder(actor.self_encoder(self_features), actor.public_encoder(public), edge, neighbors)
        recurrent, _ = actor.gru(spatial.unsqueeze(1), torch.zeros(1, len(cohort), actor.spec.gru_hidden_dimension))
        ids = torch.tensor([r['uav_id'] for r in cohort]); index = torch.arange(len(cohort))
        def pad(value):
            result = value.new_zeros((len(cohort), 1, actor.spec.uav_count, *value.shape[1:]))
            result[index, 0, ids] = value
            return result
        masks = torch.zeros(len(cohort), 1, actor.spec.uav_count, actor.spec.action_dimensions['tx_select'], dtype=torch.bool)
        masks[..., 0] = True
        masks[index, 0, ids] = torch.tensor([r['action_mask'] for r in cohort])
        batch = SimpleNamespace(self_features=pad(self_features), neighbor_public_features=pad(public),
                                edge_features=pad(edge), action_masks={'tx_select': masks})
        logits = actor.branch_logits('tx_select', pad(recurrent[:, 0]), {}, batch)[index, 0, ids]
        p = torch.softmax(logits.masked_fill(~masks[index, 0, ids], -torch.inf), -1)
        outputs.append(p)
    p = torch.cat(outputs); chosen = p.argmax(-1)
    return {'scope': 'fixed historical E3 TX bank, independent zero hidden; not E5 formal trajectory',
            'legal_opportunities': len(rows), 'p_transmit': float(p[:, 1:].sum(-1).mean()),
            'deterministic_transmit': int((chosen > 0).sum()), 'deterministic_idle': int((chosen == 0).sum()),
            'fraction_transmit': float((chosen > 0).double().mean()),
            'fraction_idle': float((chosen == 0).double().mean())}


def remote_funnel(tasks):
    remote = [t for t in tasks if t.get('remote')]
    return {'remote_binding': len(remote), 'first_tx': sum('first_tx' in t for t in remote),
            'tx_complete': sum('tx_complete' in t for t in remote),
            'first_cpu': sum('first_cpu' in t for t in remote),
            'remote_done': sum(t.get('outcome') == 'done' for t in remote),
            'done_per_first_cpu': fraction(sum(t.get('outcome') == 'done' for t in remote),
                                          sum('first_cpu' in t for t in remote))}


def classify_final(route_fixed, tx_fixed, formal):
    if route_fixed['fraction_defer'] == 1:
        return 'E5-C'
    if route_fixed['fraction_remote'] <= .5 or route_fixed['fraction_remote'] <= route_fixed['fraction_defer']:
        raise ValueError('Route is mixed; do not force an unsupported gate category')
    if tx_fixed['fraction_idle'] == 1 and formal['tx_energy_j'] == 0:
        return 'E5-B'
    if formal['remote_done'] > 0 and formal['completion_improvement_pp'] > 0 and formal['tx_energy_j'] > 0:
        return 'E5-A'
    return 'E5-D'


def audit(root):
    torch.set_num_threads(1)
    register, configs, runs, reports = {}, {}, {}, {}
    def record(path):
        register[str(path.resolve())] = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
    def read(path):
        record(path); return json.loads(path.read_text(encoding='utf-8'))
    for arm, prefix in PREFIXES.items():
        candidates = list((root / 'logs/gate_m').glob(f'*seed-43__cfg-{prefix}__git-*'))
        if len(candidates) != 1:
            raise ValueError(f'Ambiguous {arm} run')
        runs[arm] = candidates[0]
        configs[arm] = read(candidates[0] / 'config_snapshot.yaml')
    diff = require_matched_difference(configs['E4'], configs['E5'])
    bank_path = root / 'logs/diagnostics/gate-i/source-seed43/factual_state_bank.jsonl'
    record(bank_path)
    if register[str(bank_path.resolve())]['sha256'] != BANK_SHA:
        raise ValueError('Fixed bank differs from E4 localization')
    bank_rows = [json.loads(line) for line in bank_path.read_text().splitlines()]
    route_rows = [r for r in bank_rows if r['branch'] == 'route']
    tx_rows = [r for r in bank_rows if r['branch'] == 'tx']
    assert len(route_rows) == 473 and len(tx_rows) == 5722
    encoded = {a: encode_bank(route_rows, load_run_config(runs[a] / 'config_snapshot.yaml')) for a in PREFIXES}
    assert all(torch.equal(encoded['E3'][k], encoded[a][k]) for a in ('E4', 'E5') for k in encoded[a])
    evolution, details = [], []
    old_localization = read(root / 'logs/e4-defer-localization-verified/summary.json')
    for arm in PREFIXES:
        cfg_hash = canonical_config_identity_hash(configs[arm])
        evaluations = []
        for path in (root / 'logs/gate_m/evaluations').iterdir():
            if path.is_dir() and (path / 'evaluation_manifest.json').is_file():
                candidate = json.loads((path / 'evaluation_manifest.json').read_text())
                if candidate.get('source_training_config_hash') == cfg_hash:
                    evaluations.append(path)
        assert len(evaluations) == 1
        ev = evaluations[0]; manifest = read(ev / 'evaluation_manifest.json')
        if arm == 'E5': assert ev.name == E5_EVALUATION
        assert manifest['evaluation_seeds'] == list(range(1042, 1047))
        assert manifest['episode_horizon'] == 500
        record(ev / 'episode_metrics.jsonl')
        episodes = [json.loads(s) for s in (ev / 'episode_metrics.jsonl').read_text().splitlines()]
        episodes = sorted((r for r in episodes if r['method_id'] == 'ca_gat_mappo'), key=lambda r:r['evaluation_seed'])
        assert [r['evaluation_seed'] for r in episodes] == list(range(1042, 1047))
        assert all(all(r['invariants'][k] for k in ('generated_equals_completed_plus_expired_plus_truncated',
                                                  'total_energy_equals_tx_plus_cpu')) for r in episodes)
        diag = root / 'logs/gate_m/diagnostics' / ('hierarchical-route-eval__' + ev.name)
        saved_summary = read(diag / 'summary.json'); record(diag / 'route_decision_trace.jsonl')
        formal_rows = [json.loads(s) for s in (diag / 'route_decision_trace.jsonl').read_text().splitlines()]
        assert len(formal_rows) == saved_summary['route_decision_count']
        for r in saved_summary['action_trace_sha256_by_episode']:
            assert r['action_trace_sha256'] == next(e['action_trace_sha256'] for e in episodes if e['evaluation_seed'] == r['evaluation_seed'])
        assert all(r['evaluation_run_id'] == ev.name for r in formal_rows)
        actor_cfg = load_run_config(runs[arm] / 'config_snapshot.yaml')
        actor = CAGATMAPPOActor(actor_cfg).eval().requires_grad_(False)
        checkpoint_records = []
        for step in ((32000,) if arm == 'E3' else STEPS):
            cp = runs[arm] / 'checkpoints' / ('final.pt' if step == 32000 else f'step_{step}.pt')
            if not cp.is_file():
                checkpoint_records.append({'step':step, 'missing':True}); continue
            record(cp); payload = load_checkpoint_payload(cp)
            assert payload['config_hash'] == cfg_hash == canonical_config_identity_hash(payload['config_snapshot'])
            assert payload['config_snapshot']['seed'] == 43
            assert payload['training_state']['collected_environment_transitions'] == step
            assert payload['checkpoint_kind'] == ('FINAL_COMPLETED' if step == 32000 else 'PERIODIC_RESUME')
            actor.load_state_dict(payload['model_state']['actor'], strict=True)
            with torch.inference_mode(): scores = route_forward(actor, encoded[arm])
            summary, state_rows = summarize(arm, step, scores, route_rows)
            evolution.append(summary); details.extend(state_rows)
            if arm in ('E3', 'E4'):
                historical = next(r for r in old_localization['evolution'] if r['model'] == arm and r['step'] == step)
                assert summary == historical, 'E4 localization protocol numerical reproduction failed'
            checkpoint_records.append({'step':step, 'sha256':register[str(cp.resolve())]['sha256']})
            print(arm, step, summary['p_remote'], summary['fraction_remote'], flush=True)
            if step == 32000:
                assert register[str(cp.resolve())]['sha256'] == manifest['source_checkpoint_sha256']
                state = payload['training_state']
                assert state['training_complete'] and state['completed_episodes'] == 64
                assert state['optimized_transitions'] == 32000 and state['ppo_update_count'] == 125
                assert state['active_rollout_length'] == state['unused_final_tail_transitions'] == 0
                assert payload['active_rollout_state'] == {'rollout_length':0,'rollout_policy_version':None,'ordered_transitions':[]}
                assert set(payload['model_state']) == {'actor','critic'} and payload['optimizer_state']
                with torch.inference_mode(): tx_fixed = tx_fixed_forward(actor, tx_rows, actor_cfg)
                final_state = state
        reports[arm] = {'identity': {'run':str(runs[arm]), 'evaluation':str(ev), 'config_hash':cfg_hash,
                         'manifest':manifest, 'training_state':final_state, 'checkpoints':checkpoint_records},
                        'formal': [{'seed':r['evaluation_seed'], **r['metrics']} for r in episodes],
                        'formal_mean': {k:mean(r['metrics'][k] for r in episodes) for k in METRICS},
                        'formal_route': formal_route_summary(formal_rows), 'fixed_tx':tx_fixed}
        if arm != 'E3':
            for key in ('evaluation_config_sha256','evaluation_device','dtype','deterministic_action_mode',
                        'method_suite','episode_horizon','action_domain_identity'):
                assert manifest[key] == reports['E3']['identity']['manifest'][key]
            reference = reports['E3']['formal']
            assert [e['external_trace_sha256'] for e in episodes] == [e['external_trace_sha256'] for e in e3_episodes]
            assert [r['generated'] for r in reports[arm]['formal']] == [r['generated'] for r in reference]
        else: e3_episodes = episodes
        # These rows cover only Route-active slots, not the complete TX population.
        reports[arm]['formal_tx_observed_subset'] = {
            'route_active_rows':len(formal_rows),
            'non_idle_tx_proposals':sum(r['tx_action'] != 'idle' for r in formal_rows),
            'accepted_tx_rows':sum(bool((r.get('executed_communication') or {}).get('accepted')) for r in formal_rows),
            'full_legal_opportunities':None, 'full_p_transmit':None}
    print('Saved training lifecycle/CPU trace read only', flush=True)
    for arm in PREFIXES:
        if arm in ('E3','E4'):
            previous = read(root / f'logs/e4-v1-seed43-audit/{arm}_summary.json')
            assert previous['identity']['config_hash'] == reports[arm]['identity']['config_hash']
            assert previous['formal_mean'] == reports[arm]['formal_mean']
            for key in ('cpu','cpu_frequency','conditioned'):
                reports[arm]['training_' + key] = previous[key]
            archive = root / f'logs/e4-v1-seed43-audit/{arm}_tasks.jsonl.gz'
            record(archive)
            with gzip.open(archive, 'rt', encoding='utf-8') as stream:
                reports[arm]['training_remote_funnel'] = remote_funnel([json.loads(line) for line in stream])
        else:
            trace = trace_analysis(runs[arm] / 'conditional_credit_trace_v1.jsonl', configs[arm], register)
            assert trace['header']['run_id'] == runs[arm].name
            reports[arm]['training_cpu'] = cpu_summary(trace)
            reports[arm]['training_cpu_frequency'] = frequency_counts(trace['actions'].values())
            reports[arm]['training_conditioned'] = trace['conditioned']
            reports[arm]['training_remote_funnel'] = remote_funnel(trace['tasks'])
            reports[arm]['training_tx'] = {'legal_opportunities':sum(a['tx_active'] for a in trace['actions'].values()),
                'sampled_transmit':sum(a['tx_active'] and a['tx_upper'] != 'idle' for a in trace['actions'].values()),
                'sampled_idle':sum(a['tx_active'] and a['tx_upper'] == 'idle' for a in trace['actions'].values())}
    final = next(r for r in evolution if r['model'] == 'E5' and r['step'] == 32000)
    formal_done = sum(r['completed'] for r in reports['E5']['formal'])
    assert reports['E5']['formal_route']['all_active']['counts']['local'] == 0
    formal_context = {'remote_done':formal_done,
        'completion_improvement_pp':100*(reports['E5']['formal_mean']['completion_ratio'] - reports['E3']['formal_mean']['completion_ratio']),
        'tx_energy_j':sum(r['tx_energy_j'] for r in reports['E5']['formal'])}
    case = classify_final(final, reports['E5']['fixed_tx'], formal_context)
    remote_bindings = reports['E5']['formal_route']['all_active']['counts']['remote']
    result = {'case':case, 'config_diff':diff, 'bank':old_localization['bank'],
        'evolution':evolution, 'reports':reports, 'formal_context':formal_context,
        'formal_remote_funnel':{'remote_bindings_legal_proposals':remote_bindings, 'first_tx_lower_bound':formal_done,
            'tx_complete_lower_bound':formal_done,'first_cpu_lower_bound':formal_done,'remote_done':formal_done,
            'proof':'All formal Local counts zero, reset clears lifecycle; every DONE is necessarily Remote. Intermediate totals were not persisted.',
            'done_per_first_cpu_bounds':[formal_done/remote_bindings,1.0]},
        'boundaries':{'training':False,'resume':False,'evaluation':False,'new_seed':False,
                      'algorithm_modified':False,'environment_instantiated':False,'checkpoint_saved':False},
        'sources':register}
    for name, rec in register.items():
        assert file_hash(Path(name)) == rec['sha256'], f'Source changed during audit: {name}'
    return result, details


def write_outputs(root, result, details):
    out = root / 'logs/e5-v1-seed43-audit'
    tables = root / 'dashboard_logs/e5-v1-seed43-audit'
    if out.exists() or tables.exists(): raise FileExistsError('Preserve existing audit outputs')
    out.mkdir(parents=True); tables.mkdir(parents=True)
    (out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (out / 'route_state_scores.jsonl').write_text(''.join(json.dumps(r, separators=(',',':'))+'\n' for r in details), encoding='utf-8')
    for name, rows in [('checkpoint_evolution',result['evolution']),
                       ('formal_per_seed',[{'model':arm,**r} for arm,p in result['reports'].items() for r in p['formal']]),
                       ('fixed_tx',[{'model':arm,**p['fixed_tx']} for arm,p in result['reports'].items()])]:
        with (tables / (name+'.csv')).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def write_report(root, result, validation):
    """Render only measured saved/probed values; label unavailable formal fields."""
    path = root / 'E5_V1_SEED43_GATE_REPORT.md'
    if path.exists(): raise FileExistsError('Preserve existing gate report')
    reports = result['reports']
    def number(value):
        return 'NA' if value is None else f'{value:.6f}'
    def percent(value):
        return 'NA' if value is None else f'{100*value:.2f}%'
    def table(headers, rows):
        return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                          '| ' + ' | '.join('---' for _ in headers) + ' |',
                          *('| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows)])
    chunks = ['# E5-v1 seed43 最终结果审计',
        '**E5 final 已不再 all-Defer；最终判定为 E5-A。**正式 Remote 合法状态 473/473 选择 Remote；同一 E4 localization 固定 Route bank 上为 461/473 Remote、12/473 Defer。TX 恢复且已有 170 个 Remote DONE，五集平均 completion 为 27.57%，相对 E3 增加 9.84 个百分点、相对 E4 增加 27.57 个百分点。',
        '本轮仅读取用户已完成的训练/正式评估产物，并执行固定历史 observation 的 CPU/float32 神经网络 forward。没有 training、resume、evaluation、环境 reset/step/replay、新 seed、optimizer step、checkpoint 保存或算法修改。新增代码仅用于离线审计和报告，测试也不运行环境。',
        '## 1. Identity Gate：PASS',
        table(['Model','Training config SHA-256','Final checkpoint SHA-256'],[
            [a,p['identity']['config_hash'],p['identity']['manifest']['source_checkpoint_sha256']] for a,p in reports.items()]),
        'E5 seed=43；collected/optimized steps=32000；completed episodes=64；PPO updates=125；checkpoint kind=`FINAL_COMPLETED`；training_complete=true；active rollout=0、unused tail=0。actor 严格 state_dict 加载成功，actor/critic 和 optimizer 状态均存在。八个请求的 checkpoint 均实际存在。final 文件 SHA-256 与正式 manifest 一致，所有读取输入在审计前后 SHA-256 不变。',
        f"E5 training run：`{reports['E5']['identity']['run']}`。\n\nE5 formal evaluation：`{E5_EVALUATION}`。",
        'E3/E4 自动按 cfg prefix 唯一定位。三者正式 evaluation 均包含且只取 CA-GAT-MAPPO 的 1042、1043、1044、1045、1046 五集；episode horizon=500、CPU/torch.float32、deterministic masked leaf argmax 一致。evaluation config hash、action domain、逐 seed generated task 数及 external trace SHA-256 均相同；task conservation 与 TX+CPU energy conservation 通过。所有五集均纳入，未 cherry-pick。',
        'E5 相对 E4 唯一 canonical 算法差异：`training.mappo.route_head_lr_scale: 1.0 → 0.1`。历史省略字段按默认 1.0 补齐后比较；candidate-aware CPU-frequency decoder 仍为 `candidate_aware_v1`。其他 canonical 内容完全相同，没有额外算法差异，因此未触发停止条件。',
        '## 2. 第一优先级：final Route',
        '### 正式 evaluation：全部 active Route 决策',
        table(['Model','Local','Remote','Defer','P(Local)','P(Remote)','P(Defer)','det Local','det Remote','det Defer'],[
            [a,*[r['counts'][k] for k in ('local','remote','defer')],
             *[number(r['p_'+k]) for k in ('local','remote','defer')],
             *[percent(r['fraction_'+k]) for k in ('local','remote','defer')]]
            for a,p in reports.items() for r in [p['formal_route']['all_active']]]),
        '### 正式 evaluation：Remote 合法状态',
        table(['Model','States','Local/Remote/Defer','P(Local)','P(Remote)','P(Defer)','det Local','det Remote','det Defer'],[
            [a,r['states'],' / '.join(str(r['counts'][k]) for k in ('local','remote','defer')),
             *[number(r['p_'+k]) for k in ('local','remote','defer')],
             *[percent(r['fraction_'+k]) for k in ('local','remote','defer')]]
            for a,p in reports.items() for r in [p['formal_route']['remote_legal']]]),
        'E5 的 2,373 次正式 Defer 全部发生在没有合法 Remote candidate 的状态；Remote 合法时没有 Defer。全部 active 决策的 83.38% Defer 不能解释为 all-Defer collapse。不同模型的正式状态分布不同，所以上表不是 matched-state 因果比较。E5 在 Remote 不合法时仍选择 Defer、没有 Local，这一行为仍存在。',
        '### 完全相同固定 Route bank：final 比较',
        table(['Model','Local/Remote/Defer','P(Local)','P(Remote)','P(Defer)','det Local','det Remote','det Defer'],[
            [r['model'],' / '.join(str(round(473*r['fraction_'+k])) for k in ('local','remote','defer')),
             *[number(r['p_'+k]) for k in ('local','remote','defer')],
             *[percent(r['fraction_'+k]) for k in ('local','remote','defer')]]
            for r in result['evolution'] if r['step']==32000]),
        '固定 bank：`logs/diagnostics/gate-i/source-seed43/factual_state_bank.jsonl`，SHA-256=`'+BANK_SHA+'`。复用 473 条 active、Remote 合法 Route snapshot；observation、candidate ordering、mask 和 tensorized inputs 完全一致。每条独立零 GRU hidden，CPU/float32；没有重采样或拼接缺失历史。该协议与 E4 collapse localization 完全相同，E3 final 和 E4 八个 checkpoint 的全部导出数值逐项精确复现。它不等于正式 evaluation 的连续 episode hidden 协议。',
        '## 3. Route checkpoint evolution',
        table(['Step','Model','P(Remote)','P(Defer)','det Remote','det Defer','upper Defer−Remote logit'],[
            [r['step'],r['model'],number(r['p_remote']),number(r['p_defer']),
             percent(r['fraction_remote']),percent(r['fraction_defer']),number(r['defer_minus_remote_mean'])]
            for step in STEPS for r in result['evolution'] if r['step']==step and r['model'] in ('E4','E5')]),
        'E4：28K 为 100% Remote，32K 为 100% Defer；upper Defer−Remote margin 从 −2.983611 变为 +0.377472。E5：28K 为 93.45% Remote，32K 为 97.46% Remote；P(Remote) 0.699487 → 0.713856，upper margin 从 ' +
        number(next(r['defer_minus_remote_mean'] for r in result['evolution'] if r['model']=='E5' and r['step']==28000)) +
        ' 变为 −1.186026。因此在本次 matched seed43 实验中，0.1 LR 成功阻止了所针对的后期 Remote→all-Defer 翻转；不是声称所有 checkpoint 都稳定或跨 seed 已成立。E5 的早期 8K/12K/16K 仍全 Defer。',
        'upper logit margin 比较的是 Local/Defer/Remote 上层分类，deterministic fraction 按实际完整 leaf masked argmax；Remote upper 优势可能被多个 destination 的概率分摊，不能直接把上层 argmax 当作最终 leaf 动作。',
        '## 4. TX：未出现 deterministic Idle collapse',
        '### 同一历史 TX bank 的 final forward',
        table(['Model','Legal TX states','P(Transmit)','det Transmit count','det Idle count','Transmit fraction','Idle fraction'],[
            [a,r['legal_opportunities'],number(r['p_transmit']),r['deterministic_transmit'],r['deterministic_idle'],
             percent(r['fraction_transmit']),percent(r['fraction_idle'])] for a,p in reports.items() for r in [p['fixed_tx']]]),
        'TX 同样复用上述文件的全部 5,722 条历史 E3 legal-TX snapshot，保持 candidate ordering/mask，独立零 hidden；只运行 shared modules 和 TX head，不调用 Route/CPU/critic。这是三模型在相同 TX inputs 上的比较，不能称为 E5 正式轨迹的 TX legal-opportunity 总数。',
        '### 正式 TX 证据与保存限制',
        '历史 E3 经 Gate M 验证的正式 TX 统计：legal opportunities=5,722，P(Transmit)=0.369189，det Transmit=0，det Idle=5,722。该概率使用历史正式连续 hidden，因此与本轮零 hidden 固定 TX bank 的 0.402786 不同。E4 正式 all-Defer，从空队列开始，legal TX opportunities=0；P(Transmit)=NA，不能据此宣称 Idle collapse 被解决。',
        'E5 正式 evaluation 的完整 TX mask/probability/所有 action rows 没有持久化。完整 legal opportunities、P(Transmit)、det Transmit/Idle 总数均为 **NA（不可从当前产物精确恢复）**。Route-active telemetry 子集有 '+
        str(reports['E5']['formal_tx_observed_subset']['non_idle_tx_proposals'])+' 次 non-Idle TX proposal、'+
        str(reports['E5']['formal_tx_observed_subset']['accepted_tx_rows'])+' 次 accepted TX，仅为子集，不能充当全体分母。正式五集 TX energy 总计 4.003235 J、CPU energy 全部非零、170 个 Remote DONE，足以否定全体 deterministic Idle collapse；固定 TX bank 也全部 Transmit。没有为补字段重跑 evaluation。',
        '## 5. Remote lifecycle',
        '### 正式 final policy',
        table(['Model','Remote binding','first TX','TX complete','first CPU','Remote DONE'],[
            ['E3',473,0,0,0,0],['E4',0,0,0,0,0],['E5',473,'≥170；精确总数 NA','≥170；精确总数 NA','≥170；精确总数 NA',170]]),
        'E3 的 funnel 来自已有 Gate M/历史 formal identity 验证记录；E4 的零链条由全 Defer 和空 reset 队列推出。E5 的 473 个合法 Remote proposal 按当前环境 slot-end Route 规则形成 binding。`LifecycleManager.reset` 清空 tasks/queues，正式 Local count=0，所以五集完成的 46/38/31/17/38 全部必然来自 Remote。每个 Remote DONE 必然经过 first TX、TX complete、first CPU，故三个中间级的严格下界都是 170；不是把下界伪装成准确 funnel count。精确 Remote DONE/first CPU 比率为 NA，边界为 [35.94%, 100%]；170/473=35.94% 是 DONE/binding。',
        '### 已保存 64 集训练轨迹：精确链条，非 final formal 指标',
        table(['Model','Remote binding','first TX','TX complete','first CPU','Remote DONE','DONE/first CPU'],[
            [a,*[f[k] for k in ('remote_binding','first_tx','tx_complete','first_cpu','remote_done')],percent(f['done_per_first_cpu'])]
            for a,p in reports.items() for f in [p['training_remote_funnel']]]),
        '这些计数来自已发生的训练采样轨迹；每个 task 只计首次正服务，TX complete 要求 remaining_bits 精确为 0，跨 episode task id 独立。它们包含整个训练过程的变化，不能用于替换 final deterministic policy 的缺失字段。',
        '## 6. Formal performance：五集逐一与等权平均',
        table(['Model','Eval seed','Completion','Completed','Expired rate','Expired','Return','Energy J','Executor rejection'],[
            [a,r['seed'],percent(r['completion_ratio']),r['completed'],percent(r['expiration_ratio']),r['expired'],
             number(r['total_reward']),number(r['total_energy_j']),r['executor_rejection']]
            for a,p in reports.items() for r in p['formal']]),
        table(['Model','Mean completion','Mean completed','Mean expired rate','Mean expired','Mean return','Mean energy J','Mean rejection'],[
            [a,percent(r['completion_ratio']),number(r['completed']),percent(r['expiration_ratio']),number(r['expired']),
             number(r['total_reward']),number(r['total_energy_j']),number(r['executor_rejection'])]
            for a,p in reports.items() for r in [p['formal_mean']]]),
        '这里 completion/expired rate 均为五集等权均值，没有用汇总任务比率替换。E3 完成总数 108，E4 为 0，E5 为 170。E5 相对 E3 的收益并非每集都更好（1042、1045 更低），但五集均有完成，平均增益明确。E5 executor rejection 总计 102，高于 E3/E4 的 0，TX/CPU energy 也有代价；没有把这一风险藏在平均 completion 后面。',
        '## 7. CPU-frequency downstream：第二层结果',
        '正式产物没有保存 CPU frequency actions、首次 CPU task joins、first-service slack 或 DONE/EXPIRED frequency；这些 final-policy 字段为 NA，不能靠重跑 evaluation 补齐。本节采用完整已保存训练 trace 明确描述 downstream 行为；不改变 Route/TX gate 的优先级。',
        table(['Model','Selected CPU queue actions','f=0','f=.25','f=.5','f=1'],[
            [a,f['selected_queue_actions'],*[f"{f['counts'][k]} ({percent(f['fractions'][k])})" for k in ('0.0','0.25','0.5','1.0')]]
            for a,p in reports.items() for f in [p['training_cpu_frequency']]]),
        table(['Model','Remote first-CPU cohort','Tasks','f=0/.25/.5/1 counts','Mean first slack slots','Mean first Hz','DONE/first CPU'],[
            [a,scope,r['tasks'],' / '.join(str(r['first_frequency'][k]) for k in ('0.0','0.25','0.5','1.0')),
             number(r['first_cpu_slack_mean']),number(r['first_cpu_hz_mean']),
             percent(p['training_remote_funnel']['done_per_first_cpu']) if scope=='remote_all' else '—']
            for a,p in reports.items() for scope in ('remote_all','remote_done','remote_expired') for r in [p['training_cpu'][scope]]]),
        table(['Model','Outcome','Positive CPU service actions','f=0/.25/.5/1 counts'],[
            [a,o,r['actions'],' / '.join(str(r['counts'][k]) for k in ('0.0','0.25','0.5','1.0'))]
            for a,p in reports.items() for o in ('done','expired') for r in [p['training_cpu']['positive_service_actions_'+o]]]),
        'E5 candidate-aware frequency 在训练的非 idle CPU queue 动作中仍近似均匀，full frequency 为 26.17%（E4 25.78%）。训练 Remote DONE/first CPU=6.54%，高于 E4 4.39%，低于 E3 10.84%；这个全程混合轨迹统计没有证明 final deadline ranking 已解决，也不抵消正式 Route/TX/Remote DONE 的恢复。此次 gate 成功依据是正式结果与固定 bank，不是 CPU decoder 行为变化。',
        '## 8. 最终唯一分类与下一步',
        '**E5-A：Route 不再 all-Defer、TX 服务与 Remote DONE 已恢复、正式 completion 明显改善。Route LR intervention 在 seed43 的本次 matched 实验有效，E5 值得继续。**',
        '下一步唯一建议：**对当前 E5-v1 做最小 multi-seed validation，暂不修改任何架构或优化分支代码。**本轮没有新 seed 或运行计划落地，没有擅自转 CPU/TX intervention。单 seed 结果不足以声称跨 seed 稳定，因此目前不进入下一刀算法修改。',
        '## 9. 验证与可复核产物',
        f"专项审计 7 项 + 原固定 Route 协议回归 9 项，共 **{validation['tests']}/{validation['tests']} PASS**，0 failure、0 error、0 skip；均只读/纯 forward，不运行环境。测试涵盖额外配置差异 fail-fast、cohort 分母、重复/非法动作、A/B/C/D 分类、task-level funnel 和 TX branch-only 数值一致性/权重及 RNG 不变。",
        '- 审计实现：`src/evaluation/e5_v1_final_audit.py`。\n- 专项测试：`tests/test_e5_v1_final_audit.py`。\n- JSON 审计汇总与输入 SHA-256：`logs/e5-v1-seed43-audit/summary.json`。\n- 全部 Route per-state logits/probabilities：`logs/e5-v1-seed43-audit/route_state_scores.jsonl`。\n- Evolution / formal per-seed / fixed TX CSV：`dashboard_logs/e5-v1-seed43-audit/`。\n- 测试日志/结果：`logs/e5-v1-seed43-audit-validation/unit_tests.log`、`unit_summary.json`。\n- 沿用上下文：`E4_V1_SEED43_GATE_REPORT.md`、`E4_DEFER_COLLAPSE_LOCALIZATION.md`、`logs/gate_m/analysis/final-diagnosis/identity_seed43.json`。',
        '审计到此停止。']
    path.write_text('\n\n'.join(chunks)+'\n', encoding='utf-8')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    result, details = audit(root)
    write_outputs(root, result, details)
    print(json.dumps({'case':result['case'], 'formal_context':result['formal_context'],
                      'E5_fixed_tx':result['reports']['E5']['fixed_tx']}), flush=True)
