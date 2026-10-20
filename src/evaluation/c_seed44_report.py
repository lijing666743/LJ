"""Audit and explain three training seeds using the fixed 40k endpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from ..config import load_run_config
from .c40k_progress import ROOT, SEEDS, ARRIVALS, digest, read_json, write_json
from .c_seed44_progress import STEPS, audit_directory
from .checkpoint_energy_audit import analyze_saved_run
from .group_meeting_figures import summarize_episodes

PREVIOUS_TAG = '20261015_seed43_4k40k'


def endpoint_statistics(rows, endpoint=40000):
    """Count training runs, not scenes; never select each seed's best point."""
    selected = [r for r in rows if r['checkpoint_steps'] == endpoint]
    if len(selected) != 3 or {r['training_seed'] for r in selected} != {42, 43, 44}:
        raise ValueError('Require exactly seeds42/43/44 at the fixed endpoint')
    if any(r['generated'] != 605 for r in selected):
        raise ValueError('All endpoints must use the same605 generated tasks')
    result = {'endpoint': endpoint, 'independent_training_runs': 3,
              'evaluation_scenes_per_run': 5}
    for key in ('completion_pct', 'total_energy_j', 'energy_per_DONE_j', 'mean_episode_reward'):
        values = [r[key] for r in selected]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError(f'Nonfinite endpoint metric: {key}')
        result[key] = {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values),
                       'minimum': min(values), 'maximum': max(values)}
    return result


def load_previous():
    """Read hash-verified numeric evidence without rewriting historical reports."""
    manifest = read_json(ROOT / f'metadata/audits/c_seed43_progress/{PREVIOUS_TAG}/artifact_manifest.json')
    expected = {r['path']: r['sha256'] for r in manifest}
    csv_root = ROOT / f'dashboard_logs/C/multiseed/{PREVIOUS_TAG}'
    paths = [csv_root/'checkpoint_metrics_seed42_seed43.csv',
             csv_root/'per_scene_metrics_seed42_seed43.csv',
             ROOT/f'docs/results/multiseed/{PREVIOUS_TAG}/comparison_summary.json']
    for path in paths:
        assert digest(path) == expected[path.relative_to(ROOT).as_posix()]
    integer_keys = {'training_seed', 'checkpoint_steps', 'generated', 'completed', 'expired', 'truncated', 'seed'}
    loaded = []
    for path in paths[:2]:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            loaded.append([{k: (int(v) if k in integer_keys else v if k == 'method' else float(v))
                            for k, v in row.items()} for row in csv.DictReader(stream)])
    return *loaded, read_json(paths[2])['baselines'], {str(p.relative_to(ROOT)): digest(p) for p in paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='20261015_seed44_4k40k')
    parser.add_argument('--output-tag')
    args = parser.parse_args(argv)
    audit = audit_directory(args.tag)
    assert read_json(audit/'validation.json')['status'] == 'PASS'
    output_tag = args.output_tag or args.tag
    audit_directory(output_tag)
    target = ROOT/'docs/results/multiseed'/output_tag
    if target.exists():
        raise FileExistsError('Use a new output tag; existing reports are immutable')
    rows, scenes, baselines, previous_hashes = load_previous()
    for step in STEPS:
        receipt = read_json(audit/f'C44_{step//1000}k_run.json')
        assert receipt['training_seed'] == 44 and receipt['checkpoint_steps'] == step
        assert receipt['network_updates'] == receipt['training_calls'] == 0
        records_path = next(p for p in receipt['evaluation_artifacts'] if p.endswith('episode_metrics.jsonl'))
        records = [json.loads(line) for line in Path(records_path).read_text('utf-8').splitlines()]
        learned = [r for r in records if r['method_id'] == 'ca_gat_mappo']
        assert {r['evaluation_seed'] for r in learned} == set(SEEDS) and len(learned) == 5
        pooled = summarize_episodes(learned)
        assert pooled == receipt['pooled']
        rows.append({'training_seed': 44, 'method': 'C', 'checkpoint_steps': step, **pooled})
        for r in learned:
            m = r['metrics']
            scenes.append({'training_seed': 44, 'checkpoint_steps': step, 'seed': r['evaluation_seed'],
                           'generated': m['generated'], 'completed': m['completed'],
                           'completion_pct': 100*m['completed']/m['generated'],
                           'total_energy_j': m['total_energy_j'], 'cpu_energy_j': m['cpu_energy_j'],
                           'tx_energy_j': m['tx_energy_j'], 'mean_episode_reward': m['total_reward']})
    assert len(rows) == 30 and len(scenes) == 150
    assert all(r['generated'] == ARRIVALS[r['seed']] for r in scenes)
    for row in rows:
        selected = [s for s in scenes if s['training_seed'] == row['training_seed']
                    and s['checkpoint_steps'] == row['checkpoint_steps']]
        assert len(selected) == 5 and row['generated'] == 605
        assert sum(s['completed'] for s in selected) == row['completed']
        assert math.isclose(sum(s['total_energy_j'] for s in selected), row['total_energy_j'], abs_tol=1e-8)
        assert math.isclose(row['completion_pct'], 100*row['completed']/row['generated'], abs_tol=1e-10)
    stats = endpoint_statistics(rows)
    final = {s: next(r for r in rows if r['training_seed'] == s and r['checkpoint_steps'] == 40000)
             for s in (42, 43, 44)}
    heuristic = next(r for r in baselines if r['method'] == 'heuristic')
    local = next(r for r in baselines if r['method'] == 'local_only')
    spans = {s: max(r['completion_pct'] for r in rows if r['training_seed'] == s and r['checkpoint_steps'] >= 28000)
             - min(r['completion_pct'] for r in rows if r['training_seed'] == s and r['checkpoint_steps'] >= 28000)
             for s in (42, 43, 44)}
    receipt = read_json(audit/'C44_40k_run.json')
    energy, _ = analyze_saved_run(load_run_config(audit/'C44_40k_evaluation_config.json'), receipt)
    energy_summaries = [{'seed': s, **v['summary']} for s, v in energy.items()]
    assert math.isclose(sum(r['total_energy_j'] for r in energy_summaries), final[44]['total_energy_j'], abs_tol=1e-8)
    windows = read_json(audit/'training_reward_windows.json')['windows']
    target.mkdir(parents=True)
    summary = {'fixed_primary_endpoint': 40000, 'training_seeds': [42, 43, 44],
               'checkpoint_metrics': rows, 'per_scene_metrics': scenes,
               'baselines': baselines, 'endpoint_statistics': stats,
               'late_four_point_span_pp': spans, 'seed44_final_training_window': windows[-1],
               'seed44_energy_reconstruction': energy_summaries, 'previous_source_sha256': previous_hashes,
               'claim_boundary': 'Three training runs; fixed auxiliary shadow RNG; five development scenes; no held-out claim'}
    write_json(target/'comparison_summary.json', summary)
    lines = ['# C seed44评估与三训练种子对照', '',
             '2026-10-15。本轮只做冻结评估及离线分析，没有训练或优化器更新。按用户偏好，采用文字说明，不新增汇总表或图。', '',
             '## 协议与训练完整性', '',
             'seed44十个checkpoint通过完整性核验；final为40000步、80回合、156次PPO更新、39936条优化样本、64条unused tail。回合种子序列匹配seed44独立训练，没有continuation来源。',
             '每个模型使用1042—1046相同五个开发场景，共605个到达任务；CPU float32、masked_argmax、网络冻结。新增200个评估回合中，150个固定基线回合逐项匹配历史记录；全部外部随机过程匹配。未使用held-out最终测试。', '',
             '## 预先固定的40k结果', '']
    for label, r in [('Local', local), ('Heuristic', heuristic), *[(f'C seed{s}', final[s]) for s in (42, 43, 44)]]:
        lines.append(f"- {label}：完成{r['completed']}/605（{r['completion_pct']:.2f}%），总能耗{r['total_energy_j']:.3f} J，每完成任务能耗{r['energy_per_DONE_j']:.4f} J，评估回合回报均值{r['mean_episode_reward']:.3f}。")
    f = final[44]
    lines += ['', f"seed44相对seed42完成数变化{f['completed']-final[42]['completed']:+d}，相对seed43变化{f['completed']-final[43]['completed']:+d}；相对Heuristic变化{f['completed']-heuristic['completed']:+d}（{f['completion_pct']-heuristic['completion_pct']:+.2f}个百分点）。",
              f"相对Heuristic，总能耗变化{100*(f['total_energy_j']/heuristic['total_energy_j']-1):+.2f}%，每完成任务能耗变化{100*(f['energy_per_DONE_j']/heuristic['energy_per_DONE_j']-1):+.2f}%。应分别判断任务收益和能耗代价。", '',
              '## 逐场景与学习过程', '']
    for s in SEEDS:
        values = [next(r for r in scenes if r['seed'] == s and r['training_seed'] == t and r['checkpoint_steps'] == 40000)
                  for t in (42, 43, 44)]
        lines.append(f"- 场景{s}，到达{ARRIVALS[s]}：seed42/43/44分别完成{values[0]['completed']}、{values[1]['completed']}、{values[2]['completed']}个。")
    lines += ['', 'seed44全部checkpoint的完成率依次为：' + '；'.join(
              f"{r['checkpoint_steps']//1000}k：{r['completion_pct']:.2f}%" for r in rows if r['training_seed'] == 44) + '。',
              '最近四点指28k、32k、36k、40k；seed42/43/44完成率跨度依次为' +
              '、'.join(f'{spans[s]:.3f}' for s in (42, 43, 44)) + '个百分点。跨度是固定开发场景下的诊断，不是全局收敛证明。',
              f"seed44最后8回合训练回报为{windows[-1]['mean_episode_reward']:.3f}±{windows[-1]['sample_std_episode_reward']:.3f}（回合样本标准差，不是置信区间）。控制台reward_mean是每步全程均值，不能直接当作最终开发性能。", '',
              '## 跨训练种子解释与下一步', '',
              f"固定40k三次训练的完成率均值{stats['completion_pct']['mean']:.3f}%，样本标准差{stats['completion_pct']['sample_std']:.3f}个百分点；范围{stats['completion_pct']['minimum']:.2f}%—{stats['completion_pct']['maximum']:.2f}%。独立训练数为3，五个场景不扩大为15次训练。辅助shadow RNG固定，因此是改变master seed的重复性检验。",
              '保留全部训练种子，主比较统一40k，不挑各seed中途最佳点替代final。是否进入平台应结合各seed学习曲线与完成率，不仅看训练reward。完成率优势与能耗优势分开陈述。',
              '先判断三次训练是否一致超过Heuristic及波动幅度，再决定限定诊断或同预算学习基线/消融比较。A/B现有16k结果保留，不能与C40k直接证明方法优劣；本轮不自动延长训练、修改reward或使用held-out测试。', '',
              f'数值证据：`docs/results/multiseed/{output_tag}/comparison_summary.json`。审计：`metadata/audits/c_seed44_progress/{args.tag}/`。',
              f'离线重建：`python -m src.evaluation.c_seed44_report --tag {args.tag} --output-tag NEW_OUTPUT_TAG`。', '']
    (target/'C_SEED44_DEVELOPMENT_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'seed44': f, 'statistics': stats, 'late_spans': spans, 'report': str(target/'C_SEED44_DEVELOPMENT_REPORT.md')}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
