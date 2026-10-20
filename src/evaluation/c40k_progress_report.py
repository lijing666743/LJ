"""Rebuild C4k--40k development tables, energy audit and diagnostic plots."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from ..config import load_run_config
from .c40k_progress import ROOT, OLD_AUDIT, read_json, write_json, digest
from .checkpoint_energy_audit import analyze_saved_run
from .group_meeting_figures import load_meeting_data, summarize_episodes, scene_comparison_series, COLORS, LABELS


def records(path):
    return [json.loads(line) for line in Path(path).read_text('utf-8').splitlines()]


def write_csv(path, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_verified_json(path, value):
    """Reusing an audit must reproduce identical values, never replace evidence."""
    if Path(path).exists():
        normalized = json.loads(json.dumps(value))
        assert read_json(path) == normalized, f'Existing audit differs: {path}'
    else:
        write_json(path, value)


def paired_outcomes(old, new):
    """Pair the same generated task within the same frozen scene."""
    index_old = {(r['evaluation_seed'], r['task_id']): r for r in old if r['method_id'] == 'ca_gat_mappo'}
    index_new = {(r['evaluation_seed'], r['task_id']): r for r in new if r['method_id'] == 'ca_gat_mappo'}
    assert index_old.keys() == index_new.keys() and len(index_old) == 605
    result = Counter()
    changed = []
    for key, before in index_old.items():
        after = index_new[key]
        first = before['final_status'] == 'DONE'
        second = after['final_status'] == 'DONE'
        result['both_done' if first and second else 'new_only_done' if second else 'old_only_done' if first else 'neither_done'] += 1
        if first != second:
            changed.append({'seed': key[0], 'task_id': key[1], 'old_status': before['final_status'],
                            'new_status': after['final_status'], 'old_destination': before['destination_uav'],
                            'new_destination': after['destination_uav']})
    result['net_DONE_change'] = result['new_only_done'] - result['old_only_done']
    return dict(result), changed


def figures(data, windows, output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager, pyplot as plt
    import numpy as np
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
                         'font.size': 11, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.3))
    learned = data['C_rows']
    x = [r['checkpoint_steps']/1000 for r in learned]
    ax = axes[0, 0]
    ax.plot(x, [r['completion_pct'] for r in learned], '-o', color=COLORS['C'], label='C')
    for row in data['baseline_rows']:
        ax.axhline(row['completion_pct'], color=COLORS[row['method']], linestyle='--', label=LABELS[row['method']])
    ax.set(xlabel='累计训练步数（k）', ylabel='总完成数 / 总到达数（%）', title='固定开发场景：总体完成率')
    ax.legend(loc='lower right')
    ax = axes[0, 1]
    ax.plot(x, [r['energy_per_DONE_j'] for r in learned], '-o', color=COLORS['C'], label='C')
    for row in data['baseline_rows']:
        ax.axhline(row['energy_per_DONE_j'], color=COLORS[row['method']], linestyle='--', label=LABELS[row['method']])
    ax.set(xlabel='累计训练步数（k）', ylabel='总能耗 / 总完成数（J/任务）', title='固定开发场景：每完成任务能耗')
    ax.legend(loc='center right')
    ax = axes[1, 0]
    means = np.array([w['mean_episode_reward'] for w in windows])
    sd = np.array([w['sample_std_episode_reward'] for w in windows])
    wx = np.array([w['end_environment_steps']/1000 for w in windows])
    ax.plot(wx, means, '-o', color=COLORS['C'], label='8回合非重叠均值')
    ax.fill_between(wx, means-sd, means+sd, color=COLORS['C'], alpha=0.16, label='±1回合样本标准差')
    ax.set(xlabel='窗口末累计步数（k）', ylabel='训练回合回报', title='训练回报：随机场景与随机策略')
    ax.legend(loc='upper left')
    ax = axes[1, 1]
    scene = scene_comparison_series(data, checkpoint_steps=40000)
    position = np.arange(5)
    for offset, method in enumerate(('local_only', 'heuristic', 'C')):
        values = scene['series'][method]['completion_pct']
        bars = ax.bar(position+(offset-1)*0.25, values, width=0.24, color=COLORS[method], label=LABELS[method])
        ax.bar_label(bars, labels=[f'{v:.1f}' for v in values], fontsize=8, padding=2)
    ax.set_xticks(position, [f'{s}\n到达{n}' for s, n in zip(scene['scenes'], scene['arrival_counts'])])
    ax.set(ylabel='逐场景完成率（%）', title='40k与固定基线：逐场景完成率', ylim=(0, 100))
    ax.legend(loc='upper right')
    for ax in axes.flat:
        ax.grid(axis='y', alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle('C续训至40k：单训练seed的开发诊断', fontsize=17)
    fig.text(0.5, 0.015, '训练seed42；开发seeds1042—1046，共605任务。阴影不是置信区间；开发平台期不等于训练收敛。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'C_40k_development_dashboard.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)
    return scene


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='20261014_32k40k')
    parser.add_argument('--output-tag')
    args = parser.parse_args(argv)
    for tag in (args.tag, args.output_tag or args.tag):
        if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in tag):
            parser.error('Use a simple tag')
    audit = ROOT / 'metadata/audits/c40k_progress' / args.tag
    assert read_json(audit / 'validation.json')['status'] == 'PASS'
    report_tag = args.output_tag or args.tag
    csv_root = ROOT / 'dashboard_logs/C/seed42/checkpoint_progress' / report_tag
    plot_root = ROOT / 'plots/C/seed42/checkpoint_progress' / report_tag
    result_root = ROOT / 'docs/results/seed42' / report_tag
    for path in (csv_root, plot_root, result_root):
        assert not path.exists(), f'Use a fresh output-tag: {path}'
        path.mkdir(parents=True)
    data = load_meeting_data(ROOT)
    risk = list(data['energy_risk'])
    new_runs = {}
    new_tasks = {}
    energy_checks = {}
    for step in (36000, 40000):
        run = read_json(audit / f'C_{step//1000}k_run.json')
        new_runs[step] = run
        episode_path = next(p for p in run['evaluation_artifacts'] if p.endswith('episode_metrics.jsonl'))
        episode_rows = records(episode_path)
        learned = [r for r in episode_rows if r['method_id'] == 'ca_gat_mappo']
        data['C_rows'].append({'method': 'C', 'checkpoint_steps': step, **summarize_episodes(learned)})
        for row in episode_rows:
            if row['method_id'] == 'random':
                continue
            m = row['metrics']
            data['per_scene'].append({'method': 'C' if row['method_id'] == 'ca_gat_mappo' else row['method_id'],
                    'checkpoint_steps': step, 'seed': row['evaluation_seed'], 'generated': m['generated'],
                    'completed': m['completed'], 'completion_pct': 100*m['completed']/m['generated']})
        config = load_run_config(audit / f'C_{step//1000}k_evaluation_config.json')
        analysis, _ = analyze_saved_run(config, run)
        summaries = [value['summary'] for value in analysis.values()]
        write_verified_json(audit / f'C_{step//1000}k_energy_per_scene.json',
                   [{'checkpoint_steps': step, 'seed': seed, **value['summary']}
                    for seed, value in analysis.items()])
        counts = ('remote_assignments', 'published_energy_infeasible_remote_assignments',
                  'missing_public_energy_remote_assignments', 'CPU_energy_blocked_UAV_slots',
                  'post_persistent_blockage_assignments')
        risk.append({'checkpoint_steps': step, **{key: sum(s[key] for s in summaries) for key in counts}})
        energy_checks[str(step)] = {str(seed): value['checks'] for seed, value in analysis.items()}
        new_tasks[step] = records(run['service_artifacts'][1])
    old32 = read_json(OLD_AUDIT / 'C_32k_run.json')
    new_tasks[32000] = records(old32['service_artifacts'][1])
    pairs = []
    changed_tasks = []
    for before, after in ((32000, 36000), (36000, 40000), (32000, 40000)):
        comparison, changes = paired_outcomes(new_tasks[before], new_tasks[after])
        pairs.append({'old_checkpoint': before, 'new_checkpoint': after, **comparison})
        changed_tasks.extend({'old_checkpoint': before, 'new_checkpoint': after, **row} for row in changes)
    write_verified_json(audit / 'paired_task_changes.json', pairs)
    write_verified_json(audit / 'energy_reconstruction_checks.json', energy_checks)
    write_verified_json(audit / 'energy_risk.json', risk)
    rows = [*data['baseline_rows'], *data['C_rows']]
    write_csv(csv_root / 'summary_metrics.csv', rows)
    write_csv(csv_root / 'per_scene_metrics.csv', data['per_scene'])
    if changed_tasks:
        write_csv(csv_root / 'paired_changed_tasks.csv', changed_tasks)
    windows = read_json(audit / 'training_reward_windows.json')['windows']
    write_csv(csv_root / 'training_reward_windows.csv', windows)
    last_four = data['C_rows'][-4:]
    span = max(r['completion_pct'] for r in last_four) - min(r['completion_pct'] for r in last_four)
    summary = {'checkpoint_metrics': {str(r['checkpoint_steps']): r for r in data['C_rows']},
               'baselines': data['baseline_rows'], 'last_four_completion_span_pp': span,
               'descriptive_2pp_tolerance_met': span <= 2,
               'training_signal_gate': 'signal-watch', 'scope': 'one training seed, fixed development scenes',
               'paired_changes': pairs, 'new_checkpoint_sources': new_runs}
    write_verified_json(audit / 'comparison_summary.json', summary)
    scene = figures(data, windows, plot_root)
    write_json(plot_root / 'figure_manifest.json', {
        'classification': 'development_diagnostic_only', 'source_csv': str(csv_root / 'summary_metrics.csv'),
        'script': 'src/evaluation/c40k_progress_report.py', 'training_seed': 42,
        'evaluation_seeds': [1042,1043,1044,1045,1046], 'total_arrivals': 605,
        'scene_heights_and_labels': scene, 'training_band': 'episode sample SD, not confidence interval',
        'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in csv_root.glob('*.csv')}})
    table = ['|方法|步数|完成数/到达数|完成率|总能耗(J)|能耗/完成任务(J)|评估回报|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        step = f"{r['checkpoint_steps']//1000}k" if r['checkpoint_steps'] else '固定策略'
        table.append(f"|{r['method']}|{step}|{r['completed']}/{r['generated']}|{r['completion_pct']:.2f}%|{r['total_energy_j']:.3f}|{r['energy_per_DONE_j']:.4f}|{r['mean_episode_reward']:.3f}|")
    final = data['C_rows'][-1]
    previous = next(r for r in data['C_rows'] if r['checkpoint_steps'] == 32000)
    heuristic = next(r for r in data['baseline_rows'] if r['method'] == 'heuristic')
    scene_lines = []
    for seed in (1042, 1043, 1044, 1045, 1046):
        c = next(r for r in data['per_scene'] if r['method'] == 'C' and r['checkpoint_steps'] == 40000 and r['seed'] == seed)
        h = next(r for r in data['per_scene'] if r['method'] == 'heuristic' and r['checkpoint_steps'] == 40000 and r['seed'] == seed)
        scene_lines.append(f"- 场景{seed}：C完成{c['completed']}/{c['generated']}，Heuristic完成{h['completed']}/{h['generated']}，净差{c['completed']-h['completed']:+d}个。")
    report = '\n'.join(['# C续训至40k：开发评估报告', '',
        '2026-10-14。用户已完成训练；本轮仅做训练真实性核验、冻结开发评估与离线分析，没有启动新训练。', '',
        '## 连续性与协议', '',
        '32k重放与原final的actor、critic、两个Adam、策略RNG、诊断和计数完全一致。40k完成80回合、156次PPO更新，优化39936条、末尾64条未优化；36k保存160条pending。', '',
        '固定开发seeds1042—1046、CPU float32、masked_argmax；每点605任务。新增40个评估回合，含10个C回合及30个基线回合；全部基线与原32k动作、指标和外部随机过程一致，网络更新为0。未使用held-out最终测试。', '',
        '## 汇总', '', *table, '', '## 平稳性诊断', '',
        f"40k相对32k完成数变化{final['completed']-previous['completed']:+d}个，完成率变化{final['completion_pct']-previous['completion_pct']:+.3f}个百分点，总能耗变化{final['total_energy_j']-previous['total_energy_j']:+.3f} J。", '',
        f'预先约定的最近四点（28k、32k、36k、40k）完成率跨度为{span:.3f}个百分点；2个百分点的描述性容差' + ('满足。' if span <= 2 else '未满足。'), '',
        '训练回报按每8回合非重叠窗口统计，不将重放回合重复计入。最后三个窗口：' + '；'.join(f"{w['end_environment_steps']//1000}k：{w['mean_episode_reward']:.3f} ± {w['sample_std_episode_reward']:.3f}" for w in windows[-3:]) + '（±为回合样本标准差）。', '',
        '完成率容差只是开发平台期参考，不是统计显著性或收敛证明。需结合训练回报、逐场景表现和能耗；单训练seed不能证明多seed稳定，也不能把C40k与A/B16k作为同预算方法排名。', '',
        '## 与Heuristic的差距', '',
        f"40k比Heuristic多完成{final['completed']-heuristic['completed']}个任务，总能耗增加{final['total_energy_j']-heuristic['total_energy_j']:.3f} J（{100*(final['total_energy_j']/heuristic['total_energy_j']-1):.2f}%），每完成任务能耗增加{100*(final['energy_per_DONE_j']/heuristic['energy_per_DONE_j']-1):.2f}%。不能据此称能效更优。", '',
        *scene_lines, '',
        '## 逐任务与能耗依据', '',
        *[f"- {r['old_checkpoint']//1000}k→{r['new_checkpoint']//1000}k：新增完成{r.get('new_only_done',0)}个，失去完成{r.get('old_only_done',0)}个，净变化{r['net_DONE_change']:+d}个。" for r in pairs], '',
        'CPU/TX服务轨迹已按原能耗重建器逐场景核对；额外能耗来源及能量阻塞统计见审计中的energy_risk.json、energy_reconstruction_checks.json。', '',
        '数据：`dashboard_logs/C/seed42/checkpoint_progress/' + report_tag + '/`；图：`plots/C/seed42/checkpoint_progress/' + report_tag + '/`；审计：`metadata/audits/c40k_progress/' + args.tag + '/`。', '',
        '复现评估：`python -m src.evaluation.c40k_progress --tag NEW_TAG`；从已有评估重新出图：`python -m src.evaluation.c40k_progress_report --tag ' + args.tag + ' --output-tag NEW_OUTPUT_TAG`。图为开发诊断，未升级为正式论文证据。', ''])
    (result_root / 'C40K_DEVELOPMENT_REPORT.md').write_text(report, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
