"""Reproduce seed42/43 checkpoint comparisons without selecting best seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

from ..config import load_run_config
from .c40k_progress import ROOT, SEEDS, ARRIVALS, read_json, write_json, digest
from .c_seed43_progress import STEPS, audit_directory
from .checkpoint_energy_audit import analyze_saved_run
from .group_meeting_figures import COLORS, LABELS, load_meeting_data

COLOR43 = '#8B5AA3'
OLD_RESULT = ROOT / 'metadata/audits/c40k_progress/20261014_32k40k/comparison_summary.json'
OLD_SCENES = ROOT / 'dashboard_logs/C/seed42/checkpoint_progress/20261014_32k40k/per_scene_metrics.csv'


def endpoint_statistics(rows, endpoint=40000):
    """Training seeds are the independent units; five scenes are not five runs."""
    selected = [row for row in rows if row['checkpoint_steps'] == endpoint]
    seeds = [row['training_seed'] for row in selected]
    if set(seeds) != {42,43} or len(seeds) != 2:
        raise ValueError('Expected exactly seed42 and seed43 at the fixed endpoint')
    result = {'endpoint': endpoint, 'independent_training_runs': 2, 'evaluation_scenes_per_run': 5}
    for name in ('completion_pct','total_energy_j','energy_per_DONE_j','mean_episode_reward'):
        values = [row[name] for row in selected]
        result[name] = {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values),
                        'minimum': min(values), 'maximum': max(values)}
    return result


def write_csv(path, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_comparison(tag):
    audit = audit_directory(tag)
    assert read_json(audit/'validation.json')['status'] == 'PASS'
    original_manifest=read_json(ROOT/'metadata/audits/c40k_progress/20261014_32k40k/artifact_manifest.json')
    expected={r['path']:r['sha256'] for r in original_manifest}
    for path in (OLD_RESULT,OLD_SCENES,ROOT/'metadata/audits/c40k_progress/20261014_32k40k/training_reward_windows.json'):
        assert digest(path)==expected[path.relative_to(ROOT).as_posix()], 'Saved seed42 source changed'
    old = read_json(OLD_RESULT)
    reference_sources={r['step']:r['episode_metrics'] for r in load_meeting_data(ROOT)['source_runs']}
    for step in (36000,40000):
        receipt=read_json(ROOT/f'metadata/audits/c40k_progress/20261014_32k40k/C_{step//1000}k_run.json')
        reference_sources[step]=next(p for p in receipt['evaluation_artifacts'] if p.endswith('episode_metrics.jsonl'))
    reference_records={step:[json.loads(line) for line in Path(path).read_text('utf-8').splitlines()
                            if json.loads(line)['method_id']=='ca_gat_mappo']
                       for step,path in reference_sources.items()}
    rows = []
    scenes = []
    for step in STEPS:
        rows.append({'training_seed': 42, **old['checkpoint_metrics'][str(step)]})
        receipt = read_json(audit/f'C43_{step//1000}k_run.json')
        rows.append({'training_seed': 43, 'method': 'C', 'checkpoint_steps': step, **receipt['pooled']})
        path = next(p for p in receipt['evaluation_artifacts'] if p.endswith('episode_metrics.jsonl'))
        records = [json.loads(line) for line in Path(path).read_text('utf-8').splitlines()]
        for record in records:
            if record['method_id'] != 'ca_gat_mappo':
                continue
            m = record['metrics']
            scenes.append({'training_seed': 43, 'checkpoint_steps': step, 'seed': record['evaluation_seed'],
                           'generated': m['generated'], 'completed': m['completed'],
                           'completion_pct': 100*m['completed']/m['generated'],
                           'total_energy_j': m['total_energy_j'], 'cpu_energy_j':m['cpu_energy_j'],
                           'tx_energy_j':m['tx_energy_j'], 'mean_episode_reward': m['total_reward']})
    with OLD_SCENES.open(encoding='utf-8-sig',newline='') as stream:
        for r in csv.DictReader(stream):
            if r['method'] == 'C':
                record=next(v for v in reference_records[int(r['checkpoint_steps'])]
                            if v['evaluation_seed']==int(r['seed']))
                metric=record['metrics']
                assert metric['generated']==int(r['generated']) and metric['completed']==int(r['completed'])
                scenes.append({'training_seed': 42, 'checkpoint_steps': int(r['checkpoint_steps']),
                               'seed': int(r['seed']), 'generated': int(r['generated']),
                               'completed': int(r['completed']), 'completion_pct': float(r['completion_pct']),
                               'total_energy_j':metric['total_energy_j'],'cpu_energy_j':metric['cpu_energy_j'],
                               'tx_energy_j':metric['tx_energy_j'],'mean_episode_reward':metric['total_reward']})
    assert len(rows) == 20 and len(scenes) == 100
    assert all(r['generated']==605 for r in rows)
    assert all(r['generated']==ARRIVALS[r['seed']] for r in scenes)
    for row in rows:
        selected=[s for s in scenes if s['training_seed']==row['training_seed'] and s['checkpoint_steps']==row['checkpoint_steps']]
        assert sum(s['completed'] for s in selected)==row['completed']
        assert abs(sum(s['total_energy_j'] for s in selected)-row['total_energy_j'])<1e-8
    return rows, scenes, old['baselines']


def figures(rows, scenes, baselines, windows, target):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager, pyplot as plt
    import numpy as np
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,'svg.fonttype':'none'})
    colors = {42: COLORS['C'], 43: COLOR43}
    fig, axes = plt.subplots(2,2,figsize=(14.5,9.5))
    for seed in (42,43):
        values = [r for r in rows if r['training_seed']==seed]
        x=[r['checkpoint_steps']/1000 for r in values]
        axes[0,0].plot(x,[r['completion_pct'] for r in values],'-o',color=colors[seed],label=f'C seed{seed}')
        axes[0,1].plot(x,[r['energy_per_DONE_j'] for r in values],'-o',color=colors[seed],label=f'C seed{seed}')
        wx=[r['end_environment_steps']/1000 for r in windows[seed]]
        mean=np.array([r['mean_episode_reward'] for r in windows[seed]])
        sd=np.array([r['sample_std_episode_reward'] for r in windows[seed]])
        axes[1,0].plot(wx,mean,'-o',color=colors[seed],label=f'seed{seed}：8回合均值')
        axes[1,0].fill_between(wx,mean-sd,mean+sd,color=colors[seed],alpha=.12)
    for baseline in baselines:
        for ax,metric in ((axes[0,0],'completion_pct'),(axes[0,1],'energy_per_DONE_j')):
            ax.axhline(baseline[metric],color=COLORS[baseline['method']],linestyle='--',label=LABELS[baseline['method']])
    axes[0,0].set(title='同一开发协议：总体完成率',xlabel='累计训练步数（k）',ylabel='总完成数 / 总到达数（%）')
    axes[0,1].set(title='同一开发协议：每完成任务能耗',xlabel='累计训练步数（k）',ylabel='总能耗 / 总完成数（J/任务）')
    axes[1,0].set(title='训练回报：均值与回合波动',xlabel='8回合窗口末步数（k）',ylabel='训练回合回报')
    axes[0,0].legend(loc='lower right',fontsize=9,ncol=2)
    axes[0,1].legend(loc='best',fontsize=9,ncol=2)
    axes[1,0].legend(loc='upper left',fontsize=9)
    ax=axes[1,1]
    position=np.arange(5)
    with OLD_SCENES.open(encoding='utf-8-sig',newline='') as stream: previous=list(csv.DictReader(stream))
    chart=[]
    for method,seed,color,label in (('local_only',None,COLORS['local_only'],'纯本地'),
                                   ('heuristic',None,COLORS['heuristic'],'Heuristic'),
                                   ('C',42,colors[42],'C seed42'),('C',43,colors[43],'C seed43')):
        values=[]
        for scene in SEEDS:
            if seed is None:
                r=next(r for r in previous if r['method']==method and int(r['checkpoint_steps'])==40000 and int(r['seed'])==scene)
            else:
                r=next(r for r in scenes if r['training_seed']==seed and r['checkpoint_steps']==40000 and r['seed']==scene)
            value=100*int(r['completed'])/int(r['generated'])
            assert abs(value-float(r['completion_pct']))<1e-8
            values.append(value)
        chart.append({'label':label,'completion_pct':values})
        offset=len(chart)-1
        bars=ax.bar(position+(offset-1.5)*.20,values,width=.19,color=color,label=label)
        ax.bar_label(bars,labels=[f'{v:.1f}' for v in values],padding=2,fontsize=7)
    ax.set_xticks(position,[f'{s}\n到达{ARRIVALS[s]}' for s in SEEDS])
    ax.set(title='预先固定40k：逐场景完成率',ylabel='完成率（%）',ylim=(0,105))
    ax.legend(loc='best',ncol=2,fontsize=9)
    for ax in axes.flat:
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('C独立训练seed42 / seed43：4k—40k开发评估',fontsize=17)
    fig.text(.5,.015,'每个模型评估相同5个开发场景、605任务；仅2次独立训练。训练阴影为±1回合样本标准差，不是置信区间。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.95))
    for suffix in ('png','svg'):
        fig.savefig(target/f'C_seed42_seed43_4k40k.{suffix}',dpi=180,facecolor='white')
    plt.close(fig)
    return chart


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',default='20261015_seed43_4k40k')
    parser.add_argument('--output-tag')
    args=parser.parse_args(argv)
    audit=audit_directory(args.tag)
    output_tag=args.output_tag or args.tag
    audit_directory(output_tag)
    csv_root=ROOT/'dashboard_logs/C/multiseed'/output_tag
    plot_root=ROOT/'plots/C/multiseed'/output_tag
    doc_root=ROOT/'docs/results/multiseed'/output_tag
    for path in (csv_root,plot_root,doc_root):
        if path.exists():raise FileExistsError('Use a new output tag')
        path.mkdir(parents=True)
    rows,scenes,baselines=load_comparison(args.tag)
    stats=endpoint_statistics(rows)
    write_csv(csv_root/'checkpoint_metrics_seed42_seed43.csv',rows)
    write_csv(csv_root/'per_scene_metrics_seed42_seed43.csv',scenes)
    windows={42:read_json(ROOT/'metadata/audits/c40k_progress/20261014_32k40k/training_reward_windows.json')['windows'],
             43:read_json(audit/'training_reward_windows.json')['windows']}
    write_csv(csv_root/'training_windows_seed42_seed43.csv',
              [{'training_seed':seed,**r} for seed in (42,43) for r in windows[seed]])
    chart=figures(rows,scenes,baselines,windows,plot_root)
    write_json(plot_root/'figure_manifest.json',{'purpose':'two-training-seed development diagnostic',
               'training_seeds':[42,43],'evaluation_seeds':list(SEEDS),'denominator_per_run':605,
               'primary_endpoint':40000,'training_band':'episode sample SD; no inference across seeds',
               'final_scene_heights_and_labels':chart,'source_csv_hashes':{p.name:digest(p) for p in csv_root.glob('*.csv')},
               'regeneration_script':'src/evaluation/c_seed43_report.py'})
    final42=next(r for r in rows if r['training_seed']==42 and r['checkpoint_steps']==40000)
    final43=next(r for r in rows if r['training_seed']==43 and r['checkpoint_steps']==40000)
    heuristic=next(r for r in baselines if r['method']=='heuristic')
    local=next(r for r in baselines if r['method']=='local_only')
    learned43=[r for r in rows if r['training_seed']==43]
    best=max(learned43,key=lambda r:r['completion_pct'])
    latest=[r for r in learned43 if r['checkpoint_steps']>=28000]
    span=max(r['completion_pct'] for r in latest)-min(r['completion_pct'] for r in latest)
    receipt=read_json(audit/'C43_40k_run.json')
    energy,_=analyze_saved_run(load_run_config(audit/'C43_40k_evaluation_config.json'),receipt)
    energy_summaries=[{'seed':seed,**value['summary']} for seed,value in energy.items()]
    write_json(doc_root/'C43_40k_energy_per_scene.json',energy_summaries)
    report_summary={'primary_endpoint':40000,'seed42':final42,'seed43':final43,
                    'C43_minus_C42_completion_pp':final43['completion_pct']-final42['completion_pct'],
                    'C43_minus_heuristic_completed':final43['completed']-heuristic['completed'],
                    'two_seed_statistics':stats,'C43_late4_completion_span_pp':span,
                    'best_C43_checkpoint_diagnostic_only':best,'baselines':baselines,
                    'claim':'two independent training seeds, same development scenes; no final generalization claim'}
    write_json(doc_root/'comparison_summary.json',report_summary)
    lines=['# C seed43独立训练：4k—40k开发评估与seed42对照','','2026-10-15。用户已完成seed43训练；本轮只做冻结开发评估和离线分析，没有训练或优化器更新。','',
           '## 协议与完整性','','十个checkpoint均通过校验；40k完成80回合、156次PPO更新，优化39936条，unused tail为64。训练来源独立，无seed42 continuation。',
           '开发seeds1042—1046、CPU float32、masked_argmax；每模型605任务。新增200回合中C为50回合、固定基线为150回合；各点固定基线动作/指标和外部随机过程与seed42记录完全一致。未使用held-out最终测试。','',
           '## 同预算学习过程','','|步数|seed42完成率|seed43完成率|43−42（百分点）|seed42能耗(J)|seed43能耗(J)|','|---|---:|---:|---:|---:|---:|']
    for step in STEPS:
        a=next(r for r in rows if r['training_seed']==42 and r['checkpoint_steps']==step)
        b=next(r for r in rows if r['training_seed']==43 and r['checkpoint_steps']==step)
        lines.append(f"|{step//1000}k|{a['completion_pct']:.2f}%|{b['completion_pct']:.2f}%|{b['completion_pct']-a['completion_pct']:+.2f}|{a['total_energy_j']:.3f}|{b['total_energy_j']:.3f}|")
    lines+=['','## 预先固定的40k结果','','|模型/策略|完成数/到达数|完成率|总能耗(J)|每完成任务能耗(J)|评估回报|','|---|---:|---:|---:|---:|---:|']
    for label,r in [('Local',local),('Heuristic',heuristic),('C seed42',final42),('C seed43',final43)]:
        lines.append(f"|{label}|{r['completed']}/605|{r['completion_pct']:.2f}%|{r['total_energy_j']:.3f}|{r['energy_per_DONE_j']:.4f}|{r['mean_episode_reward']:.3f}|")
    lines+=['',f"seed43相对seed42完成数变化{final43['completed']-final42['completed']:+d}，完成率变化{final43['completion_pct']-final42['completion_pct']:+.3f}个百分点。seed43相对Heuristic完成数变化{final43['completed']-heuristic['completed']:+d}。",'',
            '## 40k逐场景完成数','','|开发seed|到达数|C seed42|C seed43|43−42|','|---|---:|---:|---:|---:|']
    for scene in SEEDS:
        a=next(r for r in scenes if r['training_seed']==42 and r['checkpoint_steps']==40000 and r['seed']==scene)
        b=next(r for r in scenes if r['training_seed']==43 and r['checkpoint_steps']==40000 and r['seed']==scene)
        lines.append(f"|{scene}|{a['generated']}|{a['completed']}|{b['completed']}|{b['completed']-a['completed']:+d}|")
    lines+=['',
            '## 跨训练seed的初步统计','',
            f"固定40k的完成率均值为{stats['completion_pct']['mean']:.3f}%，样本标准差为{stats['completion_pct']['sample_std']:.3f}个百分点。独立训练数为2，不是10；五个场景是每个模型的评估样本，不能作为五次独立训练。",'',
            '只有两个训练seed，均值和标准差只是描述性汇总，不据此报告可靠的总体置信区间或宣称多seed稳健。所有seed均保留，不剔除较差seed。','',
            '## 稳定性与选择规则','',
            f"seed43最近四点（28k、32k、36k、40k）完成率跨度{span:.3f}个百分点。中期最高开发完成率为{best['completion_pct']:.2f}%（{best['checkpoint_steps']//1000}k），仅用于解释学习过程；主报告仍采用预先固定的40k，不用中期最佳替换。",'',
            'seed42在固定场景上的平台期不能自动推广为不同训练seed都表现相近。训练回报受不同训练任务和策略采样影响，直接比较训练回报不等于统一场景性能比较。必须结合这里的同协议评估解释。','',
            '## 后续顺序','',
            '先根据40k跨seed差距判断是否存在明显训练敏感性，再决定补seed44或做限定范围的诊断。诊断/修改应与当前冻结版本分开，不因seed43不理想就丢弃它或立即改reward；若技术路线变更，先记录并确认。A/B仍只有seed42的16k正式结果，不与C40k直接作同预算排名。','',
            f'审计：`metadata/audits/c_seed43_progress/{args.tag}/`；CSV：`dashboard_logs/C/multiseed/{output_tag}/`；图：`plots/C/multiseed/{output_tag}/`。',
            f'复现：`python -m src.evaluation.c_seed43_progress --tag NEW_TAG --workers 2`；从已有评估重出表图：`python -m src.evaluation.c_seed43_report --tag {args.tag} --output-tag NEW_OUTPUT_TAG`。','']
    (doc_root/'C_SEED43_DEVELOPMENT_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'primary_completion42':final42['completion_pct'],'primary_completion43':final43['completion_pct'],
                     'difference_pp':report_summary['C43_minus_C42_completion_pp'],'statistics':stats,
                     'report':str(doc_root/'C_SEED43_DEVELOPMENT_REPORT.md')},ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
