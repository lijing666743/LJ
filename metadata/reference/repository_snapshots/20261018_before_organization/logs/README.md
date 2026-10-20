# A/B/C 实验日志

本目录只保留当前A/B/C实际产物。每组目录结构相同：

```text
A/（B、C同样）
└─ seed42/
   └─ 8k/
      ├─ training/       原始训练指标、学习链诊断、配置快照、checkpoints
      ├─ evaluation/     final模型的1042—1046评估指标与身份
      └─ diagnostics/    对应评估的Route逐决策trace与汇总
```

| 位置 | 内容 |
|---|---|
| `training/checkpoints/final.pt` | 当前预算终点模型，用于冻结策略评估 |
| `training/checkpoints/step_4000.pt` | 原训练中期恢复点；跨源码身份的exact-resume另有一致性要求 |
| `training/raw_metrics.jsonl` | 全8000步原始指标，体积较大但保留供核对 |
| `training/route_*` 与 `integrated_scheduler_updates_v1.jsonl` | 同一次训练的Route、信用、Q及调度诊断，不是其他方法 |
| `evaluation/aggregate_metrics.json` | 评估汇总 |
| `evaluation/episode_metrics.jsonl`、`evaluation_metrics.csv` | 逐episode评估指标 |
| `evaluation/evaluation_manifest.json` | 模型、协议、seed及来源身份 |
| `diagnostics/route_decision_trace.jsonl`、`summary.json` | 同一组评估的Route行为 |

训练CSV对应 `../dashboard_logs/{A,B,C}/seed42/8k/training_metrics.csv`，图对应 `../plots/{A,B,C}/seed42/8k/training_dashboard.png`。

汇总比较见 [seed42报告](../docs/results/seed42/seed42_abc_comparison_report.md)。历史对齐与工程记录在 [metadata](../metadata/README.md)，不混入三组日志。

目录名缩短，文件内容和实验身份保留。旧路径到当前位置见 [路径对照表](../metadata/DIRECTORY_RELOCATION.json)。后续实验进入 `组别/seed/8k/runs/时间标签/`；命令见 [后续实验说明](../docs/reproduction/REPRODUCE_ABC.md#后续实验怎样放置)。不要直接覆盖当前结果。

## 2026-10-14 补充

上表描述原8k层级。现在A/B/C各有`seed42/16k_continuation/`，C另有`seed42/32k_continuation/`；checkpoint_progress保存各阶段开发评估。训练日志可能超过1GB，属于真实复核依据，未因体积大而删除。

C全部中期模型与final、两次continuation来源、配置快照和侧日志继续保留，供后续完整续训使用。28k periodic有96条pending；32k final虽无unused tail，仍按当前规则使用periodic恢复与final核对。`engineering/continuation_tests_tmp/`旧测试输出已清理，真实训练和开发评估均保留。`group_meeting_summary/`仅保存图表统计测试记录。当前结果导航见[最新状态](../docs/results/CURRENT_EXPERIMENT_STATUS.md)。
