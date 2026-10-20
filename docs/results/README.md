# 结果报告索引

先看[当前实验状态](CURRENT_EXPERIMENT_STATUS.md)。报告保留原址，原始指标、模型和审计仍沿用原目录。

## 当前强基线

- [D seed44最终训练核验](../../metadata/audits/qmix_c_matched/20261021/seed44_40k_training_check.json)与[十检查点评估及三seed汇总](../../metadata/audits/qmix_c_matched/20261021/seed44_40k_evaluation_check.json)：三seed固定40k为425/419/409个任务。

- [D（CFC-QMIX）：4k—40k](../../sections/results/20261018_C_FEATURE_CREDIT_QMIX.md)
- [D 的 C 匹配协议](../../sections/planning/20261018_C_MATCHED_QMIX_PROTOCOL.md)
- [16k 训练核验](../../metadata/audits/qmix_c_matched/20261018/seed42_16k_training_check.json)与[评估核验](../../metadata/audits/qmix_c_matched/20261018/seed42_16k_evaluation_check.json)
- [24k 训练核验](../../metadata/audits/qmix_c_matched/20261019/seed42_24k_training_check.json)与[评估核验](../../metadata/audits/qmix_c_matched/20261019/seed42_24k_evaluation_check.json)
- [32k训练完成核验](../../metadata/audits/qmix_c_matched/20261019/seed42_32k_training_completion_check.json)与[评估核验](../../metadata/audits/qmix_c_matched/20261019/seed42_32k_evaluation_check.json)
- [40k训练核验](../../metadata/audits/qmix_c_matched/20261019/seed42_40k_training_check.json)与[评估核验](../../metadata/audits/qmix_c_matched/20261019/seed42_40k_evaluation_check.json)
- [D seed43独立8k训练核验](../../metadata/audits/qmix_c_matched/20261019/seed43_8k_training_check.json)与[4k/8k评估核验](../../metadata/audits/qmix_c_matched/20261019/seed43_8k_evaluation_check.json)
- [D seed43最终40k训练核验](../../metadata/audits/qmix_c_matched/20261020/seed43_40k_training_check.json)与[十检查点评估核验](../../metadata/audits/qmix_c_matched/20261020/seed43_40k_evaluation_check.json)
- [seed43固定40k比较Excel](../../dashboard_logs/strong_baselines/comparisons/seed43_40k/D_C_seed43_40k_comparison.xlsx)：十点进展、同预算C/D与固定基线、五场景完成数。
- [固定40k汇总CSV](../../dashboard_logs/strong_baselines/comparisons/seed43_40k/fixed_40k_comparison.csv)、[十检查点CSV](../../dashboard_logs/strong_baselines/comparisons/seed43_40k/seed43_checkpoint_progress.csv)与[五场景CSV](../../dashboard_logs/strong_baselines/comparisons/seed43_40k/fixed_40k_scenes.csv)
- [D seed43 8k冻结路由行为诊断](../../metadata/audits/qmix_c_matched/20261019/seed43_8k_route_behavior_check.json)：实际分配、任务结局和传输阶段；尚未测定同状态路由正确率。

## C 历史结果

| 报告 | 范围 |
| --- | --- |
| [C seed42 40k](seed42/20261014_32k40k/C40K_DEVELOPMENT_REPORT.md) | seed42 检查点与能耗 |
| [C seed43 40k](multiseed/20261015_seed43_4k40k/C_SEED43_DEVELOPMENT_REPORT.md) | 独立训练 seed43 |
| [C seed44 40k](multiseed/20261015_seed44_4k40k/C_SEED44_DEVELOPMENT_REPORT.md) | 独立训练 seed44、多 seed 波动 |
| [seed44 限定诊断](diagnostics/20261015_seed44_limited/SEED44_LIMITED_DIAGNOSTIC_REPORT.md) | 路由、Defer、CPU 与训练信号 |
| [C seed42原始8k摘要](seed42/C_SEED42_8K_HISTORICAL_SUMMARY.md) | 保留C原始记录及汇总口径；旧A/B比较已退役 |

## 研究过程与历史

旧 QMIX 诊断和实现报告见 [sections/results/](../../sections/results/)，候选协议见 [sections/planning/](../../sections/planning/)。旧版本只供追溯，不是当前训练入口。

已有研究总计划在 [docs/plans/](../plans/)。论文正文从 [sections 索引](../../sections/README.md)进入；原始来源见 [logs](../../logs/README.md)与 [metadata](../../metadata/README.md)。
