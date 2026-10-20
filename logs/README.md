# 实验日志与模型索引

当前结论见[实验状态](../docs/results/CURRENT_EXPERIMENT_STATUS.md)，报告见[结果索引](../docs/results/README.md)。本轮保留C/D真实模型、日志、评估和来源路径。

A/B独立日志与模型已按用户范围调整删除，不再作为活动实验。C/D全部来源模型保留，退役身份、文件哈希及最小真实历史指标见[回执](../metadata/cleanup/20261020_retire_ab/receipt.json)；不能将退役解释为算法失败。

| 位置 | 内容 |
| --- | --- |
| `C/seed42/` | 原 8k、16k/32k/40k 连续训练、评估与诊断 |
| `C/seed43/`、`C/seed44/` | 独立 40k 与每 4k 开发评估 |
| `strong_baselines/development/c_feature_credit_qmix/seed42/8k/` | D（CFC-QMIX）原 8k 和 4k/8k 评估 |
| `strong_baselines/development/c_feature_credit_qmix/seed42/16k/` | 8k→16k 完整续训、新增 12k/16k 评估及四检查点系列 |
| `strong_baselines/development/c_feature_credit_qmix/seed42/24k/` | 16k→24k 完整续训、新增 20k/24k 评估及六检查点系列 |
| `strong_baselines/development/c_feature_credit_qmix/seed42/32k/` | 24k→32k完整续训、新增28k/32k评估及八检查点系列 |
| `strong_baselines/development/c_feature_credit_qmix/seed42/40k/` | 32k→40k完整续训、新增36k/40k评估及十检查点系列 |
| `strong_baselines/development/c_feature_credit_qmix/seed43/8k/` | 独立8k、4k/8k冻结评估及两个检查点系列 |
| `strong_baselines/development/c_feature_credit_qmix/seed43/16k/`、`24k/`、`32k/`、`40k/` | 用户本地完整续训及有序补评已完成；最终40k、十检查点、200条方法/场景记录已核验 |
| `engineering/`、`group_meeting_summary/` | 既有工程与汇报统计检查，不作为开发性能结果 |

`config_snapshot.yaml` 记录实际配置；`raw_metrics.jsonl`、`aggregate_metrics.json` 记录真实运行；`checkpoints/` 保存模型；评估及系列 manifest 记录身份。CSV 在 `dashboard_logs/`，图片在 `plots/`。

D的内部标识仍为`c_feature_credit_qmix`，模型与日志不重命名；40k系列按来源复用原8k目录的4k/8k、原16k目录的12k/16k、原24k目录的20k/24k和原32k目录的28k/32k评估，不复制模型或改写来源。C三个seed、中间模型和续训来源全部保留；体积大、表现弱或日期早不自动构成删除理由。缓存和测试临时文件按明确清单单独清理。

D 的旧 256 步工程模型、图表和工程评估输出已清理；其[工程验收记录](../metadata/audits/qmix_c_matched/20261018/engineering_validation.json)与[清理前恢复依赖核对](../metadata/cleanup/20261018_method_d_cleanup/engineering_retention.json)保留。正式 8k/16k 模型和评估不受影响。
