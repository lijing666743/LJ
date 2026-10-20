# 项目目录指南

本页对应2026-10-21清理后的实际目录。先看[当前状态](results/CURRENT_EXPERIMENT_STATUS.md)和[结果索引](results/README.md)。本页说明文件位置，运行命令只在聊天提供。

## 实现、配置和测试

| 位置 | 内容 |
| --- | --- |
| `main.py` | 轻量程序入口 |
| `src/env/` | 生命周期、队列、移动、信道、服务、观察、奖励与能耗 |
| `src/models/` | C/MAPPO、QMIX、信用、回放、更新与完整恢复 |
| `src/evaluation/` | 冻结评估、检查点比较、指标与诊断 |
| `src/policies/`、`src/training/` | 固定策略、训练适配及已有训练支持 |
| `src/artifact_tools/` | 既有产物与侧文件检查工具 |
| `experiments/task-allocation/` | C活动配置与来源清单；A/B独立配置已退役 |
| `experiments/strong-baselines/` | D（CFC-QMIX）的15份分阶段配置及1份工程配置 |
| `tests/`、`tests/fixtures/` | 自动验证与固定回归输入；旧QMIX测试配置位于 fixtures/configs/qmix_legacy/ |
| `tools/`、`environment/` | 已有辅助工具与依赖记录 |

部分旧命名空间只有包入口，实际实现按上表查找。旧配置和 fixture 仍可能被测试引用，不能仅凭名称或日期删除。

## 实验产物

| 位置 | 内容 |
| --- | --- |
| `logs/C/seed42/` | 原 8k、16k/32k/40k 连续训练链及诊断 |
| `logs/C/seed43/`、`logs/C/seed44/` | 独立 40k 与每 4k 开发评估 |
| `logs/strong_baselines/development/c_feature_credit_qmix/seed42/`、`seed43/`、`seed44/` | D（CFC-QMIX）8k/16k/24k/32k/40k训练与评估，目录沿用内部标识 |
| `dashboard_logs/`、`plots/` | 对应训练与评估 CSV、Dashboard 和已有结果图片 |
| `logs/engineering/` | 工程验证记录，不作为开发性能结果 |

QMIX后续系列的4k/8k评估复用原8k目录，12k/16k位于16k目录，20k/24k位于24k目录，28k/32k位于32k目录，36k/40k位于40k目录。原始run_id、checkpoint、来源、日志和指标保持原位。

## 文档与论文

| 入口 | 用途 |
| --- | --- |
| [当前实验状态](results/CURRENT_EXPERIMENT_STATUS.md) | 最新已完成事项、指标和建议 |
| [结果索引](results/README.md) | C、多 seed、QMIX 和既有诊断 |
| [论文材料索引](../sections/README.md) | 0—7 章及方法、规划、研究结果 |
| `sections/methods/` | 35 维关系、统一评分、信用和学习链 |
| `sections/planning/`、`sections/results/` | 强基线协议、比较边界与研究结果 |
| `sections/history/` | 早期实施计划与交接材料 |
| `docs/plans/`、`docs/reproduction/` | 已有总计划与历史复现文档，按日期阅读 |
| `docs/presentations/` | 本地既有汇报材料；仅清理三份哈希相同的预检查副本，不修改正文 |

新论文和详细研究材料位于 `sections/`；工程导航位于 `docs/`。既有报告保留原址，以索引统一查找。

## 身份与历史

[metadata 索引](../metadata/README.md)区分当前导航与历史身份。最新入口是 [CURRENT_WORKSPACE_INDEX.json](../metadata/CURRENT_WORKSPACE_INDEX.json)；原 `CURRENT_EXPERIMENT_MANIFEST.json` 是 2026-10-14 的快照。

历史审计、C和共用恢复备份保持原位；A/B专属的两个本地恢复ZIP已随退役清理，原备份清单按历史日期解释，不再代表全部文件仍存在。旧混合A/B/C导出已删除，保留C原始记录及C单独摘要。见[深度清理清单](../metadata/cleanup/20261020_deep_cd_cleanup/plan.json)。`.agents/`、`knowledge/` 不参与整理；`.git/` 和 IDE 个人设置不作自动迁移。[整理前原文快照](../metadata/reference/repository_snapshots/20261018_before_organization/snapshot_manifest.json)供追溯旧进展与目录说明。
