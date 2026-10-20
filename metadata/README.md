# 身份、审计与历史索引

最新导航为 [CURRENT_WORKSPACE_INDEX.json](CURRENT_WORKSPACE_INDEX.json)，当前指标看[实验状态](../docs/results/CURRENT_EXPERIMENT_STATUS.md)。这里的审计不是额外实验结果。当前活动学习方法C/D及退役A/B身份见 [METHOD_CATALOG.json](METHOD_CATALOG.json)。

## 当前证据

| 位置 | 内容 |
| --- | --- |
| [audits/qmix_c_matched/20261018/](audits/qmix_c_matched/20261018/) | C 对齐审阅、当前 QMIX 实现、8k/16k 训练与评估核验 |
| [audits/qmix_c_matched/20261019/](audits/qmix_c_matched/20261019/) | D seed42本地24k/32k/40k、seed43独立8k训练/评估、同预算C比较及多seed配置准备核验 |
| [audits/qmix_c_matched/20261020/](audits/qmix_c_matched/20261020/) | D seed43的32k阶段与最终40k训练/十检查点评估核验、同预算C/D比较及表格核对；早期未齐全回执按时点解释 |
| [audits/qmix_c_matched/20261021/](audits/qmix_c_matched/20261021/) | D seed44最终40k、十检查点评估与三训练seed固定预算比较核验 |
| `audits/c40k_progress/` | C seed42 的 40k 与检查点评估 |
| `audits/c_seed43_progress/`、`audits/c_seed44_progress/` | C 两次独立 40k 的来源与结果 |
| `audits/checkpoint_progress/` | 既有 A/B/C 4k—16k 比较 |
| 其他 `audits/` 日期目录 | 当时的工程检查、诊断或旧候选记录 |

## 历史身份与备份

| 位置 | 如何解释 |
| --- | --- |
| `ABC_RELEASE_MANIFEST.json`、`ABC_FILE_MANIFEST.json` | 原 A/B/C 8k 发布身份，不覆盖后续全部结果 |
| `CURRENT_EXPERIMENT_MANIFEST.json` | 2026-10-14、C32k 时点的历史快照 |
| `DIRECTORY_RELOCATION.json` | 原路径与现址的历史对照 |
| `BASELINE_RECOVERY_MANIFEST.json`、`local_backups/` | 原发布恢复身份；C和共用ZIP保留，A/B专属本地ZIP已按退役范围清理 |
| `reference/` | 历史最小证据与原文快照 |
| `cleanup/` | 整理清单、删除边界和保护核验 |

旧路径、“已完成/尚未开始”等描述按原日期解释。保留旧审计不代表已删除的旧模型仍存在，也不构成运行入口。

2026-10-20按用户要求退役A/B独立实验。删除清单、真实历史指标与C/D保护核验见[本次回执](cleanup/20261020_retire_ab/receipt.json)。原ABC发布清单、混合历史审计及测试夹具来源仍为C/D提供追溯，保持原文，不作为活动A/B实验入口；没有按关键词清除共用实现或改写旧结论。

同日进一步检查发现上轮遗漏的A/B恢复ZIP及旧混合导出，已补清理；旧QMIX配置按测试依赖迁入夹具或删除，记录在[深度清理清单](cleanup/20261020_deep_cd_cleanup/plan.json)。原发布清单和历史审计不改写，其中已删除路径不作当前存在性承诺；C/D源码、活动配置、模型与结果保留。

## 本次整理

[整理回执](cleanup/20261018_repository_organization/receipt.json)记录原文件、归档、缓存及重复副本清理和整理后核验。[整理前快照](reference/repository_snapshots/20261018_before_organization/snapshot_manifest.json)逐字保留七份旧导航文档，其相对链接按原位置解释。

本次只对经哈希确认的重复诊断副本去重；保留的标准文件及旧路径映射见整理回执。历史执行日志保持原文，记录当时的重建路径。`.agents/`、`knowledge/` 不在枚举或处理范围；既有 Git 修改和删除不作恢复、提交或上传。

## D 的固定命名与进一步精简

当前版本固定称 D（CFC-QMIX），内部标识、配置和模型路径不改。[本轮回执](cleanup/20261018_method_d_cleanup/receipt.json)记录工程输出、重复图片包与空目录的清理。原组会 ZIP 的 14 个文件均在展开目录中逐字保留，验收记录保留；历史清单继续描述其原执行时点。
