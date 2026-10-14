# 当前实验入口

当前活动入口为 task-allocation/configs/ 下A/B/C × training seeds42、43、44九份独立8k配置；已完成并交付的是seed42三组。定义与正式配置hash沿用 configuration_manifest.json，其 formal_training_started 字段是Step06发布时记录，不是当前完成状态。

A为P1a分层baseline，B为统一leaf结构，C为B加35维物理关系。环境、reward、PPO、预算和公共学习链保持原协议。禁止预训练、历史权重初始化、warm-start或oracle标签训练。

完整复现方法见 [A/B/C复现说明](../docs/reproduction/REPRODUCE_ABC.md)，原始结果见 [Step07A报告](../docs/results/seed42/seed42_abc_comparison_report.md)。旧实验配置已从活动入口移除；tests/fixtures/configs中的历史配置仅供共享回归。历史P1a对齐配置与最小证据集中保存在 metadata/reference/p1a_seed42_1044/，属于审计引用。

## 2026-10-14 补充

现有配置共13份：原9份8k配置、A/B/C各1份seed42-16k-resume配置、C的1份seed42-32k-resume配置。已完成A/B16k、C32k；43/44仍仅有8k配置。早期清单保留发布时状态，不覆盖成新的训练完成证明。

完整状态续训须显式指定periodic checkpoint、延长预算和重放核对参照；只加载权重或只运行resume配置不构成连续训练。未来C续训所需中期checkpoint与全部来源已保留，下一预算尚未设置。[当前结果](../docs/results/CURRENT_EXPERIMENT_STATUS.md)与[恢复说明](../docs/reproduction/REPRODUCE_ABC.md)提供当前入口。
