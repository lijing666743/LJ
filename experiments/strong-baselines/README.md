# 强基线配置索引

当前方法固定命名为 **D（CFC-QMIX，共同特征与信用增强 QMIX）**；英文全称为 Common Feature and Credit Augmented QMIX，内部方法标识保留 `c_feature_credit_qmix`，结果与比较边界见[报告](../../sections/results/20261018_C_FEATURE_CREDIT_QMIX.md)。运行命令仅在聊天提供。

D 是完整学习基线的比较编号，不是 A/B/C 结构消融中的第四个单因素分支。原版 QMIX 与旧 `fa-gat-qmix-*` 候选不统称为 D。

## 当前配置

| 配置 | 用途与状态 |
| --- | --- |
| [c-feature-credit-qmix-seed42-8k.json](configs/c-feature-credit-qmix-seed42-8k.json) | 已完成的独立 8k，原模型和 4k/8k 评估保留 |
| [c-feature-credit-qmix-seed42-16k.json](configs/c-feature-credit-qmix-seed42-16k.json) | 已完成的 8k→累计16k 完整续训，新增 12k/16k 评估 |
| [c-feature-credit-qmix-seed42-24k.json](configs/c-feature-credit-qmix-seed42-24k.json) | 用户本地已完成的 16k→累计24k 完整续训，20k/24k 冻结评估及六检查点系列核验通过 |
| [c-feature-credit-qmix-seed42-32k.json](configs/c-feature-credit-qmix-seed42-32k.json) | 本地累计32k/64回合/993更新；28k/32k及八检查点冻结评估核验通过 |
| [c-feature-credit-qmix-seed42-40k.json](configs/c-feature-credit-qmix-seed42-40k.json) | 本地累计40k/80回合/1249更新；36k/40k及十检查点冻结评估核验通过 |
| [c-feature-credit-qmix-seed43-8k.json](configs/c-feature-credit-qmix-seed43-8k.json)及同seed的16k/24k/32k/40k | 五段本地训练及十检查点冻结评估已完成；最终40k/80回合/1249更新，完成419/605个任务 |
| [c-feature-credit-qmix-seed44-8k.json](configs/c-feature-credit-qmix-seed44-8k.json)及同seed的16k/24k/32k/40k | 独立seed44五段训练及十检查点冻结评估已核验；最终40k/80回合/1249更新，完成409/605个任务 |
| [c-feature-credit-qmix-engineering.json](configs/c-feature-credit-qmix-engineering.json) | 工程验证配置；旧试跑输出已清理，验收记录保留，不作为开发性能结果 |

24k 配置只提高 QMIX 回合数/真实步数上限并使用新的三个输出根，学习参数、环境、reward 和 seed 均不变；[恢复核验](../../metadata/audits/qmix_c_matched/20261018/continuation_24k_preflight.json)、[24k 训练核验](../../metadata/audits/qmix_c_matched/20261019/seed42_24k_training_check.json)及[评估核验](../../metadata/audits/qmix_c_matched/20261019/seed42_24k_evaluation_check.json)通过。当前 8k/16k/24k 已完成，不应当成空运行覆盖。

32k同样只增加两项QMIX预算上限并更换三个输出根，[预算与来源核验](../../metadata/audits/qmix_c_matched/20261019/continuation_32k_preflight.json)、[训练完成核验](../../metadata/audits/qmix_c_matched/20261019/seed42_32k_training_completion_check.json)及[评估核验](../../metadata/audits/qmix_c_matched/20261019/seed42_32k_evaluation_check.json)通过。所有现有有效产物保留。

40k只增加QMIX累计回合/真实步数上限并更换三个输出根，学习设置、环境、reward与seed42保持不变。[预算与来源校验](../../metadata/audits/qmix_c_matched/20261019/continuation_40k_preflight.json)、[40k训练核验](../../metadata/audits/qmix_c_matched/20261019/seed42_40k_training_check.json)及[评估核验](../../metadata/audits/qmix_c_matched/20261019/seed42_40k_evaluation_check.json)通过。seed42的40k阶段已完成，不得作为空运行覆盖。

seed43/44新增十份配置，相对seed42同预算仅改变主seed与三个输出根；八条同seed预算扩展合同、十次真实CLI配置解析、CPU零步独立初始化及八项既有续训CLI测试通过。seed43现已全部完成，见[40k训练核验](../../metadata/audits/qmix_c_matched/20261020/seed43_40k_training_check.json)与[完整评估核验](../../metadata/audits/qmix_c_matched/20261020/seed43_40k_evaluation_check.json)；seed44也已完成，见[最终训练核验](../../metadata/audits/qmix_c_matched/20261021/seed44_40k_training_check.json)与[十检查点评估及三seed比较](../../metadata/audits/qmix_c_matched/20261021/seed44_40k_evaluation_check.json)。固定复制seed42的五段预算，阶段内每4k保留模型，最终统一报告40k，不能根据某seed得分较低而更换参数或选择较短预算。当前shadow收集按本段预算截断，不能断言一次直跑40k与五段协议轨迹完全等价；详情见[协议](../../sections/planning/20261018_C_MATCHED_QMIX_PROTOCOL.md)与[配置核验](../../metadata/audits/qmix_c_matched/20261019/multiseed_43_44_config_preflight.json)。

seed43已按用户安排在本地连续完成余下四段训练，再按16k→24k→32k→40k顺序完成累计冻结评估；只调整评估时点，没有改变五段预算及信用规则。该流程的[预算、来源与命令语法核验](../../metadata/audits/qmix_c_matched/20261019/seed43_8k40k_deferred_evaluation_preflight.json)为运行前记录，最新完成状态以40k真实核验及[对比工作簿](../../dashboard_logs/strong_baselines/comparisons/seed43_40k/D_C_seed43_40k_comparison.xlsx)为准。运行命令仅在聊天提供。

## 旧候选与测试依赖

2026-10-20深度整理后，`configs/` 仅保留D的15份分阶段配置与1份工程配置。三份仍供回归使用的旧QMIX 8k/16k/40k配置已逐字迁至 [tests/fixtures/configs/qmix_legacy/](../../tests/fixtures/configs/qmix_legacy/)，十个测试模块同步读取新址；四份无测试引用的旧工程/draft配置已删除。清单与原始哈希见[整理清单](../../metadata/cleanup/20261020_deep_cd_cleanup/plan.json)。旧报告中的原路径按其执行日期解释，不再作为运行入口。

恢复模型必须通过源码指纹、结构合同和完整状态校验，不能按相近文件名混用。当前方法独立初始化，不使用 C 训练权重。
