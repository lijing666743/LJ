# 实验配置索引

当前C/D配置内容与位置保留，完成状态以[当前实验状态](../docs/results/CURRENT_EXPERIMENT_STATUS.md)和真实产物为准。

| 配置族 | 内容 | 当前状态 |
| --- | --- | --- |
| [task-allocation/configs/](task-allocation/configs/) | 8份C独立训练及续训配置 | C seed42/43/44各40k完成；A/B的8份独立配置已退役删除 |
| [strong-baselines/](strong-baselines/README.md) | 15份D分阶段配置及1份D工程配置 | D seed42/43/44各累计40k训练及十检查点冻结评估已核验 |

配置数量不等于已完成实验数量。当前`task-allocation/configuration_manifest.json`只登记C活动配置；原发布身份保留在历史ABC清单中，不能将历史清单当作当前文件存在性清单。A/B不再是训练入口，后续组件消融另从完整C/D定义。清理范围见[回执](../metadata/cleanup/20261020_retire_ab/receipt.json)。

当前D seed42/43/44的五段训练及各十检查点冻结评估均已核验；固定40k分别完成425/419/409个任务，三seed同预算比较已整理。本轮不启动新训练。工程配置仅供工程验证。实验目录现共24份配置，只保留C与D；三份测试依赖的旧QMIX配置原样移至 tests/fixtures/configs/qmix_legacy/，四份无测试引用的旧工程/draft配置已删除，见[深度清理清单](../metadata/cleanup/20261020_deep_cd_cleanup/plan.json)。

完整恢复依赖配置、源码、checkpoint和来源合同。新seed配置相对seed42同预算仅改变主seed和三个输出根；同seed后续阶段只增加预算并更换输出根，既有来源身份保留。运行命令仅在聊天提供。
