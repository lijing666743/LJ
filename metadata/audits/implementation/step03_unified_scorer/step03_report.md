# 第3步交付：统一Route评分与叶动作选择

日期：2026-10-12。状态：本步评分/选动作工程gate为PASS；第4步P1a学习链及后续工程/实验尚未执行。没有新策略效果结论。

## 已实现

新增`src/models/unified_route.py`，复用原`OptionAwareRouteDecoderV1`共享评分器及旧特征，构造随机初始化的显式工程probe。Local与每个Remote同路评分，Defer继续可学习；所有合法Route叶直接形成一个masked categorical。生产采样、确定性选择和同动作重算路径实际执行，确定性选择取合法最高叶，并列按最低合法索引处理。

新增独立初始化子域`SeedSequence([seed,100,17001])`。先构造未经加载的历史层次参照actor，再用独立初始化head替换Route，保留其他模块原随机消耗顺序；丢弃的旧head没有注册为probe参数。新head初始化只操作CPU随机域，不重置CUDA随机流。

现有actor仅补新模式所需的raw public、端点与私有Route评分路径。动作分布增加actor/config Route身份一致性检查，防止新actor套用旧层次选择。模式配置校验仍拒绝正式启用；新模式的训练用Q候选入口明确报错，等待第4步接线。工程probe保留参照config作为基线信息，不能把该config hash标为新B组实验身份。

没有修改环境、reward、CPU/TX服务或实验配置，没有强制本地比例、额外远程惩罚、预训练、权重热启动或oracle标签。完整Local CPU负担仍只用于审计，B组原特征不变。

## 实际验证

| 检查 | 结果 |
| --- | --- |
| 新模式工程测试 | 24 passed，含实际CUDA随机域检查 |
| 最终专项四模块 | 81 passed、5 subtests passed，13.55秒 |
| network/actions/private backbone回归 | 34 passed、6 subtests passed，17.69秒 |
| 公平初始化 | seed42/43/44每组123个共同actor张量逐个完全相同，critic张量相同；新head同seed复现、不同seed有差异 |
| 叶选择 | Local、Defer、各Remote胜出、并列、非法最高项、无Remote、单Remote、仅Idle/Defer均符合契约 |
| 组与叶区别 | Remote总质量最大但每个Remote叶低于Local时，实际选择Local |
| 采样/replay/RNG | 2/3/4/5 UAV实际共享评分；一次平级Route采样；chosen log-prob一致；确定性/replay/诊断不消费policy RNG |
| 历史兼容 | P1a seed42/8k、E13 seed42/43/32k的actor/critic严格加载；本步前后初始化state、全部head/hidden及加载后输出指纹一致 |
| 保护 | 开始216文件全部匹配第2步；结束211个原文件不变、5个授权源码/测试变化、2个新增源码/测试，更新清单218文件 |

合计115 passed、11 subtests passed。JUnit计数包含子测试，分别86和40，与上述总数对应。测试只验证工程契约；不作为正式策略训练、预训练样本或完成率结果。

首次专项为78 passed、5 subtests passed、1 failed。失败来自新增private-hidden测试使用episode-start夹具，按原契约hidden会被清零；测试改用非episode-start后，最终整组专项全部通过。保留`focused_01.txt/xml`，未将初次执行记为通过。生产源码在两次专项之间未改变。

历史兼容比较使用本步编辑前保存、且SHA与第2步保护清单一致的源码快照，在独立模块别名下只读加载。比较采用同一CPU float32持久化观测；不是历史策略整episode评估，也不证明当前dirty源码等同历史训练时源码。

## 文件与证据

已有源码/测试仅本步补充：`src/config.py`、`src/models/ca_gat_mappo.py`、`src/models/ca_gat_mappo_actions.py`、`tests/test_route_candidate_interface.py`、`tests/test_candidate_aware_route_decoder_v1.py`。新增`src/models/unified_route.py`、`tests/test_unified_route.py`。

README追加宏观进度，实施计划追加最新执行更新，详细方法见[评分说明](../../../../sections/unified_route_scoring_step03.md)。没有重置、覆盖其他dirty改动；`implementation_changes.patch`只由本步前快照与最终文件生成，不使用整个Git dirty diff作为本步差异。

关键记录：

- `starting_guard.json`：开始时216项完整性核查。
- `initialization_audit.json`：三组seed共同张量清单及初始化身份。
- `historical_compatibility.json`：三个历史模型严格加载与前后指纹。
- `focused_final.txt/xml`、`regression.txt/xml`及对应`*_command.json`：实际测试和命令。
- `protected_files_after.json`、`validation.json`、`implementation_changes.patch`：结束时完整性与改动边界。

未处理或修改`.agents/`和`knowledge/`内容；仅按技能要求读取相关说明。沿用现有Python/依赖/CLI及项目seed协议，不引入依赖或入口改造。技能外部共享环境说明路径在本机不可用，已沿用前步记录的仓库约定与既有运行环境。

## 下一步入口

第4步把独立初始化和共享评分纳入正式新模式构造器，接通P1a的task-event信用、chosen-only normalized Route-Q辅助、实际叶PPO比值、原系数log-barrier、明确decision_mode的日志及checkpoint/Adam/rollout恢复。必须保留shared+task、有效task系数1和CF系数0，不能为绕过校验降级credit。完成必要工程测试后继续第5步C组关系特征及第6步完整gate，之后才开始8k实验。
