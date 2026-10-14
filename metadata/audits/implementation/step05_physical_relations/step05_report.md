# 合理任务分配第5步交付报告

日期：2026-10-12。状态：第5步工程验证通过；第6步完整工程gate尚未执行。

## 本步结果

C组现可通过 `training.mappo.route_feature_mode=physical_relations_v1` 构造，Route仍为统一合法叶决策。B默认保持 `original_option_features_v1`，可以独立运行。只接入物理关系与必要的身份校验，没有启动8k/32k正式实验或任何预训练，没有发布下一步的A/B/C实验配置。

新增35维actor可见关系：任务bits/cycles/期限、本地完整CPU队列与已发布远端负担、同单位计算/排队时间、对应源TX backlog、历史速率或stale CSI传输代理、期限余量、执行端/源端能量代理，以及有效性、AoI和历史outage信息。连续值使用原配置尺度和固定signed-log变换，无evaluation数据拟合。

Local计入本地及收到的CPU任务，ego槽只计一次；Local U2U时间、backlog和能耗为0。Remote速率缺失或观测为0时，没有可用的有限传输时间估计，零值只作带false validity的占位，不解释为瞬时完成。新特征不修改环境合法mask，不强制Local比例或移除Defer。

详细方法、公式、35维字段表、代理假设及边界见 [第5步方法说明](../../../../sections/unified_route_physical_relations_step05.md)。机器可读字段与尺度见 [physical_feature_contract.json](physical_feature_contract.json)。

## 改动文件

| 文件 | 本步内容 |
| --- | --- |
| `src/models/route_physical_relations.py`（新增） | 单位还原、Local/Remote关系、validity/AoI、共享35→32→204物理投影 |
| `src/config.py` | 默认兼容的特征模式、C组合校验和原hash保护 |
| `src/models/ca_gat_mappo.py` | B恒等融合入口，C生产构造与特征身份 |
| `src/models/unified_route.py` | C额外参数独立初始化域17002、v2特征身份；B域17001与身份不变 |
| `src/models/ca_gat_mappo_actions.py` | actor/config特征模式一致性与被动日志身份 |
| `src/models/route_decision.py` | v2叶日志追加feature_mode；旧记录缺失仍为NA |
| `src/models/ca_gat_mappo_checkpoint.py` | C身份保存/核验与B/C混用拒绝 |
| `tests/test_route_physical_relations.py`（新增） | 单位、边界、候选、生产PPO/updater、CUDA及恢复测试 |
| `sections/unified_route_physical_relations_step05.md`（新增） | 详细方法与解释边界 |
| `sections/cooperative_task_allocation_implementation_plan.md`、`README.md` | 仅追加本步进度 |

本步差异来自改动前逐文件快照：[implementation_changes.patch](implementation_changes.patch)。工作区已有其他未提交工作，本报告不把整个Git dirty diff归为本步改动。

## 生产学习接线

C物理投影加到原204维候选融合后，再进入原LayerNorm与共享评分器；Local/Remote仍使用同一评分路径。原411维Q布局保留，C候选融合现包含物理关系且继续detach。Q目标、chosen-only监督、task-event信用、有效task系数1、有效actor CF系数0、原探索正则及PPO实际叶ratio未改变。

新增4个参数张量，共7,884个参数。seed42/43/44逐张量核查表明：每组B/C共同156个actor张量以及全部critic张量一致。C新增模块不消耗原CPU初始化流，也不重置CUDA采样流。B在本步前后的配置hash、actor/critic初始化与固定输入前向指纹完全一致。

C checkpoint记录 `unified_route_identity.v2`、`physical_relations_v1`、物理投影随机域与变换身份；B继续使用原v1。即使候选Q维度同为411，两个特征版本也不能当作同一种方法恢复。具体核验见 [configuration_and_initialization.json](configuration_and_initialization.json)。

## 实际测试

| 验证 | 真实结果 | 证据 |
| --- | --- | --- |
| C新增专项 | 32 passed，66.36秒 | [stdout](physical_final.txt)、[JUnit](physical_final.xml) |
| 12个相关旧模块回归 | 230 passed、67 subtests passed，307.33秒 | [stdout](regression.txt)、[JUnit](regression.xml) |
| B前后与B/C共同初始化 | seeds42/43/44全部通过 | [身份与配置审计](configuration_and_initialization.json) |
| 历史P1a 8k、E13 seed42/43 32k | 原配置hash、checkpoint SHA、strict actor/critic load与初始/加载后固定输入指纹一致 | [历史兼容核验](historical_compatibility.json) |
| 保护清单 | 原220项中214项未变，6项为上述允许源码修改；新增2项后清单222项 | [保护检查](protection_check.json)、[新清单](protected_files_after.json) |

合计262项测试与67项子测试通过，不把先前试跑重复计入。测试执行命令保存在 `physical_final_command.json` 和 `regression_command.json`。范围和状态见 [validation.json](validation.json)，运行环境和依赖边界见 [runtime_and_scope.json](runtime_and_scope.json)。

专项覆盖：

- 精确SI单位与能量值、任务/队列不同尺度、Local零U2U、Local+received CPU不漏不重、TX/UNBOUND不混入CPU负担。
- 历史速率优先、零速率与缺失区分、stale CSI/历史干扰回退、AoI、无有效公共消息、零/极小CPU能力及零预算。
- 2—5 UAV、候选ID重排、无Remote/单Remote/Idle、原合法性与非法概率为0。
- Local、Defer及每个Remote选择下，新增物理输入及全部投影参数进入实际PPO梯度；候选Q输入保持detach。CUDA前向/梯度和随机流检查通过。
- 256条保存观测与合成终局工程夹具上的四轮真实updater：物理参数实际更新，首epoch同动作ratio为1，原task/Q接线及有效CF系数不变。没有加载历史模型权重作为C初始化。
- C非空actor Adam状态、160条pending rollout、模型/候选身份保存恢复，以及恢复后下一次相同生产PPO目标的Adam更新一致。完整真实trainer及pending补齐后的集成恢复仍归第6步验收。

初次专项为27 passed、1 failed：测试夹具误把 `FINAL_COMPLETED` 用于resume，原“训练已完成”保护正确拒绝。已将夹具改为 `PERIODIC_RESUME` 和160条pending后复验通过，生产完成态保护没有放宽。初次结果保留在 `physical01.txt/xml`。

## 限制与下一步

时间代理假设配置能力和代理速率可用，不预测未来CPU/TX调度、半双工、其他源到达或资源仲裁。CPU能耗代理仅针对当前任务；源TX代理包括对应TX backlog和本任务。它们都不是完整未来能量可行性判定，不替代真实扣账。

本步没有以FULL-1k作分析或验收依据；历史8k/32k只用于兼容核验，没有整episode评估或恢复训练。测试夹具和合成终局没有用于正式训练，也没有生成新的任务完成率、效果图或论文成绩。

下一步为第6步：完整仓库回归、短真实生命周期与trainer/保存/恢复/评估集成，并发布独立A/B/C配置。该gate通过后才进入第7步从零8k对照。当前不能声称模型已学会合理分配或任务完成更多。
