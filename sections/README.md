# sections 目录说明

## 当前研究入口（2026-10-20）

当前活动学习方法只保留C/D。A/B独立实验及入口已按用户要求退役，后续从完整C/D重新定义组件消融；下方历史材料仍为C/D共用机制和来源提供说明，不表示继续开展旧A/B实验。

- [当前状态](../docs/results/CURRENT_EXPERIMENT_STATUS.md)与[结果索引](../docs/results/README.md)：最新阶段从这里进入。
- [D（CFC-QMIX）协议](planning/20261018_C_MATCHED_QMIX_PROTOCOL.md)与[D的4k—40k结果](results/20261018_C_FEATURE_CREDIT_QMIX.md)：当前强基线工作。
- `planning/`、`results/`：研究规划、比较边界与研究结果；`methods/`、`history/`：既有方法及早期实施材料。

下方 2026-10-13/14 的数量与阶段记录按历史原文保留，不代表当前全部材料。论文 0—7 章正文、公式与科学约束本轮未改；运行命令只在聊天提供。


## 从这里阅读

- [C的物理关系说明](methods/unified_route_physical_relations_step05.md)：阅读当前C方法可从这里开始。
- `methods/`：7份当前方法与继承机制说明，阅读C先看物理关系，再看统一学习链和P1a。
- `history/`：2份历史实施计划/交接记录，需要追溯时再查。
- 根目录：0—7章和本索引；0—7内容未改。

```text
sections/
├─ 0_abstract_keywords.md … 7_conclusion.md
├─ README.md
├─ methods/   当前方法、信用、Q输入与CPU语义
└─ history/   早期实施计划、原8k交接协议
```

公式统一用独立行的`$$`块和行内`$...$`。在纯文本编辑器中看到反斜杠是公式源代码；可使用支持数学预览的Markdown阅读器查看排版结果；原离线HTML当前未保留。两份旧文献说明在本轮开始时已不在工作区，因此未重建，也未列入现行材料清单。


## 当前说明文件用途（2026-10-14）

0—7章全部原样保留。下面列出其他保留材料的实际用途；文件里的旧日期、旧预算及旧启动命令属于历史阶段记录。当前结果和C续训入口统一见[当前状态](../docs/results/CURRENT_EXPERIMENT_STATUS.md)与[复现说明](../docs/reproduction/REPRODUCE_C_D.md)，不要把早期计划中的“尚未训练”当作今天的状态。

| 文件 | 为什么保留 |
|---|---|
| [实施计划](history/cooperative_task_allocation_implementation_plan.md) | 已确认的A/B/C范围、公平比较和多seed规划；历史计划身份保留 |
| [原始交接协议](history/task_allocation_experiment_handoff_step06.md) | 原九份8k配置定义及实际工程边界，历史结果报告直接引用 |
| [任务事件信用](methods/e14c_route_task_credit.md) | 当前任务事件return、Defer停止点和TRUNCATED删失语义 |
| [normalized Q输入](methods/e14d_p0_input_repair.md) | 当前辅助Q的尺度变换、候选表示与chosen-only监督契约 |
| [P1a信用来源](methods/e14e_p1a_task_event_credit.md) | 当前Route actor使用shared+task，CF对actor有效系数为0 |
| [候选与CPU语义](methods/route_candidate_interface_cpu_audit.md) | 当前Local/Remote候选及CPU负担含义，避免特征解释混淆 |
| [统一评分](methods/unified_route_scoring_step03.md) | B/C统一叶评分、动作选择及共同参数初始化公平性 |
| [统一学习链](methods/unified_route_learning_step04.md) | B/C与P1a的PPO、task-event和Q接线 |
| [C的物理关系](methods/unified_route_physical_relations_step05.md) | C实际35维特征、单位还原和信息权限边界 |

已删除单独的旧`e14d_route_counterfactual_q.md`：其Q辅助部分由normalized Q和统一学习链说明覆盖，旧CF直接进入actor的路线已被P1a task_event替代；不再把该旧路线单列为当前方法。原内容身份与替代依据见[Markdown审计](../metadata/cleanup/20261014/markdown_audit.json)。当前保留9份有用途的方法/历史材料及本索引，未改论文公式和章节正文。


## 方案源与当前实现规格

项目的唯一实施基线为：

```
knowledge/project_plan/方案2.md
```

该文件已经冻结，但位于 AGENTS.md 定义的 protected `knowledge/` 目录，本轮自动任务不得修改。当前环境实现规格以已经冻结的 `sections/2_system_model.md` 为准。`sections/` 中的论文正文不得自行改变任务状态机、资源异构定义、动作空间、信息权限、时隙因果或实验边界。

## 章节状态与职责

| 文件                                   | 当前状态           | 内容边界                                                     | 下一步                               |
| -------------------------------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------ |
| `0_abstract_keywords.md`               | 等待核心章节冻结   | 摘要与关键词，不提前写实验结论                               | 在系统模型、方法和实验协议完成后整理 |
| `1_introduction.md`                    | 等待核心章节冻结   | 研究背景、问题、贡献和文章结构                               | 在三章一致性审计后正式撰写           |
| `2_system_model.md`                    | FROZEN / COMPLETE SPECIFICATION | 系统对象、符号、异构性、任务状态机、队列、时隙因果、信道、动作语义和信息权限 | 作为 Section 3/4 接口基线            |
| `3_methods.md`                         | FROZEN / COMPLETE LEARNING INTERFACE | CA-GAT-MAPPO、Factorized-Action GAT-QMIX、训练与执行流程     | 作为 RL implementation interface      |
| `4_experimental_protocol.md`           | FROZEN / COMPLETE IMPLEMENTATION CONTRACT | Gate P–Gate 6、实验族 A/B/C、参数校准、统计和日志规范        | 作为 config/runner/Gate 0 interface    |
| `5_results_template.md`                | 结果占位           | 只接收真实运行结果、图表和统计检验                           | 编码和实验完成后填写                 |
| `6_discussion_limitations_template.md` | 讨论占位           | 结果解释、失败场景、局限性和外推边界                         | 依据真实结果填写                     |
| `7_conclusion.md`                      | 等待核心章节和结果 | 总结研究问题、方法、主要结果和局限性                         | 最后阶段完成                         |

## 核心维护规则

1. `knowledge/project_plan/方案2.md` 是既有方案源；该文件位于 protected `knowledge/` 目录，本轮自动任务不得修改。
2. `2_system_model.md` 是当前环境实现规格源，也是系统对象、符号、资源/负载异构、任务状态机、队列、时隙因果、物理机制、信息权限和动作语义的论文主源。
3. `3_methods.md` 只能说明算法如何处理已经冻结的系统，不得重新定义物理规则、队列更新、动作含义或信息权限。
4. `4_experimental_protocol.md` 只能定义如何校准、训练、评价和统计，不得重新定义系统和算法接口。
5. `route` 与 `tx_select` 必须始终分离：
   - `route` 只处理未绑定任务；
   - `tx_select` 只处理已有传输队列。
6. 主实验使用五个固定资源组，不再使用任意 RB 起始位置。
7. 主实验使用七分支离散动作：
   - `route`
   - `tx_select`
   - `resource_group`
   - `resource_width`
   - `power_level`
   - `cpu_queue`
   - `cpu_frequency`
8. Beta 和 Dirichlet 仅允许出现在可选连续动作扩展中，不得进入主模型。
9. Factorized-Action GAT-QMIX 是分支式多离散变体，不得表述为未经修改的标准 QMIX。
10. 新路由任务、传输完成任务和新到达任务均在槽末入队，下一时隙才允许服务。
11. actor 不读取真实 `lambda_i`，只读取历史任务到达率估计。
12. outage 只在实际被调度且分配非零资源的链路上统计。
13. 半双工由联合 hard-feasible 执行器保证，不能只依赖单智能体本地 mask。
14. 能量不可行时只能执行离散降档或 idle，不得映射为档位集合之外的连续值。
15. 第三 UAV 软遮挡、连续资源动作、Sionna RT、数量 OOD 和类型 embedding 均为可选扩展。
16. 结果和讨论章节只能使用真实运行结果，不得填入虚构数字、曲线、显著性或结论。
17. `2_system_model.md`、`3_methods.md` 和 `4_experimental_protocol.md` 已完成并通过当前一致性审计，初始化/specification 阶段结束，项目当前阶段为 Environment Implementation；冻结章节仍是唯一实现基线。

## 推荐工作顺序

```
Environment Implementation
    -> environment skeleton
    -> Gate 0
    -> random policy rollout
    -> heuristic policy rollout
    -> RL implementation only after Gate 0
```


## 当前A/B/C资料索引（2026-10-13）

0—7章在本次清理前后字节完全一致，不重写其中冻结的系统与论文接口。当前已完成的开发试验以task_allocation_experiment_handoff_step06.md定义A/B/C，统一结构、学习链与35维物理关系分别见unified_route_scoring_step03.md、unified_route_learning_step04.md、unified_route_physical_relations_step05.md。P1a/task-event与Q输入公共依赖说明保留。

当前seed42结果与限制见 ../docs/results/seed42/C_SEED42_8K_HISTORICAL_SUMMARY.md，复现说明见 ../docs/reproduction/REPRODUCE_C_D.md。E15/E16及已不采用的调研材料已删除，当前不保留压缩包；没有以清理名义改变0—7章的研究边界。
