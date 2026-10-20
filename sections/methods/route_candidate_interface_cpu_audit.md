# 第2步：候选接口与本地CPU负担语义核查

> 阅读定位：本文保留当前方法所继承的定义及原阶段验证记录。旧进度、预算和命令属于历史状态；当前结果与C续训入口见[复现说明](../../docs/reproduction/REPRODUCE_C_D.md)。

日期：2026-10-12。范围：实施计划第2步；新评分器、PPO适配和训练不属于本步。

## 候选与动作身份

环境Route叶顺序保持Idle、Local、Defer、按升序排列的非自身UAV。节点候选视图仅包含Local及各Remote，Local的执行节点始终是源UAV自身。Idle和Defer是控制动作，不伪装成执行节点。

新增`src/models/route_candidates.py`提供`RouteCandidateBatch`。视图保存节点ID、原环境动作索引、环境合法性mask、旧task/source/burden/validity特征，以及可选的端点嵌入、Route recurrent特征和独立hidden输出。候选顺序发生变化时，动作索引根据实际节点身份换算，不能以视图中的新位置重新编号目的地。

环境合法性与消息有效性分别记录。某Remote公共消息暂不可用时，原环境可能仍允许卸载；接口保留这个合法动作及无效消息标记，不自行删掉动作，也不读取helper即时私有队列填补缺失消息。节点候选为空支持时，原控制动作仍可合法。

`CAGATMAPPOActor.forward_with_route_interface`显式读取这个视图；普通`forward`保持旧路径。使用Route私有backbone时，端点嵌入、recurrent特征及hidden均来自该私有路径；hidden两个lane不能混用。视图不新增可训练参数，不采样动作，也不改变候选特征的梯度归属。

## 概率与选择

`RouteLeafResult`同时保存合法叶概率、实际选中叶索引/节点ID/叶log-prob、叶argmax、Local/Remote/Defer组质量、组质量argmax和给定Remote的条件目的地概率。Idle独占合法支持时，组argmax标记为-1，Remote条件概率全为0。

实际选择由生产动作分布提供，接口只读取。旧层次策略仍可先选Remote组再选目的地，其实际动作可能不等于全叶argmax；本步不重新解释或替换这个动作。只有后续新统一模式才以叶argmax作为确定性选择。这里的“实际选择”指策略proposal；环境执行后的拒绝或降级仍应由原执行日志记录，不能把本接口当成已执行服务证明。

## Local负担的发现

源码证据：

- `src/env/observation.py`的`ObservationBuilder._private_queues`：`local_cpu`只统计本地任务；`cpu_by_source[ego]`复用同一个本地队列，其他source项统计已传输完成、在本节点等待CPU的远程任务。
- `src/env/queues.py`的`QueueState.enqueue_local/enqueue_cpu`：本地任务与收到的远程任务由互斥队列拥有；远程CPU队列不能使用自身source。生命周期负责保持任务唯一所有权。
- `src/env/actions.py`的`ProposalAdapter._prepare_cpu`及`src/env/service.py`的CPU服务路径：每个UAV的单个CPU动作选择本地队列或一个远程source队列，使用同一个执行节点的频率能力；两类任务竞争该节点的CPU服务。
- `src/env/public_history.py`的`PublicMessageHistory._aggregate_cpu_load`：对外广播的CPU负担包括本地任务与所有收到的远程CPU任务，各计一次。
- `src/models/ca_gat_mappo.py`的`RouteOptionFeatureExtractor.extract`：旧显式Local burden只使用`local_cpu`，Remote burden使用公共CPU汇总。两端的显式负担口径不完全对称。

这不是说原actor完全看不到收到的远程负担：完整self输入已有`cpu_by_source`，backbone可能从中学习。发现只限于旧Route显式Local burden没有直接汇总这些负担。

正确汇总可对`cpu_by_source`各source求和，其中自身项以canonical `local_cpu`替代；不得再将`local_cpu`额外加一次。只汇总已进入CPU竞争的队列，不纳入UNBOUND及尚在传输的TX任务。队列总量也不等于当前slot立即可服务量，后续物理特征仍需尊重服务生效时序。

新增`visible_local_cpu_burden`提供该汇总的诊断视图，返回原tensorizer单位下的normalized task count与remaining cycles。本步不把它接进A/B旧burden或模型评分。

验证夹具由真实生命周期与观测构建器生成：1个本地任务100 cycles，2个接收任务200和300 cycles，再加400 cycles的发出TX任务与500 cycles的UNBOUND任务。正确Local CPU总量为3个任务、600 cycles；本地队列不会重复计数，TX/UNBOUND不会计入。旧显式Local burden仍为100 cycles，以此检验A/B语义没有被顺手修改。以上数字是工程夹具输入与断言，不是实验成绩。

## 对C组的约束

完整Local CPU负担将在第5步作为C组显式关系的输入，B仍保留旧特征语义。旧tensorizer对排队cycles与单任务cycles使用不同归一化尺度，因此旧pressure不能直接当作秒或物理等待时间。C计算排队/计算时间前，须先按各自尺度还原物理量，再以统一单位组合。

Remote仍使用原公共负担、CSI/history及有效性/AoI，不用集中状态补成即时真实负担。对私有端点、当前任务、计算/传输/期限/能量关系的扩展必须遵循既定可见性边界。

本步测试与兼容证据位于`logs/development/task_allocation/step02_candidate_interface/`。`unified_node_v1`已登记为保留接口并明确拒绝启用；只有后续评分器、信用与PPO接线通过相应工程检查后才可发布运行配置。所有新方法效果仍未知。
