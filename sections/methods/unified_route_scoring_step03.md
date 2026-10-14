# 第3步：统一Route评分与动作选择

> 阅读定位：本文保留当前方法所继承的定义及原阶段验证记录。旧进度、预算和命令属于历史状态；当前结果与C续训入口见[复现说明](../../docs/reproduction/REPRODUCE_ABC.md)。

日期：2026-10-12。范围为B组原有特征下的评分、采样和确定性选择工程验证。P1a学习链适配仍属于第4步；未进行新策略训练、预训练、oracle生成或历史模型整episode评估。

## 候选如何比较

`src/models/unified_route.py`复用已有`OptionAwareRouteDecoderV1`，Local与每个Remote通过同一组特征投影、融合和共享最终评分器。保留Local/Remote类型嵌入、端点特征、消息有效性和原有burden含义；本步不把第2步诊断出的完整Local CPU总负担替换进B组。

Idle、Local、Defer、各Remote仍使用原环境叶索引。实际评分采用每个ego下非自身节点升序的原目的地顺序；只读候选视图即使重排，也不改变实际评分或执行索引。Defer由原控制head学习；Idle沿用环境支持集。

所有合法叶的原始分数经过一次masked categorical归一化。训练侧采样接口直接从该叶分布采样；同动作重算读取同一叶的log-prob。确定性接口选合法原始分数最大者，并列时取最低合法动作索引。Remote组概率只通过第2步读数接口汇总，不能代替叶选择。

已构造Local分数2、三个Remote各1.5、Defer为-8的工程输入：Remote总质量大于Local，但实际选择Local。该检查证明执行规则，不代表随机初始化网络已经懂得合理分配。

## 初始化公平性

先完整构造未经加载的层次参照actor，保留其历史参数初始化顺序，再丢弃并替换Route head。这样能保留在旧Route head之后创建的TX等共同参数；旧head不再注册或参与优化。

新共享评分器的种子由`SeedSequence([master_seed, 100, 17001])`派生，取有符号64位范围内整数。100为既有Torch初始化流，17001为本评分器独立子域；policy采样流110不变。新评分器在CPU随机域内构造，恢复调用者CPU RNG，且不重置已初始化的CUDA RNG。

seed42、43、44均逐张量核对：每组123个共同actor张量完全相同，critic参数相同；相同seed的新head可重现，不同seed的新head不同。真实P1a的credit、reward、CPU/TX选项及config身份均保留。审计表见`logs/development/task_allocation/step03_unified_scorer/initialization_audit.json`。

## 当前入口与下一步边界

`build_unified_route_probe(reference_config)`仅提供显式工程probe，要求合法的层次参照配置及原destination路径。probe的`baseline_config`代表参照身份，**不是可用于B组实验的配置身份**。构造过程不加载checkpoint，不使用任何预训练权重。

probe把actor及动作分布的实际Route模式同时设为`unified_node_v1`，使用生产forward及既有平级采样/重算路径。普通动作分布构造器新增actor/config模式一致性检查，防止把新actor与旧层次选择规则混用。

正式配置仍明确拒绝启用`unified_node_v1`。新模式请求训练用Route-Q候选特征时也明确报错，因为411维候选表示、task-event/Q辅助、正则、telemetry、PPO与checkpoint身份尚未适配。没有把P1a降级为shared GAE，也没有删除旧组合校验。

第4步须将本步独立初始化与共享评分接入正式新模式构造器，完成P1a学习链和日志适配，并验证真正的PPO梯度及非空优化器/rollout恢复。第6步完整工程gate通过后才能发布A/B/C实验配置并开始8k实验。本步工程probe不作为实验入口或最终checkpoint格式。

## 实际验证

- 新增24项测试：已知叶分数、每个目的地身份、并列规则、Remote组质量与叶决策分离、无Remote/单Remote/仅Idle/仅Defer、2至5架UAV、真实共享评分采样与重算、批次/时间轴、RNG、初始化、旧特征等价和私有hidden路径；其中CUDA随机域检查实际通过。
- 最终专项合计81 passed、5 subtests passed；相关network、action distribution和private backbone回归34 passed、6 subtests passed。合计115 passed、11 subtests passed。
- 首次专项有1项新测试失败：持久化夹具带episode-start，会按契约重置传入hidden，无法据此断言private hidden改变输出。测试改用非episode-start输入后通过；生产源码未因该夹具问题改变。首次失败日志保留。
- P1a seed42/8k、E13 seed42和43/32k的历史actor/critic严格加载通过；使用已核验的本步前源码快照，在同一CPU float32持久化观测上比较初始state、所有head输出、hidden及加载后输出，全部字节指纹相同。

详细stdout、JUnit、执行命令、仅本步差异与保护清单见第3步日志目录。测试不是正式训练结果，不支持完成率提升、收敛或优于参照的结论。
