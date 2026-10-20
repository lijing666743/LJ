# 第4步：统一Route与P1a学习链适配

> 阅读定位：本文保留当前方法所继承的定义及原阶段验证记录。旧进度、预算和命令属于历史状态；当前结果与C续训入口见[复现说明](../../docs/reproduction/REPRODUCE_C_D.md)。

日期：2026-10-12。本步接通工程接口，不启动正式实验训练。主目标仍为相同期限、资源和能量约束下系统完成更多任务，不设Local占比目标。

## 模式与初始化

`unified_node_v1`现在可以通过配置校验并由正常actor构造器建立，沿用第3步的独立初始化子域。构造时先消耗历史层次head的初始化随机数，保留TX等后续模块的共同参数，再替换成独立初始化的共享评分器；没有注册待丢弃的旧head或加载旧权重。

P1a工程配置相对原参照的resolved差异仅为`training.mappo.route_decoder_mode`。seed42/43/44的共同actor与critic张量逐个相同。旧`option_aware_v1`仍只接受shared GAE；旧模式约束没有被直接删除。新模式允许已有层次模式支持的信用组合，并在task-event组合下继续要求私有Route backbone、branch-specific ratio、role credit和telemetry。实验A/B/C配置尚未发布，仍等第6步。

新模式明确拒绝destination预训练/在线监督或额外destination decoder。本步保留原CLI、依赖、环境、reward及CPU/TX机制。

## PPO与任务信用

Route继续只占一个logical component。PPO old/new log-prob与ratio对应实际选中的完整叶动作，不额外拆分Local-vs-Remote和Remote destination。合法mask与inactive处理复用原动作分布和PPO路径。

P1a仍使用shared advantage加task-event advantage：task有效系数1，CF对actor的有效系数0。配置中的counterfactual系数保留用于原有诊断，不据此把CF重新加回actor。Q辅助系数仍为0.5；原Q损失内部的二分之一平方误差也保持原值。

Local/Remote终局、Defer后续binding bootstrap、rollout frontier、policy-version边界和TRUNCATED删失规则继续由已有task-event模块处理。目标和advantage在当前rollout开始更新前冻结，四个epoch保持一致，不回放跨策略版本的旧actor动作。

## Q候选接线

共享评分器可显式导出411维训练用候选表示：204维当前候选融合嵌入、204维合法Remote平均上下文、3维Local/Defer/Remote类型指示。Idle为全零；Defer复用当前Local任务/源端表示，类型另行区分。结构沿用P1a Q输入布局，但嵌入来自新共享评分器，属于B组所需接线。

所有候选表示在导出时detach，集中状态也在Q网络入口detach。normalized Q的静态尺度和网络、chosen-only监督、终局目标、冻结时点均未改变。只有实际chosen叶回归到当前任务事件回报；未选叶的Q是预测，不是标签。Q诊断可计算CF，但P1a actor继续使用shared+task。

该接口不把第2步审计中的完整Local CPU负担替换进B组，也没有增加C组的时间、速率或能量关系特征。

## 正则与日志

原log-barrier仍按Local、Remote总质量、Defer三个合法组计算，初始系数0.05，按真实收集transition在6000步退火至0。同一rollout的四个epoch使用相同系数。组质量仅用于正则与统计，不决定实际选择；没有按Remote叶数量额外发奖励。

新增`src/models/route_decision.py`生成被动v2记录，使用已经产生的policy输出，不再次调用actor或采样。真实训练记录的任务ID来自当时UNBOUND队首或环境结算的任务事件；评估记录来自决策前观测的队首。记录包括：

- `decision_mode=flat_leaf_v1`、真实`task_id`和decoder身份；
- 候选节点ID、动作索引、原mask、叶分数和概率；
- 实际chosen leaf/node与leaf argmax；
- 另列的组质量及其argmax。

新模式的`actor_top_level_argmax`设为JSON null，汇总按新的真实叶argmax分类，防止把Remote组质量误读为实际确定性选择。旧v1日志行为保留；`normalize_route_decision_record`读取旧记录时把缺失任务ID、分数等保留为None/NA，不能补造历史信息。

训练rollout、task-credit和Q更新行采用相应v2 schema，原日志容器文件名仍保留`_v1.jsonl`以兼容既有路径约定，读取时应检查行内schema。评估trace及summary采用`evaluation_route_telemetry.v2`；旧模式继续输出v1。proposal记录不替代环境执行后的拒绝、降级或服务完成事实。

## Checkpoint与恢复

统一模式的新checkpoint增加必需的`route_decoder_identity`：包含decoder、原特征版本、411维Q表示版本、叶决策模式、初始化子域和派生种子。缺失或篡改身份会被拒绝；旧模式checkpoint的字段布局保持原样。

现有Adam分组、Route慢学习率、GRU双lane、policy generator和active rollout序列化继续复用。新测试使用非空actor/critic Adam状态、已推进的采样流和160条待优化transition恢复，然后补满rollout执行相同四轮更新，与未中断序列化对照逐张量比较。该数据是明确标注的工程夹具，不会被任何新实验加载为训练数据或初始权重。

## 验证与结论边界

新增学习链测试覆盖各叶生产PPO梯度、chosen ratio/clip、inactive Route、Q chosen-only与detach、非零CF不进入actor、四轮冻结、初始化公平性、task ID与日志、正则极值/退火、无Remote/单Remote、CUDA、非空Adam与pending rollout恢复。复用旧任务信用测试继续核对Defer、终局与删失规则。

历史P1a 8k和两个E13 32k的结构化checkpoint校验、actor/critic严格加载、原hash及所有输出指纹不变。新随机初始化的合成checkpoint也通过actor-only loader，无整episode评价。

完整执行数量、初次夹具失败与修复、差异及保护清单见[第4步报告](../../metadata/audits/implementation/step04_learning_chain/step04_report.md)。工程通过不证明新策略已收敛或完成更多任务。下一步只接入C组显式物理关系特征，再执行第6步完整工程gate；其后才按既定协议从零开展8k实验。
