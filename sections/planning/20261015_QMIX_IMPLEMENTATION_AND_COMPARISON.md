# 保留 C 后的强基线实现与比较协议

2026-10-15。依据用户决定，保留当前 C，停止把 seed44 的额外干预或修复作为强基线比较的前置条件。全部保留 seeds42/43/44 的 40k 和中间检查点；主比较固定 40k，不用各 seed 最佳中期点替换。已有能耗代价和种子波动继续公开。

## 2026-10-17：target同步间隔的单变量开发试验

v6的seed42已实际完成16k训练与4k至16k冻结开发评估，8k/12k/16k分别完成166/116/102个任务，共同分母605。15回合被动重放精确匹配原指标、动作、hidden与外部过程；96个回放块按当前权重前史重算后反传，所测活跃分支梯度未断开，mixer偏导未消失。CPU有积压且存在合法正频率时，Idle/零频率从1574/4855（32.42%）升至1972/4661（42.31%）。8k原本完成任务中82个后来失败，另新增18个完成，净少64个。冻结替换路由、CPU或两者时分别完成124/136/161个；这是诊断干预，不能作为训练成绩或唯一根因证明。

当前权重回放探针显示，12k/16k的target滞后对TD target的影响明显大于8k；12k之前仅在第7224步复制过一次target，12k已出现退化。对照[PyMARL原作者learner](https://github.com/oxwhirl/pymarl/blob/master/src/learners/q_learner.py)确认，其target间隔以episode计数，与本项目的优化器更新单位不同；[BenchMARL配置](https://github.com/facebookresearch/BenchMARL/blob/main/benchmarl/conf/experiment/base_experiment.yaml)明确区分软更新与按优化器步数计数的硬更新。两者均不能证明某个数值适用于本环境。

本轮预先固定仅将`training.qmix.target_update_interval_updates`从200改为50，先做独立seed42的8k开发检查，每4k冻结评估；预期225次更新内复制4次target，而不是原来的1次，这是计数推导而非性能结果。网络、reward、Double-Q、TD(λ=0.6)、当前权重前史重算、学习率、epsilon计划与分支探索规则均不变，C及其已有结果不变。原作者的完整agent动作探索与本项目分支探索存在差异，本轮不同时修改该项，以保留单变量解释边界。

`target_sync_v7`是实验配置版本，checkpoint学习语义仍为v6，配置哈希区分target间隔；不能承接旧200间隔模型作为新曲线，必须独立随机初始化。8k及预备16k配置使用新的输出根；40k模板仅作后续预备，当前不运行40k或其他训练seed。短CUDA验证只证明工程链路，参数50是否改善学习以及是否适合续训均需由新实验回答。

用户授权在诊断和新配置验证后清理不再被新实验依赖的失败QMIX模型、数据、旧配置和旧审计；C、A/B、受保护目录、新代码/配置及本轮简要依据保留。当前依据汇总在`metadata/audits/qmix_target_sync/20261017/receipt.json`。下文200间隔和旧版本运行条目均为历史记录，不是本轮候选设置；旧版本数据不用于最终论文优劣结论。

## 本轮实现对象和当前边界

实现的是冻结方法规范 Section 3.10 中的 **Factorized-Action GAT-QMIX**，属于本环境的七分支 QMIX 适配。它是完整控制基线，训练并执行 Route、TX 选择与资源参数、CPU 队列与频率。它不是 MTS 的复现，不能用本实现冒充用户补充论文的方法。

原始 QMIX 的依据为[原论文](https://proceedings.mlr.press/v80/rashid18a.html)及[作者 PyMARL](https://github.com/oxwhirl/pymarl)。七分支条件动作、循环 chunk、现有 CA-GATv2 图输入和当前有限 horizon 是本项目的适配，需要在论文中明确说明。

MTS 路由适配仍按[文献审阅](20261015_STRONG_BASELINE_REVIEW.md)另行实现，并与重新训练的 C-route 共用固定 TX/CPU 服务。现有完整 C 权重临时换服务不能充当公平训练的 C-route。MTS 的原文平均回报更新细节、当前 reward 和单队首约束须明确对照；在此之前不声称 MTS 已实现。

## 独立学习链

`src/models/factorized_gat_qmix.py` 实现共享 agent Q 网络：复用现有 actor-safe tensorizer 与 CA-GATv2 **层代码**，独立初始化和持有参数。局部 self/public/edge 输入经过图编码与 GRU；七个 Q 分支只读取 recurrent feature，不再把前序动作 one-hot 输入后续分支。合法提案集合分为相互独立的 Route、TX 参数组合和 CPU 队列/频率组合；在组内按原 ActionMasks 计算完整后缀最大值，执行与 TD target 均最大化同一个活跃分支 Q 值之和。epsilon 探索仍按合法条件分支抽样；inactive 分支的 Q 不进入 agent utility。每个动作仍交给同一个 deterministic executor。

同宽度/层数不等于参数量完全相同：C 有私有 Route backbone、物理关系评分器和专用调度机制，QMIX 采用独立的分支 Q heads；mixer 仅训练使用。论文应公开这些差异与实际参数量，不能把该比较说成只替换优化器的严格消融。

2026-10-16限定诊断发现，actor-safe边张量中的干扰/噪声比与历史质量等数值可远大于本机队列输入，图聚合后的LayerNorm不能消除注意力和值聚合阶段已经发生的尺度支配。当前QMIX在图层前对边张量逐元素使用固定的 $\operatorname{sgn}(x)\log(1+|x|)$；这里$x$为既有边输入的一个标量。变换不拟合开发集统计量、不读取中央状态或未来信息，原观测接口和replay原始张量保持不变；collection、online replay、target和冻结评估统一通过同一个encode入口，仅应用一次。C及共享tensorizer/GAT层源码不变。这是本项目QMIX适配的输入预处理，论文应明确披露。

mixer 只读取 centralized state 和 agent utilities。hypernetwork 权重取绝对值，ELU 隐层保持对各 agent utility 的非负偏导。当前分组后缀最大化与独立分支分数配合，保证在原合法提案集合内取得所学习 agent utility 的最大值；不保证真实环境回报最优、联合 executor 的可执行性或训练收敛。

当前 mixer 在 hypernetwork 前使用 **signed-log1p 后无可训练参数的 LayerNorm**，避免中央状态物理量尺度被乘法权重放大。这是明确的训练期状态预处理，不改变 actor observation、reward 或 C。2026-10-16 修复动作解码与 agent utility 不一致的问题后，checkpoint schema 升为 v3，拒绝旧条件 Q heads 的 v2 权重；历史失败产物按用户授权清理。

上述actor边输入尺度修复阶段checkpoint schema为v4，并记录`actor_edge_transform=signed_log1p_v1`。v3权重对应旧输入语义，加载及续训入口均拒绝；不得将v3的16k之后接上v4处理并拼成同一学习曲线。v5继续沿用该actor处理身份，但以新的学习目标身份区分，必须从独立随机初始化开始。

`src/models/qmix_replay.py` 保存 episode 内连续真实 chunk。保存 actor tensors、真实条件 mask 合约、proposal、active、executed summary、reward、centralized state、bootstrap、下一决策信息、episode/slot 及在线/target 各自的 collection-time chunk hidden。序列不能跨 episode；末尾不足 32 的 chunk 保留，不补零、不丢弃。

v5及更早版本采用NO BURN-IN：更新从保存的行为 hidden 开始，不能声称 hidden 已用最新权重重新计算过全部前史。target 网络也有独立保存的 collection-time hidden；target 参数更新后，这些 replay hidden 可能来自较早的 target 版本，是此前实现公开的循环 replay 近似。v6改为下述当前权重前史重算。

online 网络重评真实 proposal。自v5开始采用Double-Q，v6沿用：online网络在合法动作集合内以同一组内后缀最大化选择完整proposal，target encoder和heads重评该proposal，再由target mixer估值；不再由target网络自行选最大值。该选择/估值分工对齐[原作者PyMARL](https://github.com/oxwhirl/pymarl/blob/master/src/learners/q_learner.py)，不是新的服务启发式。

自v5开始使用chunk内TD(λ)，v6沿用，λ固定0.6，数值参考[PyMARL2的QMIX配置](https://github.com/hijkzzz/pymarl2/blob/master/src/config/algs/qmix.yaml)及[其learner](https://github.com/hijkzzz/pymarl2/blob/master/src/learners/nq_learner.py)。令$r_t$为现有全局奖励，$V^-_{t+1}$为上述Double-Q bootstrap，$b_t$表示是否存在真实非终止下一决策，$\gamma=0.99$；chunk内逆序计算$G_t=r_t+\gamma b_t[(1-\lambda)V^-_{t+1}+\lambda G_{t+1}]$。非终止chunk末端令$G_{t+1}=V^-_{t+1}$，真实terminal/truncated令$b_t=0$。该截断多步回报使用真实采样奖励，不添加完成奖励、伪造下一状态或跨回合回传，也不是完整episode回放或重要性修正trace。TD loss仍为真实transition的half-MSE均值，不同长度chunk按真实条数加权，序列逐个反传且梯度裁剪阈值10不变。

`src/models/qmix_targets.py`集中定义Double-Q与λ及其目标身份。checkpoint schema升为v5，记录`learning_target=double_q_chunk_td_lambda_0p6_v1`、λ及chunk边界约定，拒绝v4及更旧的加载/续训。actor边输入处理、七分支结构、解码、reward、Adam学习率、epsilon计划及target复制频率均沿用v4；NO BURN-IN近似仍保留。该训练目标改进不意味着旧一步TD公式非法，也不证明学习失败只有一个根因。

2026-10-16在v5的8k/16k冻结诊断中，完整重放十回合与原评估指标、动作、hidden和外部过程哈希一致。16k的路由更偏向Defer，CPU积压且存在合法正频率选择时仍Idle/零频率的比例从24.76%升至55.21%。冻结分组替换分别恢复部分完成任务，但这是反事实诊断而非重训结果。64个回放块的反传没有发现梯度断开或mixer梯度消失；同输入重算前史后，部分16k样本的路由/CPU梯度方向发生明显偏移，支持消除旧hidden近似，但不证明它是全部退化的唯一原因。

v6只修复循环回放的上下文一致性：更新前，从真实episode起点以当前online/target各自权重重放该chunk之前的actor输入，在`no_grad`下构造两套hidden并detach，再对原chunk计算Double-Q和TD(λ)。同一minibatch内按episode/offset复用已算前史，每次optimizer更新后丢弃该临时缓存；不读取未来输入、不从chunk中间伪造零状态、不让前史参与TD loss或跨完整episode反传。循环缓存覆盖仍需前史时只保存已被逐出的actor输入，episode不再驻留后释放，完整状态checkpoint包含该缓存。旧collection-time hidden继续保存为审计数据，不再初始化训练forward；collection跨更新边界携带hidden的原约定仍保留，不声称行为采集已经按当前权重重算全部前史。

该修复针对[循环经验回放中的recurrent-state staleness风险](https://willdabney.com/publication/r2d2/)，采用本项目有限horizon的完整前史重算；不是R2D2复现，也没有引入其优先回放或分布式训练。v6记录`recurrent_context=episode_prefix_current_weights_no_grad_v1`；schema与源码身份均更新，拒绝把v5权重直接续成v6同一条曲线。现有v5模型、指标及诊断证据保留，约0.69 MB的原运行源码快照供历史复核，不复制模型。下一轮仍需seed42从随机初始化做独立8k检查，4k/8k冻结评估；reward、网络结构、epsilon、λ、学习率、target复制频率及C全部不变，是否改善只能由新的真实训练回答。

不调用 C 的 actor distribution、critic、PPO、GAE、Route task credit/Q 辅助目标或 TX paired shadow。配置中遗留的 MAPPO 字段用于兼容共有 schema/tensorizer及设备字段，不能据此把这些学习机制算入 QMIX。

## 40k 配置和新增数值说明

先比较训练 seeds42/43/44，继续预留45/46用于五训练重复；开发评估1042—1046。40k 草案均关闭正式训练，代码验收与有限开发校准完成后才发布启用版本。预留20042—20061不用于本轮工程测试或调参。

- 共同环境、action、reward、外生随机流及500槽 horizon继承当前 C；所有模型从独立随机初始化开始。
- Q 编码128、单层GATv2、4 attention heads、GRU128；Adam学习率0.0005，gamma0.99。
- replay容量10000个连续chunk，chunk长度32，batch32。每收集完整chunk后，达到一个batch的warmup容量就执行一次optimizer update。
- mixer隐藏维32，hypernetwork隐藏维128；float32；half-MSE，无奖励缩放或新奖励项。
- 预算80回合/40000主环境transitions，每4000步保存；主报告使用预算终点final。
- epsilon从1到0.05，线性32000步；随后保持0.05，评估完全关闭探索。
- target每200次optimizer update硬复制，单位不是环境步或episode。

后两项是本次40k实验对 Section 4 长预算默认值100000/2000的明确覆盖。若沿用原探索长度，40k时epsilon仍为0.62；按本实现每chunk更新的频率，40k总共1280个chunk、warmup后1249次更新，2000间隔不会触发一次target更新。当前草案让最后8k进入低探索，并预期复制target6次。这些是配置推导，不是性能实验结果，也不宣称参数最优。

必要校准只使用开发数据。原计划每方法最多3套配置、总主环境预算80k；2026-10-16用户明确要求进一步修改、删除被替代失败数据并训练新版本，本轮单列追加v5 seed42的8k诊断预算，固定4k/8k评估与8k主报告点，不自动扩到40k或其他训练seed。旧实验文件删除不抵消已消耗的计算预算，修复版本和累计成本应如实披露；不将本轮追加预算描述为仍已满足原80k总上限。不得为了C获胜压低基线预算，也不得根据held-out结果调参。

## 完整恢复和冻结评估

checkpoint在完整episode边界保存 online/target、Adam、replay及其采样RNG、epsilon RNG、计数与真实诊断记录。源文件SHA-256纳入checkpoint；同预算resume要求同配置的periodic、同实现、同设备与torch版本。

2026-10-16按用户确定的8k→16k范围增加显式完整状态预算延长：`src/models/qmix_continuation.py`只接受完成原预算的final，允许改变QMIX回合/环境步上限和三个不重叠的新输出根，其余科学设置必须逐项一致。恢复online/target、Adam、整个replay、两类私有RNG、计数及原诊断历史后，逐内容哈希核对完整状态；原8k已结束完整episode且没有未处理chunk，不重跑旧步数。延长checkpoint记录祖先模型哈希、配置及续训模块指纹，后续periodic恢复继续保留该来源链。这是完整恢复，不是仅加载权重的warm-start；不提供MAPPO专属重放或oracle指令。

`src/evaluation/qmix_continuation_progress.py`在相同冻结协议下核验并复用原4k/8k评估，只新增12k/16k评估，合并四个检查点。原模型和评估原地保留，不复制大checkpoint；续训与评估扩展均放在新模块，原QMIX学习核心及原冻结评估源指纹不变。当前16k只作单seed开发学习长度检查，不替代将来的共同40k主比较。

上段记录的是v3阶段的恢复协议。该阶段8k→16k已实际完成并核对完整状态；因学习检查未通过，相关模型与评估数据按用户要求在本次修复验收后清理。同版本v4仍保留完整恢复与原地复用能力，不能跨输入处理版本恢复。新8k配置只改变输出根以区分版本，其余科学设置与原8k开发配置相同。

`src/evaluation/qmix_runner.py` 从源checkpoint的训练配置构造Q网络，只加载agent权重；不构造optimizer，不加载replay作评估输入。采用现有正式metrics和外生轨迹哈希算法，在QMIX/Local/Heuristic/Random间检查相同场景的外生过程一致性，并检查评估前后网络逐项不变。新增QMIX评估入口独立于现有C四方法入口，旧C入口保持既有实现。

工程验收覆盖单调偏导、包含 TX/CPU 耦合的完整合法动作穷举、合法条件选择、inactive梯度、循环顺序和reset、terminal/padding、真实参数更新、target复制、短尾chunk、restricted checkpoint loader、完整恢复一致、跨方法拒绝、禁止覆盖、冻结评估与主入口。旧轮运行产物已由用户授权清理；修复与清理核验入口为 `metadata/audits/qmix_repair_cleanup/20261016/receipt.json`。短工程试跑只能说明链路能够执行；基线是否充分学习、与C相比如何，必须由后续独立实验回答。

## 下一步顺序

1. 完成动作解码修复、受影响测试和环境回归，检查CUDA放置与运行成本；按用户授权清理旧失败产物，不将其当作有效基线结果。
2. 登记并执行有限开发校准，固定QMIX最终数值设置与全部实现身份；不修改C来追逐基线成绩。
3. QMIX按42/43/44执行匹配40k，按1042—1046冻结评估，与已有C40k全部三个seed及固定基线比较完成率和成本。过程曲线不替代40k主结果。
4. 单独推进MTS共同服务路由对照；补同预算A/B及C45/46后再进入最终测试和结果写作。

此协议更新后，旧计划中“必须先干预/修复seed44才推进”的建议不再是当前任务。论文继续如实保留seed44波动与能耗局限。

## 本轮实际验收记录

2026-10-16 已确认旧实现的顺序 argmax 可能低于其自身活跃分支 Q 值之和的合法组合最大值。当前修复移除跨分支前缀输入，并对 TX、CPU 条件后缀求精确最大值；完整合法动作穷举与真实训练/恢复夹具验证该一致性，修复不加入强制非 Idle、强制非 Defer 或启发式服务规则。循环 replay 的 NO BURN-IN 近似仍保留，不能据此次修复宣称所有学习问题已经解决。

v3的seed42已完成8k→16k连续开发检查，但未通过学习有效性验收。最终五场景与Heuristic共十回合的被动任务追踪精确复现原记录，失败主要停留在CPU阶段；冻结replay的输入敏感度与TD反传检查定位了边输入尺度支配这一具体数值问题，未证明它解释全部性能差距。当前v4只修复该输入处理，并保留NO BURN-IN近似；测试或短CUDA工程更新不能证明新的8k完成率改善。失败模型、评估、图表及旧审计材料在本次验收后按用户要求删除，只保留简要修复/清理回执。下一轮先做seed42的有限8k检查，每4k冻结评估，再决定延长预算。训练命令只在对话提供。

v4的16k最终冻结CPU干预显示，替换已选队列的零频率行为可增加这些开发场景的完成数，但这是干预诊断而非训练模型成绩，不能作为正式基线或保证重训提升。冻结replay还观察到online与target动作排序分歧及target自最大化高于Double-Q估值。本轮因此按用户授权预先固定v5的Double-Q+TD(λ=0.6)改进；只能用后续真实训练评估检验收益，不能把两项改动的效果分别归因或宣称已定位唯一根因。测试通过后清理v4失败产物与临时夹具，保留简要回执，再执行新8k训练；运行说明只在对话提供。
