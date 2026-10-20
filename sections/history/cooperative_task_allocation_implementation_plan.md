# 合理任务分配：无预训练实施步骤

> 历史记录：本文保存原阶段的实施计划或8k交接协议。旧进度和命令只说明当时状态；当前A/B16k、C32k与后续续训请看[当前结果](../../docs/results/CURRENT_EXPERIMENT_STATUS.md)及[复现说明](../../docs/reproduction/REPRODUCE_C_D.md)。

日期：2026-10-12。

状态：实施计划已编制；下面的代码改动、测试、训练和评估均尚未执行。用户已确认主线，后续按步骤推进；本轮只交付计划。

执行更新（2026-10-12）：第1步已实际执行，基线身份、文件保护清单、A/B/C协议和种子使用审计已保存至 `logs/development/task_allocation/step01_baseline_protocol/`。详细交付与核验见[第1步报告](../../metadata/audits/implementation/step01_baseline_protocol/step01_report.md)。原状态描述保留为计划编制时的记录；第2—10步仍未执行，下一步为统一候选信息与概率接口。

执行更新2（2026-10-12）：第2步已完成候选ID/mask、旧特征视图、实际叶选择与概率诊断接口、私有Route路径及Local CPU负担审计；新接口35项测试和受影响旧模块复验通过，三个历史模型改动前后逐张量兼容检查通过。`unified_node_v1`已登记但仍明确拒绝启用，评分器和PPO接线尚未实现。详见[第2步报告](../../metadata/audits/implementation/step02_candidate_interface/step02_report.md)及[语义核查](../methods/route_candidate_interface_cpu_audit.md)。本更新为最新进度；第3—10步尚未执行，下一步为统一候选Route评分器。

依据：历史32k/8k实验与MTS迁移分析（旧调研已撤下；仅保留历史规划依据名称，不作为当前运行入口）。不以FULL-1k作为技术判断或验收依据。

## 1. 目标、主线与边界

最终目标：模型依据任务、队列、算力、链路和能量，合理选择执行节点，使全系统在原期限与资源约束下完成更多任务。

主线：保留CA-GAT-MAPPO和恢复后的P1a任务信用，适配现有平级候选评分结构，让Local与各合法Remote直接比较；先验证结构，再验证显式物理关系特征。论文式QMIX是后续备选，第一轮不同时替换训练算法。

固定边界：

- 所有新实验随机初始化，不加载P1a/E16或其他已训练模型；P1a代表方法参照，不代表warm-start权重来源。
- 不运行destination预训练、专家蒸馏、离线oracle标签训练或简单场景预训练后迁移。
- 原任务到达、任务大小与计算量、deadline、移动、拓扑、信道、stale CSI、干扰、半双工、资源仲裁、通信/CPU服务、能量核算保持原定义。
- 第一轮保留原reward权重、task-event语义、CPU gate、TX paired-shadow、PPO/LR等设置；Route-Q仍为辅助，CF不进入actor。不借网络改动同时调奖励。
- Idle、Local、Defer、各Remote的合法性与索引映射保持原环境契约；新策略改变的是评分和选择方式，不删除合法等待或强制本地比例。
- 不覆盖历史配置、checkpoint或产物；不修改.agents/与knowledge/。已有sections材料只补充状态或引用，不重写历史结论。
- 保留CLI和轻量main.py；实现放src/，测试放tests/。

## 2. 实验组先固定

| 组别 | Route决策 | 候选特征 | 其余训练/环境 | 要回答的问题 |
| --- | --- | --- | --- | --- |
| A：P1a参照 | 原层次Local/Remote/Defer，再选Remote目的地 | 原特征 | 原P1a | 同预算、同seed的参照表现 |
| B：统一候选结构 | 直接比较Local、每个Remote及Defer | 尽量复用原OptionAware特征和共享评分 | 保留P1a有效credit、CPU/TX及超参数 | 候选比较结构本身是否有帮助 |
| C：统一候选+物理关系 | 与B相同 | 在B上增加统一单位的排队/计算/传输/期限/能量关系 | 与B相同 | 显式关系是否进一步提高系统完成数 |

B相对A是Route决策参数化消融，不宣称逐参数完全相同。C相对B才是特征消融。不得把A→C整体改善全部归因于其中一个组件。

旧P1a seed42/8k结果可作历史校验；正式比较需要方法与实现身份一致的A组。若旧结果与当前可复现版本不能对齐，创建新的A组从零run，不改写旧结果。8k新方法不得直接与E13的32k或五seed均值混报。

## 3. 逐步实施总表

| 步骤 | 核心工作 | 本步完成的判据 | 下一步 |
| --- | --- | --- | --- |
| 1 | 固定基线身份与实验协议 | 保护清单、差异边界、种子与指标已记录 | 2 |
| 2 | 统一候选信息与概率接口 | Local/Remote语义、信息边界及旧路径兼容测试通过 | 3 |
| 3 | 接入统一候选Route评分器 | B前向、采样、确定性选择合法，旧模式不变 | 4 |
| 4 | 适配PPO、task credit、Q辅助和探索 | 真实生产更新路径的数值、梯度、日志及checkpoint测试通过 | 5 |
| 5 | 增加物理关系特征 | C特征单位、可见性、边界与生产接线测试通过 | 6 |
| 6 | 工程回归和配置发布 | 必要回归、完整回归、短集成与恢复检查通过 | 7 |
| 7 | seed42、8k开发试验 | A/B/C真实训练与同协议评估完整，无工程错误 | 8 |
| 8 | 多训练seed、8k验证 | 三训练seed配对比较完成，决定保留B或C或返回诊断 | 9或返回 |
| 9 | 同预算32k与未见种子检验 | 五训练seed、配对评估及独立测试完成 | 10或定稿 |
| 10 | 按条件实施QMIX备选 | 独立方法及公平服务规则对照经测试与真实实验验证 | 单独结论 |

步骤编号代表可独立交付的任务，不代表本轮已完成。每步交付后可按用户指令继续下一步；不将后续整组训练自动混入当前工程步骤。

## 4. 第1步：固定基线与验收协议

### 工作

1. 读取P1a配置、当前实际源码与必要AGENTS说明，记录Git状态，识别已有未提交改动；不能将整个dirty diff当成本次改动。
2. 对将影响实验的源码、原配置和历史checkpoint生成SHA-256清单。保留P1a的8k身份及E13 seed42/43的32k身份，后续每步复核。
3. 保存A/B/C的允许差异列表；第一轮不得改变reward、CPU/TX机制、环境难度、学习率、探索系数或预算来补救表现。
4. 固定开发training seeds为42、43、44；固定开发evaluation seeds为1042—1046。已有1044 oracle只作已见开发诊断，不用于训练或独立测试结论。
5. 为第9步预留training seeds45、46。拟保留evaluation seeds20042—20061作未见测试；先检查项目是否已使用，若已使用，训练前一次性替换并记录，不能看结果后换seed。
6. 固定checkpoint选择：8k与32k各以预算终点final为主结论；中间checkpoint只诊断，不按测试成绩挑选最好模型。
7. 记录原P1a配置中的真实超参数和默认字段，包括Route LR scale、rollout长度、四个PPO epochs、task/CF有效系数及探索退火；以文件为准，不从论文复制数值。

### 交付

在独立logs/development/task_allocation/目录保存baseline_identity、experiment_protocol和保护哈希清单；这些文件在本步实际执行时才生成。更新README宏观状态。

### 验收与失败处理

必须能清楚回答比较的模型、配置、预算、seed、环境及指标分别是什么。身份不清先补齐，不训练。此步是读取和协议工作，不要求为了“有测试”启动环境或优化器。

## 5. 第2步：建立统一候选信息与概率接口

### 工作

1. 复核现有OptionAwareRouteDecoderV1、HierarchicalOptionAwareRouteDecoderV1与RouteOptionFeatureExtractor。已有共享评分逻辑优先复用，不再创建同义算法。
2. 增加默认关闭的新模式，例如route_decoder_mode=unified_node_v1。此名称为计划接口，尚未存在；不得现在写进旧配置运行。
3. 固定候选对齐：Local对应源UAV；Remote候选对应实际邻居ID；Idle和Defer仍为原控制动作。保持环境动作索引与mask，不靠候选排序改变目的地身份。
4. 建立统一结果接口，提供合法leaf概率、chosen log-prob、实际chosen action与可选的Local/Remote/Defer组质量。
5. 区分三种统计：实际leaf argmax；将Remote概率求和后的组质量argmax；给定Remote的条件目的地概率。只有第一种用于新模式的确定性选择，后两种用于诊断/兼容，不伪装成实际动作。
6. 精查Local CPU总负担：共享该CPU的本地任务和收到的远程任务应按实际资源竞争语义纳入可见负担，避免只数本地队列；不重复累计同一任务。Remote使用原可获得的公共负载与AoI，不读取集中状态补成“真实即时负载”。
7. 私有Route backbone、GRU hidden、候选ID及有效性传递保持明确，避免适配后悄悄退回公共backbone。

本步对CPU负担做语义审计，不顺手改变A/B原特征含义。若发现旧显式Local负担未涵盖全部共享CPU竞争，将完整负担关系列入C并记录差异；A/B仍保留既有输入语义，以免把特征修正混入结构消融。

### 测试与验收

- 2/3/4/5 UAV下候选ID、mask与输出维度一致；无Remote、单Remote、无任务、无效公共消息都可处理。
- 候选输入排列与对应ID/mask一起排列时，分数对应排列；不要求历史整网对UAV重编号完全不变。
- 不同源节点的Local始终指向自身；任何非法候选概率为0，合法总概率为1。
- 新模式关闭时旧actor前向、初始化RNG、配置hash与checkpoint加载保持原契约。

不通过时先修候选语义和边界，不进入策略训练。

## 6. 第3步：接入B组统一候选评分

### 工作

1. 复用现有平级decoder共享评分思想，Local和各Remote用同一候选评价路径，保留能表达Local无U2U传输及候选有效性的类型信息。
2. 新模式在所有合法Route叶动作上形成一个分布；训练使用这个分布采样，评估使用同一个分布的合法leaf argmax。Defer仍是合法可学习选项。
3. 不对已训练层次P1a临时改变argmax来制造新成绩；B必须重新随机初始化训练。
4. 为新decoder建立独立初始化随机域，保护同seed共同模块的初始化。需以张量比较验证，不能只声称“用了同seed所以相同”。
5. B仅改结构与所需接口；暂不增加物理关系特征，不改reward、强制Local、增加Remote惩罚或改变CPU/TX服务。

### 测试与验收

- 直接通过已知logits检查新模式选择的是合法最高leaf；检查并列时的固定规则。
- 特别构造“Remote总概率最大，但每个Remote叶概率低于Local”的输入，确认组质量不覆盖leaf决策。
- Defer占优、Local占优、不同Remote分别占优等边界都能正确选择；这是执行契约测试，不是训练有效性证明。
- 检查训练采样与PPO replay采用相同分布、same-action log-prob一致，评估不消费policy采样RNG。

## 7. 第4步：适配P1a学习链与日志

当前config明确限制旧option_aware_v1仅配shared_gae，而P1a的task-event、telemetry与log-barrier限制hierarchical模式。必须增加有测试的新组合，不能直接删除旧校验或自动降级credit。

### 工作

1. PPO继续使用Route作为一个logical component，计算实际chosen leaf的新旧log-prob比值。新模式不把Local-vs-Remote和destination拆成两次独立PPO奖励。
2. task-event保持原任务ID关联、Local/Remote终局、Defer binding bootstrap、rollout frontier和删失语义；actor仍使用shared+task-event，有效task系数1、CF系数0。
3. 适配normalized Route-Q辅助的候选表示和leaf概率输入；保留chosen-only监督与冻结时点，不把未选候选预测当真实标签。若只为兼容新decoder需要改表示，记录为B所需接线，不借机改变Q target或将CF重新加入actor。
4. 复用log-barrier对Local、Defer和Remote总质量的正则计算，保留原系数与真实transition退火日程。Remote组质量仅供正则/统计，不重新控制实际选取。候选数量增加不应变成额外Remote奖励。
5. 日志新增明确decision_mode、真实task_id、候选ID/mask、leaf概率/分数、实际chosen与leaf argmax；组质量及组质量argmax另列。避免旧actor_top_level_argmax字段在新模式下被误读。
6. Route训练与评估日志按schema版本扩展；旧schema仍可读取，缺失字段标记NA，不补造历史task_id或分数。
7. 新checkpoint明确保存新decoder/feature身份、optimizer分组及GRU/rollout恢复状态。旧模式默认行为与旧真实checkpoint严格加载继续保持。

### 必要测试

- 新模式的Route old/new log-prob、ratio、clip、chosen合法性与inactive mask。
- 同一冻结batch中的task target、advantage与四个epoch冻结行为；CF确实不进入actor。
- Local、每个Remote参数路径与Defer均有可验证的生产loss梯度；检查没有向冻结credit/Q target错误反传。
- 合法集合退化、极小概率、Remote不可用和退火结束无NaN/Inf；inactive Route不贡献虚假损失。
- telemetry开关前后动作与RNG一致；新日志实际动作与decision_mode相符。
- 新checkpoint及非空Adam状态往返、带未优化rollout的resume；旧E13/P1a strict load和hash不变。

优先复用tests/test_option_aware_route_decoder_v1.py、test_route_task_credit.py、test_route_actor_credit_source.py、test_route_option_q.py、test_route_exploration.py及已有PPO/checkpoint测试。新增针对跨接口失败风险的测试，不为改名编写重复测试。

### 通过后仍不能声称

非零梯度、测试通过或成功保存checkpoint只能证明接线可用，不能证明模型会选对节点或任务完成更多。

## 8. 第5步：接入C组显式物理关系特征

在B已通过工程测试后，才增加C；B保持独立可运行。可复用E15的单位转换与关系构造经验，但需扩展为Local和Remote统一语义，不只是替换Remote head。

| 关系 | Local | Remote | 边界 |
| --- | --- | --- | --- |
| 任务工作量与期限 | bits、cycles、slack | 同一任务 | 原决策时刻可见 |
| 计算能力与负担 | 本机可见CPU总负担与能力 | 已发布负担与能力 | 保留AoI/validity，不使用远端隐藏即时队列 |
| 计算/排队时间代理 | cycles/f、queued cycles/f | 同单位构造 | f为配置能力或可见能力，不是假设未来必定满频服务 |
| U2U传输代理 | 0，标记无需U2U | task bits、对应TX backlog、stale CSI/history估计速率 | 缺失速率单独标记，不以0秒假装可立即传完 |
| 期限余量关系 | slack与本地服务代理比较 | slack与排队、传输、计算代理比较 | 不据代理强行屏蔽原合法动作 |
| 能量余量 | 本机预算与CPU能耗代理 | 可见邻居预算、CPU代理及源端TX代价 | 代理不等于真实能量扣账 |

静态尺度来自原配置，连续大动态范围采用固定可解释变换；不在eval数据上拟合标准化器。保持原神经输入边界；不得让actor直接访问集中critic输入或oracle结果。

测试重点为单位换算、Local零U2U成本、CPU负担不遗漏/不重复、缺失/过期消息、极端能力与预算、mask与候选对应，以及C特征真实进入生产PPO梯度。不能编写“队列增大后神经分数必须降低”作为随机初始化网络的强制测试；物理特征可验证，评分偏好应由训练结果检验。

## 9. 第6步：工程回归与独立配置发布

### 工作

1. 新增独立配置目录，例如experiments/task-allocation/configs/，准备A/B/C的seed42、8k配置和后续seed模板；本计划中的路径是拟定产物，实施时才生成。
2. 各组log、dashboard CSV、plots目录独立，配置身份明确记录；禁止覆盖旧P1a/E13/FULL目录。
3. 固定8k为16×500 slots，保存4k与final；预算终点按真实collected计，同时报告optimized与unused tail。保留原P1a 64-tail行为，不只为新组增加末尾更新。
4. 在独立测试目录运行短生命周期fixture，覆盖Local→CPU→DONE与Remote→TX→CPU→DONE，以及EXPIRED/TRUNCATED结算。fixture不作为开发完成率。
5. 用很小的真实trainer集成fixture验证新模式训练、保存、resume、评估全链；输出明确标为工程fixture，不混进8k/32k指标。
6. 运行与改动相关的必要回归和仓库完整回归；修复失败后再发布工程gate。CPU/CUDA数值检查按现有设备契约执行。
7. 复核受保护源码/配置/历史checkpoint清单，确认所有差异均在本轮允许范围，README只追加宏观进度。

### 交付与验收

真实测试结果、配置diff、架构身份、环境/旧权重保护哈希、已验证的训练/评估命令与输出路径。只有工程gate通过才进入第7步；当前不提供含未实现参数的“可直接运行命令”。

## 10. 第7步：seed42的8k开发试验

### 执行顺序

1. 核对A组方法身份，必要时重建新的A seed42/8k；历史P1a只作交叉核验。
2. B seed42从零训练到8k，保存4k/final；随后进行相同CPU float32、冻结actor、合法确定性评估。
3. C工程已通过后，以相同seed/预算独立从零训练与评估。不加载A/B权重。
4. 三组使用开发evaluation seeds1042—1046、500 slots；主结果取final，4k只观察学习变化。组间核对任务生成事实、环境参数与外生轨迹指纹；新架构身份不同，不强求整个evaluation config hash相同。
5. 读取完整生命周期、CPU/TX与Route结果，不只看dashboard reward。

### 评估清单

- 每个seed：Generated、DONE、EXPIRED、TRUNCATED、完成率；检查守恒。
- Local/Remote唯一绑定任务数及各自DONE/EXPIRED/TRUNCATED；Defer另外统计决策次数与不同任务数。
- EXPIRED的UNBOUND/TX_WAIT/TX_SERVICE/CPU_WAIT/CPU_SERVICE阶段。
- 各UAV承接任务、CPU服务、queue cycles、能量消耗与不可行闲置；区分可服务时的Idle和能量不可行。
- 概率统计：真实leaf最大项、Local概率、Remote条件差距、Defer概率与其实际argmax次数；不把高熵等同正在评估探索。
- 能耗与延迟作为补充，空分母记NA；不能用少完成任务造成的低能耗掩盖失败。
- 优化更新数、有效任务事件、真实环境steps及额外TX shadow steps/耗时。

### 判断

Local=0本身不自动判失败，但需结合已见开发状态的系统价值辨别与整个episode成绩解释；Local增加而系统DONE不增加也不算目标达成。单seed改善只说明值得扩大验证。不得因为B/C表现不好临时改reward、强制Local或把Defer屏蔽后沿用同一实验名。

## 11. 第8步：三个训练seed的8k验证

扩展A/B/C到training seeds43、44，与seed42共同构成3组×3seed，最多9个8k run；原A42仅在身份完全符合时复用。每个run独立初始化。总主环境预算上限72,000 transitions，不含fixture、评估与TX shadow成本，全部成本单列。

每个训练seed内先对同一组开发evaluation seeds计算均值，再比较B−A、C−A和C−B；同时给出每个seed原始结果及任务数加权pooled口径，不混用均值定义。开发继续条件预先采用：相对A的总体配对平均完成率差为正，且至少2/3个training seeds的开发均值为正。这个门槛是决定是否值得继续的操作标准，不是统计显著性或论文优越性证明。

若B与C都通过，开发均值更高者进入32k；若均值相同，优先较简单的B。保留另一组作为消融，不删除失败run。若两者都未通过，先完成失败归因，不直接堆到32k。

失败归因顺序：工程/索引与日志一致性 → credit和梯度/冻结 → Defer未绑定 → 节点负担与能量 → 任务收益与系统收益冲突。必要时提出单独的credit消融；在记录新假设与差异后才实施，不把参数试错隐藏在原A/B/C中。

## 12. 第9步：同预算32k与未见种子检验

### 工作

1. A与开发选定的B或C，各以training seeds42—46独立从零训练32k，共10个run。新32k不加载8k实验权重；正常同一次32k训练中的中间checkpoint不属于预训练。
2. 总主环境预算320,000 transitions，另列评估、TX shadow与耗时。可按42/43/44再45/46分批执行，但不能看未见测试结果后取消不利seed或调参。
3. 固定保存8k、16k、24k、final；主比较仅用32k final，其他里程碑只作学习轨迹诊断。
4. 仍用相同环境与物理服务；开发评估与未见测试分开。选定版本冻结后再运行预留未见evaluation seeds。
5. 报告每个训练seed结果、配对差、训练seed间离散程度及全部失败run。若估计置信区间，应把训练seed视为独立训练重复，不能把同一模型的多个episode全当独立训练样本。

### 验收

对“完成更多”的结论，必须在同预算A对照与独立测试中获得一致支持，同时没有改弱约束或隐藏截断。若效应不稳或区间无法排除无改善，表述为尚无充分改进证据。最多五个训练seed并不自动保证数据充分或结论显著。

重点解释分配策略在不同源节点负担、helper拥塞、能量余量与链路条件下如何变化；不规定必须有多少Local，不按Remote比例作成功门槛。E13历史结果作为背景，不替代新的同版本A组。

## 13. 第10步：按条件实施论文式QMIX备选

触发条件：统一候选MAPPO仍不能稳定改善，或需要完整算法对照解释合作价值分解的作用。它不属于前九步的顺手小改。

1. 明确控制范围：先选择Route-only QMIX + 固定合法TX/CPU规则；同时建立相同服务规则下的Route-only MAPPO、本地限定、随机和启发式对照。这是新方法族，不能冒充原三头联合算法。
2. 使用相同Local/Remote/Defer候选契约与可见特征、GRU、联合团队reward、单调mixing network、目标网络与在线序列replay；target/mask、padding、burn-in与hidden恢复独立测试。
3. 不复用当前chosen-only Route-Q作为“已实现QMIX”；当前registry只是预留入口，需要真正的网络、trainer、checkpoint与评估接线。
4. 所有经验来自本次交互；禁止预填历史P1a/oracle数据，禁止加载已训练权重。明确多次replay更新不等于新增环境样本。
5. 先通过工程gate，再按与其MAPPO对照一致的8k、32k预算和seed协议检验；单列优化更新量与计算成本。
6. 若单调价值分解在竞争性分配中受限，保留失败结果并分析，不承诺QMIX必胜，也不继续无边界扩大算法范围。

## 14. 实施文件与产物约定

主要可能影响的文件：

- src/config.py：新模式、组合校验、默认关闭与旧hash兼容。
- src/models/ca_gat_mappo.py及必要的新候选模块：复用统一评分、私有backbone及物理关系特征。
- src/models/ca_gat_mappo_actions.py：新模式的真实采样、leaf log-prob、合法确定性选择。
- src/models/route_task_credit.py、route_option_q.py、route_exploration.py：只做新模式所需适配，保持原目标与旧路径。
- src/models/ca_gat_mappo_update.py、trainer/rollout/checkpoint及src/evaluation/：按实际依赖最小修改，接通更新、日志、恢复与评估身份。
- tests/：复用已有测试并补新模式边界与集成测试。
- experiments/task-allocation/configs/：独立A/B/C配置。

新模块与测试的具体文件名在对应步骤落实，不把本表当成必须触碰所有文件的清单。环境物理模块不列为本轮允许修改范围。

运行日志与工程报告放logs/development/task_allocation/；dashboard CSV放dashboard_logs/development/task_allocation/；图放plots/development/task_allocation/；详细方法与阶段结论放sections/；README只记录进度、范围和结论边界。每个run有独立config hash与输出身份，不覆盖旧目录。

每步交付统一包含：做了什么、实际改动文件、已执行测试及结果、保护检查、未解决问题、下一步入口。未执行的测试/训练写“未执行”，不预填PASS或指标。需要回退时关闭新mode或恢复本步改动，不整体重置dirty工作区。

## 15. 下一次实际执行的起点

从第1步开始：固定P1a基线、保护清单和A/B/C协议。第一轮工程的终点是第6步的可运行、可测试、可追溯配置；随后才进入真实8k训练。所有新方法效果目前均未知。

执行更新3（2026-10-12）：第3步已通过显式工程probe接入原特征共享评分、合法叶采样/argmax和同动作重算。seed42/43/44下每组123个共同actor张量相同，三个历史actor/critic严格加载与改动前后输出指纹一致。最终专项81 passed、5 subtests passed，相关回归34 passed、6 subtests passed。正式`unified_node_v1`配置仍拦截启用，训练用Q候选入口明确拒绝，避免P1a学习链未适配时静默使用旧规则；第4步将完成正式构造器、信用/PPO/Q、日志和恢复适配。详见[第3步说明](../methods/unified_route_scoring_step03.md)及[工程报告](../../metadata/audits/implementation/step03_unified_scorer/step03_report.md)。本更新取代前文“尚未执行”的进度描述；第4—10步仍未执行，新方法效果未知。

执行更新4（2026-10-12）：第4步完成正常统一模式构造、411维detach候选Q接线、原P1a任务信用、实际叶PPO、原log-barrier、v2日志及新checkpoint身份。28项新增专项、259项相关回归和53项子测试通过，三项增强复验通过；非空Adam、已推进采样流及160条pending rollout恢复后续更新与对照一致。三个历史模型严格加载、配置hash及输出指纹不变，seed42/43/44共同张量相同。配置层已支持新模式，A/B/C实验配置仍待第6步发布。详见[第4步说明](../methods/unified_route_learning_step04.md)和[工程报告](../../metadata/audits/implementation/step04_learning_chain/step04_report.md)。本更新为最新进度；第5—10步未执行，下一步为C组显式物理关系特征。本轮没有正式训练、预训练或新方法效果结论。

执行更新5（2026-10-12）：第5步完成C组35维显式物理关系、共享物理投影、原mask与PPO/Q接线、日志特征身份及checkpoint区分；B默认输入和固定前向保持不变。32项新增专项、230项相关回归和67项子测试通过，覆盖单位、CPU不重计、缺失/零速率、stale消息、生产梯度、四轮updater及恢复。seed42/43/44下B/C共同156个actor张量和全部critic张量一致，三个历史模型严格加载及输出指纹不变。详见[第5步说明](../methods/unified_route_physical_relations_step05.md)与[工程报告](../../metadata/audits/implementation/step05_physical_relations/step05_report.md)。本更新为最新进度；第6—10步未执行，下一步为完整工程gate、短集成与独立配置发布。正式训练、预训练与新效果评估均未启动。

执行更新6（2026-10-12）：第6步工程gate通过，A/B/C × seeds42、43、44共9份独立8k配置已发布。完整回归首轮1344 passed、6 failed、338 subtests passed；缩短Windows测试临时路径并补齐旧身份测试的中性默认字段后，失败所在模块24项全通过。最终loader相关137项及30项子测试、新增10项真实工程检查通过；生命周期断言增强后另复验7项。跨轮覆盖1360个不同测试，无未解决失败，不表述为第二次完整全绿运行。A/B/C的真实四轮PPO、260条处非空Adam及4条pending恢复、520条终点、独立评估重复与防覆盖均通过。旧V1只恢复九项已完成epoch基础诊断，扩展诊断NA的边界已记录。详见[第6步报告](../../metadata/audits/implementation/step06_engineering_gate/step06_report.md)与[实验入口](task_allocation_experiment_handoff_step06.md)。第7—10步仍未执行；本轮没有正式8k/32k训练、预训练实验或新方法效果结论，下一步按第7步从seed42独立从零运行A/B/C。
