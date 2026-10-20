# 强基线选择与十篇参考文献审阅

日期：2026-10-15。状态：文献与项目适配审阅，尚未实现或运行新基线。本文为后续实验的拟定依据，不替代冻结系统规范，也不把文献报告的性能当作本项目结果。

## 1. 建议先确定的比较结构

建议保留 C 为研究主线，采用两层比较：

1. **完整系统比较**：C 与 Local、Random、现有 Heuristic、原方案要求的 Factorized-Action GAT-QMIX 比较。A/B/C 在相同预算下承担结构与特征消融。
2. **路由模块比较**：优先适配补充的 MTS，在共同 TX/CPU 服务策略下，与重新训练的 C 路由模块比较。这里回答“任务分配方法是否有帮助”，不能替代完整系统比较。

MTS 是本次十篇中与当前问题定义最接近的文献基线候选：成功完成任务数、Dec-POMDP、CTDE、循环记忆、延迟邻居队列信息均直接相关。这是适配优先级判断，尚无本环境中谁更强的实验结论。GSTRL/G-MPPO 排第二，作为后备或后续补充，不建议同时复现十篇。

**Factorized-Action GAT-QMIX 与 MTS 不是同一个方法。** 前者是本项目针对七分支动作设计的图值分解方法；后者原文主要学习任务执行节点选择。共享 QMIX mixer 实现可以减少工程重复，但不能把一份模型分别贴两个名字，充当两个独立强基线。

本建议不增加轨迹优化、任务拆分、多跳、DAG、缓存、服务部署或联邦学习。若最终需要完整复现包含这些机制的论文，应另外记录并确认研究边界；本轮没有实施该类变更。

## 2. 本次读取范围与证据等级

已读取项目 README、当前实验总表、seed42/43/44 报告、0—7 章的结构与关键规范、A/B/C 方法说明、复现说明，以及 registry、评估 runner、Route/CPU/TX 诊断相关实现。源码阅读用于确认接口和实现状态，不等于逐行代码审计。

十篇 PDF 均完成题名、摘要及与方法选择有关的系统、算法、实验设置页读取；对主要候选进一步读取连续的方法页面。另渲染核对了 1、3、4、6、8、14 号的关键算法页，以及 MTS 的网络结构页。未逐项复算所有论文公式，也未复现其中实验。

主要候选的已核查页码使用 **PDF 页序**：MTS 第 3—9 页；1 号第 7—9 页；3 号第 3、5—9 页；4 号第 9—14 页；14 号第 7—8 页。原文页码与 PDF 页序可能不同。

部分出版社页面无法完整访问，方法判断以用户提供的全文为主。窄范围网络检索没有定位到可确认属于 MTS、GSTRL 的作者官方实现；这不等于它们没有开源。MTS 提供版为 TVT 接受稿，DOI 已从原文确认，正式卷期年份仍应在写入参考文献前向出版社核验；不采用聚合网站相互冲突的年份。

## 3. 十篇文献分别如何使用

| 来源 | 原文方法与问题 | 对本项目的价值 | 直接移植的主要障碍 | 建议 |
|---|---|---|---|---|
| 补充 MTS | RNN/DQN 与 QMIX；最大化成功处理任务数 | 最接近任务分配、局部观测和协同队列主题 | 同槽多任务子步骤、FIFO、无干扰正交链路、不同 reward 与平均回报更新 | 文献基线第一候选，先做共同服务条件下的路由比较 |
| 1 号 | TF-PPO 与 MOGS；动态协同、吞吐及任务关联 | 任务完成数、负载均衡、匹配调度 | 包含 UAV 移动与任务再次转移；原文的任务匹配不能直接替换当前目的地锁定 | 相关工作与匹配机制备选 |
| 2 号 | TRO；依赖感知潜在动作表示与 TD3，可靠性感知卸载 | 成功率与动作表示文献 | UAV/云协作、轨迹与功率混合动作、任务驱动决策、表示训练阶段 | 文献定位，不列近期必做 |
| 3 号 | GSTRL 框架及 G-MPPO；HGNN、LSTM、masked PPO | 图表示、历史建模和任务卸载的重要近邻 | 原文云端决策使用全局状态，顺序任务，主要动作是执行节点选择 | 文献基线第二候选；需明确局部观测适配或集中信息版本 |
| 4 号 | HOMARL；图 MADRL、Dirichlet 混合卸载、DP CPU 分配 | 学习决策与 CPU 优化分工，可启发当前能耗诊断 | 部分卸载、多跳、移动、搜索不确定性目标与并行频率分配 | 机制讨论参考，不直接接入当前主实验 |
| 5 号 | CLP；潜在动作空间与 TD3，联合卸载、资源与轨迹 | 混合动作及能耗/时延相关工作 | 连续动作、轨迹控制、任务优先级和表示训练 | 暂不复现 |
| 6 号 | MCCCO；通信/计算分解及两层博弈近似 | 非学习优化方法、协同负载均衡 | SDN、地面 D2D、可拆分且有依赖的任务、不同资源分配机制 | 优化基线备选，当前移植代价较高 |
| 8 号 | JTORATC；卸载分解、KKT 资源分配及 SCA 轨迹优化 | 完成时延、能耗与卸载量的权衡 | 连续 CPU 分配、轨迹控制；“卸载任务数”不等于截止期完成数 | 多目标结果讨论参考，不整套移植 |
| 11 号 | SHDRLN/DFRL；分层导航与异构联邦知识共享 | 异构性、能量效率与跨 UAV 知识共享 | 学习对象是导航，原算法使用预训练技能，还引入联邦训练 | 超出当前研究主线 |
| 14 号 | Gibbs 采样与匹配博弈；缓存/服务/卸载联合决策 | 任务—执行节点匹配机制 | 缓存、服务放置、CPU 单元配额、任务交换及 QoE 目标 | 匹配子模块备选；不能整体称为当前同条件算法 |

## 4. MTS：为什么优先，以及怎样保证比较有效

### 4.1 与当前研究接近的部分

原文以各时隙成功处理任务数为目标，任务可留在本 UAV 或卸载至邻居。各 UAV 使用本地队列及上一时隙邻居队列信息执行策略，集中控制器训练各 agent 与单调 mixer。RNN 提供跨时隙记忆，DQN 处理当前任务与本地队列信息。相比全局实时决策方法，其执行信息边界更接近当前系统。

这些机制足以支持把 MTS 列为严肃的比较对象，不能支持提前称 C 优于 MTS，也不能支持把 MTS 称为已复现。

### 4.2 必须单独登记的差异

| 项目 | MTS 原文 | 当前系统 | 拟定处理 |
|---|---|---|---|
| 一槽任务分配数量 | 按到达次序依次处理多个任务，并更新 post observation | Route 仅处理一个未绑定 EDF 队首 | 保留当前环境；明确这是单队首适配，禁止在一个槽中多次调用环境 step 或提前执行绑定 |
| 队列服务 | FIFO | 已冻结的 EDF、队列选择与执行规则 | 所有比较组采用当前服务语义，登记为环境适配 |
| 信息 | 上一槽邻居队列广播 | 原 actor 可见公共摘要、消息 AoI、stale CSI/history | 只使用同一观测接口；不能给 MTS 额外的远端实时完整队列 |
| 无线资源 | 正交信道，无链路间干扰 | 固定资源组、频谱复用、干扰、半双工、联合仲裁 | 继续使用同一物理环境与执行器 |
| 调度动作 | 为任务选本地或邻居执行节点 | Local/Remote/Defer/Idle 加六个服务分支 | MTS 路由版适配现有合法 Route 集；Defer/Idle 的新增与 inactive 语义必须记录 |
| CPU/TX 控制 | 不学习当前七分支全部资源决策 | 当前 C 学习完整动作 | 路由专项中双方使用同一个固定服务策略；完整系统另用七分支 QMIX 比较 |
| Reward | 原文任务成功/未完成相关奖励 | 已冻结的系统 reward 与信用链 | 主比较共用现有环境 reward；MTS 不继承 C 的专用任务信用或 shadow 辅助作为默认设置 |
| 更新目标 | 原文式 (23) 含平均回报项 | 当前 PPO 与有限 horizon 协议 | 核对平均回报估计及 target 更新细节，不能静默替换为常规折扣 QMIX |
| Episode 边界 | 原文讨论近似无限持续问题 | 500 槽、明确 truncated 且无虚构终态 | 适配当前结算与 bootstrap 边界并披露差异，不能照搬无限时域等价性论断 |

原文平均回报估计、RNN 与 DQN 分别更新的细节，须结合伪代码、补充材料或作者实现再核验。本次未找到足以消除全部实现歧义的官方代码，不能把未说明的更新规则猜成作者算法。

### 4.3 建议的最小对照

先实现 `MTS-Route-adapted`，保留循环记忆、任务条件 DQN、QMIX 单调 mixer、经验回放和 target network 的核心结构。最终名称须依据实际保留的学习目标确定；如果改成普通折扣目标，应称为“MTS-inspired recurrent QMIX”，并逐项说明差异。

与之比较的 `C-Route-common-service` 必须在**同一个固定 TX/CPU 服务策略下从随机初始化重新训练**。它可复用 C 的路由结构与物理关系特征，但属于新的模块实验身份。不能把原 C40k 训练后临时替换 CPU/TX 的成绩，当作同训练条件的正式模块对照。已有 C40k 替换服务策略只能作为开发干预诊断。

共同服务策略优先复用现有 Heuristic 的非 Route 分支，并事先固定该组合。各策略的 Route proposal 之后再按同一分支顺序形成服务 proposal，全部交由同一个联合执行器处理；不让当前 Route 的绑定任务在本槽接受服务。

路由专项可再加入 `Heuristic-Route-common-service`，作为无需训练的参照。模块结果与完整 C40k 结果分开报告；即使模块比较获胜，也不能单独证明完整系统获胜。

### 4.4 与七分支 QMIX 的区别

完整系统的 Factorized-Action GAT-QMIX 沿用项目冻结方法规范，控制所有七个离散分支。它用于检验 C 相对另一种协同学习范式的完整决策收益，不作为原版 MTS 的复现。

分支动作存在条件合法集合，必须核对局部动作搜索与 mixer 之间的接口；标准 QMIX 的单调性不能自动证明任意顺序分支贪心在条件 mask 下仍是局部联合 Q 的精确最大值。小型可枚举动作夹具应验证搜索规则，论文只声明实际实现满足的性质。

两类实验可共享底层 replay、mixer 与保存/恢复工具；配置、动作范围、encoder、训练目标、成本、checkpoint 身份和报告名称必须分开。

## 5. GSTRL、MOGS 和 CPU 优化为什么暂列后备

**GSTRL/G-MPPO。** 其优势是图时空建模与 masked PPO，原文同时使用 UAV、已卸载任务和请求任务等异构节点。但原文第 3 页明确由云端根据实时全局状态决策，第 7 页的动作主要是请求任务的执行节点。若为当前 actor 构造局部异构图，只能由现有可见队首和公共摘要生成，不能直接读取环境中所有任务。若删除了重要任务节点或顺序依赖，应称 HGNN-LSTM masked PPO 适配版；若保留全局状态，则属于额外信息条件的集中决策参考，不能与同信息主表混报，也不是严格上界。

**1 号 MOGS。** 原文会把高负载 UAV 已接收任务转移给低负载 UAV。当前任务一旦绑定目的地即锁定，因此不能照搬重分配过程。只保留首次绑定前的偏好排序会弱化原方法，应准确称为匹配启发的路由规则，不能把它当完整 MOGS。匹配需要的任务集合、接收确认与状态交换也须检查是否超过当前广播契约。

**4 号 HOMARL。** 其 DP 在任务子集与连续 CPU 容量之间分配资源，价值来自搜索不确定性降低及能量效率。当前 CPU 每槽服务一个已选队首、频率来自离散档位，任务无搜索价值。直接复制 DP 或 Dirichlet 部分卸载会改变任务及资源语义。适合借鉴“先诊断可完成性与能量，再判断调度是否浪费”的思路，不适合本轮替换生产 CPU 服务。

**14 号匹配及 8 号优化。** 前者的配额与服务放置、后者的连续分配及轨迹优化都需要额外映射。只抽取一个子问题时，必须说明丢失哪些机制；原论文的稳定性、最优性或复杂度结论不能直接移植到新适配器。

## 6. 强基线验收标准

1. 原文机制有可定位来源，重要差异公开记录，缺失细节列为待核验项。
2. 主比较共享环境、到达/移动/信道等外生过程、执行信息边界、期限、能量预算和执行器；训练侧特有机制单独说明。
3. 有真实的梯度/TD 更新、合法动作、循环状态、replay 序列、target 更新、终止语义、checkpoint 往返和评估冻结测试。
4. 独立随机初始化；不加载 C 权重，不用 oracle 标签，不用 held-out 结果调参。
5. 40k 是当前建议的共同主预算，不保证所有算法均已充分训练。报告学习曲线，并为基线预留同样公开的开发调参额度；不以 8k 试跑较差就判定其弱。
6. 同环境步数不等于同计算成本。分别记录主环境 steps、重放/辅助仿真、优化次数、训练耗时、推理耗时和模型规模。
7. 报告所有预定训练 seed；不能删除失败 seed 或只展示优于 Heuristic 的 seed。
8. 若使用集中额外信息、不同 reward 或不同服务规则，应分为单独实验条件，不能混入同信息同协议的主要排名。

在新基线通过这些验收前，论文只能写“计划比较”，不能写“已超过强基线”。

## 7. 来源索引

以下均为用户指定的本地全文；编号沿用原文件名。页码定位见前文。

- 补充 MTS：[原文](<C:/Users/李静/Desktop/文献/研一下  文献/对比文献/An_MARL-based_Task_Scheduling_Algorithm_for_Cooperative_Computation_in_a_multi-UAV_Edge_Computing_System.pdf>)，Luyinru Yang、Jun Zheng、Yuan Zhang，TVT 接受稿，DOI：[10.1109/TVT.2025.3635005](https://doi.org/10.1109/TVT.2025.3635005)。正式卷期待出版社复核。
- 1 号：[A Multi-UAV Cooperative Task Scheduling in Dynamic Environments: Throughput Maximization](<C:/Users/李静/Desktop/文献/参考文献及说明/1.pdf>)，TC 74(2)，2025，DOI：[10.1109/TC.2024.3483636](https://doi.org/10.1109/TC.2024.3483636)。
- 2 号：[Reliability-Aware Optimization of Task Offloading for UAV-Assisted Edge Computing](<C:/Users/李静/Desktop/文献/参考文献及说明/2.pdf>)，TC 74(11)，2025，DOI：[10.1109/TC.2025.3604463](https://doi.org/10.1109/TC.2025.3604463)。
- 3 号：[Graph-Based Spatiotemporal RL Framework for Sequential Task Offloading in Multi-UAV Systems](<C:/Users/李静/Desktop/文献/参考文献及说明/3.pdf>)，TMC 25(5)，2026；在线发表 2025，DOI：[10.1109/TMC.2025.3635085](https://doi.org/10.1109/TMC.2025.3635085)；[出版社检索入口](https://xplorestaging.ieee.org/document/11264345)。
- 4 号：[Hybrid-Mode Computation Offloading Using MADRL for Multi-UAV Cooperative Search](<C:/Users/李静/Desktop/文献/参考文献及说明/4.pdf>)，TMC Early Access，2026，DOI：[10.1109/TMC.2026.3726021](https://doi.org/10.1109/TMC.2026.3726021)；[出版社记录](https://ieeexplore.ieee.org/document/11663091/)。
- 5 号：[Joint Task Offloading, Resource Allocation, and Trajectory Design for Multi-UAV Cooperative Edge Computing With Task Priority](<C:/Users/李静/Desktop/文献/参考文献及说明/5.pdf>)，TMC 23(9)，2024，DOI：[10.1109/TMC.2024.3350078](https://doi.org/10.1109/TMC.2024.3350078)。
- 6 号：[Multi-UAV Cooperative Task Offloading and Resource Allocation in 5G Advanced and Beyond](<C:/Users/李静/Desktop/文献/参考文献及说明/6.pdf>)，TWC 23(1)，2024；在线发表 2023，DOI：[10.1109/TWC.2023.3277801](https://doi.org/10.1109/TWC.2023.3277801)。
- 8 号：[Multi-Objective Optimization for Multi-UAV-Assisted Mobile Edge Computing](<C:/Users/李静/Desktop/文献/参考文献及说明/8.pdf>)，TMC 23(12)，2024，DOI：[10.1109/TMC.2024.3446819](https://doi.org/10.1109/TMC.2024.3446819)；[出版社记录](https://ieeexplore.ieee.org/document/10643301/)。
- 11 号：[Decentralized Navigation With Heterogeneous Federated Reinforcement Learning for UAV-Enabled Mobile Edge Computing](<C:/Users/李静/Desktop/文献/参考文献及说明/11.pdf>)，TMC 23(12)，2024，DOI：[10.1109/TMC.2024.3439696](https://doi.org/10.1109/TMC.2024.3439696)。
- 14 号：[Joint Content Caching, Service Placement, and Task Offloading in UAV-Enabled Mobile Edge Computing Networks](<C:/Users/李静/Desktop/文献/参考文献及说明/14.pdf>)，JSAC 43(1)，2025；在线发表 2024，DOI：[10.1109/JSAC.2024.3460049](https://doi.org/10.1109/JSAC.2024.3460049)。

方法实现与统计参考：

- [QMIX 原论文](https://proceedings.mlr.press/v80/rashid18a.html)与[原团队 PyMARL](https://github.com/oxwhirl/pymarl/blob/master/README.md)，用于核对单调 mixer、replay 和 CTDE；不是现成的本环境多离散适配器。
- [MAPPO 原论文](https://arxiv.org/abs/2103.01955)及论文指定的[官方实现](https://github.com/marlbenchmark/on-policy)，作为 MTS 无法形成可核验适配时的标准循环 MAPPO 后备参考。不能把现有增强 P1a 的 A 组直接叫作标准 MAPPO。
- [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)，支持报告独立训练波动与区间估计的要求，不为本项目指定某个训练 seed 数或保证统计显著性。

上述网络检索只服务于候选身份、实现入口及统计原则核验；本轮没有下载或执行远端代码，没有代替用户联系作者。
