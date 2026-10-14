# 3 方法：CA-GAT-MAPPO 与 Factorized-Action GAT-QMIX

本章在 `sections/2_system_model.md` 已冻结的系统、队列、物理层评价、七分支动作、联合执行器和信息权限之上定义学习方法。本文的主方法为 CA-GAT-MAPPO，学习型对比方法为 Factorized-Action GAT-QMIX。本章冻结的是方法和代码接口，不提前冻结 Section 4 中的网络宽度、训练超参数、随机种子、场景数量或统计协议。

本章中的 actor 始终生成动作提案，而不是最终执行动作。对任意时隙，方法链路固定为

$$
\text{slot-start observation}
\rightarrow
\text{graph construction}
\rightarrow
\text{CA-GATv2}
\rightarrow
\text{GRU}
\rightarrow
\text{seven masked action heads}
\rightarrow
\text{proposed action}
\rightarrow
\text{deterministic hard-feasible executor}
\rightarrow
\text{environment transition}.
$$

其中，真实信道、真实 SINR、当前干扰、当前测量和执行器结果仍由 Section 2 的环境在动作确定后计算；它们不被反向提供给 actor 的当前决策。

## 3.1 从系统问题到学习接口

Section 2 将系统写成 Dec-POMDP

$$
\mathcal{M}
=
\left(
\mathcal{S},
\{\mathcal{O}_i\}_{i\in\mathcal{U}},
\{\mathcal{A}_i\}_{i\in\mathcal{U}},
P,r,\gamma
\right).
$$

在时隙 $t$ 的槽初，UAV $i$ 获得局部观测 $o_i(t)\in\mathcal{O}_i$，并结合自身上一时刻的循环状态生成七分支动作提案

$$
\widetilde a_i(t)
=
\left(
\widetilde a_i^{\mathrm{route}}(t),
\widetilde a_i^{\mathrm{tx}}(t),
\widetilde a_i^{\mathrm{rg}}(t),
\widetilde a_i^{\mathrm{width}}(t),
\widetilde a_i^{\mathrm{pow}}(t),
\widetilde a_i^{\mathrm{cpuq}}(t),
\widetilde a_i^{\mathrm{cpuf}}(t)
\right).
$$

波浪号表示 actor proposal；它不表示环境已经接受该动作。所有 UAV 的 proposal 组成 $\widetilde{\mathbf{a}}_t=(\widetilde a_1(t),\ldots,\widetilde a_N(t))$，随后由 Section 2 的确定性联合执行器处理：

$$
\mathbf{a}_t^{\mathrm{exec}}
=
\mathcal{E}
\left(
s_t,\widetilde{\mathbf{a}}_t,\text{fixed environment state}
\right).
$$

$\mathbf{a}_t^{\mathrm{exec}}$ 包含最终接受、拒绝、降档和 canonicalization 后的结果。环境转移、实际服务、实际能耗、outage、post-service/post-settlement reward 和下一时隙状态只使用该执行结果；PPO 的 old log-prob 仍记录 proposal 的分布概率，不能用执行器结果重写。

对同质 UAV，actor 参数在所有 $i\in\mathcal{U}$ 间共享，集中 critic 也使用一套共享参数。共享参数不意味着共享私有队列或共享局部观测。由于系统物理属性已经在节点和边特征中表达，本章不额外引入 agent-ID embedding；固定 UAV-ID 只用于离散 destination domain、确定性 tie-break 和队列索引。

## 3.2 Actor 信息边界与动态图表示

### 3.2.1 槽初可见信息

actor 输入严格等于 Section 2 允许的槽初信息及其历史消息，不包含本槽尚未形成的量。为便于实现，将 $o_i(t)$ 分解为节点侧、链路侧和可用性标记三类记录：

| 接口分组 | actor 可见内容 | 时间边界 |
|---|---|---|
| 节点与任务 | UAV $i$ 的剩余主动能量、资源能力、四类队列摘要、队列长度、队首任务的剩余 bit/cycle 与 slack、本地历史动作和资源利用率 | 仅槽初已有状态 |
| 业务历史 | $\hat\lambda_i(t)$ 与 $m_i^\lambda(t)$ | 只使用 $A_i(0),\ldots,A_i(t-1)$；$t=0$ 使用缺省值和无效 mask |
| 邻居消息 | 相对位置、相对速度、广播剩余能量、广播资源能力、CPU 负载摘要和消息 AoI | 只使用已收到的历史消息 |
| 链路历史 | 陈旧 CSI $\hat h_{ij,r}(t)$、$m_{ij}^{\mathrm{CSI}}(t)$、历史干扰摘要及其可用性/AoI、上一槽有效速率、历史实际传输尝试 outage 率 | 只使用当前槽开始前已经存在的样本 |
| RU 历史质量 | $\widehat\Gamma_{ij,r}^{\mathrm{hist}}(t)$ 与对应的 per-RU validity mask | 逐 RU 输入 actor；不聚合为 executor scalar |
| 结构标记 | 候选邻居 mask、队列存在性 mask、特征可用性 mask 和缺省值标记 | 由槽初状态确定 |

其中，缺省数值必须与 availability mask 成对出现；缺省的 $0$ 不解释为真实的零信道、零干扰或零 outage。归一化所需的 reference constants 沿用 Section 2 的固定定义，具体编码值留给 Section 4，不能使用运行中 batch 的 min/max 或当前时隙的动态最大值。

actor 明确禁止读取以下信息：

- 当前真实 $h_{ij,r}(t)$、当前真实 $\mathrm{SINR}_{ij,r}(t)$ 和当前真实 interference；
- 当前槽的 $I_{ij,r}^{\mathrm{meas}}(t)$，因为该测量在槽末才形成；
- 未来位置、未来信道、未来到达量 $A_i(t)$、真实业务到达参数 $\lambda_i$ 或尚未生成的任务；
- 真实建筑物遮挡标签；
- 其他 UAV 的完整私有队列和完整任务记录；
- 当前联合动作、$\mathcal{S}_i^{\mathrm{prop}}(t)$、$\mathcal{S}_i^{\mathrm{exec}}(t)$、最终功率、最终 SINR 或执行器接受结果；
- executor-only scalar $\widehat\Gamma_{ij}^{\mathrm{hist}}(t)$。

特别地，actor 只读取逐 RU 的 $\widehat\Gamma_{ij,r}^{\mathrm{hist}}(t)$ 及其 mask。executor-only scalar 只有在 actor 已形成 $\mathcal{S}_i^{\mathrm{prop}}(t)$ 后，才能按照 Section 2 的 valid-RU arithmetic mean 派生，并且不得回写本槽 observation。

### 3.2.2 UAV 图与服务队列表

每架 UAV 是一个节点，时隙 $t$ 的候选邻居图定义为有向图

$$
\mathcal{G}_t=(\mathcal{V},\mathcal{E}_t),
\qquad
\mathcal{V}=\mathcal{U},
\qquad
\mathcal{E}_t=
\left\{(i,j):j\in\mathcal{N}_i(t)\right\}.
$$

$\mathcal{N}_i(t)$ 使用 Section 2 的三维距离阈值在每个槽初重算；图可以随时隙改变，不保存额外的 enter/leave hysteresis。
由于 actor $i$ 不允许读取其他 UAV 的完整私有 observation，动态图输入不采用“所有节点统一编码完整 $o_j(t)$”的接口，而固定采用 ego-centered 的自身私有信息与邻居公开信息分离表示。

对于当前决策 UAV $i$，其自身节点输入记为

$$
\mathbf{x}_i^{\mathrm{self}}(t)
=
\operatorname{Enc}_{\mathrm{self}}
\left(
\mathbf{o}_i^{\mathrm{actor}}(t)
\right),
$$

其中，$\mathbf{o}_i^{\mathrm{actor}}(t)$ 仅包含 Section 2 允许 UAV $i$ 在槽初读取的本地任务、队列、资源、能量、历史到达统计以及相应的缺省值和 validity masks。

对于候选邻居 $j\in\mathcal{N}_i(t)$，定义

$$
\mathbf{p}_j^{\mathrm{pub}}(t)
$$

为 Section 2 actor information boundary 中允许其他 UAV 获取的公共或已广播字段的净化投影，并定义邻居输入

$$
\mathbf{x}_{j\rightarrow i}^{\mathrm{pub}}(t)
=
\operatorname{Enc}_{\mathrm{pub}}
\left(
\mathbf{p}_j^{\mathrm{pub}}(t)
\right).
$$

因此，actor $i$ 的图输入中不存在

$$
\operatorname{Enc}_{\mathrm{node}}
\left(
o_j(t)
\right)
$$

这一“将邻居完整 observation 直接作为节点消息”的接口。

令 $\boldsymbol{\mu}_{ij}(t)$ 表示 actor $i$ 在槽初合法可获得的、仅含历史链路消息和可用性标记的有向边记录，并定义

$$
\mathbf{e}_{ij}^{\mathrm{actor}}(t)
=
\operatorname{Enc}_{\mathrm{edge}}
\left(
\boldsymbol{\mu}_{ij}(t)
\right),
\qquad
(i,j)\in\mathcal{E}_t.
$$

其中，$\mathbf{p}_j^{\mathrm{pub}}(t)$ 只包含已经广播或由既有消息接口公开的邻居字段，例如相对位置、相对速度、广播剩余能量、广播资源能力、CPU 负载摘要以及相应消息 AoI；它不得包含 UAV $j$ 的完整私有任务列表、完整队列内容或私有队首任务记录。

有向边记录 $\boldsymbol{\mu}_{ij}(t)$ 只允许使用 actor $i$ 已合法获得的相对位置和速度、估计距离、陈旧 CSI、CSI AoI、历史干扰摘要及其 AoI、per-RU 历史质量及 validity mask、上一槽有效速率、过去 outage 统计和其他特征可用性 mask。边特征不得读取当前真实信道、当前真实 SINR、当前真实 interference、当前 $I^{\mathrm{meas}}(t)$ 或 executor-only scalar historical quality。

因此，对 actor $i$ 而言，动态图接口固定为：

$$
\boxed{
\mathbf{x}_i^{\mathrm{self}}(t)
+
\left\{
\mathbf{x}_{j\rightarrow i}^{\mathrm{pub}}(t),
\mathbf{e}_{ij}^{\mathrm{actor}}(t)
\right\}_{j\in\mathcal{N}_i(t)}
}
$$

该接口与后续 CA-GATv2 的 ego-centered message passing 保持一致，不引入新的物理广播机制。

Section 2 允许已绑定服务边在暂时超出 $\mathcal{N}_i(t)$ 后继续保留。为避免把“新任务候选图”和“已绑定服务关系”混为一谈，方法使用两个接口：

1. $\mathcal{G}_t$ 只表示当前槽初的新任务候选邻居图，供 CA-GAT 消息传递；
2. 每个发送节点另有固定 UAV-ID 索引的服务队列表，记录 $Q_{i\rightarrow j}^{\mathrm{tx}}(t)$ 是否非空及其槽初历史特征，供 `tx_select` head 使用。

服务队列表不是新的物理链路或新的动作分支；它只是使已锁定目的地在当前候选范围外时仍能被固定 destination domain 正确 mask。对不存在的候选邻居、自身 ID、空传输队列和非法目的地，`tx_select` 的对应 logits 全部 mask。

图的节点数为 $N$，边的数量随 $\mathcal{N}_i(t)$ 变化；节点特征维度、边特征维度以及任何 padding 方式由 Section 4/config 冻结，本章不凭空指定数字。资源单元按 Section 2 的既定 RU 顺序编码，actor 只能读取 per-RU 历史质量，不能输出任意 RU 子集。

## 3.3 CA-GATv2 空间编码器

### 3.3.1 CA 的定义

本文将 CA 定义为 channel/context-aware attention。CA 不是新增物理层信道模型，而是让图注意力显式条件化于 actor 已经可见的链路历史、CSI 可用性、干扰历史、AoI 和 per-RU 质量代理量。主结构固定采用 GATv2 风格的动态边条件注意力，不在代码阶段保留“GAT 或 GATv2”的二选一。

对于 actor $i$，CA-GATv2 采用 ego-centered 的“自身私有信息 + 邻居公开信息”输入接口。中心 UAV $i$ 可以使用其自身完整但合法的 actor observation，而候选邻居 $j\in\mathcal{N}_i(t)$ 只能通过第 2 章 actor information boundary 已允许的公共或已广播字段参与消息传递。

定义中心 UAV $i$ 的自身初始表示为

$$
\mathbf{z}_{i\mid i}^{(0)}(t)
=
\operatorname{Enc}_{\mathrm{self}}
\left(
\mathbf{o}_i^{\mathrm{actor}}(t)
\right),
$$

其中，$\mathbf{o}_i^{\mathrm{actor}}(t)$ 仅包含第 2 章允许 UAV $i$ 在槽初获得的本地 actor 信息。

对于候选邻居 $j\in\mathcal{N}_i(t)$，定义

$$
\mathbf{p}_j^{\mathrm{pub}}(t)
$$

为第 2 章 actor information boundary 中允许其他 UAV 获得的公共或已广播字段的净化投影。该符号仅表示对既有可见信息的投影，不引入新的物理广播机制。邻居 $j$ 在 actor $i$ 的 ego-centered 图中的初始表示固定为

$$
\mathbf{z}_{j\mid i}^{(0)}(t)
=
\operatorname{Enc}_{\mathrm{pub}}
\left(
\mathbf{p}_j^{\mathrm{pub}}(t)
\right).
$$

因此，不允许采用

$$
\operatorname{Enc}
\left(
\mathbf{o}_j^{\mathrm{actor}}(t)
\right)
$$

作为 actor $i$ 的邻居节点输入。特别地，邻居消息不得包含邻居 UAV 的完整任务列表、完整队列内容、私有队首任务记录、未授权的任务或工作负载细节，也不得包含当前真实信道、当前真实 SINR、当前真实干扰、当前 $I^{\mathrm{meas}}(t)$、未来到达、未来信道或 executor-only 的标量历史链路质量。

对于有向边 $j\rightarrow i$，记 actor $i$ 合法可见的边特征为

$$
\mathbf{e}_{ij}^{\mathrm{actor}}(t),
$$

其只能由第 2 章允许的陈旧 CSI、CSI 有效性、历史干扰、相应的 mask/AoI、per-RU 历史链路质量及其有效性掩码，以及其他已明确允许的历史或拓扑字段构造，不得包含当前真实物理层信息。

在 actor $i$ 的 ego-centered 图中，第 $\ell$ 层第 $h$ 个 attention head 定义为

$$
\begin{aligned}
\mathbf{q}_{i}^{(\ell,h)}(t)
&=
W_Q^{(\ell,h)}
\mathbf{z}_{i\mid i}^{(\ell)}(t),\\
\mathbf{k}_{ij}^{(\ell,h)}(t)
&=
W_K^{(\ell,h)}
\left[
\mathbf{z}_{j\mid i}^{(\ell)}(t)
\Vert
\psi^{(\ell,h)}
\left(
\mathbf{e}_{ij}^{\mathrm{actor}}(t)
\right)
\right],\\
\mathbf{v}_{ij}^{(\ell,h)}(t)
&=
W_V^{(\ell,h)}
\left[
\mathbf{z}_{j\mid i}^{(\ell)}(t)
\Vert
\psi^{(\ell,h)}
\left(
\mathbf{e}_{ij}^{\mathrm{actor}}(t)
\right)
\right],\\
u_{ij}^{(\ell,h)}(t)
&=
\mathbf{a}^{(\ell,h)\mathsf T}
\sigma_{\mathrm{LReLU}}
\left(
\mathbf{q}_i^{(\ell,h)}(t)
+
\mathbf{k}_{ij}^{(\ell,h)}(t)
\right).
\end{aligned}
$$

其中，$W_Q^{(\ell,h)}$、$W_K^{(\ell,h)}$、$W_V^{(\ell,h)}$ 和 $\mathbf{a}^{(\ell,h)}$ 为可学习参数，$\psi^{(\ell,h)}(\cdot)$ 为边特征投影，$\Vert$ 表示向量拼接，$\sigma_{\mathrm{LReLU}}$ 为固定形式的逐元素非线性函数。

对中心 UAV $i$ 加入 self-loop，并定义

$$
\mathcal{N}_i^{+}(t)
=
\mathcal{N}_i(t)\cup\{i\}.
$$

self-loop 使用中心 UAV 自身表示和 canonical self-edge mask；对于 $j\neq i$，所有 key、value 和 message 均只能从邻居 public/sanitized representation $\mathbf{z}_{j\mid i}^{(\ell)}(t)$ 产生。

注意力权重为

$$
\alpha_{ij}^{(\ell,h)}(t)
=
\frac{
\exp\left(u_{ij}^{(\ell,h)}(t)\right)
}{
\displaystyle
\sum_{k\in\mathcal{N}_i^{+}(t)}
\exp\left(u_{ik}^{(\ell,h)}(t)\right)
},
\qquad
j\in\mathcal{N}_i^{+}(t).
$$

不存在于 $\mathcal{N}_i^{+}(t)$ 的边不参与 softmax，因此不会通过 padding 产生消息。每个 attention head 的聚合结果为

$$
\mathbf{m}_i^{(\ell,h)}(t)
=
\sum_{j\in\mathcal{N}_i^{+}(t)}
\alpha_{ij}^{(\ell,h)}(t)
\mathbf{v}_{ij}^{(\ell,h)}(t).
$$
多头结果经过共享输出映射、非线性、残差连接与归一化后形成中心 UAV $i$ 的下一层表示。整个 CA-GATv2 的信息流始终遵守

$$
\text{self legal private observation}
+
\text{neighbor public/sanitized projection}
+
\text{actor-visible edge history}
\longrightarrow
\text{CA-GATv2}.
$$

任何邻居 UAV 的完整私有 actor observation 都不得通过节点编码、key、value 或 message 路径进入 actor $i$。

所有 UAV 共享相同的 $\operatorname{Enc}_{\mathrm{self}}$、$\operatorname{Enc}_{\mathrm{pub}}$ 和 CA-GATv2 参数，不引入 agent-ID embedding。具体网络层数、attention head 数和隐藏维度由 Section 4/config 冻结。

因此，actor $i$ 的 CA-GATv2 只接收其自身合法的本地 actor observation、候选邻居的 public/sanitized projection、actor 可见的历史链路边特征及其有效性 mask。它不读取邻居完整私有 observation，也不读取当前真实 $h(t)$、当前 SINR、当前 interference、当前 $I^{\mathrm{meas}}(t)$ 或 executor-only scalar historical quality。

最终供 actor $i$ 后续 GRU 使用的空间表征记为

$$
\mathbf{g}_i(t)
=
\mathbf{z}_{i\mid i}^{(L_{\mathrm{GAT}})}(t).
$$

### 3.3.2 缺省值、mask 与可变邻居


CSI、历史干扰、历史质量和 outage 的缺省数值先与对应 mask 拼接，再进入 $\psi(\cdot)$；mask 不得被静默丢弃。邻居图为空时，self-loop 保证每个节点仍有一条合法归一化路径。图结构的变化只改变本槽可参与聚合的边，不改变 actor 输出头的维度。

## 3.4 Temporal GRU 表征

空间编码器输出通过共享 GRU 形成跨时隙记忆。为避免与 Section 2 的复信道系数 $h_{ij,r}(t)$ 冲突，GRU 隐状态统一记为 $\mathbf{z}_i^{\mathrm{RNN}}(t)$：

$$
\mathbf{z}_i^{\mathrm{RNN}}(t)=\operatorname{GRU}_{\psi}\left(\mathbf{g}_i(t),\overline{\mathbf{z}}_i^{\mathrm{RNN}}(t-1)\right),
$$

其中 $\overline{\mathbf{z}}_i^{\mathrm{RNN}}(t-1)$ 是经过 episode boundary 处理后的上一隐状态：

$$
\overline{\mathbf{z}}_i^{\mathrm{RNN}}(t-1)=\left(1-\delta_{t-1}^{\mathrm{ep}}\right)\mathbf{z}_i^{\mathrm{RNN}}(t-1).
$$

$\delta_{t-1}^{\mathrm{ep}}\in\{0,1\}$ 表示上一 transition 后是否到达 episode boundary。每个 episode 的 $\mathbf{z}_i^{\mathrm{RNN}}(-1)$ 由全零向量初始化；如果 episode 在时隙边界终止，下一 episode 不继承该隐状态。GRU hidden dimension 留给 Section 4/config。

因此，actor 的决策表示固定为当前槽的图表征和上一时序上下文共同决定：

$$
\mathbf{u}_i(t)=\mathbf{z}_i^{\mathrm{RNN}}(t).
$$
rollout 以 episode 内连续 chunk 保存。对于每个时隙 $t$ 和 UAV $i$，buffer 必须显式保存 actor forward 之前实际使用的 recurrent hidden state，记为

$$
\mathbf{z}_i^{\mathrm{RNN,in}}(t).
$$

其中，

$$
\mathbf{z}_i^{\mathrm{RNN,in}}(t)
=
\overline{\mathbf{z}}_i^{\mathrm{RNN}}(t-1).
$$

训练连续 chunk 时，若 chunk 从 $t_{\mathrm{start}}$ 开始，则其初始 hidden state 唯一取为 buffer 中该位置已经保存的

$$
\mathbf{z}_i^{\mathrm{RNN,in}}(t_{\mathrm{start}}).
$$

本文固定采用 `NO BURN-IN`，不再维护另一套独立的 chunk-initial-hidden 表示，也不得把带 GRU 的 PPO 拆成无序独立 timestep。sequence mask 仅用于 padding 和 episode boundary 处的序列截断。

## 3.5 因子化七分支策略

### 3.5.1 七个固定动作域

策略头严格对应 Section 2 的七个分支，不新增第八分支。对 UAV $i$，固定离散 domain、语义和 canonical inactive 值如下：

| 分支 | 固定离散 domain | 语义与环境映射 | inactive 条件与 canonical 值 |
|---|---|---|---|
| `route` | $\{\mathrm{idle},\mathrm{local},\mathrm{defer}\}\cup(\mathcal{U}\setminus\{i\})$ | 只为槽初 $Q_i^{\mathrm{unb}}(t)$ 的 EDF 队首任务选择 local、defer 或一个远程目的地 | 无未绑定任务时为 `idle`；远程目的地必须在 $\mathcal{N}_i(t)$ |
| `tx_select` | $\{\mathrm{idle}\}\cup(\mathcal{U}\setminus\{i\})$ | 选择一个槽初已有且非空的 $Q_{i\rightarrow j}^{\mathrm{tx}}(t)$ | 无非空传输队列时为 `idle` |
| `resource_group` | $\{\mathrm{idle},1,2,3,4,5\}$ | 选择 Section 2 的五个固定资源组，映射为 $g_i(t)$ | `tx_select=idle` 或自身选择 `idle` 时为 `idle` |
| `resource_width` | $\{1,2\}$ | 选择一个或两个相邻固定资源组，映射为 $w_i(t)$ | 父通信分支 inactive 时固定为 $w_i(t)=1$；不引入新的 idle 类别 |
| `power_level` | $\{0,0.25,0.5,1\}$ | 映射为 $p_i^{\mathrm{prop}}(t)=a_i^{\mathrm{pow}}(t)P_i^{\max}$ | 通信 proposal inactive 时固定为既有 $0$ 档 |
| `cpu_queue` | $\{\mathrm{idle}\}\cup\mathcal{U}$ | 选择本地 CPU 队列或一个远程来源 CPU 队列；$k=i$ 表示 local | 无有效 local/remote CPU 队列时为 `idle` |
| `cpu_frequency` | $\{0,0.25,0.5,1\}$ | 映射为 $f_i(t)\in\{0,0.25f_i^{\max},0.5f_i^{\max},f_i^{\max}\}$ | `cpu_queue=idle` 时固定为 $0$ |

`resource_group` 与 `resource_width` 只构造 Section 2 的 $\mathcal{S}_i^{\mathrm{prop}}(t)$，策略网络不直接输出任意 RU 子集。第五组与 width $2$ 的非法组合由 mask 排除；执行器拒绝 proposal 后，$\mathcal{S}_i^{\mathrm{exec}}(t)=\varnothing$。`power_level` 的 $0$ 档仍是已有离散等级，不定义额外的 idle action；若 raw proposal 或 energy downgrade 后最终功率为 $0$，按 Section 2 的规则执行 communication-idle canonicalization。

### 3.5.2 唯一 sampling order 与条件依赖

七个 head 的 sampling order 固定为：

$$
\boxed{\mathrm{route}\rightarrow\mathrm{tx\_select}\rightarrow\mathrm{resource\_group}\rightarrow\mathrm{resource\_width}\rightarrow\mathrm{power\_level}\rightarrow\mathrm{cpu\_queue}\rightarrow\mathrm{cpu\_frequency}}
$$

设 $\mathbf{u}_i(t)$ 为 GRU 表征，$\operatorname{Emb}_k(\cdot)$ 为第 $k$ 个离散分支的嵌入。为避免不必要的全自回归链，本章冻结以下唯一依赖关系：

$$
\begin{aligned}
\mathbf{c}_i^{\mathrm{route}}(t)&=\mathbf{u}_i(t),\\
\mathbf{c}_i^{\mathrm{tx}}(t)&=\mathbf{u}_i(t),\\
\mathbf{c}_i^{\mathrm{rg}}(t)&=\left[\mathbf{u}_i(t)\Vert\operatorname{Emb}_{\mathrm{tx}}(\widetilde a_i^{\mathrm{tx}}(t))\right],\\
\mathbf{c}_i^{\mathrm{width}}(t)&=\left[\mathbf{u}_i(t)\Vert\operatorname{Emb}_{\mathrm{tx}}(\widetilde a_i^{\mathrm{tx}}(t))\Vert\operatorname{Emb}_{\mathrm{rg}}(\widetilde a_i^{\mathrm{rg}}(t))\right],\\
\mathbf{c}_i^{\mathrm{pow}}(t)&=\left[\mathbf{u}_i(t)\Vert\operatorname{Emb}_{\mathrm{tx}}(\widetilde a_i^{\mathrm{tx}}(t))\Vert\operatorname{Emb}_{\mathrm{rg}}(\widetilde a_i^{\mathrm{rg}}(t))\Vert\operatorname{Emb}_{\mathrm{width}}(\widetilde a_i^{\mathrm{width}}(t))\right],\\
\mathbf{c}_i^{\mathrm{cpuq}}(t)&=\mathbf{u}_i(t),\\
\mathbf{c}_i^{\mathrm{cpuf}}(t)&=\left[\mathbf{u}_i(t)\Vert\operatorname{Emb}_{\mathrm{cpuq}}(\widetilde a_i^{\mathrm{cpuq}}(t))\right].
\end{aligned}
$$

因此，`route` 与 `tx_select` 在给定 GRU 表征后条件独立，`cpu_queue` 与通信三分支在给定 GRU 表征后条件独立；资源三分支按 `tx_select → resource_group → resource_width → power_level` 条件化，CPU 频率只按 `cpu_queue` 条件化。后续 head 不能读取未来分支，也不能把执行器结果作为条件输入。

相应的因子化策略写为：

$$
\pi_\theta\!\left(\widetilde a_i(t)\mid o_i(t),\mathbf{z}_i^{\mathrm{RNN}}(t-1)\right)=\prod_{k=1}^{7}\pi_{\theta,k}\left(\widetilde a_i^{(k)}(t)\mid\mathbf{c}_i^{(k)}(t),m_i^{(k)}(t)\right).
$$

其中乘积的顺序就是上面冻结的 sampling order；条件独立只在显式写出的 latent state 和前置分支条件下成立。

## 3.6 Action mask、branch activation 与执行分离

### 3.6.1 Mask 生成与合法 categorical

第 $k$ 个分支的有效动作集合记为

$$
\mathcal{M}_i^{(k)}\left(t;\widetilde a_i^{(<k)}(t)\right)\subseteq\mathcal{D}_i^{(k)},
$$

它只由槽初 actor-visible state、已生成的历史特征和已采样的合法前置分支决定。具体包括队列是否为空、目的地是否为候选邻居或已有服务队列、固定资源组/宽度边界、CPU 队列存在性、局部 residual energy 的保守离散预检查以及已有 feature masks。半双工、联合冲突和完整联合能量可行性不能由单个 actor 的局部 mask 假定解决，仍由 deterministic executor 处理。

给定第 $k$ 个 head 的原始 logit $\ell_{i,a}^{(k)}(t)$，mask 后 logit 固定为：

$$
\ell_{i,a}^{(k),\mathrm{mask}}(t)=\begin{cases}\ell_{i,a}^{(k)}(t),&a\in\mathcal{M}_i^{(k)}(t;\widetilde a_i^{(<k)}(t)),\\-\infty,&a\notin\mathcal{M}_i^{(k)}(t;\widetilde a_i^{(<k)}(t)).\end{cases}
$$

categorical distribution 只在有效集合上归一化。每个 branch 必须至少保留一个合法 action：无任务时保留 canonical `idle`，width 保留 $1$，power 保留 $0$，CPU frequency 保留 $0$。因而不会出现 all-invalid categorical。

mask 严禁读取当前真实 $h(t)$、SINR、interference、$I^{\mathrm{meas}}(t)$、当前联合 proposal 或执行器输出。动作提案生成后才可由环境/adapter 确定 $\mathcal{S}_i^{\mathrm{prop}}(t)$；actor observation 及其 graph 不依赖该集合。

### 3.6.2 Activity indicator

令 $b_i^{(k)}(t)\in\{0,1\}$ 表示第 $k$ 个 branch 是否产生真实决策贡献。定义为：

$$
\begin{aligned}
b_i^{\mathrm{route}}(t)&=\mathbb{I}\left[Q_i^{\mathrm{unb}}(t)\ne\varnothing\right],\\
b_i^{\mathrm{tx}}(t)&=\mathbb{I}\left[\exists j\ne i:Q_{i\rightarrow j}^{\mathrm{tx}}(t)\ne\varnothing\right],\\
b_i^{\mathrm{rg}}(t)&=\mathbb{I}\left[\widetilde a_i^{\mathrm{tx}}(t)\ne\mathrm{idle}\right],\\
b_i^{\mathrm{width}}(t)&=\mathbb{I}\left[\widetilde a_i^{\mathrm{tx}}(t)\ne\mathrm{idle}\right]\mathbb{I}\left[\widetilde a_i^{\mathrm{rg}}(t)\ne\mathrm{idle}\right],\\
b_i^{\mathrm{pow}}(t)&=b_i^{\mathrm{width}}(t),\\
b_i^{\mathrm{cpuq}}(t)&=\mathbb{I}\left[Q_i^{\mathrm{loc}}(t)\ne\varnothing\ \lor\ \exists k\ne i:Q_{k\rightarrow i}^{\mathrm{cpu}}(t)\ne\varnothing\right],\\
b_i^{\mathrm{cpuf}}(t)&=\mathbb{I}\left[\widetilde a_i^{\mathrm{cpuq}}(t)\ne\mathrm{idle}\right].
\end{aligned}
$$

`resource_width` 的定义与 Section 2 的 $m_i^{\mathrm{width,act}}(t)$ 一致；`power_level` 只有在通信资源 proposal active 时计为 active。一个 active branch 即使采样到合法的 $0$ power，也仍是对已有离散功率等级的显式决策；只有父分支 inactive 时使用的 canonical $0$ 才不产生 policy contribution。`cpu_frequency` 在 CPU queue inactive 时使用 canonical $0$，不产生伪造 CPU 决策。

### 3.6.3 Proposal、executor 与 physical action 的边界

策略网络输出的唯一物理接口是 $\widetilde a_i(t)$。在它之后，环境 adapter 按 Section 2 的规则依次完成：

1. 由 `resource_group + resource_width` 构造 $\mathcal{S}_i^{\mathrm{prop}}(t)$；
2. 由 `power_level` 构造 $p_i^{\mathrm{prop}}(t)$，由 `cpu_frequency` 构造 proposed CPU frequency；
3. 对 raw zero-power proposal 执行 communication-idle canonicalization；
4. 使用 actor proposal 形成后的 per-RU historical quality 派生 executor-only scalar；
5. 按 $(\operatorname{slack},-\widehat\Gamma^{\mathrm{hist}},i,j)$ 执行固定字典序半双工仲裁；
6. 按 $1.0\rightarrow0.5\rightarrow0.25\rightarrow0$ 的既有 candidate power 顺序进行 energy downgrade，再按既有 CPU frequency 顺序处理；
7. 形成 $y_{ij}(t)$、$x_{ij,r}(t)$、$\mathcal{S}_i^{\mathrm{exec}}(t)$、$p_i^{\mathrm{exec}}(t)$ 和 $f_i(t)$。

上述 executor 是确定性环境组件，不参与 actor 的 logit、mask 或 proposal log-prob 计算。拒绝或降档不反写为 actor 的另一个动作。PPO rollout buffer 必须显式保存 actor 的 proposed action，并同时显式保存 compact executed-action summary 及 executor rejection/downgrade summary，用于环境一致性检查、诊断、调试和实验指标统计。executed-action summary 仅属于环境执行元数据，不得作为 PPO policy action，不得替代 proposed action，也不得用于重新计算 old log-prob 或 PPO probability ratio。

## 3.7 Centralized critic

主方法采用单个共享的 centralized global value：

$$
V_\phi(s_t)=V_\phi\left(\operatorname{Enc}_{\mathrm{cent}}(s_t)\right)\in\mathbb{R}.
$$

critic 的输入是决策时刻 $t$ 可合法定义的全局状态，可包含全部 UAV 的位置、速度、资源能力和剩余主动能量；全部任务记录、四类队列、剩余 bit/cycle、slack 和目的地绑定状态；当前真实信道、建筑物几何/遮挡状态、真实干扰相关环境量和业务到达参数；以及执行器在决策前可获得的全局候选、资源与约束相关状态。

这些 privileged features 只服务 centralized training。critic 不读取未来位置、未来信道、未来到达量或本次动作执行后才产生的 reward、SINR、service 和 executed action；在 $t=T-1$ 之后不存在用于 bootstrap 的 $s_T$。

critic 的标量 value 对所有 agent 共享，MAPPO 使用同一个团队 reward 和同一个 global advantage。actor 的 decentralized inference 只调用 $o_i(t)$、当前图、mask 和自身 recurrent state，不调用 $s_t$ 的 privileged 分量。

## 3.8 MAPPO 目标与优化接口

### 3.8.1 Return、GAE 与固定 horizon bootstrap

本章的 $r_t$ 直接采用 Section 2 的 frozen post-service/post-settlement reward；本章不重新定义任务完成、过期、能耗或 workload 的物理语义。令 $\gamma\in(0,1)$ 为折扣因子，$\lambda_{\mathrm{GAE}}\in[0,1]$ 为 GAE 系数，二者具体数值留给 Section 4/config。

定义 episode boundary 指示量 $\delta_t^{\mathrm{ep}}$：仅当环境在 transition 后报告 episode `terminated`/`done`，或到达固定 horizon 的 `truncated` terminal bookkeeping 时，$\delta_t^{\mathrm{ep}}=1$；普通任务完成/过期但 episode 尚未终止时不置为 $1$。固定 horizon 采用 no-bootstrap contract：

$$
b_t^{\mathrm{boot}}=\mathbb{I}[t<T-1]\mathbb{I}[\delta_t^{\mathrm{ep}}=0].
$$

因此，terminated、truncated 和固定 horizon 的最后一个 service slot 均不使用 value bootstrap。对非边界 transition，TD residual、GAE advantage 和 return target 为：

$$
\begin{aligned}
\delta_t^V&=r_t+\gamma b_t^{\mathrm{boot}}V_\phi(s_{t+1})-V_\phi(s_t),\\
\widehat A_t&=\delta_t^V+\gamma\lambda_{\mathrm{GAE}}b_t^{\mathrm{boot}}\widehat A_{t+1},\\
\widehat R_t&=\widehat A_t+V_\phi(s_t).
\end{aligned}
$$

truncated 只表示 Section 2 的 episode boundary bookkeeping，不虚构 $s_T$、$r_T$、任务完成或额外 terminal penalty。$\widehat R_t$ 的计算只在 rollout chunk 内按时间逆序进行，并应用 sequence mask。

### 3.8.2 Proposal joint log-prob 与 entropy

对每个 UAV，joint log-prob 指七个分支的 active-branch joint probability，而不是展平所有 agent 后的巨大 joint action table：

$$
\log\pi_{\theta,i}^{\mathrm{joint}}(t)=\sum_{k=1}^{7}b_i^{(k)}(t)\log\pi_{\theta,k}\left(\widetilde a_i^{(k)}(t)\mid\mathbf{c}_i^{(k)}(t),m_i^{(k)}(t)\right).
$$

inactive canonical branch 的 log-prob 不计入上式。熵采用 active-branch sum，而不是 active-branch mean：

$$
\mathcal{H}_i(t)=\sum_{k=1}^{7}b_i^{(k)}(t)\mathcal{H}\left[\pi_{\theta,k}(\cdot\mid\mathbf{c}_i^{(k)}(t),m_i^{(k)}(t))\right].
$$

当一个时隙没有 active branch 时，$\mathcal{H}_i(t)=0$。训练目标在 agent/time/rollout 有效位置上统一求平均；不再除以每个时隙的 active branch 数，因此该定义无 zero-active 歧义。

rollout 中的 old log-prob 必须来自采样 proposal 时的 masked categorical distribution。确定性 executor 的拒绝、降档、资源接受和最终功率不重新计算也不修改该 old log-prob。

### 3.8.3 PPO clipped objective

令 $\theta_{\mathrm{old}}$ 为采集当前 rollout 时的 actor 参数，probability ratio 定义为：

$$
\rho_{i,t}(\theta)=\exp\left(\log\pi_{\theta,i}^{\mathrm{joint}}(t)-\log\pi_{\theta_{\mathrm{old}},i}^{\mathrm{joint}}(t)\right).
$$

主方法使用 clipped surrogate：

$$
\mathcal{L}_{\mathrm{clip}}(\theta)=\mathbb{E}_{i,t}\left[\min\left(\rho_{i,t}(\theta)\widehat A_t,\operatorname{clip}\left(\rho_{i,t}(\theta),1-\epsilon_{\mathrm{clip}},1+\epsilon_{\mathrm{clip}}\right)\widehat A_t\right)\right].
$$

value loss、entropy regularization 和总优化目标分别为：

$$
\begin{aligned}
\mathcal{L}_V(\phi)&=\frac12\mathbb{E}_t\left[V_\phi(s_t)-\widehat R_t\right]^2,\\
\mathcal{L}_{\mathrm{total}}(\theta,\phi)&=-\mathcal{L}_{\mathrm{clip}}(\theta)+c_V\mathcal{L}_V(\phi)-c_H\mathbb{E}_{i,t}[\mathcal{H}_i(t)].
\end{aligned}
$$

$\epsilon_{\mathrm{clip}}$、$c_V$、$c_H$、optimizer、learning rate、batch size、rollout length 和 update epochs 均不在本章指定。上式不表示收敛、全局最优或策略优越性保证；它只冻结后续实现必须使用的 MAPPO 更新接口。

## 3.9 Rollout 与 recurrent state contract

### 3.9.1 连续序列存储

由于 actor 含 GRU，rollout buffer 按 episode 内连续时间 chunk 保存，而不是随机打散的独立 timestep。为保证 PPO 更新接口唯一，本工作固定采用显式快照式存储：所有训练所需的 rollout-time actor 输入、critic 输入、recurrent state、proposal action、mask、boundary 信息和 executor metadata 均在采样时直接写入 buffer，训练阶段不得重新运行环境进行重建。

每个 rollout transition 必须保存以下字段：

| 字段 | 作用 | 存储规则 |
|---|---|---|
| `self_obs` | UAV $i$ 自身合法的 slot-start actor observation | 必须显式保存 |
| `neighbor_public_features` | 候选邻居的 public/sanitized node features | 必须显式保存；不得替换为邻居完整私有 observation |
| `edge_features` | actor 可见的历史链路/边特征 | 必须显式保存 |
| `graph_adjacency_or_neighbor_mask` | 恢复采样时实际使用的动态图连接关系 | 必须显式保存，不在 PPO update 时重新从环境推断 |
| `proposal_action[7]` | 七分支 actor proposed action | 必须显式保存 |
| `branch_masks[7]` | 保证更新时各 categorical/Q domain 与采样时一致 | 必须显式保存 |
| `active_branch_indicators[7]` | 重建 active-branch joint log-prob 与 entropy | 必须显式保存 |
| `old_branch_log_prob[7]` | 记录 rollout 时各 active branch 的 proposal log-prob，用于诊断和一致性检查 | 必须显式保存 |
| `old_joint_log_prob` | PPO probability ratio | 必须显式保存；只能来自采样 proposal 时的 masked policy |
| `hidden_in` | actor forward 前实际使用的 recurrent hidden state | 每个 timestep 必须显式保存 |
| `centralized_state` | centralized critic 的决策时刻输入 $s_t$ | 必须显式保存，不在 PPO update 时重新从环境构造 |
| `old_value` | rollout 时的 $V_\phi(s_t)$ | 必须显式保存 |
| `reward` | GAE 与 return | 必须显式保存，直接使用 Section 2 frozen reward |
| `terminated_or_episode_boundary` | return、sequence 截断和 hidden reset | 必须显式保存 |
| `truncated` | 区分固定 horizon bookkeeping 与普通任务完成/过期 | 必须显式保存 |
| `bootstrap_boundary_info` | 按本章既定 no-bootstrap contract 计算 GAE/return | 必须显式保存；terminal 时不得伪造 next state |
| `executed_action_summary` | 记录最终执行通信、资源、功率、CPU 等 compact executor outcome | 必须显式保存；仅作环境元数据 |
| `rejection_or_downgrade_summary` | 记录 executor rejection、half-duplex rejection、power/CPU downgrade 等结果 | 必须显式保存；仅作诊断、审计和实验指标 |

其中，`hidden_in` 对每一个 timestep 都进行显式存储。若一个 recurrent minibatch chunk 从 $t_{\mathrm{start}}$ 开始，则 chunk 初始 hidden state 唯一取自 buffer 中

$$
\mathbf{z}_i^{\mathrm{RNN,in}}(t_{\mathrm{start}}),
$$

不再维护第二套独立的 chunk-initial-hidden 表示。

`executed_action_summary` 和 `rejection_or_downgrade_summary` 属于 executor/environment metadata。它们不得作为 PPO policy action，不得替代 `proposal_action[7]`，不得用于重新计算 `old_joint_log_prob`，也不得用于 PPO probability ratio。

对于团队级 reward 和全局 centralized state，实际内存布局允许每个 slot 只存储一份共享记录，而不在所有 UAV 上重复复制；但其逻辑归属必须保持为该 slot 的唯一 rollout record。

因此，MAPPO rollout buffer 是 rollout 时信息接口的显式 snapshot；本章不保留“保存或确定性重建”二选一实现。
其中，`executed-action summary` 可以记录 $y_{ij}$、$\mathcal{S}_i^{\mathrm{exec}}$、$p_i^{\mathrm{exec}}$、实际 CPU 频率、服务量和实际能耗的接口摘要；它属于环境输出，不属于 actor observation。当前真实 channel、当前真实 SINR 和当前 measurement 若仅用于环境结算，不得为了训练 actor 而写入 actor input。

### 3.9.2 Recurrent minibatch

MAPPO 更新时，minibatch 单位固定为带有 sequence mask 的连续 chunk。若一个 chunk 从时隙 $t_{\mathrm{start}}$ 开始，则其 GRU 初始状态唯一使用 rollout buffer 在该 timestep 显式保存的

$$
\mathbf{z}_i^{\mathrm{RNN,in}}(t_{\mathrm{start}}).
$$

随后按照真实 episode 内的时间顺序依次前向计算 CA-GATv2、GRU 和七个 conditional action heads，并保持冻结的 sampling order 不变。本文固定采用 `NO BURN-IN`，不允许另外重建或估计 chunk 初始 hidden，也不允许将随机独立 timestep 拼接为 recurrent sequence。

若 sequence 中出现 episode boundary，则 sequence mask 在该边界截断 recurrent dependency；边界后的新 episode 使用其自身 buffer 中保存的 `hidden_in`，其 reset 后初始值为全零，不继承上一 episode 的 recurrent state。
## 3.10 Factorized-Action GAT-QMIX baseline

### 3.10.1 基线的编码与分支 Q 值

Factorized-Action GAT-QMIX 与主方法共享环境、actor-visible observation、动态候选图、固定七分支 domain、mask、deterministic executor 和 reward timing，但不共享可训练参数。为控制表示容量差异，基线采用相同的 CA-GATv2 图输入和 GRU 时序接口，再连接 branch-wise Q heads；具体层数和宽度由 Section 4/config 给出。

令 $\mathbf{u}_i^Q(t)$ 为基线 GATv2+GRU 表征，$Q_{i,k}$ 为第 $k$ 个分支的 masked Q head。基线采用不枚举笛卡尔积的 additive factorized utility：

$$
Q_i\left(\widetilde a_i(t)\mid o_i(t)\right)=\sum_{k=1}^{7}b_i^{(k)}(t)Q_{i,k}\left(\widetilde a_i^{(k)}(t)\mid\mathbf{u}_i^Q(t),\widetilde a_i^{(<k)}(t),m_i^{(k)}(t)\right).
$$

inactive branch 的 Q 值不进入 $Q_i$。条件 head 可以使用已经选择的前置分支，但不能创建指数级的 flat joint action table。无效 action 不参与 argmax 或 target max；至少一个 canonical valid action 始终保留。

### 3.10.2 QMIX 单调混合

每个 agent 的 factorized utility 由共享 mixer 与 centralized state 混合：

$$
Q_{\mathrm{tot}}(\widetilde{\mathbf{a}}_t,s_t)=f_{\omega}\left(Q_1(\widetilde a_1(t)),\ldots,Q_N(\widetilde a_N(t));s_t\right).
$$

$s_t$ 只作为训练期 hypernetwork/mixer 的 centralized state 输入，不能扩大 decentralized execution 的 actor observation。mixer 的权重由非负参数化产生，从而冻结 QMIX 的单调性接口：

$$
\frac{\partial Q_{\mathrm{tot}}}{\partial Q_i}\ge0,\qquad \forall i\in\mathcal{U}.
$$

该关系只定义 QMIX 的 monotonic mixing 结构，不宣称全局最优或收敛保证。执行时每个 UAV 根据有效 mask 对七个分支按固定 order 做 decentralized valid argmax，形成 proposal 后仍交给同一个 deterministic executor。

### 3.10.3 序列 replay、target 与 TD 接口

基线使用 recurrent sequence replay，而非独立随机 transition。replay item 至少保存 actor-visible observation/graph、七分支 proposal、masks、active indicators、executed-action summary、reward、critic/mixer state、next-state 信息、episode boundary 和 chunk 初始 GRU hidden。禁止用当前真实 SINR 或未来状态替换上述 actor-visible input。

令 $Q_{\mathrm{tot}}^-$ 为 target encoder、target branch heads 和 target mixer 的组合，$\widetilde{\mathbf{a}}_{t+1}^{\star}$ 为对有效分支逐步 greedy 得到的下一时刻 proposal，固定 horizon 的 TD target 为：

$$
y_t^Q=r_t+\gamma b_t^{\mathrm{boot}}Q_{\mathrm{tot}}^-\left(\widetilde{\mathbf{a}}_{t+1}^{\star},s_{t+1}\right).
$$

当 transition 为 terminated、truncated 或最后 service slot 时，$b_t^{\mathrm{boot}}=0$；不创建虚构的 $s_T$。online factorized utility 和 target utility 的更新顺序、target-update interval、replay size、learning rate、batch size 及 exploration schedule 留给 Section 4。

### 3.10.4 训练与推理策略

QMIX 训练使用 masked epsilon-greedy：以配置的 exploration probability 在当前有效 action 集合中选择，否则选择有效 argmax；探索动作仍必须通过 deterministic executor。QMIX evaluation 使用 masked greedy action，不进行随机探索。该基线不是 PPO 风格 stochastic policy，不使用 PPO 的 log-prob、GAE 或 entropy objective。

## 3.11 训练与分散执行流程

### 3.11.1 CA-GAT-MAPPO 训练算法

**算法 1：CA-GAT-MAPPO recurrent rollout 与更新。**

**输入：** Section 2 环境、共享 actor 参数 $\theta$、共享 centralized critic 参数 $\phi$、固定的 graph/action-mask/executor 接口以及 Section 4 提供的训练配置。

**输出：** 训练后的参数快照、proposal policy、recurrent state contract 和 rollout/evaluation 记录。

1. 初始化共享 actor、critic、优化器和空 rollout buffer；不把环境真值写入 actor encoder。
2. 对每个 episode 执行完整 reset，设置 $t=0$、$\mathbf{z}_i^{\mathrm{RNN}}(-1)=\mathbf{0}$，并初始化 episode boundary mask。
3. 在每个 service slot $t$，先读取槽初 $o_i(t)$、$\mathcal{G}_t$、服务队列表和 branch masks；这些对象只由 Section 2 已允许的信息构成。
4. 用 CA-GATv2 得到 $\mathbf{g}_i(t)$，再用 GRU 得到 $\mathbf{u}_i(t)$。
5. 按 `route → tx_select → resource_group → resource_width → power_level → cpu_queue → cpu_frequency` 依次采样 masked categorical，保存七分支 proposal、branch masks、active indicators、old log-prob、entropy 和 next recurrent state。
6. 将联合 proposal 交给 Section 2 的 raw zero-power canonicalization、确定性仲裁、energy downgrade 和 final canonicalization；actor 不读取 executor-only scalar。
7. 环境用最终 executed action 完成真实信道、SINR、service、actual energy、task settlement、reward、history update 和 slot-end arrival，返回 $r_t$、$s_{t+1}$、终止标志及必要的 transition bookkeeping。
8. 将 proposal、executed summary、critic input/value、reward、boundary/truncation 和 recurrent hidden 写入连续 rollout chunk；若到达 boundary，清零下一 episode 的 hidden。
9. rollout 达到配置的更新边界后，以 contiguous sequence minibatch 计算 no-bootstrap GAE、return、PPO ratio、clipped objective、value loss 和 active-branch entropy，并更新 $\theta,\phi$。
10. 所有训练更新完成后，进入下一 rollout；评估阶段切换到 3.11.2 的 masked greedy contract，不改变环境或 executor 语义。

该算法描述训练顺序，不构成实验结果或收敛证明。episode 数、rollout 长度、更新 epoch、batch size、学习率和所有网络 numeric hyperparameters 均由 Section 4 冻结。

### 3.11.2 Training / evaluation inference contract

训练和评估只在采样规则上不同，输入和 executor 保持一致：

| 阶段 | actor/Q 网络输出规则 | executor | 随机性 |
|---|---|---|---|
| MAPPO training | masked categorical sampling，按固定七分支顺序 | 同一 deterministic executor | 仅来自策略采样和环境既有随机过程 |
| MAPPO evaluation | 每个有效 branch 取 masked argmax，仍按固定顺序 | 同一 deterministic executor | 不进行策略采样 |
| QMIX training | 有效动作集合上的 epsilon-greedy | 同一 deterministic executor | 仅使用配置的 exploration 与环境既有随机过程 |
| QMIX evaluation | 有效动作集合上的 greedy argmax | 同一 deterministic executor | 不进行 exploration |

评估不能把 proposal 直接当作 executed action，也不能绕过 half-duplex、energy downgrade、outage 或 slot-end settlement。所有方法共享 Section 2 的时间边界和 reward timing；Section 4 再定义场景、seed、baseline 组合和统计方法。

## 3.12 Implementation Interface Summary

本节将前述语义压缩为下一阶段 Python/PyTorch 实现可以直接采用的模块契约。接口中的维度符号由配置注入，不在本章硬编码具体数值。

### 3.12.1 CA-GAT-MAPPO actor

| 接口项 | 冻结内容 |
|---|---|
| Input: observation | 每个 UAV 的 actor-visible slot-start observation；包含队列、任务、能量、邻居消息、stale CSI、历史干扰、per-RU historical quality、AoI、arrival estimate 与 masks；不包含 current true physical quantities |
| Input: graph | 节点集合 $\mathcal{U}$、当前候选邻居有向边 $\mathcal{E}_t$、edge features、edge masks，以及已绑定服务队列表的固定 UAV-ID 索引 |
| Input: masks | 七个 branch 的 slot-start/conditional action masks；每个 mask 至少保留一个 canonical valid action |
| Input: recurrent state | 每个 timestep 显式保存并输入 actor forward 前的 $\mathbf{z}_i^{\mathrm{RNN,in}}(t)$；recurrent chunk 从 $t_{\mathrm{start}}$ 开始时，初始 hidden 唯一取 buffer 中保存的 $\mathbf{z}_i^{\mathrm{RNN,in}}(t_{\mathrm{start}})$；同时使用 episode boundary mask，固定 `NO BURN-IN` |
| Internal | 固定 CA-GATv2 → GRU → 七个 conditional masked heads；sampling order 唯一固定 |
| Output: proposal action | 七分支 `route`、`tx_select`、`resource_group`、`resource_width`、`power_level`、`cpu_queue`、`cpu_frequency` proposal；不输出 executed action |
| Output: probability | 七个 active-branch log-prob、每个 UAV 的 joint log-prob、active-branch entropy |
| Output: recurrent state | 下一槽 $\mathbf{z}_i^{\mathrm{RNN}}(t)$；episode boundary 后不得跨 episode 复用 |

### 3.12.2 Executor / environment adapter

| Input | Output |
|---|---|
| 槽初环境状态、联合 proposal、固定资源组映射、历史质量/mask、确定性排序规则和能量约束 | $\mathcal{S}_i^{\mathrm{prop}}$、$p_i^{\mathrm{prop}}$、executor-only scalar、最终 $y_{ij}$、$x_{ij,r}$、$\mathcal{S}_i^{\mathrm{exec}}$、$p_i^{\mathrm{exec}}$、CPU frequency、service、actual energy、outage、post-settlement reward 和 $s_{t+1}$ |

adapter 不向 actor 返回当前执行器 scalar、current SINR 或 current measurement；这些量只用于环境结算、critic 的决策时刻 true state（若该量属于 $s_t$）和 rollout 审计。

### 3.12.3 Centralized critic

| Input | Output |
|---|---|
| 决策时刻的 centralized state $s_t$ 或其确定性全局编码；可含 all-agent state、true channel、queue/task truth、energy 和 executor-relevant pre-action state，不含 future state 或 post-action outcome | 共享标量 $V_\phi(s_t)$ |

### 3.12.4 MAPPO rollout buffer

MAPPO 采用显式快照式 on-policy rollout buffer。所有 PPO 更新所需的 rollout-time actor 输入、critic 输入、recurrent state、proposal action、mask、boundary 信息和 executor metadata 均在采样时直接保存；训练阶段不得重新运行环境重建这些字段。

每个 agent-level rollout record 固定包含：

```text
self_obs
neighbor_public_features
edge_features
graph_adjacency_or_neighbor_mask

proposal_action[
    route,
    tx_select,
    resource_group,
    resource_width,
    power_level,
    cpu_queue,
    cpu_frequency
]

branch_masks[7]
active_branch_indicators[7]

old_branch_log_prob[7]
old_joint_log_prob

hidden_in
```

其中，`hidden_in` 表示该 timestep actor forward 之前实际使用的

$$
\mathbf{z}_i^{\mathrm{RNN,in}}(t).
$$

若 recurrent chunk 从 $t_{\mathrm{start}}$ 开始，则 chunk initial hidden 唯一取为

$$
\mathbf{z}_i^{\mathrm{RNN,in}}(t_{\mathrm{start}}),
$$

即 buffer 在该位置显式保存的 `hidden_in`。本文固定采用 `NO BURN-IN`，不再维护第二套 chunk-initial-hidden 表示。

每个 slot-level rollout record 固定包含：

```text
centralized_state
old_value
reward
terminated_or_episode_boundary
truncated
bootstrap_boundary_info

executed_action_summary
rejection_or_downgrade_summary
```

其中，`centralized_state` 是决策时刻 critic 实际使用的 $s_t$，`old_value` 是 rollout 时保存的 $V_\phi(s_t)$。这些字段不得在 PPO update 阶段通过重新运行环境进行重建。

`executed_action_summary` 与 `rejection_or_downgrade_summary` 必须显式保存，但仅属于 executor/environment metadata，用于环境一致性检查、诊断、调试和实验指标统计。它们：

- 不得作为 PPO policy action；
- 不得替代七分支 `proposal_action`；
- 不得用于重新计算 `old_joint_log_prob`；
- 不得用于 PPO probability ratio。

对于团队级 reward 和 centralized state，实际内存布局允许每个 slot 只保存一份共享记录，不必在所有 UAV 上重复复制；但其逻辑归属保持为该 slot 的唯一 rollout record。

因此，MAPPO rollout buffer 的正式接口为：

$$
\boxed{
\text{rollout-time explicit snapshot}
\rightarrow
\text{contiguous recurrent PPO update}
}
$$

本章不保留“保存或重建”“previous hidden 或 chunk initial hidden”等二选一实现。

### 3.12.5 Factorized-Action GAT-QMIX


| 接口项 | 冻结内容 |
|---|---|
| Input | 与主方法相同的 actor-visible observation、动态图、masks、active indicators、recurrent state；训练 mixer 另接当前 centralized state $s_t$ |
| Encoder | CA-GATv2 + GRU 的 agent-local representation；参数与主方法独立，capacity 由 Section 4 配置 |
| Branch output | 七个 masked branch-wise Q heads，保持同一 fixed domain、sampling/selection order 和 inactive canonical semantics |
| Local utility | $Q_i=\sum_k b_i^{(k)}Q_{i,k}$，不枚举 flat joint action |
| Mixer | $Q_{\mathrm{tot}}=f_\omega(Q_1,\ldots,Q_N;s_t)$；通过 non-negative mixing weights 保持 $\partial Q_{\mathrm{tot}}/\partial Q_i\ge0$ |
| Replay | recurrent contiguous sequence replay，保存 proposal、masks、active indicators、reward、boundary、hidden、next-state 和 executed summary |
| Target | target encoder、target branch heads、target mixer；使用 masked valid greedy next proposal 和 no-bootstrap boundary contract |
| Training / evaluation | training 为 masked epsilon-greedy，evaluation 为 masked greedy；二者均通过同一 deterministic executor |

### 3.12.6 本章不冻结的数字项

以下内容只在 Section 4/config 冻结：hidden dimension、GAT 层数与 head 数、GRU dimension、learning rate、$\gamma$、GAE $\lambda$、PPO clip epsilon、entropy/value coefficient、batch size、rollout length、update epochs、epsilon schedule、target-update interval、replay size、训练 episode 数、seed 和场景数量。它们的延后不影响本章已经冻结的输入、输出、mask、branch dependency、proposal/execution boundary、rollout、target 和 evaluation inference contract。
