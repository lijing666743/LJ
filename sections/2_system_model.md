# 2 系统模型与问题建模

## 2.1 系统概述

​      本文考虑由 $N$ 架 UAV 组成的动态异构多 UAV U2U-MEC 系统。UAV 既可以产生任务，也可以作为 U2U 发送节点、接收节点、本地计算节点或远程协同计算节点。系统在外生移动、动态候选邻居、资源异构、业务负载异构、建筑物遮挡、同频干扰、不完美 CSI、跨时隙队列和任务截止期共同存在的条件下运行。研究对象是单跳协同卸载与资源编排：任务只能从源 UAV 直接传输到一个目的 UAV，不能拆分到多个目的地，也不允许经由第三架 UAV 多跳转发。

​      系统时间被离散为长度为 $\Delta t$ 的时隙，时隙 $t$ 对应连续时间区间 $[t\Delta t,(t+1)\Delta t)$。槽初表示该区间开始的边界，槽内只对槽初已经存在的队列执行传输和计算服务，槽末表示本槽服务完成后的边界并承载到达、绑定、转队和完成等事件。新任务在时隙 $t_n^{\mathrm{arr}}$ 的槽末生成并到达，记录其到达事件所属时隙索引 $t_n^{\mathrm{arr}}$，因此在 $t_n^{\mathrm{arr}}+1$ 的槽初才首次进入 actor 可见的未绑定队列并允许 route；本槽 route 决策在槽末绑定目的地，任务从下一时隙起才可接受传输或 CPU 服务。传输完成任务同样在槽末转入 CPU 队列，并在下一时隙首次计算。该安排保留传输中断、排队等待、CPU 竞争和截止期违约的端到端因果关系。任务结果回传暂不单独建模，因此远程计算完成表示任务服务过程结束；若后续研究需要刻画结果数据回传，可在主模型通过附加的反向服务队列扩展，但不改变本章的单跳输入卸载接口。

​      本章的主模型包括 UAV 状态、任务生命周期、EDF 队列、外生移动、轻量 U2U 物理层评价、固定资源组、跨时隙传输和计算服务、主动通信及 CPU 能耗、七分支离散动作、半双工与节点独占无线调度的联合解析和 Dec-POMDP 信息边界。第三 UAV 软遮挡、高保真传播回放、UAV 数量 OOD 和离散硬件类型属于主实验通过后再考虑的可选扩展；连续资源动作同样不进入主模型。它们不改变主模型中任务不可多目的地拆分、单跳、固定高度、固定资源组和槽边界等冻结规则。

​      本章刻意限定建模声明边界。该模型不是完整飞控系统，不包含六自由度飞行动力学、旋翼推力、姿态控制或飞控回路；不是完整通信协议栈，不包含包级重传、HARQ、随机接入和控制信令细节；不是实测信道、NS-3 包级联动或实时射线追踪系统。无线部分属于 modeled physical-layer evaluation：它用几何遮挡、路径损耗、相关阴影、槽级衰落、不完美 CSI 和同频干扰生成可解释的链路评价量，用于比较卸载和资源编排行为，而不宣称替代实测或高保真物理层仿真。

## 2.2 符号与基本定义

​      系统采用离散时隙、有限 UAV 集合和显式任务记录。下表集中给出跨小节重复使用的符号；首次出现时仍在局部文字中补充其物理含义和单位。

| 分组 | 符号 | 含义 | 单位或定义域 |
|---|---|---|---|
| UAV 网络 | $\mathcal U$ | UAV 集合 | $\{1,\ldots,N\}$ |
| UAV 网络 | $i,j,k$ | UAV 索引 | $\mathcal U$ |
| UAV 网络 | $N$ | UAV 数量 | 正整数 |
| UAV 网络 | $\mathcal N_i(t)$ | UAV $i$ 在时隙 $t$ 的新任务候选邻居集合 | $\mathcal U\setminus\{i\}$ 的子集 |
| 时间与空间 | $t$ | 时隙索引 | $\{0,\ldots,T-1\}$ |
| 时间与空间 | $\Delta t$ | 时隙长度 | s |
| 时间与空间 | $T$ | episode 时隙数 | 正整数 |
| 时间与空间 | $\mathbf p_i(t)$ | 水平位置 | $\mathrm{m}^2$ |
| 时间与空间 | $\mathbf q_i(t)$ | 三维 UAV 位置 | $\mathrm{m}^3$ |
| 时间与空间 | $\mathbf v_i(t)$ | 三维速度 | $\mathrm{m/s}$ |
| 时间与空间 | $H$ | 固定飞行高度 | m |
| 时间与空间 | $\mathcal R$ | 主场景的矩形水平移动区域 | $[x_{\min},x_{\max}]\times[y_{\min},y_{\max}]$ |
| 时间与空间 | $\mathcal R_{\mathrm{free}}(H)$ | 固定高度 $H$ 下排除有限高度建筑实体后的可行水平飞行区域 | $\mathcal R\setminus\bigcup_{b\in\mathcal B:\,H\le H_b}\mathcal P_b$ |
| 时间与空间 | $d_{ij}(t)$ | UAV $i$ 与 $j$ 的三维距离 | m |
| UAV 平台 | $f_i^{\max}$ | UAV $i$ 最大 CPU 频率 | cycle/s |
| UAV 平台 | $P_i^{\max}$ | UAV $i$ 最大通信发射功率 | W |
| UAV 平台 | $E_i^0$ | episode 初始主动能量预算 | J |
| UAV 平台 | $E_i^{\mathrm{res}}(t)$ | 剩余主动能量 | J |
| UAV 平台 | $\kappa_i$ | CPU 动态能耗系数 | 由 $E=\kappa f^3\tau$ 定义 |
| 任务与队列 | $\tau_n$ | 任务 $n$ 的完整记录 | 记录元组 |
| 任务与队列 | $D_n,C_n$ | 任务输入数据量和计算量 | bit，cycle |
| 任务与队列 | $D_n^{\mathrm{rem}},C_n^{\mathrm{rem}}$ | 剩余待传输 bit 和待执行 cycle | bit，cycle |
| 任务与队列 | $t_n^{\mathrm{arr}}$ | 槽末到达事件所属的时隙索引 | slot |
| 任务与队列 | $t_n^{\mathrm{cmp}}$ | 完成发生并在槽末记录的服务时隙索引 | slot |
| 任务与队列 | $t_n^{\mathrm{ddl}}$ | 最后允许接受传输或计算服务的时隙索引 | slot |
| 任务与队列 | $L_n^{\mathrm{ddl}}$ | clip 后的相对 deadline 时隙预算 | slot |
| 任务与队列 | $d_n^{\mathrm{dst}}$ | 任务当前绑定的执行目的 UAV | $\mathcal U\cup\{\varnothing\}$ |
| 任务与队列 | $s_n(t)$ | 任务生命周期状态 | 见 2.5 |
| 任务与队列 | $Q_i^{\mathrm{unb}}$ | UAV $i$ 的未绑定任务队列 | 任务列表 |
| 任务与队列 | $Q_i^{\mathrm{loc}}$ | UAV $i$ 的本地待计算任务队列 | 任务列表 |
| 任务与队列 | $Q_{i\rightarrow j}^{\mathrm{tx}}$ | 从 $i$ 到 $j$ 的已绑定传输队列 | 任务列表 |
| 任务与队列 | $Q_{i\rightarrow j}^{\mathrm{cpu}}$ | 来源为 $i$、在 $j$ 上执行的 CPU 队列 | 任务列表 |
| 任务与队列 | $\operatorname{slack}_n(t)$ | 任务截止期余量 | slot |
| 仲裁与确定性 | $\Pi_{ij}^{\mathrm{tx}}(t)$ | 候选通信边的固定字典序仲裁键 | $(\text{slack},-\text{quality},i,j)$ |
| 业务到达 | $A_i(t)$ | UAV $i$ 在时隙 $t$ 槽末生成并到达的任务数 | 任务数；Bernoulli 情况为 $\{0,1\}$ |
| 业务到达 | $\lambda_i$ | UAV $i$ 的环境任务到达概率或按时隙计的强度 | task/slot；Bernoulli 概率为无量纲 |
| 业务到达 | $\hat{\lambda}_i(t)$ | 时隙 $t$ 槽初基于已结束时隙到达量形成的历史到达率估计 | task/slot |
| 业务到达 | $W_\lambda$ | 历史到达率估计的最大窗口长度 | slot |
| 业务到达 | $K_\lambda(t)$ | 时隙 $t$ 槽初实际可用的历史样本数 | 样本数 |
| 业务到达 | $m_i^\lambda(t)$ | 历史到达率估计的可用性 mask | $\{0,1\}$ |
| 无线信道 | $f_c$ | 载频 | Hz |
| 无线信道 | $B^{\mathrm{tot}}$ | 系统总通信带宽 | Hz |
| 无线信道 | $B_{\mathrm{RU}}$ | 单个等效资源单元带宽 | Hz |
| 无线信道 | $N_0$ | 线性热噪声功率谱密度 | W/Hz |
| 无线信道 | $F_{\mathrm{dB}}$ | 接收机噪声系数 | dB |
| 无线信道 | $F$ | 线性 noise factor | 无量纲 |
| 无线信道 | $P_{\mathrm{noise,RU}}$ | 单个等效资源单元的接收噪声功率 | W |
| 无线信道 | $\mathcal B$ | 建筑物集合 | 有限高度三维闭棱柱集合 |
| 无线信道 | $\mathcal P_b$ | 建筑物 $b$ 的闭合水平足迹 | $\mathcal P_b\subseteq\mathcal R$ |
| 无线信道 | $H_b$ | 建筑物 $b$ 的有限高度 | m |
| 无线信道 | $\mathcal B_b$ | 建筑物 $b$ 的有限高度闭棱柱实体 | $\mathcal P_b\times[0,H_b]$ |
| 无线信道 | $I_{ij}^{\mathrm{bld}}(t)$ | 几何遮挡指示量 | $\{0,1\}$ |
| 无线信道 | $L_{\mathrm B}$ | 轻量几何遮挡抽象中的场景级名义额外建筑遮挡损耗参数 | dB |
| 无线信道 | $PL_{ij}(t)$ | $i\rightarrow j$ 路径损耗 | dB |
| 无线信道 | $\Delta s_{ij}(t)$ | 两端 UAV 在相邻时隙间的平均水平位移；episode reset 时 $\Delta s_{ij}(0)=0$ | m |
| 无线信道 | $X_{ij}(t)$ | 相关阴影项 | dB |
| 无线信道 | $h_{ij,r}(t)$ | 资源单元 $r$ 上的复信道增益 | 复数 |
| 无线信道 | $\hat h_{ij,r}(t)$ | actor 可用的 CSI 估计 | 复数 |
| 无线信道 | $a_{ij}^{\mathrm{CSI}}(t)$ | CSI AoI | slot |
| 无线信道 | $\epsilon_{ij,r}^{\mathrm{CSI}}(t)$ | 资源单元 $r$ 上的 CSI 功率增益误差样本 | dB |
| 无线信道 | $\widehat I_{j,r}^{\mathrm{hist}}(t)$ | 接收 UAV $j$ 的历史干扰摘要 | W |
| 无线信道 | $\ell_{ij}^{\mathrm{CSI}}(t)$ | 陈旧 CSI 的原始历史索引，$t-a_{ij}^{\mathrm{CSI}}(t)$ | 整数，可为负 |
| 无线信道 | $m_{ij}^{\mathrm{CSI}}(t)$ | 陈旧 CSI 历史索引的可用性 mask | $\{0,1\}$ |
| 无线信道 | $\widehat Z_{j,r}^{\mathrm{hist}}(t)$ | 历史干扰加噪声摘要 | W |
| 无线信道 | $I_{ij,r}^{\mathrm{meas}}(t)$ | 当前已执行接收边 $i\rightarrow j$ 在资源单元 $r$ 上的 interference-only 测量；写入接收端历史时记作 $I_{j,r}^{\mathrm{meas}}(t)$ | W |
| 无线信道 | $Q_{ij}^{\mathrm{bit}}(t)$ | 槽初链路 $i\rightarrow j$ 传输队列中的剩余待传输 bit 总量 | bit |
| 无线信道 | $m_{ij,r}^{I,\mathrm{meas}}(t)$ | 当前已执行接收边 $i\rightarrow j$ 在资源单元 $r$ 上的测量可用性 mask | $\{0,1\}$ |
| 无线信道 | $m_{j,r}^{I,\mathrm{meas}}(t)$ | 当前接收端资源单元测量的可用性 mask | $\{0,1\}$ |
| 无线信道 | $\widehat\Gamma_{ij}^{\mathrm{hist}}(t)$ | actor proposal 形成后，在当前 proposed resource set 上由 valid per-RU 历史质量代理量得到的 executor-only scalar | 线性值 |
| 无线信道 | $\widehat\Gamma_{ij,r}^{\mathrm{hist}}(t)$ | 历史干扰链路质量代理量 | 线性值 |
| 无线信道 | $p_r^{\mathrm{ref}}$ | 固定参考每资源单元功率，$P_{\mathrm{ref}}/R$ | W |
| 无线信道 | $\beta_I$ | 历史干扰指数滑动平均系数 | $[0,1)$ |
| 无线信道 | $I_{j,r}^{\mathrm{def}}$ | episode 初始历史干扰缺省值 | W |
| 无线信道 | $a_j^{\mathrm{msg}}(t)$ | 接收 UAV $j$ 广播摘要的消息 AoI | slot |
| 无线信道 | $m_{j,r}^{I}(t)$ | 历史干扰摘要可用性 mask | $\{0,1\}$ |
| 无线信道 | $\mathrm{SINR}_{ij,r}(t)$ | 资源单元级信干噪比 | 线性值 |
| 无线信道 | $\gamma_{\min}^{\mathrm{dB}}$ | 无线 SINR 门限的 dB 表示 | dB |
| 无线信道 | $\gamma_{\min}^{\mathrm{lin}}$ | 由 dB 门限转换得到的线性 SINR 门限 | 线性无量纲值 |
| 无线信道 | $R_{ij}^{\mathrm{eff}}(t)$ | 链路有效速率 | bit/s |
| outage 统计 | $\chi_{ij}^{\mathrm{att}}(t)$ | 实际传输尝试指示量 | $\{0,1\}$ |
| outage 统计 | $o_{ij}(t)$ | 单次实际传输尝试的 outage 样本 | $\{0,1,\mathrm{NA}\}$ |
| outage 统计 | $\mathcal A^{\mathrm{att}}$ | 实际传输尝试样本集合 | 时隙、发送端和接收端三元组集合 |
| outage 统计 | $\widehat q_{ij}^{\mathrm{out}}(t)$ | 仅由历史实际传输尝试计算的 outage 特征 | $[0,1]$ 或缺省值 |
| outage 统计 | $m_{ij}^{\mathrm{out}}(t)$ | 历史 outage 特征可用性 mask | $\{0,1\}$ |
| outage 统计 | $\mathrm{NA}$ | 没有实际传输尝试时的不可用标记，不参与数值统计 | 非数值标记 |
| 资源与能量 | $R$ | 等效资源单元总数 | 正整数 |
| 资源与能量 | $G$ | 固定资源组数 | $G=5$ |
| 资源与能量 | $\mathcal G_g$ | 第 $g$ 个固定资源组 | 等效资源单元集合 |
| 资源与能量 | $\mathcal K_{\mathrm{RU}}$ | 等效资源单元索引集合 | $\{1,\ldots,R\}$ |
| 资源与能量 | $g_i(t)$ | 资源组构造索引，满足 $g_i(t)\equiv a_i^{\mathrm{rg}}(t)$ | $\{\mathrm{idle},1,\ldots,5\}$；仅 active 时表示组索引 |
| 资源与能量 | $w_i(t)$ | 宽度构造变量，满足 $w_i(t)\equiv a_i^{\mathrm{width}}(t)$ | $\{1,2\}$；inactive 时使用 canonical 值 $1$ |
| 资源与能量 | $m_i^{\mathrm{width,act}}(t)$ | 宽度分支 activity mask | $\{0,1\}$ |
| 资源与能量 | $\mathcal S_i^{\mathrm{prop}}(t)$ | actor 动作提案为发送 UAV $i$ 申请的资源单元集合 | 资源单元集合 |
| 资源与能量 | $\mathcal S_i^{\mathrm{exec}}(t)$ | 联合执行器仲裁后发送 UAV $i$ 实际占用的资源单元集合 | 资源单元集合 |
| 资源与能量 | $x_{ij,r}(t)$ | 链路 $i\rightarrow j$ 对资源单元 $r$ 的最终实际占用指示 | $\{0,1\}$ |
| 资源与能量 | $p_i^{\mathrm{prop}}(t)$ | actor 提出的总发射功率 | W |
| 资源与能量 | $p_i^{\mathrm{cand}}(t;\ell)$ | executor 在第 $\ell$ 个降档检查阶段使用的临时候选功率 | W；仅 executor 内部变量 |
| 资源与能量 | $p_i^{\mathrm{exec}}(t)$ | executor 确定的最终实际总发射功率 | W |
| 资源与能量 | $p_{ij,r}(t)$ | 链路在资源单元 $r$ 上的最终实际功率 | W |
| 资源与能量 | $\tau_{ij}^{\mathrm{tx}}(t)$ | 链路 $i\rightarrow j$ 的通信实际活跃时间 | s |
| 资源与能量 | $\tau_i^{\mathrm{cpu}}(t)$ | UAV $i$ 的 CPU 实际活跃时间 | s |
| 资源与能量 | $\overline\tau_i^{\mathrm{cpu}}(t)$ | CPU 预留实际活跃时间上界 | s |
| 资源与能量 | $E_i^{\mathrm{tx}}(t)$ | 主动通信能耗 | J |
| 资源与能量 | $E_i^{\mathrm{cpu}}(t)$ | CPU 动态能耗 | J |
| 资源与能量 | $E_i^{\mathrm{act}}(t)$ | 主动通信与计算总能耗 | J |
| 资源与能量 | $\overline E_i^{\mathrm{tx,cand}}(t;\ell)$ | 基于候选功率的通信预留能量上界 | J；executor 临时检查量 |
| 资源与能量 | $\overline E_j^{\mathrm{cpu}}(t)$ | CPU 预留能量上界 | J |
| 资源与能量 | $\overline E_i^{\mathrm{act,cand}}(t;\ell)$ | 基于候选功率的联合主动能量预留上界 | J；executor 临时检查量 |
| 动作接口 | $a_i(t)$ | UAV $i$ 的联合动作 | 七元组 |
| 动作接口 | $a_i^{\mathrm{route}}(t)$ | 新任务路由分支 | 离散候选集合 |
| 动作接口 | $a_i^{\mathrm{tx}}(t)$ | 已有传输队列选择分支 | 离散候选集合 |
| 动作接口 | $a_i^{\mathrm{rg}}(t)$ | 固定资源组分支 | $\{\mathrm{idle},1,\ldots,5\}$ |
| 动作接口 | $a_i^{\mathrm{width}}(t)$ | 相邻资源组宽度分支 | $\{1,2\}$ |
| 动作接口 | $a_i^{\mathrm{pow}}(t)$ | 通信功率档位分支 | $\{0,0.25,0.5,1\}$ |
| 动作接口 | $a_i^{\mathrm{cpuq}}(t)$ | CPU 队列分支 | 本地、远程或 idle |
| 动作接口 | $a_i^{\mathrm{cpuf}}(t)$ | CPU 频率档位分支 | $\{0,0.25,0.5,1\}$ |
| Dec-POMDP | $s_t$ | 全局系统状态 | 状态空间 $\mathcal S$ |
| Dec-POMDP | $o_i(t)$ | UAV $i$ 的局部观测 | 观测空间 $\mathcal O_i$ |
| Dec-POMDP | $\mathbf a_t$ | 联合动作 | $\prod_i\mathcal A_i$ |
| Dec-POMDP | $P$ | 状态转移核 | $\mathcal S\times\mathcal A\rightarrow\mathcal P(\mathcal S)$ |
| Dec-POMDP | $r_t$ | 团队即时奖励 | 实数 |
| Dec-POMDP | $\gamma$ | 折扣因子 | $(0,1)$ |
| Dec-POMDP | $\Pi_{\mathrm{hard}}$ | actor 策略、动作 mask 与确定性执行器共同诱导的闭环硬可行策略集合 | 策略集合 |
| Dec-POMDP | $J_{\mathrm{RL}}(\boldsymbol{\pi})$ | 折扣团队奖励对应的 RL 训练替代目标 | 实数 |
| 性能指标 | $G_T,C_T,X_T,Z_T$ | episode 内生成、按时完成、过期和截断任务数 | task |
| 性能指标 | $\mathcal C_T$ | episode 内按时完成的任务集合 | 任务集合，$\lvert\mathcal C_T\rvert=C_T$ |
| 性能指标 | $\Lambda_{\mathrm{succ}}(\boldsymbol{\pi})$ | 期望按时成功任务吞吐量 | task/slot |
| 性能指标 | $\eta_{\mathrm{cmp}},\eta_{\mathrm{exp}},\eta_{\mathrm{trunc}}$ | 完成、过期和截断任务比例 | $[0,1]$ 或 $\mathrm{NA}$ |
| 性能指标 | $\overline T_{\mathrm{E2E}}$ | done 任务的平均端到端时延 | s 或 $\mathrm{NA}$ |
| 性能指标 | $E_{\mathrm{tot}}^{\mathrm{act}}$ | episode 内实际主动通信与计算总能耗 | J |

任务目的地字段统一定义为：

$$
d_n^{\mathrm{dst}}(t)
=
\begin{cases}
\varnothing, & \text{任务尚未绑定，即处于 unbound 状态},\\
i_n, & \text{任务已绑定在源 UAV 本地执行},\\
j,\ j\ne i_n, & \text{任务已绑定到远程 UAV }j\text{ 执行}.
\end{cases}
$$

因此，$\varnothing$ 仅表示任务尚未绑定目的地；local 任务和 remote 任务的目的 UAV 分别为源 UAV $i_n$ 和远程 UAV $j\ne i_n$，字段定义域仍为 $\mathcal U\cup\{\varnothing\}$。

route 决策在槽末生效并使任务进入 local 或 tx 队列时，任务才完成目的地绑定；绑定后目的地字段保持不变。

​      默认参数用于编码初始接口和 Gate P 数值可行性检查前的可复现实验起点。它们不等同于已完成的物理层标定结果。

| 参数 | 编码初始值 | 单位 | 状态 |
|---|---:|---|---|
| $N$ | $6$ | 架 | 敏感性参数，主规模集合为 $\{4,6,8\}$ |
| 区域边长 | $1000\times1000$ | m | 敏感性参数 |
| $H$ | $80$ | m | 已冻结 |
| $\Delta t$ | $20$ | ms | 已冻结 |
| $T$ | $500$ | slot | 敏感性参数 |
| $f_c$ | $3.5$ | GHz | Gate P 待校准 |
| $B^{\mathrm{tot}}$ | $20$ | MHz | Gate P 待校准 |
| $B_{\mathrm{RU}}$ | $B^{\mathrm{tot}}/R=1$ | MHz | 由等宽 RU 划分得到 |
| $R$ | $20$ | 等效资源单元 | Gate P 待校准 |
| $G$ | $5$ | 组 | 已冻结 |
| $P_{\mathrm{ref}}$ | $1$ | W | Gate P 待校准 |
| $f_{\mathrm{ref}}$ | $2.0\times10^9$ | cycle/s | Gate P 待校准 |
| $\kappa_{\mathrm{ref}}$ | $1.0\times10^{-28}$ | 由能耗式确定 | Gate P 待校准 |
| $E_{\mathrm{ref}}^0$ | $30$ | J | Gate P 待校准 |
| $R_{\mathrm{ref}}$ | $20$ | Mbit/s | Gate P 待校准 |
| 天线增益 | $2$ | dBi | Gate P 待校准 |
| $F_{\mathrm{dB}}$ | $7$ | dB | Gate P 待校准 |
| $F$ | $10^{F_{\mathrm{dB}}/10}$ | 无量纲 | 由 dB 噪声系数转换得到 |
| $\gamma_{\min}^{\mathrm{dB}}$ | $-3$ | dB | 敏感性参数 |
| $\gamma_{\min}^{\mathrm{lin}}$ | $10^{-3/10}\approx0.5012$ | 线性无量纲值 | 由 dB 门限转换 |
| $\sigma_s$ | $4$ | dB | 敏感性参数 |
| $d_{\mathrm{corr}}$ | $50$ | m | 敏感性参数 |
| $L_{\mathrm B}$ | $15$ | dB | 名义主设置；$\{10,15,20,30\}$ 用于宽范围敏感性分析 |
| $R_{\mathrm{cand}}$ | $500$ | m | Gate P 待校准 |
| $a^{\mathrm{CSI}}$ | $0$ | slot | 敏感性参数，主场景可设为 $\{0,3,5\}$ |

​      “已冻结”表示本章不得改变其语义；“Gate P 待校准”在本章仅表示通过仿真链路预算、deadline 和能量 pilot 进行数值可行性检查与仿真尺度冻结，不表示现场测量、射线追踪或物理传播模型标定；“敏感性参数”表示在主模型结构不变的前提下考察结论对数值设置的依赖性，在实验结果产生前不据此声称鲁棒性。

## 2.3 UAV平台与节点功能模型

### 2.3.1 UAV空间状态

UAV 采用固定高度的三维位置表示。令 $\mathbf p_i(t)=[x_i(t),y_i(t)]^{\mathrm T}$ 为水平位置，则完整位置定义为：

$$
\mathbf q_i(t)
=
\left[
x_i(t),
y_i(t),
H
\right]^{\mathrm T}.
$$

其中 $x_i(t)$ 和 $y_i(t)$ 的单位为 m，$H$ 为固定高度，$\mathbf q_i(t)$ 用于后续距离、建筑物几何和信道评价。高度不随时隙变化，因此移动模块只更新水平位置。

对应的三维速度为：

$$
\mathbf v_i(t)
=
\left[
v_i^x(t),
v_i^y(t),
0
\right]^{\mathrm T}.
$$

其中 $v_i^x(t)$ 和 $v_i^y(t)$ 的单位为 m/s，竖直速度恒为零。该速度由外生移动模型产生，不是 actor 的动作分支，也不参与轨迹优化。

任意两架 UAV 之间的三维欧氏距离定义为：

$$
d_{ij}(t)
=
\left\|
\mathbf q_i(t)-\mathbf q_j(t)
\right\|_2.
$$

该距离统一用于候选邻居筛选、建筑物路径几何和链路大尺度评价。因而后续通信模块不将水平距离替代为完整的三维距离。

### 2.3.2 机载计算、通信、能量与队列模块

每架 UAV 由四个相互关联但功能可区分的模块组成：机载 CPU 模块、U2U 无线通信模块、主动通信和计算能量模块，以及任务与队列管理模块。CPU 模块服务本地队列或远程来源队列；无线模块服务已绑定的单跳传输链路；能量模块扣除通信发射和 CPU 动态运行产生的主动能耗；队列管理模块负责任务状态、目的地锁定、EDF 排序和槽边界更新。

UAV 的资源能力向量定义为：

$$
\boldsymbol{\chi}_i^{\mathrm{res}}
=
\left[
f_i^{\max},
P_i^{\max},
E_i^0,
\kappa_i
\right].
$$

其中，$f_i^{\max}$ 的单位是 cycle/s，表示 CPU 的最大等效处理频率；$P_i^{\max}$ 的单位是 W，表示无线发射功率上限；$E_i^0$ 的单位是 J，表示一个 episode 可用于主动通信和计算的初始能量预算；$\kappa_i$ 是 CPU 动态能耗系数，使得频率越高时单位时间能耗按三次方增长。该向量描述 UAV 本身的资源异构性，不包含业务到达率。

### 2.3.3 UAV节点角色与并行服务能力

同一架 UAV 的角色由当前任务和队列状态动态决定，而不是由固定类型标签预先决定。它可以同时承担任务源节点、U2U 发送节点、U2U 接收节点、本地计算节点和远程协同计算节点。CPU 模块与无线模块允许并行工作，因此一架 UAV 可以在同一时隙执行 CPU 服务，并作为发送端或接收端参与一条无线传输链路；两类服务仅通过既有的联合主动能量可行性约束发生耦合。

无线收发机采用半双工模式，即同一 UAV 在同一时隙不得同时发送和接收。在此基础上，主模型另行采用节点独占（node-exclusive）的一链路 MAC 调度抽象：每架 UAV 每个时隙至多关联一条最终接受的 U2U 无线服务边。该节点独占规则是附加的建模假设，并非半双工定义本身的数学推论。下文将半双工模式与节点独占无线调度的组合简称为“联合半双工规则”。每个计算 UAV 每槽至多选择一个 CPU 队列。一个链路服务可以按照 EDF 顺序连续扣除同一传输队列中的多个小任务，但不能在同一槽切换到另一个目的链路。

### 2.3.4 UAV动态节点状态

为了描述环境内部的节点状态，定义：

$$
\mathbf z_i^{\mathrm{uav}}(t)
=
\left[
\mathbf q_i(t),
\mathbf v_i(t),
E_i^{\mathrm{res}}(t),
\boldsymbol{\chi}_i^{\mathrm{res}},
\hat{\lambda}_i(t),
\mathbf q_i^{\mathrm{queue}}(t)
\right].
$$

其中 $\mathbf q_i^{\mathrm{queue}}(t)$ 是队列摘要向量，至少包括未绑定任务数、本地任务数、传输任务数、CPU 任务数、队首剩余 bit、队首剩余 cycle 和队首 slack。$\hat{\lambda}_i(t)$ 是由历史到达记录得到的估计量，而不是真实业务参数。系统级节点状态还可以保留真实建筑物、真实信道、完整任务记录和队列真值，但这些量不自动进入 actor 的局部观测。

### 2.3.5 UAV模型边界

本模型不建模六自由度飞行动力学、旋翼推力、姿态、飞控回路或轨迹优化，也不统计推进能耗。所有比较方法使用同一外生移动轨迹和同一轨迹随机数，因此性能差异不来自轨迹控制。主模型只统计卸载和资源编排直接产生的主动通信能耗与 CPU 动态能耗；不能将该能耗报告为完整 UAV 总能耗。

## 2.4 UAV资源异构与业务负载模型

资源异构和业务负载异构分开定义。资源异构来自硬件能力和能量预算，业务负载异构来自外部任务到达过程。业务负载向量写为：

$$
\boldsymbol{\chi}_i^{\mathrm{traffic}}
=
[\lambda_i].
$$

其中 $\lambda_i$ 表示 UAV $i$ 的任务到达概率或强度，只用于环境生成。若每个 UAV 每槽最多产生一个任务，则可采用：

$$
A_i(t)\sim\operatorname{Bernoulli}(\lambda_i).
$$

这里 $A_i(t)\in\{0,1\}$ 是槽末生成任务的到达指示量。负载异构实验改变各 UAV 的 $\lambda_i$ 离散程度，同时保持全局平均到达率的设定不变。

actor 不能读取真实 $\lambda_i$。执行期可用的负载线索是历史窗口内的经验估计：

$$
W_\lambda\in\mathbb N_{+},
\qquad
K_\lambda(t)=\min\{W_\lambda,t\}.
$$

其中 $W_\lambda$ 是历史窗口长度，单位为 slot，$K_\lambda(t)$ 是时隙 $t$ 槽初实际可用的历史样本数。历史到达率估计和可用性 mask 联合定义为：

$$
\hat{\lambda}_i(t)
=
\begin{cases}
0, & t=0,\\[2pt]
\displaystyle
\frac{\sum_{\tau=t-K_\lambda(t)}^{t-1}A_i(\tau)}
{K_\lambda(t)}, & t\ge 1,
\end{cases}
\qquad
m_i^\lambda(t)=\mathbb I\!\left[K_\lambda(t)>0\right].
$$

因此，$t=0$ 时没有已结束时隙的到达样本，使用固定缺省值 $\hat{\lambda}_i(0)=0$ 和 $m_i^\lambda(0)=0$；对 $t\ge1$，$m_i^\lambda(t)=1$。求和上界严格为 $t-1$：当 $1\le t<W_\lambda$ 时使用当前实际存在的短历史窗口，当 $t\ge W_\lambda$ 时使用最近 $W_\lambda$ 个已结束时隙。该估计的单位沿用按时隙计的 task/slot，不引入额外的 $\Delta t$ 换算，也不改变真实业务到达分布。

特别地，$A_i(t)$ 在时隙 $t$ 槽末才生成，因此不属于 $\hat{\lambda}_i(t)$ 的历史样本，也不得进入时隙 $t$ 的 actor 观测；它最早在时隙 $t+1$ 槽初通过历史估计或队列状态产生影响。

硬件参数采用可解释 profile 加小扰动生成，而不是把所有资源参数完全独立采样。对 profile $c(i)$，资源向量可表示为：

$$
\boldsymbol{\chi}_i^{\mathrm{res}}
=
\boldsymbol{\chi}_{c(i)}^{\mathrm{profile}}
\odot
\left(\mathbf 1+\boldsymbol{\delta}_i\right),
\qquad
\delta_{i,m}\in[-0.05,0.05].
$$

其中 $\odot$ 表示逐元素乘法，$\boldsymbol{\delta}_i$ 是由独立随机种子生成的可复现小扰动。五类硬件 profile 的相对参数如下：

| Profile | $f_i^{\max}/f_{\mathrm{ref}}$ | $P_i^{\max}/P_{\mathrm{ref}}$ | $E_i^0/E_{\mathrm{ref}}^0$ | $\kappa_i/\kappa_{\mathrm{ref}}$ | 物理含义 |
|---|---:|---:|---:|---:|---|
| Resource-poor | $0.7$ | $0.8$ | $0.8$ | $1.20$ | 计算、通信和能效均较弱 |
| Balanced | $1.0$ | $1.0$ | $1.0$ | $1.00$ | 参考 profile |
| Compute-rich | $1.5$ | $1.0$ | $1.1$ | $0.90$ | CPU 能力较强且相对节能 |
| Energy-limited | $1.0$ | $0.8$ | $0.5$ | $1.10$ | 能量预算较紧 |
| Communication-rich | $1.0$ | $1.5$ | $1.1$ | $1.00$ | 通信发射能力较强 |

主场景使用上述五类 profile，并分别设置 Homogeneous、Mild-Heterogeneous 和 Strong-Heterogeneous 三档资源/负载组合。profile 只描述资源结构，不能代替 $\lambda_i$ 的业务配置。

## 2.5 任务、生命周期与队列模型

每个任务由唯一标识、源 UAV、输入数据量、计算量、到达时隙、截止时隙、剩余工作量、目的地和生命周期状态构成。任务记录定义为：

$$
\tau_n
=
\left(
id_n,
i_n,
D_n,
C_n,
t_n^{\mathrm{arr}},
t_n^{\mathrm{ddl}},
D_n^{\mathrm{rem}},
C_n^{\mathrm{rem}},
d_n^{\mathrm{dst}},
s_n
\right).
$$

其中 $D_n$ 的单位为 bit，$C_n$ 的单位为 cycle，$D_n^{\mathrm{rem}}$ 表示剩余待传输输入 bit，$C_n^{\mathrm{rem}}$ 表示剩余待执行 cycle；$d_n^{\mathrm{dst}}=\varnothing$ 表示任务尚未绑定目的 UAV。任务生成时处于 unbound 状态，并初始化为 $D_n^{\mathrm{rem}}=D_n$、$C_n^{\mathrm{rem}}=C_n$ 和 $d_n^{\mathrm{dst}}=\varnothing$。远程任务的 $D_n^{\mathrm{rem}}$ 只按实际 U2U 传输服务量递减；任务绑定为 local 时则必须按下述生命周期规则立即清零该字段。$C_n^{\mathrm{rem}}$ 保持为 $C_n$，直到实际 CPU 服务使其递减。

输入数据量和单位 bit 计算量使用方案 2 的初始化范围：

$$
D_n\sim U(0.25,1.5)\ \mathrm{Mbit}.
$$

该式给出任务输入规模的编码初始分布，正式实验前仍需经过 Gate P 的非退化性检查。计算量与数据量的比值定义为：

$$
\frac{C_n}{D_n}
\sim U(300,1000)\ \mathrm{cycle/bit}.
$$

deadline 根据参考传输时间和参考计算时间生成。令 $\eta_n\sim U(1.5,3.0)$ 为服务裕量因子，先定义 clip 后的相对 deadline 时隙预算：

$$
L_n^{\mathrm{ddl}}
=
\operatorname{clip}
\left(
\left\lceil
\frac{
\eta_n
\left(
D_n/R_{\mathrm{ref}}+C_n/f_{\mathrm{ref}}
\right)
}{
\Delta t
}
\right\rceil,
10,
150
\right).
$$

随后定义绝对 deadline 时隙索引为：

$$
t_n^{\mathrm{ddl}}
=
t_n^{\mathrm{arr}}+L_n^{\mathrm{ddl}}.
$$

其中 $L_n^{\mathrm{ddl}}$ 表示从任务到达槽末到截止槽末的总相对时隙预算，其中包含 route 决策所占用的一个时隙，不表示纯传输或 CPU 可服务时隙数量；$R_{\mathrm{ref}}$ 的单位为 bit/s，$f_{\mathrm{ref}}$ 的单位为 cycle/s，$\Delta t$ 的单位为 s。分布表中的 Mbit 在代入传输时间前统一换算为 bit，10 至 150 只限制相对预算 $L_n^{\mathrm{ddl}}$，不表示绝对时隙索引。

任务状态集合为：

$$
\mathcal S_{\mathrm{task}}
=
\{
\mathrm{unbound},
\mathrm{local},
\mathrm{tx},
\mathrm{cpu},
\mathrm{done},
\mathrm{expired}
\}.
$$

其中，unbound 表示任务已经到达但尚未绑定本地或远程目的地，此时 $d_n^{\mathrm{dst}}=\varnothing$；local 表示任务已绑定到源 UAV $i_n$ 的本地 CPU 队列，此时 $d_n^{\mathrm{dst}}=i_n$；tx 表示任务已绑定到远程 UAV $j\ne i_n$ 并处于传输队列，此时 $d_n^{\mathrm{dst}}=j$；cpu 表示任务正在本地或远程 CPU 队列中接受计算服务，其目的地字段必须等于实际执行 UAV，即本地执行时为 $i_n$、远程执行时为 $j\ne i_n$；done 表示任务不晚于截止时隙末完成；expired 表示任务在截止时隙服务结束、且本槽服务量和剩余工作量更新后仍未完成。

合法生命周期转移由以下集合概括：

$$
\mathcal T_{\mathrm{task}}
=
\{
\mathrm{unbound}\rightarrow\mathrm{local},
\mathrm{unbound}\rightarrow\mathrm{tx},
\mathrm{unbound}\rightarrow\mathrm{expired},
\mathrm{local}\rightarrow\mathrm{cpu},
\mathrm{local}\rightarrow\mathrm{expired},
\mathrm{tx}\rightarrow\mathrm{cpu},
\mathrm{tx}\rightarrow\mathrm{expired},
\mathrm{cpu}\rightarrow\mathrm{done},
\mathrm{cpu}\rightarrow\mathrm{expired}
\}.
$$

done 和 expired 均为终止状态，该集合不包含由二者出发的转移，尤其不包含 $\mathrm{expired}\rightarrow\mathrm{done}$。

当 route 决策在槽末生效并触发 $\mathrm{unbound}\rightarrow\mathrm{local}$ 转移时，绑定操作必须同时执行：

$$
d_n^{\mathrm{dst}}\leftarrow i_n,
\qquad
D_n^{\mathrm{rem}}\leftarrow 0.
$$

当 route 决策在槽末生效并触发 $\mathrm{unbound}\rightarrow\mathrm{tx}$ 转移时，绑定操作必须同时执行：

$$
d_n^{\mathrm{dst}}\leftarrow j,
\qquad
j\ne i_n.
$$

本地任务不需要 U2U 输入传输，因此进入本地 CPU 队列后必须始终满足 $D_n^{\mathrm{rem}}=0$；其 $C_n^{\mathrm{rem}}=C_n$，直到实际 CPU 服务使其递减。远程任务绑定到 $j\ne i_n$ 后，$D_n^{\mathrm{rem}}$ 继续表示尚未完成的 U2U 输入传输量，并只按实际传输服务量递减。

该集合不包含已绑定任务改变目的 UAV 的转移，也不包含多跳转发。新任务在槽末到达并记录 $t_n^{\mathrm{arr}}$，下一时隙槽初才首次进入未绑定队列并允许 route；route 决策在槽末生效时，unbound 任务转入 local 或 tx 队列并立即锁定目的地，下一时隙才首次接受 CPU 或传输服务。此后处于 local、tx 或 cpu 状态的任务不得再次执行 route，也不得改变 $d_n^{\mathrm{dst}}$。本槽传输完成任务在槽末进入远程 CPU 队列，下一时隙才首次计算；因此新生成任务、本槽路由任务和本槽传输完成任务均不能在同一时隙获得下一阶段服务。

对于处于 done 状态的任务，若其在服务时隙 $t_n^{\mathrm{cmp}}$ 的服务过程中完成，则在该时隙槽末记录完成，完成边界为 $(t_n^{\mathrm{cmp}}+1)\Delta t$。由于到达边界为 $(t_n^{\mathrm{arr}}+1)\Delta t$，其端到端时延定义为：

$$
T_n^{\mathrm{E2E}}
=
\left(
t_n^{\mathrm{cmp}}-t_n^{\mathrm{arr}}
\right)\Delta t.
$$

合法完成记录必须满足 $t_n^{\mathrm{cmp}}\le t_n^{\mathrm{ddl}}$。任务可以在其截止时隙 $t_n^{\mathrm{ddl}}$ 内继续接受本槽传输或计算服务；仅在该槽的服务量和剩余工作量更新完成后进行终止结算。若任务在该结算点仍未全部完成，则立即转为 expired。该判定可写为：

$$
\operatorname{expired}_n
=
\mathbb 1
\left\{
\text{任务在 }t_n^{\mathrm{ddl}}\text{ 的本槽服务量和剩余工作量更新后仍未全部完成}
\right\}.
$$

若任务在相同结算点已全部完成，则转为 done，并计为按时完成。系统仅保留按时完成和过期两类结算结果；expired 任务在 $t_n^{\mathrm{ddl}}$ 槽末结算后立即从所有活动队列中移除，因此在 $t_n^{\mathrm{ddl}}+1$ 的槽初不再出现在任何活动队列中，也不再接受传输或计算服务。


为闭合槽末因果关系，任务状态转移必须先于 deadline settlement：对每个时隙 $t$，先完成本槽 service 并更新 $D_n^{\mathrm{rem}}$、$C_n^{\mathrm{rem}}$，再应用本槽已经发生但只能在槽末生效的 route、transfer-completion 和 CPU-completion 转移，完成全部队列 bookkeeping，最后在该 post-service/post-transition 状态上执行 done/expired 结算。该顺序冻结为：

$$
\boxed{
\text{slot service}
\rightarrow
\text{slot-end task transition}
\rightarrow
\text{done/expired settlement}
}
$$

其中，route 在槽末完成目的地 binding；local route 将 $d_n^{\mathrm{dst}}$ 锁定为源 UAV 并令 $D_n^{\mathrm{rem}}=0$；transmission 在本槽完成全部剩余 bit 时令 $D_n^{\mathrm{rem}}=0$，若 $C_n^{\mathrm{rem}}>0$ 则仅在槽末转入目的 UAV 的 CPU 队列、不得在同一槽再次获得 CPU service；CPU 完成全部 remaining cycles 时先形成 completion candidate。若任务在上述结算状态满足全部 workload 已完成，则转为 $\mathrm{done}$；否则若 $t=t_n^{\mathrm{ddl}}$，立即转为 $\mathrm{expired}$。不得采用 settlement 先于 route/transfer transition 的顺序。若任务转为 expired，则从所有 active service queues 和 waiting queues 中移除，且不进入下一槽的 route、tx 或 CPU service。

三个 deadline-slot 边界例子如下。若 $t=t_n^{\mathrm{ddl}}$ 的槽初任务仍为 unbound 且本槽执行合法 route，binding 可以在槽末生效，但 route 不产生本槽 service；若 workload 仍未完成，则在同一槽的 settlement 中标记为 $\mathrm{expired}$，不进入下一槽。若 deadline slot 内 transmission 恰好使 $D_n^{\mathrm{rem}}\rightarrow0$ 而 $C_n^{\mathrm{rem}}>0$，允许完成 tx $\rightarrow$ CPU 的槽末 bookkeeping，但不提供同槽 CPU service，随后立即标记为 $\mathrm{expired}$。若 deadline slot 内 CPU 恰好使 $C_n^{\mathrm{rem}}\rightarrow0$ 且全部 workload 已完成，则标记为 $\mathrm{done}$ 而不是 $\mathrm{expired}$；completion 判断优先于 expiration 判断。

## 2.6 任务队列与EDF调度

环境显式保存四类队列。未绑定队列为：

$$
Q_i^{\mathrm{unb}}(t)
=
\left(
\tau_n:\ i_n=i,\ s_n(t)=\mathrm{unbound}
\right).
$$

本地队列为：

$$
Q_i^{\mathrm{loc}}(t)
=
\left(
\tau_n:\ i_n=i,\ s_n(t)\in\{\mathrm{local},\mathrm{cpu}\},\ d_n^{\mathrm{dst}}=i
\right).
$$

这里的 $i$ 同时是任务源 UAV 和本地执行 UAV；队列内任务均满足 $D_n^{\mathrm{rem}}=0$。对 $j\in\mathcal U\setminus\{i\}$，从源 UAV $i$ 到远程目的 UAV $j$ 的传输队列为：

$$
Q_{i\rightarrow j}^{\mathrm{tx}}(t)
=
\left(
\tau_n:\ i_n=i,\ d_n^{\mathrm{dst}}=j,\ s_n(t)=\mathrm{tx}
\right).
$$

对 $j\in\mathcal U\setminus\{i\}$，来源为 $i$、在远程目的 UAV $j$ 上执行的 CPU 队列为：

$$
Q_{i\rightarrow j}^{\mathrm{cpu}}(t)
=
\left(
\tau_n:\ i_n=i,\ d_n^{\mathrm{dst}}=j,\ s_n(t)=\mathrm{cpu}
\right).
$$

因此，两类远程队列均要求 $j\ne i$；目的地为源 UAV 的本地任务不会被收入远程 CPU 队列。

每类队列均按队首 deadline 优先的 EDF 顺序排列。任务 $n$ 在槽初、且仍处于活动状态并满足 $t\le t_n^{\mathrm{ddl}}$ 时的 deadline 余量定义为：

$$
\operatorname{slack}_n(t)
=
t_n^{\mathrm{ddl}}-t+1.
$$

余量越小表示任务越紧迫；在最后允许服务的时隙 $t=t_n^{\mathrm{ddl}}$，slack 等于 $1$，活动任务不应出现 $0$ 或负数 slack。相同 slack 时采用固定任务标识作为可复现的 tie-break，不把随机打散引入队列语义。

route 只处理槽初 $Q_i^{\mathrm{unb}}(t)$ 中的最高优先级任务，并在 local、defer 或当前候选远程 UAV 中选择；选择 local 或远程目的地时，route 决策在槽末生效并一次性绑定目的地，不直接产生本槽传输服务。tx_select 只从槽初已有的 $Q_{i\rightarrow j}^{\mathrm{tx}}(t)$ 中选择一个非空传输队列，并负责本槽服务预算分配；该动作只服务已绑定的传输队列，不参与目的地选择或改变 $d_n^{\mathrm{dst}}$。两者即使在同一槽分别输出，也不能使新路由任务获得本槽传输服务。

## 2.7 外生移动与动态U2U图模型

### 2.7.1 Gauss-Markov水平移动

移动轨迹由环境外生生成。每个 episode reset 时直接生成或载入初始水平位置和速度 $\mathbf p_i(0)$、$\mathbf v_i(0)$，以及移动模型所需的其他初始状态；不定义也不访问 $\mathbf p_i(-1)$。令 $\bar{\mathbf v}_i$ 为平均水平速度，$\boldsymbol{\varepsilon}_i(t)$ 为零均值扰动，水平速度采用：

$$
\mathbf v_i^{\mathrm{hor}}(t+1)
=
\alpha\mathbf v_i^{\mathrm{hor}}(t)
+
(1-\alpha)\bar{\mathbf v}_i
+
\sqrt{1-\alpha^2}\,
\boldsymbol{\varepsilon}_i(t).
$$

其中 $\alpha=0.85$ 为编码初始相关系数，$\mathbf v_i^{\mathrm{hor}}(t)=[v_i^x(t),v_i^y(t)]^{\mathrm T}$，该式对 $t\ge0$ 使用。

主场景的水平移动区域冻结为轴对齐矩形
$\mathcal R=[x_{\min},x_{\max}]\times[y_{\min},y_{\max}]$，边界只采用 coordinate-wise specular reflection。
令 $\widetilde v_i^q(t+1)$ 为 Gauss--Markov 速度更新得到的暂态坐标速度，并令
$\widetilde q_i(t+1)=q_i(t)+\Delta t\,\widetilde v_i^q(t+1)$，其中 $q\in\{x,y\}$。对每个坐标分别执行：

$$
q_i(t+1)
=
\begin{cases}
2q_{\min}-\widetilde q_i(t+1),&\widetilde q_i(t+1)<q_{\min},\\
2q_{\max}-\widetilde q_i(t+1),&\widetilde q_i(t+1)>q_{\max},\\
\widetilde q_i(t+1),&\text{otherwise},
\end{cases}
\qquad
v_i^q(t+1)
=
\begin{cases}
-\widetilde v_i^q(t+1),&\widetilde q_i(t+1)\notin[q_{\min},q_{\max}],\\
\widetilde v_i^q(t+1),&\text{otherwise}.
\end{cases}
$$

若单个步长跨越同一坐标边界后仍越界，则重复同一镜像映射，直至坐标进入合法区间。更新后的三维位置重新写为：

$$
\mathbf q_i(t+1)
=
\left[
\mathbf p_i(t+1)^{\mathrm T},H
\right]^{\mathrm T}.
$$

位置更新式对 $t\ge0$ 使用已定义的 $\mathbf p_i(t)$；需要引用上一时隙位置的位移量从 $t=1$ 开始定义，初始时刻不通过负时隙历史反推移动状态。

为保证外生移动与 2.8.1 节有限高度建筑几何的一致性，定义固定飞行高度 $H$ 下的可行水平飞行区域为：

$$
\mathcal R_{\mathrm{free}}(H)
=
\mathcal R
\setminus
\bigcup_{\substack{b\in\mathcal B\\H\le H_b}}
\mathcal P_b.
$$

所有用于通信与服务评价的离散时隙位置必须满足：

$$
\begin{aligned}
\mathbf p_i(t)&\in\mathcal R_{\mathrm{free}}(H),
&&\forall i\in\mathcal U,\ t\in\{0,\ldots,T-1\},\\
\mathbf q_i(t)&\notin\mathcal B_b,
&&\forall i\in\mathcal U,\ b\in\mathcal B,\ t\in\{0,\ldots,T-1\}.
\end{aligned}
$$

其中，$\mathcal P_b$、$H_b$ 和 $\mathcal B_b=\mathcal P_b\times[0,H_b]$ 分别表示建筑物 $b$ 的闭合水平足迹、有限高度和三维闭实体。当 $H\le H_b$ 时，$\mathbf p_i(t)$ 不得位于 $\mathcal P_b$ 内部或边界上；由于屋顶属于闭实体，$H=H_b$ 仍执行该排除。当 $H>H_b$ 时，UAV 可以从该建筑物上方飞越。因此，建筑物不被解释为无限竖直禁飞柱。

最小安全距离是 reset 可行性约束而不是运行期 collision controller。固定参数满足
$d_{\min}^{\mathrm{safe}}>0$；reset 按 UAV ID 升序依次采用 deterministic-seed rejection sampling，当前采样位置必须位于 $\mathcal R_{\mathrm{free}}(H)$ 内且满足：
$$
\left\|\mathbf p_i(0)-\mathbf p_j(0)\right\|_2
\ge d_{\min}^{\mathrm{safe}},\qquad\forall i\ne j.
$$
初始候选位置不合法时只重新采样当前 UAV，直到得到合法位置。完整外生候选轨迹按照上述 Gauss--Markov 递推和矩形外边界规则在场景生成阶段生成，或从预生成轨迹集中载入；用于一个 episode 的完整离散位置序列必须统一满足建筑排除条件。无效候选轨迹按固定 master seed 和稳定候选顺序整体舍弃，接受后的轨迹及其标识在比较算法之间固定复用，从而保持确定性复现。该接受过程使轨迹集合成为经建筑几何条件化的 Gauss--Markov 候选集合，不应表述为完全无约束的 Gauss--Markov 轨迹分布。

episode 运行期间不新增位置 projection、clamp、random reset、在线避障控制器、RL 动作约束、奖励惩罚或策略相关的轨迹修改。若研究区域不是轴对齐矩形，本章不替换为另一套外边界规则。上述建筑排除条件只保证 $t=0,\ldots,T-1$ 的离散服务时隙位置合法；本模型未检验 $\mathbf q_i(t)$ 与 $\mathbf q_i(t+1)$ 之间的连续飞行线段，因而不宣称连续时间轨迹已满足建筑碰撞规避。

上述移动不属于动作空间，所有算法共享同一有效外生轨迹和轨迹随机数。


### 2.7.2 新任务候选邻居与服务边

新任务候选邻居由三维距离阈值定义：

$$
\mathcal N_i(t)
=
\left\{
j\in\mathcal U\setminus\{i\}:
d_{ij}(t)\le R_{\mathrm{cand}}
\right\}.
$$

其中 $R_{\mathrm{cand}}$ 是候选通信半径，初始值为 500 m，正式实验前结合链路预算校准。主场景不启用 distance hysteresis；每个时隙只依据当前槽初距离直接重算 $\mathcal N_i(t)$，不保存 enter/leave 阈值或邻居成员状态。滞回只能作为主模型之外的扩展，后续实现不得将其作为可选主规则。
本项目中，上述 500 m 仅表示全局/历史默认参考值；当前 resolved config 将 Small Scenario 的新任务候选邻居图阈值修订为 525 m，Medium/Large 保持 500 m。该阈值不等价于最大通信范围、SINR/rate/outage 阈值或物理链路预算硬截断；已绑定 service-edge 的可用性仍由 channel、SINR、rate、outage 与 executor 语义共同决定。

新任务边只用于决定未绑定任务当前可选的目的 UAV。服务边则保存已经锁定的 $i\rightarrow j$ 关系；即使锁定任务因移动暂时超出新任务候选范围，仍保留其传输队列和剩余 bit，等待链路恢复或按过期规则结算，链路 outage、移动或超出候选范围均不触发目的地变更。outage 不删除已绑定服务边。所有邻居、建筑物路径和信道计算均以 $\mathbf q_i(t)$ 的三维位置为输入。

## 2.8 建模的U2U物理层评价

### 2.8.1 建筑物几何遮挡

令 $\mathcal P_b\subseteq\mathcal R$ 为第 $b$ 个建筑物的闭合水平足迹，$H_b>0$ 为其有限高度，并将建筑物表示为三维闭棱柱实体 $\mathcal B_b=\mathcal P_b\times[0,H_b]$。令 $\zeta\in[0,1]$ 为线段参数，则 UAV $i$ 到 UAV $j$ 的传播线段为：

$$
\boldsymbol{\ell}_{ij}(\zeta,t)
=
(1-\zeta)\mathbf q_i(t)+\zeta\mathbf q_j(t),
\qquad
\zeta\in[0,1].
$$

只要存在至少一个建筑物与该闭线段相交，就判定链路发生几何遮挡：

$$
I_{ij}^{\mathrm{bld}}(t)
=
\mathbb 1
\left\{
\exists b\in\mathcal B:
\boldsymbol{\ell}_{ij}([0,1],t)
\cap\mathcal B_b
\ne\varnothing
\right\}.
$$

这里显式使用存在量词，以保证遮挡判定是“任一建筑物相交即遮挡”，而不是对建筑物进行平均或随机抽样。该指示量是环境内部真实状态，actor 不直接读取其真值。

### 2.8.2 路径损耗和建筑物附加损耗

自由空间基准路径损耗可写为：

$$
PL_{\mathrm{FS}}(d,f_c)
=
20\log_{10}
\left(
\frac{4\pi f_c d}{c_0}
\right),
$$

其中 $c_0$ 为光速，$d$ 使用三维距离，$f_c$ 使用载频。建筑物附加损耗按遮挡指示量加入：

$$
L_{\mathrm{bld}}(t)
=
L_{\mathrm B} I_{ij}^{\mathrm{bld}}(t).
$$

当 $I_{ij}^{\mathrm{bld}}(t)=0$ 时，$L_{\mathrm{bld}}(t)=0$。$L_{\mathrm B}\ge0$ 是轻量几何遮挡抽象中的场景级名义额外建筑遮挡损耗参数，用于聚合表征相对于自由空间基准、在建筑物阻断直达几何时由未显式重建的绕射、反射、散射及可用替代传播路径所产生的大尺度平均额外衰减。该参数不表示建筑入楼损耗、墙体穿透损耗或材料常数，也不是标准的 3.5 GHz 建筑损耗、普适室外遮挡损耗、本项目实测值或本项目射线追踪标定值。

主模型取 $L_{\mathrm B}=15\,\mathrm{dB}$ 作为名义建模基准，并采用：

$$
L_{\mathrm B}\in\{10,15,20,30\}\,\mathrm{dB}
$$

进行刻意设置的宽范围敏感性分析。该集合不是文献认证的轻度、中度或重度遮挡离散分类；其用途是在假定的额外遮挡严重程度变化时，检验物理 KPI、算法增益和算法相对排序是否保持稳定。在完成相应实验前，不据此声称敏感性鲁棒性。含相关阴影的总路径损耗为：

$$
PL_{ij}(t)
=
PL_{\mathrm{FS}}\left(d_{ij}(t),f_c\right)
+
L_{\mathrm B}I_{ij}^{\mathrm{bld}}(t)
+
X_{ij}(t).
$$

在上述分解中，$X_{ij}(t)$ 是 dB 域零均值随机残差，其统计均值不再包含由同一 $I_{ij}^{\mathrm{bld}}(t)$ 表示的建筑遮挡惩罚。该额外遮挡项仅叠加在自由空间基准上，不与已在 NLoS 均值中包含遮挡相关额外损耗的经验路径损耗模型无条件叠加。特别地，若未来采用状态条件化的 $PL_{\mathrm{3GPP}}^{\mathrm{NLoS}}$ 或其他 NLoS 均值模型，则在没有单独标定独立残差项时不得再加入 $L_{\mathrm B}I_{ij}^{\mathrm{bld}}(t)$，否则可能对同一遮挡机制重复计损。

该式只提供轻量链路评价所需的大尺度项，不等同于实测路径损耗或实时射线追踪输出。$L_{\mathrm B}=15\,\mathrm{dB}$ 在获得场景匹配的外部测量或射线追踪标定前始终保持名义参数身份，不应归因于 3GPP 或 ITU 推荐。

### 2.8.3 空间相关阴影

空间相关性通过相邻时隙内两端 UAV 的水平位移建立。episode reset 时直接采样初始阴影项：

$$
X_{ij}(0)
\sim
\mathcal N(0,\sigma_s^2),
$$

其中 $X_{ij}(0)$ 是阴影衰落过程的显式初始样本，不通过未定义的 $X_{ij}(-1)$ 递推获得。初始位移约定为 $\Delta s_{ij}(0)=0$。对 $t\ge1$，位移尺度为：

$$
\Delta s_{ij}(t)
=
\frac{
\left\|\mathbf p_i(t)-\mathbf p_i(t-1)\right\|_2
+
\left\|\mathbf p_j(t)-\mathbf p_j(t-1)\right\|_2
}{2}.
$$

设 $d_{\mathrm{corr}}$ 为相关距离，则相关系数为：

$$
\rho_{ij}(t)
=
\exp
\left(
-\frac{\Delta s_{ij}(t)}{d_{\mathrm{corr}}}
\right).
$$

阴影项按槽间递推：

$$
X_{ij}(t)
=
\rho_{ij}(t)X_{ij}(t-1)
+
\sqrt{1-\rho_{ij}^2(t)}\,
\sigma_s\epsilon_{ij}(t).
$$

该递推式仅对 $t\ge1$ 使用。其中 $\sigma_s$ 的单位为 dB，$\epsilon_{ij}(t)$ 是标准化零均值扰动。该模型保持短期移动连续性，并使遮挡和链路质量变化不被人为地设置为完全独立；阴影项在 dB 域保持既有定义，不将其直接当作线性功率相加。

### 2.8.4 Rician/Rayleigh槽级衰落

主模型按已确定的槽级 LoS/NLoS 状态选择小尺度衰落类型。slot $t$ 的位置、路径损耗、阴影和 LoS/NLoS 状态确定后，环境为全部有向链路 $i\rightarrow j$（$i\ne j$）及全部资源单元 $r$ 生成小尺度创新；该生成不以 actor 是否选择某条链路为条件。episode reset 直接按同一规则生成 $h_{ij,r}(0)$，后续时隙不访问负时隙信道历史。

先定义标准复高斯创新：
$$
z_{ij,r}(t)\sim\mathcal{CN}(0,1),\qquad
\Re(z_{ij,r}(t)),\Im(z_{ij,r}(t))
\stackrel{\mathrm{i.i.d.}}{\sim}\mathcal N(0,1/2),\qquad
\mathbb E[|z_{ij,r}(t)|^2]=1.
$$
NLoS 链路使用 normalized Rayleigh：
$$
g_{ij,r}(t)=z_{ij,r}(t).
$$
LoS 链路使用 normalized Rician。令 $K_{\mathrm{dB}}=6$ dB 且
$K_{\mathrm{lin}}=10^{K_{\mathrm{dB}}/10}$，则
$$
g_{ij,r}(t)=
\sqrt{\frac{K_{\mathrm{lin}}}{K_{\mathrm{lin}}+1}}
+\sqrt{\frac{1}{K_{\mathrm{lin}}+1}}\,z_{ij,r}(t).
$$
LoS deterministic component 使用单位相位 $1$；当前系统只使用 $|h_{ij,r}(t)|^2$，不额外引入未冻结的 LoS 相位模型。两种状态均满足 $\mathbb E[|g_{ij,r}(t)|^2]=1$。当 $I_{ij}^{\mathrm{bld}}(t)=0$ 时使用 Rician；当 $I_{ij}^{\mathrm{bld}}(t)=1$ 时使用 Rayleigh 特例。因此，该归一化状态切换只改变小尺度衰落的统计形态，不重复增加由 $L_{\mathrm B}I_{ij}^{\mathrm{bld}}(t)$ 表示的大尺度平均损耗。
$$
|h_{ij,r}(t)|^2=G_tG_r10^{-PL_{ij}(t)/10}|g_{ij,r}(t)|^2.
$$
小尺度创新满足跨时隙、跨有向链路和跨 RU 独立：
$$
z_{ij,r}(t)\perp z_{k\ell,s}(\tau),\qquad(i,j,r,t)\ne(k,\ell,s,\tau).
$$
该独立性只冻结 small-scale innovation，不覆盖位置、路径损耗、LoS/NLoS 或 $X_{ij}(t)$ 表达的大尺度相关。$z$ 与 arrival randomness、CSI error、interference measurement 和 executor 决策也相互独立。slot $t$ 按 $(i,j,r)$ 升序字典序为所有有向链路/RU 生成完整 small-scale tensor，再形成 $h(t)$；不得依赖 set/dict 遍历顺序、actor-selected edges 或 accepted-link count。最终 executed action 确定后，物理层使用同一 $h_{ij,r}(t)$ 计算 SINR/service；actor 不能读取当前真实信道。

### 2.8.5 不完美CSI和CSI AoI

actor 使用陈旧且带估计误差的信道信息。令 $a_{ij}^{\mathrm{CSI}}(t)$ 为 CSI AoI，$\xi_{ij,r}(t)$ 为正的乘性估计误差。先定义陈旧 CSI 的原始历史索引及可用性 mask：

$$
\ell_{ij}^{\mathrm{CSI}}(t)
=
t-a_{ij}^{\mathrm{CSI}}(t),
\qquad
m_{ij}^{\mathrm{CSI}}(t)
=
\mathbb I
\left[
\ell_{ij}^{\mathrm{CSI}}(t)\ge0
\right].
$$

陈旧 CSI 特征定义为：

$$
\hat h_{ij,r}(t)
=
\begin{cases}
h_{ij,r}\!\left(\ell_{ij}^{\mathrm{CSI}}(t)\right)\sqrt{\xi_{ij,r}(t)},
&m_{ij}^{\mathrm{CSI}}(t)=1,
\\[6pt]
0,
&m_{ij}^{\mathrm{CSI}}(t)=0.
\end{cases}
$$

当 $\ell_{ij}^{\mathrm{CSI}}(t)<0$ 时，$m_{ij}^{\mathrm{CSI}}(t)=0$，不访问负时隙信道；此时的零值只表示缺省占位，并不表示真实零信道。只有当 $\ell_{ij}^{\mathrm{CSI}}(t)\ge0$ 时才读取已经存在的历史样本。当当前时隙满足 $\ell_{ij}^{\mathrm{CSI}}(t)=0$ 时，首次读取 $h_{ij,r}(0)$。若 AoI 固定为常数 $d$，则在 $t=d$ 时有 $\ell_{ij}^{\mathrm{CSI}}(t)=0$，首次读取 $h_{ij,r}(0)$。本轮不生成 pre-episode 信道历史，也不通过 clamp 或当前真实 $h_{ij,r}(t)$ 回填指定 AoI 的历史样本。
正文参数表中的 $a^{\mathrm{CSI}}$ 是外生配置的陈旧偏移量；主场景将其作为固定的敏感性设置（例如 $0$、$3$ 或 $5$ slot），并在整个 episode 内保持不变。因此记号 $a_{ij}^{\mathrm{CSI}}(t)$ 在本模型中表示该既有配置值，而不是另行定义的刷新/递推 freshness process；其唯一作用仍是通过 $\ell_{ij}^{\mathrm{CSI}}(t)=t-a_{ij}^{\mathrm{CSI}}(t)$ 选择可读取的历史信道样本。本轮不对该固定陈旧偏移量新增 refresh 或 increment 规则，也不改变既有 CSI 历史索引语义。

CSI error 在每个 slot、每条有向链路和每个 RU 上重新采样。令
$\epsilon_{ij,r}^{\mathrm{CSI}}(t)$ 为 dB power-gain domain 的误差：
$$
\epsilon_{ij,r}^{\mathrm{CSI}}(t)\sim\mathcal N(0,\sigma_{\mathrm{CSI}}^2),\qquad
\xi_{ij,r}(t)=10^{\epsilon_{ij,r}^{\mathrm{CSI}}(t)/10}.
$$
因此 $10\log_{10}\xi_{ij,r}(t)=\epsilon_{ij,r}^{\mathrm{CSI}}(t)$，$\xi>0$ 为功率增益因子，$\sigma_{\mathrm{CSI}}$ 为 dB 域标准差；作用于复信道的线性乘性因子是 $\sqrt{\xi_{ij,r}(t)}$，从而
$|\hat h_{ij,r}(t)|^2=|h_{ij,r}(\ell_{ij}^{\mathrm{CSI}}(t))|^2\xi_{ij,r}(t)$。
$$
\epsilon_{ij,r}^{\mathrm{CSI}}(t)\perp\epsilon_{k\ell,s}^{\mathrm{CSI}}(\tau),\qquad
(i,j,r,t)\ne(k,\ell,s,\tau).
$$
CSI error 与 small-scale fading、shadowing、arrival process、interference measurement 和 executor 决策独立；本模型不引入 temporal CSI-error correlation 或 sample-hold。每个 slot 按 $(i,j,r)$ 升序字典序生成完整 CSI-error tensor，再由 availability mask 决定哪些样本进入 observation physics；mask 为 $0$ 的样本被忽略但不改变 RNG 顺序。固定 master seed 和相同配置必须产生相同的 CSI-error realization sequence。

actor 使用 $\hat h_{ij,r}(t)$ 时必须同时读取对应的 $m_{ij}^{\mathrm{CSI}}(t)$；缺省值不能脱离 mask 单独解释为真实测量。
上述 $\hat h_{ij,r}(t)$ 只描述期望链路 $i\rightarrow j$ 的陈旧信道估计，不包含本时隙其他 UAV 尚未确定的资源选择、功率或执行器接受结果；actor 也不得读取当前真实信道 $h_{ij,r}(t)$。为给槽初决策提供可实现的干扰线索，接收 UAV $j$ 对每个资源单元维护只由上一时隙或更早信息形成的历史干扰摘要。当前槽初的历史干扰量必须只由已结束时隙形成。对所有 $t\ge0$，在 actor 读取槽初观测时，$\widehat I_{j,r}^{\mathrm{hist}}(t)$、$m_{j,r}^{I}(t)$ 和 $a_j^{\mathrm{msg}}(t)$ 均不得依赖当前槽尚未形成的 $I_{j,r}^{\mathrm{meas}}(t)$；当前槽测量最早在槽末形成，并在下一槽槽初使用。令 $Q_{ij}^{\mathrm{bit}}(t)$ 表示槽初链路 $i\rightarrow j$ 传输队列中的剩余待传输 bit 总量。对当前有向接收边和 RU：
$$
m_{ij,r}^{I,\mathrm{meas}}(t)
=
\mathbb I[x_{ij,r}(t)=1]\,
\mathbb I[p_i^{\mathrm{exec}}(t)>0]\,
\mathbb I[Q_{ij}^{\mathrm{bit}}(t)>0].
$$
其中 $x_{ij,r}(t)=1$ 表示 RU 被最终 executed action 占用；接收端摘要使用既有符号
$m_{j,r}^{I,\mathrm{meas}}(t)=\max_{i\ne j}m_{ij,r}^{I,\mathrm{meas}}(t)$，节点独占无线调度规则保证同一接收端至多对应一条最终接受的入边。在槽 $t$ 的最终执行结果和真实信道已知后，历史摘要按以下下一槽递推更新：

$$
\widehat I_{j,r}^{\mathrm{hist}}(t+1)
=
\begin{cases}
\beta_I\widehat I_{j,r}^{\mathrm{hist}}(t)
+
(1-\beta_I)I_{j,r}^{\mathrm{meas}}(t),
&
m_{j,r}^{I,\mathrm{meas}}(t)=1,
\\[4pt]
\widehat I_{j,r}^{\mathrm{hist}}(t),
&
m_{j,r}^{I,\mathrm{meas}}(t)=0.
\end{cases}
$$

缺测时保持上一历史估计，不把缺测编码为 $I^{\mathrm{meas}}=0$；历史可用性 mask 只在首次获得有效测量后置为 1：

$$
m_{j,r}^{I}(t+1)
=
m_{j,r}^{I}(t)
\lor
m_{j,r}^{I,\mathrm{meas}}(t).
$$

其中 $\widehat I_{j,r}^{\mathrm{hist}}$ 始终是 interference-only 的线性功率，热噪声不进入该量。历史摘要通过既有接收端控制消息广播，现有唯一的摘要消息 AoI $a_j^{\mathrm{msg}}(t)$ 按消息生成时点更新：

$$
a_j^{\mathrm{msg}}(t+1)
=
\begin{cases}
1,
&
\exists r:\ m_{j,r}^{I,\mathrm{meas}}(t)=1,
\\[4pt]
a_j^{\mathrm{msg}}(t)+1,
&
\text{本槽无新测量且此前已有可用摘要},
\\[4pt]
\mathrm{NA},
&
\text{本槽无新测量且 episode 尚无任何可用摘要}.
\end{cases}
$$

测量在槽 $t$ 末生成、在槽 $t+1$ 槽初第一次可见，因此刷新后的消息 AoI 按既有槽时间语义为 $1$，而不是 $0$；无新测量时下一槽增加 $1$。$a_j^{\mathrm{msg}}$ 是历史干扰广播摘要的消息 AoI，不与固定配置的 CSI 陈旧偏移量 $a_{ij}^{\mathrm{CSI}}$ 混用。episode 初始时设置 $\widehat I_{j,r}^{\mathrm{hist}}(0)=I_{j,r}^{\mathrm{def}}$、$m_{j,r}^{I}(0)=0$ 和 $a_j^{\mathrm{msg}}(0)=\mathrm{NA}$。这里的测量可用性由既有接收端测量条件提供，不新增独立的信道观测过程。

若 $m_{ij,r}^{I,\mathrm{meas}}(t)=1$ 且 $I_{ij,r}^{\mathrm{meas}}(t)=0$，这是有效测量得到零干扰，零值正常进入 EMA 并刷新 AoI。若 $m=0$，则表示没有测量样本：不把 $0$ 写入 EMA，历史保持不变，AoI 递增或保持 $\mathrm{NA}$。zero-power idle、被拒绝 proposal 和空 executed resource set 均不产生 interference measurement。

历史摘要先与噪声功率合成为：

$$
\widehat Z_{j,r}^{\mathrm{hist}}(t)
=
N_0B_{\mathrm{RU}}F
+
\widehat I_{j,r}^{\mathrm{hist}}(t).
$$

为避免把该历史量误称为当前真实 SINR，定义固定参考每资源单元功率下的历史干扰链路质量代理量：

$$
\widehat\Gamma_{ij,r}^{\mathrm{hist}}(t)
=
\frac{
p_r^{\mathrm{ref}}
\left|\hat h_{ij,r}(t)\right|^2
}{
\widehat Z_{j,r}^{\mathrm{hist}}(t)
},
\qquad
p_r^{\mathrm{ref}}
=
\frac{P_{\mathrm{ref}}}{R}.
$$

其中 $p_r^{\mathrm{ref}}$ 只用于构造跨时隙、跨动作可比较的 per-RU 历史观测特征，不是 actor 在当前时隙最终选择的实际功率；$\widehat\Gamma_{ij,r}^{\mathrm{hist}}(t)$ 也不使用当前其他 UAV 的动作，因而不等于当前真实 $\mathrm{SINR}_{ij,r}(t)$。actor 在槽初观测阶段只使用 per-RU 历史质量代理量及其有效性 mask、上一槽已实现的有效速率、仅基于过去实际传输尝试的 outage 率、CSI AoI、消息 AoI 和可用性 mask。scalar historical quality 不属于 actor slot-start observation；它只能在 actor 形成 resource proposal、得到 $\mathcal S_i^{\mathrm{prop}}(t)$ 后，由 executor 为确定性仲裁派生。$\mathrm{SINR}_{ij,r}(t)$ 仍只由环境在联合动作确定后计算。CSI AoI 和历史摘要/消息 AoI 的更新是环境事件，不是 actor 可直接控制的动作。

在 actor proposal 形成并得到 $\mathcal S_i^{\mathrm{prop}}(t)$ 后，为使 deterministic executor 的输入唯一，定义 executor-only 的 valid-RU 集合：
$$
m_{ij,r}^{\mathrm{qual}}(t)=m_{ij}^{\mathrm{CSI}}(t)\land m_{j,r}^{I}(t),
\qquad
\mathcal V_{ij}(t)=
\{r\in\mathcal S_i^{\mathrm{prop}}(t):m_{ij,r}^{\mathrm{qual}}(t)=1\}.
$$
executor-only scalar historical quality 只在该 valid RU 集合上取 arithmetic mean：
$$
\widehat\Gamma_{ij}^{\mathrm{hist}}(t)
=
\begin{cases}
\dfrac{1}{|\mathcal V_{ij}(t)|}\sum_{r\in\mathcal V_{ij}(t)}
\widehat\Gamma_{ij,r}^{\mathrm{hist}}(t),&|\mathcal V_{ij}(t)|>0,\\
0,&|\mathcal V_{ij}(t)|=0.
\end{cases}
$$
聚合范围严格为 actor proposal 形成后的 $\mathcal S_i^{\mathrm{prop}}(t)$ 中具有有效 CSI 与有效历史干扰摘要的 RU；不使用全部 RU、$\mathcal S_i^{\mathrm{exec}}(t)$、其他资源组或当前真实 SINR。该 scalar 不回写 actor observation；无 valid RU 时的 $0$ 是 deterministic lowest-quality fallback。

### 2.8.6 模型边界和第三UAV软遮挡扩展

主模型只包含两端 UAV 与建筑物几何遮挡，不把第三 UAV 对 Fresnel 区域的软遮挡加入训练环境。只有主模型通过 Gate P、Gate 0 及后续机制闸门后，才可将第三 UAV 软遮挡作为几何启发式消融或离线高保真回放扩展。该扩展只能用于敏感性和趋势验证，不能被表述为实测遮挡模型，也不能反向改变主模型的 actor 信息权限。

## 2.9 固定资源组、SINR、有效速率和outage


由于 $R$ 个资源单元采用等带宽划分，单个资源单元带宽与系统总通信带宽的关系唯一确定为：

$$
B_{\mathrm{RU}}
=
\frac{B^{\mathrm{tot}}}{R}.
$$

噪声链统一在功率域中计算。$N_0$ 表示线性热噪声功率谱密度，单位为 W/Hz；若外部参数以 dBm/Hz 或 dBW/Hz 给出，必须先转换为线性 W/Hz。接收机噪声系数以 $F_{\mathrm{dB}}$（dB）给出，其线性 noise factor 为：

$$
F
=
10^{F_{\mathrm{dB}}/10},
\qquad
P_{\mathrm{noise,RU}}
=
N_0B_{\mathrm{RU}}F.
$$

因此后续 SINR 分母中的 $N_0B_{\mathrm{RU}}F$ 与 $P_{\mathrm{noise,RU}}$ 等价；同一计算链中不混用 dB 和线性功率。

系统将 $R$ 个等效资源单元预先划分为 $G=5$ 个连续、互不重叠的固定资源组：

$$
\mathcal G
=
\left\{
\mathcal G_1,\mathcal G_2,\mathcal G_3,\mathcal G_4,\mathcal G_5
\right\},
\qquad
\mathcal G_g\cap\mathcal G_{g'}=\varnothing\ (g\ne g'),
\qquad
|\mathcal G_g|=\frac{R}{G}.
$$

主场景 $R=20$ 时每组含 4 个等效资源单元；$R=10$ 和 $R=40$ 时分别为 2 个和 8 个。资源选择以固定组为粒度，不暴露单个等效资源单元的自由索引，也不使用连续带宽份额。

资源组和宽度分支只建立一个派生映射，不增加新的动作分支：

$$
g_i(t)\equiv a_i^{\mathrm{rg}}(t),
\qquad
w_i(t)\equiv a_i^{\mathrm{width}}(t).
$$

当通信父分支 active 时，$g_i(t)\in\{1,\ldots,5\}$ 且 $w_i(t)\in\{1,2\}$；定义宽度分支 activity mask 为：

$$
m_i^{\mathrm{width,act}}(t)
=
\mathbb I\left[a_i^{\mathrm{tx}}(t)\ne\mathrm{idle}\right]
\mathbb I\left[a_i^{\mathrm{rg}}(t)\ne\mathrm{idle}\right].
$$

当该 mask 为 $0$ 时，宽度分支仍保持原有定义域 $\{1,2\}$，并采用已有定义域中的 canonical inactive 编码 $w_i(t)=1$；该编码只用于保持固定维度接口，不表示实际申请一个资源组，也不触发资源申请、功率预留、实际传输或 active-branch 的统计。不得将该编码称为新的 idle action。

仅当 $m_i^{\mathrm{width,act}}(t)=1$ 时，动作提案申请的组集合定义为：

$$
\mathcal W_i(t)
=
\{g_i(t),g_i(t)+1,\ldots,g_i(t)+w_i(t)-1\},
\qquad
\mathcal S_i^{\mathrm{prop}}(t)
=
\bigcup_{g\in\mathcal W_i(t)}\mathcal G_g.
$$

$\mathcal S_i^{\mathrm{prop}}(t)$ 只表示发送 UAV $i$ 在 actor 动作提案中申请的资源单元集合，在联合执行器运行前即可由 $g_i(t),w_i(t)$ 确定，不表示资源已经实际占用。$g_i(t)=5$ 且 $w_i(t)=2$ 时该组合由动作 mask 禁止；当 $m_i^{\mathrm{width,act}}(t)=0$ 时令 $\mathcal S_i^{\mathrm{prop}}(t)=\varnothing$。即使执行器最终拒绝发送，也允许 $\mathcal S_i^{\mathrm{prop}}(t)\ne\varnothing$。

actor 的离散功率分支先映射为 proposed power：

$$
p_i^{\mathrm{prop}}(t)
=
\Phi_P\left(a_i^{\mathrm{pow}}(t)\right)
=
a_i^{\mathrm{pow}}(t)P_i^{\max}
\in
\{0,0.25,0.5,1.0\}P_i^{\max}.
$$

这里 $\Phi_P(\rho)=\rho P_i^{\max}$ 只是已有功率动作分支到物理量的映射。$p_i^{\mathrm{prop}}(t)$ 是 actor 的动作提案，不是最终实际发射功率；executor 可以拒绝该提案，或沿既有离散降档序列检查更低功率。因而资源宽度仍只表示一个组或两个相邻组，功率分支也不会产生离散集合之外的连续值。


若 actor 选择了通信 tx proposal 但 $p_i^{\mathrm{prop}}(t)=0$，则该 proposal 在进入冲突解析和资源执行语义前预先 canonicalize 为 communication idle，并从候选通信边中排除。它不占用半双工与节点独占无线调度中的无线端点、不占用执行资源、不产生能量预留、干扰、实际 transmission attempt 或 outage 样本；power 0 档位本身保留，且不增加新的 idle 类别。形式上，其最终通信结果必须满足：

$$
p_i^{\mathrm{prop}}(t)=0
\quad\Longrightarrow\quad
\text{communication idle before executor}.
$$

executor 内部允许使用候选功率，但候选功率不属于动作接口、环境持久状态、actor observation 或第八个动作分支。令 $p_i^{\mathrm{cand}}(t;\ell)$ 表示第 $\ell$ 个降档检查阶段的临时值；它从 $p_i^{\mathrm{prop}}(t)$ 开始，只沿当前已有的绝对离散功率等级 $1.0\rightarrow0.5\rightarrow0.25\rightarrow0$ 向下选择：

$$
p_i^{\mathrm{cand}}(t;\ell)
=
\rho_{i,\ell}^{\mathrm{cand}}(t)P_i^{\max},
\qquad
\rho_{i,\ell}^{\mathrm{cand}}(t)
\in\{0,0.25,0.5,1.0\},
\qquad
p_i^{\mathrm{cand}}(t;1)=p_i^{\mathrm{prop}}(t).
$$

其中候选序列从 actor 提出的绝对离散等级开始并保持降序；若 proposed 等级为 $0.5$，候选序列为 $0.5,0.25,0$，而不是再次将 $p_i^{\mathrm{prop}}$ 乘以一个比例。这样 candidate 只表示 executor 的临时检查值，不重构既有功率档位。


若通信候选已经通过前序 deterministic arbitration，但 energy feasibility search 将其 candidate power 降至 $0$，则最终执行结果同样 canonicalize 为 communication idle。该处理在本槽不触发 backtracking 或 re-arbitration；前序节点独占无线扫描中已经被拒绝的低优先级 proposal 不因该降档而复活。

$$
p_i^{\mathrm{cand}}(t;\ell^\star)=0
\ \text{and final }p_i^{\mathrm{exec}}(t)=0
\quad\Longrightarrow\quad
\text{communication idle}.
$$

节点独占无线扫描得到的接受结果在功率降档完成前只属于 executor 内部的暂定结果；下述 $y_{ij}(t)$、$x_{ij,r}(t)$、$\mathcal S_i^{\mathrm{exec}}(t)$ 和 $p_i^{\mathrm{exec}}(t)$ 均指完成该 canonicalization 后的最终执行结果。

在当前模型中，联合执行器对已选择的固定资源组内部资源单元不做部分删除、增加或重新分配；对提案链路只决定是否接受。于是链路 $i\rightarrow j$ 的最终实际资源占用指示为：

$$
x_{ij,r}(t)
=
y_{ij}(t)\mathbb 1\left[r\in\mathcal S_i^{\mathrm{prop}}(t)\right].
$$

其中 $y_{ij}(t)=1$ 表示链路经联合执行器接受；当 $y_{ij}(t)=0$ 时，$x_{ij,r}(t)=0$ 对所有 $r$ 成立，即使提案集合非空。链路在已分配资源上的总功率按实际占用资源单元数平均分配：

$$
p_{ij,r}(t)
=
\begin{cases}
\dfrac{p_i^{\mathrm{exec}}(t)}
{|\mathcal S_i^{\mathrm{exec}}(t)|},
& x_{ij,r}(t)=1,\\[6pt]
0,
& x_{ij,r}(t)=0.
\end{cases}
$$

联合执行器仲裁后的实际资源集合定义为：

$$
\mathcal S_i^{\mathrm{exec}}(t)
=
\left\{
 r\in\mathcal K_{\mathrm{RU}}:
 \sum_{j\in\mathcal U,\ j\ne i}x_{ij,r}(t)>0
\right\}.
$$

它表示发送 UAV $i$ 在执行器完成硬约束仲裁后的最终实际占用资源单元集合。由于每个发送 UAV 每槽至多有一条满足 $y_{ij}(t)=1$ 的实际接受发送链路，若该唯一链路为 $i\rightarrow j$，则：

$$
\mathcal S_i^{\mathrm{exec}}(t)
=
\left\{r\in\mathcal K_{\mathrm{RU}}:x_{ij,r}(t)=1\right\}.
$$

若发送 UAV $i$ 没有任何被接受的发送链路，则 $\mathcal S_i^{\mathrm{exec}}(t)=\varnothing$；此时 $\mathcal S_i^{\mathrm{prop}}(t)$ 仍可能非空。对满足 $y_{ij}(t)=1$ 的唯一实际接受链路，有：

$$
\left|\mathcal S_i^{\mathrm{exec}}(t)\right|>0
\Longleftrightarrow
\sum_{r=1}^{R}x_{ij,r}(t)>0,
\qquad y_{ij}(t)=1.
$$

联合执行器完成接受或降档后，定义最终实际总发射功率：

$$
p_i^{\mathrm{exec}}(t)
=
\begin{cases}
0,
& \sum_{j\in\mathcal U,\ j\ne i}y_{ij}(t)=0
\ \text{或}
|\mathcal S_i^{\mathrm{exec}}(t)|=0,\\[4pt]
p_i^{\mathrm{cand}}(t;\ell^\star),
& y_{ij^\star}(t)=1,\ |\mathcal S_i^{\mathrm{exec}}(t)|>0,
\end{cases}
$$

其中 $j^\star$ 是发送 UAV $i$ 唯一被接受的目的 UAV，$\ell^\star$ 是 executor 最终接受的候选档位。被拒绝、退化为通信 idle 或没有实际资源占用时，$p_i^{\mathrm{exec}}(t)=0$；接受降档时允许 $p_i^{\mathrm{exec}}(t)\ne p_i^{\mathrm{prop}}(t)$。$p_i^{\mathrm{exec}}(t)$ 与 $y_{ij}(t)$、$x_{ij,r}(t)$ 以及 $\mathcal S_i^{\mathrm{exec}}(t)$ 属于同一最终执行阶段。


因此，$p_i^{\mathrm{exec}}(t)=0$ 的最终语义唯一为 physical communication idle：

$$
\boxed{
p_i^{\mathrm{exec}}(t)=0
\Longrightarrow
\text{no executed communication}
}
$$

执行器必须在最终结果形成前同步清除该 UAV 的通信执行变量：

$$
p_i^{\mathrm{exec}}(t)=0
\Longrightarrow
y_{ij}(t)=0\ \forall j,
\qquad
x_{ij,r}(t)=0\ \forall j,r,
\qquad
\mathcal S_i^{\mathrm{exec}}(t)=\varnothing.
$$

不得保留 $y_{ij}(t)=1$ 或 $\mathcal S_i^{\mathrm{exec}}(t)\ne\varnothing$ 与 $p_i^{\mathrm{exec}}(t)=0$ 并存的半执行状态；因此后续 SINR、service、energy、interference measurement 和 outage 均不由该 idle 结果产生。

链路未被接受时，实际资源集合为空；后续实际服务、通信能耗和 outage 资格均使用 $\mathcal S_i^{\mathrm{exec}}(t)$ 或等价的 $x_{ij,r}(t)$，不使用 $\mathcal S_i^{\mathrm{prop}}(t)$。

当分母为零时链路未获得资源，约定所有 $p_{ij,r}(t)=0$，且不计算该链路的有效服务。

在接收噪声功率为 $N_0B_{\mathrm{RU}}F$ 时，资源单元级 SINR 定义如下。这里的 $x_{ij,r}(t)$、$p_{ij,r}(t)$ 和 $y_{ij}(t)$ 是联合执行器处理后的实际接受结果：只有在所有 UAV 生成动作提案、执行器完成半双工与节点独占无线调度、能量等硬约束处理，且实际接受变量、资源占用、功率和当前真实信道均确定后，环境才计算当前真实 SINR；该量不进入槽初 actor 的局部观测。

$$
\mathrm{SINR}_{ij,r}(t)
=
\frac{
x_{ij,r}(t)p_{ij,r}(t)|h_{ij,r}(t)|^2
}{
N_0B_{\mathrm{RU}}F
+
\sum_{(k,l)\ne(i,j)}
x_{kl,r}(t)p_{kl,r}(t)|h_{kj,r}(t)|^2
}.
$$

其中 $N_0$ 是噪声谱密度，$B_{\mathrm{RU}}$ 是单个等效资源单元带宽，$F$ 是由噪声系数换算得到的线性因子。分母中的求和仅包含同一资源单元上被联合执行器接受的其他传输边。节点独占无线调度只排除共享端点的边，并不建立全局正交资源或无干扰通信；端点不相交的并发链路若执行 RU 集合重叠，仍按上式进入相应接收端的 SINR 干扰求和项。该真实 SINR 始终是线性无量纲比值，不是 dB 值；它仅用于环境状态转移、有效速率、实际服务量、outage 以及集中训练期允许使用的真实全局状态。
为使槽末测量与上述真实 SINR 的干扰项完全一致，若当前槽存在已被联合执行器接受的接收边 $i\rightarrow j$，则在执行结果已确定后定义：

$$
I_{ij,r}^{\mathrm{meas}}(t)
=
\sum_{\substack{k,l\in\mathcal U,\ k\ne l\\(k,l)\ne(i,j)}}
x_{kl,r}(t)p_{kl,r}(t)\left|h_{kj,r}(t)\right|^2.
$$

该式直接复用真实 SINR 分母中的 interference term；由于节点独占无线调度约束，接收 UAV $j$ 至多对应一条当前已接受的入边，因此历史摘要仍以接收端形式记作 $I_{j,r}^{\mathrm{meas}}(t)\equiv I_{ij,r}^{\mathrm{meas}}(t)$。计算前提是联合执行器已经确定 $y$、$x$、$p^{\mathrm{exec}}$ 和 $p_{ij,r}$；求和只包含最终 executed 的 $x_{kl,r}(t)$ 和 $p_{kl,r}(t)$，不包含 proposed power、candidate power、被拒绝的传输或没有执行资源的链路。式中没有 $N_0B_{\mathrm{RU}}F$，所以该测量是 interference-only，不是 interference-plus-noise；热噪声只在构造 $\widehat Z_{j,r}^{\mathrm{hist}}(t)$ 时加入一次。

设 $\gamma_{\min}^{\mathrm{dB}}$ 为 dB 表示的最小可用 SINR 阈值，并定义对应的线性阈值为：

$$
\gamma_{\min}^{\mathrm{lin}}
=
10^{\gamma_{\min}^{\mathrm{dB}}/10}.
$$

本模型中 $\gamma_{\min}^{\mathrm{dB}}=-3\ \mathrm{dB}$，因此 $\gamma_{\min}^{\mathrm{lin}}=10^{-3/10}\approx0.5012$。所有与线性真实 SINR 的数值比较均使用 $\gamma_{\min}^{\mathrm{lin}}$；dB 阈值仅用于参数说明、报告和可读性展示。链路有效速率为：

$$
R_{ij}^{\mathrm{eff}}(t)
=
\sum_{r=1}^{R}
B_{\mathrm{RU}}
\log_2
\left(
1+\mathrm{SINR}_{ij,r}(t)
\right)
\mathbb 1
\left\{
\mathrm{SINR}_{ij,r}(t)\ge\gamma_{\min}^{\mathrm{lin}}
\right\}.
$$

$\gamma_{\min}^{\mathrm{dB}}$ 可在 $-6$、$0$ 和 $3$ dB 之间做敏感性分析，并按上述转换同步得到对应的 $\gamma_{\min}^{\mathrm{lin}}$。有效速率是环境真实服务计算的输入，不应与 actor 只能获得的历史干扰链路质量代理量混淆。

outage 样本只由联合执行器最终接受并实际发起传输的链路产生。$a_i^{\mathrm{tx}}(t)=j$ 仅表示 actor 的传输动作提案，不能直接生成 outage 样本。令槽初已绑定传输队列中的剩余 bit 为：

$$
Q_{ij}^{\mathrm{bit}}(t)
=
\sum_{n\in Q_{i\rightarrow j}^{\mathrm{tx}}(t)}D_n^{\mathrm{rem}}(t).
$$

实际传输尝试指示量定义为：

$$
\chi_{ij}^{\mathrm{att}}(t)
=
\mathbb 1\left[y_{ij}(t)=1\right]
\mathbb 1\left[p_i^{\mathrm{exec}}(t)>0\right]
\mathbb 1\left[\left|\mathcal S_i^{\mathrm{exec}}(t)\right|>0\right]
\mathbb 1\left[Q_{ij}^{\mathrm{bit}}(t)>0\right].
$$

在 $y_{ij}(t)=1$ 时，$\left|\mathcal S_i^{\mathrm{exec}}(t)\right|>0$ 与 $\sum_{r=1}^{R}x_{ij,r}(t)>0$ 等价，因此也可写为：

$$
\chi_{ij}^{\mathrm{att}}(t)
=
\mathbb 1\left[y_{ij}(t)=1\right]
\mathbb 1\left[p_i^{\mathrm{exec}}(t)>0\right]
\mathbb 1\left[\sum_{r=1}^{R}x_{ij,r}(t)>0\right]
\mathbb 1\left[Q_{ij}^{\mathrm{bit}}(t)>0\right].
$$

其中 $y_{ij}(t)$、$p_i^{\mathrm{exec}}(t)$ 和 $x_{ij,r}(t)$ 均指联合执行器处理后的实际接受、最终实际功率和实际资源占用结果。只有 $\chi_{ij}^{\mathrm{att}}(t)=1$ 才生成一个实际传输尝试样本；同一发送 UAV 每槽至多生成一个此类样本。actor 主动选择 idle、执行器拒绝或因半双工与节点独占无线冲突退化为 idle、能量降档至零功率、实际资源占用为空或传输队列为空时，$\chi_{ij}^{\mathrm{att}}(t)=0$，这些情况不属于无线 outage。

对单次实际传输尝试，沿用当前真实 SINR、有效速率和既有无线 outage 判定条件：

$$
o_{ij}(t)
=
\begin{cases}
1,
&
\chi_{ij}^{\mathrm{att}}(t)=1,\
R_{ij}^{\mathrm{eff}}(t)=0,
\\
0,
&
\chi_{ij}^{\mathrm{att}}(t)=1,\
R_{ij}^{\mathrm{eff}}(t)>0,
\\
\mathrm{NA},
&
\chi_{ij}^{\mathrm{att}}(t)=0.
\end{cases}
$$

因此，实际非零功率和非零资源均已发出数据、但真实有效速率为零时，仍按现有无线 outage 条件记为 $o_{ij}(t)=1$；$o_{ij}(t)=\mathrm{NA}$ 不得在任何统计中转为 0 或 1。对当前时隙 $t$，历史 outage 特征只使用已结束时隙中的实际传输尝试集合：

$$
\mathcal H_{ij}^{\mathrm{att}}(t)
=
\left\{
\tau:\,0\le\tau\le t-1,\ \chi_{ij}^{\mathrm{att}}(\tau)=1
\right\}.
$$

当 $|\mathcal H_{ij}^{\mathrm{att}}(t)|>0$ 时，outage 特征为：

$$
\hat q_{ij}^{\mathrm{out}}(t)
=
\frac{
\sum_{\tau\in\mathcal H_{ij}^{\mathrm{att}}(t)}
\mathbb 1\{o_{ij}(\tau)=1\}
}{
|\mathcal H_{ij}^{\mathrm{att}}(t)|
}.
$$

当 $|\mathcal H_{ij}^{\mathrm{att}}(t)|=0$ 时，使用固定缺省值并令 $m_{ij}^{\mathrm{out}}(t)=0$；有历史实际传输尝试时令该 mask 为 1。缺省值不表示 outage，NA 也不进入上述分子或分母。累计 outage 指标的实际尝试样本集合定义为：

$$
\mathcal A^{\mathrm{att}}
=
\left\{
(t,i,j):\chi_{ij}^{\mathrm{att}}(t)=1
\right\},
$$

并按实际传输尝试计算：

$$
P_{\mathrm{out}}
=
\begin{cases}
\displaystyle
\frac{
\sum_{(t,i,j)\in\mathcal A^{\mathrm{att}}}o_{ij}(t)
}{|\mathcal A^{\mathrm{att}}|},
& |\mathcal A^{\mathrm{att}}|>0,\\[1.2ex]
\mathrm{NA},
& |\mathcal A^{\mathrm{att}}|=0.
\end{cases}
$$

按链路、episode、场景或滑动窗口分别统计时，分母均只包含对应窗口内的实际传输尝试次数。等效 RB 级失效率另行记录为：

$$
O_{ij}^{\mathrm{RB}}(t)
=
\frac{
\sum_{r:x_{ij,r}(t)=1}
\mathbb 1\{\mathrm{SINR}_{ij,r}(t)<\gamma_{\min}^{\mathrm{lin}}\}
}{
\sum_{r=1}^{R}x_{ij,r}(t)
},
$$

且仅当 $\chi_{ij}^{\mathrm{att}}(t)=1$ 时计算；否则记为 NA。该口径使链路 outage、等效 RB 级失效率和未实际传输状态保持可区分。

## 2.10 跨时隙传输服务模型

tx_select 的候选集合只来自槽初已经存在的非空传输队列：

$$
a_i^{\mathrm{tx}}(t)
\in
\{\mathrm{idle}\}
\cup
\left\{
j:
Q_{i\rightarrow j}^{\mathrm{tx}}(t)\ne\varnothing
\right\}.
$$

该动作只选择已有服务边的目的 UAV，不负责新任务目的地绑定。新任务的 route 输出即使在同一时隙存在，也只能在槽末进入本地或传输队列。

对被选择的链路，先聚合槽初队列中所有任务的剩余 bit：

$$
Q_{ij}^{\mathrm{bit}}(t)
=
\sum_{n\in Q_{i\rightarrow j}^{\mathrm{tx}}(t)}
D_n^{\mathrm{rem}}(t).
$$

根据有效速率得到本时隙链路服务预算：

$$
S_{ij}^{\mathrm{tx}}(t)
=
\min
\left\{
Q_{ij}^{\mathrm{bit}}(t),
R_{ij}^{\mathrm{eff}}(t)\Delta t
\right\}.
$$

对联合执行器接受的链路，通信实际活跃时间定义为：

$$
\tau_{ij}^{\mathrm{tx}}(t)
=
\begin{cases}
\min\left\{
\Delta t,
\dfrac{Q_{ij}^{\mathrm{bit}}(t)}
{R_{ij}^{\mathrm{eff}}(t)}
\right\},
&
y_{ij}(t)=1,\ |\mathcal S_i^{\mathrm{exec}}(t)|>0,\ p_i^{\mathrm{exec}}(t)>0,\
R_{ij}^{\mathrm{eff}}(t)>0,
\\[8pt]
\Delta t,
&
y_{ij}(t)=1,\ |\mathcal S_i^{\mathrm{exec}}(t)|>0,\ p_i^{\mathrm{exec}}(t)>0,\
R_{ij}^{\mathrm{eff}}(t)=0,
\\[4pt]
0,
&
\text{其他情况}.
\end{cases}
$$

其中 $y_{ij}(t)=1$ 表示链路经联合执行器接受。有效速率大于零且整个传输队列在槽内提前清空时，发送端立即停止主动发射，因而实际活跃时间小于 $\Delta t$；有效速率为零但已以非零功率占用资源尝试发射时，视为整槽发射尝试。未被接受、通信 idle、功率为零或没有实际资源占用的链路均不产生通信活跃时间，也不产生 outage 样本；本定义与 $\chi_{ij}^{\mathrm{att}}(t)$ 的实际执行口径一致。

服务预算沿 $Q_{i\rightarrow j}^{\mathrm{tx}}(t)$ 的 EDF 顺序逐任务扣除：先服务队首任务，扣除其剩余 bit 与当前预算的较小值；预算仍有剩余时继续处理下一任务；直到预算耗尽或队列为空。对任务 $n$，令 $b_n^{\mathrm{tx}}(t)$ 为本槽实际扣除的 bit，则更新关系为：

$$
D_n^{\mathrm{rem}}(t+1)
=
D_n^{\mathrm{rem}}(t)-b_n^{\mathrm{tx}}(t),
\qquad
0\le b_n^{\mathrm{tx}}(t)\le D_n^{\mathrm{rem}}(t).
$$

一条被选择的链路可以在同槽完成多个小任务，但不能同时服务另一个目的链路。传输完成的任务在槽末进入目的 UAV 的远程 CPU 队列，计算量仍为其完整 $C_n$；本槽新进入 CPU 队列的任务下一时隙才可计算。因此传输完成任务不能同槽计算。

## 2.11 计算与能耗模型

每个计算 UAV 每槽最多选择一个槽初非空 CPU 队列。CPU 队列动作候选集合为：

$$
a_j^{\mathrm{cpuq}}(t)
\in
\{\mathrm{idle},\mathrm{local}\}
\cup
\left\{
i:
Q_{i\rightarrow j}^{\mathrm{cpu}}(t)\ne\varnothing
\right\}.
$$

其中 local 表示 $Q_j^{\mathrm{loc}}(t)$，远程索引 $i$ 表示来源为 $i$ 的队列 $Q_{i\rightarrow j}^{\mathrm{cpu}}(t)$。CPU 频率采用离散档位：

$$
f_j(t)
\in
\{0,0.25,0.5,1.0\}f_j^{\max}.
$$

设被选 CPU 队列的 EDF 队首任务为 $n^\star$，其本槽实际执行 cycle 为：

$$
c_j^{\mathrm{cpu}}(t)
=
\begin{cases}
\min\left\{
C_{n^\star}^{\mathrm{rem}}(t),
f_j(t)\Delta t
\right\},
&
f_j(t)>0\text{ 且选择了有效非空 CPU 队列},
\\[8pt]
0,
&
\text{其他情况}.
\end{cases}
$$

第一版主模型每个 UAV 每槽只服务一个 CPU 队首任务，不把剩余 CPU 服务预算继续分配给第二个 CPU 队列。队首剩余 cycle 更新为：

$$
C_{n^\star}^{\mathrm{rem}}(t+1)
=
C_{n^\star}^{\mathrm{rem}}(t)-c_j^{\mathrm{cpu}}(t),
\qquad
0\le c_j^{\mathrm{cpu}}(t)\le C_{n^\star}^{\mathrm{rem}}(t).
$$

CPU 实际活跃时间定义为：

$$
\tau_j^{\mathrm{cpu}}(t)
=
\begin{cases}
\dfrac{c_j^{\mathrm{cpu}}(t)}{f_j(t)},
&
f_j(t)>0\text{ 且选择了有效非空 CPU 队列},
\\[8pt]
0,
&
\text{其他情况}.
\end{cases}
$$

因此 $0\le\tau_j^{\mathrm{cpu}}(t)\le\Delta t$。任务在槽内提前完成时，CPU 随即停止本任务的动态计算活动；剩余槽时间不转而服务第二个任务，也不产生 CPU 动态能耗。

实际通信能耗由接受的传输边、其总发射功率和实际活跃时间决定：

$$
E_i^{\mathrm{tx}}(t)
=
\sum_{j\in\mathcal U}
y_{ij}(t)\,p_i^{\mathrm{exec}}(t)\,\tau_{ij}^{\mathrm{tx}}(t).
$$

其中 $p_i^{\mathrm{exec}}(t)$ 是链路在全部最终执行资源上的总发射功率；由于每个发送 UAV 每槽至多有一条接受链路，上式不会重复累计多条发送活跃时间。虽然仍有 $\sum_r p_{ij,r}(t)=p_i^{\mathrm{exec}}(t)$，实际通信能耗不再无条件乘以完整 $\Delta t$。

CPU 动态能耗按实际活跃时间计算：

$$
E_j^{\mathrm{cpu}}(t)
=
\kappa_j f_j^3(t)\tau_j^{\mathrm{cpu}}(t)
=
\kappa_j f_j^2(t)c_j^{\mathrm{cpu}}(t).
$$

当 $f_j(t)=0$ 或 CPU queue 为 idle 时，$c_j^{\mathrm{cpu}}(t)=0$、$\tau_j^{\mathrm{cpu}}(t)=0$ 且 $E_j^{\mathrm{cpu}}(t)=0$。在节点独占无线仲裁形成候选接受集合后、真实有效速率、最终发射功率和实际服务结果形成前，联合执行器仍不能知道通信实际活跃时间，因此对候选动作使用基于 candidate power 的保守能量预留上界。该阶段读取槽初已绑定队列、$\mathcal S_i^{\mathrm{prop}}(t)$ 和当前正在检查的 $p_i^{\mathrm{cand}}(t;\ell)$；不读取当前真实 SINR、有效速率、实际服务或实际活跃时间，预留仍按提案资源集合和候选功率计算：

$$
\overline E_i^{\mathrm{tx,cand}}(t;\ell)
=
\begin{cases}
p_i^{\mathrm{cand}}(t;\ell)\Delta t,
&
\begin{gathered}
\exists j:\ a_i^{\mathrm{tx}}(t)=j,\ Q_{i\rightarrow j}^{\mathrm{tx}}(t)\ne\varnothing,\ Q_{ij}^{\mathrm{bit}}(t)>0,\\
|\mathcal S_i^{\mathrm{prop}}(t)|>0,\ p_i^{\mathrm{cand}}(t;\ell)>0
\end{gathered},
\\[4pt]
0,
&
\text{其他情况}.
\end{cases}
$$

上式中的 $p_i^{\mathrm{cand}}(t;\ell)$ 仅表示 executor 当前正在检查的候选功率档位，而不是对最终接受结果的预判；预留条件不以 $y_{ij}(t)=1$、$|\mathcal S_i^{\mathrm{exec}}(t)|>0$ 或 $\sum_r x_{ij,r}(t)>0$ 为前提。每个候选档位都单独计算该临时预留；候选发送若在半双工与节点独占无线调度或能量等硬约束阶段被拒绝，或最终降档为通信 idle，则候选预留释放且 $E_i^{\mathrm{tx}}(t)=0$。若候选被接受，未实际消耗的预留在槽末释放，剩余能量只按实际主动能耗扣除。

CPU 预留可以根据槽初已知的队首任务和频率确定：

$$
\overline\tau_j^{\mathrm{cpu}}(t)
=
\begin{cases}
\min\left\{
\Delta t,
\dfrac{C_{n^\star}^{\mathrm{rem}}(t)}{f_j(t)}
\right\},
&
f_j(t)>0\text{ 且 CPU 队列有效},
\\[8pt]
0,
&
\text{其他情况},
\end{cases}
$$

$$
\overline E_j^{\mathrm{cpu}}(t)
=
\kappa_j f_j^3(t)\overline\tau_j^{\mathrm{cpu}}(t).
$$

联合主动能量预留为：

$$
\overline E_i^{\mathrm{act,cand}}(t;\ell)
=
\overline E_i^{\mathrm{tx,cand}}(t;\ell)+\overline E_i^{\mathrm{cpu}}(t),
\qquad
\overline E_i^{\mathrm{act,cand}}(t;\ell)
\le
E_i^{\mathrm{res}}(t).
$$


在节点独占无线仲裁形成接受集合后、真实服务开始前，联合执行器使用上述候选预留上界检查离散降档组合的硬可行性；真实服务完成后，实际主动能耗和剩余能量更新为：

$$
E_i^{\mathrm{act}}(t)
=
E_i^{\mathrm{tx}}(t)+E_i^{\mathrm{cpu}}(t),
\qquad
E_i^{\mathrm{act}}(t)
\le
\overline E_i^{\mathrm{act,cand}}(t;\ell^\star)
\le
E_i^{\mathrm{res}}(t),
$$

$$
E_i^{\mathrm{res}}(t+1)
=
E_i^{\mathrm{res}}(t)-E_i^{\mathrm{act}}(t).
$$

预留但未实际消耗的能量在时隙结束时释放，不从剩余能量中扣除。上述约束只保证主动通信与计算的能量守恒，不包含推进、悬停、姿态或机载传感器能耗。初始能量、参考 CPU 和参考速率必须在 Gate P pilot 后校准，不能把未校准数值直接解释为实际硬件功耗。

若联合动作的预留能量不可行，执行器只在预定义离散档位中降级，降档序列为：

$$
1.0\rightarrow0.5\rightarrow0.25\rightarrow0.
$$

执行器降档顺序冻结为：对每个 UAV，先保持其 actor 提议的 CPU 频率不变；再对接受集合中的每个通信候选沿当前已有的绝对 candidate power 档位从高到低（从 $p_i^{\mathrm{prop}}(t)$ 所在档位开始）逐档检查，并为每个候选计算 $\overline E_i^{\mathrm{act,cand}}(t;\ell)$。选择满足 $\overline E_i^{\mathrm{act,cand}}(t;\ell)\le E_i^{\mathrm{res}}(t)$ 的最高可行通信档位并停止；若包括 $0$ 在内的全部通信档位在当前 CPU 频率下均不可行，才按 actor 提议档位向下逐档检查既有 CPU 频率档位，直到找到最高可行 CPU 频率；如通信功率为 $0$ 且 CPU 也降至最低 idle 档位后仍不可行，沿用当前正文已有的最终安全 idle 处理。全流程不新增功率或 CPU 频率档位，不使用随机 tie-break，也不在找到最高可行档位后继续无必要降档。

## 2.12 七分支离散动作与硬约束

每个 UAV 的联合动作固定为七个离散分支：

$$
a_i(t)
=
\left(
a_i^{\mathrm{route}}(t),
a_i^{\mathrm{tx}}(t),
a_i^{\mathrm{rg}}(t),
a_i^{\mathrm{width}}(t),
a_i^{\mathrm{pow}}(t),
a_i^{\mathrm{cpuq}}(t),
a_i^{\mathrm{cpuf}}(t)
\right).
$$

七个分支的语义如下。route 只处理槽初未绑定 EDF 任务，候选为 local、defer 或 $\mathcal N_i(t)$ 中的一个远程目的地；tx 只选择槽初已有的非空传输队列；resource group 选择 idle 或五个固定资源组之一并映射为 $g_i(t)$；resource width 保持定义域 $\{1,2\}$ 并映射为 $w_i(t)$，父通信分支 inactive 时采用 canonical 值 $1$；power level 选择离散比例 $a_i^{\mathrm{pow}}(t)\in\{0,0.25,0.5,1\}$ 并映射为 proposed power $p_i^{\mathrm{prop}}(t)$；CPU queue 选择 local、一个远程 CPU 队列或 idle；CPU frequency 选择 $0$、$0.25f_i^{\max}$、$0.5f_i^{\max}$ 或 $f_i^{\max}$。

动作 mask 在槽初根据可用队列和资源状态生成：

- 没有未绑定任务时，route 固定为 idle；
- 不在 $\mathcal N_i(t)$ 中的远程目的地不可选；
- 已绑定到 local 或 remote 的任务不得再次进入 route 候选；route 候选始终只来自 $Q_i^{\mathrm{unb}}(t)$；
- 没有传输队列时，tx 和 resource group 固定为 idle；width 不新增 idle 类别而使用 canonical 值 $1$，power 沿用既有 $0$ 档位，使 proposed power 为 $0$，这些 inactive 编码均不触发物理资源申请；
- 第五组不能与宽度 2 组合；
- 没有本地或远程 CPU 队列时，CPU queue 和 CPU frequency 固定为 idle；
- 按预留能量上界明显超过剩余能量的离散档位提前 mask；
- 半双工与节点独占无线冲突不由单个 actor 的本地 mask 假定解决，而由联合执行器解析。

为明确联合解析的硬约束，令 $y_{ij}(t)\in\{0,1\}$ 表示传输边 $i\rightarrow j$ 是否被接受，令 $u_{jk}(t)\in\{0,1\}$ 表示 UAV $j$ 是否选择来源为 $k$ 的 CPU 队列，其中 $k=j$ 表示本地队列。则每个发送 UAV 和每个计算 UAV 分别满足：

$$
\sum_{j\in\mathcal U\setminus\{i\}}y_{ij}(t)\le1,
\qquad
\sum_{k\in\mathcal U}u_{ik}(t)\le1.
$$

为同时实施半双工收发模式和节点独占无线调度，最终接受边满足：

$$
\sum_{j\ne i}y_{ij}(t)
+
\sum_{k\ne i}y_{ki}(t)
\le 1,
\qquad
\forall i\in\mathcal U.
$$

该不等式比普通半双工约束更强：它同时禁止同一 UAV 出现两条出边、两条入边以及一条入边与一条出边。忽略方向后，每个时隙的最终接受无线边集构成一个 matching；此处不要求该 matching 为 induced matching、maximal matching 或 maximum matching。若 $\{i,j\}\cap\{k,l\}=\varnothing$，则边 $i\rightarrow j$ 与 $k\rightarrow l$ 可在同一时隙执行，因此该模型不是全局 TDMA、全局正交调度或全系统单链路调度。

CPU 与无线通信模块被视为并行资源，因此上述节点独占无线约束仅作用于无线边，不禁止同一 UAV 在发送或接收时执行 CPU 服务；二者只通过既有联合主动能量可行性约束耦合。联合执行器先收集所有合法、非 idle 且 $p_i^{\mathrm{prop}}(t)>0$ 的候选通信边；raw zero-power proposal 已在此前 canonicalize 为 idle。对每条候选边 $i\rightarrow j$，令 $n_{ij}^{\star}$ 为 $Q_{i\rightarrow j}^{\mathrm{tx}}(t)$ 的 EDF 队首任务，并令 $\widehat\Gamma_{ij}^{\mathrm{hist}}(t)$ 为 2.8.5 在该候选边的 $\mathcal S_i^{\mathrm{prop}}(t)$ 上定义的唯一 scalar historical quality。executor 不改用 per-RU 任意选择、$\mathcal S_i^{\mathrm{exec}}(t)$、当前真实 SINR 或随机 fallback。

候选通信边的唯一仲裁键冻结为：

$$
\Pi_{ij}^{\mathrm{tx}}(t)
=
\left(
\operatorname{slack}_{n_{ij}^{\star}}(t),
-\widehat\Gamma_{ij}^{\mathrm{hist}}(t),
i,
j
\right).
$$

所有候选边按上述键作升序字典序排序：slack 越小越优先，历史链路质量代理量越大越优先，随后按发送 UAV ID $i$、接收 UAV ID $j$ 升序 tie-break。排序完成后从空的接受集合开始逐项扫描。一条边被接受后同时占用其发送端和接收端；后续与已接受边共享任一端点的候选被拒绝，端点不相交的候选仍可被接受。该仲裁不依据 RU 是否重叠来排除端点不相交的边；其 RU 重叠影响由既有 SINR 干扰模型处理。被拒绝链路满足 $y_{ij}(t)=0$、$\mathcal S_i^{\mathrm{exec}}(t)=\varnothing$ 和 $p_i^{\mathrm{exec}}(t)=0$，且不生成 outage 样本。该确定性过程不承诺接受集为 induced matching、maximal matching 或 maximum matching，也不承诺任何匹配最优性。

CPU queue 分支仍由 actor 提出，执行器不新增全局 CPU 调度触发机制；每个 UAV 只执行当前已选的一个有效 CPU 队列及其 EDF 队首任务。若实现内部确需在多个有效 CPU 候选之间比较，则使用 $(\operatorname{slack}_{n^{\star}}(t),n^{\star},j)$ 的升序字典序，其中 $n^{\star}$ 是固定任务 ID、$j$ 是计算 UAV ID；这只是把既有 EDF、固定任务 ID 和 UAV ID tie-break 写明，不改变 CPU 队列语义。

因此，在固定槽初状态、固定联合动作提案和固定环境状态下，联合执行器定义为：

$$
\mathcal E:
(\text{槽初状态},\ \text{联合动作提案},\ \text{固定环境状态})
\mapsto
(\text{最终执行动作}).
$$

该映射不得依赖 Python set/dict 的偶然遍历顺序、未声明的列表顺序、随机 tie-break 或实现者自行选择的升降序；完成节点独占无线扫描后才进入上述能量降档顺序。若后续 energy downgrade 使 candidate power 降至 $0$，只执行上述 final canonicalization，不在本槽 backtrack 或 re-arbitrate。

## 2.13 时隙事件顺序

### 2.13.1 episode reset 与 slot-0 readiness

每个 episode 在进入 $\mathrm{actor}(0)$ 之前执行一次完整 reset。reset 后环境 slot index 直接为 $t=0$，不存在需要读取的 $t=-1$ 状态；actor observation 所依赖的每个 state、history 和 mask 必须已经具有合法数值，或具有明确的 $\mathrm{NA}$ 与可用性 mask，不通过 negative index、lazy initialization 或 Python 默认空值补齐。

能量初值对每个 UAV 满足：
$$
E_i^{\mathrm{res}}(0)=E_i^0,
\qquad
E_i^0>0.
$$
$E_i^0$ 是 episode 初始主动能量预算，$E_i^{\mathrm{res}}(0)$ 是 reset 后剩余主动能量，二者不与 reservation energy 混用。

在本模型不设置 episode initial backlog 的约定下，slot 0 开始前正文已有的任务与队列结构均为空：
$$
\mathcal T_0=\varnothing,
\qquad
Q_i^{\mathrm{unb}}(0)=Q_i^{\mathrm{loc}}(0)=\varnothing,
\qquad
Q_{i\rightarrow j}^{\mathrm{tx}}(0)
=
Q_{i\rightarrow j}^{\mathrm{cpu}}(0)
=
\varnothing,
\quad
i\ne j.
$$
$\mathcal T_0$ 使用 2.14 已定义的 active task collection，不引入第二套任务集合或队列名称。

正文当前只给出任务记录中的 $id_n$ 和固定任务 ID tie-break，未另行冻结起始编号。因此 reset 时 task-ID counter 置为 $0$，新任务按 $n=0,1,2,\ldots$ 获得 deterministic bookkeeping ID。槽末同一时隙的新任务先按 UAV ID 升序处理，同一 UAV 内再沿 arrival generator 的稳定生成顺序单调递增编号；编号顺序不依赖 Python 容器遍历顺序，也不改变任务分布或到达数量。

reset 同时保持已冻结的移动、阴影和真实信道边界：直接提供 $\mathbf p_i(0)$、$\mathbf v_i(0)$，设置 $\Delta s_{ij}(0)=0$，采样 $X_{ij}(0)\sim\mathcal N(0,\sigma_s^2)$，并生成首个合法 true-channel sample $h_{ij,r}(0)$。不定义或访问 $\mathbf p_i(-1)$、$X_{ij}(-1)$ 或 $h_{ij,r}(-1)$。

陈旧 CSI、历史干扰和 per-RU 历史质量特征也必须在 $\mathrm{actor}(0)$ 前合法定义：沿用既有 CSI stale index/availability、$\widehat I_{j,r}^{\mathrm{hist}}(0)=I_{j,r}^{\mathrm{def}}$、$m_{j,r}^{I}(0)=0$、$a_j^{\mathrm{msg}}(0)=\mathrm{NA}$ 以及 per-RU historical-quality proxy 与 $m_{ij,r}^{\mathrm{qual}}(0)$ 的缺省值和 mask。executor-only scalar 不在 actor(0) 前计算，而是在本槽 actor proposal 形成 $\mathcal S_i^{\mathrm{prop}}(t)$ 后按 2.8.5 派生。reset 按 2.8.4 的固定 $(i,j,r)$ 顺序生成 $h_{ij,r}(0)$，并按 2.8.5 的同一顺序生成完整 CSI-error tensor；CSI error 不作为持久 $\xi(0)$ state。固定 master seed 和配置下，reset 的信道与 CSI-error realization sequence 必须可复现。

episode-level bookkeeping 在 reset 时初始化为物理上合法的空状态：completed、expired 和 truncated counters 为 $0$；实际 transmission-attempt count、outage numerator 和 denominator 为 $0$，无样本的 outage ratio/average 为 $\mathrm{NA}$；energy accumulator 为 $0$；latency accumulator 和 latency sample list 为空。既有 normalization reference constants 不在 reset 中重新估计。

### 2.13.2 时隙事件顺序

每个 episode 的决策与服务时隙固定为 $t=0,1,\ldots,T-1$，其中 $T-1$ 是最后一个可执行 service slot；不存在 $t=T$ 的额外 actor decision、通信或 CPU service。每个时隙严格按以下顺序执行，以保持任务、bit、cycle、energy、settlement、reward 和 arrival 的因果一致性：

1. 先固定 slot $t$ 的 mobility 与 large-scale state（位置、距离、路径损耗、LoS/NLoS 和 shadowing）；随后按固定 $(i,j,r)$ 字典序为全部有向链路/RU 生成 $h_{ij,r}(t)$ 和完整 $\epsilon_{ij,r}^{\mathrm{CSI}}(t)$ tensor，再依据 stale index 和 availability mask 构造 actor 可见的陈旧 CSI 与历史观测。actor 读取 slot-start state、已有任务、四类队列、候选邻居、历史干扰摘要、延迟消息、历史到达率和对应 masks；历史到达率按 2.4 节仅使用 $A_i(0),\ldots,A_i(t-1)$，$t=0$ 使用固定缺省值和无效 mask，且 actor 不读取本步骤生成的当前真实 $h(t)$。
2. actor 根据槽初局部观测生成七分支动作提案。
3. 对通信 tx proposal 执行 raw zero-power canonicalization：若 $p_i^{\mathrm{prop}}(t)=0$，该 proposal 作为 communication idle 排除，不进入半双工与节点独占无线仲裁、资源、能量、干扰或实际 attempt 语义。
4. 对一个未绑定 EDF 任务执行 route 决策；从槽初已有的非空传输队列中执行 tx_select；解析固定资源组、资源宽度和功率动作，形成 $\mathcal S_i^{\mathrm{prop}}(t)$、$p_i^{\mathrm{prop}}(t)$，并选择一个槽初 CPU 队列和 CPU 频率档位。route 只在槽末 binding，不产生本槽下一阶段 service。
5. 在 $\mathcal S_i^{\mathrm{prop}}(t)$ 形成后，仅由 executor 根据 per-RU historical-quality proxy 及其 valid mask 计算 executor-only scalar $\widehat\Gamma_{ij}^{\mathrm{hist}}(t)$；该 scalar 不回写或修改本槽 actor observation。随后，对剩余合法且非 idle 的通信候选按既有 $\Pi_{ij}^{\mathrm{tx}}(t)$ 升序字典序执行 deterministic executor 和节点独占无线扫描；该步骤形成暂定接受结果，不改变既有仲裁键、字典序或端点占用约束。
6. 对暂定接受通信候选沿既有 $1.0\rightarrow0.5\rightarrow0.25\rightarrow0$ candidate power 顺序进行 power-first energy downgrade；只有通信功率降至 $0$ 且联合能量仍不可行时，才按既有顺序执行 CPU frequency downgrade，并选择最高可行档位。
7. 若 raw proposal 或 candidate downgrade 的最终执行功率为 $0$，执行 final zero-power canonicalization；本槽不进行 backtracking 或 re-arbitration，不复活前序已拒绝的低优先级 proposal。
8. 形成最终 $y_{ij}(t)$、$x_{ij,r}(t)$、$\mathcal S_i^{\mathrm{exec}}(t)$、$p_i^{\mathrm{exec}}(t)$ 和 $f_i(t)$；其中 $p_i^{\mathrm{exec}}(t)=0$ 时最终通信变量必须全部为 idle/zero。
9. 在最终执行变量确定后，结合当前真实槽级信道计算真实 interference、SINR、有效速率、通信 service 和通信实际活跃时间；zero-power idle 不进入真实 SINR desired edge。
10. 更新本槽实际 service 造成的 $D_n^{\mathrm{rem}}$、$C_n^{\mathrm{rem}}$，计算并扣除 actual active energy，释放未实际消耗的 reservation，并按 $\chi_{ij}^{\mathrm{att}}(t)$ 生成 outage 样本；zero-power idle 不产生 interference、$I^{\mathrm{meas}}$、transmission service、tx energy 或 actual attempt。
11. 应用本槽已经发生但只能在槽末生效的 task transition 并完成 bookkeeping：route binding 在槽末生效；local route 锁定 source UAV 并令 $D_n^{\mathrm{rem}}=0$；tx 完成全部剩余 bits 时转入目的 UAV CPU waiting queue；CPU 完成全部 remaining cycles 时形成 completion candidate；任何新转入 CPU queue 的任务都不得在同一槽再次获得 CPU service。
12. 在 post-service/post-transition 状态上执行 settlement：先判断全部 workload 是否完成，完成则记为 $\mathrm{done}$；否则若 $t=t_n^{\mathrm{ddl}}$ 则记为 $\mathrm{expired}$，并立即从所有 active service queues/waiting queues 移除，不进入下一槽 service。
13. 使用 actual energy、post-service/post-settlement task state 和既有 reward 公式计算 $r_t$；reward 仍发生在 slot-end arrival 之前，$A_i(t)$ 不影响同槽 reward。
14. 在当前真实信道和最终 executed action 已确定后，按既有规则形成当前 measurement/history 更新；该更新只使用最终 executed nonzero communication，zero-power idle 不进入 $I^{\mathrm{meas}}$。
15. 将本槽 route/transfer-completion 结果和新到达任务在槽末入队；新任务记为 $A_i(t)$、记录 $t_n^{\mathrm{arr}}=t$，下一时隙才可见和 route；新到达不影响时隙 $t$ 的 actor 或 reward。
16. 若 $t<T-1$，更新下一槽外生移动和 large-scale state，并按固定 $(i,j,r)$ 顺序预生成下一槽真实 $h(t+1)$ 与 CSI-error tensor；历史 CSI 使用既有固定配置陈旧偏移量计算 $\ell_{ij}^{\mathrm{CSI}}(t+1)=t+1-a_{ij}^{\mathrm{CSI}}(t+1)$，不新增独立的 CSI refresh/increment 过程。若 $t=T-1$，执行 terminal bookkeeping 并在边界 $T$ 结束 episode，不构造 $t=T$ 的 actor 或 service。

因此，完整事件依赖关系冻结为：
$$
\text{mobility/large-scale state}
\rightarrow
\text{fixed-order }h(t)\text{ generation}
\rightarrow
\text{fixed-order CSI-error generation}
\rightarrow
\text{slot-start historical observation/masks}
\rightarrow
\text{actor proposal}
\rightarrow
\text{raw zero-power canonicalization}
\rightarrow
\text{resource proposal }(\mathcal S^{\mathrm{prop}},p^{\mathrm{prop}})
\rightarrow
\text{executor-only scalar historical quality}
\rightarrow
\text{deterministic executor}
\rightarrow
\text{energy downgrade}
\rightarrow
\text{final zero-power canonicalization}
\rightarrow
\text{final }(y,x,\mathcal S^{\mathrm{exec}},p^{\mathrm{exec}},f)
\rightarrow
\text{true channel/SINR/service}
\rightarrow
\text{remaining workload update}
\rightarrow
\text{slot-end task transition}
\rightarrow
\text{done/expired settlement}
\rightarrow
\text{reward}
\rightarrow
\text{current measurement/history update}
\rightarrow
\text{slot-end arrival}
\rightarrow
\text{next slot or terminal bookkeeping}.
$$

固定 horizon 的终止规则如下：若任务满足 $t_n^{\mathrm{ddl}}\le T-1$，则它仍在其 deadline slot 接受最后一次合法 service；该槽服务和剩余工作量更新后，完成则记为 $\mathrm{done}$，否则立即记为 $\mathrm{expired}$，episode 在边界 $T$ 到来时不再改写这一结算结果。若任务在边界 $T$ 仍未完成且 $t_n^{\mathrm{ddl}}\ge T$，则只在 episode terminal bookkeeping 中标记为 $\mathrm{truncated}$，不把该标签加入 $\mathcal S_{\mathrm{task}}$，也不将其视为物理任务状态转移。由统一到达过程在最后一个槽末生成的 $A_i(T-1)$ 同样只用于 terminal bookkeeping，直接记为 $\mathrm{truncated}$；它们不进入 $t=T$ 的 route 或 service，也不影响 $r_{T-1}$。

$\mathrm{truncated}$ 与 $\mathrm{expired}$ 的统计边界保持分离：$\mathrm{truncated}$ 只用于 episode 终止统计、trajectory/replay 终止标记和 evaluation accounting，不计入 expired numerator，不虚构 $t_n^{\mathrm{cmp}}$ 或 $T_n^{\mathrm{E2E}}$，并单独报告其数量或比例。E2E completion latency 只对实际完成的任务统计；outage 仍只使用 $0,\ldots,T-1$ 内真实传输尝试，energy 仍只统计这些时隙内的实际主动能耗。系统不增加 $T,T+1,\ldots$ 的 post-horizon drain/service slots，不为未完成任务等待到 done/expired，也不由 truncation 自动产生 $r(T)$ 或其他额外 terminal penalty。

时间轴示例：时隙 $t$ 槽末任务到达，记录 $t_n^{\mathrm{arr}}=t$；时隙 $t+1$ 任务首次可 route，槽末完成绑定；时隙 $t+2$ 首次可接受传输或 CPU 服务。若在时隙 $t+2$ 槽末完成，则 $t_n^{\mathrm{cmp}}=t+2$，端到端时延为 $((t+2)-t)\Delta t=2\Delta t$。

## 2.14 Dec-POMDP、信息边界与系统优化问题

### 2.14.1 状态、局部观测与联合动作

全局状态包含位置、速度、真实信道、建筑物几何、真实业务参数、任务记录、队列、资源和能量等环境变量，可抽象为：

$$
s_t
=
\left(
\{\mathbf q_i(t),\mathbf v_i(t),\boldsymbol{\chi}_i^{\mathrm{res}},E_i^{\mathrm{res}}(t)\}_{i\in\mathcal U},
\mathcal T_t,
\mathcal Q_t,
\mathcal H_t,
\mathcal B,
\boldsymbol{\lambda}
\right).
$$

其中 $\mathcal T_t$ 是活动任务集合，$\mathcal Q_t$ 是四类队列集合，$\mathcal H_t$ 是真实信道与 CSI 历史，$\boldsymbol{\lambda}$ 是环境业务到达参数。该状态用于定义环境转移和集中训练期的全局信息，不表示每个 actor 都能读取所有分量。

UAV $i$ 的局部观测由状态和可用消息历史共同决定：

$$
o_i(t)
=
\mathcal O_i
\left(
s_t,
m_i(t)
\right).
$$

局部观测至少包括自身剩余能量比例、归一化资源能力、历史到达率估计 $\hat{\lambda}_i(t)$ 及其可用性 mask $m_i^\lambda(t)$、本地/传输/CPU 队列摘要、候选传输队列的任务数与剩余 bit、CPU 队列的任务数与剩余 cycle、队首 slack、上一槽动作和资源利用率；邻居特征包括相对位置、相对速度、广播剩余能量、广播资源能力、CPU 负载摘要和消息 AoI；边特征包括估计距离、陈旧期望链路增益或其归一化形式、历史干扰链路质量代理量 $\widehat\Gamma_{ij,r}^{\mathrm{hist}}(t)$、其有效性 mask、上一槽有效速率、仅基于过去实际传输尝试的 outage 率、CSI AoI、历史干扰摘要 AoI、相对速度和特征可用性 mask。executor-only scalar $\widehat\Gamma_{ij}^{\mathrm{hist}}(t)$ 不属于 $o_i(t)$，而是在 actor proposal 形成 $\mathcal S_i^{\mathrm{prop}}(t)$ 后按 2.8.5 计算并仅供 deterministic executor 使用。上一槽有效速率只表示已结束时隙的实际结果；未调度链路没有该观测时使用缺省值加 mask，NA 不视为失败；outage 率只使用 $\tau\le t-1$ 的 $\chi_{ij}^{\mathrm{att}}(\tau)=1$ 样本，不读取当前时隙尚未产生的真实 SINR 或执行结果。当前槽的 $I_{ij,r}^{\mathrm{meas}}(t)$ 同样不属于 $o_i(t)$；它只能通过槽末更新后的历史摘要在 $o_i(t+1)$ 或更晚时隙中间接出现。

actor 不得读取以下信息：

- 真实建筑物遮挡标签；
- 真实瞬时信道 $h_{ij,r}(t)$；
- 当前真实 $\mathrm{SINR}_{ij,r}(t)$ 和当前真实干扰；
- 当前其他 UAV 尚未确定的资源、功率及执行器接受结果；
- 未来位置或未来移动轨迹；
- 真实业务到达参数 $\lambda_i$；
- 当前时隙槽末才生成的到达量 $A_i(t)$；
- 尚未生成的未来任务；
- 其他 UAV 的完整私有队列与完整任务记录。

集中训练期的 critic 或等价全局评价模块可以使用真实资源参数、所有任务和队列真值、真实信道与遮挡状态、真实业务到达参数以及建筑物和全局统计，但执行期 actor 的权限不因此扩大。

联合动作是各 UAV 局部动作的笛卡尔积：

$$
\mathbf a_t
=
\left(
a_1(t),\ldots,a_N(t)
\right)
\in
\mathcal A
=
\prod_{i\in\mathcal U}\mathcal A_i.
$$

动作 mask 与联合硬可行执行器共同将原始联合动作映射为实际可执行动作，但不改变七分支的语义。

### 2.14.2 状态转移与硬可行策略集合

在外生任务到达、移动、信道演化和联合服务规则共同作用下，状态转移可写为：

$$
s_{t+1}
\sim
P
\left(
\cdot\mid s_t,\mathbf a_t
\right).
$$

其中，$\mathbf a_t$ 始终表示各 actor 给出的七分支联合动作提案，确定性执行器不创建新的 actor 动作空间。转移核在环境内部依次完成：

$$
\mathbf a_t
\longrightarrow
\text{确定性执行器}
\longrightarrow
\text{已执行通信、CPU 与资源决策}
\longrightarrow
\text{SINR、服务、能量、队列与生命周期更新}
\longrightarrow
s_{t+1}.
$$

因此，转移核包含真实信道和遮挡对服务量的影响、传输与 CPU 队列更新、能量扣除、任务完成/过期结算、槽末入队、外生移动和 CSI AoI 更新，但不把 actor 的局部估计直接当作真实环境状态，也不把执行器解析结果重新定义为 Dec-POMDP 动作。

Dec-POMDP 可用元组表示为：

$$
\mathcal M
=
\left(
\mathcal S,
\{\mathcal O_i\}_{i\in\mathcal U},
\{\mathcal A_i\}_{i\in\mathcal U},
P,
r,
\gamma
\right).
$$

联合执行必须满足 2.5--2.13 已定义的任务队列合法性、候选邻居合法性、目的地锁定、route/tx 语义、固定资源组与相邻宽度规则、离散发射功率和 CPU 频率档位、候选及实际能量可行性、半双工收发模式与节点独占无线调度、每个 CPU 每槽至多服务一个队列，以及任务生命周期和 hard-deadline 结算规则。相应的实际可行集合概括为：

$$
\mathcal A_t^{\mathrm{feas}}
=
\left\{
\begin{array}{l}
\text{由确定性执行器解析得到的通信、计算与资源决策}:\\
\text{2.5--2.13 中的队列、邻居、目的地锁定与 route/tx 规则成立},\\
\text{固定资源组、相邻宽度及离散功率/CPU 频率规则成立},\\
\overline E_i^{\mathrm{act,cand}}(t;\ell^\star)\le E_i^{\mathrm{res}}(t),\ \forall i,\\
\text{半双工收发模式与节点独占无线调度约束成立},\\
\text{每个 CPU 每槽至多服务一个合法队列，任务生命周期与截止期结算合法}
\end{array}
\right\}.
$$

动作 mask 处理可由局部信息提前识别的非法分支；确定性执行器处理必须在联合动作提案形成后才能判断的资源冲突、半双工与节点独占无线冲突和联合能量可行性。原始 actor 提案不必独立满足全部联合约束。由 actor 策略、动作 mask、确定性执行器和既有物理约束共同诱导的闭环硬可行策略集合定义为：

$$
\Pi_{\mathrm{hard}}
=
\left\{
\boldsymbol{\pi}:
\Pr_{\boldsymbol{\pi},P}
\left[
\begin{array}{c}
\text{闭环执行在每个服务时隙 }t\in\{0,\ldots,T-1\}\text{ 均属于}\\
\mathcal A_t^{\mathrm{feas}}
\end{array}
\right]
=1
\right\}.
$$

该定义中的“属于”针对执行器解析后的已执行通信、CPU 与资源变量，而不是要求原始联合提案 $\mathbf a_t$ 自身满足所有联合约束。能量硬约束在执行前使用当前 $p_i^{\mathrm{cand}}(t;\ell)$ 对应的 $\overline E_i^{\mathrm{act,cand}}(t;\ell)$；接受 $\ell^\star$ 后，实际能耗满足 $E_i^{\mathrm{act}}(t)\le\overline E_i^{\mathrm{act,cand}}(t;\ell^\star)$，未使用的预留能量释放。

### 2.14.3 Episode 级物理性能指标

令 $\mathcal C_T$ 表示 episode 内记录为 done 的按时完成任务集合。保留 $N_{\mathrm{on}}(t)$ 的既有定义，即经过时隙 $t$ 的 service 后在本槽完成且满足 hard deadline 的任务数，则 episode 内按时完成任务数为：

$$
C_T
=
|\mathcal C_T|
=
\sum_{t=0}^{T-1}N_{\mathrm{on}}(t).
$$

令 $G_T$、$X_T$ 和 $Z_T$ 分别表示 episode 内生成、expired 和 truncated 的任务数。由于固定时域在边界 $T$ 对未完成且 $t_n^{\mathrm{ddl}}\ge T$ 的任务仅作 truncated 统计标记，三类 episode 结算计数满足：

$$
G_T
=
C_T+X_T+Z_T.
$$

完成、过期和截断比例统一定义为：

$$
\left(
\eta_{\mathrm{cmp}},
\eta_{\mathrm{exp}},
\eta_{\mathrm{trunc}}
\right)
=
\begin{cases}
\left(
\dfrac{C_T}{G_T},
\dfrac{X_T}{G_T},
\dfrac{Z_T}{G_T}
\right),
& G_T>0,\\[4pt]
\left(
\mathrm{NA},
\mathrm{NA},
\mathrm{NA}
\right),
& G_T=0.
\end{cases}
$$

当 $G_T>0$ 时，$\eta_{\mathrm{cmp}}+\eta_{\mathrm{exp}}+\eta_{\mathrm{trunc}}=1$。truncated 任务是由有限仿真时域引起的删失样本，不是物理 hard-deadline 失败；因此它不进入 $X_T$，不虚构 $t_n^{\mathrm{cmp}}$ 或 $T_n^{\mathrm{E2E}}$，也不能解释为 expired。

期望按时成功任务吞吐量定义为：

$$
\Lambda_{\mathrm{succ}}(\boldsymbol{\pi})
=
\frac{1}{T}
\mathbb E_{\boldsymbol{\pi},P}
\left[C_T\right]
=
\frac{1}{T}
\mathbb E_{\boldsymbol{\pi},P}
\left[
\sum_{t=0}^{T-1}
N_{\mathrm{on}}(t)
\right].
$$

$\Lambda_{\mathrm{succ}}(\boldsymbol{\pi})$ 的单位为 task/slot。由于比较方法采用相同的固定 $T$，最大化该指标等价于最大化 $\mathbb E_{\boldsymbol{\pi},P}[C_T]$。

任务端到端时延继续采用 2.5 的定义 $T_n^{\mathrm{E2E}}=(t_n^{\mathrm{cmp}}-t_n^{\mathrm{arr}})\Delta t$，仅 done 任务具有合法样本。作为次要物理 KPI，平均端到端时延定义为：

$$
\overline T_{\mathrm{E2E}}
=
\begin{cases}
\dfrac{1}{C_T}
\displaystyle\sum_{n\in\mathcal C_T}
T_n^{\mathrm{E2E}},
& C_T>0,\\[4pt]
\mathrm{NA},
& C_T=0.
\end{cases}
$$

$\overline T_{\mathrm{E2E}}$ 的期望方向为越小越好，但本节不为其另设独立的时延最小化问题。

episode 内实际主动能耗定义为：

$$
E_{\mathrm{tot}}^{\mathrm{act}}
=
\sum_{t=0}^{T-1}
\sum_{i\in\mathcal U}
\left[
E_i^{\mathrm{tx}}(t)
+
E_i^{\mathrm{cpu}}(t)
\right].
$$

$E_{\mathrm{tot}}^{\mathrm{act}}$ 是次要物理 KPI，期望方向为越小越好。它只包含实际 U2U 发射能耗和实际 CPU 动态计算能耗，不包含推进、悬停、飞控、姿态控制、感知或静态平台能耗；各 UAV 的剩余能量可行性仍是硬约束。本节不为该 KPI 另设独立的能量最小化问题。

### 2.14.4 系统级物理优化目标

在满足通信、计算、能量预算、半双工与节点独占无线调度、队列、资源调度及任务截止期等硬约束的条件下，系统首要目标是提高单位时间内按时成功完成的任务数量。因此，唯一的主物理优化问题定义为：

$$
\mathrm{P}_{\mathrm{phys}}:
\qquad
\max_{\boldsymbol{\pi}\in\Pi_{\mathrm{hard}}}
\Lambda_{\mathrm{succ}}(\boldsymbol{\pi}).
$$

完成比例 $\eta_{\mathrm{cmp}}$ 仍是核心评价 KPI，但不作为主物理目标。原因在于 $G_T$ 包含 episode 末因有限时域截断而未进入完整服务或尚未到达物理截止时隙的任务；直接最大化 $C_T/G_T$ 会在同一分母效应中混合仿真删失与物理 deadline 失败。固定 $T$ 下的 $\Lambda_{\mathrm{succ}}$ 则直接刻画单位时隙的期望按时完成任务数。

### 2.14.5 团队即时奖励与 RL 训练替代目标

为了刻画截止期和资源代价，定义活动任务的紧迫工作量。下式中的 $W(t)$ 在时隙 $t$ 的奖励计算阶段读取，即已完成本槽 service、实际能耗计算和 completion/deadline settlement 之后、槽末新到达任务 $A_i(t)$ 生成与入队之前的 post-service/pre-next-arrival 值：

$$
W(t)
=
\sum_{n\in\mathcal T_t^{\mathrm{active}}}
\frac{
D_n^{\mathrm{rem}}(t)/R_{\mathrm{ref}}
+
C_n^{\mathrm{rem}}(t)/f_{\mathrm{ref}}
}{
\max\{1,t_n^{\mathrm{ddl}}-t+1\}
}.
$$

其中，在 reward 读取边界上，$\mathcal T_t^{\mathrm{active}}$ 是完成本槽 settlement 后仍未处于 done 或 expired 的任务集合，$D_n^{\mathrm{rem}}(t)$ 和 $C_n^{\mathrm{rem}}(t)$ 是同一 post-service 边界的剩余量；槽末刚生成且仅在 $t+1$ 可见的 $A_i(t)$ 不包含在其中。本地绑定任务因满足 $D_n^{\mathrm{rem}}=0$，不计入通信剩余工作量。$N_{\mathrm{on}}(t)$ 统计经过 slot $t$ service 后在本槽完成且满足 hard deadline 的任务数，$N_{\mathrm{exp}}(t)$ 统计同一 settlement 中在 deadline slot service/update 后仍未完成而标记为 expired 的任务数；deadline slot 本身仍可用于合法 service。$E^{\mathrm{act}}(t)$ 为全体 UAV 在本槽最终执行产生的实际主动能耗，不使用 reservation、candidate 或 proposed energy，则团队即时奖励定义为：

$$
r_t
=
w_c\widetilde N_{\mathrm{on}}(t)
-
w_d\widetilde N_{\mathrm{exp}}(t)
-
w_q\widetilde W(t)
-
w_e\widetilde E^{\mathrm{act}}(t).
$$

波浪号的参考尺度定义为：

$$
\widetilde W(t)=\frac{W(t)}{W^{\mathrm{ref}}},
\qquad
\widetilde N_{\mathrm{on}}(t)=\frac{N_{\mathrm{on}}(t)}{N^{\mathrm{ref}}},
\qquad
\widetilde N_{\mathrm{exp}}(t)=\frac{N_{\mathrm{exp}}(t)}{N^{\mathrm{ref}}},
\qquad
\widetilde E^{\mathrm{act}}(t)=\frac{E^{\mathrm{act}}(t)}{E_{\mathrm{act}}^{\mathrm{ref}}}.
$$

其中 $W^{\mathrm{ref}}>0$ 是 workload reference，单位为 s；$N^{\mathrm{ref}}>0$ 是每槽任务数 reference，单位为 task；$E_{\mathrm{act}}^{\mathrm{ref}}=\sum_{i\in\mathcal U}E_i^0>0$ 是全体 UAV 的固定 episode 初始主动能量 reference，单位为 J。三者均在实验开始前确定，在一个 episode 内、各次 reset 之间以及 evaluation 中保持不变；它们不是当前 slot 或 episode 的 observed maximum、未来 maximum、batch statistics 或训练过程中的动态 min/max，且不随 seed 改变。后续配置只负责给出这些 fixed constants 的具体数值，不重新定义其物理含义。$w_c,w_d,w_q,w_e$ 的编码初始值为 $1.0,1.0,0.2,0.05$，仅作为调试起点，不作为已验证的最优权重。

该奖励与物理目标的对应关系如下：正的 $\widetilde N_{\mathrm{on}}(t)$ 支持提高按时成功任务吞吐量这一主物理目标；负的 $\widetilde N_{\mathrm{exp}}(t)$ 惩罚 hard-deadline 失败；负的 $\widetilde W(t)$ 在稀疏的完成或过期事件发生前提供剩余工作量与 deadline 紧迫性的稠密 shaping；负的 $\widetilde E^{\mathrm{act}}(t)$ 鼓励降低实际主动通信与计算能耗。$W(t)$ 仅是 RL shaping 量，不是物理时延目标、未完成工作量的独立优化目标或 E2E 时延的等价量。

给定共享环境和各 actor 的局部策略集合 $\boldsymbol{\pi}$，折扣训练回报定义为：

$$
J_{\mathrm{RL}}(\boldsymbol{\pi})
=
\mathbb E_{\boldsymbol{\pi},P}
\left[
\sum_{t=0}^{T-1}
\gamma^t r_t
\right].
$$

其中 $\gamma\in(0,1)$ 为折扣因子。相应的 RL 训练替代目标为：

$$
\max_{\boldsymbol{\pi}\in\Pi_{\mathrm{hard}}}
J_{\mathrm{RL}}(\boldsymbol{\pi}).
$$

该训练目标与 $\max_{\boldsymbol{\pi}\in\Pi_{\mathrm{hard}}}\Lambda_{\mathrm{succ}}(\boldsymbol{\pi})$ 不具有代数等价性：$\gamma<1$ 对较早出现的奖励赋予更大权重，使不同时刻的完成奖励和能耗惩罚具有不同折扣；$W(t)$ 提供稠密 shaping；$N_{\mathrm{exp}}(t)$ 提供额外的失败惩罚；固定归一化尺度和奖励权重进一步改变了各项在训练信号中的相对贡献。因此，两类目标的关系应理解为：

$$
\text{系统级物理目标}
\longrightarrow
\text{奖励 shaping 与替代构造}
\longrightarrow
\text{RL 优化}
\longrightarrow
\text{基于物理 KPI 的最终评价}.
$$

具体学习器、策略网络和价值更新属于后续方法章节，本章不写入其更新公式，也不据此声称 RL 解与主物理问题全局等价或具有全局最优性。

### 2.14.6 评价指标与声明边界

最终评价必须联合报告 $\Lambda_{\mathrm{succ}}$、$C_T$、$G_T$、$X_T$、$Z_T$、$\eta_{\mathrm{cmp}}$、$\eta_{\mathrm{exp}}$、$\eta_{\mathrm{trunc}}$、$\overline T_{\mathrm{E2E}}$ 和 $E_{\mathrm{tot}}^{\mathrm{act}}$。其中，$\Lambda_{\mathrm{succ}}$ 对应主物理目标；$\overline T_{\mathrm{E2E}}$ 和 $E_{\mathrm{tot}}^{\mathrm{act}}$ 是越小越好的次要物理 KPI；三类比例用于审计有限 horizon 下的任务去向。$G_T=0$ 时三类比例均为 $\mathrm{NA}$，$C_T=0$ 时平均 E2E 时延为 $\mathrm{NA}$，不得以零替代无样本结果。

truncated 仅是 episode 终止时的删失统计标签，不加入任务物理状态集合，不计入 expired numerator，也不产生虚构的完成时刻、E2E 时延或额外终止奖励。当前奖励不显式优化异构 UAV 间的公平性，本节不把公平性写入 $\mathrm{P}_{\mathrm{phys}}$，也不新增公平性奖励或优化目标。上述问题定义只规定硬可行闭环决策、物理目标和 RL 替代信号，不推出全局最优、收敛保证、部署性能或未验证的公平性结论。

## 2.15 假设、局限性与实现映射

### 2.15.1 主要建模假设

1. UAV 高度固定，水平移动由外生 Gauss-Markov 过程产生；完整离散位置序列在场景生成阶段通过有限高度建筑排除检查后固定，所有算法使用相同的有效外生轨迹。
2. 任务输入不可拆分到多个目的 UAV，不允许多跳转发；目的地在 route 决策生效并进入 local 或 tx 队列时锁定。
3. 传输和 CPU 服务跨时隙持续，槽末入队的任务下一时隙才可服务。
4. 计算模块与无线模块可以并行；无线收发机采用半双工模式，并附加节点独占的一链路 MAC 调度抽象。
5. 主通信资源由五个固定资源组组成，资源组宽度只取一个或两个相邻组，功率和 CPU 频率只取离散档位。
6. 建筑物以有限高度三维闭棱柱表示；UAV 的离散服务时隙位置位于建筑实体之外，链路遮挡通过闭线段与建筑物的几何相交判定。
7. 轻量信道模型仅服务于趋势评价；真实信道用于环境转移，actor 使用陈旧期望链路信道估计、历史干扰摘要、历史干扰链路质量代理量、相应 AoI 和可用性 mask。
8. episode reset 直接初始化 $\mathbf p_i(0)$、$\mathbf v_i(0)$、$X_{ij}(0)$ 和 $h_{ij,r}(0)$，并以 $\Delta s_{ij}(0)=0$ 作为初始位移约定；不生成或访问负时隙历史，CSI 不可用时使用缺省值与 mask。
9. 初始能量、参考速率和 deadline 在 Gate P pilot 中进行非退化性校准；路径损耗相关参数在 Gate P 中只进行仿真链路预算与数值可行性检查。在获得场景匹配的外部测量或射线追踪标定前，$L_{\mathrm B}=15\,\mathrm{dB}$ 仅为名义建模基准。
10. 主动能耗只包含按实际传输活跃时间和 CPU 动态活跃时间统计的无线发射与 CPU 动态能耗，不包含推进和其他飞行器功耗；预留能量只用于执行前的硬可行性检查，不作为实际能耗指标。
11. 结果回传暂不建模，远程 CPU 完成即视为任务计算完成。

### 2.15.2 局限性和声明边界

本模型适合研究动态异构、单跳 U2U-MEC、离散资源编排和截止期队列之间的耦合机制，但不应被解释为完整 UAV 通信系统的数字孪生。主模型不冻结单无线电或单 RF 链架构，也不据此推导节点独占约束。固定高度和外生移动排除了轨迹控制与飞行动力学影响；建筑排除条件只保证离散服务时隙位置合法，不证明相邻时隙间的连续飞行线段已避开建筑；轻量物理层不提供实测信道或实时射线追踪级别的真实性保证；未建模结果回传、完整协议栈、HARQ、MIMO、RIS、DAG 任务和多跳路由；第三 UAV 软遮挡及高保真传播回放只能作为后续扩展。性能结论必须限定在已实现的环境、参数校准范围、测试场景和统计协议内。

本章的公式定义服务接口、约束和评价量，不证明任务完成率的理论最优边界，不证明策略对未测试规模或未测试分布的泛化，也不把经验趋势写成定理。任何数值、曲线、显著性或算法优越性结论都必须留给真实程序运行后的结果章节。

历史干扰摘要是接收端经既有控制消息广播的抽象槽级控制信息，不展开完整控制信令协议、测量误差硬件或信令能耗；$\widehat\Gamma_{ij,r}^{\mathrm{hist}}(t)$ 是历史干扰链路质量代理量，不代表当前真实 SINR。计划测试应覆盖当前干扰信息泄漏、历史量只使用 $t-1$ 或更早数据、无历史时的缺省值与 mask、AoI 更新以及 actor 观测不含当前真实 SINR 等信息边界，但这些测试目前尚未实现，也不构成已完成的测试结果。

### 2.15.3 系统规则到代码和测试的映射

下表只描述后续实现与测试应覆盖的接口，不声称这些模块当前已经实现。由于项目初始化阶段不编写系统实现代码，本表中的文件名是规划映射。本节的时间边界不变量是：槽末到达的任务下一槽可 route，槽末绑定或转队的任务下一槽可服务，deadline 时隙服务结束后再结算。计划测试应覆盖这些 off-by-one 与槽边界用例；以下仅为测试要求，不代表测试已经实现。

outage 口径的计划测试至少包括：

- proposed send 被执行器拒绝时，outage 为 NA；
- actor 主动选择通信 idle 时，outage 为 NA；
- 零功率或零实际资源占用时，outage 为 NA；
- 实际 $p_i^{\mathrm{exec}}(t)>0$、非零资源且队列非空，但有效速率为零时，outage 为 1；
- 成功实际传输时，沿用既有判定得到 outage 为 0；
- 累计指标的分母只包含实际传输尝试；
- 没有实际传输尝试时，累计 outage 为 NA；
- 历史 outage 特征只使用 $t-1$ 或更早的实际传输尝试样本。
- SINR 门限量纲测试：确认 $\gamma_{\min}^{\mathrm{dB}}=-3\ \mathrm{dB}$ 转换为 $\gamma_{\min}^{\mathrm{lin}}\approx0.5012$，并覆盖线性 SINR 等于、略低于和略高于该值时的有效速率门控与 outage 判定边界；

动作接口与功率阶段的计划测试至少包括：

- 验证 $g_i(t)=a_i^{\mathrm{rg}}(t)$、$w_i(t)=a_i^{\mathrm{width}}(t)$ 只建立派生映射，不增加第八或第九个动作分支；
- 验证 width 的定义域仍为 $\{1,2\}$，inactive 时使用 canonical 值 $w_i(t)=1$，且该值不改变 $\mathcal S_i^{\mathrm{prop}}(t)$、环境状态、资源申请、能量预留或实际传输；
- 验证 $\mathcal S_i^{\mathrm{prop}}(t)$ 只由 actor 的 resource group/width 分支构造，执行器拒绝后 $\mathcal S_i^{\mathrm{exec}}(t)=\varnothing$；
- 验证 $p_i^{\mathrm{prop}}(t)$ 只由 $a_i^{\mathrm{pow}}(t)$ 映射得到，且 actor 提案、candidate 临时值和 executed 最终值彼此分离；
- 执行器拒绝或退化为通信 idle 时验证 $p_i^{\mathrm{exec}}(t)=0$；
- 执行器降档时验证 $p_i^{\mathrm{exec}}(t)\ne p_i^{\mathrm{prop}}(t)$ 可以合法发生，且 candidate 只使用既有绝对离散功率等级；
- 验证通信能量预留使用 $p_i^{\mathrm{cand}}(t;\ell)$，而不是尚未确定的 $p_i^{\mathrm{exec}}(t)$；
- 验证 $p_{ij,r}(t)$ 是由最终 $p_i^{\mathrm{exec}}(t)$ 和 $\mathcal S_i^{\mathrm{exec}}(t)$ 得到的 executed per-RU power；
- 验证真实 SINR、实际服务、实际通信能耗和 outage 的非零功率条件均使用 executed power，且不存在未限定的 $p_i(t)$ 跨阶段复用。
- 固定槽初状态、联合动作提案和环境状态重复执行两次时，最终 $y_{ij}(t)$、$x_{ij,r}(t)$、$p_i^{\mathrm{exec}}(t)$、CPU 频率和拒绝结果必须完全一致；
- 具有不同 slack 的候选边按较小 slack 优先，slack 相同则按较大历史链路质量代理量优先，再按发送和接收 UAV ID 升序；
- slack、历史链路质量代理量和发送 UAV ID 均相同的候选比较时，接收 UAV ID 升序必须给出唯一顺序，且不得调用随机 tie-break；
- 节点独占无线扫描按固定顺序接受不冲突候选；共享任一端点的候选必须被拒绝，端点不相交的候选允许并发，并验证拒绝边的 $y_{ij}(t)=0$、空执行资源集、零执行功率和 NA outage；若已接受的端点不相交边执行 RU 重叠，则验证其进入相应 SINR 干扰项；
- 对同一联合动作，候选输入改用 set/dict 或不同未声明列表顺序时，排序后的 executor 输出必须不变；
- 当前 actor CPU 频率下存在可行 candidate power 时，必须选择最高可行通信档位且 CPU 频率保持不变；
- 只有通信 candidate power 已降至 $0$ 且联合能量仍不可行时，才允许按既有 CPU 频率档位降档；每次都选择最高可行 CPU 频率；
- executor 排序和降档不得读取当前真实 SINR、当前 outage 或未来执行结果；上述内容只能在最终执行动作确定后由环境计算。

历史到达率的计划测试至少包括：

- $t=0$ 时验证 $K_\lambda(0)=0$、$\hat{\lambda}_i(0)=0$ 和 $m_i^\lambda(0)=0$；
- $t=1$ 时只使用 $A_i(0)$，不访问负时隙索引；
- $1<t<W_\lambda$ 时只使用全部已存在的历史样本，不包含 $A_i(t)$；
- $t\ge W_\lambda$ 时只使用最近 $W_\lambda$ 个已结束时隙的到达量；
- 修改尚未在槽末生成的 $A_i(t)$ 不得改变时隙 $t$ 的 actor 观测；
- $A_i(t)$ 最早只能影响时隙 $t+1$ 的历史到达率或队列观测；
- 缺省值 $0$ 必须始终与 $m_i^\lambda(t)=0$ 的不可用标志配对，不能被解释为真实零到达率。

本轮 reward 与 episode 终止语义的计划测试仅用于后续实现，不表示测试已经实现：

1. reward 使用 post-service/post-settlement 状态；
2. deadline slot 在 service/update 后再决定 done 或 expired；
3. $E^{\mathrm{act}}(t)$ 使用 actual energy 而非 reservation energy；
4. $A_i(t)$ 不影响同槽 reward；
5. $A_i(t)$ 最早影响 $t+1$ 的 state/observation；
6. 所有 normalization denominator 在 episode 开始前固定；
7. normalization 不依赖运行中的 max/min 或 batch statistics；
8. 相同物理量输入得到相同 normalized value；
9. slot $T-1$ 是最后 service slot；
10. 不存在 slot $T$ 的 actor 或 service；
11. $t_n^{\mathrm{ddl}}=T-1$ 的任务可以在最后槽完成；
12. $t_n^{\mathrm{ddl}}=T-1$ 且最后槽未完成的任务记为 expired；
13. $t_n^{\mathrm{ddl}}\ge T$ 且边界仍未完成的任务记为 truncated 而非 expired；
14. $A_i(T-1)$ 不获得 route/service，并在 terminal bookkeeping 中按冻结规则处理；
15. truncated task 不虚构 $t_n^{\mathrm{cmp}}$ 或 $T_n^{\mathrm{E2E}}$；
16. truncated 不计入 expired numerator；
17. truncated 单独统计；
18. terminal 不新增虚构 reward；
19. outage denominator 只使用实际 transmission attempts；
20. 固定 seed 下 terminal accounting 可复现。


本轮 P1-01/P1-07/P1-08 专项计划测试仅记录测试要求，不表示测试代码已经实现：

1. deadline slot 中 unbound 任务执行 route 后可在槽末完成 binding，但 unresolved task 在同槽 settlement 后记为 expired，且不进入下一槽 service。
2. deadline slot 中 transmission 恰好完成全部 bits、仍有 $C_n^{\mathrm{rem}}>0$ 时，允许 tx $\rightarrow$ CPU bookkeeping，但同槽不 CPU service，随后记为 expired。
3. deadline slot 中 CPU 恰好完成全部 remaining cycles 时，completion 判断优先，任务记为 done 而不是 expired。
4. reset 后验证 $E_i^{\mathrm{res}}(0)=E_i^0>0$。
5. reset 后验证 $Q_i^{\mathrm{unb}}(0)$、$Q_i^{\mathrm{loc}}(0)$、$Q_{i\rightarrow j}^{\mathrm{tx}}(0)$、$Q_{i\rightarrow j}^{\mathrm{cpu}}(0)$ 和 active task collection 全部为空。
6. reset 后验证 task-ID counter 从 $0$ 开始，并在固定 seed 下按 UAV ID 升序及 generator 稳定顺序分配可复现 ID。
7. reset 后验证 completed/expired/truncated count、actual attempt count、outage numerator/denominator、energy accumulator 和 latency accumulator/sample list 的初值；无样本 ratio/average 为 $\mathrm{NA}$。
8. 验证 $\mathrm{actor}(0)$ 前所有 state/history/mask 均为合法数值或明确 $\mathrm{NA}$ 加 mask，且不读取 negative index。
9. 验证 raw $p_i^{\mathrm{prop}}(t)=0$ 的通信 proposal 不占用半双工或节点独占无线端点、执行资源或能量预留，也不产生 interference、$I^{\mathrm{meas}}$ 或 actual attempt。
10. 验证 candidate downgrade 到 $p_i^{\mathrm{exec}}(t)=0$ 后最终满足 $y_{ij}(t)=0$、$x_{ij,r}(t)=0$ 和 $\mathcal S_i^{\mathrm{exec}}(t)=\varnothing$。
11. 验证 candidate downgrade 到 $0$ 后不 backtrack/re-arbitrate，不复活前序已拒绝 proposal；既有 executor priority key 和最高可行档位顺序保持不变。
12. 固定槽初状态、联合动作提案和环境输入重复执行时，executor 与 task settlement 的最终输出完全确定，且 zero-power 不产生 outage 样本。


本轮 P1-02 至 P1-06 专项计划测试仅记录后续实现要求，不表示测试代码已经实现：

- 越界只触发 coordinate-wise specular reflection，并验证对应速度分量反号；reset rejection sampling 满足 $d_{\min}^{\mathrm{safe}}$，主场景不启用 hysteresis；
- normalized Rayleigh/Rician 的 $z$ 实部和虚部方差均为 $1/2$，$\mathbb E[|g|^2]=1$，跨 slot/link/RU 独立，且 fixed seed 按 $(i,j,r)$ 生成完整 tensor；
- CSI error 为 dB 域 $\mathcal N(0,\sigma_{\mathrm{CSI}}^2)$，$\xi=10^{\epsilon/10}$，每槽/link/RU 独立重采样；不可用分支使用 placeholder+mask 且不改变 RNG 顺序；
- measurement mask 只由 executed desired reception、正 executed power 和槽初正 bit 共同决定；有效零测量进入 EMA，missing 不进入 EMA，当前测量下一槽才可见；
- executor-only scalar historical quality 只对 $\mathcal S_i^{\mathrm{prop}}(t)$ 中的 valid RU 求 arithmetic mean，无 valid RU 时固定为 $0$，不使用 $\mathcal S_i^{\mathrm{exec}}(t)$ 或 current true SINR；
- 固定槽初状态、联合动作和 master seed 重复执行时，CSI、history、scalar quality 与 executor priority order 完全一致。

物理层初始时刻的计划测试至少包括：

- $t=0$ 时 reset 直接提供 $\mathbf p_i(0)$ 和 $\mathbf v_i(0)$，不得读取 $\mathbf p_i(-1)$；
- $\Delta s_{ij}(0)=0$，且 $t\ge1$ 时才使用相邻时隙位置计算位移；
- $X_{ij}(0)$ 直接从 $\mathcal N(0,\sigma_s^2)$ 采样，不读取 $X_{ij}(-1)$；
- 阴影递推仅从 $t=1$ 开始；
- 对任意 $a_{ij}^{\mathrm{CSI}}(t)>t$，不得访问负历史索引；
- CSI 不可用时，缺省值与 $m_{ij}^{\mathrm{CSI}}(t)=0$ 配套；
- 当 $\ell_{ij}^{\mathrm{CSI}}(t)=0$ 时，首次可读取已生成的 $h_{ij,r}(0)$；
- 缺省 CSI 不得被解释为真实零信道，且不得通过当前真实信道回填历史；
- actor 观测不包含当前时隙的真实 $h_{ij,r}(t)$。

P1-04 计划测试还应覆盖以下物理层可执行性边界（仅新增测试计划，不表示已经实现）：

- 等宽资源单元满足 $B_{\mathrm{RU}}=B^{\mathrm{tot}}/R$，并验证主场景由 $20$ MHz 与 $R=20$ 得到 $1$ MHz；
- 噪声系数从 $F_{\mathrm{dB}}$ 到 $F=10^{F_{\mathrm{dB}}/10}$ 的转换正确，且 $N_0$ 先以线性 W/Hz 进入 $P_{\mathrm{noise,RU}}=N_0B_{\mathrm{RU}}F$；
- SINR 分母中的热噪声只加入一次，历史 interference-only 不重复携带热噪声；
- $I_{ij,r}^{\mathrm{meas}}(t)$ 与真实 SINR 的 interference term 完全一致且不含 thermal noise；
- proposed、candidate、被拒绝传输或无执行资源的链路均不产生 $I^{\mathrm{meas}}$ 干扰，测量只使用 executed action；
- actor 在槽 $t$ 不能读取 $I^{\mathrm{meas}}(t)$，当前测量最早在槽 $t+1$ 影响历史观测；
- 当前测量缺失时历史估计保持不变，且缺测不会被编码为真实零干扰；
- EMA 只在槽末由 $I^{\mathrm{meas}}(t)$ 更新到下一槽，消息 AoI 在有效测量时刷新为 $1$、缺测时递增 $1$，初始无历史保持 $\mathrm{NA}$；
- 固定 CSI 陈旧偏移量只按既有 $\ell_{ij}^{\mathrm{CSI}}(t)=t-a_{ij}^{\mathrm{CSI}}(t)$ 读取，不新增递推 freshness process；
- $10\log_{10}\xi$ 在 dB 域描述功率增益误差，$\sqrt{\xi}$ 作为线性乘性因子作用于复信道 $h$，并验证不被再次当作线性功率误差使用；
- reset 后历史干扰估计、mask、消息 AoI、CSI mask 和历史质量代理均有合法初值，slot 0 不访问负索引；
- 若测量或 AoI 过程包含随机性，固定 seed 下应可复现。

| 系统规则 | 形式化对象或不变量 | 计划实现位置 | 计划测试或 Gate |
|---|---|---|---|
| 任务记录与生命周期 | $\tau_n$、$\mathcal S_{\mathrm{task}}$、$\mathcal T_{\mathrm{task}}$；unbound→local/tx 时一次性绑定目的地，且 local、tx、cpu 状态下目的地不可变 | src/env/tasks.py | 目的地标记与任务生命周期测试，Gate 0 |
| bit 守恒 | local 绑定后 $D_n^{\mathrm{rem}}=0$；远程任务按实际传输服务递减且不为负 | src/env/queues.py | local 绑定清零与远程 bit 守恒测试，Gate 0 |
| cycle 守恒 | $C_n^{\mathrm{rem}}$ 按实际 CPU 服务递减且不为负 | src/env/queues.py | cycle 守恒测试，Gate 0 |
| 状态机合法性 | 只允许 $\mathcal T_{\mathrm{task}}$ 中的转移 | src/env/tasks.py | 零非法状态转移测试，Gate 0 |
| route/tx 分离 | route 仅操作 $Q_i^{\mathrm{unb}}$ 并一次性绑定；tx_select 仅操作 $Q_{i\rightarrow j}^{\mathrm{tx}}$ 且不改变目的地 | src/env/action_space.py，src/env/queues.py | 同槽因果测试，Gate 0 |
| 槽边界 | 新路由、新到达和传输完成任务下一槽才可服务 | src/env/u2u_mec_env.py | 槽序列测试，Gate 0 |
| 固定高度 | $\mathbf q_i(t)=[x_i(t),y_i(t),H]^{\mathrm T}$ | src/env/mobility.py | 高度不变测试，Gate 0 |
| 动态候选邻居 | $\mathcal N_i(t)$ 与三维距离阈值 | src/env/topology.py | 邻居边界测试，Gate 1 |
| 建筑物遮挡 | 存在量词几何相交判定 | src/env/buildings.py | 遮挡几何测试，Gate 1 |
| 固定资源组 | $\mathcal G_1,\ldots,\mathcal G_5$ 及宽度 mask | src/env/resource_groups.py | 资源组边界测试，Gate 0 |
| SINR 与有效速率 | $\mathrm{SINR}_{ij,r}$、$R_{ij}^{\mathrm{eff}}$ | src/env/channel.py | 干扰和速率单元测试，Gate 1 |
| RU、噪声与历史干扰 | $B_{\mathrm{RU}}$、$P_{\mathrm{noise,RU}}$、$I_{ij,r}^{\mathrm{meas}}$、$\widehat I_{j,r}^{\mathrm{hist}}$、AoI | src/env/channel.py，src/agents/observation.py | P1-04 测量、单位链、时序、mask 和 freshness 计划测试，Gate 1 |
| outage 统计 | 仅对 $y_{ij}=1$、$p_i^{\mathrm{exec}}(t)>0$、实际资源占用非空且队列剩余 bit 大于零的实际传输尝试统计；其余为 NA | src/env/channel.py，src/evaluation/ | outage 口径测试，Gate 1 |
| 能量守恒 | $\overline E_i^{\mathrm{act,cand}}(t;\ell^\star)\le E_i^{\mathrm{res}}(t)$；$E_i^{\mathrm{res}}(t+1)=E_i^{\mathrm{res}}(t)-E_i^{\mathrm{act}}(t)$ | src/env/energy.py | candidate 预留硬约束、executed 实际能耗守恒与非负测试，Gate 0 |
| 实际服务与能耗 | $\tau_{ij}^{\mathrm{tx}}$、$\tau_i^{\mathrm{cpu}}$；提前完成后按实际活跃时间扣除，executed power 的 outage 尝试按整槽计 | src/env/energy.py，src/env/queues.py | 提前完成能耗、executed-power outage 整槽发射尝试、实际能耗不超过 candidate 预留、剩余能量按实际能耗更新，均为计划测试，Gate 0 |
| 离散能量降档 | $1.0\rightarrow0.5\rightarrow0.25\rightarrow0$ | src/env/energy.py，src/agents/action_mask.py | 档位闭集测试，Gate 0 |
| 联合半双工（半双工 + 节点独占无线调度） | 同一 UAV 不得同时发送和接收，且每个时隙至多关联一条最终接受无线边；端点不相交的边可并发 | src/env/half_duplex_resolver.py | 共享端点冲突、端点不相交并发与 RU 重叠干扰测试，Gate 0 |
| 信息权限 | actor 只使用 $o_i(t)$，禁止读取真实遮挡、当前真实信道/SINR/干扰、当前联合动作结果、未来位置、真实 $\lambda_i$ 和完整私有队列 | src/agents/observation.py | 信息泄漏、历史量时序、缺省值与 mask、AoI 更新测试，Gate 0 |
| Gate P 数值可行性检查 | 仿真链路预算、deadline 与能量处于非退化区间；不构成外部传播标定 | configs/ 与 experiments/（待建） | Gate P pilot |
| Gate 0 环境正确 | 任务、bit、cycle、energy 守恒；route/tx 分离；槽因果正确 | tests/（待建） | Gate 0 |

### 2.15.4 本章与后续章节的关系

本章是系统对象、符号、状态机、队列、物理层评价、动作接口和信息权限的唯一主源。后续方法章节只能在这些接口上说明策略网络、训练流程、集中训练和分散执行方式，不得重新定义任务状态、route/tx_select 语义、固定资源组、半双工与节点独占无线调度、离散降档或 actor 信息边界。实验协议章节只能使用本章的参数状态、Gate P/Gate 0 和评价口径组织校准、训练、测试、统计与复现，不得通过实验设置改变系统规则。

在编码实现前，必须先以不调用学习器的完整 episode 验证任务、bit、cycle 和能量守恒、零非法状态转移、槽边界、outage 统计、离散降档和联合半双工规则（半双工 + 节点独占无线调度）；只有 Gate 0 通过后，才可将本章定义的环境接口连接到后续策略和价值模块。
