# 第5步：统一候选的显式物理关系

> 阅读定位：本文保留当前方法所继承的定义及原阶段验证记录。旧进度、预算和命令属于历史状态；当前结果与C续训入口见[复现说明](../../docs/reproduction/REPRODUCE_ABC.md)。

日期：2026-10-12。本文记录C组工程实现，不构成学习效果结论。

## 版本与边界

B保留 `route_decoder_mode=unified_node_v1`、默认 `route_feature_mode=original_option_features_v1`。C仅增加 `route_feature_mode=physical_relations_v1`；该组合只允许用于统一模式。新字段的默认值从规范配置身份中移除，原P1a、E13以及第4步B配置hash不变。A/B/C独立实验配置留第6步发布。

实现位于 `src/models/route_physical_relations.py`，只接受原actor的self、public、edge张量及候选ID；不接受集中critic状态、远端实时私有队列、环境对象、oracle标签或未来信道。参考已有E15的单位还原和历史速率/CSI代理构造，但不调用其oracle、监督loss或预训练流程。

原环境、reward、task-event信用、TX paired-shadow、CPU gate、PPO、Q目标、学习率和预算未改变。C不新增合法性判断，不强制Local占比，不删除Defer。

## 单位还原

令配置给定的单任务数据上限为 $b_{\max}$，单任务计算量上限为 $c_{\max}$，UAV数为 $A$，episode长度为 $H$，归一化队列规模为 $M=\max(1,AH)$。原tensorizer分别使用：

$$
b=\widetilde b\,b_{\max},\qquad
c=\widetilde c\,c_{\max},\qquad
Q_j=\widetilde Q_j\,M c_{\max},\qquad
B_{ij}=\widetilde B_{ij}\,M b_{\max}.
$$
这里 $b$ 为当前待分配任务bits，$c$ 为cycles，$Q_j$ 为执行节点CPU队列剩余cycles，$B_{ij}$ 为源节点 $i$ 向远端 $j$ 的TX队列剩余bits。任务头与队列总量使用不同尺度，不能直接相加归一化数值。

配置参考频率为 $f_{\mathrm{ref}}$，参考能量为 $E_{\mathrm{ref}}$，参考速率为 $R_{\mathrm{ref}}$，slot时长为 $\Delta t$。从可见资源列还原频率 $f_j$、剩余能量 $E_j$、CPU系数 $\kappa_j$ 和源端最大发射功率 $P_i$；任务期限余量还原为秒 $d$。这些参数均来自原配置尺度和actor可见值。

Local CPU负担取本地队列与所有收到的CPU源队列之和。`cpu_by_source[ego]` 是Local的同一份队列，先用canonical Local替换该槽，再按源汇总一次。UNBOUND和TX队列不纳入CPU负担。Remote只取对应已发布公共CPU总负担，保留消息有效性和AoI。

## 时间与期限代理

对具有有效CPU能力的候选：

$$
\widehat t^{\mathrm{task,cpu}}_j=\frac{c}{f_j},\qquad
\widehat t^{\mathrm{queue,cpu}}_j=\frac{Q_j}{f_j}.
$$
Local的U2U传输需求为false，任务/队列U2U时间与TX能耗均为0。Remote优先使用有效历史effective rate；历史速率无效时，使用原actor的stale CSI及历史干扰估计全RU等功率速率：

$$
\widehat R_{ij}=\sum_{r=1}^{R}W_r
\log_2\left(1+\frac{(P_i/R)|\widehat h_{ijr}|^2}{N_r+\widehat I_{ijr}}\right).
$$
$R$ 为RU数，$W_r$ 为单RU带宽，$\widehat h_{ijr}$ 为过期CSI，$N_r$ 为配置噪声功率，$\widehat I_{ijr}$ 为历史干扰。干扰缺失时仅以噪声构造代理，另保留“全部RU干扰有效”标记。有效历史观测即使速率为0，也不被另一个CSI估计覆盖。

当Remote速率观测存在且为正时：

$$
\widehat t^{\mathrm{task,tx}}_{ij}=\frac{b}{\widehat R_{ij}},\qquad
\widehat t^{\mathrm{queue,tx}}_{ij}=\frac{B_{ij}}{\widehat R_{ij}},
$$
$$
\widehat t^{\mathrm{service}}_{ij}=
\widehat t^{\mathrm{task,cpu}}_j+
\widehat t^{\mathrm{queue,cpu}}_j+
\widehat t^{\mathrm{task,tx}}_{ij}+
\widehat t^{\mathrm{queue,tx}}_{ij},\qquad
\widehat m_{ij}=d-\widehat t^{\mathrm{service}}_{ij}.
$$
Local同样使用该总和，其中TX项为0。缺失速率、观测到零速率、无有效远端CPU或无当前任务时，相关不可用代理存0占位，同时 `service_valid=false`。网络同时接收占位值与validity；0占位不表示可以瞬时完成。分母数值下限为 $10^{-12}$ Hz或bps；零能力仍标无效。

这些值不是未来完成时间保证。它们没有模拟仲裁、半双工、其他源未来到达、未来CPU频率、移动/干扰变化、传输与CPU重叠或slot离散边界；历史速率也不保证未来可获得相同带宽。消息AoI、CSI AoI和有效性随特征一起传递，不以“过期”自动删除合法动作。

## 能量代理

当前任务的CPU能耗代理及源端清空对应TX backlog再传本任务的能耗代理为：

$$
\widehat E^{\mathrm{task,cpu}}_j=\kappa_j f_j^2 c,\qquad
\widehat E^{\mathrm{tx}}_{ij}=P_i
\left(\widehat t^{\mathrm{task,tx}}_{ij}+\widehat t^{\mathrm{queue,tx}}_{ij}\right).
$$
提供 $E_j-\widehat E^{\mathrm{task,cpu}}_j$ 与 $E_i-\widehat E^{\mathrm{tx}}_{ij}$ 余量，保留源端和执行端预算。这两个余量分别描述当前任务CPU代价及源TX队列代价；CPU余量未扣未来/排队任务全部耗能，不能当作完整能量可行性判定。Local源TX余量就是源剩余预算，Local CPU代价由执行端CPU余量表达。真实能量扣账继续由原环境完成。

## 35维输入与生产接线

| 连续关系（23维） | 原始单位/含义 | 固定变换尺度 |
| --- | --- | --- |
| task_bits、task_cycles | bits、cycles | 单任务配置上限 |
| slack_s、task_cpu_s、queue_cpu_s、task_tx_s、queue_tx_s、service_s、slack_margin_s | 秒 | slot时长 |
| frequency_hz、rate_bps | Hz、bps | 原参考频率/速率 |
| queued_cycles、tx_backlog_bits | cycles、bits | 单任务配置上限 |
| cpu_task_count | 实际任务数 | 1 |
| destination_energy_j、cpu_energy_j、source_energy_j、tx_energy_j、destination_energy_margin_j、source_tx_energy_margin_j | Joule | 原参考初始能量 |
| message_aoi_slots、csi_aoi_slots、outage_rate | slot数、历史outage比例 | 1 |

其余12维为task/destination/CPU有效性、needs-U2U、速率是否观测/为正、service有效性、消息AoI有效性、CSI有效性、历史速率有效性、全RU干扰有效性和outage有效性。连续关系使用固定有符号 $\operatorname{sign}(x)\log(1+|x|)$，标记保持0/1。不使用训练或evaluation统计拟合尺度。

C在B的204维候选融合向量进入原LayerNorm之前，加上共享 `35→32→204` 物理投影。Local与Remote使用相同投影及最终评分器。所有原融合参数、控制头、私有Route backbone和其他分支保持共同初始化；额外4个参数张量、7,884个参数使用独立随机域17002。B Route原有随机域17001保持不变，策略采样仍用stream110。

原411维Q表示仍为候选融合、合法Remote平均融合与3维类型标记；C的融合现在包含物理关系。Q输入仍detach，chosen-only目标与有效actor CF系数0不变，特征不会通过Q辅助伪装成oracle监督。实际叶PPO、合法softmax、Defer与log-barrier保持原契约。

日志在原v2叶决策记录中追加 `feature_mode`，旧记录缺失时读为NA；C checkpoint使用 `unified_route_identity.v2`，记录物理特征、变换与随机域身份。B仍保持v1身份。B/C互相恢复被拒绝，不以相同411维误判方法相同。

## 已执行验证与下一步

32项新增专项、230项相关回归及67项子测试通过。覆盖Local/Remote精确单位与能量值、Local零U2U、CPU不漏不重、缺失/零速率、stale CSI回退、消息AoI、零/极小CPU能力、零能量、2—5 UAV、候选重排、原mask、所有合法选择的生产PPO梯度、CUDA、四轮真实updater工程夹具、非空actor Adam及160条pending rollout恢复。没有要求随机网络对队列变化呈现人为规定的单调偏好。

seed42/43/44下B的配置、actor/critic参数及固定输入输出与改动前一致；B/C每组156个共同actor张量及全部critic张量一致。P1a 8k及E13两个32k历史模型严格加载和固定输入输出指纹与第4步一致。上述历史核验没有运行整episode评估，没有加载旧权重用于C初始化。

恢复专项初版把完成态checkpoint用于resume，被原保护正确拒绝；已改为PERIODIC_RESUME和160条pending夹具后通过，原保护未放宽。当前只验证工程接线，不评价训练信号收敛；没有正式训练、预训练或新完成率结果。第6步仍需完整仓库回归、短真实trainer/评估/恢复集成和独立A/B/C配置发布。

工程报告：`logs/development/task_allocation/step05_physical_relations/step05_report.md`。
