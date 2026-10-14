# E14d P0：Q 专用输入尺度与候选辨识修复

> 阅读定位：本文的normalized Q输入契约仍被当前A/B/C采用；其中P0反事实信用进入actor的公式及数值探针属于历史诊断。当前Route actor使用task_event，以[P1a说明](e14e_p1a_task_event_credit.md)为准。

本轮仅代码审计、独立数值探针和单元测试。没有环境 rollout、策略训练、checkpoint 策略评估或既有权重修改。所有数字均来自已有训练日志、已保存的训练张量或明确标注的合成 Q-only 问题。

## 1. 审计结论与边界

E14d 训练共有 3414 个唯一已优化 Route 决策，Local/Remote/Defer 的 chosen-Q 监督分别为 941/974/1499。Local 中 721 个发生在 Remote 合法时；三类分别覆盖全部四个 source 和 16 个 episode。按 episode/source/Remote-legality 分箱，共 64 个非空粗粒度状态格，54 个同时包含三类选择。这说明存在粗粒度覆盖重叠，不证明连续任务状态充分重叠，也不提供同状态未执行候选的反事实真值。

Q 输入来自当前 centralized state 及当前 actor encoder 的 detached candidate 表示；G_event 仅用于监督目标，未进入 Q 输入。Q 前向先 gather 合法候选，非法输入不进入归一化或 MLP。只对 chosen leaf 做原有 half-MSE；没有为未执行候选造标签。Q、旧概率、baseline、actor advantage 在原有时点冻结，保持四个 PPO epoch 不变。

代码核验未发现 mask、detach、optimizer 接线错误：Q 参数进入原 critic Adam 唯一参数组，学习率仍为 0.0003；Q loss 不向 actor encoder、主 value network 或 task-value network 传播。训练 Q 输出层已从零初始化变为非零，4k 到 final 的累计参数变化进一步证明 Q 有更新。但历史日志没有逐 update 的实际梯度或 optimizer 更新量；当前离线梯度、跨 checkpoint 参数差及合成拟合更新量不能冒充这些历史量。

此前后半程 1780/1823 个状态的合法 Q range <1e-4。当前 4k checkpoint 保存的 160 个真实训练 transition 中，有 62 个 Route-active 样本。离线前向复现尺度失衡和微弱候选响应，支持输入处理消融；尚未构成“输入尺度导致真实策略失败”的因果证明。

另外，Defer 的大量 G_event 为 binding_value_bootstrap；这可能影响信用质量，但本轮不修改 bootstrap、终局 attribution 或信用公式。当前 G_event 本身是任务事件辅助目标，不宣称其与原始 team reward 的策略梯度严格等价。

## 2. 输入契约

完整候选张量为 `[B,T,N,N+2,411]`，其中 N+2 个 leaf 按原动作顺序为 Idle、Local、Defer、N-1 个 Remote destination。Q batch gather 后为 state `[M,S]`、candidate `[M,N+2,411]`、legal `[M,N+2]`。本次 N=4、S=2469、M=62；Q 输出 `[62,6]`，其中 186 个合法 leaf。

候选 411 维严格保持原构造：前 204 维为当前 leaf 的 fused embedding，中间 204 维为合法 Remote 的平均 context，末三维依次为 Local/Defer/Remote 类型。Local 和 Defer 的前两块相同，末尾类型不同。Idle 是全零占位，Route-active 时不合法。

Remote 的目标信息来自对应目标的 endpoint、burden、validity 等 fused 表示及 source-dependent leaf→destination 映射；没有显式绝对 UAV ID one-hot。P0 不增加新候选特征、不重新排序、不平均各 Remote leaf。若不同目标的全部现有表示完全相同，当前输入本就不能保证通过绝对 ID 区分目标；本轮不把它宣称为已解决的 representation 问题。

CentralizedStateTensorizer 已对 bits、cycles、队列数量、deadline/slack、energy、CPU frequency 等做参考尺度处理，并非所有字段都是原始物理量。特别是任务总量和队列又除以 `N × horizon`，本例为 2000；部分值因此只有 1e-4 量级。CSI 是复信道幅度；历史 interference/denominator 是 noise-reference ratio；historical quality 未压缩，保存张量中最大值 48463.617。详细字段位置、单位、观测范围及变换见后面的自动生成审计表及 JSON。

## 3. 三个独立模式

新增 `training.mappo.route_q_input_mode`：

- `legacy`：默认值，原网络、初始化 RNG、state_dict keys 和前向计算顺序保持一致。默认字段从 canonical identity 中移除，原 E14d hash 保持不变。
- `normalized`：P0a，仅新增固定字段预处理，保留原拼接 MLP 及其宽度和初始化。
- `dual_branch`：P0b，在 P0a 基础上增加独立 state/candidate 投影，再用原类型的 MLP 融合。

对字段 j 定义配置常数 g_j 和固定变换 T_j：离散 mask/布尔字段恒等；标记为压缩的连续字段使用

$$
T_j(x)=\operatorname{sign}(g_jx)\log(1+|g_jx|).
$$
已归一化的有界字段保持恒等；path loss 以 100 dB 为参考线性缩放。信道幅度乘 `sqrt(reference_transmit_power_w / reference_noise_w)` 后作 signed-log；历史比值作 signed-log。任务/队列聚合先撤销 max_tasks 的公共稀释，再压缩，保留原 bits/cycles 的单任务参考单位。这里的 reference_noise_w 只由配置中 noise PSD、RU bandwidth 和 receiver noise figure 计算。

候选表示写为 `c=[e_1,e_2,t]`，两块 embedding 分别使用

$$
R(e)=e/\max\{1,\sqrt{\operatorname{mean}(e^2)}\},\qquad R(c)=[R(e_1),R(e_2),t].
$$
不中心化，零/常量 embedding 均有定义；类型位原样保留。没有 running mean/variance、训练统计拟合、eval 数据拟合或未来信息。固定 gain/compress/schema_version 为持久 buffer，随 checkpoint 保存；train/eval 均不更新。buffer 使用模型默认 dtype，中间缩放/RMS 使用 float64 后转回输入 dtype。合法非有限输入明确报错，非法候选在处理前排除，不用填零掩盖合法数据损坏。

设 `F` 为 `Linear → LeakyReLU → LayerNorm → Linear(1)`：

$$
Q_{P0a}(s,c)=F([T(s),R(c)]).
$$
P0b 的两个投影均为 `Linear(hidden=128) → LeakyReLU → LayerNorm`：

$$
z_s=P_s(T(s)),\quad z_c=P_c(R(c)),\quad
Q_{P0b}(s,c)=F([z_s,z_c,t]).
$$
类型位还直连到融合层，以保留原类型区分路径。Remote 各 leaf 的 embedding 走独立前向；候选置换具有等变性。没有 Dueling、Residual、Local bonus、Q range 正则或 reward 改动。P0a/legacy Q 有 369153 个参数，P0b 有 403073 个参数。

## 4. 不变的目标和信用

$$
L_Q=\frac{1}{2M}\sum_i(Q(s_i,c_{a_i})-G_{\mathrm{event},i})^2,
$$
$$
b_Q=\sum_{a\in\mathcal A_{legal}}\pi_{old}(a\mid s)Q(s,c_a),\quad
A_{CF}=Q(s,c_{chosen})-b_Q,\quad
A_{Route}=A_{shared}+\beta A_{CF}.
$$
β=1，原 Q loss 外部系数 0.5，actor/task/TX/CPU 机制、旧概率、PPO loss、冻结时点、optimizer 学习率、batch、训练预算全部保持原设置。本轮没有恢复 E14c 的直接 task advantage。

## 5. 数值证据与不能声称的效果

真实离线探针只读取 step_4000.pt 中已保存的训练张量；不创建环境、不选择新动作、不步进、不更新真实模型。actor 的前向只计算既有 factual action 的 log-prob 与 candidate 表示。

非线性层没有唯一的可加 state/candidate 分解。因此记录总输出、state-only 输出和加入 candidate 后的差值；这些差值不是两个独立可加的非线性分量。

实际 CF 梯度使用 autograd 对保存张量的 clipped Route surrogate 求导，不执行 actor optimizer step。所有 replay ratio 都在 clip 区间内，因而当前样本上的 CF 项对梯度是可加贡献。该部分没有保存完整 frozen shared GAE，不能恢复原训练完整 shared-gradient 比值。

P0a 的真实数据干预使用训练过的 legacy MLP 原权重，只改变输入变换；它不是训练完成的 P0a。P0b 是新初始化、零输出 head；初始 Q range=0、早期投影梯度=0 都可能是预期现象，不能据此判退化。

合成问题使用独立临时 Q，32 个 state × 5 个 candidate，state 前四维具有 50000 量级。目标直接编码在随机置换的候选数值中，不偏好任何动作类型；独立 16×5 样本检查泛化。使用 240 个独立 Q-only Adam 步、lr=0.01，仅为有限诊断，不是 MEC 训练配置。没有真实 checkpoint 被拟合，没有将合成排序视为 MEC 最优排序。

真实 chosen-Q 改善仍未证明：P0a 同权重输入干预的 MAE 略降而 RMSE 略升；P0b 尚无真实拟合后的预测。不能声称已解决 Route credit、Local/Remote 排名或完成率下降。

## 6. 测试、兼容性及回滚

本轮选择纯张量测试，执行时阻止环境构造。完整项目测试包含 rollout，故未运行完整套件。第一次扩大选择时，两项环境相关历史测试被保护拦截，未创建环境；另捕获新增 gain buffer float64 与现有 updater 单一 dtype 契约不兼容，已在新 conditioner 内修正，未修改 updater。最终通过结果和 XML 见后面的实测汇总。

覆盖 legacy 数值/RNG 等价、mask/非法 NaN 排除、类型保留、极端量纲、零方差、动态候选数、置换/独立路径、合成可学习性、梯度隔离、旧 credit/baseline 恒等式、未来标签不进入输入、buffer 保存恢复、eval 无统计变化、原 optimizer 参数归属、CPU feasibility 边界、TX paired-shadow 公式及 PPO 回归。CUDA 只运行独立张量前向/反向。

旧 E14d 配置 hash 仍为 `2cd92efd656e373ad4c563866e7a2c39f14af4e5c6eef89b633c125565adf87d`。真实旧 checkpoint 通过结构化 loader 和严格 actor/critic 加载，文件 SHA 不变。legacy 无新增模型 buffer；新模式的固定尺度 buffer 和投影参数通过完整 critic state_dict 保存/恢复。

回滚方法：选回原 E14d 配置和原 checkpoint，或新的独立 legacy 对照配置。不同输入模式的 checkpoint 必须匹配配置，不通过 `strict=False` 混载；尤其 P0b 与 legacy 的参数结构不同。既有 E13/E14a/b/c/d 文件和 checkpoint 不覆盖，不用 Git reset 丢弃仓库原有未提交工作。

工程前置条件通过后可启动独立开发对照，但本轮没有启动。下一轮应验证 pre-update chosen-Q 校准、跨 rollout 泛化及实际 CF 梯度，再检查 deterministic Local、DONE、UNBOUND、CPU coverage；不以 Q range 增大或 Local 数量增加单独判成功。

## 7. 实测汇总

| 指标 | legacy 4k 保存张量 | P0a 同权重输入干预 | P0b 新初始化 |
| --- | ---: | ---: | ---: |
| 原始 state 平均范数 | 43895.693 | 43895.693 | 43895.693 |
| 原始 candidate 平均范数 | 20.222873 | 20.222873 | 20.222873 |
| 处理后 state 平均范数 | 43895.693 | 120.89709 | 120.89709 |
| 处理后 candidate 平均范数 | 20.222873 | 20.222856 | 20.222856 |
| 融合前 state 分量/投影范数 | 11822.905 | 55.737946 | 11.313626 |
| 融合前 candidate 分量/投影范数 | 10.385785 | 10.385777 | 11.31313 |
| 融合激活后的 candidate 差量范数 | 6.5792611 | 6.648174 | 3.0391558 |
| 融合 LayerNorm 后 candidate 差量范数 | 0.0084851228 | 1.1412277 | 9.546285 |
| chosen MAE | 0.37478256 | 0.359763 | 0.47843948 |
| chosen RMSE | 0.49683666 | 0.50178552 | 0.57224172 |

不同模式的“分量/投影”含义不同：legacy/P0a 是第一层加权贡献；P0b 是分别经过投影、激活、LayerNorm 的输出。完整中间层数据见 JSON。

| 合成 Q-only（240 步） | 拟合 RMSE | 已知排序准确率 | 独立合成 RMSE | 独立合成排序准确率 |
| --- | ---: | ---: | ---: | ---: |
| legacy | 0.707003 | 92.50% | 0.535134 | 85.62% |
| normalized | 0.003079 | 100.00% | 0.305995 | 97.50% |
| dual_branch | 0.000646 | 100.00% | 0.082491 | 98.12% |

合成泛化仍有误差；尤其 P0a 的独立样本 RMSE 明显高于拟合集误差。这只能证明可学习性，不能证明真实任务价值泛化。

| 真实训练 chosen 类别 | 样本数 | Remote 合法数 | G_event 均值 | 标准差 | 范围 |
| --- | ---: | ---: | ---: | ---: | --- |
| local | 941 | 721 | -0.365256 | 0.588687 | [-0.970299, 0.970299] |
| remote | 974 | 974 | -0.407617 | 0.570401 | [-0.960596, 0.960596] |
| defer | 1499 | 1156 | -0.449840 | 0.153568 | [-0.666918, 0.000000] |

全体 G_event：mean=-0.414480，std=0.447133，范围 [-0.970299,0.970299]。

保存张量上的第一层梯度列范数：legacy state=0.14474084、candidate=7.6250354e-05；同权重 P0a 干预 state=0.10163841、candidate=0.0174662。这些是当前离线 batch 的 Q-loss 梯度，不是历史每步梯度，也不单独证明正确学习。

CF-only Route-active mean 参数梯度范数=4.0753629e-06；保存 partial batch 有 7 个原始 active component，按生产分支均值 reduction 缩放后的该贡献=5.8219469e-07。没有执行 actor step，不将其与缺失的原训练 shared gradient 作比。

4k→final Q 第一层 weight 累计变化 L2=1.080118，输出 weight 变化=0.016635。这跨越多个 update，不是单步 optimizer 更新。

最终纯张量回归：152 passed、82 subtests passed、0 failed/error/skipped（17.78 s），已包含两项 CUDA 数值测试。此前 CUDA 独立检查 2 passed、0 skipped（3.79 s）；旧 checkpoint 通过生产结构化 loader 复测 1 passed（4.63 s），均不重复计入最终独立数量。当前新增测试共 30 项。

证据目录：`D:/My_project_li/logs/development/e14d_p0_input_repair`；原始 XML 为 `tensor_regression_final.xml`、`cuda_tests.xml`、`legacy_checkpoint_test.xml`。第一次保护拦截及新增 dtype 修正前的失败记录保留在 `tensor_regression.xml`，不隐藏初次失败。

## 8. 完整字段审计表

索引为零基半开区间；观测范围来自这 62 个保存训练状态，不是全训练理论范围。字段来源均为原 CentralizedStateTensorizer 当前状态编码。

| 字段 | 区间 | 原编码单位 | 已观测范围 | Q 专用处理 |
| --- | --- | --- | --- | --- |
| slot | [0,1) | slot/horizon | [0.692, 0.998] | 线性，gain=1 |
| positions | [1,13) | position/bounds | [0.0978447, 1] | 线性，gain=1 |
| velocities | [13,25) | velocity/config std | [-2.39039, 2.52197] | signed-log，gain=1 |
| tx_power | [25,29) | W/reference W | [0.788442, 1.04459] | signed-log，gain=1 |
| cpu_frequency | [29,33) | Hz/reference Hz | [0.717014, 1.9988] | signed-log，gain=1 |
| cpu_coefficient | [33,37) | coefficient/reference | [0.931033, 1.16462] | signed-log，gain=1 |
| energy | [37,41) | J/reference J | [0.403132, 0.987347] | 线性，gain=1 |
| arrivals | [41,45) | probability | [0.02, 0.11] | 线性，gain=1 |
| neighbor_mask | [45,61) | dimensionless | [0, 1] | 保留离散编码 |
| distance | [61,77) | m/diagonal | [0, 0.742883] | 线性，gain=1 |
| true_channel_real | [77,397) | channel amplitude | [-1.39021e-05, 8.90909e-05] | signed-log，gain=7079458 |
| true_channel_imag | [397,717) | channel amplitude | [-4.96328e-05, 5.181e-05] | signed-log，gain=7079458 |
| blocked_mask | [717,733) | dimensionless | [0, 1] | 保留离散编码 |
| shadowing | [733,749) | dB/config std | [-1.77122, 2.58799] | signed-log，gain=1 |
| path_loss | [749,765) | dB | [0, 126.842] | 线性，gain=0.01 |
| stale_csi_real | [765,1085) | channel amplitude | [-1.28627e-05, 0.000136226] | signed-log，gain=7079458 |
| stale_csi_imag | [1085,1405) | channel amplitude | [-5.44539e-05, 6.36475e-05] | signed-log，gain=7079458 |
| csi_valid | [1405,1421) | dimensionless | [0, 1] | 保留离散编码 |
| csi_age | [1421,1437) | slots/horizon | [0, 0.006] | 线性，gain=1 |
| interference | [1437,1517) | W/noise W | [0, 0] | signed-log，gain=1 |
| interference_valid | [1517,1597) | dimensionless | [0, 1] | 保留离散编码 |
| message_age | [1597,1601) | (slots+1)/horizon | [0.004, 0.47] | 线性，gain=1 |
| message_valid | [1601,1605) | dimensionless | [1, 1] | 保留离散编码 |
| historical_denominator | [1605,1685) | W/noise W | [1, 1] | signed-log，gain=1 |
| historical_quality | [1685,2005) | dimensionless channel quality | [0, 48463.6] | signed-log，gain=1 |
| quality_valid | [2005,2325) | dimensionless | [0, 1] | 保留离散编码 |
| task_u0_count | [2325,2326) | task reference units/max_tasks | [0, 0.0025] | signed-log，gain=2000 |
| task_u0_bits | [2326,2327) | task reference units/max_tasks | [0, 0.00189036] | signed-log，gain=2000 |
| task_u0_cycles | [2327,2328) | task reference units/max_tasks | [0, 0.00137492] | signed-log，gain=2000 |
| task_u0_remaining_bits | [2328,2329) | task reference units/max_tasks | [0, 0.000493554] | signed-log，gain=2000 |
| task_u0_remaining_cycles | [2329,2330) | task reference units/max_tasks | [0, 0.00118696] | signed-log，gain=2000 |
| task_u0_slack | [2330,2331) | slack/max_slack | [0, 0.273333] | 线性，gain=1 |
| task_u0_unbound | [2331,2332) | task reference units/max_tasks | [0, 0.0005] | signed-log，gain=2000 |
| task_u0_local | [2332,2333) | task reference units/max_tasks | [0, 0.001] | signed-log，gain=2000 |
| task_u0_tx | [2333,2334) | task reference units/max_tasks | [0, 0.0005] | signed-log，gain=2000 |
| task_u0_cpu | [2334,2335) | task reference units/max_tasks | [0, 0.0015] | signed-log，gain=2000 |
| task_u1_count | [2335,2336) | task reference units/max_tasks | [0.001, 0.0035] | signed-log，gain=2000 |
| task_u1_bits | [2336,2337) | task reference units/max_tasks | [0.000494653, 0.00216462] | signed-log，gain=2000 |
| task_u1_cycles | [2337,2338) | task reference units/max_tasks | [0.000355464, 0.00139046] | signed-log，gain=2000 |
| task_u1_remaining_bits | [2338,2339) | task reference units/max_tasks | [0, 0.000515479] | signed-log，gain=2000 |
| task_u1_remaining_cycles | [2339,2340) | task reference units/max_tasks | [0.000336343, 0.00133786] | signed-log，gain=2000 |
| task_u1_slack | [2340,2341) | slack/max_slack | [0, 0.233333] | 线性，gain=1 |
| task_u1_unbound | [2341,2342) | task reference units/max_tasks | [0, 0.001] | signed-log，gain=2000 |
| task_u1_local | [2342,2343) | task reference units/max_tasks | [0, 0.0015] | signed-log，gain=2000 |
| task_u1_tx | [2343,2344) | task reference units/max_tasks | [0, 0.0005] | signed-log，gain=2000 |
| task_u1_cpu | [2344,2345) | task reference units/max_tasks | [0, 0.002] | signed-log，gain=2000 |
| task_u2_count | [2345,2346) | task reference units/max_tasks | [0, 0.001] | signed-log，gain=2000 |
| task_u2_bits | [2346,2347) | task reference units/max_tasks | [0, 0.000909975] | signed-log，gain=2000 |
| task_u2_cycles | [2347,2348) | task reference units/max_tasks | [0, 0.000625918] | signed-log，gain=2000 |
| task_u2_remaining_bits | [2348,2349) | task reference units/max_tasks | [0, 0.000469312] | signed-log，gain=2000 |
| task_u2_remaining_cycles | [2349,2350) | task reference units/max_tasks | [0, 0.000439364] | signed-log，gain=2000 |
| task_u2_slack | [2350,2351) | slack/max_slack | [0, 0.54] | 线性，gain=1 |
| task_u2_unbound | [2351,2352) | task reference units/max_tasks | [0, 0.0005] | signed-log，gain=2000 |
| task_u2_local | [2352,2353) | task reference units/max_tasks | [0, 0.0005] | signed-log，gain=2000 |
| task_u2_tx | [2353,2354) | task reference units/max_tasks | [0, 0.0005] | signed-log，gain=2000 |
| task_u2_cpu | [2354,2355) | task reference units/max_tasks | [0, 0.001] | signed-log，gain=2000 |
| task_u3_count | [2355,2356) | task reference units/max_tasks | [0, 0.002] | signed-log，gain=2000 |
| task_u3_bits | [2356,2357) | task reference units/max_tasks | [0, 0.00114091] | signed-log，gain=2000 |
| task_u3_cycles | [2357,2358) | task reference units/max_tasks | [0, 0.000747146] | signed-log，gain=2000 |
| task_u3_remaining_bits | [2358,2359) | task reference units/max_tasks | [0, 0.000436281] | signed-log，gain=2000 |
| task_u3_remaining_cycles | [2359,2360) | task reference units/max_tasks | [0, 0.000614084] | signed-log，gain=2000 |
| task_u3_slack | [2360,2361) | slack/max_slack | [0, 0.353333] | 线性，gain=1 |
| task_u3_unbound | [2361,2362) | task reference units/max_tasks | [0, 0.0005] | signed-log，gain=2000 |
| task_u3_local | [2362,2363) | task reference units/max_tasks | [0, 0.0005] | signed-log，gain=2000 |
| task_u3_tx | [2363,2364) | task reference units/max_tasks | [0, 0.0005] | signed-log，gain=2000 |
| task_u3_cpu | [2364,2365) | task reference units/max_tasks | [0, 0.0015] | signed-log，gain=2000 |
| bound_destination_counts | [2365,2381) | count/max_tasks | [0, 0.002] | signed-log，gain=2000 |
| queues | [2381,2445) | count/max_tasks | [0, 0.002] | signed-log，gain=2000 |
| building_0_present | [2445,2446) | dimensionless | [1, 1] | 保留离散编码 |
| building_0_geometry | [2446,2451) | position/bounds | [0.15, 1.5] | 线性，gain=1 |
| building_1_present | [2451,2452) | dimensionless | [1, 1] | 保留离散编码 |
| building_1_geometry | [2452,2457) | position/bounds | [0.15, 1.5] | 线性，gain=1 |
| building_2_present | [2457,2458) | dimensionless | [1, 1] | 保留离散编码 |
| building_2_geometry | [2458,2463) | position/bounds | [0.55, 1.5] | 线性，gain=1 |
| building_3_present | [2463,2464) | dimensionless | [1, 1] | 保留离散编码 |
| building_3_geometry | [2464,2469) | position/bounds | [0.65, 1.5] | 线性，gain=1 |

## 9. 独立配置、版本与命令（仅准备，未执行训练/评估）

Git commit：`eb1140cefc4a0ff79b8bd0c05e73c7a23528d7ba`。工作树原本已有未提交修改，仍为 dirty；不能只用该 commit 标识本轮代码。精确源码 SHA-256 与配置文件 SHA-256 已保存到 `logs/development/e14d_p0_input_repair/experiment_identity.json`。

| 模式 | 配置文件 | canonical config hash | 输出根目录 |
| --- | --- | --- | --- |
| legacy | `D:\My_project_li\experiments\cooperative-load\configs\e14d-p0-legacy-seed42-8k-dev.json` | `c9bda309c73ef8583949a7e06da0d24f595c7ccf7ffde32027f4a37e0a926f12` | `D:/My_project_li/logs/development/e14d_p0_legacy_seed42_8k` |
| normalized | `D:\My_project_li\experiments\cooperative-load\configs\e14d-p0-normalized-seed42-8k-dev.json` | `c155c181a5bbca4d8af553694f5d5e2eaa910ff67e472ae14390c5045cacbb61` | `D:/My_project_li/logs/development/e14d_p0_normalized_seed42_8k` |
| dual_branch | `D:\My_project_li\experiments\cooperative-load\configs\e14d-p0-dual-branch-seed42-8k-dev.json` | `abfad889f23110b18434568f029c29de8c14f99d52827a1644c4cccef62c0f4b` | `D:/My_project_li/logs/development/e14d_p0_dual_branch_seed42_8k` |

三组均为 seed42、16×500 slots、8000 transitions，checkpoint interval=4000；保留 step_4000.pt 和 final.pt。仅 Q mode 和独立 output roots 不同。旧 E14d/E14c 与 seed44–46 不改。以下 run_id 对应当前代码/配置身份；未来代码版本变化时应使用实际新 run 的 checkpoint，不复用旧目录。

PowerShell 中先切换目录：

```powershell
Set-Location 'D:\My_project_li'
```

### legacy：8k 训练

```powershell
& 'D:\Miniconda\python.exe' -B -X utf8 'D:\My_project_li\main.py' --mode rl --config 'D:\My_project_li\experiments\cooperative-load\configs\e14d-p0-legacy-seed42-8k-dev.json'
```

4k checkpoint 的 eval1044 开发诊断：

```powershell
& 'D:\Miniconda\python.exe' -B -X utf8 -m src.evaluation.service_diagnostics --config 'D:\My_project_li\experiments\cooperative-load\configs\e14d-p0-legacy-seed42-8k-dev.json' --evaluate-from 'D:\My_project_li\logs\development\e14d_p0_legacy_seed42_8k\ca_gat_mappo__remote_stress_v2__seed-42__cfg-c9bda309c73e__git-eb1140cefc4a\checkpoints\step_4000.pt' --evaluation-seed 1044 --evaluation-device cpu --development-checkpoint --logs-dir 'D:\My_project_li\logs\development\e14d_p0_legacy_seed42_8k\eval1044_4k'
```

8k checkpoint 的 eval1044 开发诊断：

```powershell
& 'D:\Miniconda\python.exe' -B -X utf8 -m src.evaluation.service_diagnostics --config 'D:\My_project_li\experiments\cooperative-load\configs\e14d-p0-legacy-seed42-8k-dev.json' --evaluate-from 'D:\My_project_li\logs\development\e14d_p0_legacy_seed42_8k\ca_gat_mappo__remote_stress_v2__seed-42__cfg-c9bda309c73e__git-eb1140cefc4a\checkpoints\final.pt' --evaluation-seed 1044 --evaluation-device cpu --development-checkpoint --logs-dir 'D:\My_project_li\logs\development\e14d_p0_legacy_seed42_8k\eval1044_8k'
```

### normalized：8k 训练

```powershell
& 'D:\Miniconda\python.exe' -B -X utf8 'D:\My_project_li\main.py' --mode rl --config 'D:\My_project_li\experiments\cooperative-load\configs\e14d-p0-normalized-seed42-8k-dev.json'
```

4k checkpoint 的 eval1044 开发诊断：

```powershell
& 'D:\Miniconda\python.exe' -B -X utf8 -m src.evaluation.service_diagnostics --config 'D:\My_project_li\experiments\cooperative-load\configs\e14d-p0-normalized-seed42-8k-dev.json' --evaluate-from 'D:\My_project_li\logs\development\e14d_p0_normalized_seed42_8k\ca_gat_mappo__remote_stress_v2__seed-42__cfg-c155c181a5bb__git-eb1140cefc4a\checkpoints\step_4000.pt' --evaluation-seed 1044 --evaluation-device cpu --development-checkpoint --logs-dir 'D:\My_project_li\logs\development\e14d_p0_normalized_seed42_8k\eval1044_4k'
```

8k checkpoint 的 eval1044 开发诊断：

```powershell
& 'D:\Miniconda\python.exe' -B -X utf8 -m src.evaluation.service_diagnostics --config 'D:\My_project_li\experiments\cooperative-load\configs\e14d-p0-normalized-seed42-8k-dev.json' --evaluate-from 'D:\My_project_li\logs\development\e14d_p0_normalized_seed42_8k\ca_gat_mappo__remote_stress_v2__seed-42__cfg-c155c181a5bb__git-eb1140cefc4a\checkpoints\final.pt' --evaluation-seed 1044 --evaluation-device cpu --development-checkpoint --logs-dir 'D:\My_project_li\logs\development\e14d_p0_normalized_seed42_8k\eval1044_8k'
```

### dual_branch：8k 训练

```powershell
& 'D:\Miniconda\python.exe' -B -X utf8 'D:\My_project_li\main.py' --mode rl --config 'D:\My_project_li\experiments\cooperative-load\configs\e14d-p0-dual-branch-seed42-8k-dev.json'
```

4k checkpoint 的 eval1044 开发诊断：

```powershell
& 'D:\Miniconda\python.exe' -B -X utf8 -m src.evaluation.service_diagnostics --config 'D:\My_project_li\experiments\cooperative-load\configs\e14d-p0-dual-branch-seed42-8k-dev.json' --evaluate-from 'D:\My_project_li\logs\development\e14d_p0_dual_branch_seed42_8k\ca_gat_mappo__remote_stress_v2__seed-42__cfg-abfad889f231__git-eb1140cefc4a\checkpoints\step_4000.pt' --evaluation-seed 1044 --evaluation-device cpu --development-checkpoint --logs-dir 'D:\My_project_li\logs\development\e14d_p0_dual_branch_seed42_8k\eval1044_4k'
```

8k checkpoint 的 eval1044 开发诊断：

```powershell
& 'D:\Miniconda\python.exe' -B -X utf8 -m src.evaluation.service_diagnostics --config 'D:\My_project_li\experiments\cooperative-load\configs\e14d-p0-dual-branch-seed42-8k-dev.json' --evaluate-from 'D:\My_project_li\logs\development\e14d_p0_dual_branch_seed42_8k\ca_gat_mappo__remote_stress_v2__seed-42__cfg-abfad889f231__git-eb1140cefc4a\checkpoints\final.pt' --evaluation-seed 1044 --evaluation-device cpu --development-checkpoint --logs-dir 'D:\My_project_li\logs\development\e14d_p0_dual_branch_seed42_8k\eval1044_8k'
```

独立合成探针（无需任何训练 checkpoint）：

```powershell
& 'D:\Miniconda\python.exe' -B -X utf8 -m src.evaluation.route_q_probe --synthetic-only --synthetic-steps 240 --output-dir 'D:\My_project_li\logs\development\e14d_p0_input_repair\standalone_synthetic'
```

保存训练张量的只读探针：

```powershell
& 'D:\Miniconda\python.exe' -B -X utf8 -m src.evaluation.route_q_probe --config 'D:\My_project_li\experiments\cooperative-load\configs\e14d-v1-route-counterfactual-q-seed42-8k-dev.json' --checkpoint 'D:\My_project_li\logs\development\e14d_route_counterfactual_q_seed42_8k\ca_gat_mappo__remote_stress_v2__seed-42__cfg-2cd92efd656e__git-eb1140cefc4a\checkpoints\step_4000.pt' --final-checkpoint 'D:\My_project_li\logs\development\e14d_route_counterfactual_q_seed42_8k\ca_gat_mappo__remote_stress_v2__seed-42__cfg-2cd92efd656e__git-eb1140cefc4a\checkpoints\final.pt' --training-log 'D:\My_project_li\logs\development\e14d_route_counterfactual_q_seed42_8k\ca_gat_mappo__remote_stress_v2__seed-42__cfg-2cd92efd656e__git-eb1140cefc4a\route_option_q_updates_v1.jsonl' --output-dir 'D:\My_project_li\logs\development\e14d_p0_input_repair\saved_training_probe'
```

测试选择详见 `test_selection_final.json`。测试执行时拦截环境构造；不得直接扩大为会触发 rollout 的完整套件。

