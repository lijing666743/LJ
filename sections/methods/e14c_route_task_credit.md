# E14c：任务事件 Route credit 与独立 task value

> 阅读定位：本文保留当前方法所继承的定义及原阶段验证记录。旧进度、预算和命令属于历史状态；当前结果与C续训入口见[复现说明](../../docs/reproduction/REPRODUCE_ABC.md)。

本文件记录开发方法，不包含 E14c 训练结果，也不宣称已证明 Local 应优于 Remote。

## 问题与输入核验

已有开发诊断显示 E14b-8k/eval1044 的 192 次 Route 决策中，Local top-1/top-2 都为 0；平均 Local/Remote/Defer 概率为 18.38%/41.73%/39.89%。与此同时，E14b 训练中 Local 已有 947/3545 次真实采样。该证据支持继续检验任务结果到 Route 排序的学习链，不能单凭它证明 credit 是唯一根因。

现有 RouteOptionFeatureExtractor 已含任务剩余 bits/cycles、slack、源端能量/CPU/本地及未绑定队列、目标公开 CPU 负载、目标方向 TX 队列、历史速率/outage/可见性等信息。它提供负载与可行性代理，不提供 oracle 完成时间。E14c 不添加 actor 特征，不修改 action semantics、destination decoder 或 mask。

## 相互配套的机制

1. 保留 E14b 顶层 log-barrier、0.05 初始系数及 6000 transitions 退火。
2. 复用 TaskTerminalAttribution、AgentRewardTerms 与 Route 事件身份，建立任务事件 return。
3. 新增训练专用、独立参数的集中式 V_task。冻结 rollout 期间的值用于 Route baseline 和未终结任务 bootstrap。
4. 只有 Route PPO advantage 加入任务项。原 team/role critic、reward、TX paired-shadow、CPU gate 和 evaluation masked argmax 均保留。

V_task 输入为当前 centralized state 加当前任务 18 维特征：归一化的剩余/初始 bits、剩余/初始 cycles、slack、age；四个 live phase 指示；source/destination one-hot。任务 ID 仅用于日志关联，不进入网络。网络为 Linear–LeakyReLU–LayerNorm–Linear，宽度沿用 encoder hidden=128，最终层零初始化。初始化在独立 RNG fork 内，避免改变 actor/base critic 初值及采样流。此网络不与 actor 或 base critic 共享参数。

## Return、advantage 和时间索引

令 t 为 Route 决策 slot，d 为本任务终局在 slot 末结算的 slot，K 为 rollout 最后一个 slot，gamma=0.99。C_j 为复用现有 terminal attribution 的任务 payoff：DONE 取 source_reward+destination_reward，EXPIRED 取 -(source_penalty+destination_penalty)，TRUNCATED 取 0 并标记 censored。Local/Remote 使用同一规则，不因动作类别另设奖励。

对于 Local/Remote 绑定：

$$
G^{\mathrm{bind}}_{j,t}=\begin{cases}
\gamma^{d-t} C_j,&\text{本 rollout 内观察到终局},\\
\gamma^{K+1-t}\bar V_{\mathrm{task}}(s_{K+1},j),&\text{rollout 结束时仍存活}.
\end{cases}
$$
对于 Defer，若本 rollout 内后来在 b 绑定且绑定后仍存活，则在绑定后的当前状态停止：

$$
G^{\mathrm{defer}}_{j,t}=\gamma^{b+1-t}\bar V_{\mathrm{task}}(s_{b+1},j).
$$
因此随后 DONE 的实际正终局标签不会直接复制给早期 Defer。若直到未绑定过期，则各 Defer 使用 gamma^(d-t) C_j；若未结束则使用 rollout frontier bootstrap。绑定当 slot 即过期/截断时，该终局已经发生在 binding event，使用 gamma^(b-t) C_j，避免引用不存在的 post-state。该设计允许有价值的等待得到正 value 差，但不把每次等待都当作完成任务的原因。

$$
A^{\mathrm{task}}_{j,t}=G^{\mathrm{event}}_{j,t}-\bar V_{\mathrm{task}}(s_t,j),\qquad
A^{\mathrm{Route}}_{i,t}=A^{\mathrm{shared}}_{i,t}+\alpha A^{\mathrm{task}}_{j,t},\quad \alpha=1.
$$
任务 value 使用所有 live task 的 pre-state，回归完整终局 return 或 frontier bootstrap；与 Defer actor 的 binding-event 停止规则有意不同。所有 targets/advantages 在首次 PPO epoch 前冻结。

$$
L_{\mathrm{taskV}}=\frac12\operatorname{mean}_j(V_{\mathrm{task}}-G^{\mathrm{task}})^2,\qquad
L=L_{\mathrm{existing}}+0.5L_{\mathrm{taskV}}.
$$
保持现有不归一化 advantage 的设置。任务 value 参数纳入既有 critic Adam 保存/恢复，但与 base critic 分别裁剪梯度；不新增 optimizer。actor 的原全局梯度裁剪保留，因此新增 Route 梯度可间接改变其他 actor 参数的裁剪缩放，这不能被描述为整个训练数值轨迹不变。

## PPO、边界与理论限度

行为采样始终来自原 masked actor；stored old log-prob 与采样分布一致。Route ratio 使用完整 Route leaf（Remote 包括 destination 条件概率）：exp(new_log_prob-old_log_prob)，沿用 epsilon=0.2 的 PPO clipping。仅替换 Route advantage，TX/CPU surrogate 输入规则不改。

不跨 PPO policy-version 回放旧 Route action。跨 rollout 的任务后果通过训练专用 V_task 的 bootstrap 传播；日志另保存后续 terminal facts，仅用于离线 task_id join，不补算旧 actor 梯度。episode/source/task 联合身份阻止 task ID 重用串联。终局发生在 slot 末，故 terminal 指数为 d-t，而 post-state 位于下一 slot，指数为 d+1-t；真实时序测试覆盖最后一 slot 绑定与截断。

该实现增加一个对动作类别对称的完成/过期辅助目标，重加权了 completion learning；它不是严格保持原 team objective 等价的 RUDDER，也不是 COMA counterfactual estimator，不能声称对原目标无偏或保证改善。V_task bootstrap 可能存在冷启动/估计偏差；当前 actor 特征未必充分；Defer 仍可能因原即时成本和负结果折扣而被高估。需要通过 bootstrap 比例、value loss、按终局分组的 advantage 和最终性能检验这些风险。

## Telemetry 与消融

保留 route_training_rollouts_v1.jsonl、route_diagnostic_samples_v1.jsonl；新增 route_task_credit_updates_v1.jsonl。记录每个 PPO epoch 的真实 Route old/new log-prob、ratio、clipped/unclipped surrogate、初始和当前顶层概率、shared/task/combined advantage、event return/full-task return/value、terminal/slot/delay/censor、任务身份和 destination。按 Local/Remote/Defer 与 0–2k/2–4k/4–6k/6–8k 聚合。count 为 epoch 记录数，unique_factual_count 为真实采样数，避免把 4 epochs 当作 4 次环境访问。

同时记录所有 task_terminals/route_decisions，允许跨 rollout 离线补齐最终结局。未优化 final tail 仅记录事实，optimized=false，不虚构 advantage 或 PPO ratio。8000 real transitions 沿用 E14b 的 256-step 整 rollout 规则：7936 用于更新，最后 64 个不做 PPO 更新。

消融：route_credit_mode=shared_gae 关闭 E14c；alpha=0 保留独立 task value 的训练，但不加入 Route task advantage。后者的单步合成 optimizer 测试检查 actor/base critic 与 E14b 逐位一致。不得把 mode off 的兼容性扩展为 E14c 训练结果已得到验证。

## 文献与代码对应

- [RUDDER, NeurIPS 2019](https://papers.neurips.cc/paper_files/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html)：借鉴迟到结果关联早期动作；核验[作者示例的预测差分](https://github.com/ml-jku/rudder-demonstration-code/blob/master/rudder.py)。未移植 LSTM、episode replay 或 reward redistribution 的等价性主张。
- [COMA, AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11794)：核验[PyMARL learner](https://github.com/oxwhirl/pymarl/blob/master/src/learners/coma_learner.py)的 masked policy、Q baseline 和 TD-lambda target。当前不具备已验证的各 Route 反事实 Q，因此不假装实现 COMA。
- [DAC, NeurIPS 2019](https://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/zhangnips19a.pdf)：借鉴高低层分开估计 advantage，核验[作者 PPO agent](https://github.com/ShangtongZhang/DeepRL/blob/DAC/deep_rl/agent/ASquaredC_PPO_agent.py)。未移植 options termination 或重构 actor。
- [MAPPO, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract.html)：核验[作者 clipped PPO](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/r_mappo.py)和 buffer 的 log-prob、GAE、mask。其共享 GAE 不直接回答 task-level offloading ranking。
- [Decima, SIGCOMM 2019](https://web.mit.edu/decima/)：核验[作者训练代码](https://github.com/hongzimao/decima-sim/blob/master/train.py)的事件等待 reward 累计与时间 baseline，借鉴不能遗漏等待后果。未复制平均 job completion time reward、GNN 或系统模型。
- [LyDROO, IEEE TWC 2021](https://ieeexplore.ieee.org/document/9449944)：核验[作者 resource allocation](https://github.com/revenol/LyDROO/blob/main/ResourceAllocation.py)对 local/offload 的队列/能量权衡；用于 offloading 问题对照，不作为 PPO terminal-credit 已验证的证据，未移植虚拟队列或物理优化器。

## 8k 开发验收

比较 E14b/E14c 的同阶段 Local/Remote/Defer 真实采样、actor 概率、top-1/top-2；按状态、动作和最终 DONE/EXPIRED 分组的 task advantage 与实际 PPO surrogate；特别检查 UNBOUND EXPIRED 的 Defer 是否收到负 credit，以及成功前 Defer 的 credit_stop。记录 bootstrap/censor 占比和 value loss，不能把稀少 terminal labels 的 bootstrap 预测当作已验证任务价值。

4k/8k eval1044 保持 deterministic masked argmax，核验任务 realization 一致，再看 Local top-1 是否随状态出现，并共同检查 DONE、UNBOUND、CPU_WAIT、CPU coverage、energy、TX success。只恢复 Local 采样或 Local>0 不算成功，也不能凭单个 eval1044 作正式论文结论。
