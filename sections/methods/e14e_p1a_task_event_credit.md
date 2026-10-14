# E14e-P1a：恢复任务事件 Route actor credit

> 阅读定位：本文保留当前方法所继承的定义及原阶段验证记录。旧进度、预算和命令属于历史状态；当前结果与C续训入口见[复现说明](../../docs/reproduction/REPRODUCE_ABC.md)。

## 范围与验收

本轮只切换 Route actor 的 credit 来源，保留 P0a normalized RouteOptionQ 的完整辅助训练。未执行 RL training、环境构造/rollout、checkpoint 策略评估或自动调参。单元测试中的有限 optimizer 更新只作用于一次性模型和人工标签，不产生实验结果。

验收结果及测试计数以 `logs/development/e14e_p1a_engineering/tensor_regression_accepted.xml` 和本文件最终补充为准。训练效果、Local ranking 和 DONE 是否改善仍需本人后续执行受控 8k 训练和 eval1044。

## 修改前接线审计

1. E14c `construct_task_credit` 定义 $A_{task}=G_{event}-V_{task,old}$。旧 task value 在采集时冻结，输入为当前 centralized state 与当前任务特征，无未来终局标签。
2. DONE/EXPIRED 的事件 payoff 复用 `TaskTerminalAttribution` 的 source/destination 分配；EXPIRED 为负，DONE 为正。Route 时刻为 $t$，slot $d$ 末终局的回报为 $\gamma^{d-t}r_{event}$。正 payoff 不保证任意 baseline 下 advantage 必定为正；定向测试使用可核验 baseline 验证符号。
3. TRUNCATED 为 censoring，终局 payoff 为零，不伪造成 EXPIRED；因此 advantage 为 $-V_{task,old}$，并非强行设为零。
4. repeated Defer 保留各次原始 decision identity。后续绑定有 post-step value 时，Defer 在绑定处停止：$\gamma^{b+1-t}V_{task,old}(s_{b+1})$，不复制绑定后后来 DONE 的完整 payoff。同 slot 绑定并终局、没有 post-state 时复用 $\gamma^{b-t}r_{event}$。
5. 一直未绑定并过期时，当前 rollout 内各次 Defer 接收各自折扣的负终局 target；只有一个任务终局事实，不重复插入 terminal。不同 decision 得到同一事件的折扣 credit 是原有定义，不等于重复终局记录。
6. 未终局状态在当前 rollout frontier 使用 next-state value bootstrap，指数对应 next slot。关联键是 episode/source/task；拒绝混合 policy version。**不跨 PPO 更新 replay 已优化过的 actor decision**；跨 rollout 终局事实可作诊断，但不能宣称所有历史 Route 均直接得到完整终局回报。
7. E14c 最终 Route actor 使用 shared 加 task credit。E14d/P0a 在 `prepare_route_q` 将它改写成 shared 加 frozen CF，并在 updater 将结果替换回 task-credit batch。
8. P0a/P1a 配置 `advantage_normalization=false`，Route 冻结 advantage 不再次归一化；PPO clip、旧策略叶概率及 Remote 条件概率不变。
9. TX/CPU 使用各自原有分支 advantage；TX paired-shadow 不复用 P1a Route credit。Route 私有网络、动作、mask、CPU gate 和 TX decoder 均未改。
10. 主 critic、task value、Q 仍在原 critic Adam 中训练，学习率 0.0003、各辅助 loss 系数均 0.5；独立梯度裁剪接线不变。Q 输入 detached，不更新 actor encoder；Route actor advantage detached，不反向更新 Q。

## P1a 数学定义与配置

$$
A_{task}=G_{event}-V_{task,old},\quad
A_{CF}=Q_{old}(s,o_{chosen})-\sum_o\pi_{old}(o\mid s)Q_{old}(s,o).
$$
$$
\begin{aligned}
\text{P0a: } A_{route}&=A_{shared}+\beta A_{CF},\\
\text{P1a: } A_{route}&=A_{shared}+\alpha_{task}A_{task},
\qquad \alpha_{task}=1.0.
\end{aligned}
$$
$s$ 是决策时状态，$o$ 是合法 Route 叶动作（Remote 含目的地），$\pi_{old}$ 是采集时真实旧策略。P1a 中 CF 的有效 actor 系数严格为零；原配置中的 beta=1 保留用于 Q/CF 诊断，不能把它解释为仍进入 actor。

新增 `training.mappo.route_actor_credit_source`：默认 `counterfactual_q` 完整保留 P0a 路径；`task_event` 复用 E14c 已冻结的 combined tensor，不重新实现 return。复用现有 `route_task_credit_coefficient=1.0`，未新增同义系数。原 `route_credit_mode=task_event_value_v1` 保留，负责事件采集/value 路径；新增 selector 只负责 Q 开启时的 actor 来源。配置拒绝不支持的 hybrid 或不保留 Q 的 task_event 组合。

Q 的 normalized conditioning、候选构造、chosen-only target、半平方误差、optimizer、mask 和 old-policy baseline 全部保持；同一固定 buffer 的四 epoch Q/task-value losses 与 critic 更新在两种 actor 来源下逐位相同。未选择动作仍没有真实反事实监督，P1a 不会因此证明 Q ranking 正确。

**原有全局 actor 梯度裁剪的边界**：即使 TX/CPU advantage、结构和分支 surrogate 不变，Route 梯度改变仍可能改变全局裁剪比例，并间接改变其他 actor 参数的更新幅度。不能宣称训练后 TX/CPU 权重逐位相同；本轮没有改变该既有规则。

## 修改文件

| 文件 | 原因 |
|---|---|
| `src/config.py` | selector、组合校验及默认值身份兼容；不更改旧配置 hash |
| `src/models/route_option_q.py` | 仅选择冻结 actor advantage，保留全部 Q 训练；补充四组件诊断 |
| `src/models/ca_gat_mappo_update.py` | 将 selector 传入现有准备路径；辅助 loss 接线不变 |
| `src/models/route_task_credit.py` | 增加按 epoch/动作类别统计及有效 actor 系数；return/value loss 不变 |
| `src/evaluation/route_q_probe.py` | 离线张量探针尊重新 selector，无策略评估 |
| `src/evaluation/route_credit_probe.py` | 新增保存张量上的只读真实梯度诊断 |
| `tests/test_route_actor_credit_source.py` | 来源切换、生命周期归因、隔离、CUDA、checkpoint、CPU gate 及四 epoch 生产 updater 纯张量测试 |
| `experiments/cooperative-load/configs/e14e-p1a-task-event-credit-seed42-8k-dev.json` | 新独立开发配置 |
| `sections/methods/e14e_p1a_task_event_credit.md` | 本交付、审计和待执行命令 |
| `README.md` | 宏观进度说明 |

## Telemetry

现有 `route_task_credit_updates_v1.jsonl` 和 `route_option_q_updates_v1.jsonl` 保留，追加 actor 来源、有效 task/CF 系数及 `credit_components_by_epoch`。按 Local/Remote/Defer 与 PPO epoch 分别统计 shared、task、CF、final 的 count、mean、population std、RMS、min/max 和正/负/零计数。样本保留原 return/value/ratio/surrogate 及 Q 校准字段；没有 CF 的历史 E14c 记录标为不可获得，不伪造为零。

## 实际只读梯度探针

输入为 P0a 真实 step_4000 checkpoint 内保存的 160 transitions（3840–3999），64 个 Route-active 样本；advantage 来自对应后续更新 epoch0 的真实日志，未用部分轨迹伪造完整 GAE。计算同一保存 actor 上的诊断 loss，不是 P1a 训练后结果。未构造环境、未采样动作、未执行 optimizer.step。

| Route-active 平均 surrogate | Route 私有参数梯度范数 |
|---|---:|
| shared-only | 1.861364804 |
| task-only | 0.600313274 |
| CF-only | 0.003255131 |
| shared+task | 2.021600757 |

task/shared = 0.322512424；shared/task = 3.100655748；CF/shared = 0.001748787。cos(shared,task)=0.117153446，cos(shared,CF)=−0.547565299，cos(shared,combined)=0.955526694，cos(task,combined)=0.404817104。

当前子集 ratio 在 0.999999762–1.000000238，超 clip 样本为零；combined 与两组件梯度和的残差范数 5.23e−7。生产 aggregation 此子集有 7 个活跃 component，除以 7 后 shared/task/CF/combined 范数分别为 0.265909258 / 0.085759039 / 0.000465019 / 0.288800108。探针报告只覆盖历史部分 minibatch、私有 Route 参数与裁剪前 surrogate，不代表完整 update、entropy、全局裁剪或 Adam 后的梯度贡献。

证据：`logs/development/e14e_p1a_engineering/route_credit_gradients.json`；输入 checkpoint SHA-256：`39786b3db2fc496068806e8dce4a7dafdd57c9618f57a14022e7b8d8816b0ff7`。

## 配置身份、兼容与回滚

P0a hash：`c155c181a5bbca4d8af553694f5d5e2eaa910ff67e472ae14390c5045cacbb61`。

P1a hash：`2f143a85c0e37f605bbf38b59b3a31267a305464179593d9828c7c0a1518e3fa`。

与 P0a 相比唯一算法差异是 `route_actor_credit_source: counterfactual_q（缺省） → task_event`，另有 logs/plots/dashboard 三个独立输出根目录。normalized、alpha=1、beta=1（诊断）、Q/value loss、学习率、rollout/minibatch/epoch、探索 schedule、CPU/TX 和 reward 均相同。完整 diff：`logs/development/e14e_p1a_engineering/config_diff.json`。

seed42、CUDA、16 episodes×500 slots=8000 transitions，保存 step_4000.pt 和 final.pt；预计 31 次 PPO update，保留原丢弃最后不完整 rollout 规则。新 run root 为 `logs/development/e14e_p1a_task_event_credit_seed42_8k`。

代码基线 commit 为 `eb1140cefc4a0ff79b8bd0c05e73c7a23528d7ba`，工作区含此前实验及本轮未提交修改；不能只凭 commit 重建本轮源码，应保留 engineering integrity manifest 的文件 SHA-256。selector 默认值不进入 canonical identity，旧 E14d/P0a hash 不变；无新增网络参数或 optimizer group，真实旧 checkpoint 可严格加载。新 fixture checkpoint 验证 selector、normalization buffers 与非空 Adam 状态往返。

回滚机制：使用旧 P0a 配置（默认 counterfactual_q）或在新的隔离实验配置中切回该 selector。P1a 从头训练，不 warm-start。禁止将旧 P0a periodic checkpoint 作为更改 credit 后的无缝续训；配置 hash 不同应被兼容校验拒绝。不要覆盖旧产物。

## 待本人执行的训练命令（本轮未执行）

```powershell
Set-Location 'D:\My_project_li'
$py = 'D:\Miniconda\python.exe'
$cfg = 'D:\My_project_li\experiments\cooperative-load\configs\e14e-p1a-task-event-credit-seed42-8k-dev.json'
& $py -B -X utf8 'D:\My_project_li\main.py' --mode rl --config $cfg
```

## 训练后待执行的 4k/8k eval1044（本轮未执行）

以下先按真实新 run 目录定位 checkpoint，不假定未来代码 commit 后缀；仅存在一个匹配 run 时才继续。两个 checkpoint 在本轮尚未生成。评估为 CPU、development checkpoint、500 slots、原 actor masked argmax，Q 不参与选动作。

```powershell
Set-Location 'D:\My_project_li'
$py = 'D:\Miniconda\python.exe'
$cfg = 'D:\My_project_li\experiments\cooperative-load\configs\e14e-p1a-task-event-credit-seed42-8k-dev.json'
$root = 'D:\My_project_li\logs\development\e14e_p1a_task_event_credit_seed42_8k'
$runs = @(Get-ChildItem -LiteralPath $root -Directory -Filter 'ca_gat_mappo__remote_stress_v2__seed-42__cfg-2f143a85c0e3__git-*')
if ($runs.Count -ne 1) { throw '必须明确定位唯一 P1a 训练 run；不能猜测 checkpoint' }
$run = $runs[0].FullName
$cp4 = Join-Path $run 'checkpoints\step_4000.pt'
$cp8 = Join-Path $run 'checkpoints\final.pt'
if (!(Test-Path -LiteralPath $cp4) -or !(Test-Path -LiteralPath $cp8)) { throw '4k/final checkpoint 不完整' }
& $py -B -X utf8 -m src.evaluation.service_diagnostics --config $cfg --evaluate-from $cp4 --evaluation-seed 1044 --evaluation-device cpu --development-checkpoint --logs-dir (Join-Path $root 'eval1044_4k')
& $py -B -X utf8 -m src.evaluation.service_diagnostics --config $cfg --evaluate-from $cp8 --evaluation-seed 1044 --evaluation-device cpu --development-checkpoint --logs-dir (Join-Path $root 'eval1044_8k')
```

## 后续判断

先检查四组件 credit、Defer 正负 target、Task/CF 的实际梯度、Q chosen calibration 和 task baseline。再比较 4k→8k 的 Local sampled/argmax（区分 Remote 合法性）、Defer、UNBOUND expired、DONE、CPU service coverage 和能耗。机制接通不等于策略有效；不得提前声称 P1a 优于 P0a、Local 恢复或 Q 已正确。

## 最终工程验收结果

**P1a engineering gate: PASS**。

最终执行：**182 passed、82 subtests passed、0 failed、0 errors、0 skipped，75.21 秒**。其中新增 P1a 30 项；CUDA 3 项通过（P1a Q/task-value 独立 optimizer 测试 1 项，既有 Q 输入前向/反向 2 项）。未执行包含环境 rollout 的完整项目套件；全局构造禁令保护保证选中测试不能创建环境。

证据：`logs/development/e14e_p1a_engineering/tensor_regression_accepted.xml`、`test_selection_accepted.json`。覆盖原 P0a 数值路径、E14c attribution、CF 不进入 actor、Q/task-value CPU/CUDA 更新、四 epoch target/advantage 冻结、TX/CPU surrogate 隔离、7 类 CPU gate 保存张量对照、旧真实 checkpoint 严格加载、新 selector/buffers/非空 Adam 状态往返、有限数值及配置 diff。

首轮新测试夹具出现 4 项失败；修正接口/元数据后，后续暴露 paired-offset 夹具与不成立的“全局裁剪后其他分支权重逐位相同”断言，已修正为准确的隔离要求。第一次扩大回归为 175 passed、13 setup errors：13 个旧 CPU 测试初始化尝试构造环境，被禁止构造的保护拦截，**没有执行环境 rollout**；因此改用保存张量定向测试。新 CPU 夹具首次 7 项失败（未 finalize 的 buffer 取 sequence），修正为直接构造保存 tensor chunk 后 7 项全部通过。各轮 XML 原样保留，不覆盖失败记录，最终通过结果单独命名为 accepted。

上述是工程验收，不是实验有效性结论。P1a 已完成工程实现，可以进入受控开发训练；本轮没有执行该训练。
