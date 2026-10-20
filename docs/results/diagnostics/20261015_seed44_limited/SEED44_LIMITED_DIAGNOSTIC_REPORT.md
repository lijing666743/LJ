# seed44 第一步限定诊断报告

日期：2026-10-15。状态：D1—D4 离线诊断已实际执行；未启动训练、新环境评估、最终测试或网络更新。未修改 reward、生产环境、配置和模型。本轮新增的是已有真实日志的重建与分析，不是新实验成绩。

## 当前结论

后期回落已定位到明确的轨迹行为：部分原本完成的任务在36k反复选择Defer，随后未绑定即过期，同时造成其他未绑定任务等待。CPU高能耗集中在高频服务及特定执行节点；不能主要解释为对失败任务白算。尚未证明这些行为由哪项学习机制导致，也没有证明替换动作能无损恢复完成率。

## 实际执行范围

Route覆盖10个已保存检查点评估，每个605任务，重建每槽UNBOUND EDF队首。CPU覆盖11种保存条件、55个历史评估回合，逐项核对110000个CPU槽、6655份生命周期记录。相同任务在不同条件下重复核对，不是新增独立训练。
seed44使用20k、32k、36k、40k；seed42/43使用32k、36k、40k作为描述性控制，场景均为1042—1046。固定Heuristic取自相同冻结评估；它本身没有training seed或checkpoint。训练信号读取三个种子的Dashboard与所需更新日志，并仅读取final checkpoint中的optimizer学习率，不实例化策略或环境。

## D1/D2：任务损失与Route/Defer

32k→36k失去86个原本完成的任务，新增完成18个，净变化-68。失去完成任务中49个为EXPIRED/UNBOUND，另23个EXPIRED/CPU_SERVICE、13个EXPIRED/CPU_WAIT、1个TRUNCATED/CPU_WAIT。
49个未绑定过期任务中，48个实际成为队首并反复Defer，1个始终未成为队首。前48个累计1110次Defer，Local在所有机会均合法，78次另有合法Remote。不能把唯一未获机会任务本人记为Defer。
这49个任务在32k原先有47个绑定Local、2个绑定Remote。36k累计还有1002个task-slots排在其他EDF队首后，这些队首当时也实际Defer。这里是任务等待槽数，不是互不重叠的全局时隙数。
48个获得机会任务首次决策的inclusive slack中位数为22.5槽，范围2—100槽；slack采用deadline-slot+1。绑定在槽末生效，因此slack还不能直接当作后续可服务槽数或完成保证。

各检查点Route事实：

- C44_20k：active 604次，Local 169、Remote 435、Defer 0次。
- C44_32k：active 1355次，Local 159、Remote 424、Defer 772次。
- C44_36k：active 1950次，Local 87、Remote 431、Defer 1432次。
- C44_40k：active 1486次，Local 145、Remote 418、Defer 923次。

全部实际选择与合法叶argmax一致，EDF队首也与生命周期重建一致；未发现这部分存在错误索引或把Remote组质量当作执行叶动作的证据。该核验只覆盖已读取的轨迹。

## D3：CPU能耗与服务

40k相对Heuristic多消耗总能量67.913 J，其中CPU增加62.474 J，占91.99%。按各自任务终局记账，DONE增加64.932 J，EXPIRED变化-2.070 J，TRUNCATED变化-0.388 J。因此额外CPU成本主要记在完成任务上，而非失败任务上。两策略完成的任务集合不同，这个分类不是相同任务的因果分解。
进一步配对双方都完成的286个相同任务，其实际完成cycles均为150.826×10^9，但C44与Heuristic的CPU能耗分别为121.080 J、61.698 J。这排除了该子集仅由工作量不同解释能耗差，但仍不能分离目的地、频率、时序和资源异构各自的贡献。
C44满频正服务占5537/5547（99.82%）；Heuristic为5434/8370（64.92%）。C44总服务量234.729×10^9 cycles，低于Heuristic的263.821×10^9。频率分母只包含正CPU服务槽，不包括idle。
UAV2是能耗集中点：C44 CPU 111.598 J，Heuristic 43.874 J。节点编号沿用代码0—3；这是五场景之和。
32k→36k UAV3正服务槽由1630降至683。36k只有4个有积压且局部最低CPU能量可行、却无服务的节点槽，另305个积压无服务槽的所有源队首均不满足最低CPU能量需求。40k局部可行无服务槽为230。因此36k低点不支持用大量已入CPU后的局部可行闲置作主要解释。
局部最低CPU能量检查未包含所有联合TX/CPU仲裁条件，不等于实际联合可执行性。遥测没有CPU proposal、完整合法mask和scheduler scores，不能由满频执行断定actor错误、门控错误或降频可无损。

## D4：后期训练信号

窗口按更新结束步数划分为(28k,32k]、(32k,36k]、(36k,40k]，分别16、15、16次更新，每次4个epoch；四个epoch不当作四倍独立训练样本。

- 28→32k：训练Route Defer 393/1394（28.19%），Route KL 0.0002151，联合CPU KL 0.0008518，联合CPU clip 1.136%，critic half-MSE 0.3841。
- 32→36k：训练Route Defer 1205/2145（56.18%），Route KL 0.0001501，联合CPU KL 0.0009588，联合CPU clip 1.489%，critic half-MSE 0.3808。
- 36→40k：训练Route Defer 634/1594（39.77%），Route KL 0.0000735，联合CPU KL 0.0012322，联合CPU clip 1.668%，critic half-MSE 0.3107。

32k→36k训练Defer比例明显上升，与开发轨迹方向一致，但Route KL没有相应暴增、Route clip在这些窗口均为0。36k→40k回升时联合CPU KL/clip反而更高，不能把这些标量单独作为36k低点根因。seed43末窗口训练Defer也升至48.27%，现象并非只在seed44存在。
最终checkpoint中三种子实际actor学习率组分别约3e-5、3e-4，critic为3e-4；历史每次optimizer真实LR没有独立日志，保留NA。CSV alpha来自配置，不冒充实时optimizer读数。晚窗口探索barrier实际日志全0，Route entropy系数0.01不变；这不是在32k新发生的退火事件。
本配置使用joint CPU调度；cpu_queue兼容字段承载联合log-prob，cpu_frequency的log-prob是零占位。后者零KL不表示频率策略没变化。critic_grad_norm仅覆盖value network，辅助Route value/Q的历史梯度范数未记录；Route/shared-trunk梯度来自联合actor loss，不能当独立分支梯度冲突证据。Route Q辅助未直接作为当前actor的counterfactual信用。
三个完整CSV的8个重复列逐行值一致；seed42续训重放交叠16个更新的PPO记录一致，窗口没有双计。所读取字段与辅助loss均未发现NaN/Inf，但没有扫描每个历史tensor，不能声称训练全状态已排除数值问题。

## 下一步与边界

建议下一步做小规模冻结组合干预：在已见1042—1046场景，对C44的32k/36k比较原策略、只替换Route、只替换非Route服务，以及共同固定服务条件。先验证动作顺序、mask、循环状态、proposal/execution和外生轨迹，再执行有限开发评估。这样检验绑定行为与CPU服务之间的影响，不立即改reward或追加训练预算。具体组合和成本应在实施前固定。
本轮D5干预未执行，也未实现MTS/QMIX或扩展训练。当前能确认行为与成本位置，不能宣布学习根因已查清。若下一轮干预没有形成明确可归因收益，就保留C现版本，推进同预算消融与强基线，不无限调参。

## 审计、重建与已知修正

正式Route证据使用limited_route_diagnostic.v2。首版曾把slack写为deadline-slot，已修正为inclusive定义并补测试；旧首版保留为superseded，不能作为当前正式slack证据。分析初次测试的临时目录权限问题已通过工作区隔离临时目录解决。
原始诊断与统一验证入口：

- `metadata/audits/seed44_limited_diagnostic/20261015_route_v2/route_diagnostic.json`
- `metadata/audits/seed44_limited_diagnostic/20261015/cpu_diagnostic.json`
- `metadata/audits/seed44_limited_diagnostic/20261015/training_diagnostic.json`
- `metadata/audits/seed44_limited_diagnostic/20261015/diagnostic_summary.json`
- `metadata/audits/seed44_limited_diagnostic/20261015/execution_validation.json`

重建须使用新的输出路径，防止覆盖。本轮实际命令、stdout、JUnit、前后指纹与重建一致性记录保存在同一审计目录；最终验收数量以execution_validation.json为准。
