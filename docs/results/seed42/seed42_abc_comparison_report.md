# Step 07A：seed42 A/B/C 结果汇总与历史 P1a 对齐审计
整理说明（2026-10-13）：本报告仅更新文件链接；数值、方法与结论未改。原始来源JSON中的旧路径通过 [路径对照表](../../../metadata/DIRECTORY_RELOCATION.json) 查找现址，相关取证JSON集中在 [metadata/audits/seed42](../../../metadata/audits/seed42)。

审计日期：2026-10-12。范围：已存在的 seed42 / 8k 训练产物、final 冻结策略评估、Route diagnostic，以及历史 P1a 的 1044 开发服务诊断。本轮未启动训练、episode evaluation、oracle 或预训练，未修改生产代码或实验配置。

## 组会速览
| 方法 | DONE↑ | 完成率↑ | EXPIRED↓ | Reward↑ | Energy↓ (J) | Rejection↓ |
| --- | --- | --- | --- | --- | --- | --- |
| A | 79.4 | 65.4357% | 33.2 | 37.2907 | 34.8243 | 11 |
| B | 63.4 | 52.4191% | 49 | 2.4266 | 25.5296 | 0.8 |
| C | 65.8 | 53.8175% | 46.6 | 7.7641 | 29.8235 | 3.2 |

在当前单一training seed42的五个开发评估seed上，A的DONE、完成率和Reward均值最高，A/B/C排序为 **A > C > B**。C相比B有均值恢复，但尚未超过A；恢复并非每个评估seed都出现。当前A在1044上再次出现Local=0、Remote=117、Defer=0，与从历史原始产物重新读取的结果一致。

**身份与取证限制先行披露：** A/B/C已保存的预算、配置、checkpoint和evaluation身份没有发现冲突，可以按Step06协议在当前范围内比较。原始启动命令及初始权重摘要没有保存；配置、initial_policy_version=0和无预训练/历史权重来源字段支持既定的从零协议，但仅凭final不能绝对证明整个初始化历史。历史评估明确标注 `development_service_diagnostic`、`paper_formal_result=false`；它是保存的真实开发诊断，不能改称论文正式结果。历史与当前的专项比较严格限定1044对1044。

## 1. 方法定义与可比性
| 组别 | Route结构 | 输入特征 | 学习链 | 实验作用 |
| --- | --- | --- | --- | --- |
| A | 原P1a分层：Local/Remote/Defer，再选Remote destination | 原候选特征 | 原P1a | baseline |
| B | Local、每个合法Remote、Defer统一leaf比较 | 原候选特征 | 保留P1a task-event/PPO/Q | 结构消融 |
| C | 同B | 加35维physical_relations_v1 | 同B | 特征消融 |

定义依据：[Step06正式协议](../../../sections/history/task_allocation_experiment_handoff_step06.md)。A是本协议安排的从零P1a baseline，不是另一个新算法；实际启动来源的证据限制见上。B/C只引入协议允许的结构/特征差异。三组environment、reward、动作约束和学习链公共配置一致。

| 核验项 | A | B | C |
| --- | --- | --- | --- |
| training seed / scenario | 42 / remote_stress_v2 | 42 / remote_stress_v2 | 42 / remote_stress_v2 |
| collected / optimized / tail | 8000 / 7936 / 64 | 8000 / 7936 / 64 | 8000 / 7936 / 64 |
| episodes × slots / PPO updates | 16 × 500 / 31 | 16 × 500 / 31 | 16 × 500 / 31 |
| initial / final policy version | 0 / 31 | 0 / 31 | 0 / 31 |
| checkpoint kind | FINAL_COMPLETED | FINAL_COMPLETED | FINAL_COMPLETED |
| eval seeds | 1042,1043,1044,1045,1046 | 1042,1043,1044,1045,1046 | 1042,1043,1044,1045,1046 |
| eval模式 / device / dtype | deterministic masked argmax / CPU / float32 | deterministic masked argmax / CPU / float32 | deterministic masked argmax / CPU / float32 |
| 训练device | CUDA / NVIDIA GeForce MX450 | CUDA / NVIDIA GeForce MX450 | CUDA / NVIDIA GeForce MX450 |
| 训练gate | smoke pass / signal-watch | smoke pass / signal-watch | smoke pass / signal-watch |
| 正式评估配置hash | d2463e1929ea867212edec4774f54b2c5a6dbc9ef957f00dd2978402aec4849a | d2463e1929ea867212edec4774f54b2c5a6dbc9ef957f00dd2978402aec4849a | d2463e1929ea867212edec4774f54b2c5a6dbc9ef957f00dd2978402aec4849a |

评估使用各run的 `checkpoints/final.pt`，没有根据中途表现选择checkpoint。runtime记录为Python 3.13.11、torch 2.7.1+cu118、CUDA 11.8。Git HEAD相同但工作区dirty；共同HEAD本身不能替代源码身份。当前106个源码/入口/依赖清单文件SHA与Step06清单一致，并在本轮前后保持一致。Step06 manifest的 `formal_training_started=false` 是配置发布时的状态，不能用来否认后来已生成的训练产物。

### A 身份
- config：[experiments/task-allocation/configs/a-seed42-8k.json](../../../experiments/task-allocation/configs/a-seed42-8k.json)
- config hash：`81cb1ad596cb6dcd6f212e2d08a7edc50e9b210b0a5cf226ba4a7a2cb1651ae5`
- run_id：`ca_gat_mappo__remote_stress_v2__seed-42__cfg-81cb1ad596cb__git-eb1140cefc4a`
- checkpoint：[logs/A/seed42/8k/training/checkpoints/final.pt](../../../logs/A/seed42/8k/training/checkpoints/final.pt)
- checkpoint SHA256：`5da08bb0309948ab8bbd9423364d6f76b07daf1a8dcdb7786c7dee141f958199`
- evaluation_run_id：`formal-eval__remote_stress_v2__chk-5da08bb03099__cfg-d2463e1929ea__eval-1f1e83d20fc9`
- evaluation目录：[logs/A/seed42/8k/evaluation](../../../logs/A/seed42/8k/evaluation)
- Git/source：`eb1140cefc4a0ff79b8bd0c05e73c7a23528d7ba`；dirty=true；完整架构、snapshot与tensor spec见 `checkpoint_identity_audit.json`。

### B 身份
- config：[experiments/task-allocation/configs/b-seed42-8k.json](../../../experiments/task-allocation/configs/b-seed42-8k.json)
- config hash：`1bd1c1aa959695e150ef3734a625f5d87de87e4f4325d470f6ea3845e8cdfda4`
- run_id：`ca_gat_mappo__remote_stress_v2__seed-42__cfg-1bd1c1aa9596__git-eb1140cefc4a`
- checkpoint：[logs/B/seed42/8k/training/checkpoints/final.pt](../../../logs/B/seed42/8k/training/checkpoints/final.pt)
- checkpoint SHA256：`2cab254ff8d527726779b7ec18383ffc0cd7710f5edb6d0ac229da2c554feec5`
- evaluation_run_id：`formal-eval__remote_stress_v2__chk-2cab254ff8d5__cfg-d2463e1929ea__eval-a4b04e518632`
- evaluation目录：[logs/B/seed42/8k/evaluation](../../../logs/B/seed42/8k/evaluation)
- Git/source：`eb1140cefc4a0ff79b8bd0c05e73c7a23528d7ba`；dirty=true；完整架构、snapshot与tensor spec见 `checkpoint_identity_audit.json`。

### C 身份
- config：[experiments/task-allocation/configs/c-seed42-8k.json](../../../experiments/task-allocation/configs/c-seed42-8k.json)
- config hash：`9e76466588b68a6002d24d80f5e1424f370f26e97f8d4854c392f41da682ad15`
- run_id：`ca_gat_mappo__remote_stress_v2__seed-42__cfg-9e76466588b6__git-eb1140cefc4a`
- checkpoint：[logs/C/seed42/8k/training/checkpoints/final.pt](../../../logs/C/seed42/8k/training/checkpoints/final.pt)
- checkpoint SHA256：`58da11c49bde4a517471e1386842b90568fdb1ffe51d9c290caa7e3d442ff1b5`
- evaluation_run_id：`formal-eval__remote_stress_v2__chk-58da11c49bde__cfg-d2463e1929ea__eval-e866f9e0ccb7`
- evaluation目录：[logs/C/seed42/8k/evaluation](../../../logs/C/seed42/8k/evaluation)
- Git/source：`eb1140cefc4a0ff79b8bd0c05e73c7a23528d7ba`；dirty=true；完整架构、snapshot与tensor spec见 `checkpoint_identity_audit.json`。

## 2. 训练阶段总表
| 指标 | 单位 | A | B | C |
| --- | --- | --- | --- | --- |
| Collected transitions | transitions | 8000 | 8000 | 8000 |
| Optimized transitions | transitions | 7936 | 7936 | 7936 |
| Unused tail | transitions | 64 | 64 | 64 |
| PPO updates | updates | 31 | 31 | 31 |
| Episodes | episodes | 16 | 16 | 16 |
| Training reward_mean（全8000步） | dimensionless | -0.1642 | -0.1526 | -0.1574 |
| Smoke gate | status | pass | pass | pass |
| Signal gate | status | signal-watch | signal-watch | signal-watch |
| all16平均episode_reward | dimensionless | -82.1064 | -76.313 | -78.6985 |
| first8平均episode_reward | dimensionless | -79.3264 | -77.8269 | -83.3063 |
| last8平均episode_reward | dimensionless | -84.8865 | -74.7992 | -74.0906 |
| all16平均completion_count | tasks | 22.375 | 25.125 | 24 |
| first8平均completion_count | tasks | 23.75 | 24.5 | 21.875 |
| last8平均completion_count | tasks | 21 | 25.75 | 26.125 |
| all16平均expiration_count | tasks | 90.375 | 87.6875 | 88.9375 |
| first8平均expiration_count | tasks | 89.375 | 88.75 | 91.375 |
| last8平均expiration_count | tasks | 91.375 | 86.625 | 86.5 |
| 最高单episode completion | tasks | 36 | 42 | 36 |
| 最低单episode completion | tasks | 10 | 13 | 15 |
| 最好episode reward | dimensionless | -42.8762 | -29.5671 | -40.2592 |
| 最差episode reward | dimensionless | -109.9829 | -103.2478 | -102.0234 |
| final_actor_loss | dimensionless | 0.2964 | 0.2219 | 0.2031 |
| all124_mean_actor_loss | dimensionless | 0.259 | 0.3167 | 0.3325 |
| final_critic_loss | dimensionless | 0.2665 | 0.2651 | 0.2626 |
| all124_mean_critic_loss | dimensionless | 0.3917 | 0.3998 | 0.4068 |
| final_entropy | dimensionless | 1.5449 | 1.7405 | 1.6437 |
| all124_mean_entropy | dimensionless | 1.4914 | 1.5079 | 1.526 |
| 末episode reward_mean | dimensionless | -0.1825 | -0.1715 | -0.1666 |

`training_reward_mean` 取正式aggregate的 `reward.mean_per_transition`，覆盖全部8000 transitions；末episode的reward_mean另列。all16/first8/last8分别使用episode_index 0—15/0—7/8—15。final loss/entropy指最后一个PPO epoch（update_index=30、ppo_epoch_index=3、series_index=124）；124个epoch的均值另列，避免将全程均值误称末值。原raw与dashboard CSV的episode/epoch指标已交叉核对。

**训练episode指标只反映训练过程，不用于直接宣布最终A/B/C胜负。** B全程平均完成数25.125高于C的24和A的22.375；A后8个episode的完成数和reward均比前8个差，B/C则有所改善。这些训练采样过程数据与冻结确定性策略评估排序不同，不能相互替代。smoke pass只反映执行/信号工程检查；signal-watch应保留，不能表述为学习效果验收通过。

## 3. Final evaluation 核心表
| 指标 | 单位 | A | B | C | 最优 |
| --- | --- | --- | --- | --- | --- |
| Generated | tasks | 121 | 121 | 121 | 相同 |
| DONE | tasks | 79.4 | 63.4 | 65.8 | A |
| Completion Ratio | percent | 65.4357% | 52.4191% | 53.8175% | A |
| EXPIRED | tasks | 33.2 | 49 | 46.6 | A |
| Expiration Ratio | percent | 27.8508% | 40.6736% | 39.2589% | A |
| TRUNCATED | tasks | 8.4 | 8.6 | 8.6 | A |
| Truncation Ratio | percent | 6.7135% | 6.9072% | 6.9236% | A |
| Total Reward | dimensionless | 37.2907 | 2.4266 | 7.7641 | A |
| Completed Task E2E Delay | s | 0.4111 | 0.424 | 0.43 | A |
| Total Energy | J | 34.8243 | 25.5296 | 29.8235 | B |
| CPU Energy | J | 34.3486 | 25.4418 | 29.4468 | B |
| TX Energy | J | 0.4757 | 0.0878 | 0.3766 | B |
| Energy per Completed Task | J/task | 0.4315 | 0.4053 | 0.4478 | B |
| Executor Rejection | events | 11 | 0.8 | 3.2 | B |
| Executor Downgrade | events | 0 | 0 | 0 | A/B/C |
| Backlog Final Tasks | tasks | 8.8 | 8.8 | 8.8 | A/B/C |
| Backlog Mean Tasks | tasks | 6.5132 | 7.1576 | 7.0912 | A |
| Backlog Max Tasks | tasks | 14.2 | 14.2 | 15 | A/B |
| Workload Penalty | dimensionless | 8.8927 | 11.9612 | 11.4217 | A |
| Outage Rate | percent | 0% | 0% | 0.177% | A/B |

主结果只取 `ca_gat_mappo`。除Completed Task E2E Delay外，本表为五个episode的算术均值（空值按正式口径排除）；完成率是每个episode比率的均值，不是pooled比率。Delay的正式aggregate按所有DONE任务汇总，样本数A=397、B=317、C=329，因此没有再对五个episode的delay简单平均。Energy per Completed Task是五个episode各自Energy/DONE的均值，不是合计Energy/合计DONE。能量为J，时延为s，比例统一显示百分比。各组完成任务集合不同，delay/每任务能量比较具有选择集合差异。

最优按用户指定方向逐指标标记，Generated相同不设优劣。B的能量和Rejection较低，同时DONE明显低于A；不能因此将B判为整体更好。Outage是已尝试链路上的条件指标，三组分母不同，零outage不等于全部任务都能及时远程完成。

## 4. Pooled 结果
| 指标 | A | B | C |
| --- | --- | --- | --- |
| generated | 605 | 605 | 605 |
| completed | 397 | 317 | 329 |
| expired | 166 | 245 | 233 |
| truncated | 42 | 43 | 43 |
| completion_ratio | 65.6198% | 52.3967% | 54.3802% |
| expiration_ratio | 27.438% | 40.4959% | 38.5124% |
| truncation_ratio | 6.9421% | 7.1074% | 7.1074% |

三组总Generated都是605。pooled完成率=五个episode合计DONE/合计Generated；每组均核验Generated=DONE+EXPIRED+TRUNCATED。A的主表完成率为65.4357%，pooled为65.6198%，两种权重不同，并非统计冲突。

## 5. A/B/C 差值
| 对比 | ΔDONE | Δ完成率 (pp) | ΔEXPIRED | ΔReward | ΔEnergy (J) | ΔRejection |
| --- | --- | --- | --- | --- | --- | --- |
| A->B | -16 | -13.0166 pp | +15.8 | -34.8641 | -9.2947 | -10.2 |
| B->C | +2.4 | +1.3984 pp | -2.4 | +5.3375 | +4.2939 | +2.4 |
| A->C | -13.6 | -11.6182 pp | +13.4 | -29.5266 | -5.0008 | -7.8 |

所有差值均为箭头后组减箭头前组；完成率差值为百分点，没有使用相对百分比。

## 6. 逐 evaluation seed
| Eval seed | Generated | A DONE | B DONE | C DONE | A完成率 | B完成率 | C完成率 | A Reward | B Reward | C Reward | DONE最高 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1042 | 135 | 100 | 81 | 104 | 74.0741% | 60% | 77.037% | 68.87 | 27.3192 | 77.6651 | C |
| 1043 | 122 | 74 | 56 | 71 | 60.6557% | 45.9016% | 58.1967% | 27.5485 | -13.0579 | 18.0832 | A |
| 1044 | 117 | 94 | 65 | 53 | 80.3419% | 55.5556% | 45.2991% | 69.6324 | 6.499 | -16.7163 | A |
| 1045 | 102 | 59 | 56 | 50 | 57.8431% | 54.902% | 49.0196% | 10.3793 | 3.598 | -9.4842 | A |
| 1046 | 129 | 70 | 59 | 51 | 54.2636% | 45.7364% | 39.5349% | 10.0232 | -12.2254 | -30.7273 | A |

| 方法 | DONE最高的seed数 |
| --- | --- |
| A | 4 |
| B | 0 |
| C | 1 |

A对B的DONE在五个seed上均更高；A对C在四个seed更高，1042上C比A多4个DONE。C−B的逐seed DONE差分别为 +23, +15, -12, -6, -8；均值恢复由1042和1043的正差贡献，1044—1046为负差。因此C每个场景都恢复的说法不符合现有表格。五个evaluation seed是同一训练实例的环境变化，不能当作五个独立training seed。

## 7. 固定基线一致性与完整排序
`local_only`、`heuristic`、`random` 的aggregate子树全部字段在A/B/C间完全一致。逐episode全部字段（仅排除必然不同的evaluation_run_id）也一致，包含action/external/hidden trace SHA、指标、有效样本计数及invariants；没有对不一致基线进行平均。

| 方法（按DONE降序） | DONE | Completion Ratio | EXPIRED | Reward | Energy (J) |
| --- | --- | --- | --- | --- | --- |
| A | 79.4 | 65.4357% | 33.2 | 37.2907 | 34.8243 |
| heuristic | 74.2 | 61.0377% | 38 | 26.0223 | 21.8334 |
| C | 65.8 | 53.8175% | 46.6 | 7.7641 | 29.8235 |
| local_only | 64.6 | 53.7248% | 48.6 | 4.8678 | 23.7326 |
| B | 63.4 | 52.4191% | 49 | 2.4266 | 25.5296 |
| random | 5.4 | 4.5377% | 105.4 | -118.5192 | 9.667 |

当前排序为A > Heuristic > C > Local-only > B > Random。A相对固定基线的这一排序仅适用于此实例和开发评估范围。

## 8. Route 实际选择与目的地
| 方法 | Route总数 | Local | Remote | Defer | Local% | Remote% | Defer% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 604 | 131 | 473 | 0 | 21.6887% | 78.3113% | 0% |
| B | 614 | 529 | 75 | 10 | 86.1564% | 12.215% | 1.6287% |
| C | 654 | 273 | 330 | 51 | 41.7431% | 50.4587% | 7.7982% |

分母是active Route决策次数，不是生成任务数。Defer可对同一任务重复发生；三组决策总数不必相同。这里Remote是实际选中并绑定的Route目的地，不等于同槽通信服务已经成功传输，也不等于完成任务的远程服务归因。不能使用 `executed_communication.receiver_uav` 代替当前Route目的地，该服务可能属于更早排队的任务。

| 方法 | Remote→UAV0 | Remote→UAV1 | Remote→UAV2 | Remote→UAV3 | 其他/NA |
| --- | --- | --- | --- | --- | --- |
| A | 152 | 120 | 129 | 72 | 0 |
| B | 14 | 15 | 39 | 7 | 0 |
| C | 103 | 63 | 121 | 43 | 0 |

逐行使用actual final_route_action_index、动作domain和selected_remote_id对齐真实节点ID；全部Remote均合法、非source=self，非法self排除数为0。未将candidate position当作UAV id。B/C还核验chosen leaf与legal deterministic leaf argmax一致；Defer的node=null、数组占位=-1均按无目的地解释。

| Eval seed | 方法 | Local | Remote | Defer | Route总数 |
| --- | --- | --- | --- | --- | --- |
| 1042 | A | 55 | 80 | 0 | 135 |
| 1042 | B | 124 | 11 | 2 | 137 |
| 1042 | C | 81 | 54 | 3 | 138 |
| 1043 | A | 0 | 121 | 0 | 121 |
| 1043 | B | 96 | 25 | 2 | 123 |
| 1043 | C | 24 | 97 | 0 | 121 |
| 1044 | A | 0 | 117 | 0 | 117 |
| 1044 | B | 92 | 25 | 0 | 117 |
| 1044 | C | 29 | 87 | 47 | 163 |
| 1045 | A | 39 | 63 | 0 | 102 |
| 1045 | B | 95 | 7 | 3 | 105 |
| 1045 | C | 61 | 41 | 0 | 102 |
| 1046 | A | 37 | 92 | 0 | 129 |
| 1046 | B | 122 | 7 | 3 | 132 |
| 1046 | C | 78 | 51 | 1 | 130 |

| 方法 | Local且存在合法Remote的次数 | 其余Local（无合法Remote） |
| --- | --- | --- |
| A | 0 | 131 |
| B | 398 | 131 |
| C | 142 | 131 |

A的131次Local均发生在没有合法Remote的决策中；在这份冻结策略trace中，有合法Remote时A全部选择Remote。B/C分别有398/142次在合法Remote存在时仍选Local。这描述现有mask与动作，不推断某个动作的反事实成功率，也不将无合法Remote推广为远程服务不可行的普遍物理结论。

## 9. Leaf argmax 与组概率质量
| 方法 | Local argmax% | Remote argmax% | Defer argmax% | mean P(Local) | mean P(Remote group) | mean P(Defer) |
| --- | --- | --- | --- | --- | --- | --- |
| A | NA | NA | NA | 26.0296% | 39.848% | 34.1224% |
| B | 86.1564% | 12.215% | 1.6287% | 42.3815% | 34.5228% | 23.0957% |
| C | 41.7431% | 50.4587% | 7.7982% | 39.2024% | 37.2793% | 23.5183% |

B/C的argmax是合法leaf中最大概率对应的单个动作；与实际chosen leaf逐行一致。Remote group是多个Remote leaf概率之和，不能将组质量最大解释为选中某个Remote leaf。平均概率按全部active Route决策平均，包括Remote不合法而组质量为0的行；均值不等于argmax频率。A使用原top-level/destination两层语义，leaf argmax列为NA；A的三项mean P是top-level概率，不伪造成flat leaf输出。

| 方法 | Local组质量argmax次数 | Remote组质量argmax次数 | Defer组质量argmax次数 | chosen与组质量argmax不同次数 |
| --- | --- | --- | --- | --- |
| A | NA | NA | NA | NA |
| B | 369 | 237 | 8 | 162 |
| C | 233 | 417 | 4 | 87 |

C的Remote组质量最大417次，实际Remote leaf选择330次；B分别237次与75次。组质量聚合与单叶比较属于不同统计口径，这一差异本身不是执行错误。

## 10. 历史 P1a-1044 与当前 A-1044
历史定位依据为Step01 baseline_identity中的P1a 8k项，再回到历史checkpoint、evaluation_manifest、episode_metrics和Route trace重新取数；用户给出的参考值未作为数据来源。历史保存产物的用途是development_service_diagnostic，paper_formal_result=false。

| 指标 | 单位 | 历史P1a-1044 | 当前A-1044 | 是否一致/相近 |
| --- | --- | --- | --- | --- |
| generated | tasks | 117 | 117 | 一致 |
| completed | tasks | 94 | 94 | 一致 |
| expired | tasks | 19 | 19 | 一致 |
| truncated | tasks | 4 | 4 | 一致 |
| completion_ratio | percent | 80.3419% | 80.3419% | 一致 |
| Local | decisions | 0 | 0 | 一致 |
| Remote | decisions | 117 | 117 | 一致 |
| Defer | decisions | 0 | 0 | 一致 |
| remote_ratio | percent | 100% | 100% | 一致 |
| total_reward | dimensionless | 69.6324 | 69.6324 | 一致 |

两边全部25个episode metrics字段逐项完全相同，action trace SHA和external trace SHA完全相同。117条Route非概率字段在排除evaluation_run_id、episode_id后一致。概率日志并非逐位相同：remote_probability 54行, remote_destination_probabilities 55行, local_probability 12行, defer_probability 13行；最大绝对差为8.94069671631e-08，小于1e-6。现有产物不足以唯一解释旧/新概率记录微差来源；不将它扩大为行为或性能差异。

### 历史与当前身份区别
**历史P1a**

- checkpoint：[metadata/reference/p1a_seed42_1044/training/checkpoints/final.pt](../../../metadata/reference/p1a_seed42_1044/training/checkpoints/final.pt)
- checkpoint SHA256：`d303878ed57dcfca0ee9d04354ab678046fbf22b3546aef38a853b53bae7a062`
- config：[metadata/reference/p1a_seed42_1044/original_config.json](../../../metadata/reference/p1a_seed42_1044/original_config.json)
- config hash：`2f143a85c0e37f605bbf38b59b3a31267a305464179593d9828c7c0a1518e3fa`
- run_id：`ca_gat_mappo__remote_stress_v2__seed-42__cfg-2f143a85c0e3__git-eb1140cefc4a`
- evaluation_run_id：`development-service-eval__remote_stress_v2__chk-d303878ed57d__cfg-2e7a1dd6b92d__eval-302b95721c5e`
- Git HEAD：`eb1140cefc4a0ff79b8bd0c05e73c7a23528d7ba`；seed42；8000 collected / 7936 optimized / 64 tail；16×500；31 PPO updates。

**当前A**

- checkpoint：[logs/A/seed42/8k/training/checkpoints/final.pt](../../../logs/A/seed42/8k/training/checkpoints/final.pt)
- checkpoint SHA256：`5da08bb0309948ab8bbd9423364d6f76b07daf1a8dcdb7786c7dee141f958199`
- config：[experiments/task-allocation/configs/a-seed42-8k.json](../../../experiments/task-allocation/configs/a-seed42-8k.json)
- config hash：`81cb1ad596cb6dcd6f212e2d08a7edc50e9b210b0a5cf226ba4a7a2cb1651ae5`
- run_id：`ca_gat_mappo__remote_stress_v2__seed-42__cfg-81cb1ad596cb__git-eb1140cefc4a`
- evaluation_run_id：`formal-eval__remote_stress_v2__chk-5da08bb03099__cfg-d2463e1929ea__eval-1f1e83d20fc9`
- Git HEAD：`eb1140cefc4a0ff79b8bd0c05e73c7a23528d7ba`；seed42；8000 collected / 7936 optimized / 64 tail；16×500；31 PPO updates。

| 身份层次 | 核验结果 |
| --- | --- |
| 方法定义 | 相同P1a：分层Route、task-event信用、原公共学习链 |
| checkpoint文件 | 路径不同，SHA不同，不是同一文件 |
| 训练config | hash不同；环境/reward/架构定义相同；输出位置差异保留 |
| evaluation config hash（历史） | 2e7a1dd6b92d34ac40b02349c8fcb47ef2a90eb2d0e55018647ceeba137ae88a |
| evaluation config hash（当前） | d2463e1929ea867212edec4774f54b2c5a6dbc9ef957f00dd2978402aec4849a |
| eval口径 | 历史只1044且为开发服务诊断；当前1042—1046 formal wrapper；专项只对1044 |
| 架构 | checkpoint配置一致；旧manifest缺省destination_decoder_mode=legacy，不能据缺省键误报不同 |
| source边界 | 共同dirty Git HEAD不能证明旧工作树全部内容相同；当前源码与Step06清单相同 |
| 随机初始化取证 | 配置与初始policy version支持从零协议；缺少启动命令/初始权重摘要，无法绝对复原来源 |

| 模型部分 | 历史tensor数 | 当前tensor数 | 历史/当前元素数 | 名称集合 | shape/dtype | 逐tensor完全相同 |
| --- | --- | --- | --- | --- | --- | --- |
| actor | 158 | 158 | 522462/522462 | 相同 | 相同 | 158/158 |
| critic | 21 | 21 | 1009873/1009873 | 相同 | 相同 | 21/21 |

离线比较只读CPU加载checkpoint并检查actor/critic state_dict，没有创建或运行episode。元素数是全部state_dict张量元素数，不额外声称均为trainable parameters。actor 158/158、critic 21/21张量逐位相同，没有不同张量。**方法相同不等于checkpoint文件相同；本例文件不同，但学习到的actor/critic状态相同。** 不能把这两个保存实例当作统计上独立的不同训练重复，也不能仅凭相同张量推断复制历史权重或存在bug。

历史单seed的DONE=94只与当前A-1044的94比较；未与当前A五seed均值79.4混比。

## 11. 关键回答
**Q1：当前A仍是Remote-dominant吗？** 是。五seed合并Route中Remote=473/604（78.3113%）、Local=131/604（21.6887%）、Defer=0。它不是所有seed都Local=0。

**Q2：是否接近历史Local=0现象？** 1044上当前A为Local0/Remote117/Defer0，Remote比例100%；历史原产物也是相同计数。1043也Local0，但不能推广到全部评估seed。

**Q3：DONE差异是否伴随Remote行为差异？** 是。A使用Remote最多、DONE均值最高；B更偏Local，C相比B增加Remote并出现更多Defer。A在有合法Remote时全选Remote，B/C仍有主动Local选择。这是当前实例的行为相关性，尚不能断言Remote造成DONE优势；任务可行性、服务排队、信用/动作排序及训练随机性仍可能共同作用。

**Q4：Remote-dominant一定正确吗？** 不能这样判断。在当前remote_stress_v2、seed42训练实例和1042—1046开发评估上，A的Remote-dominant与最高DONE同时出现，说明Local=0本身不能被自动判为策略失败。合理性仍需结合任务级可行性、资源负担和多training-seed证据；不设Local比例达标目标。

## 12. 分层阶段结论
1. **训练阶段：** B全程平均完成数最高，B/C后8个episode较前8个改善，A则下降；训练采样过程不能代替final确定性策略评估。
2. **Final evaluation：** 冻结策略A/B/C排序A>C>B；A在4/5个开发评估seed的DONE最高。完整固定基线排序A>Heuristic>C>Local-only>B>Random。
3. **A→B：** DONE减少16，完成率下降13.0166pp；当前seed42不支持统一候选结构提高完成性能。
4. **B→C：** DONE均值增加2.4，完成率恢复1.3984pp；physical relations伴随正向均值恢复，但1044—1046仍退化，不能表述为普遍改善。
5. **A→C：** C的DONE均值比A少13.6，完成率低11.6182pp；完整C尚未超过当前A。
6. **历史P1a→当前A：** 方法/架构相同、checkpoint文件不同，但actor/critic张量相同；1044性能指标及实际Route行为重现，概率记录仅有极小浮点差。历史仍保留开发诊断身份。

结论限定为一个training seed的开发比较。未做多训练seed不确定性估计，未用未见确认集，未根据本轮结果调整方法。

## 13. 后续待诊断问题（本轮不执行）
- B在存在合法Remote时大量选择Local，DONE比A低；需任务级可行性、候选叶排序与信用证据区分合理本地选择和服务机会损失，不能凭占比直接改reward或强制Remote。
- C的均值恢复集中在1042/1043，1044出现47次Defer且DONE降至53；需核查重复决策对应的任务和期限状态，不能将47次直接当47个不同任务。
- A的Local仅出现在没有合法Remote时；以后获授权的诊断需区分mask限制与策略偏好并联系真实资源/完成事件。当前未运行反事实或oracle。
- A的能量和Rejection高于B/C；需在完成任务优先目标下审视资源代价。零Outage未覆盖所有未选择/未及时服务的任务。
- 从零来源缺少启动命令和初始权重摘要；后续另行授权的工程步骤可补足记录。本轮不更改记录系统，也不把配置证据夸大为完整来源证明。
- 历史与当前A概率日志微差、旧dirty源码完整快照不可重建，以及同张量状态不构成独立重复，均需保留审计边界。

## 14. 来源、验证与交付
全部49项输入逐个记录路径、SHA256、字节数和用途，读取前后hash一致。`source_manifest.json` 的446项关键数字/身份条目记录run_id、checkpoint SHA、evaluation_run_id、seed、字段选择和计算公式；身份与源码清单见 `checkpoint_identity_audit.json`。

验证包括：训练raw/dashboard一致；eval JSONL/CSV一致；aggregate均值与pooled从episode重算一致；任务守恒、能量分解和reward组件等式成立；同seed Generated与外生轨迹一致；固定基线逐字段一致；Route summary与trace重算一致；目的地非self/合法且真实ID一致；B/C chosen leaf等于合法leaf argmax；历史1044全部metrics/动作一致且单独记录概率微差。CSV百分比为数值百分比、差值为pp；保留完整精度，显示表仅四舍五入。

- [seed42_abc_comparison_report.md](seed42_abc_comparison_report.md)
- [seed42_abc_training_summary.csv](seed42_abc_training_summary.csv)
- [seed42_abc_evaluation_summary.csv](seed42_abc_evaluation_summary.csv)
- [seed42_abc_per_eval_seed.csv](seed42_abc_per_eval_seed.csv)
- [seed42_abc_route_behavior.csv](seed42_abc_route_behavior.csv)
- [seed42_abc_remote_destinations.csv](seed42_abc_remote_destinations.csv)
- [seed42_abc_deltas.csv](seed42_abc_deltas.csv)
- [historical_p1a_vs_current_a_1044.csv](historical_p1a_vs_current_a_1044.csv)
- [source_manifest.json](../../../metadata/audits/seed42/source_manifest.json)
- [validation.json](../../../metadata/audits/seed42/validation.json)
- [checkpoint_identity_audit.json](../../../metadata/audits/seed42/checkpoint_identity_audit.json)
- [analysis_data.json](../../../metadata/audits/seed42/analysis_data.json)
- [summary_tables.json](../../../metadata/audits/seed42/summary_tables.json)

只读审计完成后停止。未进入原因修复，未训练seed43/44，未启动新实验。
