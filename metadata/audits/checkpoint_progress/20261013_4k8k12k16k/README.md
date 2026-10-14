# A/B/C seed42：4k→8k→12k→16k 固定开发场景诊断

checkpoint审计与功能评估协议身份均为PASS。只新增六组12k/16k frozen deterministic evaluation，直接复用既有4k/8k。未训练、未resume、未更新网络，未使用oracle、pretraining或held-out seeds20042–20061。

结论：C仍改善；A在8k后下降、16k部分恢复但未回到8k；B在12k改善后于16k回落。当前没有足够依据将任一方法判断为经验性能平台，更不能宣布收敛。16k仍是预定final，没有选择或重新定义12k为最终模型。

## 1. Checkpoint与协议

实际路径和内容解析见checkpoint_inventory.json。三组training seed=42、scenario=remote_stress_v2。A为P1a hierarchical Route；B为unified leaf+original option features；C为unified leaf+physical_relations_v1。12k为PERIODIC_RESUME，16k为FINAL_COMPLETED。三组8k的verified_exact记录已持久化，本轮只读再次比较actor、critic、两个Adam、策略RNG、诊断历史及计数与原8k final一致；未伪造终端stdout。

权威协议来自原4k/8k实际审计、manifest与config：eval seeds1042–1046、500步、CPU float32、masked argmax、update_network=false、同一四方法suite与随机流。六组新评估协议哈希均为 d2463e1929ea867212edec4774f54b2c5a6dbc9ef957f00dd2978402aec4849a。

同一eval seed的原始外生trace跨所有方法和checkpoint一致。相同action-mask实现和语义不要求不同策略状态下的每步mask数组相同；动作导致的队列和干扰变化不被强制抹平。

历史元数据区别已披露：旧8k入口保留requires_grad=True，但actor.eval()+torch.no_grad()，runner验证前后权重一致；旧4k与本轮12k/16k诊断入口显式requires_grad=False，同样no_grad推理。该标志不是逐字一致的元数据，功能推理协议哈希、动作语义及无更新约束一致。旧8k的formal-eval标签不表示使用held-out或成为paper-final结果。

## 2. 核心对比表

DONE/EXPIRED的mean为五回合算术均值；pooled DONE为五回合总数，pooled completion为总DONE/总generated。每个checkpoint总generated均为605。total energy为五回合能耗之和，另列每回合均值。energy per completed task同时给出回合比值均值与总能耗/总DONE。E2E delay只统计DONE任务并按任务池化。Route比例以Local+Remote+Defer有效决策为分母，Idle单列。

| Method | Metric | 4k | 8k | 12k | 16k |
|---|---|---|---|---|---|
| A | mean DONE | 62.8000 | 79.4000 | 65.4000 | 70.8000 |
| A | pooled DONE | 314 | 397 | 327 | 354 |
| A | mean completion ratio (%) | 51.1032 | 65.4357 | 53.8385 | 58.1709 |
| A | pooled completion ratio (%) | 51.9008 | 65.6198 | 54.0496 | 58.5124 |
| A | mean EXPIRED | 50.0000 | 33.2000 | 46.4000 | 41.4000 |
| A | mean evaluation reward | 0.9083 | 37.2907 | 5.9324 | 18.8668 |
| A | mean TRUNCATED | 8.2000 | 8.4000 | 9.2000 | 8.8000 |
| A | mean completed-task E2E delay (s, task-pooled) | 0.4485 | 0.4111 | 0.4915 | 0.4055 |
| A | total energy (J, sum of five episodes) | 150.8462 | 174.1215 | 158.2684 | 175.6387 |
| A | mean total energy (J/episode) | 30.1692 | 34.8243 | 31.6537 | 35.1277 |
| A | mean CPU energy (J/episode) | 29.7958 | 34.3486 | 30.3965 | 33.4041 |
| A | mean TX energy (J/episode) | 0.3734 | 0.4757 | 1.2572 | 1.7237 |
| A | pooled energy per completed task (J/task) | 0.4804 | 0.4386 | 0.4840 | 0.4962 |
| A | mean episode energy per completed task (J/task) | 0.4757 | 0.4315 | 0.4744 | 0.4973 |
| A | Local count (pooled) | 246 | 131 | 131 | 131 |
| A | Remote count (pooled) | 254 | 473 | 473 | 473 |
| A | Defer count (pooled) | 3046 | 0 | 2 | 0 |
| A | Local ratio (%) | 6.9374 | 21.6887 | 21.6172 | 21.6887 |
| A | Remote ratio (%) | 7.1630 | 78.3113 | 78.0528 | 78.3113 |
| A | Defer ratio (%) | 85.8996 | 0.0000 | 0.3300 | 0.0000 |
| B | mean DONE | 51.2000 | 63.4000 | 81.6000 | 68.6000 |
| B | pooled DONE | 256 | 317 | 408 | 343 |
| B | mean completion ratio (%) | 42.3202 | 52.4191 | 67.4660 | 56.3068 |
| B | pooled completion ratio (%) | 42.3140 | 52.3967 | 67.4380 | 56.6942 |
| B | mean EXPIRED | 61.8000 | 49.0000 | 31.2000 | 43.8000 |
| B | mean evaluation reward | -23.8216 | 2.4266 | 41.9584 | 13.3506 |
| B | mean TRUNCATED | 8.0000 | 8.6000 | 8.2000 | 8.6000 |
| B | mean completed-task E2E delay (s, task-pooled) | 0.4945 | 0.4240 | 0.3953 | 0.4206 |
| B | total energy (J, sum of five episodes) | 132.9725 | 127.6480 | 178.1162 | 131.6814 |
| B | mean total energy (J/episode) | 26.5945 | 25.5296 | 35.6232 | 26.3363 |
| B | mean CPU energy (J/episode) | 26.2949 | 25.4418 | 35.2816 | 25.4383 |
| B | mean TX energy (J/episode) | 0.2996 | 0.0878 | 0.3417 | 0.8980 |
| B | pooled energy per completed task (J/task) | 0.5194 | 0.4027 | 0.4366 | 0.3839 |
| B | mean episode energy per completed task (J/task) | 0.5243 | 0.4053 | 0.4304 | 0.3692 |
| B | Local count (pooled) | 209 | 529 | 318 | 131 |
| B | Remote count (pooled) | 261 | 75 | 286 | 473 |
| B | Defer count (pooled) | 3286 | 10 | 0 | 0 |
| B | Local ratio (%) | 5.5644 | 86.1564 | 52.6490 | 21.6887 |
| B | Remote ratio (%) | 6.9489 | 12.2150 | 47.3510 | 78.3113 |
| B | Defer ratio (%) | 87.4867 | 1.6287 | 0.0000 | 0.0000 |
| C | mean DONE | 56.0000 | 65.8000 | 70.6000 | 77.6000 |
| C | pooled DONE | 280 | 329 | 353 | 388 |
| C | mean completion ratio (%) | 46.4100 | 53.8175 | 57.9384 | 63.8533 |
| C | pooled completion ratio (%) | 46.2810 | 54.3802 | 58.3471 | 64.1322 |
| C | mean EXPIRED | 56.8000 | 46.6000 | 41.6000 | 35.0000 |
| C | mean evaluation reward | -12.7028 | 7.7641 | 17.9278 | 33.4842 |
| C | mean TRUNCATED | 8.2000 | 8.6000 | 8.8000 | 8.4000 |
| C | mean completed-task E2E delay (s, task-pooled) | 0.4747 | 0.4300 | 0.4318 | 0.4056 |
| C | total energy (J, sum of five episodes) | 142.6859 | 149.1173 | 162.1704 | 179.3118 |
| C | mean total energy (J/episode) | 28.5372 | 29.8235 | 32.4341 | 35.8624 |
| C | mean CPU energy (J/episode) | 27.9452 | 29.4468 | 31.8827 | 35.0529 |
| C | mean TX energy (J/episode) | 0.5920 | 0.3766 | 0.5514 | 0.8095 |
| C | pooled energy per completed task (J/task) | 0.5096 | 0.4532 | 0.4594 | 0.4621 |
| C | mean episode energy per completed task (J/task) | 0.5167 | 0.4478 | 0.4463 | 0.4580 |
| C | Local count (pooled) | 146 | 273 | 144 | 138 |
| C | Remote count (pooled) | 301 | 330 | 460 | 466 |
| C | Defer count (pooled) | 2718 | 51 | 0 | 0 |
| C | Local ratio (%) | 4.6130 | 41.7431 | 23.8411 | 22.8477 |
| C | Remote ratio (%) | 9.5103 | 50.4587 | 76.1589 | 77.1523 |
| C | Defer ratio (%) | 85.8768 | 7.7982 | 0.0000 | 0.0000 |

完整25项逐seed评估指标含任务结局、delay、CPU/TX/total energy、rejection/downgrade、backlog、outage与workload penalty，见per_checkpoint_seed_metrics.csv以及各evaluation的episode_metrics.jsonl。Route逐seed计数、比例、目的地分布见route_behavior_summary.json。

## 3. 每个开发seed的成对DONE

| Eval seed | Method | 4k | 8k | 12k | 16k |
|---|---|---|---|---|---|
| 1042 | A | 108 | 100 | 78 | 100 |
| 1042 | B | 74 | 81 | 101 | 100 |
| 1042 | C | 60 | 104 | 100 | 100 |
| 1043 | A | 42 | 74 | 69 | 65 |
| 1043 | B | 45 | 56 | 82 | 64 |
| 1043 | C | 60 | 71 | 71 | 78 |
| 1044 | A | 62 | 94 | 74 | 64 |
| 1044 | B | 51 | 65 | 82 | 52 |
| 1044 | C | 52 | 53 | 79 | 80 |
| 1045 | A | 44 | 59 | 46 | 58 |
| 1045 | B | 47 | 56 | 70 | 59 |
| 1045 | C | 50 | 50 | 49 | 59 |
| 1046 | A | 58 | 70 | 60 | 67 |
| 1046 | B | 39 | 59 | 73 | 68 |
| 1046 | C | 58 | 51 | 54 | 71 |

per_seed_comparison.json提供按method与按eval seed两种视图，含全部checkpoint的metrics及8k−4k、12k−8k、16k−12k、16k−8k四组DONE、completion-ratio、reward、expired差值；CSV为per_seed_DONE_and_deltas.csv。

## 4. 阶段变化和场景方向

| Method | Delta | mean DONE delta | mean completion delta (pp) | mean reward delta | mean EXPIRED delta | DONE ↑/=/↓ |
|---|---|---|---|---|---|---|
| A | 8k-4k | 16.6000 | 14.3324 | 36.3824 | -16.8000 | 4/0/1 |
| A | 12k-8k | -14 | -11.5971 | -31.3583 | 13.2000 | 0/0/5 |
| A | 16k-12k | 5.4000 | 4.3323 | 12.9344 | -5 | 3/0/2 |
| A | 16k-8k | -8.6000 | -7.2648 | -18.4239 | 8.2000 | 0/1/4 |
| B | 8k-4k | 12.2000 | 10.0990 | 26.2482 | -12.8000 | 5/0/0 |
| B | 12k-8k | 18.2000 | 15.0469 | 39.5318 | -17.8000 | 5/0/0 |
| B | 16k-12k | -13 | -11.1592 | -28.6078 | 12.6000 | 0/0/5 |
| B | 16k-8k | 5.2000 | 3.8877 | 10.9241 | -5.2000 | 4/0/1 |
| C | 8k-4k | 9.8000 | 7.4075 | 20.4669 | -10.2000 | 3/1/1 |
| C | 12k-8k | 4.8000 | 4.1209 | 10.1637 | -5 | 2/1/2 |
| C | 16k-12k | 7 | 5.9149 | 15.5564 | -6.6000 | 4/1/0 |
| C | 16k-8k | 11.8000 | 10.0358 | 25.7201 | -11.6000 | 4/0/1 |

A：8k→12k五个场景DONE/reward均下降、expired均增加。12k→16k有3升2降、均值回升5.4 DONE；16k仍比8k少8.6 DONE，mean completion低7.2648个百分点。A的12k均值没有高于16k，但存在相对8k的后期下降。

B：8k→12k五个场景全部改善，12k→16k五个场景DONE/completion/reward全部下降、expired全部增加，是明确的固定开发场景late degradation。16k仍高于8k，但不能用12k替换final。能耗存在取舍：总能耗178.1162→131.6814 J，池化每DONE能耗0.436559→0.383911 J/task，退化判断特指任务完成和reward，不表示所有指标都恶化。

C：8k→12k均值改善但场景方向混合；12k→16k DONE为4升1平、reward为5升，mean completion增加5.9149个百分点、expired减少6.6，仍在改善。16k相对8k提升11.8 DONE，mean completion增加10.0358个百分点。

C相对B的mean DONE差为4k:+4.8、8k:+2.4、12k:−11.0、16k:+9.0。末点的相对恢复信号扩大，但不是持续优势；单个training seed不能证明physical_relations_v1具有稳定或因果优势。

## 5. Training rollout与frozen evaluation分开解释

下表来自原16k final保存的每8个500步训练回合历史。训练为随机策略动作和变化的episode seeds；开发评估为固定五个seed和确定性动作。两种reward不作为同一序列，也不比较绝对水平来判断训练失败或收敛。

| Method | 0–4k rollout reward | 4–8k rollout reward | 8–12k rollout reward | 12–16k rollout reward |
|---|---|---|---|---|
| A | -79.3264 | -84.8865 | -68.1217 | -93.9810 |
| B | -77.8269 | -74.7992 | -82.0598 | -89.4669 |
| C | -83.3063 | -74.0906 | -78.5608 | -89.4954 |

B的训练块reward均值−74.799→−82.060下降时，frozen evaluation reward却从2.4266→41.9584提高；随后16k固定评估确实回落。C训练块reward下降时，固定评估仍改善。这是两种采样/决策条件下的观测趋势不同，不是本轮已证实的实现错误。

## 6. Route行为与既有诊断范围

| Method | Checkpoint | Local | Remote | Defer | Local % | Remote % | Defer % | Local when Remote legal | Remote destination counts |
|---|---|---|---|---|---|---|---|---|---|
| A | 4k | 246 | 254 | 3046 | 6.9374 | 7.1630 | 85.8996 | 117 | {"0": 77, "1": 58, "2": 84, "3": 35} |
| A | 8k | 131 | 473 | 0 | 21.6887 | 78.3113 | 0.0000 | 0 | {"0": 152, "1": 120, "2": 129, "3": 72} |
| A | 12k | 131 | 473 | 2 | 21.6172 | 78.0528 | 0.3300 | 0 | {"0": 124, "1": 98, "2": 150, "3": 101} |
| A | 16k | 131 | 473 | 0 | 21.6887 | 78.3113 | 0.0000 | 0 | {"0": 121, "1": 88, "2": 166, "3": 98} |
| B | 4k | 209 | 261 | 3286 | 5.5644 | 6.9489 | 87.4867 | 79 | {"0": 101, "1": 83, "2": 56, "3": 21} |
| B | 8k | 529 | 75 | 10 | 86.1564 | 12.2150 | 1.6287 | 398 | {"0": 14, "1": 15, "2": 39, "3": 7} |
| B | 12k | 318 | 286 | 0 | 52.6490 | 47.3510 | 0.0000 | 187 | {"0": 75, "1": 41, "2": 133, "3": 37} |
| B | 16k | 131 | 473 | 0 | 21.6887 | 78.3113 | 0.0000 | 0 | {"0": 146, "1": 152, "2": 79, "3": 96} |
| C | 4k | 146 | 301 | 2718 | 4.6130 | 9.5103 | 85.8768 | 17 | {"0": 114, "1": 60, "2": 70, "3": 57} |
| C | 8k | 273 | 330 | 51 | 41.7431 | 50.4587 | 7.7982 | 142 | {"0": 103, "1": 63, "2": 121, "3": 43} |
| C | 12k | 144 | 460 | 0 | 23.8411 | 76.1589 | 0.0000 | 13 | {"0": 146, "1": 103, "2": 150, "3": 61} |
| C | 16k | 138 | 466 | 0 | 22.8477 | 77.1523 | 0.0000 | 7 | {"0": 155, "1": 106, "2": 146, "3": 59} |

B的Local/Remote从12k的318/286变为16k的131/473，目的地分布同时改变；这与DONE回落同时发生，但不能凭相关性定位成因或设置Local比例门槛。A的Local/Remote总数在8k/12k/16k相近，但Remote目的地分布不同，不能声称整个Route策略未变。

B/C原trace含task_id、group_mass、leaf_probabilities等字段，可直接统计重复Defer。4k时B/C的Defer决策约98.20%/97.94%来自被Defer至少两次的任务；8k时B为0%、C约78.43%；12k/16k均无Defer。A的hierarchical v1 trace没有Route task_id，重复等待任务无法可靠判断，没有用TX head ID替代或新增影响动作的采集。原始概率字段见各route_decision_trace.jsonl，汇总见route_behavior_summary.json。

## 7. 平台、32k与reward判断

目前没有足够依据认定任何方法进入经验性能平台：A/B有明显反转，C的DONE/completion/reward仍增长、expired仍降低。本轮未预先定义等效界值，不做事后阈值式convergence声明。结论只适用于seed42 + development seeds1042–1046，不推出模型收敛、稳定方法优劣或充分训练后谁一定最好。

没有必须将三组统一延至32k的证据。C持续改善提供后续定向观察动机，不证明32k必要或必然有益；A/B应先复核后期退化并检验多training-seed重复性。本轮不训练。

当前没有可归因证据要求修改reward，科学配置保持冻结。本轮校验未发现reward公式身份不一致；训练波动或checkpoint反转本身不足以支持reward改动。

## 8. 所有新artifact路径

审计和表格目录：D:\My_project_li\metadata\audits\checkpoint_progress\20261013_4k8k12k16k

主要文件：checkpoint_inventory.json、evaluation_protocol.json、comparison_summary.json、per_seed_comparison.json、route_behavior_summary.json、platform_evidence.json、interpretation.json、training_rollout_context.json、core_comparison_table.csv、per_seed_DONE_and_deltas.csv、per_checkpoint_seed_metrics.csv、validation.json、inputs_before.json、inputs_after.json、evaluation_execution.json、evaluation_tests.xml、figure_artifacts.json。完整逐文件路径、大小与SHA256见artifact_manifest.json。

- A-12k：D:\My_project_li\logs\A\seed42\checkpoint_progress\20261013_4k8k12k16k\12k\evaluations\development-service-eval__remote_stress_v2__chk-2ee8be084711__cfg-d2463e1929ea__eval-d893cd3c1376
- A-16k：D:\My_project_li\logs\A\seed42\checkpoint_progress\20261013_4k8k12k16k\16k\evaluations\development-service-eval__remote_stress_v2__chk-c94352238cdc__cfg-d2463e1929ea__eval-a72d8cc2e9a5
- B-12k：D:\My_project_li\logs\B\seed42\checkpoint_progress\20261013_4k8k12k16k\12k\evaluations\development-service-eval__remote_stress_v2__chk-d0b1d9fa22ba__cfg-d2463e1929ea__eval-eeb8813de801
- B-16k：D:\My_project_li\logs\B\seed42\checkpoint_progress\20261013_4k8k12k16k\16k\evaluations\development-service-eval__remote_stress_v2__chk-49f962672e2a__cfg-d2463e1929ea__eval-c575e2bbe14f
- C-12k：D:\My_project_li\logs\C\seed42\checkpoint_progress\20261013_4k8k12k16k\12k\evaluations\development-service-eval__remote_stress_v2__chk-7c9034e00432__cfg-d2463e1929ea__eval-f819edd6abf8
- C-16k：D:\My_project_li\logs\C\seed42\checkpoint_progress\20261013_4k8k12k16k\16k\evaluations\development-service-eval__remote_stress_v2__chk-dc8754d78711__cfg-d2463e1929ea__eval-fffa0489957e

诊断图（PNG/SVG）与数据/生成规格：
- D:\My_project_li\plots\checkpoint_progress\20261013_4k8k12k16k\checkpoint_progress_dashboard.png
- D:\My_project_li\plots\checkpoint_progress\20261013_4k8k12k16k\checkpoint_progress_dashboard.svg
- D:\My_project_li\plots\checkpoint_progress\20261013_4k8k12k16k\figure_data_and_recipe.json

## 9. 测试与校验

评估相关26项测试及8项子测试通过。新增120个评估回合，其中30个learned-policy、90个固定基线；90个新基线逐字段与原8k一致。全部新旧12个方法×checkpoint组合的240个episode记录通过任务、能耗、reward与随机流检查，复算300个learned-policy标量汇总。256个输入文件前后SHA256一致，生产源码文件集合未变。训练/resume/optimizer/backward/paired-shadow守卫调用次数均为0；本轮未使用held-out seeds。

图已核对坐标、单位、图例及JSON数值。只连接四个离散checkpoint，无平滑、无虚构误差条、无最佳点筛选；不是paper held-out或多training-seed证据。
