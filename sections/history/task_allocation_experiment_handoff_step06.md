# 合理任务分配第6步：工程验收与实验入口

> 历史记录：本文保存原阶段的实施计划或8k交接协议。旧进度和命令只说明当时状态；当前A/B16k、C32k与后续续训请看[当前结果](../../docs/results/CURRENT_EXPERIMENT_STATUS.md)及[复现说明](../../docs/reproduction/REPRODUCE_ABC.md)。

本步只验证工程链路并准备独立配置。工程验收结论与原始记录统一见 `logs/development/task_allocation/step06_engineering_gate/validation.json` 和 `step06_report.md`。工程夹具的完成率、reward、loss和图不能作为8k/32k效果证据。目标仍是在相同物理约束下增加系统完成数，不设置Local比例门槛。

## A/B/C配置

| 组 | Route结构 | 输入 | 其他学习与服务规则 |
|---|---|---|---|
| A | 原P1a分层结构 | 原候选特征 | 原P1a |
| B | Local与各Remote统一候选评分，保留Defer | 原候选特征 | 原P1a |
| C | 同B | 加入第5步35维物理关系 | 原P1a |

配置放在 `experiments/task-allocation/configs/`，共A/B/C × training seeds42、43、44九份。逐字段差异与架构身份见工程目录的 `config_diff_and_identity.json`；文件SHA256、配置hash、run_id和输出路径见 `experiments/task-allocation/configuration_manifest.json`。A也是独立输出身份，不静默复用历史P1a结果。

每份8k配置固定16个episode × 500 slots，共8000条主环境transition，4k保存periodic，预算终点保存final。原256条rollout行为对应31次PPO update、7936条optimized和64条unused tail；A/B/C都不为尾段单独补更新。保留原reward、task-event credit、CPU/TX、学习率、网络宽度、四轮PPO、chosen-only Q和原TX paired shadow预算。

所有run从随机初始化开始。禁止预训练、历史权重初始化、warm-start、蒸馏和oracle标签训练。32k配置在开发组选定后再生成；同一次run恢复自己的periodic checkpoint属于断点续训，不能把8k权重作为新32k起点。

## 本步验证的边界

生命周期夹具使用真实环境服务，单独调整极小任务、8-slot horizon和确定性环境条件，核对Local→CPU→DONE、Remote→TX→CPU→DONE，以及EXPIRED/TRUNCATED和任务/工作量守恒。它只证明两条完成路径存在。

A/B/C各使用128维生产网络及真实四轮PPO，单路径520条transition。控制路径连续运行；另一条在260条处保存，此时已经完成一次更新、Adam状态非空、rollout还有4条，随后恢复到520条。逐项比较actor、critic、两套Adam、policy RNG、保存的更新诊断和恢复后第二次完整更新。每路径优化512条、尾段8条。

随后使用各自final，经真实评估入口在seed1042上运行模型、本地限定、启发式、随机四种方法，并重复评估。核对相同外部轨迹、逐episode结果一致、任务守恒、checkpoint不变，以及训练CSV/PNG和评估主文件、Route诊断、服务诊断的实际存在与防覆盖。短集成使用CPU；CUDA相关梯度和更新由回归覆盖，不声称CUDA完整轨迹逐位一致。

现有V1 checkpoint只保存已完成PPO epoch的九项基础标量。中断前更新的KL、clip fraction、entropy schedule、Route/agent-credit等扩展诊断恢复后为NA；不能宣称完整诊断摘要逐位恢复。本步保留该历史schema，单独核对其全部已保存字段与所有学习状态，恢复后的新更新仍完整比较。恢复训练的辅助Route outcome关联也不作为完整服务统计来源；效果分析使用独立评估的真实任务生命周期。

## 下一步启动方式

以下为PowerShell入口，工作目录固定为 `D:\My_project_li`。本步只解析核验命令，没有执行正式训练。工程报告为PASS后，第7步先执行seed42的A/B/C，三个run分别从零开始，按顺序运行便于观察显存与进度；不要并发挤占MX450。

```powershell
& D:\Miniconda\python.exe -B -X utf8 main.py --config experiments/task-allocation/configs/a-seed42-8k.json
& D:\Miniconda\python.exe -B -X utf8 main.py --config experiments/task-allocation/configs/b-seed42-8k.json
& D:\Miniconda\python.exe -B -X utf8 main.py --config experiments/task-allocation/configs/c-seed42-8k.json
```

训练日志分别位于 `logs/development/task_allocation/experiments/{A,B,C}/seed42/8k/<run_id>/`；Dashboard CSV在对应 `dashboard_logs/` 目录，PNG在对应 `plots/` 目录。seed43/44使用同目录下对应配置，属于后续第8步。

完整训练、恢复、评估参数数组见工程目录的 `handoff_commands.json`，共27组。训练命令的执行上下文已通过校验；恢复和评估在本步应当报告“未来checkpoint尚不存在”，没有用伪文件绕过校验。实际短夹具已验证真实文件的恢复和评估。

例如B seed42的预算终点评估入口为：

```powershell
& D:\Miniconda\python.exe -B -X utf8 main.py --config experiments/task-allocation/configs/b-seed42-8k.json --mode evaluation --set launch_profile=null --evaluate-from "D:\My_project_li\logs\development\task_allocation\experiments\B\seed42\8k\ca_gat_mappo__remote_stress_v2__seed-42__cfg-1bd1c1aa9596__git-eb1140cefc4a\checkpoints\final.pt" --evaluation-device cpu
```

此路径绑定当前工作区HEAD与已发布配置；实际启动时以解析出的run_id及清单为准，源码提交后需重新记录身份。开发评估固定使用1042—1046，预留未见种子20042—20061仍不运行。主比较只取预算终点final，不按最好中间checkpoint挑选。

## 第7步的记录要求

分别记录主环境collected、optimized、unused tail、PPO updates、TX shadow和评估成本、耗时及配置/源码/checkpoint身份。比较系统DONE、EXPIRED、TRUNCATED、完成率、时延与能量，再用Local/Remote/Defer、helper负担和真实服务链解释行为；不以Remote下降或Local上升直接判定成功。

seed42结果只决定是否值得扩展开发试验。三训练seed的继续门槛、B/C选择规则以及独立32k预算沿用[实施计划](cooperative_task_allocation_implementation_plan.md)。目前没有新方法优越性或任务完成更多的结论。
