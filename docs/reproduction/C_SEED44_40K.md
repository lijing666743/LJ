# C seed44独立训练至40k

执行后更新（2026-10-15）：用户已完成训练，十个checkpoint及4k—40k冻结开发评估均已完成。固定40k完成370/605（61.16%），Heuristic为371/605；后四点完成率跨度11.24个百分点。见[三训练seed评估说明](../results/multiseed/20261015_seed44_4k40k/C_SEED44_DEVELOPMENT_REPORT.md)。下方为启动前历史记录；不要向已有输出重复启动同一训练。

准备日期：2026-10-15。用户自行启动；本次准备没有启动训练。配置为[c-seed44-40k.json](../../experiments/task-allocation/configs/c-seed44-40k.json)。

本轮重点研究C，并保留A/B历史结果。A/B当前预算16k，因此不将C40k的结果用于同预算方法优劣结论。

## 训练设置和身份

seed44是新的独立训练重复，从随机初始化到40000步，不加载seed42/43权重、optimizer或rollout。与seed43的40k配置相比，仅改变master seed及logs、plots、dashboard_logs三个输出目录；reward、环境、网络、PPO和辅助采样设置不变。包括`tx_shadow_rng_seed=130043`在内的辅助设置保留，故本次检验的是固定这些设置、改变master seed的重复性。

配置哈希前缀为`4e86aa1bef24`，当前运行身份为`ca_gat_mappo__remote_stress_v2__seed-44__cfg-4e86aa1bef24__git-eb1140cefc4a`。

训练设备CUDA；预算80回合，每回合500步。4k、8k、12k、16k、20k、24k、28k、32k、36k保存完整periodic checkpoint；40k保存`final.pt`，不另生成`step_40000.pt`。预期final为40000步、156次PPO更新、39936条已优化样本、64条unused tail；这些是预算推导，尚非实验结果。未来若延长预算，需保留36k完整checkpoint，通过恢复36k、重放核对40k再延长，不直接加载final权重冒充连续训练。

## 已完成的检查

实际配置解析、checkpoint几何检查和独立输出目录冲突检查通过。真实构造CUDA actor、critic和两个Adam，确认训练步数、回合数、PPO更新和pending均为0，Adam状态为空，无continuation来源。准备检查中禁止并确认没有训练、环境step、Adam.step及checkpoint加载调用。

Gate 0及checkpoint契约共63项测试、8项子测试通过。详细证据见[启动前审计](../../metadata/audits/training_launch/20261015_c_seed44_40k_ready/)。未修改reward、环境、学习率或训练源码，未安装依赖或上传GitHub。

## PowerShell启动命令

整段执行，不沿用之前会话的`$runArgs`：

```powershell
Set-Location 'D:\My_project_li'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
foreach ($c44Output in @('logs/C/seed44/40k', 'dashboard_logs/C/seed44/40k', 'plots/C/seed44/40k')) {
    if (Test-Path -LiteralPath $c44Output) { throw "Output already exists: $c44Output. Inspect before restarting." }
}
& 'D:\Miniconda\python.exe' -u -B -X utf8 main.py --config 'experiments/task-allocation/configs/c-seed44-40k.json'
if ($LASTEXITCODE -ne 0) { throw 'C seed44 training failed. Inspect the error before restarting.' }
```

启动应显示`seed=44`、`cfg-4e86aa1bef24`、`requested=cuda, resolved=cuda`，首回合`episode=1/80`。输出位于`logs/C/seed44/40k/`、`dashboard_logs/C/seed44/40k/`及`plots/C/seed44/40k/`。

接通电源并避免睡眠，保持终端打开，同一输出目录只运行一个训练进程。如果中断，保留日志和checkpoint，先检查恢复路径，不删除后重跑。

## 完成后的下一步

该命令只训练并保存checkpoint，不自动执行冻结评估；`evaluation_interval_steps=4000`也不会自动启动评估。完成后先核验十个checkpoint、训练计数、回合种子和指标有限性，再统一评估4k—40k。

沿用1042—1046五个开发场景、CPU float32、masked_argmax、冻结网络及相同任务到达轨迹。40k是预先固定的主要比较点；中期checkpoint用于观察学习过程，不挑各seed最佳点替代40k。与seed42/43分别比较完成率、总能耗、每完成任务能耗及逐场景变化；Local与Heuristic采用相同场景口径。三次训练可增加重复性证据，不能自动证明全局收敛或所有场景优势。最终held-out测试继续保留，不用于本轮调参。
