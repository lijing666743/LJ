# C seed43独立训练至40k

执行后更新（2026-10-15）：用户已完成训练，十个checkpoint校验与4k—40k开发评估均已完成。40k完成391/605（64.63%），seed42同预算为412/605（68.10%）。见[完整对照报告](../results/multiseed/20261015_seed43_4k40k/C_SEED43_DEVELOPMENT_REPORT.md)。下方保留启动前审计与历史命令，不要向已有输出重复启动同一训练。

准备日期：2026-10-14。用户决定夜间自行运行：C seed43从随机初始化训练至40k，每4k保存一次，全部训练完成后再统一评估。本轮没有启动训练。

配置：[c-seed43-40k.json](../../experiments/task-allocation/configs/c-seed43-40k.json)。运行身份为`ca_gat_mappo__remote_stress_v2__seed-43__cfg-ec33289ec8c8__git-eb1140cefc4a`。

## 实验设置

- 与C seed42的40k配置相比，仅改变master seed42→43和logs、plots、dashboard_logs三个输出根目录；reward、环境、网络、PPO、学习率、辅助采样设置和开发评估协议全部保留。
- 这是独立的seed43实验，不加载seed42权重、optimizer或rollout；也不继承seed42的训练计数。`main.py`没有传入resume或warm-start参数。
- 总预算40000个环境步，80个完整回合，每回合500步；训练设备CUDA，保持原Windows/Python3.13.11/torch2.7.1+cu118/MX450环境。
- 4000、8000、12000、16000、20000、24000、28000、32000、36000步保存完整periodic checkpoint；40000步保存`final.pt`，不另生成`step_40000.pt`。
- 预期final计数为40000步、80回合、156次PPO更新、39936条优化样本、64条unused tail。这些是配置推导值，不是尚未产生的实验结果。未来如继续训练，保留36k periodic，不把final权重加载当作完整恢复。

## 已完成的启动前检查

已读取并解析实际配置，通过checkpoint保存几何、CLI `--show-config`和新输出冲突检查。真实构造seed43的actor、critic和两个Adam，确认训练步数、回合数、PPO更新数及pending均为0，两个Adam状态为空、没有continuation来源；拦截并确认训练、环境step和Adam.step调用次数全部为0。

Gate 0与checkpoint契约测试通过，详细结果和526个既有实验输入的冻结核验见[启动前审计](../../metadata/audits/training_launch/20261014_c_seed43_40k_ready/)。本轮没有安装依赖、修改科学源码或产生seed43训练/评估结果。

## 用户运行命令

在PowerShell中整段运行。此命令直接使用新配置，不沿用之前seed42会话中的`$runArgs`。

```powershell
Set-Location 'D:\My_project_li'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& 'D:\Miniconda\python.exe' -u -B -X utf8 main.py --config 'experiments/task-allocation/configs/c-seed43-40k.json'
if ($LASTEXITCODE -ne 0) { throw 'C seed43 training failed. Inspect the error before restarting.' }
```

启动信息应为`seed=43`、配置哈希前缀`ec33289ec8c8`、`requested=cuda, resolved=cuda`。进度应从`episode=1/80`开始，而不是seed42的重放回合。

结果分别进入`logs/C/seed43/40k/`、`dashboard_logs/C/seed43/40k/`和`plots/C/seed43/40k/`。保持电源接通、终端打开，避免电脑进入睡眠。不要同时启动多个使用同一配置/输出目录的进程；若报错，保存错误与现有checkpoint后再检查恢复方案，不直接覆盖重跑。

## 训练完成后统一看效果

配置沿用`evaluation_interval_steps=4000`，但现有训练入口不会因此自动执行冻结开发评估。本命令只训练并保存checkpoint，不在训练途中或结束后自行启动评估。

训练完成后，先核验final计数、所有periodic及训练信号，再统一评估4k—40k检查点。采用原开发seeds1042—1046、CPU float32、masked_argmax和相同到达任务；与seed42对应步数比较，40k为预先固定的主要报告点，中期结果用于学习曲线，不逐seed挑选最佳检查点替代40k结果。

Local和Heuristic在相同冻结场景中的结果应保持一致。最终报告完成率、总能耗、每完成任务能耗和逐场景变化，并分别展示seed42/43，两个训练seed仍是初步重复性证据；本轮不使用held-out最终测试、不修改reward、不上传或改动云端32k基线。
