# C从32k连续训练到40k

执行后更新（2026-10-14）：用户已完成40k训练，原32k学习状态重放核对通过；36k、40k冻结开发评估也已完成。结果见[40k开发评估报告](../results/seed42/20261014_32k40k/C40K_DEVELOPMENT_REPORT.md)。以下保留启动前审计与历史命令，不要再次向已有40k输出目录运行同一命令。

准备日期：2026-10-14。用户选择总预算40k，并自行启动。本轮仅准备配置、恢复核验和命令，没有执行正式训练、真实重放或新增开发评估。

## 本轮范围

这是既有C seed42训练链的第三次预算延长，新增32k—40k共8000个环境步、16个回合，不是重新初始化。只修改训练总步数32000→40000、总回合数64→80，以及logs、plots、dashboard_logs三个新输出根目录；reward、环境、网络、策略采样规则、PPO参数和学习率均沿用32k设置。

配置：[c-seed42-40k-resume.json](../../experiments/task-allocation/configs/c-seed42-40k-resume.json)。目标配置哈希为`708b8aaa4d67e8ff24a3c8ae4c44d97f89ab79b28fc1b7927547f83c1916ba4a`。

原工作区Git HEAD仍为`eb1140cefc4a0ff79b8bd0c05e73c7a23528d7ba`。本命令针对当前电脑原工作区；不要先切换至GitHub发布提交，也不能把新克隆目录直接当作已核验续训环境。

## 连续性路线与尾部

1. 从32k训练目录中的`step_28000.pt`恢复actor、critic、两个Adam、策略RNG、计数器、诊断及96条pending rollout。恢复点已有27904条优化样本、109次PPO更新。
2. 重放28k—32k的4000步。96条pending被保留并继续填充，第一批满256条rollout的更新发生在28160步。
3. 在32000步逐项核对原32k `final.pt`的模型、两个优化器、策略随机状态、诊断和计数器。原32k final优化32000条、125次更新、unused tail为0，所以此边界没有需补回的被丢弃样本。若不完全一致，停止，不进入32k之后。
4. 核对通过后继续至40k。一次启动实际执行12000步，其中4000为重放，8000为新增训练；这不是在32k基础上再训练40000步。

| 总步数 | 保存类型 | 完成回合 | 优化样本数 | PPO更新数 | 保存pending | final丢弃tail |
|---|---|---:|---:|---:|---:|---:|
| 28000 | 既有periodic恢复点 | 56 | 27904 | 109 | 96 | 0 |
| 32000 | 新目录中的periodic边界核对点 | 64 | 32000 | 125 | 0 | 0 |
| 36000 | 新periodic检查点 | 72 | 35840 | 140 | 160 | 0 |
| 40000 | 新final | 80 | 39936 | 156 | 0 | 64 |

表中是预算和rollout几何推导的预期计数，不是新增实验结果。40k末尾64条按既有规则不做不完整PPO更新；未来如继续延长，应保留36k periodic并重放核对40k、再保留再生的64条，不能用40k final权重替代完整恢复。

## 已完成的准备检查

- 实际调用CLI的`--check-resume`恢复路径，逐项状态比较通过；拦截并确认环境step、Adam.step、trainer._run调用次数全部为0。
- 环境Gate 0、checkpoint契约及预算延长允许项检查：64项测试、14项子测试通过。历史第二次延长与不中断真实PPO对照验证记录仍保留；本轮未重新执行该耗时测试，也未在正式CUDA训练链上提前重放4000步。
- 新配置相对32k配置只有五个允许字段变化，三个正式输出根目录均不存在，旧32k final SHA256与保存基线一致。
- 当前实测环境仍为Python3.13.11、torch2.7.1+cu118、CUDA11.8、MX450。依赖以`environment/requirements-abc-tested.txt`及原环境为准，没有安装或更新依赖。
- 原始checkpoint不是通用进程快照：全局Python/NumPy/Torch RNG未逐一保存，恢复范围限于当前已核验的固定训练路径、原环境与原Git身份；环境随机流按原回合种子重建。

详细记录：[本轮审计目录](../../metadata/audits/continuation_launch/20261014_c40k_ready/)。启动前完整状态检查通过，不等于真实28k—32k CUDA重放已通过；后者是启动时必须通过的门槛。

## 由用户执行的训练命令

在PowerShell中执行，保持原工作目录；这条命令会启动真实重放和训练。本次准备没有执行它。

```powershell
Set-Location 'D:\My_project_li'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$cp = 'logs/C/seed42/32k_continuation/ca_gat_mappo__remote_stress_v2__seed-42__cfg-02c7757520b8__git-eb1140cefc4a/checkpoints'
$runArgs = @(
    '--config', 'experiments/task-allocation/configs/c-seed42-40k-resume.json',
    '--resume-from', "$cp/step_28000.pt",
    '--extend-training-budget',
    '--verify-replay-against', "$cp/final.pt"
)
& 'D:\Miniconda\python.exe' -u -B -X utf8 main.py @runArgs
if ($LASTEXITCODE -ne 0) { throw 'C 40k continuation failed. Inspect the error before restarting.' }
```

到32k时应看到`Replay verification passed at 32000; pending tail retained`。这句话中的tail是统一日志措辞；本轮32k实际长度为0。若返回非零或出现mismatch，不跳过检查、不覆盖输出重跑；先保存错误和已写文件，再决定从哪个完整periodic恢复。

新结果进入`logs/C/seed42/40k_continuation/`、`dashboard_logs/C/seed42/40k_continuation/`和`plots/C/seed42/40k_continuation/`。训练中应持续看到回合或PPO进度，以及文件增长。用户手动运行期间建议约每10分钟检查控制台、日志或checkpoint是否更新；连续两次都没有更新时记录停滞嫌疑并检查进程，不直接认定训练失败。本命令只进行训练，保存checkpoint不会自动完成冻结开发评估。

## 训练后怎样判断是否趋于平稳

完成后先核验32k重放状态与40k计数，再对36k periodic和40k final分别执行CPU float32、masked_argmax、开发seeds1042—1046的冻结评估。沿用原评估路径；不要将periodic直接当作final加载。五场景总到达数应仍为605，逐场景分别为135、122、117、102、129，否则先检查协议。同一场景的Local和Heuristic复用已核验固定结果，不重定义基线。

主要比较20k、24k、28k、32k、36k、40k的总体完成率、评估回报、CPU/TX总能耗、每完成任务能耗及逐场景变化。总体完成率按总完成数/总到达数，能耗按五场景求和。训练阶段仍用每8个完整回合的非重叠均值及标准差观察回报，并查看actor/critic损失、策略熵、KL等已记录指标是否有限、有无异常尖峰；不因随机训练回合任务数不同就直接用完成数判定性能变化。

在看到新结果前约定以下诊断口径：最近四个检查点（28k、32k、36k、40k）的完成率跨度不超过2个百分点，可作为平台期参考，但还要检查是否持续单向改善/下降、评估回报与能耗趋势、逐场景是否掩盖退化，以及训练回报均值/波动是否仍变化。2个百分点只是本轮描述性容差，不是显著性检验或收敛证明。若训练回报仍持续提升，或个别场景继续退化，应报告这种分歧，不能只凭总体曲线变平就下结论。

已有训练回报末段仍为负且在改善，训练信号暂按`signal-watch`观察；不为满足正回报或平滑外观改变reward。只延长C seed42、五个开发场景的证据最多支持“该训练链在这些场景上趋于平稳”，不能证明理论收敛、多seed稳定或与A/B同预算优势。本轮不使用held-out最终测试、不展开大量场景扫描、不生成预期结果数值。
