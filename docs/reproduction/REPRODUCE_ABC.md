# A/B/C 当前版本复现说明

更新日期：2026-10-14。本文区分读取既有结果、工程验证、重新运行实验和完整状态续训。整理期间没有启动正式训练或新增环境评估。

## 当前保存了什么

- A/B/C：training seed42 的原始 8k 训练及 4k checkpoint。
- A/B/C：通过完整恢复、重放核对获得的 16k continuation。
- C：通过第二次完整恢复、重放核对获得的 32k continuation，含 16k、20k、24k、28k 中期 checkpoint 和 32k final。
- 开发评估：seeds1042—1046；五个场景的任务到达数分别为 135、122、117、102、129，共 605 个。
- seeds43/44 目前只有原始 8k 配置，不代表已训练。held-out seeds20042—20061 尚未用于最终测试。

完整结果入口见[当前实验状态](../results/CURRENT_EXPERIMENT_STATUS.md)。原始 8k 发布身份仍由 `metadata/ABC_RELEASE_MANIFEST.json` 描述；20261014 清理清单描述此次本地整理，二者不是同一个版本快照。

## 环境与运行入口

已验证的环境：Windows、Python3.13.11、NumPy2.3.5、torch2.7.1+cu118、matplotlib3.10.8、PyYAML6.0.3、pytest9.1.1；训练设备为 NVIDIA GeForce MX450，冻结评估为 CPU float32。

`requirements.txt` 是宽松直接运行依赖；`environment/requirements-abc-tested.txt` 记录实测版本，其中 CUDA wheel 需对应来源，不能当作跨平台自动安装锁文件。PyYAML 为配置读取的可选完整解析器；工程验证另需 pytest。

以下命令在 PowerShell 中执行。其他电脑需替换项目位置和 Python 路径，先使用 `--show-config` 核对解析结果；旧证据内含原机器绝对路径，迁移后不能直接假定所有路径有效。

```powershell
Set-Location 'D:\My_project_li'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& 'D:\Miniconda\python.exe' -B -X utf8 main.py --help
& 'D:\Miniconda\python.exe' -B -X utf8 main.py --config experiments/task-allocation/configs/c-seed42-32k-resume.json --show-config
```

以上仅查看帮助或配置，不训练。

## 查看和重建已有汇报

直接查看 `plots/C/seed42/group_meeting/20261014_scene_fix/` 的五张 PNG/SVG 与修正版压缩包；对应 CSV 在 `dashboard_logs/C/seed42/group_meeting/20261014_scene_fix/`。旧 `20261014/` 图片包已被替代，删除记录保留在本轮清理清单中。

如需从既有评估文件重新出图，使用一个未用过的标签：

```powershell
& 'D:\Miniconda\python.exe' -B -X utf8 -m src.evaluation.group_meeting_figures --tag NEW_UNIQUE_TAG
```

该入口读取已保存的评估数据，核对开发协议、任务到达与固定基线，生成图片、CSV 和来源记录，不启动训练或环境评估。不要省略 `--tag`：默认标签对应已被修正版替代的历史目录。

总体完成率用 `sum(DONE)/sum(generated)`；总能耗用五个场景实际 CPU 与 TX 能耗之和；每完成任务能耗用 `sum(energy)/sum(DONE)`。这些口径与每场景比率的算术平均不同。相同场景内任务到达、任务属性及外部随机过程一致，Local/Heuristic 又是固定策略，因此它们跨 checkpoint 的结果相同是预期现象。

## 工程测试

可在项目内创建专用临时目录，避免 Windows 默认临时目录的权限与长路径问题。每次采用新目录，不复用正式实验输出。

```powershell
$testTag = Get-Date -Format 'yyyyMMdd_HHmmss'
$testRoot = Join-Path (Get-Location).Path "tmp/tests_$testTag"
New-Item -ItemType Directory -Path $testRoot | Out-Null
$env:TEMP = $testRoot
$env:TMP = $testRoot
$env:TMPDIR = $testRoot
$env:MPLCONFIGDIR = Join-Path $testRoot 'matplotlib'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& 'D:\Miniconda\python.exe' -B -X utf8 -m pytest tests -q -p no:cacheprovider --basetemp "$testRoot/pytest" --maxfail=1
```

完整测试包含短回合 PPO 工程夹具，用来验证真实学习链和恢复连续性；这些输出不属于正式实验结果。`tools/run_pytest.py` 是另一种启动方式，但其 `--temp-root` 明确要求在项目外，本次受限工作区验证采用上面的项目内隔离方式。测试结束后只清理该次创建的临时目录；部分 Step06 测试还会在 `plots/development/`、`dashboard_logs/development/` 下生成工程产物，应与真实 A/B/C 结果区分。

## 从零重做与续训的区别

原始九份 `*-8k.json` 是从随机初始化开始的独立训练配置；四份 `*-resume.json` 是延长已有训练的目标配置。仅运行 resume 配置而不传入恢复参数，会成为重新初始化的训练，不能称为连续训练。

如果需要独立重现原始 C8k，以下是从零训练入口示例，使用独立输出位置；本次没有执行该命令：

```powershell
$newTag = Get-Date -Format 'yyyyMMdd_HHmmss'
& 'D:\Miniconda\python.exe' -u -B -X utf8 main.py --config experiments/task-allocation/configs/c-seed42-8k.json --set "output.logs_dir=logs/C/seed42/8k/runs/$newTag" --set "output.plots_dir=plots/C/seed42/8k/runs/$newTag" --set "output.dashboard_logs_dir=dashboard_logs/C/seed42/8k/runs/$newTag"
```

重新执行现有 C32k final 的冻结开发评估，可使用下面入口，仍采用1042—1046协议并写入新目录；本次没有执行该命令，也不把它作为held-out测试：

```powershell
$evalTag = Get-Date -Format 'yyyyMMdd_HHmmss'
$finalC32 = 'logs/C/seed42/32k_continuation/ca_gat_mappo__remote_stress_v2__seed-42__cfg-02c7757520b8__git-eb1140cefc4a/checkpoints/final.pt'
& 'D:\Miniconda\python.exe' -B -X utf8 main.py --config metadata/audits/c32k_progress/20261014_16k32k/C_32k_evaluation_config.json --evaluate-from $finalC32 --evaluation-device cpu --set "output.logs_dir=logs/C/seed42/evaluation/runs/$evalTag" --set "output.plots_dir=plots/C/seed42/evaluation/runs/$evalTag" --set "output.dashboard_logs_dir=dashboard_logs/C/seed42/evaluation/runs/$evalTag"
```

原始 8k final 丢弃了 64 条未优化尾部，原始 16k final 丢弃了 128 条；不能以加载 final 权重代替完整恢复。已经完成的实际路线为：

| 路线 | 恢复点 | 重放核对边界 | 边界尾部处理 |
|---|---|---|---|
| A/B/C →16k | 原训练 step4000 | 与原 8k final 的已保存学习状态核对 | 重放再生的 64 条 pending 留给后续 PPO |
| C →32k | 16k continuation 的 step12000 | 与原 16k final 核对 | 重放再生的 128 条 pending 留给后续 PPO |

完整恢复包括 actor、critic、两个 Adam、独立策略 RNG、计数器、pending rollout 与已保存诊断；环境随机流按原回合种子重建。旧格式没有保存全局 Python/NumPy/Torch RNG，支持范围限定为已核验的训练路径与运行环境，不能泛称为任意进程状态快照。

### 只恢复检查，不训练

下面示例检查 C 的既有 12k→32k 路线；`--check-resume` 只恢复并核验，不走环境步、不更新 optimizer，也不执行 12k→16k 重放。

```powershell
$checkpointRoot = 'logs/C/seed42/16k_continuation/ca_gat_mappo__remote_stress_v2__seed-42__cfg-551432924db1__git-eb1140cefc4a/checkpoints'
$checkTag = Get-Date -Format 'yyyyMMdd_HHmmss'
& 'D:\Miniconda\python.exe' -B -X utf8 main.py --config experiments/task-allocation/configs/c-seed42-32k-resume.json --resume-from "$checkpointRoot/step_12000.pt" --extend-training-budget --verify-replay-against "$checkpointRoot/final.pt" --check-resume --set "output.logs_dir=logs/C/seed42/restore_check_$checkTag" --set "output.plots_dir=plots/C/seed42/restore_check_$checkTag" --set "output.dashboard_logs_dir=dashboard_logs/C/seed42/restore_check_$checkTag"
```

正式启动命令的历史原文位于 `metadata/audits/continuation_launch/20261013_c32k_ready/launch_command.json`，A/B/C16k 恢复参数位于 `metadata/audits/continuation_launch/20261013_ready/preflight.json`。这些路线已经完成，不应再次向同一输出位置启动。

### 后续继续训练 C 应保留什么

C32k 的全部 checkpoint、配置快照、训练侧日志和两层 continuation 来源均保留。特别保留 `step_28000.pt` 和 32k `final.pt`：前者含完整恢复状态与 96 条 pending，后者作为已完成 32k 的核对参照。32k final 的 unused tail 虽为 0，当前加载约束仍区分 final 与 periodic，不能直接假定 final 可续训。

下一轮先确定更大预算与新输出目录，再审计从 28k 恢复、重放到 32k、逐项核对后继续的方案。现有 `c-seed42-32k-resume.json` 不是更大预算配置；本轮没有准备或执行新的正式训练命令。

## 后续实验怎样放置

配置进入 `experiments/task-allocation/configs/`；实际训练日志和模型进入 `logs/{A,B,C}/seedXX/预算_阶段/唯一run_id/`；CSV 与图片进入对应的 `dashboard_logs/` 和 `plots/`；详细分析进入 `docs/results/`，取证与输入身份进入 `metadata/audits/`。所有新实验使用独立输出位置，不覆盖旧结果。

上传前另做目标仓库、文件大小、复现资料与敏感信息检查。当前 `.gitignore` 默认排除运行产物和 checkpoint，不能把“推送源码”理解为“实验模型和结果也已上传”。本次未创建提交或向任何 GitHub 仓库写入。
