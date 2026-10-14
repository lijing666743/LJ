# 合理任务分配第6步工程报告

状态：**PASS，配置已发布**。本步完成工程验收；第7步正式8k训练未启动。不存在新方法提高任务完成数的结果声明。

## 实际改动

- `src/evaluation/actor_loader.py`：评估架构校验补入Route特征模式，明确拒绝B/C混用；默认原特征不写入身份字典，保留旧A/B评估身份。
- `tests/test_task_allocation_engineering_gate.py`：新增10项配置、真实生命周期、真实PPO/保存恢复/评估/产物集成检查。
- `tests/test_route_slot_workload_timing.py`：补齐历史配置比较夹具中的10个后续新增中性默认字段，保留原严格结构和hash断言。
- `experiments/task-allocation/configs/`：新增九份A/B/C × seeds42/43/44独立8k配置；新增配置身份清单。
- README追加宏观进度；实施计划追加执行更新6；详细命令及边界见 `sections/task_allocation_experiment_handoff_step06.md`。

业务源码只改动评估loader。本步没有修改环境物理源码、reward、CPU/TX、P1a信用和优化目标、学习超参数、旧配置与旧checkpoint。与本步开始状态比较的补丁见 `source_and_existing_test.patch`；仓库原有dirty改动不归为本步改动。

## 测试结果与失败处理

| 验证 | 实际结果 |
|---|---|
| 完整仓库回归首轮 | 1344 passed、6 failed、338 subtests passed，2013.99秒 |
| 六项失败所在两个完整模块复验 | 24 passed |
| 最终修改后的loader及相关回归 | 137 passed、30 subtests passed |
| 新增A/B/C工程检查 | 10 passed，567.43秒 |
| 最后生命周期断言增强复验 | 7 passed、3 deselected |

跨上述运行覆盖1360个不同测试，六项原回归失败均已解决，没有未解决失败。这是完整回归加受影响模块复验，不声称又执行了一次全绿完整回归。完整首轮在loader修改前启动；修改后的loader由137项相关回归和三组真实集成覆盖。最后一次新增测试文件修改只增强生命周期断言，已单独复验。

首轮五项侧日志修复测试失败源于本次测试根目录过长，Windows普通路径打开失败；改用仓库内较短的 `logs/t6tmp/` 后原测试通过，未改生产修复工具。另一项历史配置测试仍保留后续新增的中性默认项；补齐夹具省略清单后原结构/hash检查通过。

新集成首次对整个训练摘要作相等断言失败：旧V1 checkpoint仅保存已完成epoch的九项基础标量，扩展诊断恢复后NA。改为严格比较全部已保存诊断、全部学习状态及恢复后第二次完整更新，同时显式记录缺失字段，没有放宽参数/Adam/RNG相等要求。随后修正测试中signal gate名称拼写，并让长评估路径的读取使用项目已有Windows扩展路径适配。所有首次失败记录保留在 `integration01/02/03.txt`；最终全流程结果为 `integration04.txt/xml`。

## 真实集成证据

三组均采用128维网络及真实四轮PPO。工程夹具只覆盖horizon=20、预算520、26 episodes、checkpoint间隔260、CPU和隔离输出目录；服务、reward、学习/信用参数保持原配置。

每组两条路径均采集520、优化512、尾段8、更新2次。中断在260条处，已有一次真实更新和非空actor/critic Adam，pending rollout为4。恢复终点的actor、critic、两套Adam、policy RNG与连续路径逐位相同；保存的全部基础诊断和恢复后的第二次完整更新一致。

每组final通过真实评估入口，在seed1042上运行模型、本地限定、启发式和随机四方法，共80 slots，随后重复80 slots；逐episode结果一致，方法间外部随机轨迹一致，任务守恒，checkpoint SHA256不变。训练核心文件/CSV/PNG及侧日志、评估4个主产物、2个Route产物、4个服务产物已实际生成；重复发布拒绝且原字节不变。B/C交叉加载在特征身份检查处明确拒绝。

四个生命周期夹具中，任务0在Local路径由source0 CPU于slot2完成，没有TX；Remote路径在slot2传输1 bit至helper1，并由helper1 CPU于slot3完成。两者都真实服务1 cycle。另验证EXPIRED、TRUNCATED及任务/bit/cycle守恒。极小任务仅用于工程测试，不代表原工作负载上的策略性能。

每组原始证据在 `pytest_integration04/test_real_trainer_resume_and_e{0,1,2}/engineering_evidence.json`；汇总和文件校验值见 `integration_summary.json`。最终增强生命周期证据在 `pytest_lifecycle_final/`。

## 身份与成本

九份配置严格由原P1a复制，仅变更组别所需Route模式、seed和独立输出目录；逐字段diff和架构身份见 `config_diff_and_identity.json`。均为16×500=8000主环境transition、4k periodic与final、7936 optimized、64 unused tail、31次PPO更新。没有加载历史权重、warm-start或oracle标签训练。正式产物目标仍不存在。

三个历史P1a/E13 8k/32k模型经最终评估loader严格加载，默认特征身份保持旧口径、checkpoint SHA256不变，见 `historical_evaluation_loader.json`。未对它们重跑episode，未分析FULL-1k数据。完整回归中的旧E15/E16单元夹具仅检查已有兼容行为，不参与A/B/C训练或初始化。

成功集成共3120主环境transitions、480评估slots、352实测TX shadow slots，其中A控制/恢复各56、B各60、C各60。包括失败重试的本步新集成累计8320主环境transitions，五轮生命周期累计160 slots，评估及预检查累计720 slots。早期失败重试未独立持久化shadow合计，记NA而非0；既有完整回归夹具的模拟总成本也未汇总。成本、耗时和口径见 `fixture_cost.json`，工程成本不计入正式8k/32k预算。

Python 3.13.11、PyTorch 2.7.1+cu118、CUDA可用设备MX450；CPU完整一致性已验证，CUDA相关回归通过，不声称CUDA全轨迹逐位一致。依赖与入口不变，运行环境记录见 `runtime_and_scope.json`。技能引用的外部环境契约文件在当前机器不存在，沿用已记录的Miniconda与项目requirements，未安装新依赖。

## 限制与下一步

旧V1已完成更新的KL、clip fraction、entropy schedule、Route/agent-credit等扩展诊断恢复为NA；本步没有扩展历史checkpoint schema。恢复训练的辅助Route outcome关联不当作全程服务统计，正式结果应使用独立评估的真实任务生命周期。短夹具只有8个PPO epochs，信号gate为 `insufficient-horizon`，不能据此判断学习效果。

当前dirty源码不能仅用HEAD复现，必须连同本步文件SHA256清单保留。`protected_files_after.json` 与 `delivery_check.json`提供本步边界核验。

第7步入口见 `sections/task_allocation_experiment_handoff_step06.md`，27组命令解析记录见 `handoff_commands.json`。恢复/评估命令等待各自真实checkpoint生成，未伪造文件绕过校验。下一步只按计划从seed42独立从零启动A/B/C 8k，再比较系统DONE；本步不启动第7步，也不提前使用预留未见评估种子。
