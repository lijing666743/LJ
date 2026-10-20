# 项目文件夹说明（本地整理后）

## 现行管理主干（用户已确认，2026-10-14）

| 目录 | 职责 |
|---|---|
| `src/` | 环境、算法、训练、完整恢复、评估与分析实现 |
| `tests/` | 自动验证及固定输入 |
| `docs/` | 当前结果、复现方法、目录导航及管理说明 |
| `logs/` | 按方法/seed/预算/run身份保存真实运行记录和模型 |
| `plots/` | 训练与结果图片、组会图及显示检查图 |

辅助目录继续按本项目约定保留：experiments为配置，dashboard_logs为Dashboard CSV，sections为0—7章和方法材料，metadata为身份与审计，environment为依赖说明。当前实验仍在推进，本轮没有冻结整个项目、复制多套源码或搬动C模型路径。详见[管理参考与后续上传约定](PUBLICATION_PREPARATION.md)。


## sections现行整理（2026-10-14）

`sections/`根目录现在只放0—7章和README；7份方法材料进入`sections/methods/`，2份早期实施记录进入`sections/history/`。下方旧快照中的单独文件名可以按[现行索引](../sections/README.md)定位。

[现行方法说明索引](../sections/README.md)：7份方法Markdown仍在，旧P0与当前P1a信用路线已标注区别。此前生成的离线HTML当前不在工作区，索引以实际保留材料为准。[路径迁移记录](../metadata/cleanup/20261014_sections_formatting/relocation_receipt.json)保留旧文件到新位置的映射；旧审计快照不改写。


## 当前目录补充（2026-10-14）

下方原说明描述20261013的8k整理快照，按历史保留。当前已扩展至A/B16k与C32k，当前结果以[18行总表](results/CURRENT_EXPERIMENT_STATUS.md)和[复现说明](reproduction/REPRODUCE_ABC.md)为入口。

| 位置 | 当前用途 |
|---|---|
| `logs/{A,B,C}/seed42/8k/` | 原8k训练、4k恢复点与历史开发评估，继续保留 |
| `logs/{A,B,C}/seed42/16k_continuation/` | 三组真实连续训练到16k的全部状态与日志 |
| `logs/C/seed42/32k_continuation/` | C续训至32k；16/20/24/28k中期模型和32k final全部保留 |
| `logs/{A,B,C}/seed42/checkpoint_progress/` | 中期与终点开发评估、逐任务和服务诊断 |
| `plots/C/seed42/group_meeting/20261014_scene_fix/` | 当前有效组会五图、来源数据、图表说明及修正版压缩包 |
| `dashboard_logs/C/seed42/group_meeting/20261014_scene_fix/` | 对应总表与逐场景CSV |
| `metadata/audits/c32k_progress/20261014_16k32k/` | C32k训练完整性、开发比较、能耗和任务级诊断 |
| `metadata/cleanup/20261014/` | 本次依赖审计、删除清单、续训文件检查、冻结输入与验证记录 |

配置共13份：原9份8k、3份16k resume和1份C32k resume。`configuration_manifest.json`仍是原发布时快照。`.agents/`、`knowledge/`不参与处理；文献阅读笔记引用的37张页面图继续保留，临时PDF提取文本已清理。IDE个人设置保留并由既有忽略规则排除。

旧组会`20261014/`图片/CSV已被修正版替代并清理；旧审计记录维持历史原文，删除与替代映射见本轮清单。工程测试产物不作为实验结果。

## 2026-10-13目录快照（历史说明）


本说明对应2026-10-13整理后的目录。当前已保存的实验是 **A/B/C × training seed42 × 8k**；seed43/44只有配置，尚无正式训练结果。A是原P1a分层Route，B是统一候选叶比较，C是B加35维物理关系特征。

本次只整理本地项目，不处理上传。已有模型、指标、正式配置、生产源码及0—7章保持原内容。下面将运行代码、实验配置、真实产物、说明材料和工程证据分别说明。

## 一、根目录里的每个文件夹

| 文件夹 | 里面是什么 | 平时什么时候用 |
|---|---|---|
| `src/` | 程序源码：环境、模型、学习、评估、策略及入口调度 | 修改环境或算法、排查运行逻辑 |
| `experiments/` | A/B/C的九份8k配置和配置发布清单 | 选择方法、training seed和实验预算 |
| `logs/` | A/B/C现有训练、final评估与Route诊断的原始文件、模型 | 查原始结果、加载final、核对训练行为 |
| `dashboard_logs/` | A/B/C训练过程的CSV | 看逐步/训练汇总数据、做曲线或交叉核对 |
| `plots/` | A/B/C训练Dashboard图片 | 快速看已有训练曲线 |
| `docs/` | 本目录说明、当前复现说明、seed42结果报告和比较CSV | 看项目怎么用、当前结果是什么 |
| `sections/` | 0—7章及当前A/B/C的方法、协议和实施说明 | 写论文、核对系统与方法边界 |
| `metadata/` | 文件身份、路径迁移、实施审计、历史P1a对齐证据和本次清理/测试记录 | 确认结果来源、文件是否被改变、为什么保留某个历史文件 |
| `tests/` | 自动化工程测试及其固定输入 | 修改代码后检查是否破坏环境、PPO、Route、保存恢复等行为 |
| `tools/` | 测试启动和侧日志修复工具 | 执行工程测试或处理明确的checkpoint侧日志修复需求 |
| `environment/` | 实际验证过的直接依赖版本记录 | 配置或核对Python环境；这里不是环境仿真源码 |
| `knowledge/` | 参考论文与原研究方案，受保护 | 查文献、查原方案；不作为A/B/C训练数据 |
| `.agents/` | 项目内Codex科研技能与模板，受保护 | 辅助文献、科研作图、论文措辞和代码规范；训练程序不依赖它 |
| `.git/` | Git版本记录及工作区身份 | 版本管理、读取真实commit；正式评估会读取身份 |
| `.idea/` | 本机IDE的工作区与代码检查设置 | 使用原IDE时恢复个人工作区；不是实验结果或算法依赖 |

## 二、src：真正的程序放在哪里

根目录的 `main.py` 是轻量入口，实际调用链在这里。

| 子目录 | 当前实际内容与作用 |
|---|---|
| `src/env/` | 真实环境实现：任务到达、生命周期、队列、移动、拓扑、信道、动作与执行器、通信/CPU服务、能耗、观察、reward、指标及状态校验。实际信道/移动/队列源码分别在此目录的channel.py、mobility.py、queues.py等文件中。 |
| `src/models/` | 当前CA-GAT-MAPPO网络与学习链：actor/critic、动作分布、rollout、GAE/PPO、trainer、checkpoint、task-event信用、Route Q输入、统一候选结构、35维物理关系、TX和CPU相关模块。A/B/C主要算法实现集中在这里。 |
| `src/evaluation/` | final模型严格加载、评估runner、指标、Route/服务诊断及共享兼容诊断模块。部分文件名保留旧阶段名称；保留源码不表示本轮执行过对应诊断或oracle。 |
| `src/policies/` | 策略接口及local-only、heuristic、random固定对照策略。评估中会用到这些对照。 |
| `src/training/` | 现有离线destination_distillation工具及包入口；当前A/B/C正式训练由models中的trainer执行，不使用该离线工具。该目录不是本轮预训练产物。 |
| `src/algorithms/` | 目前只有包入口__init__.py，是保留的代码命名空间；当前算法实现不在这里。 |
| `src/baselines/` | 目前只有包入口；真实固定对照实现位于src/policies/。 |
| `src/channel/` | 目前只有包入口；真实信道实现在src/env/channel.py。 |
| `src/mobility/` | 目前只有包入口；真实移动实现在src/env/mobility.py。 |
| `src/queues/` | 目前只有包入口；真实队列实现在src/env/queues.py及生命周期相关文件。 |
| `src/utils/` | 目前只有包入口，保留原源码结构。 |

只有__init__.py的包不是空文件夹，但也不能据目录名称声称其中已实现功能。本次为保留当前源码身份，没有重构这些包，也没有按E13/E14/destination/oracle等文件名直接删除共享源码。

`src/`直属文件中：config.py管理配置，cli.py解析输入，registry.py/runner.py/execution.py负责运行注册与调度，rollout.py负责策略交互，artifacts.py/training_artifacts.py负责产物写入；matched_short_diagnostics.py是已有短诊断支持。当前不用的诊断工具不会仅因存在就自动执行。

## 三、experiments：选择运行哪一种方法

```text
experiments/
├─ README.md
└─ task-allocation/
   ├─ configuration_manifest.json
   └─ configs/
      ├─ a-seed42-8k.json / a-seed43-8k.json / a-seed44-8k.json
      ├─ b-seed42-8k.json / b-seed43-8k.json / b-seed44-8k.json
      └─ c-seed42-8k.json / c-seed43-8k.json / c-seed44-8k.json
```

- `task-allocation/`：当前合理任务分配实验的配置集。
- `configs/`：九份已发布配置；每份指定方法、training seed、8k预算和其他原协议参数。
- configuration_manifest.json：发布时记录的配置hash、差异与当时状态；其中“尚未正式训练”的字段是当时记录，当前完成状态看根README和真实产物。

**后续实验请按 [复现说明](reproduction/REPRODUCE_ABC.md#后续实验怎样放置) 使用输出覆盖参数。** 原配置的历史输出路径未改，直接运行它会重新生成旧层级。覆盖输出只改变存储位置及其对应身份，不改变方法、环境、reward或预算；本次没有启动seed43/44。

## 四、logs：三组真实产物的结构完全对应

```text
logs/
├─ README.md
├─ A/seed42/8k/
│  ├─ training/checkpoints/
│  ├─ evaluation/
│  └─ diagnostics/
├─ B/seed42/8k/（同样结构）
└─ C/seed42/8k/（同样结构）
```

`A/`、`B/`、`C/`区分方法；`seed42/`是训练随机种子；`8k/`表示8000 collected transitions预算，不是评估seed。每一层都有对应用途，未创建没有结果的seed43/44空目录。

| 每组8k下面的目录 | 包含什么 | 应怎样理解 |
|---|---|---|
| `training/` | config_snapshot.yaml、aggregate_metrics.json、raw_metrics.jsonl及Route信用/Q/rollout/调度侧日志 | 同一个8k训练run的不同记录，多个JSONL不代表多个方法或新实验。raw_metrics较大，保留是为了能回到原始数据核对。 |
| `training/checkpoints/` | final.pt、step_4000.pt | final是预算终点模型；step_4000是原中期恢复点。final可用于冻结策略评估；恢复中期训练还需满足配置及Git身份约束。 |
| `evaluation/` | aggregate_metrics.json、episode_metrics.jsonl、evaluation_metrics.csv、evaluation_manifest.json | 该组final在1042—1046的既有评估。manifest记录真实模型、协议与来源。 |
| `diagnostics/` | route_decision_trace.jsonl、summary.json | 与上面的评估对应，描述Local/Remote/Defer与目的地选择；不是另一组训练。 |

评估文件内部除了ca_gat_mappo，还包含local_only、heuristic、random，这是相同协议的固定对照。因此每组evaluation不只有神经模型的五条episode；这些对照不属于之前失败的旧训练版本，也不应为了“只剩ABC”而删去原评估中的对照行。

目录短名没有替换run_id/evaluation_run_id等身份。原始文件中的旧绝对路径保留；现址查metadata/DIRECTORY_RELOCATION.json，不把历史来源改写成新实验。

## 五、dashboard_logs与plots：数据和图分开放

| 目录 | 实际内容 |
|---|---|
| `dashboard_logs/A/seed42/8k/` | A的training_metrics.csv |
| `dashboard_logs/B/seed42/8k/` | B的training_metrics.csv |
| `dashboard_logs/C/seed42/8k/` | C的training_metrics.csv |
| `plots/A/seed42/8k/` | A的training_dashboard.png |
| `plots/B/seed42/8k/` | B的training_dashboard.png |
| `plots/C/seed42/8k/` | C的training_dashboard.png |

各自的A/B/C、seed42、8k中间目录与logs一一对应。CSV是训练过程的表格输出，PNG是已生成的训练图，final评估指标仍在logs相应evaluation目录；不要把训练曲线当成final评估结果。

旧development/task_allocation/experiments空树已清理。step06_engineering_gate/pytest中的CSV/PNG是本次工程回归临时生成物，完成后清除，不保留为正式实验结果。目录根部的.gitkeep只是原来的空目录占位文件，不是结果。

## 六、docs：怎么看当前项目与结果

| 子目录/文件 | 内容 |
|---|---|
| `DIRECTORY_GUIDE.md` | 本说明，解释整个项目目录 |
| `reproduction/` | 只有当前REPRODUCE_ABC.md：环境、校验、测试、后续从零训练和评估的命令 |
| `results/` | 已有结果的阅读与比较材料 |
| `results/seed42/` | Step07A比较报告及7份CSV：训练汇总、评估汇总、逐评估seed、Route行为、Remote目的地、方法差值和历史P1a-1044对齐 |

旧design/remote_stress设计和e13/的seed43 pilot、4k准备报告已核对引用后删除。它们不是当前remote_stress_v2 A/B/C的运行依赖。当前生产实现和测试保留。

## 七、sections：论文与当前方法说明

本目录没有下层目录，文件按主题命名。

| 文件范围 | 内容 |
|---|---|
| `0_abstract_keywords.md`—`7_conclusion.md` | 摘要、引言、系统模型、方法、实验协议、结果模板、讨论模板、结论；八份内容原样保留，模板不代表已有正式论文结论。 |
| `cooperative_task_allocation_implementation_plan.md` | 当前任务分配的实施步骤与执行进度 |
| `task_allocation_experiment_handoff_step06.md` | A/B/C定义、预算与交接协议 |
| `route_candidate_interface_cpu_audit.md` | 候选动作与CPU接口审计 |
| `unified_route_scoring_step03.md` | 统一候选比较结构 |
| `unified_route_learning_step04.md` | 对应学习链 |
| `unified_route_physical_relations_step05.md` | C使用的35维物理关系特征 |
| `e14c_route_task_credit.md`、`e14d_p0_input_repair.md`、`e14e_p1a_task_event_credit.md` | 当前方法继承的信用、Q输入和P1a公共学习机制说明；不是这些旧实验的整套失败产物。 |
| `README.md` | 章节职责及当前资料索引 |

## 八、metadata：为什么还需要少量历史内容

| 子目录/文件 | 内容与保留原因 |
|---|---|
| `ABC_RELEASE_MANIFEST.json` | A/B/C当前配置、源码与模型身份、现址、预期已有指标、运行环境和证据边界 |
| `ABC_FILE_MANIFEST.json` | 本地整理完成时的文件SHA256清单；后续新增实验不自动进入这份冻结清单 |
| `DIRECTORY_RELOCATION.json` | old→new路径、迁移前hash、删除条目；可查旧审计指向的新位置 |
| `audits/` | 与当前实验有关的审核依据 |
| `audits/seed42/` | Step07A的source_manifest、checkpoint_identity_audit、analysis_data、summary_tables、validation；保留原数字、来源路径和取证结果 |
| `audits/implementation/` | Step01—06最终实施证据；删除了重复before/after源码和过时调试迭代 |
| `audits/implementation/step01_baseline_protocol/` | baseline身份、开发seed使用和实验协议 |
| `audits/implementation/step02_candidate_interface/` | 候选接口实现与兼容性结论 |
| `audits/implementation/step03_unified_scorer/` | 统一候选结构与初始化/兼容审计 |
| `audits/implementation/step04_learning_chain/` | 真实学习链及从零配置检查 |
| `audits/implementation/step05_physical_relations/` | 物理特征契约、配置和兼容检查 |
| `audits/implementation/step06_engineering_gate/` | 配置发布、集成汇总与工程门结果 |
| `reference/` | 只有支撑当前报告所需的最小历史对齐输入 |
| `reference/p1a_seed42_1044/` | 历史P1a的original_config.json以及三类证据；不是第四种当前方法 |
| `reference/p1a_seed42_1044/training/` | 原训练aggregate和config snapshot |
| `reference/p1a_seed42_1044/training/checkpoints/` | 原final，支持与当前A作真实张量比较 |
| `reference/p1a_seed42_1044/evaluation/` | 原1044开发服务诊断的manifest、aggregate与episode记录 |
| `reference/p1a_seed42_1044/diagnostics/` | 原1044的Route trace与summary |
| `cleanup/`、`cleanup/20261013/` | 本次清理范围、旧文件删除、路径/冻结文件校验、模型加载、完整回归输出与最终报告 |

原审计中的状态只表示当时执行结果。例如旧清理校验曾记录临时压缩包，但这些包已按要求删除，记录不表示现在还保留归档。旧报告引用的调试临时目录也不承诺继续存在；当前输入现址通过迁移表核对。

这些历史文件分别用于支撑“A与原P1a对齐”或测试共享学习链，不用于A/B/C训练初始化。删除整套旧失败运行与保留少量可核对证据是不同用途。

## 九、tests与tools：测试输入不是训练数据

| 子目录/文件 | 内容 |
|---|---|
| `tests/`直属test_*.py | 自动化测试：环境/生命周期/守恒、网络与PPO、信用与Q、候选结构/物理特征、模型保存恢复、评估与防覆盖等 |
| `tests/fixtures/` | 固定测试输入及用途说明 |
| `tests/fixtures/configs/` | 9份旧配置，供现有共享行为与身份兼容测试读取；不是正式启动入口 |
| `tests/fixtures/historical/` | 只用于工程测试的历史输入 |
| `tests/fixtures/historical/checkpoints/` | 两份包含pending-rollout张量的旧checkpoint |
| `…/e14d_p0_normalized_seed42_8k/` | step_4000.pt，供历史输入/PPO等兼容测试 |
| `…/e14d_route_counterfactual_q_seed42_8k/` | step_4000.pt，供Q输入/学习链兼容测试 |
| `tools/run_pytest.py` | 已有测试启动器，选择隔离临时目录 |
| `tools/repair_route_sidecar_for_resume.py` | 有明确需求时修复checkpoint恢复相关的Route侧日志；本次没有用它改正式结果 |

旧命名的测试可能覆盖当前仍在用的共享代码。不能只按测试名的E编号删除它。当前从零训练不读上述两个测试checkpoint，也没有预训练或oracle数据集输入。

## 十、environment、knowledge、.agents与隐藏目录

`environment/`仅含requirements-abc-tested.txt，记录Python 3.13.11环境下验证过的直接依赖版本。它不是包含全部传递依赖的跨平台锁文件；仿真源码在src/env/。根requirements.txt继续保留原依赖范围。

`knowledge/`受保护，本次只为解释目录读取名称，没有修改、删除或移动内容：

- `knowledge/papers/`：13份参考论文PDF，包括用户提供的多UAV协同计算任务调度论文及其他相关工作。
- `knowledge/project_plan/`：原方案2.md，是研究方案资料。

`.agents/`同样受保护，训练程序不把这里当输入数据：

| 技能目录（均在.agents/skills/下） | 用途及下层目录 |
|---|---|
| `sci-figures-python/` | 科研数据图；agents/存调用配置，assets/存辅助资产，references/存规范参考 |
| `sci-image2-diagram-drafter/` | 科研结构/机制图草稿；agents/存调用配置，references/存画图规则，templates/存模板 |
| `sci-literature-review/` | 文献检索与证据核对；agents/存调用配置，references/存核对规范 |
| `sci-manuscript-formalization-augmenter/` | 公式、符号、算法与形式化表达；agents/存调用配置，assets/存辅助资产，references/存规范 |
| `sci-manuscript-phrase-refiner/` | 学术措辞润色；agents/存调用配置，references/存规范，templates/存模板 |
| `sci-student-research-code-builder/` | 科研代码组织规范；agents/存调用配置，references/存规范，assets/存程序模板 |
| `sci-student-research-code-builder/assets/research-program-template/`及其`src/` | 技能附带的通用程序模板；不是当前项目的第二套运行源码或A/B/C实验 |

`.agents/skills/`是上述六种技能的容器；每个技能的SKILL.md规定其使用方式。保护目录中即便存在空目录，也没有自动清理。

`.idea/inspectionProfiles/`保存IDE的代码检查配置，workspace.xml保存个人工作区状态。它与论文指标无关，本次未把个人IDE配置作为失败实验删除。

`.git/`由Git管理：hooks/是钩子模板或脚本，info/是本地辅助配置，logs/是版本引用变动记录，objects/存版本对象，refs/存分支等引用，worktrees/存工作树管理信息；index/HEAD/config等是版本索引与配置。这里的logs/与项目根logs/不是同一用途，不应按“旧实验日志”清理。没有改动Git内部文件。

## 十一、根目录文件与以后怎么用

| 根目录文件 | 用途 |
|---|---|
| `main.py` | 轻量程序入口 |
| `README.md` | 当前宏观状态、范围与导航 |
| `AGENTS.md` | 仓库工作约定和保护边界 |
| `requirements.txt` | 原运行依赖范围 |
| `.gitignore` | 原有Git忽略规则；本次未处理上传，也未改它 |

日常看结果先读docs/results/seed42；看原始文件去logs/A、B、C；继续实验先读docs/reproduction/REPRODUCE_ABC.md，选择配置后用新输出目录。当前冻结结果不覆盖，后续运行建议统一存成：

```text
logs/A（或B、C）/seed43（或42、44）/8k/runs/时间标签/...
dashboard_logs/A（或B、C）/seed43（或42、44）/8k/runs/时间标签/...
plots/A（或B、C）/seed43（或42、44）/8k/runs/时间标签/...
```

重新运行工程测试可能产生新的临时CSV/图片；它们不自动变成正式A/B/C证据。原始产物里的多个诊断文件、固定对照行和测试夹具各有用途，不按数量误判为多种正式方法。

2026-10-14补充Markdown审计：0—7章原样保留；其余文档逐项用途见[sections当前索引](../sections/README.md)。旧e14d_route_counterfactual_q.md已由P1a/task_event、normalized Q及统一学习链说明替代而删除。另清理3个空目录；全部受保护目录排除在扫描之外。
