# 后续仓库整理与上传说明

日期：2026-10-14。本次只完成本地整理，上传目标等待用户另行提供。

只读参考：[M24_LiJingRu](https://github.com/didgmd/SUTResearchHub/tree/main/M24_LiJingRu)。已查看其顶层版本分组及 P1v6、P2v1 目录；P1v6 把论文图片与指标 CSV 放在 paper_results，P2v1 包含若干比较与消融入口。本项目借鉴“源码、配置、结果和绘图入口可定位”的组织方式，继续遵守本项目已有 src/tests/logs/plots/dashboard_logs 约定。没有复制其代码、改变本项目技术路线，或向参考仓库写入。

后续面向用户提供的新仓库，拟按以下内容组织：

| 内容 | 本地位置 | 后续处理 |
|---|---|---|
| 入口、源码、固定测试与配置 | main.py、src、tests、experiments | 保留依赖完整性，提交前核对目标仓库要求 |
| 运行环境与使用说明 | requirements.txt、environment、docs/reproduction | 随源码提供，明确机器路径与CUDA构建条件 |
| 当前结果导航 | docs/results/CURRENT_EXPERIMENT_STATUS.md | 包含真实18行总表和解释边界 |
| 组会图片、汇总CSV | plots/C/seed42/group_meeting/20261014_scene_fix、dashboard_logs对应目录 | 现被忽略规则排除，上传时应有意识选择要发布的交付件 |
| checkpoint与原始日志 | logs/A、B、C | 全部本地保留，尤其C续训链；确定目标后再选择合适的发布方式 |
| 论文章节、研究资料和技能 | sections、knowledge、.agents | sections保留；knowledge与.agents为受保护目录，不自动处理或上传 |
| 本机设置与临时产物 | .idea、tmp、缓存 | 个人设置不动，临时输出清理并忽略 |

当前原始日志有超过1GB的文件，因此不能直接把整个本地工作区当作普通Git文件包。现有 Git 工作区也已有大量未提交修改及历史删除。用户提供目标后，先形成明确的发布清单并核对文件大小、路径与必要结果，再处理提交；不向只读参考仓库推送。

本次清理记录：[清理报告](../metadata/cleanup/20261014/cleanup_report.md)。后续C继续训练与上传是两项独立工作；清理不会删除恢复点，上传也不能用加载权重替代完整续训。

## 用户确认的进行中项目管理方式（2026-10-14）

用户已确认以 `src、tests、docs、logs、plots` 为主干继续管理实验。当前工作区已具备这些目录，本轮补齐职责与后续规则，不搬动已有模型或实验输出路径。

| 主目录 | 当前职责 | 后续如何使用 |
|---|---|---|
| `src/` | 环境、策略、模型、PPO、完整续训、评估与分析实现 | 实现和修改集中在此；根目录main.py保持轻量入口 |
| `tests/` | 环境守恒、模型、恢复连续性、指标与图表口径验证，及固定夹具 | 代码变更运行相关验证；工程夹具产物与正式实验区分 |
| `docs/` | 当前状态、结果入口、复现说明、目录说明及发布准备 | 优先维护已有索引，避免为同一事项反复新增相似说明 |
| `logs/` | 原始运行记录、配置快照、checkpoint与真实开发评估 | 沿用方法/seed/预算/独立run身份，续训写入新位置，保留来源链 |
| `plots/` | 训练曲线、分析图、组会图与文档显示检查图 | 图能追溯到真实CSV/JSON及生成入口，当前有效版本明确标注 |

按本项目已有约定，继续保留 `experiments/` 的正式配置、`dashboard_logs/` 的Dashboard CSV、`sections/` 的0—7章与方法材料、`metadata/` 的身份和审计，以及 `environment/` 的实测依赖说明。`.agents/`、`knowledge/`仍受保护。本轮不将这些必要内容强行并入五个主目录。

后续版本与实验管理遵循以下方式：

1. 共享一套运行源码，A/B/C差异由各自配置与方法模块说明；预算延长按独立run保存，不因8k、16k、32k各复制一套源码。
2. 每次正式运行记录方法、training seed、预算、配置身份、源码身份和输出位置。当前A/B16k、C32k仍是seed42开发结果，不能把目录整理当作多seed或收敛证据。
3. C后续续训保留periodic、final、配置和侧日志；确定新预算后再准备目标配置、零步恢复检查和旧预算边界重放核对。此次没有选择新预算或启动训练。
4. 结果查看统一从[当前总表](results/CURRENT_EXPERIMENT_STATUS.md)进入；运行方式从[复现说明](reproduction/REPRODUCE_C_D.md)进入；方法公式从[材料索引](../sections/README.md)进入。
5. 用户提供目标仓库后，再确定源码版本、要展示的紧凑CSV/图片、需要共享的恢复文件及大文件保存方式。当前忽略规则会排除logs/plots/dashboard_logs，上传时应明确选择交付范围；本轮没有改忽略规则，也没有提交或推送。

## 对老师仓库的只读参考范围

本轮读取[仓库总说明](https://github.com/didgmd/SUTResearchHub/blob/main/README.md)、[管理模板](https://github.com/didgmd/SUTResearchHub/blob/main/AGENTS.md)和忽略规则；查看全部27个学生/团队目录第一层，并深入39个项目/源码/结果等目录，另阅读14份项目说明、索引或路径协议文件。这里只比较可见的组织方式，不声称逐行审计了全部算法、实验质量或提交历史。

参考仓库的治理文件作为学习材料，不替代本项目AGENTS.md或用户已确认的约定。所有访问均为读取，没有克隆参考仓库、运行其中脚本、复制算法代码或修改远端内容。

| 学生/团队目录 | 实际观察到的组织 | 对本项目的启示 |
|---|---|---|
| [M22_LianXiaoHui](https://github.com/didgmd/SUTResearchHub/tree/main/M22_LianXiaoHui) | 参数、状态、动作等功能文件平放 | 借鉴按研究概念拆模块，现统一收在src |
| [M22_LiuXue](https://github.com/didgmd/SUTResearchHub/tree/main/M22_LiuXue) | P1/P2及较多含DRL/UP/BR/参数条件的目录，部分含Log | 条件要清楚可辨；我们用配置记录差异，减少重复源码 |
| [M22_WangWeiRan](https://github.com/didgmd/SUTResearchHub/tree/main/M22_WangWeiRan) | 功能模块、Validation、PictureDrawer与Log | 计算、验证和绘图职责分开 |
| [M23_GuoHongZhang](https://github.com/didgmd/SUTResearchHub/tree/main/M23_GuoHongZhang/P1v9) | P1v系列；所看P1v9有src/docs/logs/plots及current_state/next_task | 用当前状态和下一步入口导航；日志文件携带方法/seed/时间 |
| [M23_KongYuWei](https://github.com/didgmd/SUTResearchHub/tree/main/M23_KongYuWei/P1v5) | P1v系列；docs/logs/plots、分组修订实验及figure-ready CSV | 图表与导出数据对应，比较/消融入口可定位 |
| [M23_YueZiCheng](https://github.com/didgmd/SUTResearchHub/tree/main/M23_YueZiCheng/P1v5) | P1v系列、NS3、统一实验合同及较多审计/绘图脚本 | 统一路径、配置和运行身份，保留图表数据来源 |
| [M24_LiJingRu](https://github.com/didgmd/SUTResearchHub/tree/main/M24_LiJingRu) | P1v/P2v版本；paper_results集中图片和方法指标CSV，P2v有消融目录 | 结果展示集中且可回查；方法配置和对照关系明确 |
| [M24_WangWeiHao](https://github.com/didgmd/SUTResearchHub/tree/main/M24_WangWeiHao) | base/v2模块、比较/消融脚本、project_paths与experiment_protocol | 输出路径与实验协议集中管理 |
| [M24_ZhangLei](https://github.com/didgmd/SUTResearchHub/tree/main/M24_ZhangLei) | P1v1、Test、建模公式；P1v1按agent/config/physics拆分 | 模型、参数与环境分开；测试和学术说明各有位置 |
| [M25_HeSiYu](https://github.com/didgmd/SUTResearchHub/tree/main/M25_HeSiYu) | UAV主题目录含system_model、agent及test文件 | 环境与智能体分开并有针对性测试 |
| [M25_LiJing](https://github.com/didgmd/SUTResearchHub/tree/main/M25_LiJing) | 所看Practice1仅有占位脚本 | 未见可借鉴的完整实验管理结构，不据目录名推断完成度 |
| [M25_WangRongXia](https://github.com/didgmd/SUTResearchHub/tree/main/M25_WangRongXia) | 所看Practice1仅有占位脚本 | 同上，不把占位目录当作管理模板 |
| [U19_LiPengZhao](https://github.com/didgmd/SUTResearchHub/tree/main/U19_LiPengZhao/HybridPCI) | original/rewrite/modified多个脚本与CSV、dataOPT | 版本关系要明确；后续我们由Git版本与配置标识承接 |
| [U20_ChenSiXue](https://github.com/didgmd/SUTResearchHub/tree/main/U20_ChenSiXue) | MEC、车辆、场景、QNet及绘图模块平放 | 保留概念分层，归入src相应模块 |
| [U20_LiJiaNi](https://github.com/didgmd/SUTResearchHub/tree/main/U20_LiJiaNi/P3) | P1/P2/P3；P3有core/docs/experiments/results/tests/tools及结果总索引 | 借鉴一个结果入口连到阶段摘要、manifest和图，不照搬core命名 |
| [U21_LiLeTian](https://github.com/didgmd/SUTResearchHub/tree/main/U21_LiLeTian/P1) | P0.5/P1；P1功能模块及Topology.svg | 拓扑等可视化有用途，本项目统一输出到plots |
| [U22_ChenGuangBo](https://github.com/didgmd/SUTResearchHub/tree/main/U22_ChenGuangBo/P1v2) | P1v系列；所看P1v2有比较脚本、模型和比较图 | 模型与比较图需要明确身份；是否普通Git跟踪另由当前发布策略决定 |
| [U22_HanSu](https://github.com/didgmd/SUTResearchHub/tree/main/U22_HanSu/P1v3) | P1v系列；Logs及用户数/移动速度条件目录 | 场景参数必须可追溯，不能只凭图名解释设置 |
| [U22_WangRui](https://github.com/didgmd/SUTResearchHub/tree/main/U22_WangRui/P1v3) | P1v系列；功能模块与logs | 保留运行记录，后续记录配置与seed身份 |
| [U23_BuLinXue](https://github.com/didgmd/SUTResearchHub/tree/main/U23_BuLinXue/P1v5) | 轻量main、src/logs/plots/sections/knowledge；README明确不使用docs | 同学间规则有差异；我们遵从用户已选的docs主干，不照搬其例外 |
| [U23_MingYiBo](https://github.com/didgmd/SUTResearchHub/tree/main/U23_MingYiBo/P1) | 所看P1仅有初始脚本 | 未见完整实验结构，不据此提出管理规则 |
| [U23_QinYiXiang](https://github.com/didgmd/SUTResearchHub/tree/main/U23_QinYiXiang/P1v2) | P1v系列；README/docs/src/logs/plots | 说明、实现与输出分离；目录齐全不等于入口已经轻量化 |
| [U23_StatisticalModeling](https://github.com/didgmd/SUTResearchHub/tree/main/U23_StatisticalModeling) | 训练、Excel读取、网络与reward等功能模块 | 数据读取、模型和目标逻辑分别维护 |
| [U24_ChenSiQi](https://github.com/didgmd/SUTResearchHub/tree/main/U24_ChenSiQi) | src/tests/docs/logs/plots，另有data/checkpoints；README含文档地图 | 最接近本轮主干选择：当前状态、协议和下一步有统一导航 |
| [U24_DaiPeng](https://github.com/didgmd/SUTResearchHub/tree/main/U24_DaiPeng) | 仅.gitkeep占位 | 无完整模板可评估 |
| [U24_GongYuQing](https://github.com/didgmd/SUTResearchHub/tree/main/U24_GongYuQing) | 轻量main、src/tests/docs/artifacts；小型证据从运行输出中挑选 | 区分持续运行工作区与要展示/发布的紧凑证据，避免把全部缓存上传 |
| [U24_GuoJiuLin](https://github.com/didgmd/SUTResearchHub/tree/main/U24_GuoJiuLin/p1v3) | p1v系列；统一main、src内runner/路径/统计等入口 | 主入口统一、默认值集中，结果保存和统计职责明确 |

还查看了Common的目录层级，它保存共享模板、文档与日常材料。本项目不会因此复制Common或扩大受保护资料的处理范围。

总体采用的做法是：主干清楚、当前入口集中、实验身份完整、结果和图对应、运行输出按需发布。既有版本目录和平铺脚本可供理解历史，但不要求我们复制相同数量的目录或说明文件。

本轮读取清单与本地验证见[参考审计](../metadata/audits/repository_management/20261014_hub_reference/reference_survey.json)及[检查记录](../metadata/audits/repository_management/20261014_hub_reference/validation.json)。下一步等待用户提供自己的仓库，先查看目标结构与约定，再准备明确的提交范围；参考仓库继续保持只读。

## 用户指定的保存目标（2026-10-14）

用户已授权把当前进度保存到https://github.com/lijing666743/LJ；该仓库初始为空。采用独立基线提交与标签baseline-c32k-20261014，并将完整实验恢复资料保存为同标签Release资产。旧SUTResearchHub只读边界不变。Git上限50MiB，配置/源码/测试/紧凑结果跟踪，真实训练模型与大原始日志使用恢复包；已调整本地忽略规则，使有意保留的小型结果可跟踪。当前工作区HEAD与真实索引保持原状，避免影响旧checkpoint的严格身份检查。[保存与回溯说明](reproduction/RESTORE_BASELINE.md)给出使用边界。


## 当前C/D发布范围（2026-10-21）

本轮继续发布到用户指定的LJ仓库。旧main为2dc941a7f6e253bef820e9dff9120252c5f53ae3，保留baseline-c32k-20261014标签及旧Release。当前版本覆盖C/D三训练seed各40k的实现、24份活动配置、测试、论文材料及紧凑真实证据，按原目录保存；大模型、完整训练流和本地汇报工作区不进入本轮普通Git。此版本不等于完整模型恢复包，旧Release也不覆盖新增C40k及D模型。发布清单和实际远端核验以metadata/publication/20261021_cd40k为准。

采用独立临时发布副本，原实验工作区HEAD与索引保持不变。Git继续采用原始字节换行策略，避免源码指纹被自动转换；中文文档使用UTF-8，主要入口与相对链接单独检查。发布不放宽checkpoint的Git身份、设备、路径和完整状态校验，新克隆不自动具备现有模型的严格续训条件。运行命令仅在聊天提供。
