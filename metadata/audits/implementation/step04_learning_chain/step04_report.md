# 第4步交付：P1a学习链、日志与恢复

日期：2026-10-12。状态：本步工程gate为PASS；第5—10步未执行。没有正式实验训练、预训练、oracle生成或新方法效果结论。

## 已接通的行为

`unified_node_v1`通过正常配置与actor构造器建立，不再仅限第3步probe。新评分器使用独立初始化子域，保留历史层次head的原随机消耗顺序，随后替换该head；不加载任何历史权重。旧共享特征、CPU/TX、reward及环境机制保持原值。

Route在PPO中仍只占一个logical component，old/new log-prob、ratio和clip对应实际chosen叶。P1a actor保持shared+task-event，task有效系数1、CF有效系数0；normalized Q辅助系数0.5和原损失、目标、chosen-only监督不变。终局、Defer binding bootstrap、frontier与删失规则复用已有实现。

新增统一评分器训练用411维候选表示，沿用“候选嵌入+合法Remote平均上下文+类型指示”的原布局。导出表示detach，Q集中状态入口detach；未选叶是预测，不作真实标签。新模式继续要求相应P1a结构校验，不把旧option-aware模式的shared GAE限制直接删除，也不自动降级信用。

原log-barrier保持Local/Remote总质量/Defer分组、0.05初始系数和6000真实transition退火。组质量只供正则与统计，不控制实际叶动作。

v2日志记录真实task ID、候选节点/动作索引、mask、分数与叶概率、实际chosen和leaf argmax，另列组质量。统一模式的旧`actor_top_level_argmax`为null；汇总按真实叶argmax分类。旧schema缺失字段由兼容读数保留None/NA，不补造历史数据。

新checkpoint必需`route_decoder_identity`，明确decoder、原特征、411维Q表示、叶决策及初始化身份。旧模式字段布局不变，新模式缺失/篡改身份拒绝加载；保留现有Adam分组、GRU双lane、policy RNG及pending rollout恢复。

## 实际验证

| 检查 | 结果 |
| --- | --- |
| 新学习链专项 | 28 passed，包含CUDA实际叶PPO/Q梯度 |
| 13个模块相关回归 | 259 passed、53 subtests passed，530.49秒 |
| 三项补强复验 | 3 passed、25 deselected；非零CF、首epoch ratio及推进过的RNG恢复 |
| 实际叶PPO | Local、Defer和每个Remote均有生产loss梯度；ratio=1、ratio=2的clip、inactive agent检查通过 |
| Q隔离 | 411维表示与类型对应正确；候选/集中状态detach，只有chosen预测有监督梯度；非零CF在beta=99时仍不进入task-event actor |
| 四轮冻结 | task/Q目标、frozen Q、CF诊断和最终Route advantage跨4个epoch不变；首轮实际chosen ratio约为1 |
| 退化与正则 | 无Remote、单Remote、Idle退化、极低分数及6000步退火结束有限；无虚假inactive Route贡献 |
| 日志 | 真实生命周期任务ID、leaf/group分离、旧schema缺失为NA、记录开关不改变后续采样；JSON有限性通过 |
| 保存/恢复 | 非空actor/critic Adam、推进17次随机数后的policy RNG、160条pending rollout及GRU恢复；补满rollout后继续4轮更新，模型与对照逐张量一致 |
| 新actor-only加载 | 绑定实际工作区Git身份的随机初始化合成checkpoint严格加载且文件hash不变，无episode评价 |
| 历史兼容 | P1a seed42/8k、E13 seed42/43/32k结构化校验及actor/critic strict load通过，初始/加载后state和所有head/hidden指纹与第3步相同 |
| 初始化与配置 | seed42/43/44共同actor/critic张量逐个完全相同；P1a工程配置resolved差异仅Route模式，原canonical hash不变 |

共有287项不同pytest测试和53项子测试通过。初期81项复验包含在后续259项回归内，不重复相加。28项专项通过后，为补强恢复测试增加了三个断言，仅重跑受影响三项；其余25项测试及生产数值路径没有变化。本步没有执行整仓库完整gate，该工作仍留第6步。

首次学习链专项有12个失败、4个通过：夹具使用普通环境观测，缺少P1a CPU防饥饿所需的精确deadline context。改用相同P1a工程配置生成观测后，第二次为6个失败、10个通过，剩余失败是测试未提供配对TX信用的显式advantage/mask。补齐后第三次为17个通过、1个失败：测试把所有policy component都置inactive，而原PPO按契约拒绝全inactive目标。改为仅一个agent的Route inactive、其余agent真实active后，最终28项通过。上述修正只涉及新增夹具，不改变CPU/TX或PPO机制；失败stdout/XML均保留。

新增actor-only加载初次探查被原provenance guard拒绝，因为合成resume夹具的Git身份为unknown。保留拒绝记录；使用新建的合成checkpoint绑定实际HEAD及dirty状态后通过，没有放宽加载校验或修改原夹具文件。

## 工程夹具边界

专项的PPO/Adam/恢复检查使用可丢弃的合成目标与重复观测输入，不生成实验成绩。部分输入复用既有8k checkpoint里的观测/rollout夹具，但没有把历史model state加载进新actor/critic；当前新模式参数始终新随机初始化。夹具更新产生的权重、任务标签和buffer均不会用于后续实验的初始权重或训练集。

历史P1a/E13模型只在独立兼容检查中加载，使用持久化张量比较，无整episode策略评价。相关旧测试含短环境与合成更新，均属于工程回归；没有启动新的8k/32k实验，也没有根据任何评估结果选模或调参。

## 改动与保护

本步改动13个已有受保护清单文件：

- `src/config.py`：新模式组合校验，拒绝destination监督/预训练。
- `src/models/ca_gat_mappo.py`：正常构造器初始化公平性与detach候选Q表示。
- `src/models/ca_gat_mappo_actions.py`：输出真实decoder身份。
- `src/models/unified_route.py`：新checkpoint身份及工程probe说明。
- `src/models/route_exploration.py`、`route_task_credit.py`、`route_option_q.py`：真实任务/叶日志与v2更新schema。
- `src/models/ca_gat_mappo_trainer.py`：从决策前观测传递真实队首task ID。
- `src/models/ca_gat_mappo_checkpoint.py`：新身份保存/严格校验与actor/config一致性。
- `src/evaluation/route_telemetry.py`：新模式叶记录与summary v2。
- 三个既有测试文件：补充新模式由保留状态转为可构造后的预期，旧模式断言保留。

新增`src/models/route_decision.py`、`tests/test_unified_route_learning.py`。README仅追加宏观进度；实施计划追加最新进度；详细方法放[第4步说明](../../../../sections/unified_route_learning_step04.md)。没有修改PPO目标函数、updater、Q归一化、环境物理或CPU/TX执行模块。

开始时218个文件全部匹配第3步清单；结束时205个原文件不变，13个授权源码/测试变化，加2个新源码/测试，清单为220文件。原实验JSON、所列历史checkpoint/snapshot、main.py和AGENTS保持原内容；`.agents/`及`knowledge/`未修改或批量处理。差异由已核验的本步前快照生成，不把原dirty改动归为本步工作，没有整体回退文件。

沿用既有Python/torch/numpy及requirements，不增依赖，不改CLI和项目seed协议。相关技能的通用默认值由本项目已接受协议覆盖，外部共享环境说明路径仍不可用，记录于`runtime_and_scope.json`。

## 证据与下一步

关键产物包括`validation.json`、`protected_files_after.json`、`implementation_changes.patch`、`learning_final.txt/xml`、`strengthened.txt/xml`、`regression.txt/xml`及执行命令；身份与兼容证据为`historical_compatibility.json`、`formal_configuration_and_initialization.json`和`synthetic_actor_loader_check.json`。测试临时checkpoint均留在本步专用fixture目录。

下一步为第5步C组显式物理关系特征：完整且不重复的Local CPU负担、统一单位的排队/计算时间代理、Remote传输代价、期限余量与能量关系。B仍保留原特征作为独立对照。第6步完整工程gate及A/B/C配置发布完成后，才开始既定从零8k实验；工程测试通过不证明合理分配已学会或任务完成更多。
