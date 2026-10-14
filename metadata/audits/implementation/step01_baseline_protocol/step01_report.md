# 合理任务分配第1步交付：基线身份与实验协议

日期：2026-10-12。状态：第1步完成；第2—10步未执行。

本步将“合理分配以增加全系统DONE”的目标落实为可追溯的比较协议。没有实现新策略，也没有产生新方法训练或评估结果。未使用FULL-1k表现制定或验证方法；种子核查仅读取身份/配置字段。

## 已固定的基线

当前工作区HEAD为 `eb1140cefc4a0ff79b8bd0c05e73c7a23528d7ba`，存在原有未提交修改。HEAD不能独立复现当前源码，已保存开始时Git状态与214个文件SHA-256；不将既有dirty改动归为本步修改，也不声称当前源码与历史训练源码完全一致。

P1a方法配置的canonical hash为 `2f143a85c0e37f605bbf38b59b3a31267a305464179593d9828c7c0a1518e3fa`。原配置、历史snapshot与final checkpoint中的配置身份一致；E13两个32k模型也分别通过同样核对。模型仅以CPU restricted loader读取元数据，未创建actor、加载优化器或恢复训练。

| 历史参照 | 采集主环境transitions | 实际优化transitions | PPO更新 | 未用尾部 | final SHA-256前缀 |
| --- | ---: | ---: | ---: | ---: | --- |
| P1a seed42/8k | 8000 | 7936 | 31 | 64 | d303878ed57d |
| E13 seed42/32k | 32000 | 32000 | 125 | 0 | 8d0f66bfbf46 |
| E13 seed43/32k | 32000 | 32000 | 125 | 0 | 25c576fe6974 |

历史4k checkpoint、三个final、原配置与snapshot均保持原内容。完整哈希、训练状态与runtime provenance见 `baseline_identity.json`。旧P1a只作为方法参照；新A是否可复用历史A42须在后续确认实现身份，否则同预算从零重建。

## A/B/C差异与约束

- A：原P1a层次Route及原候选特征。
- B：统一Local/每个Remote的候选评分与真实leaf采样/argmax，复用旧特征语义，只做必要兼容接线。
- C：在B上增加可见的排队、计算、传输、期限和能量关系；若旧Local显式负担不完整，其修正归C。

原环境、reward、CPU gate、TX paired-shadow和训练超参数固定，不设Local占比奖励或验收下限。A→B是结构及必要接口消融，B→C是特征消融。B/C的 `unified_node_v1` 仍是待实现接口，本步没有发布含该字段的可运行配置。

已从配置与源码核对：actor/critic学习率均0.0003，Route head倍率0.1，rollout256、recurrent chunk32、PPO epochs4；Route私有backbone保留。任务信用有效系数1；配置虽含CF系数1，但P1a选择task_event，实际CF actor系数为0。Route-Q辅助损失系数0.5保留。探索log-barrier初始系数0.05、退火6000步。destination decoder/supervision均legacy，oracle rank系数0；所有省略的MAPPO默认字段也已单独保存。

所有新run独立随机初始化。禁止历史模型加载作为初始化、8k模型续成新的32k实验、离线预训练、专家蒸馏、oracle标签训练及简单场景预训练迁移。TX paired-shadow是原有在线信用机制，保留并单列额外仿真成本。

## 种子、预算与判断

开发训练seed42/43/44，开发评估seed1042—1046；已有1044 oracle属于已见诊断，不作为训练标签或未见测试。32k确认用训练seed42—46；预留评估seed20042—20061。

已扫描84份现有logs manifest、config snapshot、oracle summary、实验JSON配置与顶层metadata的结构化seed字段，未发现预留20个种子。配置声明也保守计为使用。该结论只覆盖扫描记录，不能证明仓库外或已删除、未记录的运行未用过。扫描路径、哈希与抽取值见 `seed_usage_audit.json`。版本冻结后才使用预留种子，不能用它调参或挑模型。

开发最多9个8k run，主环境预算72000 transitions；确认A与胜出B/C各5个32k run，共320000 transitions。二者均不含评估、fixture及TX shadow；额外成本单列。主要比较只用预算终点final，不按测试成绩选checkpoint。

开发每个训练seed先算五个评估episode完成率的均值，再求相对A的配对差。继续条件为配对差总体均值>0且至少2/3训练seed为正；B/C都通过时选开发均值更高者，完全相同则选B。该门槛是继续投入的操作规则，不是显著性证明。原始DONE/Generated、任务守恒、过期阶段、Local/Remote绑定与系统服务成本均需同时报告。

## 本步核验与交付

19项身份/协议一致性检查通过，214个清单文件全部保持相同SHA-256；84份种子审计证据再次校验一致。具体结果见 `validation.json`。本步没有源码改动，因此未运行模型单元或集成测试；该PASS不代表新decoder、训练链路或方法有效性已经通过测试。

新增文件仅在本目录：`baseline_identity.json`、`experiment_protocol.json`、`protected_files_sha256.json`、`seed_usage_audit.json`、`git_status_before.txt`、`validation.json`与本报告。已有文件只补充README进度与实施计划状态。源码、测试、实验配置、历史checkpoint未改动；`.agents/`与`knowledge/`未遍历处理或修改。未启动训练、策略评估、oracle或预训练。

下一步：按计划第2步建立候选ID/mask与概率接口，审计Local实际共享CPU负担，保持旧路径兼容；完成相应测试后再进入评分器和PPO适配。所有新方法效果仍未知。
