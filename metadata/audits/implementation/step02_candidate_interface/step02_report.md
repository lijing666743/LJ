# 第2步交付：统一候选信息与概率接口

日期：2026-10-12。状态：第2步工程gate通过；第3—10步未实施。新统一评分器和PPO接线仍待实现，当前没有新方法效果结论。

## 本步交付

新增 `src/models/route_candidates.py`：

- `RouteCandidateBatch`对齐Local/Remote的真实节点ID、原环境动作索引、合法mask和原有语义特征；可携带私有Route端点嵌入、recurrent特征和hidden。
- `build_route_candidates`保留候选身份。Local执行节点为自身；候选视图可以排列，但动作索引仍对应原目的地。无Remote、单Remote、无任务和无效公共消息均显式处理。
- `RouteLeafResult/route_leaf_result`保留实际选择、chosen leaf log-prob、叶argmax、Local/Remote/Defer组质量、组质量argmax及条件Remote概率。非法叶概率为0，合法叶总和为1；无Remote时条件概率为0。
- `visible_local_cpu_burden`汇总可见的本地与接收CPU队列，返回原tensorizer归一化单位下的任务数与cycles；不接入旧A/B评分。

`CAGATMAPPOActor.forward_with_route_interface`是显式启用的只读视图入口。默认forward不创建这个视图。生产动作分布输出新增 `route_leaf_result()`，直接读取已选proposal，不重新选择动作。该proposal不替代环境执行后的拒绝/降级记录。

Local和各Remote节点视图不包含Idle/Defer；它们在完整Route叶分布中保持原控制动作身份。Remote公共消息有效性与环境合法性分别保留；无效公共消息不会被即时helper私有状态替换。

`unified_node_v1`已在配置枚举登记，但配置校验明确拒绝启用，避免接口尚未接通时进入训练。旧模式默认值及canonical hash不变。本步没有新decoder参数或新的采样规则；新模式的确定性leaf argmax属于第3步。

## CPU语义核查结论

旧显式Local burden只统计本地队列；Remote burden使用对外广播的完整CPU负担，因此两端显式口径不完全对称。完整self输入已经包含接收CPU队列，不能据此声称旧模型完全不可见这些任务。

`cpu_by_source[ego]`已经复用本地队列，正确汇总不能再额外加一次 `local_cpu`。工程夹具通过真实生命周期和观测构建器验证：本地100 cycles加两个收到的200/300 cycles任务，CPU总负担为3个任务、600 cycles；发出TX的400 cycles与UNBOUND的500 cycles不计入。旧显式Local burden继续为100 cycles，以保证本步没有改变A/B特征语义。

完整负担及统一物理单位关系按协议留到C组第5步。旧pressure不是秒；排队cycles和单任务cycles使用不同归一化尺度，后续必须先还原各自物理量再组合。详细证据见[语义核查](../../../../sections/route_candidate_interface_cpu_audit.md)。

## 实际验证

| 检查 | 实际结果 |
| --- | --- |
| 新接口测试 | 最终版本35项全部通过，包含2/3/4/5 UAV、排列与mask、信息边界、CPU唯一计数、概率极值、实际选择及私有hidden/梯度 |
| 首轮专项 | 33 passed；此前3个失败来自新增测试夹具的旧构造器参数顺序，已修正 |
| 12个模块的目标回归 | 237 passed、137 subtests passed；1个子测试因旧CLI假设所有枚举值可运行而失败 |
| 修正后的受影响模块+新接口复验 | 47 passed、5 subtests passed；四种旧模式继续接受，新保留模式明确拒绝启用 |
| 三个历史模型 | P1a seed42/8k、E13 seed42/43/32k actor/critic均严格加载；改动前后初始state、全部head logits、hidden及加载模型输出的字节指纹相同 |
| 历史配置与保护清单 | P1a canonical hash保持原值；原214个清单文件中210个不变，仅4个授权源码/测试文件变化 |

回归中的唯一失败已在最后复验覆盖并解决。修正只涉及旧测试对新保留模式的预期，生产源码在目标回归和最后复验之间没有变化，因此没有再重复整组回归。不能把最初含失败的回归描述成单次全部通过；完整stdout、JUnit XML、执行命令与综合核验见本目录记录及[validation.json](validation.json)。

两个早期回归尝试因临时文件创建长时间不返回而停止，没有记作通过。单独faulthandler记录定位到 `tempfile._mkstemp_inner -> NamedTemporaryFile -> atomic_save_checkpoint`；把TEMP/TMP/TMPDIR及pytest basetemp定向到本步工作区专用目录后，相关检查点测试和目标回归顺利完成。没有修改生产checkpoint保存逻辑。诊断原文保存在 `checkpoint_fixture_diagnostic.txt`。

测试包含可丢弃的合成PPO/Adam与短环境夹具，属于工程核验；其权重、任务标签或输出均不作为训练数据或新方法实验成绩。本步未启动新策略实验训练、预训练、蒸馏、oracle生成或历史模型整episode评估。新接口主要在CPU验证，已有回归也覆盖旧路径CUDA分支；本步不声称新统一模式已可执行GPU训练。

## 改动与保护

已有文件只补充所需内容：

- `src/config.py`：保留模式枚举与明确的暂不可运行校验。
- `src/models/ca_gat_mappo.py`：默认关闭的候选读数入口和可选输出字段。
- `src/models/ca_gat_mappo_actions.py`：实际选中proposal的概率读数方法。
- `tests/test_candidate_aware_route_decoder_v1.py`：补充保留模式CLI拒绝断言，旧模式断言保留。
- README与实施计划：追加第2步进度。

新增候选模块、新接口测试、CPU语义材料及本目录工程记录。环境物理、reward、CPU/TX机制、原实验JSON配置、历史snapshot/checkpoint、依赖与main.py保持原内容；未遍历处理或修改`.agents/`、`knowledge/`。

本次变化与已有dirty改动已分离：只逆向移除本步新增段落，3个生产源码文件与1个旧测试文件均可在内存中恢复至第1步记录的SHA-256，没有实际回退任何文件。可审阅 `implementation_changes.patch` 与 `legacy_test_expectation.patch`，不将整个Git dirty diff归为本步修改。更新后的216个源码/测试/历史文件保护清单见 `protected_files_after.json`。

关键交付：`before_compatibility.json`、`historical_compatibility_final.json`、`regression_tests.txt/xml`、`final_focused_tests.txt/xml`、`validation.json`及两份本步差异patch。`test_tmp`、`pytest_tmp`、`pytest_retry_tmp`仅承载工程测试临时文件。

## 下一步入口

第3步复用现有共享评分逻辑，让Local与各Remote在合法叶上直接比较，并明确Defer/Idle控制项。节点视图评分输出必须按 `action_indices` 对齐回原环境叶索引，不能在候选重排后重新编号目的地。继续保护私有backbone与共同模块初始化RNG；评分器、采样与确定性选择先过测试，第4步才接入P1a信用/PPO/Q辅助等学习链路，第6步通过前不开始实验训练。
