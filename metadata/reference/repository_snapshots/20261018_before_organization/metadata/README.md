# 身份与工程记录

这里的内容用于核对当前版本，不作为额外训练结果。

| 位置 | 用途 |
|---|---|
| `ABC_RELEASE_MANIFEST.json` | A/B/C配置、源码、final模型、评估及现址 |
| `ABC_FILE_MANIFEST.json` | 当前本地文件的SHA256清单 |
| `DIRECTORY_RELOCATION.json` | 本次文件old→new路径与迁移前后校验；删除条目另列 |
| `audits/seed42/` | Step07A原始取证JSON，数值与原路径保留；旧路径用迁移表查找 |
| `audits/implementation/` | Step01—06最终报告、方法初始化/兼容与保护检查；旧调试迭代和源码副本已删除 |
| `reference/p1a_seed42_1044/` | A baseline与历史P1a-1044对齐所需的最小原始证据；不用于训练初始化 |
| `cleanup/20261013/` | 本次清理清单、删除记录、模型加载及完整工程回归结果 |

旧审计中的路径和“当时文件存在/验证通过”等状态描述的是其执行时刻，不能当作整理后路径。当前存在性和字节一致性以最终清理验证为准。清理过程中曾创建的压缩包已按用户要求删除；相关旧校验记录仅描述当时操作，不表示现在仍有归档。

论文0—7章保持原样，其他当前方法说明在sections/。测试必须使用的旧张量放在tests/fixtures/，不在logs/制造第四组实验。

## 2026-10-14 当前导航

`ABC_RELEASE_MANIFEST.json`、`ABC_FILE_MANIFEST.json`保留原8k发布身份；当前本地整理另见`cleanup/20261014/`，不得把旧清单理解为包含续训后所有文件。

- `audits/continuation_implementation/`与`audits/continuation_launch/`：完整恢复实现和两次续训准备证据。
- `audits/checkpoint_progress/20261013_4k8k12k16k/`：A/B/C开发阶段比较。
- `audits/c_heuristic_gap/`、`c_heuristic_timeline/`、`c_route_energy_inputs/`、`c_energy_progress/`：C16k任务与能耗诊断。
- `audits/c32k_progress/20261014_16k32k/`：C32k完整性、能耗和逐场景进度。
- `audits/group_meeting_summary/20261014_scene_fix/`：当前有效修正版图表验证；原20261014审计为历史记录。
- `cleanup/20261014/c_continuation_retention.json`：本轮实际读取的C32k全部checkpoint状态。

清理前Git工作区已有大量修改和删除，其中包括knowledge内的既有删除状态；本轮不恢复、不处理这些受保护文件，基线状态单独记录，不能归因于本轮清理。

当前本地身份快照见[CURRENT_EXPERIMENT_MANIFEST.json](CURRENT_EXPERIMENT_MANIFEST.json)，覆盖13份正式配置、107个src模块和20个实验checkpoint；本轮结论见[清理报告](cleanup/20261014/cleanup_report.md)。该清单记录20261014时点，后续续训应创建新的身份记录。

方法说明后续归类与公式排版记录见`cleanup/20261014_sections_formatting/`：旧sections根文件已移入methods/history，旧清单中的路径通过relocation_receipt.json定位，历史审计身份不覆盖。[现行材料索引](../sections/README.md)提供直接入口。此前离线HTML当前不在工作区，历史渲染校验仍保留其执行时刻的记录。
