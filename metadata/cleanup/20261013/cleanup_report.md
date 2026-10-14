# 当前A/B/C本地清理报告

完成日期：2026-10-13。本轮按用户要求整理本地目录，继续保留后续实验能力；不处理GitHub、Git LFS或上传设置，不保留压缩包。

## 最终目录

- logs/仅有A、B、C三组及目录说明；每组seed42/8k下分training、evaluation、diagnostics，模型在training/checkpoints。
- dashboard_logs/和plots/仅保留A、B、C实际CSV/PNG及原.gitkeep；旧experiments空树、step06工程测试临时CSV/PNG均删除。
- docs/只保留本次完整目录说明、reproduction与results/seed42；旧design及e13三份报告核对引用后删除。
- metadata/集中保存身份、路径迁移、Step01—06最终证据、Step07A取证、历史P1a最小对齐输入和本次清理/回归记录。
- tests/fixtures的9份旧配置和2份pending-rollout checkpoint保留，供当前共享学习链测试；不是训练数据、正式实验入口或预训练初始化。
- sections/0—7章与当前方法说明保留；knowledge/、.agents/及原.git未修改。个人IDE配置保留。

逐个文件夹的说明见 [完整目录说明](../../../docs/DIRECTORY_GUIDE.md)。

## 清理数量与归档状态

早前旧实验清理：914个原文件、71个缓存文件、565个空目录；Windows超长路径审计追加删除63个旧残留文件。原计划与长路径清单保留在本目录供核对。

此次目录再整理：120个文件按SHA256核对后迁移；删除清单另有160个文件条目，包含153个旧工程迭代/源码副本、3份过时文档、1份已作废打包清单和3个压缩包。另清掉112个空目录（与上述早前数量分开记录），删除3处工程测试临时目录。

三个压缩包合计2,648,491,962字节，已删除，不存在可恢复归档。此前archive_validation等文件记录的是创建/验证当时的操作，不能解释为当前仍有ZIP。当前没有再次打包。

## 内容与功能验证

| 验证 | 实际结果 |
|---|---|
| 完整工程回归 | **1261 passed，338 subtests passed**，2038.98秒；0 failure、0 error、0 skip |
| 冻结源码/配置/0—7章/正式结果 | 178项SHA256全部一致 |
| 原Step07A取证输入 | 49项经迁移表定位，内容hash一致 |
| 关键数字来源 | 446项来源条目涉及的26个输入文件均可定位 |
| A/B/C原配置身份 | 三组seed42原config hash不变；九份正式配置均保留原内容 |
| 三个final模型 | 新路径下严格CPU加载成功，checkpoint SHA256不变 |
| 评估及Route结果 | aggregate与原保存期望逐字段一致，Route计数一致 |
| 独立目录可加载性 | 此前在独立拷贝及其真实新Git commit中，三组show-config与final加载通过；临时拷贝已删除 |
| 当前目录收尾 | 空目录、ZIP和测试残留的最终检查见final_validation.json |

测试是已有工程回归，含小预算PPO与短环境评估夹具，不是新的8k正式实验。其关键证据JSON单独保留，模型/CSV/图片等临时输出已删除；JSON里的临时路径只标记测试来源，不要求这些目录继续存在。外部TEMP启动器在当前沙箱中被重定向，未启动测试；之后使用显式隔离的仓库内临时目录完成整套回归，没有修改生产源码绕过限制。

## 实验边界与后续使用

未改变环境、reward、PPO、Route结构、模型权重、正式配置或论文0—7章；未启动新的正式训练、正式episode evaluation、预训练或oracle。只读加载模型用于验证文件搬迁可用。

A/B/C现有单training-seed结果仍为A>C>B，B/C尚未超过A。本次整理不产生新的性能结论。历史P1a只用于1044对齐，不当作独立训练重复；缺少原始启动命令/初始权重摘要的证据限制继续保留。

未来实验按 [后续实验命令](../../../docs/reproduction/REPRODUCE_ABC.md#后续实验怎样放置) 使用新的组别/seed/预算/时间目录。原配置为保留历史hash未改输出字段；直接运行原配置会重建旧层级，使用文档给出的输出覆盖即可。每次运行保留程序真实生成的新身份，不能冒充历史run_id。

Step07A报告仅更新链接并增加迁移说明，表格数字和结论不改；其CSV和原始取证JSON不改。原始snapshot/manifest中的旧路径通过 [路径对照表](../../DIRECTORY_RELOCATION.json) 查找。完整文件hash清单为 [ABC_FILE_MANIFEST.json](../../ABC_FILE_MANIFEST.json)。
