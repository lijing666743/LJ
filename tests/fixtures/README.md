# 当前 A/B/C 的测试夹具

本目录只存放工程回归输入，不作为正式实验入口、正式结果或预训练数据。

- configs/ 中包含历史配置，用于验证当前共享CPU、TX、P1a信用、Q输入及统一Route接口。保留其原内容和配置hash；本轮只迁移测试引用路径。
- historical/checkpoints/ 中两份 step_4000.pt 是从旧run逐字节复制的原始checkpoint，供B/C学习链测试读取pending-rollout张量和兼容性校验。当前A/B/C正式训练不从这些模型初始化。
- 来源路径、SHA256及迁移说明记录在 metadata/ABC_RELEASE_MANIFEST.json 的 test_dependencies 字段。
- 测试中的人工terminal标签仅用于检查工程行为，不能当作新收集的实验数据或学习效果。

正式配置在 experiments/task-allocation/configs/，真实seed42训练与评估产物在 logs/{A,B,C}/seed42/8k/。
