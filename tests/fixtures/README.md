# C/D 共用实现的测试夹具

本目录只存放工程回归输入，不作为正式实验入口、正式结果或预训练数据。

- configs/ 中包含历史配置，用于验证当前共享CPU、TX、P1a信用、Q输入及统一Route接口。保留其原内容和配置hash；本轮只迁移测试引用路径。
- historical/checkpoints/ 中两份 step_4000.pt 是从旧run逐字节复制的原始checkpoint，供共享学习链测试读取pending-rollout张量和兼容性校验。C/D当前实验不从这些模型初始化；历史名称不表示A/B仍是活动实验。
- 原夹具来源路径、SHA256及迁移说明记录在 metadata/ABC_RELEASE_MANIFEST.json 的 test_dependencies 字段。
- configs/qmix_legacy/ 中三份旧QMIX配置是固定回归输入，保留原始字节与配置hash，已从活动实验目录迁出。对应十个测试模块只更换读取路径，迁移清单见 metadata/cleanup/20261020_deep_cd_cleanup/plan.json；它们不作为D的新训练候选。
- 测试中的人工terminal标签仅用于检查工程行为，不能当作新收集的实验数据或学习效果。

当前C配置在 experiments/task-allocation/configs/，D配置在 experiments/strong-baselines/configs/；真实产物分别位于 logs/C/ 和 logs/strong_baselines/development/c_feature_credit_qmix/。A/B专属实验已退役。
