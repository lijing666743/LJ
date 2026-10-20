# C/D现有实验与恢复证据

更新日期：2026-10-21。这里只登记保留对象与恢复边界，运行命令继续仅在聊天提供。

## 当前保留

- C seed42/43/44各40k模型、所有中间模型、原始训练日志、评估和配置快照。
- D（CFC-QMIX）seed42/43/44各40k、五段来源状态、每seed十检查点冻结评估，均已核验；当前结果见三seed报告。
- Local、Heuristic、Random及共用环境、任务生命周期、路由/TX/CPU实现和测试夹具。
- 依赖版本仍由environment/requirements-abc-tested.txt记录；其旧文件名不表示A/B仍是活动实验。

## 恢复边界

C seed42既有预算扩展依赖periodic状态恢复、旧边界重放核对及pending rollout保留，不能仅加载final权重就称为精确续训。C的原8k、16k、32k、40k来源及全部周期模型保留；任何新预算仍需事先核验实际入口合同。C seed43/44独立训练，不从A/B权重初始化。

D采用v10完整online/target/Adam/replay/随机流/信用状态恢复；现有五段预算和shadow窗口边界属于协议，不能未经核验改为一次直跑40k。来源与数值核验见metadata/audits/qmix_c_matched/20261019和20261020。

## 共享历史与退役边界

原ABC_RELEASE_MANIFEST.json与ABC_FILE_MANIFEST.json保留原始字节，供C原始身份及测试夹具追溯；它们是历史清单，A/B删除后不能当作当前文件存在性清单。P1a、统一路由和35维特征的共用实现仍被C/D依赖，未修改或删除。A/B专属实验及入口按用户要求退役，详见[清理清单](../../metadata/cleanup/20261020_retire_ab/plan.json)。

深度整理另移除A/B两个专属本地恢复ZIP、旧混合导出及无依赖的QMIX草案。C和support恢复ZIP保持原始字节；BASELINE_RECOVERY_MANIFEST.json仍描述2026-10-14发布时点，不能用其A/B条目推断本地备份仍在。三份旧QMIX回归配置移入tests/fixtures/configs/qmix_legacy/，原始哈希与文件映射见[本轮清单](../../metadata/cleanup/20261020_deep_cd_cleanup/plan.json)。

当前可用结果见[实验状态](../results/CURRENT_EXPERIMENT_STATUS.md)，方法与协议见[论文材料索引](../../sections/README.md)。源码指纹、配置、模型及来源身份必须匹配；不覆盖有效产物，不把工程验证计作正式实验。
