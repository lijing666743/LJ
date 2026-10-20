# QMIX CPU联合效用实现与工程验证

## 当前结果

`cpu_pair_q_v9`已完成实现和工程验证，尚未启动8k开发训练，没有新性能结果。230项回归测试及22项子测试通过（JUnit共252项记录）；另一次首轮专项运行60项通过，已被完整回归覆盖，不重复计入230项。

实际CUDA工程训练完成224步、7回合、55次优化器更新和13次target复制；online有87个参数张量变化，其中频率效用分支的四个参数张量全部变化。参数和更新数值有限，真实前史重算被使用，训练raw JSONL、aggregate JSON和Dashboard CSV一致，真实Dashboard PNG可解析。随后用保存的最终模型在CPU上进行四种方法各一回合冻结评估，评估更新为0，参数及checkpoint不变，外部过程、任务、工作量和能量守恒核验通过。短程工程结果不能用来判断新结构是否有效。

## 限定变化

旧v8将队列Q与一个全局频率Q相加；新实现保留队列Q，为每个非Idle队列分别输出一行频率Q，按所选队列取对应频率行，再将两者相加。原频率域中的0/.25/.5/1、Idle、原mask及七分支proposal接口均保留；没有引入强制服务、最高频率规则或Heuristic部署替代，也未增加actor可见信息。

该参数化可以表达同状态不同队列的相反频率偏好；执行时先对每个合法队列取合法频率最大效用，再比较队列及Idle，teacher replay按保存的队列与频率取值，Double-Q使用online选对、target评估同一对。测试覆盖独立穷举、相反偏好、批次/时间/agent轴、选中输出梯度、非有限Q拒绝、零频率回退、Idle最优、旧checkpoint拒绝及既有循环/恢复/Gate0/CLI/preflight。

总参数由1,142,112增加至1,143,660，增加1,548个（0.13554%）。除了整个频率分支之外，其余初始参数的精确哈希与v8一致。保留环境、reward、C/A/B、GAT/GRU主体、QMIX单调mixer、Double-Q、TD(lambda=0.6)、当前权重前史、agent级探索、学习率、epsilon退火及开发target间隔200。新checkpoint schema为v8，并强制检查CPU效用合同；两个加载入口均已实际拒绝旧v8实验的schema v7模型。

## 清理与保留验证

新实现验证通过后，按既有授权删除失败v8运行数据、三份旧配置、旧审计目录、两份重复旧材料及本轮工程/测试临时产物，共14个明确目标、389个文件、3,291,311,470字节。原实验结果、冻结诊断、历史对照和原测试回执以紧凑真实文件转存到`metadata/audits/qmix_cpu_pair/20261017/`。

清理脚本的一项PowerShell路径过滤表达式发生非终止报错。删除前已有独立明确路径白名单，并执行了实际路径清单、保留文件重叠及reparse检查；删除后另用独立程序逐项核验14个实际目标和全部子文件均在白名单范围内，且872项保留文件的大小与SHA256全部匹配，没有计划外删除。删除前另已核验837项旧输入完全不变、15项为本次明确且已验证的源码/测试/模板修改、4项旧输入列入清理。C/A/B、共享环境/reward/C源码和受保护目录均保留。

## 下一轮开发预算

新8k/16k独立配置已加载，8k至16k仅预算扩展检查通过；8k零步初始化的optimizer及回放为空、online=target，未加载旧模型。7份当前配置可加载，8k/16k输出根不存在。现在先从零训练seed42 8k，预期16回合、225次更新，4k/8k保存并在同样五个开发场景冻结评估；不直接启动16k/40k。

8k配置为`experiments/strong-baselines/configs/fa-gat-qmix-seed42-8k-cpu-pair-development.json`，hash为`ce44bb693b2b3db39069ebbdb1ce098b8dc82b97a1da0a35291426eb46da3d77`。主要同设置对照是失败v8的69/605及0.7719256610 J/完成；推荐16k检查仍要求超过历史v6的166个且J/完成不高于0.4324177329。该开发门槛不等于强基线最终验收或跨种子稳定性证明。详细科学协议为`sections/planning/20261017_QMIX_CPU_PAIR_Q_TRIAL.md`，运行命令只在对话提供。

实际证据：审计目录中的`regression_tests.xml`、`initialization_check.json`、`cuda_validation.json`、`preservation_before_cleanup.json`、`deletion_inventory.json`、`deletion_result.json`及`preservation_after_cleanup.json`。
