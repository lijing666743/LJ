# Codex 工作约定

本文件规定 Codex 在本仓库中的工作方式。所有修改都应遵循以下边界：

- `.agents/` 和 `knowledge/` 是受保护目录，不得自动修改、删除、移动、重命名、覆盖或批量处理其中的内容。
- 修改任何已有文件前，必须先读取相关文件，并只补充缺失内容。
- `README.md` 只维护项目的宏观进度、范围和阶段性说明。
- 论文公式、章节材料和详细学术内容统一放入 `sections/`。
- 真正的源代码必须放入 `src/`；`main.py` 必须保持轻量，只作为程序入口。
- 测试代码放入 `tests/`。
- 图片输出到 `plots/`；Dashboard CSV 输出到 `dashboard_logs/`；普通运行日志输出到 `logs/`。
- 禁止伪造实验结果、指标、CSV、PNG 或其他运行产物。
- 禁止在没有测试的情况下声称模块已经完成。
- 每完成一个任务，都要更新 `README.md` 中的进度说明。
- Python 文件、类和函数使用英文命名；README 和论文说明使用中文。
- 公式使用标准 LaTeX，并为符号提供清晰定义。
- 不得随意扩大研究范围；若需变更技术路线或任务边界，应先记录并确认。
- 当前项目已切换至 Environment Implementation 阶段。允许开始实现 configuration、CLI、registry、runner、task lifecycle、queues、mobility、topology、channel、stale CSI/history、interference measurement、deterministic executor、communication service、CPU service、energy accounting、observation、reward、reset、step、metrics、Gate 0 tests、environment sanity、random policy 和 heuristic policy；Gate 0 通过前不得启动正式 RL training，不得声称 MAPPO/QMIX 已完成或生成正式训练结果。
