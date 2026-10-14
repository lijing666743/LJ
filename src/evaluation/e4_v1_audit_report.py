"""Concise Chinese audit report rendered from actual saved-artifact summaries."""
from __future__ import annotations

from pathlib import Path


def fmt(value, digits=4):
    return "NA" if value is None else f"{value:.{digits}f}"


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *["| " + " | ".join(map(cell, row)) + " |" for row in rows]])


def write_report(reports, provenance, root):
    r3, r4 = reports["E3"], reports["E4"]
    delta = r4["formal_mean"]["completion_ratio"] - r3["formal_mean"]["completion_ratio"]
    assert r4["formal_mean"]["completed"] == 0 and delta < 0
    text = ["# E4-v1 seed43 vs E3 seed43 matched evaluation audit", "",
        "**唯一 Gate 判定：E4-E，正式性能明显恶化，淘汰 E4-v1，不扩 seed。**",
        f"五集 mean completion：{fmt(r3['formal_mean']['completion_ratio'] * 100, 2)}% → 0%；"
        "Remote lifecycle 未恢复；final Route 全部 Defer，使 TX 与 CPU 失去访问机会。不能把空 TX 分母解释为 TX Idle collapse 已解决。",
        "",
        "本轮只读取已有训练、checkpoint、formal evaluation 和诊断产物，进行离线统计/制图/测试；没有 training、resume、evaluation、actor forward、optimizer step 或算法修改。"
        "五个 evaluation seeds 是同一训练 seed 的五个测试 episode，不是五个 training replicates，也不作显著性/跨 seed 结论。",
        "", "## 1. Identity Gate：PASS", "",
        table(["项", "E3", "E4-v1"], [
            ["Training config SHA-256", r3["identity"]["config_hash"], r4["identity"]["config_hash"]],
            ["Final checkpoint SHA-256", r3["identity"]["checkpoint_sha256"], r4["identity"]["checkpoint_sha256"]],
            ["Source training commit", r3["identity"]["git_commit"], r4["identity"]["git_commit"]],
            ["seed / collected = optimized / episodes / PPO updates", "43 / 32000 / 64 / 125", "43 / 32000 / 64 / 125"],
            ["Final", "FINAL_COMPLETED；tail=0，active rollout=0", "FINAL_COMPLETED；tail=0，active rollout=0"],
            ["Evaluation seeds / horizon / decoding", "1042–1046 / 500 / deterministic masked argmax", "1042–1046 / 500 / deterministic masked argmax"],
            ["Evaluation device / dtype", "CPU / float32", "CPU / float32"],
            ["Evaluator git dirty", r3["identity"]["evaluator_git_dirty"], r4["identity"]["evaluator_git_dirty"]]]),
        "", "Canonical config 严格比较只发现一个 treatment（legacy 默认省略先归一化为 legacy）：", "",
        "```text\ntraining.mappo.cpu_frequency_decoder_mode: legacy -> candidate_aware_v1\n```", "",
        "未排除 output 路径或 telemetry 来隐藏差异。Environment、reward、critic、GAE、PPO、LR、entropy、Route/TX/CPU queue、mask、executor、预算及其余 canonical 字段一致。"
        "checkpoint 中 actor/critic/optimizer/RNG payload 完整，训练状态及 trace 125 update-end、500 PPO epoch（optimizer identity 1–500）吻合。读取 optimizer_state 只用于完整性检查，未执行优化。", "",
        "两个 evaluation resolved config hash 同为 `" + r3["identity"]["manifest"]["evaluation_config_sha256"] + "`。"
        "训练身份来自 evaluation manifest 的 source checkpoint/config/commit，而不是将 evaluation cfg hash 当 training hash。"
        "五个 seed 的外生 trace hash 逐项匹配，两模型使用相同环境随机轨迹和评估协议。", "",
        "E4 evaluator provenance 为 git_dirty=True：commit 不能独立代表当时完整源码。单变量**配置** Gate 已通过，CPU-frequency treatment 由 checkpoint actor scorer 参数与 config 对应；"
        "本报告不把 dirty commit 宣称为完全冻结的源码证明。E3 历史 Gate G 的额外 TX 诊断在本轮再次核验历史 checkpoint actor/critic 逐位相同，"
        "五集 metrics/action/external trace identity 与本次 E3 完全相同后引用；未重新执行任何 probe。", "",
        "## 2. 正式性能：逐 seed 与五集等权 mean", ""]
    rows = []
    for mode, report in reports.items():
        for r in report["formal"]:
            rows.append([mode, r["evaluation_seed"], fmt(r["completion_ratio"]), r["completed"],
                fmt(r["expiration_ratio"]), r["expired"], fmt(r["total_reward"], 2),
                fmt(r["total_energy_j"], 3), r["executor_rejection"]])
        r = report["formal_mean"]
        rows.append([mode, "mean", fmt(r["completion_ratio"]), fmt(r["completed"], 1), fmt(r["expiration_ratio"]),
                     fmt(r["expired"], 1), fmt(r["total_reward"], 2), fmt(r["total_energy_j"], 3), fmt(r["executor_rejection"], 1)])
    text += [table(["Model", "Eval seed", "Completion rate", "DONE/episode", "Expired rate", "Expired/episode", "Return", "Energy J", "Rejection"], rows), "",
        f"Completion mean 下降 {fmt(-delta * 100, 2)} 个百分点；E3 的 DONE 全部是 Local，Remote DONE 原本为 0。"
        "E4 五集所有任务未完成，损失了 E3 尚存的 Local completion。零能耗表示没有实际计算/传输服务，不能作为效率改善。", "",
        "## 3. Final Route / TX / Remote lifecycle", ""]
    rows = []
    for mode, report in reports.items():
        r, tx = report["route"], report["tx"]
        rows.append([mode, "/".join(str(r["counts"].get(k, 0)) for k in ("local", "remote", "defer")),
            r["remote_legal"], fmt(r["p_remote"]), fmt(r["deterministic_remote"]),
            tx["legal_opportunities"], fmt(tx["p_transmit"]),
            fmt(tx["deterministic_transmit"] / tx["legal_opportunities"] if tx["legal_opportunities"] else None),
            fmt(tx["deterministic_idle"] / tx["legal_opportunities"] if tx["legal_opportunities"] else None)])
    text += [table(["Model", "Local/Remote/Defer", "Remote legal decisions", "Mean P(Remote)", "Argmax Remote | legal", "Legal TX opportunities", "Mean P(Transmit)", "Argmax Transmit | legal", "Argmax Idle | legal"], rows), "",
        "Route 按 decision 计数，同一任务 Defer 后的重访不是新的 task；Remote legal 按该 decision 至少一个合法 Remote candidate。"
        "P(Remote) 是所有 Remote candidates 概率总和，仅在 Remote 合法状态平均。TX 概率只在合法 TX opportunities 定义。E4 从空队列 reset 起全程 Defer，"
        "没有 Local/Remote binding，因而合法 TX/CPU 机会为 0；这里 NA 为分母不存在，不能补成 0。", "",
        table(["Model", "Remote binding", "First TX", "TX complete", "Destination first CPU", "Remote on-time DONE"],
              [[m, *r["formal_funnel"].values()] for m, r in reports.items()]), "",
        "**E4 没有解决 E3 的 final deterministic TX Idle collapse。** E3 在 5,722 个合法机会全部 argmax Idle；E4 未在任何合法 TX 状态检验到恢复，而是 upstream Route 全 Defer。"
        "只能认定 Remote/TX 行为未恢复、最终整体退化；不能声称 E4 在同一 TX state 下仍 argmax Idle，也不能声称修好了它。", "",
        "## 4. CPU-frequency treatment：训练访问状态中的真实行为", "",
        "正式评估文件未保存逐 slot CPU 频率/队列轨迹。E4 formal 因全 Defer 没有 CPU 服务，conditional frequency/slack/wait 为 NA；E3 formal 有 Local CPU 能耗，但没有可恢复的逐动作频率分布。"
        "以下 CPU 分布/状态/结局比较严格限于两次已完成的 32K **训练** factual actions/services，不冒充 final-policy 同状态 probe。"
        "仅 CPU queue 非 Idle 的动作进入 frequency 分母；合法频率 0 与 inactive canonical 0 分开。", ""]
    rows = []
    for mode, report in reports.items():
        f = report["cpu_frequency"]
        rows.append([mode, f["selected_queue_actions"], *[fmt(f["fractions"][str(k)]) for k in (0., .25, .5, 1.)],
                     fmt(report["cpu_energy_j"], 3), fmt(report["training_total_energy_j"], 3)])
    text += [table(["Model", "Selected queue actions", "Zero", "Low .25", "Medium .5", "Full 1", "CPU energy J / 32K", "Total energy J / 32K"], rows), "",
        "**CPU frequency 的实际采样分布已改变，但没有形成有用的 deadline ranking。** E4 四档接近均匀，"
        "每个4K窗 full fraction 仅约24%–26%；并非几乎一直最高频，也不与 legacy 分布相同。"
        "这是一项实际行为差异，不等于治疗成功；没有证据说明它学会了紧迫任务提高频率、轻松任务主动节能。", "",
        "CPU 能耗按冻结 executor 公式 `kappa * executed_frequency_hz² * actual_served_cycles` 从已保存执行服务恢复；"
        "每个 UAV kappa 来自冻结 profile ratios/perturbations。Total energy 从逐 episode energy penalty 与固定 active-energy reference 还原，"
        "并与 aggregate reward energy penalty 总数校验；不是强行按整 slot 计费。", ""]
    rows = []
    for mode, report in reports.items():
        for bucket in ("tight_large", "easy_small", "other"):
            r = report["conditioned"].get(bucket, {})
            rows.append([mode, bucket, r.get("actions", 0), *[fmt(r.get(k)) for k in ("zero_fraction", "low_fraction", "medium_fraction", "full_fraction")]])
    text += [table(["Model", "Current head state", "Actions", "Zero", "Low", "Medium", "Full"], rows), "",
        "tight_large：当前 slack≤5 slots 且 head remaining cycles 大于该 UAV 一个 full-frequency slot 能力；easy_small：slack>15 且剩余量≤一个 full slot。"
        "这只是透明的描述分组，不是人工 deadline override。E4 tight_large full=25.88%，与整体25.78%几乎一致，"
        "缺乏清晰的 urgency 排序；easy_small low+medium=54.67%，但只有75次动作，且仍近均匀分布，不能声称学会因果节能。"
        "不同 run 访问的 tasks/states 不同，所以不能从这些频率比例识别同状态 scorer 的因果 effect。", "",
        "Remote first positive CPU service：每个 task 仅首次计数；DONE/EXPIRED/censored 是事后结局标签，只用于分析。", ""]
    rows = []
    for mode, report in reports.items():
        for outcome in ("all", "done", "expired", "censored"):
            r = report["cpu"]["remote_" + outcome]
            n = r["tasks"]
            rows.append([mode, outcome, n, fmt(r["first_cpu_slack_mean"], 2),
                fmt(r["first_remaining_cycles_mean"] / 1e6 if n else None, 2), fmt(r["tx_cpu_wait_mean"], 2),
                "/".join(str(r["first_frequency"][str(f)]) for f in (0., .25, .5, 1.)),
                fmt(r["first_cpu_hz_mean"] / 1e9 if n else None, 3)])
    text += [table(["Model", "Remote outcome", "Tasks", "First CPU slack slots", "Remaining Mcycles", "TX→CPU wait slots", "First frequency counts 0/.25/.5/1", "First actual GHz"], rows), "",
        "所有获得正 CPU 服务的 task 的 outcome 频率（每次正服务动作作分母，不是首次 task）：", ""]
    text += [table(["Model", "Outcome", "CPU service actions", "Mean normalized frequency", "Counts 0/.25/.5/1"],
        [[m, o, r["cpu"]["positive_service_actions_" + o]["actions"],
          fmt(r["cpu"]["positive_service_actions_" + o]["level_mean"]),
          "/".join(str(r["cpu"]["positive_service_actions_" + o]["counts"][str(f)]) for f in (0., .25, .5, 1.))]
         for m, r in reports.items() for o in ("done", "expired", "censored")]), "",
        "task slack 使用保存的 deadline_slot 减实际服务 slot；remaining cycles 使用该 slot 的执行 head_remaining_cycles，"
        "TX→CPU wait 按首次正 CPU global index 减严格 remaining_bits==0 的 TX complete global index。终局分组非 actor 输入，未向策略注入未来 outcome。", "",
        "训练中 Remote CPU first-service task 数 4854→4598、Remote DONE 526→202；Remote DONE/CPU 从10.84%降至4.39%。"
        "first-service mean slack 18.83→18.03 slots，remaining约573.36→579.07 Mcycles，TX→CPU wait 12.49→12.24 slots。"
        "等待小幅减少没有转化成更多 DONE，不支持 CPU downstream 改善。", "",
        "## 5. 训练轨迹：固定八个 4K 窗", ""]
    rows = []
    for mode, report in reports.items():
        for s in report["stages"]:
            rows.append([mode, f"{s['start']//1000}–{s['end']//1000}K", fmt(s["completed"], 2), fmt(s["expired"], 2),
                fmt(s["return"], 2), fmt(s["entropy"], 3), fmt(s["route_remote"]),
                s["tx_opportunities"], fmt(s["tx_transmit_fraction"]),
                "/".join(fmt(s["cpu_frequency"]["fractions"][str(f)], 3) for f in (0., .25, .5, 1.))])
    text += [table(["Model", "Window", "DONE/episode", "Expired/episode", "Return/episode", "Entropy", "Route Remote", "TX legal actions", "Sampled TX | legal", "CPU 0/low/medium/full"], rows), "",
        "每窗8集，episode 终点划窗；entropy 为该窗 update 终点对应真实 PPO epoch entropy 的等权平均（跨界 update 不拆）。"
        "Route/CPU/TX 以实际 collection index 划窗。Train sampled TX ≠ final deterministic TX；CPU 分布以 selected queue actions 为分母。"
        "本轮只取 E4 已存在 action/service/epoch identity，不补造 Gate L PPO credit、advantage 或 surrogate。", "",
        "训练 Remote cohort 漏斗：", "",
        table(["Model", "Remote binding", "First TX", "TX complete", "First CPU", "Remote DONE"],
              [[m, len(ts), *[sum(k in t for t in ts) for k in ("first_tx", "tx_complete", "first_cpu")],
                sum(t.get("outcome") == "done" for t in ts)] for m, r in reports.items()
               for ts in [[t for t in r["tasks"] if t.get("remote")]]]), "",
        "## 6. Gate、证据边界与交付", "",
        "选 **E4-E**，不选 E4-A/B/C/D：正式性能明显恶化（不表示统计显著），所有 evaluation episodes 零完成，"
        "最终全 Defer；无论训练中的 CPU 局部频率是否改变，都不能覆盖这个淘汰条件。此结果否定当前 E4-v1 实现作为保留/扩 seed 候选，"
        "不证明 CPU decoder 是唯一根因，也不证明所有 deadline-conditioned decoder 必然无效。", "",
        "**唯一下一步建议：回到已保存的 Route/TX 轨迹，核查 E4 全 Defer 形成的机制；先完成这个离线解释，再决定新的单变量假设。**"
        "当前不扩 seed、不继续 E4-v1、不启动任何训练或评估。", "",
        "六张图（PNG + SVG；图注/分母见 `plots/e4-v1-seed43-audit/captions.json`）：", ""]
    for name in ("01_formal_performance", "02_remote_lifecycle_funnel", "03_route_tx_final_behavior",
                 "04_cpu_frequency_distribution", "05_done_expired_cpu_frequency_slack", "06_training_stage_comparison"):
        text.append(f"![{name}]({(root / 'plots/e4-v1-seed43-audit' / (name + '.png')).as_posix()})")
        text.append("")
    text += ["输入路径、SHA-256、canonical diff、checkpoint/评估 provenance 与完整核验输出：`logs/e4-v1-seed43-audit/input_manifest.json`、`E3_summary.json`、`E4_summary.json`。"
        "真实派生表：`dashboard_logs/e4-v1-seed43-audit/`。Source：`src/evaluation/e4_v1_matched_audit.py`、`e4_v1_audit_figures.py`、`e4_v1_audit_report.py`。"
        "测试结果另存 `logs/e4-v1-seed43-audit/test_result.json`；图像逐张检查，未重跑 formal evaluation。", ""]
    result_path = root / "logs/e4-v1-seed43-audit/test_result.json"
    if result_path.exists():
        import json
        result = json.loads(result_path.read_text(encoding="utf-8"))
        text += [f"离线 unit/regression tests：{result['tests_run']}/{result['tests_run']} PASS，"
                 f"failures={result['failures']}、errors={result['errors']}、skipped={result['skipped']}。"
                 "首次沙箱运行的3个 Windows 临时目录权限错误已通过允许测试夹具写入后复跑消除，"
                 "初次记录保留在 `test_result_sandbox.json`。", ""]
    (root / "E4_V1_SEED43_GATE_REPORT.md").write_text("\n".join(text), encoding="utf-8")
