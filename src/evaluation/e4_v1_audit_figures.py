"""Six diagnostic figures from saved E3/E4 audit tables only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"E3": "#2878B5", "E4": "#D95319"}
LEVELS = (0., .25, .5, 1.)


def build_figures(reports, output):
    output = Path(output)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    captions = {}

    def save(fig, name, caption):
        fig.tight_layout(rect=(0, .055, 1, .94))
        fig.text(.02, .015, "One matched training seed (43); saved-artifact diagnostics; no significance/causal claim.", fontsize=9)
        for ext in ("png", "svg"):
            fig.savefig(output / (name + "." + ext), dpi=190, facecolor="white")
        plt.close(fig)
        captions[name] = caption

    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, key, title in zip(axs.flat, ("completion_ratio", "completed", "expiration_ratio", "expired", "total_reward", "total_energy_j"),
                               ("Completion rate", "Completed / episode", "Expired rate", "Expired / episode", "Return / episode", "Energy (J) / episode")):
        for mode, report in reports.items():
            rows = report["formal"]
            ax.plot([r["evaluation_seed"] for r in rows], [r[key] for r in rows], "o-", color=COLORS[mode], label=mode)
            ax.axhline(report["formal_mean"][key], color=COLORS[mode], ls="--", alpha=.65)
        ax.set_title(title)
        ax.set_xticks(range(1042, 1047))
        ax.grid(alpha=.2)
    axs[0, 0].legend()
    fig.suptitle("Formal evaluation: five matched exogenous episodes; dashed = episode mean")
    save(fig, "01_formal_performance", "每个点为一个已完成的正式评估 episode；虚线为五集等权均值，非 pooled task ratio。两个模型 executor rejection 均为 0。")

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    labels = ["Binding", "First TX", "TX complete", "First CPU", "On-time DONE"]
    keys = ["remote_binding", "first_tx_service", "tx_complete", "destination_cpu_first_service", "remote_on_time"]
    for mode, report in reports.items():
        y = [report["formal_funnel"][k] for k in keys]
        axs[0].plot(range(5), y, "o-", color=COLORS[mode], label=mode)
        for i, n in enumerate(y):
            axs[0].annotate(str(n), (i, n), xytext=(0, 6 if mode == "E3" else -17), textcoords="offset points", ha="center", color=COLORS[mode])
        ts = [t for t in report["tasks"] if t.get("remote")]
        y = [len(ts), *[sum(k in t for t in ts) for k in ("first_tx", "tx_complete", "first_cpu")], sum(t.get("outcome") == "done" for t in ts)]
        axs[1].plot(range(5), y, "o-", color=COLORS[mode], label=mode)
        for i, n in enumerate(y):
            axs[1].annotate(str(n), (i, n), xytext=(0, 8 if mode == "E3" else -18),
                            textcoords="offset points", ha="center", color=COLORS[mode])
    for ax, title in zip(axs, ("Formal final policy (five episodes)", "Training task cohorts (0-32K)")):
        ax.set_title(title)
        ax.set_xticks(range(5), labels, rotation=20)
        ax.set_ylabel("Distinct Remote tasks")
        ax.legend()
        ax.grid(alpha=.2)
        ax.margins(y=.12)
    fig.suptitle("Remote lifecycle: positive service only; exact TX completion")
    save(fig, "02_remote_lifecycle_funnel", "左图正式 final 漏斗；E4 无绑定，零值不是未记录。E3 下游零值由历史同权重/同正式 action hash 诊断及本次零 TX 能耗交叉支持。右图训练唯一 task cohort，DONE 为 deadline 内完成。")

    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(3)
    for j, (mode, report) in enumerate(reports.items()):
        counts = [report["route"]["counts"].get(k, 0) for k in ("local", "remote", "defer")]
        axs[0].bar(x + (j - .5) * .32, counts, .32, color=COLORS[mode], label=mode)
        for i, n in enumerate(counts):
            axs[0].text(i + (j - .5) * .32, n + 90, str(n), ha="center", fontsize=9)
    axs[0].set_xticks(x, ["Local", "Remote", "Defer"])
    axs[0].set_title("Formal Route decisions (counts)")
    axs[0].legend()
    axs[1].bar([0, 1], [reports[m]["route"]["p_remote"] for m in reports], color=list(COLORS.values()))
    axs[1].set_xticks([0, 1], ["E3", "E4"])
    axs[1].set_ylim(0, 1.07)
    axs[1].set_title("Mean P(Remote) | Remote legal")
    axs[1].set_ylabel("Probability")
    for i, m in enumerate(reports):
        r = reports[m]["route"]
        axs[1].text(i, r["p_remote"] + .03, f"n={r['remote_legal']}", ha="center")
    tx = reports["E3"]["tx"]
    axs[2].bar([0, 1, 2], [tx["p_transmit"], 0, 1], color=COLORS["E3"])
    axs[2].set_xticks([0, 1, 2], ["P(TX)", "argmax TX", "argmax Idle"])
    axs[2].set_ylim(0, 1.4)
    axs[2].set_title(f"E3 TX | legal (n={tx['legal_opportunities']})")
    axs[2].text(.5, 1.17, "E4: 0 legal TX opportunities\nP(TX), TX/Idle fractions = NA", transform=axs[2].get_xaxis_transform(), ha="center", fontsize=9)
    fig.suptitle("Final behavior: E4 all-Defer removes TX visitation; this is not TX recovery")
    save(fig, "03_route_tx_final_behavior", "Route denominator 为每次决策，Defer 同一 task 可重访。P(Remote) 仅 Remote 合法状态；E4 TX conditional 概率/比例不可定义，不能写成恢复后的 0% Idle。E3 TX 概率来自保存的 Gate G 同策略诊断。")

    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    for j, (mode, report) in enumerate(reports.items()):
        counts = report["cpu_frequency"]
        ys = [counts["fractions"][str(f)] for f in LEVELS]
        axs[0].bar(np.arange(4) + (j - .5) * .32, ys, .32, color=COLORS[mode], label=f"{mode} n={counts['selected_queue_actions']}")
        for ax, bucket, title in zip(axs[1:], ("tight_large", "easy_small"), ("Tight slack + large head", "Easy slack + small head")):
            c = report["conditioned"].get(bucket, {})
            ys = [c.get(k, np.nan) for k in ("zero_fraction", "low_fraction", "medium_fraction", "full_fraction")]
            ax.bar(np.arange(4) + (j - .5) * .32, ys, .32, color=COLORS[mode], label=f"{mode} n={c.get('actions', 0)}")
            ax.set_title(title)
    for ax in axs:
        ax.set_xticks(range(4), ["Zero", "Low .25", "Medium .5", "Full 1"])
        ax.set_ylabel("Fraction of selected-queue actions")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
    axs[0].set_title("All training CPU frequency choices")
    fig.suptitle("Actual training treatment behavior (0-32K); formal E4 has no CPU service")
    save(fig, "04_cpu_frequency_distribution", "CPU queue 非 Idle 的动作作分母，含合法 0 频率，排除 inactive canonical 0。tight=slack≤5 且 remaining>1 full-frequency slot capacity；easy=slack>15 且 remaining≤1 full slot。不同 run 是不同访问状态，非同状态配对比较。")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for j, (mode, report) in enumerate(reports.items()):
        for i, outcome in enumerate(("done", "expired")):
            c = report["cpu"]["remote_" + outcome]
            axs[0, 0].bar(i + (j - .5) * .32, c["first_cpu_slack_mean"] or 0, .32, color=COLORS[mode], label=mode if i == 0 else None)
            axs[0, 0].text(i + (j - .5) * .32, (c["first_cpu_slack_mean"] or 0) + 1, f"n={c['tasks']}", ha="center", fontsize=9)
            axs[0, 1].bar(i + (j - .5) * .32, c["first_remaining_cycles_mean"] / 1e6 if c["tasks"] else np.nan, .32, color=COLORS[mode])
            axs[1, 0].bar(i + (j - .5) * .32, c["tx_cpu_wait_mean"] if c["tasks"] else np.nan, .32, color=COLORS[mode])
            axs[1, 1].bar(i + (j - .5) * .32, c["first_frequency"]["1.0"] / c["tasks"] if c["tasks"] else np.nan, .32, color=COLORS[mode])
    for ax, title, ylabel in zip(axs.flat, ("First CPU service slack", "First CPU remaining cycles", "TX complete to first CPU wait", "Full frequency at first CPU service"),
                                ("Mean slots", "Mean Mcycles", "Mean slots", "Task fraction")):
        ax.set_xticks([0, 1], ["DONE", "EXPIRED"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.2)
    axs[0, 0].legend()
    axs[1, 1].set_ylim(0, 1)
    fig.suptitle("Remote training tasks receiving positive CPU service; retrospective outcome grouping")
    save(fig, "05_done_expired_cpu_frequency_slack", "每个 Remote task 首次正 CPU 服务只计一次；DONE/EXPIRED 是事后分组，不是 actor 输入，也不能证明频率造成 outcome。未进入 CPU 的任务不在分母。remaining 与 slack 由该动作的执行 head 和已保存 lifecycle 还原。")

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    keys = ("completed", "expired", "return", "entropy", "route_remote", "tx_opportunities", "tx_transmit_fraction", "cpu_full")
    titles = ("Completed / episode", "Expired / episode", "Return / episode", "PPO epoch entropy", "Sampled Route Remote fraction", "Legal TX actions / 4K", "Sampled TX | legal", "Full CPU frequency | selected queue")
    for ax, key, title in zip(axs.flat, keys, titles):
        for mode, report in reports.items():
            ys = [s["cpu_frequency"]["fractions"]["1.0"] if key == "cpu_full" else s[key] for s in report["stages"]]
            ax.plot(range(8), ys, "o-", color=COLORS[mode], label=mode)
        ax.set_xticks(range(8), [f"{i*4}-{i*4+4}" for i in range(8)], rotation=45)
        ax.set_xlabel("Training transitions (K)")
        ax.set_title(title)
        ax.grid(alpha=.2)
    axs[0, 0].legend()
    fig.suptitle("All eight fixed 4K windows; episode ends and PPO update ends assigned to windows")
    save(fig, "06_training_stage_comparison", "每窗8个完整500-slot episode。性能按 episode end，entropy按 PPO update end 的实际 epoch mean，动作按 collection time。边界跨窗 update 不拆分。并非 Gate L credit，未重建任何未保存 surrogate/advantage。")
    (output / "captions.json").write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")
