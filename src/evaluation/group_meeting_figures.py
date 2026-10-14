"""Reproduce the C4k-32k meeting table and figures from saved evaluations only."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


CHECKPOINTS = (4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000)
SCENES = (1042, 1043, 1044, 1045, 1046)
PROTOCOL = "d2463e1929ea867212edec4774f54b2c5a6dbc9ef957f00dd2978402aec4849a"
COLORS = {"C": "#2C6093", "heuristic": "#C48744", "local_only": "#638575"}
LABELS = {"C": "方法 C", "heuristic": "Heuristic", "local_only": "纯本地"}


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def summarize_episodes(episodes):
    """Pool counts/energy, average episode reward, and weight delay by DONE."""
    if not episodes:
        raise ValueError("No episode records")
    for record in episodes:
        m = record["metrics"]
        if m["generated"] != m["completed"] + m["expired"] + m["truncated"]:
            raise ValueError("Task count conservation failed")
        if not math.isclose(m["total_energy_j"], m["cpu_energy_j"] + m["tx_energy_j"], abs_tol=1e-8):
            raise ValueError("Energy conservation failed")
        if m["generated"] <= 0:
            raise ValueError("Generated task count must be positive")
    fields = ("generated", "completed", "expired", "truncated", "total_energy_j", "cpu_energy_j", "tx_energy_j")
    result = {field: sum(r["metrics"][field] for r in episodes) for field in fields}
    result["completion_pct"] = 100 * result["completed"] / result["generated"]
    result["mean_per_scene_completion_pct"] = statistics.mean(100 * r["metrics"]["completed"]
        / r["metrics"]["generated"] for r in episodes)
    result["mean_episode_reward"] = statistics.mean(r["metrics"]["total_reward"] for r in episodes)
    result["energy_per_DONE_j"] = result["total_energy_j"] / result["completed"] if result["completed"] else None
    result["completed_task_delay_s"] = (sum(r["metrics"]["completed"] * r["metrics"]["completed_task_e2e_delay_s"]
        for r in episodes if r["metrics"]["completed"]) / result["completed"] if result["completed"] else None)
    return result


def load_meeting_data(root):
    """Verify all eight saved protocols and invariant baselines before plotting."""
    inputs = set()

    def read(path):
        path = Path(path)
        inputs.add(path)
        return json.loads(path.read_text(encoding="utf-8-sig"))

    older = read(root / "metadata/audits/checkpoint_progress/20261013_4k8k12k16k/comparison_summary.json")
    latest_root = root / "metadata/audits/c32k_progress/20261014_16k32k"
    latest = read(latest_root / "comparison_summary.json")
    newer_runs = read(latest_root / "source_evaluation_runs.json")
    results, per_scene, source_runs = [], [], []
    baseline_reference = None
    external_reference = None
    for step in CHECKPOINTS:
        if step <= 16000:
            source = next(r for r in older["source_datasets"] if r["method"] == "C" and r["checkpoint_steps"] == step)
            metrics_path, manifest_path = Path(source["episode_metrics"]), Path(source["manifest"])
        else:
            source = newer_runs[str(step)]
            metrics_path, manifest_path = map(Path, source["evaluation_artifacts"][1::-1])
        manifest = read(manifest_path)
        if manifest["evaluation_config_sha256"] != PROTOCOL:
            raise ValueError("Functional evaluation protocol changed")
        inputs.add(metrics_path)
        rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
        index = {(r["method_id"], r["evaluation_seed"]): r for r in rows}
        expected = {(method, seed) for method in ("ca_gat_mappo", "local_only", "heuristic", "random") for seed in SCENES}
        if set(index) != expected or len(rows) != 20:
            raise ValueError("Missing or duplicated method/scene records")
        baselines = {key: {k: v for k, v in row.items() if k != "evaluation_run_id"}
                     for key, row in index.items() if key[0] != "ca_gat_mappo"}
        external = {(method, seed): row["external_trace_sha256"] for (method, seed), row in index.items()}
        if baseline_reference is None:
            baseline_reference, external_reference = baselines, external
        if baselines != baseline_reference or external != external_reference:
            raise ValueError("Baseline result or exogenous realization changed")
        learned = [index["ca_gat_mappo", seed] for seed in SCENES]
        summary = summarize_episodes(learned)
        if summary["generated"] != 605:
            raise ValueError("Total denominator must remain605")
        if step <= 16000:
            expected_done = older["methods"]["C"][str(step)]["pooled"]["DONE"]
        else:
            expected_done = latest["checkpoint_metrics"][str(step)]["completed"]
        if summary["completed"] != expected_done:
            raise ValueError("Published and recomputed DONE differ")
        results.append({"method": "C", "checkpoint_steps": step, **summary})
        for method in ("ca_gat_mappo", "local_only", "heuristic"):
            for seed in SCENES:
                m = index[method, seed]["metrics"]
                per_scene.append({"method": "C" if method == "ca_gat_mappo" else method,
                    "checkpoint_steps": step, "seed": seed, "generated": m["generated"],
                    "completed": m["completed"], "completion_pct": 100 * m["completed"] / m["generated"]})
        source_runs.append({"step": step, "episode_metrics": str(metrics_path), "manifest": str(manifest_path),
                            "evaluation_run_id": manifest["evaluation_run_id"]})
    baseline_rows = []
    for method in ("local_only", "heuristic"):
        summary = summarize_episodes([index[method, seed] for seed in SCENES])
        baseline_rows.append({"method": method, "checkpoint_steps": None, **summary})
    risk = [{"checkpoint_steps": int(step), **values} for step, values in latest["checkpoint_metrics"].items()]
    risk.sort(key=lambda r: r["checkpoint_steps"])
    data = {"training_seed": 42, "evaluation_seeds": list(SCENES), "protocol_sha256": PROTOCOL,
            "baseline_rows": baseline_rows, "C_rows": results, "per_scene": per_scene,
            "energy_risk": risk, "source_runs": source_runs,
            "source_hashes": {str(p.relative_to(root)): file_sha256(p) for p in sorted(inputs)}}
    return data


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scene_comparison_series(data, checkpoint_steps=32000, scene_ids=SCENES):
    """Tie bar heights and labels to the same percentage metric."""
    series = {}
    arrivals = None
    for method in ("local_only", "heuristic", "C"):
        selected = []
        for seed in scene_ids:
            matches = [r for r in data["per_scene"] if r["method"] == method
                       and r["seed"] == seed and r["checkpoint_steps"] == checkpoint_steps]
            if len(matches) != 1:
                raise ValueError("Expected one method/checkpoint/scene record")
            row = matches[0]
            if row["generated"] <= 0 or not 0 <= row["completed"] <= row["generated"]:
                raise ValueError("Invalid task counts")
            ratio = 100 * row["completed"] / row["generated"]
            if not math.isclose(ratio, row["completion_pct"], abs_tol=1e-9):
                raise ValueError("Saved percentage differs from task counts")
            selected.append(row)
        counts = [r["generated"] for r in selected]
        if arrivals is None:
            arrivals = counts
        if counts != arrivals:
            raise ValueError("Methods have different arrivals in the same scene")
        percentages = [100 * r["completed"] / r["generated"] for r in selected]
        series[method] = {"completion_pct": percentages,
                          "bar_labels": [f"{v:.2f}%" for v in percentages],
                          "completed": [r["completed"] for r in selected]}
    return {"scenes": list(scene_ids), "arrival_counts": arrivals,
            "bar_height_metric": "completion_pct", "bar_label_metric": "completion_pct", "series": series}


def generate_figures(data, output):
    """Create Chinese meeting visuals with no smoothing or invented uncertainty."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt
    import numpy as np

    font = Path("C:/Windows/Fonts/msyh.ttc")
    if not font.is_file():
        raise FileNotFoundError("Chinese plot font is unavailable")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False,
        "font.size": 12, "axes.titlesize": 17, "axes.labelsize": 13, "legend.fontsize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    x = [r["checkpoint_steps"] / 1000 for r in data["C_rows"]]
    baselines = {r["method"]: r for r in data["baseline_rows"]}
    metadata = []

    def finish(fig, stem, caption, plotted):
        fig.text(.5, .025, caption, ha="center", va="bottom", fontsize=10, color="#555555")
        label_check = None
        if stem == "03_scene_comparison":
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            boxes = [text.get_window_extent(renderer) for text in fig.axes[0].texts if "%" in text.get_text()]
            if any(a.overlaps(b) for index, a in enumerate(boxes) for b in boxes[index + 1:]):
                raise ValueError("Scene percentage labels overlap")
            label_check = {"label_count": len(boxes), "overlap_pairs": 0}
        for extension in ("png", "svg"):
            fig.savefig(output / f"{stem}.{extension}", dpi=240, facecolor="white")
        metadata.append({"figure": stem, "caption": caption, "plotted_data": plotted, "label_overlap_check": label_check})
        plt.close(fig)

    def curve_panel(ax, key, title, ylabel):
        y = [r[key] for r in data["C_rows"]]
        ax.plot(x, y, color=COLORS["C"], marker="o", lw=2.5, markersize=6, label=LABELS["C"])
        for method, linestyle in (("heuristic", "--"), ("local_only", ":")):
            ax.axhline(baselines[method][key], color=COLORS[method], linestyle=linestyle, lw=2,
                       label=LABELS[method] + "（固定基线）")
        ax.set(title=title, xlabel="C 的累计训练步数（k）", ylabel=ylabel, xticks=x, xlim=(3, 34))
        ax.grid(axis="y", alpha=.22)
        ax.legend(loc="best", frameon=False)
        final_label = f"{y[-1]:.4f}" if key == "energy_per_DONE_j" else f"{y[-1]:.2f}"
        ax.annotate(final_label, (x[-1], y[-1]), xytext=(5, 9), textcoords="offset points",
                    color=COLORS["C"], weight="bold", fontsize=11)

    common = "训练 seed42；固定开发 seeds1042–1046；每组605个相同任务；基线无需训练；无平滑、无误差条"
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    fig.subplots_adjust(left=.075, right=.965, top=.86, bottom=.22, wspace=.28)
    curve_panel(axes[0], "completion_pct", "任务完成率", "总完成数 / 605（%）")
    axes[0].set_ylim(40, 77)
    curve_panel(axes[1], "mean_episode_reward", "平均评估奖励", "5个评估回合的平均奖励")
    finish(fig, "01_completion_and_reward", common,
           {"x_k": x, "C_completion_pct": [r["completion_pct"] for r in data["C_rows"]],
            "C_mean_reward": [r["mean_episode_reward"] for r in data["C_rows"]], "baselines": baselines})

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    fig.subplots_adjust(left=.075, right=.965, top=.86, bottom=.22, wspace=.28)
    curve_panel(axes[0], "total_energy_j", "总能耗", "5个评估回合的总能耗（J）")
    curve_panel(axes[1], "energy_per_DONE_j", "每完成任务的能耗", "总能耗 / 总完成数（J/任务）")
    finish(fig, "02_energy_cost", common,
           {"x_k": x, "C_total_energy_j": [r["total_energy_j"] for r in data["C_rows"]],
            "C_energy_per_DONE_j": [r["energy_per_DONE_j"] for r in data["C_rows"]], "baselines": baselines})

    fig, ax = plt.subplots(figsize=(11.8, 4.9))
    fig.subplots_adjust(left=.08, right=.97, top=.83, bottom=.23)
    centers = np.arange(len(SCENES))
    scene_data = scene_comparison_series(data)
    for offset, method in zip((-.27, 0, .27), ("local_only", "heuristic", "C")):
        series = scene_data["series"][method]
        bars = ax.bar(centers + offset, series["completion_pct"], width=.25,
                      color=COLORS[method], label="C 32k" if method == "C" else LABELS[method])
        ax.bar_label(bars, labels=series["bar_labels"], padding=4, fontsize=9)
    counts = scene_data["arrival_counts"]
    ax.set_xticks(centers, [f"{seed}\n到达 {n} 个任务" for seed, n in zip(SCENES, counts)])
    ax.set(title="C 32k 与两种基线的逐场景完成率", ylabel="任务完成率（%）", ylim=(0, 100))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.02), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    finish(fig, "03_scene_comparison", "柱高与柱顶标注均为完成率（完成数/该场景到达数）；每个场景使用自己的分母",
           scene_data)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.subplots_adjust(left=.06, right=.975, top=.85, bottom=.23, wspace=.31)
    risk_keys = (("published_energy_infeasible_pct_of_remote", "低能量远端分配占比", "占全部远端分配（%）"),
                 ("CPU_energy_blocked_UAV_slots", "CPU 能量阻断", "阻断 UAV-时隙数（每组10,000个）"),
                 ("post_persistent_blockage_assignments", "持续阻断后的新分配", "新分配任务数"))
    rx = [r["checkpoint_steps"] / 1000 for r in data["energy_risk"]]
    for ax, (key, title, ylabel) in zip(axes, risk_keys):
        y = [r[key] for r in data["energy_risk"]]
        ax.plot(rx, y, marker="o", color=COLORS["C"], linewidth=2.3)
        for xx, yy in zip(rx, y):
            ax.annotate(f"{yy:.2f}" if "pct" in key else str(yy), (xx, yy),
                        xytext=(0, 8), textcoords="offset points", ha="center", fontsize=10)
        ax.set(title=title, xlabel="C 的累计训练步数（k）", ylabel=ylabel, xticks=rx)
        ax.margins(y=.28); ax.grid(axis="y", alpha=.22)
    finish(fig, "04_energy_risk", "仅12k–32k已有诊断；公共能量按日志因果顺序重建，缺失消息单列；早期未诊断不等于风险为零",
           {"x_k": rx, "series": {key: [r[key] for r in data["energy_risk"]] for key, _, _ in risk_keys}})

    display_rows = data["baseline_rows"] + data["C_rows"]
    headers = ["方法 / 模型", "完成数", "完成率\n（%）", "过期数", "截断数", "平均评估\n奖励", "总能耗\n（J）", "能耗 / DONE\n（J/任务）"]
    cells = []
    for row in display_rows:
        label = f"C {row['checkpoint_steps']//1000}k" if row["method"] == "C" else LABELS[row["method"]]
        cells.append([label, str(row["completed"]), f"{row['completion_pct']:.2f}", str(row["expired"]),
            str(row["truncated"]), f"{row['mean_episode_reward']:.2f}", f"{row['total_energy_j']:.2f}",
            f"{row['energy_per_DONE_j']:.4f}"])
    fig, ax = plt.subplots(figsize=(16, 6.2)); ax.axis("off")
    fig.subplots_adjust(left=.025, right=.975, top=.85, bottom=.16)
    table = ax.table(cellText=cells, colLabels=headers, cellLoc="center", loc="center",
                     colWidths=[.15, .09, .12, .10, .10, .13, .14, .17])
    table.auto_set_font_size(False); table.set_fontsize(14); table.scale(1, 2.1)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D8DEE5"); cell.set_linewidth(.6)
        if row == 0:
            cell.set_facecolor("#223C54"); cell.set_text_props(color="white", weight="bold"); cell.set_height(cell.get_height()*1.35)
        elif row == len(display_rows):
            cell.set_facecolor("#E2EEF7"); cell.set_text_props(weight="bold", color="#244D71")
        elif row % 2 == 0:
            cell.set_facecolor("#F4F6F8")
    fig.suptitle("C 4k–32k 与固定基线：阶段性实验总表", fontsize=21, y=.95)
    finish(fig, "00_summary_table", "各行均为5个场景、605个任务；完成率=总完成数/605；能耗/DONE=总能耗/总完成数；截断指评估结束时未完成",
           {"headers": headers, "cells": cells})
    return metadata


def build(root, tag):
    if Path(tag).name != tag or tag in {".", ".."}:
        raise ValueError("Output tag must be one directory name")
    data = load_meeting_data(root)
    figure_root = root / "plots/C/seed42/group_meeting" / tag
    csv_root = root / "dashboard_logs/C/seed42/group_meeting" / tag
    audit_root = root / "metadata/audits/group_meeting_summary" / tag
    if any(p.exists() for p in (figure_root, csv_root, audit_root)):
        raise FileExistsError("Use a new tag; existing artifacts are never overwritten")
    for p in (figure_root, csv_root, audit_root):
        p.mkdir(parents=True)
    rows = data["baseline_rows"] + data["C_rows"]
    write_csv(csv_root / "summary_metrics.csv", rows)
    write_csv(csv_root / "per_scene_metrics.csv", data["per_scene"])
    (figure_root / "figure_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    figures = generate_figures(data, figure_root)
    recipe = {"purpose": "group-meeting development diagnostic; not final manuscript evidence",
        "script": str(Path(__file__).resolve()), "script_sha256": file_sha256(__file__),
        "regeneration": "D:/Miniconda/python.exe -B -X utf8 -m src.evaluation.group_meeting_figures --tag NEW_TAG",
        "data": str(figure_root / "figure_data.json"), "summary_csv": str(csv_root / "summary_metrics.csv"),
        "source_hashes": data["source_hashes"], "figures": figures,
        "palette": COLORS, "font": "Microsoft YaHei", "smoothing": None, "error_bars": None,
        "table_order": "two fixed baselines, followed by C4k-32k",
        "claim_limits": ["One training seed and five development scenes", "No convergence or energy-efficiency advantage claim",
                         "No ranking against A/B at different budgets", "No paper held-out evaluation or additional training"]}
    (figure_root / "figure_recipe.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (audit_root / "data_validation.json").write_text(json.dumps({"status": "PASS", "saved_episode_records_checked": 160,
        "same_baseline_episode_records_checked": 120, "training_seed": 42, "evaluation_seeds": list(SCENES),
        "protocol_sha256": PROTOCOL, "source_hashes": data["source_hashes"], "new_training_or_evaluation_executed": False,
        "input_files_unchanged": all(file_sha256(root / p)==digest for p,digest in data["source_hashes"].items())},
        ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"table_rows": rows, "plots_dir": str(figure_root), "csv_dir": str(csv_root),
                      "audit_dir": str(audit_root), "figure_count": len(figures)}, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="20261014")
    args = parser.parse_args(argv)
    build(Path(__file__).resolve().parents[2], args.tag)


if __name__ == "__main__":
    main()
