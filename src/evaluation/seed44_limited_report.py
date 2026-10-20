"""Combine verified offline diagnostics without executing a policy or environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "metadata/audits/seed44_limited_diagnostic"


def paired_cpu_tasks(learned, reference):
    """Compare the same task identities; outcome groups are descriptive."""
    def task_map(case):
        result = {}
        for episode in case["episodes"]:
            scene = episode["identity"]["evaluation_seed"]
            for task in episode["tasks"]:
                key = (scene, task["task_id"])
                if key in result:
                    raise ValueError("Duplicated scene/task identity")
                result[key] = task
        return result

    left, right = task_map(learned), task_map(reference)
    if set(left) != set(right):
        raise ValueError("Task identities differ across CPU comparison")
    for key in left:
        if left[key]["required_cycles"] != right[key]["required_cycles"]:
            raise ValueError("Paired task workload differs")
    groups = {}
    for label, predicate in (
        ("both_done", lambda a, b: a == b == "DONE"),
        ("learned_only_done", lambda a, b: a == "DONE" and b != "DONE"),
        ("reference_only_done", lambda a, b: a != "DONE" and b == "DONE"),
        ("neither_done", lambda a, b: a != "DONE" and b != "DONE"),
    ):
        selected = [key for key in left if predicate(left[key]["final_status"], right[key]["final_status"])]
        groups[label] = {"task_count": len(selected),
            "learned_cpu_energy_j": math.fsum(left[k]["cpu_energy_j"] for k in selected),
            "reference_cpu_energy_j": math.fsum(right[k]["cpu_energy_j"] for k in selected),
            "learned_served_cycles": math.fsum(left[k]["served_cycles"] for k in selected),
            "reference_served_cycles": math.fsum(right[k]["served_cycles"] for k in selected)}
    common = groups["both_done"]
    if not math.isclose(common["learned_served_cycles"], common["reference_served_cycles"], abs_tol=1e-4):
        raise ValueError("Common completed tasks have inconsistent served workload")
    return {"paired_tasks": len(left), "groups": groups,
            "interpretation": "Actual debits for matched identities; not an alternate-action counterfactual."}


def build_summary(route, cpu, training):
    """Require cross-diagnostic agreement before reporting any conclusions."""
    if route["schema_version"] != "limited_route_diagnostic.v2":
        raise ValueError("Require corrected inclusive-slack Route schema v2")
    if cpu["status"] != "PASS" or training["schema"] != "seed44_limited_training_diagnostic.v1":
        raise ValueError("Unexpected diagnostic identity/status")
    if not route["all_30_inputs_unchanged_during_analysis"] or not route["all_tasks_workload_and_external_traces_match"]:
        raise ValueError("Route source/workload verification failed")
    if any(training[k] for k in ("training_started", "environment_evaluation_started", "heldout_used")):
        raise ValueError("This report requires offline-only diagnostics")
    cases = {(c["method_id"], c["training_seed"], c["checkpoint_steps"]): c for c in cpu["cases"]}
    if len(cases) != len(cpu["cases"]) or len(cases) != 11:
        raise ValueError("Require eleven unique CPU cases")
    for (method, seed, step), case in cases.items():
        if method != "ca_gat_mappo":
            continue
        r = route["runs"][f"C{seed}_{step // 1000}k"]["summary"]
        p = case["pooled"]
        if r["tasks"] != p["generated"] or r["completed"] != p["completed"] or p["generated"] != 605:
            raise ValueError("Route/CPU task accounting disagrees")
        if not r["exact_edf_head_reconstruction_passed"]:
            raise ValueError("EDF reconstruction did not pass")
    primary = route["comparisons"]["C44_32k_to_C44_36k"]
    if primary["gained"] - primary["lost"] != primary["net_completed_change"]:
        raise ValueError("Paired completion arithmetic disagrees")
    learned = cases["ca_gat_mappo", 44, 40000]
    heuristic = cases["heuristic", 44, 40000]
    p, h = learned["pooled"], heuristic["pooled"]
    cpu_delta = p["cpu_energy_j"] - h["cpu_energy_j"]
    total_delta = p["total_energy_j"] - h["total_energy_j"]
    outcome_delta = {key: p["outcome_distribution"][key]["cpu_energy_j"] - h["outcome_distribution"][key]["cpu_energy_j"]
                     for key in ("DONE", "EXPIRED", "TRUNCATED")}
    if not math.isclose(math.fsum(outcome_delta.values()), cpu_delta, abs_tol=1e-8):
        raise ValueError("Outcome energy differences do not conserve total")
    windows = training["seeds"]["44"]["windows"]
    for w in windows:
        if w["ppo_epoch_count"] != 4 * w["update_count"]:
            raise ValueError("PPO epoch/update accounting differs")
        if sum(w["route_selection_unique_rollout_counts"].values()) != w["branches"]["route"]["unique_rollout_active_samples"]:
            raise ValueError("Route sample accounting differs")
    return {"schema_version": "seed44_limited_summary.v1", "date": "2026-10-15",
        "scope": "New offline analysis of existing development logs; no new training/evaluation",
        "primary_pair": {key: primary[key] for key in ("lost", "gained", "net_completed_change")},
        "lost_unbound": primary["lost_new_terminal_groups"]["EXPIRED/UNBOUND"],
        "lost_unbound_previous": primary["lost_before_by_new_terminal_group"]["EXPIRED/UNBOUND"],
        "route_run_summaries": {key: value["summary"] for key, value in route["runs"].items()},
        "cpu_case_summaries": [{"method_id": c["method_id"], "training_seed_provenance": c["training_seed"],
                                "checkpoint_steps_provenance": c["checkpoint_steps"], "pooled": c["pooled"]}
                               for c in cpu["cases"]],
        "energy_difference_vs_heuristic": {"cpu_j": cpu_delta, "total_j": total_delta,
            "cpu_share_pct": 100 * cpu_delta / total_delta, "cpu_j_by_outcome": outcome_delta},
        "matched_cpu_comparison": paired_cpu_tasks(learned, heuristic),
        "seed44_training_windows": [{k: v for k, v in w.items() if k != "epoch_detail"} for w in windows],
        "verification_scope": {"route_runs": len(route["runs"]), "cpu_cases": len(cases),
            "saved_cpu_episodes": sum(len(c["episodes"]) for c in cpu["cases"]),
            "cpu_slot_checks": sum(e["checks"]["queue_and_energy_slot"] for c in cpu["cases"] for e in c["episodes"]),
            "task_cycle_checks": sum(e["checks"]["task_cycle_and_first_service"] for c in cpu["cases"] for e in c["episodes"])},
        "limits": ["No causal intervention has been run", "No proof that replacing Defer with Local completes a task",
            "Actual CPU frequency does not expose proposal/mask/scheduler-score causes",
            "Matched-task energy differs jointly with execution destination, timing, and frequency",
            "Historical optimizer LR and missing auxiliary gradient norms remain unavailable"]}


def render_report(s):
    """Render only checked measurements, with separate facts and limitations."""
    u, before, scope = s["lost_unbound"], s["lost_unbound_previous"], s["verification_scope"]
    cases = {(c["method_id"], c["training_seed_provenance"], c["checkpoint_steps_provenance"]): c["pooled"] for c in s["cpu_case_summaries"]}
    p, h = cases["ca_gat_mappo", 44, 40000], cases["heuristic", 44, 40000]
    c32, c36 = cases["ca_gat_mappo", 44, 32000], cases["ca_gat_mappo", 44, 36000]
    energy, pair = s["energy_difference_vs_heuristic"], s["matched_cpu_comparison"]["groups"]["both_done"]
    lines = ["# seed44 第一步限定诊断报告", "", "日期：2026-10-15。状态：D1—D4 离线诊断已实际执行；未启动训练、新环境评估、最终测试或网络更新。未修改 reward、生产环境、配置和模型。本轮新增的是已有真实日志的重建与分析，不是新实验成绩。", "",
        "## 当前结论", "", "后期回落已定位到明确的轨迹行为：部分原本完成的任务在36k反复选择Defer，随后未绑定即过期，同时造成其他未绑定任务等待。CPU高能耗集中在高频服务及特定执行节点；不能主要解释为对失败任务白算。尚未证明这些行为由哪项学习机制导致，也没有证明替换动作能无损恢复完成率。", "",
        "## 实际执行范围", "", f"Route覆盖{scope['route_runs']}个已保存检查点评估，每个605任务，重建每槽UNBOUND EDF队首。CPU覆盖{scope['cpu_cases']}种保存条件、{scope['saved_cpu_episodes']}个历史评估回合，逐项核对{scope['cpu_slot_checks']}个CPU槽、{scope['task_cycle_checks']}份生命周期记录。相同任务在不同条件下重复核对，不是新增独立训练。",
        "seed44使用20k、32k、36k、40k；seed42/43使用32k、36k、40k作为描述性控制，场景均为1042—1046。固定Heuristic取自相同冻结评估；它本身没有training seed或checkpoint。训练信号读取三个种子的Dashboard与所需更新日志，并仅读取final checkpoint中的optimizer学习率，不实例化策略或环境。", "",
        "## D1/D2：任务损失与Route/Defer", "", f"32k→36k失去{s['primary_pair']['lost']}个原本完成的任务，新增完成{s['primary_pair']['gained']}个，净变化{s['primary_pair']['net_completed_change']}。失去完成任务中49个为EXPIRED/UNBOUND，另23个EXPIRED/CPU_SERVICE、13个EXPIRED/CPU_WAIT、1个TRUNCATED/CPU_WAIT。",
        f"49个未绑定过期任务中，{u['visible_defer']}个实际成为队首并反复Defer，{u['never_head']}个始终未成为队首。前48个累计{u['defer_decisions']}次Defer，Local在所有机会均合法，{u['defer_decisions_with_legal_remote']}次另有合法Remote。不能把唯一未获机会任务本人记为Defer。",
        f"这49个任务在32k原先有{before['lifecycle_binding_choices']['Local']}个绑定Local、{before['lifecycle_binding_choices']['Remote']}个绑定Remote。36k累计还有{u['queued_behind_other_edf_head_slots']}个task-slots排在其他EDF队首后，这些队首当时也实际Defer。这里是任务等待槽数，不是互不重叠的全局时隙数。",
        f"48个获得机会任务首次决策的inclusive slack中位数为{u['first_decision_slack_slots']['median']}槽，范围{u['first_decision_slack_slots']['minimum']}—{u['first_decision_slack_slots']['maximum']}槽；slack采用deadline-slot+1。绑定在槽末生效，因此slack还不能直接当作后续可服务槽数或完成保证。", "",
        "各检查点Route事实：", ""]
    for label in ("C44_20k", "C44_32k", "C44_36k", "C44_40k"):
        r = s["route_run_summaries"][label]
        lines.append(f"- {label}：active {r['active_decisions']}次，Local {r['choice_counts'].get('Local',0)}、Remote {r['choice_counts'].get('Remote',0)}、Defer {r['choice_counts'].get('Defer',0)}次。")
    lines += ["", "全部实际选择与合法叶argmax一致，EDF队首也与生命周期重建一致；未发现这部分存在错误索引或把Remote组质量当作执行叶动作的证据。该核验只覆盖已读取的轨迹。", "",
        "## D3：CPU能耗与服务", "", f"40k相对Heuristic多消耗总能量{energy['total_j']:.3f} J，其中CPU增加{energy['cpu_j']:.3f} J，占{energy['cpu_share_pct']:.2f}%。按各自任务终局记账，DONE增加{energy['cpu_j_by_outcome']['DONE']:.3f} J，EXPIRED变化{energy['cpu_j_by_outcome']['EXPIRED']:+.3f} J，TRUNCATED变化{energy['cpu_j_by_outcome']['TRUNCATED']:+.3f} J。因此额外CPU成本主要记在完成任务上，而非失败任务上。两策略完成的任务集合不同，这个分类不是相同任务的因果分解。",
        f"进一步配对双方都完成的{pair['task_count']}个相同任务，其实际完成cycles均为{pair['learned_served_cycles']/1e9:.3f}×10^9，但C44与Heuristic的CPU能耗分别为{pair['learned_cpu_energy_j']:.3f} J、{pair['reference_cpu_energy_j']:.3f} J。这排除了该子集仅由工作量不同解释能耗差，但仍不能分离目的地、频率、时序和资源异构各自的贡献。",
        f"C44满频正服务占{int(p['frequency_distribution']['1.0']['positive_service_slots'])}/{p['positive_service_slots']}（{100*p['frequency_distribution']['1.0']['positive_service_slots']/p['positive_service_slots']:.2f}%）；Heuristic为{int(h['frequency_distribution']['1.0']['positive_service_slots'])}/{h['positive_service_slots']}（{100*h['frequency_distribution']['1.0']['positive_service_slots']/h['positive_service_slots']:.2f}%）。C44总服务量{p['served_cycles']/1e9:.3f}×10^9 cycles，低于Heuristic的{h['served_cycles']/1e9:.3f}×10^9。频率分母只包含正CPU服务槽，不包括idle。",
        f"UAV2是能耗集中点：C44 CPU {p['uav_distribution']['2']['cpu_energy_j']:.3f} J，Heuristic {h['uav_distribution']['2']['cpu_energy_j']:.3f} J。节点编号沿用代码0—3；这是五场景之和。",
        f"32k→36k UAV3正服务槽由{int(c32['uav_distribution']['3']['positive_service_slots'])}降至{int(c36['uav_distribution']['3']['positive_service_slots'])}。36k只有{c36['backlogged_no_service_energy_feasible_slots']}个有积压且局部最低CPU能量可行、却无服务的节点槽，另{c36['backlogged_no_service_all_heads_energy_blocked_slots']}个积压无服务槽的所有源队首均不满足最低CPU能量需求。40k局部可行无服务槽为{p['backlogged_no_service_energy_feasible_slots']}。因此36k低点不支持用大量已入CPU后的局部可行闲置作主要解释。",
        "局部最低CPU能量检查未包含所有联合TX/CPU仲裁条件，不等于实际联合可执行性。遥测没有CPU proposal、完整合法mask和scheduler scores，不能由满频执行断定actor错误、门控错误或降频可无损。", "",
        "## D4：后期训练信号", "", "窗口按更新结束步数划分为(28k,32k]、(32k,36k]、(36k,40k]，分别16、15、16次更新，每次4个epoch；四个epoch不当作四倍独立训练样本。", ""]
    for w in s["seed44_training_windows"]:
        counts, b = w["route_selection_unique_rollout_counts"], w["branches"]
        active = sum(counts.values())
        lines.append(f"- {w['start_exclusive']//1000}→{w['end_inclusive']//1000}k：训练Route Defer {counts['defer']}/{active}（{100*counts['defer']/active:.2f}%），Route KL {b['route']['kl_sample_weighted_mean']:.7f}，联合CPU KL {b['cpu_queue']['kl_sample_weighted_mean']:.7f}，联合CPU clip {100*b['cpu_queue']['clip_sample_weighted_mean']:.3f}%，critic half-MSE {w['scalars']['critic_loss']['mean']:.4f}。")
    lines += ["", "32k→36k训练Defer比例明显上升，与开发轨迹方向一致，但Route KL没有相应暴增、Route clip在这些窗口均为0。36k→40k回升时联合CPU KL/clip反而更高，不能把这些标量单独作为36k低点根因。seed43末窗口训练Defer也升至48.27%，现象并非只在seed44存在。",
        "最终checkpoint中三种子实际actor学习率组分别约3e-5、3e-4，critic为3e-4；历史每次optimizer真实LR没有独立日志，保留NA。CSV alpha来自配置，不冒充实时optimizer读数。晚窗口探索barrier实际日志全0，Route entropy系数0.01不变；这不是在32k新发生的退火事件。",
        "本配置使用joint CPU调度；cpu_queue兼容字段承载联合log-prob，cpu_frequency的log-prob是零占位。后者零KL不表示频率策略没变化。critic_grad_norm仅覆盖value network，辅助Route value/Q的历史梯度范数未记录；Route/shared-trunk梯度来自联合actor loss，不能当独立分支梯度冲突证据。Route Q辅助未直接作为当前actor的counterfactual信用。",
        "三个完整CSV的8个重复列逐行值一致；seed42续训重放交叠16个更新的PPO记录一致，窗口没有双计。所读取字段与辅助loss均未发现NaN/Inf，但没有扫描每个历史tensor，不能声称训练全状态已排除数值问题。", "",
        "## 下一步与边界", "", "建议下一步做小规模冻结组合干预：在已见1042—1046场景，对C44的32k/36k比较原策略、只替换Route、只替换非Route服务，以及共同固定服务条件。先验证动作顺序、mask、循环状态、proposal/execution和外生轨迹，再执行有限开发评估。这样检验绑定行为与CPU服务之间的影响，不立即改reward或追加训练预算。具体组合和成本应在实施前固定。",
        "本轮D5干预未执行，也未实现MTS/QMIX或扩展训练。当前能确认行为与成本位置，不能宣布学习根因已查清。若下一轮干预没有形成明确可归因收益，就保留C现版本，推进同预算消融与强基线，不无限调参。", "",
        "## 审计、重建与已知修正", "", "正式Route证据使用limited_route_diagnostic.v2。首版曾把slack写为deadline-slot，已修正为inclusive定义并补测试；旧首版保留为superseded，不能作为当前正式slack证据。分析初次测试的临时目录权限问题已通过工作区隔离临时目录解决。",
        "原始诊断与统一验证入口：", "", "- `metadata/audits/seed44_limited_diagnostic/20261015_route_v2/route_diagnostic.json`",
        "- `metadata/audits/seed44_limited_diagnostic/20261015/cpu_diagnostic.json`", "- `metadata/audits/seed44_limited_diagnostic/20261015/training_diagnostic.json`",
        "- `metadata/audits/seed44_limited_diagnostic/20261015/diagnostic_summary.json`", "- `metadata/audits/seed44_limited_diagnostic/20261015/execution_validation.json`", "",
        "重建须使用新的输出路径，防止覆盖。本轮实际命令、stdout、JUnit、前后指纹与重建一致性记录保存在同一审计目录；最终验收数量以execution_validation.json为准。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", type=Path, default=AUDIT / "20261015_route_v2/route_diagnostic.json")
    parser.add_argument("--cpu", type=Path, default=AUDIT / "20261015/cpu_diagnostic.json")
    parser.add_argument("--training", type=Path, default=AUDIT / "20261015/training_diagnostic.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args(argv)
    summary_path = args.output_dir.resolve() / "diagnostic_summary.json"
    report_path = args.report_path.resolve()
    if not summary_path.is_relative_to(AUDIT) or not report_path.is_relative_to(ROOT / "docs/results/diagnostics"):
        parser.error("Use the diagnostic audit and result-document directories")
    for target in (summary_path, report_path):
        if target.exists():
            raise FileExistsError(f"Refusing overwrite: {target}")
    inputs = [args.route, args.cpu, args.training]
    summary = build_summary(*(json.loads(p.read_text(encoding="utf-8")) for p in inputs))
    summary["diagnostic_source_sha256"] = {str(p.resolve().relative_to(ROOT)).replace("\\", "/"):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    from ..artifacts import atomic_write_text_group
    atomic_write_text_group(((summary_path, json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                             (report_path, render_report(summary))))
    print(json.dumps({"summary": str(summary_path), "report": str(report_path),
                      "matched_completed_tasks": summary["matched_cpu_comparison"]["groups"]["both_done"]["task_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
