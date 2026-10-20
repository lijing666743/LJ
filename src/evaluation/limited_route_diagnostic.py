"""Pure offline task-paired Route diagnosis from existing development telemetry.

This module imports no environment, actor, optimizer, or random-number package.
It reconstructs the factual UNBOUND EDF queue from arrival/binding/expiry times
and verifies every expected head against the recorded active Route decisions.
Queue blocking is descriptive; no counterfactual completion claim is inferred.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


METHOD = "ca_gat_mappo"
SCENES = (1042, 1043, 1044, 1045, 1046)
IDENTITY_FIELDS = ("evaluation_seed", "task_id", "source_uav", "generation_slot",
                   "deadline_slot", "input_bits", "required_cycles")
RUN_SPECS = ((44, 20), (44, 32), (44, 36), (44, 40),
             (42, 32), (42, 36), (42, 40), (43, 32), (43, 36), (43, 40))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def task_key(row: dict) -> tuple[int, int]:
    return int(row["evaluation_seed"]), int(row["task_id"])


def describe(values) -> dict:
    values = list(values)
    if not values:
        return {"n": 0, "minimum": None, "median": None, "mean": None, "maximum": None}
    return {"n": len(values), "minimum": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "maximum": max(values)}


def max_consecutive(slots) -> int:
    best = current = 0
    previous = None
    for slot in sorted(set(slots)):
        current = current + 1 if previous is not None and slot == previous + 1 else 1
        best = max(best, current)
        previous = slot
    return best


def leaf_metrics(row: dict) -> dict:
    """Use legal flat leaves, never Remote aggregate mass as leaf argmax."""
    scores, probs, mask = row["leaf_scores"], row["leaf_probabilities"], row["route_legal_mask"]
    domain = row["route_action_domain"]
    if not len(scores) == len(probs) == len(mask) == len(domain):
        raise ValueError("Route leaf arrays have incompatible lengths")
    if row.get("route_action_mask", mask) != mask:
        raise ValueError("Factual and leaf legal masks differ")
    legal = [i for i, active in enumerate(mask) if active]
    if not legal or domain[0] != "idle" or mask[0]:
        raise ValueError("Trace must contain an active non-idle Route decision")
    if any(not math.isfinite(float(v)) for v in [*scores, *probs]):
        raise ValueError("Nonfinite Route leaf value")
    if any(p < 0 for p in probs) or not math.isclose(sum(probs), 1.0, abs_tol=1e-5):
        raise ValueError("Invalid leaf probabilities")
    if any(probs[i] != 0.0 for i in range(len(mask)) if not mask[i]):
        raise ValueError("Illegal leaf has probability mass")
    chosen = int(row["chosen_leaf_index"])
    winner = max(legal, key=lambda i: scores[i])
    if chosen != winner or int(row["leaf_argmax_index"]) != winner:
        raise ValueError("Recorded choice differs from deterministic legal leaf argmax")
    top = "Local" if chosen == 1 else "Defer" if chosen == 2 else "Remote"
    if row["selected_top_level"] != top or row["actor_decision_argmax"] != top:
        raise ValueError("Route action label disagrees with selected leaf")
    binding = [i for i in legal if i != 2]
    if not binding or 2 not in legal:
        raise ValueError("Expected local/Defer legal alternatives")
    best_bind = max(binding, key=lambda i: scores[i])
    remote = [i for i in binding if i >= 3]
    best_remote = max(remote, key=lambda i: scores[i]) if remote else None
    runner_up = max((i for i in legal if i != chosen), key=lambda i: scores[i])
    return {
        "legal_leaf_count": len(legal), "legal_binding_count": len(binding),
        "legal_remote_count": len(remote), "chosen_leaf_label": domain[chosen],
        "chosen_minus_runner_up_score": scores[chosen] - scores[runner_up],
        "chosen_minus_runner_up_probability": probs[chosen] - probs[runner_up],
        "defer_minus_best_binding_score": scores[2] - scores[best_bind],
        "defer_minus_best_binding_probability": probs[2] - probs[best_bind],
        "local_minus_best_remote_score": None if best_remote is None else scores[1] - scores[best_remote],
        "local_minus_best_remote_probability": None if best_remote is None else probs[1] - probs[best_remote],
        "remote_mass_argmax_differs_from_actual_leaf": row["group_mass_argmax"] != top,
    }


def analyze_run(lifecycle_rows: list[dict], route_rows: list[dict], horizons: dict[int, int]) -> dict:
    """Analyze factual task/head history, rejecting missing or inconsistent traces."""
    lifecycle_rows = [r for r in lifecycle_rows if r["method_id"] == METHOD]
    tasks = {task_key(r): r for r in lifecycle_rows}
    if len(tasks) != len(lifecycle_rows):
        raise ValueError("Duplicate task identity")
    by_task = defaultdict(list)
    by_source_slot = {}
    leaves = []
    for row in route_rows:
        key = task_key(row)
        if key not in tasks:
            raise ValueError("Route task is missing from lifecycle")
        task = tasks[key]
        slot = int(row["step"])
        if (int(row["uav_id"]) != task["source_uav"]
                or not task["generation_slot"] < slot <= task["deadline_slot"]
                or slot >= horizons[key[0]]
                or (task["route_bind_slot"] is not None and slot > task["route_bind_slot"])):
            raise ValueError("Route decision is outside the task's factual UNBOUND window")
        source_slot = (key[0], int(row["uav_id"]), slot)
        if source_slot in by_source_slot:
            raise ValueError("Duplicate source/slot Route decision")
        by_source_slot[source_slot] = row
        metrics = leaf_metrics(row)
        selected = row["selected_top_level"]
        if selected != "Defer":
            if task["route_bind_slot"] != slot:
                raise ValueError("Selected binding does not match lifecycle binding time")
            destination = task["source_uav"] if selected == "Local" else row["selected_remote_id"]
            if destination != task["destination_uav"]:
                raise ValueError("Selected binding does not match lifecycle destination")
        elif task["route_bind_slot"] == slot:
            raise ValueError("Defer row conflicts with lifecycle binding")
        by_task[key].append({**row, "offline_leaf_metrics": metrics})
        leaves.append(metrics)

    # At slot s arrivals from s-1 exist, binding at s is still UNBOUND at
    # slot start, and deadline s settles only after that slot's decision.
    waiting = defaultdict(list)
    head_slots = {}
    source_tasks = defaultdict(list)
    for key, task in tasks.items():
        source_tasks[(key[0], task["source_uav"])].append(task)
    expected_decisions = 0
    for (scene, source), source_rows in sorted(source_tasks.items()):
        for slot in range(horizons[scene]):
            eligible = [t for t in source_rows if t["generation_slot"] < slot <= t["deadline_slot"]
                        and (t["route_bind_slot"] is None or t["route_bind_slot"] >= slot)]
            row = by_source_slot.get((scene, source, slot))
            if not eligible:
                if row is not None:
                    raise ValueError("Route decision without an eligible UNBOUND task")
                continue
            head = min(eligible, key=lambda t: (t["deadline_slot"], t["task_id"]))
            if row is None or row["task_id"] != head["task_id"]:
                raise ValueError("Reconstructed EDF head differs from Route trace or trace is missing")
            expected_decisions += 1
            head_slots[(scene, source, slot)] = row
            for task in eligible:
                if task["task_id"] != head["task_id"]:
                    waiting[(scene, task["task_id"])].append({
                        "slot": slot, "head_task_id": head["task_id"],
                        "head_action": row["selected_top_level"],
                    })
    if expected_decisions != len(route_rows):
        raise ValueError("Expected active head count differs from Route row count")

    task_details = {}
    for key, task in sorted(tasks.items()):
        decisions = sorted(by_task[key], key=lambda r: r["step"])
        defer = [r for r in decisions if r["selected_top_level"] == "Defer"]
        first = decisions[0]["step"] if decisions else None
        end = task["route_bind_slot"]
        if end is None:
            end = min(task["deadline_slot"], horizons[key[0]] - 1)
        eligible_slots = list(range(task["generation_slot"] + 1, end + 1))
        observed = {r["step"] for r in decisions}
        wait_rows = waiting[key]
        wait_slots = {r["slot"] for r in wait_rows}
        if observed | wait_slots != set(eligible_slots) or observed & wait_slots:
            raise ValueError("Task eligible slots not partitioned into head and queued slots")
        # Waiting behind a deferred EDF head is observed queue occupancy, not
        # proof that an alternative head decision would have completed this task.
        task_details[f"{key[0]}:{key[1]}"] = {
            "lifecycle": dict(task), "active_decision_count": len(decisions),
            "defer_count": len(defer), "ever_visible_defer": bool(defer),
            "never_observed_as_head": not decisions,
            "all_observed_decisions_defer": bool(decisions) and len(defer) == len(decisions),
            "first_decision_slot": first,
            "first_decision_slack_slots": None if first is None else task["deadline_slot"] - first + 1,
            "first_decision_deadline_offset_slots": None if first is None else task["deadline_slot"] - first,
            "arrival_to_first_decision_slots": None if first is None else first - task["generation_slot"],
            "first_eligible_to_first_decision_slots": None if first is None else first - task["generation_slot"] - 1,
            "binding_slack_slots": None if task["route_bind_slot"] is None else task["deadline_slot"] - task["route_bind_slot"] + 1,
            "post_binding_remaining_service_slots": None if task["route_bind_slot"] is None else task["deadline_slot"] - task["route_bind_slot"],
            "eligible_unbound_slots": len(eligible_slots),
            "no_eligible_route_slot_before_terminal": not eligible_slots,
            "eligible_slots_all_behind_other_head": bool(eligible_slots) and not decisions,
            "max_consecutive_defer_slots": max_consecutive(r["step"] for r in defer),
            "queued_behind_other_edf_head_slots": len(wait_rows),
            "queued_behind_deferred_edf_head_slots": sum(r["head_action"] == "Defer" for r in wait_rows),
            "max_consecutive_slots_without_head_opportunity": max_consecutive(wait_slots),
            "earlier_head_tasks": dict(Counter(str(r["head_task_id"]) for r in wait_rows)),
            "first_decision": None if not decisions else compact_decision(decisions[0], task),
            "last_decision": None if not decisions else compact_decision(decisions[-1], task),
            "decision_history": [compact_decision(r, task) for r in decisions],
        }
    counts = Counter(r["selected_top_level"] for r in route_rows)
    detail_rows = list(task_details.values())
    summary = {
        "tasks": len(tasks), "completed": sum(t["final_status"] == "DONE" for t in tasks.values()),
        "terminal_outcomes": dict(Counter(f"{t['final_status']}/{t.get('terminal_stage')}" for t in tasks.values())),
        "active_decisions": len(route_rows), "choice_counts": dict(counts),
        "unique_deferred_tasks": sum(r["ever_visible_defer"] for r in detail_rows),
        "never_head_tasks": sum(r["never_observed_as_head"] for r in detail_rows),
        "no_eligible_route_slot_before_terminal_tasks": sum(r["no_eligible_route_slot_before_terminal"] for r in detail_rows),
        "legal_remote_count_distribution": dict(Counter(str(r["legal_remote_count"]) for r in leaves)),
        "group_mass_argmax_differs_from_actual_leaf_count": sum(r["remote_mass_argmax_differs_from_actual_leaf"] for r in leaves),
        "first_decision_slack_slots": describe(r["first_decision_slack_slots"] for r in detail_rows if r["first_decision_slack_slots"] is not None),
        "binding_slack_slots": describe(r["binding_slack_slots"] for r in detail_rows if r["binding_slack_slots"] is not None),
        "max_consecutive_defer_slots": describe(r["max_consecutive_defer_slots"] for r in detail_rows),
        "queued_behind_other_edf_head_slots": sum(r["queued_behind_other_edf_head_slots"] for r in detail_rows),
        "queued_behind_deferred_edf_head_slots": sum(r["queued_behind_deferred_edf_head_slots"] for r in detail_rows),
        "defer_minus_best_binding_score_on_defer": describe(r["offline_leaf_metrics"]["defer_minus_best_binding_score"] for d in by_task.values() for r in d if r["selected_top_level"] == "Defer"),
        "chosen_minus_runner_up_probability": describe(r["chosen_minus_runner_up_probability"] for r in leaves),
        "exact_edf_head_reconstruction_passed": True,
    }
    per_scene = {}
    for scene in sorted(horizons):
        selected = [r for r in route_rows if r["evaluation_seed"] == scene]
        scene_tasks = [t for k, t in tasks.items() if k[0] == scene]
        per_scene[str(scene)] = {
            "tasks": len(scene_tasks), "completed": sum(t["final_status"] == "DONE" for t in scene_tasks),
            "active_decisions": len(selected),
            "choice_counts": dict(Counter(r["selected_top_level"] for r in selected)),
        }
    return {"summary": summary, "per_scene": per_scene, "tasks": task_details}


def compact_decision(row: dict, task: dict) -> dict:
    return {"slot": row["step"], "slack_slots": task["deadline_slot"] - row["step"] + 1,
            "deadline_offset_slots": task["deadline_slot"] - row["step"],
            "action": row["selected_top_level"], "selected_remote_id": row["selected_remote_id"],
            "legal_candidates": row["legal_route_candidates"],
            "legal_leaf_scores": {label: score for label, score, legal in zip(row["route_action_domain"], row["leaf_scores"], row["route_legal_mask"]) if legal},
            "legal_leaf_probabilities": {label: prob for label, prob, legal in zip(row["route_action_domain"], row["leaf_probabilities"], row["route_legal_mask"]) if legal},
            "group_mass": row["group_mass"], "group_mass_argmax": row["group_mass_argmax"],
            **row["offline_leaf_metrics"]}


def group_details(rows: list[dict]) -> dict:
    binding_choices = Counter("UNBOUND" if r["lifecycle"]["destination_uav"] is None else
                              "Local" if r["lifecycle"]["destination_uav"] == r["lifecycle"]["source_uav"] else
                              "Remote" for r in rows)
    return {
        "tasks": len(rows), "never_head": sum(r["never_observed_as_head"] for r in rows),
        "no_eligible_route_slot_before_terminal": sum(r["no_eligible_route_slot_before_terminal"] for r in rows),
        "eligible_slots_all_behind_other_head": sum(r["eligible_slots_all_behind_other_head"] for r in rows),
        "lifecycle_binding_choices": dict(binding_choices),
        "visible_defer": sum(r["ever_visible_defer"] for r in rows),
        "all_observed_decisions_defer": sum(r["all_observed_decisions_defer"] for r in rows),
        "repeated_defer_at_least_two": sum(r["defer_count"] >= 2 for r in rows),
        "defer_decisions": sum(r["defer_count"] for r in rows),
        "defer_decisions_with_legal_remote": sum(d["action"] == "Defer" and d["legal_remote_count"] > 0 for r in rows for d in r["decision_history"]),
        "active_decisions": sum(r["active_decision_count"] for r in rows),
        "queued_behind_other_edf_head_slots": sum(r["queued_behind_other_edf_head_slots"] for r in rows),
        "queued_behind_deferred_edf_head_slots": sum(r["queued_behind_deferred_edf_head_slots"] for r in rows),
        "first_decision_slack_slots": describe(r["first_decision_slack_slots"] for r in rows if r["first_decision_slack_slots"] is not None),
        "arrival_to_first_decision_slots": describe(r["arrival_to_first_decision_slots"] for r in rows if r["arrival_to_first_decision_slots"] is not None),
        "binding_slack_slots": describe(r["binding_slack_slots"] for r in rows if r["binding_slack_slots"] is not None),
        "defer_count": describe(r["defer_count"] for r in rows),
        "max_consecutive_defer_slots": describe(r["max_consecutive_defer_slots"] for r in rows),
    }


def compare_runs(old: dict, new: dict) -> dict:
    if old["tasks"].keys() != new["tasks"].keys():
        raise ValueError("Task identities differ across runs")
    lost, gained, both, neither = [], [], 0, 0
    for key, before in old["tasks"].items():
        after = new["tasks"][key]
        a, b = before["lifecycle"], after["lifecycle"]
        if any(a[field] != b[field] for field in IDENTITY_FIELDS):
            raise ValueError("Paired task workload/deadline differs")
        old_done, new_done = a["final_status"] == "DONE", b["final_status"] == "DONE"
        if old_done and not new_done:
            lost.append(key)
        elif new_done and not old_done:
            gained.append(key)
        elif old_done:
            both += 1
        else:
            neither += 1
    groups, old_groups = defaultdict(list), defaultdict(list)
    for key in lost:
        row = new["tasks"][key]
        task = row["lifecycle"]
        label = f"{task['final_status']}/{task['terminal_stage']}"
        groups[label].append(row)
        old_groups[label].append(old["tasks"][key])
    return {"task_identity_and_workload_exact": True, "tasks": len(old["tasks"]),
            "both_done": both, "lost": len(lost), "gained": len(gained), "neither_done": neither,
            "net_completed_change": len(gained) - len(lost),
            "lost_task_ids": lost, "gained_task_ids": gained,
            "lost_new_terminal_groups": {key: group_details(rows) for key, rows in sorted(groups.items())},
            "lost_before_by_new_terminal_group": {key: group_details(rows) for key, rows in sorted(old_groups.items())},
            "lost_before": group_details([old["tasks"][key] for key in lost]),
            "lost_after": group_details([new["tasks"][key] for key in lost]),
            "gained_before": group_details([old["tasks"][key] for key in gained]),
            "gained_after": group_details([new["tasks"][key] for key in gained]),
            "causal_intervention_performed": False}


def only_path(paths, label: str) -> Path:
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f"Expected one {label}, found {len(paths)}")
    return paths[0]


def build_diagnostic(root: Path) -> dict:
    runs, inputs = {}, {}
    for seed, checkpoint in RUN_SPECS:
        run_name = f"C{seed}_{checkpoint}k"
        search = root / "logs" / "C" / f"seed{seed}" / "checkpoint_progress"
        files = {}
        for filename in ("route_decision_trace.jsonl", "task_lifecycle.jsonl", "episode_metrics.jsonl"):
            path = only_path((p for p in search.glob(f"*/{checkpoint}k/**/{filename}")), f"{run_name} {filename}")
            files[filename] = path
            inputs[path.relative_to(root).as_posix()] = {"sha256_before": sha256(path), "bytes": path.stat().st_size}
        episodes = [r for r in read_jsonl(files["episode_metrics.jsonl"]) if r["method_id"] == METHOD]
        horizons = {int(r["evaluation_seed"]): int(r["episode_horizon"]) for r in episodes}
        if tuple(sorted(horizons)) != SCENES or len(episodes) != 5:
            raise ValueError("Require exactly the five existing development scenes")
        result = analyze_run(read_jsonl(files["task_lifecycle.jsonl"]), read_jsonl(files["route_decision_trace.jsonl"]), horizons)
        if result["summary"]["tasks"] != 605:
            raise ValueError("Require605 task-paired development arrivals")
        if sum(r["metrics"]["completed"] for r in episodes) != result["summary"]["completed"]:
            raise ValueError("Reconstructed DONE count differs from recorded evaluation")
        result["episode_horizons"] = horizons
        result["external_trace_sha256"] = {r["evaluation_seed"]: r["external_trace_sha256"] for r in episodes}
        result["inputs"] = {k: str(v.relative_to(root).as_posix()) for k, v in files.items()}
        runs[run_name] = result
    # All ten checkpoints must share task workloads and external processes.
    reference = runs["C44_32k"]
    for run in runs.values():
        compare_runs(reference, run)
        if run["external_trace_sha256"] != reference["external_trace_sha256"]:
            raise ValueError("External evaluation trace mismatch")
    pairs = [(f"C{s}_{a}k", f"C{s}_{b}k") for s in (42, 43, 44) for a, b in ((32, 36), (36, 40))]
    pairs.append(("C44_20k", "C44_40k"))
    comparisons = {f"{a}_to_{b}": compare_runs(runs[a], runs[b]) for a, b in pairs}
    for path, record in inputs.items():
        record["sha256_after"] = sha256(root / path)
        if record["sha256_before"] != record["sha256_after"]:
            raise ValueError("Input changed during offline diagnosis")
    return {
        "schema_version": "limited_route_diagnostic.v2", "analysis_date": "2026-10-15",
        "scope": "Existing five development scenes only; pure offline reconstruction; no actor calls, environment steps, training, checkpoint selection or heldout access",
        "interpretation": {
            "slack_definition": "Inclusive environment convention: deadline_slot - decision_or_binding_slot + 1. Route binding occurs at slot end; post_binding_remaining_service_slots equals deadline_slot - binding_slot.",
            "facts": "Arrival, deadline, binding, final stage, actual legal leaf decisions and exact reconstructed EDF heads",
            "descriptive_association": "Waiting behind another EDF head (including a deferred head) is factual queue occupancy; alternative completion is unknown",
            "unidentified": "Causal completion effects of Defer/EDF/one-binding-per-slot, alternate-route feasibility, and completion feasibility from deadline slack alone",
        },
        "all_30_inputs_unchanged_during_analysis": True,
        "all_tasks_workload_and_external_traces_match": True,
        "inputs": inputs, "runs": runs, "comparisons": comparisons,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output.resolve()
    allowed = root / "metadata" / "audits" / "seed44_limited_diagnostic"
    if not output.is_relative_to(allowed) or output.name != "route_diagnostic.json":
        raise ValueError("Output must be a new route_diagnostic.json under the dedicated audit directory")
    if output.exists():
        raise FileExistsError("Offline diagnosis never overwrites existing output")
    result = build_diagnostic(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    pair = result["comparisons"]["C44_32k_to_C44_36k"]
    print(json.dumps({"output": str(output), "sha256": sha256(output),
                      "input_files": len(result["inputs"]),
                      "run_summaries": {k: {field: v["summary"][field] for field in
                                           ("tasks", "completed", "active_decisions", "choice_counts", "exact_edf_head_reconstruction_passed")}
                                        for k, v in result["runs"].items()},
                      "primary_pair": {field: pair[field] for field in
                                       ("lost", "gained", "net_completed_change", "lost_new_terminal_groups", "lost_before_by_new_terminal_group")}},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
