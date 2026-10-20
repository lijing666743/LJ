"""Strictly offline CPU accounting for saved development telemetry.

This module never constructs an environment, actor, or optimizer.  Energy is
summed from actual CPU debits, then checked against the physical identity and
episode metrics.  Slack is a local optimistic bound, never a policy observation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from ..config import load_run_config
from .checkpoint_energy_audit import (
    EnergyAuditError, cpu_energy_blocked, hardware_resources, require_close,
)


SERVICE_SCHEMA = "evaluation_service_telemetry.v1"
OUTCOMES = ("DONE", "EXPIRED", "TRUNCATED")
IDENTITY_KEYS = ("evaluation_run_id", "method_id", "evaluation_seed", "episode_id")
ROOT = Path(__file__).resolve().parents[2]


def _finite_nonnegative(value, label):
    if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise EnergyAuditError(f"Invalid nonnegative {label}: {value!r}")
    return float(value)


def validate_identity(rows, identity):
    """Reject mixed identities instead of silently folding methods together."""
    if set(identity) != set(IDENTITY_KEYS):
        raise EnergyAuditError("Incomplete declared episode identity")
    for row in rows:
        if row.get("schema_version") != SERVICE_SCHEMA:
            raise EnergyAuditError("Unexpected service schema")
        if any(row.get(key) != identity[key] for key in IDENTITY_KEYS):
            raise EnergyAuditError("Mixed or missing episode identity")


def _distribution(rows):
    """Additive accounting, with explicit zeroes for empty outcome buckets."""
    return {
        "positive_service_slots": len(rows),
        "served_cycles": math.fsum(r["served_cycles"] for r in rows),
        "cpu_energy_j": math.fsum(r["cpu_energy_j"] for r in rows),
    }


def analyze_cpu_episode(config, tasks, cpu_rows, tx_rows, metrics, *, identity):
    """Reconstruct one saved method/scene; no policy or simulation is run."""
    for rows in (tasks, cpu_rows, tx_rows):
        validate_identity(rows, identity)
    env = config.environment
    horizon, count = env.episode_horizon, env.uav_count
    resources = hardware_resources(config)
    levels = tuple(config.action.cpu_frequency_levels)
    minimum_level = min(level for level in levels if level > 0)
    by_id = {t["task_id"]: t for t in tasks}
    if len(by_id) != len(tasks) or len(tasks) != metrics["generated"]:
        raise EnergyAuditError("Duplicated tasks or generated-count mismatch")
    grid = {(r["slot"], r["uav_id"]): r for r in cpu_rows}
    expected_grid = {(slot, uav) for slot in range(horizon) for uav in range(count)}
    if len(grid) != len(cpu_rows) or set(grid) != expected_grid:
        raise EnergyAuditError("CPU grid missing, extra, or duplicated")
    tx_grid = {}
    for row in tx_rows:
        key = (row["slot"], row["uav_id"])
        if key not in expected_grid:
            raise EnergyAuditError("TX row outside episode grid")
        if row["link_summary"]:
            if key in tx_grid:
                raise EnergyAuditError("Duplicated TX energy summary")
            tx_grid[key] = row
    if set(tx_grid) != expected_grid:
        raise EnergyAuditError("Missing TX energy summary")
    eligible, terminal, remaining = {}, {}, {}
    outcomes = Counter()
    for tid, task in by_id.items():
        if task["final_status"] not in OUTCOMES:
            raise EnergyAuditError("Nonterminal lifecycle row")
        outcomes[task["final_status"]] += 1
        remaining[tid] = _finite_nonnegative(task["required_cycles"], "required cycles")
        _finite_nonnegative(task["remaining_cycles"], "remaining cycles")
        bind = task["route_bind_slot"]
        local = task["source_uav"] == task["destination_uav"]
        entry = bind if local else task["tx_complete_slot"]
        eligible[tid] = entry + 1 if entry is not None else horizon
        terminal[tid] = (task["done_slot"] if task["final_status"] == "DONE" else
                         task["expired_slot"] if task["final_status"] == "EXPIRED" else horizon - 1)
        if terminal[tid] is None or not 0 <= terminal[tid] < horizon:
            raise EnergyAuditError("Invalid task terminal slot")
        if task["final_status"] == "EXPIRED" and terminal[tid] != task["deadline_slot"]:
            raise EnergyAuditError("Expiry differs from deadline")
        if task["final_status"] == "DONE" and terminal[tid] > task["deadline_slot"]:
            raise EnergyAuditError("Task completed after deadline")
    for outcome, name in zip(OUTCOMES, ("completed", "expired", "truncated")):
        if outcomes[outcome] != metrics[name]:
            raise EnergyAuditError("Lifecycle outcomes differ from metrics")
    previous_energy = {uav: r["initial_energy"] for uav, r in resources.items()}
    service_rows, idle_rows, checks = [], [], Counter()
    per_task = {tid: {"task_id": tid, "source_uav": t["source_uav"],
                     "destination_uav": t["destination_uav"], "final_status": t["final_status"],
                     "terminal_stage": t["terminal_stage"], "required_cycles": t["required_cycles"],
                     "remaining_cycles": t["remaining_cycles"], "cpu_energy_j": 0.0,
                     "served_cycles": 0.0, "positive_service_slots": 0,
                     "first_service_slot": None, "last_service_slot": None,
                     "negative_optimistic_slack_service_slots": 0}
                for tid, t in by_id.items()}
    for slot in range(horizon):
        for uav in range(count):
            row, resource = grid[slot, uav], resources[uav]
            energy = _finite_nonnegative(row["cpu_energy_j"], "CPU energy")
            cycles = _finite_nonnegative(row["served_cycles"], "served cycles")
            frequency = _finite_nonnegative(row["cpu_frequency"], "CPU frequency")
            tx_energy = _finite_nonnegative(tx_grid[slot, uav]["tx_energy_j"], "TX energy")
            before_energy = row["residual_energy_j"] + energy + tx_energy
            require_close(before_energy, previous_energy[uav], "Energy balance", absolute=1e-8)
            previous_energy[uav] = row["residual_energy_j"]
            if previous_energy[uav] < -env.energy_tolerance_j:
                raise EnergyAuditError("Negative residual energy")
            level = min(levels, key=lambda value: abs(frequency / resource["frequency"] - value))
            require_close(frequency, level * resource["frequency"], "Allowed frequency", absolute=1e-5)
            queue = [t for t in tasks if t["destination_uav"] == uav
                     and eligible[t["task_id"]] <= slot <= terminal[t["task_id"]]
                     and remaining[t["task_id"]] > 0]
            if len(queue) != row["cpu_queue_length"]:
                raise EnergyAuditError(f"Queue count mismatch at {slot}/{uav}")
            require_close(math.fsum(remaining[t["task_id"]] for t in queue),
                          row["cpu_queue_cycles"], "Queue cycles", absolute=1e-4)
            heads = {}
            for task in sorted(queue, key=lambda t: (t["deadline_slot"], t["task_id"])):
                heads.setdefault(task["source_uav"], task["task_id"])
            blocked, cheapest = cpu_energy_blocked(
                [remaining[tid] for tid in heads.values()], energy=before_energy,
                frequency=resource["frequency"], coefficient=resource["coefficient"],
                minimum_level=minimum_level, slot_duration=env.slot_duration_s,
                tolerance=env.energy_tolerance_j)
            if cycles > 0:
                tid = row["served_task_id"]
                if tid not in heads.values() or row["selected_task_id"] != tid or frequency <= 0:
                    raise EnergyAuditError("Service does not select a source-queue EDF head")
                require_close(cycles, min(remaining[tid], frequency * env.slot_duration_s),
                              "Service amount", absolute=1e-4)
                require_close(energy, resource["coefficient"] * frequency ** 2 * cycles,
                              "Actual CPU energy identity", absolute=1e-8)
                if energy > before_energy + env.energy_tolerance_j:
                    raise EnergyAuditError("Service exceeds energy budget")
                task = by_id[tid]
                inclusive_slots = task["deadline_slot"] - slot + 1
                optimistic_slots = math.ceil(remaining[tid] / (resource["frequency"] * env.slot_duration_s))
                slack = inclusive_slots - optimistic_slots
                service_rows.append({"slot": slot, "uav_id": uav, "task_id": tid,
                    "final_status": task["final_status"], "frequency_level": level,
                    "cpu_frequency_hz": frequency, "served_cycles": cycles, "cpu_energy_j": energy,
                    "deadline_remaining_inclusive_slots": inclusive_slots,
                    "optimistic_compute_slack_slots": slack})
                record = per_task[tid]
                record["cpu_energy_j"] += energy
                record["served_cycles"] += cycles
                record["positive_service_slots"] += 1
                record["negative_optimistic_slack_service_slots"] += int(slack < 0)
                if record["first_service_slot"] is None:
                    record["first_service_slot"] = slot
                record["last_service_slot"] = slot
                remaining[tid] = max(0.0, remaining[tid] - cycles)
                checks["positive_service_identity"] += 1
            else:
                if row["served_task_id"] is not None or energy != 0:
                    raise EnergyAuditError("Zero service has task attribution or positive debit")
                if queue:
                    idle_rows.append({"slot": slot, "uav_id": uav,
                        "queue_length": len(queue), "energy_before_j": before_energy,
                        "all_source_heads_energy_blocked": blocked,
                        "minimum_head_reservation_j": cheapest,
                        "executed_no_selected_task": row["selected_task_id"] is None,
                        "executed_frequency_zero": frequency == 0})
            checks["queue_and_energy_slot"] += 1
    for tid, task in by_id.items():
        require_close(remaining[tid], task["remaining_cycles"], "Final task remaining cycles", absolute=1e-4)
        require_close(per_task[tid]["served_cycles"], task["required_cycles"] - task["remaining_cycles"],
                      "Task cycle conservation", absolute=1e-4)
        if per_task[tid]["first_service_slot"] != task["cpu_first_service_slot"]:
            raise EnergyAuditError("First service differs from lifecycle")
        if task["final_status"] == "DONE" and task["remaining_cycles"] != 0:
            raise EnergyAuditError("DONE task has remaining cycles")
        checks["task_cycle_and_first_service"] += 1
    cpu_energy = math.fsum(r["cpu_energy_j"] for r in cpu_rows)
    tx_energy = math.fsum(r["tx_energy_j"] for r in tx_grid.values())
    require_close(cpu_energy, metrics["cpu_energy_j"], "Episode CPU total", absolute=1e-8)
    require_close(tx_energy, metrics["tx_energy_j"], "Episode TX total", absolute=1e-8)
    require_close(cpu_energy + tx_energy, metrics["total_energy_j"], "Episode total energy", absolute=1e-8)
    require_close(math.fsum(t["cpu_energy_j"] for t in per_task.values()), cpu_energy,
                  "Per-task CPU attribution", absolute=1e-8)
    summary = {"generated": len(tasks), "completed": outcomes["DONE"], "expired": outcomes["EXPIRED"],
        "truncated": outcomes["TRUNCATED"], "cpu_energy_j": cpu_energy, "tx_energy_j": tx_energy,
        "total_energy_j": cpu_energy + tx_energy,
        "served_cycles": math.fsum(r["served_cycles"] for r in cpu_rows),
        "positive_service_slots": len(service_rows), "backlogged_no_service_slots": len(idle_rows),
        "backlogged_no_service_all_heads_energy_blocked_slots": sum(r["all_source_heads_energy_blocked"] for r in idle_rows),
        "backlogged_no_service_energy_feasible_slots": sum(not r["all_source_heads_energy_blocked"] for r in idle_rows),
        "backlogged_no_service_no_selected_task_slots": sum(r["executed_no_selected_task"] for r in idle_rows),
        "negative_optimistic_slack_service_slots": sum(r["optimistic_compute_slack_slots"] < 0 for r in service_rows),
        "cpu_energy_on_negative_optimistic_slack_j": math.fsum(r["cpu_energy_j"] for r in service_rows if r["optimistic_compute_slack_slots"] < 0),
        "frequency_distribution": {str(level): _distribution([r for r in service_rows if r["frequency_level"] == level]) for level in levels if level > 0},
        "outcome_distribution": {outcome: {**_distribution([r for r in service_rows if r["final_status"] == outcome]),
            "generated_tasks": outcomes[outcome], "tasks_ever_served": sum(t["positive_service_slots"] > 0 and t["final_status"] == outcome for t in per_task.values())}
            for outcome in OUTCOMES},
        "uav_distribution": {str(uav): {**_distribution([r for r in service_rows if r["uav_id"] == uav]),
            "final_energy_j": previous_energy[uav], "maximum_frequency_hz": resources[uav]["frequency"],
            "backlogged_no_service_slots": sum(r["uav_id"] == uav for r in idle_rows)} for uav in range(count)}}
    return {"identity": dict(identity), "summary": summary, "checks": dict(checks),
            "tasks": list(per_task.values()), "backlogged_no_service": idle_rows}


def pool_cpu_summaries(episodes):
    """Pool actual scene totals without treating scenes as training repeats."""
    rows = [episode["summary"] for episode in episodes]
    scalar_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    result = {key: math.fsum(row[key] for row in rows) for key in scalar_keys}
    for key in scalar_keys:
        if all(isinstance(row[key], int) for row in rows):
            result[key] = int(result[key])
    for name in ("frequency_distribution", "outcome_distribution", "uav_distribution"):
        result[name] = {}
        for group in rows[0][name]:
            fields = rows[0][name][group]
            result[name][group] = {key: math.fsum(row[name][group][key] for row in rows) for key in fields
                                   if key != "maximum_frequency_hz"}
            if "maximum_frequency_hz" in fields:
                values = {row[name][group]["maximum_frequency_hz"] for row in rows}
                if len(values) != 1:
                    raise EnergyAuditError("UAV resources differ across scenes")
                result[name][group]["maximum_frequency_hz"] = values.pop()
    result["completion_pct"] = 100 * result["completed"] / result["generated"]
    result["cpu_j_per_billion_served_cycles"] = 1e9 * result["cpu_energy_j"] / result["served_cycles"] if result["served_cycles"] else None
    return result


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def analyze_cpu_saved_run(config_path, receipt_path, *, method_id="ca_gat_mappo"):
    """Load saved artifacts with source/run/schema/scene checks and file hashes."""
    receipt = _json(receipt_path)
    config = load_run_config(config_path)
    manifest_path = next(p for p in receipt["evaluation_artifacts"] if Path(p).name == "evaluation_manifest.json")
    metrics_path = next(p for p in receipt["evaluation_artifacts"] if Path(p).name == "episode_metrics.jsonl")
    service_paths = receipt["service_artifacts"]
    manifest, service_manifest = _json(manifest_path), _json(service_paths[0])
    for item in (manifest, service_manifest):
        if item["evaluation_run_id"] != receipt["evaluation_run_id"]:
            raise EnergyAuditError("Manifest run does not match receipt")
        if item["source_checkpoint_sha256"] != receipt["source_checkpoint_sha256"]:
            raise EnergyAuditError("Manifest source checkpoint does not match receipt")
        if tuple(item["evaluation_seeds"]) != tuple(config.evaluation.evaluation_seeds):
            raise EnergyAuditError("Manifest evaluation seeds differ from configuration")
    if manifest["evaluation_config_sha256"] != receipt["evaluation_config_sha256"]:
        raise EnergyAuditError("Protocol hash differs from receipt")
    if service_manifest["source_training_config_hash"] != manifest["source_training_config_hash"]:
        raise EnergyAuditError("Source training configuration hashes differ")
    if service_manifest["schema_version"] != SERVICE_SCHEMA or service_manifest["frequency_unit"] != "Hz":
        raise EnergyAuditError("Unsupported service schema or frequency unit")
    if receipt["network_updates"] != 0 or receipt.get("training_calls", 0) != 0:
        raise EnergyAuditError("Receipt does not describe frozen evaluation")
    groups = []
    paths = [config_path, receipt_path, manifest_path, metrics_path, *service_paths]
    before = {str(Path(p).resolve()): _sha(p) for p in paths}
    suite = set(service_manifest["method_suite"])
    scenes = tuple(config.evaluation.evaluation_seeds)
    for path in [*service_paths[1:], metrics_path]:
        grouped = defaultdict(list)
        for row in _jsonl(path):
            if row.get("evaluation_run_id") != receipt["evaluation_run_id"]:
                raise EnergyAuditError("Foreign run in mixed source artifact")
            if row.get("method_id") not in suite or row.get("evaluation_seed") not in scenes:
                raise EnergyAuditError("Foreign method/scene in mixed source artifact")
            if path != metrics_path and row.get("episode_id") != scenes.index(row["evaluation_seed"]):
                raise EnergyAuditError("Source episode id differs from scene index")
            if path != metrics_path and row.get("schema_version") != SERVICE_SCHEMA:
                raise EnergyAuditError("Foreign service schema in mixed source artifact")
            grouped[row["method_id"], row["evaluation_seed"]].append(row)
        if set(grouped) != {(method, seed) for method in suite for seed in scenes}:
            raise EnergyAuditError("Mixed source artifact has an incomplete suite")
        groups.append(grouped)
    for grouped, count_key in zip(groups[:3], ("task_count", "cpu_record_count", "tx_record_count")):
        if sum(map(len, grouped.values())) != service_manifest[count_key]:
            raise EnergyAuditError("Artifact row count differs from service manifest")
    episodes = []
    workload = {}
    for index, seed in enumerate(scenes):
        records = groups[3][method_id, seed]
        if len(records) != 1:
            raise EnergyAuditError("Expected one metrics row per method/scene")
        identity = dict(zip(IDENTITY_KEYS, (receipt["evaluation_run_id"], method_id, seed, index)))
        episodes.append(analyze_cpu_episode(config, groups[0][method_id, seed], groups[1][method_id, seed],
                                            groups[2][method_id, seed], records[0]["metrics"], identity=identity))
        workload[str(seed)] = sorted((t["task_id"], t["source_uav"], t["generation_slot"], t["deadline_slot"],
                                      t["input_bits"], t["required_cycles"]) for t in groups[0][method_id, seed])
        trace = manifest["shared_external_trace_by_seed"][str(seed)]
        if len(set(trace["method_trace_sha256"].values()) | {trace["shared_external_trace_sha256"]}) != 1:
            raise EnergyAuditError("External process traces differ across methods")
    if any(_sha(p) != sha for p, sha in before.items()):
        raise EnergyAuditError("An input changed during offline analysis")
    return {"method_id": method_id, "training_seed": receipt.get("training_seed", config.seed),
            "checkpoint_steps": receipt["checkpoint_steps"], "pooled": pool_cpu_summaries(episodes),
            "episodes": episodes, "input_sha256": before, "workload_identity": workload,
            "external_trace_identity": {str(seed): manifest["shared_external_trace_by_seed"][str(seed)]["shared_external_trace_sha256"] for seed in scenes}}


def build_cpu_diagnostic(root=ROOT):
    """Analyze the predeclared checkpoints and one fixed historical Heuristic."""
    root = Path(root)
    specs = []
    for seed, steps, directory in (
        (44, (20, 32, 36, 40), "c_seed44_progress/20261015_seed44_4k40k"),
        (43, (32, 36, 40), "c_seed43_progress/20261015_seed43_4k40k"),
        (42, (32,), "c32k_progress/20261014_16k32k"),
        (42, (36, 40), "c40k_progress/20261014_32k40k"),
    ):
        for step in steps:
            prefix = f"C{seed}" if seed in (43, 44) else "C"
            base = root / "metadata/audits" / directory
            specs.append((base / f"{prefix}_{step}k_evaluation_config.json", base / f"{prefix}_{step}k_run.json"))
    results = [analyze_cpu_saved_run(config, receipt) for config, receipt in specs]
    baseline_config, baseline_receipt = specs[3]
    results.append(analyze_cpu_saved_run(baseline_config, baseline_receipt, method_id="heuristic"))
    workload = results[0]["workload_identity"]
    external = results[0]["external_trace_identity"]
    if any(r["workload_identity"] != workload or r["external_trace_identity"] != external for r in results):
        raise EnergyAuditError("Workloads or external processes differ across comparisons")
    for result in results:
        del result["workload_identity"]
    return {"schema_version": "limited_cpu_diagnostic.v1", "status": "PASS", "date": "2026-10-15",
            "scope": "Saved development telemetry only; no rollout, training, reward/config changes, or held-out scenes",
            "slack_definition": "deadline_slot-slot+1 minus ceil(remaining_cycles/(destination_max_frequency*slot_duration)); optimistic local bound ignoring contention, energy and later decisions",
            "idle_definition": "CPU queue present at slot start but zero actual served cycles; local minimum reservation feasibility is not a joint action feasibility or lost-completion proof",
            "outcome_energy_boundary": "Energy attributed to eventual DONE/EXPIRED/TRUNCATED tasks is descriptive; no claim it was avoidable or could be lowered without affecting completions",
            "cases": results, "paired_workloads_checked_per_case": sum(map(len, workload.values()))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Existing diagnostic is immutable; choose a new output path")
    result = build_cpu_diagnostic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]),
                      "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
