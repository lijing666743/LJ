"""Offline energy-risk reconstruction from immutable service telemetry.

No actor, environment, optimizer, or alternative policy is executed here.
Public-energy values are reconstructed scalars, not logged actor tensors.
"""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path


class EnergyAuditError(ValueError):
    """Telemetry violates the reconstruction's declared identities."""


def require_close(actual, expected, label, *, absolute=1e-5):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=absolute):
        raise EnergyAuditError(f"{label}: {actual} != {expected}")


def hardware_resources(config):
    """Use the fixed, published profile mapping without constructing an env."""
    env = config.environment
    result = {}
    for uav in range(env.uav_count):
        f, _, energy, coefficient = env.profile_ratios[env.profile_assignment[uav]]
        df, _, de, dc = env.profile_perturbations[uav]
        result[uav] = {
            "frequency": env.reference_cpu_frequency_hz * f * (1 + df),
            "coefficient": env.reference_cpu_coefficient * coefficient * (1 + dc),
            "initial_energy": env.reference_initial_energy_j * energy * (1 + de),
        }
    return result


def cpu_energy_blocked(head_cycles, *, energy, frequency, coefficient,
                       minimum_level, slot_duration, tolerance):
    """Only source-queue heads are eligible; final partial service is reserved."""
    if not head_cycles:
        return False, None
    minimum_frequency = minimum_level * frequency
    reservations = [coefficient * minimum_frequency ** 2
                    * min(cycles, minimum_frequency * slot_duration)
                    for cycles in head_cycles]
    cheapest = min(reservations)
    return cheapest > energy + tolerance, cheapest


def persistent_blockage_start(blocked_slots, positive_slots, backlogged_slots):
    """First blocked suffix after the last positive service; gaps are allowed."""
    blocked = set(blocked_slots)
    last_positive = max(positive_slots, default=-1)
    last_feasible_backlog = max((s for s in backlogged_slots if s not in blocked), default=-1)
    floor = max(last_positive, last_feasible_backlog)
    return min((s for s in blocked if s > floor), default=None)


def reconstruct_episode(config, tasks, cpu_rows, tx_rows, route_rows, metrics):
    """Return measured and reconstructed metrics for one method and scene."""
    env = config.environment
    count, horizon = env.uav_count, env.episode_horizon
    tolerance = env.energy_tolerance_j
    minimum_level = min(x for x in config.action.cpu_frequency_levels if x > 0)
    resources = hardware_resources(config)
    task_by_id = {t["task_id"]: t for t in tasks}
    if len(task_by_id) != len(tasks) or len(tasks) != metrics["generated"]:
        raise EnergyAuditError("Task identity or generated count differs")
    cpu = {(r["slot"], r["uav_id"]): r for r in cpu_rows}
    if len(cpu) != len(cpu_rows) or len(cpu) != count * horizon:
        raise EnergyAuditError("CPU slot grid is incomplete or duplicated")
    tx_by_slot = defaultdict(list)
    for row in tx_rows:
        tx_by_slot[row["slot"]].append(row)
    bindings = defaultdict(list)
    eligible = {}
    terminal = {}
    for task in tasks:
        tid = task["task_id"]
        if task["route_bind_slot"] is not None:
            bindings[task["route_bind_slot"]].append(task)
            entry = (task["route_bind_slot"] if task["destination_uav"] == task["source_uav"]
                     else task["tx_complete_slot"])
            eligible[tid] = entry + 1 if entry is not None else horizon
        else:
            eligible[tid] = horizon
        terminal[tid] = (task["done_slot"] if task["done_slot"] is not None else
                         task["expired_slot"] if task["expired_slot"] is not None else horizon - 1)
    route_bindings = {}
    for row in route_rows:
        if row["selected_top_level"] in {"Local", "Remote"}:
            if row["task_id"] in route_bindings:
                raise EnergyAuditError("Task has multiple route bindings")
            route_bindings[row["task_id"]] = row
    for task in tasks:
        if task["route_bind_slot"] is None:
            continue
        row = route_bindings.get(task["task_id"])
        if row is None or row["step"] != task["route_bind_slot"] or row["uav_id"] != task["source_uav"]:
            raise EnergyAuditError("Route and lifecycle binding differ")
        destination = row["selected_remote_id"] if row["selected_top_level"] == "Remote" else row["uav_id"]
        if destination != task["destination_uav"]:
            raise EnergyAuditError("Route and lifecycle destination differ")

    remaining = {tid: t["required_cycles"] for tid, t in task_by_id.items()}
    bits_remaining = {tid: t["input_bits"] for tid, t in task_by_id.items()}
    previous_energy = {uav: resource["initial_energy"] for uav, resource in resources.items()}
    public = {uav: None for uav in range(count)}
    remote_rows, slot_rows = [], []
    blocked = defaultdict(list)
    positive = defaultdict(list)
    backlogged = defaultdict(list)
    exposed = set()
    full_frequency_slots = 0
    checks = Counter()
    for slot in range(horizon):
        links = {r["uav_id"]: r for r in tx_by_slot[slot] if r["link_summary"]}
        if len(links) != count or sum(r["link_summary"] for r in tx_by_slot[slot]) != count:
            raise EnergyAuditError("TX summary must have exactly one row per UAV and slot")
        before_energy = {}
        for uav in range(count):
            row = cpu[slot, uav]
            before_energy[uav] = row["residual_energy_j"] + row["cpu_energy_j"] + links[uav]["tx_energy_j"]
            require_close(before_energy[uav], previous_energy[uav], "Energy balance", absolute=1e-8)
            checks["energy_balance"] += 1
            queue = [t for t in tasks if t["destination_uav"] == uav
                     and eligible[t["task_id"]] <= slot <= terminal[t["task_id"]]
                     and remaining[t["task_id"]] > 0]
            if len(queue) != row["cpu_queue_length"]:
                raise EnergyAuditError(f"CPU queue length at {slot}/{uav}")
            require_close(sum(remaining[t["task_id"]] for t in queue), row["cpu_queue_cycles"], "Queue cycles")
            checks["CPU_queue"] += 1
            heads = {}
            for task in sorted(queue, key=lambda t: (t["deadline_slot"], t["task_id"])):
                heads.setdefault(task["source_uav"], task["task_id"])
            resource = resources[uav]
            is_blocked, cheapest = cpu_energy_blocked([remaining[tid] for tid in heads.values()],
                energy=before_energy[uav], frequency=resource["frequency"], coefficient=resource["coefficient"],
                minimum_level=minimum_level, slot_duration=env.slot_duration_s, tolerance=tolerance)
            if queue:
                backlogged[uav].append(slot)
            if is_blocked:
                blocked[uav].append(slot)
                exposed.update(t["task_id"] for t in queue)
            slot_rows.append({"slot": slot, "uav_id": uav, "CPU_queue_length": len(queue),
                "energy_before_j": before_energy[uav], "minimum_head_reservation_j": cheapest,
                "all_heads_energy_blocked": is_blocked, "served_cycles": row["served_cycles"]})
            served = row["served_task_id"]
            if row["served_cycles"] > 0:
                if served not in heads.values() or row["selected_task_id"] != served or is_blocked:
                    raise EnergyAuditError("Positive service did not select an energy-feasible source-queue EDF head")
                expected = min(remaining[served], row["cpu_frequency"] * env.slot_duration_s)
                require_close(row["served_cycles"], expected, "CPU service amount")
                require_close(row["cpu_energy_j"], resource["coefficient"] * row["cpu_frequency"] ** 2
                              * row["served_cycles"], "CPU energy", absolute=1e-8)
                remaining[served] = max(0, remaining[served] - row["served_cycles"])
                positive[uav].append(slot)
                if math.isclose(row["cpu_frequency"], resource["frequency"], rel_tol=1e-12):
                    full_frequency_slots += 1
                checks["EDF_and_service_amount"] += 1
        for task in bindings[slot]:
            if task["destination_uav"] == task["source_uav"]:
                continue
            destination = task["destination_uav"]
            resource, message = resources[destination], public[destination]
            lower_bound = resource["coefficient"] * (minimum_level * resource["frequency"]) ** 2 * task["required_cycles"]
            full_cost = resource["coefficient"] * resource["frequency"] ** 2 * task["required_cycles"]
            published = message[1] if message is not None else None
            remote_rows.append({"task_id": task["task_id"], "source": task["source_uav"],
                "destination": destination, "bind_slot": slot, "public_message_valid": message is not None,
                "public_message_source_slot": message[0] if message is not None else None,
                "message_AoI_slots": slot - message[0] if message is not None else None,
                "published_energy_j": published, "actual_destination_energy_j": before_energy[destination],
                "minimum_frequency_completion_energy_bound_j": lower_bound,
                "published_energy_below_minimum_completion_bound": published is not None and published + tolerance < lower_bound,
                "published_full_frequency_margin_negative": published is not None and published + tolerance < full_cost,
                "actual_energy_below_minimum_completion_bound": before_energy[destination] + tolerance < lower_bound,
                "final_status": task["final_status"], "terminal_stage": task["terminal_stage"]})
        # Current-slot reception is first visible at the next actor slot.
        refreshed = set()
        for link in links.values():
            head = link["task_id"]
            attempted = (link["accepted"] and link["executed_power_w"] > 0 and link["executed_ru_indices"]
                         and head is not None and bits_remaining[head] > 0)
            if attempted:
                refreshed.add(link["destination_uav"])
            checks["TX_attempt_gate"] += 1
        for row in tx_by_slot[slot]:
            if row["task_id"] is not None:
                bits_remaining[row["task_id"]] = max(0, bits_remaining[row["task_id"]] - row["transmitted_bits"])
        for uav in refreshed:
            public[uav] = (slot, cpu[slot, uav]["residual_energy_j"])
        previous_energy = {uav: cpu[slot, uav]["residual_energy_j"] for uav in range(count)}
    for tid, task in task_by_id.items():
        require_close(remaining[tid], task["remaining_cycles"], "Final remaining cycles")
        if task["destination_uav"] is not None and task["source_uav"] != task["destination_uav"]:
            require_close(bits_remaining[tid], task["remaining_bits"], "Final remaining TX bits")
        checks["final_task_cycles"] += 1
    cpu_energy = math.fsum(r["cpu_energy_j"] for r in cpu_rows)
    tx_energy = math.fsum(r["tx_energy_j"] for r in tx_rows if r["link_summary"])
    require_close(cpu_energy, metrics["cpu_energy_j"], "Episode CPU energy", absolute=1e-8)
    require_close(tx_energy, metrics["tx_energy_j"], "Episode TX energy", absolute=1e-8)
    nodes, post_tasks = [], []
    for uav in range(count):
        start = persistent_blockage_start(blocked[uav], positive[uav], backlogged[uav])
        nodes.append({"uav_id": uav, "blocked_slots": blocked[uav], "positive_service_slots": positive[uav],
                      "persistent_blockage_start": start, "final_energy_j": previous_energy[uav]})
        if start is not None:
            post_tasks.extend({"task_id": t["task_id"], "destination": uav, "bind_slot": t["route_bind_slot"],
                "persistent_blockage_start": start, "final_status": t["final_status"], "terminal_stage": t["terminal_stage"]}
                for t in tasks if t["destination_uav"] == uav and t["route_bind_slot"] is not None
                and t["route_bind_slot"] >= start)
    counts = Counter(t["final_status"] for t in tasks)
    for outcome, metric in (("DONE", "completed"), ("EXPIRED", "expired"), ("TRUNCATED", "truncated")):
        if counts[outcome] != metrics[metric]:
            raise EnergyAuditError("Task outcomes differ from episode metrics")
    summary = {"generated": len(tasks), "completed": counts["DONE"], "expired": counts["EXPIRED"],
        "truncated": counts["TRUNCATED"], "completion_pct": counts["DONE"] / len(tasks) * 100,
        "remote_assignments": len(remote_rows), "local_assignments": sum(t["route_bind_slot"] is not None
            and t["destination_uav"] == t["source_uav"] for t in tasks),
        "missing_public_energy_remote_assignments": sum(not r["public_message_valid"] for r in remote_rows),
        "published_energy_infeasible_remote_assignments": sum(r["published_energy_below_minimum_completion_bound"] for r in remote_rows),
        "published_full_frequency_margin_negative_remote_assignments": sum(r["published_full_frequency_margin_negative"] for r in remote_rows),
        "actual_energy_infeasible_remote_assignments": sum(r["actual_energy_below_minimum_completion_bound"] for r in remote_rows),
        "CPU_energy_blocked_UAV_slots": sum(len(v) for v in blocked.values()),
        "CPU_tasks_exposed_to_blockage": len(exposed),
        "exposed_task_final_status": dict(Counter(task_by_id[tid]["final_status"] for tid in exposed)),
        "post_persistent_blockage_assignments": len(post_tasks),
        "post_persistent_blockage_final_status": dict(Counter(t["final_status"] for t in post_tasks)),
        "positive_CPU_service_UAV_slots": sum(len(v) for v in positive.values()),
        "full_frequency_positive_CPU_UAV_slots": full_frequency_slots,
        "CPU_energy_j": cpu_energy, "TX_energy_j": tx_energy, "total_energy_j": cpu_energy + tx_energy,
        "energy_per_DONE_j": (cpu_energy + tx_energy) / counts["DONE"] if counts["DONE"] else None,
        "reward": metrics["total_reward"]}
    summary["published_energy_infeasible_pct_of_remote"] = (100 * summary["published_energy_infeasible_remote_assignments"]
        / len(remote_rows) if remote_rows else None)
    return {"summary": summary, "checks": dict(checks), "remote_assignments": remote_rows,
            "CPU_slots": slot_rows, "node_statistics": nodes, "post_blockage_tasks": post_tasks}


def analyze_saved_run(config, run_record):
    """Load only saved evaluation artifacts and reconstruct each learned scene."""
    groups = []
    for path in [*run_record["service_artifacts"][1:], run_record["route_artifacts"][1],
                 run_record["evaluation_artifacts"][1]]:
        group = defaultdict(list)
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("method_id", "ca_gat_mappo") == "ca_gat_mappo":
                    group[row["evaluation_seed"]].append(row)
        groups.append(group)
    results = {}
    for seed in config.evaluation.evaluation_seeds:
        if len(groups[4][seed]) != 1:
            raise EnergyAuditError("Expected exactly one saved learned episode per scene")
        results[seed] = reconstruct_episode(config, groups[0][seed], groups[1][seed], groups[2][seed],
                                           groups[3][seed], groups[4][seed][0]["metrics"])
    return results, groups[0]
