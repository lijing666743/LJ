"""Read-only matched E4 audit of existing artifacts; no policy/env execution.

Training observations and deterministic formal episodes are separate populations.
No counterfactual ranking or missing probability is reconstructed by inference.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from src.config import canonical_config_identity_hash, canonicalize_config_identity

RUNS = {
    "E3": "ca_gat_mappo__remote_stress_v2__seed-43__cfg-735fb97120ce__git-97b387271007",
    "E4": "ca_gat_mappo__remote_stress_v2__seed-43__cfg-a216fc6b1306__git-21624c6cd274",
}
EVALS = {
    "E3": "formal-eval__remote_stress_v2__chk-3832e046d1be__cfg-d6d90df7321f__eval-8418b5c559b1",
    "E4": "formal-eval__remote_stress_v2__chk-0eb5cd7d8fdf__cfg-d6d90df7321f__eval-41ef8dc58d02",
}
METRICS = ("completion_ratio", "completed", "expiration_ratio", "expired",
           "total_reward", "total_energy_j", "executor_rejection")
LEVELS = (0., .25, .5, 1.)


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def stage(step):
    if not 0 <= step < 32000:
        raise ValueError("Outside fixed training budget")
    return step // 4000


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [row for key in sorted(a.keys() | b.keys())
                for row in differences(a.get(key), b.get(key), f"{prefix}.{key}".strip("."))]
    return [] if a == b else [{"field": prefix, "E3": a, "E4": b}]


def frequency_counts(actions):
    selected = [a for a in actions if a["cpu_queue"] != "idle"]
    counts = Counter(float(a["cpu_frequency"]) for a in selected)
    return {"selected_queue_actions": len(selected),
            "counts": {str(f): counts[f] for f in LEVELS},
            "fractions": {str(f): fraction(counts[f], len(selected)) for f in LEVELS}}


def resources(config):
    env = config["environment"]
    values = []
    for i, profile in enumerate(env["profile_assignment"]):
        fr, _, er, kr = env["profile_ratios"][profile]
        fd, _, ed, kd = env["profile_perturbations"][i]
        values.append({"fmax": env["reference_cpu_frequency_hz"] * fr * (1 + fd),
                       "kappa": env["reference_cpu_coefficient"] * kr * (1 + kd),
                       "initial_energy": env["reference_initial_energy_j"] * er * (1 + ed)})
    return values


def cpu_state_bucket(slack, remaining, full_slot_capacity):
    """Explicit descriptive bins, not a decoder rule or a feasibility oracle."""
    if slack <= 5 and remaining > full_slot_capacity:
        return "tight_large"
    if slack > 15 and remaining <= full_slot_capacity:
        return "easy_small"
    return "other"


def summarize_cpu_actions(actions, tasks, config):
    hardware = resources(config)
    buckets = defaultdict(list)
    result = []
    for a in actions:
        if a["cpu_queue"] == "idle":
            continue
        cpu = a["executed_cpu"]
        key = (a["episode"], cpu["head_task_id"])
        task = tasks.get(key)
        if task is None or "snapshot" not in task:
            raise ValueError("Selected CPU head has no lifecycle snapshot")
        slack = task["snapshot"]["deadline_slot"] - a["slot"]
        remaining = cpu["head_remaining_cycles"]
        bucket = cpu_state_bucket(slack, remaining,
                                  hardware[a["agent"]]["fmax"] * config["environment"]["slot_duration_s"])
        row = {"step": a["action_key"][0], "episode": a["episode"],
               "task_id": cpu["head_task_id"], "remote": task.get("remote", False),
               "outcome": task.get("outcome", "censored"), "level": float(a["cpu_frequency"]),
               "slack": slack, "remaining_cycles": remaining, "bucket": bucket,
               "positive_service": a.get("served_cycles", 0) > 0,
               "served_cycles": a.get("served_cycles", 0), "energy_j": a.get("cpu_energy_j", 0)}
        result.append(row)
        buckets[bucket].append(row)
    summaries = {key: {"actions": len(rows), "full_fraction": mean(r["level"] == 1 for r in rows),
                       "low_fraction": mean(r["level"] == .25 for r in rows),
                       "medium_fraction": mean(r["level"] == .5 for r in rows),
                       "zero_fraction": mean(r["level"] == 0 for r in rows)}
                 for key, rows in buckets.items()}
    return result, summaries


def trace_analysis(path, config, register):
    actions, tasks, counts = {}, defaultdict(dict), Counter()
    updates, epochs, optimizer_steps = set(), set(), set()
    hasher = hashlib.sha256()
    hardware = resources(config)
    with path.open("rb") as stream:
        for line in stream:
            hasher.update(line)
            # Saved per-action PPO credit is intentionally outside this audit.
            if b'"event_type": "ppo_agent_epoch"' in line[:80]:
                counts["ppo_agent_epoch"] += 1
                continue
            r = json.loads(line)
            kind = r["event_type"]
            counts[kind] += 1
            if kind == "header":
                header = r
            elif kind == "agent_action":
                key = tuple(r["action_key"])
                if key in actions:
                    raise ValueError("Duplicate action")
                actions[key] = r
            elif kind == "task_route" and r["route"]["applied"]:
                route = r["route"]
                tasks[tuple(r["task_key"])].update(binding=r["global_index"],
                    remote=route["destination"] != route["uav_id"])
            elif kind == "task_service" and r["amount"] > 0:
                task = tasks[tuple(r["task_key"])]
                key = tuple(r["action_key"])
                task.setdefault("first_" + r["stage"], key[0])
                if r["stage"] == "tx" and r["remaining_bits_after_slot"] == 0.:
                    task.setdefault("tx_complete", key[0])
                if r["stage"] == "cpu":
                    action = actions[key]
                    task.setdefault("first_cpu_level", float(action["cpu_frequency"]))
                    task.setdefault("first_cpu_hz", r["executed_frequency_hz"])
                    task.setdefault("first_remaining_cycles", action["executed_cpu"]["head_remaining_cycles"])
                    energy = hardware[key[1]]["kappa"] * r["executed_frequency_hz"] ** 2 * r["amount"]
                    action["served_cycles"] = action.get("served_cycles", 0.) + r["amount"]
                    action["cpu_energy_j"] = action.get("cpu_energy_j", 0.) + energy
            elif kind in ("task_terminal", "task_censored"):
                task = tasks[tuple(r["task_key"])]
                outcome = r.get("outcome", "censored")
                task.update(outcome="censored" if outcome == "truncated" else outcome,
                            saved_outcome=outcome)
                if "task_snapshot" in r:
                    task["snapshot"] = r["task_snapshot"]
                if "global_index" in r:
                    task["terminal"] = r["global_index"]
            elif kind == "update_end":
                updates.add(r["update_index"])
            elif kind == "ppo_epoch":
                epochs.add((r["update_index"], r["epoch_index"]))
                optimizer_steps.add(r["optimizer_step"])
            elif kind == "run_end":
                end = r
    register[str(path.resolve())] = {"sha256": hasher.hexdigest(), "bytes": path.stat().st_size}
    assert set(actions) == {(s, a) for s in range(32000) for a in range(4)}
    assert updates == set(range(125)) and epochs == {(u, e) for u in range(125) for e in range(4)}
    assert optimizer_steps == set(range(1, 501)) and counts["run_end"] == 1
    assert counts["ppo_agent_epoch"] == 512000
    assert end["event_counts_before_end"] == {k: v for k, v in counts.items() if k != "run_end"}
    assert header["config_hash"] == canonical_config_identity_hash(config)
    for key, task in tasks.items():
        task.update(episode=key[0], task_id=key[1])
        if "first_cpu" in task:
            assert math.isclose(task["first_remaining_cycles"], task["snapshot"]["cpu_cycles"], abs_tol=1e-5)
            task["first_cpu_slack"] = task["snapshot"]["deadline_slot"] - task["first_cpu"] % 500
            task["tx_cpu_wait"] = (task["first_cpu"] - task["tx_complete"]
                                   if "tx_complete" in task else None)
    cpu_rows, conditioned = summarize_cpu_actions(actions.values(), tasks, config)
    return {"header": header, "run_end": end, "counts": dict(counts), "actions": actions,
            "tasks": list(tasks.values()), "cpu_rows": cpu_rows, "conditioned": conditioned}


def file_record(path, register):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    register[str(path.resolve())] = {"sha256": h.hexdigest(), "bytes": path.stat().st_size}
    return h.hexdigest()


def load_json(path, register):
    file_record(path, register)
    return json.loads(path.read_text(encoding="utf-8"))


def formal_route(path, episodes, register):
    counts, legal, probability, remote_selected = Counter(), 0, [], 0
    per_seed = defaultdict(Counter)
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for line in f:
            hasher.update(line)
            r = json.loads(line)
            counts[r["selected_top_level"].lower()] += 1
            per_seed[r["evaluation_seed"]][r["selected_top_level"].lower()] += 1
            if any(str(c).lower().startswith("remote") for c in r["legal_route_candidates"]):
                legal += 1
                probability.append(r["remote_probability"])
                remote_selected += r["selected_top_level"].lower() == "remote"
    register[str(path.resolve())] = {"sha256": hasher.hexdigest(), "bytes": path.stat().st_size}
    # All Defer from empty resets implies no bindings, no queues, no TX/CPU opportunities.
    return {"counts": dict(counts), "decisions": sum(counts.values()), "remote_legal": legal,
            "p_remote": mean(probability), "deterministic_remote": fraction(remote_selected, legal),
            "per_seed": {str(s): dict(c) for s, c in per_seed.items()}}


def training_dashboard(path, register):
    file_record(path, register)
    csv.field_size_limit(10_000_000)
    episodes, entropy = [], []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["record_type"] == "episode":
                episodes.append({"end": (int(r["episode_index"]) + 1) * 500,
                    "completed": float(r["completed_task_count"]),
                    "expired": float(r["expired_task_count"]), "return": float(r["reward"]),
                    "energy_penalty": float(r["energy_penalty"])})
            elif r["record_type"] == "ppo_epoch" and r["entropy"]:
                entropy.append({"end": (int(r["update_index"]) + 1) * 256,
                                "entropy": float(r["entropy"])})
    assert len(episodes) == 64 and len(entropy) == 500
    return episodes, entropy


def training_route(path, register):
    stages = [Counter() for _ in range(8)]
    h = hashlib.sha256()
    with path.open("rb") as f:
        for line in f:
            h.update(line)
            r = json.loads(line)
            if r["record_type"] != "route_diagnostic_sample":
                continue
            s = stages[stage(r["global_environment_step"])]
            choice = r["policy"]["actual_route"]
            s["remote" if isinstance(choice, int) else choice] += 1
            s["decisions"] += 1
    register[str(path.resolve())] = {"sha256": h.hexdigest(), "bytes": path.stat().st_size}
    return stages


def stage_summaries(trace, episodes, entropy, route, config):
    result = []
    hardware = resources(config)
    energy_reference = sum(r["initial_energy"] for r in hardware)
    for index in range(8):
        start, end = index * 4000, (index + 1) * 4000
        eps = [r for r in episodes if start < r["end"] <= end]
        ent = [r["entropy"] for r in entropy if start < r["end"] <= end]
        actions = [a for key, a in trace["actions"].items() if start <= key[0] < end]
        tx = [a for a in actions if a["tx_active"]]
        freq = frequency_counts(actions)
        result.append({"start": start, "end": end, "episodes": len(eps),
            **{k: mean(e[k] for e in eps) for k in ("completed", "expired", "return")},
            "entropy": mean(ent), "entropy_epoch_rows": len(ent),
            "route_remote": fraction(route[index]["remote"], route[index]["decisions"]),
            "route_counts": dict(route[index]), "tx_opportunities": len(tx),
            "tx_transmit": sum(a["tx_upper"] != "idle" for a in tx),
            "tx_transmit_fraction": mean(a["tx_upper"] != "idle" for a in tx),
            "cpu_frequency": freq,
            "cpu_energy_j": sum(a.get("cpu_energy_j", 0) for a in actions),
            "total_energy_j": sum(e["energy_penalty"] for e in eps) * energy_reference /
                              config["environment"]["reward_energy_weight"]})
    return result


def cpu_summary(trace):
    summaries = {}
    for scope in ("all", "remote"):
        subset = [t for t in trace["tasks"] if "first_cpu" in t and (scope == "all" or t.get("remote"))]
        for outcome in ("all", "done", "expired", "censored"):
            ts = [t for t in subset if outcome == "all" or t.get("outcome") == outcome]
            summaries[scope + "_" + outcome] = {"tasks": len(ts),
                "first_frequency": {str(f): sum(t["first_cpu_level"] == f for t in ts) for f in LEVELS},
                **{k + "_mean": mean(t.get(k) for t in ts if t.get(k) is not None)
                   for k in ("first_cpu_slack", "first_remaining_cycles", "tx_cpu_wait", "first_cpu_hz")}}
    for outcome in ("done", "expired", "censored"):
        rs = [r for r in trace["cpu_rows"] if r["outcome"] == outcome and r["positive_service"]]
        summaries["positive_service_actions_" + outcome] = {"actions": len(rs),
            "level_mean": mean(r["level"] for r in rs),
            "counts": {str(f): sum(r["level"] == f for r in rs) for f in LEVELS}}
    return summaries


def audit(root):
    import torch  # Deserialize saved payload only; never instantiate or run a model.
    register, reports, configs = {}, {}, {}
    for mode in RUNS:
        run, ev = root / "logs/gate_m" / RUNS[mode], root / "logs/gate_m/evaluations" / EVALS[mode]
        cfg = load_json(run / "config_snapshot.yaml", register)
        configs[mode] = cfg
        aggregate = load_json(run / "aggregate_metrics.json", register)
        manifest = load_json(ev / "evaluation_manifest.json", register)
        checkpoint = run / "checkpoints/final.pt"
        sha = file_record(checkpoint, register)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg_hash = canonical_config_identity_hash(cfg)
        assert sha == manifest["source_checkpoint_sha256"]
        assert cfg_hash == payload["config_hash"] == manifest["source_training_config_hash"]
        assert canonical_config_identity_hash(payload["config_snapshot"]) == cfg_hash
        assert payload["git_commit"] == manifest["source_training_git_commit"]
        assert cfg["seed"] == 43 and payload["checkpoint_kind"] == "FINAL_COMPLETED"
        state = payload["training_state"]
        assert state["training_complete"] and state["completed_episodes"] == 64
        assert state["collected_environment_transitions"] == state["optimized_transitions"] == 32000
        assert state["ppo_update_count"] == state["policy_version"] == 125
        assert state["active_rollout_length"] == state["unused_final_tail_transitions"] == 0
        assert payload["active_rollout_state"] == {"rollout_length": 0,
            "rollout_policy_version": None, "ordered_transitions": []}
        assert set(payload["model_state"]) == {"actor", "critic"} and payload["optimizer_state"]
        assert manifest["evaluation_seeds"] == list(range(1042, 1047))
        assert manifest["evaluation_run_id"] == EVALS[mode] and manifest["episode_horizon"] == 500
        file_record(ev / "episode_metrics.jsonl", register)
        rows = [json.loads(line) for line in (ev / "episode_metrics.jsonl").read_text().splitlines()]
        formal = [r for r in rows if r["method_id"] == "ca_gat_mappo"]
        assert sorted(r["evaluation_seed"] for r in formal) == list(range(1042, 1047))
        assert all(all(r["invariants"][k] for k in
                       ("generated_equals_completed_plus_expired_plus_truncated", "total_energy_equals_tx_plus_cpu"))
                   for r in formal)
        route_dir = root / "logs/gate_m/diagnostics" / ("hierarchical-route-eval__" + EVALS[mode])
        route = formal_route(route_dir / "route_decision_trace.jsonl", formal, register)
        route_manifest = load_json(route_dir / "summary.json", register)
        assert route_manifest["evaluation_run_id"] == EVALS[mode]
        assert route["decisions"] == route_manifest["route_decision_count"]
        for r in route_manifest["action_trace_sha256_by_episode"]:
            assert r["action_trace_sha256"] == next(e["action_trace_sha256"] for e in formal if e["evaluation_seed"] == r["evaluation_seed"])
        trace = trace_analysis(run / "conditional_credit_trace_v1.jsonl", cfg, register)
        assert trace["header"]["run_id"] == RUNS[mode] and trace["header"]["git_commit"] == payload["git_commit"]
        episodes, entropy = training_dashboard(root / "dashboard_logs/gate_m" / (RUNS[mode] + "_metrics.csv"), register)
        route_stages = training_route(run / "route_diagnostic_samples_v1.jsonl", register)
        assert sum(s["decisions"] for s in route_stages) == trace["counts"]["task_route"]
        stages = stage_summaries(trace, episodes, entropy, route_stages, cfg)
        assert math.isclose(sum(e["return"] for e in episodes), aggregate["reward"]["total"], abs_tol=1e-7)
        assert sum(e["completed"] for e in episodes) == aggregate["reward"]["completion_component_total"]
        total_energy = sum(s["total_energy_j"] for s in stages)
        assert math.isclose(sum(e["energy_penalty"] for e in episodes),
                            aggregate["reward"]["energy_penalty_total"], abs_tol=1e-9)
        assert sum(s["cpu_energy_j"] for s in stages) <= total_energy + 1e-6
        reports[mode] = {"identity": {"config_hash": cfg_hash, "checkpoint_sha256": sha,
            "git_commit": payload["git_commit"], "git_dirty": cfg.get("_metadata", {}).get("git_dirty"),
            "evaluator_git_dirty": manifest["evaluator_git_dirty"], "training_state": state,
            "runtime": payload["runtime_provenance"], "manifest": manifest,
            "actor_parameter_keys": list(payload["model_state"]["actor"])},
            "formal": [{"evaluation_seed": r["evaluation_seed"], **r["metrics"]} for r in formal],
            "formal_mean": {k: mean(r["metrics"][k] for r in formal) for k in METRICS},
            "route": route, "stages": stages, "cpu": cpu_summary(trace),
            "conditioned": trace["conditioned"], "cpu_frequency": frequency_counts(trace["actions"].values()),
            "cpu_energy_j": sum(s["cpu_energy_j"] for s in stages), "training_total_energy_j": total_energy,
            "trace_counts": trace["counts"], "tasks": trace["tasks"], "cpu_rows": trace["cpu_rows"]}
        print(f"{mode}: identity verified; saved trace read; formal completion={reports[mode]['formal_mean']['completion_ratio']:.6f}", flush=True)
        if mode == "E3":
            e3_payload, e3_formal = payload, formal
        else:
            assert manifest["evaluation_config_sha256"] == reports["E3"]["identity"]["manifest"]["evaluation_config_sha256"]
            for a, b in zip(e3_formal, formal):
                assert a["evaluation_seed"] == b["evaluation_seed"]
                assert a["external_trace_sha256"] == b["external_trace_sha256"]
            for key in ("deterministic_action_mode", "dtype", "evaluation_device", "method_suite", "episode_horizon"):
                assert manifest[key] == reports["E3"]["identity"]["manifest"][key]
    a, b = (canonicalize_config_identity(configs[m]) for m in ("E3", "E4"))
    a["training"]["mappo"].setdefault("cpu_frequency_decoder_mode", "legacy")
    diff = differences(a, b)
    assert diff == [{"field": "training.mappo.cpu_frequency_decoder_mode", "E3": "legacy", "E4": "candidate_aware_v1"}]
    # Revalidate saved historical E3 extra TX diagnostics against identical model weights and formal actions.
    old_run = Path("D:/My_project_li_gate_g_frozen/logs/ca_gat_mappo__remote_stress_v2__seed-43__cfg-cca983e51ebe__git-b105a225c7a8")
    old = torch.load(old_run / "checkpoints/final.pt", map_location="cpu", weights_only=False)
    file_record(old_run / "checkpoints/final.pt", register)
    for model in ("actor", "critic"):
        assert e3_payload["model_state"][model].keys() == old["model_state"][model].keys()
        assert all(torch.equal(t, old["model_state"][model][k]) for k, t in e3_payload["model_state"][model].items())
    historical = load_json(Path("D:/My_project_li_gate_g_frozen/logs/diagnostics/gate-g-summaries/e3-seed43.json"), register)
    assert historical["checkpoint_sha256"] == register[str((old_run / "checkpoints/final.pt").resolve())]["sha256"]
    old_eval = old_run.parent / "evaluations" / historical["evaluation_run_id"]
    old_rows = [json.loads(line) for line in (old_eval / "episode_metrics.jsonl").read_text().splitlines()]
    file_record(old_eval / "episode_metrics.jsonl", register)
    old_rows = [r for r in old_rows if r["method_id"] == "ca_gat_mappo"]
    for a, b in zip(e3_formal, old_rows):
        assert a["metrics"] == b["metrics"] and a["action_trace_sha256"] == b["action_trace_sha256"]
        assert a["external_trace_sha256"] == b["external_trace_sha256"] and a["evaluation_seed"] == b["evaluation_seed"]
    reports["E3"]["tx"] = historical["tx_totals"]
    reports["E3"]["formal_funnel"] = historical["remote_lifecycle_totals"]
    assert reports["E4"]["route"]["counts"] == {"defer": reports["E4"]["route"]["decisions"]}
    assert all(r["total_energy_j"] == r["completed"] == 0 for r in reports["E4"]["formal"])
    reports["E4"]["tx"] = {"legal_opportunities": 0, "p_transmit": None,
                            "deterministic_transmit": 0, "deterministic_idle": 0}
    reports["E4"]["formal_funnel"] = {k: 0 for k in reports["E3"]["formal_funnel"]}
    return reports, {"operation": "offline saved-artifact analysis only", "config_diff": diff,
                     "checks": "PASS", "historical_E3_weights_and_formal_actions_equal": True,
                     "sources": register}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_outputs(reports, provenance, root):
    output = root / "logs/e4-v1-seed43-audit"
    tables = root / "dashboard_logs/e4-v1-seed43-audit"
    plots = root / "plots/e4-v1-seed43-audit"
    for p in (output, tables, plots):
        p.mkdir(parents=True, exist_ok=True)
    for mode, r in reports.items():
        with gzip.open(output / f"{mode}_tasks.jsonl.gz", "wt", encoding="utf-8") as f:
            for task in r["tasks"]:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")
        write_csv(tables / f"{mode}_cpu_actions.csv", r["cpu_rows"])
        write_csv(tables / f"{mode}_formal.csv", r["formal"])
        write_csv(tables / f"{mode}_stages.csv", [{k: v for k, v in s.items() if not isinstance(v, dict)} for s in r["stages"]])
        (output / f"{mode}_summary.json").write_text(json.dumps({k: v for k, v in r.items() if k not in ("tasks", "cpu_rows")}, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    from src.evaluation.e4_v1_audit_figures import build_figures
    build_figures(reports, plots)
    from src.evaluation.e4_v1_audit_report import write_report
    write_report(reports, provenance, root)
    sources = [Path(__file__), Path(__file__).with_name("e4_v1_audit_figures.py"),
               Path(__file__).with_name("e4_v1_audit_report.py")]
    provenance["analysis_source_sha256"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (output / "input_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Offline audit written to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    if any((root / p).exists() for p in ("logs/e4-v1-seed43-audit", "dashboard_logs/e4-v1-seed43-audit",
                                       "plots/e4-v1-seed43-audit")):
        raise FileExistsError("Preserve existing audit outputs")
    reports, provenance = audit(root)
    write_outputs(reports, provenance, root)


if __name__ == "__main__":
    main()
