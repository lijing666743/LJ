"""Pure offline reductions of frozen C training artifacts; never runs a policy.

The three late windows use rollout *end* counters, (start, end].  Samples in
the rollout crossing a window boundary are attributed to its end, and four
PPO epochs are repeated optimization views, not four independent samples.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

WINDOWS = ((28000, 32000), (32000, 36000), (36000, 40000))
MISSING = (None, "", "NA", "None", "null")
SCALARS = (
    "approx_kl", "clip_fraction", "entropy", "actor_loss", "critic_loss",
    "actor_grad_norm_before_clip", "critic_grad_norm_before_clip",
    "route_head_total_grad_norm", "shared_trunk_grad_norm", "alpha",
    "critic_learning_rate", "route_entropy_coefficient", "route_entropy_schedule_progress",
)
BRANCHES = ("route", "cpu_queue", "cpu_frequency")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def number(value):
    """Keep absent values absent and reject non-finite historical observations."""
    if value in MISSING:
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean is not a numeric diagnostic")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite historical diagnostic")
    return result


def describe(values) -> dict:
    present = [number(v) for v in values]
    present = [v for v in present if v is not None]
    return {"count": len(present), "mean": statistics.mean(present) if present else None,
            "min": min(present) if present else None,
            "max": max(present) if present else None,
            "sample_std": statistics.stdev(present) if len(present) > 1 else None}


def weighted_mean(pairs):
    valid = [(number(v), number(n)) for v, n in pairs]
    valid = [(v, n) for v, n in valid if v is not None and n is not None and n > 0]
    return sum(v*n for v, n in valid) / sum(n for _, n in valid) if valid else None


def read_dashboard(path: Path) -> tuple[list[dict], dict]:
    """Stream CSV, verify every duplicate column, retain only epoch/episode rows."""
    selected, counts, versions = [], Counter(), defaultdict(set)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        locations = defaultdict(list)
        for index, name in enumerate(header):
            locations[name].append(index)
        repeated = {name: indices for name, indices in locations.items() if len(indices) > 1}
        for line, cells in enumerate(reader, 2):
            if len(cells) != len(header):
                raise ValueError(f"CSV width mismatch at {path}:{line}")
            for name, indices in repeated.items():
                if len({cells[index] for index in indices}) != 1:
                    raise ValueError(f"Conflicting duplicate column {name} at {path}:{line}")
            row = {name: cells[indices[0]] for name, indices in locations.items()}
            counts[row["record_type"]] += 1
            for field in ("diagnostics_schema_version", "route_telemetry_schema_version"):
                if row.get(field) not in MISSING:
                    versions[field].add(row[field])
            if row["record_type"] in ("ppo_epoch", "episode"):
                selected.append(row)
    return selected, {"record_counts": dict(counts), "duplicate_columns": list(repeated),
                      "all_duplicate_cells_identical": True,
                      "schema_versions": {k: sorted(v) for k, v in versions.items()}}


def read_compact_jsonl(path: Path, kind: str) -> tuple[list[dict], dict]:
    """Stream large per-sample JSONL and retain logged reductions, not samples."""
    rows, schemas = [], Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            schemas[raw["schema"]] += 1
            if kind == "integrated":
                row = {key: raw[key] for key in ("environment_steps", "update_index", "ppo_epochs")}
            else:
                row = {"environment_steps": raw["environment_step_end"],
                       "optimized": raw.get("optimized", True)}
                if kind == "route":
                    row.update(ppo_epochs=raw["ppo_epochs"], groups=raw["groups"])
                elif kind == "task":
                    row.update(losses=raw["task_value_loss_by_epoch"],
                               credit_components=raw["credit_components_by_epoch"])
                elif kind == "q":
                    row.update(losses=raw["q_loss_by_epoch"],
                               calibration=raw["calibration_by_epoch"],
                               actor_credit_source=raw["actor_credit_source"],
                               effective_actor_counterfactual_coefficient=raw[
                                   "effective_actor_counterfactual_coefficient"])
                else:
                    raise ValueError(f"Unknown JSONL kind {kind}")
            rows.append(row)
    return rows, {"schema_counts": dict(schemas), "record_count": len(rows),
                  "first_environment_steps": rows[0]["environment_steps"] if rows else None,
                  "last_environment_steps": rows[-1]["environment_steps"] if rows else None}


def validate_replay_overlap(earlier: list[dict], later: list[dict]) -> dict:
    """Only compare semantic epoch telemetry, excluding run identity and timers."""
    old = {row["environment_steps"]: row for row in earlier}
    new = {row["environment_steps"]: row for row in later}
    if len(old) != len(earlier) or len(new) != len(later):
        raise ValueError("Duplicate update end within a source")
    common = sorted(old.keys() & new.keys())
    for step in common:
        if (old[step]["update_index"] != new[step]["update_index"] or
                old[step]["ppo_epochs"] != new[step]["ppo_epochs"]):
            raise ValueError(f"Replay telemetry conflict at {step}")
    return {"overlap_update_count": len(common), "overlap_steps": common,
            "all_epoch_payloads_exactly_equal": True,
            "excluded_comparison_fields": ["run_id", "config_hash", "paired_samples", "timing"],
            "selection": "Use final continuation only for all late windows; never sum segments."}


def _window(rows, start, end):
    return [r for r in rows if start < int(r["environment_steps"]) <= end
            and r.get("optimized", True)]


def _by_update(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["update_index"])].append(row)
    return grouped


def _branch_summary(rows, branch) -> dict:
    prefix = f"{branch}_branch_ppo_"
    first = [r for r in rows if int(r["ppo_epoch_index"]) == 0]
    count = prefix + "active_count"
    counts = [number(r.get(count)) for r in first]
    present_counts = [n for n in counts if n is not None]

    def peak(batch, field):
        values = [number(r.get(prefix+field)) for r in batch]
        values = [v for v in values if v is not None]
        return max(values) if values else None

    return {
        "unique_rollout_active_samples": sum(present_counts) if present_counts else None,
        "advantage_sample_weighted_mean_epoch0": weighted_mean(
            (r.get(prefix+"advantage_mean"), r.get(count)) for r in first),
        "kl_sample_weighted_mean": weighted_mean(
            (r.get(prefix+"approx_kl_mean"), r.get(count)) for r in rows),
        "clip_sample_weighted_mean": weighted_mean(
            (r.get(prefix+"clip_fraction"), r.get(count)) for r in rows),
        "epoch_kl": describe(r.get(prefix+"approx_kl_mean") for r in rows),
        "epoch_clip_fraction": describe(r.get(prefix+"clip_fraction") for r in rows),
        "update_peak_kl": describe(peak(batch, "approx_kl_mean") for batch in _by_update(rows).values()),
        "update_peak_clip_fraction": describe(peak(batch, "clip_fraction")
                                               for batch in _by_update(rows).values()),
    }


def summarize_window(dashboard, integrated, route, task, q, start, end, clip_norm) -> dict:
    epochs = [r for r in _window(dashboard, start, end) if r["record_type"] == "ppo_epoch"]
    episodes = [r for r in _window(dashboard, start, end) if r["record_type"] == "episode"]
    groups = _by_update(epochs)
    for update_index, batch in groups.items():
        if sorted(int(r["ppo_epoch_index"]) for r in batch) != [0, 1, 2, 3]:
            raise ValueError(f"Missing or duplicate PPO epoch for update {update_index}")
        if any(int(r["environment_steps"]) != 256*(update_index+1) for r in batch):
            raise ValueError("Global update counter disagrees with environment steps")
    by_kind = {kind: _window(rows, start, end) for kind, rows in (
        ("integrated", integrated), ("route", route), ("task", task), ("q", q))}
    expected_steps = {int(r["environment_steps"]) for r in epochs}
    for kind, rows in by_kind.items():
        steps = [r["environment_steps"] for r in rows]
        if len(steps) != len(set(steps)) or set(steps) != expected_steps:
            raise ValueError(f"{kind} window coverage differs from PPO epoch history")
    first = [r for r in epochs if int(r["ppo_epoch_index"]) == 0]
    branch = {name: _branch_summary(epochs, name) for name in BRANCHES}
    cpu_rows = [(r, e["cpu_joint"]) for r in by_kind["integrated"] for e in r["ppo_epochs"]]
    route_rows = [e["route"] for r in by_kind["integrated"] for e in r["ppo_epochs"]]
    for row in by_kind["integrated"]:
        # compact_scheduler_summary is an independent route/CPU count source.
        matches = [r for r in first if int(r["environment_steps"]) == row["environment_steps"]]
        if len(row["ppo_epochs"]) != 4 or len(matches) != 1:
            raise ValueError("Integrated summary epoch count invalid")
        if row["ppo_epochs"][0]["cpu_joint"]["active_samples"] != int(
                matches[0]["cpu_queue_branch_ppo_active_count"]):
            raise ValueError("Integrated CPU joint count does not match logged CPU queue branch")
    credit = {}
    for category in ("local", "remote", "defer"):
        components = [r["credit_components"]["epoch0/"+category] for r in by_kind["task"]]
        credit[category] = {name: {
            "count": sum(c[name]["count"] for c in components),
            "sample_weighted_mean": weighted_mean((c[name]["mean"], c[name]["count"]) for c in components),
            "positive_count": sum(c[name]["positive_count"] for c in components),
            "negative_count": sum(c[name]["negative_count"] for c in components),
        } for name in ("shared_advantage", "task_advantage", "counterfactual_advantage", "final_route_advantage")}
    q_calibration = {}
    for epoch in ("0", "3"):
        q_calibration[epoch] = {}
        for category in ("local", "remote", "defer"):
            cells = [r["calibration"][epoch][category] for r in by_kind["q"]]
            mse = weighted_mean(((c["rmse"]**2 if c["rmse"] is not None else None),
                                 c["epoch_sample_count"]) for c in cells)
            q_calibration[epoch][category] = {
                "sample_count": sum(c["epoch_sample_count"] for c in cells),
                "mae": weighted_mean((c["mae"], c["epoch_sample_count"]) for c in cells),
                "rmse": math.sqrt(mse) if mse is not None else None,
            }
    epoch_detail = [{"update_index": int(r["update_index"]),
                     "epoch_index": int(r["ppo_epoch_index"]),
                     "environment_steps": int(r["environment_steps"]),
                     **{field: number(r.get(field)) for field in SCALARS},
                     **{f"{name}_{field}": number(r.get(f"{name}_branch_ppo_{field}"))
                        for name in BRANCHES for field in ("approx_kl_mean", "clip_fraction")}}
                    for r in epochs]
    return {
        "start_exclusive": start, "end_inclusive": end,
        "update_count": len(groups), "ppo_epoch_count": len(epochs),
        "first_update_end": min(expected_steps), "last_update_end": max(expected_steps),
        "episode_count": len(episodes),
        "episode_reward": describe(r["reward"] for r in episodes),
        "scalars": {field: describe(r.get(field) for r in epochs) for field in SCALARS},
        "gradient_above_clip": {field: {"epoch_count": sum(number(r[field]) > clip_norm for r in epochs),
                                        "denominator": len(epochs), "threshold": clip_norm}
                                for field in ("actor_grad_norm_before_clip", "critic_grad_norm_before_clip")},
        "branches": branch,
        "route_entropy_sample_weighted_mean": weighted_mean((r["entropy"], r["active_samples"]) for r in route_rows),
        "route_selection_unique_rollout_counts": {
            action: sum(int(r[f"route_selected_{action}_count"]) for r in first)
            for action in ("local", "remote", "defer")},
        "route_mean_actor_defer_probability": weighted_mean(
            (r["mean_defer_probability_all_route_active"], r["route_branch_active_count"]) for r in first),
        "cpu_joint_entropy_sample_weighted_mean": weighted_mean((c["entropy"], c["active_samples"]) for _, c in cpu_rows),
        "cpu_joint_idle_sample_weighted_fraction": weighted_mean(
            (r["ppo_epochs"][0]["cpu_joint"]["idle_fraction"],
             r["ppo_epochs"][0]["cpu_joint"]["active_samples"]) for r in by_kind["integrated"]),
        "cpu_joint_frequency_distribution_sample_weighted": {
            level: weighted_mean((r["ppo_epochs"][0]["cpu_joint"]["frequency_distribution"][level],
                                  r["ppo_epochs"][0]["cpu_joint"]["active_samples"])
                                 for r in by_kind["integrated"]) for level in ("0.0", "0.25", "0.5", "1.0")},
        "route_credit_components_unique_epoch0": credit,
        "route_task_value_half_mse_by_epoch": {str(e): describe(r["losses"][e] for r in by_kind["task"]) for e in range(4)},
        "route_q_half_mse_by_epoch": {str(e): describe(r["losses"][e] for r in by_kind["q"]) for e in range(4)},
        "route_q_calibration": q_calibration,
        "exploration_actual_logged_coefficient": describe(e["coefficient"] for r in by_kind["route"] for e in r["ppo_epochs"]),
        "epoch_detail": epoch_detail,
        "NA": {"cpu_branch_gradient_norm": None, "route_value_gradient_norm": None,
               "route_q_gradient_norm": None, "historical_optimizer_param_group_lrs": None,
               "post_final_epoch_policy_kl": None},
    }


def build_diagnostic(root: Path) -> dict:
    root = root.resolve()
    inputs = {}

    def use(path):
        path = Path(path)
        inputs[path.relative_to(root).as_posix()] = {"sha256": digest(path), "bytes": path.stat().st_size}
        return path

    def unique(pattern):
        paths = list(root.glob(pattern))
        if len(paths) != 1:
            raise ValueError(f"Expected one input for {pattern}, got {len(paths)}")
        return use(paths[0])

    result = {"schema": "seed44_limited_training_diagnostic.v1", "training_started": False,
              "environment_evaluation_started": False, "heldout_used": False,
              "claim_boundary": "Descriptive frozen training-log comparison; no causal or convergence conclusion.",
              "window_rule": "Rollout end counter in (start,end]; crossing rollout is assigned to its end.",
              "denominator_rule": "Unique activity/advantage counts use epoch0 once; KL/entropy are sample-weighted epoch views.",
              "seeds": {}}
    for seed in (42, 43, 44):
        folder = "40k_continuation" if seed == 42 else "40k"
        pattern = f"logs/C/seed{seed}/{folder}/*/"
        dashboard_path = unique(f"dashboard_logs/C/seed{seed}/{folder}/*_metrics.csv")
        dashboard, csv_audit = read_dashboard(dashboard_path)
        config = json.loads(unique(pattern+"config_snapshot.yaml").read_text("utf-8"))
        aggregate = json.loads(unique(pattern+"aggregate_metrics.json").read_text("utf-8"))
        compact, file_audit = {}, {}
        for kind, name in (("integrated", "integrated_scheduler_updates_v1.jsonl"),
                           ("route", "route_training_rollouts_v1.jsonl"),
                           ("task", "route_task_credit_updates_v1.jsonl"),
                           ("q", "route_option_q_updates_v1.jsonl")):
            compact[kind], file_audit[kind] = read_compact_jsonl(unique(pattern+name), kind)
        replay = None
        if seed == 42:
            old, old_audit = read_compact_jsonl(unique("logs/C/seed42/32k_continuation/*/integrated_scheduler_updates_v1.jsonl"), "integrated")
            replay = validate_replay_overlap(old, compact["integrated"])
            replay["earlier_log"] = old_audit
        # Trusted local checkpoint read only: no network, environment or optimizer construction.
        import torch
        checkpoint = torch.load(unique(pattern+"checkpoints/final.pt"), map_location="cpu", weights_only=False)
        final_lrs = {name: [{"group_index": i, "lr": group["lr"], "group_name": group.get("name")}
                           for i, group in enumerate(state["param_groups"])]
                     for name, state in checkpoint["optimizer_state"].items()}
        mappo = config["training"]["mappo"]
        if (aggregate["seed"] != seed or config["seed"] != seed
                or checkpoint["config_hash"] != config["_metadata"]["config_hash"]):
            raise ValueError("Training artifact identity mismatch")
        history = [r for r in dashboard if r["record_type"] == "ppo_epoch"]
        if len(history) != aggregate["training"]["ppo_epoch_diagnostic_count"]:
            raise ValueError("PPO epoch history count disagrees with aggregate")
        result["seeds"][str(seed)] = {
            "csv_audit": csv_audit, "jsonl_audit": file_audit, "replay_audit": replay,
            "aggregate_checks": aggregate["checks"], "training_accounting": aggregate["training"],
            "final_checkpoint_training_state": checkpoint["training_state"],
            "final_actual_optimizer_param_group_lrs": final_lrs,
            "configured_learning_rates": {key: mappo[key] for key in (
                "actor_learning_rate", "critic_learning_rate", "route_head_lr_scale", "learning_rate_schedule_enabled")},
            "learning_rate_log_semantics": "alpha/critic_learning_rate are config values in training_artifacts.py; actual optimizer LRs verified only at final checkpoint.",
            "credit_boundary": {key: mappo[key] for key in (
                "route_credit_mode", "route_value_mode", "route_actor_credit_source", "route_exploration_mode",
                "route_exploration_initial_coefficient", "route_exploration_anneal_steps")},
            "windows": [summarize_window(dashboard, **compact, start=start, end=end,
                                         clip_norm=mappo["gradient_clip_norm"]) for start, end in WINDOWS],
        }
        del checkpoint
    for rel in ("src/training_artifacts.py", "src/models/ca_gat_mappo_ppo.py",
                "src/models/ca_gat_mappo_update.py", "src/models/ca_gat_mappo_trainer.py",
                "src/models/integrated_scheduler.py", "src/models/route_task_credit.py",
                "src/models/route_option_q.py", "src/models/route_exploration.py"):
        use(root/rel)
    result["inputs"] = inputs
    result["limitations"] = [
        "No raw_metrics scan: all source CSV duplicate cells checked; selected numeric fields and auxiliary loss reductions reject non-finite values; aggregate checks are also reported. This is not a claim about every historical tensor.",
        "Epoch diagnostics are pre-optimizer-step views. Epoch3 is before the fourth step; final post-step KL is unlogged.",
        "CPU queue legacy telemetry name matches integrated CPU joint active count; CPU frequency is a conditional legacy subview and must not be added to CPU joint counts.",
        "Four PPO epochs reuse the same rollout. Activity, visitation and advantage denominators use epoch0; replay segments are not independent training seeds.",
        "Route Q is auxiliary prediction with actor_credit_source=task_event; its MAE/RMSE cannot establish a policy root cause or correctness of unselected actions.",
        "CPU/Route counts are active decision positions, not independent tasks; training episodes and development scenes differ.",
        "Gradient clipping frequency is descriptive; clipping above threshold is not proof of divergence.",
    ]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Offline audit never overwrites an existing output")
    result = build_diagnostic(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"Offline training diagnostic written: {args.output}; {len(result['inputs'])} frozen inputs")
    for seed, row in result["seeds"].items():
        print(f"seed{seed}: 624 logged epochs / 156 updates; "
              f"late windows {[w['update_count'] for w in row['windows']]} updates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
