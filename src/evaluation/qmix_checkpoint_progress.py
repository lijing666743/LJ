"""Evaluate a completed QMIX run at each saved milestone, without training.

All checkpoints are required before evaluation starts. Completed checkpoint
evaluations may be reused only after their source identities and results pass
validation. The preselected primary report point remains the budget end.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from ..artifacts import atomic_write_text_group, _filesystem_path, _path_entry_exists
from ..config import load_run_config
from ..models.factorized_gat_qmix import METHOD_ID, validate_qmix_config
from ..models.qmix_trainer import load_checkpoint
from .actor_loader import checkpoint_sha256
from .metrics import FORMAL_METRIC_FIELDS
from .qmix_runner import QMIXEvaluationRunner, SUITE, evaluation_fingerprints,method_suite


def checkpoint_milestones(config):
    q = config.training.qmix
    return (*range(q.checkpoint_interval_steps, q.max_training_environment_steps,
                   q.checkpoint_interval_steps), q.max_training_environment_steps)


def _csv_text(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _validate_records(config, records, aggregate):
    suite=method_suite(config)
    expected = [(method, int(seed)) for seed in config.evaluation.evaluation_seeds for method in suite]
    if [(row["method_id"], row["evaluation_seed"]) for row in records] != expected:
        raise ValueError("checkpoint evaluation has missing, duplicate or reordered method/scene records")
    for row in records:
        metrics = row["metrics"]
        if row["episode_horizon"] != config.environment.episode_horizon or set(metrics) != set(FORMAL_METRIC_FIELDS):
            raise ValueError("checkpoint evaluation horizon/metric schema differs")
        if any(value is not None and not math.isfinite(float(value)) for value in metrics.values()):
            raise ValueError("checkpoint evaluation contains non-finite metrics")
        if metrics["generated"] != sum(metrics[key] for key in ("completed", "expired", "truncated")):
            raise ValueError("checkpoint evaluation task conservation differs")
        if not math.isclose(metrics["total_energy_j"], metrics["cpu_energy_j"] + metrics["tx_energy_j"], abs_tol=1.e-9):
            raise ValueError("checkpoint evaluation energy identity differs")
        if any(not row["invariants"][key] for key in (
                "generated_equals_completed_plus_expired_plus_truncated",
                "total_energy_equals_tx_plus_cpu", "reward_equals_component_identity",
                "workload_bit_conservation", "workload_cycle_conservation")):
            raise ValueError("checkpoint evaluation invariant failed")
    for seed in config.evaluation.evaluation_seeds:
        if len({row["external_trace_sha256"] for row in records if row["evaluation_seed"] == seed}) != 1:
            raise ValueError("external environment realizations differ across methods")
    if tuple(aggregate["method_suite"]) != suite:
        raise ValueError("aggregate method suite differs")
    for method in suite:
        group = [row for row in records if row["method_id"] == method]
        pooled = aggregate["methods"][method]["pooled"]
        if any(pooled[key] != sum(row["metrics"][key] for row in group)
               for key in ("generated", "completed", "expired", "truncated")):
            raise ValueError("aggregate counts differ from episode records")


def _read_completed(runner):
    paths = runner.artifact_paths()
    manifest = json.loads(_filesystem_path(paths["manifest"]).read_text(encoding="utf-8"))
    required = {
        "evaluation_run_id": runner.run_id, "checkpoint_sha256": runner.checkpoint_hash,
        "config_hash": runner.config.config_hash, "training_steps": runner.training_steps,
        "source_training_config_hash": runner.source_config_hash,
        "source_training_budget_steps": runner.training_budget,
        "checkpoint_kind": runner.checkpoint_kind,
        "evaluation_purpose": "development_checkpoint", "network_updates": 0,
        "exploration": False, "device": str(runner.device), "dtype": "float32",
        "evaluation_source_sha256": evaluation_fingerprints(),
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("existing checkpoint evaluation source/identity differs; refusing reuse")
    if tuple(manifest["method_suite"]) != method_suite(runner.config) or tuple(manifest["evaluation_seeds"]) != runner.config.evaluation.evaluation_seeds:
        raise ValueError("existing checkpoint evaluation method/scene contract differs")
    records = [json.loads(line) for line in _filesystem_path(paths["episodes"]).read_text(encoding="utf-8").splitlines()]
    if any(row["evaluation_run_id"] != runner.run_id for row in records):
        raise ValueError("existing episode records belong to another evaluation")
    aggregate = json.loads(_filesystem_path(paths["aggregate"]).read_text(encoding="utf-8"))
    _validate_records(runner.config, records, aggregate)
    return records


class QMIXCheckpointProgressRunner:
    def __init__(self, config, *, checkpoint_directory=None, output_directory=None,
                 device="cpu", reuse_completed=False):
        validate_qmix_config(config)
        if config.mode != "rl":
            raise ValueError("checkpoint series requires the original QMIX training configuration")
        self.config = config
        self.device = str(torch.device(device))
        self.reuse_completed = reuse_completed
        self.checkpoint_directory = Path(checkpoint_directory or config.artifact_paths()["checkpoint_directory"])
        self.output_directory = Path(output_directory or self.checkpoint_directory.parent / "checkpoint_progress")
        for directory in (self.output_directory, Path(config.output.dashboard_logs_dir)):
            if any(part.lower() in {".agents", "knowledge"} for part in directory.resolve().parts):
                raise ValueError("checkpoint evaluation cannot write into a protected directory")

    def _preflight(self):
        milestones = checkpoint_milestones(self.config)
        files = [(step, self.checkpoint_directory / (
            "final.pt" if step == milestones[-1] else f"step_{step:06d}.pt")) for step in milestones]
        missing = [str(path) for _, path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError("complete training before checkpoint evaluation; missing: " + ", ".join(missing))
        inventory, runners, existing = [], [], []
        for step, path in files:
            # Validate source config without retaining all replay payloads.
            self._validate_source(path, step, milestones[-1])
            output = replace(self.config.output, logs_dir=str(self.output_directory / f"step_{step:06d}"))
            evaluation = replace(self.config, mode="evaluation", launch_profile=None,
                training=replace(self.config.training, formal_rl_enabled=False), output=output)
            runner = QMIXEvaluationRunner(evaluation, path, device=self.device,
                checkpoint_purpose="development_checkpoint", expected_training_steps=step)
            flags = [_path_entry_exists(target) for target in runner.artifact_paths().values()]
            if any(flags) and not all(flags):
                raise FileExistsError("incomplete checkpoint evaluation artifact group; inspect before reuse")
            if any(flags) and not self.reuse_completed:
                raise FileExistsError("checkpoint evaluation already exists; use explicit --reuse-completed")
            saved = _read_completed(runner) if all(flags) else None
            inventory.append({"training_steps": step, "kind": runner.checkpoint_kind,
                "path": str(path.resolve()), "sha256": runner.checkpoint_hash,
                "evaluation_run_id": runner.run_id})
            runners.append(runner)
            existing.append(saved)
        return inventory, runners, existing

    def _validate_source(self, path, step, budget):
        data = load_checkpoint(path)
        if data["config_hash"] != self.config.config_hash:
            raise ValueError("checkpoint series source config differs from the training configuration")
        expected_kind = "final" if step == budget else "periodic"
        if data["steps"] != step or data["kind"] != expected_kind:
            raise ValueError("checkpoint series kind/step differs from the milestone")
        horizon = self.config.environment.episode_horizon
        if data["next_episode"] * horizon != step:
            raise ValueError("checkpoint episode counter differs from its step")
        q = self.config.training.qmix
        chunks = data["next_episode"] * math.ceil(horizon / q.recurrent_sequence_length_slots)
        if data["update_count"] != max(0, chunks-q.sequence_batch_size+1):
            raise ValueError("checkpoint optimizer counter differs from collected chunks")

    def run(self, *, progress=None):
        inventory, runners, existing = self._preflight()
        identity = {"config": self.config.config_hash, "checkpoints": inventory,
                    "device": self.device, "output": str(self.output_directory.resolve()),
                    "evaluator": evaluation_fingerprints()}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        series_id = f"qmix-checkpoints__seed-{self.config.seed}__{digest}"
        final_paths = {
            "manifest": self.output_directory / "series_manifest.json",
            "summary": self.output_directory / "summary.json",
            "csv": Path(self.config.output.dashboard_logs_dir) / f"{series_id}_metrics.csv",
            "scene_csv": Path(self.config.output.dashboard_logs_dir) / f"{series_id}_per_scene.csv",
        }
        flags = [_path_entry_exists(path) for path in final_paths.values()]
        if any(flags) and (not all(flags) or not self.reuse_completed):
            raise FileExistsError("checkpoint series summary already exists or is incomplete")
        if all(flags) and any(records is None for records in existing):
            raise ValueError("series summary exists but a checkpoint evaluation is missing")
        rows, scene_rows, artifacts = [], [], []
        baseline, external = {}, {}
        baseline_matches = reused = 0
        for index, (source, runner, saved) in enumerate(zip(inventory, runners, existing)):
            if checkpoint_sha256(source["path"]) != source["sha256"]:
                raise ValueError("checkpoint changed after preflight")
            if saved is None:
                result = runner.run(write_artifacts=True, progress=progress)
                records = [item.artifact_record(runner.run_id) for item in result["episodes"]]
                _validate_records(runner.config, records, result["aggregate"])
            else:
                records = saved
                reused += 1
            artifacts.extend(str(path) for path in runner.artifact_paths().values())
            for record in records:
                seed, method = record["evaluation_seed"], record["method_id"]
                signature = (record["external_trace_sha256"], record["metrics"]["generated"])
                if seed in external and external[seed] != signature:
                    raise ValueError("workload/external process changed across checkpoints")
                external[seed] = signature
                if method != self.config.method_id:
                    key = (method, seed)
                    fixed = (record["metrics"], record["action_trace_sha256"], record["external_trace_sha256"])
                    if key in baseline:
                        if baseline[key] != fixed:
                            raise ValueError("fixed baseline changed across checkpoints")
                        baseline_matches += 1
                    baseline[key] = fixed
                scene_rows.append({"training_steps": source["training_steps"], "training_seed": self.config.seed,
                    "method_id": method, "evaluation_seed": seed, **record["metrics"],
                    "checkpoint_sha256": source["sha256"], "action_trace_sha256": record["action_trace_sha256"],
                    "external_trace_sha256": record["external_trace_sha256"]})
            for method in method_suite(self.config):
                metrics = [row["metrics"] for row in records if row["method_id"] == method]
                generated = sum(row["generated"] for row in metrics)
                completed = sum(row["completed"] for row in metrics)
                energy = sum(row["total_energy_j"] for row in metrics)
                rows.append({"training_steps": source["training_steps"], "training_seed": self.config.seed,
                    "method_id": method, "generated": generated, "completed": completed,
                    "completion_ratio": completed/generated if generated else None,
                    "expired": sum(row["expired"] for row in metrics),
                    "truncated": sum(row["truncated"] for row in metrics),
                    "total_energy_j": energy, "cpu_energy_j": sum(row["cpu_energy_j"] for row in metrics),
                    "tx_energy_j": sum(row["tx_energy_j"] for row in metrics),
                    "energy_per_completed_task_j": energy/completed if completed else None,
                    "mean_reward": sum(row["total_reward"] for row in metrics)/len(metrics),
                    "checkpoint_sha256": source["sha256"]})
            if checkpoint_sha256(source["path"]) != source["sha256"]:
                raise ValueError("checkpoint changed during evaluation")
            if progress is not None:
                progress({"checkpoint_finished": source["training_steps"], "index": index+1,
                          "of": len(inventory), "reused": saved is not None})
        manifest = {"schema": "qmix_checkpoint_progress.v1", "series_id": series_id,
            "training_seed": self.config.seed, "training_config_hash": self.config.config_hash,
            "checkpoints": inventory, "evaluation_seeds": list(self.config.evaluation.evaluation_seeds),
            "method_suite": list(method_suite(self.config)), "primary_report_step": inventory[-1]["training_steps"],
            "checkpoint_selection": "fixed_budget_end; intermediate points describe learning only",
            "network_updates": 0, "exploration": False, "device": self.device,
            "evaluation_source_sha256": evaluation_fingerprints(), "fixed_baseline_repeat_matches": baseline_matches,
            "episode_count": len(scene_rows), "claim_boundary": "Development evaluation; no held-out or superiority claim"}
        summary = {"schema": "qmix_checkpoint_progress_summary.v1", "series_id": series_id,
                   "primary_report_step": inventory[-1]["training_steps"], "rows": rows}
        payloads = [(final_paths["manifest"], json.dumps(manifest, ensure_ascii=False, indent=2)),
                    (final_paths["summary"], json.dumps(summary, ensure_ascii=False, indent=2)),
                    (final_paths["csv"], _csv_text(rows)), (final_paths["scene_csv"], _csv_text(scene_rows))]
        if all(flags):
            if any(_filesystem_path(path).read_bytes() != content.encode("utf-8") for path, content in payloads):
                raise ValueError("existing series summary differs from validated checkpoint results")
        else:
            atomic_write_text_group(payloads, group_name="QMIX checkpoint series summary")
        return {"manifest": manifest, "rows": rows, "reused_checkpoints": reused,
                "artifacts": tuple([*artifacts, *(str(path) for path in final_paths.values())])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="original QMIX training config")
    parser.add_argument("--checkpoint-directory")
    parser.add_argument("--output-directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--reuse-completed", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        runner = QMIXCheckpointProgressRunner(load_run_config(args.config),
            checkpoint_directory=args.checkpoint_directory, output_directory=args.output_directory,
            device=args.device, reuse_completed=args.reuse_completed)
        result = runner.run(progress=lambda record: print(json.dumps(record, ensure_ascii=False), flush=True))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"QMIX checkpoint evaluation failed: {exc}", flush=True)
        return 1
    print(json.dumps({"status": "completed", "primary_report_step": result["manifest"]["primary_report_step"],
        "checkpoint_count": len(result["manifest"]["checkpoints"]), "episode_count": result["manifest"]["episode_count"],
        "network_updates": 0, "artifacts": result["artifacts"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
