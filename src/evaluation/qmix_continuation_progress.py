"""Combine verified prior QMIX evaluations with new continuation milestones.

Original 4k/8k artifacts and checkpoint bytes are reused in place. Only new
milestones are simulated. This module deliberately leaves the original frozen
evaluator and its source fingerprints unchanged.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from ..artifacts import _path_entry_exists
from ..config import load_run_config
from ..models.qmix_continuation import load_continuation_checkpoint, source_config, file_sha256
from .qmix_checkpoint_progress import QMIXCheckpointProgressRunner, checkpoint_milestones, _read_completed
from .qmix_runner import QMIXEvaluationRunner


class QMIXContinuationProgressRunner(QMIXCheckpointProgressRunner):
    def _preflight(self):
        final_path = self.checkpoint_directory / "final.pt"
        if not final_path.is_file():
            raise FileNotFoundError("complete continued training before evaluation; final checkpoint is missing")
        data = load_continuation_checkpoint(final_path)
        if data["config_hash"] != self.config.config_hash or data["kind"] != "final":
            raise ValueError("continued final checkpoint differs from the requested training configuration")
        provenance = data.get("budget_extension")
        if provenance is None:
            raise ValueError("checkpoint has no verified QMIX budget-extension provenance")
        sources = [(source_config(entry), Path(entry["path"]), entry["sha256"])
                   for entry in provenance["ancestors"]]
        plans = []
        for step in checkpoint_milestones(self.config):
            inherited = next((item for item in sources if step <= item[0].training.qmix.max_training_environment_steps), None)
            config, endpoint, _ = inherited if inherited else (self.config, final_path, None)
            budget = config.training.qmix.max_training_environment_steps
            checkpoint = endpoint if step == budget else endpoint.parent / f"step_{step:06d}.pt"
            if inherited:
                directory = endpoint.parent.parent / "checkpoint_progress"
                # Config hashes retain the original relative/absolute spelling
                # of output roots. Preserve it when locating cached results.
                if not Path(config.output.logs_dir).is_absolute():
                    directory = directory.relative_to(Path.cwd())
            else:
                directory = self.output_directory
            plans.append((step, config, checkpoint, directory, inherited is not None))
        missing = [str(path) for _, _, path, _, _ in plans if not path.is_file()]
        if missing:
            raise FileNotFoundError("checkpoint continuation series is incomplete: " + ", ".join(missing))
        inventory, runners, existing = [], [], []
        for step, source, path, directory, inherited in plans:
            QMIXCheckpointProgressRunner(source)._validate_source(
                path, step, source.training.qmix.max_training_environment_steps)
            if not inherited:
                # New checkpoints must preserve the same ancestry as final.
                checkpoint_data = load_continuation_checkpoint(path)
                if checkpoint_data.get("budget_extension") != provenance:
                    raise ValueError("continued checkpoint ancestry differs across milestones")
            output = replace(source.output, logs_dir=str(directory / f"step_{step:06d}"))
            evaluation = replace(source, mode="evaluation", launch_profile=None,
                training=replace(source.training, formal_rl_enabled=False), output=output)
            runner = QMIXEvaluationRunner(evaluation, path, device=self.device,
                checkpoint_purpose="development_checkpoint", expected_training_steps=step)
            flags = [_path_entry_exists(p) for p in runner.artifact_paths().values()]
            if any(flags) and not all(flags):
                raise FileExistsError("incomplete checkpoint evaluation group; refusing reuse")
            if inherited and not all(flags):
                raise FileNotFoundError("prior milestone evaluation must already be complete; original outputs are immutable")
            if all(flags) and not inherited and not self.reuse_completed:
                raise FileExistsError("continued checkpoint evaluation already exists; use explicit --reuse-completed")
            saved = _read_completed(runner) if all(flags) else None
            inventory.append({"training_steps": step, "kind": runner.checkpoint_kind,
                "path": str(path.resolve()), "sha256": runner.checkpoint_hash,
                "evaluation_run_id": runner.run_id, "source_training_config_hash": source.config_hash,
                "inherited": inherited, "continuation_evaluator_sha256": file_sha256(__file__)})
            runners.append(runner)
            existing.append(saved)
        return inventory, runners, existing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--reuse-completed", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        result = QMIXContinuationProgressRunner(load_run_config(args.config), device=args.device,
            reuse_completed=args.reuse_completed).run(
                progress=lambda value: print(json.dumps(value, ensure_ascii=False), flush=True))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"QMIX continuation evaluation failed: {exc}", flush=True)
        return 1
    print(json.dumps({"status": "completed", "primary_report_step": result["manifest"]["primary_report_step"],
        "checkpoint_count": len(result["manifest"]["checkpoints"]),
        "episode_count": result["manifest"]["episode_count"], "network_updates": 0,
        "reused_checkpoints": result["reused_checkpoints"], "artifacts": result["artifacts"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
