"""Frozen Heuristic route/CPU replacements to localize a learned-policy deficit."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from ..artifacts import _filesystem_path, atomic_write_text_group
from ..config import load_run_config
from ..models.factorized_gat_qmix import METHOD_ID
from ..models.qmix_continuation import state_digest
from .actor_loader import checkpoint_sha256
from .qmix_cpu_deadline_diagnostic import frozen_episode, aggregate_scenes
from .qmix_runner import QMIXEvaluationRunner


def run(config_path, output_path, progress=print):
    config = load_run_config(config_path)
    output_path = Path(output_path)
    if _filesystem_path(output_path).exists():
        raise FileExistsError("refusing to overwrite intervention evidence")
    final = Path(config.artifact_paths()["final_checkpoint"])
    evaluation = replace(config, mode="evaluation", launch_profile=None,
                         training=replace(config.training, formal_rl_enabled=False))
    runner = QMIXEvaluationRunner(evaluation, final, device="cpu", checkpoint_purpose="development_checkpoint",
                                  expected_training_steps=8000)
    before = state_digest(runner.agent.state_dict())
    before_file = checkpoint_sha256(final)
    directory = final.parent.parent / "checkpoint_progress" / "step_008000"
    files = list(_filesystem_path(directory).glob("*/episode_metrics.jsonl"))
    if len(files) != 1:
        raise ValueError("requires exactly one saved 8k evaluation")
    saved = {row["evaluation_seed"]: row for row in map(json.loads, files[0].read_text(encoding="utf-8").splitlines())
             if row["method_id"] == METHOD_ID}
    records = []
    for groups in (("cpu",), ("route",), ("route", "cpu")):
        scenes = []
        for seed in config.evaluation.evaluation_seeds:
            episode, trace = frozen_episode(runner, seed, heuristic_groups=groups)
            if episode.external_trace_sha256 != saved[seed]["external_trace_sha256"]:
                raise ValueError("intervention changed external task/channel inputs")
            trace["action_trace_sha256"] = episode.action_trace_sha256
            trace["external_trace_sha256"] = episode.external_trace_sha256
            scenes.append(trace)
            progress(json.dumps({"stage": "frozen_heuristic_group_intervention", "groups": groups,
                                 "seed": seed, "completed": episode.metrics["completed"],
                                 "optimizer_updates": 0}), flush=True)
        total = {key: sum(scene["metrics"][key] for scene in scenes)
                 for key in ("generated", "completed", "expired", "truncated", "total_energy_j", "cpu_energy_j", "tx_energy_j")}
        total["energy_per_completed_task_j"] = total["total_energy_j"]/total["completed"] if total["completed"] else None
        records.append({"heuristic_groups": groups, "scenes": scenes, "totals": total, **aggregate_scenes(scenes)})
    if before != state_digest(runner.agent.state_dict()) or before_file != checkpoint_sha256(final):
        raise ValueError("intervention changed checkpoint parameters")
    result = {"schema": "qmix_frozen_heuristic_group_intervention.v1", "optimizer_updates": 0,
              "episodes": sum(len(record["scenes"]) for record in records),
              "checkpoint_parameters_unchanged": True, "external_traces_match": True,
              "baseline_completed": sum(row["metrics"]["completed"] for row in saved.values()),
              "claim_boundary": "Closed-loop diagnostic replacements from the existing actor-visible deterministic Heuristic. These are hybrid interventions, not trained QMIX scores or independent causal effects: changed observations can change the retained learned branches later.",
              "interventions": records}
    atomic_write_text_group([(output_path, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))],
                           group_name="QMIX frozen group intervention diagnostic")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.config, args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
