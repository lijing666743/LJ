"""Read-only comparison checks; this module never constructs or trains a model.

Drafts scale exploration/target intervals to the explicit 40k comparison
budget. A configuration pass is deliberately not a launch gate.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

from ..config import RunConfig, load_run_config
from ..registry import build_default_registry, unavailable_handler


METHOD_ID = "factorized_action_gat_qmix"
TRAIN_SEEDS = (42, 43, 44)
DEVELOPMENT_SEEDS = (1042, 1043, 1044, 1045, 1046)
MATCHED_STEPS = 40000
CHECKPOINT_INTERVAL = 4000


def build_qmix_draft(reference: RunConfig, seed: int) -> dict[str, Any]:
    """Create an explicitly disabled draft without changing the reference."""
    reference.validate()
    if type(seed) is not int or seed not in TRAIN_SEEDS:
        raise ValueError(f"seed must be one of {TRAIN_SEEDS}")
    draft = copy.deepcopy(reference.resolved_dict())
    draft.update(mode="rl", method_id=METHOD_ID, launch_profile=None, seed=seed)
    draft["training"]["formal_rl_enabled"] = False
    draft["training"]["qmix"].update(
        max_training_environment_steps=MATCHED_STEPS,
        max_training_episodes=MATCHED_STEPS // reference.environment.episode_horizon,
        checkpoint_interval_steps=CHECKPOINT_INTERVAL,
        epsilon_decay_steps=32000,
        target_update_interval_updates=200,
    )
    for field, directory in (
        ("logs_dir", "logs"),
        ("plots_dir", "plots"),
        ("dashboard_logs_dir", "dashboard_logs"),
    ):
        draft["output"][field] = (
            f"{directory}/strong_baselines/fa_gat_qmix/seed{seed}/40k"
        )
    return draft


def linear_epsilon(config: RunConfig, steps: int) -> float:
    """Report the configured schedule; no random stream is consumed."""
    qmix = config.training.qmix
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if type(qmix.epsilon_decay_steps) is not int or qmix.epsilon_decay_steps <= 0:
        raise ValueError("epsilon_decay_steps must be a positive integer")
    fraction = min(1.0, steps / qmix.epsilon_decay_steps)
    return float(qmix.epsilon_start + fraction * (qmix.epsilon_end - qmix.epsilon_start))


def inspect_comparison(
    reference: RunConfig, candidate: RunConfig, project_root: str | Path
) -> dict[str, Any]:
    """Check fairness and filesystem identity without executing a handler."""
    reference.validate()
    issues: list[str] = []
    warnings: list[str] = []
    try:
        candidate.validate()
    except ValueError as exc:
        issues.append(f"candidate config invalid: {exc}")
    source = reference.resolved_dict()
    other = candidate.resolved_dict()
    if reference.method_id != "ca_gat_mappo":
        issues.append("reference must identify CA-GAT-MAPPO/C")
    if (reference.training.mappo.route_decoder_mode != "unified_node_v1"
            or reference.training.mappo.route_feature_mode != "physical_relations_v1"):
        issues.append("reference must retain the current C route identity")
    if (candidate.mode, candidate.method_id) != ("rl", METHOD_ID):
        issues.append("candidate must identify the independent QMIX training path")
    if candidate.seed not in TRAIN_SEEDS:
        issues.append("candidate seed is outside the first comparison seed set")
    for field in ("scenario_id", "environment", "action", "reward", "reproducibility"):
        if source[field] != other[field]:
            issues.append(f"shared contract differs: {field}")
    if tuple(candidate.evaluation.evaluation_seeds) != DEVELOPMENT_SEEDS:
        issues.append("development evaluation seeds must remain 1042--1046")
    if candidate.evaluation.update_network:
        issues.append("evaluation must not update a network")
    if candidate.evaluation.inference_rule_qmix != "masked_greedy":
        issues.append("QMIX evaluation requires masked_greedy")
    qmix = candidate.training.qmix
    integer_fields = (
        "encoder_hidden_dimension", "gat_layer_count", "attention_head_count",
        "gru_hidden_dimension", "replay_capacity_sequences",
        "recurrent_sequence_length_slots", "sequence_batch_size",
        "epsilon_decay_steps", "target_update_interval_updates",
        "max_training_episodes", "max_training_environment_steps",
        "checkpoint_interval_steps",
    )
    valid_integers = True
    for name in integer_fields:
        value = getattr(qmix, name)
        if type(value) is not int or value <= 0:
            issues.append(f"training.qmix.{name} must be a positive integer")
            valid_integers = False
    for name in ("learning_rate", "gamma", "epsilon_start", "epsilon_end"):
        value = getattr(qmix, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            issues.append(f"training.qmix.{name} must be finite numeric")
        elif (value <= 0 if name == "learning_rate" else not 0 <= value <= 1):
            issues.append(f"training.qmix.{name} outside its allowed range")
    finite_epsilon = all(
        type(getattr(qmix, name)) in (int, float) and math.isfinite(getattr(qmix, name))
        for name in ("epsilon_start", "epsilon_end")
    )
    if finite_epsilon and qmix.epsilon_start < qmix.epsilon_end:
        issues.append("epsilon_start must be at least epsilon_end")
    if qmix.optimizer != "Adam":
        issues.append("QMIX optimizer differs from the Section 4 Adam contract")
    if qmix.max_training_environment_steps != MATCHED_STEPS:
        issues.append("QMIX primary budget must be 40000 environment transitions")
    if (type(qmix.max_training_episodes) is int
            and qmix.max_training_episodes * candidate.environment.episode_horizon != MATCHED_STEPS):
        issues.append("episode limit and horizon must cover exactly 40000 transitions")
    if qmix.checkpoint_interval_steps != CHECKPOINT_INTERVAL:
        issues.append("QMIX checkpoint interval must be 4000 transitions")
    for name in ("encoder_hidden_dimension", "gat_layer_count", "attention_head_count", "gru_hidden_dimension"):
        if getattr(qmix, name) != getattr(reference.training.mappo, name):
            issues.append(f"encoder capacity differs from C: {name}")
    if valid_integers:
        if qmix.encoder_hidden_dimension % qmix.attention_head_count:
            issues.append("attention heads must divide encoder dimension")
        if qmix.sequence_batch_size > qmix.replay_capacity_sequences:
            issues.append("replay capacity is smaller than one sequence batch")
    root = Path(project_root).resolve()
    outputs: dict[str, str] = {}
    for field, directory in (("logs_dir", "logs"), ("plots_dir", "plots"), ("dashboard_logs_dir", "dashboard_logs")):
        target = (root / getattr(candidate.output, field)).resolve()
        expected = root / directory / "strong_baselines" / "fa_gat_qmix" / f"seed{candidate.seed}" / "40k"
        outputs[field] = str(target)
        if target != expected:
            issues.append(f"{field} must use the independent baseline output path")
        if target.exists():
            issues.append(f"{field} already exists; inspect it before any launch")
    registry = build_default_registry()
    handlers = {
        mode: "unavailable" if registry.resolve(mode, METHOD_ID) is unavailable_handler else "registered_unverified"
        for mode in ("rl", "evaluation")
    }
    epsilon_at_40k = None
    if finite_epsilon and type(qmix.epsilon_decay_steps) is int and qmix.epsilon_decay_steps > 0:
        epsilon_at_40k = linear_epsilon(candidate, MATCHED_STEPS)
        if qmix.epsilon_decay_steps > MATCHED_STEPS:
            warnings.append("Inherited exploration schedule extends beyond 40k; review during bounded development calibration")
    warnings.append("Inherited MAPPO fields are schema/reference data; QMIX must not use PPO, GAE, actor credit, or TX shadow updates")
    return {
        "schema_version": "strong_baseline_preflight.v1",
        "configuration_status": "FAIL" if issues else "PASS",
        "method_id": candidate.method_id,
        "seed": candidate.seed,
        "config_hash": candidate.config_hash if not issues else None,
        "primary_environment_transitions": qmix.max_training_environment_steps,
        "epsilon_at_40k": epsilon_at_40k,
        "handler_status": handlers,
        "formal_training_enabled": candidate.training.formal_rl_enabled,
        "launch_ready": False,
        "launch_gate": "This read-only check never certifies model, replay, checkpoint, evaluation, learning, or Gate 0 acceptance",
        "issues": issues,
        "warnings": warnings,
        "output_paths": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    result = inspect_comparison(
        load_run_config(args.reference), load_run_config(args.candidate), args.project_root
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["configuration_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
