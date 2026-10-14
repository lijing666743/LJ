"""Explicit budget extension and fail-closed replay checks for periodic saves."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping
import hashlib

import numpy as np
import torch

from ..config import (
    CHECKPOINT_KIND_FINAL_COMPLETED,
    CHECKPOINT_KIND_PERIODIC_RESUME,
    RunConfig,
    canonical_config_identity_hash,
    canonicalize_config_identity,
)
from .ca_gat_mappo_checkpoint import CheckpointError, validate_checkpoint_payload


def budget_extension_source_config(config: RunConfig, payload: Mapping[str, Any]) -> RunConfig:
    """Rebuild the source identity allowing only larger caps and new output roots."""

    if payload["checkpoint_kind"] != CHECKPOINT_KIND_PERIODIC_RESUME:
        raise CheckpointError("budget extension requires PERIODIC_RESUME; final tail is unavailable")
    source = canonicalize_config_identity(payload["config_snapshot"])
    if canonical_config_identity_hash(source) != payload["config_hash"]:
        raise CheckpointError("source config snapshot content hash mismatch")
    source_mappo = source["training"]["mappo"]
    target_mappo = config.training.mappo
    for name in ("max_training_episodes", "max_training_environment_steps"):
        if getattr(target_mappo, name) <= source_mappo[name]:
            raise CheckpointError(f"budget extension must increase {name}")
    if (target_mappo.learning_rate_schedule_enabled
            or target_mappo.entropy_coefficient_schedule_enabled):
        raise CheckpointError("budget-dependent learning schedules cannot be extended")
    if (target_mappo.conditional_credit_telemetry_enabled
            and target_mappo.tx_upper_decoder_mode == "legacy"
            and target_mappo.cpu_scheduler_mode == "legacy_hierarchical"):
        raise CheckpointError("budget extension does not support legacy conditional-credit sidecars")
    output_fields = ("logs_dir", "plots_dir", "dashboard_logs_dir")
    for name in output_fields:
        if Path(getattr(config.output, name)).resolve() == Path(source["output"][name]).resolve():
            raise CheckpointError(f"budget extension requires a new output.{name}")
    source_config = replace(
        config,
        training=replace(config.training, mappo=replace(
            target_mappo,
            max_training_episodes=source_mappo["max_training_episodes"],
            max_training_environment_steps=source_mappo["max_training_environment_steps"],
        )),
        output=replace(config.output, **{name: source["output"][name] for name in output_fields}),
    )
    # These fields are init=False and dataclasses.replace intentionally omits them.
    for name in ("git_commit", "git_branch", "git_dirty"):
        object.__setattr__(source_config, name, getattr(config, name))
    if source_config.resolved_dict() != source:
        raise CheckpointError(
            "budget extension may change only episode/step caps and three output roots; "
            "scientific configuration differs"
        )
    source_config.validate()
    config.validate()
    validate_checkpoint_payload(payload, config=source_config)
    validate_checkpoint_payload(payload, config=config)
    return source_config


def checkpoint_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_continuation_metadata(payload: Mapping[str, Any]) -> None:
    """Reject missing or inconsistent replay status before restoring any model."""

    provenance = payload["config_snapshot"]["_metadata"].get("continuation")
    if provenance is None:
        return
    required = {
        "schema_version", "source_checkpoint", "source_sha256", "source_config_hash",
        "target_config_hash", "restored_environment_steps", "restored_pending_transitions",
        "target_environment_steps", "diagnostic_sidecars_start_at", "previous", "replay_reference",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != required or provenance["schema_version"] != 1:
        raise CheckpointError("invalid continuation provenance schema")
    for name in ("source_sha256", "source_config_hash", "target_config_hash"):
        digest = provenance[name]
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise CheckpointError(f"invalid continuation digest: {name}")
    if provenance["target_config_hash"] != payload["config_hash"]:
        raise CheckpointError("continuation target config hash mismatch")
    for name in ("restored_environment_steps", "restored_pending_transitions", "target_environment_steps", "diagnostic_sidecars_start_at"):
        if type(provenance[name]) is not int or provenance[name] < 0:
            raise CheckpointError(f"invalid continuation count: {name}")
    current = payload["training_state"]["collected_environment_transitions"]
    restored = provenance["restored_environment_steps"]
    mappo = payload["config_snapshot"]["training"]["mappo"]
    target = min(mappo["max_training_environment_steps"],
                 mappo["max_training_episodes"] * payload["config_snapshot"]["environment"]["episode_horizon"])
    if (not 0 < restored <= current <= target
            or provenance["target_environment_steps"] != target
            or provenance["restored_pending_transitions"] != restored % 256
            or provenance["diagnostic_sidecars_start_at"] != restored):
        raise CheckpointError("continuation transition accounting mismatch")
    if not isinstance(provenance["source_checkpoint"], str) or not provenance["source_checkpoint"]:
        raise CheckpointError("continuation source path is missing")
    previous = provenance["previous"]
    if previous is not None and (not isinstance(previous, Mapping)
                                or previous.get("target_config_hash") != provenance["source_config_hash"]):
        raise CheckpointError("continuation lineage hash mismatch")
    info = provenance["replay_reference"]
    if info is None:
        return
    if (not isinstance(info, Mapping) or set(info) != {"path", "sha256", "environment_steps", "status"}
            or info["status"] not in {"pending", "verified_exact"}
            or type(info["environment_steps"]) is not int
            or not isinstance(info["path"], str) or not info["path"]
            or not isinstance(info["sha256"], str) or len(info["sha256"]) != 64):
        raise CheckpointError("invalid replay reference provenance")
    step = info["environment_steps"]
    if not restored < step < target:
        raise CheckpointError("replay reference lies outside the continuation interval")
    if ((info["status"] == "pending" and current >= step)
            or (info["status"] == "verified_exact" and current < step)):
        raise CheckpointError("replay verification status contradicts checkpoint step")


def assert_state_equal(actual: Any, expected: Any, path: str) -> None:
    """Compare nested state exactly, including tensor dtype, shape and values."""

    if isinstance(expected, torch.Tensor):
        equal = (isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype
                 and actual.shape == expected.shape
                 and torch.equal(actual.detach().cpu(), expected.detach().cpu()))
    elif isinstance(expected, np.ndarray):
        equal = (isinstance(actual, np.ndarray) and actual.dtype == expected.dtype
                 and np.array_equal(actual, expected))
    elif isinstance(expected, Mapping):
        equal = isinstance(actual, Mapping) and actual.keys() == expected.keys()
        if equal:
            for key in expected:
                assert_state_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        equal = isinstance(actual, type(expected)) and len(actual) == len(expected)
        if equal:
            for index, item in enumerate(expected):
                assert_state_equal(actual[index], item, f"{path}[{index}]")
    else:
        equal = type(actual) is type(expected) and actual == expected
    if not equal:
        raise CheckpointError(f"replay verification mismatch: {path}; continuation stopped")


def verify_replay_boundary(current: Mapping[str, Any], reference: Mapping[str, Any]) -> None:
    """Verify the factual prefix while retaining the regenerated pending tail."""

    if reference["checkpoint_kind"] != CHECKPOINT_KIND_FINAL_COMPLETED:
        raise CheckpointError("replay reference must be FINAL_COMPLETED")
    for name in ("model_state", "optimizer_state", "policy_rng_state", "diagnostics_state"):
        assert_state_equal(current[name], reference[name], name)
    for name in (
        "policy_version", "started_episodes", "completed_episodes", "next_episode_index",
        "collected_environment_transitions", "optimized_transitions", "ppo_update_count",
    ):
        assert_state_equal(current["training_state"][name], reference["training_state"][name], name)
    if current["active_rollout_state"]["rollout_length"] != reference["training_state"]["unused_final_tail_transitions"]:
        raise CheckpointError("replay pending tail length differs from original unused tail")


__all__ = ["budget_extension_source_config", "checkpoint_sha256", "assert_state_equal",
           "validate_continuation_metadata", "verify_replay_boundary"]
