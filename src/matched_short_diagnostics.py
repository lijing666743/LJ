"""Durable live diagnostics for matched from-scratch short training.

JSONL is the canonical source.  Each record is appended, flushed, and fsynced
independently so a later exception or interrupt does not erase prior evidence.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import RunConfig
from .models.ca_gat_mappo_route_telemetry import RouteOutcomeAssociation
from .models.ca_gat_mappo_update import RecurrentPPOUpdateOutput


LIVE_DIAGNOSTICS_SCHEMA_VERSION = 1
MATCHED_GROUPS = ("baseline", "a1", "a2")


class LiveTrainingDiagnosticsError(RuntimeError):
    """Raised when live diagnostic records violate their contract."""


def matched_group_from_config(config: RunConfig) -> str:
    coefficient = float(config.training.mappo.route_choice_stability_coef)
    for group, expected in (("baseline", 0.0), ("a1", 0.02), ("a2", 0.05)):
        if math.isclose(coefficient, expected, rel_tol=0.0, abs_tol=1.0e-12):
            return group
    return f"coef_{coefficient:.12g}"


def live_diagnostics_path(config: RunConfig) -> Path:
    return Path(config.output.logs_dir) / config.run_id / "live_training_diagnostics.jsonl"


_LIVE_FIELDS = (
    "run_id",
    "group",
    "record_type",
    "environment_step",
    "rollout_index",
    "update_index",
    "ppo_epoch",
    "episode_index",
    "timestamp",
    "route_active_count",
    "local_count",
    "remote_count",
    "defer_count",
    "idle_count",
    "local_share",
    "remote_share",
    "conditional_remote_mass",
    "remote_local_binary_entropy",
    "floor_violation_fraction",
    "stability_valid_count",
    "unscaled_stability_loss",
    "scaled_stability_loss",
    "remote_scorer_grad_norm",
    "local_route_row_grad_norm",
    "remote_local_grad_norm_ratio",
    "gradient_ratio_valid",
    "approx_kl",
    "approx_kl_mean",
    "approx_kl_max",
    "clip_fraction",
    "clip_fraction_mean",
    "clip_fraction_max",
    "policy_loss",
    "critic_loss",
    "route_entropy",
    "global_grad_norm_before_clip",
    "gradient_clip_triggered",
    "gradient_clip_fraction",
    "episode_return",
    "completed_tasks",
    "expired_tasks",
    "truncated_tasks",
    "generated_tasks",
    "actual_energy_joules",
    "completion_rate",
    "expiration_rate",
    "truncation_rate",
    "energy_j_per_env_step",
    "remote_completed",
    "remote_expired",
    "remote_truncated",
    "local_completed",
    "local_expired",
    "local_truncated",
)


def _base_record(
    config: RunConfig,
    group: str,
    *,
    record_type: str,
    environment_step: int,
) -> dict[str, Any]:
    record = {field: None for field in _LIVE_FIELDS}
    record.update(
        {
            "schema_version": LIVE_DIAGNOSTICS_SCHEMA_VERSION,
            "run_id": config.run_id,
            "group": group,
            "record_type": record_type,
            "environment_step": environment_step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": config.seed,
            "git_commit": config.git_commit,
            "config_hash": config.config_hash,
        }
    )
    return record


def _finite_or_none(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LiveTrainingDiagnosticsError(f"{name} must be numeric or null")
    result = float(value)
    if not math.isfinite(result):
        raise LiveTrainingDiagnosticsError(f"{name} must be finite or null")
    return result


def _mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return None if not present else math.fsum(present) / len(present)


def _maximum(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return None if not present else max(present)


def _median(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return None if not present else float(statistics.median(present))


def _weighted_mean(
    pairs: Iterable[tuple[float | None, int]],
) -> float | None:
    numerator = 0.0
    denominator = 0
    for value, count in pairs:
        if value is None or count <= 0:
            continue
        numerator += float(value) * count
        denominator += count
    return None if denominator == 0 else numerator / denominator


def _route_counts(telemetry: Any, *, uav_count: int) -> dict[str, Any]:
    if telemetry is None:
        return {
            "route_active_count": 0,
            "local_count": 0,
            "remote_count": 0,
            "defer_count": 0,
            "idle_count": 0,
            "local_share": None,
            "remote_share": None,
        }
    active = int(telemetry.route_branch_active_count)
    local = int(telemetry.route_selected_local_count)
    remote = int(telemetry.route_selected_remote_count)
    defer = int(telemetry.route_selected_defer_count)
    matrix = tuple(getattr(telemetry, "active_branch_matrix", ()) or ())
    idle = 0
    if matrix:
        for timestep in matrix:
            if len(timestep) != uav_count:
                raise LiveTrainingDiagnosticsError(
                    "route activity matrix agent count is inconsistent"
                )
            idle += sum(not bool(agent[0]) for agent in timestep)
    return {
        "route_active_count": active,
        "local_count": local,
        "remote_count": remote,
        "defer_count": defer,
        "idle_count": idle,
        "local_share": None if active == 0 else local / active,
        "remote_share": None if active == 0 else remote / active,
    }


def build_update_records(
    config: RunConfig,
    group: str,
    output: RecurrentPPOUpdateOutput,
    *,
    update_index: int,
    environment_step: int,
    uav_count: int,
) -> tuple[dict[str, Any], ...]:
    """Build one deduplicated rollout, four epoch, and one aggregate record."""

    epochs = tuple(output.epoch_diagnostics)
    if len(epochs) != 4:
        raise LiveTrainingDiagnosticsError("matched diagnostics require four PPO epochs")
    count_rows = [_route_counts(epoch.route_telemetry, uav_count=uav_count) for epoch in epochs]
    if any(row != count_rows[0] for row in count_rows[1:]):
        raise LiveTrainingDiagnosticsError(
            "route behavior counts changed across PPO epochs for one rollout"
        )
    rollout = _base_record(
        config, group, record_type="rollout", environment_step=environment_step
    )
    rollout.update(
        {
            "rollout_index": update_index,
            "update_index": update_index,
            **count_rows[0],
        }
    )

    epoch_records: list[dict[str, Any]] = []
    stability_rows = []
    for epoch in epochs:
        combined_grad_norm = math.hypot(
            epoch.actor_grad_norm_before_clip,
            epoch.critic_grad_norm_before_clip,
        )
        clipping_triggered = (
            epoch.actor_grad_norm_before_clip > epoch.clip_max_norm
            or epoch.critic_grad_norm_before_clip > epoch.clip_max_norm
        )
        record = _base_record(
            config, group, record_type="ppo_epoch", environment_step=environment_step
        )
        record.update(
            {
                "rollout_index": update_index,
                "update_index": update_index,
                "ppo_epoch": epoch.epoch_index,
                "approx_kl": epoch.approx_kl,
                "clip_fraction": epoch.clip_fraction,
                "policy_loss": epoch.actor_loss,
                "critic_loss": epoch.critic_loss,
                "route_entropy": (
                    None
                    if epoch.route_telemetry is None
                    else epoch.route_telemetry.route_entropy_mean
                ),
                "global_grad_norm_before_clip": combined_grad_norm,
                "gradient_clip_triggered": clipping_triggered,
            }
        )
        stability = epoch.route_choice_stability
        if stability is not None:
            gradient_valid = (
                stability.remote_scorer_gradient_norm is not None
                and stability.local_route_row_gradient_norm is not None
                and stability.local_route_row_gradient_norm > 0.0
                and stability.gradient_norm_ratio is not None
            )
            record.update(
                {
                    "conditional_remote_mass": stability.conditional_remote_mass,
                    "remote_local_binary_entropy": stability.remote_local_binary_entropy,
                    "floor_violation_fraction": stability.floor_violation_fraction,
                    "stability_valid_count": stability.valid_sample_count,
                    "unscaled_stability_loss": stability.unscaled_stability_loss,
                    "scaled_stability_loss": stability.scaled_stability_loss,
                    "remote_scorer_grad_norm": stability.remote_scorer_gradient_norm,
                    "local_route_row_grad_norm": stability.local_route_row_gradient_norm,
                    "remote_local_grad_norm_ratio": (
                        stability.gradient_norm_ratio if gradient_valid else None
                    ),
                    "gradient_ratio_valid": gradient_valid,
                }
            )
            stability_rows.append(stability)
        epoch_records.append(record)

    aggregate = _base_record(
        config, group, record_type="update", environment_step=environment_step
    )
    valid_counts = [item.valid_sample_count for item in stability_rows]
    aggregate.update(
        {
            "rollout_index": update_index,
            "update_index": update_index,
            "stability_valid_count": sum(valid_counts),
            "conditional_remote_mass": _weighted_mean(
                (item.conditional_remote_mass, item.valid_sample_count)
                for item in stability_rows
            ),
            "remote_local_binary_entropy": _weighted_mean(
                (item.remote_local_binary_entropy, item.valid_sample_count)
                for item in stability_rows
            ),
            "floor_violation_fraction": _weighted_mean(
                (item.floor_violation_fraction, item.valid_sample_count)
                for item in stability_rows
            ),
            "unscaled_stability_loss": _weighted_mean(
                (item.unscaled_stability_loss, item.valid_sample_count)
                for item in stability_rows
            ),
            "scaled_stability_loss": _weighted_mean(
                (item.scaled_stability_loss, item.valid_sample_count)
                for item in stability_rows
            ),
            "remote_scorer_grad_norm": _median(
                item.remote_scorer_gradient_norm for item in stability_rows
            ),
            "local_route_row_grad_norm": _median(
                item.local_route_row_gradient_norm for item in stability_rows
            ),
            "remote_local_grad_norm_ratio": _median(
                item.gradient_norm_ratio
                for item in stability_rows
                if item.local_route_row_gradient_norm is not None
                and item.local_route_row_gradient_norm > 0.0
            ),
            "gradient_ratio_valid": any(
                item.gradient_norm_ratio is not None
                and item.local_route_row_gradient_norm is not None
                and item.local_route_row_gradient_norm > 0.0
                for item in stability_rows
            ),
            "approx_kl_mean": _mean(epoch.approx_kl for epoch in epochs),
            "approx_kl_max": _maximum(epoch.approx_kl for epoch in epochs),
            "clip_fraction_mean": _mean(epoch.clip_fraction for epoch in epochs),
            "clip_fraction_max": _maximum(epoch.clip_fraction for epoch in epochs),
            "policy_loss": _mean(epoch.actor_loss for epoch in epochs),
            "critic_loss": _mean(epoch.critic_loss for epoch in epochs),
            "route_entropy": _mean(
                None if epoch.route_telemetry is None else epoch.route_telemetry.route_entropy_mean
                for epoch in epochs
            ),
            "global_grad_norm_before_clip": _mean(
                math.hypot(
                    epoch.actor_grad_norm_before_clip,
                    epoch.critic_grad_norm_before_clip,
                )
                for epoch in epochs
            ),
            "gradient_clip_fraction": sum(
                epoch.actor_grad_norm_before_clip > epoch.clip_max_norm
                or epoch.critic_grad_norm_before_clip > epoch.clip_max_norm
                for epoch in epochs
            )
            / len(epochs),
        }
    )
    return (rollout, *epoch_records, aggregate)


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else numerator / denominator


def build_episode_record(
    config: RunConfig,
    group: str,
    *,
    episode_index: int,
    environment_step: int,
    transition_count: int,
    episode_return: float,
    metrics: Mapping[str, Any],
    route_outcomes: Sequence[RouteOutcomeAssociation] = (),
) -> dict[str, Any]:
    """Build one completed-episode KPI record from existing environment facts."""

    generated = int(metrics["generated_task_count"])
    completed = int(metrics["completed_task_count"])
    expired = int(metrics["expired_task_count"])
    truncated = int(metrics["truncated_task_count"])
    energy = _finite_or_none(metrics["total_active_energy_j"], "total_active_energy_j")
    if energy is None:
        raise LiveTrainingDiagnosticsError("completed episode energy must be present")
    outcome_counts = {
        "remote_completed": 0,
        "remote_expired": 0,
        "remote_truncated": 0,
        "local_completed": 0,
        "local_expired": 0,
        "local_truncated": 0,
    }
    for outcome in route_outcomes:
        if outcome.episode_index != episode_index or outcome.category not in {"remote", "local"}:
            continue
        key = f"{outcome.category}_{outcome.outcome}"
        if key in outcome_counts:
            outcome_counts[key] += 1
    record = _base_record(
        config, group, record_type="episode", environment_step=environment_step
    )
    record.update(
        {
            "episode_index": episode_index,
            "episode_return": _finite_or_none(episode_return, "episode_return"),
            "completed_tasks": completed,
            "expired_tasks": expired,
            "truncated_tasks": truncated,
            "generated_tasks": generated,
            "actual_energy_joules": energy,
            "completion_rate": _rate(completed, generated),
            "expiration_rate": _rate(expired, generated),
            "truncation_rate": _rate(truncated, generated),
            "energy_j_per_env_step": energy / transition_count,
            **outcome_counts,
        }
    )
    return record


class LiveTrainingDiagnosticsWriter:
    """Exclusive-create, append-only, per-record durable JSONL writer."""

    def __init__(
        self,
        config: RunConfig,
        *,
        group: str | None = None,
        path: str | Path | None = None,
    ) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        self.config = config
        self.group = matched_group_from_config(config) if group is None else group
        self.path = live_diagnostics_path(config) if path is None else Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        os.close(descriptor)
        self._lock = threading.Lock()

    def append(self, record: Mapping[str, Any]) -> None:
        if record.get("run_id") != self.config.run_id or record.get("group") != self.group:
            raise LiveTrainingDiagnosticsError("live record identity mismatch")
        line = json.dumps(
            dict(record), ensure_ascii=False, sort_keys=True, allow_nan=False
        ) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())

    def write_update(
        self,
        output: RecurrentPPOUpdateOutput,
        *,
        update_index: int,
        environment_step: int,
        uav_count: int,
    ) -> None:
        for record in build_update_records(
            self.config,
            self.group,
            output,
            update_index=update_index,
            environment_step=environment_step,
            uav_count=uav_count,
        ):
            self.append(record)

    def write_episode(
        self,
        *,
        episode_index: int,
        environment_step: int,
        transition_count: int,
        episode_return: float,
        metrics: Mapping[str, Any],
        route_outcomes: Sequence[RouteOutcomeAssociation] = (),
    ) -> None:
        self.append(
            build_episode_record(
                self.config,
                self.group,
                episode_index=episode_index,
                environment_step=environment_step,
                transition_count=transition_count,
                episode_return=episode_return,
                metrics=metrics,
                route_outcomes=route_outcomes,
            )
        )


__all__ = [
    "LIVE_DIAGNOSTICS_SCHEMA_VERSION",
    "LiveTrainingDiagnosticsError",
    "LiveTrainingDiagnosticsWriter",
    "build_episode_record",
    "build_update_records",
    "live_diagnostics_path",
    "matched_group_from_config",
]
