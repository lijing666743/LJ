"""Formal evaluation metric mapping over the environment's existing truth."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..env.environment import U2UMECEnvironment


FORMAL_EVALUATION_SCHEMA_VERSION = "formal_evaluation.v1"
FORMAL_METRIC_FIELDS = (
    "generated",
    "completed",
    "expired",
    "truncated",
    "completion_ratio",
    "expiration_ratio",
    "truncation_ratio",
    "completed_task_e2e_delay_s",
    "tx_energy_j",
    "cpu_energy_j",
    "total_energy_j",
    "energy_per_completed_task_j",
    "backlog_mean_tasks",
    "backlog_max_tasks",
    "backlog_final_tasks",
    "outage_numerator",
    "outage_denominator",
    "outage_rate",
    "executor_rejection",
    "executor_downgrade",
    "completion_component",
    "expiration_penalty",
    "workload_penalty",
    "energy_penalty",
    "total_reward",
)


class EvaluationMetricsError(RuntimeError):
    """Raised when environment-owned metric identities do not hold."""


@dataclass(frozen=True)
class EvaluationEpisodeResult:
    """One method/seed episode with audit traces retained in memory."""

    method_id: str
    evaluation_seed: int
    episode_horizon: int
    metrics: Mapping[str, int | float | None]
    completed_delay_samples_s: tuple[float, ...]
    action_trace: tuple[tuple[Mapping[str, Any], ...], ...]
    action_trace_sha256: str
    external_trace: tuple[Mapping[str, Any], ...]
    external_trace_sha256: str
    hidden_trace_sha256: tuple[str, ...]
    invariants: Mapping[str, bool | float]

    def artifact_record(self, evaluation_run_id: str) -> dict[str, Any]:
        """Return the stable episode JSONL schema without bulky slot payloads."""

        return {
            "evaluation_schema_version": FORMAL_EVALUATION_SCHEMA_VERSION,
            "evaluation_run_id": evaluation_run_id,
            "method_id": self.method_id,
            "evaluation_seed": self.evaluation_seed,
            "episode_horizon": self.episode_horizon,
            "action_trace_sha256": self.action_trace_sha256,
            "external_trace_sha256": self.external_trace_sha256,
            "hidden_trace_sha256": list(self.hidden_trace_sha256),
            "metrics": dict(self.metrics),
            "metric_valid_sample_counts": {
                "completed_task_e2e_delay_s": len(self.completed_delay_samples_s),
                "energy_per_completed_task_j": (
                    1
                    if self.metrics["energy_per_completed_task_j"] is not None
                    else 0
                ),
                "outage_rate": int(self.metrics["outage_denominator"]),
            },
            "invariants": dict(self.invariants),
        }


def build_episode_metrics(
    *,
    method_id: str,
    evaluation_seed: int,
    environment: U2UMECEnvironment,
    slot_infos: Sequence[Mapping[str, Any]],
    action_trace: tuple[tuple[Mapping[str, Any], ...], ...],
    action_trace_sha256: str,
    external_trace: tuple[Mapping[str, Any], ...],
    external_trace_sha256: str,
    hidden_trace_sha256: tuple[str, ...] = (),
) -> EvaluationEpisodeResult:
    """Map existing EpisodeMetrics and StepResult info to the formal schema."""

    if environment.metrics is None:
        raise EvaluationMetricsError("environment metrics are unavailable")
    snapshot = environment.metrics.snapshot()
    generated = int(snapshot["generated_task_count"])
    completed = int(snapshot["completed_task_count"])
    expired = int(snapshot["expired_task_count"])
    truncated = int(snapshot["truncated_task_count"])
    if generated != completed + expired + truncated:
        raise EvaluationMetricsError("generated task conservation failed")

    tx_energy = float(snapshot["total_tx_energy_j"])
    cpu_energy = float(snapshot["total_cpu_energy_j"])
    total_energy = float(snapshot["total_active_energy_j"])
    energy_error = total_energy - math.fsum((tx_energy, cpu_energy))
    if not _identity_close(energy_error, total_energy, tx_energy, cpu_energy):
        raise EvaluationMetricsError("total energy identity failed")

    delay_samples = tuple(
        float(value) for value in snapshot["completed_e2e_latency_samples_s"]
    )
    if len(delay_samples) != completed:
        raise EvaluationMetricsError(
            "completed-task latency sample count differs from completed tasks"
        )
    backlog_values = tuple(
        int(record["task_count"]) for record in snapshot["slot_start_backlog"]
    )
    if len(backlog_values) != environment.config.environment.episode_horizon:
        raise EvaluationMetricsError(
            "backlog snapshots do not cover the configured episode horizon"
        )

    reward_names = (
        "completion_component",
        "expiration_penalty",
        "workload_penalty",
        "energy_penalty",
        "reward",
    )
    reward_totals = {
        name: math.fsum(float(info["reward"][name]) for info in slot_infos)
        for name in reward_names
    }
    expected_reward = math.fsum(
        (
            reward_totals["completion_component"],
            -reward_totals["expiration_penalty"],
            -reward_totals["workload_penalty"],
            -reward_totals["energy_penalty"],
        )
    )
    reward_error = reward_totals["reward"] - expected_reward
    if not _identity_close(reward_error, reward_totals["reward"], expected_reward):
        raise EvaluationMetricsError("reward component identity failed")

    rejection_count = sum(len(info["rejection"]) for info in slot_infos)
    downgrade_count = sum(len(info["downgrade"]) for info in slot_infos)
    outage_denominator = int(snapshot["actual_attempt_count"])
    outage_numerator = int(snapshot["outage_count"])
    outage_rate = snapshot["outage_ratio"]
    if outage_denominator == 0:
        if outage_rate is not None:
            raise EvaluationMetricsError("zero-attempt outage rate must be NA")
    else:
        expected_outage = outage_numerator / outage_denominator
        if outage_rate != expected_outage:
            raise EvaluationMetricsError("outage numerator/denominator identity failed")

    metrics: dict[str, int | float | None] = {
        "generated": generated,
        "completed": completed,
        "expired": expired,
        "truncated": truncated,
        "completion_ratio": snapshot["completion_rate"],
        "expiration_ratio": snapshot["expiration_rate"],
        "truncation_ratio": snapshot["truncation_rate"],
        "completed_task_e2e_delay_s": (
            statistics.fmean(delay_samples) if delay_samples else None
        ),
        "tx_energy_j": tx_energy,
        "cpu_energy_j": cpu_energy,
        "total_energy_j": total_energy,
        "energy_per_completed_task_j": (
            total_energy / completed if completed > 0 else None
        ),
        "backlog_mean_tasks": statistics.fmean(backlog_values),
        "backlog_max_tasks": max(backlog_values),
        "backlog_final_tasks": backlog_values[-1],
        "outage_numerator": outage_numerator,
        "outage_denominator": outage_denominator,
        "outage_rate": outage_rate,
        "executor_rejection": rejection_count,
        "executor_downgrade": downgrade_count,
        "completion_component": reward_totals["completion_component"],
        "expiration_penalty": reward_totals["expiration_penalty"],
        "workload_penalty": reward_totals["workload_penalty"],
        "energy_penalty": reward_totals["energy_penalty"],
        "total_reward": reward_totals["reward"],
    }
    if tuple(metrics) != FORMAL_METRIC_FIELDS:
        raise EvaluationMetricsError("formal episode metric schema drifted")
    if completed == 0 and metrics["energy_per_completed_task_j"] is not None:
        raise EvaluationMetricsError(
            "zero-completion energy-per-task metric must be NA"
        )

    conservation = snapshot["conservation"]
    invariants: dict[str, bool | float] = {
        "generated_equals_completed_plus_expired_plus_truncated": True,
        "total_energy_equals_tx_plus_cpu": True,
        "reward_equals_component_identity": True,
        "workload_bit_conservation": bool(conservation["bit_conserved"]),
        "workload_cycle_conservation": bool(conservation["cycle_conserved"]),
        "energy_identity_error": energy_error,
        "reward_identity_error": reward_error,
    }
    if not invariants["workload_bit_conservation"] or not invariants[
        "workload_cycle_conservation"
    ]:
        raise EvaluationMetricsError("environment workload conservation failed")
    return EvaluationEpisodeResult(
        method_id=method_id,
        evaluation_seed=evaluation_seed,
        episode_horizon=environment.config.environment.episode_horizon,
        metrics=metrics,
        completed_delay_samples_s=delay_samples,
        action_trace=action_trace,
        action_trace_sha256=action_trace_sha256,
        external_trace=external_trace,
        external_trace_sha256=external_trace_sha256,
        hidden_trace_sha256=hidden_trace_sha256,
        invariants=invariants,
    )


def aggregate_episode_metrics(
    episodes: Sequence[EvaluationEpisodeResult],
    method_suite: Sequence[str],
) -> dict[str, Any]:
    """Aggregate every formal scalar over seeds with explicit NA counts."""

    result: dict[str, Any] = {
        "evaluation_schema_version": FORMAL_EVALUATION_SCHEMA_VERSION,
        "method_suite": list(method_suite),
        "methods": {},
    }
    for method_id in method_suite:
        method_episodes = tuple(
            episode for episode in episodes if episode.method_id == method_id
        )
        if not method_episodes:
            raise EvaluationMetricsError(
                f"no evaluation episodes were recorded for {method_id}"
            )
        metric_statistics = {
            field: _statistics(
                [episode.metrics[field] for episode in method_episodes]
            )
            for field in FORMAL_METRIC_FIELDS
        }
        all_delays = [
            delay
            for episode in method_episodes
            for delay in episode.completed_delay_samples_s
        ]
        metric_statistics["completed_task_e2e_delay_s"] = _statistics(all_delays)

        total_generated = sum(int(item.metrics["generated"]) for item in method_episodes)
        total_completed = sum(int(item.metrics["completed"]) for item in method_episodes)
        total_expired = sum(int(item.metrics["expired"]) for item in method_episodes)
        total_truncated = sum(int(item.metrics["truncated"]) for item in method_episodes)
        total_outages = sum(
            int(item.metrics["outage_numerator"]) for item in method_episodes
        )
        total_attempts = sum(
            int(item.metrics["outage_denominator"]) for item in method_episodes
        )
        result["methods"][method_id] = {
            "episode_count": len(method_episodes),
            "evaluation_seeds": [item.evaluation_seed for item in method_episodes],
            "metrics": metric_statistics,
            "pooled": {
                "generated": total_generated,
                "completed": total_completed,
                "expired": total_expired,
                "truncated": total_truncated,
                "completion_ratio": (
                    total_completed / total_generated if total_generated else None
                ),
                "expiration_ratio": (
                    total_expired / total_generated if total_generated else None
                ),
                "truncation_ratio": (
                    total_truncated / total_generated if total_generated else None
                ),
                "outage_numerator": total_outages,
                "outage_denominator": total_attempts,
                "outage_rate": (
                    total_outages / total_attempts if total_attempts else None
                ),
            },
        }
    return result


def _statistics(values: Sequence[int | float | None]) -> dict[str, int | float | None]:
    valid = [float(value) for value in values if value is not None]
    return {
        "mean": statistics.fmean(valid) if valid else None,
        "std": statistics.pstdev(valid) if valid else None,
        "valid_sample_count": len(valid),
    }


def _identity_close(error: float, *values: float) -> bool:
    scale = max(1.0, *(abs(float(value)) for value in values))
    return math.isfinite(error) and abs(error) <= 1.0e-12 * scale


__all__ = [
    "FORMAL_EVALUATION_SCHEMA_VERSION",
    "FORMAL_METRIC_FIELDS",
    "EvaluationEpisodeResult",
    "EvaluationMetricsError",
    "aggregate_episode_metrics",
    "build_episode_metrics",
]
