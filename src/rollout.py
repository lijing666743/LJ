"""Unified, artifact-producing policy rollout over ``U2UMECEnvironment``."""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_text_group, preflight_rollout_artifacts
from .config import RunConfig
from .env.environment import U2UMECEnvironment
from .policies.base import Policy


ENVIRONMENT_STREAM_NAMES = (
    "reset_mobility",
    "task_arrival",
    "task_workload",
    "channel_fading",
    "csi_error",
    "interference_measurement",
)

DASHBOARD_COLUMNS = (
    "run_id",
    "method_id",
    "scenario_id",
    "seed",
    "environment_seed",
    "policy_seed",
    "policy_stream_id",
    "git_commit",
    "config_hash",
    "slot",
    "terminated",
    "truncated",
    "bootstrap_allowed",
    "reward",
    "completion_component",
    "expiration_penalty",
    "workload_penalty",
    "energy_penalty",
    "generated_task_count",
    "completed_task_count",
    "expired_task_count",
    "truncated_task_count",
    "completion_rate",
    "expiration_rate",
    "truncation_rate",
    "actual_attempt_count",
    "outage_count",
    "outage_rate",
    "tx_energy_j",
    "cpu_energy_j",
    "total_energy_j",
    "queue_backlog",
    "rejection_count",
    "downgrade_count",
    "canonicalization_count",
    "proposal_json",
    "executed_json",
    "rejection_or_downgrade_json",
)


class RolloutError(RuntimeError):
    """Raised when a real rollout or artifact audit is incomplete."""


@dataclass(frozen=True)
class RolloutOutcome:
    """In-memory and persisted outputs from one complete real episode."""

    summary: dict[str, Any]
    raw_records: tuple[dict[str, Any], ...]
    artifacts: tuple[str, ...]


class RolloutRunner:
    """Run any actor-safe policy through the single environment backend."""

    def __init__(self, config: RunConfig, policy: Policy) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        if policy.method_id != config.method_id:
            raise RolloutError(
                f"policy method_id {policy.method_id!r} does not match config {config.method_id!r}"
            )
        config.validate()
        self.config = config
        self.policy = policy

    def run(self, *, write_artifacts: bool = True) -> RolloutOutcome:
        """Execute one full finite-horizon episode and optionally persist it."""

        if write_artifacts:
            preflight_rollout_artifacts(self.config.artifact_paths())
        environment = U2UMECEnvironment(self.config)
        reset = environment.reset()
        observations = reset.observations
        raw_records: list[dict[str, Any]] = []

        while True:
            proposals = tuple(self.policy.act(observation) for observation in observations)
            step = environment.step(proposals)
            raw_records.append(self._raw_record(step.info))
            if step.terminated or step.truncated:
                if step.observations is not None or step.centralized_state is not None:
                    raise RolloutError("terminal rollout exposed a fake next decision state")
                break
            if step.observations is None:
                raise RolloutError("nonterminal rollout step omitted next actor observations")
            observations = step.observations

        conservation = environment.assert_invariants()
        if not conservation.is_conserved:
            raise RolloutError("rollout ended with failed bit/cycle conservation")
        if not raw_records:
            raise RolloutError("rollout produced no real slot records")
        if len(raw_records) != self.config.environment.episode_horizon:
            raise RolloutError("rollout did not execute the configured finite horizon")

        assert environment.metrics is not None
        summary = self._aggregate(
            raw_records,
            environment.metrics.snapshot(),
            conservation.to_dict(),
        )
        artifacts = self._write_artifacts(raw_records, summary) if write_artifacts else ()
        return RolloutOutcome(summary, tuple(raw_records), artifacts)

    def _metadata(self) -> dict[str, Any]:
        stream_ids = self.config.reproducibility.stream_ids
        return {
            "run_id": self.config.run_id,
            "method_id": self.config.method_id,
            "scenario_id": self.config.scenario_id,
            "seed": self.config.seed,
            "master_seed": self.config.seed,
            "environment_seed": self.config.seed,
            "environment_stream_ids": {
                name: int(stream_ids[name]) for name in ENVIRONMENT_STREAM_NAMES
            },
            "policy_seed": self.policy.policy_seed,
            "policy_stream_id": self.policy.policy_stream_id,
            "git_branch": self.config.git_branch,
            "git_commit": self.config.git_commit,
            "git_dirty": self.config.git_dirty,
            "config_hash": self.config.config_hash,
        }

    def _raw_record(self, info: dict[str, Any]) -> dict[str, Any]:
        cumulative = info["metrics"]
        backlog = next(
            record
            for record in reversed(cumulative["slot_start_backlog"])
            if int(record["slot"]) == int(info["slot"])
        )
        slot_metrics = cumulative["slot_records"][-1]
        compact_metrics = {
            key: cumulative[key]
            for key in (
                "generated_task_count",
                "completed_task_count",
                "expired_task_count",
                "truncated_task_count",
                "actual_attempt_count",
                "outage_count",
                "outage_ratio",
                "total_tx_energy_j",
                "total_cpu_energy_j",
                "total_active_energy_j",
                "completion_rate",
                "expiration_rate",
                "truncation_rate",
                "mean_completed_e2e_latency_s",
                "generated_bits",
                "generated_cycles",
                "actual_served_bits",
                "actual_served_cycles",
                "local_binding_cleared_bits",
            )
        }
        compact_metrics["completed_e2e_latency_valid_sample_count"] = len(
            cumulative["completed_e2e_latency_samples_s"]
        )
        compact_metrics["queue_backlog"] = int(backlog["task_count"])
        compact_metrics["slot_start_backlog"] = backlog
        compact_metrics["slot_record"] = slot_metrics
        compact_metrics["conservation"] = cumulative["conservation"]
        return {
            **self._metadata(),
            "record_type": "slot",
            "slot": info["slot"],
            "boundary_slot": info["boundary_slot"],
            "next_slot": info["next_slot"],
            "next_decision_slot": info["next_decision_slot"],
            "bootstrap_allowed": info["bootstrap_allowed"],
            "terminated": info["terminated"],
            "truncated": info["truncated"],
            "proposal_action": info["proposal"],
            "executed_action_summary": info["executed"],
            "rejection_or_downgrade_summary": {
                "rejection": info["rejection"],
                "downgrade": info["downgrade"],
                "canonicalization": info["canonicalization"],
            },
            "service": info["service"],
            "energy": info["energy"],
            "outage": info["outage"],
            "reward": info["reward"],
            "arrival": info["arrival"],
            "metrics": compact_metrics,
            "conservation": info["conservation"],
        }

    def _aggregate(
        self,
        raw_records: list[dict[str, Any]],
        metrics: dict[str, Any],
        conservation: dict[str, Any],
    ) -> dict[str, Any]:
        latency_samples = [float(item) for item in metrics["completed_e2e_latency_samples_s"]]
        backlog_values = [
            int(record["task_count"]) for record in metrics["slot_start_backlog"]
        ]
        reward_keys = (
            "normalized_completed",
            "normalized_expired",
            "normalized_workload",
            "normalized_energy",
            "completion_component",
            "expiration_penalty",
            "workload_penalty",
            "energy_penalty",
            "reward",
        )
        reward_totals = {
            key: math.fsum(float(record["reward"][key]) for record in raw_records)
            for key in reward_keys
        }

        rejection_reasons: Counter[str] = Counter()
        downgrade_reasons: Counter[str] = Counter()
        canonicalization_reasons: Counter[str] = Counter()
        downgraded_action_count = 0
        for record in raw_records:
            audit = record["rejection_or_downgrade_summary"]
            rejection_reasons.update(str(item["reason"]) for item in audit["rejection"])
            canonicalization_reasons.update(
                str(item["reason"]) for item in audit["canonicalization"]
            )
            for item in audit["downgrade"]:
                downgraded_action_count += 1
                if item["communication"] is not None:
                    downgrade_reasons.update([f"communication:{item['communication']}"])
                if item["cpu"] is not None:
                    downgrade_reasons.update([f"cpu:{item['cpu']}"])

        attempts = int(metrics["actual_attempt_count"])
        summary = {
            **self._metadata(),
            "record_type": "aggregate",
            "episode": {
                "episode_count": 1,
                "configured_horizon": self.config.environment.episode_horizon,
                "slots_executed": len(raw_records),
                "terminated": bool(raw_records[-1]["terminated"]),
                "truncated": bool(raw_records[-1]["truncated"]),
                "bootstrap_allowed_at_boundary": bool(raw_records[-1]["bootstrap_allowed"]),
                "fake_terminal_state_created": False,
            },
            "tasks": {
                "generated": int(metrics["generated_task_count"]),
                "completed": int(metrics["completed_task_count"]),
                "expired": int(metrics["expired_task_count"]),
                "truncated": int(metrics["truncated_task_count"]),
                "completion_rate": metrics["completion_rate"],
                "expiration_rate": metrics["expiration_rate"],
                "truncation_rate": metrics["truncation_rate"],
            },
            "e2e_latency_s": {
                "mean": statistics.fmean(latency_samples) if latency_samples else None,
                "median": statistics.median(latency_samples) if latency_samples else None,
                "std": statistics.pstdev(latency_samples) if latency_samples else None,
                "valid_sample_count": len(latency_samples),
            },
            "energy_j": {
                "tx": float(metrics["total_tx_energy_j"]),
                "cpu": float(metrics["total_cpu_energy_j"]),
                "total": float(metrics["total_active_energy_j"]),
            },
            "transmission": {
                "actual_attempt_count": attempts,
                "outage_count": int(metrics["outage_count"]),
                "outage_rate": metrics["outage_ratio"],
                "outage_valid_sample_count": attempts,
            },
            "backlog": {
                "mean": statistics.fmean(backlog_values) if backlog_values else None,
                "maximum": max(backlog_values) if backlog_values else None,
                "final": backlog_values[-1] if backlog_values else None,
                "valid_sample_count": len(backlog_values),
            },
            "reward": reward_totals,
            "executor": {
                "rejection_count": sum(rejection_reasons.values()),
                "rejection_by_reason": dict(sorted(rejection_reasons.items())),
                "downgraded_action_count": downgraded_action_count,
                "downgrade_event_count": sum(downgrade_reasons.values()),
                "downgrade_by_reason": dict(sorted(downgrade_reasons.items())),
                "canonicalization_count": sum(canonicalization_reasons.values()),
                "canonicalization_by_reason": dict(
                    sorted(canonicalization_reasons.items())
                ),
            },
            "conservation": conservation,
            "claim_boundary": (
                "single-seed diagnostic rollout; not evidence of robustness, "
                "superiority, convergence, or manuscript readiness"
            ),
        }
        json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False)
        return summary

    def _write_artifacts(
        self,
        raw_records: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> tuple[str, ...]:
        paths = self.config.artifact_paths()
        snapshot_path = Path(paths["config_snapshot"])
        raw_path = Path(paths["raw_metrics"])
        aggregate_path = Path(paths["aggregate_metrics"])
        csv_path = Path(paths["dashboard_csv"])

        snapshot_text = json.dumps(
            self.config.snapshot_dict(), ensure_ascii=False, indent=2, allow_nan=False
        ) + "\n"
        raw_text = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for record in raw_records
        )
        aggregate_text = json.dumps(
            summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        csv_text = self._dashboard_csv(raw_records)

        atomic_write_text_group(
            (
                (snapshot_path, snapshot_text),
                (raw_path, raw_text),
                (aggregate_path, aggregate_text),
                (csv_path, csv_text),
            ),
            group_name="rollout artifact group",
        )
        return tuple(str(path) for path in (snapshot_path, raw_path, aggregate_path, csv_path))

    def _dashboard_csv(self, raw_records: list[dict[str, Any]]) -> str:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=DASHBOARD_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for record in raw_records:
            metrics = record["metrics"]
            reward = record["reward"]
            audit = record["rejection_or_downgrade_summary"]
            slot_metrics = metrics["slot_record"]
            writer.writerow({
                "run_id": record["run_id"],
                "method_id": record["method_id"],
                "scenario_id": record["scenario_id"],
                "seed": record["seed"],
                "environment_seed": record["environment_seed"],
                "policy_seed": self._csv_value(record["policy_seed"]),
                "policy_stream_id": self._csv_value(record["policy_stream_id"]),
                "git_commit": record["git_commit"],
                "config_hash": record["config_hash"],
                "slot": record["slot"],
                "terminated": record["terminated"],
                "truncated": record["truncated"],
                "bootstrap_allowed": record["bootstrap_allowed"],
                "reward": reward["reward"],
                "completion_component": reward["completion_component"],
                "expiration_penalty": reward["expiration_penalty"],
                "workload_penalty": reward["workload_penalty"],
                "energy_penalty": reward["energy_penalty"],
                "generated_task_count": metrics["generated_task_count"],
                "completed_task_count": metrics["completed_task_count"],
                "expired_task_count": metrics["expired_task_count"],
                "truncated_task_count": metrics["truncated_task_count"],
                "completion_rate": self._csv_value(metrics["completion_rate"]),
                "expiration_rate": self._csv_value(metrics["expiration_rate"]),
                "truncation_rate": self._csv_value(metrics["truncation_rate"]),
                "actual_attempt_count": metrics["actual_attempt_count"],
                "outage_count": metrics["outage_count"],
                "outage_rate": self._csv_value(metrics["outage_ratio"]),
                "tx_energy_j": slot_metrics["tx_energy_j"],
                "cpu_energy_j": slot_metrics["cpu_energy_j"],
                "total_energy_j": (
                    float(slot_metrics["tx_energy_j"]) + float(slot_metrics["cpu_energy_j"])
                ),
                "queue_backlog": metrics["queue_backlog"],
                "rejection_count": len(audit["rejection"]),
                "downgrade_count": len(audit["downgrade"]),
                "canonicalization_count": len(audit["canonicalization"]),
                "proposal_json": self._compact_json(record["proposal_action"]),
                "executed_json": self._compact_json(record["executed_action_summary"]),
                "rejection_or_downgrade_json": self._compact_json(audit),
            })
        return stream.getvalue()

    @staticmethod
    def _compact_json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @staticmethod
    def _csv_value(value: Any) -> Any:
        return "NA" if value is None else value



__all__ = [
    "DASHBOARD_COLUMNS",
    "ENVIRONMENT_STREAM_NAMES",
    "RolloutError",
    "RolloutOutcome",
    "RolloutRunner",
]
