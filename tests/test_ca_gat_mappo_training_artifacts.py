"""Smoke Training artifact tests for real CA-GAT-MAPPO diagnostics."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.config import load_run_config
from src.models.ca_gat_mappo_trainer import (
    CAGATMAPPOLossDiagnostics,
    CAGATMAPPOEpisodeDiagnostics,
    CAGATMAPPORewardDiagnostics,
    CAGATMAPPOTrainingResult,
    CAGATMAPPOUpdateDiagnostics,
    ScalarTrainingDiagnostics,
)
from src.models.ca_gat_mappo_update import (
    AgentCreditPPOEpochTelemetry,
    RecurrentPPOEpochDiagnostics,
    RecurrentPPOUpdateOutput,
)
from src.training_artifacts import write_cagat_mappo_training_artifacts


def scalar(count: int, total: float) -> ScalarTrainingDiagnostics:
    return ScalarTrainingDiagnostics(
        count=count,
        total=total,
        mean=0.0 if count == 0 else total / count,
    )


def reward_diagnostics(count: int, multiplier: float = 1.0) -> CAGATMAPPORewardDiagnostics:
    completion = 8.0 * multiplier
    expiration = 2.0 * multiplier
    workload = 1.0 * multiplier
    energy = 1.0 * multiplier
    return CAGATMAPPORewardDiagnostics(
        reward=scalar(count, completion - expiration - workload - energy),
        completed_task_count=scalar(count, 8.0 * multiplier),
        expired_task_count=scalar(count, 2.0 * multiplier),
        urgent_workload_s=scalar(count, 3.0 * multiplier),
        actual_energy_j=scalar(count, 4.0 * multiplier),
        normalized_completed=scalar(count, 8.0 * multiplier),
        normalized_expired=scalar(count, 2.0 * multiplier),
        normalized_workload=scalar(count, 1.0 * multiplier),
        normalized_energy=scalar(count, 1.0 * multiplier),
        completion_component=scalar(count, completion),
        expiration_penalty=scalar(count, expiration),
        workload_penalty=scalar(count, workload),
        energy_penalty=scalar(count, energy),
    )


def training_result() -> CAGATMAPPOTrainingResult:
    episodes = tuple(
        CAGATMAPPOEpisodeDiagnostics(
            episode_index=index,
            environment_seed=1000 + index,
            transition_count=32,
            completed_boundary=True,
            reward=reward_diagnostics(32),
        )
        for index in range(8)
    )
    epoch_rows = tuple(
        RecurrentPPOEpochDiagnostics(
            epoch_index=index,
            actor_loss=-0.10 - 0.01 * index,
            critic_loss=1.00 - 0.10 * index,
            entropy_mean=2.00 - 0.05 * index,
            total_loss=0.38 - 0.06 * index,
            ratio_mean=1.00 + 0.01 * index,
            actor_grad_norm_before_clip=0.70 + 0.01 * index,
            critic_grad_norm_before_clip=0.80 + 0.01 * index,
            clip_max_norm=0.5,
        )
        for index in range(4)
    )
    update = CAGATMAPPOUpdateDiagnostics(
        update_index=0,
        rollout_policy_version=0,
        policy_version_after_update=1,
        output=RecurrentPPOUpdateOutput(
            epoch_diagnostics=epoch_rows,
            chunk_count=8,
            chunk_length=32,
            valid_transition_count=256,
            old_policy_snapshot_preserved=True,
        ),
    )

    def aggregate(name: str) -> ScalarTrainingDiagnostics:
        values = [getattr(epoch, name) for epoch in epoch_rows]
        return scalar(len(values), sum(values))

    return CAGATMAPPOTrainingResult(
        total_environment_transitions=256,
        optimized_transitions=256,
        unused_final_tail_transitions=0,
        started_episode_count=8,
        completed_episode_count=8,
        ppo_update_count=1,
        initial_policy_version=0,
        final_policy_version=1,
        episode_seeds=tuple(1000 + index for index in range(8)),
        reward=reward_diagnostics(256, multiplier=8.0),
        ppo=CAGATMAPPOLossDiagnostics(
            actor_loss=aggregate("actor_loss"),
            critic_loss=aggregate("critic_loss"),
            entropy=aggregate("entropy_mean"),
            total_loss=aggregate("total_loss"),
            ratio=aggregate("ratio_mean"),
            actor_grad_norm_before_clip=aggregate("actor_grad_norm_before_clip"),
            critic_grad_norm_before_clip=aggregate("critic_grad_norm_before_clip"),
        ),
        episodes=episodes,
        updates=(update,),
    )


class CAGATMAPPOTrainingArtifactTests(unittest.TestCase):
    def test_smoke_result_writes_auditable_csv_summary_and_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_run_config(
                cli_overrides={
                    "mode": "rl",
                    "method_id": "ca_gat_mappo",
                    "training.formal_rl_enabled": True,
                    "training.mappo.training_device": "cpu",
                    "environment.episode_horizon": 32,
                    "training.mappo.max_training_episodes": 8,
                    "training.mappo.max_training_environment_steps": 256,
                    "training.mappo.evaluation_interval_steps": 256,
                    "training.mappo.checkpoint_interval_steps": 128,
                    "output.logs_dir": str(root / "logs"),
                    "output.dashboard_logs_dir": str(root / "dashboard_logs"),
                    "output.plots_dir": str(root / "plots"),
                }
            )

            outcome = write_cagat_mappo_training_artifacts(config, training_result())

            self.assertEqual(outcome.smoke_gate_status, "pass")
            self.assertEqual(outcome.signal_gate_status, "insufficient-horizon")
            self.assertEqual(len(outcome.artifacts), 5)
            self.assertTrue(all(Path(path).exists() for path in outcome.artifacts))

            paths = config.artifact_paths()
            summary = json.loads(Path(paths["aggregate_metrics"]).read_text(encoding="utf-8"))
            self.assertEqual(summary["smoke_training_gate_status"], "pass")
            self.assertEqual(summary["signal_gate_status"], "insufficient-horizon")
            self.assertTrue(all(summary["checks"].values()))
            self.assertIn("single-seed", summary["claim_boundary"])

            with Path(paths["dashboard_csv"]).open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            self.assertEqual(sum(row["record_type"] == "episode" for row in rows), 8)
            self.assertEqual(sum(row["record_type"] == "ppo_epoch" for row in rows), 4)
            self.assertTrue(all(row["agent_credit_mode"] == "team" for row in rows))
            self.assertTrue(
                all(
                    row["agent_value_mean"] == ""
                    for row in rows
                    if row["record_type"] == "ppo_epoch"
                )
            )
            self.assertTrue(all(row["signal_gate_status"] == "insufficient-horizon" for row in rows))
            self.assertEqual(
                Path(paths["dashboard_png"]).read_bytes()[:8],
                b"\x89PNG\r\n\x1a\n",
            )

    def test_role_credit_telemetry_is_additive_under_schema_v3(self) -> None:
        telemetry = AgentCreditPPOEpochTelemetry(
            per_agent_value_mean=(0.1, 0.2, 0.3, 0.4),
            per_agent_td_residual_mean=(1.1, 1.2, 1.3, 1.4),
            per_agent_advantage_mean=(2.1, 2.2, 2.3, 2.4),
            per_agent_return_mean=(3.1, 3.2, 3.3, 3.4),
            same_timestep_advantage_equality_rate=0.25,
            route_category_agent_advantage_mean={
                "local": (1.0, None, 3.0, 4.0),
                "remote": (None, 2.0, None, 4.0),
                "defer": (1.0, 2.0, 3.0, None),
            },
            per_head_critic_loss=(0.5, 0.6, 0.7, 0.8),
        )
        original = training_result()
        update = original.updates[0]
        epoch_rows = tuple(
            replace(epoch, agent_credit_telemetry=telemetry)
            for epoch in update.output.epoch_diagnostics
        )
        role_result = replace(
            original,
            updates=(
                replace(
                    update,
                    output=replace(
                        update.output,
                        epoch_diagnostics=epoch_rows,
                    ),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_run_config(
                cli_overrides={
                    "mode": "rl",
                    "method_id": "ca_gat_mappo",
                    "training.formal_rl_enabled": True,
                    "training.mappo.training_device": "cpu",
                    "training.mappo.agent_credit_mode": "role_decomposed",
                    "environment.episode_horizon": 32,
                    "training.mappo.max_training_episodes": 8,
                    "training.mappo.max_training_environment_steps": 256,
                    "training.mappo.evaluation_interval_steps": 256,
                    "training.mappo.checkpoint_interval_steps": 128,
                    "output.logs_dir": str(root / "logs"),
                    "output.dashboard_logs_dir": str(root / "dashboard_logs"),
                    "output.plots_dir": str(root / "plots"),
                }
            )
            write_cagat_mappo_training_artifacts(config, role_result)
            paths = config.artifact_paths()
            with Path(paths["dashboard_csv"]).open(
                encoding="utf-8",
                newline="",
            ) as handle:
                rows = [
                    row
                    for row in csv.DictReader(handle)
                    if row["record_type"] == "ppo_epoch"
                ]
            self.assertEqual(len(rows), 4)
            self.assertTrue(
                all(row["agent_credit_mode"] == "role_decomposed" for row in rows)
            )
            self.assertEqual(
                json.loads(rows[-1]["agent_value_mean"]),
                [0.1, 0.2, 0.3, 0.4],
            )
            self.assertEqual(
                json.loads(rows[-1]["per_head_critic_loss"]),
                [0.5, 0.6, 0.7, 0.8],
            )
            summary = json.loads(
                Path(paths["aggregate_metrics"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["training"]["agent_credit_mode"],
                "role_decomposed",
            )
            self.assertEqual(
                summary["ppo"]["agent_credit_latest"]["per_agent_return_mean"],
                [3.1, 3.2, 3.3, 3.4],
            )


if __name__ == "__main__":
    unittest.main()
