"""Fast tests for the Formal Experiment artifact non-overwrite gate."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from src.artifacts import (
    ArtifactConflictError,
    ArtifactWriteError,
    ORDINARY_ARTIFACT_KEYS,
    atomic_write_bytes_group,
    preflight_formal_training_artifacts,
)
from src.config import load_run_config, write_config_snapshot
from src.execution import ExecutionContext
from src.policies.heuristic_policy import HeuristicPolicy
from src.policies.local_only_policy import LocalOnlyPolicy
from src.policies.random_policy import RandomPolicy
from src.registry import build_default_registry, ca_gat_mappo_training_handler
from src.rollout import RolloutRunner
from src.training_artifacts import write_cagat_mappo_training_artifacts
from tests.test_ca_gat_mappo_training_artifacts import training_result


def _file_state(path: Path) -> tuple[bytes, str, int]:
    content = path.read_bytes()
    return content, hashlib.sha256(content).hexdigest(), path.stat().st_mtime_ns


def _output_overrides(root: Path) -> dict[str, str]:
    return {
        "output.logs_dir": str(root / "logs"),
        "output.dashboard_logs_dir": str(root / "dashboard_logs"),
        "output.plots_dir": str(root / "plots"),
    }


def _formal_config(root: Path):
    return load_run_config(
        cli_overrides={
            "mode": "rl",
            "method_id": "ca_gat_mappo",
            "launch_profile": "rl-formal",
            "training.formal_rl_enabled": True,
            "training.mappo.training_device": "cpu",
            **_output_overrides(root),
        }
    )


def _training_artifact_config(root: Path):
    return load_run_config(
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
            **_output_overrides(root),
        }
    )


def _rollout_config(root: Path, mode: str, method_id: str):
    return load_run_config(
        cli_overrides={
            "mode": mode,
            "method_id": method_id,
            "scenario_id": "small",
            "seed": 42,
            "environment.episode_horizon": 1,
            "environment.arrival_probabilities": [0.0, 0.0, 0.0, 0.0],
            **_output_overrides(root),
        }
    )


class AtomicArtifactWriterTests(unittest.TestCase):
    def test_absent_targets_are_written_with_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.bin"
            second = root / "nested" / "second.bin"

            written = atomic_write_bytes_group(
                ((first, b"first\n"), (second, b"second\x00value")),
                group_name="test artifact group",
            )

            self.assertEqual(written, (first, second))
            self.assertEqual(first.read_bytes(), b"first\n")
            self.assertEqual(second.read_bytes(), b"second\x00value")

    def test_existing_target_preserves_bytes_hash_mtime_and_blocks_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new_target = root / "new.bin"
            existing = root / "existing.bin"
            existing.write_bytes(b"immutable original")
            fixed_time = 1_700_000_000_123_456_700
            os.utime(existing, ns=(fixed_time, fixed_time))
            before = _file_state(existing)

            with self.assertRaisesRegex(
                ArtifactConflictError,
                "already exists; refusing overwrite",
            ):
                atomic_write_bytes_group(
                    ((new_target, b"new"), (existing, b"replacement")),
                    group_name="test artifact group",
                )

            self.assertFalse(new_target.exists())
            self.assertEqual(_file_state(existing), before)

    def test_staging_failure_cleans_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "failure.bin"
            with patch("src.artifacts.os.fsync", side_effect=OSError("synthetic fsync")):
                with self.assertRaisesRegex(ArtifactWriteError, "synthetic fsync"):
                    atomic_write_bytes_group(((target, b"payload"),))
            self.assertFalse(target.exists())
            self.assertEqual(tuple(root.glob("*.tmp")), ())

    def test_cleanup_is_best_effort_and_does_not_mask_primary_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "failure.bin"
            with patch(
                "src.artifacts.os.fsync",
                side_effect=OSError("primary failure"),
            ), patch(
                "src.artifacts.Path.unlink",
                side_effect=OSError("cleanup failure"),
            ) as unlink:
                with self.assertRaisesRegex(ArtifactWriteError, "primary failure"):
                    atomic_write_bytes_group(((target, b"payload"),))
            unlink.assert_called_once()
            self.assertFalse(target.exists())

    def test_config_snapshot_is_immutable_after_first_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config_snapshot.yaml"
            config = load_run_config()
            self.assertEqual(write_config_snapshot(config, target), target)
            before = _file_state(target)

            with self.assertRaises(ArtifactConflictError):
                write_config_snapshot(config, target)

            self.assertEqual(_file_state(target), before)


class TrainingArtifactGroupTests(unittest.TestCase):
    def test_one_conflict_blocks_every_other_training_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _training_artifact_config(Path(directory))
            paths = config.artifact_paths()
            conflict = Path(paths["aggregate_metrics"])
            conflict.parent.mkdir(parents=True, exist_ok=True)
            conflict.write_bytes(b"existing aggregate")
            fixed_time = 1_700_000_000_223_456_700
            os.utime(conflict, ns=(fixed_time, fixed_time))
            before = _file_state(conflict)

            with patch("src.training_artifacts._plot_dashboard") as plot:
                with self.assertRaises(ArtifactConflictError):
                    write_cagat_mappo_training_artifacts(config, training_result())

            plot.assert_not_called()
            self.assertEqual(_file_state(conflict), before)
            for key in (
                "config_snapshot",
                "raw_metrics",
                "dashboard_csv",
                "dashboard_png",
            ):
                self.assertFalse(Path(paths[key]).exists(), key)


class FormalPreflightTests(unittest.TestCase):
    @staticmethod
    def _fake_training_modules(training):
        models_package = ModuleType("src.models")
        models_package.__path__ = []
        trainer_module = ModuleType("src.models.ca_gat_mappo_trainer")
        trainer_module.CAGATMAPPOTrainer = training
        return models_package, trainer_module

    def test_fresh_formal_rejects_interrupted_run_before_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _formal_config(Path(directory))
            paths = config.artifact_paths()
            periodic = Path(paths["checkpoint_directory"]) / "step_50000.pt"
            periodic.parent.mkdir(parents=True, exist_ok=True)
            periodic.write_bytes(b"periodic checkpoint")
            trainer_type = Mock(side_effect=AssertionError("trainer must not start"))
            models_package, trainer_module = self._fake_training_modules(trainer_type)

            with patch.dict(
                sys.modules,
                {
                    "src.models": models_package,
                    "src.models.ca_gat_mappo_trainer": trainer_module,
                },
            ):
                result = ca_gat_mappo_training_handler(config)

            self.assertEqual(result.status, "failed")
            self.assertIn("already exists", result.message)
            trainer_type.assert_not_called()
            self.assertEqual(periodic.read_bytes(), b"periodic checkpoint")

    def test_resume_allows_periodic_checkpoint_and_existing_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _formal_config(Path(directory))
            paths = config.artifact_paths()
            periodic = Path(paths["checkpoint_directory"]) / "step_50000.pt"
            periodic.parent.mkdir(parents=True, exist_ok=True)
            periodic.write_bytes(b"periodic checkpoint")
            resumed_trainer = Mock()
            training_result_stub = SimpleNamespace(
                total_environment_transitions=500_000,
                completed_episode_count=1_000,
                ppo_update_count=1_953,
            )
            resumed_trainer.train_with_checkpoints.return_value = training_result_stub
            trainer_type = Mock()
            trainer_type.resume_from_checkpoint.return_value = resumed_trainer
            models_package, trainer_module = self._fake_training_modules(trainer_type)
            artifacts_module = ModuleType("src.training_artifacts")
            artifact_writer = Mock(
                return_value=SimpleNamespace(
                    artifacts=(),
                    smoke_gate_status="pass",
                    signal_gate_status="signal-watch",
                    reward_mean=0.1,
                    actor_loss=-0.2,
                    critic_loss=0.3,
                    entropy=0.4,
                )
            )
            artifacts_module.write_cagat_mappo_training_artifacts = artifact_writer

            with patch.dict(
                sys.modules,
                {
                    "src.models": models_package,
                    "src.models.ca_gat_mappo_trainer": trainer_module,
                    "src.training_artifacts": artifacts_module,
                },
            ):
                result = ca_gat_mappo_training_handler(
                    config,
                    execution_context=ExecutionContext.from_resume_path(periodic),
                )

            self.assertEqual(result.status, "completed")
            trainer_type.assert_not_called()
            trainer_type.resume_from_checkpoint.assert_called_once_with(config, periodic)
            resumed_trainer.train_with_checkpoints.assert_called_once_with()
            artifact_writer.assert_called_once_with(config, training_result_stub)

    def test_resume_rejects_final_checkpoint_before_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _formal_config(Path(directory))
            paths = config.artifact_paths()
            periodic = Path(paths["checkpoint_directory"]) / "step_50000.pt"
            final = Path(paths["final_checkpoint"])
            periodic.parent.mkdir(parents=True, exist_ok=True)
            periodic.write_bytes(b"periodic checkpoint")
            final.write_bytes(b"immutable final checkpoint")
            fixed_time = 1_700_000_000_323_456_700
            os.utime(final, ns=(fixed_time, fixed_time))
            before = _file_state(final)
            trainer_type = Mock(side_effect=AssertionError("trainer must not start"))
            models_package, trainer_module = self._fake_training_modules(trainer_type)

            with patch.dict(
                sys.modules,
                {
                    "src.models": models_package,
                    "src.models.ca_gat_mappo_trainer": trainer_module,
                },
            ):
                result = ca_gat_mappo_training_handler(
                    config,
                    execution_context=ExecutionContext.from_resume_path(periodic),
                )

            self.assertEqual(result.status, "failed")
            self.assertIn("final.pt", result.message)
            trainer_type.assert_not_called()
            self.assertEqual(_file_state(final), before)

    def test_resume_rejects_every_reserved_final_artifact_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for index, key in enumerate((*ORDINARY_ARTIFACT_KEYS, "final_checkpoint")):
                with self.subTest(key=key):
                    config = _formal_config(base / str(index))
                    paths = config.artifact_paths()
                    conflict = Path(paths[key])
                    conflict.parent.mkdir(parents=True, exist_ok=True)
                    conflict.write_bytes(key.encode("utf-8"))

                    with self.assertRaisesRegex(
                        ArtifactConflictError,
                        "already exists",
                    ):
                        preflight_formal_training_artifacts(paths, resume=True)


class BaselineIdentityTests(unittest.TestCase):
    def test_random_heuristic_and_local_only_repeat_identity_fail_before_rollout(self) -> None:
        cases = (
            ("random", "random", lambda config: RandomPolicy(config.seed)),
            ("heuristic", "heuristic", HeuristicPolicy),
            ("baseline", "local_only", LocalOnlyPolicy),
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            registry = build_default_registry()
            for mode, method_id, policy_factory in cases:
                with self.subTest(method_id=method_id):
                    config = _rollout_config(base / method_id, mode, method_id)
                    policy = policy_factory(config)
                    first = RolloutRunner(config, policy).run()
                    before = {
                        Path(path): _file_state(Path(path)) for path in first.artifacts
                    }

                    with patch("src.rollout.U2UMECEnvironment") as environment:
                        second = registry.resolve(mode, method_id)(config)

                    self.assertEqual(second.status, "failed")
                    self.assertIn("already exists", second.message)
                    environment.assert_not_called()
                    self.assertEqual(
                        {path: _file_state(path) for path in before},
                        before,
                    )

    def test_reserved_figure_input_conflict_fails_before_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _rollout_config(Path(directory), "random", "random")
            figure_input = Path(config.artifact_paths()["figure_input"])
            figure_input.parent.mkdir(parents=True, exist_ok=True)
            figure_input.write_bytes(b"existing figure input")

            with patch("src.rollout.U2UMECEnvironment") as environment:
                with self.assertRaises(ArtifactConflictError):
                    RolloutRunner(config, RandomPolicy(config.seed)).run()

            environment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
