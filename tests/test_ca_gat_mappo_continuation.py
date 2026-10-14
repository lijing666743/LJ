"""Small engineering fixtures for stateful extension; no formal experiments."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import pytest

from src.cli import build_arg_parser, build_execution_context_from_args
from src.config import CHECKPOINT_KIND_PERIODIC_RESUME, ConfigError
from src.execution import validate_execution_context
from src.models.ca_gat_mappo_checkpoint import (
    CheckpointError, atomic_save_checkpoint, load_checkpoint_payload,
    validate_checkpoint_payload, validate_periodic_checkpoint_compatibility,
)
from src.models.ca_gat_mappo_continuation import assert_state_equal, verify_replay_boundary
from src.models.ca_gat_mappo_trainer import CAGATMAPPOTrainer
from tests.test_ca_gat_mappo_checkpoint import (
    deterministic_updater_factory, make_checkpoint_config, make_payload,
)


def extension_config(source, root, budget=1040):
    return replace(source,
        training=replace(source.training, mappo=replace(source.training.mappo,
            max_training_environment_steps=budget,
            max_training_episodes=budget // source.environment.episode_horizon)),
        output=replace(source.output, logs_dir=str(root / "logs"),
            plots_dir=str(root / "plots"), dashboard_logs_dir=str(root / "dashboard_logs")))


class ContinuationTests(unittest.TestCase):
    def test_only_explicit_larger_caps_and_new_roots_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_checkpoint_config(root / "source", horizon=130, interval=260, budget=520)
            payload, *_ = make_payload(source, CHECKPOINT_KIND_PERIODIC_RESUME)
            target = extension_config(source, root / "target")
            def check(config, candidate=payload, explicit=True):
                return validate_periodic_checkpoint_compatibility(
                    config, candidate, dtype=torch.float32, cuda_available=False,
                    extend_training_budget=explicit)
            check(target)
            with self.assertRaisesRegex(CheckpointError, "config_hash mismatch"):
                check(target, explicit=False)
            mutations = [
                replace(target, seed=43),
                replace(target, environment=replace(target.environment, reward_completion_weight=2.0)),
                replace(target, training=replace(target.training, mappo=replace(target.training.mappo,
                    actor_learning_rate=0.0004))),
                replace(target, training=replace(target.training, mappo=replace(target.training.mappo,
                    checkpoint_interval_steps=130))),
                replace(target, output=replace(target.output, logs_dir=source.output.logs_dir)),
                extension_config(source, root / "same_budget", budget=520),
            ]
            for config in mutations:
                with self.subTest(config=config.config_hash), self.assertRaises(CheckpointError):
                    check(config)
            tampered = copy.deepcopy(payload)
            tampered["config_snapshot"]["seed"] = 999
            with self.assertRaisesRegex(CheckpointError, "content hash mismatch"):
                check(target, tampered)

    def test_extension_replay_tail_and_uninterrupted_training_are_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_checkpoint_config(root / "source", horizon=130, interval=260, budget=520)
            paused = CAGATMAPPOTrainer(source, updater_factory=deterministic_updater_factory)
            checkpoint = paused.train_until_periodic_checkpoint()
            source_restored = CAGATMAPPOTrainer.resume_from_checkpoint(
                source, checkpoint, updater_factory=deterministic_updater_factory)
            source_restored.train_with_checkpoints()
            reference = Path(source.artifact_paths()["final_checkpoint"])
            with self.assertRaisesRegex(CheckpointError, "PERIODIC_RESUME"):
                CAGATMAPPOTrainer.resume_from_checkpoint(
                    extension_config(source, root / "bad_final"), reference, extend_training_budget=True)

            target = extension_config(source, root / "target")
            resumed = CAGATMAPPOTrainer.resume_from_checkpoint(
                target, checkpoint, extend_training_budget=True, verify_replay_against=reference,
                updater_factory=deterministic_updater_factory)
            original = load_checkpoint_payload(checkpoint)
            self.assertEqual(len(resumed.rollout_buffer), 4)
            assert_state_equal(resumed.updater.optimizers.actor_optimizer.state_dict(),
                               original["optimizer_state"]["actor_adam"], "restored_adam")
            result = resumed.train_with_checkpoints()
            expected = CAGATMAPPOTrainer(target, updater_factory=deterministic_updater_factory)
            expected_result = expected.train()
            self.assertEqual(expected_result, result)
            for name in ("actor", "critic"):
                assert_state_equal(getattr(expected, name).state_dict(), getattr(resumed, name).state_dict(), name)
            for name in ("actor_optimizer", "critic_optimizer"):
                assert_state_equal(getattr(expected.updater.optimizers, name).state_dict(),
                                   getattr(resumed.updater.optimizers, name).state_dict(), name)
            self.assertTrue(torch.equal(expected.policy_generator.get_state(), resumed.policy_generator.get_state()))
            # At the old final boundary, keep the 8 regenerated samples; the old final had none.
            boundary = load_checkpoint_payload(Path(target.artifact_paths()["final_checkpoint"]).parent / "step_520.pt")
            self.assertEqual(boundary["active_rollout_state"]["rollout_length"], 8)
            self.assertEqual(boundary["config_snapshot"]["_metadata"]["continuation"]["replay_reference"]["status"],
                             "verified_exact")
            verify_replay_boundary(boundary, load_checkpoint_payload(reference))
            corrupted = copy.deepcopy(boundary)
            corrupted["config_snapshot"]["_metadata"]["continuation"]["replay_reference"]["status"] = "pending"
            with self.assertRaisesRegex(CheckpointError, "status contradicts"):
                validate_checkpoint_payload(corrupted)
            # This checkpoint supports ordinary strict resume without another budget change.
            strict = CAGATMAPPOTrainer.resume_from_checkpoint(
                target, Path(target.artifact_paths()["final_checkpoint"]).parent / "step_520.pt",
                updater_factory=deterministic_updater_factory)
            self.assertEqual(len(strict.rollout_buffer), 8)
            self.assertEqual(strict._continuation_provenance["replay_reference"]["status"], "verified_exact")

    def test_replay_mismatch_stops_at_boundary_before_any_further_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_checkpoint_config(root / "source", horizon=130, interval=260, budget=520)
            paused = CAGATMAPPOTrainer(source, updater_factory=deterministic_updater_factory)
            checkpoint = paused.train_until_periodic_checkpoint()
            original = CAGATMAPPOTrainer.resume_from_checkpoint(
                source, checkpoint, updater_factory=deterministic_updater_factory)
            original.train_with_checkpoints()
            bad = copy.deepcopy(load_checkpoint_payload(source.artifact_paths()["final_checkpoint"]))
            key = next(iter(bad["model_state"]["actor"]))
            bad["model_state"]["actor"][key].flatten()[0] += 1
            reference = atomic_save_checkpoint(bad, root / "mismatched_final.pt")
            target = extension_config(source, root / "target")
            resumed = CAGATMAPPOTrainer.resume_from_checkpoint(
                target, checkpoint, extend_training_budget=True, verify_replay_against=reference,
                updater_factory=deterministic_updater_factory)
            with self.assertRaisesRegex(CheckpointError, "replay verification mismatch"):
                resumed.train_with_checkpoints()
            self.assertEqual(resumed._transitions, 520)
            self.assertEqual(resumed.policy_version, 2)
            self.assertEqual(len(resumed.rollout_buffer), 8)
            self.assertFalse(Path(target.artifact_paths()["final_checkpoint"]).exists())

    def test_cli_check_never_dispatches_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_checkpoint_config(root / "source", horizon=130, interval=260, budget=520)
            paused = CAGATMAPPOTrainer(source, updater_factory=deterministic_updater_factory)
            checkpoint = paused.train_until_periodic_checkpoint()
            target = extension_config(source, root / "target")
            from src.execution import ExecutionContext
            from src.cli import _check_resume
            context = ExecutionContext(resume_from=checkpoint, extend_training_budget=True, check_resume=True)
            messages = []
            with patch.object(CAGATMAPPOTrainer, "_run", side_effect=AssertionError("training called")):
                self.assertEqual(_check_resume(target, context, messages.append), 0, messages)
            self.assertIn('"environment_steps_executed": 0', messages[0])
            self.assertFalse(Path(target.output.logs_dir).exists())
            args = build_arg_parser().parse_args(["--check-resume", "--extend-training-budget"])
            with self.assertRaisesRegex(ConfigError, "require --resume-from"):
                validate_execution_context(source, build_execution_context_from_args(args))


@pytest.mark.parametrize("group", list("ABC"))
def test_real_abc_ppo_extension_preserves_learning_and_regenerates_tail(group, tmp_path):
    """Real PPO and shadow labels, with small horizons only for engineering."""

    from src.config import load_run_config
    root = Path(__file__).resolve().parents[1]
    original = load_run_config(root / f"experiments/task-allocation/configs/{group.lower()}-seed42-8k.json")
    source = replace(original,
        environment=replace(original.environment, episode_horizon=20),
        training=replace(original.training, mappo=replace(original.training.mappo,
            training_device="cpu", max_training_environment_steps=520, max_training_episodes=26,
            checkpoint_interval_steps=260, evaluation_interval_steps=260)),
        output=replace(original.output, logs_dir=str(tmp_path / "source/logs"),
            plots_dir=str(tmp_path / "source/plots"), dashboard_logs_dir=str(tmp_path / "source/dashboard_logs")))
    object.__setattr__(source, "git_commit", original.git_commit)
    source.validate()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        paused = CAGATMAPPOTrainer(source)
        checkpoint = paused.train_until_periodic_checkpoint()
        old_run = CAGATMAPPOTrainer.resume_from_checkpoint(source, checkpoint)
        old_run.train_with_checkpoints()
        reference = Path(source.artifact_paths()["final_checkpoint"])
        target = extension_config(source, tmp_path / "target", budget=780)
        object.__setattr__(target, "git_commit", original.git_commit)
        resumed = CAGATMAPPOTrainer.resume_from_checkpoint(
            target, checkpoint, extend_training_budget=True, verify_replay_against=reference)
        actual = resumed.train_with_checkpoints()
        control_config = replace(target, output=replace(target.output,
            logs_dir=str(tmp_path / "control/logs"), plots_dir=str(tmp_path / "control/plots"),
            dashboard_logs_dir=str(tmp_path / "control/dashboard_logs")))
        object.__setattr__(control_config, "git_commit", original.git_commit)
        control = CAGATMAPPOTrainer(control_config)
        expected = control.train()
        assert (actual.total_environment_transitions, actual.optimized_transitions,
                actual.unused_final_tail_transitions, actual.ppo_update_count) == (780, 768, 12, 3)
        # Optional historical diagnostics are not persisted by V1; all saved diagnostics are checked.
        assert replace(actual, updates=expected.updates) == expected
        assert_state_equal(control._diagnostics_checkpoint_state(), resumed._diagnostics_checkpoint_state(), "diagnostics")
        for name in ("actor", "critic"):
            assert_state_equal(getattr(control, name).state_dict(), getattr(resumed, name).state_dict(), name)
        for name in ("actor_optimizer", "critic_optimizer"):
            assert_state_equal(getattr(control.updater.optimizers, name).state_dict(),
                               getattr(resumed.updater.optimizers, name).state_dict(), name)
        assert torch.equal(control.policy_generator.get_state(), resumed.policy_generator.get_state())
        boundary = load_checkpoint_payload(Path(target.artifact_paths()["final_checkpoint"]).parent / "step_520.pt")
        verify_replay_boundary(boundary, load_checkpoint_payload(reference))
        assert boundary["active_rollout_state"]["rollout_length"] == 8
        assert boundary["config_snapshot"]["_metadata"]["continuation"]["replay_reference"]["status"] == "verified_exact"
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
