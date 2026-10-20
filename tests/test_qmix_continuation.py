"""Real small-environment continuity and immutable inherited-evaluation checks."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.artifacts import _filesystem_path
from src.cli import main
from src.config import load_run_config
from src.evaluation.qmix_checkpoint_progress import QMIXCheckpointProgressRunner
from src.evaluation.qmix_continuation_progress import QMIXContinuationProgressRunner
from src.evaluation.qmix_runner import QMIXEvaluationRunner
from src.execution import ExecutionContext, validate_execution_context
from src.models.qmix_continuation import (CONTINUITY_KEYS, QMIXContinuationTrainer,
    load_continuation_checkpoint, state_digest, validate_budget_extension)
from src.models.qmix_trainer import QMIXTrainer, checkpoint_bytes


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def pair(tmp_path):
    source = load_run_config(ROOT / "tests/fixtures/configs/qmix_legacy/fa-gat-qmix-seed42-8k-episode-return-development.json",
        cli_overrides={"training.mappo.training_device": "cpu", "environment.episode_horizon": 4,
            "training.qmix.max_training_episodes": 2, "training.qmix.max_training_environment_steps": 8,
            "training.qmix.checkpoint_interval_steps": 4, "training.qmix.recurrent_sequence_length_slots": 2,
            "training.qmix.sequence_batch_size": 2, "training.qmix.target_update_interval_updates": 2,
            "evaluation.evaluation_seeds": [1042], "output.logs_dir": str(tmp_path / "a"),
            "output.plots_dir": str(tmp_path / "pa"), "output.dashboard_logs_dir": str(tmp_path / "da")})
    target = replace(source, training=replace(source.training, qmix=replace(source.training.qmix,
        max_training_episodes=4, max_training_environment_steps=16)), output=replace(source.output,
        logs_dir=str(tmp_path / "b"), plots_dir=str(tmp_path / "pb"), dashboard_logs_dir=str(tmp_path / "db")))
    trainer = QMIXTrainer(source)
    trainer.train(write_checkpoints=True)
    path = Path(source.artifact_paths()["final_checkpoint"])
    return source, target, trainer, path


def numerical_state(trainer):
    payload = trainer.payload("final")
    result = {key: payload[key] for key in CONTINUITY_KEYS}
    # Preserve each run's provenance while comparing all numerical history.
    result["episodes"] = [{k: v for k, v in row.items() if k != "run_id"} for row in result["episodes"]]
    return state_digest(result)


def original_hashes(config):
    paths = [p for p in Path(config.output.logs_dir).rglob("*") if _filesystem_path(p).is_file()]
    paths += list(Path(config.output.dashboard_logs_dir).glob("*.csv"))
    return {str(p): hashlib.sha256(_filesystem_path(p).read_bytes()).hexdigest() for p in paths}


def test_final_extension_matches_uninterrupted_model_adam_replay_rng_and_history(pair):
    source, target, old, path = pair
    before = original_hashes(source)
    restored = QMIXContinuationTrainer.from_checkpoint(target, path, extend=True)
    assert restored.restore_verification["environment_steps_executed"] == 0
    assert restored.steps == 8 and restored.learner.updates == 3
    assert restored.episodes == old.episodes
    uninterrupted = QMIXTrainer(target)
    uninterrupted.train(stop_after_episodes=2)
    assert numerical_state(restored) == numerical_state(uninterrupted)
    restored.train(write_checkpoints=True)
    uninterrupted.train()
    assert numerical_state(restored) == numerical_state(uninterrupted)
    assert restored.episodes[:2] == old.episodes
    assert restored.steps == 16 and restored.learner.updates == 7
    assert restored.budget_extension["replayed_environment_steps"] == 0
    assert original_hashes(source) == before


def test_continued_periodic_resume_preserves_ancestry_and_exact_state(pair):
    _, target, _, path = pair
    uninterrupted = QMIXContinuationTrainer.from_checkpoint(target, path, extend=True)
    uninterrupted.train()
    first = QMIXContinuationTrainer.from_checkpoint(target, path, extend=True)
    first.train(stop_after_episodes=3, write_checkpoints=True)
    checkpoint = Path(target.artifact_paths()["checkpoint_directory"]) / "step_000012.pt"
    resumed = QMIXContinuationTrainer.from_checkpoint(target, checkpoint)
    resumed.train()
    assert numerical_state(resumed) == numerical_state(uninterrupted)
    assert resumed.budget_extension == uninterrupted.budget_extension


@pytest.mark.parametrize("change", ["seed", "learning_rate", "epsilon", "reward", "device", "alias"])
def test_extension_rejects_any_scientific_or_output_overlap_change(pair, change):
    source, target, _, path = pair
    if change == "seed":
        target = replace(target, seed=43)
    elif change in {"learning_rate", "epsilon"}:
        key = "learning_rate" if change == "learning_rate" else "epsilon_decay_steps"
        value = .001 if change == "learning_rate" else 16000
        target = replace(target, training=replace(target.training, qmix=replace(target.training.qmix, **{key: value})))
    elif change == "reward":
        target = replace(target, reward=replace(target.reward, beta_completion_credit=.4))
    elif change == "device":
        target = replace(target, training=replace(target.training, mappo=replace(target.training.mappo, training_device="cuda")))
    else:
        target = replace(target, output=replace(target.output, logs_dir=source.output.logs_dir))
    with patch("src.models.qmix_continuation.QMIXContinuationTrainer.__init__", side_effect=AssertionError("must reject before model construction")):
        with pytest.raises(ValueError, match="scientific|overlap"):
            QMIXContinuationTrainer.from_checkpoint(target, path, extend=True)


def test_final_requires_explicit_extension_and_periodic_cannot_extend(pair):
    source, target, _, path = pair
    with pytest.raises(ValueError, match="periodic checkpoint"):
        QMIXContinuationTrainer.from_checkpoint(target, path)
    periodic = Path(source.artifact_paths()["checkpoint_directory"]) / "step_000004.pt"
    with pytest.raises(ValueError, match="completed source final"):
        QMIXContinuationTrainer.from_checkpoint(target, periodic, extend=True)


def test_real_main_check_resume_has_zero_steps_updates_and_outputs(pair, tmp_path):
    source, target, _, path = pair
    config_path = tmp_path / "extension.json"
    config_path.write_text(json.dumps(target.resolved_dict()), encoding="utf-8")
    before = original_hashes(source)
    output = []
    with patch.object(QMIXTrainer, "train", side_effect=AssertionError("check must not train")), \
         patch("src.models.qmix_trainer.U2UMECEnvironment", side_effect=AssertionError("check must not construct environment")):
        assert main(["--config", str(config_path), "--resume-from", str(path),
            "--extend-training-budget", "--check-resume"], output_fn=output.append) == 0
    assert any("resume_check_passed" in row for row in output)
    assert not Path(target.output.logs_dir).exists()
    assert original_hashes(source) == before


def test_real_main_and_no_argument_menu_resume_same_final(pair, tmp_path):
    _, target, _, path = pair
    config_path = tmp_path / "extension.json"
    config_path.write_text(json.dumps(target.resolved_dict()), encoding="utf-8")
    with patch.object(QMIXContinuationTrainer, "train", side_effect=RuntimeError("fixture-stop-before-training")) as train:
        output = []
        answers = iter(["10", str(config_path), "", "", str(path), "extend"])
        assert main([], input_fn=lambda _: next(answers), output_fn=output.append) == 1
        assert train.call_count == 1
        assert any("fixture-stop-before-training" in row for row in output)
    assert not Path(target.output.logs_dir).exists()


def test_new_milestones_reuse_old_evaluations_without_copying_or_resimulating(pair):
    source, target, _, path = pair
    QMIXCheckpointProgressRunner(source).run()
    before = original_hashes(source)
    trainer = QMIXContinuationTrainer.from_checkpoint(target, path, extend=True)
    trainer.train(write_checkpoints=True)
    seen = []
    result = QMIXContinuationProgressRunner(target).run(progress=seen.append)
    assert result["reused_checkpoints"] == 2
    assert [x["training_steps"] for x in result["manifest"]["checkpoints"]] == [4, 8, 12, 16]
    assert result["manifest"]["episode_count"] == 16  # four methods, one scene, four points
    assert len([x for x in seen if "method_id" in x]) == 8
    assert result["manifest"]["primary_report_step"] == 16
    assert result["manifest"]["fixed_baseline_repeat_matches"] == 9
    assert original_hashes(source) == before
    with patch.object(QMIXEvaluationRunner, "episode", side_effect=AssertionError("must reuse all results")):
        assert QMIXContinuationProgressRunner(target, reuse_completed=True).run()["reused_checkpoints"] == 4


def test_missing_old_evaluation_and_changed_ancestry_fail_before_new_simulation(pair):
    source, target, _, path = pair
    trainer = QMIXContinuationTrainer.from_checkpoint(target, path, extend=True)
    trainer.train(write_checkpoints=True)
    with patch.object(QMIXEvaluationRunner, "episode", side_effect=AssertionError("must fail preflight")):
        with pytest.raises(FileNotFoundError, match="prior milestone evaluation"):
            QMIXContinuationProgressRunner(target).run()
    final = Path(target.artifact_paths()["final_checkpoint"])
    data = load_continuation_checkpoint(final)
    data["budget_extension"]["ancestors"][0]["sha256"] = "0" * 64
    tampered = final.parent / "tampered.pt"
    tampered.write_bytes(checkpoint_bytes(data))
    with pytest.raises(ValueError, match="ancestor checkpoint"):
        load_continuation_checkpoint(tampered)


def test_actual_8k16k_configs_change_only_caps_and_outputs():
    source = load_run_config(ROOT / "tests/fixtures/configs/qmix_legacy/fa-gat-qmix-seed42-8k-episode-return-development.json")
    target = load_run_config(ROOT / "tests/fixtures/configs/qmix_legacy/fa-gat-qmix-seed42-16k-episode-return-development.json")
    validate_budget_extension(source, target)
    assert target.training.qmix.max_training_environment_steps == 16000
    assert target.training.qmix.epsilon_decay_steps == 32000
    assert target.training.qmix.checkpoint_interval_steps == 4000
    assert source.training.qmix.target_update_interval_updates == target.training.qmix.target_update_interval_updates == 200
