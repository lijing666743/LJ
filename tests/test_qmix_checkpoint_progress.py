"""Real tiny-environment milestone fixtures; never paper experiment results."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.config import load_run_config
from src.artifacts import _filesystem_path
from src.evaluation.qmix_checkpoint_progress import QMIXCheckpointProgressRunner, checkpoint_milestones, main
from src.evaluation.qmix_runner import QMIXEvaluationRunner
from src.models.qmix_trainer import QMIXTrainer, checkpoint_bytes


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def trained(tmp_path):
    config = load_run_config(ROOT / "tests/fixtures/configs/qmix_legacy/fa-gat-qmix-seed42-40k.json",
        cli_overrides={"training.mappo.training_device": "cpu", "environment.episode_horizon": 4,
            "training.qmix.max_training_episodes": 10, "training.qmix.max_training_environment_steps": 40,
            "training.qmix.checkpoint_interval_steps": 4, "training.qmix.recurrent_sequence_length_slots": 2,
            "training.qmix.sequence_batch_size": 2, "training.qmix.target_update_interval_updates": 2,
            "evaluation.evaluation_seeds": [1042], "output.logs_dir": str(tmp_path / "logs"),
            "output.plots_dir": str(tmp_path / "plots"), "output.dashboard_logs_dir": str(tmp_path / "dashboard_logs")})
    trainer = QMIXTrainer(config)
    trainer.train(write_checkpoints=True)
    return config, trainer


def hash_files(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob("*.pt")}


def test_actual_40k_milestone_geometry_has_nine_periodic_and_final():
    config = load_run_config(ROOT / "tests/fixtures/configs/qmix_legacy/fa-gat-qmix-seed42-40k.json")
    assert checkpoint_milestones(config) == tuple(range(4000, 40001, 4000))


def test_periodic_requires_explicit_development_purpose_and_correct_step(trained):
    config, _ = trained
    checkpoint = Path(config.artifact_paths()["checkpoint_directory"]) / "step_000004.pt"
    evaluation = replace(config, mode="evaluation")
    with pytest.raises(ValueError, match="final checkpoint"):
        QMIXEvaluationRunner(evaluation, checkpoint)
    runner = QMIXEvaluationRunner(evaluation, checkpoint, checkpoint_purpose="development_checkpoint",
                                 expected_training_steps=4)
    assert runner.training_steps == 4 and runner.training_budget == 40 and runner.checkpoint_kind == "periodic"
    with pytest.raises(ValueError, match="milestone"):
        QMIXEvaluationRunner(evaluation, checkpoint, checkpoint_purpose="development_checkpoint",
                             expected_training_steps=8)


def test_all_ten_checkpoints_evaluate_without_optimizer_or_source_mutation(trained):
    config, _ = trained
    directory = Path(config.artifact_paths()["checkpoint_directory"])
    before = hash_files(directory)
    progress = []
    with patch("torch.optim.Adam", side_effect=AssertionError("evaluation must not construct an optimizer")):
        result = QMIXCheckpointProgressRunner(config).run(progress=progress.append)
    assert hash_files(directory) == before
    assert len(result["manifest"]["checkpoints"]) == 10
    assert result["manifest"]["episode_count"] == 40
    assert result["manifest"]["primary_report_step"] == 40
    assert result["manifest"]["network_updates"] == 0
    assert result["manifest"]["fixed_baseline_repeat_matches"] == 27
    assert len(result["rows"]) == 40
    assert len([x for x in progress if "checkpoint_finished" in x]) == 10
    with Path(result["artifacts"][-2]).open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 40
    # Rerunning without an explicit reuse instruction never overwrites results.
    with pytest.raises(FileExistsError):
        QMIXCheckpointProgressRunner(config).run()
    with patch.object(QMIXEvaluationRunner, "episode", side_effect=AssertionError("reuse must not simulate again")):
        reused = QMIXCheckpointProgressRunner(config, reuse_completed=True).run()
    assert reused["reused_checkpoints"] == 10
    assert reused["rows"] == result["rows"]


def test_missing_middle_checkpoint_fails_before_environment(trained):
    config, _ = trained
    # A separate directory contains actual copies, leaving original inputs alone.
    original = Path(config.artifact_paths()["checkpoint_directory"])
    incomplete = original.parent / "incomplete"
    incomplete.mkdir()
    for path in original.glob("*.pt"):
        if path.name != "step_000020.pt":
            (incomplete / path.name).write_bytes(path.read_bytes())
    with patch.object(QMIXEvaluationRunner, "episode", side_effect=AssertionError("must not evaluate partial training")):
        with pytest.raises(FileNotFoundError, match="step_000020.pt"):
            QMIXCheckpointProgressRunner(config, checkpoint_directory=incomplete).run()
    assert not (incomplete.parent / "checkpoint_progress").exists()


def test_partial_completed_series_can_reuse_only_verified_checkpoints(trained):
    config, _ = trained
    campaign = QMIXCheckpointProgressRunner(config)
    _, runners, _ = campaign._preflight()
    runners[0].run()
    seen = []
    result = QMIXCheckpointProgressRunner(config, reuse_completed=True).run(progress=seen.append)
    assert result["reused_checkpoints"] == 1
    assert len([x for x in seen if "method_id" in x]) == 36


def test_reuse_rejects_tampered_metric_records(trained):
    config, _ = trained
    _, runners, _ = QMIXCheckpointProgressRunner(config)._preflight()
    runners[0].run()
    path = _filesystem_path(runners[0].artifact_paths()["episodes"])
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[0]["metrics"]["completed"] += 1
    path.write_text("\n".join(json.dumps(row) for row in records)+"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conservation"):
        QMIXCheckpointProgressRunner(config, reuse_completed=True).run()


def test_renamed_checkpoint_and_foreign_config_are_rejected(trained):
    config, _ = trained
    directory = Path(config.artifact_paths()["checkpoint_directory"])
    renamed = directory / "step_000008_alias.pt"
    renamed.write_bytes((directory / "step_000004.pt").read_bytes())
    with pytest.raises(ValueError, match="filename"):
        QMIXEvaluationRunner(replace(config, mode="evaluation"), renamed,
                             checkpoint_purpose="development_checkpoint")
    with pytest.raises(ValueError, match="source config"):
        QMIXCheckpointProgressRunner(replace(config, seed=43), checkpoint_directory=directory).run()


def test_cli_runs_the_complete_series_and_explicit_reuse(trained, tmp_path, monkeypatch):
    config, _ = trained
    path = tmp_path / "training.json"
    path.write_text(json.dumps(config.resolved_dict()), encoding="utf-8")
    # Git provenance is excluded from config hash; pass the actual directory.
    monkeypatch.setattr("sys.argv", ["qmix_checkpoint_progress", "--config", str(path),
        "--checkpoint-directory", config.artifact_paths()["checkpoint_directory"]])
    assert main() == 0
    monkeypatch.setattr("sys.argv", ["qmix_checkpoint_progress", "--config", str(path),
        "--checkpoint-directory", config.artifact_paths()["checkpoint_directory"], "--reuse-completed"])
    assert main() == 0
