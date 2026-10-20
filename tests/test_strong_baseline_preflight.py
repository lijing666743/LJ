"""Fair comparison checks, including failures that defaults alone miss."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import load_run_config
from src.registry import unavailable_handler
from src.training.strong_baseline_preflight import (
    build_qmix_draft, inspect_comparison, linear_epsilon,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "experiments/task-allocation/configs/c-seed43-40k.json"


@pytest.fixture
def reference():
    return load_run_config(REFERENCE)


def load_draft(reference, tmp_path, seed=42):
    path = tmp_path / f"qmix{seed}.json"
    path.write_text(json.dumps(build_qmix_draft(reference, seed)), encoding="utf-8")
    return load_run_config(path)


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_disabled_draft_is_fair_and_does_not_mutate_c(reference, tmp_path, seed):
    original = copy.deepcopy(reference.resolved_dict())
    candidate = load_draft(reference, tmp_path, seed)
    result = inspect_comparison(reference, candidate, tmp_path)
    assert result["configuration_status"] == "PASS"
    assert not result["launch_ready"]
    assert not candidate.training.formal_rl_enabled
    assert result["handler_status"] == {"rl": "registered_unverified", "evaluation": "registered_unverified"}
    assert reference.resolved_dict() == original
    assert candidate.training.qmix.max_training_episodes == 80
    assert result["epsilon_at_40k"] == pytest.approx(.05)


@pytest.mark.parametrize("field,value", [
    ("max_training_environment_steps", 500000),
    ("max_training_episodes", 1000),
    ("checkpoint_interval_steps", 50000),
    ("epsilon_decay_steps", 0),
    ("sequence_batch_size", True),
    ("learning_rate", float("nan")),
    ("gamma", 1.2),
    ("epsilon_end", -.1),
    ("epsilon_start", .01),
    ("target_update_interval_updates", "2000"),
    ("max_training_episodes", "80"),
    ("optimizer", "SGD"),
    ("attention_head_count", 3),
    ("replay_capacity_sequences", 1),
])
def test_invalid_qmix_values_are_reported(reference, tmp_path, field, value):
    candidate = load_draft(reference, tmp_path)
    candidate = replace(candidate, training=replace(
        candidate.training, qmix=replace(candidate.training.qmix, **{field: value})
    ))
    result = inspect_comparison(reference, candidate, tmp_path)
    assert result["configuration_status"] == "FAIL"
    assert result["issues"]


@pytest.mark.parametrize("change", ["reward", "environment", "action", "eval_seed", "method"])
def test_comparison_contract_drift_is_rejected(reference, tmp_path, change):
    candidate = load_draft(reference, tmp_path)
    if change == "reward":
        candidate = replace(candidate, reward=replace(candidate.reward, beta_completion_credit=.4))
    elif change == "environment":
        candidate = replace(candidate, environment=replace(candidate.environment, reward_energy_weight=.1))
    elif change == "action":
        candidate = replace(candidate, action=replace(candidate.action, power_levels=(0., .5, 1.)))
    elif change == "eval_seed":
        candidate = replace(candidate, evaluation=replace(candidate.evaluation, evaluation_seeds=(20042,)))
    else:
        candidate = replace(candidate, method_id="ca_gat_mappo")
    assert inspect_comparison(reference, candidate, tmp_path)["configuration_status"] == "FAIL"


@pytest.mark.parametrize("path", ["logs/C/seed43/40k", "../outside", ".agents/forbidden"])
def test_output_alias_and_escape_rejected_without_creation(reference, tmp_path, path):
    candidate = load_draft(reference, tmp_path)
    candidate = replace(candidate, output=replace(candidate.output, logs_dir=path))
    assert inspect_comparison(reference, candidate, tmp_path)["configuration_status"] == "FAIL"
    assert not (tmp_path / ".agents").exists()


def test_existing_output_is_preserved(reference, tmp_path):
    candidate = load_draft(reference, tmp_path)
    output = tmp_path / candidate.output.logs_dir
    output.mkdir(parents=True)
    marker = output / "preserve.txt"
    marker.write_bytes(b"existing run")
    result = inspect_comparison(reference, candidate, tmp_path)
    assert result["configuration_status"] == "FAIL"
    assert marker.read_bytes() == b"existing run"


def test_no_handlers_environment_or_training_invoked(reference, tmp_path):
    candidate = load_draft(reference, tmp_path)
    with patch("src.training.strong_baseline_preflight.build_default_registry") as registry:
        registry.return_value.resolve.return_value = unavailable_handler
        result = inspect_comparison(reference, candidate, tmp_path)
        assert result["configuration_status"] == "PASS"
        assert registry.return_value.resolve.call_count == 2
    assert not (tmp_path / "logs").exists()


def test_registered_handler_does_not_certify_algorithm(reference, tmp_path):
    candidate = load_draft(reference, tmp_path)
    with patch("src.training.strong_baseline_preflight.build_default_registry") as registry:
        registry.return_value.resolve.return_value = lambda *a: pytest.fail("must not execute")
        result = inspect_comparison(reference, candidate, tmp_path)
    assert result["handler_status"]["rl"] == "registered_unverified"
    assert not result["launch_ready"]


@pytest.mark.parametrize("step,expected", [(0, 1.), (16000, .525), (32000, .05), (40000, .05)])
def test_linear_schedule_boundaries(reference, tmp_path, step, expected):
    assert linear_epsilon(load_draft(reference, tmp_path), step) == pytest.approx(expected)


@pytest.mark.parametrize("seed", [True, -1, 45])
def test_first_wave_seed_scope(reference, seed):
    with pytest.raises(ValueError):
        build_qmix_draft(reference, seed)
