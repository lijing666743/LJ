"""Repeated C continuation with real PPO; small fixtures are not experiment scores."""

from dataclasses import replace
from pathlib import Path

import torch

from src.config import load_run_config
from src.models.ca_gat_mappo_checkpoint import load_checkpoint_payload
from src.models.ca_gat_mappo_continuation import assert_state_equal, verify_replay_boundary
from src.models.ca_gat_mappo_trainer import CAGATMAPPOTrainer
from tests.test_ca_gat_mappo_continuation import extension_config


def test_second_c_extension_preserves_lineage_tail_and_uninterrupted_ppo(tmp_path):
    original = load_run_config(Path(__file__).resolve().parents[1]
                               / "experiments/task-allocation/configs/c-seed42-16k-resume.json")

    def preserve_git(config):
        for name in ("git_commit", "git_branch", "git_dirty"):
            object.__setattr__(config, name, getattr(original, name))
        return config

    source = preserve_git(replace(original,
        environment=replace(original.environment, episode_horizon=20),
        training=replace(original.training, mappo=replace(original.training.mappo,
            training_device="cpu", max_training_environment_steps=520,
            max_training_episodes=26, checkpoint_interval_steps=260,
            evaluation_interval_steps=260)),
        output=replace(original.output, logs_dir=str(tmp_path / "s/logs"),
            plots_dir=str(tmp_path / "s/plots"),
            dashboard_logs_dir=str(tmp_path / "s/dashboard_logs"))))
    source.validate()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        first_checkpoint = CAGATMAPPOTrainer(source).train_until_periodic_checkpoint()
        old = CAGATMAPPOTrainer.resume_from_checkpoint(source, first_checkpoint)
        old.train_with_checkpoints()
        old_final = Path(source.artifact_paths()["final_checkpoint"])

        first_config = preserve_git(extension_config(source, tmp_path / "e1", budget=780))
        first = CAGATMAPPOTrainer.resume_from_checkpoint(first_config, first_checkpoint,
            extend_training_budget=True, verify_replay_against=old_final)
        first.train_with_checkpoints()
        first_final = Path(first_config.artifact_paths()["final_checkpoint"])
        second_checkpoint = first_final.parent / "step_520.pt"
        source_state = load_checkpoint_payload(second_checkpoint)
        assert source_state["active_rollout_state"]["rollout_length"] == 8
        assert source_state["config_snapshot"]["_metadata"]["continuation"][
            "replay_reference"]["status"] == "verified_exact"

        second_config = preserve_git(extension_config(first_config, tmp_path / "e2", budget=1040))
        second = CAGATMAPPOTrainer.resume_from_checkpoint(second_config, second_checkpoint,
            extend_training_budget=True, verify_replay_against=first_final)
        assert_state_equal(second._continuation_provenance["previous"],
            source_state["config_snapshot"]["_metadata"]["continuation"], "previous_lineage")
        assert len(second.rollout_buffer) == 8
        actual = second.train_with_checkpoints()
        final = load_checkpoint_payload(second_config.artifact_paths()["final_checkpoint"])
        boundary_path = Path(second_config.artifact_paths()["final_checkpoint"]).parent / "step_780.pt"
        boundary = load_checkpoint_payload(boundary_path)
        verify_replay_boundary(boundary, load_checkpoint_payload(first_final))
        assert boundary["active_rollout_state"]["rollout_length"] == 12
        assert boundary["config_snapshot"]["_metadata"]["continuation"][
            "replay_reference"]["status"] == "verified_exact"
        strict = CAGATMAPPOTrainer.resume_from_checkpoint(second_config, boundary_path)
        assert len(strict.rollout_buffer) == 12
        assert_state_equal(strict._continuation_provenance,
            boundary["config_snapshot"]["_metadata"]["continuation"], "strict_lineage")

        control_config = preserve_git(replace(second_config, output=replace(second_config.output,
            logs_dir=str(tmp_path / "control/logs"), plots_dir=str(tmp_path / "control/plots"),
            dashboard_logs_dir=str(tmp_path / "control/dashboard_logs"))))
        control = CAGATMAPPOTrainer(control_config)
        expected = control.train()
        assert (actual.total_environment_transitions, actual.optimized_transitions,
                actual.unused_final_tail_transitions, actual.ppo_update_count) == (1040, 1024, 16, 4)
        # V1 omits some historical optional diagnostics; compare every persisted diagnostic.
        assert replace(actual, updates=expected.updates) == expected
        assert_state_equal(control._diagnostics_checkpoint_state(), final["diagnostics_state"], "diagnostics")
        for name in ("actor", "critic"):
            assert_state_equal(getattr(control, name).state_dict(), final["model_state"][name], name)
        for name, key in (("actor_optimizer", "actor_adam"), ("critic_optimizer", "critic_adam")):
            assert_state_equal(getattr(control.updater.optimizers, name).state_dict(),
                final["optimizer_state"][key], name)
        assert torch.equal(control.policy_generator.get_state(),
                           final["policy_rng_state"]["generator_state"])
    finally:
        torch.set_num_threads(previous_threads)
