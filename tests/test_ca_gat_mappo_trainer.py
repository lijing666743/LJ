"""Production Trainer Gate tests for CA-GAT-MAPPO."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import math
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.config import ConfigError, RunConfig
from src.env.environment import StepResult, U2UMECEnvironment
from src.env.randomness import derive_training_episode_seed
from src.models.ca_gat_mappo_gae import compute_rollout_gae
from src.models.ca_gat_mappo_trainer import (
    CAGATMAPPOTrainer,
    CAGATMAPPOTrainerError,
    build_training_episode_config,
)
from src.models.ca_gat_mappo_update import (
    CAGATMAPPORecurrentPPOUpdater,
    RecurrentPPOEpochDiagnostics,
    RecurrentPPOUpdateOutput,
)


def make_trainer_config(
    *,
    horizon: int = 500,
    max_steps: int = 1,
    max_episodes: int = 1000,
    formal_rl_enabled: bool = True,
    device: str = "cpu",
) -> RunConfig:
    base = RunConfig()
    environment = replace(
        base.environment,
        episode_horizon=horizon,
        uav_count=1,
        arrival_probabilities=(0.25,),
        profile_assignment=("Balanced",),
        profile_perturbations=((0.0, 0.0, 0.0, 0.0),),
        building_layout=(),
        candidate_neighbor_radius_m=2_000.0,
        velocity_std_mps=(0.0, 0.0),
        shadowing_std_db=0.0,
        csi_error_std_db=0.0,
        fixed_csi_aoi_slots=1,
    )
    mappo = replace(
        base.training.mappo,
        encoder_hidden_dimension=8,
        attention_head_count=1,
        gru_hidden_dimension=8,
        training_device=device,
        max_training_episodes=max_episodes,
        max_training_environment_steps=max_steps,
    )
    config = replace(
        base,
        environment=environment,
        training=replace(
            base.training,
            mappo=mappo,
            formal_rl_enabled=formal_rl_enabled,
        ),
    )
    config.validate()
    return config


def update_output(offset: float = 0.0) -> RecurrentPPOUpdateOutput:
    diagnostics = tuple(
        RecurrentPPOEpochDiagnostics(
            epoch_index=index,
            actor_loss=1.0 + offset + index,
            critic_loss=2.0 + offset + index,
            entropy_mean=0.5 + offset + index,
            total_loss=3.0 + offset + index,
            ratio_mean=1.0 + 0.01 * index,
            actor_grad_norm_before_clip=4.0 + offset + index,
            critic_grad_norm_before_clip=5.0 + offset + index,
            clip_max_norm=0.5,
        )
        for index in range(4)
    )
    return RecurrentPPOUpdateOutput(
        epoch_diagnostics=diagnostics,
        chunk_count=8,
        chunk_length=32,
        valid_transition_count=256,
        old_policy_snapshot_preserved=True,
    )


class RecordingEnvironment:
    def __init__(self, config: RunConfig, owner: "RecordingFactory") -> None:
        self.delegate = U2UMECEnvironment(config)
        self.owner = owner
        self.step_inputs = []
        self.step_results: list[StepResult] = []
        self.legal_inputs: list[bool] = []

    def reset(self):
        self.owner.events.append("reset")
        return self.delegate.reset()

    def step(self, proposals):
        copied = tuple(proposals)
        observations = self.delegate.current_observations
        assert observations is not None
        self.legal_inputs.append(
            all(
                observation.action_masks.is_legal(proposal)
                for observation, proposal in zip(observations, copied)
            )
        )
        self.step_inputs.append(copied)
        self.owner.events.append("step")
        result = self.delegate.step(copied)
        self.step_results.append(result)
        return result


class RecordingFactory:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = [] if events is None else events
        self.configs: list[RunConfig] = []
        self.environments: list[RecordingEnvironment] = []

    def __call__(self, config: RunConfig) -> RecordingEnvironment:
        self.configs.append(config)
        environment = RecordingEnvironment(config, self)
        self.environments.append(environment)
        return environment

    @property
    def all_results(self) -> tuple[StepResult, ...]:
        return tuple(
            result
            for environment in self.environments
            for result in environment.step_results
        )


class CapturingUpdater:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events
        self.calls = 0
        self.transitions = ()
        self.finalized = False

    def update(self, buffer):
        self.calls += 1
        self.finalized = buffer.finalized
        self.transitions = tuple(
            buffer.transition_at(index) for index in range(len(buffer))
        )
        if self.events is not None:
            self.events.append("update")
        return update_output(float(self.calls - 1))


class ProductionCollectionLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.events: list[str] = []
        cls.config = make_trainer_config(max_steps=257)
        cls.factory = RecordingFactory(cls.events)
        cls.updater = CapturingUpdater(cls.events)
        cls.trainer = CAGATMAPPOTrainer(
            cls.config,
            environment_factory=cls.factory,
            updater_factory=lambda *_: cls.updater,
        )
        cls.hidden_inputs: list[torch.Tensor] = []
        cls.generator_ids: list[int] = []
        cls.sample_versions: list[int] = []
        cls.reset_mock = Mock(
            side_effect=AssertionError("policy generator must not reset")
        )
        cls.trainer.action_distribution.reset_sampling_generators = (
            cls.reset_mock
        )
        original_sample = cls.trainer.action_distribution.sample_actions

        def sample(*args, **kwargs):
            cls.events.append("sample")
            cls.hidden_inputs.append(args[2].detach().cpu().clone())
            cls.generator_ids.append(id(kwargs["generator"]))
            cls.sample_versions.append(cls.trainer.policy_version)
            return original_sample(*args, **kwargs)

        cls.trainer.action_distribution.sample_actions = sample
        original_append = cls.trainer.rollout_buffer.append_step

        def append(**kwargs):
            cls.events.append("append")
            return original_append(**kwargs)

        cls.trainer.rollout_buffer.append_step = append
        cls.initial_generator_state = (
            cls.trainer.policy_generator.get_state().clone()
        )
        cls.result = cls.trainer.train()

    def test_01_exact_transition_update_and_tail_accounting(self) -> None:
        self.assertEqual(self.result.total_environment_transitions, 257)
        self.assertEqual(self.result.optimized_transitions, 256)
        self.assertEqual(self.result.unused_final_tail_transitions, 1)
        self.assertEqual(self.result.ppo_update_count, 1)
        self.assertEqual(len(self.trainer.rollout_buffer), 0)
        self.assertEqual(self.result.collected_environment_transitions, 257)
        self.assertEqual(self.result.completed_episodes, 0)

    def test_02_policy_version_barrier_increments_once(self) -> None:
        self.assertEqual(self.result.initial_policy_version, 0)
        self.assertEqual(self.result.final_policy_version, 1)
        update = self.result.updates[0]
        self.assertEqual(update.rollout_policy_version, 0)
        self.assertEqual(update.policy_version_after_update, 1)
        self.assertEqual(
            tuple(self.sample_versions[:256]), (0,) * 256
        )
        self.assertEqual(self.sample_versions[256], 1)

    def test_03_updater_receives_one_finalized_full_rollout(self) -> None:
        self.assertEqual(self.updater.calls, 1)
        self.assertTrue(self.updater.finalized)
        self.assertEqual(len(self.updater.transitions), 256)
        self.assertEqual(
            tuple(item.slot for item in self.updater.transitions),
            tuple(range(256)),
        )

    def test_04_update_boundary_carries_collector_hidden(self) -> None:
        self.assertTrue(torch.count_nonzero(self.hidden_inputs[255]).item())
        self.assertTrue(torch.count_nonzero(self.hidden_inputs[256]).item())
        self.assertFalse(
            torch.equal(
                self.hidden_inputs[256],
                torch.zeros_like(self.hidden_inputs[256]),
            )
        )

    def test_05_policy_generator_is_one_continuous_object(self) -> None:
        self.assertEqual(len(set(self.generator_ids)), 1)
        self.assertNotEqual(
            self.initial_generator_state.tolist(),
            self.trainer.policy_generator.get_state().tolist(),
        )
        self.reset_mock.assert_not_called()
        self.assertEqual(self.trainer.action_distribution._generators, {})
        self.assertEqual(self.trainer.action_distribution.policy_stream_id, 110)

    def test_06_order_is_sample_step_append_then_update(self) -> None:
        events = tuple(item for item in self.events if item != "reset")
        self.assertEqual(events[:3], ("sample", "step", "append"))
        self.assertEqual(events[255 * 3 : 256 * 3], ("sample", "step", "append"))
        self.assertEqual(events[256 * 3], "update")
        self.assertEqual(
            events[256 * 3 + 1 :],
            ("sample", "step", "append"),
        )

    def test_07_real_masks_and_proposals_reach_real_environment(self) -> None:
        environment = self.factory.environments[0]
        self.assertEqual(len(environment.step_inputs), 257)
        self.assertTrue(all(environment.legal_inputs))
        self.assertTrue(
            all(
                len(proposals) == self.config.environment.uav_count
                for proposals in environment.step_inputs
            )
        )

    def test_08_real_executor_metadata_is_stored(self) -> None:
        first_transition = self.updater.transitions[0]
        first_step = self.factory.all_results[0]
        self.assertEqual(
            first_transition.executed_action_summary,
            tuple(first_step.info["executed"]),
        )
        self.assertIn(
            "rejection",
            first_transition.rejection_or_downgrade_summary,
        )
        self.assertIn(
            "downgrade",
            first_transition.rejection_or_downgrade_summary,
        )

    def test_09_step_result_reward_is_the_rollout_reward(self) -> None:
        for transition, step in zip(
            self.updater.transitions,
            self.factory.all_results[:256],
        ):
            self.assertEqual(
                transition.reward.item(),
                torch.tensor(step.reward, dtype=torch.float32).item(),
            )

    def test_10_non_boundary_bootstrap_uses_real_next_state(self) -> None:
        self.assertTrue(
            all(item.bootstrap_allowed for item in self.updater.transitions)
        )
        self.assertTrue(
            all(
                item.bootstrap_value is not None
                and torch.isfinite(item.bootstrap_value)
                for item in self.updater.transitions
            )
        )

    def test_11_reward_sums_and_means_are_auditable(self) -> None:
        rewards = [item.reward for item in self.factory.all_results]
        self.assertAlmostEqual(self.result.reward.reward.total, math.fsum(rewards))
        self.assertAlmostEqual(
            self.result.reward.reward.mean,
            math.fsum(rewards) / len(rewards),
        )
        completion = [
            item.info["reward"]["completion_component"]
            for item in self.factory.all_results
        ]
        self.assertAlmostEqual(
            self.result.reward.completion_component.total,
            math.fsum(completion),
        )

    def test_12_ppo_diagnostics_aggregate_four_epochs(self) -> None:
        self.assertEqual(self.result.ppo.actor_loss.count, 4)
        self.assertAlmostEqual(self.result.ppo.actor_loss.total, 10.0)
        self.assertAlmostEqual(self.result.ppo.ratio.mean, 1.015)
        self.assertEqual(
            self.result.ppo.critic_grad_norm_before_clip.count, 4
        )

    def test_13_models_are_explicit_cpu_float32(self) -> None:
        for module in (self.trainer.actor, self.trainer.critic):
            self.assertTrue(
                all(
                    parameter.device.type == "cpu"
                    and parameter.dtype == torch.float32
                    for parameter in module.parameters()
                )
            )

    def test_14_partial_budget_stops_mid_episode_without_extra_reset(self) -> None:
        self.assertEqual(self.result.completed_episode_count, 0)
        self.assertEqual(self.result.started_episode_count, 1)
        self.assertFalse(self.result.episodes[0].completed_boundary)
        self.assertEqual(len(self.factory.environments), 1)

    def test_15_result_is_immutable_and_trainer_is_one_shot(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            self.result.total_environment_transitions = 0
        with self.assertRaisesRegex(
            CAGATMAPPOTrainerError, "exactly one training run"
        ):
            self.trainer.train()


class EpisodeBoundaryLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = make_trainer_config(horizon=3, max_steps=4)
        cls.factory = RecordingFactory()
        cls.updater = CapturingUpdater()
        cls.progress_lines: list[str] = []
        cls.trainer = CAGATMAPPOTrainer(
            cls.config,
            environment_factory=cls.factory,
            updater_factory=lambda *_: cls.updater,
            progress_logger=cls.progress_lines.append,
        )
        cls.hidden_inputs: list[torch.Tensor] = []
        cls.generator_ids: list[int] = []
        original_sample = cls.trainer.action_distribution.sample_actions

        def sample(*args, **kwargs):
            cls.hidden_inputs.append(args[2].detach().cpu().clone())
            cls.generator_ids.append(id(kwargs["generator"]))
            return original_sample(*args, **kwargs)

        cls.trainer.action_distribution.sample_actions = sample
        cls.trainer.rollout_buffer.clear = lambda: None
        cls.result = cls.trainer.train()
        cls.transitions = tuple(
            cls.trainer.rollout_buffer.transition_at(index)
            for index in range(len(cls.trainer.rollout_buffer))
        )
        cls.chunk = cls.trainer.rollout_buffer.finalize()
        cls.gae = compute_rollout_gae(cls.chunk, cls.config)

    def test_16_hidden_resets_only_at_true_episode_boundary(self) -> None:
        self.assertTrue(torch.equal(
            self.hidden_inputs[0], torch.zeros_like(self.hidden_inputs[0])
        ))
        self.assertTrue(torch.count_nonzero(self.hidden_inputs[1]).item())
        self.assertTrue(torch.equal(
            self.hidden_inputs[3], torch.zeros_like(self.hidden_inputs[3])
        ))

    def test_17_boundary_transition_has_no_fake_bootstrap(self) -> None:
        self.assertEqual(
            tuple(item.slot for item in self.transitions), (0, 1, 2, 0)
        )
        boundary = self.transitions[2]
        self.assertTrue(boundary.truncated)
        self.assertTrue(boundary.episode_boundary)
        self.assertFalse(boundary.bootstrap_allowed)
        self.assertIsNone(boundary.bootstrap_value)
        self.assertTrue(self.transitions[3].episode_start)

    def test_18_gae_recursion_does_not_cross_episode_boundary(self) -> None:
        self.assertFalse(self.gae.bootstrap_mask[2].item())
        self.assertAlmostEqual(
            self.gae.advantage[2].item(),
            self.gae.td_residual[2].item(),
            places=6,
        )

    def test_19_episode_seed_changes_without_base_config_mutation(self) -> None:
        self.assertEqual(len(self.factory.configs), 2)
        expected = tuple(
            derive_training_episode_seed(self.config.seed, index)
            for index in range(2)
        )
        self.assertEqual(self.result.episode_seeds, expected)
        self.assertEqual(
            tuple(item.seed for item in self.factory.configs), expected
        )
        for episode_config in self.factory.configs:
            self.assertEqual(
                replace(episode_config, seed=self.config.seed), self.config
            )

    def test_20_policy_stream_continues_across_episode_reset(self) -> None:
        self.assertEqual(len(set(self.generator_ids)), 1)
        self.assertEqual(self.updater.calls, 0)
        self.assertEqual(self.result.unused_final_tail_transitions, 4)

    def test_21_episode_diagnostics_include_complete_and_partial(self) -> None:
        self.assertEqual(self.result.started_episode_count, 2)
        self.assertEqual(self.result.completed_episode_count, 1)
        self.assertEqual(
            tuple(item.transition_count for item in self.result.episodes),
            (3, 1),
        )
        self.assertEqual(
            tuple(item.completed_boundary for item in self.result.episodes),
            (True, False),
        )

    def test_22_progress_logger_reports_only_completed_episodes(self) -> None:
        self.assertEqual(len(self.progress_lines), 1)
        line = self.progress_lines[0]
        episode = self.result.episodes[0]
        self.assertEqual(
            line,
            "CA-GAT-MAPPO progress: "
            "episode=1/1000, collected_transitions=3/4, ppo_updates=0, "
            f"episode_reward={episode.reward.reward.total:.6g}, "
            "completion_count="
            f"{episode.reward.completed_task_count.total:.6g}, "
            "expiration_count="
            f"{episode.reward.expired_task_count.total:.6g}, "
            "actor_loss=n/a, critic_loss=n/a, entropy=n/a, device=cpu",
        )


class StartupAndFailureGuardTests(unittest.TestCase):
    def test_progress_logger_reports_latest_completed_update_means(self) -> None:
        progress_lines: list[str] = []
        updater = CapturingUpdater()
        trainer = CAGATMAPPOTrainer(
            make_trainer_config(
                horizon=256, max_steps=256, max_episodes=1
            ),
            updater_factory=lambda *_: updater,
            progress_logger=progress_lines.append,
        )
        result = trainer.train()

        self.assertEqual(result.completed_episode_count, 1)
        self.assertEqual(result.ppo_update_count, 1)
        self.assertEqual(len(progress_lines), 1)
        line = progress_lines[0]
        self.assertIn(
            "episode=1/1, collected_transitions=256/256, ppo_updates=1",
            line,
        )
        self.assertIn("actor_loss=2.5", line)
        self.assertIn("critic_loss=3.5", line)
        self.assertIn("entropy=2", line)
        self.assertTrue(line.endswith("device=cpu"))

    def test_22_formal_gate_precedes_model_and_environment_work(self) -> None:
        config = make_trainer_config(formal_rl_enabled=False)
        factory = Mock()
        with patch(
            "src.models.ca_gat_mappo_trainer.CAGATMAPPOActor"
        ) as actor:
            with self.assertRaisesRegex(ConfigError, "before environment reset"):
                CAGATMAPPOTrainer(config, environment_factory=factory)
        actor.assert_not_called()
        factory.assert_not_called()

    def test_23_unavailable_cuda_fails_without_cpu_fallback(self) -> None:
        config = make_trainer_config(device="cuda")
        with patch("torch.cuda.is_available", return_value=False):
            with patch(
                "src.models.ca_gat_mappo_trainer.CAGATMAPPOActor"
            ) as actor:
                with self.assertRaisesRegex(
                    ConfigError, "silent CPU fallback is forbidden"
                ):
                    CAGATMAPPOTrainer(config)
        actor.assert_not_called()

    def test_24_auto_device_is_rejected(self) -> None:
        config = make_trainer_config()
        config = replace(
            config,
            training=replace(
                config.training,
                mappo=replace(
                    config.training.mappo,
                    training_device="auto",
                ),
            ),
        )
        with self.assertRaisesRegex(ConfigError, "training_device"):
            CAGATMAPPOTrainer(config)

    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA is unavailable"
    )
    def test_25_explicit_cuda_trainer_collects_on_cuda(self) -> None:
        config = make_trainer_config(device="cuda")
        updater = CapturingUpdater()
        trainer = CAGATMAPPOTrainer(
            config,
            updater_factory=lambda *_: updater,
        )
        result = trainer.train()
        self.assertEqual(result.total_environment_transitions, 1)
        self.assertEqual(result.unused_final_tail_transitions, 1)
        self.assertEqual(trainer.policy_generator.device.type, "cuda")
        for module in (trainer.actor, trainer.critic):
            self.assertTrue(
                all(
                    parameter.device.type == "cuda"
                    for parameter in module.parameters()
                )
            )

    def test_24_episode_config_uses_frozen_seed_helper_only(self) -> None:
        config = make_trainer_config()
        object.__setattr__(
            config, "source_config_path", "provenance-must-survive.yaml"
        )
        derived = build_training_episode_config(config, 7)
        self.assertEqual(
            derived.seed,
            derive_training_episode_seed(config.seed, 7),
        )
        self.assertEqual(
            derived.source_config_path, config.source_config_path
        )
        self.assertEqual(
            replace(derived, seed=config.seed).resolved_dict(),
            config.resolved_dict(),
        )
        self.assertEqual(config.seed, RunConfig().seed)

    def test_25_episode_seed_does_not_consume_global_numpy_rng(self) -> None:
        config = make_trainer_config()
        np.random.seed(20260816)
        before = np.random.get_state()
        build_training_episode_config(config, 9)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_25_default_wiring_reuses_existing_recurrent_updater(self) -> None:
        trainer = CAGATMAPPOTrainer(make_trainer_config())
        self.assertIsInstance(
            trainer.updater, CAGATMAPPORecurrentPPOUpdater
        )
        self.assertIs(
            trainer.updater.action_distribution,
            trainer.action_distribution,
        )

    def test_26_episode_budget_stops_without_extra_episode(self) -> None:
        config = make_trainer_config(
            horizon=3, max_steps=99, max_episodes=1
        )
        factory = RecordingFactory()
        updater = CapturingUpdater()
        result = CAGATMAPPOTrainer(
            config,
            environment_factory=factory,
            updater_factory=lambda *_: updater,
        ).train()
        self.assertEqual(result.total_environment_transitions, 3)
        self.assertEqual(result.completed_episode_count, 1)
        self.assertEqual(len(factory.environments), 1)
        self.assertEqual(updater.calls, 0)

    def test_27_failed_update_does_not_increment_policy_version(self) -> None:
        class FailingUpdater:
            def update(self, _buffer):
                raise RuntimeError("synthetic updater failure")

        trainer = CAGATMAPPOTrainer(
            ProductionCollectionLifecycleTests.config,
            updater_factory=lambda *_: FailingUpdater(),
        )
        for transition in (
            ProductionCollectionLifecycleTests.updater.transitions
        ):
            trainer.rollout_buffer.append(transition)
        trainer._rollout_policy_version = trainer.policy_version
        with self.assertRaisesRegex(RuntimeError, "synthetic updater failure"):
            trainer._perform_update([])
        self.assertEqual(trainer.policy_version, 0)

    def test_28_updater_cannot_consume_policy_sampling_generator(self) -> None:
        class ConsumingUpdater:
            trainer = None

            def update(self, _buffer):
                torch.rand(
                    (),
                    generator=self.trainer.policy_generator,
                )
                return update_output()

        updater = ConsumingUpdater()
        trainer = CAGATMAPPOTrainer(
            ProductionCollectionLifecycleTests.config,
            updater_factory=lambda *_: updater,
        )
        updater.trainer = trainer
        for transition in (
            ProductionCollectionLifecycleTests.updater.transitions
        ):
            trainer.rollout_buffer.append(transition)
        trainer._rollout_policy_version = trainer.policy_version
        with self.assertRaisesRegex(
            CAGATMAPPOTrainerError,
            "consumed the policy sampling stream",
        ):
            trainer._perform_update([])
        self.assertEqual(trainer.policy_version, 0)

    def test_29_rollout_policy_version_change_fails_fast(self) -> None:
        config = make_trainer_config(max_steps=2)
        updater = CapturingUpdater()
        trainer = CAGATMAPPOTrainer(
            config,
            updater_factory=lambda *_: updater,
        )
        original_sample = trainer.action_distribution.sample_actions
        sample_count = 0

        def sample(*args, **kwargs):
            nonlocal sample_count
            output = original_sample(*args, **kwargs)
            sample_count += 1
            if sample_count == 2:
                trainer.policy_version += 1
            return output

        trainer.action_distribution.sample_actions = sample
        with self.assertRaisesRegex(
            CAGATMAPPOTrainerError,
            "changed during transition collection",
        ):
            trainer.train()
        self.assertEqual(len(trainer.rollout_buffer), 1)
        self.assertEqual(updater.calls, 0)


if __name__ == "__main__":
    unittest.main()
