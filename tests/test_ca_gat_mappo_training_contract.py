"""Contract tests for the frozen recurrent PPO training configuration."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.config import ConfigError, load_run_config


class RecurrentPPOTrainingContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mappo = load_run_config().training.mappo

    def test_optimizer_contract_is_explicit(self) -> None:
        self.assertEqual(self.mappo.optimizer, "Adam")
        self.assertEqual(self.mappo.optimizer_topology, "separate_actor_critic")
        self.assertEqual(self.mappo.actor_learning_rate, 3.0e-4)
        self.assertEqual(self.mappo.critic_learning_rate, 3.0e-4)
        self.assertEqual(
            (self.mappo.adam_beta1, self.mappo.adam_beta2),
            (0.9, 0.999),
        )
        self.assertEqual(self.mappo.adam_eps, 1.0e-8)
        self.assertEqual(self.mappo.weight_decay, 0.0)
        self.assertTrue(self.mappo.zero_grad_set_to_none)
        self.assertEqual(self.mappo.gradient_clip_norm, 0.5)
        self.assertEqual(
            self.mappo.gradient_clip_scope,
            "separate_actor_critic_global_norm",
        )

    def test_full_rollout_has_one_deterministic_recurrent_minibatch(self) -> None:
        self.assertEqual(self.mappo.rollout_length_slots, 256)
        self.assertEqual(self.mappo.recurrent_chunk_length_slots, 32)
        self.assertEqual(
            self.mappo.rollout_length_slots
            % self.mappo.recurrent_chunk_length_slots,
            0,
        )
        self.assertEqual(self.mappo.recurrent_chunk_count, 8)
        self.assertEqual(self.mappo.sequence_minibatch_size, 8)
        self.assertEqual(
            self.mappo.recurrent_chunk_count,
            self.mappo.sequence_minibatch_size,
        )
        self.assertEqual(self.mappo.update_epochs, 4)
        self.assertTrue(self.mappo.require_full_rollout)
        self.assertFalse(self.mappo.chunk_shuffle)
        self.assertFalse(self.mappo.timestep_shuffle)
        self.assertFalse(self.mappo.allow_incomplete_minibatch)
        self.assertFalse(self.mappo.sequence_padding)
        self.assertFalse(self.mappo.padded_actor_forward)

    def test_advantage_and_optional_mechanisms_remain_disabled(self) -> None:
        self.assertFalse(self.mappo.advantage_normalization)
        self.assertFalse(self.mappo.value_clipping)
        self.assertFalse(self.mappo.target_kl_enabled)
        self.assertFalse(self.mappo.kl_early_stopping)
        self.assertFalse(self.mappo.learning_rate_schedule_enabled)
        self.assertFalse(self.mappo.entropy_coefficient_schedule_enabled)
        self.assertFalse(self.mappo.gradient_accumulation)
        self.assertFalse(self.mappo.mixed_precision)

    def test_invalid_chunk_geometry_and_shuffle_fail_fast(self) -> None:
        with self.assertRaises(ConfigError):
            load_run_config(
                cli_overrides={"training.mappo.rollout_length_slots": 255}
            )
        with self.assertRaises(ConfigError):
            load_run_config(cli_overrides={"training.mappo.chunk_shuffle": True})

    def test_section4_states_the_execution_contract(self) -> None:
        protocol_path = (
            Path(__file__).resolve().parents[1]
            / "sections"
            / "4_experimental_protocol.md"
        )
        protocol = protocol_path.read_text(encoding="utf-8")
        for required_text in (
            "separate `actor_optimizer` and `critic_optimizer`",
            "actor_optimizer.zero_grad(set_to_none=True)",
            "critic_optimizer.zero_grad(set_to_none=True)",
            "NO SHUFFLE",
            "NO PADDING",
            "NO BURN-IN",
            "advantage_normalization=False",
            "256/32/8",
        ):
            self.assertIn(required_text, protocol)


if __name__ == "__main__":
    unittest.main()
