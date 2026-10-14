"""Contract tests for the frozen CA-GAT-MAPPO Trainer lifecycle."""

from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from src.config import (
    ConfigError,
    MAPPO_INITIAL_POLICY_VERSION,
    STREAM_IDS,
    compute_mappo_training_accounting,
    load_run_config,
    require_formal_rl_enabled,
    validate_mappo_training_device,
)
from src.env.randomness import derive_training_episode_seed
from src.policies.random_policy import RandomPolicy


ROOT = Path(__file__).resolve().parents[1]
SECTION_2 = ROOT / "sections" / "2_system_model.md"
SECTION_3 = ROOT / "sections" / "3_methods.md"
SECTION_4 = ROOT / "sections" / "4_experimental_protocol.md"


def _rollout_segments(start: int, length: int, horizon: int) -> tuple[tuple[int, int, int], ...]:
    segments: list[tuple[int, int, int]] = []
    cursor = start
    remaining = length
    while remaining:
        episode, slot = divmod(cursor, horizon)
        take = min(remaining, horizon - slot)
        segments.append((episode, slot, slot + take))
        cursor += take
        remaining -= take
    return tuple(segments)


def _replace_mappo(config, **changes):
    return replace(
        config,
        training=replace(
            config.training,
            mappo=replace(config.training.mappo, **changes),
        ),
    )


def _git_blob_sha1(path: Path) -> str:
    data = path.read_text(encoding="utf-8").encode("utf-8")
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


class CAGATMAPPOTrainerLifecycleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_run_config()
        cls.mappo = cls.config.training.mappo
        cls.protocol = SECTION_4.read_text(encoding="utf-8")

    def test_01_episode_and_rollout_boundaries_are_independent(self) -> None:
        self.assertEqual(self.config.environment.episode_horizon, 500)
        self.assertEqual(self.mappo.rollout_length_slots, 256)
        self.assertNotEqual(500, 256)
        self.assertIn("rollout boundary 不得伪造 `terminated` 或 `truncated`", self.protocol)

    def test_02_rollout_zero_geometry_is_episode_zero_slots_zero_to_255(self) -> None:
        self.assertEqual(_rollout_segments(0, 256, 500), ((0, 0, 256),))
        self.assertIn("rollout 0 = episode0[0:256]", self.protocol)

    def test_03_rollout_one_crosses_the_episode_boundary(self) -> None:
        self.assertEqual(
            _rollout_segments(256, 256, 500),
            ((0, 256, 500), (1, 0, 12)),
        )
        self.assertIn(
            "rollout 1 = episode0[256:500] + episode1[0:12]",
            self.protocol,
        )

    def test_04_update_boundary_carries_instead_of_resets_hidden(self) -> None:
        self.assertTrue(self.mappo.carry_hidden_across_update_boundary)
        self.assertIn("CARRY COLLECTOR HIDDEN", self.protocol)
        self.assertIn(
            "policy-update boundary IS NOT recurrent-reset boundary",
            self.protocol,
        )

    def test_05_only_episode_boundary_resets_hidden(self) -> None:
        self.assertIn("只有真实 episode boundary 才将 hidden reset 为全零", self.protocol)
        methods = SECTION_3.read_text(encoding="utf-8")
        self.assertIn("不继承上一 episode 的 recurrent state", methods)

    def test_06_policy_generator_is_not_reset_per_episode(self) -> None:
        self.assertFalse(self.mappo.policy_sampling_reset_each_episode)
        self.assertIn("每个 episode 禁止调用 `reset_sampling_generators()`", self.protocol)

    def test_07_episode_seed_sequence_is_reproducible(self) -> None:
        first = tuple(derive_training_episode_seed(42, index) for index in range(8))
        second = tuple(derive_training_episode_seed(42, index) for index in range(8))
        self.assertEqual(first, second)

    def test_08_episode_indices_produce_distinct_seeds(self) -> None:
        seeds = tuple(derive_training_episode_seed(42, index) for index in range(1000))
        self.assertEqual(len(set(seeds)), 1000)
        self.assertNotEqual(seeds[0], seeds[1])

    def test_09_episode_seed_derivation_does_not_consume_policy_rng(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(123456)
        before = generator.get_state().clone()
        derive_training_episode_seed(42, 17)
        self.assertTrue(torch.equal(generator.get_state(), before))
        self.assertEqual(STREAM_IDS["torch_policy_sampling"], 110)

    def test_10_baseline_and_evaluation_seed_contract_is_unchanged(self) -> None:
        random_policy = RandomPolicy(42)
        self.assertEqual(random_policy.policy_seed, 42)
        self.assertEqual(random_policy.policy_stream_id, 110)
        self.assertEqual(self.config.evaluation.train_seeds, (42, 43, 44, 45, 46))
        self.assertEqual(
            self.config.evaluation.evaluation_seeds,
            (1042, 1043, 1044, 1045, 1046),
        )
        self.assertEqual(
            tuple(STREAM_IDS[name] for name in (
                "reset_mobility",
                "task_arrival",
                "task_workload",
                "channel_fading",
                "csi_error",
                "interference_measurement",
            )),
            (10, 20, 30, 40, 50, 60),
        )

    def test_11_one_rollout_uses_one_policy_version(self) -> None:
        self.assertEqual(MAPPO_INITIAL_POLICY_VERSION, 0)
        self.assertIn("一个 256-transition rollout 的所有 transition", self.protocol)
        self.assertIn("同一 actor parameter version", self.protocol)
        self.assertIn("同一 critic parameter version", self.protocol)

    def test_12_bootstrap_and_finalize_precede_optimizer_step(self) -> None:
        barrier = (
            "transition 255、合法 rollout-tail `bootstrap_value` snapshot 和 "
            "buffer finalize，之后才允许任何 `optimizer.step()`"
        )
        self.assertIn(barrier, self.protocol)

    def test_13_successful_update_increments_policy_version_once(self) -> None:
        self.assertIn("policy_version += 1", self.protocol)
        self.assertIn("失败或未完成的 update 不得增加 version", self.protocol)

    def test_14_training_budget_has_1953_full_rollouts_plus_32(self) -> None:
        self.assertEqual(divmod(500000, 256), (1953, 32))
        self.assertIn("500000 = 1953 * 256 + 32", self.protocol)

    def test_15_collected_transition_count_is_500000(self) -> None:
        accounting = compute_mappo_training_accounting(self.config)
        self.assertEqual(accounting.collected_environment_transitions, 500000)
        self.assertEqual(self.mappo.max_environment_transitions, 500000)

    def test_16_optimized_transition_count_is_499968(self) -> None:
        accounting = compute_mappo_training_accounting(self.config)
        self.assertEqual(accounting.optimized_transitions, 499968)
        self.assertEqual(accounting.optimized_transitions, 1953 * 256)

    def test_17_unused_final_tail_count_is_32(self) -> None:
        accounting = compute_mappo_training_accounting(self.config)
        self.assertEqual(accounting.unused_final_tail_transitions, 32)
        self.assertTrue(self.mappo.discard_final_partial_rollout)

    def test_18_final_tail_is_not_padded(self) -> None:
        self.assertFalse(self.mappo.sequence_padding)
        self.assertFalse(self.mappo.padded_actor_forward)
        self.assertIn("不 padding、不复制", self.protocol)

    def test_19_final_tail_does_not_trigger_partial_ppo_update(self) -> None:
        accounting = compute_mappo_training_accounting(self.config)
        self.assertEqual(accounting.ppo_update_count, 1953)
        self.assertTrue(self.mappo.require_full_rollout)
        self.assertIn("不计算 partial GAE/PPO update", self.protocol)

    def test_20_final_tail_does_not_extend_the_training_budget(self) -> None:
        accounting = compute_mappo_training_accounting(self.config)
        self.assertLess(accounting.unused_final_tail_transitions, 256)
        self.assertEqual(accounting.collected_environment_transitions, 500000)
        self.assertIn("不补采超过 500000 的 transition", self.protocol)

    def test_21_formal_training_device_is_explicit_cuda(self) -> None:
        self.assertEqual(self.mappo.training_device, "cuda")
        self.assertEqual(validate_mappo_training_device(self.config, cuda_available=True), "cuda")

    def test_22_unavailable_cuda_fails_fast_without_cpu_fallback(self) -> None:
        with self.assertRaisesRegex(ConfigError, "silent CPU fallback is forbidden"):
            validate_mappo_training_device(self.config, cuda_available=False)
        cpu_config = _replace_mappo(self.config, training_device="cpu")
        self.assertEqual(validate_mappo_training_device(cpu_config, cuda_available=False), "cpu")

    def test_23_auto_device_is_not_a_valid_configuration(self) -> None:
        with self.assertRaisesRegex(ConfigError, "training_device"):
            load_run_config(
                cli_overrides={"training.mappo.training_device": "auto"}
            )
        self.assertNotIn("auto", {"cpu", "cuda"})

    def test_24_formal_rl_gate_blocks_before_training_work(self) -> None:
        self.assertFalse(self.config.training.formal_rl_enabled)
        with self.assertRaisesRegex(ConfigError, "before environment reset"):
            require_formal_rl_enabled(self.config)
        enabled = replace(
            self.config,
            training=replace(self.config.training, formal_rl_enabled=True),
        )
        self.assertIsNone(require_formal_rl_enabled(enabled))

    def test_25_section_two_and_three_remain_at_frozen_content(self) -> None:
        self.assertEqual(
            _git_blob_sha1(SECTION_2),
            "50412164c3b14eea57c1e6d80860fc33a0214009",
        )
        self.assertEqual(
            _git_blob_sha1(SECTION_3),
            "9e004aa1b9aa2891c098f62a7039e0f54f87ce6a",
        )


if __name__ == "__main__":
    unittest.main()
