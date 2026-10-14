"""Contract tests for Checkpoint V1 without checkpoint I/O."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import unittest

from src.config import (
    CHECKPOINT_KIND_FINAL_COMPLETED,
    CHECKPOINT_KIND_PERIODIC_RESUME,
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_V1_ACTIVE_ROLLOUT_FIELDS,
    CHECKPOINT_V1_ALLOWED_PAYLOAD_VALUE_KINDS,
    CHECKPOINT_V1_DIAGNOSTICS_STATE_FIELDS,
    CHECKPOINT_V1_EXACTNESS,
    CHECKPOINT_V1_MODEL_STATE_FIELDS,
    CHECKPOINT_V1_OPTIMIZER_STATE_FIELDS,
    CHECKPOINT_V1_POLICY_RNG_STATE_FIELDS,
    CHECKPOINT_V1_RESUME_BOUNDARY,
    CHECKPOINT_V1_RUNTIME_PROVENANCE_FIELDS,
    CHECKPOINT_V1_SAFE_BOUNDARY_ORDER,
    CHECKPOINT_V1_TOP_LEVEL_FIELDS,
    CHECKPOINT_V1_TRAINING_STATE_FIELDS,
    CHECKPOINT_V1_TRANSITION_FIELDS,
    ConfigError,
    EVALUATION_SNAPSHOT_V1_EXCLUDED_FIELDS,
    EVALUATION_SNAPSHOT_V1_REQUIRED_FIELDS,
    STREAM_IDS,
    compute_mappo_checkpoint_active_rollout_length,
    compute_mappo_periodic_checkpoint_steps,
    compute_mappo_training_accounting,
    load_run_config,
    mappo_checkpoint_kind_at,
    mappo_final_checkpoint_path,
    mappo_periodic_checkpoint_path,
    validate_mappo_checkpoint_resume_compatibility,
)
from src.env.randomness import derive_training_episode_seed


ROOT = Path(__file__).resolve().parents[1]
SECTION_2 = ROOT / "sections" / "2_system_model.md"
SECTION_3 = ROOT / "sections" / "3_methods.md"
SECTION_4 = ROOT / "sections" / "4_experimental_protocol.md"


def _git_blob_sha1(path: Path) -> str:
    data = path.read_text(encoding="utf-8").encode("utf-8")
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


class CAGATMAPPOCheckpointContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_run_config(
            cli_overrides={
                "mode": "rl",
                "method_id": "ca_gat_mappo",
                "training.mappo.training_device": "cpu",
            }
        )
        cls.mappo = cls.config.training.mappo
        cls.protocol = SECTION_4.read_text(encoding="utf-8")
        cls.contract = cls.protocol.split(
            "### 4.11.5 Checkpoint / Exact-Resume Contract V1",
            maxsplit=1,
        )[1].split(
            "## 4.12 Factorized-Action GAT-QMIX numerical training contract",
            maxsplit=1,
        )[0]

    def _validate_compatibility(self, **changes: object) -> None:
        metadata: dict[str, object] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_kind": CHECKPOINT_KIND_PERIODIC_RESUME,
            "method_id": self.config.method_id,
            "git_commit": self.config.git_commit,
            "config_hash": self.config.config_hash,
            "training_device": self.mappo.training_device,
            "cuda_available": False,
        }
        metadata.update(changes)
        validate_mappo_checkpoint_resume_compatibility(
            self.config,
            **metadata,
        )

    def test_01_checkpoint_interval_is_50000(self) -> None:
        self.assertEqual(self.mappo.checkpoint_interval_steps, 50000)

    def test_02_formal_interval_is_episode_aligned(self) -> None:
        self.assertEqual(self.config.environment.episode_horizon, 500)
        self.assertEqual(
            self.mappo.checkpoint_interval_steps
            % self.config.environment.episode_horizon,
            0,
        )

    def test_03_invalid_checkpoint_geometry_fails_fast(self) -> None:
        base = {
            "mode": "rl",
            "method_id": "ca_gat_mappo",
            "training.mappo.training_device": "cpu",
        }
        for interval in (0, 49999, 500000):
            with self.subTest(interval=interval), self.assertRaises(ConfigError):
                load_run_config(
                    cli_overrides={
                        **base,
                        "training.mappo.checkpoint_interval_steps": interval,
                    }
                )

    def test_04_periodic_schedule_is_50k_through_450k(self) -> None:
        expected = tuple(
            range(
                self.mappo.checkpoint_interval_steps,
                self.mappo.max_environment_transitions,
                self.mappo.checkpoint_interval_steps,
            )
        )
        self.assertEqual(
            compute_mappo_periodic_checkpoint_steps(self.config),
            expected,
        )
        self.assertEqual((expected[0], expected[-1]), (50000, 450000))

    def test_05_500k_is_final_completed_not_periodic(self) -> None:
        self.assertEqual(
            mappo_checkpoint_kind_at(self.config, 500000),
            CHECKPOINT_KIND_FINAL_COMPLETED,
        )
        self.assertNotIn(500000, compute_mappo_periodic_checkpoint_steps(self.config))

    def test_06_safe_boundary_order_is_machine_readable_and_documented(self) -> None:
        self.assertEqual(
            CHECKPOINT_V1_SAFE_BOUNDARY_ORDER,
            (
                "terminal_environment_step_complete",
                "terminal_settlement_and_reward_complete",
                "terminal_transition_appended",
                "episode_accounting_updated",
                "due_full_rollout_ppo_update_completed",
                "training_stop_checked",
                "periodic_checkpoint_due_checked",
                "periodic_resume_snapshot_saved",
                "next_episode_reset",
            ),
        )
        fragments = (
            "terminal transition append",
            "episode counters",
            "完成 recurrent PPO update",
            "保存 `PERIODIC_RESUME` snapshot",
            "reset 下一 episode",
        )
        positions = tuple(self.contract.index(fragment) for fragment in fragments)
        self.assertEqual(positions, tuple(sorted(positions)))

    def test_07_pending_full_rollout_is_forbidden(self) -> None:
        self.assertIn("pending full rollout", self.contract)
        self.assertIn("PPO update 尚未执行", self.contract)

    def test_08_partial_rollout_and_structured_payload_are_required(self) -> None:
        self.assertEqual(
            CHECKPOINT_V1_ACTIVE_ROLLOUT_FIELDS,
            ("rollout_length", "rollout_policy_version", "ordered_transitions"),
        )
        self.assertIn("plain structured state", self.contract)
        self.assertIn("每个 periodic checkpoint 必须保存 active partial rollout", self.contract)
        self.assertIn("detached_cpu_tensor", CHECKPOINT_V1_ALLOWED_PAYLOAD_VALUE_KINDS)

    def test_09_50k_partial_rollout_remainder_is_80(self) -> None:
        self.assertEqual(
            compute_mappo_checkpoint_active_rollout_length(self.config, 50000),
            80,
        )

    def test_10_100k_partial_rollout_remainder_is_160(self) -> None:
        self.assertEqual(
            compute_mappo_checkpoint_active_rollout_length(self.config, 100000),
            160,
        )

    def test_11_all_periodic_remainders_are_formula_derived(self) -> None:
        steps = compute_mappo_periodic_checkpoint_steps(self.config)
        expected = tuple(step % self.mappo.rollout_length_slots for step in steps)
        actual = tuple(
            compute_mappo_checkpoint_active_rollout_length(self.config, step)
            for step in steps
        )
        self.assertEqual(actual, expected)
        self.assertEqual(actual, (80, 160, 240, 64, 144, 224, 48, 128, 208))

    def test_12_policy_rng_state_and_device_type_are_required(self) -> None:
        self.assertEqual(
            CHECKPOINT_V1_POLICY_RNG_STATE_FIELDS,
            ("generator_state", "device_type"),
        )
        self.assertEqual(STREAM_IDS["torch_policy_sampling"], 110)
        self.assertIn("torch.Generator.get_state()", self.contract)
        self.assertIn("set_state(saved_state)", self.contract)

    def test_13_master_seed_alone_is_not_exact_resume(self) -> None:
        self.assertIn("只保存 master seed 不足以 exact resume", self.contract)
        self.assertIn("禁止从 master seed 重新初始化 policy generator", self.contract)

    def test_14_environment_rng_state_is_not_saved_at_boundary(self) -> None:
        self.assertIn("environment RNG state 均 **NOT SAVED**", self.contract)
        self.assertEqual(
            derive_training_episode_seed(42, 17),
            derive_training_episode_seed(42, 17),
        )
        self.assertIn("derive_training_episode_seed()", self.contract)

    def test_15_online_hidden_is_not_saved_at_episode_boundary(self) -> None:
        self.assertIn("在线 collector hidden 也 **NOT SAVED**", self.contract)
        self.assertIn("下一 episode 的唯一合法值是 exact zero", self.contract)

    def test_16_partial_rollout_hidden_in_is_required(self) -> None:
        self.assertIn("hidden_in", CHECKPOINT_V1_TRANSITION_FIELDS)
        self.assertIn("partial rollout 中每条历史 transition 的 `hidden_in`", self.contract)

    def test_17_actor_and_critic_state_are_required(self) -> None:
        self.assertEqual(CHECKPOINT_V1_MODEL_STATE_FIELDS, ("actor", "critic"))
        self.assertIn("actor state_dict、critic state_dict", self.contract)

    def test_18_both_adam_states_are_required(self) -> None:
        self.assertEqual(
            CHECKPOINT_V1_OPTIMIZER_STATE_FIELDS,
            ("actor_adam", "critic_adam"),
        )
        self.assertIn("actor Adam state_dict 和 critic Adam state_dict", self.contract)

    def test_19_trainer_counters_are_required(self) -> None:
        required = {
            "policy_version",
            "started_episodes",
            "completed_episodes",
            "next_episode_index",
            "collected_environment_transitions",
            "optimized_transitions",
            "ppo_update_count",
            "unused_final_tail_transitions",
            "training_complete",
            "active_rollout_length",
            "active_rollout_policy_version",
        }
        self.assertEqual(set(CHECKPOINT_V1_TRAINING_STATE_FIELDS), required)

    def test_20_full_existing_diagnostics_are_required(self) -> None:
        self.assertEqual(
            CHECKPOINT_V1_DIAGNOSTICS_STATE_FIELDS,
            ("reward_accumulators", "episode_history", "ppo_update_history"),
        )
        self.assertIn("SAVE FULL EXISTING DIAGNOSTIC HISTORY", self.contract)

    def test_21_checkpoint_schema_is_independent_of_section_version(self) -> None:
        self.assertEqual(self.mappo.checkpoint_schema_version, 1)
        self.assertEqual(CHECKPOINT_SCHEMA_VERSION, 1)
        self.assertEqual(CHECKPOINT_V1_RESUME_BOUNDARY, "EPISODE_BOUNDARY")
        self.assertEqual(CHECKPOINT_V1_EXACTNESS, "SAME_RUNTIME_STATE_EXACT")
        with self.assertRaisesRegex(ConfigError, "schema version mismatch"):
            self._validate_compatibility(schema_version=True)
        self.assertNotEqual(str(CHECKPOINT_SCHEMA_VERSION), self.config.config_version)
        self.assertEqual(len(CHECKPOINT_V1_TOP_LEVEL_FIELDS), 13)
        self.assertEqual(
            set(CHECKPOINT_V1_RUNTIME_PROVENANCE_FIELDS),
            {
                "python_version",
                "torch_version",
                "cuda_runtime_version",
                "device_type",
                "device_name",
                "dtype",
            },
        )

    def test_22_method_id_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ConfigError, "method_id mismatch"):
            self._validate_compatibility(method_id="random")

    def test_23_git_commit_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ConfigError, "git_commit mismatch"):
            self._validate_compatibility(git_commit="different-commit")

    def test_24_full_canonical_config_hash_equality_passes(self) -> None:
        self.assertIsNone(self._validate_compatibility())
        self.assertIn("FULL CANONICAL CONFIG HASH EQUALITY", self.contract)

    def test_25_config_hash_mismatch_fails_without_allowlist(self) -> None:
        changed = replace(
            self.config,
            output=replace(self.config.output, logs_dir="different-logs"),
        )
        with self.assertRaisesRegex(ConfigError, "canonical config_hash mismatch"):
            validate_mappo_checkpoint_resume_compatibility(
                changed,
                schema_version=CHECKPOINT_SCHEMA_VERSION,
                checkpoint_kind=CHECKPOINT_KIND_PERIODIC_RESUME,
                method_id=changed.method_id,
                git_commit=changed.git_commit,
                config_hash=self.config.config_hash,
                training_device=changed.training.mappo.training_device,
                cuda_available=False,
            )

    def test_26_device_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ConfigError, "training device mismatch"):
            self._validate_compatibility(training_device="cuda")

    def test_27_auto_and_cross_device_resume_are_unsupported(self) -> None:
        with self.assertRaises(ConfigError):
            load_run_config(
                cli_overrides={
                    "mode": "rl",
                    "method_id": "ca_gat_mappo",
                    "training.mappo.training_device": "auto",
                }
            )
        self.assertIn("CPU→CPU 或 CUDA→CUDA", self.contract)
        self.assertIn("禁止 CUDA→CPU、CPU→CUDA", self.contract)

    def test_28_atomic_save_sequence_is_explicit(self) -> None:
        for fragment in (
            "same-directory temporary file",
            "binary serialization",
            "`flush`",
            "`os.fsync(file)`",
            "`os.replace(temp, final)`",
        ):
            self.assertIn(fragment, self.contract)
        self.assertIn("不得直接覆盖写 final", self.contract)

    def test_29_periodic_path_reuses_run_id_and_logs_root(self) -> None:
        expected = (
            Path(self.config.output.logs_dir)
            / self.config.run_id
            / "checkpoints"
            / "step_50000.pt"
        )
        self.assertEqual(
            Path(mappo_periodic_checkpoint_path(self.config, 50000)),
            expected,
        )
        self.assertEqual(
            Path(mappo_final_checkpoint_path(self.config)),
            expected.parent / "final.pt",
        )

    def test_30_latest_alias_is_forbidden(self) -> None:
        self.assertIn("不实现 latest symlink/copy", self.contract)

    def test_31_existing_checkpoint_must_not_be_overwritten(self) -> None:
        self.assertIn("同名文件已存在时 fail fast", self.contract)
        self.assertIn("不自动 overwrite", self.contract)

    def test_32_corrupt_checkpoint_has_no_automatic_fallback(self) -> None:
        self.assertIn("损坏或不兼容时 fail fast", self.contract)
        self.assertIn("不自动 fallback 到上一文件", self.contract)

    def test_33_final_collected_count_is_500000(self) -> None:
        accounting = compute_mappo_training_accounting(self.config)
        self.assertEqual(accounting.collected_environment_transitions, 500000)

    def test_34_final_optimized_count_is_499968(self) -> None:
        accounting = compute_mappo_training_accounting(self.config)
        self.assertEqual(accounting.optimized_transitions, 499968)

    def test_35_final_unused_tail_is_32(self) -> None:
        accounting = compute_mappo_training_accounting(self.config)
        self.assertEqual(accounting.unused_final_tail_transitions, 32)

    def test_36_final_ppo_update_count_is_1953(self) -> None:
        accounting = compute_mappo_training_accounting(self.config)
        self.assertEqual(accounting.ppo_update_count, 1953)

    def test_37_final_active_rollout_length_is_zero(self) -> None:
        self.assertEqual(
            compute_mappo_checkpoint_active_rollout_length(self.config, 500000),
            0,
        )
        self.assertIn("active_rollout_length = 0", self.contract)

    def test_38_final_kind_is_distinct_and_has_no_periodic_path(self) -> None:
        self.assertNotEqual(
            CHECKPOINT_KIND_FINAL_COMPLETED,
            CHECKPOINT_KIND_PERIODIC_RESUME,
        )
        with self.assertRaisesRegex(ConfigError, "PERIODIC_RESUME"):
            mappo_periodic_checkpoint_path(self.config, 500000)
        self.assertIn("不生成 `step_500000.pt`", self.contract)

    def test_39_final_completed_cannot_continue_same_run(self) -> None:
        with self.assertRaisesRegex(ConfigError, "training already complete"):
            self._validate_compatibility(
                checkpoint_kind=CHECKPOINT_KIND_FINAL_COMPLETED
            )

    def test_40_evaluation_snapshot_excludes_training_resume_state(self) -> None:
        self.assertIn("actor_state", EVALUATION_SNAPSHOT_V1_REQUIRED_FIELDS)
        self.assertEqual(
            set(EVALUATION_SNAPSHOT_V1_EXCLUDED_FIELDS),
            {
                "critic_state",
                "optimizer_state",
                "policy_rng_state",
                "active_rollout_state",
                "training_state",
                "diagnostics_state",
            },
        )
        self.assertIn("Evaluation 使用 frozen masked argmax", self.contract)

    def test_41_sections_two_and_three_remain_frozen(self) -> None:
        self.assertEqual(
            _git_blob_sha1(SECTION_2),
            "50412164c3b14eea57c1e6d80860fc33a0214009",
        )
        self.assertEqual(
            _git_blob_sha1(SECTION_3),
            "9e004aa1b9aa2891c098f62a7039e0f54f87ce6a",
        )

    def test_42_checkpoint_implementation_is_isolated(self) -> None:
        core_sources = (
            ROOT / "src" / "config.py",
            ROOT / "src" / "models" / "ca_gat_mappo_trainer.py",
            ROOT / "src" / "models" / "ca_gat_mappo_rollout.py",
            ROOT / "src" / "models" / "ca_gat_mappo_update.py",
        )
        production = "\n".join(
            path.read_text(encoding="utf-8") for path in core_sources
        )
        self.assertNotIn("torch.save(", production)
        self.assertNotIn("torch.load(", production)
        checkpoint = (
            ROOT / "src" / "models" / "ca_gat_mappo_checkpoint.py"
        )
        checkpoint_source = checkpoint.read_text(encoding="utf-8")
        self.assertIn("def resume_from_checkpoint", production)
        self.assertIn("def atomic_save_checkpoint", checkpoint_source)
        self.assertIn("def load_checkpoint_payload", checkpoint_source)
        self.assertEqual(
            # Offline evaluation scripts can discuss checkpoints without
            # implementing model persistence. Check the learning modules here.
            tuple((ROOT / "src" / "models").rglob("*checkpoint*.py")),
            (checkpoint,),
        )


if __name__ == "__main__":
    unittest.main()
