"""Checkpoint V1 implementation and same-runtime exact-resume tests."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.config import (
    CHECKPOINT_KIND_FINAL_COMPLETED,
    CHECKPOINT_KIND_PERIODIC_RESUME,
    CHECKPOINT_SCHEMA_VERSION,
    ChannelAblationMode,
    RunConfig,
    RouteDecoderMode,
)
from src.models.ca_gat_mappo import (
    ActorObservationTensorizer,
    CAGATMAPPOActor,
    MAPPOCentralizedCritic,
)
from src.models.ca_gat_mappo_actions import (
    CAGATMAPPOActionDistribution,
    SequentialActionMaskBatch,
)
from src.models.ca_gat_mappo_checkpoint import (
    CheckpointError,
    atomic_save_checkpoint,
    build_checkpoint_payload,
    load_checkpoint_payload,
    restore_active_rollout,
    restore_model_optimizer_and_rng,
    restore_policy_rng,
    serialize_active_rollout,
    serialize_rollout_transition,
    validate_checkpoint_payload,
    validate_periodic_checkpoint_compatibility,
)
from src.models.ca_gat_mappo_rollout import CAGATMAPPORolloutBuffer
from src.models.ca_gat_mappo_trainer import (
    CAGATMAPPOTrainer,
    CAGATMAPPOTrainingResult,
)
from src.models.ca_gat_mappo_update import (
    CAGATMAPPORecurrentPPOUpdater,
    RecurrentPPOEpochDiagnostics,
    RecurrentPPOUpdateOutput,
    build_ca_gat_mappo_optimizers,
)
from tests.test_ca_gat_mappo_rollout_buffer import RolloutCollectorFixture
from tests.test_ca_gat_mappo_update import (
    align_old_policy_snapshots,
    make_seed_transition,
    make_synthetic_buffer,
)


def make_checkpoint_config(
    root: Path,
    *,
    horizon: int = 4,
    interval: int = 4,
    budget: int = 8,
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
        max_training_episodes=max(2, budget // horizon),
        max_training_environment_steps=budget,
        checkpoint_interval_steps=interval,
    )
    config = replace(
        base,
        mode="rl",
        method_id="ca_gat_mappo",
        environment=environment,
        training=replace(
            base.training,
            mappo=mappo,
            formal_rl_enabled=True,
        ),
        output=replace(base.output, logs_dir=str(root / "logs")),
    )
    config.validate()
    return config


def fixed_update_output() -> RecurrentPPOUpdateOutput:
    return RecurrentPPOUpdateOutput(
        epoch_diagnostics=tuple(
            RecurrentPPOEpochDiagnostics(
                epoch_index=index,
                actor_loss=1.0 + index,
                critic_loss=2.0 + index,
                entropy_mean=0.5 + index,
                total_loss=3.0 + index,
                ratio_mean=1.0 + 0.01 * index,
                actor_grad_norm_before_clip=4.0 + index,
                critic_grad_norm_before_clip=5.0 + index,
                clip_max_norm=0.5,
            )
            for index in range(4)
        ),
        chunk_count=8,
        chunk_length=32,
        valid_transition_count=256,
        old_policy_snapshot_preserved=True,
    )


class DeterministicAdamUpdater:
    """Fast test updater with real Adam state and no additional RNG."""

    def __init__(self, actor, critic, config) -> None:
        self.optimizers = build_ca_gat_mappo_optimizers(actor, critic, config)

    def update(self, buffer) -> RecurrentPPOUpdateOutput:
        if not buffer.full or not buffer.finalized:
            raise AssertionError("fixture updater requires one finalized rollout")
        actor = self.optimizers.actor_optimizer
        critic = self.optimizers.critic_optimizer
        actor.zero_grad(set_to_none=True)
        critic.zero_grad(set_to_none=True)
        actor_loss = sum(
            parameter.square().sum()
            for parameter in self.optimizers.actor_parameters
        ) * 1.0e-7
        critic_loss = sum(
            parameter.square().sum()
            for parameter in self.optimizers.critic_parameters
        ) * 1.0e-7
        actor_loss.backward()
        critic_loss.backward()
        actor.step()
        critic.step()
        return fixed_update_output()


def deterministic_updater_factory(actor, critic, config, _distribution):
    return DeterministicAdamUpdater(actor, critic, config)


def collect_partial_buffer(
    config: RunConfig,
    length: int,
) -> CAGATMAPPORolloutBuffer:
    collector = RolloutCollectorFixture(config)
    buffer = CAGATMAPPORolloutBuffer(config)
    for _ in range(length):
        buffer.append(collector.collect().transition)
    return buffer


def training_state(
    config: RunConfig,
    kind: str,
    active_length: int,
) -> dict[str, object]:
    if kind == CHECKPOINT_KIND_PERIODIC_RESUME:
        collected = config.training.mappo.checkpoint_interval_steps
        unused = 0
        complete = False
    else:
        collected = config.training.mappo.max_environment_transitions
        unused = collected % config.training.mappo.rollout_length_slots
        complete = True
    optimized = collected - active_length - unused
    updates = optimized // config.training.mappo.rollout_length_slots
    completed = collected // config.environment.episode_horizon
    return {
        "policy_version": updates,
        "started_episodes": completed,
        "completed_episodes": completed,
        "next_episode_index": completed,
        "collected_environment_transitions": collected,
        "optimized_transitions": optimized,
        "ppo_update_count": updates,
        "unused_final_tail_transitions": unused,
        "training_complete": complete,
        "active_rollout_length": active_length,
        "active_rollout_policy_version": updates if active_length else None,
    }


def empty_diagnostics() -> dict[str, object]:
    return {
        "reward_accumulators": {},
        "episode_history": [],
        "ppo_update_history": [],
    }


def make_payload(
    config: RunConfig,
    kind: str,
    *,
    actor: CAGATMAPPOActor | None = None,
    critic: MAPPOCentralizedCritic | None = None,
    updater: CAGATMAPPORecurrentPPOUpdater | None = None,
) -> tuple[dict[str, object], CAGATMAPPOActor, MAPPOCentralizedCritic, object]:
    device = torch.device(config.training.mappo.training_device)
    actor = CAGATMAPPOActor(config).to(device) if actor is None else actor
    critic = MAPPOCentralizedCritic(config).to(device) if critic is None else critic
    distribution = CAGATMAPPOActionDistribution(actor, config)
    updater = (
        CAGATMAPPORecurrentPPOUpdater(
            actor, critic, config, action_distribution=distribution
        )
        if updater is None
        else updater
    )
    generator = torch.Generator(device=device.type)
    generator.manual_seed(distribution.policy_seed)
    if kind == CHECKPOINT_KIND_PERIODIC_RESUME:
        active_length = (
            config.training.mappo.checkpoint_interval_steps
            % config.training.mappo.rollout_length_slots
        )
        buffer = collect_partial_buffer(config, active_length)
        version = training_state(config, kind, active_length)["policy_version"]
    else:
        active_length = 0
        buffer = CAGATMAPPORolloutBuffer(config)
        version = None
    active = serialize_active_rollout(buffer, version, kind)
    payload = build_checkpoint_payload(
        config=config,
        checkpoint_kind=kind,
        actor=actor,
        critic=critic,
        optimizers=updater.optimizers,
        policy_generator=generator,
        training_state=training_state(config, kind, active_length),
        active_rollout_state=active,
        diagnostics_state=empty_diagnostics(),
        dtype=torch.float32,
    )
    return payload, actor, critic, updater


def assert_nested_equal(test: unittest.TestCase, left, right) -> None:
    if isinstance(left, torch.Tensor):
        test.assertIsInstance(right, torch.Tensor)
        test.assertEqual(left.dtype, right.dtype)
        test.assertEqual(left.shape, right.shape)
        test.assertTrue(torch.equal(left.cpu(), right.cpu()))
    elif isinstance(left, dict):
        test.assertEqual(tuple(left), tuple(right))
        for key in left:
            assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(type(left), type(right))
        test.assertEqual(len(left), len(right))
        for first, second in zip(left, right):
            assert_nested_equal(test, first, second)
    else:
        test.assertEqual(left, right)


class StructuredCheckpointTests(unittest.TestCase):
    def test_route_diagnostic_flag_does_not_enter_training_state_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = make_checkpoint_config(root / "control")
            treatment = replace(
                control,
                training=replace(
                    control.training,
                    mappo=replace(
                        control.training.mappo,
                        route_diagnostic_samples_enabled=True,
                    ),
                ),
            )
            treatment.validate()
            torch.manual_seed(3301)
            torch_state_control = make_payload(control, CHECKPOINT_KIND_PERIODIC_RESUME)[0]
            torch.manual_seed(3301)
            torch_state_treatment = make_payload(treatment, CHECKPOINT_KIND_PERIODIC_RESUME)[0]
            self.assertEqual(tuple(torch_state_control), tuple(torch_state_treatment))
            self.assertNotEqual(torch_state_control["config_hash"], torch_state_treatment["config_hash"])
            for field in (
                "training_state",
                "model_state",
                "optimizer_state",
                "policy_rng_state",
                "active_rollout_state",
                "diagnostics_state",
            ):
                assert_nested_equal(
                    self,
                    torch_state_control[field],
                    torch_state_treatment[field],
                )

    def test_01_partial_rollout_roundtrip_is_field_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_checkpoint_config(
                Path(directory), horizon=100, interval=100, budget=200
            )
            buffer = collect_partial_buffer(config, 5)
            state = serialize_active_rollout(
                buffer, 0, CHECKPOINT_KIND_PERIODIC_RESUME
            )
            restored, version = restore_active_rollout(
                config,
                state,
                checkpoint_kind=CHECKPOINT_KIND_PERIODIC_RESUME,
                policy_version=0,
            )
            self.assertEqual(len(restored), 5)
            self.assertEqual(version, 0)
            for index in range(5):
                assert_nested_equal(
                    self,
                    serialize_rollout_transition(buffer.transition_at(index)),
                    serialize_rollout_transition(restored.transition_at(index)),
                )

    def test_02_full_or_corrupt_partial_rollout_fails_without_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_checkpoint_config(Path(directory))
            buffer = CAGATMAPPORolloutBuffer(config, capacity=1)
            buffer.append(collect_partial_buffer(config, 1).transition_at(0))
            with self.assertRaisesRegex(CheckpointError, "pending full rollout"):
                serialize_active_rollout(
                    buffer, 0, CHECKPOINT_KIND_PERIODIC_RESUME
                )
            valid = serialize_active_rollout(
                collect_partial_buffer(config, 1),
                0,
                CHECKPOINT_KIND_PERIODIC_RESUME,
            )
            valid["rollout_length"] = 2
            with self.assertRaisesRegex(CheckpointError, "count differs"):
                restore_active_rollout(
                    config,
                    valid,
                    checkpoint_kind=CHECKPOINT_KIND_PERIODIC_RESUME,
                    policy_version=0,
                )

    def test_03_policy_rng_roundtrip_preserves_multiple_next_proposals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_checkpoint_config(root)
            payload, actor, _critic, _updater = make_payload(
                config, CHECKPOINT_KIND_PERIODIC_RESUME
            )
            path = atomic_save_checkpoint(payload, root / "policy.pt")
            loaded = load_checkpoint_payload(path)
            first = restore_policy_rng(loaded["policy_rng_state"], "cpu")
            second = restore_policy_rng(loaded["policy_rng_state"], "cpu")
            collector = RolloutCollectorFixture(config)
            batch = collector.actor_tensorizer.encode_step(
                collector.observations, episode_start=True
            )
            masks = SequentialActionMaskBatch.from_observations(
                collector.observations
            )
            hidden = actor.initial_hidden(1)
            distribution = CAGATMAPPOActionDistribution(actor, config)
            proposals_a = []
            proposals_b = []
            for _ in range(4):
                proposals_a.append(
                    distribution.sample_actions(
                        batch, masks, hidden, generator=first
                    ).proposals
                )
                proposals_b.append(
                    distribution.sample_actions(
                        batch, masks, hidden, generator=second
                    ).proposals
                )
            self.assertEqual(proposals_a, proposals_b)
            self.assertTrue(torch.equal(first.get_state(), second.get_state()))


class CheckpointIdentityRegressionTests(unittest.TestCase):
    """Keep historical default elision strict and narrowly scoped."""

    def _payload(self, current: RunConfig, source: RunConfig) -> dict:
        snapshot = copy.deepcopy(source.snapshot_dict())
        snapshot["_metadata"]["config_hash"] = current.config_hash
        snapshot["_metadata"]["run_id"] = current.run_id
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_kind": CHECKPOINT_KIND_PERIODIC_RESUME,
            "method_id": current.method_id,
            "git_commit": current.git_commit,
            "config_hash": current.config_hash,
            "config_snapshot": snapshot,
            "runtime_provenance": {
                "device_type": "cpu",
                "dtype": "torch.float32",
            },
        }

    def _config(self, *, channel: ChannelAblationMode, route: RouteDecoderMode) -> RunConfig:
        base = make_checkpoint_config(Path("identity-regression"))
        config = replace(
            base,
            environment=replace(base.environment, channel_ablation_mode=channel),
            training=replace(
                base.training,
                mappo=replace(base.training.mappo, route_decoder_mode=route),
            ),
        )
        config.validate()
        return config

    def _validate(self, current: RunConfig, source: RunConfig) -> None:
        payload = self._payload(current, source)
        with patch(
            "src.models.ca_gat_mappo_checkpoint.validate_checkpoint_payload",
            side_effect=lambda value, config=None: value,
        ):
            validate_periodic_checkpoint_compatibility(
                current,
                payload,
                dtype=torch.float32,
                cuda_available=False,
            )

    def test_default_channel_elision_and_decoder_identity_are_strict(self) -> None:
        legacy_full = self._config(
            channel=ChannelAblationMode.FULL,
            route=RouteDecoderMode.LEGACY,
        )
        simplified_legacy = self._config(
            channel=ChannelAblationMode.SIMPLIFIED_DETERMINISTIC,
            route=RouteDecoderMode.LEGACY,
        )
        option_full = self._config(
            channel=ChannelAblationMode.FULL,
            route=RouteDecoderMode.OPTION_AWARE_V1,
        )

        with self.subTest(case="full snapshot vs omitted canonical default"):
            self._validate(legacy_full, legacy_full)
        with self.subTest(case="simplified snapshot vs omitted canonical default"):
            with self.assertRaisesRegex(
                CheckpointError, "checkpoint config snapshot differs"
            ):
                self._validate(legacy_full, simplified_legacy)
        with self.subTest(case="full snapshot vs simplified current config"):
            with self.assertRaisesRegex(
                CheckpointError, "checkpoint config snapshot differs"
            ):
                self._validate(simplified_legacy, legacy_full)
        with self.subTest(case="legacy snapshot vs option-aware current config"):
            with self.assertRaisesRegex(
                CheckpointError, "route_decoder_mode mismatch"
            ):
                self._validate(option_full, legacy_full)


class AtomicAndCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = make_checkpoint_config(self.root)
        self.periodic, *_ = make_payload(
            self.config, CHECKPOINT_KIND_PERIODIC_RESUME
        )
        self.final, *_ = make_payload(
            self.config, CHECKPOINT_KIND_FINAL_COMPLETED
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _raw_save(self, payload, name: str) -> Path:
        path = self.root / name
        with path.open("wb") as handle:
            torch.save(payload, handle)
        return path

    def test_04_atomic_save_load_and_existing_target_protection(self) -> None:
        path = atomic_save_checkpoint(self.periodic, self.root / "step_4.pt")
        loaded = load_checkpoint_payload(path)
        assert_nested_equal(self, self.periodic, loaded)
        before = path.read_bytes()
        with self.assertRaisesRegex(CheckpointError, "already exists"):
            atomic_save_checkpoint(self.periodic, path)
        self.assertEqual(path.read_bytes(), before)

    def test_05_serialization_failure_leaves_no_final_or_temp(self) -> None:
        target = self.root / "failure.pt"
        with patch(
            "src.models.ca_gat_mappo_checkpoint.torch.save",
            side_effect=RuntimeError("synthetic serialization failure"),
        ):
            with self.assertRaisesRegex(CheckpointError, "atomic checkpoint save failed"):
                atomic_save_checkpoint(self.periodic, target)
        self.assertFalse(target.exists())
        self.assertEqual(tuple(self.root.glob(".failure.pt.*.tmp")), ())

    def test_06_corruption_is_explicit_and_never_falls_back(self) -> None:
        atomic_save_checkpoint(self.periodic, self.root / "step_4.pt")
        corrupt = self.root / "step_8.pt"
        corrupt.write_bytes(b"not a torch checkpoint")
        with self.assertRaisesRegex(CheckpointError, "checkpoint load failed"):
            load_checkpoint_payload(corrupt)

    def test_07_unknown_schema_kind_and_missing_key_fail_fast(self) -> None:
        cases = []
        wrong_schema = copy.deepcopy(self.periodic)
        wrong_schema["schema_version"] = 2
        cases.append((wrong_schema, "schema version mismatch"))
        wrong_kind = copy.deepcopy(self.periodic)
        wrong_kind["checkpoint_kind"] = "UNKNOWN"
        cases.append((wrong_kind, "unknown checkpoint kind"))
        missing = copy.deepcopy(self.periodic)
        del missing["model_state"]
        cases.append((missing, "fields differ"))
        for index, (payload, message) in enumerate(cases):
            with self.subTest(message=message):
                path = self._raw_save(payload, f"invalid_{index}.pt")
                with self.assertRaisesRegex(CheckpointError, message):
                    load_checkpoint_payload(path)

    def test_08_strict_metadata_failures_precede_model_construction(self) -> None:
        cases = []
        for name, value in (
            ("method_id", "random"),
            ("git_commit", "different-commit"),
            ("config_hash", "0" * 64),
        ):
            payload = copy.deepcopy(self.periodic)
            payload[name] = value
            cases.append((name, payload))
        device = copy.deepcopy(self.periodic)
        device["runtime_provenance"]["device_type"] = "cuda"
        device["policy_rng_state"]["device_type"] = "cuda"
        cases.append(("device", device))
        dtype = copy.deepcopy(self.periodic)
        dtype["runtime_provenance"]["dtype"] = "torch.float64"
        cases.append(("dtype", dtype))
        for index, (name, payload) in enumerate(cases):
            with self.subTest(name=name):
                path = self._raw_save(payload, f"metadata_{index}.pt")
                with patch(
                    "src.models.ca_gat_mappo_trainer.CAGATMAPPOActor"
                ) as actor:
                    with self.assertRaises(CheckpointError):
                        CAGATMAPPOTrainer.resume_from_checkpoint(
                            self.config, path
                        )
                actor.assert_not_called()

    def test_09_final_completed_is_not_a_resume_checkpoint(self) -> None:
        path = atomic_save_checkpoint(self.final, self.root / "final.pt")
        with patch(
            "src.models.ca_gat_mappo_trainer.CAGATMAPPOActor"
        ) as actor:
            with self.assertRaisesRegex(CheckpointError, "training already complete"):
                CAGATMAPPOTrainer.resume_from_checkpoint(self.config, path)
        actor.assert_not_called()


class OptimizerAndTrainerIntegrationTests(unittest.TestCase):
    def test_10_true_ppo_adam_state_roundtrip_is_nonempty_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_checkpoint_config(
                root, horizon=260, interval=260, budget=520
            )
            seed = make_seed_transition(config)
            synthetic = make_synthetic_buffer(config, seed)
            aligned = align_old_policy_snapshots(config, synthetic)
            actor = CAGATMAPPOActor(config)
            critic = MAPPOCentralizedCritic(config)
            distribution = CAGATMAPPOActionDistribution(actor, config)
            updater = CAGATMAPPORecurrentPPOUpdater(
                actor, critic, config, action_distribution=distribution
            )
            updater.update(aligned)
            self.assertTrue(updater.optimizers.actor_optimizer.state)
            self.assertTrue(updater.optimizers.critic_optimizer.state)
            payload, *_ = make_payload(
                config,
                CHECKPOINT_KIND_PERIODIC_RESUME,
                actor=actor,
                critic=critic,
                updater=updater,
            )
            path = atomic_save_checkpoint(payload, root / "adam.pt")
            loaded = load_checkpoint_payload(path)
            restored_actor = CAGATMAPPOActor(config)
            restored_critic = MAPPOCentralizedCritic(config)
            restored_updater = CAGATMAPPORecurrentPPOUpdater(
                restored_actor, restored_critic, config
            )
            restore_model_optimizer_and_rng(
                config=config,
                payload=loaded,
                actor=restored_actor,
                critic=restored_critic,
                optimizers=restored_updater.optimizers,
                dtype=torch.float32,
            )
            assert_nested_equal(
                self,
                updater.optimizers.actor_optimizer.state_dict(),
                restored_updater.optimizers.actor_optimizer.state_dict(),
            )
            assert_nested_equal(
                self,
                updater.optimizers.critic_optimizer.state_dict(),
                restored_updater.optimizers.critic_optimizer.state_dict(),
            )
            first_state = next(
                iter(restored_updater.optimizers.actor_optimizer.state.values())
            )
            self.assertIn("exp_avg", first_state)
            self.assertIn("exp_avg_sq", first_state)
            self.assertIn("step", first_state)

    def test_11_cpu_uninterrupted_equals_save_load_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_checkpoint_config(
                root, horizon=130, interval=260, budget=520
            )
            uninterrupted = CAGATMAPPOTrainer(
                config, updater_factory=deterministic_updater_factory
            )
            result_a = uninterrupted.train()

            paused = CAGATMAPPOTrainer(
                config, updater_factory=deterministic_updater_factory
            )
            checkpoint = paused.train_until_periodic_checkpoint()
            self.assertEqual(checkpoint.name, "step_260.pt")
            periodic = load_checkpoint_payload(checkpoint)
            self.assertEqual(
                periodic["active_rollout_state"]["rollout_length"], 4
            )
            self.assertEqual(periodic["training_state"]["policy_version"], 1)

            resumed = CAGATMAPPOTrainer.resume_from_checkpoint(
                config,
                checkpoint,
                updater_factory=deterministic_updater_factory,
            )
            result_b = resumed.train_with_checkpoints()
            self.assertIsInstance(result_b, CAGATMAPPOTrainingResult)
            self.assertEqual(result_a, result_b)
            assert_nested_equal(
                self, uninterrupted.actor.state_dict(), resumed.actor.state_dict()
            )
            assert_nested_equal(
                self, uninterrupted.critic.state_dict(), resumed.critic.state_dict()
            )
            assert_nested_equal(
                self,
                uninterrupted.updater.optimizers.actor_optimizer.state_dict(),
                resumed.updater.optimizers.actor_optimizer.state_dict(),
            )
            assert_nested_equal(
                self,
                uninterrupted.updater.optimizers.critic_optimizer.state_dict(),
                resumed.updater.optimizers.critic_optimizer.state_dict(),
            )
            self.assertTrue(
                torch.equal(
                    uninterrupted.policy_generator.get_state(),
                    resumed.policy_generator.get_state(),
                )
            )
            self.assertEqual(uninterrupted.policy_version, resumed.policy_version)
            final_path = Path(config.artifact_paths()["final_checkpoint"])
            self.assertTrue(final_path.is_file())
            self.assertFalse((final_path.parent / "step_520.pt").exists())
            final = load_checkpoint_payload(final_path)
            self.assertTrue(final["training_state"]["training_complete"])
            self.assertEqual(final["active_rollout_state"]["rollout_length"], 0)

    def test_12_formal_final_geometry_is_formula_derived(self) -> None:
        base = RunConfig()
        collected = base.training.mappo.max_environment_transitions
        rollout = base.training.mappo.rollout_length_slots
        self.assertEqual(collected, 500000)
        self.assertEqual((collected // rollout) * rollout, 499968)
        self.assertEqual(collected % rollout, 32)
        self.assertEqual(collected // rollout, 1953)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CUDACheckpointTests(unittest.TestCase):
    def test_13_cuda_state_rng_optimizer_and_continuation_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_checkpoint_config(root, device="cuda")
            actor = CAGATMAPPOActor(config).cuda()
            critic = MAPPOCentralizedCritic(config).cuda()
            optimizers = build_ca_gat_mappo_optimizers(actor, critic, config)
            optimizers.actor_optimizer.zero_grad(set_to_none=True)
            optimizers.critic_optimizer.zero_grad(set_to_none=True)
            sum(item.square().sum() for item in actor.parameters()).backward()
            sum(item.square().sum() for item in critic.parameters()).backward()
            optimizers.actor_optimizer.step()
            optimizers.critic_optimizer.step()
            distribution = CAGATMAPPOActionDistribution(actor, config)
            updater = CAGATMAPPORecurrentPPOUpdater(
                actor,
                critic,
                config,
                optimizers=optimizers,
                action_distribution=distribution,
            )
            payload, *_ = make_payload(
                config,
                CHECKPOINT_KIND_PERIODIC_RESUME,
                actor=actor,
                critic=critic,
                updater=updater,
            )
            path = atomic_save_checkpoint(payload, root / "cuda.pt")
            loaded = load_checkpoint_payload(path)
            restored_actor = CAGATMAPPOActor(config).cuda()
            restored_critic = MAPPOCentralizedCritic(config).cuda()
            restored_updater = CAGATMAPPORecurrentPPOUpdater(
                restored_actor, restored_critic, config
            )
            generator = restore_model_optimizer_and_rng(
                config=config,
                payload=loaded,
                actor=restored_actor,
                critic=restored_critic,
                optimizers=restored_updater.optimizers,
                dtype=torch.float32,
            )
            assert_nested_equal(self, actor.state_dict(), restored_actor.state_dict())
            assert_nested_equal(self, critic.state_dict(), restored_critic.state_dict())
            for optimizer in (
                restored_updater.optimizers.actor_optimizer,
                restored_updater.optimizers.critic_optimizer,
            ):
                self.assertTrue(
                    all(
                        value.device.type == "cuda"
                        for state in optimizer.state.values()
                        for value in state.values()
                        if isinstance(value, torch.Tensor)
                    )
                )
            expected = restore_policy_rng(loaded["policy_rng_state"], "cuda")
            self.assertTrue(
                torch.equal(
                    torch.rand(8, device="cuda", generator=generator),
                    torch.rand(8, device="cuda", generator=expected),
                )
            )
            buffer, version = restore_active_rollout(
                config,
                loaded["active_rollout_state"],
                checkpoint_kind=CHECKPOINT_KIND_PERIODIC_RESUME,
                policy_version=0,
            )
            self.assertEqual((len(buffer), version), (4, 0))

            collector = RolloutCollectorFixture(config)
            tensorizer = ActorObservationTensorizer(config)
            actor_batch = tensorizer.encode_step(
                collector.observations,
                device="cuda",
                episode_start=True,
            )
            masks = SequentialActionMaskBatch.from_observations(
                collector.observations
            )
            output = CAGATMAPPOActionDistribution(
                restored_actor, config
            ).sample_actions(
                actor_batch,
                masks,
                restored_actor.initial_hidden(1, device="cuda"),
                generator=generator,
            )
            self.assertTrue(
                all(
                    observation.action_masks.is_legal(proposal)
                    for observation, proposal in zip(
                        collector.observations, output.proposals[0][0]
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
