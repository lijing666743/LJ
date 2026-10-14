from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from src.artifacts import ArtifactConflictError
from src.cli import (
    build_arg_parser,
    build_execution_context_from_args,
    build_run_config_from_args,
    main as cli_main,
)
from src.config import (
    ActorRatioMode,
    AgentCreditMode,
    CHECKPOINT_KIND_FINAL_COMPLETED,
    CHECKPOINT_KIND_PERIODIC_RESUME,
    ChannelAblationMode,
    ConfigError,
    RewardCreditConfig,
    RouteDecoderMode,
    RunConfig,
    canonical_config_identity_hash,
)
from src.evaluation.actor_loader import (
    EvaluationCheckpointError,
    load_final_actor_for_evaluation,
)
from src.evaluation.metrics import FORMAL_METRIC_FIELDS
from src.evaluation.runner import FORMAL_METHOD_SUITE, FormalEvaluationRunner
from src.execution import ExecutionContext, validate_execution_context
from src.models.ca_gat_mappo_checkpoint import atomic_save_checkpoint
from src.registry import RunResult, build_default_registry
from tests.test_ca_gat_mappo_checkpoint import make_checkpoint_config, make_payload


def make_evaluation_fixture(
    root: Path,
    *,
    evaluation_seeds: tuple[int, ...] = (1042,),
    arrivals: tuple[float, ...] | None = None,
    kind: str = CHECKPOINT_KIND_FINAL_COMPLETED,
    actor_ratio_mode: ActorRatioMode = ActorRatioMode.JOINT,
    agent_credit_mode: AgentCreditMode = AgentCreditMode.TEAM,
    channel_ablation_mode: ChannelAblationMode = ChannelAblationMode.FULL,
    route_decoder_mode: RouteDecoderMode = RouteDecoderMode.LEGACY,
    reward: RewardCreditConfig | None = None,
    omit_snapshot_channel_mode: bool = False,
    multi_uav: bool = False,
):
    source = make_checkpoint_config(root, horizon=4, interval=4, budget=8)
    if multi_uav:
        environment = RunConfig().environment
        source = replace(
            source,
            environment=replace(
                environment,
                episode_horizon=4,
                arrival_probabilities=(1.0,) * environment.uav_count,
                building_layout=(),
                candidate_neighbor_radius_m=2_000.0,
                velocity_std_mps=(0.0, 0.0),
                shadowing_std_db=0.0,
                csi_error_std_db=0.0,
                fixed_csi_aoi_slots=1,
            ),
        )
    source = replace(
        source,
        training=replace(
            source.training,
            mappo=replace(
                source.training.mappo,
                actor_ratio_mode=actor_ratio_mode,
                agent_credit_mode=agent_credit_mode,
                route_decoder_mode=route_decoder_mode,
            ),
        ),
        environment=replace(
            source.environment,
            channel_ablation_mode=channel_ablation_mode,
        ),
        reward=source.reward if reward is None else reward,
    )
    if arrivals is not None:
        source = replace(
            source,
            environment=replace(source.environment, arrival_probabilities=arrivals),
        )
    object.__setattr__(source, "git_branch", "test-branch")
    object.__setattr__(source, "git_commit", "a" * 40)
    object.__setattr__(source, "git_dirty", False)
    payload, source_actor, _critic, _updater = make_payload(source, kind)
    if omit_snapshot_channel_mode:
        payload["config_snapshot"]["environment"].pop(
            "channel_ablation_mode", None
        )
    checkpoint = root / ("final.pt" if kind == CHECKPOINT_KIND_FINAL_COMPLETED else "periodic.pt")
    atomic_save_checkpoint(payload, checkpoint)
    evaluation = replace(
        source,
        mode="evaluation",
        method_id="ca_gat_mappo",
        launch_profile=None,
        evaluation=replace(source.evaluation, evaluation_seeds=evaluation_seeds),
    )
    object.__setattr__(evaluation, "git_branch", "test-evaluator")
    object.__setattr__(evaluation, "git_commit", "b" * 40)
    object.__setattr__(evaluation, "git_dirty", False)
    evaluation.validate()
    return source, evaluation, checkpoint, source_actor


class TestEvaluationExecutionContext(unittest.TestCase):
    def test_direct_cli_parses_explicit_evaluation_context_outside_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_a = Path(directory) / "a.pt"
            checkpoint_b = Path(directory) / "b.pt"
            checkpoint_a.write_bytes(b"a")
            checkpoint_b.write_bytes(b"b")
            parser = build_arg_parser()
            common = [
                "--mode",
                "evaluation",
                "--method-id",
                "ca_gat_mappo",
                "--evaluation-device",
                "cpu",
            ]
            args_a = parser.parse_args([*common, "--evaluate-from", str(checkpoint_a)])
            args_b = parser.parse_args([*common, "--evaluate-from", str(checkpoint_b)])
            config_a = build_run_config_from_args(args_a)
            config_b = build_run_config_from_args(args_b)
            context = build_execution_context_from_args(args_a)

            self.assertEqual(context.evaluate_from, checkpoint_a)
            self.assertEqual(context.evaluation_device, "cpu")
            self.assertIsNone(context.resume_from)
            self.assertEqual(config_a.config_hash, config_b.config_hash)
            self.assertEqual(config_a.run_id, config_b.run_id)
            self.assertNotIn("evaluate_from", config_a.snapshot_dict())

    def test_resume_and_evaluate_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_bytes(b"checkpoint")
            config = build_run_config_from_args(
                build_arg_parser().parse_args(
                    [
                        "--mode",
                        "evaluation",
                        "--method-id",
                        "ca_gat_mappo",
                    ]
                )
            )
            context = ExecutionContext.from_paths(
                resume_from=path,
                evaluate_from=path,
                evaluation_device="cpu",
            )
            with self.assertRaisesRegex(ConfigError, "mutually exclusive"):
                validate_execution_context(config, context)

    def test_missing_directory_and_missing_device_fail_before_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = build_run_config_from_args(
                build_arg_parser().parse_args(
                    [
                        "--mode",
                        "evaluation",
                        "--method-id",
                        "ca_gat_mappo",
                    ]
                )
            )
            cases = (
                (root / "missing.pt", "cpu", "does not exist"),
                (root, "cpu", "not a regular file"),
            )
            for path, device, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ConfigError, message):
                        validate_execution_context(
                            config,
                            ExecutionContext.from_evaluation_path(path, device),
                        )
            checkpoint = root / "final.pt"
            checkpoint.write_bytes(b"payload")
            with self.assertRaisesRegex(ConfigError, "explicit"):
                validate_execution_context(
                    config,
                    ExecutionContext.from_evaluation_path(checkpoint, None),
                )

    def test_interactive_menu_passes_the_same_evaluation_context_to_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "final.pt"
            checkpoint.write_bytes(b"synthetic")
            answers = iter(("7", "small", "", "42", str(checkpoint), "cpu"))
            completed = RunResult(
                status="completed",
                run_id="evaluation-id",
                mode="evaluation",
                method_id="ca_gat_mappo",
                message="stubbed",
            )
            with patch("src.cli.Runner.run", return_value=completed) as run:
                status = cli_main(
                    [],
                    input_fn=lambda _prompt: next(answers),
                    output_fn=lambda _line: None,
                )
            self.assertEqual(status, 0)
            context = run.call_args.kwargs["execution_context"]
            self.assertEqual(context.evaluate_from, checkpoint)
            self.assertEqual(context.evaluation_device, "cpu")

    def test_registry_enables_the_real_formal_handler(self) -> None:
        self.assertEqual(
            build_default_registry().resolve(
                "evaluation", "ca_gat_mappo"
            ).__name__,
            "formal_evaluation_handler",
        )


class TestFinalActorOnlyLoader(unittest.TestCase):
    def test_historical_snapshot_missing_default_full_channel_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, config, checkpoint, _actor = make_evaluation_fixture(
                Path(directory), omit_snapshot_channel_mode=True
            )
            loaded = load_final_actor_for_evaluation(
                config,
                checkpoint,
                evaluation_device="cpu",
            )

            self.assertEqual(
                loaded.source_training_config_original_hash,
                source.config_hash,
            )
            self.assertEqual(
                loaded.source_training_config_raw_snapshot_hash,
                loaded.source_training_config_canonical_hash,
            )

    def test_final_checkpoint_loads_only_actor_and_enters_eval_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _source, config, checkpoint, source_actor = make_evaluation_fixture(
                Path(directory)
            )
            with patch(
                "src.models.ca_gat_mappo_checkpoint.restore_model_optimizer_and_rng"
            ) as restore_all, patch(
                "src.models.ca_gat_mappo_checkpoint.restore_policy_rng"
            ) as restore_rng, patch(
                "src.models.ca_gat_mappo_checkpoint.restore_active_rollout"
            ) as restore_rollout:
                loaded = load_final_actor_for_evaluation(
                    config,
                    checkpoint,
                    evaluation_device="cpu",
                )
            restore_all.assert_not_called()
            restore_rng.assert_not_called()
            restore_rollout.assert_not_called()
            self.assertFalse(loaded.actor.training)
            self.assertEqual(loaded.source_checkpoint_kind, "FINAL_COMPLETED")
            self.assertEqual(loaded.source_training_actor_ratio_mode, "joint")
            self.assertEqual(loaded.source_training_agent_credit_mode, "team")
            self.assertEqual(loaded.source_training_device, "cpu")
            self.assertEqual(loaded.evaluation_device, "cpu")
            self.assertEqual(
                loaded.source_training_config_original_hash,
                loaded.source_training_config_canonical_hash,
            )
            self.assertNotEqual(
                loaded.source_training_config_raw_snapshot_hash,
                loaded.source_training_config_canonical_hash,
            )
            self.assertEqual(
                loaded.source_checkpoint_metadata["config_hash"],
                loaded.source_training_config_original_hash,
            )
            for name, value in source_actor.state_dict().items():
                self.assertTrue(torch.equal(value.cpu(), loaded.actor.state_dict()[name].cpu()))

    def test_nondefault_deterministic_channel_is_distinct_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, config, checkpoint, _actor = make_evaluation_fixture(
                Path(directory),
                channel_ablation_mode=(
                    ChannelAblationMode.SIMPLIFIED_DETERMINISTIC
                ),
            )
            full_identity = source.snapshot_dict()
            full_identity["environment"]["channel_ablation_mode"] = "full"
            self.assertNotEqual(
                canonical_config_identity_hash(source.snapshot_dict()),
                canonical_config_identity_hash(full_identity),
            )
            full_evaluator = replace(
                config,
                environment=replace(
                    config.environment,
                    channel_ablation_mode=ChannelAblationMode.FULL,
                ),
            )
            full_evaluator.validate()

            with self.assertRaisesRegex(
                EvaluationCheckpointError,
                "environment/observation domain",
            ):
                load_final_actor_for_evaluation(
                    full_evaluator,
                    checkpoint,
                    evaluation_device="cpu",
                )

    def test_legacy_decoder_checkpoint_is_rejected_by_option_aware_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _source, config, checkpoint, _actor = make_evaluation_fixture(
                Path(directory),
                route_decoder_mode=RouteDecoderMode.LEGACY,
            )
            option_evaluator = replace(
                config,
                training=replace(
                    config.training,
                    mappo=replace(
                        config.training.mappo,
                        route_decoder_mode=RouteDecoderMode.OPTION_AWARE_V1,
                    ),
                ),
            )
            option_evaluator.validate()

            with self.assertRaisesRegex(
                EvaluationCheckpointError,
                "route_decoder_mode",
            ):
                load_final_actor_for_evaluation(
                    option_evaluator,
                    checkpoint,
                    evaluation_device="cpu",
                )

    def test_reward_provenance_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _source, config, checkpoint, _actor = make_evaluation_fixture(
                Path(directory)
            )
            incompatible = replace(
                config,
                reward=replace(
                    config.reward,
                    beta_completion_credit=0.75,
                ),
            )
            incompatible.validate()

            with self.assertRaisesRegex(
                EvaluationCheckpointError,
                "reward provenance mismatch",
            ):
                load_final_actor_for_evaluation(
                    incompatible,
                    checkpoint,
                    evaluation_device="cpu",
                )

    def test_branch_specific_training_provenance_reaches_formal_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _source, config, checkpoint, _actor = make_evaluation_fixture(
                Path(directory),
                actor_ratio_mode=ActorRatioMode.BRANCH_SPECIFIC,
            )
            runner = FormalEvaluationRunner(
                config,
                checkpoint,
                evaluation_device="cpu",
            )
            self.assertEqual(
                runner.loaded_actor.source_training_actor_ratio_mode,
                "branch_specific",
            )
            result = runner.run(write_artifacts=False)
            self.assertEqual(
                result.manifest["source_training_actor_ratio_mode"],
                "branch_specific",
            )

    def test_role_credit_training_provenance_reaches_formal_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _source, config, checkpoint, _actor = make_evaluation_fixture(
                Path(directory),
                agent_credit_mode=AgentCreditMode.ROLE_DECOMPOSED,
            )
            runner = FormalEvaluationRunner(
                config,
                checkpoint,
                evaluation_device="cpu",
            )
            self.assertEqual(
                runner.loaded_actor.source_training_agent_credit_mode,
                "role_decomposed",
            )
            result = runner.run(write_artifacts=False)
            self.assertEqual(
                result.manifest["source_training_agent_credit_mode"],
                "role_decomposed",
            )

    def test_periodic_resume_is_rejected_for_formal_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _source, config, checkpoint, _actor = make_evaluation_fixture(
                Path(directory), kind=CHECKPOINT_KIND_PERIODIC_RESUME
            )
            with self.assertRaisesRegex(
                EvaluationCheckpointError, "FINAL_COMPLETED"
            ):
                load_final_actor_for_evaluation(
                    config,
                    checkpoint,
                    evaluation_device="cpu",
                )

    def test_architecture_and_action_domain_mismatch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _source, config, checkpoint, _actor = make_evaluation_fixture(
                Path(directory)
            )
            incompatible = replace(
                config,
                environment=replace(config.environment, ru_count=10),
            )
            with self.assertRaises((ConfigError, EvaluationCheckpointError)):
                load_final_actor_for_evaluation(
                    incompatible,
                    checkpoint,
                    evaluation_device="cpu",
                )


class TestFormalEvaluationRunner(unittest.TestCase):
    def test_hierarchical_route_telemetry_is_action_neutral_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _source, config, checkpoint, _actor = make_evaluation_fixture(
                Path(directory),
                route_decoder_mode=RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1,
                multi_uav=True,
            )
            control = FormalEvaluationRunner(
                config,
                checkpoint,
                evaluation_device="cpu",
                collect_route_telemetry=False,
            ).run(write_artifacts=False)
            instrumented = FormalEvaluationRunner(
                config,
                checkpoint,
                evaluation_device="cpu",
                collect_route_telemetry=True,
            ).run(write_artifacts=False)

            self.assertEqual(control.aggregate_metrics, instrumented.aggregate_metrics)
            self.assertEqual(
                [item.action_trace_sha256 for item in control.episodes],
                [item.action_trace_sha256 for item in instrumented.episodes],
            )
            self.assertEqual(
                [item.artifact_record(control.evaluation_run_id) for item in control.episodes],
                [
                    item.artifact_record(instrumented.evaluation_run_id)
                    for item in instrumented.episodes
                ],
            )
            self.assertTrue(instrumented.route_decision_trace)
            required = {
                "episode_id",
                "step",
                "uav_id",
                "local_probability",
                "remote_probability",
                "defer_probability",
                "remote_destination_probabilities",
                "selected_top_level",
                "selected_remote_id",
                "final_route_action_index",
                "legal_route_candidates",
                "route_action_mask",
                "tx_action",
                "resource_group",
                "power_action",
                "executed_communication",
            }
            for record in instrumented.route_decision_trace:
                self.assertTrue(required.issubset(record))
                self.assertAlmostEqual(
                    record["local_probability"]
                    + record["remote_probability"]
                    + record["defer_probability"],
                    1.0,
                    places=6,
                )
                destinations = record["remote_destination_probabilities"]
                if destinations:
                    self.assertAlmostEqual(sum(destinations.values()), 1.0, places=6)
            summary = instrumented.route_summary
            assert summary is not None
            self.assertEqual(
                summary["route_decision_count"],
                len(instrumented.route_decision_trace),
            )
            self.assertEqual(
                summary["route_decision_count"],
                summary["local_count"]
                + summary["remote_count"]
                + summary["defer_count"]
                + summary["idle_count"],
            )

    def test_route_telemetry_remains_compatible_with_all_decoder_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in (
                RouteDecoderMode.LEGACY,
                RouteDecoderMode.OPTION_AWARE_V1,
                RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1,
            ):
                with self.subTest(route_decoder_mode=mode.value):
                    case_root = root / mode.value
                    case_root.mkdir()
                    _source, config, checkpoint, _actor = make_evaluation_fixture(
                        case_root,
                        route_decoder_mode=mode,
                        multi_uav=True,
                    )
                    result = FormalEvaluationRunner(
                        config,
                        checkpoint,
                        evaluation_device="cpu",
                    ).run(write_artifacts=False)
                    self.assertTrue(result.route_decision_trace)
                    self.assertTrue(
                        all(
                            record["decoder_mode"] == mode.value
                            for record in result.route_decision_trace
                        )
                    )

    def test_cpu_evaluation_is_repeatable_actor_safe_and_fair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _source, config, checkpoint, _actor = make_evaluation_fixture(
                Path(directory), evaluation_seeds=(1042, 1043)
            )
            runner = FormalEvaluationRunner(
                config,
                checkpoint,
                evaluation_device="cpu",
            )
            deterministic = runner.action_distribution.deterministic_actions

            def checked_deterministic(*args, **kwargs):
                self.assertFalse(torch.is_grad_enabled())
                self.assertFalse(runner.loaded_actor.actor.training)
                return deterministic(*args, **kwargs)

            with patch.object(
                runner.action_distribution,
                "deterministic_actions",
                side_effect=checked_deterministic,
            ) as deterministic_call, patch.object(
                torch.Tensor, "backward", side_effect=AssertionError("backward forbidden")
            ), patch.object(
                torch.optim.Optimizer,
                "step",
                side_effect=AssertionError("optimizer.step forbidden"),
            ):
                first = runner.run(write_artifacts=False)
            second = runner.run(write_artifacts=False)

            self.assertGreater(deterministic_call.call_count, 0)
            self.assertEqual(first.artifacts, ())
            self.assertEqual(first.aggregate_metrics, second.aggregate_metrics)
            self.assertEqual(
                [item.artifact_record(first.evaluation_run_id) for item in first.episodes],
                [item.artifact_record(second.evaluation_run_id) for item in second.episodes],
            )
            self.assertEqual(len(first.episodes), 2 * len(FORMAL_METHOD_SUITE))
            for seed in config.evaluation.evaluation_seeds:
                selected = [item for item in first.episodes if item.evaluation_seed == seed]
                self.assertEqual(len({item.external_trace_sha256 for item in selected}), 1)
            for method_id in FORMAL_METHOD_SUITE:
                left = [item for item in first.episodes if item.method_id == method_id]
                right = [item for item in second.episodes if item.method_id == method_id]
                self.assertEqual(
                    [item.action_trace for item in left],
                    [item.action_trace for item in right],
                )
            actor_episodes = [
                item for item in first.episodes if item.method_id == "ca_gat_mappo"
            ]
            self.assertEqual(
                actor_episodes[0].hidden_trace_sha256[0],
                actor_episodes[1].hidden_trace_sha256[0],
            )
            self.assertTrue(
                all(
                    len(item.hidden_trace_sha256) == item.episode_horizon + 1
                    for item in actor_episodes
                )
            )
            self.assertFalse(runner.loaded_actor.actor.training)

    def test_complete_metric_schema_and_na_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _source, config, checkpoint, _actor = make_evaluation_fixture(
                Path(directory), arrivals=(0.0,)
            )
            result = FormalEvaluationRunner(
                config,
                checkpoint,
                evaluation_device="cpu",
            ).run(write_artifacts=False)
            for episode in result.episodes:
                self.assertEqual(tuple(episode.metrics), FORMAL_METRIC_FIELDS)
                self.assertEqual(episode.metrics["generated"], 0)
                self.assertEqual(episode.metrics["completed"], 0)
                self.assertIsNone(episode.metrics["completion_ratio"])
                self.assertIsNone(episode.metrics["completed_task_e2e_delay_s"])
                self.assertIsNone(episode.metrics["energy_per_completed_task_j"])
                self.assertEqual(episode.metrics["outage_denominator"], 0)
                self.assertIsNone(episode.metrics["outage_rate"])
                self.assertTrue(
                    episode.invariants[
                        "generated_equals_completed_plus_expired_plus_truncated"
                    ]
                )
                self.assertTrue(
                    episode.invariants["total_energy_equals_tx_plus_cpu"]
                )
                self.assertTrue(
                    episode.invariants["reward_equals_component_identity"]
                )
            for method_id in FORMAL_METHOD_SUITE:
                stats = result.aggregate_metrics["methods"][method_id]["metrics"]
                for value in stats.values():
                    self.assertEqual(
                        tuple(value), ("mean", "std", "valid_sample_count")
                    )
                self.assertEqual(
                    stats["completed_task_e2e_delay_s"]["valid_sample_count"],
                    0,
                )

    def test_artifacts_are_isolated_atomic_and_non_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _source, config, checkpoint, _actor = make_evaluation_fixture(
                Path(directory)
            )
            calls = 0

            def environment_factory(run_config):
                nonlocal calls
                calls += 1
                from src.env.environment import U2UMECEnvironment

                return U2UMECEnvironment(run_config)

            runner = FormalEvaluationRunner(
                config,
                checkpoint,
                evaluation_device="cpu",
                environment_factory=environment_factory,
            )
            first = runner.run(write_artifacts=True)
            self.assertEqual(len(first.artifacts), 4)
            self.assertEqual(len(first.diagnostic_artifacts), 2)
            self.assertTrue(
                all(
                    "/diagnostics/hierarchical-route-eval__" in path.replace("\\", "/")
                    for path in first.diagnostic_artifacts
                )
            )
            summary = json.loads(
                Path(first.diagnostic_artifacts[0]).read_text(encoding="utf-8")
            )
            trace = [
                json.loads(line)
                for line in Path(first.diagnostic_artifacts[1])
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(summary["route_decision_count"], len(trace))
            before = {Path(path): Path(path).read_bytes() for path in first.artifacts}
            call_count = calls
            with self.assertRaises(ArtifactConflictError):
                runner.run(write_artifacts=True)
            self.assertEqual(calls, call_count)
            self.assertEqual(
                before,
                {path: path.read_bytes() for path in before},
            )
            manifest = json.loads(Path(first.artifacts[0]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["method_suite"], list(FORMAL_METHOD_SUITE))
            self.assertEqual(manifest["source_checkpoint_kind"], "FINAL_COMPLETED")
            self.assertEqual(manifest["evaluation_device"], "cpu")
            self.assertTrue(manifest["masked_argmax"])
            self.assertIn("source_checkpoint_sha256", manifest)
            self.assertIn("source_training_config_hash", manifest)
            self.assertIn("source_training_config_original_hash", manifest)
            self.assertIn("source_training_config_raw_snapshot_hash", manifest)
            self.assertIn("source_training_config_canonical_hash", manifest)
            self.assertIn("source_checkpoint_metadata", manifest)
            self.assertEqual(manifest["source_training_actor_ratio_mode"], "joint")
            self.assertEqual(manifest["source_training_agent_credit_mode"], "team")
            self.assertIn("shared_external_trace_by_seed", manifest)
            self.assertIn("/evaluations/", first.artifacts[0].replace("\\", "/"))


if __name__ == "__main__":
    unittest.main()
