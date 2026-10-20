"""Unit tests for the Config/CLI/Runner implementation wave."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from src.cli import MENU_ROUTES, build_arg_parser, build_run_config_from_args, main
from src.config import (
    ChannelAblationMode,
    ConfigError,
    _detected_runtime_versions,
    load_run_config,
)
from src.execution import ExecutionContext
from src.registry import RunResult, build_default_registry, ca_gat_mappo_training_handler
from src.runner import Runner


class ConfigTests(unittest.TestCase):
    def test_defaults_and_scenario_derivation(self) -> None:
        config = load_run_config()
        self.assertEqual(config.config_version, "section4.v1")
        self.assertEqual(config.seed, 42)
        self.assertIsNone(config.launch_profile)
        self.assertEqual(config.environment.uav_count, 4)
        self.assertEqual(config.environment.ru_bandwidth_hz, 1_000_000.0)
        self.assertEqual(config.derived_stream_ids["task_arrival"], 20)
        runtime_versions = config.snapshot_dict()["_metadata"]["runtime_versions"]
        self.assertEqual(set(runtime_versions), {"python", "numpy", "torch", "cuda", "miniconda"})
        for value in runtime_versions.values():
            self.assertNotIn("TO VERIFY", value)

        medium = load_run_config(cli_overrides={"scenario_id": "medium"})
        self.assertEqual(medium.environment.uav_count, 6)
        self.assertEqual(len(medium.environment.profile_assignment), 6)

    def test_channel_ablation_mode_loading_identity_snapshot_and_validation(self) -> None:
        default = load_run_config()
        explicit_full = load_run_config(
            cli_overrides={"environment.channel_ablation_mode": "full"}
        )
        simplified = load_run_config(
            cli_overrides={
                "environment.channel_ablation_mode": "simplified_deterministic"
            }
        )

        self.assertEqual(default.environment.channel_ablation_mode, ChannelAblationMode.FULL)
        self.assertEqual(explicit_full.environment.channel_ablation_mode, ChannelAblationMode.FULL)
        self.assertEqual(
            simplified.environment.channel_ablation_mode,
            ChannelAblationMode.SIMPLIFIED_DETERMINISTIC,
        )
        self.assertEqual(default.config_hash, explicit_full.config_hash)
        self.assertNotEqual(default.config_hash, simplified.config_hash)
        self.assertNotIn("channel_ablation_mode", default.resolved_dict()["environment"])
        self.assertEqual(
            default.snapshot_dict()["environment"]["channel_ablation_mode"],
            "full",
        )
        self.assertEqual(
            simplified.snapshot_dict()["environment"]["channel_ablation_mode"],
            "simplified_deterministic",
        )
        with self.assertRaisesRegex(ConfigError, "channel_ablation_mode"):
            load_run_config(
                cli_overrides={"environment.channel_ablation_mode": "invalid"}
            )

    def test_oracle_defaults_are_disabled_and_zero_rate(self) -> None:
        config = load_run_config()
        self.assertFalse(config.training.mappo.route_oracle_counterfactual_enabled)
        self.assertEqual(config.training.mappo.route_oracle_selection_rate_ppm, 0)

    def test_oracle_cli_override_accepts_boundary_rates(self) -> None:
        for selection_rate in (1, 1_000_000):
            with self.subTest(selection_rate=selection_rate):
                config = load_run_config(
                    cli_overrides={
                        "training.mappo.route_oracle_counterfactual_enabled": True,
                        "training.mappo.route_oracle_selection_rate_ppm": selection_rate,
                    }
                )
                self.assertTrue(config.training.mappo.route_oracle_counterfactual_enabled)
                self.assertEqual(config.training.mappo.route_oracle_selection_rate_ppm, selection_rate)
                resolved_mappo = config.resolved_dict()["training"]["mappo"]
                self.assertTrue(resolved_mappo["route_oracle_counterfactual_enabled"])
                self.assertEqual(resolved_mappo["route_oracle_selection_rate_ppm"], selection_rate)

    def test_oracle_enabled_zero_rate_fails_validation(self) -> None:
        with self.assertRaisesRegex(ConfigError, "positive selection rate"):
            load_run_config(
                cli_overrides={
                    "training.mappo.route_oracle_counterfactual_enabled": True,
                    "training.mappo.route_oracle_selection_rate_ppm": 0,
                }
            )

    def test_oracle_selection_rate_above_maximum_fails_validation(self) -> None:
        with self.assertRaisesRegex(ConfigError, "must be an integer in"):
            load_run_config(
                cli_overrides={
                    "training.mappo.route_oracle_counterfactual_enabled": True,
                    "training.mappo.route_oracle_selection_rate_ppm": 1_000_001,
                }
            )

    def test_unknown_oracle_config_path_remains_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unknown override field"):
            load_run_config(
                cli_overrides={"training.mappo.nonexistent_oracle_field": True}
            )

    def test_oracle_settings_have_scoped_config_identity(self) -> None:
        default = load_run_config()
        explicit_disabled = load_run_config(
            cli_overrides={
                "training.mappo.route_oracle_counterfactual_enabled": False,
                "training.mappo.route_oracle_selection_rate_ppm": 0,
            }
        )
        enabled_low = load_run_config(
            cli_overrides={
                "training.mappo.route_oracle_counterfactual_enabled": True,
                "training.mappo.route_oracle_selection_rate_ppm": 1,
            }
        )
        enabled_high = load_run_config(
            cli_overrides={
                "training.mappo.route_oracle_counterfactual_enabled": True,
                "training.mappo.route_oracle_selection_rate_ppm": 1_000_000,
            }
        )
        self.assertEqual(default.config_hash, explicit_disabled.config_hash)
        self.assertEqual(default.run_id, explicit_disabled.run_id)
        self.assertNotEqual(enabled_low.config_hash, enabled_high.config_hash)
        self.assertNotEqual(enabled_low.run_id, enabled_high.run_id)
        self.assertNotEqual(default.config_hash, enabled_low.config_hash)

    def test_file_cli_and_interactive_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps({"mode": "random", "seed": 9, "scenario_id": "medium"}),
                encoding="utf-8",
            )
            config = load_run_config(
                path,
                cli_overrides={"seed": 11},
                interactive_overrides={"scenario_id": "large"},
            )
        self.assertEqual(config.mode, "random")
        self.assertEqual(config.method_id, "random")
        self.assertEqual(config.seed, 11)
        self.assertEqual(config.scenario_id, "large")
        self.assertEqual(config.environment.uav_count, 8)
        self.assertEqual(config.source_config_path, str(path))

    def test_unknown_field_is_rejected_before_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"not_a_field": 1}), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_run_config(path)

    def test_launch_profile_is_validated_and_changes_canonical_identity(self) -> None:
        shared = {
            "mode": "rl",
            "method_id": "ca_gat_mappo",
            "training.formal_rl_enabled": True,
        }
        unprofiled = load_run_config(cli_overrides=shared)
        formal = load_run_config(cli_overrides={**shared, "launch_profile": "rl-formal"})
        formal_snapshot = formal.snapshot_dict()

        self.assertIsNone(unprofiled.snapshot_dict()["launch_profile"])
        self.assertEqual(formal.launch_profile, "rl-formal")
        self.assertEqual(formal_snapshot["launch_profile"], "rl-formal")
        self.assertEqual(formal_snapshot["_metadata"]["config_hash"], formal.config_hash)
        self.assertNotEqual(unprofiled.config_hash, formal.config_hash)
        self.assertNotEqual(unprofiled.run_id, formal.run_id)
        with self.assertRaisesRegex(ConfigError, "launch_profile"):
            load_run_config(cli_overrides={"launch_profile": "pilot"})

    def test_cuda_provenance_parses_annotated_torch_version_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package_root = Path(directory) / "torch"
            package_root.mkdir()
            package_init = package_root / "__init__.py"
            package_init.write_text("", encoding="utf-8")
            (package_root / "version.py").write_text(
                "__version__: str = '2.8.0+cu128'\n"
                "cuda: str | None = '12.8'\n",
                encoding="utf-8",
            )
            fake_spec = SimpleNamespace(origin=str(package_init))
            _detected_runtime_versions.cache_clear()
            try:
                with patch("src.config.importlib.util.find_spec", return_value=fake_spec):
                    versions = _detected_runtime_versions()
            finally:
                _detected_runtime_versions.cache_clear()
        self.assertEqual(versions["torch"], "2.8.0+cu128")
        self.assertEqual(versions["cuda"], "12.8")


class RunnerAndCliTests(unittest.TestCase):
    def _run_stubbed_training_handler(self, config):
        training_result = SimpleNamespace(
            total_environment_transitions=500_000,
            completed_episode_count=1_000,
            ppo_update_count=1_953,
        )
        models_package = ModuleType("src.models")
        models_package.__path__ = []
        trainer_module = ModuleType("src.models.ca_gat_mappo_trainer")
        trainer_type = Mock()
        trainer_type.return_value.train.return_value = training_result
        trainer_type.return_value.train_with_checkpoints.return_value = training_result
        trainer_module.CAGATMAPPOTrainer = trainer_type
        artifacts_module = ModuleType("src.training_artifacts")
        artifact_writer = Mock(
            return_value=SimpleNamespace(
                artifacts=(),
                smoke_gate_status="pass",
                signal_gate_status="signal-pass",
                reward_mean=0.1,
                actor_loss=-0.2,
                critic_loss=0.3,
                entropy=0.4,
            )
        )
        artifacts_module.write_cagat_mappo_training_artifacts = artifact_writer
        with patch.dict(
            sys.modules,
            {
                "src.models": models_package,
                "src.models.ca_gat_mappo_trainer": trainer_module,
                "src.training_artifacts": artifacts_module,
            },
        ):
            result = ca_gat_mappo_training_handler(config)

        self.assertEqual(result.status, "completed")
        trainer_type.assert_called_once_with(config)
        artifact_writer.assert_called_once_with(config, training_result)
        return trainer_type

    def test_local_only_baseline_config_registry_and_menu(self) -> None:
        default = load_run_config(cli_overrides={"mode": "baseline"})
        explicit = load_run_config(cli_overrides={
            "mode": "baseline",
            "method_id": "local_only",
        })
        qmix = load_run_config(cli_overrides={
            "mode": "baseline",
            "method_id": "factorized_action_gat_qmix",
        })
        registry = build_default_registry()
        self.assertEqual(default.method_id, "local_only")
        self.assertEqual(explicit.method_id, "local_only")
        self.assertEqual(MENU_ROUTES["5"], ("baseline", "local_only"))
        self.assertEqual(
            registry.resolve("baseline", "local_only").__name__,
            "local_only_rollout_handler",
        )
        self.assertEqual(Runner().run(qmix).status, "unavailable")
        self.assertEqual(
            registry.resolve("random", "random").__name__,
            "random_rollout_handler",
        )
        self.assertEqual(
            registry.resolve("heuristic", "heuristic").__name__,
            "heuristic_rollout_handler",
        )
        self.assertEqual(
            registry.resolve("rl", "ca_gat_mappo").__name__,
            "ca_gat_mappo_training_handler",
        )

    def test_direct_and_menu_local_only_paths_use_common_runner(self) -> None:
        direct_output: list[str] = []
        direct_status = main([
            "--mode", "baseline",
            "--method-id", "local_only",
            "--scenario-id", "small",
            "--seed", "42",
            "--show-config",
        ], output_fn=direct_output.append)
        self.assertEqual(direct_status, 0)
        direct_config = json.loads(direct_output[-1])
        self.assertEqual(direct_config["mode"], "baseline")
        self.assertEqual(direct_config["method_id"], "local_only")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "local_only.json"
            path.write_text(json.dumps({
                "environment": {
                    "episode_horizon": 3,
                    "arrival_probabilities": [1.0, 1.0, 1.0, 1.0],
                },
                "output": {
                    "logs_dir": str(root / "logs"),
                    "dashboard_logs_dir": str(root / "dashboard_logs"),
                    "plots_dir": str(root / "plots"),
                },
            }), encoding="utf-8")
            answers = iter(("5", "small", str(path), "42"))
            interactive_output: list[str] = []
            interactive_status = main(
                [],
                input_fn=lambda _prompt: next(answers),
                output_fn=interactive_output.append,
            )
        self.assertEqual(interactive_status, 0)
        self.assertTrue(any(
            "mode=baseline, method_id=local_only" in line
            for line in interactive_output
        ))
        self.assertTrue(any(
            "status=completed" in line for line in interactive_output
        ))

    def test_disabled_qmix_runner_does_not_fabricate_artifacts(self) -> None:
        config = load_run_config(cli_overrides={
            "mode": "rl",
            "method_id": "factorized_action_gat_qmix",
        })
        result = Runner().run(config)
        self.assertEqual(result.status, "failed")
        self.assertIn("formal RL training is disabled", result.message)
        for artifact in config.artifact_paths().values():
            self.assertFalse(Path(artifact).exists())

        output: list[str] = []
        status = main(
            ["--mode", "rl", "--method-id", "factorized_action_gat_qmix"],
            output_fn=output.append,
        )
        self.assertEqual(status, 1)
        self.assertTrue(any("status=failed" in line for line in output))

    def test_rl_smoke_and_formal_profiles_resolve_frozen_devices_and_budgets(self) -> None:
        smoke_output: list[str] = []
        self.assertEqual(
            main(["--profile", "rl-smoke", "--show-config"], output_fn=smoke_output.append),
            0,
        )
        smoke = json.loads(smoke_output[-1])
        self.assertEqual((smoke["mode"], smoke["method_id"]), ("rl", "ca_gat_mappo"))
        self.assertTrue(smoke["training"]["formal_rl_enabled"])
        self.assertEqual(smoke["training"]["mappo"]["training_device"], "cpu")
        self.assertEqual(smoke["training"]["mappo"]["max_training_environment_steps"], 256)
        self.assertEqual(smoke["training"]["mappo"]["rollout_length_slots"], 256)

        long_smoke_output: list[str] = []
        self.assertEqual(
            main(
                ["--profile", "rl-long-smoke", "--show-config"],
                output_fn=long_smoke_output.append,
            ),
            0,
        )
        long_smoke = json.loads(long_smoke_output[-1])
        self.assertEqual(
            (
                long_smoke["mode"],
                long_smoke["method_id"],
                long_smoke["scenario_id"],
                long_smoke["seed"],
            ),
            ("rl", "ca_gat_mappo", "small", 42),
        )
        self.assertTrue(long_smoke["training"]["formal_rl_enabled"])
        self.assertEqual(long_smoke["environment"]["episode_horizon"], 500)
        long_smoke_mappo = long_smoke["training"]["mappo"]
        self.assertEqual(long_smoke_mappo["training_device"], "cuda")
        self.assertEqual(long_smoke_mappo["max_training_episodes"], 100)
        self.assertEqual(long_smoke_mappo["max_training_environment_steps"], 50000)
        self.assertEqual(long_smoke_mappo["evaluation_interval_steps"], 50000)
        self.assertEqual(long_smoke_mappo["checkpoint_interval_steps"], 2500)
        self.assertEqual(long_smoke_mappo["rollout_length_slots"], 256)
        self.assertEqual(long_smoke_mappo["recurrent_chunk_length_slots"], 32)
        self.assertEqual(long_smoke_mappo["sequence_minibatch_size"], 8)
        self.assertEqual(long_smoke_mappo["update_epochs"], 4)
        self.assertEqual(long_smoke_mappo["ppo_clip_epsilon"], 0.2)
        self.assertEqual(long_smoke_mappo["gamma"], 0.99)
        self.assertEqual(long_smoke_mappo["gae_lambda"], 0.95)
        self.assertEqual(long_smoke_mappo["gradient_clip_norm"], 0.5)

        formal_output: list[str] = []
        self.assertEqual(
            main(["--profile", "rl-formal", "--show-config"], output_fn=formal_output.append),
            0,
        )
        formal = json.loads(formal_output[-1])
        self.assertTrue(formal["training"]["formal_rl_enabled"])
        self.assertEqual(formal["training"]["mappo"]["training_device"], "cuda")
        self.assertEqual(formal["training"]["mappo"]["max_training_environment_steps"], 500000)

    def test_rl_profile_aliases_normalize_to_canonical_config_and_hash(self) -> None:
        aliases = (
            ("smoke", "rl-smoke"),
            ("long-smoke", "rl-long-smoke"),
            ("formal", "rl-formal"),
        )
        for alias, canonical in aliases:
            with self.subTest(alias=alias):
                snapshots = []
                for spelling in (alias, canonical):
                    output: list[str] = []
                    self.assertEqual(
                        main(["--profile", spelling, "--show-config"], output_fn=output.append),
                        0,
                    )
                    snapshot = json.loads(output[-1])
                    self.assertEqual(snapshot["launch_profile"], canonical)
                    snapshots.append(snapshot)
                self.assertEqual(
                    snapshots[0]["_metadata"]["config_hash"],
                    snapshots[1]["_metadata"]["config_hash"],
                )
                for snapshot in snapshots:
                    snapshot.pop("_metadata")
                self.assertEqual(snapshots[0], snapshots[1])

    def test_runner_forwards_resume_context_without_changing_config(self) -> None:
        config = load_run_config(cli_overrides={
            "mode": "rl",
            "method_id": "ca_gat_mappo",
            "launch_profile": "rl-formal",
            "training.formal_rl_enabled": True,
        })
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "periodic.pt"
            checkpoint.write_bytes(b"stub checkpoint")
            context = ExecutionContext.from_resume_path(checkpoint)
            handler = Mock(return_value=RunResult(
                status="completed",
                run_id=config.run_id,
                mode=config.mode,
                method_id=config.method_id,
                message="stubbed resume",
            ))
            registry = Mock()
            registry.resolve.return_value = handler

            result = Runner(registry=registry).run(
                config,
                execution_context=context,
            )

        self.assertEqual(result.status, "completed")
        registry.resolve.assert_called_once_with(config.mode, config.method_id)
        handler.assert_called_once_with(config, execution_context=context)

    def test_resume_parser_and_config_identity_exclude_resume_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "periodic.pt"
            checkpoint.write_bytes(b"stub checkpoint")
            parser = build_arg_parser()
            fresh_args = parser.parse_args(["--profile", "rl-formal"])
            resumed_args = parser.parse_args([
                "--profile",
                "rl-formal",
                "--resume-from",
                str(checkpoint),
            ])

            self.assertEqual(resumed_args.resume_from, str(checkpoint))
            fresh = build_run_config_from_args(fresh_args)
            resumed = build_run_config_from_args(resumed_args)

        self.assertEqual(fresh.config_hash, resumed.config_hash)
        self.assertEqual(fresh.run_id, resumed.run_id)
        self.assertNotIn("resume_from", resumed.resolved_dict())
        self.assertNotIn(str(checkpoint), json.dumps(resumed.snapshot_dict()))
        self.assertNotIn("resume_from", resumed.snapshot_dict()["_metadata"]["cli_overrides"])

    def test_resume_is_allowed_for_formal_profile_aliases(self) -> None:
        for profile in ("formal", "rl-formal"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "periodic.pt"
                checkpoint.write_bytes(b"stub checkpoint")
                args = build_arg_parser().parse_args([
                    "--profile",
                    profile,
                    "--resume-from",
                    str(checkpoint),
                ])
                config = build_run_config_from_args(args)
                self.assertEqual(config.launch_profile, "rl-formal")
                training_result = SimpleNamespace(
                    total_environment_transitions=500_000,
                    completed_episode_count=1_000,
                    ppo_update_count=1_953,
                )
                models_package = ModuleType("src.models")
                models_package.__path__ = []
                trainer_module = ModuleType("src.models.ca_gat_mappo_trainer")
                trainer_type = Mock()
                resumed_trainer = Mock()
                resumed_trainer.train_with_checkpoints.return_value = training_result
                trainer_type.resume_from_checkpoint.return_value = resumed_trainer
                trainer_module.CAGATMAPPOTrainer = trainer_type
                artifacts_module = ModuleType("src.training_artifacts")
                artifact_writer = Mock(return_value=SimpleNamespace(
                    artifacts=(),
                    smoke_gate_status="pass",
                    signal_gate_status="signal-pass",
                    reward_mean=0.1,
                    actor_loss=-0.2,
                    critic_loss=0.3,
                    entropy=0.4,
                ))
                artifacts_module.write_cagat_mappo_training_artifacts = artifact_writer
                with patch.dict(sys.modules, {
                    "src.models": models_package,
                    "src.models.ca_gat_mappo_trainer": trainer_module,
                    "src.training_artifacts": artifacts_module,
                }):
                    result = ca_gat_mappo_training_handler(
                        config,
                        execution_context=ExecutionContext.from_resume_path(checkpoint),
                    )

                self.assertEqual(result.status, "completed")
                trainer_type.assert_not_called()
                trainer_type.resume_from_checkpoint.assert_called_once_with(config, checkpoint)
                resumed_trainer.train.assert_not_called()
                resumed_trainer.train_with_checkpoints.assert_called_once_with()
                artifact_writer.assert_called_once_with(config, training_result)

    def test_resume_is_rejected_for_non_formal_profiles_and_methods(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "periodic.pt"
            checkpoint.write_bytes(b"stub checkpoint")
            rejected = (
                (["--profile", "smoke"], "launch_profile"),
                (["--profile", "rl-smoke"], "launch_profile"),
                (["--profile", "long-smoke"], "launch_profile"),
                (["--profile", "rl-long-smoke"], "launch_profile"),
                (["--mode", "rl", "--method-id", "ca_gat_mappo", "--set", "training.formal_rl_enabled=true"], "launch_profile"),
            )
            for base_args, expected_text in rejected:
                with self.subTest(base_args=base_args):
                    output: list[str] = []
                    status = main(
                        [*base_args, "--resume-from", str(checkpoint)],
                        output_fn=output.append,
                    )
                    self.assertEqual(status, 2)
                    self.assertTrue(any(expected_text in line for line in output))

    def test_resume_path_must_be_an_existing_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                root / "missing.pt",
                root / "checkpoint-directory",
            )
            cases[1].mkdir()
            for checkpoint in cases:
                with self.subTest(checkpoint=checkpoint):
                    output: list[str] = []
                    status = main(
                        ["--profile", "rl-formal", "--resume-from", str(checkpoint)],
                        output_fn=output.append,
                    )
                    self.assertEqual(status, 2)
                    self.assertTrue(any("checkpoint path" in line for line in output))

    def test_formal_handler_uses_checkpoint_training(self) -> None:
        config = load_run_config(
            cli_overrides={
                "mode": "rl",
                "method_id": "ca_gat_mappo",
                "launch_profile": "rl-formal",
                "training.formal_rl_enabled": True,
            }
        )

        trainer_type = self._run_stubbed_training_handler(config)

        trainer_type.return_value.train_with_checkpoints.assert_called_once_with()
        trainer_type.return_value.train.assert_not_called()

    def test_non_formal_handlers_keep_non_checkpoint_training(self) -> None:
        shared = {
            "mode": "rl",
            "method_id": "ca_gat_mappo",
            "training.formal_rl_enabled": True,
        }
        for launch_profile in ("rl-smoke", "rl-long-smoke", None):
            with self.subTest(launch_profile=launch_profile):
                overrides = dict(shared)
                if launch_profile is not None:
                    overrides["launch_profile"] = launch_profile
                config = load_run_config(cli_overrides=overrides)

                trainer_type = self._run_stubbed_training_handler(config)

                trainer_type.return_value.train.assert_called_once_with()
                trainer_type.return_value.train_with_checkpoints.assert_not_called()

    def test_root_main_long_smoke_show_config_reports_cuda(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "main.py"),
                "--profile",
                "rl-long-smoke",
                "--show-config",
            ],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=environment,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        config = json.loads(completed.stdout)
        self.assertEqual(config["training"]["mappo"]["training_device"], "cuda")

    def test_rl_menu_reaches_real_trainer_train_without_starting_training_in_gate(self) -> None:
        training_result = SimpleNamespace(
            total_environment_transitions=256,
            completed_episode_count=8,
            ppo_update_count=1,
        )
        answers = iter(("6", "smoke", "small", "", "42"))
        output: list[str] = []
        models_package = ModuleType("src.models")
        models_package.__path__ = []
        trainer_module = ModuleType("src.models.ca_gat_mappo_trainer")
        trainer_type = Mock()
        trainer_type.return_value.train.return_value = training_result
        trainer_module.CAGATMAPPOTrainer = trainer_type
        artifacts_module = ModuleType("src.training_artifacts")
        artifact_writer = Mock(
            return_value=SimpleNamespace(
                artifacts=(),
                smoke_gate_status="pass",
                signal_gate_status="insufficient-horizon",
                reward_mean=-0.1,
                actor_loss=-0.2,
                critic_loss=0.3,
                entropy=0.4,
            )
        )
        artifacts_module.write_cagat_mappo_training_artifacts = artifact_writer
        with patch.dict(
            sys.modules,
            {
                "src.models": models_package,
                "src.models.ca_gat_mappo_trainer": trainer_module,
                "src.training_artifacts": artifacts_module,
            },
        ):
            status = main(
                [],
                input_fn=lambda _prompt: next(answers),
                output_fn=output.append,
            )

        self.assertEqual(status, 0)
        trainer_type.assert_called_once()
        trainer_type.return_value.train.assert_called_once_with()
        trainer_type.return_value.train_with_checkpoints.assert_not_called()
        dispatched = trainer_type.call_args.args[0]
        artifact_writer.assert_called_once_with(dispatched, training_result)
        self.assertEqual((dispatched.mode, dispatched.method_id), ("rl", "ca_gat_mappo"))
        self.assertEqual(dispatched.launch_profile, "rl-smoke")
        self.assertTrue(dispatched.training.formal_rl_enabled)
        self.assertEqual(dispatched.training.mappo.training_device, "cpu")
        self.assertTrue(any("requested=cpu, resolved=cpu" in line for line in output))
        self.assertTrue(any("status=completed" in line for line in output))

    def test_rl_trainer_failure_returns_nonzero_exit_code(self) -> None:
        models_package = ModuleType("src.models")
        models_package.__path__ = []
        trainer_module = ModuleType("src.models.ca_gat_mappo_trainer")
        trainer_type = Mock(side_effect=RuntimeError("gate failure"))
        trainer_module.CAGATMAPPOTrainer = trainer_type
        output: list[str] = []
        with patch.dict(
            sys.modules,
            {
                "src.models": models_package,
                "src.models.ca_gat_mappo_trainer": trainer_module,
            },
        ):
            status = main(["--profile", "rl-smoke"], output_fn=output.append)

        self.assertEqual(status, 1)
        self.assertTrue(any("status=failed" in line for line in output))
        self.assertTrue(any("gate failure" in line for line in output))

    def test_direct_cli_uses_common_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "random.json"
            path.write_text(json.dumps({
                "environment": {
                    "episode_horizon": 3,
                    "arrival_probabilities": [1.0, 1.0, 1.0, 1.0],
                },
                "output": {
                    "logs_dir": str(root / "logs"),
                    "dashboard_logs_dir": str(root / "dashboard_logs"),
                    "plots_dir": str(root / "plots"),
                },
            }), encoding="utf-8")
            output: list[str] = []
            status = main(
                ["--mode", "random", "--config", str(path), "--seed", "7"],
                output_fn=output.append,
            )
            self.assertEqual(status, 0)
            self.assertTrue(any("mode=random" in line for line in output))
            self.assertTrue(any("status=completed" in line for line in output))
            self.assertTrue(any("raw_metrics.jsonl" in line for line in output))

    def test_real_no_argument_path_can_exit_from_menu(self) -> None:
        answers = iter(["0"])
        output: list[str] = []
        status = main([], input_fn=lambda _prompt: next(answers), output_fn=output.append)
        self.assertEqual(status, 0)
        self.assertTrue(any("U2U-MEC Experiment Launcher" in line for line in output))


if __name__ == "__main__":
    unittest.main()
