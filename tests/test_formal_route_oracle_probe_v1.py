from __future__ import annotations

from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

from src.cli import (
    build_arg_parser,
    build_execution_context_from_args,
    build_run_config_from_args,
)
from src.config import ConfigError
from src.execution import ExecutionContext, validate_execution_context
from src.evaluation.route_oracle_probe import (
    FORMAL_ROUTE_ORACLE_PERFORMANCE_SCHEMA_VERSION,
    FormalRouteOracleProbeRunner,
    FormalRouteStateCandidate,
    _formal_selection_key,
    _sha256_json,
    aggregate_formal_gate_status,
    build_conditional_statistics,
    calculate_formal_oracle,
    select_formal_route_states,
)


def candidate(
    *,
    seed: int,
    episode: int,
    slot: int,
    uav: int,
    key: str,
) -> FormalRouteStateCandidate:
    return FormalRouteStateCandidate(
        episode_id=episode,
        evaluation_seed=seed,
        slot=slot,
        uav_id=uav,
        task_id=1000 + slot * 10 + uav,
        observation_hash=f"observation-{key}",
        joint_observation_hash=f"joint-{seed}-{slot}",
        hidden_in_hash=f"hidden-in-{seed}-{slot}",
        hidden_hash=f"hidden-{seed}-{slot}",
        policy_action="local",
        legal_routes=("local", 1),
        selection_key=key,
    )


class FormalSamplerTests(unittest.TestCase):
    def test_sampler_is_deterministic_balanced_and_one_uav_per_root(self) -> None:
        candidates = (
            candidate(seed=1042, episode=0, slot=1, uav=0, key="d"),
            candidate(seed=1042, episode=0, slot=1, uav=1, key="a"),
            candidate(seed=1042, episode=0, slot=2, uav=0, key="b"),
            candidate(seed=1042, episode=0, slot=3, uav=0, key="c"),
            candidate(seed=1043, episode=1, slot=1, uav=0, key="e"),
            candidate(seed=1043, episode=1, slot=2, uav=0, key="f"),
            candidate(seed=1043, episode=1, slot=3, uav=0, key="g"),
        )
        first, metadata = select_formal_route_states(
            candidates,
            target_count=4,
            evaluation_seeds=(1042, 1043),
        )
        second, _ = select_formal_route_states(
            tuple(reversed(candidates)),
            target_count=4,
            evaluation_seeds=(1042, 1043),
        )
        self.assertEqual(
            [item.selection_key for item in first],
            [item.selection_key for item in second],
        )
        self.assertEqual(len({item.root_key for item in first}), len(first))
        self.assertIn("a", {item.selection_key for item in first})
        self.assertNotIn("d", {item.selection_key for item in first})
        self.assertEqual(metadata["same_root_candidates_removed"], 1)
        self.assertEqual(metadata["per_seed"]["1042"]["selected_count"], 2)
        self.assertEqual(metadata["per_seed"]["1043"]["selected_count"], 2)
        self.assertEqual(
            metadata["selected_state_ids"],
            ["a", "b", "e", "f"],
        )

    def test_sampler_reports_incomplete_selection(self) -> None:
        selected, metadata = select_formal_route_states(
            (candidate(seed=1042, episode=0, slot=1, uav=0, key="a"),),
            target_count=2,
            evaluation_seeds=(1042,),
        )
        self.assertEqual(len(selected), 1)
        self.assertFalse(metadata["selection_complete"])

    def test_state_hash_and_sampling_id_golden_values(self) -> None:
        self.assertEqual(
            _sha256_json({"slot": 2, "uav_id": 1, "task_id": 2}),
            "49179d319a1813ab011c76f09b51381e199f922911d3e0fc8740603888662355",
        )
        self.assertEqual(
            _formal_selection_key(
                {
                    "checkpoint_sha256": "checkpoint",
                    "resolved_config_hash": "config",
                    "episode": 0,
                    "evaluation_seed": 1042,
                    "slot": 2,
                    "uav_id": 1,
                    "task_id": 2,
                    "observation_hash": "observation",
                    "hidden_hash": "hidden",
                }
            ),
            "4b85238c6b5bbf05cfa3de8f35d1e4df3b3f8d6ac0f176efd423c8729436a965",
        )


class FormalOracleMetricTests(unittest.TestCase):
    def test_oracle_best_action_and_regret_use_route_return(self) -> None:
        result = calculate_formal_oracle(
            {"local": 1.25, 1: 2.0, 2: 1.5},
            policy_action="local",
        )
        self.assertEqual(result["oracle_best_action"], 1)
        self.assertEqual(result["oracle_best_action_set"], [1])
        self.assertAlmostEqual(result["regret"], 0.75)
        self.assertTrue(result["oracle_remote_better"])
        self.assertTrue(result["policy_miss"])
        self.assertFalse(result["tie"])

    def test_tie_has_no_unique_best_and_policy_in_set_is_not_miss(self) -> None:
        result = calculate_formal_oracle(
            {"local": 2.0, 1: 2.0 + 5.0e-10, 2: 1.0},
            policy_action="local",
        )
        self.assertIsNone(result["oracle_best_action"])
        self.assertEqual(result["oracle_best_action_set"], ["local", 1])
        self.assertTrue(result["tie"])
        self.assertTrue(result["local_remote_tie"])
        self.assertFalse(result["policy_miss"])


class FormalGateAndConditionalTests(unittest.TestCase):
    @staticmethod
    def state(value: float, *, remote: bool, better: bool, miss: bool):
        return {
            "statistics_valid": True,
            "policy_is_remote": remote,
            "oracle_remote_better": better,
            "policy_miss": miss,
            "condition_features": {
                "local_backlog": value,
                "deadline_slack": value,
                "helper_queue": value,
                "channel_quality": value,
            },
            "gates": {
                "same_root": True,
                "only_route_diff": True,
                "same_hidden": True,
                "crn": True,
                "repeatability": True,
            },
        }

    def test_gate_aggregation_is_fail_closed(self) -> None:
        states = [self.state(1.0, remote=False, better=True, miss=True)]
        passed = aggregate_formal_gate_status(
            states,
            requested_state_count=1,
            actor_weights_unchanged=True,
            sampling_replay_match=True,
        )
        self.assertEqual(passed["overall"], "PASS")
        failed = aggregate_formal_gate_status(
            states,
            requested_state_count=2,
            actor_weights_unchanged=True,
            sampling_replay_match=True,
        )
        self.assertEqual(failed["overall"], "FAIL")
        self.assertFalse(failed["sample_complete"])

    def test_conditional_bins_report_required_rates(self) -> None:
        states = (
            self.state(1.0, remote=False, better=True, miss=True),
            self.state(2.0, remote=True, better=True, miss=False),
            self.state(3.0, remote=False, better=False, miss=False),
            self.state(4.0, remote=True, better=False, miss=True),
        )
        statistics = build_conditional_statistics(states)
        self.assertEqual(statistics["valid_state_count"], 4)
        bins = statistics["dimensions"]["local_backlog"]["bins"]
        self.assertEqual(sum(item["sample_count"] for item in bins), 4)
        for item in bins:
            self.assertIn("policy_remote_ratio", item)
            self.assertIn("oracle_remote_better_ratio", item)
            self.assertIn("miss_rate", item)


class FormalProbeCliCompatibilityTests(unittest.TestCase):
    @staticmethod
    def parse(*extra: str):
        return build_arg_parser().parse_args(
            [
                "--mode",
                "evaluation",
                "--method-id",
                "ca_gat_mappo",
                "--evaluate-from",
                "main.py",
                "--evaluation-device",
                "cpu",
                "--oracle-probe",
                *extra,
            ]
        )

    def test_smoke_defaults_remain_unchanged(self) -> None:
        context = build_execution_context_from_args(self.parse())
        self.assertEqual(context.oracle_probe_mode, "smoke")
        self.assertEqual(context.oracle_probe_states, 3)

    def test_formal_defaults_to_two_hundred_states(self) -> None:
        context = build_execution_context_from_args(
            self.parse("--oracle-probe-mode", "formal")
        )
        self.assertEqual(context.oracle_probe_mode, "formal")
        self.assertEqual(context.oracle_probe_states, 200)

    def test_formal_accepts_explicit_supported_count(self) -> None:
        context = build_execution_context_from_args(
            self.parse(
                "--oracle-probe-mode",
                "formal",
                "--oracle-probe-states",
                "100",
            )
        )
        self.assertEqual(context.oracle_probe_states, 100)

    def test_formal_validation_rejects_cuda_and_out_of_range_count(self) -> None:
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
        with self.assertRaisesRegex(ConfigError, "requires --evaluation-device cpu"):
            validate_execution_context(
                config,
                ExecutionContext.from_evaluation_path(
                    "main.py",
                    "cuda",
                    oracle_probe_states=100,
                    oracle_probe_mode="formal",
                ),
            )
        with self.assertRaisesRegex(ConfigError, r"\[100, 300\]"):
            validate_execution_context(
                config,
                ExecutionContext.from_evaluation_path(
                    "main.py",
                    "cpu",
                    oracle_probe_states=99,
                    oracle_probe_mode="formal",
                ),
            )


class FormalProbeObservabilityTests(unittest.TestCase):
    @staticmethod
    def bare_runner() -> FormalRouteOracleProbeRunner:
        runner = FormalRouteOracleProbeRunner.__new__(FormalRouteOracleProbeRunner)
        runner.probe_id = "formal-route-oracle-v1__test"
        runner.state_count = 25
        runner._branch_rollout_times = [2.0, 4.0]
        runner._stage_a_actor_call_count = 2500
        runner._stage_b_factual_actor_call_count = 2500
        runner._shadow_actor_call_count = 12
        runner._preprocessing_cache_hits = 7
        runner._preprocessing_cache_misses = 5
        runner._stage_b_started_at = 100.0
        return runner

    def test_progress_logging_uses_fixed_interval_and_final_state(self) -> None:
        runner = self.bare_runner()
        output = io.StringIO()
        with (
            patch(
                "src.evaluation.route_oracle_probe.time.perf_counter",
                side_effect=(110.0, 125.0),
            ),
            redirect_stdout(output),
        ):
            runner._log_state_progress(9)
            runner._log_state_progress(10)
            runner._log_state_progress(25)
        lines = output.getvalue().strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("10/25 completed", lines[0])
        self.assertIn("elapsed=10.0s", lines[0])
        self.assertIn("avg=1.000s/state", lines[0])
        self.assertIn("eta=15.0s", lines[0])
        self.assertIn("25/25 completed", lines[1])
        self.assertIn("eta=0.0s", lines[1])

    def test_performance_metadata_is_separate_and_complete(self) -> None:
        runner = self.bare_runner()
        with patch(
            "src.evaluation.route_oracle_probe.time.perf_counter",
            return_value=130.0,
        ):
            metadata = runner._performance_report(
                run_started_at=90.0,
                stage_a_runtime_sec=20.0,
                stage_b_runtime_sec=10.0,
                states_completed=2,
            )
        self.assertEqual(
            metadata["schema_version"],
            FORMAL_ROUTE_ORACLE_PERFORMANCE_SCHEMA_VERSION,
        )
        self.assertEqual(metadata["total_runtime_sec"], 40.0)
        self.assertEqual(metadata["states_completed"], 2)
        self.assertEqual(metadata["avg_seconds_per_state"], 5.0)
        self.assertEqual(metadata["branch_rollout_count"], 2)
        self.assertEqual(metadata["avg_branch_rollout_time_sec"], 3.0)
        self.assertEqual(metadata["avg_actor_call_count"], 6.0)
        self.assertAlmostEqual(metadata["preprocessing_cache_hit_rate"], 7.0 / 12.0)


if __name__ == "__main__":
    unittest.main()
