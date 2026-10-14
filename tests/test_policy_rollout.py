"""Implementation 06 tests for masked random policy and unified rollout."""

from __future__ import annotations

import csv
import inspect
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.cli import main as cli_main
from src.config import STREAM_IDS, load_run_config
from src.env.actions import ActionProposal
from src.env.environment import U2UMECEnvironment
from src.env.observation import ActionMasks
from src.policies.heuristic_policy import (
    HeuristicPolicy,
    masked_historical_quality,
)
from src.policies.local_only_policy import LocalOnlyPolicy
from src.policies.random_policy import RandomPolicy
from src.registry import build_default_registry
from src.runner import Runner
from src.rollout import DASHBOARD_COLUMNS, RolloutRunner


BRANCH_ORDER = (
    "route",
    "tx_select",
    "resource_group",
    "resource_width",
    "power_level",
    "cpu_queue",
    "cpu_frequency",
)


def make_random_config(
    *,
    horizon: int = 8,
    arrivals: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
    output_root: Path | None = None,
):
    overrides: dict[str, object] = {
        "mode": "random",
        "method_id": "random",
        "scenario_id": "small",
        "seed": 42,
        "environment.episode_horizon": horizon,
        "environment.arrival_probabilities": list(arrivals),
    }
    if output_root is not None:
        overrides.update({
            "output.logs_dir": str(output_root / "logs"),
            "output.dashboard_logs_dir": str(output_root / "dashboard_logs"),
            "output.plots_dir": str(output_root / "plots"),
        })
    return load_run_config(cli_overrides=overrides)


def make_heuristic_config(
    *,
    horizon: int = 8,
    arrivals: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    output_root: Path | None = None,
):
    overrides: dict[str, object] = {
        "mode": "heuristic",
        "method_id": "heuristic",
        "scenario_id": "small",
        "seed": 42,
        "environment.episode_horizon": horizon,
        "environment.arrival_probabilities": list(arrivals),
    }
    if output_root is not None:
        overrides.update({
            "output.logs_dir": str(output_root / "logs"),
            "output.dashboard_logs_dir": str(output_root / "dashboard_logs"),
            "output.plots_dir": str(output_root / "plots"),
        })
    return load_run_config(cli_overrides=overrides)


def make_local_only_config(
    *,
    horizon: int = 8,
    arrivals: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
):
    return load_run_config(cli_overrides={
        "mode": "baseline",
        "method_id": "local_only",
        "scenario_id": "small",
        "seed": 42,
        "environment.episode_horizon": horizon,
        "environment.arrival_probabilities": list(arrivals),
    })


def make_heuristic_observation(
    *,
    unbound_slack: int | None = None,
    unbound_bits: float = 20.0,
    unbound_cycles: float = 20.0,
    local_backlog_cycles: float = 0.0,
    route_remotes: tuple[int, ...] = (),
    tx_candidates: tuple[int, ...] = (),
    tx_slacks: dict[int, int] | None = None,
    cpu_candidates: tuple[int, ...] = (),
    cpu_slacks: dict[int, int] | None = None,
    cpu_task_ids: dict[int, int] | None = None,
    cpu_cycles: dict[int, float] | None = None,
    cpu_task_counts: dict[int, int] | None = None,
    cpu_total_cycles: dict[int, float] | None = None,
    quality_values: dict[int, tuple[float, ...]] | None = None,
    quality_masks: dict[int, tuple[bool, ...]] | None = None,
    destination_max_cpu_frequency_hz: dict[int, float] | None = None,
    destination_queued_cycles: dict[int, float] | None = None,
    destination_rates_bps: dict[int, float] | None = None,
    destination_public_valid: dict[int, bool] | None = None,
    destination_rate_valid: dict[int, bool] | None = None,
    destination_stale_csi: dict[int, complex] | None = None,
    destination_csi_valid: dict[int, bool] | None = None,
    power_legal: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0),
    cpu_frequency_legal: dict[int, tuple[float, ...]] | None = None,
    max_cpu_frequency_hz: float = 1000.0,
):
    config = make_heuristic_config()
    count = 4
    ru_count = 20
    route_domain = ("idle", "local", "defer", 1, 2, 3)
    tx_domain = ("idle", 1, 2, 3)
    group_domain = ("idle", 1, 2, 3, 4, 5)
    width_domain = (1, 2)
    power_domain = (0.0, 0.25, 0.5, 1.0)
    cpu_queue_domain = ("idle", 0, 1, 2, 3)
    cpu_frequency_domain = (0.0, 0.25, 0.5, 1.0)

    route_mask = np.zeros(len(route_domain), dtype=np.bool_)
    if unbound_slack is None:
        route_mask[0] = True
    else:
        route_mask[1:3] = True
        for destination in route_remotes:
            route_mask[route_domain.index(destination)] = True

    tx_mask = np.zeros(len(tx_domain), dtype=np.bool_)
    tx_mask[0] = True
    for destination in tx_candidates:
        tx_mask[tx_domain.index(destination)] = True

    cpu_queue_mask = np.zeros(len(cpu_queue_domain), dtype=np.bool_)
    cpu_queue_mask[0] = True
    for source in cpu_candidates:
        cpu_queue_mask[cpu_queue_domain.index(source)] = True

    power_mask = np.array(
        [level in power_legal for level in power_domain],
        dtype=np.bool_,
    )
    cpu_frequency_mask = np.ones(
        (count, len(cpu_frequency_domain)),
        dtype=np.bool_,
    )
    for source, legal_levels in (cpu_frequency_legal or {}).items():
        cpu_frequency_mask[source] = [
            level in legal_levels for level in cpu_frequency_domain
        ]

    masks = ActionMasks(
        uav_id=0,
        paper_uav_id=1,
        sampling_order=BRANCH_ORDER,
        route_domain=route_domain,
        tx_select_domain=tx_domain,
        resource_group_domain=group_domain,
        resource_width_domain=width_domain,
        power_level_domain=power_domain,
        cpu_queue_domain=cpu_queue_domain,
        cpu_frequency_domain=cpu_frequency_domain,
        route_mask=route_mask,
        tx_select_mask=tx_mask,
        cpu_queue_mask=cpu_queue_mask,
        power_energy_mask=power_mask,
        cpu_frequency_energy_mask=cpu_frequency_mask,
        route_branch_active=unbound_slack is not None,
        tx_branch_active=bool(tx_candidates),
        cpu_queue_branch_active=bool(cpu_candidates),
        resource_group_count=5,
        tx_idle_action="idle",
        cpu_idle_action="idle",
        canonical_width=1,
        canonical_power=0.0,
        canonical_cpu_frequency=0.0,
    )

    tx_slacks = tx_slacks or {}
    cpu_slacks = cpu_slacks or {}
    cpu_task_ids = cpu_task_ids or {}
    cpu_cycles = cpu_cycles or {}
    cpu_task_counts = cpu_task_counts or {}
    cpu_total_cycles = cpu_total_cycles or {}
    destination_max_cpu_frequency_hz = (
        destination_max_cpu_frequency_hz or {}
    )
    destination_queued_cycles = destination_queued_cycles or {}
    destination_rates_bps = destination_rates_bps or {}
    destination_public_valid = destination_public_valid or {}
    destination_rate_valid = destination_rate_valid or {}
    destination_stale_csi = destination_stale_csi or {}
    destination_csi_valid = destination_csi_valid or {}

    def indexed_queues(
        candidates: tuple[int, ...],
        slacks: dict[int, int],
        task_ids: dict[int, int],
        cycles: dict[int, float],
        task_counts: dict[int, int],
        total_cycles: dict[int, float],
    ):
        valid = np.zeros(count, dtype=np.bool_)
        counts = np.zeros(count, dtype=np.int64)
        aggregate_cycles = np.zeros(count, dtype=np.float64)
        head_slack = np.zeros(count, dtype=np.int64)
        head_task_id = np.full(count, -1, dtype=np.int64)
        head_cycles = np.zeros(count, dtype=np.float64)
        for source in candidates:
            valid[source] = True
            counts[source] = task_counts.get(source, 1)
            head_slack[source] = slacks.get(source, 3)
            head_task_id[source] = task_ids.get(source, source)
            head_cycles[source] = cycles.get(source, 20.0)
            aggregate_cycles[source] = total_cycles.get(
                source,
                head_cycles[source],
            )
        return SimpleNamespace(
            task_count=counts,
            remaining_cycles=aggregate_cycles,
            head_valid_mask=valid,
            head_slack_slots=head_slack,
            head_task_id=head_task_id,
            head_remaining_cycles=head_cycles,
        )

    quality = np.zeros((count, ru_count), dtype=np.float64)
    quality_valid = np.zeros((count, ru_count), dtype=np.bool_)
    public_valid = np.zeros(count, dtype=np.bool_)
    destination_cpu_ratio = np.zeros(count, dtype=np.float64)
    destination_cpu_cycles = np.zeros(count, dtype=np.float64)
    effective_rate = np.zeros(count, dtype=np.float64)
    effective_rate_valid = np.zeros(count, dtype=np.bool_)
    stale_csi = np.zeros((count, ru_count), dtype=np.complex128)
    csi_valid = np.zeros(count, dtype=np.bool_)
    for destination in route_remotes:
        public_valid[destination] = destination_public_valid.get(
            destination, True
        )
        destination_cpu_ratio[destination] = (
            destination_max_cpu_frequency_hz.get(
                destination, max_cpu_frequency_hz
            )
            / config.environment.reference_cpu_frequency_hz
        )
        destination_cpu_cycles[destination] = (
            destination_queued_cycles.get(destination, 0.0)
        )
        effective_rate[destination] = destination_rates_bps.get(
            destination, 1.0e9
        )
        effective_rate_valid[destination] = destination_rate_valid.get(
            destination, True
        )
        stale_csi[destination] = destination_stale_csi.get(
            destination,
            0.0j,
        )
        csi_valid[destination] = destination_csi_valid.get(
            destination,
            False,
        )
    for destination, values in (quality_values or {}).items():
        if len(values) != ru_count:
            raise ValueError("quality fixture must contain 20 RUs")
        quality[destination] = values
    for destination, values in (quality_masks or {}).items():
        if len(values) != ru_count:
            raise ValueError("quality-mask fixture must contain 20 RUs")
        quality_valid[destination] = values

    unbound = SimpleNamespace(
        head_valid_mask=unbound_slack is not None,
        head_slack_slots=0 if unbound_slack is None else unbound_slack,
        head_remaining_bits=0.0 if unbound_slack is None else unbound_bits,
        head_remaining_cycles=0.0 if unbound_slack is None else unbound_cycles,
    )
    observation = SimpleNamespace(
        uav_id=0,
        self_resources=SimpleNamespace(
            max_cpu_frequency_hz=max_cpu_frequency_hz,
        ),
        private_queues=SimpleNamespace(
            unbound=unbound,
            local_cpu=SimpleNamespace(
                remaining_cycles=local_backlog_cycles,
            ),
            tx_by_destination=indexed_queues(
                tx_candidates,
                tx_slacks,
                {},
                {},
                {},
                {},
            ),
            cpu_by_source=indexed_queues(
                cpu_candidates,
                cpu_slacks,
                cpu_task_ids,
                cpu_cycles,
                cpu_task_counts,
                cpu_total_cycles,
            ),
        ),
        neighbor_public=SimpleNamespace(
            valid_mask=public_valid,
            max_cpu_frequency_ratio=destination_cpu_ratio,
            cpu_load_remaining_cycles=destination_cpu_cycles,
        ),
        edge_history=SimpleNamespace(
            historical_quality=quality,
            quality_valid_mask=quality_valid,
            last_effective_rate_bps=effective_rate,
            last_rate_valid_mask=effective_rate_valid,
            stale_csi=stale_csi,
            csi_valid_mask=csi_valid,
        ),
        action_masks=masks,
    )
    return config, observation


def observations_with_active_queues():
    config = make_random_config(horizon=6)
    environment = U2UMECEnvironment(config)
    reset = environment.reset()
    environment.step(environment.canonical_proposals())
    assert environment.current_observations is not None

    proposals: list[ActionProposal] = []
    remote_used = False
    canonical = config.action.canonical_inactive_values
    for observation in environment.current_observations:
        masks = observation.action_masks
        legal_routes = [
            value
            for value, allowed in zip(masks.route_domain, masks.route_mask)
            if bool(allowed)
        ]
        remote = next((value for value in legal_routes if isinstance(value, int)), None)
        route = remote if remote is not None and not remote_used else "local"
        remote_used = remote_used or remote is not None
        proposals.append(ActionProposal(
            uav_id=observation.uav_id,
            route=route,
            tx_select=canonical["tx_select"],
            resource_group=canonical["resource_group"],
            resource_width=int(canonical["resource_width"]),
            power_level=float(canonical["power_level"]),
            cpu_queue=canonical["cpu_queue"],
            cpu_frequency=float(canonical["cpu_frequency"]),
        ))
    environment.step(proposals)
    assert environment.current_observations is not None
    return config, environment, environment.current_observations


def assert_finite_or_none(test: unittest.TestCase, value: object) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        test.assertTrue(math.isfinite(value))
        return
    if isinstance(value, dict):
        for nested in value.values():
            assert_finite_or_none(test, nested)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            assert_finite_or_none(test, nested)


class TestRandomPolicy(unittest.TestCase):
    def test_sampling_uses_exact_seven_branch_order(self) -> None:
        config = make_random_config(horizon=2, arrivals=(0.0, 0.0, 0.0, 0.0))
        observation = U2UMECEnvironment(config).reset().observations[0]
        original = observation.action_masks
        calls: list[str] = []

        class RecordingMasks:
            sampling_order = original.sampling_order

            @staticmethod
            def domain_for(branch: str):
                return original.domain_for(branch)

            @staticmethod
            def mask_for(branch: str, previous=None):
                calls.append(branch)
                return original.mask_for(branch, previous)

            @staticmethod
            def is_legal(proposal: ActionProposal) -> bool:
                return original.is_legal(proposal)

        actor_only = SimpleNamespace(
            uav_id=observation.uav_id,
            action_masks=RecordingMasks(),
        )
        proposal = RandomPolicy(42).act(actor_only)
        self.assertEqual(calls, list(BRANCH_ORDER))
        self.assertEqual(len(proposal.branches), 7)
        self.assertEqual(original.sampling_order, BRANCH_ORDER)

    def test_only_valid_masked_actions_are_sampled(self) -> None:
        _, _, observations = observations_with_active_queues()
        policy = RandomPolicy(42)
        for _ in range(64):
            for observation in observations:
                proposal = policy.act(observation)
                self.assertTrue(observation.action_masks.is_legal(proposal))

    def test_inactive_branches_use_frozen_canonical_values(self) -> None:
        config = make_random_config(horizon=1, arrivals=(0.0, 0.0, 0.0, 0.0))
        reset = U2UMECEnvironment(config).reset()
        canonical = config.action.canonical_inactive_values
        for observation in reset.observations:
            proposal = RandomPolicy(42).act(observation)
            self.assertEqual(proposal.route, canonical["route"])
            self.assertEqual(proposal.tx_select, canonical["tx_select"])
            self.assertEqual(proposal.resource_group, canonical["resource_group"])
            self.assertEqual(proposal.resource_width, canonical["resource_width"])
            self.assertEqual(proposal.power_level, canonical["power_level"])
            self.assertEqual(proposal.cpu_queue, canonical["cpu_queue"])
            self.assertEqual(proposal.cpu_frequency, canonical["cpu_frequency"])

    def test_same_seed_reproduces_and_different_seed_changes_trajectory(self) -> None:
        _, _, observations = observations_with_active_queues()
        first = RandomPolicy(42)
        second = RandomPolicy(42)
        different = RandomPolicy(43)
        first_trajectory = [
            first.act(observations[index % len(observations)]).branches
            for index in range(64)
        ]
        second_trajectory = [
            second.act(observations[index % len(observations)]).branches
            for index in range(64)
        ]
        different_trajectory = [
            different.act(observations[index % len(observations)]).branches
            for index in range(64)
        ]
        self.assertEqual(first_trajectory, second_trajectory)
        self.assertNotEqual(first_trajectory, different_trajectory)
        self.assertEqual(first.policy_stream_id, STREAM_IDS["torch_policy_sampling"])

    def test_policy_rng_consumption_does_not_change_environment_rng(self) -> None:
        config = make_random_config(horizon=3)
        first = U2UMECEnvironment(config)
        second = U2UMECEnvironment(config)
        first_reset = first.reset()
        second.reset()
        policy = RandomPolicy(42)
        for _ in range(100):
            for observation in first_reset.observations:
                policy.act(observation)
        first.step(first.canonical_proposals())
        second.step(second.canonical_proposals())
        self.assertEqual(first.snapshot(), second.snapshot())

    def test_policy_receives_no_centralized_or_post_action_argument(self) -> None:
        parameters = tuple(inspect.signature(RandomPolicy.act).parameters)
        self.assertEqual(parameters, ("self", "observation"))
        _, _, observations = observations_with_active_queues()
        observation = observations[0]
        actor_only = SimpleNamespace(
            uav_id=observation.uav_id,
            action_masks=observation.action_masks,
        )
        proposal = RandomPolicy(42).act(actor_only)
        self.assertTrue(observation.action_masks.is_legal(proposal))


class TestRolloutRunner(unittest.TestCase):
    def test_complete_episode_uses_real_backend_and_no_fake_terminal_state(self) -> None:
        config = make_random_config(horizon=12)
        outcome = RolloutRunner(config, RandomPolicy(42)).run(write_artifacts=False)
        self.assertEqual(len(outcome.raw_records), 12)
        self.assertEqual(outcome.artifacts, ())
        self.assertTrue(outcome.raw_records[-1]["truncated"])
        self.assertIsNone(outcome.raw_records[-1]["next_decision_slot"])
        self.assertFalse(outcome.raw_records[-1]["bootstrap_allowed"])
        self.assertFalse(outcome.summary["episode"]["fake_terminal_state_created"])
        self.assertTrue(outcome.summary["conservation"]["is_conserved"])
        self.assertGreater(outcome.summary["tasks"]["generated"], 0)
        assert_finite_or_none(self, outcome.summary)

    def test_na_semantics_are_preserved_without_samples(self) -> None:
        config = make_random_config(
            horizon=1,
            arrivals=(0.0, 0.0, 0.0, 0.0),
        )
        outcome = RolloutRunner(config, RandomPolicy(42)).run(write_artifacts=False)
        self.assertIsNone(outcome.summary["tasks"]["completion_rate"])
        self.assertIsNone(outcome.summary["tasks"]["expiration_rate"])
        self.assertIsNone(outcome.summary["e2e_latency_s"]["mean"])
        self.assertEqual(outcome.summary["e2e_latency_s"]["valid_sample_count"], 0)
        self.assertIsNone(outcome.summary["transmission"]["outage_rate"])
        self.assertEqual(outcome.summary["transmission"]["outage_valid_sample_count"], 0)

    def test_real_artifacts_have_fixed_schema_and_no_plot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_random_config(horizon=5, output_root=root)
            outcome = RolloutRunner(config, RandomPolicy(42)).run()
            self.assertEqual(len(outcome.artifacts), 4)
            for artifact in outcome.artifacts:
                self.assertTrue(Path(artifact).is_file())
            raw_path = Path(config.artifact_paths()["raw_metrics"])
            records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 5)
            self.assertEqual(records[0]["run_id"], config.run_id)
            self.assertEqual(records[0]["policy_seed"], 42)
            self.assertEqual(records[0]["policy_stream_id"], 110)
            self.assertIn("proposal_action", records[0])
            self.assertIn("executed_action_summary", records[0])
            self.assertIn("rejection_or_downgrade_summary", records[0])

            aggregate = json.loads(
                Path(config.artifact_paths()["aggregate_metrics"]).read_text(encoding="utf-8")
            )
            self.assertEqual(aggregate, outcome.summary)
            with Path(config.artifact_paths()["dashboard_csv"]).open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(tuple(rows[0]), DASHBOARD_COLUMNS)
            self.assertEqual(len(rows), 5)
            self.assertFalse(Path(config.artifact_paths()["dashboard_png"]).exists())
            self.assertFalse(Path(config.artifact_paths()["figure_input"]).exists())

    def test_direct_and_interactive_cli_share_random_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "smoke.json"
            config_path.write_text(json.dumps({
                "scenario_id": "small",
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
            direct_output: list[str] = []
            direct_status = cli_main(
                ["--mode", "random", "--config", str(config_path), "--seed", "42"],
                output_fn=direct_output.append,
            )
            answers = iter(("3", "small", str(config_path), "42"))
            interactive_output: list[str] = []
            interactive_status = cli_main(
                [],
                input_fn=lambda _prompt: next(answers),
                output_fn=interactive_output.append,
            )
            self.assertEqual(direct_status, 0)
            self.assertEqual(interactive_status, 1)
            self.assertTrue(any("status=completed" in line for line in direct_output))
            self.assertTrue(any("status=failed" in line for line in interactive_output))
            self.assertTrue(any("already exists" in line for line in interactive_output))
            registry = build_default_registry()
            self.assertEqual(
                registry.resolve("random", "random").__name__,
                "random_rollout_handler",
            )


class TestHeuristicPolicy(unittest.TestCase):
    def test_masked_history_excludes_invalid_and_distinguishes_missing(self) -> None:
        partial = masked_historical_quality(
            (2.0, 0.0, 4.0),
            (True, False, True),
        )
        self.assertTrue(partial.history_valid)
        self.assertEqual(partial.mean, 3.0)

        missing = masked_historical_quality(
            (0.0, 0.0, 0.0),
            (False, False, False),
        )
        valid_zero = masked_historical_quality(
            (0.0, 0.0, 0.0),
            (True, False, False),
        )
        self.assertFalse(missing.history_valid)
        self.assertIsNone(missing.mean)
        self.assertTrue(valid_zero.history_valid)
        self.assertEqual(valid_zero.mean, 0.0)

    def test_route_idle_defer_local_proxy_and_fallback(self) -> None:
        config, observation = make_heuristic_observation()
        self.assertEqual(HeuristicPolicy(config).act(observation).route, "idle")

        config, observation = make_heuristic_observation(
            unbound_slack=1,
            route_remotes=(1,),
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).route, "defer")

        config, observation = make_heuristic_observation(
            unbound_slack=3,
            unbound_cycles=20.0,
            local_backlog_cycles=20.0,
            route_remotes=(1,),
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).route, "local")

        config, observation = make_heuristic_observation(
            unbound_slack=2,
            unbound_cycles=20.0,
            local_backlog_cycles=100.0,
            route_remotes=(1,),
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).route, "local")

        config, observation = make_heuristic_observation(
            unbound_slack=3,
            unbound_cycles=20.0,
            local_backlog_cycles=100.0,
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).route, "local")

    def test_route_remote_history_valid_mean_and_id_rules(self) -> None:
        invalid = (False,) * 20
        first_only = (True,) + (False,) * 19
        zeros = (0.0,) * 20

        config, observation = make_heuristic_observation(
            unbound_slack=3,
            local_backlog_cycles=100.0,
            route_remotes=(1, 2),
            quality_values={1: zeros, 2: zeros},
            quality_masks={1: invalid, 2: first_only},
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).route, 2)

        config, observation = make_heuristic_observation(
            unbound_slack=3,
            local_backlog_cycles=100.0,
            route_remotes=(1, 2),
            quality_values={1: (1.0,) * 20, 2: (2.0,) * 20},
            quality_masks={1: (True,) * 20, 2: (True,) * 20},
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).route, 2)

        config, observation = make_heuristic_observation(
            unbound_slack=3,
            local_backlog_cycles=100.0,
            route_remotes=(1, 2),
            quality_values={1: (2.0,) * 20, 2: (2.0,) * 20},
            quality_masks={1: (True,) * 20, 2: (True,) * 20},
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).route, 1)

    def test_route_excludes_compute_poor_infeasible_destination(self) -> None:
        config, observation = make_heuristic_observation(
            unbound_slack=4,
            unbound_cycles=20.0,
            local_backlog_cycles=1_000.0,
            route_remotes=(1,),
            destination_max_cpu_frequency_hz={1: 100.0},
            destination_queued_cycles={1: 5.0},
        )

        self.assertEqual(
            HeuristicPolicy(config).act(observation).route,
            "local",
        )

    def test_route_feasibility_precedes_historical_quality_ranking(self) -> None:
        valid = (True,) * 20
        config, observation = make_heuristic_observation(
            unbound_slack=4,
            unbound_cycles=20.0,
            local_backlog_cycles=1_000.0,
            route_remotes=(1, 2),
            destination_max_cpu_frequency_hz={1: 1_000.0, 2: 1_000.0},
            destination_queued_cycles={1: 25.0, 2: 0.0},
            quality_values={1: (10.0,) * 20, 2: (1.0,) * 20},
            quality_masks={1: valid, 2: valid},
        )

        self.assertEqual(HeuristicPolicy(config).act(observation).route, 2)

    def test_route_all_remote_infeasible_preserves_local_fallback(self) -> None:
        config, observation = make_heuristic_observation(
            unbound_slack=3,
            unbound_bits=1_000.0,
            local_backlog_cycles=1_000.0,
            route_remotes=(1, 2),
            destination_rates_bps={1: 1.0, 2: 1.0},
        )

        self.assertEqual(
            HeuristicPolicy(config).act(observation).route,
            "local",
        )

    def test_route_cold_start_uses_actor_visible_historical_quality(self) -> None:
        config, observation = make_heuristic_observation(
            unbound_slack=3,
            local_backlog_cycles=1_000.0,
            route_remotes=(1,),
            destination_rate_valid={1: False},
            quality_values={1: (0.25,) * 20},
            quality_masks={1: (True,) * 20},
        )

        policy = HeuristicPolicy(config)
        expected = (
            max(config.action.resource_width_options)
            * config.environment.ru_bandwidth_hz
            * math.log2(1.0 + 0.25)
        )

        self.assertAlmostEqual(
            policy._estimated_route_rate_bps(observation, 1),
            expected,
        )
        self.assertEqual(policy.act(observation).route, 1)

    def test_route_valid_last_rate_is_used_without_bootstrap_fallback(self) -> None:
        config, observation = make_heuristic_observation(
            unbound_slack=4,
            local_backlog_cycles=1_000.0,
            route_remotes=(1,),
            destination_rates_bps={1: 1_234.0},
            destination_rate_valid={1: True},
            quality_values={1: (100.0,) * 20},
            quality_masks={1: (True,) * 20},
        )

        policy = HeuristicPolicy(config)

        self.assertEqual(
            policy._estimated_route_rate_bps(observation, 1),
            1_234.0,
        )
        self.assertEqual(policy.act(observation).route, 1)

    def test_route_cold_start_uses_actor_visible_stale_csi(self) -> None:
        config, observation = make_heuristic_observation(
            unbound_slack=3,
            local_backlog_cycles=1_000.0,
            route_remotes=(1,),
            destination_public_valid={1: False},
            destination_rate_valid={1: False},
            destination_stale_csi={1: 1.0e-5 + 0.0j},
            destination_csi_valid={1: True},
        )

        policy = HeuristicPolicy(config)
        estimated = policy._estimated_route_rate_bps(observation, 1)

        self.assertTrue(math.isfinite(estimated))
        self.assertGreater(estimated, 0.0)
        self.assertLessEqual(
            estimated,
            config.environment.reference_rate_bps,
        )
        self.assertEqual(policy.act(observation).route, 1)

    def test_route_invalid_all_history_has_finite_masked_fallback(self) -> None:
        config, observation = make_heuristic_observation(
            unbound_slack=3,
            local_backlog_cycles=1_000.0,
            route_remotes=(1,),
            destination_rate_valid={1: False},
            destination_public_valid={1: False},
            max_cpu_frequency_hz=2.1e9,
        )

        policy = HeuristicPolicy(config)
        expected = min(
            max(config.action.resource_width_options)
            * config.environment.ru_bandwidth_hz
            * math.log2(
                1.0 + config.environment.outage_threshold_linear
            ),
            config.environment.reference_rate_bps,
        )
        estimated = policy._estimated_route_rate_bps(observation, 1)

        self.assertTrue(math.isfinite(estimated))
        self.assertGreater(estimated, 0.0)
        self.assertAlmostEqual(estimated, expected)
        self.assertEqual(policy.act(observation).route, "local")

    def test_tx_lexicographic_slack_history_mean_and_destination(self) -> None:
        valid = (True,) * 20
        invalid = (False,) * 20

        config, observation = make_heuristic_observation(
            tx_candidates=(1, 2),
            tx_slacks={1: 2, 2: 3},
            quality_values={1: (1.0,) * 20, 2: (10.0,) * 20},
            quality_masks={1: valid, 2: valid},
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).tx_select, 1)

        config, observation = make_heuristic_observation(
            tx_candidates=(1, 2),
            tx_slacks={1: 2, 2: 2},
            quality_values={1: (0.0,) * 20, 2: (0.0,) * 20},
            quality_masks={1: invalid, 2: valid},
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).tx_select, 2)

        config, observation = make_heuristic_observation(
            tx_candidates=(1, 2),
            tx_slacks={1: 2, 2: 2},
            quality_values={1: (1.0,) * 20, 2: (2.0,) * 20},
            quality_masks={1: valid, 2: valid},
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).tx_select, 2)

        config, observation = make_heuristic_observation(
            tx_candidates=(1, 2),
            tx_slacks={1: 2, 2: 2},
            quality_values={1: (2.0,) * 20, 2: (2.0,) * 20},
            quality_masks={1: valid, 2: valid},
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).tx_select, 1)

    def test_resource_group_masked_history_ties_and_width_independence(self) -> None:
        values = [0.0] * 20
        masks = [False] * 20
        masks[4] = True
        config, observation = make_heuristic_observation(
            tx_candidates=(1,),
            quality_values={1: tuple(values)},
            quality_masks={1: tuple(masks)},
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).resource_group, 2)

        values = [0.0] * 20
        masks = [False] * 20
        values[0], values[2] = 10.0, 2.0
        masks[0], masks[2] = True, True
        values[4:8] = [5.0] * 4
        masks[4:8] = [True] * 4
        config, observation = make_heuristic_observation(
            tx_candidates=(1,),
            quality_values={1: tuple(values)},
            quality_masks={1: tuple(masks)},
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).resource_group, 1)

        values = [0.0] * 20
        masks = [False] * 20
        values[0:4] = [5.0] * 4
        values[4:8] = [5.0] * 4
        masks[0:8] = [True] * 8
        config, observation = make_heuristic_observation(
            tx_candidates=(1,),
            quality_values={1: tuple(values)},
            quality_masks={1: tuple(masks)},
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).resource_group, 1)

        config, observation = make_heuristic_observation(
            tx_candidates=(1,),
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).resource_group, 1)

        values = [0.0] * 20
        masks = [False] * 20
        values[8:12] = [9.0] * 4
        masks[8:12] = [True] * 4
        config, observation = make_heuristic_observation(
            tx_candidates=(1,),
            quality_values={1: tuple(values)},
            quality_masks={1: tuple(masks)},
        )
        normal = HeuristicPolicy(config).act(observation)
        width_one = SimpleNamespace(**vars(observation))
        width_one.action_masks = replace(
            observation.action_masks,
            resource_width_domain=(1,),
        )
        constrained = HeuristicPolicy(config).act(width_one)
        self.assertEqual(normal.resource_group, 3)
        self.assertEqual(constrained.resource_group, 3)
        self.assertEqual(constrained.resource_width, 1)

    def test_resource_width_maximum_and_inactive_canonical(self) -> None:
        config, observation = make_heuristic_observation(
            tx_candidates=(1,),
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).resource_width, 2)

        config, observation = make_heuristic_observation()
        self.assertEqual(HeuristicPolicy(config).act(observation).resource_width, 1)

    def test_power_maximum_zero_fallback_and_inactive_canonical(self) -> None:
        config, observation = make_heuristic_observation(
            tx_candidates=(1,),
            power_legal=(0.0, 0.25, 0.5),
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).power_level, 0.5)

        config, observation = make_heuristic_observation(
            tx_candidates=(1,),
            power_legal=(0.0,),
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).power_level, 0.0)

        config, observation = make_heuristic_observation()
        self.assertEqual(HeuristicPolicy(config).act(observation).power_level, 0.0)

    def test_cpu_queue_slack_task_id_and_source_tie_breaks(self) -> None:
        config, observation = make_heuristic_observation(
            cpu_candidates=(1, 2),
            cpu_slacks={1: 2, 2: 3},
            cpu_task_ids={1: 9, 2: 1},
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).cpu_queue, 1)

        config, observation = make_heuristic_observation(
            cpu_candidates=(1, 2),
            cpu_slacks={1: 2, 2: 2},
            cpu_task_ids={1: 9, 2: 1},
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).cpu_queue, 2)

        config, observation = make_heuristic_observation(
            cpu_candidates=(1, 2),
            cpu_slacks={1: 2, 2: 2},
            cpu_task_ids={1: 1, 2: 1},
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).cpu_queue, 1)

        config, observation = make_heuristic_observation(
            cpu_candidates=(0, 1),
            cpu_slacks={0: 2, 1: 2},
            cpu_task_ids={0: 1, 1: 1},
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).cpu_queue, 0)

    def test_cpu_frequency_uses_slack_exact_threshold_and_fallbacks(self) -> None:
        config, observation = make_heuristic_observation(
            cpu_candidates=(0,),
            cpu_slacks={0: 2},
            cpu_cycles={0: 20.0},
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).cpu_frequency, 0.5)

        config, observation = make_heuristic_observation(
            cpu_candidates=(0,),
            cpu_slacks={0: 2},
            cpu_cycles={0: 10.0},
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).cpu_frequency, 0.25)

        config, observation = make_heuristic_observation(
            cpu_candidates=(0,),
            cpu_slacks={0: 2},
            cpu_cycles={0: 21.0},
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).cpu_frequency, 1.0)

        config, observation = make_heuristic_observation(
            cpu_candidates=(0,),
            cpu_slacks={0: 1},
            cpu_cycles={0: 100.0},
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).cpu_frequency, 1.0)

        config, observation = make_heuristic_observation(
            cpu_candidates=(0,),
            cpu_slacks={0: 2},
            cpu_cycles={0: 20.0},
            cpu_frequency_legal={0: (0.0,)},
        )
        self.assertEqual(HeuristicPolicy(config).act(observation).cpu_frequency, 0.0)

    def test_cpu_frequency_cumulative_demand_requires_higher_level(self) -> None:
        config, observation = make_heuristic_observation(
            cpu_candidates=(0,),
            cpu_slacks={0: 2},
            cpu_cycles={0: 10.0},
            cpu_task_counts={0: 2},
            cpu_total_cycles={0: 20.0},
        )

        policy = HeuristicPolicy(config)
        proposal = policy.act(observation)

        self.assertEqual(proposal.cpu_frequency, 0.5)
        telemetry = policy.cpu_frequency_telemetry[0]
        self.assertTrue(telemetry.deadline_feasible)
        self.assertFalse(telemetry.infeasible_fallback)
        self.assertEqual(telemetry.pending_task_count, 2)

    def test_cpu_frequency_two_small_tasks_need_two_service_slots(self) -> None:
        config, observation = make_heuristic_observation(
            cpu_candidates=(0,),
            cpu_slacks={0: 1},
            cpu_cycles={0: 1.0},
            cpu_task_counts={0: 2},
            cpu_total_cycles={0: 2.0},
        )

        policy = HeuristicPolicy(config)
        proposal = policy.act(observation)

        self.assertEqual(proposal.cpu_frequency, 1.0)
        telemetry = policy.cpu_frequency_telemetry[0]
        self.assertFalse(telemetry.deadline_feasible)
        self.assertTrue(telemetry.infeasible_fallback)

    def test_cpu_frequency_checks_each_discrete_deadline_prefix(self) -> None:
        config, observation = make_heuristic_observation(
            cpu_candidates=(0, 1),
            cpu_slacks={0: 1, 1: 4},
            cpu_task_ids={0: 0, 1: 1},
            cpu_cycles={0: 5.0, 1: 6.0},
            cpu_task_counts={0: 1, 1: 2},
            cpu_total_cycles={0: 5.0, 1: 17.0},
        )

        policy = HeuristicPolicy(config)
        proposal = policy.act(observation)

        self.assertEqual(proposal.cpu_frequency, 0.5)
        self.assertTrue(
            policy.cpu_frequency_telemetry[0].deadline_feasible
        )

    def test_cpu_frequency_single_task_matches_old_minimum_sufficient(self) -> None:
        config, observation = make_heuristic_observation(
            cpu_candidates=(0,),
            cpu_slacks={0: 2},
            cpu_cycles={0: 10.0},
            cpu_task_counts={0: 1},
            cpu_total_cycles={0: 10.0},
        )

        policy = HeuristicPolicy(config)
        proposal = policy.act(observation)

        self.assertEqual(proposal.cpu_frequency, 0.25)
        self.assertTrue(
            policy.cpu_frequency_telemetry[0].deadline_feasible
        )

    def test_cpu_frequency_deadline_slot_counts_as_serviceable(self) -> None:
        config, observation = make_heuristic_observation(
            cpu_candidates=(0,),
            cpu_slacks={0: 1},
            cpu_cycles={0: 20.0},
            cpu_task_counts={0: 1},
            cpu_total_cycles={0: 20.0},
        )

        policy = HeuristicPolicy(config)
        proposal = policy.act(observation)

        self.assertEqual(proposal.cpu_frequency, 1.0)
        self.assertTrue(
            policy.cpu_frequency_telemetry[0].deadline_feasible
        )

    def test_cpu_queue_aggregate_deadline_order_is_deterministic(self) -> None:
        config, observation = make_heuristic_observation(
            cpu_candidates=(0, 1, 2),
            cpu_slacks={0: 3, 1: 2, 2: 2},
            cpu_task_ids={0: 1, 1: 9, 2: 1},
        )

        policy = HeuristicPolicy(config)
        first = policy._visible_cpu_demands(observation)
        second = policy._visible_cpu_demands(observation)

        self.assertEqual(first, second)
        self.assertEqual(tuple(item.source for item in first), (2, 1, 0))

    def test_canonical_values_order_actor_boundary_and_seed_independence(self) -> None:
        config, observation = make_heuristic_observation()
        proposal = HeuristicPolicy(config).act(observation)
        self.assertEqual(
            proposal.branches,
            ("idle", "idle", "idle", 1, 0.0, "idle", 0.0),
        )
        self.assertEqual(len(proposal.branches), 7)
        self.assertEqual(
            tuple(inspect.signature(HeuristicPolicy.act).parameters),
            ("self", "observation"),
        )
        for privileged in (
            "centralized_state",
            "true_channel",
            "true_sinr",
            "current_interference",
            "I_meas",
            "executed_action",
        ):
            self.assertFalse(hasattr(observation, privileged))

        calls: list[str] = []
        original = observation.action_masks

        class RecordingMasks:
            sampling_order = original.sampling_order

            @staticmethod
            def domain_for(branch: str):
                return original.domain_for(branch)

            @staticmethod
            def mask_for(branch: str, previous=None):
                calls.append(branch)
                return original.mask_for(branch, previous)

            @staticmethod
            def is_legal(candidate: ActionProposal) -> bool:
                return original.is_legal(candidate)

        actor_only = SimpleNamespace(**vars(observation))
        actor_only.action_masks = RecordingMasks()
        first = HeuristicPolicy(config, policy_seed=42).act(actor_only)
        second = HeuristicPolicy(config, policy_seed=999).act(actor_only)
        self.assertEqual(calls[:7], list(BRANCH_ORDER))
        self.assertEqual(first, second)
        self.assertIsNone(HeuristicPolicy(config).policy_stream_id)

    def test_real_rollout_artifacts_direct_and_interactive_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_heuristic_config(
                horizon=5,
                arrivals=(1.0, 1.0, 1.0, 1.0),
                output_root=root,
            )
            outcome = RolloutRunner(
                config,
                HeuristicPolicy(config),
            ).run()
            self.assertEqual(len(outcome.raw_records), 5)
            self.assertEqual(len(outcome.artifacts), 4)
            self.assertEqual(outcome.summary["method_id"], "heuristic")
            self.assertEqual(outcome.summary["policy_seed"], 42)
            self.assertIsNone(outcome.summary["policy_stream_id"])
            self.assertTrue(outcome.summary["conservation"]["is_conserved"])
            for artifact in outcome.artifacts:
                self.assertTrue(Path(artifact).is_file())
            self.assertFalse(
                Path(config.artifact_paths()["dashboard_png"]).exists()
            )

            config_path = root / "heuristic.json"
            config_path.write_text(json.dumps({
                "environment": {
                    "episode_horizon": 3,
                    "arrival_probabilities": [1.0, 1.0, 1.0, 1.0],
                },
                "output": {
                    "logs_dir": str(root / "cli_logs"),
                    "dashboard_logs_dir": str(root / "cli_dashboard_logs"),
                    "plots_dir": str(root / "cli_plots"),
                },
            }), encoding="utf-8")
            direct_output: list[str] = []
            direct_status = cli_main(
                ["--mode", "heuristic", "--config", str(config_path), "--seed", "42"],
                output_fn=direct_output.append,
            )
            answers = iter(("4", "small", str(config_path), "42"))
            interactive_output: list[str] = []
            interactive_status = cli_main(
                [],
                input_fn=lambda _prompt: next(answers),
                output_fn=interactive_output.append,
            )
            self.assertEqual(direct_status, 0)
            self.assertEqual(interactive_status, 1)
            self.assertTrue(any("status=completed" in line for line in direct_output))
            self.assertTrue(any("status=failed" in line for line in interactive_output))
            self.assertTrue(any("already exists" in line for line in interactive_output))
            self.assertEqual(
                build_default_registry().resolve(
                    "heuristic", "heuristic"
                ).__name__,
                "heuristic_rollout_handler",
            )


class TestLocalOnlyPolicy(unittest.TestCase):
    @staticmethod
    def _policy(config):
        return LocalOnlyPolicy(
            replace(config, mode="baseline", method_id="local_only")
        )

    def test_route_and_communication_follow_local_only_masks(self) -> None:
        config, observation = make_heuristic_observation(
            unbound_slack=3,
            route_remotes=(1, 2),
            tx_candidates=(1, 2),
        )
        proposal = self._policy(config).act(observation)
        self.assertEqual(proposal.route, "local")
        self.assertEqual(proposal.tx_select, "idle")
        self.assertEqual(proposal.resource_group, "idle")
        self.assertEqual(proposal.resource_width, 1)
        self.assertEqual(proposal.power_level, 0.0)
        self.assertTrue(observation.action_masks.is_legal(proposal))

        config, inactive = make_heuristic_observation(tx_candidates=(1, 2))
        inactive_proposal = self._policy(config).act(inactive)
        self.assertEqual(inactive_proposal.route, "idle")
        self.assertEqual(inactive_proposal.tx_select, "idle")
        self.assertTrue(inactive.action_masks.is_legal(inactive_proposal))

    def test_cpu_serves_only_self_at_highest_legal_frequency(self) -> None:
        config, observation = make_heuristic_observation(
            cpu_candidates=(0, 1),
            cpu_frequency_legal={0: (0.0, 0.25, 0.5)},
        )
        proposal = self._policy(config).act(observation)
        self.assertEqual(proposal.cpu_queue, observation.uav_id)
        self.assertEqual(proposal.cpu_frequency, 0.5)
        self.assertTrue(observation.action_masks.is_legal(proposal))

        config, zero_only = make_heuristic_observation(
            cpu_candidates=(0, 1),
            cpu_frequency_legal={0: (0.0,)},
        )
        zero_proposal = self._policy(config).act(zero_only)
        self.assertEqual(zero_proposal.cpu_queue, zero_only.uav_id)
        self.assertEqual(zero_proposal.cpu_frequency, 0.0)
        self.assertTrue(zero_only.action_masks.is_legal(zero_proposal))

        config, remote_only = make_heuristic_observation(cpu_candidates=(1,))
        idle_proposal = self._policy(config).act(remote_only)
        self.assertEqual(idle_proposal.cpu_queue, "idle")
        self.assertEqual(idle_proposal.cpu_frequency, 0.0)
        self.assertTrue(remote_only.action_masks.is_legal(idle_proposal))

    def test_determinism_actor_boundary_and_no_rng(self) -> None:
        config, observation = make_heuristic_observation(
            unbound_slack=3,
            route_remotes=(1,),
            tx_candidates=(1,),
            cpu_candidates=(0, 1),
        )
        baseline = replace(config, mode="baseline", method_id="local_only")
        actor_only = SimpleNamespace(
            uav_id=observation.uav_id,
            action_masks=observation.action_masks,
        )
        first_policy = LocalOnlyPolicy(baseline, policy_seed=42)
        second_policy = LocalOnlyPolicy(baseline, policy_seed=999)
        first = first_policy.act(actor_only)
        self.assertEqual(first, first_policy.act(actor_only))
        self.assertEqual(first, second_policy.act(actor_only))
        self.assertIsNone(first_policy.policy_stream_id)
        self.assertFalse(hasattr(first_policy, "_rng"))
        self.assertEqual(
            tuple(inspect.signature(LocalOnlyPolicy.act).parameters),
            ("self", "observation"),
        )
        self.assertTrue(observation.action_masks.is_legal(first))

    def test_short_real_rollout_preserves_local_only_invariants(self) -> None:
        config = make_local_only_config(horizon=8)
        outcome = RolloutRunner(config, LocalOnlyPolicy(config)).run(
            write_artifacts=False
        )
        self.assertEqual(len(outcome.raw_records), 8)
        self.assertEqual(outcome.summary["episode"]["slots_executed"], 8)
        self.assertTrue(outcome.summary["conservation"]["is_conserved"])
        self.assertAlmostEqual(
            outcome.summary["energy_j"]["tx"],
            0.0,
            delta=config.environment.energy_tolerance_j,
        )
        self.assertEqual(
            outcome.summary["transmission"]["actual_attempt_count"], 0
        )
        self.assertEqual(
            outcome.summary["transmission"]["outage_valid_sample_count"], 0
        )
        self.assertIsNone(outcome.summary["transmission"]["outage_rate"])
        for record in outcome.raw_records:
            for proposal in record["proposal_action"]:
                self.assertIn(proposal["route"], {"idle", "local"})
                self.assertEqual(proposal["tx_select"], "idle")
                self.assertEqual(proposal["resource_group"], "idle")
                self.assertEqual(proposal["resource_width"], 1)
                self.assertEqual(proposal["power_level"], 0.0)
                self.assertIn(
                    proposal["cpu_queue"],
                    {"idle", proposal["uav_id"]},
                )


if __name__ == "__main__":
    unittest.main()
