from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.cli import build_arg_parser, build_execution_context_from_args
from src.config import RunConfig
from src.env.actions import ActionProposal
from src.env.environment import StepResult, U2UMECEnvironment
from src.evaluation import route_oracle as route_oracle_module
from src.evaluation.route_oracle import (
    FrozenOraclePolicyRunnerCache,
    Oracle1Collector,
    OracleBranchResult,
)
from src.evaluation.route_oracle_probe import (
    action_diff_report,
    build_counterfactual_actions,
    common_random_numbers_gate,
    repeatability_gate,
    same_hidden_gate,
    same_root_gate,
)
from src.models.ca_gat_mappo import CAGATMAPPOActor


def proposal(uav_id: int, *, route: str | int = "defer") -> ActionProposal:
    return ActionProposal(
        uav_id=uav_id,
        route=route,
        tx_select="idle",
        resource_group="idle",
        resource_width=1,
        power_level=0.0,
        cpu_queue="idle",
        cpu_frequency=0.0,
    )


def branch(
    route: str | int,
    *,
    state: str = "state",
    rng: str = "rng",
    hidden: str = "hidden",
    exogenous: tuple[str, ...] = ("exo-1", "exo-2"),
    reward: float = 1.0,
) -> OracleBranchResult:
    return OracleBranchResult(
        route=route,
        terminal_class="done",
        terminal_slot=2,
        route_slot=0,
        route_to_terminal_slots=2,
        completion_delay_s=0.2,
        source_tx_energy_j=0.1,
        focal_cpu_energy_j=0.2,
        helper_cpu_energy_j=0.0,
        team_active_energy_j=0.3,
        initial_state_fingerprint=state,
        initial_rng_fingerprint=rng,
        initial_exogenous_fingerprint="root-exogenous",
        initial_hidden_digest=hidden,
        final_rng_fingerprint="final-rng",
        trace=({"slot": 0, "reward": reward},),
        exogenous_trace=exogenous,
        focal_task_id=7,
    )


class RouteOracleProbeGateTests(unittest.TestCase):
    def test_same_root_gate_passes_and_detects_root_or_factual_mutation(self) -> None:
        decision = SimpleNamespace(
            initial_state_fingerprint="state",
            initial_rng_fingerprint="rng",
            branches=(branch("local"), branch(1)),
        )
        self.assertTrue(
            same_root_gate(
                decision,
                factual_state_after="state",
                factual_rng_after="rng",
            )
        )
        changed_branch = replace(branch(1), initial_rng_fingerprint="changed")
        changed = SimpleNamespace(
            initial_state_fingerprint="state",
            initial_rng_fingerprint="rng",
            branches=(branch("local"), changed_branch),
        )
        self.assertFalse(
            same_root_gate(
                changed,
                factual_state_after="state",
                factual_rng_after="rng",
            )
        )
        self.assertFalse(
            same_root_gate(
                decision,
                factual_state_after="mutated",
                factual_rng_after="rng",
            )
        )

    def test_only_route_diff_gate_passes_and_detects_other_action_change(self) -> None:
        factual = (proposal(0), proposal(1))
        actions = build_counterfactual_actions(
            factual,
            focal_uav=0,
            routes=("local", 1),
        )
        passed, report = action_diff_report(actions, focal_uav=0)
        self.assertTrue(passed)
        self.assertEqual(
            report["local_vs_remote_uav_1"],
            [{"uav_id": 0, "field": "route", "local": "local", "branch": 1}],
        )
        contaminated = dict(actions)
        remote = list(contaminated["remote_uav_1"])
        remote[1] = replace(remote[1], cpu_frequency=0.5)
        contaminated["remote_uav_1"] = tuple(remote)
        self.assertFalse(action_diff_report(contaminated, focal_uav=0)[0])

    def test_same_hidden_and_crn_gates_detect_mismatch(self) -> None:
        branches = (branch("local"), branch(1))
        self.assertTrue(same_hidden_gate(branches, "hidden"))
        self.assertTrue(common_random_numbers_gate(branches))
        self.assertFalse(
            same_hidden_gate((branches[0], replace(branches[1], initial_hidden_digest="other")), "hidden")
        )
        self.assertFalse(
            common_random_numbers_gate(
                (branches[0], replace(branches[1], exogenous_trace=("different", "exo-2")))
            )
        )

    def test_repeatability_gate_requires_exact_branch_replay(self) -> None:
        first = (branch("local"), branch(1))
        passed, evidence = repeatability_gate(first, first)
        self.assertTrue(passed)
        self.assertEqual(
            evidence["local"]["first_digest"],
            evidence["local"]["second_digest"],
        )
        changed = (branch("local", reward=1.5), branch(1))
        self.assertFalse(repeatability_gate(first, changed)[0])

        terminal_changed = replace(
            branch("local"),
            trace=(
                {
                    "slot": 1,
                    "state_fingerprint": "different-final-state",
                    "rng_fingerprint": "final-rng",
                    "reward": 1.0,
                },
            ),
        )
        self.assertFalse(
            repeatability_gate(first, (terminal_changed, branch(1)))[0]
        )


class FormalProbeRuntimeOptimizationTests(unittest.TestCase):
    class _ActionMasks:
        @staticmethod
        def is_legal(_proposal) -> bool:
            return True

    class _Observation:
        action_masks = None

        def __init__(self) -> None:
            self.action_masks = FormalProbeRuntimeOptimizationTests._ActionMasks()

    class _Shadow:
        def __init__(self) -> None:
            self.slot = 0
            self.state_calls = 0
            self.rng_calls = 0
            self.exogenous_calls = 0

        def step(self, _proposals) -> StepResult:
            self.slot += 1
            return StepResult(
                observations=(FormalProbeRuntimeOptimizationTests._Observation(),),
                centralized_state=None,
                reward=float(self.slot),
                terminated=False,
                truncated=False,
                info={},
            )

        def state_fingerprint(self) -> str:
            self.state_calls += 1
            return f"state-{self.slot}"

        def rng_fingerprint(self) -> str:
            self.rng_calls += 1
            return f"rng-{self.slot}"

        def exogenous_fingerprint(self) -> str:
            self.exogenous_calls += 1
            return f"exogenous-{self.slot}"

    class _Runner:
        def act(self, _observations, hidden, *, episode_start=False):
            del episode_start
            return SimpleNamespace(
                proposals=(proposal(0, route="local"),),
                hidden_out=hidden.detach().clone(),
            )

    @staticmethod
    def _rollout(mode: str):
        collector = Oracle1Collector.__new__(Oracle1Collector)
        collector.budget = None
        collector.runner = FormalProbeRuntimeOptimizationTests._Runner()
        collector.trace_fingerprint_mode = mode
        shadow = FormalProbeRuntimeOptimizationTests._Shadow()

        def terminal_payload(environment, _task_id, route_slot):
            if environment.slot < 3:
                return None
            return {
                "terminal_class": "done",
                "terminal_slot": environment.slot,
                "route_to_terminal_slots": environment.slot - route_slot,
                "completion_delay_s": 0.3,
                "terminal_reason": "completed",
            }

        with patch.object(
            route_oracle_module,
            "_task_terminal_payload",
            side_effect=terminal_payload,
        ):
            result = collector._rollout_branch(
                shadow,
                (FormalProbeRuntimeOptimizationTests._Observation(),),
                (proposal(0, route="local"),),
                source_uav=0,
                focal_task_id=7,
                route="local",
                route_slot=0,
                initial_hidden=torch.zeros((1, 1, 1), dtype=torch.float32),
                initial_state="root-state",
                initial_rng="root-rng",
                initial_exogenous="root-exogenous",
                initial_hidden_digest="hidden",
                role="COMMITTED",
            )
        return shadow, result

    def test_boundary_fingerprints_only_terminal_step_without_outcome_change(self) -> None:
        step_shadow, step_result = self._rollout("step")
        boundary_shadow, boundary_result = self._rollout("boundary")

        self.assertEqual(step_shadow.state_calls, 3)
        self.assertEqual(step_shadow.rng_calls, 3)
        self.assertEqual(boundary_shadow.state_calls, 1)
        self.assertEqual(boundary_shadow.rng_calls, 1)
        self.assertEqual(boundary_shadow.exogenous_calls, 3)
        self.assertTrue(
            all(item["state_fingerprint"] is None for item in boundary_result.trace[:-1])
        )
        self.assertIsNotNone(boundary_result.trace[-1]["state_fingerprint"])
        self.assertIsNotNone(boundary_result.trace[-1]["rng_fingerprint"])
        self.assertEqual(
            [item["reward"] for item in step_result.trace],
            [item["reward"] for item in boundary_result.trace],
        )
        self.assertEqual(step_result.exogenous_trace, boundary_result.exogenous_trace)
        self.assertEqual(step_result.terminal_class, boundary_result.terminal_class)
        self.assertEqual(step_result.terminal_slot, boundary_result.terminal_slot)
        self.assertEqual(step_result.team_active_energy_j, boundary_result.team_active_energy_j)

    def test_preprocessing_cache_never_caches_actor_output(self) -> None:
        config = RunConfig()
        config.validate()
        environment = U2UMECEnvironment(config)
        observations = tuple(environment.reset().observations)
        actor = CAGATMAPPOActor(config).eval()
        runner = FrozenOraclePolicyRunnerCache(config).get(actor, 0)
        hidden = actor.initial_hidden(1, device="cpu", dtype=torch.float32)

        with (
            patch.object(
                runner._tensorizer,
                "encode_step",
                wraps=runner._tensorizer.encode_step,
            ) as encode_step,
            patch.object(
                runner._distribution,
                "deterministic_actions",
                wraps=runner._distribution.deterministic_actions,
            ) as deterministic_actions,
        ):
            runner.start_preprocessing_cache(max_entries=4)
            first = runner.act(observations, hidden)
            second = runner.act(observations, hidden)
            stats = runner.stop_preprocessing_cache()

        self.assertEqual(encode_step.call_count, 1)
        self.assertEqual(deterministic_actions.call_count, 2)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(first.proposals, second.proposals)
        self.assertTrue(torch.equal(first.hidden_out, second.hidden_out))


class RouteOracleProbeCliTests(unittest.TestCase):
    def test_direct_flags_build_bounded_probe_context(self) -> None:
        checkpoint = "main.py"
        args = build_arg_parser().parse_args(
            [
                "--mode",
                "evaluation",
                "--method-id",
                "ca_gat_mappo",
                "--evaluate-from",
                checkpoint,
                "--evaluation-device",
                "cpu",
                "--oracle-probe",
                "--oracle-probe-states",
                "2",
            ]
        )
        context = build_execution_context_from_args(args)
        self.assertEqual(context.oracle_probe_states, 2)
        self.assertEqual(context.evaluation_device, "cpu")

if __name__ == "__main__":
    unittest.main()
