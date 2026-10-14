from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass, replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.config import CHECKPOINT_KIND_FINAL_COMPLETED, RunConfig
from src.env.actions import ActionProposal
from src.env.environment import (
    EnvironmentFingerprintError,
    U2UMECEnvironment,
    _fingerprint_digest,
)
from src.env.tasks import Task
from src.evaluation.route_oracle import (
    ATTRIBUTION_ALLOCATED_PROXY,
    ATTRIBUTION_EXACT,
    ATTRIBUTION_NOT_AVAILABLE,
    FrozenOraclePolicyRunnerCache,
    FrozenPolicyIdentity,
    FrozenRunnerIdentity,
    FrozenRunnerProvenance,
    Oracle1Collector,
    OracleBranchResult,
    OracleCollectionBudget,
    OracleDecisionResult,
    OracleReliabilityError,
    RouteOracleArtifactWriter,
    UNRESOLVED,
    VALID_BRANCH_STATUS,
    INVALID_BRANCH_STATUS,
    classify_oracle_routes,
    compare_branch_outcomes,
    inspect_route_oracle_artifact,
    oracle_schema_header,
    _energy_from_step,
    _task_terminal_payload,
)
from src.models.ca_gat_mappo import CAGATMAPPOActor, MAPPOCentralizedCritic
from src.models.ca_gat_mappo_checkpoint import (
    build_checkpoint_payload,
    serialize_active_rollout,
)
from src.models.ca_gat_mappo_trainer import CAGATMAPPOTrainer
from src.models.ca_gat_mappo_update import (
    RecurrentPPOEpochDiagnostics,
    RecurrentPPOUpdateOutput,
    build_ca_gat_mappo_optimizers,
)


def make_oracle_config(*, horizon: int = 2, output_dir: str | None = None) -> RunConfig:
    base = RunConfig()
    env = replace(
        base.environment,
        episode_horizon=horizon,
        uav_count=2,
        arrival_probabilities=(1.0, 0.0),
        profile_assignment=("Balanced", "Balanced"),
        profile_perturbations=((0.0, 0.0, 0.0, 0.0),) * 2,
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
        training_device="cpu",
        route_oracle_counterfactual_enabled=True,
        route_oracle_selection_rate_ppm=1_000_000,
    )
    config = replace(
        base,
        environment=env,
        output=(
            replace(base.output, logs_dir=output_dir)
            if output_dir is not None
            else base.output
        ),
        training=replace(base.training, mappo=mappo),
    )
    config.validate()
    return config


def make_trainer_oracle_config(output_dir: str, *, enabled: bool) -> RunConfig:
    config = make_oracle_config(horizon=128, output_dir=output_dir)
    mappo = replace(
        config.training.mappo,
        max_training_episodes=2,
        max_training_environment_steps=256,
        checkpoint_interval_steps=128,
        route_oracle_counterfactual_enabled=enabled,
        route_oracle_selection_rate_ppm=1_000_000 if enabled else 0,
    )
    config = replace(
        config,
        mode="rl",
        method_id="ca_gat_mappo",
        training=replace(
            config.training,
            formal_rl_enabled=True,
            mappo=mappo,
        ),
    )
    config.validate()
    return config


def checkpoint_payload_for(trainer: CAGATMAPPOTrainer) -> dict[str, object]:
    return build_checkpoint_payload(
        config=trainer.config,
        checkpoint_kind=CHECKPOINT_KIND_FINAL_COMPLETED,
        actor=trainer.actor,
        critic=trainer.critic,
        optimizers=trainer._optimizer_bundle(),
        policy_generator=trainer.policy_generator,
        training_state=trainer._training_checkpoint_state(
            CHECKPOINT_KIND_FINAL_COMPLETED
        ),
        active_rollout_state=serialize_active_rollout(
            trainer.rollout_buffer,
            trainer._rollout_policy_version,
            CHECKPOINT_KIND_FINAL_COMPLETED,
        ),
        diagnostics_state=trainer._diagnostics_checkpoint_state(),
        dtype=trainer.dtype,
    )


def active_fixture(*, factual_route: str | int = "defer"):
    config = make_oracle_config()
    environment = U2UMECEnvironment(config)
    environment.reset()
    first = environment.step(environment.canonical_proposals())
    assert first.observations is not None
    observations = tuple(first.observations)
    remote = next(
        value
        for value, available in zip(
            observations[0].action_masks.route_domain,
            observations[0].action_masks.route_mask,
        )
        if bool(available) and isinstance(value, int) and not isinstance(value, bool)
    )
    if factual_route == "remote":
        factual_route = remote
    proposals = list(environment.canonical_proposals())
    proposals[0] = replace(proposals[0], route=factual_route)
    return config, environment, observations, tuple(proposals), remote


def assert_nested_exact(test: unittest.TestCase, left, right, path: str = "root") -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        test.assertIsInstance(left, torch.Tensor, path)
        test.assertIsInstance(right, torch.Tensor, path)
        test.assertEqual(left.dtype, right.dtype, path)
        test.assertEqual(tuple(left.shape), tuple(right.shape), path)
        test.assertTrue(torch.equal(left, right), path)
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        test.assertIsInstance(left, np.ndarray, path)
        test.assertIsInstance(right, np.ndarray, path)
        test.assertEqual(left.dtype, right.dtype, path)
        np.testing.assert_array_equal(left, right, err_msg=path)
        return
    if is_dataclass(left) or is_dataclass(right):
        test.assertTrue(is_dataclass(left) and is_dataclass(right), path)
        test.assertEqual(type(left), type(right), path)
        for item in fields(left):
            assert_nested_exact(test, getattr(left, item.name), getattr(right, item.name), f"{path}.{item.name}")
        return
    if isinstance(left, dict) or isinstance(right, dict):
        test.assertIsInstance(left, dict, path)
        test.assertIsInstance(right, dict, path)
        test.assertEqual(set(left), set(right), path)
        for key in left:
            assert_nested_exact(test, left[key], right[key], f"{path}[{key!r}]")
        return
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        test.assertEqual(type(left), type(right), path)
        test.assertEqual(len(left), len(right), path)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            assert_nested_exact(test, left_item, right_item, f"{path}[{index}]")
        return
    test.assertEqual(left, right, path)


class RecordingEnvironment(U2UMECEnvironment):
    def __init__(self, config: RunConfig) -> None:
        super().__init__(config)
        self.factual_actions: list[tuple[ActionProposal, ...]] = []
        self.factual_results = []
        self.factual_state_fingerprints: list[str] = []
        self.factual_rng_fingerprints: list[str] = []

    def state_fingerprint(self) -> str:
        # Recording lists belong to the test observer, not environment state.
        saved = (
            self.factual_actions,
            self.factual_results,
            self.factual_state_fingerprints,
            self.factual_rng_fingerprints,
        )
        del self.factual_actions
        del self.factual_results
        del self.factual_state_fingerprints
        del self.factual_rng_fingerprints
        saved_config = self.config
        normalized_mappo = replace(
            saved_config.training.mappo,
            route_oracle_counterfactual_enabled=False,
            route_oracle_selection_rate_ppm=0,
        )
        normalized_output = replace(
            saved_config.output,
            logs_dir="__oracle_neutrality__",
        )
        self.config = replace(
            saved_config,
            training=replace(saved_config.training, mappo=normalized_mappo),
            output=normalized_output,
        )
        try:
            # The integration gate compares factual runtime state, while the
            # full object-graph fingerprint is covered by the dedicated clone
            # gate.  The public snapshot avoids duplicated immutable config
            # references held by service objects.
            return _fingerprint_digest(self.snapshot())
        finally:
            self.config = saved_config
            (
                self.factual_actions,
                self.factual_results,
                self.factual_state_fingerprints,
                self.factual_rng_fingerprints,
            ) = saved

    def step(self, proposals):
        joint = tuple(proposals)
        self.factual_actions.append(joint)
        result = super().step(joint)
        self.factual_results.append(result)
        self.factual_state_fingerprints.append(self.state_fingerprint())
        self.factual_rng_fingerprints.append(self.rng_fingerprint())
        return result


class RecordingFactory:
    def __init__(self) -> None:
        self.environments: list[RecordingEnvironment] = []

    def __call__(self, config: RunConfig) -> RecordingEnvironment:
        environment = RecordingEnvironment(config)
        self.environments.append(environment)
        return environment


class DeterministicAdamUpdater:
    """Use real optimizer state while keeping the integration fixture bounded."""

    def __init__(self, actor, critic, config, _distribution) -> None:
        self.optimizers = build_ca_gat_mappo_optimizers(actor, critic, config)
        self.captured_transitions = ()
        self.calls = 0

    def update(self, buffer):
        self.captured_transitions = tuple(
            copy.deepcopy(buffer.transition_at(index))
            for index in range(len(buffer))
        )
        self.calls += 1
        actor_optimizer = self.optimizers.actor_optimizer
        critic_optimizer = self.optimizers.critic_optimizer
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        actor_loss = sum(parameter.square().sum() for parameter in self.optimizers.actor_parameters) * 1.0e-7
        critic_loss = sum(parameter.square().sum() for parameter in self.optimizers.critic_parameters) * 1.0e-7
        actor_loss.backward()
        critic_loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()
        diagnostics = tuple(
            RecurrentPPOEpochDiagnostics(
                epoch_index=index,
                actor_loss=1.0 + index,
                critic_loss=2.0 + index,
                entropy_mean=0.5 + index,
                total_loss=3.0 + index,
                ratio_mean=1.0,
                actor_grad_norm_before_clip=4.0,
                critic_grad_norm_before_clip=5.0,
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


def make_branch(
    route: str | int,
    *,
    terminal: str = "done",
    slots: int = 1,
    energy: float | None = 1.0,
    valid: bool = True,
) -> OracleBranchResult:
    if valid:
        return OracleBranchResult(
            route=route,
            terminal_class=terminal,
            terminal_slot=slots,
            route_slot=0,
            route_to_terminal_slots=slots,
            completion_delay_s=float(slots) if terminal == "done" else None,
            source_tx_energy_j=energy,
            focal_cpu_energy_j=energy,
            team_active_energy_j=energy,
        )
    return OracleBranchResult(
        route=route,
        valid=False,
        status=INVALID_BRANCH_STATUS,
    )


class RouteOracleDiagnosticTests(unittest.TestCase):
    def test_clone_structure_and_identical_action_replay_gates(self) -> None:
        config = make_oracle_config(horizon=3)
        factual = U2UMECEnvironment(config)
        shadow = U2UMECEnvironment(config)
        factual.reset()
        shadow.reset()
        self.assertEqual(factual.state_fingerprint(), shadow.state_fingerprint())
        self.assertEqual(factual.rng_fingerprint(), shadow.rng_fingerprint())
        for _ in range(config.environment.episode_horizon):
            proposals = factual.canonical_proposals()
            expected = factual.step(proposals)
            observed = shadow.step(proposals)
            self.assertEqual(expected.reward, observed.reward)
            self.assertEqual(expected.terminated, observed.terminated)
            self.assertEqual(expected.truncated, observed.truncated)
            self.assertEqual(expected.info, observed.info)
            self.assertEqual(factual.state_fingerprint(), shadow.state_fingerprint())
            self.assertEqual(factual.rng_fingerprint(), shadow.rng_fingerprint())
            if expected.truncated:
                break

    def test_state_fingerprint_no_repr_and_mutation_sensitivity_gates(self) -> None:
        first = {"values": [1, 2, {"bytes": b"abc"}]}
        second = {"values": [1, 2, {"bytes": b"abc"}]}
        self.assertEqual(_fingerprint_digest(first), _fingerprint_digest(second))
        self.assertEqual(_fingerprint_digest(first), _fingerprint_digest(first))

        class UnsupportedFingerprintObject:
            __slots__ = ()

        with self.assertRaises(EnvironmentFingerprintError):
            _fingerprint_digest(UnsupportedFingerprintObject())

        config = make_oracle_config(horizon=3)
        environment = U2UMECEnvironment(config)
        environment.reset()
        before = environment.state_fingerprint()
        self.assertEqual(before, environment.state_fingerprint())
        environment._actual_attempt_counts[0, 1] += 1
        self.assertNotEqual(before, environment.state_fingerprint())

    def test_real_clone_for_shadow_structure_and_mutation_gate(self) -> None:
        _, factual, _, _, _ = active_fixture()
        shadow = factual.clone_for_shadow()
        self.assertEqual(factual.state_fingerprint(), shadow.state_fingerprint())
        self.assertEqual(factual.rng_fingerprint(), shadow.rng_fingerprint())

        for left, right in (
            (factual.lifecycle, shadow.lifecycle),
            (factual.lifecycle.queues, shadow.lifecycle.queues),
            (factual.lifecycle.task_id_generator, shadow.lifecycle.task_id_generator),
            (factual.resource_states, shadow.resource_states),
            (factual.public_history, shadow.public_history),
            (factual.previous_actions, shadow.previous_actions),
            (factual.traffic, shadow.traffic),
            (factual.traffic._history, shadow.traffic._history),
            (factual.channel_history, shadow.channel_history),
            (factual.channel_history._true_channels, shadow.channel_history._true_channels),
            (factual.channel_history.interference, shadow.channel_history.interference),
            (factual.metrics, shadow.metrics),
            (factual._last_effective_rate_bps, shadow._last_effective_rate_bps),
            (factual._actual_attempt_counts, shadow._actual_attempt_counts),
            (factual._outage_counts, shadow._outage_counts),
        ):
            self.assertIsNot(left, right)
        for name, factual_rng in factual._environment_rngs().items():
            self.assertIsNot(factual_rng, shadow._environment_rngs()[name])

        self.assertIs(shadow.executor.lifecycle, shadow.lifecycle)
        self.assertIs(shadow.physical_service.lifecycle, shadow.lifecycle)
        self.assertIsNot(shadow.physical_service.resource_states, factual.resource_states)

        factual_state = factual.state_fingerprint()
        factual_rng = factual.rng_fingerprint()
        shadow._actual_attempt_counts[0, 1] += 1
        self.assertEqual(factual_state, factual.state_fingerprint())
        self.assertEqual(factual_rng, factual.rng_fingerprint())

    def test_factual_local_remote_reuse_and_defer_shadow_branch_gates(self) -> None:
        for route in ("local", "remote", "defer"):
            config, environment, observations, proposals, remote = active_fixture(
                factual_route=route
            )
            actor = CAGATMAPPOActor(config).eval()
            runner = FrozenOraclePolicyRunnerCache(config).get(actor, 0)
            hidden = actor.initial_hidden(1, device="cpu", dtype=torch.float32)
            initial_state = environment.state_fingerprint()
            decisions = Oracle1Collector(config, runner).collect_pre_step(
                environment,
                observations,
                proposals,
                hidden,
                episode_id=0,
                global_environment_step=1,
                policy_version=0,
            )
            self.assertEqual(len(decisions), 1)
            decision = decisions[0]
            routes = tuple(branch.route for branch in decision.branches)
            if route == "defer":
                self.assertEqual(routes, ("local", remote, "defer"))
                self.assertEqual(decision.branches[-1].role, "FACTUAL_DEFER_SHADOW")
            elif route == "local":
                self.assertEqual(routes, ("local", remote))
                self.assertEqual(decision.branches[0].role, "FACTUAL_ROUTE_REUSE")
            else:
                self.assertEqual(routes, ("local", remote))
                self.assertEqual(decision.branches[1].role, "FACTUAL_ROUTE_REUSE")
            self.assertEqual(initial_state, environment.state_fingerprint())

    def test_common_randomness_and_branch_local_hidden_gates(self) -> None:
        config, environment, observations, proposals, _ = active_fixture()
        actor = CAGATMAPPOActor(config).eval()
        runner = FrozenOraclePolicyRunnerCache(config).get(actor, 0)
        hidden = actor.initial_hidden(1, device="cpu", dtype=torch.float32)
        decisions = Oracle1Collector(config, runner).collect_pre_step(
            environment, observations, proposals, hidden,
            episode_id=0, global_environment_step=1, policy_version=0,
        )
        branches = decisions[0].branches
        self.assertTrue(all(branch.status == VALID_BRANCH_STATUS for branch in branches))
        self.assertTrue(all(branch.initial_hidden_digest == branches[0].initial_hidden_digest for branch in branches))
        self.assertEqual(len({id(branch.trace) for branch in branches}), len(branches))
        self.assertEqual(len({branch.exogenous_trace[0] for branch in branches}), 1)

    def test_crn_fail_fast_gate(self) -> None:
        left = replace(make_branch("local"), exogenous_trace=("baseline",))
        right = replace(make_branch(1), exogenous_trace=("mismatch",))
        with self.assertRaisesRegex(OracleReliabilityError, "COMMON_RANDOMNESS_GATE = FAIL"):
            Oracle1Collector._check_common_randomness((left, right))

    def test_focal_terminal_time_assertion_gate(self) -> None:
        config = make_oracle_config(horizon=8)
        environment = U2UMECEnvironment(config)
        environment.reset()
        assert environment.lifecycle is not None

        completed = Task.create(
            task_id=100,
            source_uav=0,
            data_bits=1.0,
            cpu_cycles=1.0,
            arrival_slot=0,
            deadline_budget_slots=10,
        )
        completed.bind_local(binding_slot=1)
        completed.enter_cpu(transition_slot=2, service_eligible_slot=3)
        completed.remaining_cycles = 0.0
        completed.complete(3)
        environment.lifecycle.tasks[completed.task_id] = completed
        done_payload = _task_terminal_payload(environment, completed.task_id, 1)
        self.assertIsNotNone(done_payload)
        self.assertEqual(done_payload["terminal_class"], "done")
        self.assertEqual(done_payload["terminal_slot"], 3)
        self.assertEqual(done_payload["route_to_terminal_slots"], 2)
        self.assertEqual(
            done_payload["completion_delay_s"],
            (completed.completion_slot - completed.arrival_slot)
            * config.environment.slot_duration_s,
        )

        expired = Task.create(
            task_id=101,
            source_uav=0,
            data_bits=1.0,
            cpu_cycles=1.0,
            arrival_slot=0,
            deadline_budget_slots=5,
        )
        expired.expire(5)
        environment.lifecycle.tasks[expired.task_id] = expired
        environment.slot = 5
        expired_payload = _task_terminal_payload(environment, expired.task_id, 1)
        self.assertEqual(expired_payload["terminal_class"], "expired")
        self.assertEqual(expired_payload["terminal_slot"], 5)
        self.assertEqual(expired_payload["route_to_terminal_slots"], 4)
        self.assertIsNone(expired_payload["completion_delay_s"])

        truncated = Task.create(
            task_id=102,
            source_uav=0,
            data_bits=1.0,
            cpu_cycles=1.0,
            arrival_slot=0,
            deadline_budget_slots=10,
        )
        truncated.mark_truncated()
        environment.lifecycle.tasks[truncated.task_id] = truncated
        environment.slot = 7
        truncated_payload = _task_terminal_payload(environment, truncated.task_id, 1)
        self.assertEqual(truncated_payload["terminal_class"], "truncated")
        self.assertEqual(truncated_payload["terminal_slot"], 7)
        self.assertEqual(truncated_payload["route_to_terminal_slots"], 6)
        self.assertIsNone(truncated_payload["completion_delay_s"])

    def test_energy_attribution_and_classification_priority_gate(self) -> None:
        exact = _energy_from_step(
            {
                "energy": {"debits": [{"total_energy_j": 4.0}]},
                "service": {
                    "links": [
                        {
                            "transmit_energy_j": 2.0,
                            "task_services": [{"task_id": 7, "amount": 1.0}],
                        }
                    ],
                    "cpu": [{"task_id": 7, "cpu_energy_j": 3.0, "executor_uav": 0}],
                },
            },
            focal_task_id=7,
            source_uav=0,
        )
        self.assertEqual(exact["source_tx_status"], ATTRIBUTION_EXACT)
        self.assertEqual(exact["focal_cpu_status"], ATTRIBUTION_EXACT)

        proxy = _energy_from_step(
            {
                "energy": {"debits": [{"total_energy_j": 5.0}]},
                "service": {
                    "links": [
                        {
                            "transmit_energy_j": 4.0,
                            "task_services": [
                                {"task_id": 7, "amount": 1.0},
                                {"task_id": 8, "amount": 3.0},
                            ],
                        }
                    ],
                    "cpu": [{"task_id": 7, "cpu_energy_j": 3.0, "executor_uav": 1}],
                },
            },
            focal_task_id=7,
            source_uav=0,
        )
        self.assertEqual(proxy["source_tx_status"], ATTRIBUTION_ALLOCATED_PROXY)
        self.assertEqual(proxy["focal_cpu_status"], ATTRIBUTION_EXACT)
        self.assertEqual(proxy["helper_cpu"], 3.0)

        unavailable = _energy_from_step({}, focal_task_id=7, source_uav=0)
        self.assertEqual(unavailable["source_tx_status"], ATTRIBUTION_NOT_AVAILABLE)
        self.assertEqual(unavailable["focal_cpu_status"], ATTRIBUTION_NOT_AVAILABLE)

        completed = make_branch("local", terminal="done", slots=4, energy=100.0)
        expired = make_branch(1, terminal="expired", slots=0, energy=0.0)
        self.assertEqual(
            compare_branch_outcomes(
                completed,
                expired,
                slot_duration_s=1.0,
                energy_tolerance_j=0.1,
            ),
            1,
        )
        fast = make_branch("local", slots=1, energy=100.0)
        slow = make_branch(1, slots=3, energy=0.0)
        self.assertEqual(
            compare_branch_outcomes(
                fast,
                slow,
                slot_duration_s=1.0,
                energy_tolerance_j=0.1,
            ),
            1,
        )
        lower_energy = make_branch("local", slots=1, energy=1.0)
        higher_energy = make_branch(1, slots=2, energy=2.0)
        self.assertEqual(
            compare_branch_outcomes(
                lower_energy,
                higher_energy,
                slot_duration_s=1.0,
                energy_tolerance_j=0.1,
            ),
            1,
        )

    def test_route_active_idle_is_rejected_and_no_remote_is_ineligible(self) -> None:
        config, environment, observations, proposals, _ = active_fixture()
        actor = CAGATMAPPOActor(config).eval()
        runner = FrozenOraclePolicyRunnerCache(config).get(actor, 0)
        hidden = actor.initial_hidden(1, device="cpu", dtype=torch.float32)
        idle = list(proposals)
        idle[0] = replace(idle[0], route="idle")
        with self.assertRaises(Exception):
            Oracle1Collector(config, runner).collect_pre_step(
                environment, observations, tuple(idle), hidden,
                episode_id=0, global_environment_step=1, policy_version=0,
            )
        contract = observations[0].action_masks
        route_mask = [value in {"local", "defer"} for value in contract.route_domain]
        no_remote_observation = replace(
            observations[0],
            action_masks=replace(contract, route_mask=route_mask),
        )
        no_remote = (no_remote_observation, observations[1])
        self.assertEqual(
            Oracle1Collector(config, runner).collect_pre_step(
                environment, no_remote, proposals, hidden,
                episode_id=0, global_environment_step=1, policy_version=0,
            ),
            (),
        )

    def test_policy_miss_classification_synthetic_cases(self) -> None:
        cases = (
            ("defer_local_better", "defer", (make_branch("local", slots=1), make_branch(1, slots=3), make_branch("defer", terminal="expired", slots=2)), "local", "YES"),
            ("defer_remote_better", "defer", (make_branch("local", slots=3), make_branch(1, slots=1), make_branch("defer", terminal="expired", slots=2)), 1, "YES"),
            ("local_remote_better", "local", (make_branch("local", slots=3), make_branch(1, slots=1)), 1, "YES"),
            ("remote_zero_remote_one_better", 0, (make_branch("local", slots=4), make_branch(0, slots=3), make_branch(1, slots=1)), 1, "YES"),
            ("factual_already_best", "local", (make_branch("local", slots=1), make_branch(1, slots=3)), "local", "NO"),
        )
        for name, factual, branches, expected_best, expected_miss in cases:
            with self.subTest(name=name):
                committed = tuple(branch.route for branch in branches if branch.route != "defer")
                result = classify_oracle_routes(
                    branches,
                    factual_route=factual,
                    committed_routes=committed,
                    evaluated_route_set=tuple(branch.route for branch in branches),
                )
                self.assertEqual(result["BEST_COMMITTED_ROUTE"], expected_best)
                self.assertEqual(result["ORACLE1_ROUTE_POLICY_MISS"], expected_miss)

    def test_tie_invalid_unresolved_and_illegal_remote_classification_gates(self) -> None:
        tie = classify_oracle_routes(
            (make_branch("local", slots=1, energy=1.0), make_branch(1, slots=1, energy=1.0)),
            factual_route="local",
            committed_routes=("local", 1),
        )
        self.assertEqual(tie["ORACLE_TIE"], "YES")
        self.assertEqual(tie["ORACLE1_ROUTE_POLICY_MISS"], "NO")

        invalid = classify_oracle_routes(
            (make_branch("local", valid=False), make_branch(1, slots=1)),
            factual_route="local",
            committed_routes=("local", 1),
        )
        self.assertEqual(invalid["ORACLE1_ROUTE_POLICY_MISS"], UNRESOLVED)

        unresolved = classify_oracle_routes(
            (make_branch("local", slots=1, energy=None), make_branch(1, slots=1, energy=1.0)),
            factual_route="local",
            committed_routes=("local", 1),
        )
        self.assertEqual(unresolved["ORACLE1_ROUTE_POLICY_MISS"], UNRESOLVED)

        excluded = classify_oracle_routes(
            (make_branch("local", slots=1), make_branch(1, slots=3), make_branch(2, slots=0)),
            factual_route="local",
            committed_routes=("local", 1),
            evaluated_route_set=("local", 1),
        )
        self.assertEqual(excluded["evaluated_route_set"], ["local", 1])
        self.assertNotIn(2, excluded["BEST_COMMITTED_ROUTE_SET"])

    def test_best_route_scope_gate_for_factual_defer(self) -> None:
        branches = (
            make_branch("local", slots=1),
            make_branch(1, slots=3),
            make_branch("defer", terminal="expired", slots=1),
        )
        result = classify_oracle_routes(
            branches,
            factual_route="defer",
            committed_routes=("local", 1),
            evaluated_route_set=("local", 1, "defer"),
        )
        self.assertEqual(result["ORACLE1_BEST_ROUTE"], "local")
        self.assertEqual(result["BEST_COMMITTED_ROUTE"], "local")
        self.assertEqual(result["ORACLE1_BEST_ROUTE_SCOPE"], "EVALUATED_ORACLE1_BRANCHES_ONLY")

    def test_frozen_runner_version_and_policy_isolation_gates(self) -> None:
        config, _, observations, _, _ = active_fixture()
        actor = CAGATMAPPOActor(config).eval()
        cache = FrozenOraclePolicyRunnerCache(config)
        first = cache.get(actor, 0)
        same = cache.get(actor, 0)
        self.assertIs(first, same)
        before = [parameter.detach().clone() for parameter in actor.parameters()]
        hidden = actor.initial_hidden(1, device="cpu", dtype=torch.float32)
        one = first.act(observations, hidden)
        two = first.act(observations, hidden)
        self.assertFalse(one.hidden_out.data_ptr() == two.hidden_out.data_ptr())
        two_before = two.hidden_out.detach().clone()
        one.hidden_out.add_(1.0)
        self.assertFalse(torch.equal(one.hidden_out, two.hidden_out))
        self.assertTrue(torch.equal(two.hidden_out, two_before))
        self.assertFalse(first.actor.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in first.actor.parameters()))
        self.assertEqual([parameter.detach().clone().tolist() for parameter in actor.parameters()], [value.tolist() for value in before])
        provenance_same_math = cache.get(
            actor,
            0,
            checkpoint_sha256="1" * 64,
            source="fresh_checkpoint",
        )
        resumed_provenance = cache.get(
            actor,
            0,
            checkpoint_sha256="2" * 64,
            source="resumed_checkpoint",
        )
        self.assertIs(first, provenance_same_math)
        self.assertIs(first, resumed_provenance)
        self.assertEqual(
            first.runner_identity,
            FrozenRunnerIdentity(0, first.actor_state_digest),
        )
        self.assertEqual(
            cache.current_identity.runner_provenance,
            FrozenRunnerProvenance(
                checkpoint_sha256="2" * 64,
                source="resumed_checkpoint",
            ),
        )
        with torch.no_grad():
            next(iter(actor.parameters())).add_(0.001)
        with self.assertRaises(OracleReliabilityError):
            cache.get(actor, 0)
        changed_digest = cache.get(actor, 1)
        self.assertIsNot(first, changed_digest)
        version_changed = cache.get(actor, 1)
        self.assertIs(changed_digest, version_changed)

    def test_tiny_decision_budget_gate(self) -> None:
        budget = OracleCollectionBudget(
            target_selected_decisions=2,
            max_factual_environment_transitions=8,
            max_shadow_transitions=20,
        )
        self.assertFalse(budget.should_stop())
        budget.record_selected_decision()
        budget.observe_factual_transitions(1)
        budget.record_selected_decision()
        self.assertTrue(budget.should_stop())
        self.assertEqual(budget.stop_reason, "TARGET_SELECTED_DECISIONS")
        empty = OracleCollectionBudget()
        self.assertIsNone(empty.target_selected_decisions)
        self.assertIsNone(empty.max_factual_environment_transitions)
        self.assertIsNone(empty.max_shadow_transitions)

    def test_oracle_sidecar_schema_and_isolation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_oracle_config(output_dir=directory)
            identity = FrozenPolicyIdentity(0, "0" * 64)
            header = oracle_schema_header(config, identity)
            writer = RouteOracleArtifactWriter(config, header)
            branch = make_branch("local", slots=1)
            decision = type("Decision", (), {})
            # Build the real immutable record without invoking a shadow rollout.
            item = OracleDecisionResult(
                decision_key="decision-0",
                episode_id=0,
                global_environment_step=0,
                source_uav=0,
                task_id=0,
                route_slot=0,
                factual_route="local",
                committed_routes=("local", 1),
                evaluated_route_set=("local", 1),
                policy_identity=identity,
                initial_state_fingerprint="a" * 64,
                initial_rng_fingerprint="b" * 64,
                initial_exogenous_fingerprint="c" * 64,
                initial_hidden_digest="d" * 64,
                branches=(branch, make_branch(1, slots=2)),
                classification=classify_oracle_routes(
                    (branch, make_branch(1, slots=2)),
                    factual_route="local",
                    committed_routes=("local", 1),
                ),
                selector_input={"schema_version": 1},
            )
            writer.write(item)
            summary = inspect_route_oracle_artifact(config)
            self.assertEqual(summary["decision_count"], 1)
            self.assertEqual(summary["branch_count"], 2)
            content = Path(summary["path"]).read_text(encoding="utf-8")
            self.assertIn("evaluated_route_set", content)
            self.assertNotIn("state_dump", content)
            self.assertFalse(Path(directory, "environment").exists())

            record = item.to_decision_record()
            self.assertEqual(record["factual_route_branch_validity"], VALID_BRANCH_STATUS)
            self.assertTrue(record["factual_branch_reused"])
            defer_branch = make_branch("defer", terminal="expired", slots=1)
            defer_branches = (branch, make_branch(1, slots=2), defer_branch)
            defer_item = replace(
                item,
                factual_route="defer",
                evaluated_route_set=("local", 1, "defer"),
                branches=defer_branches,
                classification=classify_oracle_routes(
                    defer_branches,
                    factual_route="defer",
                    committed_routes=("local", 1),
                    evaluated_route_set=("local", 1, "defer"),
                ),
            )
            defer_record = defer_item.to_decision_record()
            self.assertEqual(
                defer_record["factual_route_branch_validity"], VALID_BRANCH_STATUS
            )
            self.assertFalse(defer_record["factual_branch_reused"])

    def test_checkpoint_resume_policy_identity_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_oracle_config(output_dir=directory)
            identity_before = FrozenPolicyIdentity(
                0,
                "a" * 64,
                checkpoint_sha256="1" * 64,
                source="initial_checkpoint",
            )
            identity_after = FrozenPolicyIdentity(
                0,
                "a" * 64,
                checkpoint_sha256="2" * 64,
                source="resumed_checkpoint",
            )
            self.assertEqual(identity_before.runner_identity, identity_after.runner_identity)
            self.assertNotEqual(
                identity_before.runner_provenance,
                identity_after.runner_provenance,
            )
            writer_before = RouteOracleArtifactWriter(
                config,
                oracle_schema_header(config, identity_before),
            )
            branch = make_branch("local", slots=1)
            branches = (branch, make_branch(1, slots=2))
            decision = OracleDecisionResult(
                decision_key="resume-decision",
                episode_id=0,
                global_environment_step=0,
                source_uav=0,
                task_id=0,
                route_slot=0,
                factual_route="local",
                committed_routes=("local", 1),
                evaluated_route_set=("local", 1),
                policy_identity=identity_before,
                initial_state_fingerprint="a" * 64,
                initial_rng_fingerprint="b" * 64,
                initial_exogenous_fingerprint="c" * 64,
                initial_hidden_digest="d" * 64,
                branches=branches,
                classification=classify_oracle_routes(
                    branches,
                    factual_route="local",
                    committed_routes=("local", 1),
                ),
                selector_input={"schema_version": 1},
            )
            writer_before.write(decision)
            writer_after = RouteOracleArtifactWriter(
                config,
                oracle_schema_header(config, identity_after),
                allow_existing=True,
            )
            self.assertEqual(writer_after.header, writer_before.header)
            summary = inspect_route_oracle_artifact(config)
            self.assertEqual(summary["decision_count"], 1)

    def test_factual_behavior_neutrality_and_checkpoint_training_state_gate(self) -> None:
        config, environment, observations, proposals, _ = active_fixture()
        control = U2UMECEnvironment(config)
        control.reset()
        control.step(control.canonical_proposals())
        actor = CAGATMAPPOActor(config).eval()
        runner = FrozenOraclePolicyRunnerCache(config).get(actor, 0)
        hidden = actor.initial_hidden(1, device="cpu", dtype=torch.float32)
        Oracle1Collector(config, runner).collect_pre_step(
            environment, observations, proposals, hidden,
            episode_id=0, global_environment_step=1, policy_version=0,
        )
        treatment = environment.step(proposals)
        control_result = control.step(proposals)
        self.assertEqual(treatment.reward, control_result.reward)
        self.assertEqual(treatment.info, control_result.info)
        self.assertEqual(environment.state_fingerprint(), control.state_fingerprint())
        self.assertEqual(environment.rng_fingerprint(), control.rng_fingerprint())

    def test_training_behavior_neutrality_and_checkpoint_state_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            off_config = make_trainer_oracle_config(
                str(Path(directory) / "off"), enabled=False
            )
            on_config = make_trainer_oracle_config(
                str(Path(directory) / "on"), enabled=True
            )
            torch.manual_seed(20260903)
            template_actor = CAGATMAPPOActor(off_config)
            template_critic = MAPPOCentralizedCritic(off_config)

            def run(config, budget):
                factory = RecordingFactory()
                updater_holder: list[DeterministicAdamUpdater] = []

                def updater_factory(actor, critic, updater_config, distribution):
                    updater = DeterministicAdamUpdater(
                        actor, critic, updater_config, distribution
                    )
                    updater_holder.append(updater)
                    return updater

                trainer = CAGATMAPPOTrainer(
                    config,
                    actor=copy.deepcopy(template_actor),
                    critic=copy.deepcopy(template_critic),
                    environment_factory=factory,
                    updater_factory=updater_factory,
                    progress_logger=lambda _: None,
                    oracle_budget=budget,
                )
                result = trainer.train()
                self.assertEqual(len(updater_holder), 1)
                return trainer, factory, updater_holder[0], result

            off = run(off_config, None)
            on = run(
                on_config,
                OracleCollectionBudget(
                    target_selected_decisions=1,
                    max_factual_environment_transitions=256,
                    max_shadow_transitions=512,
                ),
            )
            off_trainer, off_factory, off_updater, off_result = off
            on_trainer, on_factory, on_updater, on_result = on

            self.assertEqual(off_result, on_result)
            self.assertEqual(len(off_factory.environments), len(on_factory.environments))
            for off_environment, on_environment in zip(
                off_factory.environments, on_factory.environments
            ):
                assert_nested_exact(
                    self,
                    off_environment.factual_actions,
                    on_environment.factual_actions,
                    "factual_actions",
                )
                self.assertEqual(
                    len(off_environment.factual_results),
                    len(on_environment.factual_results),
                )
                for off_step, on_step in zip(
                    off_environment.factual_results,
                    on_environment.factual_results,
                ):
                    self.assertEqual(off_step.reward, on_step.reward)
                    self.assertEqual(off_step.terminated, on_step.terminated)
                    self.assertEqual(off_step.truncated, on_step.truncated)
                    off_info = copy.deepcopy(off_step.info)
                    on_info = copy.deepcopy(on_step.info)
                    off_info.pop("run_id", None)
                    on_info.pop("run_id", None)
                    off_info.pop("config_hash", None)
                    on_info.pop("config_hash", None)
                    assert_nested_exact(self, off_info, on_info, "step.info")
                self.assertEqual(
                    off_environment.factual_state_fingerprints,
                    on_environment.factual_state_fingerprints,
                )
                self.assertEqual(
                    off_environment.factual_rng_fingerprints,
                    on_environment.factual_rng_fingerprints,
                )

            assert_nested_exact(
                self,
                off_updater.captured_transitions,
                on_updater.captured_transitions,
                "rollout_buffer",
            )
            assert_nested_exact(
                self,
                off_trainer.actor.state_dict(),
                on_trainer.actor.state_dict(),
                "actor_state",
            )
            assert_nested_exact(
                self,
                off_trainer.critic.state_dict(),
                on_trainer.critic.state_dict(),
                "critic_state",
            )
            assert_nested_exact(
                self,
                off_updater.optimizers.actor_optimizer.state_dict(),
                on_updater.optimizers.actor_optimizer.state_dict(),
                "actor_optimizer_state",
            )
            assert_nested_exact(
                self,
                off_updater.optimizers.critic_optimizer.state_dict(),
                on_updater.optimizers.critic_optimizer.state_dict(),
                "critic_optimizer_state",
            )
            self.assertEqual(off_trainer.policy_version, on_trainer.policy_version)
            self.assertEqual(off_trainer._transitions, on_trainer._transitions)
            self.assertEqual(off_trainer._optimized, on_trainer._optimized)
            self.assertEqual(
                off_trainer.policy_generator.get_state().tolist(),
                on_trainer.policy_generator.get_state().tolist(),
            )
            self.assertEqual(
                [environment.rng_fingerprint() for environment in off_factory.environments],
                [environment.rng_fingerprint() for environment in on_factory.environments],
            )

            off_payload = checkpoint_payload_for(off_trainer)
            on_payload = checkpoint_payload_for(on_trainer)
            for key in (
                "training_state",
                "model_state",
                "optimizer_state",
                "policy_rng_state",
                "active_rollout_state",
                "diagnostics_state",
            ):
                assert_nested_exact(
                    self,
                    off_payload[key],
                    on_payload[key],
                    f"checkpoint.{key}",
                )
            off_snapshot = copy.deepcopy(off_payload["config_snapshot"])
            on_snapshot = copy.deepcopy(on_payload["config_snapshot"])
            for snapshot in (off_snapshot, on_snapshot):
                snapshot["training"]["mappo"].pop(
                    "route_oracle_counterfactual_enabled", None
                )
                snapshot["training"]["mappo"].pop(
                    "route_oracle_selection_rate_ppm", None
                )
                snapshot["output"].pop("route_oracle_counterfactual_filename", None)
                snapshot["output"]["logs_dir"] = "__oracle_neutrality__"
                snapshot["_metadata"].pop("run_id", None)
                snapshot["_metadata"].pop("config_hash", None)
            self.assertEqual(off_snapshot, on_snapshot)
