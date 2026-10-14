"""Static and numerical gates for rollout-capped route-event N-step V1."""

from __future__ import annotations

from dataclasses import replace
import math
import unittest

import torch

from src.config import (
    ActorRatioMode,
    AgentCreditMode,
    CHECKPOINT_KIND_PERIODIC_RESUME,
    CHECKPOINT_SCHEMA_VERSION,
    ConfigError,
    RouteCreditMode,
    RouteDecoderMode,
    RunConfig,
    validate_mappo_checkpoint_resume_compatibility,
)
from src.models.ca_gat_mappo_checkpoint import (
    restore_rollout_transition,
    serialize_rollout_transition,
)
from src.models.ca_gat_mappo_ppo import (
    compute_configured_ppo_objective_and_loss,
)
from src.models.ca_gat_mappo_rollout import CAGATMAPPORolloutBuffer
from src.models.ca_gat_mappo_route_nstep import (
    RouteNstepError,
    RouteNstepSample,
    build_route_credit_step_metadata,
    compute_rollout_capped_route_event_nstep,
)
from src.models.ca_gat_mappo_route_telemetry import (
    collect_route_telemetry,
)
from src.models.ca_gat_mappo_trainer import CAGATMAPPOTrainer
from src.models.ca_gat_mappo_update import build_recurrent_ppo_minibatch
from src.training_artifacts import _base_record
from tests.test_ca_gat_mappo_route_specific_gae import ppo_inputs
from tests.test_ca_gat_mappo_route_telemetry import (
    active_route_fixture,
    evaluate_route_steps,
)
from tests.test_ca_gat_mappo_update import (
    make_seed_transition,
    make_update_config,
)


def nstep_config(*, route_lambda: float = 1.0) -> RunConfig:
    base = replace(RunConfig(), mode="rl", method_id="ca_gat_mappo")
    mappo = replace(
        base.training.mappo,
        actor_ratio_mode=ActorRatioMode.BRANCH_SPECIFIC,
        agent_credit_mode=AgentCreditMode.ROLE_DECOMPOSED,
        route_credit_mode=(
            RouteCreditMode.ROLLOUT_CAPPED_ROUTE_EVENT_NSTEP
        ),
        route_gae_lambda=route_lambda,
        training_device="cpu",
    )
    result = replace(
        base, training=replace(base.training, mappo=mappo)
    )
    result.validate()
    return result


def decoder_credit_config(
    route_decoder_mode: RouteDecoderMode,
    route_credit_mode: RouteCreditMode,
) -> RunConfig:
    """Return a config that satisfies the N-step mode's independent gates."""

    base = replace(RunConfig(), mode="rl", method_id="ca_gat_mappo")
    mappo = replace(
        base.training.mappo,
        actor_ratio_mode=ActorRatioMode.BRANCH_SPECIFIC,
        agent_credit_mode=AgentCreditMode.ROLE_DECOMPOSED,
        route_decoder_mode=route_decoder_mode,
        route_credit_mode=route_credit_mode,
        training_device="cpu",
    )
    return replace(base, training=replace(base.training, mappo=mappo))


def decision(
    episode: int,
    source: int,
    task_id: object,
    index: int,
    *,
    category: str = "local",
    policy_version: int = 0,
) -> dict[str, object]:
    key = [episode, source, task_id, index]
    return {
        "episode_id": episode,
        "source_uav": source,
        "task_id": task_id,
        "category": category,
        "branch_event_key": key,
        "formal_route_event_key": (
            None if category == "defer" else key
        ),
        "global_rollout_index": index,
        "policy_version": policy_version,
    }


def terminal(
    episode: int,
    source: int,
    task_id: object,
    index: int,
    kind: str,
) -> dict[str, object]:
    return {
        "episode_id": episode,
        "source_uav": source,
        "task_id": task_id,
        "kind": kind,
        "global_rollout_index": index,
    }


def step(
    episode: int,
    index: int,
    *,
    decisions: tuple[dict[str, object], ...] = (),
    terminals: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    return {
        "rejection": (),
        "downgrade": (),
        "canonicalization": (),
        "route_credit_step": {
            "episode_id": episode,
            "global_rollout_index": index,
            "policy_version": 0,
            "decisions": list(decisions),
            "terminals": list(terminals),
        },
    }


def estimate(
    *,
    reward: torch.Tensor,
    old_value: torch.Tensor,
    bootstrap_value: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    episode_boundary: torch.Tensor,
    route_active: torch.Tensor,
    metadata: tuple[dict[str, object], ...],
    base_advantage: torch.Tensor | None = None,
):
    return compute_rollout_capped_route_event_nstep(
        reward=reward,
        old_value=old_value,
        bootstrap_value=bootstrap_value,
        bootstrap_mask=bootstrap_mask,
        episode_boundary=episode_boundary,
        route_active=route_active,
        base_advantage=(
            torch.zeros_like(reward)
            if base_advantage is None
            else base_advantage
        ),
        transition_metadata=metadata,
        gamma=0.99,
    )


class RolloutCappedRouteNstepConfigTests(unittest.TestCase):
    def test_hierarchical_decoder_accepts_shared_gae(self) -> None:
        config = decoder_credit_config(
            RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1,
            RouteCreditMode.SHARED_GAE,
        )
        config.validate()

    def test_hierarchical_decoder_accepts_rollout_capped_nstep(self) -> None:
        config = decoder_credit_config(
            RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1,
            RouteCreditMode.ROLLOUT_CAPPED_ROUTE_EVENT_NSTEP,
        )
        config.validate()

    def test_hierarchical_decoder_rejects_route_specific_gae(self) -> None:
        config = decoder_credit_config(
            RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1,
            RouteCreditMode.ROUTE_SPECIFIC_GAE,
        )
        with self.assertRaisesRegex(
            ConfigError,
            "hierarchical_option_aware_v1 requires.*shared_gae.*rollout_capped",
        ):
            config.validate()

    def test_candidate_decoder_rejects_rollout_capped_nstep(self) -> None:
        config = decoder_credit_config(
            RouteDecoderMode.CANDIDATE_AWARE_V1,
            RouteCreditMode.ROLLOUT_CAPPED_ROUTE_EVENT_NSTEP,
        )
        with self.assertRaisesRegex(ConfigError, "requires.*shared_gae"):
            config.validate()

    def test_option_decoder_rejects_rollout_capped_nstep(self) -> None:
        config = decoder_credit_config(
            RouteDecoderMode.OPTION_AWARE_V1,
            RouteCreditMode.ROLLOUT_CAPPED_ROUTE_EVENT_NSTEP,
        )
        with self.assertRaisesRegex(ConfigError, "requires.*shared_gae"):
            config.validate()

    def test_mode_contract_and_irrelevant_route_lambda(self) -> None:
        first = nstep_config(route_lambda=0.0)
        second = nstep_config(route_lambda=1.0)
        self.assertEqual(first.config_hash, second.config_hash)
        self.assertNotIn(
            "route_gae_lambda",
            first.resolved_dict()["training"]["mappo"],
        )
        self.assertIsNone(
            _base_record(first, "diagnostic_only")[
                "effective_route_gae_lambda"
            ]
        )

    def test_mode_requires_branch_specific_role_credit_and_frozen_gamma_lambda(
        self,
    ) -> None:
        valid = nstep_config()
        invalid = (
            replace(valid.training.mappo, gamma=0.98),
            replace(valid.training.mappo, gae_lambda=0.9),
            replace(
                valid.training.mappo,
                actor_ratio_mode=ActorRatioMode.JOINT,
            ),
            replace(
                valid.training.mappo,
                agent_credit_mode=AgentCreditMode.TEAM,
            ),
        )
        for mappo in invalid:
            with self.assertRaises(ConfigError):
                replace(
                    valid,
                    training=replace(valid.training, mappo=mappo),
                ).validate()

    def test_legacy_default_hash_payload_is_still_exact(self) -> None:
        detected = RunConfig()
        frozen = replace(
            detected,
            reproducibility=replace(
                detected.reproducibility,
                python_version="3.12.13",
                numpy_version="2.3.5",
                torch_version="not-installed",
                cuda_version="not-available",
                miniconda_version="not-installed-or-not-active",
            ),
        )
        self.assertEqual(
            frozen.config_hash,
            "5f37790be06c534decdf1f618daa60bb10f49a425214dcd8ed9071708d98bbeb",
        )
        payload = frozen.resolved_dict()["training"]["mappo"]
        self.assertNotIn("route_credit_mode", payload)
        self.assertNotIn("route_gae_lambda", payload)


class RolloutCappedRouteNstepHandCalculationTests(unittest.TestCase):
    def _one_step(self, kind: str, *, boundary: bool) -> None:
        output = estimate(
            reward=torch.tensor([[2.0]]),
            old_value=torch.tensor([[0.5]]),
            bootstrap_value=torch.tensor([[3.0]]),
            bootstrap_mask=torch.tensor([not boundary]),
            episode_boundary=torch.tensor([boundary]),
            route_active=torch.tensor([[True]]),
            metadata=(
                step(
                    0,
                    10,
                    decisions=(decision(0, 0, 7, 10),),
                    terminals=(terminal(0, 0, 7, 10, kind),),
                ),
            ),
        )
        expected = 2.0 - 0.5
        if not boundary:
            expected += 0.99 * 3.0
        self.assertAlmostEqual(
            output.advantage[0, 0].item(), expected, places=6
        )
        sample = output.samples[0][0]
        assert sample is not None
        self.assertEqual(sample.horizon, 1)
        self.assertEqual(
            sample.stop_kind,
            {
                "completed": "task_completed",
                "expired": "task_expired",
                "truncated": "task_truncated_episode_boundary",
            }[kind],
        )
        self.assertEqual(sample.bootstrap_used, not boundary)

    def test_n1_completed_bootstraps(self) -> None:
        self._one_step("completed", boundary=False)

    def test_n1_expired_bootstraps(self) -> None:
        self._one_step("expired", boundary=False)

    def test_n1_truncated_episode_boundary_does_not_bootstrap(self) -> None:
        self._one_step("truncated", boundary=True)

    def test_multistep_includes_terminal_reward_and_saved_next_value(self) -> None:
        output = estimate(
            reward=torch.tensor([[1.0], [2.0], [4.0]]),
            old_value=torch.tensor([[0.25], [0.5], [0.75]]),
            bootstrap_value=torch.tensor([[0.5], [0.75], [5.0]]),
            bootstrap_mask=torch.tensor([True, True, True]),
            episode_boundary=torch.zeros(3, dtype=torch.bool),
            route_active=torch.tensor([[True], [False], [False]]),
            metadata=(
                step(0, 0, decisions=(decision(0, 0, 7, 0),)),
                step(0, 1),
                step(
                    0,
                    2,
                    terminals=(terminal(0, 0, 7, 2, "completed"),),
                ),
            ),
        )
        expected = (
            1.0 + 0.99 * 2.0 + 0.99**2 * 4.0
            + 0.99**3 * 5.0 - 0.25
        )
        self.assertAlmostEqual(
            output.advantage[0, 0].item(), expected, places=5
        )
        sample = output.samples[0][0]
        assert sample is not None
        self.assertEqual(sample.horizon, 3)

    def test_rollout_cap_bootstraps_at_q_without_future_data(self) -> None:
        output = estimate(
            reward=torch.tensor([[1.0], [2.0]]),
            old_value=torch.tensor([[0.25], [0.5]]),
            bootstrap_value=torch.tensor([[0.5], [7.0]]),
            bootstrap_mask=torch.tensor([True, True]),
            episode_boundary=torch.zeros(2, dtype=torch.bool),
            route_active=torch.tensor([[True], [False]]),
            metadata=(
                step(
                    0, 50, decisions=(decision(0, 0, 8, 50),)
                ),
                step(0, 51),
            ),
        )
        expected = 1.0 + 0.99 * 2.0 + 0.99**2 * 7.0 - 0.25
        self.assertAlmostEqual(
            output.advantage[0, 0].item(), expected, places=5
        )
        sample = output.samples[0][0]
        assert sample is not None
        self.assertEqual(sample.stop_kind, "rollout_boundary")
        self.assertFalse(sample.terminal_before_rollout)
        frozen = output.advantage.clone()
        _ = estimate(
            reward=torch.tensor([[99.0]]),
            old_value=torch.tensor([[1.0]]),
            bootstrap_value=torch.tensor([[2.0]]),
            bootstrap_mask=torch.tensor([True]),
            episode_boundary=torch.tensor([False]),
            route_active=torch.tensor([[False]]),
            metadata=(step(0, 52),),
        )
        self.assertTrue(torch.equal(output.advantage, frozen))

    def test_direct_return_matches_td_telescope_and_has_no_off_by_one(self) -> None:
        reward = torch.tensor([[0.5], [1.5], [2.5]])
        values = torch.tensor([[1.0], [2.0], [3.0]])
        next_values = torch.tensor([[2.0], [3.0], [4.0]])
        output = estimate(
            reward=reward,
            old_value=values,
            bootstrap_value=next_values,
            bootstrap_mask=torch.ones(3, dtype=torch.bool),
            episode_boundary=torch.zeros(3, dtype=torch.bool),
            route_active=torch.tensor([[True], [False], [False]]),
            metadata=(
                step(0, 0, decisions=(decision(0, 0, 9, 0),)),
                step(0, 1),
                step(
                    0,
                    2,
                    terminals=(terminal(0, 0, 9, 2, "expired"),),
                ),
            ),
        )
        residual = reward[:, 0] + 0.99 * next_values[:, 0] - values[:, 0]
        td_sum = sum(
            (0.99**index) * residual[index]
            for index in range(3)
        )
        self.assertAlmostEqual(
            output.advantage[0, 0].item(),
            float(td_sum.item()),
            places=5,
        )
        sample = output.samples[0][0]
        assert sample is not None
        self.assertEqual(sample.horizon, 3)


class RolloutCappedRouteNstepIdentityTests(unittest.TestCase):
    def test_local_remote_and_defer_share_one_formula(self) -> None:
        outputs = []
        for category in ("local", "remote", "defer"):
            outputs.append(
                estimate(
                    reward=torch.tensor([[1.0], [2.0]]),
                    old_value=torch.tensor([[0.5], [0.75]]),
                    bootstrap_value=torch.tensor([[0.75], [3.0]]),
                    bootstrap_mask=torch.tensor([True, True]),
                    episode_boundary=torch.tensor([False, False]),
                    route_active=torch.tensor([[True], [False]]),
                    metadata=(
                        step(
                            0,
                            0,
                            decisions=(
                                decision(
                                    0, 0, 9, 0, category=category
                                ),
                            ),
                        ),
                        step(
                            0,
                            1,
                            terminals=(
                                terminal(
                                    0, 0, 9, 1, "completed"
                                ),
                            ),
                        ),
                    ),
                )
            )
        self.assertEqual(
            {item.advantage[0, 0].item() for item in outputs},
            {outputs[0].advantage[0, 0].item()},
        )

    def test_repeated_defer_events_overlap_but_keep_independent_keys(self) -> None:
        output = estimate(
            reward=torch.tensor([[1.0], [2.0], [3.0]]),
            old_value=torch.zeros((3, 1)),
            bootstrap_value=torch.tensor([[0.0], [0.0], [4.0]]),
            bootstrap_mask=torch.ones(3, dtype=torch.bool),
            episode_boundary=torch.zeros(3, dtype=torch.bool),
            route_active=torch.tensor([[True], [True], [True]]),
            metadata=(
                step(
                    0,
                    20,
                    decisions=(
                        decision(
                            0, 0, 5, 20, category="defer"
                        ),
                    ),
                ),
                step(
                    0,
                    21,
                    decisions=(
                        decision(
                            0, 0, 5, 21, category="defer"
                        ),
                    ),
                ),
                step(
                    0,
                    22,
                    decisions=(
                        decision(0, 0, 5, 22, category="local"),
                    ),
                    terminals=(terminal(0, 0, 5, 22, "completed"),),
                ),
            ),
        )
        first = output.samples[0][0]
        second = output.samples[1][0]
        third = output.samples[2][0]
        assert first is not None and second is not None and third is not None
        self.assertEqual((first.horizon, second.horizon), (3, 2))
        self.assertEqual(third.horizon, 1)
        self.assertNotEqual(first.branch_event_key, second.branch_event_key)
        self.assertIsNone(first.formal_route_event_key)
        self.assertIsNone(second.formal_route_event_key)
        self.assertIsNotNone(third.formal_route_event_key)

    def test_episode_identity_prevents_same_task_id_cross_join(self) -> None:
        output = estimate(
            reward=torch.tensor([[1.0], [2.0], [10.0], [20.0]]),
            old_value=torch.zeros((4, 1)),
            bootstrap_value=torch.tensor(
                [[0.0], [0.0], [0.0], [2.0]]
            ),
            bootstrap_mask=torch.tensor([True, False, True, True]),
            episode_boundary=torch.tensor([False, True, False, False]),
            route_active=torch.tensor(
                [[True], [False], [True], [False]]
            ),
            metadata=(
                step(
                    0, 0, decisions=(decision(0, 0, 1, 0),)
                ),
                step(
                    0,
                    1,
                    terminals=(terminal(0, 0, 1, 1, "truncated"),),
                ),
                step(
                    1, 2, decisions=(decision(1, 0, 1, 2),)
                ),
                step(
                    1,
                    3,
                    terminals=(terminal(1, 0, 1, 3, "completed"),),
                ),
            ),
        )
        first = output.samples[0][0]
        second = output.samples[2][0]
        assert first is not None and second is not None
        self.assertEqual(first.horizon, 2)
        self.assertEqual(second.horizon, 2)
        self.assertAlmostEqual(
            output.advantage[0, 0].item(), 1.0 + 0.99 * 2.0
        )
        self.assertAlmostEqual(
            output.advantage[2, 0].item(),
            10.0 + 0.99 * 20.0 + 0.99**2 * 2.0,
            places=5,
        )

    def test_missing_terminal_at_episode_boundary_fails_closed(self) -> None:
        with self.assertRaises(RouteNstepError):
            estimate(
                reward=torch.ones((2, 1)),
                old_value=torch.zeros((2, 1)),
                bootstrap_value=torch.zeros((2, 1)),
                bootstrap_mask=torch.tensor([True, False]),
                episode_boundary=torch.tensor([False, True]),
                route_active=torch.tensor([[True], [False]]),
                metadata=(
                    step(
                        0, 0, decisions=(decision(0, 0, 1, 0),)
                    ),
                    step(0, 1),
                ),
            )

    def test_active_route_rejects_invalid_task_identity(self) -> None:
        for invalid_task_id in (-1, None, True):
            with self.subTest(task_id=invalid_task_id):
                with self.assertRaisesRegex(RouteNstepError, "task_id"):
                    estimate(
                        reward=torch.ones((1, 1)),
                        old_value=torch.zeros((1, 1)),
                        bootstrap_value=torch.zeros((1, 1)),
                        bootstrap_mask=torch.tensor([True]),
                        episode_boundary=torch.tensor([False]),
                        route_active=torch.tensor([[True]]),
                        metadata=(
                            step(
                                0,
                                0,
                                decisions=(
                                    decision(0, 0, invalid_task_id, 0),
                                ),
                            ),
                        ),
                    )


class RolloutCappedRouteNstepIntegrationTests(unittest.TestCase):
    def test_disabled_execution_metadata_is_exactly_legacy(self) -> None:
        executed, summary = CAGATMAPPOTrainer._execution_metadata(
            {"executed": [{"uav_id": 0}]},
            episode_id=0,
            global_rollout_index=0,
            policy_version=0,
            route_credit_enabled=False,
        )
        self.assertEqual(executed, [{"uav_id": 0}])
        self.assertEqual(
            summary,
            {
                "rejection": (),
                "downgrade": (),
                "canonicalization": (),
            },
        )

    def test_real_step_metadata_distinguishes_formal_and_defer_keys(self) -> None:
        metadata = build_route_credit_step_metadata(
            {
                "slot": 12,
                "service": {
                    "routing": [
                        {
                            "uav_id": 0,
                            "task_id": 7,
                            "proposal": "local",
                            "applied": True,
                        },
                        {
                            "uav_id": 1,
                            "task_id": 8,
                            "proposal": "defer",
                            "applied": False,
                        },
                    ],
                    "settled_tasks": [
                        {
                            "source_uav": 0,
                            "task_id": 7,
                            "status": "done",
                            "outcome": "done",
                        }
                    ],
                    "truncated_tasks": [],
                }
            },
            episode_id=3,
            global_rollout_index=99,
            policy_version=4,
        )
        local, defer = metadata["decisions"]
        self.assertEqual(local["formal_route_event_key"], [3, 7, 12])
        self.assertIsNone(defer["formal_route_event_key"])
        self.assertNotEqual(
            local["branch_event_key"], defer["branch_event_key"]
        )
        self.assertEqual(metadata["terminals"][0]["kind"], "completed")

    def test_ppo_changes_only_route_branch_advantage(self) -> None:
        inputs = ppo_inputs()
        route_advantage = inputs["advantage"] + 7.0
        output = compute_configured_ppo_objective_and_loss(
            **inputs,
            config=nstep_config(),
            route_advantage=route_advantage,
        )
        assert output.branch_advantage is not None
        self.assertTrue(
            torch.equal(
                output.branch_advantage[..., 0], route_advantage
            )
        )
        for branch in range(1, output.branch_advantage.shape[-1]):
            self.assertTrue(
                torch.equal(
                    output.branch_advantage[..., branch],
                    inputs["advantage"],
                )
            )

    def test_treatment_keeps_critic_entropy_ratios_and_other_surrogates_exact(
        self,
    ) -> None:
        from tests.test_ca_gat_mappo_route_specific_gae import low_level_ppo

        inputs = ppo_inputs()
        control = low_level_ppo(inputs, route_advantage=None)
        treatment = low_level_ppo(
            inputs, route_advantage=inputs["advantage"] + 7.0
        )
        self.assertTrue(torch.equal(control.ratio, treatment.ratio))
        self.assertTrue(
            torch.equal(control.branch_ratio, treatment.branch_ratio)
        )
        self.assertTrue(
            torch.equal(control.critic_loss, treatment.critic_loss)
        )
        self.assertTrue(
            torch.equal(control.entropy_mean, treatment.entropy_mean)
        )
        assert control.branch_surrogate is not None
        assert treatment.branch_surrogate is not None
        self.assertTrue(
            torch.equal(
                control.branch_surrogate[..., 1:],
                treatment.branch_surrogate[..., 1:],
            )
        )
        self.assertFalse(
            torch.equal(
                control.branch_surrogate[..., 0],
                treatment.branch_surrogate[..., 0],
            )
        )

    def test_checkpoint_roundtrip_preserves_nested_route_metadata(self) -> None:
        config = make_update_config()
        seed = make_seed_transition(config)
        route_step = {
            "episode_id": 0,
            "global_rollout_index": 1,
            "policy_version": 2,
            "decisions": [decision(0, 0, 7, 1)],
            "terminals": [],
        }
        transition = replace(
            seed,
            rejection_or_downgrade_summary={
                **seed.rejection_or_downgrade_summary,
                "route_credit_step": route_step,
            },
        )
        serialized = serialize_rollout_transition(transition)
        restored = restore_rollout_transition(config, serialized)
        self.assertEqual(
            restored.rejection_or_downgrade_summary[
                "route_credit_step"
            ],
            route_step,
        )

    def test_resume_rejects_cross_mode_and_accepts_same_mode(self) -> None:
        config = nstep_config()
        kwargs = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_kind": CHECKPOINT_KIND_PERIODIC_RESUME,
            "method_id": config.method_id,
            "git_commit": config.git_commit,
            "config_hash": config.config_hash,
            "training_device": "cpu",
            "cuda_available": False,
            "checkpoint_actor_ratio_mode": "branch_specific",
            "checkpoint_agent_credit_mode": "role_decomposed",
        }
        validate_mappo_checkpoint_resume_compatibility(
            config,
            checkpoint_route_credit_mode=(
                "rollout_capped_route_event_nstep"
            ),
            **kwargs,
        )
        with self.assertRaisesRegex(ConfigError, "route_credit_mode mismatch"):
            validate_mappo_checkpoint_resume_compatibility(
                config,
                checkpoint_route_credit_mode="route_specific_gae",
                **kwargs,
            )

    def test_minibatch_uses_saved_rollout_only_and_overwrites_active_route(
        self,
    ) -> None:
        base = make_update_config()
        config = replace(
            base,
            training=replace(
                base.training,
                mappo=replace(
                    base.training.mappo,
                    actor_ratio_mode=ActorRatioMode.BRANCH_SPECIFIC,
                    agent_credit_mode=AgentCreditMode.ROLE_DECOMPOSED,
                    route_credit_mode=(
                        RouteCreditMode.ROLLOUT_CAPPED_ROUTE_EVENT_NSTEP
                    ),
                    training_device="cpu",
                ),
            ),
        )
        config.validate()
        seed = make_seed_transition(base)
        agents = config.environment.uav_count
        buffer = CAGATMAPPORolloutBuffer(config, capacity=256)
        for index in range(256):
            active = seed.active_branch_indicators[:, 0]
            decisions = []
            for agent in range(agents):
                if not bool(active[agent]):
                    continue
                route = seed.proposal_actions[agent].route
                category = (
                    "local"
                    if route == "local"
                    else "defer"
                    if route == "defer"
                    else "remote"
                )
                decisions.append(
                    decision(
                        0,
                        agent,
                        index * agents + agent,
                        index,
                        category=category,
                        policy_version=3,
                    )
                )
            metadata = {
                **seed.rejection_or_downgrade_summary,
                "route_credit_step": step(
                    0,
                    index,
                    decisions=tuple(decisions),
                )["route_credit_step"],
            }
            buffer.append(
                replace(
                    seed,
                    slot=index,
                    episode_start=index == 0,
                    hidden_in=(
                        torch.zeros_like(seed.hidden_in)
                        if index == 0
                        else seed.hidden_in.clone()
                    ),
                    old_value=torch.linspace(
                        0.1, 0.1 * agents, agents
                    ),
                    reward=torch.linspace(
                        1.0, float(agents), agents
                    ),
                    bootstrap_value=torch.linspace(
                        0.5, 0.5 * agents, agents
                    ),
                    rejection_or_downgrade_summary=metadata,
                )
            )
        buffer.finalize()
        minibatch = build_recurrent_ppo_minibatch(buffer, config)
        self.assertIsNotNone(minibatch.route_advantage)
        self.assertIsNotNone(minibatch.route_nstep_samples)
        assert minibatch.route_advantage is not None
        active = minibatch.active_branch_indicators[..., 0]
        self.assertTrue(
            torch.equal(
                minibatch.route_advantage.masked_select(~active),
                minibatch.advantage.masked_select(~active),
            )
        )
        first = next(
            sample
            for row in minibatch.route_nstep_samples[0]
            for sample in row
            if sample is not None
        )
        self.assertEqual(first.global_rollout_index, 0)
        self.assertEqual(first.policy_version, 3)
        self.assertEqual(first.stop_kind, "rollout_boundary")

    def test_collector_reports_exact_nstep_sample_and_group_p95(self) -> None:
        config, environment, _, observations = active_route_fixture()
        kinds = ("local", "remote", "defer", "remote")
        actor, _, masks, proposals, policy = evaluate_route_steps(
            config, environment, (observations,), (kinds,)
        )
        agents = len(observations)
        base = torch.arange(
            1, agents + 1, dtype=torch.float32
        ).reshape(1, 1, agents)
        route = base + 10.0
        events = []
        for agent, category in enumerate(kinds):
            key = (2, agent, 100 + agent, 500)
            events.append(
                RouteNstepSample(
                    episode_id=2,
                    source_uav=agent,
                    task_id=100 + agent,
                    category=category,
                    branch_event_key=key,
                    formal_route_event_key=(
                        None if category == "defer" else key
                    ),
                    global_rollout_index=500,
                    policy_version=9,
                    base_advantage=float(base[0, 0, agent]),
                    nstep_advantage=float(route[0, 0, agent]),
                    advantage_delta=10.0,
                    horizon=3,
                    stop_kind="rollout_boundary",
                    terminal_before_rollout=False,
                    bootstrap_used=True,
                )
            )
        telemetry = collect_route_telemetry(
            policy=policy,
            action_mask_batch=masks,
            proposals=proposals,
            advantage=route,
            return_target=torch.zeros_like(route),
            sequence_valid_mask=torch.ones((1, 1), dtype=torch.bool),
            route_head=actor.action_heads["route"],
            base_advantage=base,
            route_advantage=route,
            route_nstep_samples=((tuple(events),),),
        )
        self.assertEqual(
            telemetry.nstep_route_advantage_valid_sample_count, agents
        )
        for sample in telemetry.samples:
            self.assertEqual(
                sample.advantage, sample.nstep_route_advantage
            )
            self.assertEqual(sample.nstep_horizon, 3)
            self.assertEqual(sample.policy_version, 9)
            record = sample.record()
            self.assertEqual(
                record["route_sample_branch_credit_event_key"],
                list(sample.branch_event_key),
            )
            self.assertEqual(
                record["route_sample_stop_kind"],
                sample.nstep_stop_kind,
            )
            if sample.category == "defer":
                self.assertIsNone(sample.formal_route_event_key)
        record = telemetry.record()
        for category in ("local", "remote", "defer"):
            self.assertIn(
                f"{category}_route_nstep_route_advantage_"
                "p95_absolute_magnitude",
                record,
            )
            self.assertGreater(
                record[
                    f"{category}_route_nstep_route_advantage_"
                    "valid_sample_count"
                ],
                0,
            )


if __name__ == "__main__":
    unittest.main()
