"""Final-seal tests for route-specific lambda-one GAE V1."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch

from src.config import (
    ActorRatioMode,
    AgentCreditMode,
    CHECKPOINT_KIND_PERIODIC_RESUME,
    CHECKPOINT_SCHEMA_VERSION,
    ConfigError,
    RouteCreditMode,
    RunConfig,
    load_run_config,
    validate_mappo_checkpoint_resume_compatibility,
)
from src.models.ca_gat_mappo import (
    ACTION_BRANCH_ORDER,
    CAGATMAPPOActor,
    MAPPOCentralizedCritic,
)
from src.models.ca_gat_mappo_gae import (
    compute_per_agent_gae_and_returns,
    compute_route_specific_advantage,
)
from src.models.ca_gat_mappo_ppo import (
    compute_configured_ppo_objective_and_loss,
    compute_ppo_objective_and_loss,
)
from src.models.ca_gat_mappo_route_telemetry import (
    ROUTE_TELEMETRY_SCHEMA_VERSION,
    RouteSampleTelemetry,
    collect_route_telemetry,
)
from src.training_artifacts import _base_record, _route_sample_records
from tests.test_ca_gat_mappo_route_telemetry import (
    active_route_fixture,
    evaluate_route_steps,
)


FROZEN_DEFAULT_CONFIG_HASH = (
    "5f37790be06c534decdf1f618daa60bb10f49a425214dcd8ed9071708d98bbeb"
)


def route_credit_config(*, treatment: bool) -> RunConfig:
    base = replace(RunConfig(), mode="rl", method_id="ca_gat_mappo")
    mappo = replace(
        base.training.mappo,
        actor_ratio_mode=ActorRatioMode.BRANCH_SPECIFIC,
        agent_credit_mode=AgentCreditMode.ROLE_DECOMPOSED,
        training_device="cpu",
        route_credit_mode=(
            RouteCreditMode.ROUTE_SPECIFIC_GAE
            if treatment
            else RouteCreditMode.SHARED_GAE
        ),
        route_gae_lambda=1.0,
    )
    config = replace(base, training=replace(base.training, mappo=mappo))
    config.validate()
    return config


def ppo_inputs() -> dict[str, torch.Tensor]:
    time_steps, agents, branches = 3, 2, len(ACTION_BRANCH_ORDER)
    old_branch = torch.zeros((time_steps, agents, branches), dtype=torch.float32)
    new_branch = torch.tensor(
        [
            [[0.08, -0.04, 0.02, 0.01, -0.03, 0.05, -0.02]] * agents,
            [[-0.07, 0.03, -0.01, 0.06, 0.02, -0.04, 0.01]] * agents,
            [[0.04, 0.02, -0.05, 0.03, -0.01, 0.01, -0.03]] * agents,
        ],
        dtype=torch.float32,
    )
    active = torch.ones_like(old_branch, dtype=torch.bool)
    return {
        "new_joint_log_prob": torch.zeros((time_steps, agents)),
        "old_joint_log_prob": torch.zeros((time_steps, agents)),
        "advantage": torch.tensor(
            [[1.0, -0.5], [0.4, 0.8], [-0.3, 0.6]], dtype=torch.float32
        ),
        "current_value": torch.tensor(
            [[0.2, -0.1], [0.3, 0.5], [-0.2, 0.1]], dtype=torch.float32
        ),
        "return_target": torch.tensor(
            [[0.7, 0.2], [0.4, 0.9], [-0.1, 0.8]], dtype=torch.float32
        ),
        "entropy": torch.full((time_steps, agents), 0.7),
        "sequence_valid_mask": torch.ones(time_steps, dtype=torch.bool),
        "new_branch_log_probs": new_branch,
        "old_branch_log_probs": old_branch,
        "active_branch_indicators": active,
    }


def low_level_ppo(
    inputs: dict[str, torch.Tensor],
    *,
    route_advantage: torch.Tensor | None,
):
    return compute_ppo_objective_and_loss(
        **inputs,
        epsilon_clip=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        actor_ratio_mode=ActorRatioMode.BRANCH_SPECIFIC.value,
        agent_credit_mode=AgentCreditMode.ROLE_DECOMPOSED.value,
        route_advantage=route_advantage,
    )


class RouteSpecificGAEConfigTests(unittest.TestCase):
    def test_default_hash_and_legacy_payload_are_exactly_preserved(self) -> None:
        detected = RunConfig()
        # The approved hash was sealed in the bundled 3.12 runtime.  Pin only
        # its already-canonical runtime provenance while executing numerical
        # tests in the installed PyTorch runtime.
        config = replace(
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
        resolved_mappo = config.resolved_dict()["training"]["mappo"]
        self.assertEqual(config.config_hash, FROZEN_DEFAULT_CONFIG_HASH)
        self.assertNotIn("route_credit_mode", resolved_mappo)
        self.assertNotIn("route_gae_lambda", resolved_mappo)
        self.assertFalse(hasattr(config.training.mappo, "gamma_route"))

    def test_treatment_contract_fails_fast_for_every_illegal_combination(self) -> None:
        valid = route_credit_config(treatment=True)
        cases = {
            "gamma": replace(valid.training.mappo, gamma=0.98),
            "base_lambda": replace(valid.training.mappo, gae_lambda=0.94),
            "route_lambda": replace(valid.training.mappo, route_gae_lambda=0.99),
            "ratio": replace(
                valid.training.mappo,
                actor_ratio_mode=ActorRatioMode.JOINT,
            ),
            "credit": replace(
                valid.training.mappo,
                agent_credit_mode=AgentCreditMode.TEAM,
            ),
        }
        for name, mappo in cases.items():
            with self.subTest(name=name), self.assertRaises(ConfigError):
                replace(
                    valid,
                    training=replace(valid.training, mappo=mappo),
                ).validate()

    def test_cli_loader_accepts_the_frozen_treatment_interface(self) -> None:
        config = load_run_config(
            cli_overrides={
                "training.mappo.actor_ratio_mode": "branch_specific",
                "training.mappo.agent_credit_mode": "role_decomposed",
                "training.mappo.route_credit_mode": "route_specific_gae",
                "training.mappo.route_gae_lambda": 1.0,
            }
        )
        self.assertEqual(
            config.training.mappo.route_credit_mode,
            RouteCreditMode.ROUTE_SPECIFIC_GAE,
        )
        self.assertEqual(config.training.mappo.route_gae_lambda, 1.0)


class RouteSpecificGAEHandCalculationTests(unittest.TestCase):
    def test_base_and_route_use_one_exact_td_source_with_different_recursions(self) -> None:
        residual = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            dtype=torch.float32,
        )
        bootstrap = torch.tensor([True, False, True, True])
        valid = torch.ones(4, dtype=torch.bool)
        frozen_source = residual.clone()
        base = compute_route_specific_advantage(
            base_td_residual=residual,
            bootstrap_mask=bootstrap,
            sequence_mask=valid,
            gamma=0.99,
            route_gae_lambda=0.95,
        )
        route = compute_route_specific_advantage(
            base_td_residual=residual,
            bootstrap_mask=bootstrap,
            sequence_mask=valid,
            gamma=0.99,
            route_gae_lambda=1.0,
        )
        expected_base = torch.stack(
            (
                residual[0] + 0.99 * 0.95 * residual[1],
                residual[1],
                residual[2] + 0.99 * 0.95 * residual[3],
                residual[3],
            )
        )
        expected_route = torch.stack(
            (
                residual[0] + 0.99 * residual[1],
                residual[1],
                residual[2] + 0.99 * residual[3],
                residual[3],
            )
        )
        self.assertTrue(torch.equal(residual, frozen_source))
        self.assertTrue(torch.allclose(base, expected_base, rtol=0.0, atol=1e-7))
        self.assertTrue(torch.allclose(route, expected_route, rtol=0.0, atol=1e-7))
        self.assertEqual(tuple(route.shape), (4, 2))
        self.assertFalse(torch.equal(base, route))

    def test_rollout_tail_uses_saved_per_agent_next_value_and_boundary_stops(self) -> None:
        reward = torch.tensor([[0.1, 0.2], [0.3, 0.4], [1.0, 2.0]])
        old_value = torch.tensor([[0.5, 0.7], [0.6, 0.9], [0.8, 1.1]])
        next_value = torch.tensor([[0.6, 0.9], [0.8, 1.1], [2.0, 3.0]])
        terminated = torch.tensor([False, True, False])
        truncated = torch.zeros(3, dtype=torch.bool)
        boundary = terminated | truncated
        output = compute_per_agent_gae_and_returns(
            reward=reward,
            old_value=old_value,
            bootstrap_value=next_value,
            terminated=terminated,
            truncated=truncated,
            episode_boundary=boundary,
            bootstrap_allowed=~boundary,
            gamma=0.99,
            gae_lambda=0.95,
        )
        self.assertTrue(
            torch.equal(
                output.td_residual[1],
                reward[1] - old_value[1],
            )
        )
        expected_tail = reward[2] + 0.99 * next_value[2] - old_value[2]
        self.assertTrue(torch.equal(output.td_residual[2], expected_tail))
        route = compute_route_specific_advantage(
            base_td_residual=output.td_residual,
            bootstrap_mask=output.bootstrap_mask,
            sequence_mask=output.sequence_mask,
            gamma=0.99,
            route_gae_lambda=1.0,
        )
        self.assertTrue(torch.equal(route[1], output.td_residual[1]))
        self.assertTrue(torch.equal(route[2], expected_tail))
        self.assertTrue(torch.equal(output.return_target, output.advantage + old_value))

    def test_lambda_one_remains_gamma_discounted_not_undiscounted_return(self) -> None:
        residual = torch.tensor([[0.0], [0.0], [1.0]])
        route = compute_route_specific_advantage(
            base_td_residual=residual,
            bootstrap_mask=torch.tensor([True, True, False]),
            sequence_mask=torch.ones(3, dtype=torch.bool),
            gamma=0.99,
            route_gae_lambda=1.0,
        )
        expected = torch.tensor([[0.99**2], [0.99], [1.0]])
        self.assertTrue(torch.allclose(route, expected, rtol=0.0, atol=1e-7))
        self.assertFalse(torch.equal(route, torch.ones_like(route)))


class RouteSpecificGAEPPOIsolationTests(unittest.TestCase):
    def test_configured_treatment_requires_and_consumes_route_advantage(self) -> None:
        config = route_credit_config(treatment=True)
        inputs = ppo_inputs()
        route_advantage = torch.full((3, 2), 4.0)
        with self.assertRaisesRegex(ValueError, "requires route_advantage"):
            compute_configured_ppo_objective_and_loss(**inputs, config=config)
        output = compute_configured_ppo_objective_and_loss(
            **inputs,
            config=config,
            route_advantage=route_advantage,
        )
        self.assertTrue(torch.equal(output.branch_advantage[..., 0], route_advantage))

    def test_only_route_branch_advantage_and_surrogate_change(self) -> None:
        inputs = ppo_inputs()
        route_advantage = torch.tensor(
            [[2.0, 1.5], [1.4, 1.8], [0.7, 1.6]], dtype=torch.float32
        )
        shared = low_level_ppo(inputs, route_advantage=None)
        treatment = low_level_ppo(inputs, route_advantage=route_advantage)
        self.assertTrue(torch.equal(shared.branch_ratio, treatment.branch_ratio))
        self.assertTrue(
            torch.equal(
                treatment.branch_advantage[..., 0],
                route_advantage,
            )
        )
        self.assertTrue(
            torch.equal(
                shared.branch_advantage[..., 1:],
                treatment.branch_advantage[..., 1:],
            )
        )
        self.assertTrue(
            torch.equal(
                shared.branch_unclipped_surrogate[..., 1:],
                treatment.branch_unclipped_surrogate[..., 1:],
            )
        )
        self.assertTrue(
            torch.equal(
                shared.branch_clipped_surrogate[..., 1:],
                treatment.branch_clipped_surrogate[..., 1:],
            )
        )
        self.assertTrue(
            torch.equal(
                shared.branch_surrogate[..., 1:],
                treatment.branch_surrogate[..., 1:],
            )
        )
        self.assertFalse(
            torch.equal(
                shared.branch_surrogate[..., 0],
                treatment.branch_surrogate[..., 0],
            )
        )
        self.assertTrue(torch.equal(shared.critic_loss, treatment.critic_loss))
        self.assertTrue(torch.equal(shared.entropy_mean, treatment.entropy_mean))

    def test_local_remote_and_defer_positions_all_use_route_advantage(self) -> None:
        inputs = ppo_inputs()
        route_advantage = torch.tensor(
            [[11.0, 12.0], [21.0, 22.0], [31.0, 32.0]], dtype=torch.float32
        )
        output = low_level_ppo(inputs, route_advantage=route_advantage)
        # Rows model Local, Remote, and Defer route selections respectively;
        # the branch objective is intentionally category-agnostic.
        self.assertTrue(torch.equal(output.branch_advantage[..., 0], route_advantage))

    def test_inactive_route_has_zero_route_actor_surrogate(self) -> None:
        inputs = ppo_inputs()
        inputs["active_branch_indicators"] = inputs[
            "active_branch_indicators"
        ].clone()
        inputs["active_branch_indicators"][0, 0, 0] = False
        output = low_level_ppo(
            inputs,
            route_advantage=torch.full((3, 2), 10.0),
        )
        self.assertEqual(float(output.branch_surrogate[0, 0, 0]), 0.0)
        self.assertEqual(float(output.branch_unclipped_surrogate[0, 0, 0]), 0.0)
        self.assertEqual(float(output.branch_clipped_surrogate[0, 0, 0]), 0.0)

    def test_shared_mode_matches_legacy_objective_rng_and_optimizer_step(self) -> None:
        config = route_credit_config(treatment=False)
        base_inputs = ppo_inputs()
        parameter_legacy = torch.nn.Parameter(
            base_inputs["new_branch_log_probs"].clone()
        )
        parameter_shared = torch.nn.Parameter(
            base_inputs["new_branch_log_probs"].clone()
        )
        optimizer_legacy = torch.optim.Adam([parameter_legacy], lr=3e-4)
        optimizer_shared = torch.optim.Adam([parameter_shared], lr=3e-4)
        legacy_inputs = dict(base_inputs)
        legacy_inputs["new_branch_log_probs"] = parameter_legacy
        shared_inputs = dict(base_inputs)
        shared_inputs["new_branch_log_probs"] = parameter_shared
        legacy = low_level_ppo(legacy_inputs, route_advantage=None)
        rng_before = torch.random.get_rng_state().clone()
        shared = compute_configured_ppo_objective_and_loss(
            **shared_inputs,
            config=config,
        )
        rng_after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(rng_before, rng_after))
        for name in (
            "log_ratio",
            "ratio",
            "expanded_advantage",
            "surrogate",
            "branch_ratio",
            "branch_advantage",
            "branch_surrogate",
            "critic_loss",
            "entropy_mean",
            "total_loss",
        ):
            self.assertTrue(
                torch.equal(getattr(legacy, name), getattr(shared, name)),
                name,
            )
        legacy.total_loss.backward()
        shared.total_loss.backward()
        optimizer_legacy.step()
        optimizer_shared.step()
        self.assertTrue(torch.equal(parameter_legacy, parameter_shared))
        legacy_state = optimizer_legacy.state[parameter_legacy]
        shared_state = optimizer_shared.state[parameter_shared]
        self.assertEqual(tuple(legacy_state), tuple(shared_state))
        for name in legacy_state:
            self.assertTrue(
                torch.equal(legacy_state[name], shared_state[name]),
                name,
            )

    def test_critic_parameter_schema_is_unchanged_by_route_credit_mode(self) -> None:
        shared = route_credit_config(treatment=False)
        treatment = route_credit_config(treatment=True)
        shared_critic = MAPPOCentralizedCritic(shared)
        treatment_critic = MAPPOCentralizedCritic(treatment)
        self.assertEqual(
            {key: tuple(value.shape) for key, value in shared_critic.state_dict().items()},
            {key: tuple(value.shape) for key, value in treatment_critic.state_dict().items()},
        )
        shared_actor = CAGATMAPPOActor(shared)
        treatment_actor = CAGATMAPPOActor(treatment)
        self.assertEqual(
            {key: tuple(value.shape) for key, value in shared_actor.state_dict().items()},
            {key: tuple(value.shape) for key, value in treatment_actor.state_dict().items()},
        )


class RouteSpecificGAETelemetryTests(unittest.TestCase):
    def test_active_samples_report_effective_credit_and_trace_provenance(self) -> None:
        config, environment, _, observations = active_route_fixture()
        agents = config.environment.uav_count
        kinds = (
            tuple("local" for _ in range(agents)),
            tuple("defer" if index == 0 else "remote" for index in range(agents)),
        )
        actor, _, masks, proposals, policy = evaluate_route_steps(
            config,
            environment,
            (observations, observations),
            kinds,
        )
        base = torch.arange(1, 1 + 2 * agents, dtype=torch.float32).reshape(
            1, 2, agents
        )
        new = base + 2.5
        branch_advantage = base.unsqueeze(-1).expand(
            1, 2, agents, len(ACTION_BRANCH_ORDER)
        ).clone()
        branch_advantage[..., 0] = new
        telemetry = collect_route_telemetry(
            policy=policy,
            action_mask_batch=masks,
            proposals=proposals,
            advantage=new,
            base_advantage=base,
            route_advantage=new,
            branch_advantage=branch_advantage,
            return_target=base + 10.0,
            td_residual=base - 1.0,
            sequence_valid_mask=torch.ones((1, 2), dtype=torch.bool),
            bootstrap_mask=torch.tensor([[True, False]]),
            effective_route_gae_lambda=1.0,
            route_head=actor.action_heads["route"],
            old_branch_log_probs=torch.stack(
                [policy.branch_log_probs[name].detach() for name in ACTION_BRANCH_ORDER],
                dim=-1,
            ),
            actor_ratio_mode=ActorRatioMode.BRANCH_SPECIFIC.value,
        )
        self.assertEqual(telemetry.base_route_advantage_valid_sample_count, 2 * agents)
        self.assertEqual(telemetry.new_route_advantage_valid_sample_count, 2 * agents)
        self.assertEqual(telemetry.route_advantage_delta_valid_sample_count, 2 * agents)
        self.assertAlmostEqual(telemetry.route_advantage_delta_mean, 2.5)
        self.assertAlmostEqual(telemetry.route_advantage_delta_std, 0.0)
        self.assertAlmostEqual(
            telemetry.new_route_advantage_p90_absolute_magnitude,
            float(torch.quantile(new.reshape(-1).abs(), 0.90)),
        )
        categories = {sample.category for sample in telemetry.samples}
        self.assertTrue({"local", "remote", "defer"}.issubset(categories))
        for sample in telemetry.samples:
            self.assertEqual(sample.advantage, sample.new_route_advantage)
            self.assertAlmostEqual(
                sample.route_advantage_delta,
                sample.new_route_advantage - sample.base_route_advantage,
            )
            self.assertEqual(sample.effective_route_gae_lambda, 1.0)
            self.assertEqual(sample.trace_end_kind, "episode_boundary")
            self.assertEqual(sample.bootstrap_source, "none_episode_boundary")
            self.assertEqual(
                sample.credit_trace_length_slots,
                2 if sample.time_index == 0 else 1,
            )
        self.assertIsNotNone(telemetry.route_ppo_dynamics)
        self.assertIsNotNone(telemetry.branch_ppo_dynamics)

    def test_route_sample_artifact_uses_global_index_and_rollout_policy_version(self) -> None:
        config = route_credit_config(treatment=True)
        sample = RouteSampleTelemetry(
            batch_index=3,
            time_index=7,
            agent_index=1,
            source_uav=1,
            category="defer",
            legal_remote_destinations=(),
            remote_probability_by_destination={},
            local_probability=0.4,
            defer_probability=0.6,
            total_remote_probability_mass=0.0,
            best_remote_probability=None,
            number_of_legal_remote_destinations=0,
            selected_route_action_index=1,
            selected_destination_uav=None,
            old_route_log_prob=-0.5,
            new_route_log_prob=-0.4,
            route_log_ratio=0.1,
            route_ratio=math.exp(0.1),
            route_approx_kl=math.exp(0.1) - 1.1,
            route_clip_indicator=False,
            advantage=3.0,
            return_target=4.0,
            td_residual=1.0,
            base_route_advantage=2.0,
            new_route_advantage=3.0,
            route_advantage_delta=1.0,
            effective_route_gae_lambda=1.0,
            credit_trace_length_slots=5,
            trace_end_kind="rollout_boundary",
            bootstrap_source="saved_source_agent_v_next",
        )
        telemetry = SimpleNamespace(
            schema_version=ROUTE_TELEMETRY_SCHEMA_VERSION,
            samples=(sample,),
        )
        epoch = SimpleNamespace(epoch_index=0, route_telemetry=telemetry)
        update = SimpleNamespace(
            update_index=12,
            rollout_policy_version=41,
            policy_version_after_update=42,
            output=SimpleNamespace(epoch_diagnostics=(epoch,)),
        )
        training = SimpleNamespace(
            updates=(update,),
            total_environment_transitions=10_000,
        )
        record = _route_sample_records(config, training, "diagnostic_only")[0]
        self.assertEqual(record["global_rollout_index"], 12 * 256 + 3 * 32 + 7)
        self.assertEqual(record["policy_version"], 41)
        self.assertEqual(record["route_sample_category"], "defer")
        self.assertEqual(record["route_sample_advantage"], 3.0)
        self.assertEqual(record["route_credit_mode"], "route_specific_gae")
        self.assertEqual(record["effective_route_gae_lambda"], 1.0)

    def test_shared_artifact_provenance_reports_base_lambda(self) -> None:
        record = _base_record(RunConfig(), "diagnostic_only")
        self.assertEqual(record["route_credit_mode"], "shared_gae")
        self.assertEqual(record["effective_route_gae_lambda"], 0.95)


class RouteSpecificGAECheckpointTests(unittest.TestCase):
    def _validate(self, config: RunConfig, source_route_mode: str | None) -> None:
        mappo = config.training.mappo
        validate_mappo_checkpoint_resume_compatibility(
            config,
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            checkpoint_kind=CHECKPOINT_KIND_PERIODIC_RESUME,
            method_id=config.method_id,
            git_commit=config.git_commit,
            config_hash=config.config_hash,
            training_device=mappo.training_device,
            cuda_available=False,
            checkpoint_actor_ratio_mode=ActorRatioMode(mappo.actor_ratio_mode).value,
            checkpoint_agent_credit_mode=AgentCreditMode(mappo.agent_credit_mode).value,
            checkpoint_route_credit_mode=source_route_mode,
        )

    def test_old_checkpoint_without_mode_is_shared_and_same_mode_resumes(self) -> None:
        self._validate(route_credit_config(treatment=False), None)
        treatment = route_credit_config(treatment=True)
        self._validate(treatment, RouteCreditMode.ROUTE_SPECIFIC_GAE.value)

    def test_cross_route_credit_mode_resume_fails_fast(self) -> None:
        treatment = route_credit_config(treatment=True)
        with self.assertRaisesRegex(ConfigError, "route_credit_mode mismatch"):
            self._validate(treatment, RouteCreditMode.SHARED_GAE.value)
        shared = route_credit_config(treatment=False)
        with self.assertRaisesRegex(ConfigError, "route_credit_mode mismatch"):
            self._validate(shared, RouteCreditMode.ROUTE_SPECIFIC_GAE.value)


class RouteSpecificGAEScaleSafetyTests(unittest.TestCase):
    def test_synthetic_lambda_one_scale_and_ppo_diagnostics_are_finite(self) -> None:
        time_steps, agents = 256, 3
        phase = torch.arange(time_steps, dtype=torch.float32).unsqueeze(-1)
        agent_offset = torch.arange(agents, dtype=torch.float32).unsqueeze(0)
        residual = 0.04 * torch.sin(phase / 11.0 + agent_offset)
        bootstrap = torch.ones(time_steps, dtype=torch.bool)
        bootstrap[63] = False
        bootstrap[127] = False
        bootstrap[191] = False
        base = compute_route_specific_advantage(
            base_td_residual=residual,
            bootstrap_mask=bootstrap,
            sequence_mask=torch.ones(time_steps, dtype=torch.bool),
            gamma=0.99,
            route_gae_lambda=0.95,
        )
        route = compute_route_specific_advantage(
            base_td_residual=residual,
            bootstrap_mask=bootstrap,
            sequence_mask=torch.ones(time_steps, dtype=torch.bool),
            gamma=0.99,
            route_gae_lambda=1.0,
        )
        delta = route - base
        p90_base = torch.quantile(base.abs(), 0.90)
        p90_route = torch.quantile(route.abs(), 0.90)
        ratio = p90_route / p90_base
        self.assertTrue(torch.isfinite(base).all())
        self.assertTrue(torch.isfinite(route).all())
        self.assertTrue(torch.isfinite(delta).all())
        self.assertTrue(torch.isfinite(ratio))
        self.assertLess(float(ratio), 10.0)

        inputs = ppo_inputs()
        route_leaf = torch.nn.Parameter(route[:3, :2].clone())
        inputs["new_branch_log_probs"] = torch.nn.Parameter(
            inputs["new_branch_log_probs"].clone()
        )
        output = low_level_ppo(inputs, route_advantage=route_leaf)
        output.total_loss.backward()
        route_gradient_norm = torch.linalg.vector_norm(
            inputs["new_branch_log_probs"].grad[..., 0]
        )
        route_surrogate_magnitude = output.branch_surrogate[..., 0].abs().mean()
        self.assertTrue(torch.isfinite(route_gradient_norm))
        self.assertTrue(torch.isfinite(route_surrogate_magnitude))
        self.assertTrue(math.isfinite(output.diagnostics.approx_kl))
        self.assertTrue(math.isfinite(output.diagnostics.clipped_fraction))


if __name__ == "__main__":
    unittest.main()
