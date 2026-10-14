from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

import torch

from src.config import (
    RunConfig,
    compute_effective_route_entropy_schedule,
    compute_route_only_entropy_schedule,
    load_run_config,
)
from src.models.ca_gat_mappo_ppo import (
    compute_configured_ppo_objective_and_loss,
    compute_ppo_objective_and_loss,
)


CONFIG_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "tests" / "fixtures" / "configs"
)
BASELINE_CONFIG = CONFIG_DIRECTORY / (
    "cooperative-load-v2-hierarchical-optionaware-v1-"
    "remote-stress-v2-32k.json"
)
TREATMENT_CONFIG = CONFIG_DIRECTORY / (
    "cooperative-load-v2-hierarchical-optionaware-v1-"
    "remote-stress-v2-route-entropy-v1-32k.json"
)
TREATMENT_FIELDS = {
    "route_only_entropy_initial_coefficient",
    "route_only_entropy_final_coefficient",
    "route_only_entropy_anneal_steps",
}


def _config_with_route_only_entropy(
    *, initial: float, final: float, anneal_steps: int
) -> RunConfig:
    base = RunConfig()
    return replace(
        base,
        training=replace(
            base.training,
            mappo=replace(
                base.training.mappo,
                route_only_entropy_initial_coefficient=initial,
                route_only_entropy_final_coefficient=final,
                route_only_entropy_anneal_steps=anneal_steps,
            ),
        ),
    )


def _loss_inputs(
    route_entropy: torch.Tensor,
    other_entropy: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "new_joint_log_prob": torch.zeros((2, 2)),
        "old_joint_log_prob": torch.zeros((2, 2)),
        "advantage": torch.tensor([1.0, -0.5]),
        "current_value": torch.tensor([0.2, -0.1]),
        "return_target": torch.tensor([0.3, 0.4]),
        "entropy": route_entropy + other_entropy,
        "route_entropy": route_entropy,
        "sequence_valid_mask": torch.ones(2, dtype=torch.bool),
    }


class RouteOnlyEntropyConfigTests(unittest.TestCase):
    def test_remote_stress_v2_treatment_parses_and_only_changes_treatment(self) -> None:
        baseline = load_run_config(BASELINE_CONFIG)
        treatment = load_run_config(TREATMENT_CONFIG)
        mappo = treatment.training.mappo

        self.assertEqual(treatment.scenario_id, "remote_stress_v2")
        self.assertEqual(treatment.seed, baseline.seed)
        self.assertEqual(
            mappo.max_training_environment_steps,
            baseline.training.mappo.max_training_environment_steps,
        )
        self.assertEqual(
            mappo.route_decoder_mode,
            baseline.training.mappo.route_decoder_mode,
        )
        self.assertEqual(treatment.environment, baseline.environment)
        self.assertEqual(treatment.reward, baseline.reward)
        self.assertEqual(mappo.route_only_entropy_initial_coefficient, 0.02)
        self.assertEqual(mappo.route_only_entropy_final_coefficient, 0.0)
        self.assertEqual(mappo.route_only_entropy_anneal_steps, 32000)

        baseline_mapping = json.loads(json.dumps(baseline.resolved_dict()))
        treatment_mapping = json.loads(json.dumps(treatment.resolved_dict()))
        treatment_mappo = treatment_mapping["training"]["mappo"]
        for name in TREATMENT_FIELDS:
            treatment_mappo.pop(name)
        self.assertEqual(treatment_mapping, baseline_mapping)

    def test_linear_annealing_uses_collected_environment_steps(self) -> None:
        mappo = load_run_config(TREATMENT_CONFIG).training.mappo
        expected_extra = {0: 0.02, 16000: 0.01, 32000: 0.0, 40000: 0.0}
        expected_effective = {0: 0.03, 16000: 0.02, 32000: 0.01, 40000: 0.01}
        for step in expected_extra:
            with self.subTest(step=step):
                self.assertAlmostEqual(
                    compute_route_only_entropy_schedule(mappo, step).coefficient,
                    expected_extra[step],
                    places=15,
                )
                self.assertAlmostEqual(
                    compute_effective_route_entropy_schedule(
                        mappo, step
                    ).coefficient,
                    expected_effective[step],
                    places=15,
                )


class RouteOnlyEntropyLossTests(unittest.TestCase):
    def test_zero_route_only_coefficient_is_bitwise_legacy_loss(self) -> None:
        route_entropy = torch.full((2, 2), 0.4)
        other_entropy = torch.full((2, 2), 1.2)
        inputs = _loss_inputs(route_entropy, other_entropy)
        config = _config_with_route_only_entropy(
            initial=0.0,
            final=0.0,
            anneal_steps=0,
        )
        configured = compute_configured_ppo_objective_and_loss(
            **inputs,
            config=config,
        )
        legacy = compute_ppo_objective_and_loss(
            new_joint_log_prob=inputs["new_joint_log_prob"],
            old_joint_log_prob=inputs["old_joint_log_prob"],
            advantage=inputs["advantage"],
            current_value=inputs["current_value"],
            return_target=inputs["return_target"],
            entropy=inputs["entropy"],
            sequence_valid_mask=inputs["sequence_valid_mask"],
            epsilon_clip=config.training.mappo.ppo_clip_epsilon,
            value_coefficient=config.training.mappo.value_coefficient,
            entropy_coefficient=config.training.mappo.entropy_coefficient,
        )
        self.assertTrue(torch.equal(configured.actor_loss, legacy.actor_loss))
        self.assertTrue(torch.equal(configured.critic_loss, legacy.critic_loss))
        self.assertTrue(torch.equal(configured.entropy_mean, legacy.entropy_mean))
        self.assertTrue(torch.equal(configured.total_loss, legacy.total_loss))

    def test_extra_entropy_gradient_changes_route_branch_only(self) -> None:
        baseline_route = torch.full((2, 2), 0.4, requires_grad=True)
        baseline_other = torch.full((2, 2), 1.2, requires_grad=True)
        baseline = compute_configured_ppo_objective_and_loss(
            **_loss_inputs(baseline_route, baseline_other),
            config=RunConfig(),
        )
        baseline.total_loss.backward()

        treatment_route = torch.full((2, 2), 0.4, requires_grad=True)
        treatment_other = torch.full((2, 2), 1.2, requires_grad=True)
        treatment = compute_configured_ppo_objective_and_loss(
            **_loss_inputs(treatment_route, treatment_other),
            config=_config_with_route_only_entropy(
                initial=0.02,
                final=0.0,
                anneal_steps=100,
            ),
            collected_environment_steps=0,
        )
        treatment.total_loss.backward()

        self.assertAlmostEqual(treatment.route_entropy_coefficient, 0.03)
        torch.testing.assert_close(
            treatment.total_loss,
            baseline.total_loss - torch.tensor(0.02 * 0.4),
        )
        torch.testing.assert_close(treatment_other.grad, baseline_other.grad)
        torch.testing.assert_close(
            treatment_route.grad,
            baseline_route.grad - torch.full((2, 2), 0.02 / 4.0),
        )


if __name__ == "__main__":
    unittest.main()
