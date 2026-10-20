"""Differentiate capacity from energy legality without changing the frozen policy."""
from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import load_run_config
from src.env.actions import ActionProposal
from src.env.environment import U2UMECEnvironment
from src.evaluation.qmix_cpu_deadline_diagnostic import (
    CPUDeadlineProbe, capacity_atoms, frozen_episode, replace_proposal_groups,
)
from src.evaluation.qmix_limited_diagnostic import assert_saved_episode
from src.evaluation.qmix_runner import QMIXEvaluationRunner
from src.models.ca_gat_mappo import ActorObservationTensorizer
from src.models.factorized_gat_qmix import FactorizedGATQNetwork, METHOD_ID
from src.models.qmix_continuation import state_digest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_run_config(ROOT / "tests/fixtures/configs/qmix_legacy/fa-gat-qmix-seed42-8k-episode-return-development.json",
                           cli_overrides={"environment.episode_horizon": 128,
                                          "training.qmix.max_training_environment_steps": 2048,
                                          "training.qmix.checkpoint_interval_steps": 1024})


@pytest.fixture
def observation(config):
    original = U2UMECEnvironment(config).reset().observations[0]
    mask = original.action_masks
    return replace(original, action_masks=replace(mask, cpu_queue_branch_active=True,
        cpu_queue_mask=np.array([True, True, True, False, False]),
        cpu_frequency_energy_mask=np.ones_like(mask.cpu_frequency_energy_mask),
        cpu_deadline_context=((40.e6, 2, 1.e9), (400.e6, 2, 1.e9), (0., 0, 1.e9), (0., 0, 1.e9))))


def proposal(observation, queue="idle", frequency=0.):
    return ActionProposal(uav_id=observation.uav_id, route="defer", tx_select="idle",
                          resource_group="idle", resource_width=1, power_level=0.,
                          cpu_queue=queue, cpu_frequency=frequency)


def observe(observation, queue="idle", frequency=0., *, queue_q=None, frequency_q=None):
    probe = CPUDeadlineProbe(4, .02)
    probe.decision(observation, proposal(observation, queue, frequency),
                   [2., 0., 1., 0., 0.] if queue_q is None else queue_q,
                   [[1., 0., 0., 0.]]*4 if frequency_q is None else frequency_q)
    return probe


def test_energy_available_is_not_deadline_capacity(observation):
    atoms = capacity_atoms(observation.action_masks, .02)
    assert len(atoms) == 6
    assert [(atom["queue"], atom["frequency"]) for atom in atoms if atom["capacity_feasible"]] == [(0, 1.)]
    mask = replace(observation.action_masks, cpu_deadline_context=((400.e6, 2, 1.e9),)*4)
    counts = observe(replace(observation, action_masks=mask)).counts
    assert counts["energy_positive_available"] == 1
    assert counts["capacity_feasible_head_available"] == 0
    assert counts["feasible_available_but_idle_or_zero"] == 0


@pytest.mark.parametrize("cycles,slack,feasible", [(20.e6, 1, True), (math.nextafter(20.e6, math.inf), 1, False),
                                                    (1., 10, False), (1., 11, True)])
def test_exact_float_and_inclusive_deadline(observation, cycles, slack, feasible):
    fmax = 5. if cycles == 1. else 1.e9
    mask = replace(observation.action_masks, cpu_deadline_context=((cycles, slack, fmax),)*4)
    maximum = [atom for atom in capacity_atoms(mask, .02) if atom["queue"] == 0 and atom["frequency"] == 1.][0]
    assert maximum["capacity_feasible"] == feasible


def test_masked_maximum_is_not_assumed_available(observation):
    energy = observation.action_masks.cpu_frequency_energy_mask.copy()
    energy[:, -1] = False
    mask = replace(observation.action_masks, cpu_frequency_energy_mask=energy)
    assert not any(atom["capacity_feasible"] for atom in capacity_atoms(mask, .02))


def test_idle_zero_slow_and_wrong_head_are_distinct(observation):
    idle = observe(observation)
    assert idle.counts["last_chance_available_but_idle_or_zero"] == 1
    assert idle.counts["feasible_strictly_outranked_by_selected"] == 1
    assert idle.margins["best_capacity_feasible_minus_chosen"] == [-2.]
    zero = observe(observation, 0, 0.)
    assert zero.counts["selected_queue_zero_frequency"] == 1
    slow = observe(observation, 0, .5)
    assert slow.counts["positive_chosen_head_feasible_but_frequency_insufficient"] == 1
    wrong = observe(observation, 1, 1.)
    assert wrong.counts["positive_impossible_head_when_other_head_feasible"] == 1
    correct = observe(observation, 0, 1.)
    assert correct.counts["positive_chosen_pair_capacity_feasible"] == 1


def test_missing_context_is_rejected(observation):
    with pytest.raises(ValueError, match="exact slot-start"):
        capacity_atoms(replace(observation.action_masks, cpu_deadline_context=None), .02)


def test_q_probe_uses_each_alternative_queues_own_frequency_values(observation):
    frequencies = [[0., 0., 0., 10.], [0., 0., 0., 1.], [0.]*4, [0.]*4]
    probe = observe(observation, 1, 1., queue_q=[0.]*5, frequency_q=frequencies)
    assert probe.margins["best_capacity_feasible_minus_chosen"] == [9.]


def test_completion_requires_actual_service():
    probe = CPUDeadlineProbe(4, .02)
    with pytest.raises(ValueError, match="positive actual CPU"):
        probe.settlement({"service": {"cpu": [], "settled_tasks": [{"task_id": 7, "outcome": "done"}]}})


@pytest.mark.parametrize("groups", [(), ("cpu",), ("route",), ("route", "cpu")])
def test_group_replacement_preserves_every_other_branch(observation, groups):
    original = proposal(observation)
    donor = replace(original, route="local", cpu_queue=0, cpu_frequency=1., tx_select=1, power_level=1.)
    actual = replace_proposal_groups(original, donor, groups)
    assert actual.route == (donor.route if "route" in groups else original.route)
    for field in ("cpu_queue", "cpu_frequency"):
        assert getattr(actual, field) == getattr(donor if "cpu" in groups else original, field)
    for field in ("uav_id", "tx_select", "resource_group", "resource_width", "power_level"):
        assert getattr(actual, field) == getattr(original, field)


def test_unsupported_group_is_rejected(observation):
    with pytest.raises(ValueError, match="only route and CPU"):
        replace_proposal_groups(proposal(observation), proposal(observation), ("reward",))


def test_frozen_audit_reproduces_ordinary_runner_without_mutation(config):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        runner = QMIXEvaluationRunner.__new__(QMIXEvaluationRunner)
        runner.config = replace(config, mode="evaluation", training=replace(config.training, formal_rl_enabled=False))
        runner.device = torch.device("cpu")
        runner.agent = FactorizedGATQNetwork(config).eval().requires_grad_(False)
        runner.adapter = ActorObservationTensorizer(config)
        before = state_digest(runner.agent.state_dict())
        ordinary = runner.episode(METHOD_ID, 1042)
        audited, record = frozen_episode(runner, 1042)
        assert_saved_episode(audited, ordinary.artifact_record("engineering-test"))
        assert before == state_digest(runner.agent.state_dict())
        assert record["actual_completion_cycle_energy_accounting_verified"]
        assert record["counts"]["backlogged_decisions"] > 0
    finally:
        torch.set_num_threads(previous)
