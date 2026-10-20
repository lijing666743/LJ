"""Independent target arithmetic and opposing online/target action rankings."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.config import load_run_config
from src.env.environment import U2UMECEnvironment
from src.models.ca_gat_mappo import ActorObservationTensorizer, stack_actor_time
from src.models.factorized_gat_qmix import td_target
from src.models.qmix_replay import QMIXLearner, ReplaySequence
from src.models.qmix_targets import double_q_next_values, td_lambda_targets
from src.models.qmix_trainer import QMIXTrainer, checkpoint_bytes, load_checkpoint
from src.models.qmix_continuation import load_continuation_checkpoint


def test_lambda_zero_equals_original_one_step_target():
    reward = torch.tensor([[1., 2., 3.]])
    nxt = torch.tensor([[4., 5., float("nan")]])
    bootstrap = torch.tensor([[True, True, False]])
    assert torch.equal(td_lambda_targets(reward, nxt, bootstrap, .9, 0.),
                       td_target(reward, nxt, bootstrap, .9))


def test_lambda_one_matches_discounted_terminal_return():
    actual = td_lambda_targets(torch.tensor([[1., 2., 3.]]), torch.zeros(1, 3),
                              torch.tensor([[True, True, False]]), .5, 1.)
    assert torch.equal(actual, torch.tensor([[2.75, 3.5, 3.]]))


def test_delayed_positive_reward_reaches_earlier_slots_without_target_copy():
    actual = td_lambda_targets(torch.tensor([[0., 0., 1.]]), torch.zeros(1, 3),
                              torch.tensor([[True, True, False]]), 1., .6)
    assert torch.allclose(actual, torch.tensor([[.36, .6, 1.]]))


def test_nonterminal_chunk_bootstraps_real_next_state_at_boundary():
    actual = td_lambda_targets(torch.zeros(1, 2), torch.tensor([[3., 4.]]),
                              torch.ones(1, 2, dtype=torch.bool), 1., .6)
    assert torch.allclose(actual, torch.tensor([[3.6, 4.]]))


def test_terminal_cuts_trace_and_targets_are_detached():
    nxt = torch.tensor([[10., float("nan"), 5.]], requires_grad=True)
    actual = td_lambda_targets(torch.tensor([[0., 2., 4.]], requires_grad=True), nxt,
                              torch.tensor([[True, False, True]]), 1., 1.)
    assert torch.equal(actual, torch.tensor([[2., 2., 9.]]))
    assert not actual.requires_grad


@pytest.mark.parametrize("gamma,trace", [(-.1,.6),(.99,1.1),(float("nan"),.6),(.99,float("inf"))])
def test_invalid_trace_parameters_are_rejected(gamma, trace):
    with pytest.raises(ValueError):
        td_lambda_targets(torch.zeros(1, 2), torch.zeros(1, 2),
                          torch.ones(1, 2, dtype=torch.bool), gamma, trace)


class SumMixer(nn.Module):
    def forward(self, utility, state):
        return utility.sum(-1)


@pytest.fixture
def sample(tmp_path):
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    config = load_run_config(Path(__file__).resolve().parents[1] /
        "tests/fixtures/configs/qmix_legacy/fa-gat-qmix-seed42-8k-episode-return-development.json",
        cli_overrides={"training.mappo.training_device":"cpu", "environment.episode_horizon":4,
            "training.qmix.max_training_episodes":2,"training.qmix.max_training_environment_steps":8,
            "training.qmix.checkpoint_interval_steps":4,"output.logs_dir":str(tmp_path/'logs'),
            "output.plots_dir":str(tmp_path/'plots'),"output.dashboard_logs_dir":str(tmp_path/'dash')})
    learner = QMIXLearner(config)
    learner.online.mixer = SumMixer()
    learner.target.mixer = SumMixer()
    with torch.no_grad():
        for system in [learner.online, learner.target]:
            for parameter in system.agent.heads.parameters():
                parameter.zero_()
            system.agent.heads['route'][-1].bias[0] = 1000. # illegal Idle
        learner.online.agent.heads['route'][-1].bias[1:3] = torch.tensor([5., 2.])
        learner.target.agent.heads['route'][-1].bias[1:3] = torch.tensor([1., 10.])
    obs = U2UMECEnvironment(config).reset().observations
    adapter = ActorObservationTensorizer(config)
    first = adapter.encode_step(obs, episode_start=True)
    second = adapter.encode_step(obs, episode_start=False)
    masks = []
    for item in obs:
        route = np.zeros_like(item.action_masks.route_mask)
        route[1:3] = True
        masks.append(replace(item.action_masks, route_mask=route, route_branch_active=True))
    contracts = ((tuple(masks), tuple(masks)),)
    actor = stack_actor_time((first, second))
    proposals = learner.online.agent.actions(actor, learner.online.agent.initial_hidden(), contracts)
    sequence = ReplaySequence(actor=actor, contracts=contracts, proposals=proposals.proposals,
        state=torch.zeros(1,2,learner.online.agent.spec.centralized_state_dim), reward=torch.zeros(1,2),
        bootstrap=torch.tensor([[True,False]]), hidden=learner.online.agent.initial_hidden(),
        target_hidden=learner.target.agent.initial_hidden(), next_actor=None, next_contracts=None,
        next_state=None, episode=0,first_slot=0,executed=({},{}),active=proposals.active)
    sequence.validate()
    yield config, learner, sequence
    torch.set_num_threads(before)


def test_double_q_evaluates_online_actions_instead_of_target_max(sample):
    _, learner, sequence = sample
    # Online chooses Local; target would choose Defer. Four agents each return 1.
    nxt = double_q_next_values(learner.online, learner.target, sequence, 'cpu')
    assert torch.equal(nxt, torch.tensor([[4., 0.]]))
    assert not nxt.requires_grad
    assert all(parameter.grad is None for parameter in learner.target.parameters())


def test_real_update_uses_current_lambda_and_double_q_target(sample):
    _, learner, sequence = sample
    row = learner.update((sequence,))
    assert row['td_target_mean'] == pytest.approx(.99*.05*4/2)
    assert row['q_taken_mean'] == pytest.approx(20.)
    assert row['real_transitions'] == 2 and row['update'] == 1
    assert all(parameter.grad is None for parameter in learner.target.parameters())


@pytest.mark.parametrize("loader", [load_checkpoint, load_continuation_checkpoint])
def test_missing_learning_target_identity_is_rejected(sample, tmp_path, loader):
    config, _, _ = sample
    payload = QMIXTrainer(config).payload('periodic')
    payload['implementation_contract'].pop('learning_target')
    path = tmp_path/'missing.pt'
    path.write_bytes(checkpoint_bytes(payload))
    with pytest.raises(ValueError, match='learning target'):
        loader(path)
