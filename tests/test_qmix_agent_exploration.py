"""Agent-level epsilon decisions preserve conditional masks and exact greediness."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import load_run_config
from src.env.actions import ActionProposal
from src.env.environment import U2UMECEnvironment
from src.env.observation import BRANCH_ORDER
from src.models.ca_gat_mappo import ActorObservationTensorizer, stack_actor_time
from src.models.factorized_gat_qmix import EXPLORATION_CONTRACT, FactorizedGATQNetwork
from src.models.qmix_continuation import load_continuation_checkpoint
from src.models.qmix_trainer import QMIXTrainer, checkpoint_bytes, load_checkpoint


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config(tmp_path):
    return load_run_config(ROOT / "experiments/task-allocation/configs/c-seed43-40k.json", cli_overrides={
        "method_id": "factorized_action_gat_qmix", "launch_profile": None,
        "training.formal_rl_enabled": True, "training.mappo.training_device": "cpu",
        "environment.episode_horizon": 8,
        "training.qmix.max_training_episodes": 4,
        "training.qmix.max_training_environment_steps": 32,
        "training.qmix.checkpoint_interval_steps": 8,
        "training.qmix.recurrent_sequence_length_slots": 4,
        "training.qmix.sequence_batch_size": 2,
        "training.qmix.target_update_interval_updates": 2,
        "training.qmix.epsilon_start": .5, "training.qmix.epsilon_end": .5,
        "evaluation.evaluation_seeds": [1042],
        "output.logs_dir": str(tmp_path / "logs"),
        "output.plots_dir": str(tmp_path / "plots"),
        "output.dashboard_logs_dir": str(tmp_path / "dashboard_logs"),
    })


@pytest.fixture
def setup(config):
    observations = U2UMECEnvironment(config).reset().observations
    network = FactorizedGATQNetwork(config)
    masks = []
    for observation in observations:
        mask = observation.action_masks
        route = np.zeros(len(mask.route_domain), dtype=bool)
        route[1:3] = True
        tx = np.zeros(len(mask.tx_select_domain), dtype=bool)
        tx[:2] = True
        cpu = np.zeros(len(mask.cpu_queue_domain), dtype=bool)
        cpu[:3] = True
        frequencies = np.zeros_like(mask.cpu_frequency_energy_mask)
        frequencies[:, 0] = True
        frequencies[0, 1] = True
        frequencies[1, -1] = True
        power = np.zeros(len(mask.power_level_domain), dtype=bool)
        power[[0, 1, -1]] = True
        masks.append(replace(mask, route_mask=route, tx_select_mask=tx, cpu_queue_mask=cpu,
                             cpu_frequency_energy_mask=frequencies, power_energy_mask=power,
                             route_branch_active=True, tx_branch_active=True, cpu_queue_branch_active=True))
    batch = ActorObservationTensorizer(config).encode_step(observations)
    return network, batch, ((tuple(masks),),)


class ScriptedRNG:
    """Count Bernoulli draws; deterministic last-legal choices expose mixing."""

    def __init__(self, draws):
        self.draws = tuple(draws)
        self.random_calls = 0
        self.choices = []

    def random(self):
        assert self.random_calls < len(self.draws), "unexpected additional epsilon trigger"
        result = self.draws[self.random_calls]
        self.random_calls += 1
        return result

    def choice(self, legal):
        legal = tuple(int(index) for index in legal)
        assert len(legal) > 1, "forced branches need no random choice"
        self.choices.append(legal)
        return legal[-1]


def set_scores(network, scores=None):
    with torch.no_grad():
        for branch, head in network.heads.items():
            for parameter in head.parameters():
                parameter.zero_()
            if scores is not None:
                value = torch.as_tensor(scores[branch])
                if branch == "cpu_frequency":
                    value = value.repeat(network.cpu_queue_count)
                head[-1].bias.copy_(value)


def enumerate_legal(mask, prefix=None):
    """Independent complete-action oracle, without the network's suffix decoder."""
    prefix = {} if prefix is None else prefix
    if len(prefix) == len(BRANCH_ORDER):
        yield ActionProposal(uav_id=mask.uav_id, **prefix)
        return
    branch = BRANCH_ORDER[len(prefix)]
    for value, legal in zip(mask.domain_for(branch), mask.mask_for(branch, prefix)):
        if legal:
            yield from enumerate_legal(mask, {**prefix, branch: value})


def test_one_trigger_per_batch_time_agent_controls_every_branch(setup):
    network, step, contracts = setup
    set_scores(network)
    batch = stack_actor_time([step, step]).repeat_batch(2)
    contracts = tuple((contracts[0][0], contracts[0][0]) for _ in range(2))
    draws = [.1, .9, .2, .8, .8, .1, .9, .2, .3, .7, .7, .3, .9, .9, .1, .1]
    rng = ScriptedRNG(draws)
    hidden = network.initial_hidden(2)
    greedy = network.actions(batch, hidden, contracts)
    result = network.actions(batch, hidden, contracts, epsilon=.5, rng=rng)
    expected = torch.as_tensor(np.array(draws).reshape(2, 2, 4) < .5)
    assert result.exploration_flags.dtype == torch.bool
    assert torch.equal(result.exploration_flags, expected)
    assert rng.random_calls == expected.numel()
    random_choices = 0
    for b in range(2):
        for t in range(2):
            for a, mask in enumerate(contracts[b][t]):
                proposal = result.proposals[b][t][a]
                assert mask.is_legal(proposal)
                if not expected[b, t, a]:
                    assert proposal == greedy.proposals[b][t][a]
                    assert result.utilities[b, t, a] == greedy.utilities[b, t, a]
                else:
                    context = {}
                    for branch in BRANCH_ORDER:
                        legal = np.flatnonzero(mask.mask_for(branch, context))
                        value = getattr(proposal, branch)
                        if len(legal) > 1:
                            assert value == mask.domain_for(branch)[legal[-1]]
                            random_choices += 1
                        context[branch] = value
    assert len(rng.choices) == random_choices


@pytest.mark.parametrize("score_seed", [42, 43, 44])
def test_nonexploring_agents_still_reach_exhaustive_complete_action_maximum(setup, score_seed):
    network, batch, contracts = setup
    scores_rng = np.random.default_rng(score_seed)
    scores = {branch: scores_rng.normal(size=network.spec.action_dimensions[branch]) for branch in BRANCH_ORDER}
    set_scores(network, scores)
    result = network.actions(batch, network.initial_hidden(), contracts, epsilon=.5,
                             rng=ScriptedRNG([.9, .1, .9, .1]))
    for agent in (0, 2):
        mask = contracts[0][0][agent]
        maximum = -float("inf")
        for proposal in enumerate_legal(mask):
            utility = sum(float(result.branch_values[branch][0, 0, agent,
                          mask.domain_for(branch).index(getattr(proposal, branch))])
                          for branch, active in zip(BRANCH_ORDER, mask.active_indicators(proposal)) if active)
            maximum = max(maximum, utility)
        assert float(result.utilities[0, 0, agent].detach()) == pytest.approx(maximum, abs=1.e-6)


def test_greedy_and_teacher_paths_do_not_consume_exploration_rng(setup):
    network, batch, contracts = setup
    rng = ScriptedRNG([])
    greedy = network.actions(batch, network.initial_hidden(), contracts, epsilon=0., rng=rng)
    teacher = network.actions(batch, network.initial_hidden(), contracts,
                              proposals=greedy.proposals, epsilon=.7, rng=rng)
    assert not greedy.exploration_flags.any()
    assert not teacher.exploration_flags.any()
    assert rng.random_calls == 0 and not rng.choices
    assert teacher.proposals == greedy.proposals
    assert torch.equal(teacher.utilities, greedy.utilities)


def test_private_rng_state_restores_the_same_exploration_flags_and_actions(setup):
    network, batch, contracts = setup
    first_rng = np.random.default_rng(1042)
    saved = copy.deepcopy(first_rng.bit_generator.state)
    first = network.actions(batch, network.initial_hidden(), contracts, epsilon=.5, rng=first_rng)
    next_state = copy.deepcopy(first_rng.bit_generator.state)
    restored_rng = np.random.default_rng(0)
    restored_rng.bit_generator.state = saved
    restored = network.actions(batch, network.initial_hidden(), contracts, epsilon=.5, rng=restored_rng)
    assert first.proposals == restored.proposals
    assert torch.equal(first.exploration_flags, restored.exploration_flags)
    assert torch.equal(first.utilities, restored.utilities)
    assert restored_rng.bit_generator.state == next_state


def test_fully_forced_agents_still_receive_one_agent_level_trigger(setup):
    network, batch, contracts = setup
    forced = []
    for mask in contracts[0][0]:
        route = np.zeros_like(mask.route_mask)
        route[1] = True
        tx = np.zeros_like(mask.tx_select_mask)
        tx[0] = True
        cpu = np.zeros_like(mask.cpu_queue_mask)
        cpu[0] = True
        forced.append(replace(mask, route_mask=route, tx_select_mask=tx, cpu_queue_mask=cpu))
    rng = ScriptedRNG([.1] * len(forced))
    result = network.actions(batch, network.initial_hidden(), ((tuple(forced),),), epsilon=.5, rng=rng)
    assert result.exploration_flags.all()
    assert rng.random_calls == len(forced) and not rng.choices
    for mask, proposal in zip(forced, result.proposals[0][0]):
        assert mask.is_legal(proposal)
        assert proposal.tx_select == "idle" and proposal.cpu_queue == "idle"
        assert proposal.resource_group == "idle"
        assert proposal.resource_width == mask.canonical_width
        assert proposal.power_level == mask.canonical_power
        assert proposal.cpu_frequency == mask.canonical_cpu_frequency


def test_exploring_nonidle_services_preserve_zero_energy_fallback(setup):
    network, batch, contracts = setup
    fallback = []
    for mask in contracts[0][0]:
        frequency = np.zeros_like(mask.cpu_frequency_energy_mask)
        frequency[:, 0] = True
        power = np.zeros_like(mask.power_energy_mask)
        power[0] = True
        fallback.append(replace(mask, cpu_frequency_energy_mask=frequency, power_energy_mask=power))
    rng = ScriptedRNG([.1] * len(fallback))
    result = network.actions(batch, network.initial_hidden(), ((tuple(fallback),),), epsilon=1., rng=rng)
    assert result.exploration_flags.all()
    for mask, proposal in zip(fallback, result.proposals[0][0]):
        assert mask.is_legal(proposal)
        assert proposal.tx_select != "idle" and proposal.cpu_queue != "idle"
        assert proposal.power_level == 0. and proposal.cpu_frequency == 0.


@pytest.mark.parametrize("loader", [load_checkpoint, load_continuation_checkpoint])
@pytest.mark.parametrize("mutation", ["missing", "altered"])
def test_checkpoint_loaders_reject_missing_or_changed_exploration_contract(config, tmp_path, loader, mutation):
    payload = QMIXTrainer(config).payload("periodic")
    assert payload["implementation_contract"]["exploration"] == EXPLORATION_CONTRACT
    if mutation == "missing":
        payload["implementation_contract"].pop("exploration")
    else:
        payload["implementation_contract"]["exploration"] = {"obsolete": "independent_branch_bernoulli"}
    path = tmp_path / f"{loader.__name__}_{mutation}.pt"
    path.write_bytes(checkpoint_bytes(payload))
    with pytest.raises(ValueError, match="exploration"):
        loader(path)


@pytest.mark.parametrize("loader", [load_checkpoint, load_continuation_checkpoint])
def test_checkpoint_loaders_reject_old_branchwise_schema_v6(config, tmp_path, loader):
    payload = QMIXTrainer(config).payload("periodic")
    payload["schema"] = "factorized_gat_qmix.checkpoint.v6"
    path = tmp_path / f"{loader.__name__}_v6.pt"
    path.write_bytes(checkpoint_bytes(payload))
    with pytest.raises(ValueError, match="(checkpoint|schema)"):
        loader(path)


def test_real_collection_episode_counts_match_returned_agent_exploration_flags(config, monkeypatch):
    trainer = QMIXTrainer(config)
    original = trainer.learner.online.agent.actions
    collection_flags = []

    def record_collection(*args, **kwargs):
        result = original(*args, **kwargs)
        if "epsilon" in kwargs and kwargs.get("proposals") is None:
            collection_flags.append(result.exploration_flags.detach().cpu().clone())
        return result

    monkeypatch.setattr(trainer.learner.online.agent, "actions", record_collection)
    trainer.train(stop_after_episodes=2)
    horizon = config.environment.episode_horizon
    agents = trainer.learner.online.agent.spec.uav_count
    assert len(collection_flags) == 2 * horizon
    for episode, row in enumerate(trainer.episodes):
        flags = torch.stack(collection_flags[episode * horizon:(episode + 1) * horizon])
        exploring = int(flags.sum())
        assert row["exploration_agent_decisions"] == exploring
        assert row["greedy_agent_decisions"] == flags.numel() - exploring
        assert row["exploration_agent_decisions"] + row["greedy_agent_decisions"] == horizon * agents
