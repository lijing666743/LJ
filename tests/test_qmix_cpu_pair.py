"""Independent legal-pair oracles and opposing online/target CPU preferences."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.config import load_run_config
from src.env.environment import U2UMECEnvironment
from src.models.ca_gat_mappo import ActorObservationTensorizer, stack_actor_time
from src.models.factorized_gat_qmix import CPU_UTILITY_CONTRACT, FactorizedGATQNetwork
from src.models.qmix_continuation import load_continuation_checkpoint
from src.models.qmix_replay import QMIXLearner, ReplaySequence
from src.models.qmix_targets import double_q_next_values
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
    return load_run_config(ROOT / "tests/fixtures/configs/qmix_legacy/fa-gat-qmix-seed42-8k-episode-return-development.json",
        cli_overrides={"training.mappo.training_device": "cpu", "environment.episode_horizon": 8,
            "training.qmix.max_training_episodes": 4, "training.qmix.max_training_environment_steps": 32,
            "training.qmix.checkpoint_interval_steps": 8, "training.qmix.recurrent_sequence_length_slots": 4,
            "training.qmix.sequence_batch_size": 2, "output.logs_dir": str(tmp_path/'logs'),
            "output.plots_dir": str(tmp_path/'plots'), "output.dashboard_logs_dir": str(tmp_path/'dashboard')})


@pytest.fixture
def sample(config):
    observations = U2UMECEnvironment(config).reset().observations
    batch = ActorObservationTensorizer(config).encode_step(observations, episode_start=True)
    masks = tuple(replace(obs.action_masks, cpu_queue_branch_active=True,
        cpu_queue_mask=np.ones_like(obs.action_masks.cpu_queue_mask),
        cpu_frequency_energy_mask=np.ones_like(obs.action_masks.cpu_frequency_energy_mask)) for obs in observations)
    return FactorizedGATQNetwork(config), batch, ((masks,),)


def constant_cpu(network, queues, pairs):
    with torch.no_grad():
        for parameter in network.heads.parameters():
            parameter.zero_()
        network.heads["cpu_queue"][-1].bias.copy_(torch.as_tensor(queues))
        network.heads["cpu_frequency"][-1].bias.copy_(torch.as_tensor(pairs).flatten())


def queue_only(contracts, source):
    masks = []
    for mask in contracts[0][0]:
        legal = np.zeros_like(mask.cpu_queue_mask)
        legal[source+1] = True
        masks.append(replace(mask, cpu_queue_mask=legal))
    return ((tuple(masks),),)


def test_same_state_can_prefer_opposite_frequencies_for_two_queues(sample):
    network, batch, contracts = sample
    pairs = np.full((4, 4), -10.)
    pairs[0, 1], pairs[1, 3] = 7., 8.
    constant_cpu(network, [0.]*5, pairs)
    low = network.actions(batch, network.initial_hidden(), queue_only(contracts, 0))
    high = network.actions(batch, network.initial_hidden(), queue_only(contracts, 1))
    assert all(p.cpu_queue == 0 and p.cpu_frequency == .25 for p in low.proposals[0][0])
    assert all(p.cpu_queue == 1 and p.cpu_frequency == 1. for p in high.proposals[0][0])
    assert torch.equal(low.utilities, torch.full((1, 1, 4), 7.))
    assert torch.equal(high.utilities, torch.full((1, 1, 4), 8.))


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_joint_greedy_matches_independent_enumeration_with_distinct_queue_rows(sample, seed):
    network, batch, contracts = sample
    rng = np.random.default_rng(seed)
    queues, pairs = rng.normal(size=5), rng.normal(size=(4, 4))
    constant_cpu(network, queues, pairs)
    masks = []
    for mask in contracts[0][0]:
        energy = rng.random((4, 4)) > .5
        energy[:, 0] = True
        available = rng.random(5) > .3
        available[0] = True
        masks.append(replace(mask, cpu_frequency_energy_mask=energy, cpu_queue_mask=available))
    contracts = ((tuple(masks),),)
    out = network.actions(batch, network.initial_hidden(), contracts)
    for agent, mask in enumerate(masks):
        candidates = [(queues[0], "idle", 0.)]
        for qi, source in enumerate(mask.cpu_queue_domain[1:], 1):
            if mask.cpu_queue_mask[qi]:
                for fi, frequency in enumerate(mask.cpu_frequency_domain):
                    if mask.cpu_frequency_energy_mask[qi-1, fi]:
                        candidates.append((queues[qi]+pairs[qi-1, fi], source, frequency))
        expected = max(candidates, key=lambda row: row[0])
        actual = out.proposals[0][0][agent]
        assert (actual.cpu_queue, actual.cpu_frequency) == expected[1:]
        assert float(out.utilities[0, 0, agent].detach()) == pytest.approx(expected[0], abs=1e-6)
        assert mask.is_legal(actual)
    teacher = network.actions(batch, network.initial_hidden(), contracts, proposals=out.proposals)
    assert torch.equal(teacher.utilities, out.utilities)


@pytest.mark.parametrize("winner", ["idle", "zero", "positive"])
def test_original_idle_zero_and_positive_actions_can_each_win(sample, winner):
    network, batch, contracts = sample
    queues, pairs = np.full(5, -3.), np.full((4, 4), -5.)
    expected = {"idle": ("idle", 0.), "zero": (0, 0.), "positive": (0, 1.)}[winner]
    if winner == "idle":
        queues[0] = 9.
    else:
        pairs[0, 0 if winner == "zero" else 3] = 12.
    constant_cpu(network, queues, pairs)
    out = network.actions(batch, network.initial_hidden(), contracts)
    assert all((p.cpu_queue, p.cpu_frequency) == expected for p in out.proposals[0][0])


def test_high_illegal_frequency_score_is_excluded_and_zero_fallback_survives(sample):
    network, batch, contracts = sample
    pairs = np.zeros((4, 4)); pairs[0, 3] = 1e6
    constant_cpu(network, [0., 2., 0., 0., 0.], pairs)
    masks = []
    for mask in contracts[0][0]:
        energy = np.zeros_like(mask.cpu_frequency_energy_mask); energy[:, 0] = True
        masks.append(replace(mask, cpu_frequency_energy_mask=energy))
    out = network.actions(batch, network.initial_hidden(), ((tuple(masks),),))
    assert all(p.cpu_queue == 0 and p.cpu_frequency == 0. for p in out.proposals[0][0])


def test_teacher_selects_the_correct_row_on_batch_time_agent_axes(sample):
    network, step, contracts = sample
    pairs = np.arange(16.).reshape(4, 4)
    queues = [100., 1., 2., 3., 4.]
    constant_cpu(network, queues, pairs)
    batch = stack_actor_time((step, step)).repeat_batch(2)
    contracts = tuple((contracts[0][0], contracts[0][0]) for _ in range(2))
    greedy = network.actions(batch, network.initial_hidden(2), contracts)
    proposals = tuple(tuple(tuple(replace(greedy.proposals[b][t][a], cpu_queue=(b+t+a)%4,
        cpu_frequency=(0., .25, .5, 1.)[(b+2*t+a)%4]) for a in range(4)) for t in range(2)) for b in range(2))
    out = network.actions(batch, network.initial_hidden(2), contracts, proposals=proposals)
    assert out.cpu_frequency_values.shape == (2, 2, 4, 4, 4)
    for b in range(2):
        for t in range(2):
            for a, p in enumerate(proposals[b][t]):
                fi = (0., .25, .5, 1.).index(p.cpu_frequency)
                assert out.utilities[b, t, a] == queues[p.cpu_queue+1]+pairs[p.cpu_queue, fi]
                assert torch.equal(out.branch_values["cpu_frequency"][b,t,a], torch.as_tensor(pairs[p.cpu_queue], dtype=torch.float32))


def test_teacher_gradient_reaches_only_the_selected_pair_output(sample):
    network, batch, contracts = sample
    constant_cpu(network, [0.]*5, np.zeros((4, 4)))
    original = network.actions(batch, network.initial_hidden(), contracts)
    proposals = ((tuple(replace(p, cpu_queue=2, cpu_frequency=.5) for p in original.proposals[0][0]),),)
    out = network.actions(batch, network.initial_hidden(), contracts, proposals=proposals)
    out.utilities.sum().backward()
    gradient = network.heads["cpu_frequency"][-1].bias.grad.reshape(4, 4)
    expected = torch.zeros(4, 4); expected[2, 2] = 4.
    assert torch.equal(gradient, expected)
    assert network.heads["cpu_queue"][-1].bias.grad[3] == 4.


def test_inactive_cpu_adds_no_pair_utility_or_gradient(sample):
    network, batch, contracts = sample
    masks = tuple(replace(mask, cpu_queue_branch_active=False,
        cpu_queue_mask=np.array([True, False, False, False, False])) for mask in contracts[0][0])
    constant_cpu(network, [100.]*5, np.full((4, 4), 100.))
    out = network.actions(batch, network.initial_hidden(), ((masks,),))
    assert not out.active.any() and torch.count_nonzero(out.utilities) == 0
    out.utilities.sum().backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in network.heads.parameters())


def test_nonfinite_unselected_pair_is_rejected(sample):
    network, batch, contracts = sample
    pairs = np.zeros((4, 4)); pairs[3, 3] = np.inf
    constant_cpu(network, [0.]*5, pairs)
    with pytest.raises(ValueError, match="non-finite Q"):
        network.actions(batch, network.initial_hidden(), queue_only(contracts, 0))


class SumMixer(nn.Module):
    def forward(self, utilities, state):
        return utilities.sum(-1)


def test_double_q_evaluates_online_cpu_pair_even_when_target_prefers_another(config, sample):
    _, step, contracts = sample
    learner = QMIXLearner(config)
    learner.online.mixer, learner.target.mixer = SumMixer(), SumMixer()
    online_pairs, target_pairs = np.full((4, 4), -10.), np.full((4, 4), -10.)
    online_pairs[0, 1], online_pairs[1, 3] = 5., 4.
    target_pairs[0, 1], target_pairs[1, 3] = -2., 20.
    constant_cpu(learner.online.agent, [0.]*5, online_pairs)
    constant_cpu(learner.target.agent, [0.]*5, target_pairs)
    actor = stack_actor_time((step, replace(step, episode_starts=torch.zeros_like(step.episode_starts))))
    contracts = ((contracts[0][0], contracts[0][0]),)
    chosen = learner.online.agent.actions(actor, learner.online.agent.initial_hidden(), contracts)
    seq = ReplaySequence(actor=actor, contracts=contracts, proposals=chosen.proposals,
        state=torch.zeros(1,2,learner.online.agent.spec.centralized_state_dim), reward=torch.zeros(1,2),
        bootstrap=torch.tensor([[True,False]]), hidden=learner.online.agent.initial_hidden(),
        target_hidden=learner.target.agent.initial_hidden(), next_actor=None, next_contracts=None,
        next_state=None, episode=0, first_slot=0, executed=({},{}), active=chosen.active)
    actual = double_q_next_values(learner.online, learner.target, seq, "cpu")
    assert torch.equal(actual, torch.tensor([[-8., 0.]]))
    assert not actual.requires_grad
    assert all(p.grad is None for p in learner.target.parameters())


@pytest.mark.parametrize("loader", [load_checkpoint, load_continuation_checkpoint])
@pytest.mark.parametrize("mutation", ["missing", "altered"])
def test_checkpoint_requires_matching_cpu_utility_contract(config, tmp_path, loader, mutation):
    payload = QMIXTrainer(config).payload("periodic")
    assert payload["implementation_contract"]["cpu_utility"] == CPU_UTILITY_CONTRACT
    if mutation == "missing":
        payload["implementation_contract"].pop("cpu_utility")
    else:
        payload["implementation_contract"]["cpu_utility"] = {"kind": "independent_frequency"}
    path = tmp_path / "incompatible.pt"; path.write_bytes(checkpoint_bytes(payload))
    with pytest.raises(ValueError, match="CPU utility"):
        loader(path)


@pytest.mark.parametrize("loader", [load_checkpoint, load_continuation_checkpoint])
def test_previous_v8_trial_schema_v7_is_rejected(config, tmp_path, loader):
    payload = QMIXTrainer(config).payload("periodic")
    payload["schema"] = "factorized_gat_qmix.checkpoint.v7"
    path = tmp_path / "old.pt"; path.write_bytes(checkpoint_bytes(payload))
    with pytest.raises(ValueError, match="(schema|checkpoint)"):
        loader(path)
