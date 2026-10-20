"""Independent neural, replay, update, real environment, and resume gates."""
from __future__ import annotations

import copy
import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import load_run_config
from src.env.environment import U2UMECEnvironment
from src.models.ca_gat_mappo import ActorObservationTensorizer, stack_actor_time
from src.models.factorized_gat_qmix import FactorizedGATQNetwork, MonotonicMixer, masked_half_mse, td_target
from src.models.qmix_trainer import QMIXTrainer, checkpoint_bytes, load_checkpoint
from src.models.qmix_replay import ReplaySequence, SequenceReplay
from src.evaluation.qmix_runner import QMIXEvaluationRunner


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
        "evaluation.evaluation_seeds": [1042],
        "output.logs_dir": str(tmp_path / "logs"),
        "output.plots_dir": str(tmp_path / "plots"),
        "output.dashboard_logs_dir": str(tmp_path / "dashboard_logs"),
    })


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_mixer_monotonic_gradients_and_state_only_training():
    mixer = MonotonicMixer(4, 9)
    utility = torch.randn(2, 3, 4, requires_grad=True)
    state = torch.randn(2, 3, 9)
    gradient = torch.autograd.grad(mixer(utility, state).sum(), utility)[0]
    assert (gradient >= 0).all()
    assert not torch.equal(mixer(utility, state), mixer(utility, state+torch.arange(9)))


def test_mixer_state_scale_is_bounded_without_changing_monotonicity():
    mixer = MonotonicMixer(4, 9)
    utility = torch.ones(2, 3, 4, requires_grad=True)
    state = torch.tensor([0., 1., -1., 1.e6, 2.e5, 0., 3., 0., 10.]).expand(2, 3, 9)
    total = mixer(utility, state)
    assert torch.isfinite(total).all() and total.abs().max() < 100.
    assert (torch.autograd.grad(total.sum(), utility)[0] >= 0).all()


def test_td_boundaries_and_padding_do_not_bootstrap_or_leak():
    reward = torch.tensor([1., 2., 3.])
    nxt = torch.tensor([10., float("nan"), 9.], requires_grad=True)
    target = td_target(reward, nxt, torch.tensor([True, False, False]), .9)
    assert torch.equal(target, torch.tensor([10., 2., 3.]))
    prediction = torch.tensor([8., float("nan"), 1.], requires_grad=True)
    loss = masked_half_mse(prediction, target, torch.tensor([True, False, True]))
    assert loss.item() == 2.
    loss.backward()
    assert nxt.grad is None
    assert torch.equal(prediction.grad, torch.tensor([-1., 0., -1.]))


def test_actor_greedy_exploration_and_teacher_have_same_legal_contract(config):
    env = U2UMECEnvironment(config)
    obs = env.reset().observations
    network = FactorizedGATQNetwork(config)
    adapter = ActorObservationTensorizer(config)
    hidden = network.initial_hidden()
    rng = np.random.default_rng(120)
    for _ in range(8):
        batch = adapter.encode_step(obs)
        contracts = ((tuple(item.action_masks for item in obs),),)
        before_rng = copy.deepcopy(rng.bit_generator.state)
        greedy = network.actions(batch, hidden, contracts, rng=rng)
        assert rng.bit_generator.state == before_rng
        explored = network.actions(batch, hidden, contracts, epsilon=1., rng=rng)
        assert all(item.action_masks.is_legal(p) for item, p in zip(obs, explored.proposals[0][0]))
        teacher = network.actions(batch, hidden, contracts, proposals=greedy.proposals)
        assert torch.equal(teacher.utilities, greedy.utilities)
        expected = torch.stack([greedy.branch_values[name].gather(-1, greedy.indices[name].unsqueeze(-1)).squeeze(-1)
                                for name in greedy.indices], dim=-1)
        assert torch.equal(greedy.utilities, (expected*greedy.active).sum(-1))
        result = env.step(explored.proposals[0][0])
        hidden = explored.hidden_out.detach()
        if result.truncated or result.terminated:
            break
        obs = result.observations


def test_small_route_domain_exhaustive_greedy_and_inactive_gradients(config):
    obs = U2UMECEnvironment(config).reset().observations
    network = FactorizedGATQNetwork(config)
    batch = ActorObservationTensorizer(config).encode_step(obs)
    contracts = ((tuple(item.action_masks for item in obs),),)
    inactive = network.actions(batch, network.initial_hidden(), contracts)
    inactive.utilities.sum().backward()
    assert not inactive.active.any()
    assert all(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
               for parameter in network.heads.parameters())
    mask = obs[0].action_masks
    route_mask = np.zeros(len(mask.route_domain), dtype=bool)
    route_mask[1:3] = True  # local/defer; other branches retain canonical Idle.
    changed = replace(mask, route_mask=route_mask, route_branch_active=True)
    contracts = (((changed, *contracts[0][0][1:]),),)
    with torch.no_grad():
        # An illegal Idle score dominates numerically but must be excluded.
        network.heads['route'][-1].bias[0] = 1.e6
    greedy = network.actions(batch, network.initial_hidden(), contracts)
    utilities = []
    for index in np.flatnonzero(route_mask):
        proposal = replace(greedy.proposals[0][0][0], route=mask.route_domain[index])
        teacher = (((proposal, *greedy.proposals[0][0][1:]),),)
        utilities.append(network.actions(batch, network.initial_hidden(), contracts,
                                        proposals=teacher).utilities[0, 0, 0])
    assert torch.equal(greedy.utilities[0, 0, 0], torch.stack(utilities).max())


def test_initialization_preserves_global_rng_and_repeats_by_seed(config):
    before = torch.random.get_rng_state().clone()
    first = FactorizedGATQNetwork(config)
    assert torch.equal(before, torch.random.get_rng_state())
    second = FactorizedGATQNetwork(config)
    assert_nested_equal(first.state_dict(), second.state_dict())


def test_recurrent_time_order_and_episode_reset(config):
    obs = U2UMECEnvironment(config).reset().observations
    adapter = ActorObservationTensorizer(config)
    first = adapter.encode_step(obs, episode_start=True)
    second = adapter.encode_step(obs, episode_start=False)
    network = FactorizedGATQNetwork(config)
    h = network.initial_hidden()
    together, final = network.encode(stack_actor_time((first, second)), h)
    one, midway = network.encode(first, h)
    two, sequential = network.encode(second, midway)
    assert torch.allclose(together, torch.cat((one, two), 1), atol=1e-6)
    assert torch.allclose(final, sequential, atol=1e-6)
    a, _ = network.encode(first, h)
    b, _ = network.encode(first, h+100.)
    assert torch.equal(a, b)


def test_real_updates_replay_and_target_copies(config):
    trainer = QMIXTrainer(config)
    before = copy.deepcopy(trainer.learner.online.state_dict())
    trainer.train()
    assert trainer.steps == 32 and trainer.next_episode == 4
    assert trainer.learner.updates == 7
    assert len(trainer.replay.sequences) == 8
    assert any(not torch.equal(before[k], v) for k, v in trainer.learner.online.state_dict().items())
    assert any(update["target_updated"] for update in trainer.updates)
    assert all(update["loss"] >= 0 and np.isfinite(update["gradient_norm"]) for update in trainer.updates)
    assert all(p.grad is None for p in trainer.learner.target.parameters())
    for seq in trainer.replay.sequences:
        assert seq.length == 4 and len(seq.executed) == 4
        assert bool(seq.bootstrap[0, -1]) == (seq.first_slot == 0)
        assert_nested_equal(seq.to_state(), ReplaySequence.from_state(seq.to_state()).to_state())


def test_checkpoint_restricted_loader_and_exact_resume(config, tmp_path):
    uninterrupted = QMIXTrainer(config)
    uninterrupted.train()
    first = QMIXTrainer(config)
    first.train(stop_after_episodes=2)
    path = tmp_path / "periodic.pt"
    path.write_bytes(checkpoint_bytes(first.payload("periodic")))
    loaded = load_checkpoint(path)
    assert loaded["steps"] == 16
    restored = QMIXTrainer(config)
    restored.restore(path)
    restored.train()
    assert_nested_equal(uninterrupted.payload("final"), restored.payload("final"))


def test_replay_rng_and_owned_storage_survive_serialization(config):
    trainer = QMIXTrainer(config)
    trainer.train(stop_after_episodes=2)
    saved = trainer.replay.to_state()
    restored = SequenceReplay.from_state(saved)
    assert [seq.first_slot for seq in trainer.replay.sample(3)] == [seq.first_slot for seq in restored.sample(3)]
    saved["sequences"][0]["reward"].add_(99.)
    assert not torch.equal(saved["sequences"][0]["reward"], restored.sequences[0].reward)


def test_final_evaluation_is_frozen_repeatable_and_matches_external_processes(config, tmp_path):
    trainer = QMIXTrainer(config)
    trainer.train()
    path = tmp_path / "final.pt"
    path.write_bytes(checkpoint_bytes(trainer.payload("final")))
    evaluation = replace(config, mode="evaluation")
    runner = QMIXEvaluationRunner(evaluation, path)
    before = copy.deepcopy(runner.agent.state_dict())
    one = runner.run(write_artifacts=False)
    two = runner.run(write_artifacts=False)
    assert [item.artifact_record("test") for item in one["episodes"]] == [item.artifact_record("test") for item in two["episodes"]]
    assert len(one["episodes"]) == 4
    assert len({item.external_trace_sha256 for item in one["episodes"]}) == 1
    assert_nested_equal(before, runner.agent.state_dict())
    assert all(p.grad is None and not p.requires_grad for p in runner.agent.parameters())


def test_cross_method_checkpoint_and_budget_drift_are_rejected(config, tmp_path):
    invalid = tmp_path / "other.pt"
    torch.save({"method_id": "ca_gat_mappo"}, invalid)
    with pytest.raises(ValueError, match="not a Factorized"):
        load_checkpoint(invalid)
    trainer = QMIXTrainer(config)
    trainer.train(stop_after_episodes=1)
    path = tmp_path / "periodic.pt"
    path.write_bytes(checkpoint_bytes(trainer.payload("periodic")))
    changed = replace(config, seed=44)
    with pytest.raises(ValueError, match="identical configuration"):
        QMIXTrainer(changed).restore(path)
    with pytest.raises(ValueError, match="final checkpoint"):
        QMIXEvaluationRunner(replace(config, mode="evaluation"), path)


def test_checkpoint_rejects_changed_implementation(config, tmp_path):
    trainer = QMIXTrainer(config)
    data = trainer.payload("periodic")
    data["implementation_source_sha256"] = {}
    path = tmp_path / "changed_source.pt"
    path.write_bytes(checkpoint_bytes(data))
    with pytest.raises(ValueError, match="implementation source differs"):
        load_checkpoint(path)


def test_formal_disabled_draft_cannot_start_training(config):
    disabled = replace(config, training=replace(config.training, formal_rl_enabled=False))
    with pytest.raises(Exception, match="formal RL training is disabled"):
        QMIXTrainer(disabled)


def test_short_terminal_chunks_are_kept_with_no_fake_next_state(config):
    config = replace(config, environment=replace(config.environment, episode_horizon=10),
                     training=replace(config.training, qmix=replace(config.training.qmix,
                         max_training_episodes=2, max_training_environment_steps=20, checkpoint_interval_steps=10)))
    trainer = QMIXTrainer(config)
    trainer.train()
    assert trainer.steps == 20
    assert [seq.length for seq in trainer.replay.sequences] == [4, 4, 2, 4, 4, 2]
    for seq in trainer.replay.sequences:
        if seq.length == 2:
            assert seq.next_actor is None and seq.next_state is None
            assert not seq.bootstrap[0, -1]


def test_checkpoint_publication_rejects_overwrite_before_collection(config, tmp_path):
    path = Path(config.artifact_paths()["checkpoint_directory"]) / "step_000008.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"preserve checkpoint")
    trainer = QMIXTrainer(config)
    with pytest.raises(FileExistsError, match="before collection"):
        trainer.train(write_checkpoints=True)
    assert trainer.steps == 0 and path.read_bytes() == b"preserve checkpoint"
