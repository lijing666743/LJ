"""Independent long-return oracles, causal actor boundary and replay isolation."""
import copy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.config import load_run_config
from src.models.qmix_continuation import load_continuation_checkpoint, state_digest
from src.models.qmix_recurrent import reconstruct_replay_sequences
from src.models.qmix_replay import SequenceReplay
from src.models.qmix_targets import (TD_LAMBDA, double_q_next_values,
    join_return_context, replay_return_targets, td_lambda_targets)
from src.models.qmix_trainer import QMIXTrainer, checkpoint_bytes, load_checkpoint


@pytest.fixture
def trained(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    config = load_run_config(Path(__file__).resolve().parents[1] /
        'tests/fixtures/configs/qmix_legacy/fa-gat-qmix-seed42-8k-episode-return-development.json',
        cli_overrides={'training.mappo.training_device':'cpu','environment.episode_horizon':12,
            'training.qmix.max_training_episodes':2,'training.qmix.max_training_environment_steps':24,
            'training.qmix.checkpoint_interval_steps':12,'training.qmix.recurrent_sequence_length_slots':4,
            'training.qmix.sequence_batch_size':2,'output.logs_dir':str(tmp_path/'l'),
            'output.plots_dir':str(tmp_path/'p'),'output.dashboard_logs_dir':str(tmp_path/'d')})
    trainer = QMIXTrainer(config)
    trainer.train()
    yield trainer
    torch.set_num_threads(previous)


def forward_mixture(rewards, next_values, bootstrap, gamma, lam):
    """Explicit mixture of n-step returns, independent of reverse recurrence."""
    result = []
    for start in range(len(rewards)):
        last = next((j for j in range(start, len(rewards)) if not bootstrap[j]), len(rewards)-1)
        terms = []
        for stop in range(start, last+1):
            length = stop-start+1
            value = sum(gamma**(j-start)*rewards[j] for j in range(start, stop+1))
            if bootstrap[stop]:
                value += gamma**length*next_values[stop]
            weight = lam**(length-1)*(1-lam if stop < last else 1)
            terms.append(weight*value)
        result.append(sum(terms))
    return torch.tensor([result])


@pytest.mark.parametrize('terminal',[False,True])
@pytest.mark.parametrize('lam',[0.,.6,.95,1.])
def test_reverse_returns_match_independent_forward_n_step_mixture(terminal,lam):
    rewards = [0.,-.2,1.,.4,-.1]
    nxt = [3.,2.,-1.,.5,8.]
    bootstrap = [True]*4+[not terminal]
    actual = td_lambda_targets(torch.tensor([rewards]),torch.tensor([nxt]),
        torch.tensor([bootstrap]),.99,lam)
    assert torch.allclose(actual,forward_mixture(rewards,nxt,bootstrap,.99,lam),atol=2e-6)


def test_completion_after_gradient_chunk_reaches_earlier_route(trained):
    chunks = [replace(seq,reward=torch.zeros_like(seq.reward)) for seq in trained.replay.sequences[:3]]
    chunks[2].reward[0,-1] = 1.
    replay = SequenceReplay(20,42)
    for seq in chunks:replay.add(seq)
    rebuilt,_ = reconstruct_replay_sequences(trained.learner.online,trained.learner.target,(chunks[0],),replay)
    with patch('src.models.qmix_targets.double_q_next_values',
               side_effect=lambda online,target,seq,device:torch.zeros_like(seq.reward)):
        labels,stats = replay_return_targets(trained.learner.online,trained.learner.target,
            rebuilt,'cpu',.99,replay=replay)
    assert labels[0][0,0] == pytest.approx((.99*.95)**11)
    assert labels[0].shape == (1,4) and not labels[0].requires_grad
    assert stats == {'td_return_context_slots':12,'td_return_extra_slots':8}
    old = td_lambda_targets(chunks[0].reward,torch.zeros(1,4),chunks[0].bootstrap,.99,.6)
    assert old[0,0] == 0.


def test_grouped_suffix_matches_full_episode_double_q_oracle_in_sample_order(trained):
    learner,replay = trained.learner,trained.replay
    chosen = (replay.sequences[2],replay.sequences[0],replay.sequences[4])
    rebuilt,_ = reconstruct_replay_sequences(learner.online,learner.target,chosen,replay)
    with patch('src.models.qmix_targets.double_q_next_values',wraps=double_q_next_values) as called:
        labels,stats = replay_return_targets(learner.online,learner.target,rebuilt,'cpu',.99,replay=replay)
    assert called.call_count == 2
    for seq,actual in zip(chosen,labels):
        first = replay.sequences[seq.episode*3]
        full = join_return_context(replay.future_sequences(first))
        full = replace(full,hidden=learner.online.agent.initial_hidden(),
                       target_hidden=learner.target.agent.initial_hidden())
        oracle = td_lambda_targets(full.reward,double_q_next_values(learner.online,learner.target,full,'cpu'),
                                  full.bootstrap,.99)
        assert torch.allclose(actual,oracle[:,seq.first_slot:seq.first_slot+seq.length],atol=2e-6)
    assert stats['td_return_context_slots'] == 20


def test_live_episode_tail_bootstraps_only_real_collected_next_state(trained):
    first,second = trained.replay.sequences[:2]
    replay = SequenceReplay(20,42)
    replay.add(first)
    assert replay.future_sequences(first) == (first,)
    a,_ = replay_return_targets(trained.learner.online,trained.learner.target,(first,),'cpu',.99,replay=replay)
    oracle = td_lambda_targets(first.reward,double_q_next_values(trained.learner.online,
        trained.learner.target,first,'cpu'),first.bootstrap,.99)
    assert torch.equal(a[0],oracle)
    replay.add(second)
    assert join_return_context(replay.future_sequences(first)).next_actor is replay.sequences[-1].next_actor


def test_future_rewards_change_labels_but_never_earlier_actor_or_rng(trained):
    learner,replay = trained.learner,trained.replay
    first = replay.sequences[0]
    before = state_digest((learner.online.state_dict(),learner.target.state_dict(),replay.rng.bit_generator.state))
    rng = torch.random.get_rng_state().clone()
    with torch.no_grad():
        actor_before = learner.online.agent.actions(first.actor,first.hidden,first.contracts).utilities.clone()
    a,_ = replay_return_targets(learner.online,learner.target,(first,),'cpu',.99,replay=replay)
    replay.sequences[2] = replace(replay.sequences[2],reward=replay.sequences[2].reward+1)
    b,_ = replay_return_targets(learner.online,learner.target,(first,),'cpu',.99,replay=replay)
    with torch.no_grad():
        actor_after = learner.online.agent.actions(first.actor,first.hidden,first.contracts).utilities
    assert not torch.equal(a[0],b[0])
    assert torch.equal(actor_before,actor_after)
    assert state_digest((learner.online.state_dict(),learner.target.state_dict(),replay.rng.bit_generator.state)) == before
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(p.grad is None for p in learner.target.parameters())


@pytest.mark.parametrize('fault',['gap','duplicate','episode','terminal','reset'])
def test_invalid_suffix_cannot_silently_bridge_missing_or_terminal_data(trained,fault):
    first,second = trained.replay.sequences[:2]
    if fault == 'gap':second = replace(second,first_slot=5)
    if fault == 'duplicate':second = replace(second,first_slot=0)
    if fault == 'episode':second = replace(second,episode=1)
    if fault == 'terminal':first = replace(first,bootstrap=torch.tensor([[True,True,True,False]]),
        next_actor=None,next_contracts=None,next_state=None)
    if fault == 'reset':second = replace(second,actor=replace(second.actor,
        episode_starts=torch.ones_like(second.actor.episode_starts)))
    with pytest.raises(ValueError):join_return_context((first,second))


def test_replay_suffix_gap_is_rejected_before_parameters_update(trained):
    learner,replay = trained.learner,trained.replay
    first = replay.sequences[0]
    replay.sequences.pop(1)
    before = state_digest(learner.online.state_dict())
    updates = learner.updates
    with pytest.raises(ValueError,match='missing'):learner.update((first,),replay=replay)
    assert state_digest(learner.online.state_dict()) == before and learner.updates == updates


def test_only_sampled_chunk_gets_gradients_and_loss_weight_stays_real_length(trained):
    learner,replay = trained.learner,trained.replay
    original = learner.online.agent.actions
    lengths = []
    def observe(batch,*args,**kwargs):
        if torch.is_grad_enabled():lengths.append(batch.self_features.shape[1])
        return original(batch,*args,**kwargs)
    with patch.object(learner.online.agent,'actions',side_effect=observe):
        row = learner.update((replay.sequences[0],),replay=replay)
    assert lengths == [4]
    assert row['real_transitions'] == 4 and row['td_return_context_slots'] == 12
    assert row['td_return_extra_slots'] == 8 and torch.isfinite(torch.tensor(row['loss']))
    assert all(p.grad is None for p in learner.target.parameters())


@pytest.mark.parametrize('loader',[load_checkpoint,load_continuation_checkpoint])
@pytest.mark.parametrize('fault',['old_schema','old_target'])
def test_old_chunk_return_training_cannot_resume_new_semantics(trained,tmp_path,loader,fault):
    payload = trained.payload('final')
    if fault == 'old_schema':payload['schema'] = 'factorized_gat_qmix.checkpoint.v8'
    else:payload['implementation_contract']['learning_target'] = 'double_q_chunk_td_lambda_0p6_v1'
    path = tmp_path/'old.pt';path.write_bytes(checkpoint_bytes(payload))
    with pytest.raises(ValueError,match='(schema|checkpoint|learning target)'):loader(path)


def test_metadata_names_actual_return_and_gradient_boundaries(trained):
    contract = trained.payload('final')['implementation_contract']
    assert TD_LAMBDA == contract['td_lambda'] == .95
    assert 'available_same_episode_suffix' in contract['trace_boundary']
    assert contract['gradient_boundary'] == 'sampled_real_chunk_detached_prefix'


def test_completion_credit_auditor_reconstructs_real_tiny_training(trained,tmp_path):
    from src.evaluation.qmix_completion_credit import run
    from src.models.qmix_trainer import checkpoint_bytes
    cfg = tmp_path/'config.json'
    import json
    cfg.write_text(json.dumps(trained.config.resolved_dict()),encoding='utf-8')
    checkpoint = Path(trained.config.artifact_paths()['final_checkpoint'])
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    checkpoint.write_bytes(checkpoint_bytes(trained.payload('final')))
    result = run(cfg,tmp_path/'credit.json')
    assert result['optimizer_updates'] == trained.learner.updates
    assert result['replay_draws_exactly_reconstructed']
    assert result['completed_tasks'] == sum(row['completed'] for row in trained.episodes)
    assert result['optimizer_updates_during_audit'] == 0
