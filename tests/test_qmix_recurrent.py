"""Current-weight context, prefix eviction ownership, and exact restoration."""
import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.config import load_run_config
from src.models.ca_gat_mappo import stack_actor_time
from src.models.qmix_continuation import load_continuation_checkpoint, state_digest
from src.models.qmix_recurrent import reconstruct_replay_sequences
from src.models.qmix_replay import SequenceReplay
from src.models.qmix_trainer import QMIXTrainer, checkpoint_bytes, load_checkpoint


@pytest.fixture
def trainer(tmp_path):
    previous=torch.get_num_threads();torch.set_num_threads(1)
    config=load_run_config(Path(__file__).resolve().parents[1]/
        'tests/fixtures/configs/qmix_legacy/fa-gat-qmix-seed42-8k-episode-return-development.json',
        cli_overrides={'training.mappo.training_device':'cpu','environment.episode_horizon':12,
            'training.qmix.max_training_episodes':2,'training.qmix.max_training_environment_steps':24,
            'training.qmix.checkpoint_interval_steps':12,'training.qmix.recurrent_sequence_length_slots':4,
            'training.qmix.sequence_batch_size':2,'output.logs_dir':str(tmp_path/'l'),
            'output.plots_dir':str(tmp_path/'p'),'output.dashboard_logs_dir':str(tmp_path/'d')})
    result=QMIXTrainer(config);result.train()
    yield result
    torch.set_num_threads(previous)


def test_context_matches_current_weight_full_prefix_and_reuses_batch_work(trainer):
    first,second,third=trainer.replay.sequences[:3]
    system,target=trainer.learner.online,trainer.learner.target
    with torch.no_grad():
        system.agent.gru.bias_ih_l0.add_(.13)
        target.agent.gru.bias_ih_l0.sub_(.21)
    before=state_digest(system.state_dict()); rng_before=torch.random.get_rng_state().clone()
    actual,slots=reconstruct_replay_sequences(system,target,(third,second),trainer.replay)
    with torch.no_grad():
        prefix=stack_actor_time((first.actor,second.actor))
        _,expected=system.agent.encode(prefix,system.agent.initial_hidden())
        _,expected_target=target.agent.encode(prefix,target.agent.initial_hidden())
    assert slots==8
    assert actual[0].first_slot==8 and actual[1].first_slot==4
    assert torch.allclose(actual[0].hidden,expected,atol=1e-6)
    assert torch.allclose(actual[0].target_hidden,expected_target,atol=1e-6)
    assert not actual[0].hidden.requires_grad and not actual[0].target_hidden.requires_grad
    assert not torch.allclose(actual[0].hidden,actual[0].target_hidden)
    assert state_digest(system.state_dict())==before
    assert torch.equal(torch.random.get_rng_state(),rng_before)


def test_update_is_invariant_to_obsolete_saved_hidden(trainer):
    left=copy.deepcopy(trainer.learner);right=copy.deepcopy(trainer.learner)
    sequence=trainer.replay.sequences[1]
    corrupted=replace(sequence,hidden=torch.full_like(sequence.hidden,999.),
                      target_hidden=torch.full_like(sequence.target_hidden,-999.))
    a=left.update((sequence,),replay=trainer.replay)
    b=right.update((corrupted,),replay=trainer.replay)
    assert a==b
    assert state_digest(left.online.state_dict())==state_digest(right.online.state_dict())
    assert all(p.grad is None for p in left.target.parameters())


def test_evicted_prefix_survives_serialization_then_is_purged(trainer):
    first,second,_,new_episode=trainer.replay.sequences[:4]
    replay=SequenceReplay(1,42);replay.add(first);replay.add(second)
    assert len(replay.prefix_cache)==1
    state=replay.to_state();restored=SequenceReplay.from_state(state)
    assert state_digest(replay.to_state())==state_digest(restored.to_state())
    state['prefix_cache'][0]['actor']['self_features'].add_(100.)
    assert not torch.equal(state['prefix_cache'][0]['actor']['self_features'],restored.prefix_actors(second)[0].self_features)
    a,_=reconstruct_replay_sequences(trainer.learner.online,trainer.learner.target,(second,),replay)
    b,_=reconstruct_replay_sequences(trainer.learner.online,trainer.learner.target,(second,),restored)
    assert torch.equal(a[0].hidden,b[0].hidden)
    restored.add(new_episode)
    assert restored.prefix_cache=={}


def test_missing_prefix_is_rejected_without_optimizer_update(trainer):
    sequence=trainer.replay.sequences[1]
    before=state_digest(trainer.learner.online.state_dict());updates=trainer.learner.updates
    with pytest.raises(ValueError,match='complete episode prefixes'):
        trainer.learner.update((sequence,))
    broken=SequenceReplay(1,42);broken.add(sequence)
    with pytest.raises(ValueError,match='incomplete'):
        trainer.learner.update((sequence,),replay=broken)
    assert state_digest(trainer.learner.online.state_dict())==before
    assert trainer.learner.updates==updates


def test_noninitial_chunk_cannot_reset_hidden(trainer):
    sequence=trainer.replay.sequences[1]
    batch=replace(sequence.actor,episode_starts=torch.ones_like(sequence.actor.episode_starts))
    with pytest.raises(ValueError,match='reset'):
        reconstruct_replay_sequences(trainer.learner.online,trainer.learner.target,
            (replace(sequence,actor=batch),),trainer.replay)


def test_target_copy_uses_current_target_prefix_without_saved_target_state(trainer):
    sequence=trainer.replay.sequences[2]
    trainer.learner.target.load_state_dict(trainer.learner.online.state_dict())
    result,_=reconstruct_replay_sequences(trainer.learner.online,trainer.learner.target,(sequence,),trainer.replay)
    assert torch.equal(result[0].hidden,result[0].target_hidden)


@pytest.mark.parametrize('loader',[load_checkpoint,load_continuation_checkpoint])
def test_missing_context_identity_rejected(trainer,tmp_path,loader):
    payload=trainer.payload('final');payload['implementation_contract'].pop('recurrent_context')
    checkpoint=tmp_path/'bad.pt';checkpoint.write_bytes(checkpoint_bytes(payload))
    with pytest.raises(ValueError,match='recurrent context'):
        loader(checkpoint)


def test_full_state_resume_with_evicted_prefix_matches_uninterrupted(trainer,tmp_path):
    config=replace(trainer.config,training=replace(trainer.config.training,
        qmix=replace(trainer.config.training.qmix,replay_capacity_sequences=1,sequence_batch_size=1)))
    uninterrupted=QMIXTrainer(config);uninterrupted.train()
    first=QMIXTrainer(config);first.train(stop_after_episodes=1)
    assert len(first.replay.prefix_cache)==2
    checkpoint=tmp_path/'periodic.pt';checkpoint.write_bytes(checkpoint_bytes(first.payload('periodic')))
    resumed=QMIXTrainer(config);resumed.restore(checkpoint);resumed.train()
    assert state_digest(resumed.payload('final'))==state_digest(uninterrupted.payload('final'))
