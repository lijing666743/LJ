"""Evidence checks for passive behavior tracing and backward-only replay probes."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import load_run_config
from src.env.tasks import Task
from src.evaluation.qmix_limited_diagnostic import (
    DecisionProbe, TracedQMIXRunner, assert_saved_episode, describe,
    gradient_probe, hidden_probe, terminal_stage,
    group_swap_episode, branch_gradient_comparison,
)
from src.evaluation.qmix_runner import QMIXEvaluationRunner
from src.models.qmix_continuation import state_digest
from src.models.qmix_trainer import QMIXTrainer, checkpoint_bytes


@pytest.fixture
def trained(tmp_path):
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    config=load_run_config(Path(__file__).resolve().parents[1]/
        'tests/fixtures/configs/qmix_legacy/fa-gat-qmix-seed42-8k-episode-return-development.json',
        cli_overrides={'training.mappo.training_device':'cpu','environment.episode_horizon':8,
            'training.qmix.max_training_episodes':2,'training.qmix.max_training_environment_steps':16,
            'training.qmix.checkpoint_interval_steps':8,'training.qmix.sequence_batch_size':2,
            'training.qmix.recurrent_sequence_length_slots':4,
            'output.logs_dir':str(tmp_path/'logs'),'output.plots_dir':str(tmp_path/'plots'),
            'output.dashboard_logs_dir':str(tmp_path/'dash')})
    trainer=QMIXTrainer(config); trainer.train(write_checkpoints=False)
    checkpoint=tmp_path/'final.pt'; checkpoint.write_bytes(checkpoint_bytes(trainer.payload('final')))
    yield config,trainer,checkpoint
    torch.set_num_threads(previous)


def test_frozen_diagnostic_exactly_matches_original_evaluation(trained):
    config,_,checkpoint=trained
    evaluation=replace(config,mode='evaluation',training=replace(config.training,formal_rl_enabled=False))
    original=QMIXEvaluationRunner(evaluation,checkpoint,device='cpu')
    expected=original.episode(config.method_id,1042)
    traced=TracedQMIXRunner(evaluation,checkpoint,device='cpu')
    before=state_digest(traced.agent.state_dict())
    episode,trace=traced.traced_episode(1042)
    assert_saved_episode(episode,expected.artifact_record('fixture'))
    assert trace['metrics']==expected.metrics
    assert sum(trace['expired_by_stage'].values())==expected.metrics['expired']
    assert state_digest(traced.agent.state_dict())==before
    assert all(p.grad is None for p in traced.agent.parameters())


def test_backward_probe_changes_no_parameters_or_rng(trained):
    config,trainer,_=trained
    system,target=trainer.learner.online,trainer.learner.target
    before=state_digest(system.state_dict()); target_before=state_digest(target.state_dict())
    random_before=torch.random.get_rng_state().clone()
    sequence=trainer.replay.sequences[-1]
    row=gradient_probe(system,target,sequence,config.training.qmix.gamma)
    assert row['length']==4 and row['loss']>=0
    assert row['mixer_sensitivity']['min']>=0
    assert row['active_branch_counts'].keys()==system.agent.heads.keys()
    assert 0<=row['saved_vs_greedy_agent_joint_match_fraction']<=1
    assert state_digest(system.state_dict())==before
    assert state_digest(target.state_dict())==target_before
    assert torch.equal(torch.random.get_rng_state(),random_before)
    assert all(p.grad is None for p in target.parameters())
    assert trainer.learner.updates==3


def test_hidden_probe_uses_actual_prefix_and_updates_nothing(trained):
    config,trainer,_=trained
    first,second=trainer.replay.sequences[:2]
    result=hidden_probe(trainer.learner.online,trainer.learner.target,second,[first],config.training.qmix.gamma)
    assert result['online_hidden_rms_change']>=0
    assert 0<=result['greedy_agent_joint_disagreement_fraction']<=1
    assert trainer.learner.updates==3


@pytest.mark.parametrize('destination,bits,expected',[(None,10.,'unbound'),(0,0.,'local'),(1,10.,'tx'),(1,0.,'cpu')])
def test_expiration_stage_survives_terminal_status(destination,bits,expected):
    task=Task.create(task_id=0,source_uav=0,data_bits=10.,cpu_cycles=10.,arrival_slot=0,deadline_slot=5)
    task.destination=destination; task.remaining_bits=bits
    task.expire(5)
    assert terminal_stage(task)==expected


def test_saved_metric_tampering_is_rejected(trained):
    config,_,checkpoint=trained
    evaluation=replace(config,mode='evaluation',training=replace(config.training,formal_rl_enabled=False))
    runner=QMIXEvaluationRunner(evaluation,checkpoint)
    result=runner.episode(config.method_id,1042)
    saved=result.artifact_record('fixture'); saved['metrics']['completed']+=1
    with pytest.raises(ValueError,match='metrics'):
        assert_saved_episode(result,saved)


def test_describe_keeps_empty_values_undefined_and_rejects_nonfinite():
    assert describe([])=={'count':0,'mean':None,'min':None,'max':None}
    with pytest.raises(ValueError,match='non-finite'):
        describe([np.nan])


@pytest.mark.parametrize('groups',[(),('route',),('cpu',),('route','cpu')])
def test_equal_donor_group_swaps_reproduce_original_and_remain_frozen(trained,groups):
    config,_,checkpoint=trained
    evaluation=replace(config,mode='evaluation',training=replace(config.training,formal_rl_enabled=False))
    base=QMIXEvaluationRunner(evaluation,checkpoint)
    donor=QMIXEvaluationRunner(evaluation,checkpoint)
    expected=base.episode(config.method_id,1042)
    actual=group_swap_episode(base,donor,1042,groups)
    assert_saved_episode(actual,expected.artifact_record('fixture'))
    assert all(p.grad is None for p in base.agent.parameters())


def test_reconstructed_zero_prefix_has_same_gradient_direction(trained):
    config,trainer,_=trained
    seq=trainer.replay.sequences[0]
    before=state_digest(trainer.learner.online.state_dict())
    result=branch_gradient_comparison(trainer.learner.online,trainer.learner.target,seq,[],config.training.qmix.gamma)
    for row in result.values():
        if row['old_norm']>0:
            assert row['cosine']==pytest.approx(1.,abs=1e-5)
    assert state_digest(trainer.learner.online.state_dict())==before


def test_v6_backward_probe_ignores_poisoned_collection_hidden(trained):
    config,trainer,_=trained
    sequence=trainer.replay.sequences[-1]
    poisoned=replace(sequence,hidden=torch.full_like(sequence.hidden,100.),
                     target_hidden=torch.full_like(sequence.target_hidden,-100.))
    first=gradient_probe(trainer.learner.online,trainer.learner.target,sequence,
                         config.training.qmix.gamma,replay=trainer.replay)
    second=gradient_probe(trainer.learner.online,trainer.learner.target,poisoned,
                          config.training.qmix.gamma,replay=trainer.replay)
    assert first==second
    assert first['recurrent_context_slots']==sequence.first_slot>0
    assert first['recurrent_context']=='episode_prefix_current_weights_no_grad_v1'


def test_target_lag_probe_is_zero_for_synchronized_systems(trained):
    import copy
    config,trainer,_=trained
    synchronized=copy.deepcopy(trainer.learner.online).requires_grad_(False).eval()
    result=gradient_probe(trainer.learner.online,synchronized,trainer.replay.sequences[-1],
                          config.training.qmix.gamma,replay=trainer.replay)
    assert result['target_lag_mean_abs_td_target_change']==0.
    assert all(row['different']==0 for row in result['online_target_active_branch_disagreements'].values())
