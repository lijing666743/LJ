"""Step06 real engineering fixtures, never development experiment scores."""
import hashlib
import json
import unittest
from dataclasses import asdict, fields, replace
from pathlib import Path

import pytest
import torch

from src.artifacts import ArtifactConflictError, _filesystem_path
from src.cli import build_arg_parser, build_run_config_from_args
from src.config import load_run_config
from src.env.actions import ActionProposal
from src.env.environment import U2UMECEnvironment
from src.evaluation.actor_loader import EvaluationCheckpointError, load_final_actor_for_evaluation
from src.evaluation.runner import FormalEvaluationRunner
from src.models.ca_gat_mappo import ACTION_BRANCH_ORDER
from src.models.ca_gat_mappo_checkpoint import load_checkpoint_payload
from src.models.ca_gat_mappo_trainer import (
    CAGATMAPPOTrainer, _PPO_EPOCH_DIAGNOSTIC_FIELDS, _update_checkpoint_state,
)
from src.training_artifacts import write_cagat_mappo_training_artifacts
from tests.test_ca_gat_mappo_checkpoint import assert_nested_equal
from tests.test_route_candidate_interface import P1A


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / 'experiments/task-allocation/configs'


@pytest.fixture(scope='module', autouse=True)
def single_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def digest(path):
    return hashlib.sha256(_filesystem_path(Path(path)).read_bytes()).hexdigest()


def preserve_provenance(config, source):
    # dataclasses.replace resets init=False metadata. Carry the actual loaded
    # repository provenance; never invent a commit to bypass evaluation checks.
    for name in ['git_branch','git_commit','git_dirty']:
        object.__setattr__(config,name,getattr(source,name))
    return config


@pytest.mark.parametrize('seed', [42,43,44])
def test_released_configs_only_change_declared_method_seed_and_output(seed):
    reference = load_run_config(P1A)
    expected = reference.resolved_dict()
    paths = []
    for group in 'ABC':
        path = CONFIG_ROOT / f'{group.lower()}-seed{seed}-8k.json'
        cfg = load_run_config(path); cfg.validate()
        via_cli = build_run_config_from_args(build_arg_parser().parse_args(['--config',str(path)]))
        assert cfg.config_hash == via_cli.config_hash
        assert cfg.environment == reference.environment and cfg.reward == reference.reward
        assert cfg.training.mappo.max_environment_transitions == 8000
        assert cfg.training.mappo.max_training_episodes == 16
        assert cfg.training.mappo.checkpoint_interval_steps == 4000
        actual = cfg.resolved_dict()
        actual['seed'] = expected['seed']; actual['output'] = expected['output']
        if group in 'BC': actual['training']['mappo']['route_decoder_mode'] = expected['training']['mappo']['route_decoder_mode']
        if group == 'C': actual['training']['mappo'].pop('route_feature_mode')
        assert actual == expected
        assert cfg.training.mappo.destination_supervision_mode == 'legacy'
        assert cfg.training.mappo.destination_oracle_rank_coefficient == 0
        paths.extend(cfg.artifact_paths().values())
    assert len(paths) == len(set(paths))


def choose_service_actions(environment, case):
    proposals = []
    for obs in environment.current_observations:
        values = {}
        for branch in ACTION_BRANCH_ORDER:
            domain = obs.action_masks.domain_for(branch)
            mask = obs.action_masks.mask_for(branch, values)
            legal = [value for value,ok in zip(domain,mask) if ok]
            if branch == 'route':
                if obs.uav_id == 0 and obs.private_queues.unbound.head_task_id == 0:
                    preferred = 'local' if case == 'local_done' else 1 if case == 'remote_done' else 'defer'
                else: preferred = 'defer'
            elif branch == 'tx_select': preferred = 1 if case == 'remote_done' and obs.uav_id == 0 else 'idle'
            elif branch == 'cpu_queue': preferred = 0 if obs.uav_id == (1 if case == 'remote_done' else 0) else 'idle'
            elif branch == 'resource_group': preferred = next((v for v in legal if v != 'idle'), 'idle')
            elif branch in {'resource_width','power_level','cpu_frequency'}: preferred = max(legal)
            values[branch] = preferred if preferred in legal else legal[0]
        proposal = ActionProposal(obs.uav_id, **values)
        assert obs.action_masks.is_legal(proposal)
        proposals.append(proposal)
    return proposals


@pytest.mark.parametrize('case,outcome', [
    ('local_done','done'),('remote_done','done'),('expired','expired'),('truncated','truncated')])
def test_real_environment_local_remote_and_terminal_lifecycle(case,outcome,tmp_path):
    base = load_run_config(P1A)
    env = replace(base.environment, episode_horizon=8, arrival_probabilities=(1.,0.,0.,0.),
        task_data_bits_min=1., task_data_bits_max=1., task_cycles_per_bit_min=1.,task_cycles_per_bit_max=1.,
        minimum_task_slack_slots=2 if case=='expired' else 10,
        maximum_task_slack_slots=2 if case=='expired' else 10,
        candidate_neighbor_radius_m=2000., building_layout=(), velocity_std_mps=(0.,0.),
        shadowing_std_db=0.,csi_error_std_db=0.)
    cfg = replace(base,environment=env);cfg.validate()
    environment = U2UMECEnvironment(cfg);environment.reset();events=[]
    for slot in range(8):
        result=environment.step(choose_service_actions(environment,case))
        events.append(result.info)
    target=environment.lifecycle.get_task(0)
    assert target.outcome.value == outcome
    metrics=environment.metrics.snapshot()
    assert metrics['generated_task_count']==metrics['completed_task_count']+metrics['expired_task_count']+metrics['truncated_task_count']
    assert metrics['conservation']['is_conserved']
    if outcome=='done':
        expected_executor=0 if case=='local_done' else 1
        assert target.destination==expected_executor
        cpu=[(event['slot'],row) for event in events for row in event['service']['cpu'] if row['task_id']==0]
        assert len(cpu)==1 and cpu[0][1]['executor_uav']==expected_executor
        assert cpu[0][1]['service_cycles']==target.cpu_cycles
        assert cpu[0][0]==target.completion_slot and target.binding_slot<target.completion_slot
        if case=='local_done':
            assert metrics['actual_served_bits']==0 and metrics['total_tx_energy_j']==0
            assert metrics['local_binding_cleared_bits']==target.data_bits
        else:
            assert metrics['actual_served_bits']==target.data_bits and metrics['total_tx_energy_j']>0
            assert target.binding_slot<target.cpu_entry_slot<target.completion_slot
    (tmp_path/'lifecycle_evidence.json').write_text(json.dumps({
        'kind':'ENGINEERING_FIXTURE','case':case,'target':target.snapshot(),'metrics':metrics,
        'step_info':events},ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


@pytest.mark.parametrize('group', ['A','B','C'])
def test_real_trainer_resume_and_evaluation_artifacts(group,tmp_path,monkeypatch):
    """Real PPO, two updates, partial-rollout resume and immutable evaluation.

    Only fixture horizon/budget/device/output differ from the released config.
    Reward, service, model dimensions and loss/shadow coefficients stay fixed.
    """
    base=load_run_config(CONFIG_ROOT/f'{group.lower()}-seed42-8k.json')
    env=replace(base.environment,episode_horizon=20)
    model=replace(base.training.mappo,training_device='cpu',max_training_episodes=26,
        max_training_environment_steps=520,checkpoint_interval_steps=260,evaluation_interval_steps=520)
    fixture_id=f'{tmp_path.parent.name}/{group}'
    output=replace(base.output,logs_dir=str(tmp_path/'resume/logs'),
        plots_dir=str(ROOT/'plots/development/task_allocation/step06_engineering_gate'/fixture_id),
        dashboard_logs_dir=str(ROOT/'dashboard_logs/development/task_allocation/step06_engineering_gate'/fixture_id))
    cfg=replace(base,environment=env,training=replace(base.training,mappo=model),output=output)
    preserve_provenance(cfg,base)
    cfg.validate()
    control_cfg=replace(cfg,output=replace(output,logs_dir=str(tmp_path/'control/logs'),
        plots_dir=str(tmp_path/'control/plots'),dashboard_logs_dir=str(tmp_path/'control/dashboard_logs')))
    preserve_provenance(control_cfg,base)
    from src.models import tx_paired_shadow
    original_run_pair=tx_paired_shadow.run_pair
    shadow_cost={'control':0,'resumed_path':0}
    phase='control'
    def count_real_shadows(*args,**kwargs):
        result=original_run_pair(*args,**kwargs)
        shadow_cost[phase]+=result['shadow_steps']
        return result
    monkeypatch.setattr(tx_paired_shadow,'run_pair',count_real_shadows)
    uninterrupted=CAGATMAPPOTrainer(control_cfg);expected=uninterrupted.train()
    phase='resumed_path'
    paused=CAGATMAPPOTrainer(cfg);periodic_path=paused.train_until_periodic_checkpoint()
    periodic=load_checkpoint_payload(periodic_path)
    assert periodic['active_rollout_state']['rollout_length']==4
    assert periodic['training_state']['ppo_update_count']==1
    assert periodic['optimizer_state']['actor_adam']['state'] and periodic['optimizer_state']['critic_adam']['state']
    resumed=CAGATMAPPOTrainer.resume_from_checkpoint(cfg,periodic_path)
    actual=resumed.train_with_checkpoints()
    # V1 intentionally restores only nine epoch scalars for completed updates.
    # Check every persisted field and all learning state exactly; expose missing
    # optional diagnostics rather than pretending the full summaries round-trip.
    assert replace(actual,updates=expected.updates)==expected
    assert len(actual.updates)==len(expected.updates)
    for left,right in zip(actual.updates,expected.updates):
        assert _update_checkpoint_state(left)==_update_checkpoint_state(right)
    omitted=[]
    for left,right in zip(actual.updates[0].output.epoch_diagnostics,expected.updates[0].output.epoch_diagnostics):
        for field in fields(left):
            if field.name not in _PPO_EPOCH_DIAGNOSTIC_FIELDS:
                assert getattr(left,field.name) is None
                if getattr(right,field.name) is not None:omitted.append(field.name)
    assert omitted
    assert actual.updates[1]==expected.updates[1]
    assert (actual.total_environment_transitions,actual.optimized_transitions,actual.unused_final_tail_transitions)==(520,512,8)
    assert actual.ppo_update_count==2 and actual.completed_episode_count==26
    for left,right in [(uninterrupted.actor.state_dict(),resumed.actor.state_dict()),
                       (uninterrupted.critic.state_dict(),resumed.critic.state_dict()),
                       (uninterrupted.updater.optimizers.actor_optimizer.state_dict(),resumed.updater.optimizers.actor_optimizer.state_dict()),
                       (uninterrupted.updater.optimizers.critic_optimizer.state_dict(),resumed.updater.optimizers.critic_optimizer.state_dict())]:
        assert_nested_equal(unittest.TestCase(),left,right)
    assert torch.equal(uninterrupted.policy_generator.get_state(),resumed.policy_generator.get_state())
    assert shadow_cost['control']==shadow_cost['resumed_path']
    training_artifacts=write_cagat_mappo_training_artifacts(cfg,actual)
    assert training_artifacts.smoke_gate_status=='pass'
    assert training_artifacts.signal_gate_status=='insufficient-horizon'
    assert all(Path(p).is_file() for p in training_artifacts.artifacts)
    assert Path(cfg.artifact_paths()['dashboard_png']).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    training_hashes={str(p):digest(p) for p in training_artifacts.artifacts}
    with pytest.raises(ArtifactConflictError):write_cagat_mappo_training_artifacts(cfg,actual)
    assert training_hashes=={str(p):digest(p) for p in training_artifacts.artifacts}
    final_path=Path(cfg.artifact_paths()['final_checkpoint']);final=load_checkpoint_payload(final_path)
    assert final['training_state']['training_complete'] and final['active_rollout_state']['rollout_length']==0
    evaluation=replace(cfg,mode='evaluation',launch_profile=None,
        evaluation=replace(cfg.evaluation,evaluation_seeds=(1042,)),
        output=replace(cfg.output,logs_dir=str(tmp_path/'evaluation/logs')))
    preserve_provenance(evaluation,base)
    before=digest(final_path)
    evaluator=FormalEvaluationRunner(evaluation,final_path,evaluation_device='cpu',collect_service_telemetry=True)
    result=evaluator.run(write_artifacts=True)
    repeat=FormalEvaluationRunner(evaluation,final_path,evaluation_device='cpu',collect_service_telemetry=True).run(write_artifacts=False)
    assert digest(final_path)==before
    assert result.episodes==repeat.episodes
    assert len(result.episodes)==4
    assert len({episode.external_trace_sha256 for episode in result.episodes})==1
    for episode in result.episodes:
        m=episode.metrics
        assert m['generated']==m['completed']+m['expired']+m['truncated']
    artifacts=[*result.artifacts,*result.diagnostic_artifacts,*result.service_diagnostic_artifacts]
    assert len(artifacts)==10 and all(_filesystem_path(Path(p)).is_file() for p in artifacts)
    hashes={str(p):digest(p) for p in artifacts}
    with pytest.raises(ArtifactConflictError):evaluator.run(write_artifacts=True)
    assert hashes=={str(p):digest(p) for p in artifacts}
    architecture=result.manifest['actor_architecture_identity']['network_config']
    if group=='C':assert architecture['route_feature_mode']=='physical_relations_v1'
    else:assert 'route_feature_mode' not in architecture
    if group in 'BC':
        opposite='original_option_features_v1' if group=='C' else 'physical_relations_v1'
        incompatible=replace(evaluation,training=replace(evaluation.training,mappo=replace(
            evaluation.training.mappo,route_feature_mode=opposite)))
        with pytest.raises(EvaluationCheckpointError,match='route_feature_mode'):
            load_final_actor_for_evaluation(incompatible,final_path,evaluation_device='cpu')
        assert result.route_decision_trace
        assert all(row['feature_mode']==cfg.training.mappo.route_feature_mode for row in result.route_decision_trace)
    evidence={'kind':'ENGINEERING_FIXTURE_NOT_EXPERIMENT','group':group,'source_config':str(CONFIG_ROOT/f'{group.lower()}-seed42-8k.json'),
        'fixture_config_hash':cfg.config_hash,'fixture_overrides':{'horizon':20,'budget':520,'episodes':26,'interval':260,'device':'cpu'},
        'fresh_initialization':True,'pretraining':False,'collected_control':520,'collected_resumed_path':520,
        'optimized_per_path':512,'unused_tail_per_path':8,'periodic_pending':4,'updates_per_path':2,
        'model_optimizer_rng_exact':True,'evaluation_episodes':4,'evaluation_steps':80,
        'repeat_evaluation_steps':80,'evaluation_repeat_exact':True,'task_conservation':True,
        'periodic_path':str(periodic_path),'final_path':str(final_path),'final_sha256':before,
        'evaluation_artifact_sha256':hashes,'training_artifact_sha256':training_hashes,
        'shadow_environment_steps':shadow_cost,'restored_optional_diagnostics_unavailable':sorted(set(omitted)),
        'result':asdict(actual)}
    evidence_path=tmp_path/'engineering_evidence.json'
    evidence_path.write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('ENGINEERING_FIXTURE_EVIDENCE',evidence_path,flush=True)
