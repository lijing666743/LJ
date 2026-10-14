"""One authorized frozen-policy diagnostic batch, then static E15 calibration.

No trainer/optimizer or policy update is constructed. A single precommitted
256-slot factual batch uses training seed42, never eval1044. Every Remote first
binding in that batch receives all-legal, full-500-slot shadow oracle labels.
The batch and model snapshot are saved BEFORE any gradient measurement.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch

from ..config import load_run_config
from ..env.environment import U2UMECEnvironment
from ..models.ca_gat_mappo import (ActorObservationTensorizer,CentralizedStateTensorizer,
    CAGATMAPPOActor,MAPPOCentralizedCritic)
from ..models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution,SequentialActionMaskBatch
from ..models.ca_gat_mappo_checkpoint import serialize_rollout_transition,restore_rollout_transition
from ..models.ca_gat_mappo_rollout import CAGATMAPPORolloutBuffer,CAGATMAPPORolloutChunk
from ..models.ca_gat_mappo_gae import compute_rollout_gae
from ..models.ca_gat_mappo_trainer import build_training_episode_config
from ..models.route_task_credit import KEY as TASK_KEY,capture_task_values,make_step_metadata,prepare_task_credit
from ..models.destination_ranking import (KEY,ranking_batch_loss,destination_ppo_contribution,
    destination_gradient_probe,PROTOCOL)
from .destination_system_oracle import collect_destination_labels
from .route_oracle import FrozenOraclePolicyRunnerCache,actor_state_digest


def collect_once(config_path,output,*,prefix_path=None,slots=256):
    cfg=load_run_config(config_path,cli_overrides={'training.mappo.training_device':'cpu'})
    if cfg.seed!=42 or cfg.environment.episode_horizon!=500 or cfg.scenario_id!='remote_stress_v2':
        raise ValueError('calibration protocol requires training seed42/remote_stress_v2/500 slots')
    actor=CAGATMAPPOActor(cfg);actor.eval();critic=MAPPOCentralizedCritic(cfg).eval()
    before=actor_state_digest(actor)
    distribution=CAGATMAPPOActionDistribution(actor,cfg)
    generator=torch.Generator(device='cpu');generator.manual_seed(distribution.policy_seed)
    adapter=ActorObservationTensorizer(cfg);central=CentralizedStateTensorizer(cfg)
    episode_config=build_training_episode_config(cfg,0)
    env=U2UMECEnvironment(episode_config);env.reset();hidden=actor.initial_hidden(1)
    runner=FrozenOraclePolicyRunnerCache(cfg).get(actor,0,source='E15_initial_frozen_calibration_no_optimizer')
    buffer=CAGATMAPPORolloutBuffer(cfg)
    prefix = torch.load(prefix_path,map_location='cpu',weights_only=False) if prefix_path else None
    if prefix and (prefix['config_snapshot']!=cfg.snapshot_dict() or prefix['actor_state_digest']!=before
                   or prefix.get('environment_seed')!=episode_config.seed):
        raise ValueError('cannot change seed, policy or config when extending the same frozen batch')
    started=time.perf_counter()
    for slot in range(slots):
        obs=env.current_observations;batch=adapter.encode_step(obs,episode_start=slot==0)
        masks=SequentialActionMaskBatch.from_observations(obs)
        state=central.encode_step(env.current_state)
        with torch.no_grad():
            policy=distribution.sample_actions(batch,masks,hidden,generator=generator)
            old_value=critic(state)[0,0]
        task_before=capture_task_values(env,state.features,critic,cfg)
        proposals=policy.proposals[0][0]
        saved_transition = (restore_rollout_transition(cfg,prefix['transitions'][slot])
                            if prefix and slot<len(prefix['transitions']) else None)
        if saved_transition is not None:
            labels=saved_transition.rejection_or_downgrade_summary[KEY]
            for row in labels:
                if env.state_fingerprint()!=row['provenance']['root_state']:
                    raise ValueError('saved factual prefix root does not reproduce exactly')
        else:
            labels=collect_destination_labels(env,obs,proposals,policy.hidden_out,runner,
                episode_id=0,environment_step=slot)
        step=env.step(proposals)
        next_state=central.encode_step(env.current_state)
        with torch.no_grad():bootstrap=critic(next_state)[0,0]
        task_after=capture_task_values(env,next_state.features,critic,cfg)
        metadata={TASK_KEY:make_step_metadata(before=task_before,after=task_after,info=step.info,
            policy=policy,episode_id=0,environment_step=slot,policy_version=0,capture_route_leaves=True),
            KEY:labels,'rejection':step.info.get('rejection',()),
            'downgrade':step.info.get('downgrade',()),'canonicalization':step.info.get('canonicalization',())}
        transition=buffer.append_step(slot=slot,actor_batch=batch,action_mask_batch=masks,action_output=policy,
            hidden_in=hidden,centralized_state=state,old_value=old_value,
            reward=torch.tensor(step.info['agent_reward'],dtype=torch.float32),
            terminated=False,truncated=False,episode_boundary=False,bootstrap_allowed=True,
            bootstrap_value=bootstrap,executed_action_summary=step.info['executed'],
            rejection_or_downgrade_summary=metadata)
        if saved_transition is not None:
            if (transition.proposal_actions!=saved_transition.proposal_actions or
                    any(not torch.equal(getattr(transition,k),getattr(saved_transition,k))
                        for k in ('action_indices','old_joint_log_prob','old_branch_log_probs',
                                  'hidden_in','centralized_state','old_value','reward','bootstrap_value'))):
                raise ValueError('same factual calibration prefix changed')
        hidden=policy.hidden_out.detach()
        print(f'calibration factual slot {slot+1}/{slots}; oracle labels={len(labels)}; shadow steps={sum(r["shadow_transitions"] for r in labels)}',flush=True)
    if actor_state_digest(actor)!=before:raise ValueError('frozen calibration changed actor')
    saved={'schema':'e15_frozen_diagnostic_batch.v1','not_training_checkpoint':True,
        'optimizer_created':False,'training_started':False,'training_seed':42,
        'episode_index':0,'environment_seed':episode_config.seed,
        'environment_seed_protocol':'production build_training_episode_config(master_seed=42, episode_index=0)',
        'factual_slots':slots,'actor_state_digest':before,'protocol':PROTOCOL,
        'reused_verified_prefix_slots':0 if prefix is None else len(prefix['transitions']),
        'config_path':str(Path(config_path).resolve()),'config_snapshot':cfg.snapshot_dict(),
        'actor':actor.state_dict(),'critic':critic.state_dict(),
        'transitions':[serialize_rollout_transition(buffer.transition_at(i)) for i in range(len(buffer))],
        'wall_seconds':time.perf_counter()-started}
    torch.save(saved,output)
    return saved


def static_calibration(config_path,batch_path):
    saved=torch.load(batch_path,map_location='cpu',weights_only=False)
    cfg=load_run_config(config_path,cli_overrides={'training.mappo.training_device':'cpu'})
    expected_seed=build_training_episode_config(cfg,0).seed
    if (saved.get('schema')!='e15_frozen_diagnostic_batch.v1'
            or saved.get('environment_seed')!=expected_seed or saved.get('episode_index')!=0
            or saved.get('training_seed')!=cfg.seed or saved.get('optimizer_created') is not False
            or saved.get('training_started') is not False):
        raise ValueError('calibration requires a frozen batch from the production episode-zero seed protocol')
    # The only permitted post-calibration edit is the resulting coefficient.
    # Source/override/run-id metadata is derived, not an architecture input.
    current,captured=copy.deepcopy(cfg.snapshot_dict()),copy.deepcopy(saved['config_snapshot'])
    for identity in (current,captured):
        identity.pop('_metadata',None)
        identity['training']['mappo'].pop('destination_oracle_rank_coefficient',None)
    if current!=captured:raise ValueError('saved calibration config differs beyond rank coefficient')
    actor=CAGATMAPPOActor(cfg);actor.load_state_dict(saved['actor'],strict=True)
    critic=MAPPOCentralizedCritic(cfg);critic.load_state_dict(saved['critic'],strict=True)
    buffer=CAGATMAPPORolloutBuffer(cfg)
    for row in saved['transitions']:buffer.append(restore_rollout_transition(cfg,row))
    chunk=CAGATMAPPORolloutChunk(tuple(buffer._transitions))
    shared=compute_rollout_gae(chunk,cfg)
    task=prepare_task_credit(buffer,shared.advantage,cfg)
    policy=CAGATMAPPOActionDistribution(actor,cfg).evaluate_actions(
        chunk.actor_batch,chunk.action_mask_batch,chunk.proposals,chunk.initial_hidden)
    flags=torch.tensor([r.rejection_or_downgrade_summary['policy_components_v1']['active'] for r in buffer._transitions])
    divisor=int((flags.sum((0,1))>0).sum())
    ppo=destination_ppo_contribution(actor,policy,chunk.old_branch_log_probs[...,0].unsqueeze(0),
        task.advantage.unsqueeze(0),policy.active_branches['route'],divisor,cfg.training.mappo.ppo_clip_epsilon)
    # Exactly the same joint PPO surrogate numbers, for telemetry only.
    route_surrogate=task.advantage.unsqueeze(0)
    rank,records=ranking_batch_loss(actor,policy,buffer,0,route_surrogate,
        policy.active_branches['route'],divisor,0.)
    first=destination_gradient_probe(actor,ppo,rank,1.,cfg.training.mappo.gradient_clip_norm)
    if not records or min(first['ppo_destination_gradient_norm'],first['rank_gradient_norm'])<=1e-12:
        raise ValueError('fixed calibration batch has no nonzero PPO/rank gradient; do not sweep seeds/batches')
    # One deterministic norm-matching rule, no optimizer steps or sweep.
    coefficient=first['ppo_destination_gradient_norm']/first['rank_gradient_norm']
    measured=destination_gradient_probe(actor,ppo,rank,coefficient,cfg.training.mappo.gradient_clip_norm)
    return {'schema':'e15_static_coefficient_calibration.v1','coefficient':coefficient,
        'coefficient_rule':'norm(PPO-destination)/norm(unweighted oracle CE); one fixed saved batch; target norm ratio=1',
        'batch_path':str(Path(batch_path).resolve()),'training_seed':42,
        'environment_seed':saved['environment_seed'],'eval_data_used':False,
        'factual_transitions':len(buffer),'oracle_labels':len(records),
        'multi_candidate_labels':sum(len(r['legal_destinations'])>1 for r in records),
        'candidate_count':sum(len(r['legal_destinations']) for r in records),
        'shadow_transitions':sum(r['shadow_transitions'] for r in records),
        'shadow_wall_seconds':sum(r['shadow_wall_seconds'] for r in records),
        'rank_loss':float(rank.detach()),'ppo_destination_loss':float(ppo.detach()),
        'probe':measured,'initial_unweighted_probe':first,
        'label_best_set_histogram':dict(Counter(','.join(map(str,r['best_set'])) for r in records)),
        'label_signal':{'informative_multi_candidate_labels':sum(len(r['best_set'])<len(r['legal_destinations']) for r in records),
                        'all_tie_multi_candidate_labels':sum(len(r['legal_destinations'])>1 and len(r['best_set'])==len(r['legal_destinations']) for r in records),
                        'target_outcomes':dict(Counter(b['target_outcome'] for r in records for b in r['branches'])),
                        'warning':'Nonzero CE gradient on an all-tie target is uniformization, not evidence of learned destination superiority.'},
        'credit':{'definition':'unchanged production shared GAE + task-event; real batch frontier bootstrap',
                  'shared_advantage_rms':float(shared.advantage.square().mean().sqrt()),
                  'task_advantage_rms':float(task.task_advantage.square().mean().sqrt())},
        'limitations':['Initial random actor; single diagnostic cohort, not a PPO training update.',
                      'Global clipping reported for destination-only components; production runtime separately probes the complete actor loss.',
                      'Norm matching does not establish useful label alignment or performance gain.'],
        'optimizer_created':False,'training_started':False,'actor_unchanged':actor_state_digest(actor)==saved['actor_state_digest']}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output-dir',required=True)
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--verified-prefix',help='extend the identical saved frozen prefix; no policy/seed changes')
    source.add_argument('--saved-batch',help='static remeasurement only: never construct an environment or resample')
    args=parser.parse_args(argv);torch.set_num_threads(1)
    out=Path(args.output_dir)
    if out.exists():raise FileExistsError('calibration is one-shot; choose no existing artifact directory')
    out.mkdir(parents=True)
    with patch('torch.optim.Adam.__init__',side_effect=AssertionError('optimizer forbidden in frozen calibration')):
        batch=Path(args.saved_batch) if args.saved_batch else out/'frozen_real_batch.pt'
        if not args.saved_batch:
            collect_once(args.config,batch,prefix_path=args.verified_prefix)
        result=static_calibration(args.config,batch)
    (out/'calibration.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
