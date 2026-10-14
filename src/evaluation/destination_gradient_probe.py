"""E15 static gradient inspection on saved factual training tensors only.

Never creates an environment or executes optimizer.step. Refuses coefficient
calibration unless the saved real batch has multi-destination oracle labels.
An optional synthetic structural check is explicitly separate from calibration.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from .route_q_probe import load_saved_tensors
from ..models.ca_gat_mappo import CAGATMAPPOActor
from ..models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution
from ..models.destination_ranking import (KEY, destination_ppo_contribution,
    destination_gradient_probe, listwise_loss)


def probe_saved_batch(config_path,checkpoint,training_log):
    original,payload,legacy,_,chunk,_,_=load_saved_tensors(config_path,checkpoint)
    cfg=replace(original,training=replace(original.training,mappo=replace(original.training.mappo,
        destination_decoder_mode='system_aware_v1',destination_oracle_rank_coefficient=0.)))
    actor=CAGATMAPPOActor(cfg)
    missing,unexpected=actor.load_state_dict(legacy.state_dict(),strict=False)
    if unexpected or any('system_destination_scorer' not in k for k in missing):
        raise ValueError('historical common weights do not match E15')
    policy=CAGATMAPPOActionDistribution(actor,cfg).evaluate_actions(
        chunk.actor_batch,chunk.action_mask_batch,chunk.proposals,chunk.initial_hidden)
    start=payload['training_state']['collected_environment_transitions']-chunk.length
    logged={}
    for line in Path(training_log).open(encoding='utf8'):
        r=json.loads(line)
        if r['config_hash']!=payload['config_hash']:
            raise ValueError('credit log/checkpoint identity mismatch')
        for s in r['samples']:
            if s['epoch_index']==0 and start<=s['global_rollout_index']<start+chunk.length:
                key=(s['global_rollout_index']-start,s['source_uav'])
                if key in logged:raise ValueError('duplicate saved frozen credit')
                logged[key]=s
    active=policy.active_branches['route']
    if len(logged)!=int(active.sum()):raise ValueError('incomplete saved actual credit')
    advantage=torch.zeros_like(policy.branch_log_probs['route'])
    for (t,u),r in logged.items():
        advantage[0,t,u]=r['combined_advantage']
        old=float(chunk.old_branch_log_probs[t,u,0])
        if abs(r['old_log_prob']-old)>1e-6:raise ValueError('stored old probability mismatch')
    components=torch.tensor([t.rejection_or_downgrade_summary['policy_components_v1']['active'] for t in chunk.transitions])
    divisor=int((components.sum((0,1))>0).sum())
    old=chunk.old_branch_log_probs[...,0].unsqueeze(0)
    ppo=destination_ppo_contribution(actor,policy,old,advantage,active,divisor,cfg.training.mappo.ppo_clip_epsilon)
    mask=policy.action_masks['route'][...,3:]
    scores=actor.route_decoder.system_destination_scorer(policy.destination_features)
    counts=Counter(mask.sum(-1).flatten().tolist())
    # A singleton legal set has an exact valid oracle target regardless of
    # outcomes; CE and gradient are identically zero. No invented best action.
    single=mask.sum(-1)==1
    singleton_rank=listwise_loss(scores[single],mask[single],mask[single].float()).mean()
    result=destination_gradient_probe(actor,ppo,singleton_rank,0.,cfg.training.mappo.gradient_clip_norm)
    records=[r for t in chunk.transitions for r in t.rejection_or_downgrade_summary.get(KEY,[])]
    result.update(checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        source_config_hash=payload['config_hash'],scope='actual saved P1a partial training batch; E15 scorer only is freshly initialized',
        transitions=chunk.length,agent_states=chunk.length*actor.spec.uav_count,
        legal_destination_count_histogram=dict(counts),actual_oracle_label_count=len(records),
        real_remote_action_count=int(((policy.action_indices['route']>=3)&active).sum()),
        calibration_status='BLOCKED_NO_MULTI_DESTINATION_REAL_ORACLE_BATCH',
        calibrated_coefficient=None,environment_rollout=False,training_started=False,
        explanation='No nonzero destination PPO/rank gradient exists on singleton legal sets. No coefficient is invented; no eval1044 data is used.')
    # Nonzero wiring check on the very same saved feature tensors, with an
    # explicitly synthetic mask/label. Never used for coefficient calibration.
    synthetic_mask=torch.ones_like(mask)
    target=torch.zeros_like(scores);target[...,0]=1.
    structural=listwise_loss(scores,synthetic_mask,target).mean()
    result['synthetic_structure_only']=destination_gradient_probe(actor,ppo,structural,1.,cfg.training.mappo.gradient_clip_norm)
    result['synthetic_structure_only']['not_real_oracle_supervision']=True
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--training-log',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv);torch.set_num_threads(1)
    path=Path(args.output)
    if path.exists():raise FileExistsError(path)
    result=probe_saved_batch(args.config,args.checkpoint,args.training_log)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
