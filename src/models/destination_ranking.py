"""E15 destination-only scorer and system-oracle listwise supervision.

No environment state, oracle label, absolute UAV identity or candidate pooling
is accepted by the scorer. Inputs are detached from the historical backbone.
The existing joint Route PPO surrogate is deliberately left unchanged.
"""
from __future__ import annotations

import math
from collections import Counter
import json
from pathlib import Path

import torch
from torch import nn

KEY = "destination_system_oracle_v1"
# Globally anchored tolerances, fixed before any training or calibration.
PROTOCOL = {"version": KEY, "latency_tie_slots": 1,
            "energy_absolute_tolerance_j": .001, "energy_relative_tolerance": .01,
            "continuation": "frozen_masked_argmax_full_episode"}


def masked_log_probs(scores, legal):
    if scores.shape != legal.shape or legal.dtype != torch.bool:
        raise ValueError("destination scores/mask mismatch")
    if not torch.isfinite(scores).all():
        raise ValueError("non-finite destination score")
    fallback = torch.zeros_like(legal); fallback[..., 0] = True
    safe = torch.where(legal.any(-1, keepdim=True), legal, fallback)
    return scores.masked_fill(~safe, torch.finfo(scores.dtype).min).log_softmax(-1)


class SystemAwareDestinationScorer(nn.Module):
    """Shared independent scorer, with physical ratios in seconds/Joules.

    All constants below are task/configuration scales, never fitted on eval.
    Features are aligned by the caller's destination IDs; identity is preserved
    by that wiring, not by an absolute-ID embedding or learned UAV bias.
    """
    def __init__(self, config, spec):
        super().__init__()
        from .ca_gat_mappo import ActorObservationTensorizer, RouteOptionFeatureExtractor
        adapter = ActorObservationTensorizer(config)
        env = config.environment
        self.extractor = RouteOptionFeatureExtractor(spec)
        self.uav_count = spec.uav_count
        self.ru_count = spec.ru_count
        self.bits_scale = adapter._max_task_bits
        self.cycles_scale = adapter._max_task_cycles
        self.backlog_scale = adapter._max_queue_cycles
        self.slack_scale = adapter._slack_scale
        self.frequency_scale = env.reference_cpu_frequency_hz
        self.rate_scale = env.reference_rate_bps
        self.bandwidth = env.ru_bandwidth_hz
        self.noise_w = adapter._noise_reference_w
        self.power_w = env.reference_transmit_power_w
        self.energy_scale = env.reference_initial_energy_j
        self.cpu_coefficient = env.reference_cpu_coefficient
        self.slot_duration = env.slot_duration_s
        self.input_dimensions = (5, spec.gru_hidden_dimension + spec.encoder_hidden_dimension + 14,
                                 spec.encoder_hidden_dimension + spec.neighbor_public_feature_dim,
                                 spec.edge_feature_dim, 12)
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(n, 32), nn.LayerNorm(32), nn.LeakyReLU())
            for n in self.input_dimensions])
        self.scorer = nn.Sequential(nn.Linear(160, 64), nn.LeakyReLU(), nn.Linear(64, 1))

    @staticmethod
    def _signed_log(value):
        return value.sign() * torch.log1p(value.abs())

    def features(self, recurrent, source_endpoint, endpoints, self_features, public, edge, ids):
        ex = self.extractor
        task, source, _, _, _, _ = ex.extract(self_features, public, edge, ids)
        candidate_public = ex._gather_candidates(public, ids)
        candidate_edge = ex._gather_candidates(edge, ids)
        candidate_endpoint = ex._gather_candidates(endpoints, ids)
        count = ids.shape[-1]
        def expand(x):
            return x.unsqueeze(-2).expand(*x.shape[:-1], count, x.shape[-1])
        f = candidate_public[..., 10].clamp_min(1e-8) * self.frequency_scale
        sf = source[..., 2].clamp_min(1e-8) * self.frequency_scale
        bits = expand(task)[..., 0] * self.bits_scale
        cycles = expand(task)[..., 1] * self.cycles_scale
        slack_s = expand(task)[..., 3] * self.slack_scale * self.slot_duration
        backlog = candidate_public[..., 13] * self.backlog_scale
        # Stale CSI and historical interference only; no true/future channel.
        real = candidate_edge[..., 2:2+self.ru_count]
        imag = candidate_edge[..., 2+self.ru_count:2+2*self.ru_count]
        gain = real.square() + imag.square()
        interference_start = 4 + 2*self.ru_count
        interference = candidate_edge[..., interference_start:interference_start+self.ru_count]
        interference_valid = candidate_edge[..., interference_start+self.ru_count:interference_start+2*self.ru_count]
        noise = self.noise_w * (1 + torch.where(interference_valid > .5,
                                               interference.clamp_min(0), 0))
        power = expand(source)[..., 1:2] * self.power_w / self.ru_count
        proxy = (self.bandwidth * torch.log2(1 + power * gain / noise)).sum(-1)
        csi_valid = candidate_edge[..., 2+2*self.ru_count] > .5
        historical_valid = candidate_edge[..., -3] > .5
        rate = torch.where(historical_valid, candidate_edge[..., -4] * self.rate_scale,
                           torch.where(csi_valid, proxy, torch.zeros_like(proxy)))
        rate_valid = historical_valid | csi_valid
        tx_s = torch.where(rate_valid, bits / rate.clamp_min(1.), torch.zeros_like(bits))
        cpu_s = cycles / f
        queue_s = backlog / f
        candidate_energy = candidate_public[..., 7] * self.energy_scale
        cpu_energy_proxy = self.cpu_coefficient * candidate_public[..., 11] * f.square() * cycles
        relations = torch.stack((expand(sf.unsqueeze(-1))[..., 0]/f, cpu_s,
            queue_s, tx_s, slack_s/(cpu_s+queue_s+tx_s).clamp_min(self.slot_duration),
            candidate_energy/(cpu_energy_proxy+self.energy_scale*1e-8),
            candidate_public[..., 8], expand(source)[..., 0], rate/self.rate_scale,
            rate_valid.to(f.dtype), slack_s, candidate_public[..., 0]), -1)
        # Raw booleans/valid flags remain explicit; only continuous blocks with
        # large dynamic ranges use a fixed signed-log transform.
        link = candidate_edge.clone()
        link[..., interference_start:interference_start+self.ru_count] = self._signed_log(interference)
        blocks = (expand(task), expand(torch.cat((recurrent, source_endpoint, source), -1)),
                  torch.cat((candidate_endpoint, candidate_public), -1), link,
                  torch.cat((self._signed_log(relations[..., :6]), relations[..., 6:]), -1))
        result = torch.cat(blocks, -1).detach()
        if not torch.isfinite(result).all():
            raise ValueError("E15 observable destination features contain NaN/Inf")
        return result

    def forward(self, features):
        if features.shape[-1] != sum(self.input_dimensions):
            raise ValueError("destination feature dimension mismatch")
        blocks = features.detach().split(self.input_dimensions, -1)
        return self.scorer(torch.cat([p(x) for p,x in zip(self.projections, blocks)], -1)).squeeze(-1)


def system_oracle_label(rows, legal_destinations):
    """Sequential global-anchor filters implement a transitive optimal set."""
    legal = tuple(legal_destinations)
    if not legal or len(set(legal)) != len(legal):
        raise ValueError("invalid legal destination set")
    if len(rows) != len(legal) or {r['destination_uav'] for r in rows} != set(legal):
        raise ValueError("oracle must enumerate every legal Remote exactly once")
    for r in rows:
        if any(not isinstance(r[k], int) or r[k] < 0 for k in ('DONE','EXPIRED','TRUNCATED')):
            raise ValueError("invalid system outcome count")
        if r['target_outcome'] not in {'DONE','EXPIRED','TRUNCATED'}:
            raise ValueError("invalid target terminal outcome")
        if r['target_outcome'] == 'DONE' and (r['target_latency_slots'] is None
                or not math.isfinite(r['target_latency_slots']) or r['target_latency_slots'] < 0):
            raise ValueError("DONE target requires latency")
        if not math.isfinite(r['system_service_energy_j']) or r['system_service_energy_j'] < 0:
            raise ValueError("invalid system energy")
    if len({r['DONE']+r['EXPIRED']+r['TRUNCATED'] for r in rows}) != 1:
        raise ValueError("shadow task realization differs")
    keep = list(rows)
    for key, maximize in (('DONE',True), ('EXPIRED',False), ('TRUNCATED',False)):
        value = (max if maximize else min)(r[key] for r in keep)
        keep = [r for r in keep if r[key] == value]
    target_done = any(r['target_outcome'] == 'DONE' for r in keep)
    keep = [r for r in keep if (r['target_outcome']=='DONE') == target_done]
    if target_done:
        fastest = min(r['target_latency_slots'] for r in keep)
        keep = [r for r in keep if r['target_latency_slots'] <= fastest+PROTOCOL['latency_tie_slots']]
    minimum = min(r['system_service_energy_j'] for r in keep)
    tolerance = max(PROTOCOL['energy_absolute_tolerance_j'],
                    PROTOCOL['energy_relative_tolerance']*minimum)
    keep = [r for r in keep if r['system_service_energy_j'] <= minimum+tolerance]
    best = sorted(r['destination_uav'] for r in keep)
    return {'best_set': best, 'target': [1/len(best) if d in best else 0. for d in legal],
            'protocol': dict(PROTOCOL)}


def listwise_loss(scores, legal, targets):
    targets = targets.detach().to(scores)
    if (not torch.isfinite(targets).all() or (targets < 0).any()
            or (targets[~legal] != 0).any()
            or not torch.allclose(targets.sum(-1), torch.ones_like(targets.sum(-1)), atol=1e-6)):
        raise ValueError("invalid legal-only listwise target")
    return -(targets * masked_log_probs(scores, legal)).sum(-1)


def destination_ppo_contribution(actor, policy, old_route_log_prob, advantage, active, divisor, epsilon):
    """Exact destination-parameter part of unchanged joint Route PPO.

    Keep the current top-level log-probability numerically intact but detach
    its gradient for this *diagnostic* calculation. Old log-prob is the actual
    stored joint Route behavior probability, not a fabricated conditional one.
    The caller still optimizes the historical joint Route objective.
    """
    scores=actor.route_decoder.system_destination_scorer(policy.destination_features)
    mask=policy.action_masks['route'][...,3:]
    index=policy.action_indices['route']
    remote=(index>=3)&active
    conditional=masked_log_probs(scores,mask).gather(-1,(index-3).clamp_min(0).unsqueeze(-1)).squeeze(-1)
    new=policy.branch_log_probs['route'].detach()+conditional-conditional.detach()
    ratio=(new-old_route_log_prob.detach().to(new)).exp()
    advantage=advantage.detach().to(new)
    surrogate=torch.minimum(ratio*advantage,ratio.clamp(1-epsilon,1+epsilon)*advantage)
    return -surrogate.masked_fill(~remote,0.).sum()/(active.sum().clamp_min(1)*max(1,divisor))


def destination_gradient_probe(actor, ppo_loss, rank_loss, coefficient, clip_norm, base_total_loss=None):
    """Autograd vectors, not advantage RMS; never writes .grad or steps Adam."""
    named=[(n,p) for n,p in actor.named_parameters() if p.requires_grad]
    params=[p for _,p in named]
    def vector(loss):
        gradients=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten().double()
                          for p,g in zip(params,gradients)])
    ppo,rank=vector(ppo_loss),vector(rank_loss)
    combined=ppo+coefficient*rank
    norm=lambda x:float(x.norm())
    denominator=norm(ppo)*norm(rank)
    base=vector(base_total_loss) if base_total_loss is not None else ppo
    actual=base+coefficient*rank
    factor=min(1.,clip_norm/(norm(actual)+1e-6))
    offset=0;outside=0.
    for name,p in named:
        stop=offset+p.numel()
        if 'system_destination_scorer' not in name:outside+=float(rank[offset:stop].square().sum())
        offset=stop
    return {'ppo_destination_gradient_norm':norm(ppo),'rank_gradient_norm':norm(rank),
        'combined_destination_gradient_norm':norm(combined),
        'ppo_rank_cosine':float(ppo.dot(rank))/denominator if denominator else None,
        'coefficient':coefficient,'weighted_rank_to_ppo_norm':coefficient*norm(rank)/norm(ppo) if norm(ppo)>0 else None,
        'rank_gradient_outside_destination_norm':math.sqrt(outside),
        'global_clip_scope':'production total actor loss' if base_total_loss is not None else 'destination-only diagnostic (other losses unavailable)',
        'global_actor_norm_before_clip':norm(actual),'global_clip_factor':factor,
        'global_actor_norm_after_clip':norm(actual)*factor,
        'weighted_rank_norm_before_clip':coefficient*norm(rank),
        'weighted_rank_norm_after_clip':coefficient*norm(rank)*factor,
        'ppo_destination_norm_after_clip':norm(ppo)*factor,
        'weighted_ratio_after_clip':coefficient*norm(rank)/norm(ppo) if norm(ppo)>0 else None,
        'optimizer_step':False}


def ranking_batch_loss(actor, policy, buffer, epoch, route_surrogate, active, divisor, coefficient):
    """Replay immutable labels indexed by rollout position/source, never targets as inputs."""
    features = policy.destination_features
    masks = policy.action_masks['route'][..., 3:]
    scorer = actor.route_decoder.system_destination_scorer
    if features is None or scorer is None:
        raise ValueError('E15 ranking requires system-aware destination features')
    scores = scorer(features)
    flat_scores, flat_masks = scores.reshape(-1,*scores.shape[-2:]), masks.reshape(-1,*masks.shape[-2:])
    surrogate = route_surrogate.reshape(-1, actor.spec.uav_count)
    active = active.reshape_as(surrogate)
    denominator = int(active.sum()) * divisor
    records, losses = [], []
    for t, transition in enumerate(buffer._transitions):
        for row in transition.rejection_or_downgrade_summary.get(KEY, []):
            u = row['source_uav']
            destinations = actor.route_decoder.remote_candidate_ids[u].tolist()
            legal_ids = [d for d,m in zip(destinations,flat_masks[t,u].tolist()) if m]
            if legal_ids != row['legal_destinations']:
                raise ValueError('stored oracle mask/identity differs from replay')
            if transition.proposal_actions[u].route != row['factual_chosen_destination']:
                raise ValueError('oracle factual destination differs from rollout')
            target = scores.new_tensor([row['target'][legal_ids.index(d)] if d in legal_ids else 0. for d in destinations])
            ce = listwise_loss(flat_scores[t,u],flat_masks[t,u],target)
            losses.append(ce)
            probabilities = masked_log_probs(flat_scores[t,u],flat_masks[t,u]).exp().detach()
            chosen = destinations[int(probabilities.argmax())]
            ppo = float((-surrogate[t,u]/max(1,denominator)).detach())
            records.append({**row, 'epoch_index':epoch,
                'destination_logits': flat_scores[t,u].detach().cpu().tolist(),
                'destination_probabilities': probabilities.cpu().tolist(),
                'candidate_ids':destinations, 'rank_loss':float(ce.detach()),
                'ppo_destination_loss_contribution':ppo,
                'total_destination_loss_contribution': ppo+coefficient*float(ce.detach())/max(1,sum(len(x.rejection_or_downgrade_summary.get(KEY,[])) for x in buffer._transitions)),
                'actor_top1_destination':chosen,'top1_in_oracle_set':chosen in row['best_set'],
                'ppo_definition':'existing joint Route clipped surrogate; Remote cohort contribution; no extra PPO objective'})
    if not losses:
        return scores.sum()*0., records
    return torch.stack(losses).mean(), records


def write_destination_telemetry(config, records, environment_steps):
    path = Path(config.output.logs_dir)/config.run_id/'destination_ranking_updates_v1.jsonl'
    path.parent.mkdir(parents=True,exist_ok=True)
    first = [r for r in records if r['epoch_index']==0]
    multi = [r for r in first if len(r['legal_destinations']) > 1]
    informative = [r for r in multi if len(r['best_set']) < len(r['legal_destinations'])]
    confusion = Counter(f"{r['factual_chosen_destination']}->{','.join(map(str,r['best_set']))}" for r in first)
    summary = {'labels':len(first), 'candidate_count':sum(len(r['legal_destinations']) for r in first),
        'multi_candidate_labels':len(multi),'informative_labels':len(informative),
        'all_tie_multi_candidate_labels':len(multi)-len(informative),
        'informative_fraction_of_multi':len(informative)/len(multi) if multi else None,
        'oracle_top1_set_accuracy_informative':sum(r['top1_in_oracle_set'] for r in informative)/len(informative) if informative else None,
        'shadow_transitions':sum(r['shadow_transitions'] for r in first),
        'shadow_wall_seconds':sum(r['shadow_wall_seconds'] for r in first),
        'oracle_top1_set_accuracy':sum(r['top1_in_oracle_set'] for r in first)/len(first) if first else None,
        'factual_in_set_fraction':sum(r['factual_chosen_destination'] in r['best_set'] for r in first)/len(first) if first else None,
        'mean_rank_loss':sum(r['rank_loss'] for r in first)/len(first) if first else None,
        'cross_entropy':sum(r['rank_loss'] for r in first)/len(first) if first else None,
        'destination_confusion':dict(confusion),
        'recoverable_failure_proxy':sum(r['recoverable_failure_proxy'] for r in first)}
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps({'schema':KEY,'config_hash':config.config_hash,'environment_steps':environment_steps,
                            'coefficient':config.training.mappo.destination_oracle_rank_coefficient,
                            'summary':summary,'samples':records},allow_nan=False)+'\n')
