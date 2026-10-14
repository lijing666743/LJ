"""E8 actor-visible candidate decoders; no environment or learning side effects."""
from __future__ import annotations

import math
import torch
from torch import nn


def policy_component_names(config):
    m = config.training.mappo
    tx = ("tx_upper", "tx_destination") if m.tx_upper_decoder_mode != "legacy" else ("tx_select",)
    cpu = ("cpu_joint",) if m.cpu_scheduler_mode != "legacy_hierarchical" else ("cpu_queue", "cpu_frequency")
    return ("route", *tx, "resource_group", "resource_width", "power_level", *cpu)


class TXUpperPrivateBinaryDecoderV1(nn.Module):
    """Raw observation only: shared destination utility, masked max, binary MLP.

    No recurrent/graph context and no candidate-index embedding. Illegal
    destinations neither affect the candidate maximum nor the source summary.
    """
    def __init__(self, spec):
        super().__init__()
        self.candidate_scorer = nn.Sequential(
            nn.Linear(8 + spec.neighbor_public_feature_dim + spec.edge_feature_dim, 32),
            nn.Tanh(), nn.Linear(32, 1))
        self.binary = nn.Sequential(nn.Linear(19, 32), nn.Tanh(), nn.Linear(32, 2))

    def score(self, source, candidates, legal):
        utility = self.candidate_scorer(candidates).squeeze(-1)
        available = legal.any(-1)
        best = utility.masked_fill(~legal, -torch.inf).amax(-1)
        best = torch.where(available, best, torch.zeros_like(best))
        queues = candidates[..., :8]
        summary = queues.masked_fill(~legal.unsqueeze(-1), 0).sum(-2)
        count = legal.sum(-1).to(source.dtype)
        summary = summary / count.clamp_min(1).unsqueeze(-1)
        logits = self.binary(torch.cat((source, summary, best.unsqueeze(-1),
                                       (count / legal.shape[-1]).unsqueeze(-1)), -1))
        return logits.masked_fill(~torch.stack((torch.ones_like(available), available), -1),
                                  torch.finfo(logits.dtype).min)

    def forward(self, batch, legal):
        # The caller provides the same stable non-self destination order as the
        # existing TX decoder; raw queue/public/edge inputs are explicitly bound.
        a = batch.self_features.shape[2]
        ids = torch.tensor([[j for j in range(a) if j != i] for i in range(a)],
                           device=batch.self_features.device)
        def gather(x):
            return x.gather(-2, ids.view(1, 1, a, a-1, 1).expand(*x.shape[:3], a-1, x.shape[-1]))
        queues = batch.self_features[..., 25:25+8*a].reshape(*batch.self_features.shape[:3], a, 8)
        raw = torch.cat((gather(queues), gather(batch.neighbor_public_features),
                         gather(batch.edge_features)), -1)
        # Signed log scaling keeps stale interference/CSI finite without
        # changing their visibility contract or imposing a feasibility rule.
        candidates = raw.sign() * torch.log1p(raw.abs())
        return self.score(batch.self_features[..., :9], candidates, legal)


class DeadlineAwareJointCPUSchedulerV1(nn.Module):
    """One utility per (source queue, positive frequency), plus a single Idle."""
    FEATURE_NAMES = ("head_cycles", "head_bits", "inclusive_slack", "queue_count",
                     "queue_workload", "fmax", "frequency_level", "energy_ratio",
                     "cpu_coefficient", "rho", "required_slots", "deadline_margin",
                     "capacity_per_cycle", "workload_pressure", "head_valid")

    def __init__(self, config, spec):
        super().__init__()
        env = config.environment
        self.max_cycles = env.task_data_bits_max * env.task_cycles_per_bit_max
        self.max_tasks = max(1, env.uav_count * env.episode_horizon)
        self.slack_scale = max(1, env.maximum_task_slack_slots)
        self.reference_hz = env.reference_cpu_frequency_hz
        self.delta_t = env.slot_duration_s
        self.anti_starvation_mode = config.training.mappo.cpu_anti_starvation_mode
        levels = [f for f in config.action.cpu_frequency_levels if f > 0]
        self.register_buffer("positive_levels", torch.tensor(levels), persistent=False)
        self.scorer = nn.Sequential(nn.Linear(spec.gru_hidden_dimension + len(self.FEATURE_NAMES), 64),
                                    nn.Tanh(), nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
        self.idle_scorer = nn.Sequential(nn.Linear(spec.gru_hidden_dimension + 9, 32),
                                        nn.Tanh(), nn.Linear(32, 1))

    def candidate_features(self, batch):
        x = batch.self_features
        a = x.shape[2]
        queues = x[..., 25+8*a:25+16*a].reshape(*x.shape[:3], a, 8)
        local = x[..., 17:25]
        queues = torch.where(torch.eye(a, dtype=torch.bool, device=x.device).view(1,1,a,a,1),
                             local.unsqueeze(-2), queues)
        def expand(v): return v.unsqueeze(-1).expand(*v.shape, len(self.positive_levels))
        cycles = expand(queues[..., 5] * self.max_cycles)
        workload = expand(queues[..., 2] * self.max_cycles * self.max_tasks)
        slack = expand(queues[..., 6] * self.slack_scale)
        fmax = expand(x[..., 5].unsqueeze(-1).expand(*x.shape[:3], a) * self.reference_hz)
        levels = self.positive_levels.to(x).view(1,1,1,1,-1).expand_as(cycles)
        capacity = fmax * levels * self.delta_t
        eps = torch.finfo(x.dtype).tiny
        rho = cycles / (slack.clamp_min(1) * fmax * self.delta_t).clamp_min(eps)
        required = cycles / capacity.clamp_min(eps)
        margin = slack - required
        pressure = workload / (slack.clamp_min(1) * fmax * self.delta_t).clamp_min(eps)
        ratio = capacity / cycles.clamp_min(1)
        def scaled(v): return torch.log1p(v.clamp(0, 1.e6))
        resource = lambda i: expand(x[..., i].unsqueeze(-1).expand(*x.shape[:3], a))
        features = torch.stack((cycles/self.max_cycles, expand(queues[..., 4]),
                                slack/self.slack_scale, expand(queues[..., 0]),
                                workload/(self.max_cycles*self.max_tasks), fmax/self.reference_hz,
                                levels, resource(2), resource(8), scaled(rho),
                                required.clamp_max(1.e6)/self.slack_scale,
                                margin.clamp(-1.e6,1.e6)/self.slack_scale,
                                scaled(ratio), scaled(pressure), expand(queues[..., 7])), -1)
        return features, {"rho": rho, "required_slots": required, "margin": margin,
                          "slack": slack, "workload": workload, "cycles": cycles,
                          "capacity": capacity, "frequency_hz": fmax*levels}

    def score_candidates(self, recurrent, candidates):
        context = recurrent.unsqueeze(-2).unsqueeze(-2).expand(*candidates.shape[:-1], recurrent.shape[-1])
        return self.scorer(torch.cat((context, candidates), -1)).squeeze(-1)

    def forward(self, recurrent, batch):
        features, physical = self.candidate_features(batch)
        scores = self.score_candidates(recurrent, features).flatten(-2)
        idle = self.idle_scorer(torch.cat((recurrent, batch.self_features[..., :9]), -1))
        return torch.cat((idle, scores), -1), physical


def cpu_deadline_feasibility(remaining_cycles, slack_slots, frequency_hz, delta_t):
    """Exact constant-frequency service slots and inclusive completion margin.

    Service uses min(remaining, frequency * dt), then subtracts in float64.
    Replay that arithmetic through the deadline rather than rounding float32
    features or granting an epsilon that the environment does not grant.
    A result of inf means more than the available slots would be needed.
    This is an optimistic, single-head/current-frequency test; it does not
    predict future energy, competing jobs, or joint TX/CPU budget downgrades.
    """
    capacity = frequency_hz * delta_t
    if remaining_cycles <= 0 or slack_slots <= 0 or capacity <= 0:
        return math.inf, -math.inf
    remaining = float(remaining_cycles)
    for required_slots in range(1, slack_slots + 1):
        remaining = max(0.0, remaining - min(remaining, capacity))
        if remaining == 0.0:
            return required_slots, slack_slots - required_slots
    return math.inf, -math.inf


def cpu_idle_deadline_critical(contract, delta_t):
    """True only if a legal head loses its last completion chance by idling."""
    if contract.cpu_deadline_context is None:
        raise ValueError("deadline_feasibility_v1 requires exact CPU deadline context")
    for qi, queue in enumerate(contract.cpu_queue_domain[1:], 1):
        if not contract.cpu_queue_mask[qi]:
            continue
        frequency_mask = contract.mask_for("cpu_frequency", {"cpu_queue": queue})
        levels = [float(level) for k, level in enumerate(contract.cpu_frequency_domain)
                  if level > 0 and frequency_mask[k]]
        if not levels:
            continue
        cycles, slack, fmax = contract.cpu_deadline_context[qi - 1]
        # Use the fastest currently legal level. A slower atom being urgent
        # must not hide the waiting margin still provided by a faster atom.
        _, completion_margin = cpu_deadline_feasibility(
            cycles, slack, max(levels) * fmax, delta_t)
        if completion_margin == 0:
            return True
    return False


def cpu_joint_statistics(actor, recurrent, batch, contracts, external, mode, generator):
    """Sample/evaluate ONE categorical, then decode to the existing executor pair."""
    from .ca_gat_mappo_actions import ActionDistributionError
    logits, physical = actor.cpu_joint_scheduler(recurrent, batch)
    levels = actor.cpu_joint_scheduler.positive_levels.tolist()
    # Float32 buffer values need not equal Python .1 exactly. Use config/domain
    # indices established by the scheduler rather than float equality.
    rows = []
    pairs = []
    for contract in contracts:
        domain = contract.cpu_frequency_domain
        positive = [i for i, f in enumerate(domain) if f > 0]
        if len(positive) != len(levels):
            raise ActionDistributionError("CPU joint frequency domain mismatch")
        row = [bool(contract.cpu_queue_mask[0])]
        atoms = [(0, domain.index(contract.canonical_cpu_frequency))]
        for qi, queue in enumerate(contract.cpu_queue_domain[1:], 1):
            frequency_mask = contract.mask_for("cpu_frequency", {"cpu_queue": queue})
            row.extend(bool(contract.cpu_queue_mask[qi]) and bool(frequency_mask[k]) for k in positive)
            atoms.extend((qi, k) for k in positive)
        if (actor.cpu_joint_scheduler.anti_starvation_mode == "deadline_feasibility_v1"
                and cpu_idle_deadline_critical(contract, actor.cpu_joint_scheduler.delta_t)):
            row[0] = False
        rows.append(row)
        pairs.append(atoms)
    shape = recurrent.shape[:3]
    mask = torch.tensor(rows, dtype=torch.bool, device=logits.device).reshape_as(logits)
    distribution = torch.distributions.Categorical(logits=logits.masked_fill(~mask, -torch.inf))
    if external is not None:
        indices = []
        for contract, proposal, atoms, row in zip(contracts, external, pairs, rows):
            queue = proposal.uav_id if proposal.cpu_queue == "local" else proposal.cpu_queue
            pair = (contract.cpu_queue_domain.index(queue), contract.cpu_frequency_domain.index(proposal.cpu_frequency))
            if pair not in atoms or not row[atoms.index(pair)]:
                raise ActionDistributionError("proposal outside E8 CPU joint support (Idle or legal positive frequency)")
            indices.append(atoms.index(pair))
        index = torch.tensor(indices, device=logits.device).reshape(shape)
    elif mode == "deterministic":
        index = distribution.probs.argmax(-1)
    else:
        index = torch.multinomial(distribution.probs.reshape(-1, logits.shape[-1]), 1,
                                 generator=generator).reshape(shape)
    decoded = [atoms[i] for atoms, i in zip(pairs, index.flatten().tolist())]
    queue_index = torch.tensor([p[0] for p in decoded], device=logits.device).reshape(shape)
    frequency_index = torch.tensor([p[1] for p in decoded], device=logits.device).reshape(shape)
    qcount = batch.self_features.shape[2]
    probs = distribution.probs
    candidate_probs = probs[..., 1:].reshape(*shape, qcount, len(levels))
    queue_probs = torch.cat((probs[..., :1], candidate_probs.sum(-1)), -1)
    frequency_probs = probs.new_zeros((*shape, len(contracts[0].cpu_frequency_domain)))
    chosen = candidate_probs.gather(-2, (queue_index-1).clamp_min(0).unsqueeze(-1).unsqueeze(-1)
                                   .expand(*shape, 1, len(levels))).squeeze(-2)
    chosen = chosen / chosen.sum(-1, keepdim=True).clamp_min(torch.finfo(probs.dtype).tiny)
    positive = [i for i, f in enumerate(contracts[0].cpu_frequency_domain) if f > 0]
    frequency_probs[..., positive] = chosen
    idle_freq = contracts[0].cpu_frequency_domain.index(contracts[0].canonical_cpu_frequency)
    idle_prob = torch.zeros_like(frequency_probs); idle_prob[..., idle_freq] = 1
    frequency_probs = torch.where((queue_index == 0).unsqueeze(-1), idle_prob, frequency_probs)
    selected = {}
    for name, values in physical.items():
        flat = values.flatten(-2)
        selected[name] = flat.gather(-1, (index-1).clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return {"distribution": distribution, "index": index, "queue_index": queue_index,
            "frequency_index": frequency_index, "queue_probs": queue_probs,
            "frequency_probs": frequency_probs, "selected": selected,
            "mask": mask, "probabilities": probs}


def rollout_policy_components(buffer, config, device):
    """Read frozen old component statistics from checkpoint-preserved metadata."""
    names = list(policy_component_names(config))
    logs, active = [], []
    for transition in buffer.get_sequence(0, 256).transitions:
        saved = transition.rejection_or_downgrade_summary.get("policy_components_v1")
        if saved is None or saved["names"] != names:
            raise ValueError("missing/mismatched E8 old policy components")
        log = torch.tensor(saved["log_probs"], dtype=torch.float32)
        flag = torch.tensor(saved["active"], dtype=torch.bool)
        if log.shape != (buffer.spec.uav_count, len(names)) or flag.shape != log.shape:
            raise ValueError("invalid E8 old policy component axes")
        if not torch.isfinite(log).all() or torch.any(log[~flag] != 0):
            raise ValueError("invalid old E8 logprob/activity")
        if not torch.allclose(log.sum(-1), transition.old_joint_log_prob, atol=2e-6):
            raise ValueError("old logical / executor joint logprob mismatch")
        logs.append(log); active.append(flag)
    return (torch.stack(logs).reshape(8,32,buffer.spec.uav_count,-1).to(device),
            torch.stack(active).reshape(8,32,buffer.spec.uav_count,-1).to(device))


def compact_scheduler_summary(policy, loss, config):
    """Detached reductions only; never write per-action extra telemetry."""
    names = list(policy.component_log_probs)
    result = {}
    surrogate = loss.flattened.branch_surrogate.reshape(*policy.joint_log_prob.shape, len(names))
    def mean(x, mask):
        values = x.detach()[mask]
        return float(values.float().mean().cpu()) if values.numel() else None
    for name in names:
        flag = policy.component_active[name]
        result[name] = {"active_samples": int(flag.sum()),
                        "entropy": mean(policy.component_entropies[name], flag),
                        "ppo_loss": mean(-surrogate[..., names.index(name)], flag)}
    metrics = policy.scheduler_metrics
    if "tx_upper" in names:
        flag = policy.component_active["tx_upper"]
        result["tx_upper"].update(sampled_transmit_fraction=mean(metrics["tx_sampled_transmit"], flag),
                                  p_transmit_mean=mean(metrics["tx_p_transmit"], flag),
                                  deterministic_transmit_fraction=mean(metrics["tx_deterministic_transmit"], flag))
        stability = loss.flattened.tx_upper_stability
        if stability is not None:
            result["tx_upper"].update(stability.compact_telemetry())
            result["tx_upper"]["objective_branch_divisor"] = loss.flattened.tx_upper_objective_divisor
            result["tx_upper"]["scaled_margin_loss"] = (
                .05 * float(stability.margin_loss.detach().cpu()) / loss.flattened.tx_upper_objective_divisor)
    if "cpu_joint" in names:
        flag = policy.component_active["cpu_joint"]
        positive = flag & ~metrics["cpu_joint_idle"]
        cpu = result["cpu_joint"]
        cpu["idle_fraction"] = mean(metrics["cpu_joint_idle"], flag)
        cpu["frequency_distribution"] = {str(f): mean(metrics["cpu_frequency"] == i, flag)
                                           for i,f in enumerate(config.action.cpu_frequency_levels)}
        for name in ("rho", "margin", "slack", "workload"):
            values = metrics[name].detach()[positive]
            cpu[name] = {"mean": mean(metrics[name], positive),
                         "p10_p50_p90": (torch.quantile(values.float(), values.new_tensor([.1,.5,.9])).cpu().tolist()
                                         if values.numel() else None)}
    return result
