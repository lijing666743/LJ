"""Detached Double-Q returns over available, contiguous episode suffixes.

Gradient chunks remain short. Future collected rewards enter labels only;
they never become inputs to an earlier actor decision.
"""
from dataclasses import replace
import math

import torch

from .ca_gat_mappo import stack_actor_time


TD_LAMBDA = 0.95
LEARNING_TARGET_CONTRACT = "double_q_available_episode_suffix_td_lambda_0p95_v1"


def td_lambda_targets(reward, next_total, bootstrap, gamma, trace_lambda=TD_LAMBDA):
    """Detached forward-view returns, bootstrapped at the last available slot."""
    if (reward.ndim != 2 or reward.shape != next_total.shape or reward.shape != bootstrap.shape
            or bootstrap.dtype != torch.bool or reward.shape[1] == 0):
        raise ValueError("TD(lambda) requires matching nonempty [batch,time] tensors")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (gamma, trace_lambda)):
        raise ValueError("gamma and trace lambda must lie in [0,1]")
    if not torch.isfinite(reward).all() or not torch.isfinite(next_total[bootstrap]).all():
        raise ValueError("TD(lambda) inputs must be finite on real bootstrap slots")
    with torch.no_grad():
        next_values = torch.where(bootstrap, next_total, torch.zeros_like(next_total))
        result = torch.empty_like(reward)
        following = next_values[:, -1]
        for time in reversed(range(reward.shape[1])):
            continuation = ((1 - trace_lambda) * next_values[:, time]
                            + trace_lambda * following)
            following = reward[:, time] + gamma * torch.where(
                bootstrap[:, time], continuation, torch.zeros_like(continuation))
            result[:, time] = following
    return result


def join_return_context(chunks):
    """Join real same-episode chunks, preserving the rebuilt initial hidden."""
    if not chunks:
        raise ValueError("return context requires at least one real chunk")
    first, last = chunks[0], chunks[-1]
    end = first.first_slot
    for index, chunk in enumerate(chunks):
        chunk.validate()
        if chunk.episode != first.episode or chunk.first_slot != end:
            raise ValueError("return context has a missing, duplicate or cross-episode chunk")
        if index and (not bool(chunks[index-1].bootstrap[0, -1])
                      or chunk.actor.episode_starts.any()):
            raise ValueError("return context crosses a terminal/reset boundary")
        end += chunk.length
    result = replace(first,
        actor=stack_actor_time([chunk.actor for chunk in chunks]),
        contracts=(sum((chunk.contracts[0] for chunk in chunks), ()),),
        proposals=(sum((chunk.proposals[0] for chunk in chunks), ()),),
        state=torch.cat([chunk.state for chunk in chunks], dim=1),
        reward=torch.cat([chunk.reward for chunk in chunks], dim=1),
        bootstrap=torch.cat([chunk.bootstrap for chunk in chunks], dim=1),
        active=torch.cat([chunk.active for chunk in chunks], dim=1),
        executed=sum((chunk.executed for chunk in chunks), ()),
        next_actor=last.next_actor, next_contracts=last.next_contracts,
        next_state=last.next_state)
    result.validate()
    return result


def replay_return_targets(online, target, sequences, device, gamma, *, replay=None):
    """Evaluate a suffix once per episode and slice detached sampled labels.

    Inputs must already carry current-weight prefix hidden states. Only
    resident collected transitions are used; the live episode tail retains
    its real next-state bootstrap. No gradients or RNG draws occur here.
    """
    grouped = {}
    for sequence in sequences:
        grouped.setdefault(sequence.episode, []).append(sequence)
    labels, evaluated_slots, extra_slots = {}, 0, 0
    with torch.no_grad():
        for episode, selected in sorted(grouped.items()):
            # Without a replay store, each standalone chunk is its complete
            # known context; do not infer that other selected chunks fill gaps.
            groups = [selected] if replay is not None else [[seq] for seq in selected]
            for group in groups:
                first = min(group, key=lambda seq: seq.first_slot)
                chunks = (first,) if replay is None else replay.future_sequences(first)
                context = join_return_context(chunks)
                next_values = double_q_next_values(online, target, context, device)
                returns = td_lambda_targets(context.reward.to(device), next_values,
                    context.bootstrap.to(device), gamma)
                evaluated_slots += context.length
                for sequence in group:
                    offset = sequence.first_slot-context.first_slot
                    if offset < 0 or offset+sequence.length > context.length:
                        raise ValueError("sampled chunk lies outside its return context")
                    labels[episode, sequence.first_slot] = returns[:, offset:offset+sequence.length]
                    extra_slots += context.length-offset-sequence.length
    return tuple(labels[seq.episode, seq.first_slot] for seq in sequences), {
        "td_return_context_slots": evaluated_slots,
        "td_return_extra_slots": extra_slots,
    }


def double_q_next_values(online, target, sequence, device):
    """Online exact legal selection, evaluated by target agent and target mixer."""
    with torch.no_grad():
        actor = sequence.actor.to(device)
        batch = actor if sequence.next_actor is None else stack_actor_time(
            (actor, sequence.next_actor.to(device)))
        contracts = sequence.contracts if sequence.next_contracts is None else (
            sequence.contracts[0] + (sequence.next_contracts,),)
        selected = online.agent.actions(batch, sequence.hidden.to(device), contracts)
        evaluated = target.agent.actions(batch, sequence.target_hidden.to(device), contracts,
                                        proposals=selected.proposals)
        states = sequence.state.to(device)
        if sequence.next_state is not None:
            states = torch.cat((states, sequence.next_state.to(device)), dim=1)
        totals = target.mixer(evaluated.utilities, states)
        result = torch.zeros_like(sequence.reward, device=device)
        result[:, :-1] = totals[:, 1:sequence.length]
        if sequence.next_state is not None:
            result[:, -1] = totals[:, -1]
        return result
