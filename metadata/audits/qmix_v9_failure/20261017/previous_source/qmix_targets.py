"""Double-Q bootstrap and chunk-bounded TD(lambda) for the QMIX adaptation.

TD_LAMBDA follows the PyMARL2 QMIX configuration. The original global reward,
legal action set and real nonterminal chunk-end bootstrap remain unchanged.
"""
import math

import torch

from .ca_gat_mappo import stack_actor_time


TD_LAMBDA = 0.6
LEARNING_TARGET_CONTRACT = "double_q_chunk_td_lambda_0p6_v1"


def td_lambda_targets(reward, next_total, bootstrap, gamma, trace_lambda=TD_LAMBDA):
    """Detached forward-view returns over real slots, bootstrapped at chunk end."""
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
