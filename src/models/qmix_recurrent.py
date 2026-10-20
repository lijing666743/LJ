"""Current-weight recurrent context for replay, detached before each TD chunk."""
from dataclasses import replace

import torch

from .ca_gat_mappo import stack_actor_time


RECURRENT_CONTEXT_CONTRACT = "episode_prefix_current_weights_no_grad_v1"


def reconstruct_replay_sequences(online, target, sequences, replay=None):
    """Replay real prefixes once per episode/batch, without future inputs or BPTT.

    Contexts are discarded after each optimizer update. Collection-time hidden
    tensors are retained for diagnostics, but never initialize training forwards.
    """
    grouped = {}
    for sequence in sequences:
        if sequence.first_slot < 0:
            raise ValueError("negative recurrent chunk offset")
        starts = sequence.actor.episode_starts[:, 0]
        if bool(starts.all()) != (sequence.first_slot == 0):
            raise ValueError("recurrent chunk reset differs from its episode offset")
        if sequence.first_slot and replay is None:
            raise ValueError("noninitial recurrent replay requires complete episode prefixes")
        prefixes = () if sequence.first_slot == 0 else replay.prefix_actors(sequence)
        grouped.setdefault(sequence.episode, {})[sequence.first_slot] = prefixes
    contexts = {}
    encoded_slots = 0
    device = next(online.parameters()).device
    with torch.no_grad():
        for episode, offsets in sorted(grouped.items()):
            online_hidden = online.agent.initial_hidden()
            target_hidden = target.agent.initial_hidden()
            covered = 0
            for offset, prefixes in sorted(offsets.items()):
                selected = []
                start = 0
                for actor in prefixes:
                    length = actor.self_features.shape[1]
                    if start >= covered:
                        selected.append(actor)
                    elif start + length > covered:
                        raise ValueError("prefix reconstruction split a replay chunk")
                    start += length
                if start != offset:
                    raise ValueError("episode prefix is incomplete or includes future slots")
                if offset > covered:
                    batch = stack_actor_time(selected).to(device)
                    _, online_hidden = online.agent.encode(batch, online_hidden)
                    _, target_hidden = target.agent.encode(batch, target_hidden)
                    encoded_slots += offset-covered
                contexts[episode, offset] = (online_hidden.detach(), target_hidden.detach())
                covered = offset
    result = tuple(replace(sequence, hidden=contexts[sequence.episode, sequence.first_slot][0],
        target_hidden=contexts[sequence.episode, sequence.first_slot][1]) for sequence in sequences)
    return result, encoded_slots
