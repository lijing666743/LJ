"""Versioned, field-aware Q-only conditioning of centralized training inputs.

No fitted/running statistics. Schema mirrors CentralizedStateTensorizer order;
physical reference scales come only from the training configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class StateField:
    name: str
    start: int
    stop: int
    unit: str
    gain: float = 1.
    signed_log: bool = False
    discrete: bool = False


def centralized_q_fields(config):
    """Exact scalar layout, including interleaved task aggregates."""
    e = config.environment
    n, r = e.uav_count, e.ru_count
    fields = []; offset = 0

    def add(name, size, unit="dimensionless", gain=1., log=False, discrete=False):
        nonlocal offset
        fields.append(StateField(name, offset, offset+size, unit, gain, log, discrete))
        offset += size

    noise = 10**((e.noise_psd_dbm_hz-30)/10)*e.ru_bandwidth_hz*10**(e.receiver_noise_figure_db/10)
    channel_gain = math.sqrt(e.reference_transmit_power_w/max(noise, 1e-300))
    add("slot", 1, "slot/horizon")
    add("positions", 3*n, "position/bounds")
    add("velocities", 3*n, "velocity/config std", log=True)
    add("tx_power", n, "W/reference W", log=True)
    add("cpu_frequency", n, "Hz/reference Hz", log=True)
    add("cpu_coefficient", n, "coefficient/reference", log=True)
    add("energy", n, "J/reference J")
    add("arrivals", n, "probability")
    add("neighbor_mask", n*n, discrete=True)
    add("distance", n*n, "m/diagonal")
    for part in ("real", "imag"):
        add("true_channel_"+part, n*n*r, "channel amplitude", channel_gain, True)
    add("blocked_mask", n*n, discrete=True)
    add("shadowing", n*n, "dB/config std", log=True)
    add("path_loss", n*n, "dB", .01)
    for part in ("real", "imag"):
        add("stale_csi_"+part, n*n*r, "channel amplitude", channel_gain, True)
    add("csi_valid", n*n, discrete=True)
    add("csi_age", n*n, "slots/horizon")
    add("interference", n*r, "W/noise W", log=True)
    add("interference_valid", n*r, discrete=True)
    add("message_age", n, "(slots+1)/horizon")
    add("message_valid", n, discrete=True)
    add("historical_denominator", n*r, "W/noise W", log=True)
    add("historical_quality", n*n*r, "dimensionless channel quality", log=True)
    add("quality_valid", n*n*r, discrete=True)
    max_tasks = max(1, n*e.episode_horizon)
    for u in range(n):
        for j, name in enumerate(("count", "bits", "cycles", "remaining_bits", "remaining_cycles",
                                   "slack", "unbound", "local", "tx", "cpu")):
            # Encoder divides aggregate totals by max_tasks. Undo this common
            # dilution for Q only; bits/cycles retain their per-task references.
            add(f"task_u{u}_{name}", 1, "slack/max_slack" if j == 5 else "task reference units/max_tasks",
                1. if j == 5 else max_tasks, j != 5)
    add("bound_destination_counts", n*n, "count/max_tasks", max_tasks, True)
    add("queues", 4*n*n, "count/max_tasks", max_tasks, True)
    for b in range(len(e.building_layout)):
        add(f"building_{b}_present", 1, discrete=True)
        add(f"building_{b}_geometry", 5, "position/bounds")
    return tuple(fields)


class RouteQInputConditioner(nn.Module):
    """Stateless transforms; checkpoint buffers fully define the fixed scales."""

    def __init__(self, state_dim, candidate_dim, fields):
        super().__init__()
        if candidate_dim < 5 or (candidate_dim-3) % 2:
            raise ValueError("conditioned candidates require two equal embeddings and three type bits")
        if not fields or fields[0].start != 0 or fields[-1].stop != state_dim:
            raise ValueError("Q input schema does not cover centralized state")
        gain = torch.ones(state_dim, dtype=torch.get_default_dtype())
        compress = torch.zeros(state_dim, dtype=torch.bool)
        offset = 0
        for f in fields:
            if f.start != offset or f.stop < f.start or not math.isfinite(f.gain) or f.gain <= 0:
                raise ValueError("invalid Q field layout/scale")
            if f.discrete and (f.gain != 1 or f.signed_log):
                raise ValueError("discrete Q fields must retain their encoding")
            gain[f.start:f.stop] = f.gain
            compress[f.start:f.stop] = f.signed_log
            offset = f.stop
        self.register_buffer("gain", gain)
        self.register_buffer("compress", compress)
        self.register_buffer("schema_version", torch.tensor(1, dtype=torch.int64))
        self.embedding_width = (candidate_dim-3)//2

    def forward(self, state, candidates):
        if not torch.isfinite(state).all() or not torch.isfinite(candidates).all():
            raise ValueError("nonfinite legal Route Q input")
        # Float64 intermediate protects finite float32 extremes during scaling
        # and RMS computation. Cast back before trainable layers.
        x = state.double()*self.gain.double()
        state_out = torch.where(self.compress, x.sign()*torch.log1p(x.abs()), x).to(state.dtype)
        blocks = candidates[..., :-3].double().split(self.embedding_width, -1)
        conditioned = [b / b.square().mean(-1, keepdim=True).sqrt().clamp_min(1.) for b in blocks]
        candidate_out = torch.cat((*[b.to(candidates.dtype) for b in conditioned], candidates[..., -3:]), -1)
        if not torch.isfinite(state_out).all():
            raise ValueError("conditioned Route Q state overflow")
        return state_out, candidate_out
