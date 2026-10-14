"""C-only actor-visible physical proxies; no environment service or oracle calls."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .ca_gat_mappo import ActorObservationTensorizer, OptionAwareRouteDecoderV1
from .route_candidates import remote_node_ids, visible_local_cpu_burden


CONTINUOUS_FIELDS = (
    "task_bits", "task_cycles", "slack_s", "frequency_hz", "queued_cycles",
    "cpu_task_count", "task_cpu_s", "queue_cpu_s", "tx_backlog_bits", "rate_bps",
    "task_tx_s", "queue_tx_s", "service_s", "slack_margin_s", "destination_energy_j",
    "cpu_energy_j", "source_energy_j", "tx_energy_j", "destination_energy_margin_j",
    "source_tx_energy_margin_j", "message_aoi_slots", "csi_aoi_slots", "outage_rate",
)
FLAG_FIELDS = (
    "task_valid", "destination_valid", "cpu_valid", "needs_u2u", "rate_observed",
    "rate_positive", "service_valid", "message_aoi_valid", "csi_valid",
    "history_rate_valid", "interference_all_valid", "outage_valid",
)
FEATURE_FIELDS = CONTINUOUS_FIELDS + FLAG_FIELDS


@dataclass(frozen=True)
class PhysicalRelationBatch:
    node_ids: Tensor
    raw: dict[str, Tensor]  # SI units except counts, slot ages and flags.
    features: Tensor      # Fixed signed log1p(value/config scale), then flags.


class RoutePhysicalRelationExtractor:
    """Restore tensorizer scales before comparing Local and published Remote.

    Time/energy are full-frequency/full-power proxies, not predicted schedules.
    Zero placeholders for missing estimates are meaningful only with flags.
    A measured zero rate is observed, but has no finite transfer-time estimate.
    """

    def __init__(self, config, spec):
        adapter = ActorObservationTensorizer(config)
        env = config.environment
        from .ca_gat_mappo import RouteOptionFeatureExtractor
        self.extractor = RouteOptionFeatureExtractor(spec)
        self.spec = spec
        self.bits = adapter._max_task_bits
        self.cycles = adapter._max_task_cycles
        self.queue_bits = adapter._max_queue_bits
        self.queue_cycles = adapter._max_queue_cycles
        self.max_tasks = adapter._max_tasks
        self.slack_slots = adapter._slack_scale
        self.horizon = adapter._time_scale
        self.frequency = env.reference_cpu_frequency_hz
        self.rate = env.reference_rate_bps
        self.energy = env.reference_initial_energy_j
        self.coefficient = env.reference_cpu_coefficient
        self.power = env.reference_transmit_power_w
        self.slot_s = env.slot_duration_s
        self.bandwidth = env.ru_bandwidth_hz
        self.noise_w = adapter._noise_reference_w

    def extract(self, self_features: Tensor, public: Tensor, edge: Tensor,
                ids: Tensor | None = None) -> PhysicalRelationBatch:
        ex, spec = self.extractor, self.spec
        canonical = remote_node_ids(spec.uav_count, device=self_features.device)
        ids = canonical if ids is None else ids
        if (ids.shape != canonical.shape or ids.dtype != torch.long
                or ids.device != canonical.device
                or not torch.equal(ids.sort(-1).values, canonical)):
            raise ValueError("physical relation IDs must enumerate each non-self node")
        if any(not x.is_floating_point() or x.dtype != self_features.dtype
               or x.device != self_features.device or not torch.isfinite(x).all()
               for x in (self_features, public, edge)):
            raise ValueError("physical relations require finite aligned actor tensors")
        task, _, _, _, _, _ = ex.extract(self_features, public, edge, ids)
        p = ex._gather_candidates(public, ids)
        e = ex._gather_candidates(edge, ids)
        tx = ex._gather_candidates(ex._indexed_queues(self_features, ex.tx_start), ids)
        local_cpu = visible_local_cpu_burden(self_features, spec)
        def join(local, remote):
            return torch.cat((local.unsqueeze(-1), remote), -1)
        def expand(value):
            return value.unsqueeze(-1).expand(*value.shape, spec.uav_count)
        zero = torch.zeros_like(task[..., 0])
        one = torch.ones_like(zero)
        remote_valid = p[..., 0] > .5
        dest_valid = join(one.bool(), remote_valid)
        needs = join(zero.bool(), torch.ones_like(remote_valid))
        task_valid = expand(task[..., 4] > .5)
        bits = expand(task[..., 0].clamp_min(0) * self.bits)
        cycles = expand(task[..., 1].clamp_min(0) * self.cycles)
        slack = expand(task[..., 3] * self.slack_slots * self.slot_s)
        # Local frequency/coefficient use actual resource columns (not an
        # accidental comparison of differently normalized queue/task units).
        f = join(self_features[..., 5], p[..., 10]).clamp_min(0) * self.frequency
        f = torch.where(dest_valid, f, 0.)
        kappa = join(self_features[..., 7], p[..., 11]).clamp_min(0) * self.coefficient
        backlog = join(local_cpu.normalized_remaining_cycles, p[..., 13]).clamp_min(0) * self.queue_cycles
        count = join(local_cpu.normalized_task_count, p[..., 12]).clamp_min(0) * self.max_tasks
        backlog = torch.where(dest_valid, backlog, 0.)
        count = torch.where(dest_valid, count, 0.)
        cpu_valid = dest_valid & (f > 0)
        # Tiny positive/zero capacity stays finite; validity never edits masks.
        cpu_s = torch.where(cpu_valid, cycles / f.clamp_min(1e-12), 0.)
        queue_s = torch.where(cpu_valid, backlog / f.clamp_min(1e-12), 0.)
        r = spec.ru_count
        csi_valid = e[..., 2+2*r] > .5
        history_valid = e[..., -3] > .5
        i_start = 4 + 2*r
        interference_valid = e[..., i_start+r:i_start+2*r] > .5
        interference = e[..., i_start:i_start+r].clamp_min(0)
        noise = self.noise_w * (1 + torch.where(interference_valid, interference, 0.))
        gain = e[..., 2:2+r].square() + e[..., 2+r:2+2*r].square()
        source_power = self_features[..., 3].clamp_min(0) * self.power
        # Full-RU equal power, stale CSI and historical interference. Missing
        # interference uses noise-only and remains explicitly flagged.
        csi_rate = (self.bandwidth * torch.log2(1 +
                    source_power.unsqueeze(-1).unsqueeze(-1) / r * gain / noise)).sum(-1)
        remote_rate = torch.where(history_valid, e[..., -4].clamp_min(0) * self.rate,
                                  torch.where(csi_valid, csi_rate, 0.))
        rate = join(zero, remote_rate)
        observed = join(zero.bool(), history_valid | csi_valid)
        positive = observed & (rate > 0)
        rate_ok = ~needs | positive
        tx_backlog = join(zero, tx[..., 1].clamp_min(0) * self.queue_bits)
        task_tx_s = torch.where(positive, bits / rate.clamp_min(1e-12), 0.)
        queue_tx_s = torch.where(positive, tx_backlog / rate.clamp_min(1e-12), 0.)
        service_valid = task_valid & cpu_valid & rate_ok
        service = torch.where(service_valid, cpu_s + queue_s + task_tx_s + queue_tx_s, 0.)
        margin = torch.where(service_valid, slack - service, 0.)
        dest_energy = join(self_features[..., 0], p[..., 7]).clamp_min(0) * self.energy
        dest_energy = torch.where(dest_valid, dest_energy, 0.)
        source_energy = expand(self_features[..., 0].clamp_min(0) * self.energy)
        cpu_energy = torch.where(cpu_valid, kappa * f.square() * cycles, 0.)
        tx_energy = expand(source_power) * (task_tx_s + queue_tx_s)
        dest_margin = torch.where(cpu_valid, dest_energy - cpu_energy, 0.)
        source_margin = torch.where(rate_ok, source_energy - tx_energy, 0.)
        message_valid = join(one.bool(), p[..., 16] > .5)
        message_aoi = join(zero, (p[..., 15] * self.horizon - 1).clamp_min(0))
        message_aoi = torch.where(message_valid, message_aoi, 0.)
        csi_aoi = join(zero, e[..., 3+2*r].clamp_min(0) * self.horizon)
        csi_flag = join(zero.bool(), csi_valid)
        csi_aoi = torch.where(csi_flag, csi_aoi, 0.)
        outage_valid = join(zero.bool(), e[..., -1] > .5)
        outage = torch.where(outage_valid, join(zero, e[..., -2]), 0.)
        values = (bits, cycles, slack, f, backlog, count, cpu_s, queue_s,
                  tx_backlog, rate, task_tx_s, queue_tx_s, service, margin,
                  dest_energy, cpu_energy, source_energy, tx_energy, dest_margin,
                  source_margin, message_aoi, csi_aoi, outage)
        flags = (task_valid, dest_valid, cpu_valid, needs, observed, positive,
                 service_valid, message_valid, csi_flag, join(zero.bool(), history_valid),
                 join(one.bool(), interference_valid.all(-1)), outage_valid)
        scales = (self.bits, self.cycles, self.slot_s, self.frequency, self.cycles,
                  1., self.slot_s, self.slot_s, self.bits, self.rate,
                  self.slot_s, self.slot_s, self.slot_s, self.slot_s, self.energy,
                  self.energy, self.energy, self.energy, self.energy, self.energy,
                  1., 1., 1.)
        raw = dict(zip(FEATURE_FIELDS, (*values, *flags)))
        continuous = torch.stack([v / scale for v, scale in zip(values, scales)], -1)
        features = torch.cat((continuous.sign() * torch.log1p(continuous.abs()),
                              torch.stack(flags, -1).to(continuous)), -1)
        if not torch.isfinite(features).all():
            raise ValueError("physical relation features contain NaN/Inf")
        ego = torch.arange(spec.uav_count, device=ids.device).unsqueeze(-1)
        return PhysicalRelationBatch(torch.cat((ego, ids), -1), raw, features)


class PhysicalOptionAwareRouteDecoderV1(OptionAwareRouteDecoderV1):
    """B plus a shared physical projection into its existing 204-dim fusion.

    The chosen-only Q layout stays 411-dim but now reflects C's fused inputs.
    Controls, masks, objectives and optimizer grouping retain their contracts.
    """

    def __init__(self, *, spec, config):
        super().__init__(spec=spec)
        self.physical_extractor = RoutePhysicalRelationExtractor(config, spec)
        from .unified_route import physical_route_initialization_seed
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(physical_route_initialization_seed(config.seed))
            self.physical_projection = nn.Sequential(
                nn.Linear(len(FEATURE_FIELDS), 32), nn.LeakyReLU(), nn.Linear(32, 204))

    def augment_option_features(self, fused, self_features, public_features, edge_features):
        relations = self.physical_extractor.extract(
            self_features, public_features, edge_features, self.remote_candidate_ids)
        return fused + self.physical_projection(relations.features)
