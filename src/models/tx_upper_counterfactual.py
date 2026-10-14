"""On-policy, training-only binary TX Q credit; no actor/oracle inputs added."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TXQTargetNormalizerV1(nn.Module):
    """Shared exact running moments for active team returns; no learned state."""

    EPS = 1e-5

    def __init__(self):
        super().__init__()
        self.register_buffer("count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("M2", torch.zeros((), dtype=torch.float64))

    def _apply(self, fn, recurse=True):
        # Model .to(float32) must not downcast accumulated Welford moments.
        moments = {name: getattr(self, name) for name in ("mean", "M2")}
        super()._apply(fn, recurse)
        for name, old in moments.items():
            self._buffers[name] = old.to(device=getattr(self, name).device, dtype=torch.float64)
        return self

    @property
    def std(self):
        if int(self.count) == 0:
            return self.mean.new_ones(())
        return (self.M2 / self.count).clamp_min(0).sqrt().clamp_min(self.EPS)

    def normalize(self, value):
        return (value - self.mean.to(value)) / self.std.to(value)

    def denormalize(self, value):
        return self.std.to(value) * value + self.mean.to(value)

    @torch.no_grad()
    def update_and_compensate(self, target, active, output_layer):
        """Merge one update's active targets, then preserve both physical Qs."""
        if target.shape != active.shape or active.dtype != torch.bool:
            raise ValueError("normalizer target/activity axes differ")
        samples = target.detach()[active].double()
        if not samples.numel():
            return False
        if not torch.isfinite(samples).all():
            raise ValueError("nonfinite active TX normalization targets")
        old_mean, old_std = self.mean.clone(), self.std.clone()
        n = samples.numel()
        batch_mean = samples.mean()
        batch_M2 = (samples - batch_mean).square().sum()
        total = self.count + n
        total_float = total.double()
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (n / total_float)
        new_M2 = self.M2 + batch_M2 + delta.square() * (self.count * n / total_float)
        new_std = (new_M2 / total).clamp_min(0).sqrt().clamp_min(self.EPS)
        weight = output_layer.weight.double() * (old_std / new_std)
        bias = (old_std * output_layer.bias.double() + old_mean - new_mean) / new_std
        if not (torch.isfinite(weight).all() and torch.isfinite(bias).all()):
            raise ValueError("nonfinite TX PopArt compensation")
        output_layer.weight.copy_(weight)
        output_layer.bias.copy_(bias)
        self.count.copy_(total)
        self.mean.copy_(new_mean)
        self.M2.copy_(new_M2)
        return True

    def target_telemetry(self, target, active):
        samples = target.detach()[active]
        normalized = self.normalize(samples)
        return {
            "target_raw_mean": float(samples.mean()) if samples.numel() else None,
            "target_raw_std": float(samples.std(unbiased=False)) if samples.numel() else None,
            "running_count": int(self.count),
            "running_mean": float(self.mean),
            "running_std": float(self.std),
            "normalized_target_mean": float(normalized.mean()) if samples.numel() else None,
            "normalized_target_std": float(normalized.std(unbiased=False)) if samples.numel() else None,
        }


def model_precision_tensors(module):
    """Exclude only float64 Welford moments from the float32 model contract."""
    moment_ids = {id(t) for child in module.modules()
                  if isinstance(child, TXQTargetNormalizerV1)
                  for t in (child.mean, child.M2)}
    return tuple(t for t in (*module.parameters(), *module.buffers()) if id(t) not in moment_ids)


class TXUpperCounterfactualCriticV1(nn.Module):
    """Centralized state -> per-agent [Idle, Transmit] action values.

    Reuse the existing centralized feature tensor, not actor GAT/GRU weights
    or the original V critic's trainable encoder. Agent-specific output rows
    identify the sender; other agents follow the collected on-policy actions.
    """

    def __init__(self, state_dimension: int, hidden: int, agents: int, seed: int,
                 normalize_targets: bool = False):
        super().__init__()
        self.agents = agents
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 104729)
            self.network = nn.Sequential(
                nn.Linear(state_dimension, hidden), nn.LeakyReLU(),
                nn.Linear(hidden, agents * 2),
            )
        if normalize_targets:
            self.tx_q_target_norm = TXQTargetNormalizerV1()

    def forward_normalized(self, features: Tensor) -> Tensor:
        values = self.network(features.detach())
        return values.reshape(*features.shape[:-1], self.agents, 2)

    def forward(self, features: Tensor) -> Tensor:
        values = self.forward_normalized(features)
        if hasattr(self, "tx_q_target_norm"):
            return self.tx_q_target_norm.denormalize(values)
        return values

    def update_target_statistics(self, target, active):
        return self.tx_q_target_norm.update_and_compensate(target, active, self.network[-1])


@dataclass(frozen=True)
class TXCounterfactualCredit:
    q: Tensor
    selected_q: Tensor
    baseline: Tensor
    raw_advantage: Tensor
    normalized_advantage: Tensor
    active: Tensor
    transmitted: Tensor

    def telemetry(self) -> dict[str, float | int | None]:
        def mean(x):
            return float(x[self.active].mean().cpu()) if self.active.any() else None
        def std(x):
            return float(x[self.active].std(unbiased=False).cpu()) if self.active.any() else None
        return {
            "credit_active_samples": int(self.active.sum()),
            "mean_q_idle": mean(self.q[..., 0]),
            "mean_q_transmit": mean(self.q[..., 1]),
            "mean_q_selected": mean(self.selected_q),
            "mean_counterfactual_baseline": mean(self.baseline),
            "raw_a_tx_mean": mean(self.raw_advantage),
            "raw_a_tx_std": std(self.raw_advantage),
            "normalized_a_tx_mean": mean(self.normalized_advantage),
            "normalized_a_tx_std": std(self.normalized_advantage),
            "positive_raw_a_tx_transmit_count": int((self.active & self.transmitted & (self.raw_advantage > 0)).sum()),
            "positive_raw_a_tx_idle_count": int((self.active & ~self.transmitted & (self.raw_advantage > 0)).sum()),
            "positive_normalized_a_tx_transmit_count": int((self.active & self.transmitted & (self.normalized_advantage > 0)).sum()),
            "positive_normalized_a_tx_idle_count": int((self.active & ~self.transmitted & (self.normalized_advantage > 0)).sum()),
            "mean_q_transmit_minus_idle": mean(self.q[..., 1] - self.q[..., 0]),
        }


def counterfactual_credit(q: Tensor, probabilities: Tensor, transmitted: Tensor,
                          active: Tensor) -> TXCounterfactualCredit:
    """Freeze Q and policy baseline once/update; normalize only active atoms."""
    if q.shape != probabilities.shape or q.shape[-1] != 2:
        raise ValueError("TX Q/probabilities must share [...,2] axes")
    if active.shape != q.shape[:-1] or transmitted.shape != active.shape:
        raise ValueError("TX credit action/activity axes differ")
    if active.dtype != torch.bool or transmitted.dtype != torch.bool:
        raise ValueError("TX credit masks/actions must be boolean")
    if not torch.isfinite(q[active]).all() or not torch.isfinite(probabilities[active]).all():
        raise ValueError("nonfinite active TX Q/probabilities")
    p = probabilities.detach()
    if torch.any(p[active] < 0) or not torch.allclose(p[active].sum(-1), torch.ones_like(p[..., 0][active]), atol=2e-6, rtol=0):
        raise ValueError("TX policy probabilities must sum to one")
    fixed_q = torch.where(active.unsqueeze(-1), q.detach(), torch.zeros_like(q))
    p = torch.where(active.unsqueeze(-1), p, torch.zeros_like(p))
    selected = fixed_q.gather(-1, transmitted.long().unsqueeze(-1)).squeeze(-1)
    baseline = (p * fixed_q).sum(-1)
    raw = torch.where(active, selected - baseline, torch.zeros_like(selected)).detach()
    normalized = torch.zeros_like(raw)
    if active.any():
        cohort = raw[active]
        normalized[active] = (cohort - cohort.mean()) / cohort.std(unbiased=False).clamp_min(1e-8)
    return TXCounterfactualCredit(fixed_q.detach(), selected.detach(), baseline.detach(),
                                  raw, normalized.detach(), active.detach().clone(),
                                  transmitted.detach().clone())


def team_return_target(return_target: Tensor, agent_count: int) -> Tensor:
    """Sum conserved per-agent lambda returns; shared-team returns pass through.

    Gamma, lambda, rollout bootstrap and episode-boundary semantics come from
    the existing GAE implementation; no new horizon or reward is introduced.
    """
    team = return_target.sum(-1) if return_target.ndim == 3 else return_target
    return team.detach().unsqueeze(-1).expand(*team.shape, agent_count)


def selected_action_q_loss(q: Tensor, transmitted: Tensor, target: Tensor,
                          active: Tensor) -> Tensor:
    """Only observed TX actions get Huber labels; empty cohorts get no graph."""
    if q.shape[:-1] != active.shape or q.shape[-1] != 2 or target.shape != active.shape or transmitted.shape != active.shape:
        raise ValueError("TX Q regression axes differ")
    if active.dtype != torch.bool or transmitted.dtype != torch.bool:
        raise ValueError("TX Q regression requires boolean actions/mask")
    if not active.any():
        return q.new_zeros(())
    selected = q[active].gather(-1, transmitted[active].long().unsqueeze(-1)).squeeze(-1)
    fixed_target = target.detach()[active]
    if not torch.isfinite(selected).all() or not torch.isfinite(fixed_target).all():
        raise ValueError("nonfinite active TX Q regression input")
    return F.smooth_l1_loss(selected, fixed_target)


def q_regression_telemetry(q: Tensor, transmitted: Tensor, target: Tensor,
                           active: Tensor, loss: Tensor) -> dict[str, float | None]:
    """Current selected-action fit, distinct from the frozen PPO credit Q."""
    if not active.any():
        return {"q_critic_loss": float(loss.detach().cpu()),
                "q_selected_explained_variance": None, "q_selected_target_correlation": None}
    selected = q.detach()[active].gather(-1, transmitted[active].long().unsqueeze(-1)).squeeze(-1)
    y = target.detach()[active]
    var = y.var(unbiased=False)
    ev = float((1 - (y - selected).var(unbiased=False) / var).cpu()) if var > 1e-12 else None
    sx = selected.std(unbiased=False); sy = y.std(unbiased=False)
    corr = float(((selected - selected.mean()) * (y - y.mean())).mean().div(sx * sy).cpu()) if sx > 1e-12 and sy > 1e-12 else None
    return {"q_critic_loss": float(loss.detach().cpu()),
            "q_selected_explained_variance": ev, "q_selected_target_correlation": corr}
