"""E9 binary actor objective; no reward, critic, decoder or optimizer changes."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class TXUpperStabilityOutput:
    normalized_advantage: Tensor
    active: Tensor
    margin_loss: Tensor
    entropy_mean: Tensor
    entropy_without_upper: Tensor
    beta: float
    gap: Tensor
    transmitted: Tensor
    advantage_mean: Tensor
    advantage_std: Tensor

    def compact_telemetry(self) -> dict[str, object]:
        active = self.active
        positive = active & (self.normalized_advantage > 0)
        def mean(value):
            selected = value.detach()[active]
            return float(selected.mean().cpu()) if selected.numel() else None
        return {
            "advantage_cohort_mean": float(self.advantage_mean.cpu()),
            "advantage_cohort_std": float(self.advantage_std.cpu()),
            "mean_abs_p_transmit_minus_half": mean((self.gap.sigmoid()-.5).abs()),
            "mean_abs_logit_gap": mean(self.gap.abs()),
            "positive_advantage_transmit_count": int((positive & self.transmitted).sum()),
            "positive_advantage_idle_count": int((positive & ~self.transmitted).sum()),
            "margin_loss": float(self.margin_loss.detach().cpu()),
            "beta_tx": self.beta,
        }


class TXUpperDeterministicStabilityPPOV1:
    """Fixed, action-symmetric auxiliary semantics on the TX-active cohort.

    A positive advantage means positive *after* TX-cohort standardization,
    exactly the advantage used by the upper clipped PPO surrogate. Population
    std is used; a singleton/constant cohort has zero standardized advantage.
    Weight clipping bounds outlier influence without an action preference.
    """
    MARGIN_LOGIT = .5
    MARGIN_WEIGHT = .05
    POSITIVE_WEIGHT_MAX = 5.
    NORMALIZATION_EPS = 1.e-8

    @staticmethod
    def entropy_coefficient(base: float, steps: int, budget: int) -> float:
        if (isinstance(steps, bool) or not isinstance(steps, int) or steps < 0
                or isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0
                or not isinstance(base, (int, float)) or isinstance(base, bool)
                or not math.isfinite(base) or base < 0):
            raise ValueError("invalid TX entropy schedule identity")
        progress = min(steps / budget, 1.)
        return float(base) * (1. if progress <= .25 else
                              0. if progress >= .75 else (.75-progress)/.5)

    @classmethod
    def normalize_advantage(cls, advantage: Tensor, active: Tensor):
        if advantage.shape != active.shape or active.dtype != torch.bool:
            raise ValueError("TX advantage and active mask must share axes")
        selected = advantage.detach()[active]
        if selected.numel() and not torch.isfinite(selected).all():
            raise ValueError("nonfinite TX-active advantage")
        zero = advantage.new_zeros(())
        mean = selected.mean() if selected.numel() else zero
        std = selected.std(unbiased=False) if selected.numel() else zero
        result = torch.zeros_like(advantage)
        if selected.numel():
            result[active] = (selected-mean) / std.clamp_min(cls.NORMALIZATION_EPS)
        return result.detach(), mean.detach(), std.detach()

    @classmethod
    def binary_margin(cls, gap: Tensor, transmitted: Tensor,
                      advantage: Tensor, active: Tensor) -> Tensor:
        if (gap.shape != active.shape or transmitted.shape != active.shape
                or advantage.shape != active.shape or active.dtype != torch.bool
                or transmitted.dtype != torch.bool):
            raise ValueError("binary margin inputs must share axes")
        # Mask before softplus, including all-illegal upper logits, so inactive
        # atoms have exactly zero loss/gradient without inf*0 arithmetic.
        safe_gap = torch.where(active, gap, torch.zeros_like(gap))
        signed = torch.where(transmitted, safe_gap, -safe_gap)
        weights = torch.where(active, advantage.detach().clamp(0, cls.POSITIVE_WEIGHT_MAX),
                              torch.zeros_like(advantage))
        terms = weights * F.softplus(cls.MARGIN_LOGIT-signed)
        return terms.sum() / active.sum().clamp_min(1).to(gap.dtype)

    @classmethod
    def compute(cls, *, advantage: Tensor, active: Tensor, gap: Tensor,
                transmitted: Tensor, entropy: Tensor, entropy_without_upper: Tensor,
                base_entropy_coefficient: float, steps: int, budget: int):
        for value in (gap, entropy, entropy_without_upper):
            if (value.shape != active.shape or value.device != advantage.device
                    or value.dtype != advantage.dtype or not torch.isfinite(value[active]).all()):
                raise ValueError("invalid current TX-upper statistics")
        normalized, mean, std = cls.normalize_advantage(advantage, active)
        margin = cls.binary_margin(gap, transmitted, normalized, active)
        h = torch.where(active, entropy, torch.zeros_like(entropy)).sum()
        h = h / active.sum().clamp_min(1).to(entropy.dtype)
        return TXUpperStabilityOutput(normalized, active.detach(), margin, h,
                                      entropy_without_upper,
                                      cls.entropy_coefficient(base_entropy_coefficient, steps, budget),
                                      gap, transmitted.detach(), mean, std)
