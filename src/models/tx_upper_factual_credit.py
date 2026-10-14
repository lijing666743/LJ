"""Detached factual team GAE credit for the sampled TX-upper branch."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..config import AgentCreditMode


@dataclass(frozen=True)
class TXFactualCredit:
    """One rollout's fixed credit; no learned or cross-update state."""

    team_return: Tensor
    team_old_value: Tensor
    raw_advantage: Tensor
    normalized_advantage: Tensor
    active: Tensor

    def telemetry(self, transmitted: Tensor) -> dict:
        if transmitted.shape != self.active.shape or transmitted.dtype != torch.bool:
            raise ValueError("TX factual telemetry requires boolean sampled actions")

        def mean(value):
            cohort = value[self.active]
            return float(cohort.mean().cpu()) if cohort.numel() else None

        def std(value):
            cohort = value[self.active]
            return float(cohort.std(unbiased=False).cpu()) if cohort.numel() else None

        return {
            "credit_mode": "factual_team_gae_v1",
            "credit_active_samples": int(self.active.sum()),
            "sampled_transmit_count": int((self.active & transmitted).sum()),
            "sampled_idle_count": int((self.active & ~transmitted).sum()),
            "team_return_mean": mean(self.team_return),
            "team_return_std": std(self.team_return),
            "team_old_value_mean": mean(self.team_old_value),
            "team_old_value_std": std(self.team_old_value),
            "raw_a_tx_mean": mean(self.raw_advantage),
            "raw_a_tx_std": std(self.raw_advantage),
            "normalization_mean": mean(self.raw_advantage),
            "normalization_std": std(self.raw_advantage),
            "normalization_epsilon": 1e-8,
            "normalized_a_tx_mean": mean(self.normalized_advantage),
            "normalized_a_tx_std": std(self.normalized_advantage),
            "credit_frozen_for_update": True,
            "q_status": "inactive",
            "popart_status": "inactive",
            **dict.fromkeys((
                "mean_q_idle", "mean_q_transmit", "mean_q_selected",
                "mean_counterfactual_baseline", "mean_q_transmit_minus_idle",
                "q_critic_loss", "q_critic_grad_norm_before_clip",
                "q_selected_explained_variance", "q_selected_target_correlation",
                "running_count", "running_mean", "running_std",
                "normalized_target_mean", "normalized_target_std",
                "normalized_selected_q_mean", "normalized_selected_q_std",
            )),
        }


@torch.no_grad()
def factual_team_credit(
    return_target: Tensor,
    old_value: Tensor,
    agent_credit_mode: AgentCreditMode | str,
    tx_active: Tensor,
    sequence_valid: Tensor,
) -> TXFactualCredit:
    """Sum RAW role returns/values, then normalize the active-valid cohort.

    The caller supplies the existing full-rollout lambda returns. This helper
    neither recomputes GAE nor introduces task, episode or recurrent boundaries.
    TEAM inputs have shape [...]; ROLE_DECOMPOSED inputs have shape [..., A].
    Idle and Transmit labels are deliberately absent from the credit interface.
    """
    mode = AgentCreditMode(agent_credit_mode)
    if (tx_active.dtype != torch.bool or sequence_valid.dtype != torch.bool
            or tx_active.ndim < 2
            or tx_active.shape[:-1] != sequence_valid.shape):
        raise ValueError("TX factual activity/sequence axes or dtypes differ")
    expected = tx_active.shape if mode is AgentCreditMode.ROLE_DECOMPOSED else sequence_valid.shape
    if return_target.shape != expected or old_value.shape != expected:
        raise ValueError("TX factual return/value axes do not match credit mode")
    if (not return_target.is_floating_point() or old_value.dtype != return_target.dtype
            or any(x.device != return_target.device
                   for x in (old_value, tx_active, sequence_valid))):
        raise ValueError("TX factual inputs must share device and floating dtype")
    active = tx_active.detach().clone() & sequence_valid.unsqueeze(-1)
    team_return = return_target.detach()
    team_value = old_value.detach()
    if mode is AgentCreditMode.ROLE_DECOMPOSED:
        team_return = team_return.sum(-1)
        team_value = team_value.sum(-1)
    team_return = team_return.unsqueeze(-1).expand_as(tx_active)
    team_value = team_value.unsqueeze(-1).expand_as(tx_active)
    if not (torch.isfinite(team_return[active]).all()
            and torch.isfinite(team_value[active]).all()):
        raise ValueError("nonfinite active factual team return/value")
    team_return = torch.where(active, team_return, torch.zeros_like(team_return))
    team_value = torch.where(active, team_value, torch.zeros_like(team_value))
    raw = team_return - team_value
    if not torch.isfinite(raw).all():
        raise ValueError("nonfinite factual advantage")
    normalized = torch.zeros_like(raw)
    if active.any():
        cohort = raw[active]
        normalized[active] = (cohort - cohort.mean()) / cohort.std(unbiased=False).clamp_min(1e-8)
    if not torch.isfinite(normalized).all():
        raise ValueError("nonfinite normalized factual advantage")
    return TXFactualCredit(team_return, team_value, raw, normalized, active)
