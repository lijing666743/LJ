"""PPO objective and loss computation for CA-GAT-MAPPO.

This module only combines already-computed current policy statistics with
fixed rollout snapshots.  It does not perform recurrent re-evaluation,
backpropagation, parameter updates, gradient clipping, or training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from .tx_upper_stability import TXUpperStabilityOutput, TXUpperDeterministicStabilityPPOV1

from ..config import (
    ActorRatioMode,
    AgentCreditMode,
    PolicyLossAggregationMode,
    RouteCreditMode,
    RunConfig,
    compute_effective_route_entropy_schedule,
)


class PPOObjectiveError(ValueError):
    """Raised when PPO statistics violate the frozen objective contract."""


def _finite_coefficient(
    value: float,
    name: str,
    *,
    strictly_positive: bool,
    less_than_one: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PPOObjectiveError(f"{name} must be a finite real scalar")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise PPOObjectiveError(f"{name} must be finite")
    if strictly_positive:
        if resolved <= 0.0:
            raise PPOObjectiveError(f"{name} must be positive")
    elif resolved < 0.0:
        raise PPOObjectiveError(f"{name} must be non-negative")
    if less_than_one and resolved >= 1.0:
        raise PPOObjectiveError(f"{name} must be less than one")
    return resolved


def _floating_tensor(tensor: Tensor, name: str, rank: int) -> Tensor:
    if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
        raise PPOObjectiveError(f"{name} must be a floating tensor")
    if tensor.ndim != rank:
        shape = "[T,A]" if rank == 2 else "[T]"
        raise PPOObjectiveError(f"{name} must have shape {shape}")
    if tensor.numel() == 0:
        raise PPOObjectiveError(f"{name} cannot be empty")
    if not torch.isfinite(tensor).all():
        raise PPOObjectiveError(f"{name} contains NaN or Inf")
    return tensor


def _sequence_mask(tensor: Tensor, time_steps: int, device: torch.device) -> Tensor:
    if not isinstance(tensor, Tensor) or tensor.dtype != torch.bool:
        raise PPOObjectiveError("sequence_valid_mask must be a boolean tensor")
    if tensor.ndim != 1 or tensor.shape[0] != time_steps:
        raise PPOObjectiveError("sequence_valid_mask must have shape [T]")
    if tensor.device != device:
        raise PPOObjectiveError("sequence_valid_mask must share the statistics device")
    valid_count = int(tensor.sum().item())
    if valid_count == 0:
        raise PPOObjectiveError("sequence_valid_mask has no valid positions")
    expected = torch.arange(time_steps, device=device) < valid_count
    if not torch.equal(tensor, expected):
        raise PPOObjectiveError(
            "sequence_valid_mask must describe a contiguous prefix with trailing padding"
        )
    return tensor


def _masked_mean(values: Tensor, mask: Tensor, name: str) -> Tensor:
    if values.shape != mask.shape:
        raise PPOObjectiveError(f"{name} values and mask must have identical shapes")
    selected = values.masked_select(mask)
    if selected.numel() == 0:
        raise PPOObjectiveError(f"{name} has no valid positions")
    result = selected.mean()
    if not torch.isfinite(result):
        raise PPOObjectiveError(f"{name} reduction produced NaN or Inf")
    return result


@dataclass(frozen=True)
class PPOObjectiveDiagnostics:
    """Detached scalar diagnostics for one PPO objective evaluation."""

    valid_timestep_count: int
    valid_actor_position_count: int
    ratio_mean: float
    ratio_min: float
    ratio_max: float
    clipped_fraction: float
    approx_kl: float
    surrogate_mean: float
    value_squared_error_mean: float
    entropy_mean: float
    per_head_critic_loss: tuple[float, ...] = ()
    same_timestep_advantage_equality_rate: float | None = None

    def __post_init__(self) -> None:
        if self.valid_timestep_count <= 0 or self.valid_actor_position_count <= 0:
            raise PPOObjectiveError("diagnostic valid counts must be positive")
        scalars = (
            self.ratio_mean,
            self.ratio_min,
            self.ratio_max,
            self.clipped_fraction,
            self.approx_kl,
            self.surrogate_mean,
            self.value_squared_error_mean,
            self.entropy_mean,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise PPOObjectiveError("PPO diagnostics contain NaN or Inf")
        if not 0.0 <= self.clipped_fraction <= 1.0:
            raise PPOObjectiveError("clipped_fraction must lie in [0, 1]")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in self.per_head_critic_loss
        ):
            raise PPOObjectiveError("per_head_critic_loss must be finite and non-negative")
        if self.same_timestep_advantage_equality_rate is not None and not (
            math.isfinite(self.same_timestep_advantage_equality_rate)
            and 0.0 <= self.same_timestep_advantage_equality_rate <= 1.0
        ):
            raise PPOObjectiveError(
                "same_timestep_advantage_equality_rate must lie in [0, 1]"
            )


@dataclass(frozen=True)
class CAGATMAPPOLossOutput:
    """Auditable PPO matrices, scalar losses, and detached diagnostics."""

    log_ratio: Tensor
    ratio: Tensor
    clipped_ratio: Tensor
    expanded_advantage: Tensor
    unclipped_surrogate: Tensor
    clipped_surrogate: Tensor
    surrogate: Tensor
    actor_valid_mask: Tensor
    critic_valid_mask: Tensor
    clipped_objective: Tensor
    actor_loss: Tensor
    critic_loss: Tensor
    entropy_mean: Tensor
    total_loss: Tensor
    epsilon_clip: float
    value_coefficient: float
    entropy_coefficient: float
    route_entropy_coefficient: float
    route_entropy_schedule_progress: float
    route_entropy_loss_contribution: Tensor
    other_branch_entropy_loss_contribution: Tensor
    global_entropy_loss_contribution: Tensor
    collected_environment_steps: int | None
    diagnostics: PPOObjectiveDiagnostics
    actor_ratio_mode: str = ActorRatioMode.JOINT.value
    agent_credit_mode: str = AgentCreditMode.TEAM.value
    branch_log_ratio: Tensor | None = None
    branch_ratio: Tensor | None = None
    branch_clipped_ratio: Tensor | None = None
    branch_unclipped_surrogate: Tensor | None = None
    branch_clipped_surrogate: Tensor | None = None
    branch_surrogate: Tensor | None = None
    branch_advantage: Tensor | None = None
    active_branch_indicators: Tensor | None = None
    active_branch_count: Tensor | None = None
    tx_upper_stability: TXUpperStabilityOutput | None = None
    tx_upper_objective_divisor: int = 1

    def __post_init__(self) -> None:
        matrices = (
            self.log_ratio,
            self.ratio,
            self.clipped_ratio,
            self.expanded_advantage,
            self.unclipped_surrogate,
            self.clipped_surrogate,
            self.surrogate,
        )
        if any(not isinstance(tensor, Tensor) or tensor.ndim != 2 for tensor in matrices):
            raise PPOObjectiveError("PPO actor statistics must be [T,A] tensors")
        matrix_shape = matrices[0].shape
        if any(tensor.shape != matrix_shape for tensor in matrices[1:]):
            raise PPOObjectiveError("PPO actor statistics must share one [T,A] shape")
        if self.actor_valid_mask.shape != matrix_shape or self.actor_valid_mask.dtype != torch.bool:
            raise PPOObjectiveError("actor_valid_mask must be boolean [T,A]")
        if (
            self.critic_valid_mask.ndim != 1
            or self.critic_valid_mask.shape[0] != matrix_shape[0]
            or self.critic_valid_mask.dtype != torch.bool
        ):
            raise PPOObjectiveError("critic_valid_mask must be boolean [T]")
        for tensor in matrices:
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise PPOObjectiveError("PPO actor statistics contain NaN or Inf")
        losses = (
            self.clipped_objective,
            self.actor_loss,
            self.critic_loss,
            self.entropy_mean,
            self.total_loss,
        )
        if any(
            not isinstance(tensor, Tensor)
            or tensor.ndim != 0
            or not torch.isfinite(tensor)
            for tensor in losses
        ):
            raise PPOObjectiveError("PPO objective outputs must be finite scalars")
        entropy_contributions = (
            self.route_entropy_loss_contribution,
            self.other_branch_entropy_loss_contribution,
            self.global_entropy_loss_contribution,
        )
        if any(
            not isinstance(tensor, Tensor)
            or tensor.ndim != 0
            or not torch.isfinite(tensor)
            or float(tensor.detach().cpu().item()) < -1.0e-8
            for tensor in entropy_contributions
        ):
            raise PPOObjectiveError(
                "entropy loss contributions must be finite non-negative scalars"
            )
        route_weight = _finite_coefficient(
            self.route_entropy_coefficient,
            "route_entropy_coefficient",
            strictly_positive=False,
        )
        if (
            not math.isfinite(self.route_entropy_schedule_progress)
            or not 0.0 <= self.route_entropy_schedule_progress <= 1.0
        ):
            raise PPOObjectiveError(
                "route_entropy_schedule_progress must lie in [0, 1]"
            )
        if self.collected_environment_steps is not None and (
            isinstance(self.collected_environment_steps, bool)
            or not isinstance(self.collected_environment_steps, int)
            or self.collected_environment_steps < 0
        ):
            raise PPOObjectiveError(
                "collected_environment_steps must be a non-negative integer or NA"
            )
        if not torch.allclose(
            self.global_entropy_loss_contribution,
            self.route_entropy_loss_contribution
            + self.other_branch_entropy_loss_contribution,
        ):
            raise PPOObjectiveError(
                "global entropy contribution must equal route plus other branches"
            )
        expected_total_loss = (
            self.actor_loss
            + self.value_coefficient * self.critic_loss
            - self.global_entropy_loss_contribution
        )
        if self.tx_upper_stability is not None:
            if self.tx_upper_objective_divisor < 1:
                raise PPOObjectiveError("invalid TX upper branch aggregation divisor")
            expected_total_loss = expected_total_loss + (
                TXUpperDeterministicStabilityPPOV1.MARGIN_WEIGHT
                * self.tx_upper_stability.margin_loss / self.tx_upper_objective_divisor)
        if not torch.allclose(self.total_loss, expected_total_loss):
            raise PPOObjectiveError(
                "total_loss must subtract the decomposed entropy contribution"
            )
        if route_weight != self.route_entropy_coefficient:
            raise PPOObjectiveError("route entropy coefficient normalization failed")
        if not torch.allclose(self.actor_loss, -self.clipped_objective):
            raise PPOObjectiveError("actor_loss must equal -clipped_objective")
        try:
            ratio_mode = ActorRatioMode(self.actor_ratio_mode)
        except (TypeError, ValueError) as exc:
            raise PPOObjectiveError("actor_ratio_mode is invalid") from exc
        try:
            AgentCreditMode(self.agent_credit_mode)
        except (TypeError, ValueError) as exc:
            raise PPOObjectiveError("agent_credit_mode is invalid") from exc
        branch_tensors = (
            self.branch_log_ratio,
            self.branch_ratio,
            self.branch_clipped_ratio,
            self.branch_unclipped_surrogate,
            self.branch_clipped_surrogate,
            self.branch_surrogate,
            self.branch_advantage,
        )
        if ratio_mode is ActorRatioMode.JOINT:
            if any(tensor is not None for tensor in branch_tensors) or any(
                tensor is not None
                for tensor in (
                    self.active_branch_indicators,
                    self.active_branch_count,
                )
            ):
                raise PPOObjectiveError(
                    "joint mode must not expose branch-specific objective tensors"
                )
            return
        if any(not isinstance(tensor, Tensor) or tensor.ndim != 3 for tensor in branch_tensors):
            raise PPOObjectiveError(
                "branch-specific objective statistics must be [T,A,7] tensors"
            )
        branch_shape = branch_tensors[0].shape
        if branch_shape[:2] != matrix_shape or branch_shape[-1] < 1:
            raise PPOObjectiveError(
                "branch-specific objective statistics must have shape [T,A,7]"
            )
        if any(tensor.shape != branch_shape for tensor in branch_tensors[1:]):
            raise PPOObjectiveError("branch-specific statistics must share one shape")
        if (
            not isinstance(self.active_branch_indicators, Tensor)
            or self.active_branch_indicators.shape != branch_shape
            or self.active_branch_indicators.dtype != torch.bool
        ):
            raise PPOObjectiveError(
                "active_branch_indicators must be boolean [T,A,7]"
            )
        if (
            not isinstance(self.active_branch_count, Tensor)
            or self.active_branch_count.shape != matrix_shape
        ):
            raise PPOObjectiveError("active_branch_count must have shape [T,A]")
        if any(
            not tensor.is_floating_point() or not torch.isfinite(tensor).all()
            for tensor in branch_tensors
        ):
            raise PPOObjectiveError("branch-specific statistics contain NaN or Inf")
        inactive = ~self.active_branch_indicators
        for tensor in (
            self.branch_unclipped_surrogate,
            self.branch_clipped_surrogate,
            self.branch_surrogate,
        ):
            if torch.any(tensor.masked_select(inactive) != 0.0):
                raise PPOObjectiveError(
                    "inactive branches must have zero surrogate contribution"
                )


def compute_ppo_objective_and_loss(
    *,
    new_joint_log_prob: Tensor,
    old_joint_log_prob: Tensor,
    advantage: Tensor,
    current_value: Tensor,
    return_target: Tensor,
    entropy: Tensor,
    sequence_valid_mask: Tensor,
    epsilon_clip: float,
    value_coefficient: float,
    entropy_coefficient: float,
    actor_ratio_mode: str = ActorRatioMode.JOINT.value,
    policy_loss_aggregation_mode: str = PolicyLossAggregationMode.ACTIVE_BRANCH_MEAN_V1.value,
    agent_credit_mode: str = AgentCreditMode.TEAM.value,
    new_branch_log_probs: Tensor | None = None,
    old_branch_log_probs: Tensor | None = None,
    active_branch_indicators: Tensor | None = None,
    route_entropy: Tensor | None = None,
    route_entropy_coefficient: float | None = None,
    route_entropy_schedule_progress: float = 0.0,
    collected_environment_steps: int | None = None,
    route_advantage: Tensor | None = None,
    policy_component_count: int = 7,
    tx_upper_stability: TXUpperStabilityOutput | None = None,
    tx_upper_component_index: int | None = None,
    tx_upper_advantage: Tensor | None = None,
    tx_upper_paired_mask: Tensor | None = None,
) -> CAGATMAPPOLossOutput:
    """Compute the frozen PPO objective from proposal-policy statistics.

    Policy inputs use explicit ``[T,A]`` axes.  The raw shared-team advantage,
    current centralized value, return target, and sequence mask use ``[T]``.
    Rollout quantities are detached fixed targets; gradients remain connected
    only to current log-probabilities, current values, and current entropy.
    """

    try:
        ratio_mode = ActorRatioMode(actor_ratio_mode)
    except (TypeError, ValueError) as exc:
        raise PPOObjectiveError(
            "actor_ratio_mode must be 'joint' or 'branch_specific'"
        ) from exc
    try:
        aggregation_mode = PolicyLossAggregationMode(policy_loss_aggregation_mode)
    except (TypeError, ValueError) as exc:
        raise PPOObjectiveError("policy_loss_aggregation_mode is invalid") from exc
    if (
        aggregation_mode is PolicyLossAggregationMode.BRANCHWISE_ACTIVE_MEAN_V1
        and ratio_mode is not ActorRatioMode.BRANCH_SPECIFIC
    ):
        raise PPOObjectiveError(
            "branchwise_active_mean_v1 requires branch_specific actor ratios"
        )
    if tx_upper_stability is not None and (
            aggregation_mode is not PolicyLossAggregationMode.BRANCHWISE_ACTIVE_MEAN_V1
            or tx_upper_component_index is None
            or not 0 <= tx_upper_component_index < policy_component_count):
        raise PPOObjectiveError("TX stability requires a valid branchwise upper component")
    try:
        credit_mode = AgentCreditMode(agent_credit_mode)
    except (TypeError, ValueError) as exc:
        raise PPOObjectiveError(
            "agent_credit_mode must be 'team' or 'role_decomposed'"
        ) from exc

    epsilon = _finite_coefficient(
        epsilon_clip,
        "epsilon_clip",
        strictly_positive=True,
        less_than_one=True,
    )
    value_weight = _finite_coefficient(
        value_coefficient,
        "value_coefficient",
        strictly_positive=False,
    )
    entropy_weight = _finite_coefficient(
        entropy_coefficient,
        "entropy_coefficient",
        strictly_positive=False,
    )
    route_entropy_weight = (
        entropy_weight
        if route_entropy_coefficient is None
        else _finite_coefficient(
            route_entropy_coefficient,
            "route_entropy_coefficient",
            strictly_positive=False,
        )
    )
    if (
        isinstance(route_entropy_schedule_progress, bool)
        or not isinstance(route_entropy_schedule_progress, (int, float))
        or not math.isfinite(float(route_entropy_schedule_progress))
        or not 0.0 <= float(route_entropy_schedule_progress) <= 1.0
    ):
        raise PPOObjectiveError(
            "route_entropy_schedule_progress must lie in [0, 1]"
        )
    schedule_progress = float(route_entropy_schedule_progress)
    if collected_environment_steps is not None and (
        isinstance(collected_environment_steps, bool)
        or not isinstance(collected_environment_steps, int)
        or collected_environment_steps < 0
    ):
        raise PPOObjectiveError(
            "collected_environment_steps must be a non-negative integer or NA"
        )

    new_log_prob = _floating_tensor(new_joint_log_prob, "new_joint_log_prob", 2)
    old_log_prob = _floating_tensor(old_joint_log_prob, "old_joint_log_prob", 2)
    credit_rank = 1 if credit_mode is AgentCreditMode.TEAM else 2
    raw_advantage = _floating_tensor(advantage, "advantage", credit_rank)
    raw_route_advantage = (
        None
        if route_advantage is None
        else _floating_tensor(route_advantage, "route_advantage", 2)
    )
    values = _floating_tensor(current_value, "current_value", credit_rank)
    targets = _floating_tensor(return_target, "return_target", credit_rank)
    active_entropy = _floating_tensor(entropy, "entropy", 2)
    active_route_entropy = (
        None
        if route_entropy is None
        else _floating_tensor(route_entropy, "route_entropy", 2)
    )

    time_steps, agent_count = new_log_prob.shape
    if time_steps == 0 or agent_count == 0:
        raise PPOObjectiveError("policy statistics cannot have empty time or agent axes")
    if old_log_prob.shape != new_log_prob.shape:
        raise PPOObjectiveError("new and old joint log-prob must share shape [T,A]")
    if active_entropy.shape != new_log_prob.shape:
        raise PPOObjectiveError("entropy must share the policy shape [T,A]")
    if (
        active_route_entropy is not None
        and active_route_entropy.shape != new_log_prob.shape
    ):
        raise PPOObjectiveError("route_entropy must share the policy shape [T,A]")
    if active_route_entropy is None and route_entropy_weight != entropy_weight:
        raise PPOObjectiveError(
            "a distinct route entropy coefficient requires route_entropy"
        )
    expected_credit_shape = (
        (time_steps,)
        if credit_mode is AgentCreditMode.TEAM
        else (time_steps, agent_count)
    )
    for name, tensor in (
        ("advantage", raw_advantage),
        ("current_value", values),
        ("return_target", targets),
    ):
        if tensor.shape != expected_credit_shape:
            raise PPOObjectiveError(
                f"{name} shape differs from agent_credit_mode"
            )
    if raw_route_advantage is not None:
        if ratio_mode is not ActorRatioMode.BRANCH_SPECIFIC:
            raise PPOObjectiveError(
                "route_advantage requires branch_specific actor ratios"
            )
        if credit_mode is not AgentCreditMode.ROLE_DECOMPOSED:
            raise PPOObjectiveError(
                "route_advantage requires role_decomposed agent credit"
            )
        if raw_route_advantage.shape != (time_steps, agent_count):
            raise PPOObjectiveError("route_advantage must have shape [T,A]")
    floating_inputs = (
        old_log_prob,
        raw_advantage,
        values,
        targets,
        active_entropy,
    )
    if active_route_entropy is not None:
        floating_inputs = (*floating_inputs, active_route_entropy)
    if raw_route_advantage is not None:
        floating_inputs = (*floating_inputs, raw_route_advantage)
    if any(tensor.device != new_log_prob.device for tensor in floating_inputs):
        raise PPOObjectiveError("all PPO statistics must share one device")
    if any(tensor.dtype != new_log_prob.dtype for tensor in floating_inputs):
        raise PPOObjectiveError("all PPO statistics must share one floating dtype")

    critic_valid_mask = _sequence_mask(
        sequence_valid_mask,
        time_steps,
        new_log_prob.device,
    )
    actor_valid_mask = critic_valid_mask.reshape(time_steps, 1).expand(
        time_steps,
        agent_count,
    )

    fixed_old_log_prob = old_log_prob.detach()
    fixed_advantage = raw_advantage.detach()
    fixed_return_target = targets.detach()
    expanded_advantage = (
        fixed_advantage.reshape(time_steps, 1).expand(time_steps, agent_count)
        if credit_mode is AgentCreditMode.TEAM
        else fixed_advantage
    )
    branch_log_ratio = None
    branch_ratio = None
    branch_clipped_ratio = None
    branch_unclipped_surrogate = None
    branch_clipped_surrogate = None
    branch_surrogate = None
    branch_advantage = None
    fixed_active_branches = None
    active_branch_count = None
    if ratio_mode is ActorRatioMode.JOINT:
        # Keep this path operation-for-operation equivalent to the pre-Fix-4
        # implementation.  Optional branch inputs are deliberately ignored.
        log_ratio = new_log_prob - fixed_old_log_prob
        if not torch.isfinite(log_ratio).all():
            raise PPOObjectiveError("new-old log-prob difference contains NaN or Inf")
        ratio = torch.exp(log_ratio)
        if not torch.isfinite(ratio).all():
            raise PPOObjectiveError("exp(log-ratio) produced NaN or Inf")
        clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
        unclipped_surrogate = ratio * expanded_advantage
        clipped_surrogate = clipped_ratio * expanded_advantage
        surrogate = torch.minimum(unclipped_surrogate, clipped_surrogate)
        actor_diagnostic_mask = actor_valid_mask
    else:
        for name, tensor in (
            ("new_branch_log_probs", new_branch_log_probs),
            ("old_branch_log_probs", old_branch_log_probs),
        ):
            if (
                not isinstance(tensor, Tensor)
                or not tensor.is_floating_point()
                or tuple(tensor.shape) != (time_steps, agent_count, policy_component_count)
            ):
                raise PPOObjectiveError(f"{name} must have shape [T,A,7]")
            if tensor.device != new_log_prob.device or tensor.dtype != new_log_prob.dtype:
                raise PPOObjectiveError(
                    f"{name} must share the PPO statistics device and dtype"
                )
            if not torch.isfinite(tensor).all():
                raise PPOObjectiveError(f"{name} contains NaN or Inf")
        if (
            not isinstance(active_branch_indicators, Tensor)
            or active_branch_indicators.dtype != torch.bool
            or tuple(active_branch_indicators.shape)
            != (time_steps, agent_count, policy_component_count)
        ):
            raise PPOObjectiveError(
                "active_branch_indicators must be boolean [T,A,7]"
            )
        if active_branch_indicators.device != new_log_prob.device:
            raise PPOObjectiveError(
                "active_branch_indicators must share the PPO statistics device"
            )
        assert new_branch_log_probs is not None
        assert old_branch_log_probs is not None
        fixed_active_branches = active_branch_indicators.detach()
        if tx_upper_paired_mask is not None:
            if (tx_upper_advantage is None or tx_upper_component_index is None
                    or tx_upper_paired_mask.shape != new_log_prob.shape
                    or tx_upper_paired_mask.dtype != torch.bool
                    or tx_upper_paired_mask.device != new_log_prob.device
                    or (tx_upper_paired_mask & ~fixed_active_branches[..., tx_upper_component_index]).any()
                    or (tx_upper_paired_mask & ~actor_valid_mask).any()):
                raise PPOObjectiveError("invalid explicit TX paired mask")
            fixed_active_branches = fixed_active_branches.clone()
            fixed_active_branches[..., tx_upper_component_index] &= tx_upper_paired_mask.detach()
        raw_branch_log_ratio = (
            new_branch_log_probs - old_branch_log_probs.detach()
        )
        selected_log_ratio = raw_branch_log_ratio.masked_select(
            fixed_active_branches
        )
        if selected_log_ratio.numel() == 0:
            raise PPOObjectiveError("branch-specific objective has no active branches")
        if not torch.isfinite(selected_log_ratio).all():
            raise PPOObjectiveError("active branch log-ratio contains NaN or Inf")
        selected_ratio = torch.exp(selected_log_ratio)
        if not torch.isfinite(selected_ratio).all():
            raise PPOObjectiveError("active branch exp(log-ratio) produced NaN or Inf")
        branch_log_ratio = torch.zeros_like(raw_branch_log_ratio).masked_scatter(
            fixed_active_branches,
            selected_log_ratio,
        )
        branch_ratio = torch.ones_like(raw_branch_log_ratio).masked_scatter(
            fixed_active_branches,
            selected_ratio,
        )
        branch_clipped_ratio = torch.clamp(
            branch_ratio,
            1.0 - epsilon,
            1.0 + epsilon,
        )
        if raw_route_advantage is None:
            branch_advantage = expanded_advantage.unsqueeze(-1).expand_as(
                branch_ratio
            )
        else:
            fixed_route_advantage = raw_route_advantage.detach()
            branch_advantage = torch.cat(
                (
                    fixed_route_advantage.unsqueeze(-1),
                    expanded_advantage.unsqueeze(-1).expand_as(branch_ratio)[
                        ..., 1:
                    ],
                ),
                dim=-1,
            )
        if tx_upper_stability is not None:
            if tx_upper_component_index is None:
                raise PPOObjectiveError("missing TX upper component index")
            branch_advantage = branch_advantage.clone()
            branch_advantage[..., tx_upper_component_index] = tx_upper_stability.normalized_advantage
        if tx_upper_advantage is not None:
            if (tx_upper_stability is not None or tx_upper_component_index is None
                    or not 0 <= tx_upper_component_index < policy_component_count
                    or tx_upper_advantage.shape != expanded_advantage.shape
                    or not torch.isfinite(tx_upper_advantage if tx_upper_paired_mask is None
                                          else tx_upper_advantage[tx_upper_paired_mask]).all()):
                raise PPOObjectiveError("invalid/conflicting TX branch advantage")
            branch_advantage = branch_advantage.clone()
            # Arithmetic padding only; explicit activity excludes absent labels
            # from surrogate, count and denominator. Credit storage remains NA.
            branch_advantage[..., tx_upper_component_index] = (
                tx_upper_advantage.detach() if tx_upper_paired_mask is None else
                torch.where(tx_upper_paired_mask, tx_upper_advantage.detach(), torch.zeros_like(tx_upper_advantage)))
        branch_unclipped_surrogate = torch.where(
            fixed_active_branches,
            branch_ratio * branch_advantage,
            torch.zeros_like(branch_ratio),
        )
        branch_clipped_surrogate = torch.where(
            fixed_active_branches,
            branch_clipped_ratio * branch_advantage,
            torch.zeros_like(branch_ratio),
        )
        branch_surrogate = torch.minimum(
            branch_unclipped_surrogate,
            branch_clipped_surrogate,
        )
        active_branch_count = fixed_active_branches.sum(dim=-1)
        denominator = active_branch_count.clamp_min(1).to(dtype=new_log_prob.dtype)
        log_ratio = branch_log_ratio.sum(dim=-1) / denominator
        ratio = (
            branch_ratio.masked_fill(~fixed_active_branches, 0.0).sum(dim=-1)
            / denominator
        )
        clipped_ratio = (
            branch_clipped_ratio.masked_fill(
                ~fixed_active_branches, 0.0
            ).sum(dim=-1)
            / denominator
        )
        unclipped_surrogate = branch_unclipped_surrogate.sum(dim=-1) / denominator
        clipped_surrogate = branch_clipped_surrogate.sum(dim=-1) / denominator
        surrogate = branch_surrogate.sum(dim=-1) / denominator
        actor_diagnostic_mask = (
            actor_valid_mask.unsqueeze(-1).expand_as(fixed_active_branches)
            & fixed_active_branches
        )
    if not torch.isfinite(unclipped_surrogate).all() or not torch.isfinite(
        clipped_surrogate
    ).all():
        raise PPOObjectiveError("PPO surrogate produced NaN or Inf")
    if aggregation_mode is PolicyLossAggregationMode.BRANCHWISE_ACTIVE_MEAN_V1:
        assert branch_surrogate is not None
        assert fixed_active_branches is not None
        valid_active = fixed_active_branches & actor_valid_mask.unsqueeze(-1)
        branch_counts = valid_active.sum(dim=(0, 1))
        nonempty = branch_counts > 0
        if not bool(nonempty.any()):
            raise PPOObjectiveError("branchwise objective has no active valid branches")
        branch_totals = branch_surrogate.masked_fill(~valid_active, 0.0).sum(
            dim=(0, 1)
        )
        if tx_upper_paired_mask is None:
            clipped_objective = (branch_totals[nonempty] / branch_counts[nonempty]).mean()
        else:
            # Preserve every other component's original aggregation weight,
            # even when the TX paired cohort is empty in this rollout.
            original_counts = (active_branch_indicators & actor_valid_mask.unsqueeze(-1)).sum(dim=(0, 1))
            clipped_objective = (branch_totals[nonempty] / branch_counts[nonempty]).sum() / (original_counts > 0).sum()
    else:
        # Preserve the historical reduction operation-for-operation.
        clipped_objective = _masked_mean(
            surrogate,
            actor_valid_mask,
            "clipped_objective",
        )
    actor_loss = -clipped_objective
    tx_upper_divisor = 1 if tx_upper_stability is None else int(nonempty.sum().item())

    value_squared_error = (values - fixed_return_target).square()
    critic_element_mask = (
        critic_valid_mask
        if credit_mode is AgentCreditMode.TEAM
        else critic_valid_mask.reshape(time_steps, 1).expand(
            time_steps, agent_count
        )
    )
    critic_loss = 0.5 * _masked_mean(
        value_squared_error,
        critic_element_mask,
        "critic_loss",
    )
    entropy_mean = _masked_mean(active_entropy, actor_valid_mask, "entropy_mean")
    route_entropy_mean = (
        entropy_mean.new_zeros(())
        if active_route_entropy is None
        else _masked_mean(
            active_route_entropy,
            actor_valid_mask,
            "route_entropy_mean",
        )
    )
    other_branch_entropy_mean = entropy_mean - route_entropy_mean
    if tx_upper_stability is not None:
        other_branch_entropy_mean = _masked_mean(
            tx_upper_stability.entropy_without_upper, actor_valid_mask,
            "entropy_without_tx_upper") - route_entropy_mean
    route_entropy_loss_contribution = (
        route_entropy_weight * route_entropy_mean
    )
    other_branch_entropy_loss_contribution = (
        entropy_weight * other_branch_entropy_mean
    )
    if tx_upper_stability is not None:
        other_branch_entropy_loss_contribution = other_branch_entropy_loss_contribution + (
            tx_upper_stability.beta * tx_upper_stability.entropy_mean / tx_upper_divisor)
    global_entropy_loss_contribution = (
        route_entropy_loss_contribution
        + other_branch_entropy_loss_contribution
    )
    if tx_upper_stability is not None:
        total_loss = (actor_loss + value_weight * critic_loss - global_entropy_loss_contribution
                      + TXUpperDeterministicStabilityPPOV1.MARGIN_WEIGHT
                      * tx_upper_stability.margin_loss / tx_upper_divisor)
    elif route_entropy_weight == entropy_weight:
        # Preserve the legacy disabled/base-coefficient loss operation exactly.
        total_loss = (
            actor_loss
            + value_weight * critic_loss
            - entropy_weight * entropy_mean
        )
    else:
        total_loss = (
            actor_loss
            + value_weight * critic_loss
            - global_entropy_loss_contribution
        )
    if not torch.isfinite(total_loss):
        raise PPOObjectiveError("total_loss produced NaN or Inf")

    if ratio_mode is ActorRatioMode.JOINT:
        valid_ratios = ratio.masked_select(actor_diagnostic_mask).detach()
        valid_log_ratios = log_ratio.masked_select(actor_diagnostic_mask).detach()
        clipped_positions = (ratio != clipped_ratio) & actor_diagnostic_mask
        diagnostic_denominator = actor_diagnostic_mask.sum()
    else:
        assert branch_ratio is not None
        assert branch_log_ratio is not None
        assert branch_clipped_ratio is not None
        valid_ratios = branch_ratio.masked_select(actor_diagnostic_mask).detach()
        valid_log_ratios = branch_log_ratio.masked_select(
            actor_diagnostic_mask
        ).detach()
        clipped_positions = (
            (branch_ratio != branch_clipped_ratio) & actor_diagnostic_mask
        )
        diagnostic_denominator = actor_diagnostic_mask.sum()
    valid_surrogate = surrogate.masked_select(actor_valid_mask).detach()
    valid_value_error = value_squared_error.masked_select(critic_element_mask).detach()
    per_head_critic_loss: tuple[float, ...] = ()
    advantage_equality_rate = None
    if credit_mode is AgentCreditMode.ROLE_DECOMPOSED:
        per_head_critic_loss = tuple(
            float(
                (
                    0.5
                    * value_squared_error[:, agent_index]
                    .masked_select(critic_valid_mask)
                    .mean()
                )
                .detach()
                .cpu()
                .item()
            )
            for agent_index in range(agent_count)
        )
        valid_advantage = fixed_advantage[critic_valid_mask]
        equality = torch.isclose(
            valid_advantage,
            valid_advantage[:, :1],
            rtol=0.0,
            atol=1.0e-7,
        ).all(dim=-1)
        advantage_equality_rate = float(
            equality.to(dtype=torch.float32).mean().cpu().item()
        )
    diagnostics = PPOObjectiveDiagnostics(
        valid_timestep_count=int(critic_valid_mask.sum().item()),
        valid_actor_position_count=int(actor_valid_mask.sum().item()),
        ratio_mean=float(valid_ratios.mean().cpu().item()),
        ratio_min=float(valid_ratios.min().cpu().item()),
        ratio_max=float(valid_ratios.max().cpu().item()),
        clipped_fraction=float(
            clipped_positions.sum().to(dtype=torch.float32).div(
                diagnostic_denominator
            ).cpu().item()
        ),
        approx_kl=float(
            ((valid_ratios - 1.0) - valid_log_ratios).mean().cpu().item()
        ),
        surrogate_mean=(
            float(clipped_objective.detach().cpu().item())
            if aggregation_mode is PolicyLossAggregationMode.BRANCHWISE_ACTIVE_MEAN_V1
            else float(valid_surrogate.mean().cpu().item())
        ),
        value_squared_error_mean=float(valid_value_error.mean().cpu().item()),
        entropy_mean=float(entropy_mean.detach().cpu().item()),
        per_head_critic_loss=per_head_critic_loss,
        same_timestep_advantage_equality_rate=advantage_equality_rate,
    )

    return CAGATMAPPOLossOutput(
        log_ratio=log_ratio,
        ratio=ratio,
        clipped_ratio=clipped_ratio,
        expanded_advantage=expanded_advantage,
        unclipped_surrogate=unclipped_surrogate,
        clipped_surrogate=clipped_surrogate,
        surrogate=surrogate,
        actor_valid_mask=actor_valid_mask,
        critic_valid_mask=critic_valid_mask,
        clipped_objective=clipped_objective,
        actor_loss=actor_loss,
        critic_loss=critic_loss,
        entropy_mean=entropy_mean,
        total_loss=total_loss,
        epsilon_clip=epsilon,
        value_coefficient=value_weight,
        entropy_coefficient=entropy_weight,
        route_entropy_coefficient=route_entropy_weight,
        route_entropy_schedule_progress=schedule_progress,
        route_entropy_loss_contribution=route_entropy_loss_contribution,
        other_branch_entropy_loss_contribution=(
            other_branch_entropy_loss_contribution
        ),
        global_entropy_loss_contribution=global_entropy_loss_contribution,
        collected_environment_steps=collected_environment_steps,
        diagnostics=diagnostics,
        actor_ratio_mode=ratio_mode.value,
        agent_credit_mode=credit_mode.value,
        branch_log_ratio=branch_log_ratio,
        branch_ratio=branch_ratio,
        branch_clipped_ratio=branch_clipped_ratio,
        branch_unclipped_surrogate=branch_unclipped_surrogate,
        branch_clipped_surrogate=branch_clipped_surrogate,
        branch_surrogate=branch_surrogate,
        branch_advantage=branch_advantage,
        active_branch_indicators=fixed_active_branches,
        active_branch_count=active_branch_count,
        tx_upper_stability=tx_upper_stability,
        tx_upper_objective_divisor=tx_upper_divisor,
    )


def compute_configured_ppo_objective_and_loss(
    *,
    new_joint_log_prob: Tensor,
    old_joint_log_prob: Tensor,
    advantage: Tensor,
    current_value: Tensor,
    return_target: Tensor,
    entropy: Tensor,
    sequence_valid_mask: Tensor,
    config: RunConfig,
    new_branch_log_probs: Tensor | None = None,
    old_branch_log_probs: Tensor | None = None,
    active_branch_indicators: Tensor | None = None,
    route_entropy: Tensor | None = None,
    collected_environment_steps: int | None = None,
    route_advantage: Tensor | None = None,
    tx_upper_logit_gap: Tensor | None = None,
    tx_upper_transmitted: Tensor | None = None,
    tx_upper_entropy: Tensor | None = None,
    entropy_without_tx_upper: Tensor | None = None,
    tx_upper_advantage: Tensor | None = None,
    tx_upper_paired_mask: Tensor | None = None,
) -> CAGATMAPPOLossOutput:
    """Use the Section 4 values already centralized in ``RunConfig``."""

    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    config.validate()
    mappo = config.training.mappo
    route_credit_mode = RouteCreditMode(mappo.route_credit_mode)
    if route_credit_mode in {
        RouteCreditMode.ROUTE_SPECIFIC_GAE,
        RouteCreditMode.ROLLOUT_CAPPED_ROUTE_EVENT_NSTEP,
        RouteCreditMode.TASK_EVENT_VALUE_V1,
    }:
        if route_advantage is None:
            raise PPOObjectiveError(
                f"{route_credit_mode.value} requires route_advantage"
            )
    elif route_advantage is not None:
        raise PPOObjectiveError(
            "shared_gae must not receive route_advantage"
        )
    if mappo.route_entropy_step_schedule_enabled:
        if route_entropy is None:
            raise PPOObjectiveError(
                "enabled route entropy schedule requires route_entropy"
            )
        if collected_environment_steps is None:
            raise PPOObjectiveError(
                "enabled route entropy schedule requires collected_environment_steps"
            )
    schedule = compute_effective_route_entropy_schedule(
        mappo,
        0
        if collected_environment_steps is None
        else collected_environment_steps,
    )
    from .integrated_scheduler import policy_component_names
    tx_stability = None
    tx_index = None
    if mappo.tx_upper_credit_mode in {"counterfactual_q_v1", "factual_team_gae_v1", "paired_shadow_v1"}:
        if tx_upper_advantage is None:
            raise PPOObjectiveError(f"{mappo.tx_upper_credit_mode} requires explicit TX advantage")
        tx_index = policy_component_names(config).index("tx_upper")
    elif tx_upper_advantage is not None:
        raise PPOObjectiveError("shared_agent_advantage must not receive separate TX credit")
    if (tx_upper_paired_mask is None) != (mappo.tx_upper_credit_mode != "paired_shadow_v1"):
        raise PPOObjectiveError("paired TX credit requires its explicit mask; legacy modes must not receive one")
    if mappo.tx_upper_optimization_mode == "deterministic_stability_v1":
        if collected_environment_steps is None:
            raise PPOObjectiveError("TX deterministic stability requires collected_environment_steps")
        if any(x is None for x in (tx_upper_logit_gap, tx_upper_transmitted,
                                    tx_upper_entropy, entropy_without_tx_upper)):
            raise PPOObjectiveError("TX deterministic stability requires explicit binary statistics")
        if active_branch_indicators is None:
            raise PPOObjectiveError("TX deterministic stability requires logical active indicators")
        tx_index = policy_component_names(config).index("tx_upper")
        valid = _sequence_mask(sequence_valid_mask, new_joint_log_prob.shape[0], new_joint_log_prob.device)
        active = active_branch_indicators[..., tx_index] & valid.unsqueeze(-1)
        expanded = (advantage.unsqueeze(-1).expand_as(new_joint_log_prob)
                    if advantage.ndim == 1 else advantage)
        try:
            tx_stability = TXUpperDeterministicStabilityPPOV1.compute(
                advantage=expanded, active=active, gap=tx_upper_logit_gap,
                transmitted=tx_upper_transmitted, entropy=tx_upper_entropy,
                entropy_without_upper=entropy_without_tx_upper,
                base_entropy_coefficient=mappo.entropy_coefficient,
                steps=collected_environment_steps, budget=mappo.max_training_environment_steps)
        except ValueError as exc:
            raise PPOObjectiveError(str(exc)) from exc
    return compute_ppo_objective_and_loss(
        new_joint_log_prob=new_joint_log_prob,
        old_joint_log_prob=old_joint_log_prob,
        advantage=advantage,
        current_value=current_value,
        return_target=return_target,
        entropy=entropy,
        sequence_valid_mask=sequence_valid_mask,
        policy_component_count=len(policy_component_names(config)),
        epsilon_clip=mappo.ppo_clip_epsilon,
        value_coefficient=mappo.value_coefficient,
        entropy_coefficient=mappo.entropy_coefficient,
        actor_ratio_mode=mappo.actor_ratio_mode,
        policy_loss_aggregation_mode=mappo.policy_loss_aggregation_mode,
        agent_credit_mode=mappo.agent_credit_mode,
        new_branch_log_probs=new_branch_log_probs,
        old_branch_log_probs=old_branch_log_probs,
        active_branch_indicators=active_branch_indicators,
        route_entropy=route_entropy,
        route_entropy_coefficient=schedule.coefficient,
        route_entropy_schedule_progress=schedule.progress,
        collected_environment_steps=collected_environment_steps,
        route_advantage=route_advantage,
        tx_upper_stability=tx_stability,
        tx_upper_component_index=tx_index,
        tx_upper_advantage=tx_upper_advantage,
        tx_upper_paired_mask=tx_upper_paired_mask,
    )


__all__ = [
    "CAGATMAPPOLossOutput",
    "PPOObjectiveDiagnostics",
    "PPOObjectiveError",
    "compute_configured_ppo_objective_and_loss",
    "compute_ppo_objective_and_loss",
]
