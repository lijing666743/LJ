"""Boundary-aware GAE and return targets for CA-GAT-MAPPO.

This module consumes rollout-time value snapshots only.  It does not implement
advantage normalization, PPO objectives, losses, optimizers, or training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..config import AgentCreditMode, RunConfig
from .ca_gat_mappo_rollout import CAGATMAPPORolloutChunk


class GAEComputationError(ValueError):
    """Raised when rollout statistics violate the frozen GAE contract."""


def _coefficient(value: float, name: str, *, include_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GAEComputationError(f"{name} must be a finite real scalar")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise GAEComputationError(f"{name} must be finite")
    lower_valid = resolved >= 0.0 if include_zero else resolved > 0.0
    upper_valid = resolved <= 1.0 if include_zero else resolved < 1.0
    if not (lower_valid and upper_valid):
        interval = "[0, 1]" if include_zero else "(0, 1)"
        raise GAEComputationError(f"{name} must lie in {interval}")
    return resolved


def _float_vector(tensor: Tensor, name: str) -> Tensor:
    if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
        raise GAEComputationError(f"{name} must be a floating tensor")
    if tensor.ndim != 1:
        raise GAEComputationError(f"{name} must have shape [T]")
    if tensor.numel() == 0:
        raise GAEComputationError("rollout cannot be empty")
    if not torch.isfinite(tensor).all():
        raise GAEComputationError(f"{name} contains NaN or Inf")
    return tensor.detach().to(device="cpu", dtype=torch.float32).clone().contiguous()


def _bool_vector(tensor: Tensor, name: str) -> Tensor:
    if not isinstance(tensor, Tensor) or tensor.dtype != torch.bool:
        raise GAEComputationError(f"{name} must be a boolean tensor")
    if tensor.ndim != 1:
        raise GAEComputationError(f"{name} must have shape [T]")
    if tensor.numel() == 0:
        raise GAEComputationError("rollout cannot be empty")
    return tensor.detach().to(device="cpu").clone().contiguous()


def _float_matrix(tensor: Tensor, name: str) -> Tensor:
    if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
        raise GAEComputationError(f"{name} must be a floating tensor")
    if tensor.ndim != 2:
        raise GAEComputationError(f"{name} must have shape [T,A]")
    if tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise GAEComputationError("rollout cannot have empty time or agent axes")
    if not torch.isfinite(tensor).all():
        raise GAEComputationError(f"{name} contains NaN or Inf")
    return tensor.detach().to(device="cpu", dtype=torch.float32).clone().contiguous()


@dataclass(frozen=True)
class CAGATMAPPOGAEOutput:
    """Raw shared-team GAE statistics with one scalar per rollout position."""

    bootstrap_mask: Tensor
    td_residual: Tensor
    advantage: Tensor
    return_target: Tensor
    sequence_mask: Tensor

    def __post_init__(self) -> None:
        fields = {
            "bootstrap_mask": self.bootstrap_mask,
            "td_residual": self.td_residual,
            "advantage": self.advantage,
            "return_target": self.return_target,
            "sequence_mask": self.sequence_mask,
        }
        if any(not isinstance(tensor, Tensor) for tensor in fields.values()):
            raise GAEComputationError("GAE outputs must be tensors")
        shapes = {tuple(tensor.shape) for tensor in fields.values()}
        if len(shapes) != 1 or len(next(iter(shapes))) != 1:
            raise GAEComputationError("GAE outputs must share one [T] shape")
        for name in ("bootstrap_mask", "sequence_mask"):
            tensor = fields[name]
            if tensor.dtype != torch.bool or tensor.device.type != "cpu":
                raise GAEComputationError(f"{name} must use CPU bool storage")
        for name in ("td_residual", "advantage", "return_target"):
            tensor = fields[name]
            if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
                raise GAEComputationError(f"{name} must use CPU float32 storage")
            if not torch.isfinite(tensor).all():
                raise GAEComputationError(f"{name} contains NaN or Inf")
        if torch.any(self.bootstrap_mask & ~self.sequence_mask):
            raise GAEComputationError("padding positions cannot enable bootstrap")


@dataclass(frozen=True)
class CAGATMAPPOPerAgentGAEOutput:
    """Role-decomposed GAE with shared [T] boundary masks and [T,A] values."""

    bootstrap_mask: Tensor
    td_residual: Tensor
    advantage: Tensor
    return_target: Tensor
    sequence_mask: Tensor

    def __post_init__(self) -> None:
        for name in ("bootstrap_mask", "sequence_mask"):
            tensor = getattr(self, name)
            if (
                not isinstance(tensor, Tensor)
                or tensor.ndim != 1
                or tensor.dtype != torch.bool
                or tensor.device.type != "cpu"
            ):
                raise GAEComputationError(f"{name} must be CPU bool [T]")
        numeric_shape = None
        for name in ("td_residual", "advantage", "return_target"):
            tensor = getattr(self, name)
            if (
                not isinstance(tensor, Tensor)
                or tensor.ndim != 2
                or tensor.dtype != torch.float32
                or tensor.device.type != "cpu"
                or not torch.isfinite(tensor).all()
            ):
                raise GAEComputationError(f"{name} must be finite CPU float32 [T,A]")
            numeric_shape = tuple(tensor.shape) if numeric_shape is None else numeric_shape
            if tuple(tensor.shape) != numeric_shape:
                raise GAEComputationError("per-agent GAE values must share [T,A]")
        assert numeric_shape is not None
        if numeric_shape[0] != self.sequence_mask.shape[0]:
            raise GAEComputationError("GAE mask and value time axes differ")
        if torch.any(self.bootstrap_mask & ~self.sequence_mask):
            raise GAEComputationError("padding positions cannot enable bootstrap")


def compute_gae_and_returns(
    *,
    reward: Tensor,
    old_value: Tensor,
    bootstrap_value: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    episode_boundary: Tensor,
    bootstrap_allowed: Tensor,
    gamma: float,
    gae_lambda: float,
    sequence_mask: Tensor | None = None,
) -> CAGATMAPPOGAEOutput:
    """Compute raw shared-team GAE and return targets in true time order.

    ``bootstrap_value[t]`` is the rollout-time snapshot of ``V(s_{t+1})`` for
    non-boundary transitions.  Boundary entries may contain any finite masked
    placeholder; they never contribute to the TD residual or recursion.
    """

    resolved_gamma = _coefficient(gamma, "gamma", include_zero=False)
    resolved_lambda = _coefficient(
        gae_lambda,
        "gae_lambda",
        include_zero=True,
    )
    rewards = _float_vector(reward, "reward")
    values = _float_vector(old_value, "old_value")
    next_values = _float_vector(bootstrap_value, "bootstrap_value")
    terminated_mask = _bool_vector(terminated, "terminated")
    truncated_mask = _bool_vector(truncated, "truncated")
    boundaries = _bool_vector(episode_boundary, "episode_boundary")
    allowed = _bool_vector(bootstrap_allowed, "bootstrap_allowed")
    if sequence_mask is None:
        valid = torch.ones_like(boundaries)
    else:
        valid = _bool_vector(sequence_mask, "sequence_mask")

    vectors = (
        values,
        next_values,
        terminated_mask,
        truncated_mask,
        boundaries,
        allowed,
        valid,
    )
    if any(tuple(vector.shape) != tuple(rewards.shape) for vector in vectors):
        raise GAEComputationError("all GAE inputs must share one [T] shape")
    valid_length = int(valid.sum().item())
    if valid_length == 0:
        raise GAEComputationError("rollout sequence has no valid positions")
    expected_valid = torch.arange(valid.numel()) < valid_length
    if not torch.equal(valid, expected_valid):
        raise GAEComputationError("sequence_mask must describe one contiguous prefix")

    padding = ~valid
    flags = terminated_mask | truncated_mask | boundaries | allowed
    if torch.any(flags & padding):
        raise GAEComputationError("padding positions must not carry boundary flags")
    if torch.any(terminated_mask & truncated_mask & valid):
        raise GAEComputationError("terminated and truncated must remain distinct")
    expected_boundaries = terminated_mask | truncated_mask
    if not torch.equal(boundaries[valid], expected_boundaries[valid]):
        raise GAEComputationError(
            "episode_boundary must equal terminated or truncated"
        )
    if not torch.equal(allowed[valid], ~boundaries[valid]):
        raise GAEComputationError(
            "bootstrap is allowed exactly on non-boundary transitions"
        )

    bootstrap_mask = valid & allowed & ~boundaries
    bootstrap_float = bootstrap_mask.to(dtype=torch.float32)
    raw_residual = (
        rewards
        + resolved_gamma * bootstrap_float * next_values
        - values
    )
    td_residual = torch.where(valid, raw_residual, torch.zeros_like(raw_residual))

    advantage = torch.zeros_like(td_residual)
    next_advantage = torch.zeros((), dtype=torch.float32)
    recurrence_scale = resolved_gamma * resolved_lambda
    for index in range(valid_length - 1, -1, -1):
        advantage[index] = (
            td_residual[index]
            + recurrence_scale
            * bootstrap_float[index]
            * next_advantage
        )
        next_advantage = advantage[index]

    return_target = torch.where(
        valid,
        advantage + values,
        torch.zeros_like(advantage),
    )
    for name, tensor in (
        ("td_residual", td_residual),
        ("advantage", advantage),
        ("return_target", return_target),
    ):
        if not torch.isfinite(tensor).all():
            raise GAEComputationError(f"{name} contains NaN or Inf")

    return CAGATMAPPOGAEOutput(
        bootstrap_mask=bootstrap_mask,
        td_residual=td_residual,
        advantage=advantage,
        return_target=return_target,
        sequence_mask=valid,
    )


def compute_per_agent_gae_and_returns(
    *,
    reward: Tensor,
    old_value: Tensor,
    bootstrap_value: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    episode_boundary: Tensor,
    bootstrap_allowed: Tensor,
    gamma: float,
    gae_lambda: float,
    sequence_mask: Tensor | None = None,
) -> CAGATMAPPOPerAgentGAEOutput:
    """Compute deterministic per-agent GAE without silent broadcasting."""

    resolved_gamma = _coefficient(gamma, "gamma", include_zero=False)
    resolved_lambda = _coefficient(gae_lambda, "gae_lambda", include_zero=True)
    rewards = _float_matrix(reward, "reward")
    values = _float_matrix(old_value, "old_value")
    next_values = _float_matrix(bootstrap_value, "bootstrap_value")
    if values.shape != rewards.shape or next_values.shape != rewards.shape:
        raise GAEComputationError(
            "reward, old_value, and bootstrap_value must share [T,A]"
        )
    terminated_mask = _bool_vector(terminated, "terminated")
    truncated_mask = _bool_vector(truncated, "truncated")
    boundaries = _bool_vector(episode_boundary, "episode_boundary")
    allowed = _bool_vector(bootstrap_allowed, "bootstrap_allowed")
    valid = (
        torch.ones_like(boundaries)
        if sequence_mask is None
        else _bool_vector(sequence_mask, "sequence_mask")
    )
    time_steps = rewards.shape[0]
    flags = (terminated_mask, truncated_mask, boundaries, allowed, valid)
    if any(tensor.shape != (time_steps,) for tensor in flags):
        raise GAEComputationError("boundary inputs must align with the [T,A] time axis")
    valid_length = int(valid.sum().item())
    if valid_length == 0:
        raise GAEComputationError("rollout sequence has no valid positions")
    expected_valid = torch.arange(time_steps) < valid_length
    if not torch.equal(valid, expected_valid):
        raise GAEComputationError("sequence_mask must describe one contiguous prefix")
    padding = ~valid
    combined_flags = terminated_mask | truncated_mask | boundaries | allowed
    if torch.any(combined_flags & padding):
        raise GAEComputationError("padding positions must not carry boundary flags")
    if torch.any(terminated_mask & truncated_mask & valid):
        raise GAEComputationError("terminated and truncated must remain distinct")
    if not torch.equal(boundaries[valid], (terminated_mask | truncated_mask)[valid]):
        raise GAEComputationError("episode_boundary must equal terminated or truncated")
    if not torch.equal(allowed[valid], ~boundaries[valid]):
        raise GAEComputationError(
            "bootstrap is allowed exactly on non-boundary transitions"
        )

    bootstrap_mask = valid & allowed & ~boundaries
    bootstrap_float = bootstrap_mask.to(dtype=torch.float32).unsqueeze(-1)
    raw_residual = rewards + resolved_gamma * bootstrap_float * next_values - values
    td_residual = torch.where(
        valid.unsqueeze(-1), raw_residual, torch.zeros_like(raw_residual)
    )
    advantage = torch.zeros_like(td_residual)
    next_advantage = torch.zeros(rewards.shape[1], dtype=torch.float32)
    recurrence_scale = resolved_gamma * resolved_lambda
    for index in range(valid_length - 1, -1, -1):
        advantage[index] = (
            td_residual[index]
            + recurrence_scale * bootstrap_float[index] * next_advantage
        )
        next_advantage = advantage[index]
    return_target = torch.where(
        valid.unsqueeze(-1), advantage + values, torch.zeros_like(advantage)
    )
    for name, tensor in (
        ("td_residual", td_residual),
        ("advantage", advantage),
        ("return_target", return_target),
    ):
        if not torch.isfinite(tensor).all():
            raise GAEComputationError(f"{name} contains NaN or Inf")
    return CAGATMAPPOPerAgentGAEOutput(
        bootstrap_mask=bootstrap_mask,
        td_residual=td_residual,
        advantage=advantage,
        return_target=return_target,
        sequence_mask=valid,
    )


def compute_route_specific_advantage(
    *,
    base_td_residual: Tensor,
    bootstrap_mask: Tensor,
    sequence_mask: Tensor,
    gamma: float,
    route_gae_lambda: float,
) -> Tensor:
    """Recur a route-only per-agent advantage from the frozen base residual.

    The function deliberately accepts the already-computed production TD
    residual instead of reward/value inputs.  This makes it impossible for the
    route estimator to introduce a second reward, value, or bootstrap
    definition and leaves the production critic target outside this path.
    """

    resolved_gamma = _coefficient(gamma, "gamma", include_zero=False)
    resolved_lambda = _coefficient(
        route_gae_lambda,
        "route_gae_lambda",
        include_zero=True,
    )
    residual = _float_matrix(base_td_residual, "base_td_residual")
    bootstrap = _bool_vector(bootstrap_mask, "bootstrap_mask")
    valid = _bool_vector(sequence_mask, "sequence_mask")
    if bootstrap.shape != valid.shape or residual.shape[0] != valid.shape[0]:
        raise GAEComputationError(
            "route advantage inputs must share the same time axis"
        )
    valid_length = int(valid.sum().item())
    if valid_length == 0:
        raise GAEComputationError("route advantage sequence has no valid positions")
    expected_valid = torch.arange(valid.numel()) < valid_length
    if not torch.equal(valid, expected_valid):
        raise GAEComputationError(
            "route sequence_mask must describe one contiguous prefix"
        )
    if torch.any(bootstrap & ~valid):
        raise GAEComputationError(
            "route bootstrap mask cannot enable padding positions"
        )

    advantage = torch.zeros_like(residual)
    next_advantage = torch.zeros(residual.shape[1], dtype=torch.float32)
    recurrence_scale = resolved_gamma * resolved_lambda
    bootstrap_float = bootstrap.to(dtype=torch.float32).unsqueeze(-1)
    for index in range(valid_length - 1, -1, -1):
        advantage[index] = (
            residual[index]
            + recurrence_scale * bootstrap_float[index] * next_advantage
        )
        next_advantage = advantage[index]
    if not torch.isfinite(advantage).all():
        raise GAEComputationError("route advantage contains NaN or Inf")
    return advantage


def compute_rollout_gae(
    chunk: CAGATMAPPORolloutChunk,
    config: RunConfig,
) -> CAGATMAPPOGAEOutput | CAGATMAPPOPerAgentGAEOutput:
    """Adapt one validated rollout chunk to the frozen GAE computation."""

    if not isinstance(chunk, CAGATMAPPORolloutChunk):
        raise TypeError("chunk must be a CAGATMAPPORolloutChunk")
    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    config.validate()
    horizon = config.environment.episode_horizon
    for transition in chunk.transitions:
        if transition.slot >= horizon:
            raise GAEComputationError("rollout slot lies beyond the episode horizon")
        if transition.slot == horizon - 1 and not transition.episode_boundary:
            raise GAEComputationError(
                "the fixed-horizon final service slot must be a no-bootstrap boundary"
            )
    mappo = config.training.mappo
    if AgentCreditMode(mappo.agent_credit_mode) is AgentCreditMode.ROLE_DECOMPOSED:
        return compute_per_agent_gae_and_returns(
            reward=chunk.reward,
            old_value=chunk.old_value,
            bootstrap_value=chunk.bootstrap_values,
            terminated=chunk.terminated,
            truncated=chunk.truncated,
            episode_boundary=chunk.episode_boundary,
            bootstrap_allowed=chunk.bootstrap_allowed,
            gamma=mappo.gamma,
            gae_lambda=mappo.gae_lambda,
            sequence_mask=chunk.sequence_mask,
        )
    return compute_gae_and_returns(
        reward=chunk.reward,
        old_value=chunk.old_value,
        bootstrap_value=chunk.bootstrap_values,
        terminated=chunk.terminated,
        truncated=chunk.truncated,
        episode_boundary=chunk.episode_boundary,
        bootstrap_allowed=chunk.bootstrap_allowed,
        gamma=mappo.gamma,
        gae_lambda=mappo.gae_lambda,
        sequence_mask=chunk.sequence_mask,
    )


__all__ = [
    "CAGATMAPPOGAEOutput",
    "CAGATMAPPOPerAgentGAEOutput",
    "GAEComputationError",
    "compute_gae_and_returns",
    "compute_per_agent_gae_and_returns",
    "compute_route_specific_advantage",
    "compute_rollout_gae",
]
