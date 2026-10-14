"""Rollout-capped route-event N-step actor credit.

This module only consumes immutable rollout snapshots. It never queries the
environment, replays a transition, or evaluates the critic while constructing
an actor target.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


ROUTE_NSTEP_STOP_KINDS = (
    "task_completed",
    "task_expired",
    "task_truncated_episode_boundary",
    "rollout_boundary",
)


class RouteNstepError(ValueError):
    """Raised when route-event metadata or a frozen target is inconsistent."""


@dataclass(frozen=True)
class RouteNstepSample:
    """Detached provenance for one route-active actor-credit overwrite."""

    episode_id: int
    source_uav: int
    task_id: object
    category: str
    branch_event_key: tuple[object, ...]
    formal_route_event_key: tuple[object, ...] | None
    global_rollout_index: int
    policy_version: int
    base_advantage: float
    nstep_advantage: float
    advantage_delta: float
    horizon: int
    stop_kind: str
    terminal_before_rollout: bool
    bootstrap_used: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("episode_id", self.episode_id),
            ("source_uav", self.source_uav),
            ("task_id", self.task_id),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RouteNstepError(
                    f"{name} must be a non-negative integer"
                )
        if self.category not in {"local", "remote", "defer"}:
            raise RouteNstepError("route-event category is invalid")
        if self.stop_kind not in ROUTE_NSTEP_STOP_KINDS:
            raise RouteNstepError("route-event stop_kind is invalid")
        if self.horizon <= 0:
            raise RouteNstepError("route-event horizon must be positive")
        if len(self.branch_event_key) != 4:
            raise RouteNstepError(
                "branch_event_key must have four identity fields"
            )
        if self.category == "defer" and self.formal_route_event_key is not None:
            raise RouteNstepError(
                "defer must not receive a formal route-event key"
            )
        if self.category != "defer" and self.formal_route_event_key is None:
            raise RouteNstepError(
                "applied local/remote routes require a formal key"
            )
        if any(
            not math.isfinite(value)
            for value in (
                self.base_advantage,
                self.nstep_advantage,
                self.advantage_delta,
            )
        ):
            raise RouteNstepError("route-event advantages must be finite")


@dataclass(frozen=True)
class RolloutCappedRouteNstepOutput:
    """Route actor advantage plus a [T][A] provenance matrix."""

    advantage: Tensor
    samples: tuple[tuple[RouteNstepSample | None, ...], ...]


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RouteNstepError(f"{name} must be a non-negative integer")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RouteNstepError(f"{name} must be a sequence")
    return value


def _route_category(proposal: object) -> str:
    if proposal == "local":
        return "local"
    if proposal == "defer":
        return "defer"
    if isinstance(proposal, int) and not isinstance(proposal, bool):
        return "remote"
    raise RouteNstepError(
        "route-active proposal is not local, remote, or defer"
    )


def build_route_credit_step_metadata(
    info: Mapping[str, Any],
    *,
    episode_id: int,
    global_rollout_index: int,
    policy_version: int,
) -> dict[str, Any]:
    """Extract compact JSON-safe route/lifecycle facts from one real step."""

    if not isinstance(info, Mapping):
        raise TypeError("info must be a mapping")
    episode = _nonnegative_int(episode_id, "episode_id")
    rollout_index = _nonnegative_int(
        global_rollout_index, "global_rollout_index"
    )
    version = _nonnegative_int(policy_version, "policy_version")
    service = info.get("service")
    if not isinstance(service, Mapping):
        raise RouteNstepError("StepResult.info requires service telemetry")

    decisions: list[dict[str, Any]] = []
    seen_agents: set[int] = set()
    for raw in _sequence(service.get("routing", ()), "service.routing"):
        if not isinstance(raw, Mapping):
            raise RouteNstepError("routing records must be mappings")
        source = _nonnegative_int(raw.get("uav_id"), "routing.uav_id")
        if source in seen_agents:
            raise RouteNstepError(
                "routing contains duplicate source UAV records"
            )
        seen_agents.add(source)
        task_id = raw.get("task_id")
        if task_id is None:
            continue
        task_id = _nonnegative_int(task_id, "routing.task_id")
        category = _route_category(raw.get("proposal"))
        applied = raw.get("applied")
        expected_applied = category != "defer"
        if type(applied) is not bool or applied != expected_applied:
            raise RouteNstepError(
                "routing applied flag disagrees with route category"
            )
        branch_key = (episode, source, task_id, rollout_index)
        formal_key = (
            None
            if category == "defer"
            else (
                episode,
                task_id,
                _nonnegative_int(info.get("slot"), "info.slot"),
            )
        )
        decisions.append(
            {
                "episode_id": episode,
                "source_uav": source,
                "task_id": task_id,
                "category": category,
                "branch_event_key": list(branch_key),
                "formal_route_event_key": (
                    None if formal_key is None else list(formal_key)
                ),
                "global_rollout_index": rollout_index,
                "policy_version": version,
            }
        )

    terminals: list[dict[str, Any]] = []
    terminal_keys: set[tuple[int, object]] = set()
    terminal_sources = (
        ("settled_tasks", service.get("settled_tasks", ())),
        ("truncated_tasks", service.get("truncated_tasks", ())),
    )
    for collection_name, raw_collection in terminal_sources:
        for raw in _sequence(
            raw_collection, f"service.{collection_name}"
        ):
            if not isinstance(raw, Mapping):
                raise RouteNstepError(
                    "terminal task snapshots must be mappings"
                )
            source = _nonnegative_int(
                raw.get("source_uav"), "terminal.source_uav"
            )
            task_id = raw.get("task_id")
            if task_id is None:
                raise RouteNstepError(
                    "terminal task snapshot lacks task_id"
                )
            task_id = _nonnegative_int(task_id, "terminal.task_id")
            key = (source, task_id)
            if key in terminal_keys:
                raise RouteNstepError(
                    "task has duplicate terminal records in one step"
                )
            terminal_keys.add(key)
            outcome = raw.get("outcome")
            status = raw.get("status")
            if collection_name == "truncated_tasks" or outcome == "truncated":
                kind = "truncated"
            elif outcome in {"done", "completed"} or status == "done":
                kind = "completed"
            elif outcome == "expired" or status == "expired":
                kind = "expired"
            else:
                raise RouteNstepError(
                    "terminal task has an unknown lifecycle outcome"
                )
            terminals.append(
                {
                    "episode_id": episode,
                    "source_uav": source,
                    "task_id": task_id,
                    "kind": kind,
                    "global_rollout_index": rollout_index,
                }
            )
    return {
        "episode_id": episode,
        "global_rollout_index": rollout_index,
        "policy_version": version,
        "decisions": decisions,
        "terminals": terminals,
    }


def _float_matrix(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise RouteNstepError(f"{name} must have shape [T,A]")
    result = value.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(result).all():
        raise RouteNstepError(f"{name} contains NaN or Inf")
    return result


def _bool_vector(value: Tensor, name: str, length: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or tuple(value.shape) != (length,)
        or value.dtype != torch.bool
    ):
        raise RouteNstepError(f"{name} must be boolean [T]")
    return value.detach().to(device="cpu")


def _bool_matrix(
    value: Tensor, name: str, shape: tuple[int, int]
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or tuple(value.shape) != shape
        or value.dtype != torch.bool
    ):
        raise RouteNstepError(f"{name} must be boolean [T,A]")
    return value.detach().to(device="cpu")


def _step_metadata(
    raw_summary: Mapping[str, Any],
    time_index: int,
) -> Mapping[str, Any]:
    if not isinstance(raw_summary, Mapping):
        raise RouteNstepError("rollout audit summary must be a mapping")
    step = raw_summary.get("route_credit_step")
    if not isinstance(step, Mapping):
        raise RouteNstepError(
            f"rollout position {time_index} lacks route_credit_step metadata"
        )
    _nonnegative_int(
        step.get("episode_id"), "route_credit_step.episode_id"
    )
    _nonnegative_int(
        step.get("global_rollout_index"),
        "route_credit_step.global_rollout_index",
    )
    _nonnegative_int(
        step.get("policy_version"), "route_credit_step.policy_version"
    )
    _sequence(step.get("decisions"), "route_credit_step.decisions")
    _sequence(step.get("terminals"), "route_credit_step.terminals")
    return step


def compute_rollout_capped_route_event_nstep(
    *,
    reward: Tensor,
    old_value: Tensor,
    bootstrap_value: Tensor,
    bootstrap_mask: Tensor,
    episode_boundary: Tensor,
    route_active: Tensor,
    base_advantage: Tensor,
    transition_metadata: Sequence[Mapping[str, Any]],
    gamma: float,
) -> RolloutCappedRouteNstepOutput:
    """Overwrite route-active [T,A] credit with direct capped returns."""

    if (
        isinstance(gamma, bool)
        or not isinstance(gamma, (int, float))
        or not math.isfinite(float(gamma))
        or not 0.0 < float(gamma) <= 1.0
    ):
        raise RouteNstepError("gamma must lie in (0, 1]")
    rewards = _float_matrix(reward, "reward")
    values = _float_matrix(old_value, "old_value")
    next_values = _float_matrix(bootstrap_value, "bootstrap_value")
    base = _float_matrix(base_advantage, "base_advantage")
    if values.shape != rewards.shape or next_values.shape != rewards.shape:
        raise RouteNstepError(
            "reward/value snapshots must share shape [T,A]"
        )
    if base.shape != rewards.shape:
        raise RouteNstepError(
            "base_advantage must share reward shape [T,A]"
        )
    time_steps, agents = rewards.shape
    bootstrap = _bool_vector(
        bootstrap_mask, "bootstrap_mask", time_steps
    )
    boundaries = _bool_vector(
        episode_boundary, "episode_boundary", time_steps
    )
    active = _bool_matrix(
        route_active, "route_active", (time_steps, agents)
    )
    if torch.any(bootstrap & boundaries):
        raise RouteNstepError("episode boundaries must not bootstrap")
    if len(transition_metadata) != time_steps:
        raise RouteNstepError(
            "transition metadata must align with the time axis"
        )
    steps = tuple(
        _step_metadata(summary, index)
        for index, summary in enumerate(transition_metadata)
    )

    decisions: list[dict[int, Mapping[str, Any]]] = []
    terminals: list[tuple[Mapping[str, Any], ...]] = []
    for step in steps:
        per_agent: dict[int, Mapping[str, Any]] = {}
        for raw in _sequence(
            step["decisions"], "route_credit_step.decisions"
        ):
            if not isinstance(raw, Mapping):
                raise RouteNstepError(
                    "route decision metadata must be mappings"
                )
            source = _nonnegative_int(
                raw.get("source_uav"), "decision.source_uav"
            )
            if source in per_agent:
                raise RouteNstepError(
                    "route step has duplicate agent decisions"
                )
            _nonnegative_int(raw.get("episode_id"), "decision.episode_id")
            _nonnegative_int(raw.get("task_id"), "decision.task_id")
            per_agent[source] = raw
        decisions.append(per_agent)
        terminal_rows = tuple(
            raw
            for raw in _sequence(
                step["terminals"], "route_credit_step.terminals"
            )
            if isinstance(raw, Mapping)
        )
        if len(terminal_rows) != len(step["terminals"]):
            raise RouteNstepError("terminal metadata must be mappings")
        for raw in terminal_rows:
            _nonnegative_int(raw.get("episode_id"), "terminal.episode_id")
            _nonnegative_int(raw.get("source_uav"), "terminal.source_uav")
            _nonnegative_int(raw.get("task_id"), "terminal.task_id")
        terminals.append(terminal_rows)

    output = base.clone()
    sample_rows: list[list[RouteNstepSample | None]] = [
        [None for _ in range(agents)] for _ in range(time_steps)
    ]
    resolved_gamma = float(gamma)
    for start in range(time_steps):
        for agent in range(agents):
            decision = decisions[start].get(agent)
            if bool(active[start, agent]) != (decision is not None):
                raise RouteNstepError(
                    "route activity and lifecycle decision metadata disagree"
                )
            if decision is None:
                continue
            episode = _nonnegative_int(
                decision.get("episode_id"), "decision.episode_id"
            )
            source = _nonnegative_int(
                decision.get("source_uav"), "decision.source_uav"
            )
            task_id = _nonnegative_int(
                decision.get("task_id"), "decision.task_id"
            )
            category = str(decision.get("category"))
            branch_key = tuple(
                _sequence(
                    decision.get("branch_event_key"), "branch_event_key"
                )
            )
            formal_raw = decision.get("formal_route_event_key")
            formal_key = (
                None
                if formal_raw is None
                else tuple(
                    _sequence(formal_raw, "formal_route_event_key")
                )
            )
            global_index = _nonnegative_int(
                decision.get("global_rollout_index"),
                "decision.global_rollout_index",
            )
            policy_version = _nonnegative_int(
                decision.get("policy_version"),
                "decision.policy_version",
            )

            boundary_index = next(
                (
                    index
                    for index in range(start, time_steps)
                    if bool(boundaries[index])
                    and steps[index].get("episode_id") == episode
                ),
                None,
            )
            terminal_index: int | None = None
            terminal_kind: str | None = None
            for index in range(start, time_steps):
                match = next(
                    (
                        item
                        for item in terminals[index]
                        if item.get("episode_id") == episode
                        and item.get("source_uav") == source
                        and item.get("task_id") == task_id
                    ),
                    None,
                )
                if match is not None:
                    terminal_index = index
                    terminal_kind = str(match.get("kind"))
                    break
                if boundary_index is not None and index >= boundary_index:
                    break

            if terminal_index is not None:
                end = terminal_index
                stop_kind = {
                    "completed": "task_completed",
                    "expired": "task_expired",
                    "truncated": "task_truncated_episode_boundary",
                }.get(terminal_kind)
                if stop_kind is None:
                    raise RouteNstepError(
                        "terminal metadata has an invalid kind"
                    )
                terminal_before_rollout = True
            elif boundary_index is not None:
                raise RouteNstepError(
                    "route task reached an episode boundary without "
                    "a terminal record"
                )
            else:
                end = time_steps - 1
                stop_kind = "rollout_boundary"
                terminal_before_rollout = False

            horizon = end - start + 1
            discounted_reward = torch.zeros((), dtype=torch.float32)
            discount = 1.0
            for index in range(start, end + 1):
                discounted_reward = (
                    discounted_reward
                    + discount * rewards[index, source]
                )
                discount *= resolved_gamma
            bootstrap_used = bool(bootstrap[end])
            if (
                stop_kind == "task_truncated_episode_boundary"
                and bootstrap_used
            ):
                raise RouteNstepError(
                    "truncated episode-boundary task cannot bootstrap"
                )
            if (
                stop_kind in {"task_completed", "task_expired"}
                and not bool(boundaries[end])
                and not bootstrap_used
            ):
                raise RouteNstepError(
                    "non-episode task terminal must bootstrap"
                )
            target = discounted_reward
            if bootstrap_used:
                target = target + discount * next_values[end, source]
            nstep_advantage = target - values[start, source]
            output[start, source] = nstep_advantage
            base_value = float(base[start, source].item())
            nstep_value = float(nstep_advantage.item())
            sample_rows[start][source] = RouteNstepSample(
                episode_id=episode,
                source_uav=source,
                task_id=task_id,
                category=category,
                branch_event_key=branch_key,
                formal_route_event_key=formal_key,
                global_rollout_index=global_index,
                policy_version=policy_version,
                base_advantage=base_value,
                nstep_advantage=nstep_value,
                advantage_delta=nstep_value - base_value,
                horizon=horizon,
                stop_kind=stop_kind,
                terminal_before_rollout=terminal_before_rollout,
                bootstrap_used=bootstrap_used,
            )
    if not torch.isfinite(output).all():
        raise RouteNstepError(
            "rollout-capped route advantage contains NaN or Inf"
        )
    return RolloutCappedRouteNstepOutput(
        advantage=output,
        samples=tuple(tuple(row) for row in sample_rows),
    )


__all__ = [
    "ROUTE_NSTEP_STOP_KINDS",
    "RolloutCappedRouteNstepOutput",
    "RouteNstepError",
    "RouteNstepSample",
    "build_route_credit_step_metadata",
    "compute_rollout_capped_route_event_nstep",
]
