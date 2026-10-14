"""Behavior-neutral route diagnostics for CA-GAT-MAPPO PPO updates.

The functions in this module only read already-computed policy statistics,
fixed rollout targets, action-mask snapshots, and existing gradients.  They do
not sample actions, advance random-number generators, build loss terms, run
backward, clip gradients, or step optimizers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..config import ActorRatioMode, RunConfig, WorkloadTimingMode
from ..env.actions import ActionProposal
from ..env.reward import TaskWorkloadSnapshot
from ..env.tasks import Task, TaskOutcome
from .ca_gat_mappo import (
    ACTION_BRANCH_ORDER,
    CandidateAwareRouteDecoderV1,
    HierarchicalOptionAwareRouteDecoderV1,
    OptionAwareRouteDecoderV1,
)
from .ca_gat_mappo_actions import (
    SequentialActionDistributionOutput,
    SequentialActionMaskBatch,
)
from .ca_gat_mappo_route_nstep import RouteNstepSample


ROUTE_TELEMETRY_SCHEMA_VERSION = 4
TRAJECTORY_CREDIT_SCHEMA_VERSION = 1
ROUTE_DIAGNOSTIC_SCHEMA_VERSION = 1
ROUTE_DIAGNOSTIC_UTILITY_LABEL = "OBSERVABLE_DIAGNOSTIC_UTILITY"
TRAJECTORY_CREDIT_EVENT_TYPES = ("route", "terminal", "credit")
TRAJECTORY_LEDGER_TOLERANCE = 1.0e-9
ROUTE_GRADIENT_STATUSES = ("none", "zero", "finite_nonzero", "nonfinite")
ROUTE_ACTION_CATEGORIES = ("local", "remote", "defer")
ROUTE_DISTRIBUTION_GROUPS = (
    "route_active",
    "local_route",
    "remote_route",
    "defer_route",
    "legal_remote_local",
    "legal_remote_remote",
)
ROUTE_PPO_RECORD_FIELDS = (
    "route_ppo_epsilon_clip",
    "route_ppo_overall_count",
    "route_ppo_overall_old_log_prob_mean",
    "route_ppo_overall_new_log_prob_mean",
    "route_ppo_overall_log_ratio_mean",
    "route_ppo_overall_ratio_mean",
    "route_ppo_overall_approx_kl_mean",
    "route_ppo_overall_clip_indicator_fraction",
    "route_ppo_overall_clip_fraction",
    "local_route_ppo_count",
    "local_route_ppo_old_log_prob_mean",
    "local_route_ppo_new_log_prob_mean",
    "local_route_ppo_log_ratio_mean",
    "local_route_ppo_ratio_mean",
    "local_route_ppo_approx_kl_mean",
    "local_route_ppo_clip_indicator_fraction",
    "local_route_ppo_clip_fraction",
    "remote_route_ppo_count",
    "remote_route_ppo_old_log_prob_mean",
    "remote_route_ppo_new_log_prob_mean",
    "remote_route_ppo_log_ratio_mean",
    "remote_route_ppo_ratio_mean",
    "remote_route_ppo_approx_kl_mean",
    "remote_route_ppo_clip_indicator_fraction",
    "remote_route_ppo_clip_fraction",
    "defer_route_ppo_count",
    "defer_route_ppo_old_log_prob_mean",
    "defer_route_ppo_new_log_prob_mean",
    "defer_route_ppo_log_ratio_mean",
    "defer_route_ppo_ratio_mean",
    "defer_route_ppo_approx_kl_mean",
    "defer_route_ppo_clip_indicator_fraction",
    "defer_route_ppo_clip_fraction",
)
BRANCH_PPO_STATISTIC_NAMES = (
    "active_count",
    "ratio_mean",
    "ratio_median",
    "ratio_std",
    "approx_kl_mean",
    "clip_fraction",
    "surrogate_contribution_mean",
    "advantage_mean",
)
BRANCH_PPO_RECORD_FIELDS = tuple(
    f"{branch}_branch_ppo_{statistic}"
    for branch in ACTION_BRANCH_ORDER
    for statistic in BRANCH_PPO_STATISTIC_NAMES
)


class RouteTelemetryError(ValueError):
    """Raised when route telemetry inputs or derived invariants are invalid."""


@dataclass(frozen=True)
class GradientMeasurement:
    """One read-only gradient measurement with explicit missing semantics."""

    status: str
    norm: float | None
    mean: float | None = None
    signed_sum: float | None = None

    def __post_init__(self) -> None:
        if self.status not in ROUTE_GRADIENT_STATUSES:
            raise RouteTelemetryError("unknown route-head gradient status")
        if self.status == "none":
            if self.norm is not None:
                raise RouteTelemetryError("missing gradient must use an NA norm")
            return
        if self.status == "nonfinite":
            if self.norm is not None:
                raise RouteTelemetryError("nonfinite gradient must use an NA norm")
            return
        if self.norm is None or not math.isfinite(self.norm) or self.norm < 0.0:
            raise RouteTelemetryError("present gradient norm must be finite and non-negative")
        if (self.status == "zero") != (self.norm == 0.0):
            raise RouteTelemetryError("gradient status and norm disagree")


@dataclass(frozen=True)
class RouteHeadGradientTelemetry:
    """Weight, bias, and combined route-head gradient diagnostics."""

    weight: GradientMeasurement
    bias: GradientMeasurement
    total: GradientMeasurement
    rows: tuple[Any, ...] = ()


@dataclass(frozen=True)
class BranchPPODynamicsGroup:
    """Detached statistics for one active action branch."""

    active_count: int = 0
    ratio_mean: float | None = None
    ratio_median: float | None = None
    ratio_std: float | None = None
    approx_kl_mean: float | None = None
    clip_fraction: float | None = None
    surrogate_contribution_mean: float | None = None
    advantage_mean: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.active_count, bool)
            or not isinstance(self.active_count, int)
            or self.active_count < 0
        ):
            raise RouteTelemetryError("branch PPO active_count is invalid")
        values = tuple(
            getattr(self, name)
            for name in BRANCH_PPO_STATISTIC_NAMES
            if name != "active_count"
        )
        if self.active_count == 0:
            if any(value is not None for value in values):
                raise RouteTelemetryError(
                    "branch PPO statistics must be NA without active samples"
                )
            return
        if any(value is None or not math.isfinite(value) for value in values):
            raise RouteTelemetryError(
                "branch PPO statistics must be finite with active samples"
            )
        if self.clip_fraction is None or not 0.0 <= self.clip_fraction <= 1.0:
            raise RouteTelemetryError("branch PPO clip_fraction is outside [0, 1]")

    def record(self, branch: str) -> dict[str, object]:
        return {
            f"{branch}_branch_ppo_{name}": getattr(self, name)
            for name in BRANCH_PPO_STATISTIC_NAMES
        }


@dataclass(frozen=True)
class BranchPPODynamics:
    """All seven branch statistics from one existing PPO evaluation."""

    epsilon_clip: float
    groups: Mapping[str, BranchPPODynamicsGroup] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.epsilon_clip) or not 0.0 < self.epsilon_clip < 1.0:
            raise RouteTelemetryError("branch PPO epsilon_clip is invalid")
        if tuple(self.groups) != ACTION_BRANCH_ORDER:
            raise RouteTelemetryError("branch PPO groups must use frozen branch order")

    def record(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for branch in ACTION_BRANCH_ORDER:
            result.update(self.groups[branch].record(branch))
        return result


@dataclass(frozen=True)
class RouteTelemetry:
    """Detached per-PPO-epoch route telemetry with explicit valid counts."""

    schema_version: int
    route_branch_active_count: int
    legal_remote_route_count: int
    route_selected_local_count: int
    route_selected_remote_count: int
    route_selected_defer_count: int
    remote_selection_rate_given_legal_remote: float | None
    route_entropy_valid_sample_count: int
    route_entropy_mean: float | None
    legal_remote_probability_valid_sample_count: int
    mean_selected_route_probability: float | None
    mean_local_probability: float | None
    mean_best_remote_probability: float | None
    mean_local_minus_best_remote_probability: float | None
    mean_local_minus_best_remote_logit_margin: float | None
    route_active_advantage_valid_sample_count: int
    route_active_advantage_mean: float | None
    route_active_advantage_std: float | None
    route_active_return_mean: float | None
    route_active_return_std: float | None
    local_route_valid_sample_count: int
    local_route_advantage_mean: float | None
    local_route_return_mean: float | None
    remote_route_valid_sample_count: int
    remote_route_advantage_mean: float | None
    remote_route_return_mean: float | None
    defer_route_valid_sample_count: int
    defer_route_advantage_mean: float | None
    defer_route_return_mean: float | None
    legal_remote_local_valid_sample_count: int
    legal_remote_local_advantage_mean: float | None
    legal_remote_local_return_mean: float | None
    legal_remote_remote_valid_sample_count: int
    legal_remote_remote_advantage_mean: float | None
    legal_remote_remote_return_mean: float | None
    route_head_weight_grad_status: str
    route_head_weight_grad_norm: float | None
    route_head_bias_grad_status: str
    route_head_bias_grad_norm: float | None
    route_head_total_grad_status: str
    route_head_total_grad_norm: float | None
    route_probability_valid_sample_count: int = 0
    base_route_advantage_valid_sample_count: int = 0
    base_route_advantage_mean: float | None = None
    base_route_advantage_std: float | None = None
    base_route_advantage_median: float | None = None
    base_route_advantage_p90_absolute_magnitude: float | None = None
    base_route_advantage_p95_absolute_magnitude: float | None = None
    new_route_advantage_valid_sample_count: int = 0
    new_route_advantage_mean: float | None = None
    new_route_advantage_std: float | None = None
    new_route_advantage_median: float | None = None
    new_route_advantage_p90_absolute_magnitude: float | None = None
    new_route_advantage_p95_absolute_magnitude: float | None = None
    nstep_route_advantage_valid_sample_count: int = 0
    nstep_route_advantage_mean: float | None = None
    nstep_route_advantage_std: float | None = None
    nstep_route_advantage_median: float | None = None
    nstep_route_advantage_p90_absolute_magnitude: float | None = None
    nstep_route_advantage_p95_absolute_magnitude: float | None = None
    route_advantage_delta_valid_sample_count: int = 0
    route_advantage_delta_mean: float | None = None
    route_advantage_delta_std: float | None = None
    route_advantage_delta_median: float | None = None
    route_advantage_delta_p90_absolute_magnitude: float | None = None
    route_advantage_delta_p95_absolute_magnitude: float | None = None
    mean_selected_route_probability_all_route_active: float | None = None
    mean_local_probability_all_route_active: float | None = None
    mean_defer_probability_all_route_active: float | None = None
    mean_total_remote_probability_mass: float | None = None
    mean_number_of_legal_remote_destinations: float | None = None
    mean_local_minus_total_remote_probability: float | None = None
    mean_local_minus_total_remote_logit_margin: float | None = None
    mean_remote_probability_by_destination: Mapping[str, float] | None = None
    remote_probability_destination_valid_samples: Mapping[str, int] | None = None
    route_head_weight_grad_mean: float | None = None
    route_head_weight_grad_signed_sum: float | None = None
    route_head_bias_grad_mean: float | None = None
    route_head_bias_grad_signed_sum: float | None = None
    shared_trunk_grad_status: str = "none"
    shared_trunk_grad_norm: float | None = None
    shared_trunk_gradient_scope: str = "joint_actor_loss"
    shared_trunk_interference_status: str = "NOT OBSERVABLE WITHOUT ALGORITHM-PERTURBING EXTRA BACKWARD"
    route_head_gradient_rows: tuple[Any, ...] = ()
    samples: tuple[RouteSampleTelemetry, ...] = ()
    route_ppo_dynamics: RoutePPODynamics | None = None
    actor_ratio_mode: str = ActorRatioMode.JOINT.value
    branch_ppo_dynamics: BranchPPODynamics | None = None
    active_branch_matrix: tuple[tuple[tuple[bool, ...], ...], ...] = ()
    _advantage_distributions: Mapping[str, RouteDistributionSummary] = field(default_factory=dict, repr=False)
    _return_distributions: Mapping[str, RouteDistributionSummary] = field(default_factory=dict, repr=False)
    _td_residual_distributions: Mapping[str, RouteDistributionSummary] = field(default_factory=dict, repr=False)
    _route_credit_distributions: Mapping[
        str, Mapping[str, RouteDistributionSummary]
    ] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.schema_version != ROUTE_TELEMETRY_SCHEMA_VERSION:
            raise RouteTelemetryError("route telemetry schema version is invalid")
        try:
            ActorRatioMode(self.actor_ratio_mode)
        except (TypeError, ValueError) as exc:
            raise RouteTelemetryError("actor_ratio_mode is invalid") from exc
        if self.branch_ppo_dynamics is not None and not isinstance(
            self.branch_ppo_dynamics, BranchPPODynamics
        ):
            raise TypeError("branch_ppo_dynamics must be BranchPPODynamics or None")
        if self.active_branch_matrix:
            agent_count = len(self.active_branch_matrix[0])
            if agent_count == 0 or any(
                len(time_step) != agent_count
                or any(
                    len(agent_row) != len(ACTION_BRANCH_ORDER)
                    or any(type(value) is not bool for value in agent_row)
                    for agent_row in time_step
                )
                for time_step in self.active_branch_matrix
            ):
                raise RouteTelemetryError(
                    "active_branch_matrix must have consistent [T,A,7] boolean rows"
                )
        count_names = (
            item.name
            for item in fields(self)
            if item.name.endswith("_count")
        )
        for name in count_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RouteTelemetryError(f"{name} must be a non-negative integer")
        if self.route_entropy_valid_sample_count != self.route_branch_active_count:
            raise RouteTelemetryError("route entropy count must equal route-active count")
        if (
            self.route_active_advantage_valid_sample_count
            != self.route_branch_active_count
        ):
            raise RouteTelemetryError("route advantage count must equal route-active count")
        if (
            self.legal_remote_probability_valid_sample_count
            != self.legal_remote_route_count
        ):
            raise RouteTelemetryError("legal-remote probability count is inconsistent")
        if self.legal_remote_route_count > self.route_branch_active_count:
            raise RouteTelemetryError("legal-remote count exceeds route-active count")
        if self.route_selected_remote_count > self.legal_remote_route_count:
            raise RouteTelemetryError("remote selections exceed legal-remote samples")
        if (
            self.route_selected_local_count
            + self.route_selected_remote_count
            + self.route_selected_defer_count
            != self.route_branch_active_count
        ):
            raise RouteTelemetryError("route selection counts do not partition active samples")

        self._validate_optional_group(
            self.legal_remote_route_count,
            (
                "remote_selection_rate_given_legal_remote",
                "mean_selected_route_probability",
                "mean_local_probability",
                "mean_best_remote_probability",
                "mean_local_minus_best_remote_probability",
                "mean_local_minus_best_remote_logit_margin",
            ),
        )
        self._validate_optional_group(
            self.route_branch_active_count,
            (
                "route_entropy_mean",
                "route_active_advantage_mean",
                "route_active_advantage_std",
                "route_active_return_mean",
                "route_active_return_std",
            ),
        )
        for prefix in (
            "base_route_advantage",
            "new_route_advantage",
            "nstep_route_advantage",
            "route_advantage_delta",
        ):
            count = getattr(self, f"{prefix}_valid_sample_count")
            if count not in (0, self.route_branch_active_count):
                raise RouteTelemetryError(
                    f"{prefix} count must be zero or equal route-active count"
                )
            self._validate_optional_group(
                count,
                (
                    f"{prefix}_mean",
                    f"{prefix}_std",
                    f"{prefix}_median",
                    f"{prefix}_p90_absolute_magnitude",
                    f"{prefix}_p95_absolute_magnitude",
                ),
            )
        for prefix in (
            "local_route",
            "remote_route",
            "defer_route",
            "legal_remote_local",
            "legal_remote_remote",
        ):
            self._validate_optional_group(
                getattr(self, f"{prefix}_valid_sample_count"),
                (f"{prefix}_advantage_mean", f"{prefix}_return_mean"),
            )
        if self.remote_selection_rate_given_legal_remote is not None and not (
            0.0 <= self.remote_selection_rate_given_legal_remote <= 1.0
        ):
            raise RouteTelemetryError("conditional remote selection rate is outside [0, 1]")
        for prefix in ("weight", "bias", "total"):
            GradientMeasurement(
                status=getattr(self, f"route_head_{prefix}_grad_status"),
                norm=getattr(self, f"route_head_{prefix}_grad_norm"),
            )

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        names = tuple(
            item.name
            for item in fields(cls)
            if not item.name.startswith("_")
            and item.name not in {
                "samples",
                "route_ppo_dynamics",
                "actor_ratio_mode",
                "branch_ppo_dynamics",
                "active_branch_matrix",
            }
        )
        distribution_names = []
        for group in ROUTE_DISTRIBUTION_GROUPS:
            distribution_names.extend(
                f"{group}_advantage_{suffix}"
                for suffix in (
                    "valid_sample_count",
                    "mean",
                    "std",
                    "median",
                    "p25",
                    "p75",
                    "positive_fraction",
                    "negative_fraction",
                )
            )
            distribution_names.extend(
                f"{group}_return_target_{suffix}"
                for suffix in (
                    "valid_sample_count",
                    "mean",
                    "std",
                    "median",
                    "p25",
                    "p75",
                    "positive_fraction",
                    "negative_fraction",
                )
            )
            distribution_names.extend(
                f"{group}_td_residual_{suffix}"
                for suffix in (
                    "valid_sample_count",
                    "mean",
                    "std",
                    "median",
                    "p25",
                    "p75",
                    "positive_fraction",
                    "negative_fraction",
                )
            )
        for group in ("local_route", "remote_route", "defer_route"):
            for credit in (
                "base_route_advantage",
                "nstep_route_advantage",
                "route_advantage_delta",
            ):
                distribution_names.extend(
                    f"{group}_{credit}_{suffix}"
                    for suffix in (
                        "valid_sample_count",
                        "mean",
                        "std",
                        "median",
                        "p90_absolute_magnitude",
                        "p95_absolute_magnitude",
                    )
                )
        return (
            names
            + tuple(ROUTE_PPO_RECORD_FIELDS)
            + tuple(BRANCH_PPO_RECORD_FIELDS)
            + tuple(distribution_names)
        )

    def record(self) -> dict[str, object]:
        """Return a deterministic field-ordered flat record for CSV/JSONL."""

        result = {
            name: getattr(self, name)
            for name in self.field_names()
            if hasattr(self, name)
        }
        result["schema_version"] = self.schema_version
        result["route_telemetry_schema_version"] = ROUTE_TELEMETRY_SCHEMA_VERSION
        result["actor_ratio_mode"] = self.actor_ratio_mode
        rows = []
        for row in self.route_head_gradient_rows or ():
            rows.append(row.record() if hasattr(row, "record") else row)
        result["route_head_gradient_rows"] = rows
        if self.route_ppo_dynamics is not None:
            result.update(self.route_ppo_dynamics.record())
        else:
            result.update({name: None for name in ROUTE_PPO_RECORD_FIELDS})
        if self.branch_ppo_dynamics is not None:
            result.update(self.branch_ppo_dynamics.record())
        else:
            result.update({name: None for name in BRANCH_PPO_RECORD_FIELDS})
        for prefix, summaries, include_quantiles in (
            ("advantage", self._advantage_distributions, True),
            ("return_target", self._return_distributions, True),
            ("td_residual", self._td_residual_distributions, True),
        ):
            for group in ROUTE_DISTRIBUTION_GROUPS:
                summary = (summaries or {}).get(group, RouteDistributionSummary())
                result.update(summary.record(
                    f"{group}_{prefix}", include_quantiles=include_quantiles
                ))
        for group in ("local_route", "remote_route", "defer_route"):
            summaries = (self._route_credit_distributions or {}).get(
                group, {}
            )
            for credit in (
                "base_route_advantage",
                "nstep_route_advantage",
                "route_advantage_delta",
            ):
                summary = summaries.get(credit, RouteDistributionSummary())
                result.update(
                    summary.record(
                        f"{group}_{credit}",
                        include_quantiles=False,
                        include_absolute_quantiles=True,
                    )
                )
        return result

    def _validate_optional_group(
        self,
        count: int,
        names: Sequence[str],
    ) -> None:
        for name in names:
            value = getattr(self, name)
            if count == 0:
                if value is not None:
                    raise RouteTelemetryError(f"{name} must be NA without valid samples")
            elif value is None or not math.isfinite(value):
                raise RouteTelemetryError(f"{name} must be finite with valid samples")


def _gradient_measurement(parameter: nn.Parameter) -> GradientMeasurement:
    gradient = parameter.grad
    if gradient is None:
        return GradientMeasurement("none", None)
    detached = gradient.detach()
    if not torch.isfinite(detached).all():
        raise RouteTelemetryError("route-head gradient contains NaN or Inf")
    norm = float(torch.linalg.vector_norm(detached).cpu().item())
    return GradientMeasurement(
        "zero" if norm == 0.0 else "finite_nonzero",
        norm,
        mean=float(detached.mean().cpu().item()),
        signed_sum=float(detached.sum().cpu().item()),
    )


def _aggregate_gradient_measurement(
    parameters: Sequence[nn.Parameter],
) -> GradientMeasurement:
    gradients = tuple(
        parameter.grad.detach()
        for parameter in parameters
        if parameter.grad is not None
    )
    if not gradients:
        return GradientMeasurement("none", None)
    if any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise RouteTelemetryError("route-head gradient contains NaN or Inf")
    squared_norm = math.fsum(
        float(torch.sum(gradient * gradient).cpu().item())
        for gradient in gradients
    )
    signed_sum = math.fsum(
        float(gradient.sum().cpu().item()) for gradient in gradients
    )
    count = sum(gradient.numel() for gradient in gradients)
    norm = math.sqrt(squared_norm)
    return GradientMeasurement(
        "zero" if norm == 0.0 else "finite_nonzero",
        norm,
        mean=signed_sum / count,
        signed_sum=signed_sum,
    )


def _gradient_tensor_measurement(gradient: Tensor | None) -> GradientMeasurement:
    if gradient is None:
        return GradientMeasurement("none", None)
    detached = gradient.detach()
    if not torch.isfinite(detached).all():
        raise RouteTelemetryError("route-head gradient contains NaN or Inf")
    norm = float(torch.linalg.vector_norm(detached).cpu().item())
    return GradientMeasurement(
        "zero" if norm == 0.0 else "finite_nonzero",
        norm,
        mean=float(detached.mean().cpu().item()),
        signed_sum=float(detached.sum().cpu().item()),
    )


def _candidate_route_gradient_rows(
    route_head: CandidateAwareRouteDecoderV1,
    route_domains: Any,
) -> tuple[RouteHeadGradientRowTelemetry, ...]:
    domains = list(route_domains or ())
    if domains and isinstance(domains[0], str):
        domains = [tuple(domains)]
    rows: list[RouteHeadGradientRowTelemetry] = []
    fixed_roles = ("idle", "local", "defer")
    fixed_weight_gradient = route_head.fixed_head.weight.grad
    fixed_bias_gradient = route_head.fixed_head.bias.grad
    for row_index, default_role in enumerate(fixed_roles):
        values = [
            domain[row_index]
            for domain in domains
            if row_index < len(domain)
        ]
        role = default_role
        for candidate_role in fixed_roles:
            if any(
                type(value) is str and value == candidate_role
                for value in values
            ):
                role = candidate_role
                break
        measurement = _gradient_tensor_measurement(
            None
            if fixed_weight_gradient is None
            else fixed_weight_gradient[row_index]
        )
        bias_value = (
            None
            if fixed_bias_gradient is None
            else float(fixed_bias_gradient[row_index].detach().cpu().item())
        )
        rows.append(
            RouteHeadGradientRowTelemetry(
                row_index=row_index,
                semantic_role=role,
                destination_uavs=(),
                grad_status=measurement.status,
                grad_norm=measurement.norm,
                grad_mean=measurement.mean,
                grad_signed_sum=measurement.signed_sum,
                bias_grad=bias_value,
                parameter_update_direction_mean=(
                    -measurement.mean if measurement.mean is not None else None
                ),
                parameter_update_direction_signed_sum=(
                    -measurement.signed_sum
                    if measurement.signed_sum is not None
                    else None
                ),
                bias_update_direction=(
                    -bias_value if bias_value is not None else None
                ),
            )
        )
    destinations = tuple(
        sorted(
            {
                value
                for domain in domains
                for value in domain
                if isinstance(value, int) and not isinstance(value, bool)
            }
            or set(range(route_head.uav_count))
        )
    )
    remote_measurement = _gradient_measurement(route_head.remote_scorer.weight)
    remote_bias_gradient = route_head.remote_scorer.bias.grad
    remote_bias_value = (
        None
        if remote_bias_gradient is None
        else float(remote_bias_gradient.detach().cpu().item())
    )
    rows.append(
        RouteHeadGradientRowTelemetry(
            row_index=3,
            semantic_role="remote",
            destination_uavs=destinations,
            grad_status=remote_measurement.status,
            grad_norm=remote_measurement.norm,
            grad_mean=remote_measurement.mean,
            grad_signed_sum=remote_measurement.signed_sum,
            bias_grad=remote_bias_value,
            parameter_update_direction_mean=(
                -remote_measurement.mean
                if remote_measurement.mean is not None
                else None
            ),
            parameter_update_direction_signed_sum=(
                -remote_measurement.signed_sum
                if remote_measurement.signed_sum is not None
                else None
            ),
            bias_update_direction=(
                -remote_bias_value if remote_bias_value is not None else None
            ),
        )
    )
    return tuple(rows)


def _option_route_gradient_rows(
    route_head: OptionAwareRouteDecoderV1,
    route_domains: Any,
) -> tuple[RouteHeadGradientRowTelemetry, ...]:
    domains = list(route_domains or ())
    if domains and isinstance(domains[0], str):
        domains = [tuple(domains)]

    def row(
        *,
        row_index: int,
        semantic_role: str,
        destinations: tuple[int, ...],
        weight_gradient: Tensor | None,
        bias_gradient: Tensor | None,
    ) -> RouteHeadGradientRowTelemetry:
        measurement = _gradient_tensor_measurement(weight_gradient)
        bias_value = (
            None
            if bias_gradient is None
            else float(bias_gradient.detach().reshape(-1)[0].cpu().item())
        )
        return RouteHeadGradientRowTelemetry(
            row_index=row_index,
            semantic_role=semantic_role,
            destination_uavs=destinations,
            grad_status=measurement.status,
            grad_norm=measurement.norm,
            grad_mean=measurement.mean,
            grad_signed_sum=measurement.signed_sum,
            bias_grad=bias_value,
            parameter_update_direction_mean=(
                -measurement.mean if measurement.mean is not None else None
            ),
            parameter_update_direction_signed_sum=(
                -measurement.signed_sum
                if measurement.signed_sum is not None
                else None
            ),
            bias_update_direction=(
                -bias_value if bias_value is not None else None
            ),
        )

    control_weight = route_head.control_head.weight.grad
    control_bias = route_head.control_head.bias.grad
    rows = [
        row(
            row_index=0,
            semantic_role="idle",
            destinations=(),
            weight_gradient=(
                None if control_weight is None else control_weight[0]
            ),
            bias_gradient=None if control_bias is None else control_bias[0],
        ),
        row(
            row_index=2,
            semantic_role="defer",
            destinations=(),
            weight_gradient=(
                None if control_weight is None else control_weight[1]
            ),
            bias_gradient=None if control_bias is None else control_bias[1],
        ),
    ]
    destinations = tuple(
        sorted(
            {
                value
                for domain in domains
                for value in domain
                if isinstance(value, int) and not isinstance(value, bool)
            }
            or set(range(route_head.uav_count))
        )
    )
    final_scorer = route_head.shared_option_scorer[-1]
    if not isinstance(final_scorer, nn.Linear) or final_scorer.bias is None:
        raise RouteTelemetryError("option-aware shared scorer has an invalid final layer")
    rows.insert(
        1,
        row(
            row_index=1,
            semantic_role="local_remote_shared",
            destinations=destinations,
            weight_gradient=(
                None
                if final_scorer.weight.grad is None
                else final_scorer.weight.grad[0]
            ),
            bias_gradient=final_scorer.bias.grad,
        ),
    )
    return tuple(rows)


def _hierarchical_option_route_gradient_rows(
    route_head: HierarchicalOptionAwareRouteDecoderV1,
    route_domains: Any,
) -> tuple[RouteHeadGradientRowTelemetry, ...]:
    """Map both hierarchical decision heads to stable route semantics."""

    domains = list(route_domains or ())
    if domains and isinstance(domains[0], str):
        domains = [tuple(domains)]
    destinations = tuple(
        sorted(
            {
                value
                for domain in domains
                for value in domain
                if isinstance(value, int) and not isinstance(value, bool)
            }
            or set(range(route_head.uav_count))
        )
    )

    top_level_output = route_head.top_level_head[-1]
    remote_destination_output = route_head.remote_destination_head[-1]
    if (
        not isinstance(top_level_output, nn.Linear)
        or top_level_output.bias is None
        or top_level_output.out_features != 3
    ):
        raise RouteTelemetryError(
            "hierarchical option-aware top-level head has an invalid final layer"
        )
    if (
        not isinstance(remote_destination_output, nn.Linear)
        or remote_destination_output.bias is None
        or remote_destination_output.out_features != 1
    ):
        raise RouteTelemetryError(
            "hierarchical option-aware remote-destination head has an invalid final layer"
        )

    def row(
        *,
        row_index: int,
        semantic_role: str,
        destination_uavs: tuple[int, ...],
        weight_gradient: Tensor | None,
        bias_gradient: Tensor | None,
    ) -> RouteHeadGradientRowTelemetry:
        measurement = _gradient_tensor_measurement(weight_gradient)
        bias_value = (
            None
            if bias_gradient is None
            else float(bias_gradient.detach().reshape(-1)[0].cpu().item())
        )
        return RouteHeadGradientRowTelemetry(
            row_index=row_index,
            semantic_role=semantic_role,
            destination_uavs=destination_uavs,
            grad_status=measurement.status,
            grad_norm=measurement.norm,
            grad_mean=measurement.mean,
            grad_signed_sum=measurement.signed_sum,
            bias_grad=bias_value,
            parameter_update_direction_mean=(
                -measurement.mean if measurement.mean is not None else None
            ),
            parameter_update_direction_signed_sum=(
                -measurement.signed_sum
                if measurement.signed_sum is not None
                else None
            ),
            bias_update_direction=(
                -bias_value if bias_value is not None else None
            ),
        )

    top_weight = top_level_output.weight.grad
    top_bias = top_level_output.bias.grad
    destination_weight = remote_destination_output.weight.grad
    destination_bias = remote_destination_output.bias.grad
    return (
        row(
            row_index=1,
            semantic_role="local",
            destination_uavs=(),
            weight_gradient=None if top_weight is None else top_weight[0],
            bias_gradient=None if top_bias is None else top_bias[0],
        ),
        row(
            row_index=2,
            semantic_role="defer",
            destination_uavs=(),
            weight_gradient=None if top_weight is None else top_weight[1],
            bias_gradient=None if top_bias is None else top_bias[1],
        ),
        row(
            row_index=3,
            semantic_role="remote",
            destination_uavs=destinations,
            weight_gradient=None if top_weight is None else top_weight[2],
            bias_gradient=None if top_bias is None else top_bias[2],
        ),
        row(
            row_index=3,
            semantic_role="remote_destination",
            destination_uavs=destinations,
            weight_gradient=(
                None if destination_weight is None else destination_weight[0]
            ),
            bias_gradient=destination_bias,
        ),
    )


def measure_route_head_gradients(
    route_head: nn.Module, route_domains: Any = None
) -> RouteHeadGradientTelemetry:
    """Read existing route-head gradients without changing or clipping them."""

    if isinstance(
        route_head,
        (
            CandidateAwareRouteDecoderV1,
            OptionAwareRouteDecoderV1,
            HierarchicalOptionAwareRouteDecoderV1,
        ),
    ):
        named_parameters = tuple(route_head.named_parameters())
        if isinstance(route_head, HierarchicalOptionAwareRouteDecoderV1):
            # Keep the existing total field while giving it an exact hierarchical
            # meaning: ||g_route|| = sqrt(||g_top||^2 + ||g_destination||^2).
            named_parameters = tuple(
                (name, parameter)
                for name, parameter in named_parameters
                if name.startswith(
                    ("top_level_head.", "remote_destination_head.")
                )
            )
        weight_parameters = tuple(
            parameter
            for name, parameter in named_parameters
            if not name.endswith("bias")
        )
        bias_parameters = tuple(
            parameter
            for name, parameter in named_parameters
            if name.endswith("bias")
        )
        with torch.no_grad():
            return RouteHeadGradientTelemetry(
                weight=_aggregate_gradient_measurement(weight_parameters),
                bias=_aggregate_gradient_measurement(bias_parameters),
                total=_aggregate_gradient_measurement(
                    tuple(parameter for _, parameter in named_parameters)
                ),
                rows=(
                    _candidate_route_gradient_rows(route_head, route_domains)
                    if isinstance(route_head, CandidateAwareRouteDecoderV1)
                    else (
                        _option_route_gradient_rows(route_head, route_domains)
                        if isinstance(route_head, OptionAwareRouteDecoderV1)
                        else _hierarchical_option_route_gradient_rows(
                            route_head, route_domains
                        )
                    )
                ),
            )
    if not isinstance(route_head, nn.Linear) or route_head.bias is None:
        raise TypeError(
            "route_head must be a Linear, CandidateAwareRouteDecoderV1, "
            "OptionAwareRouteDecoderV1, or HierarchicalOptionAwareRouteDecoderV1"
        )
    with torch.no_grad():
        weight = _gradient_measurement(route_head.weight)
        bias = _gradient_measurement(route_head.bias)
        present = tuple(
            item.norm for item in (weight, bias) if item.norm is not None
        )
        if not present:
            total = GradientMeasurement("none", None)
        else:
            total_norm = math.sqrt(math.fsum(value * value for value in present))
            total = GradientMeasurement(
                "zero" if total_norm == 0.0 else "finite_nonzero",
                total_norm,
            )
    return RouteHeadGradientTelemetry(
        weight=weight,
        bias=bias,
        total=total,
        rows=measure_route_head_gradient_rows(route_head, route_domains),
    )


def measure_route_head_gradient_rows(
    route_head: nn.Module, route_domains: Any = None
) -> tuple[RouteHeadGradientRowTelemetry, ...]:
    """Map existing route-head row gradients to local/defer/remote semantics."""

    if isinstance(route_head, CandidateAwareRouteDecoderV1):
        with torch.no_grad():
            return _candidate_route_gradient_rows(route_head, route_domains)
    if isinstance(route_head, OptionAwareRouteDecoderV1):
        with torch.no_grad():
            return _option_route_gradient_rows(route_head, route_domains)
    if isinstance(route_head, HierarchicalOptionAwareRouteDecoderV1):
        with torch.no_grad():
            return _hierarchical_option_route_gradient_rows(
                route_head, route_domains
            )
    if not isinstance(route_head, nn.Linear) or route_head.bias is None:
        raise TypeError("route_head has an unsupported module type")
    domains = list(route_domains or ())
    if domains and isinstance(domains[0], str):
        domains = [tuple(domains)]
    if not domains:
        domains = [tuple(range(route_head.out_features))]
    rows: list[RouteHeadGradientRowTelemetry] = []
    for row_index in range(route_head.out_features):
        values = [
            domain[row_index]
            for domain in domains
            if row_index < len(domain)
        ]
        destinations = tuple(sorted({
            value for value in values
            if isinstance(value, int) and not isinstance(value, bool)
        }, key=str))
        if any(type(value) is str and value == "local" for value in values):
            role = "local"
        elif any(type(value) is str and value == "defer" for value in values):
            role = "defer"
        elif destinations:
            role = "remote"

        elif any(type(value) is str and value == "idle" for value in values):
            role = "idle"
        else:
            role = "unknown"
        gradient = route_head.weight.grad[row_index] if route_head.weight.grad is not None else None
        measurement = GradientMeasurement("none", None)
        if gradient is not None:
            detached = gradient.detach()
            if not torch.isfinite(detached).all():
                measurement = GradientMeasurement("none", None)
            else:
                norm = float(torch.linalg.vector_norm(detached).cpu().item())
                measurement = GradientMeasurement(
                    "zero" if norm == 0.0 else "finite_nonzero", norm,
                    mean=float(detached.mean().cpu().item()),
                    signed_sum=float(detached.sum().cpu().item()),
                )
        bias_value = None
        if route_head.bias.grad is not None:
            bias_value = float(route_head.bias.grad[row_index].detach().cpu().item())
        rows.append(RouteHeadGradientRowTelemetry(
            row_index=row_index,
            semantic_role=role,
            destination_uavs=destinations,
            grad_status=measurement.status,
            grad_norm=measurement.norm,
            grad_mean=measurement.mean,
            grad_signed_sum=measurement.signed_sum,
            bias_grad=bias_value,
            parameter_update_direction_mean=-measurement.mean if measurement.mean is not None else None,
            parameter_update_direction_signed_sum=-measurement.signed_sum if measurement.signed_sum is not None else None,
            bias_update_direction=-bias_value if bias_value is not None else None,
        ))
    return tuple(rows)

def _count(mask: Tensor) -> int:

    return int(mask.sum().detach().cpu().item())


def _optional_mean(values: Tensor, mask: Tensor) -> float | None:
    selected = values.masked_select(mask)
    if selected.numel() == 0:
        return None
    return float(selected.mean().detach().cpu().item())


def _optional_population_std(values: Tensor, mask: Tensor) -> float | None:
    selected = values.masked_select(mask)
    if selected.numel() == 0:
        return None
    return float(selected.std(unbiased=False).detach().cpu().item())


def _flatten_proposals(
    proposals: Sequence[Sequence[Sequence[ActionProposal]]],
) -> tuple[ActionProposal, ...]:
    return tuple(
        proposal
        for batch in proposals
        for time_step in batch
        for proposal in time_step
    )


def collect_route_telemetry(
    *,
    policy: SequentialActionDistributionOutput,
    action_mask_batch: SequentialActionMaskBatch,
    proposals: Sequence[Sequence[Sequence[ActionProposal]]],
    advantage: Tensor,
    return_target: Tensor,
    sequence_valid_mask: Tensor,
    route_head: nn.Module,
    old_route_log_prob: Tensor | None = None,
    old_branch_log_probs: Tensor | None = None,
    td_residual: Tensor | None = None,
    shared_trunk: Any = None,
    epsilon_clip: float = 0.2,
    actor_ratio_mode: str = ActorRatioMode.JOINT.value,
    base_advantage: Tensor | None = None,
    route_advantage: Tensor | None = None,
    branch_advantage: Tensor | None = None,
    bootstrap_mask: Tensor | None = None,
    effective_route_gae_lambda: float | None = None,
    route_nstep_samples: Sequence[
        Sequence[Sequence[RouteNstepSample | None]]
    ] | None = None,
) -> RouteTelemetry:
    """Aggregate route telemetry without changing the PPO algorithm state."""

    if not isinstance(policy, SequentialActionDistributionOutput):
        raise TypeError("policy must be SequentialActionDistributionOutput")
    if not isinstance(action_mask_batch, SequentialActionMaskBatch):
        raise TypeError("action_mask_batch must be SequentialActionMaskBatch")
    try:
        ratio_mode = ActorRatioMode(actor_ratio_mode)
    except (TypeError, ValueError) as exc:
        raise RouteTelemetryError("actor_ratio_mode is invalid") from exc
    route_active = policy.active_branches["route"]
    if route_active.dtype != torch.bool or route_active.ndim != 3:
        raise RouteTelemetryError("route activity must be boolean [B,T,A]")
    batch, time, agents = route_active.shape
    if action_mask_batch.shape != (batch, time, agents):
        raise RouteTelemetryError("action-mask contracts differ from route axes")
    valid_credit_shapes = {(batch, time), (batch, time, agents)}
    if tuple(advantage.shape) not in valid_credit_shapes:
        raise RouteTelemetryError("advantage must have shape [B,T] or [B,T,A]")
    if tuple(return_target.shape) != tuple(advantage.shape):
        raise RouteTelemetryError(
            "return_target must share the scalar or per-agent advantage shape"
        )
    if td_residual is not None and tuple(td_residual.shape) != tuple(advantage.shape):
        raise RouteTelemetryError(
            "td_residual must share the scalar or per-agent advantage shape"
        )
    if base_advantage is not None and tuple(base_advantage.shape) not in valid_credit_shapes:
        raise RouteTelemetryError(
            "base_advantage must have shape [B,T] or [B,T,A]"
        )
    if route_advantage is not None and tuple(route_advantage.shape) != (
        batch,
        time,
        agents,
    ):
        raise RouteTelemetryError("route_advantage must have shape [B,T,A]")
    if branch_advantage is not None and tuple(branch_advantage.shape) != (
        batch,
        time,
        agents,
        len(ACTION_BRANCH_ORDER),
    ):
        raise RouteTelemetryError("branch_advantage must have shape [B,T,A,7]")
    if bootstrap_mask is not None and (
        tuple(bootstrap_mask.shape) != (batch, time)
        or bootstrap_mask.dtype != torch.bool
    ):
        raise RouteTelemetryError("bootstrap_mask must be boolean [B,T]")
    if effective_route_gae_lambda is not None and (
        isinstance(effective_route_gae_lambda, bool)
        or not isinstance(effective_route_gae_lambda, (int, float))
        or not math.isfinite(float(effective_route_gae_lambda))
        or not 0.0 <= float(effective_route_gae_lambda) <= 1.0
    ):
        raise RouteTelemetryError(
            "effective_route_gae_lambda must lie in [0, 1] or be NA"
        )
    if route_nstep_samples is not None and (
        len(route_nstep_samples) != batch
        or any(len(chunk) != time for chunk in route_nstep_samples)
        or any(
            len(time_step) != agents
            for chunk in route_nstep_samples
            for time_step in chunk
        )
    ):
        raise RouteTelemetryError(
            "route_nstep_samples must have shape [B][T][A]"
        )
    if (
        tuple(sequence_valid_mask.shape) != (batch, time)
        or sequence_valid_mask.dtype != torch.bool
    ):
        raise RouteTelemetryError("sequence_valid_mask must be boolean [B,T]")

    route_probabilities = policy.probabilities["route"].detach()
    route_logits = policy.raw_logits["route"].detach()
    route_masks = policy.action_masks["route"].detach().bool()
    route_indices = policy.action_indices["route"].detach().long()
    route_entropy = policy.branch_entropies["route"].detach()
    dimension = route_probabilities.shape[-1]
    expected_vector_shape = (batch, time, agents, dimension)
    if (
        tuple(route_probabilities.shape) != expected_vector_shape
        or tuple(route_logits.shape) != expected_vector_shape
        or tuple(route_masks.shape) != expected_vector_shape
        or tuple(route_indices.shape) != (batch, time, agents)
        or tuple(route_entropy.shape) != (batch, time, agents)
    ):
        raise RouteTelemetryError("route policy tensors have incompatible shapes")

    contracts = action_mask_batch.flattened()
    flattened_proposals = _flatten_proposals(proposals)
    if len(contracts) != batch * time * agents or len(flattened_proposals) != len(contracts):
        raise RouteTelemetryError("proposal or contract count differs from route axes")

    remote_domain_rows: list[list[bool]] = []
    local_domain_rows: list[list[bool]] = []
    selected_local_rows: list[bool] = []
    selected_remote_rows: list[bool] = []
    selected_defer_rows: list[bool] = []
    domains: list[tuple[object, ...]] = []
    source_uavs: list[object] = []
    for contract, proposal in zip(contracts, flattened_proposals):
        domain = tuple(contract.route_domain)
        if len(domain) != dimension:
            raise RouteTelemetryError("route domain differs from policy dimension")
        remote_row = [
            isinstance(value, int) and not isinstance(value, bool) for value in domain
        ]
        local_row = [type(value) is str and value == "local" for value in domain]
        if sum(local_row) != 1:
            raise RouteTelemetryError("route domain must contain exactly one local action")
        domains.append(domain)
        source_uavs.append(
            getattr(contract, "source_uav", getattr(contract, "uav_id", None))
        )
        remote_domain_rows.append(remote_row)
        local_domain_rows.append(local_row)
        selected = proposal.route
        selected_local_rows.append(type(selected) is str and selected == "local")
        selected_defer_rows.append(type(selected) is str and selected == "defer")
        selected_remote_rows.append(
            isinstance(selected, int) and not isinstance(selected, bool)
        )

    device = route_probabilities.device
    remote_domain = torch.tensor(remote_domain_rows, dtype=torch.bool, device=device).reshape(
        expected_vector_shape
    )
    local_domain = torch.tensor(local_domain_rows, dtype=torch.bool, device=device).reshape(
        expected_vector_shape
    )
    selected_local = torch.tensor(selected_local_rows, dtype=torch.bool, device=device).reshape(
        batch, time, agents
    )
    selected_remote = torch.tensor(selected_remote_rows, dtype=torch.bool, device=device).reshape(
        batch, time, agents
    )
    selected_defer = torch.tensor(selected_defer_rows, dtype=torch.bool, device=device).reshape(
        batch, time, agents
    )

    with torch.no_grad():
        valid_actor = sequence_valid_mask.to(device=device).unsqueeze(-1).expand_as(route_active)
        active = route_active.detach() & valid_actor
        selected_partition = selected_local | selected_remote | selected_defer
        if torch.any(active & ~selected_partition):
            raise RouteTelemetryError("active route sample is not local, remote, or defer")
        legal_remote_actions = route_masks & remote_domain
        legal_remote = active & torch.any(legal_remote_actions, dim=-1)
        if torch.any(active & selected_remote & ~legal_remote):
            raise RouteTelemetryError("selected remote lacks a legal remote action")

        selected_probability = torch.gather(
            route_probabilities, -1, route_indices.unsqueeze(-1)
        ).squeeze(-1)
        local_probability = (route_probabilities * local_domain.to(route_probabilities.dtype)).sum(dim=-1)
        defer_domain = torch.tensor(
            [[type(value) is str and value == "defer" for value in domain] for domain in domains],
            dtype=torch.bool,
            device=device,
        ).reshape(expected_vector_shape)
        defer_probability = (
            route_probabilities * defer_domain.to(route_probabilities.dtype)
        ).sum(dim=-1)
        legal_remote_probability = route_probabilities.masked_fill(
            ~legal_remote_actions, 0.0
        )
        total_remote_probability = legal_remote_probability.sum(dim=-1)
        best_remote_probability = route_probabilities.masked_fill(
            ~legal_remote_actions, -torch.inf
        ).max(dim=-1).values
        best_remote_logit = route_logits.masked_fill(
            ~legal_remote_actions, -torch.inf
        ).max(dim=-1).values
        legal_remote_logit_sum = torch.logsumexp(
            route_logits.masked_fill(~legal_remote_actions, -torch.inf), dim=-1
        )
        local_logit = (route_logits * local_domain.to(route_logits.dtype)).sum(dim=-1)
        probability_margin = local_probability - best_remote_probability
        logit_margin = local_logit - best_remote_logit
        total_logit_margin = local_logit - legal_remote_logit_sum
        if not torch.isfinite(best_remote_probability.masked_select(legal_remote)).all():
            raise RouteTelemetryError("best legal-remote probability is not finite")
        if not torch.isfinite(best_remote_logit.masked_select(legal_remote)).all():
            raise RouteTelemetryError("best legal-remote logit is not finite")

        expanded_advantage = advantage.detach().to(device=device)
        expanded_return = return_target.detach().to(device=device)
        if expanded_advantage.ndim == 2:
            expanded_advantage = expanded_advantage.unsqueeze(-1).expand_as(route_active)
            expanded_return = expanded_return.unsqueeze(-1).expand_as(route_active)
        expanded_base_advantage = (
            advantage if base_advantage is None else base_advantage
        ).detach().to(device=device)
        if expanded_base_advantage.ndim == 2:
            expanded_base_advantage = expanded_base_advantage.unsqueeze(-1).expand_as(
                route_active
            )
        expanded_new_route_advantage = (
            advantage if route_advantage is None else route_advantage
        ).detach().to(device=device)
        if expanded_new_route_advantage.ndim == 2:
            expanded_new_route_advantage = (
                expanded_new_route_advantage.unsqueeze(-1).expand_as(route_active)
            )
        expanded_route_advantage_delta = (
            expanded_new_route_advantage - expanded_base_advantage
        )
        raw_branch_active = torch.stack(
            [
                policy.active_branches[branch].detach().to(device=device)
                for branch in ACTION_BRANCH_ORDER
            ],
            dim=-1,
        )
        if tuple(raw_branch_active.shape) != (
            batch,
            time,
            agents,
            len(ACTION_BRANCH_ORDER),
        ):
            raise RouteTelemetryError("active branch tensor must have shape [B,T,A,7]")
        valid_branch_active = raw_branch_active & valid_actor.unsqueeze(-1)
        new_branch_log_probs = torch.stack(
            [
                policy.branch_log_probs[branch].detach().to(device=device)
                for branch in ACTION_BRANCH_ORDER
            ],
            dim=-1,
        )
        branch_ppo_advantage = (
            expanded_advantage.unsqueeze(-1).expand_as(new_branch_log_probs)
            if branch_advantage is None
            else branch_advantage.detach().to(device=device)
        )
        branch_ppo = compute_branch_ppo_dynamics(
            old_branch_log_probs,
            new_branch_log_probs,
            valid_branch_active,
            branch_ppo_advantage,
            epsilon_clip,
        )
        active_branch_matrix = tuple(
            tuple(
                tuple(bool(value) for value in agent_row)
                for agent_row in time_step
            )
            for time_step in raw_branch_active.reshape(
                batch * time,
                agents,
                len(ACTION_BRANCH_ORDER),
            ).detach().cpu().tolist()
        )
        td_values = td_residual.detach().to(device=device) if td_residual is not None else None
        if td_values is not None and td_values.ndim == 2:
            td_values = td_values.unsqueeze(-1).expand_as(route_active)
        local_group = active & selected_local
        remote_group = active & selected_remote
        defer_group = active & selected_defer
        legal_local_group = legal_remote & selected_local
        legal_remote_group = legal_remote & selected_remote
        category_masks = {
            "local": local_group,
            "remote": remote_group,
            "defer": defer_group,
        }
        gradients = measure_route_head_gradients(route_head, domains)
        trunk_measurement = measure_shared_trunk_gradients(
            shared_trunk if shared_trunk is not None else ()
        )

        legal_count = _count(legal_remote)
        active_count = _count(active)
        local_count = _count(local_group)
        remote_count = _count(remote_group)
        defer_count = _count(defer_group)
        legal_local_count = _count(legal_local_group)
        legal_selected_remote_count = _count(legal_remote_group)
        old_route = old_route_log_prob
        if old_route is None and old_branch_log_probs is not None:
            old_route = old_branch_log_probs[..., 0]
        if old_route is not None:
            old_route = old_route.detach().to(device=device).reshape(batch, time, agents)
        new_route = policy.branch_log_probs["route"].detach().to(device=device).reshape(
            batch, time, agents
        )
        route_ppo = compute_route_only_ppo_dynamics(
            old_route, new_route, active, category_masks, epsilon_clip
        )

        advantage_distributions = {}
        return_distributions = {}
        td_distributions = {}
        group_masks = {
            "route_active": active,
            "local_route": local_group,
            "remote_route": remote_group,
            "defer_route": defer_group,
            "legal_remote_local": legal_local_group,
            "legal_remote_remote": legal_remote_group,
        }
        for name, mask in group_masks.items():
            advantage_distributions[name] = summarize_distribution(expanded_advantage, mask)
            return_distributions[name] = summarize_distribution(expanded_return, mask)
            td_distributions[name] = (
                summarize_distribution(td_values, mask)
                if td_values is not None
                else RouteDistributionSummary()
            )
        base_route_advantage_summary = summarize_distribution(
            expanded_base_advantage,
            active,
        )
        new_route_advantage_summary = summarize_distribution(
            expanded_new_route_advantage,
            active,
        )
        route_advantage_delta_summary = summarize_distribution(
            expanded_route_advantage_delta,
            active,
        )
        route_credit_distributions = {}
        if route_nstep_samples is not None:
            for name, mask in (
                ("local_route", local_group),
                ("remote_route", remote_group),
                ("defer_route", defer_group),
            ):
                route_credit_distributions[name] = {
                    "base_route_advantage": summarize_distribution(
                        expanded_base_advantage, mask
                    ),
                    "nstep_route_advantage": summarize_distribution(
                        expanded_new_route_advantage, mask
                    ),
                    "route_advantage_delta": summarize_distribution(
                        expanded_route_advantage_delta, mask
                    ),
                }

        def mean_active(values: Tensor) -> float | None:
            return _optional_mean(values, active)

        def mean_legal(values: Tensor) -> float | None:
            return _optional_mean(values, legal_remote)

        destination_sum: dict[str, float] = {}
        destination_count: dict[str, int] = {}
        flat_remote = legal_remote_actions.reshape(-1, dimension)
        flat_probability = route_probabilities.reshape(-1, dimension)
        flat_legal = legal_remote.reshape(-1)
        for flat_index in range(flat_probability.shape[0]):
            if not bool(flat_legal[flat_index]):
                continue
            for row_index in range(dimension):
                if bool(flat_remote[flat_index, row_index]):
                    key = str(domains[flat_index][row_index])
                    destination_sum[key] = destination_sum.get(key, 0.0) + float(
                        flat_probability[flat_index, row_index].item()
                    )
                    destination_count[key] = destination_count.get(key, 0) + 1

        samples: list[RouteSampleTelemetry] = []
        flat_active = active.reshape(-1)
        flat_indices = route_indices.reshape(-1)
        flat_selected = selected_probability.reshape(-1)
        flat_old = old_route.reshape(-1) if old_route is not None else None
        flat_new = new_route.reshape(-1)
        flat_adv = expanded_advantage.reshape(-1)
        flat_base_adv = expanded_base_advantage.reshape(-1)
        flat_new_route_adv = expanded_new_route_advantage.reshape(-1)
        flat_route_adv_delta = expanded_route_advantage_delta.reshape(-1)
        flat_ret = expanded_return.reshape(-1)
        flat_td = td_values.reshape(-1) if td_values is not None else None
        flat_total_remote = total_remote_probability.reshape(-1)
        flat_local = local_probability.reshape(-1)
        flat_defer = defer_probability.reshape(-1)
        flat_best = best_remote_probability.reshape(-1)
        flat_nstep_samples = (
            None
            if route_nstep_samples is None
            else tuple(
                sample
                for chunk in route_nstep_samples
                for time_step in chunk
                for sample in time_step
            )
        )
        if flat_nstep_samples is not None:
            for flat_index, event in enumerate(flat_nstep_samples):
                if bool(flat_active[flat_index]) != (event is not None):
                    raise RouteTelemetryError(
                        "route activity and N-step sample provenance disagree"
                    )
        flat_bootstrap = (
            None
            if bootstrap_mask is None
            else bootstrap_mask.detach().to(device="cpu").reshape(-1)
        )
        base_credit_is_per_agent = (
            (advantage if base_advantage is None else base_advantage).ndim == 3
        )

        def trace_provenance(index: int) -> tuple[int | None, str | None, str | None]:
            if flat_bootstrap is None:
                return None, None, None
            boundary_index = next(
                (
                    candidate
                    for candidate in range(index, flat_bootstrap.numel())
                    if not bool(flat_bootstrap[candidate])
                ),
                None,
            )
            if boundary_index is not None:
                return (
                    boundary_index - index + 1,
                    "episode_boundary",
                    "none_episode_boundary",
                )
            return (
                flat_bootstrap.numel() - index,
                "rollout_boundary",
                (
                    "saved_source_agent_v_next"
                    if base_credit_is_per_agent
                    else "saved_team_v_next"
                ),
            )
        for flat_index in torch.nonzero(flat_active, as_tuple=False).reshape(-1).tolist():
            b = flat_index // (time * agents)
            remainder = flat_index % (time * agents)
            t = remainder // agents
            a = remainder % agents
            action_index = int(flat_indices[flat_index].item())
            domain = domains[flat_index]
            selected_value = flattened_proposals[flat_index].route
            category = (
                "local" if bool(selected_local.reshape(-1)[flat_index])
                else "remote" if bool(selected_remote.reshape(-1)[flat_index])
                else "defer"
            )
            legal_indices = [
                index for index, value in enumerate(domain)
                if isinstance(value, int) and not isinstance(value, bool) and bool(legal_remote_actions.reshape(-1, dimension)[flat_index, index])
            ]
            destination_probabilities = {
                str(domain[index]): float(flat_probability[flat_index, index].item())
                for index in legal_indices
            }
            old_value = float(flat_old[flat_index].item()) if flat_old is not None else None
            new_value = float(flat_new[flat_index].item())
            log_ratio = new_value - old_value if old_value is not None else None
            ratio = math.exp(log_ratio) if log_ratio is not None else None
            trace_length, trace_end_kind, bootstrap_source = trace_provenance(
                b * time + t
            )
            nstep_event = (
                None
                if flat_nstep_samples is None
                else flat_nstep_samples[flat_index]
            )
            if nstep_event is not None:
                if (
                    nstep_event.source_uav != source_uavs[flat_index]
                    or nstep_event.category != category
                    or not math.isclose(
                        nstep_event.base_advantage,
                        float(flat_base_adv[flat_index].item()),
                        rel_tol=0.0,
                        abs_tol=1.0e-6,
                    )
                    or not math.isclose(
                        nstep_event.nstep_advantage,
                        float(flat_new_route_adv[flat_index].item()),
                        rel_tol=0.0,
                        abs_tol=1.0e-6,
                    )
                    or not math.isclose(
                        nstep_event.advantage_delta,
                        float(flat_route_adv_delta[flat_index].item()),
                        rel_tol=0.0,
                        abs_tol=1.0e-6,
                    )
                ):
                    raise RouteTelemetryError(
                        "N-step provenance differs from the actor advantage"
                    )
                trace_length = nstep_event.horizon
                trace_end_kind = nstep_event.stop_kind
                bootstrap_source = (
                    "saved_source_agent_v_next"
                    if nstep_event.bootstrap_used
                    else "none_episode_boundary"
                )
            samples.append(RouteSampleTelemetry(
                batch_index=b,
                time_index=t,
                agent_index=a,
                source_uav=source_uavs[flat_index],
                category=category,
                legal_remote_destinations=tuple(domain[index] for index in legal_indices),
                remote_probability_by_destination=destination_probabilities,
                local_probability=float(flat_local[flat_index].item()),
                defer_probability=float(flat_defer[flat_index].item()),
                total_remote_probability_mass=float(flat_total_remote[flat_index].item()),
                best_remote_probability=(
                    float(flat_best[flat_index].item()) if legal_indices else None
                ),
                number_of_legal_remote_destinations=len(legal_indices),
                selected_route_action_index=action_index,
                selected_destination_uav=(
                    source_uavs[flat_index] if category == "local"
                    else selected_value if category == "remote" else None
                ),
                old_route_log_prob=old_value,
                new_route_log_prob=new_value,
                route_log_ratio=log_ratio,
                route_ratio=ratio,
                route_approx_kl=(ratio - 1.0 - log_ratio) if ratio is not None else None,
                route_clip_indicator=(
                    (ratio < 1.0 - epsilon_clip or ratio > 1.0 + epsilon_clip)
                    if ratio is not None else None
                ),
                advantage=float(flat_adv[flat_index].item()),
                return_target=float(flat_ret[flat_index].item()),
                td_residual=float(flat_td[flat_index].item()) if flat_td is not None else None,
                base_route_advantage=float(flat_base_adv[flat_index].item()),
                new_route_advantage=float(flat_new_route_adv[flat_index].item()),
                route_advantage_delta=float(flat_route_adv_delta[flat_index].item()),
                effective_route_gae_lambda=(
                    None
                    if effective_route_gae_lambda is None
                    else float(effective_route_gae_lambda)
                ),
                credit_trace_length_slots=trace_length,
                trace_end_kind=trace_end_kind,
                bootstrap_source=bootstrap_source,
                task_id=(
                    None if nstep_event is None else nstep_event.task_id
                ),
                branch_event_key=(
                    None
                    if nstep_event is None
                    else nstep_event.branch_event_key
                ),
                formal_route_event_key=(
                    None
                    if nstep_event is None
                    else nstep_event.formal_route_event_key
                ),
                global_rollout_index=(
                    None
                    if nstep_event is None
                    else nstep_event.global_rollout_index
                ),
                policy_version=(
                    None
                    if nstep_event is None
                    else nstep_event.policy_version
                ),
                nstep_route_advantage=(
                    None
                    if nstep_event is None
                    else nstep_event.nstep_advantage
                ),
                nstep_horizon=(
                    None if nstep_event is None else nstep_event.horizon
                ),
                nstep_stop_kind=(
                    None if nstep_event is None else nstep_event.stop_kind
                ),
                terminal_before_rollout=(
                    None
                    if nstep_event is None
                    else nstep_event.terminal_before_rollout
                ),
                bootstrap_used=(
                    None
                    if nstep_event is None
                    else nstep_event.bootstrap_used
                ),
            ))

        return RouteTelemetry(
            schema_version=ROUTE_TELEMETRY_SCHEMA_VERSION,
            route_branch_active_count=active_count,
            legal_remote_route_count=legal_count,
            route_selected_local_count=local_count,
            route_selected_remote_count=remote_count,
            route_selected_defer_count=defer_count,
            remote_selection_rate_given_legal_remote=(
                None if legal_count == 0 else remote_count / legal_count
            ),
            route_entropy_valid_sample_count=active_count,
            route_entropy_mean=_optional_mean(route_entropy, active),
            legal_remote_probability_valid_sample_count=legal_count,
            mean_selected_route_probability=_optional_mean(
                selected_probability, legal_remote
            ),
            mean_local_probability=_optional_mean(local_probability, legal_remote),
            mean_best_remote_probability=_optional_mean(
                best_remote_probability, legal_remote
            ),
            mean_local_minus_best_remote_probability=_optional_mean(
                probability_margin, legal_remote
            ),
            mean_local_minus_best_remote_logit_margin=_optional_mean(
                logit_margin, legal_remote
            ),
            route_active_advantage_valid_sample_count=active_count,
            route_active_advantage_mean=_optional_mean(expanded_advantage, active),
            route_active_advantage_std=_optional_population_std(
                expanded_advantage, active
            ),
            route_active_return_mean=_optional_mean(expanded_return, active),
            route_active_return_std=_optional_population_std(expanded_return, active),
            local_route_valid_sample_count=local_count,
            local_route_advantage_mean=_optional_mean(expanded_advantage, local_group),
            local_route_return_mean=_optional_mean(expanded_return, local_group),
            remote_route_valid_sample_count=remote_count,
            remote_route_advantage_mean=_optional_mean(expanded_advantage, remote_group),
            remote_route_return_mean=_optional_mean(expanded_return, remote_group),
            defer_route_valid_sample_count=defer_count,
            defer_route_advantage_mean=_optional_mean(expanded_advantage, defer_group),
            defer_route_return_mean=_optional_mean(expanded_return, defer_group),
            legal_remote_local_valid_sample_count=legal_local_count,
            legal_remote_local_advantage_mean=_optional_mean(
                expanded_advantage, legal_local_group
            ),
            legal_remote_local_return_mean=_optional_mean(
                expanded_return, legal_local_group
            ),
            legal_remote_remote_valid_sample_count=legal_selected_remote_count,
            legal_remote_remote_advantage_mean=_optional_mean(
                expanded_advantage, legal_remote_group
            ),
            legal_remote_remote_return_mean=_optional_mean(
                expanded_return, legal_remote_group
            ),
            route_head_weight_grad_status=gradients.weight.status,
            route_head_weight_grad_norm=gradients.weight.norm,
            route_head_bias_grad_status=gradients.bias.status,
            route_head_bias_grad_norm=gradients.bias.norm,
            route_head_total_grad_status=gradients.total.status,
            route_head_total_grad_norm=gradients.total.norm,
            route_probability_valid_sample_count=active_count,
            base_route_advantage_valid_sample_count=(
                base_route_advantage_summary.count
            ),
            base_route_advantage_mean=base_route_advantage_summary.mean,
            base_route_advantage_std=base_route_advantage_summary.std,
            base_route_advantage_median=base_route_advantage_summary.median,
            base_route_advantage_p90_absolute_magnitude=(
                base_route_advantage_summary.p90_absolute_magnitude
            ),
            base_route_advantage_p95_absolute_magnitude=(
                base_route_advantage_summary.p95_absolute_magnitude
            ),
            new_route_advantage_valid_sample_count=(
                new_route_advantage_summary.count
            ),
            new_route_advantage_mean=new_route_advantage_summary.mean,
            new_route_advantage_std=new_route_advantage_summary.std,
            new_route_advantage_median=new_route_advantage_summary.median,
            new_route_advantage_p90_absolute_magnitude=(
                new_route_advantage_summary.p90_absolute_magnitude
            ),
            new_route_advantage_p95_absolute_magnitude=(
                new_route_advantage_summary.p95_absolute_magnitude
            ),
            nstep_route_advantage_valid_sample_count=(
                new_route_advantage_summary.count
                if route_nstep_samples is not None
                else 0
            ),
            nstep_route_advantage_mean=(
                new_route_advantage_summary.mean
                if route_nstep_samples is not None
                else None
            ),
            nstep_route_advantage_std=(
                new_route_advantage_summary.std
                if route_nstep_samples is not None
                else None
            ),
            nstep_route_advantage_median=(
                new_route_advantage_summary.median
                if route_nstep_samples is not None
                else None
            ),
            nstep_route_advantage_p90_absolute_magnitude=(
                new_route_advantage_summary.p90_absolute_magnitude
                if route_nstep_samples is not None
                else None
            ),
            nstep_route_advantage_p95_absolute_magnitude=(
                new_route_advantage_summary.p95_absolute_magnitude
                if route_nstep_samples is not None
                else None
            ),
            route_advantage_delta_valid_sample_count=(
                route_advantage_delta_summary.count
            ),
            route_advantage_delta_mean=route_advantage_delta_summary.mean,
            route_advantage_delta_std=route_advantage_delta_summary.std,
            route_advantage_delta_median=route_advantage_delta_summary.median,
            route_advantage_delta_p90_absolute_magnitude=(
                route_advantage_delta_summary.p90_absolute_magnitude
            ),
            route_advantage_delta_p95_absolute_magnitude=(
                route_advantage_delta_summary.p95_absolute_magnitude
            ),
            mean_selected_route_probability_all_route_active=mean_active(selected_probability),
            mean_local_probability_all_route_active=mean_active(local_probability),
            mean_defer_probability_all_route_active=mean_active(defer_probability),
            mean_total_remote_probability_mass=mean_active(total_remote_probability),
            mean_number_of_legal_remote_destinations=_optional_mean(
                legal_remote_actions.sum(dim=-1).float(), active
            ),
            mean_local_minus_total_remote_probability=mean_active(
                local_probability - total_remote_probability
            ),
            mean_local_minus_total_remote_logit_margin=mean_legal(total_logit_margin),
            mean_remote_probability_by_destination={
                key: destination_sum[key] / destination_count[key]
                for key in destination_sum if destination_count[key]
            },
            remote_probability_destination_valid_samples=destination_count,
            route_head_gradient_rows=gradients.rows,
            route_ppo_dynamics=route_ppo,
            actor_ratio_mode=ratio_mode.value,
            branch_ppo_dynamics=branch_ppo,
            active_branch_matrix=active_branch_matrix,
            _advantage_distributions=advantage_distributions,
            _return_distributions=return_distributions,
            _td_residual_distributions=td_distributions,
            _route_credit_distributions=route_credit_distributions,
            shared_trunk_grad_status=trunk_measurement.status,
            shared_trunk_grad_norm=trunk_measurement.norm,
            route_head_weight_grad_mean=gradients.weight.mean,
            route_head_weight_grad_signed_sum=gradients.weight.signed_sum,
            route_head_bias_grad_mean=gradients.bias.mean,
            route_head_bias_grad_signed_sum=gradients.bias.signed_sum,
            samples=tuple(samples),
        )



@dataclass(frozen=True)
class RouteDistributionSummary:
    """Distribution summary for a fixed, already-selected route group."""

    count: int = 0
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    p25: float | None = None
    p75: float | None = None
    positive_fraction: float | None = None
    negative_fraction: float | None = None
    p90_absolute_magnitude: float | None = None
    p95_absolute_magnitude: float | None = None

    def record(
        self,
        prefix: str,
        *,
        include_quantiles: bool = True,
        include_absolute_quantiles: bool = False,
    ) -> dict[str, object]:
        result = {
            f"{prefix}_valid_sample_count": self.count,
            f"{prefix}_mean": self.mean,
            f"{prefix}_std": self.std,
            f"{prefix}_median": self.median,
        }
        if include_quantiles:
            result.update({
                f"{prefix}_p25": self.p25,
                f"{prefix}_p75": self.p75,
                f"{prefix}_positive_fraction": self.positive_fraction,
                f"{prefix}_negative_fraction": self.negative_fraction,
            })
        if include_absolute_quantiles:
            result.update({
                f"{prefix}_p90_absolute_magnitude": (
                    self.p90_absolute_magnitude
                ),
                f"{prefix}_p95_absolute_magnitude": (
                    self.p95_absolute_magnitude
                ),
            })
        return result


def summarize_distribution(values: Tensor, mask: Tensor | None = None) -> RouteDistributionSummary:
    selected = values.detach().float()
    if mask is not None:
        selected = selected.masked_select(mask.to(device=selected.device, dtype=torch.bool))
    selected = selected[torch.isfinite(selected)]
    if selected.numel() == 0:
        return RouteDistributionSummary()
    ordered = selected.sort().values
    return RouteDistributionSummary(
        count=int(selected.numel()),
        mean=float(selected.mean().cpu().item()),
        std=float(selected.std(unbiased=False).cpu().item()),
        median=float(median(selected.cpu().tolist())),
        p25=float(torch.quantile(selected, 0.25).cpu().item()),
        p75=float(torch.quantile(selected, 0.75).cpu().item()),
        positive_fraction=float((selected > 0).float().mean().cpu().item()),
        negative_fraction=float((selected < 0).float().mean().cpu().item()),
        p90_absolute_magnitude=float(
            torch.quantile(selected.abs(), 0.90).cpu().item()
        ),
        p95_absolute_magnitude=float(
            torch.quantile(selected.abs(), 0.95).cpu().item()
        ),
    )


@dataclass(frozen=True)
class RoutePPODynamicsGroup:
    count: int = 0
    old_log_prob_mean: float | None = None
    new_log_prob_mean: float | None = None
    log_ratio_mean: float | None = None
    ratio_mean: float | None = None
    approx_kl_mean: float | None = None
    clip_indicator_fraction: float | None = None
    clip_fraction: float | None = None

    def record(self, prefix: str) -> dict[str, object]:
        return {
            f"{prefix}_count": self.count,
            f"{prefix}_old_log_prob_mean": self.old_log_prob_mean,
            f"{prefix}_new_log_prob_mean": self.new_log_prob_mean,
            f"{prefix}_log_ratio_mean": self.log_ratio_mean,
            f"{prefix}_ratio_mean": self.ratio_mean,
            f"{prefix}_approx_kl_mean": self.approx_kl_mean,
            f"{prefix}_clip_indicator_fraction": self.clip_indicator_fraction,
            f"{prefix}_clip_fraction": self.clip_fraction,
        }


@dataclass(frozen=True)
class RoutePPODynamics:
    epsilon_clip: float
    overall: RoutePPODynamicsGroup
    groups: Mapping[str, RoutePPODynamicsGroup] = field(default_factory=dict)

    def record(self) -> dict[str, object]:
        result: dict[str, object] = {"route_ppo_epsilon_clip": self.epsilon_clip}
        result.update(self.overall.record("route_ppo_overall"))
        for category in ROUTE_ACTION_CATEGORIES:
            result.update(self.groups.get(category, RoutePPODynamicsGroup()).record(
                f"{category}_route_ppo"
            ))
        return result


@dataclass(frozen=True)
class RouteHeadGradientRowTelemetry:
    row_index: int
    semantic_role: str
    destination_uavs: tuple[object, ...] = ()
    grad_status: str = "none"
    grad_norm: float | None = None
    grad_mean: float | None = None
    grad_signed_sum: float | None = None
    bias_grad: float | None = None
    parameter_update_direction_mean: float | None = None
    parameter_update_direction_signed_sum: float | None = None
    bias_update_direction: float | None = None

    def record(self) -> dict[str, object]:
        return {
            "row_index": self.row_index,
            "semantic_role": self.semantic_role,
            "destination_uavs": list(self.destination_uavs),
            "grad_status": self.grad_status,
            "grad_norm": self.grad_norm,
            "grad_mean": self.grad_mean,
            "grad_signed_sum": self.grad_signed_sum,
            "bias_grad": self.bias_grad,
            "parameter_update_direction_mean": self.parameter_update_direction_mean,
            "parameter_update_direction_signed_sum": self.parameter_update_direction_signed_sum,
            "bias_update_direction": self.bias_update_direction,
        }


@dataclass(frozen=True)
class RouteSampleTelemetry:
    batch_index: int
    time_index: int
    agent_index: int
    source_uav: object
    category: str
    legal_remote_destinations: tuple[object, ...]
    remote_probability_by_destination: Mapping[str, float]
    local_probability: float | None
    defer_probability: float | None
    total_remote_probability_mass: float | None
    best_remote_probability: float | None
    number_of_legal_remote_destinations: int
    selected_route_action_index: int
    selected_destination_uav: object
    old_route_log_prob: float | None
    new_route_log_prob: float | None
    route_log_ratio: float | None
    route_ratio: float | None
    route_approx_kl: float | None
    route_clip_indicator: bool | None
    advantage: float | None
    return_target: float | None
    td_residual: float | None
    base_route_advantage: float | None = None
    new_route_advantage: float | None = None
    route_advantage_delta: float | None = None
    effective_route_gae_lambda: float | None = None
    credit_trace_length_slots: int | None = None
    trace_end_kind: str | None = None
    bootstrap_source: str | None = None
    task_id: object | None = None
    branch_event_key: tuple[object, ...] | None = None
    formal_route_event_key: tuple[object, ...] | None = None
    global_rollout_index: int | None = None
    policy_version: int | None = None
    nstep_route_advantage: float | None = None
    nstep_horizon: int | None = None
    nstep_stop_kind: str | None = None
    terminal_before_rollout: bool | None = None
    bootstrap_used: bool | None = None

    def __post_init__(self) -> None:
        if self.number_of_legal_remote_destinations != len(self.legal_remote_destinations):
            raise RouteTelemetryError("legal remote destination count is inconsistent")
        if self.total_remote_probability_mass is not None:
            total = sum(self.remote_probability_by_destination.values())
            if not math.isclose(total, self.total_remote_probability_mass, rel_tol=1e-6, abs_tol=1e-6):
                raise RouteTelemetryError("total remote probability is not the legal-destination sum")
        nstep_fields = (
            self.branch_event_key,
            self.global_rollout_index,
            self.policy_version,
            self.nstep_route_advantage,
            self.nstep_horizon,
            self.nstep_stop_kind,
            self.terminal_before_rollout,
            self.bootstrap_used,
        )
        if any(value is not None for value in nstep_fields):
            if any(value is None for value in nstep_fields):
                raise RouteTelemetryError(
                    "N-step route-sample provenance must be complete"
                )
            if self.task_id is None:
                raise RouteTelemetryError(
                    "N-step route-sample provenance requires task_id"
                )
            if self.category == "defer":
                if self.formal_route_event_key is not None:
                    raise RouteTelemetryError(
                        "defer must not expose a formal route-event key"
                    )
            elif self.formal_route_event_key is None:
                raise RouteTelemetryError(
                    "local/remote N-step samples require a formal key"
                )
            if self.advantage is None or not math.isclose(
                self.advantage,
                float(self.nstep_route_advantage),
                rel_tol=0.0,
                abs_tol=1.0e-6,
            ):
                raise RouteTelemetryError(
                    "actual route sample advantage differs from N-step value"
                )

    def record(self) -> dict[str, object]:
        return {
            "route_sample_batch_index": self.batch_index,
            "route_sample_time_index": self.time_index,
            "route_sample_agent_index": self.agent_index,
            "route_sample_source_uav": self.source_uav,
            "route_sample_category": self.category,
            "route_sample_legal_remote_destinations": list(self.legal_remote_destinations),
            "route_sample_remote_probability_by_destination": dict(self.remote_probability_by_destination),
            "route_sample_local_probability": self.local_probability,
            "route_sample_defer_probability": self.defer_probability,
            "route_sample_total_remote_probability_mass": self.total_remote_probability_mass,
            "route_sample_best_remote_probability": self.best_remote_probability,
            "route_sample_number_of_legal_remote_destinations": self.number_of_legal_remote_destinations,
            "route_sample_selected_route_action_index": self.selected_route_action_index,
            "route_sample_selected_destination_uav": self.selected_destination_uav,
            "route_sample_old_route_log_prob": self.old_route_log_prob,
            "route_sample_new_route_log_prob": self.new_route_log_prob,
            "route_sample_log_ratio": self.route_log_ratio,
            "route_sample_ratio": self.route_ratio,
            "route_sample_approx_kl": self.route_approx_kl,
            "route_sample_clip_indicator": self.route_clip_indicator,
            "route_sample_advantage": self.advantage,
            "route_sample_return_target": self.return_target,
            "route_sample_td_residual": self.td_residual,
            "route_sample_base_route_advantage": self.base_route_advantage,
            "route_sample_new_route_advantage": self.new_route_advantage,
            "route_sample_advantage_delta": self.route_advantage_delta,
            "route_sample_effective_route_gae_lambda": self.effective_route_gae_lambda,
            "route_sample_credit_trace_length_slots": self.credit_trace_length_slots,
            "route_sample_trace_end_kind": self.trace_end_kind,
            "route_sample_bootstrap_source": self.bootstrap_source,
            "route_sample_task_id": self.task_id,
            "route_sample_branch_event_key": (
                None
                if self.branch_event_key is None
                else list(self.branch_event_key)
            ),
            "route_sample_branch_credit_event_key": (
                None
                if self.branch_event_key is None
                else list(self.branch_event_key)
            ),
            "route_sample_formal_route_event_key": (
                None
                if self.formal_route_event_key is None
                else list(self.formal_route_event_key)
            ),
            "route_sample_nstep_route_advantage": (
                self.nstep_route_advantage
            ),
            "route_sample_nstep_horizon": self.nstep_horizon,
            "route_sample_nstep_stop_kind": self.nstep_stop_kind,
            "route_sample_stop_kind": self.nstep_stop_kind,
            "route_sample_terminal_before_rollout": (
                self.terminal_before_rollout
            ),
            "route_sample_bootstrap_used": self.bootstrap_used,
        }


def _branch_ppo_group(
    old: Tensor,
    new: Tensor,
    active: Tensor,
    advantage: Tensor,
    epsilon_clip: float,
) -> BranchPPODynamicsGroup:
    selected_old = old.masked_select(active)
    selected_new = new.masked_select(active)
    selected_advantage = advantage.masked_select(active)
    if selected_old.numel() == 0:
        return BranchPPODynamicsGroup()
    log_ratio = selected_new - selected_old
    ratio = torch.exp(log_ratio)
    if not torch.isfinite(ratio).all():
        raise RouteTelemetryError("branch PPO ratio contains NaN or Inf")
    clipped_ratio = torch.clamp(
        ratio,
        1.0 - epsilon_clip,
        1.0 + epsilon_clip,
    )
    surrogate = torch.minimum(
        ratio * selected_advantage,
        clipped_ratio * selected_advantage,
    )
    clipped = ratio != clipped_ratio
    ratio_values = ratio.detach().cpu().tolist()
    return BranchPPODynamicsGroup(
        active_count=int(ratio.numel()),
        ratio_mean=float(ratio.mean().cpu().item()),
        ratio_median=float(median(ratio_values)),
        ratio_std=float(ratio.std(unbiased=False).cpu().item()),
        approx_kl_mean=float(
            ((ratio - 1.0) - log_ratio).mean().cpu().item()
        ),
        clip_fraction=float(clipped.float().mean().cpu().item()),
        surrogate_contribution_mean=float(surrogate.mean().cpu().item()),
        advantage_mean=float(selected_advantage.mean().cpu().item()),
    )


def compute_branch_ppo_dynamics(
    old_branch_log_probs: Tensor | None,
    new_branch_log_probs: Tensor | None,
    active_branch_indicators: Tensor,
    expanded_advantage: Tensor,
    epsilon_clip: float,
) -> BranchPPODynamics | None:
    """Summarize each branch from detached tensors already used by PPO."""

    if old_branch_log_probs is None or new_branch_log_probs is None:
        return None
    if (
        not isinstance(active_branch_indicators, Tensor)
        or active_branch_indicators.dtype != torch.bool
        or active_branch_indicators.ndim != 4
        or active_branch_indicators.shape[-1] != len(ACTION_BRANCH_ORDER)
    ):
        raise RouteTelemetryError(
            "active_branch_indicators must be boolean [B,T,A,7]"
        )
    expected_shape = active_branch_indicators.shape
    for name, tensor in (
        ("old_branch_log_probs", old_branch_log_probs),
        ("new_branch_log_probs", new_branch_log_probs),
        ("expanded_advantage", expanded_advantage),
    ):
        if not isinstance(tensor, Tensor) or tensor.shape != expected_shape:
            raise RouteTelemetryError(f"{name} must have shape [B,T,A,7]")
    new = new_branch_log_probs.detach().float()
    old = old_branch_log_probs.detach().to(device=new.device, dtype=new.dtype)
    advantage = expanded_advantage.detach().to(
        device=new.device,
        dtype=new.dtype,
    )
    active = active_branch_indicators.detach().to(device=new.device)
    finite = torch.isfinite(old) & torch.isfinite(new) & torch.isfinite(advantage)
    active = active & finite
    groups = {
        branch: _branch_ppo_group(
            old[..., index],
            new[..., index],
            active[..., index],
            advantage[..., index],
            epsilon_clip,
        )
        for index, branch in enumerate(ACTION_BRANCH_ORDER)
    }
    return BranchPPODynamics(epsilon_clip=epsilon_clip, groups=groups)


def _ppo_group(old: Tensor, new: Tensor, mask: Tensor, epsilon_clip: float) -> RoutePPODynamicsGroup:
    selected_old = old.masked_select(mask)
    selected_new = new.masked_select(mask)
    if selected_old.numel() == 0:
        return RoutePPODynamicsGroup()
    log_ratio = selected_new - selected_old
    ratio = torch.exp(log_ratio)
    approx_kl = (ratio - 1.0) - log_ratio
    clipped = (ratio < 1.0 - epsilon_clip) | (ratio > 1.0 + epsilon_clip)
    return RoutePPODynamicsGroup(
        count=int(ratio.numel()),
        old_log_prob_mean=float(selected_old.mean().item()),
        new_log_prob_mean=float(selected_new.mean().item()),
        log_ratio_mean=float(log_ratio.mean().item()),
        ratio_mean=float(ratio.mean().item()),
        approx_kl_mean=float(approx_kl.mean().item()),
        clip_indicator_fraction=float(clipped.float().mean().item()),
        clip_fraction=float(clipped.float().mean().item()),
    )


def compute_route_only_ppo_dynamics(
    old_route_log_prob: Tensor | None,
    new_route_log_prob: Tensor | None,
    route_active_mask: Tensor,
    category_masks: Mapping[str, Tensor],
    epsilon_clip: float,
) -> RoutePPODynamics | None:
    if old_route_log_prob is None or new_route_log_prob is None:
        return None
    active = route_active_mask.detach().bool()
    old = old_route_log_prob.detach().float()
    new = new_route_log_prob.detach().float()
    finite = torch.isfinite(old) & torch.isfinite(new)
    active = active & finite
    groups = {
        category: _ppo_group(old, new, active & category_masks[category].detach().bool(), epsilon_clip)
        for category in ROUTE_ACTION_CATEGORIES
    }
    return RoutePPODynamics(
        epsilon_clip=epsilon_clip,
        overall=_ppo_group(old, new, active, epsilon_clip),
        groups=groups,
    )


def measure_shared_trunk_gradients(actor_or_parameters: Any) -> GradientMeasurement:
    if hasattr(actor_or_parameters, "named_parameters"):
        parameters = [
            parameter
            for name, parameter in actor_or_parameters.named_parameters()
            if name.startswith(("self_encoder", "public_encoder", "graph_encoder", "gru"))
        ]
    else:
        parameters = list(actor_or_parameters or ())
    gradients = [
        parameter.grad.detach().reshape(-1)
        for parameter in parameters
        if getattr(parameter, "grad", None) is not None
    ]
    if not gradients:
        return GradientMeasurement("none", None)
    joined = torch.cat(gradients)
    if not torch.isfinite(joined).all():
        return GradientMeasurement("nonfinite", None)
    norm = float(torch.linalg.vector_norm(joined).cpu().item())
    return GradientMeasurement(
        "zero" if norm == 0.0 else "finite_nonzero",
        norm,
        mean=float(joined.mean().cpu().item()),
        signed_sum=float(joined.sum().cpu().item()),
    )


@dataclass(frozen=True)
class RouteOutcomeAssociation:
    episode_index: int
    task_id: object
    source_uav: object
    category: str
    selected_destination_uav: object
    route_slot: int | None
    tx_start_slot: int | None
    tx_complete_slot: int | None
    cpu_service_start_slot: int | None
    outcome: str
    completion_slot: int | None
    end_to_end_latency_slots: int | None
    end_to_end_latency_s: float | None
    immediate_reward: float | None = None
    workload_penalty_component: float | None = None
    completion_component: float | None = None
    expiration_component: float | None = None
    energy_component: float | None = None
    reward_component_scope: str = "team_level"
    route_decision_sequence: int = 1
    route_decision_slot: int | None = None
    selected_route_action_index: int | None = None
    terminal_slot: int | None = None
    terminal_age_slots: int | None = None

    def record(self) -> dict[str, object]:
        return {
            f"route_outcome_{item.name}": getattr(self, item.name)
            for item in fields(self)
        }


def _route_items(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _route_category(value: object) -> str:
    if type(value) is str and value == "local":
        return "local"
    if isinstance(value, int) and not isinstance(value, bool):
        return "remote"
    return "defer"


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def route_diagnostic_schema_header(config: RunConfig) -> dict[str, Any]:
    """Return the static, versioned contract for the diagnostic sidecar."""

    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    environment = config.environment
    action = config.action
    return {
        "record_type": "schema",
        "schema_version": ROUTE_DIAGNOSTIC_SCHEMA_VERSION,
        "sidecar": "route_diagnostic_samples_v1",
        "utility_label": ROUTE_DIAGNOSTIC_UTILITY_LABEL,
        "run_id": config.run_id,
        "config_hash": config.config_hash,
        "git_commit": config.git_commit,
        "capture_source": "ActorObservation_and_existing_action_output",
        "capture_timing": "after_policy_route_decision_before_environment_step",
        "route_domain_fixed_actions": list(action.route_fixed_actions),
        "diagnostic_contract": {
            "slot_duration_s": environment.slot_duration_s,
            "ru_ids": list(range(1, environment.ru_count + 1)),
            "ru_bandwidth_hz": environment.ru_bandwidth_hz,
            "ru_count": environment.ru_count,
            "uav_count": environment.uav_count,
            "resource_group_count": environment.resource_group_count,
            "resource_groups": [list(group) for group in environment.resource_groups],
            "resource_width_options": list(action.resource_width_options),
            "power_levels": [float(level) for level in action.power_levels],
            "reference_transmit_power_w": environment.reference_transmit_power_w,
            "outage_threshold_linear": environment.outage_threshold_linear,
            "energy_tolerance_j": environment.energy_tolerance_j,
            "reference_cpu_frequency_hz": environment.reference_cpu_frequency_hz,
            "reference_cpu_coefficient": environment.reference_cpu_coefficient,
        },
        "ordering": {
            "agents": "ascending zero-based uav_id",
            "candidates": "route_domain order excluding fixed string actions",
            "rus": "ascending one-based contiguous RU IDs",
            "json": "sorted object keys and one decision per line",
        },
        "unavailable_fields": {
            "task.arrival_slot": "not_exposed_actor_safe",
            "task.arrival_slot_exact": "not_exposed_actor_safe",
            "current_true_channel": "forbidden",
            "current_true_sinr": "forbidden",
            "future_outcome": "joined_after_capture",
        },
    }


def _diagnostic_masked_values(
    values: Sequence[Any],
    valid: Sequence[Any],
    *,
    cast: Callable[[Any], Any] = float,
) -> list[Any]:
    return [cast(value) if bool(is_valid) else None for value, is_valid in zip(values, valid)]


def _diagnostic_tensor_rows(output: Any, field_name: str, agent_count: int) -> list[Any]:
    values = getattr(output, field_name, None)
    if not isinstance(values, Mapping) or "route" not in values:
        raise RouteTelemetryError(f"route diagnostic output lacks {field_name}['route']")
    tensor = values["route"]
    if not isinstance(tensor, Tensor) or tensor.ndim != 4:
        raise RouteTelemetryError(f"route diagnostic {field_name} must be [B,T,A,D]")
    if tuple(tensor.shape[:3]) != (1, 1, agent_count):
        raise RouteTelemetryError(
            f"route diagnostic {field_name} shape does not match one environment step"
        )
    return tensor.detach().cpu().tolist()[0][0]


def _diagnostic_agent_tensor(output: Any, field_name: str, agent_count: int) -> list[Any]:
    values = getattr(output, field_name, None)
    if not isinstance(values, Mapping) or "route" not in values:
        raise RouteTelemetryError(f"route diagnostic output lacks {field_name}['route']")
    tensor = values["route"]
    if not isinstance(tensor, Tensor) or tensor.ndim != 3:
        raise RouteTelemetryError(f"route diagnostic {field_name} must be [B,T,A]")
    if tuple(tensor.shape) != (1, 1, agent_count):
        raise RouteTelemetryError(
            f"route diagnostic {field_name} shape does not match one environment step"
        )
    return tensor.detach().cpu().tolist()[0][0]


@dataclass
class _RouteDecisionState:
    episode_index: int
    task_id: object
    decision_sequence: int
    source_uav: object = None
    category: str = "defer"
    destination: object = None
    route_slot: int | None = None
    route_decision_slot: int | None = None
    selected_route_action_index: int | None = None
    tx_start_slot: int | None = None
    tx_complete_slot: int | None = None
    cpu_service_start_slot: int | None = None
    completion_slot: int | None = None
    terminal_slot: int | None = None
    terminal_age_slots: int | None = None
    outcome: str | None = None
    arrival_slot: int | None = None
    tx_bits: float = 0.0
    reward: dict[str, float | None] = field(default_factory=dict)


class RouteOutcomeTracker:
    """Join route decisions to later service/terminal snapshots without changing them."""

    def __init__(self, slot_duration_s: float | None = None) -> None:
        self.slot_duration_s = slot_duration_s
        self._states: dict[tuple[int, object, int], _RouteDecisionState] = {}
        self._decision_keys: list[tuple[int, object, int]] = []
        self._task_decision_keys: dict[tuple[int, object], list[tuple[int, object, int]]] = {}

    @staticmethod
    def _reward_terms(info: Mapping[str, Any]) -> dict[str, float | None]:
        raw = info.get("reward", {})
        if not isinstance(raw, Mapping):
            return {}
        return {
            "immediate_reward": _safe_float(raw.get("reward")),
            "workload_penalty_component": _safe_float(raw.get(
                "workload_penalty", raw.get("workload_penalty_component")
            )),
            "completion_component": _safe_float(raw.get("completion_component")),
            "expiration_component": _safe_float(raw.get(
                "expiration_penalty", raw.get("expiration_component")
            )),
            "energy_component": _safe_float(raw.get(
                "energy_penalty", raw.get("energy_component")
            )),
        }

    def _task_states(self, task_key: tuple[int, object]) -> list[_RouteDecisionState]:
        return [self._states[key] for key in self._task_decision_keys.get(task_key, ())]

    @staticmethod
    def _optional_int(value: object) -> int | None:
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    @staticmethod
    def _normalized_outcome(value: object, default: str) -> str:
        raw = str(value if value is not None else default)
        if raw in {"done", "completed"}:
            return "completed"
        if raw in {"expired", "truncated"}:
            return raw
        return raw

    @staticmethod
    def _set_terminal(
        state: _RouteDecisionState,
        outcome: str,
        item: Mapping[str, Any],
        observed_slot: int | None,
    ) -> None:
        state.outcome = outcome
        state.arrival_slot = item.get("arrival_slot", state.arrival_slot)
        if outcome in {"expired", "truncated"}:
            state.completion_slot = None
            state.terminal_slot = observed_slot
        else:
            completion_slot = item.get("completion_slot", observed_slot)
            state.completion_slot = completion_slot
            state.terminal_slot = completion_slot
        if state.arrival_slot is not None and state.terminal_slot is not None:
            state.terminal_age_slots = int(state.terminal_slot) - int(state.arrival_slot)

    def observe_step(
        self,
        episode_index: int,
        info: Mapping[str, Any],
        *,
        route_action_indices: Mapping[int, int] | None = None,
    ) -> None:
        if not isinstance(info, Mapping):
            return
        try:
            slot = int(info.get("slot", info.get("slot_index")))
        except (TypeError, ValueError):
            slot = None
        reward = self._reward_terms(info)
        service = info.get("service", {})
        if not isinstance(service, Mapping):
            service = {}
        for route in _route_items(service.get("routing", service.get("routing_records"))):
            task_id = route.get("task_id")
            if task_id is None:
                continue
            proposal_record = route.get("proposal")
            if isinstance(proposal_record, Mapping):
                proposed_route = proposal_record.get(
                    "destination", proposal_record.get("route")
                )
            else:
                proposed_route = proposal_record
            destination = route.get("destination", proposed_route)
            if (
                (type(proposed_route) is str and proposed_route in {"local", "defer", "idle"})
                or (isinstance(proposed_route, int) and not isinstance(proposed_route, bool))
            ):
                category_value = proposed_route
            else:
                category_value = destination
            task_key = (episode_index, task_id)
            decision_keys = self._task_decision_keys.setdefault(task_key, [])
            decision_sequence = len(decision_keys) + 1
            key = (episode_index, task_id, decision_sequence)
            state = _RouteDecisionState(
                episode_index=episode_index,
                task_id=task_id,
                decision_sequence=decision_sequence,
                source_uav=route.get("uav_id", route.get("source_uav")),
                category=_route_category(category_value),
                destination=destination if _route_category(category_value) == "remote" else None,
                route_slot=slot,
                route_decision_slot=slot,
                selected_route_action_index=self._optional_int(route.get(
                    "selected_route_action_index",
                    route.get(
                        "selected_action_index",
                        route.get("action_index", (route_action_indices or {}).get(
                            route.get("uav_id", route.get("source_uav"))
                        )),
                    ),
                )),
                reward=dict(reward),
            )
            self._states[key] = state
            decision_keys.append(key)
            self._decision_keys.append(key)
        raw_task_services = service.get("task_services")
        if raw_task_services is None:
            raw_task_services = service.get("links")
        task_service_items: list[Mapping[str, Any]] = []
        for link_or_item in _route_items(raw_task_services):
            nested = link_or_item.get("task_services")
            task_service_items.extend(
                _route_items(nested) if nested is not None else [link_or_item]
            )
        for item in task_service_items:
            task_key = (episode_index, item.get("task_id"))
            states = self._task_states(task_key)
            if not states:
                continue
            amount = _safe_float(item.get(
                "amount", item.get("served_bits", item.get("bits"))
            )) or 0.0
            data_bits = _safe_float(item.get("data_bits", item.get("required_bits")))
            for state in states:
                if state.tx_start_slot is None:
                    state.tx_start_slot = slot
                state.tx_bits += amount
                if data_bits is not None and state.tx_bits >= data_bits:
                    state.tx_complete_slot = slot
        for item in _route_items(service.get("cpu_services", service.get("cpu"))):
            states = self._task_states((episode_index, item.get("task_id")))
            for state in states:
                if state.cpu_service_start_slot is None:
                    state.cpu_service_start_slot = slot
        for section, default_outcome in (
            ("settled_tasks", "completed"), ("truncated_tasks", "truncated")
        ):
            for item in _route_items(service.get(section)):
                states = self._task_states((episode_index, item.get("task_id")))
                if not states:
                    continue
                outcome = self._normalized_outcome(item.get("outcome"), default_outcome)
                for state in states:
                    self._set_terminal(state, outcome, item, slot)
                    if state.category == "remote" and state.tx_complete_slot is None:
                        cpu_entry_slot = item.get("cpu_entry_slot")
                        if isinstance(cpu_entry_slot, int) and not isinstance(cpu_entry_slot, bool):
                            state.tx_complete_slot = cpu_entry_slot
        metrics = info.get("metrics", {})
        if isinstance(metrics, Mapping):
            for item in _route_items(metrics.get("expired_tasks")):
                states = self._task_states((episode_index, item.get("task_id")))
                for state in states:
                    self._set_terminal(state, "expired", item, slot)

    def finalize(self) -> tuple[RouteOutcomeAssociation, ...]:
        result = []
        for key in sorted(self._decision_keys, key=lambda item: (item[0], str(item[1]), item[2])):
            state = self._states[key]
            outcome = state.outcome or "truncated"
            latency_slots = None
            if outcome == "completed" and state.arrival_slot is not None and state.completion_slot is not None:
                latency_slots = int(state.completion_slot) - int(state.arrival_slot)
            result.append(RouteOutcomeAssociation(
                episode_index=state.episode_index,
                task_id=state.task_id,
                source_uav=state.source_uav,
                category=state.category,
                selected_destination_uav=(
                    state.destination if state.category == "remote"
                    else state.source_uav if state.category == "local" else None
                ),
                route_slot=state.route_slot,
                tx_start_slot=state.tx_start_slot,
                tx_complete_slot=state.tx_complete_slot,
                cpu_service_start_slot=state.cpu_service_start_slot,
                outcome=outcome,
                completion_slot=state.completion_slot,
                end_to_end_latency_slots=latency_slots,
                end_to_end_latency_s=(
                    latency_slots * self.slot_duration_s
                    if latency_slots is not None and self.slot_duration_s is not None
                    else None
                ),
                immediate_reward=state.reward.get("immediate_reward"),
                workload_penalty_component=state.reward.get("workload_penalty_component"),
                completion_component=state.reward.get("completion_component"),
                expiration_component=state.reward.get("expiration_component"),
                energy_component=state.reward.get("energy_component"),
                reward_component_scope="team_level",
                route_decision_sequence=state.decision_sequence,
                route_decision_slot=state.route_decision_slot,
                selected_route_action_index=state.selected_route_action_index,
                terminal_slot=state.terminal_slot,
                terminal_age_slots=state.terminal_age_slots,
            ))
        return tuple(result)


class RouteDiagnosticSampleCollector:
    """Capture actor-safe route decisions without computing utility in training."""

    def __init__(self, config: RunConfig) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        self.config = config

    def schema_header(self) -> dict[str, Any]:
        return route_diagnostic_schema_header(self.config)

    @staticmethod
    def _source_queue(observation: Any) -> dict[str, Any]:
        queue = observation.private_queues.unbound
        if not bool(queue.head_valid_mask):
            raise RouteTelemetryError(
                "route-active observation has no actor-visible unbound head task"
            )
        slot = int(observation.slot)
        slack = int(queue.head_slack_slots)
        return {
            "task_id": int(queue.head_task_id),
            "remaining_bits": float(queue.head_remaining_bits),
            "remaining_cycles": float(queue.head_remaining_cycles),
            "arrival_slot": None,
            "arrival_slot_valid": False,
            "arrival_slot_source": "not_exposed_actor_safe",
            "deadline_slot": slot + slack - 1,
            "deadline_slot_valid": True,
            "deadline_slot_source": "ActorObservation.private_queues.unbound.head_slack_slots",
            "remaining_deadline_slots": slack,
            "remaining_deadline_slots_valid": True,
            "lifecycle": "unbound",
            "lifecycle_source": "ActorObservation.action_masks.route_branch_active",
            "head_valid": True,
            "provenance": {
                "task_id": "ActorObservation.private_queues.unbound.head_task_id",
                "remaining_bits": "ActorObservation.private_queues.unbound.head_remaining_bits",
                "remaining_cycles": "ActorObservation.private_queues.unbound.head_remaining_cycles",
                "arrival_slot": "not_exposed_actor_safe",
                "deadline_slot": "ActorObservation.private_queues.unbound.head_slack_slots",
                "remaining_deadline_slots": "ActorObservation.private_queues.unbound.head_slack_slots",
            },
        }

    @staticmethod
    def _source_primitives(observation: Any) -> dict[str, Any]:
        resources = observation.self_resources
        local = observation.private_queues.local_cpu
        tx = observation.private_queues.tx_by_destination
        destinations = []
        for destination in range(tx.count):
            destinations.append({
                "destination_uav": destination,
                "destination_is_self": destination == int(observation.uav_id),
                "task_count": int(tx.task_count[destination]),
                "remaining_bits": float(tx.remaining_bits[destination]),
                "remaining_cycles": float(tx.remaining_cycles[destination]),
                "provenance": {
                    "task_count": "ActorObservation.private_queues.tx_by_destination.task_count",
                    "remaining_bits": "ActorObservation.private_queues.tx_by_destination.remaining_bits",
                    "remaining_cycles": "ActorObservation.private_queues.tx_by_destination.remaining_cycles",
                },
            })
        return {
            "residual_energy_j": float(resources.residual_energy_j),
            "max_transmit_power_w": float(resources.max_transmit_power_w),
            "max_cpu_frequency_hz": float(resources.max_cpu_frequency_hz),
            "cpu_coefficient": float(resources.cpu_coefficient),
            "local_cpu_backlog": {
                "task_count": int(local.task_count),
                "remaining_cycles": float(local.remaining_cycles),
                "provenance": {
                    "task_count": "ActorObservation.private_queues.local_cpu.task_count",
                    "remaining_cycles": "ActorObservation.private_queues.local_cpu.remaining_cycles",
                },
            },
            "tx_queue_by_destination": destinations,
            "provenance": {
                "residual_energy_j": "ActorObservation.self_resources.residual_energy_j",
                "max_transmit_power_w": "ActorObservation.self_resources.max_transmit_power_w",
                "max_cpu_frequency_hz": "ActorObservation.self_resources.max_cpu_frequency_hz",
                "cpu_coefficient": "ActorObservation.self_resources.cpu_coefficient",
            },
        }

    def _candidate(
        self,
        observation: Any,
        destination: int,
        action_index: int,
        legal: bool,
    ) -> dict[str, Any]:
        public = observation.neighbor_public
        edge = observation.edge_history
        helper_valid = bool(public.valid_mask[destination])
        helper = {
            "valid": helper_valid,
            "source": "ActorObservation.neighbor_public" if helper_valid else "not_exposed_actor_safe",
            "residual_energy_j": float(public.residual_energy_j[destination]) if helper_valid else None,
            "max_cpu_frequency_hz": (
                float(public.max_cpu_frequency_ratio[destination])
                * self.config.environment.reference_cpu_frequency_hz
                if helper_valid else None
            ),
            "cpu_coefficient": (
                float(public.cpu_coefficient_ratio[destination])
                * self.config.environment.reference_cpu_coefficient
                if helper_valid else None
            ),
            "cpu_load_task_count": int(public.cpu_load_task_count[destination]) if helper_valid else None,
            "cpu_load_remaining_cycles": (
                float(public.cpu_load_remaining_cycles[destination]) if helper_valid else None
            ),
            "message_aoi_slots": (
                int(public.message_aoi_slots[destination])
                if bool(public.message_aoi_valid_mask[destination]) else None
            ),
            "message_aoi_valid": bool(public.message_aoi_valid_mask[destination]),
            "provenance": {
                "valid": "ActorObservation.neighbor_public.valid_mask",
                "residual_energy_j": "ActorObservation.neighbor_public.residual_energy_j",
                "max_cpu_frequency_hz": "ActorObservation.neighbor_public.max_cpu_frequency_ratio * static reference_cpu_frequency_hz",
                "cpu_coefficient": "ActorObservation.neighbor_public.cpu_coefficient_ratio * static reference_cpu_coefficient",
                "cpu_load_task_count": "ActorObservation.neighbor_public.cpu_load_task_count",
                "cpu_load_remaining_cycles": "ActorObservation.neighbor_public.cpu_load_remaining_cycles",
                "message_aoi_slots": "ActorObservation.neighbor_public.message_aoi_slots",
            },
        }
        quality_valid = np.asarray(edge.quality_valid_mask[destination], dtype=np.bool_)
        interference_valid = np.asarray(edge.interference_valid_mask[destination], dtype=np.bool_)
        visible = bool(edge.visible_mask[destination])
        csi_valid = bool(edge.csi_valid_mask[destination])
        link = {
            "source": "ActorObservation.edge_history",
            "visible": visible,
            "estimated_distance_m": float(edge.estimated_distance_m[destination]) if visible else None,
            "csi_valid": csi_valid,
            "csi_aoi_slots": int(edge.csi_aoi_slots[destination]) if csi_valid else None,
            "historical_quality_by_ru": _diagnostic_masked_values(
                edge.historical_quality[destination], quality_valid
            ),
            "historical_quality_valid_mask": quality_valid.tolist(),
            "historical_interference_w_by_ru": _diagnostic_masked_values(
                edge.interference_history_w[destination], interference_valid
            ),
            "historical_interference_valid_mask": interference_valid.tolist(),
            "message_aoi_slots": (
                int(edge.message_aoi_slots[destination])
                if bool(edge.message_aoi_valid_mask[destination]) else None
            ),
            "message_aoi_valid": bool(edge.message_aoi_valid_mask[destination]),
            "last_effective_rate_bps": (
                float(edge.last_effective_rate_bps[destination])
                if bool(edge.last_rate_valid_mask[destination]) else None
            ),
            "last_rate_valid": bool(edge.last_rate_valid_mask[destination]),
            "outage_rate": (
                float(edge.outage_rate[destination])
                if bool(edge.outage_valid_mask[destination]) else None
            ),
            "outage_valid": bool(edge.outage_valid_mask[destination]),
            "ru_ids": list(range(1, self.config.environment.ru_count + 1)),
            "provenance": {
                "visible": "ActorObservation.edge_history.visible_mask",
                "estimated_distance_m": "ActorObservation.edge_history.estimated_distance_m",
                "csi_valid": "ActorObservation.edge_history.csi_valid_mask",
                "csi_aoi_slots": "ActorObservation.edge_history.csi_aoi_slots",
                "historical_quality_by_ru": "ActorObservation.edge_history.historical_quality",
                "historical_quality_valid_mask": "ActorObservation.edge_history.quality_valid_mask",
                "historical_interference_w_by_ru": "ActorObservation.edge_history.interference_history_w",
                "historical_interference_valid_mask": "ActorObservation.edge_history.interference_valid_mask",
                "message_aoi_slots": "ActorObservation.edge_history.message_aoi_slots",
                "last_effective_rate_bps": "ActorObservation.edge_history.last_effective_rate_bps",
                "outage_rate": "ActorObservation.edge_history.outage_rate",
            },
        }
        tx = observation.private_queues.tx_by_destination
        return {
            "candidate_uav": destination,
            "route_action_index": action_index,
            "legal_mask": bool(legal),
            "topology_available": bool(observation.candidate_neighbor_mask[destination]),
            "self_excluded": destination == int(observation.uav_id),
            "helper_public": helper,
            "link_history": link,
            "source_tx_queue": {
                "task_count": int(tx.task_count[destination]),
                "remaining_bits": float(tx.remaining_bits[destination]),
                "remaining_cycles": float(tx.remaining_cycles[destination]),
                "provenance": "ActorObservation.private_queues.tx_by_destination",
            },
        }

    def collect(
        self,
        *,
        episode_id: int,
        global_environment_step: int,
        policy_version: int,
        observations: Sequence[Any],
        proposals: Sequence[ActionProposal],
        action_output: SequentialActionDistributionOutput,
    ) -> tuple[dict[str, Any], ...]:
        """Capture one pre-step sample for every route-active source UAV."""

        if not observations or any(not hasattr(item, "action_masks") for item in observations):
            raise RouteTelemetryError("route diagnostic capture requires ActorObservation values")
        if any(int(item.slot) != int(observations[0].slot) for item in observations):
            raise RouteTelemetryError("route diagnostic observations must share one slot")
        if len(proposals) != len(observations):
            raise RouteTelemetryError("route diagnostic proposals are misaligned")
        if any(proposal.uav_id != index for index, proposal in enumerate(proposals)):
            raise RouteTelemetryError("route diagnostic proposals must use stable UAV order")
        for name, value in (
            ("episode_id", episode_id),
            ("global_environment_step", global_environment_step),
            ("policy_version", policy_version),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RouteTelemetryError(f"{name} must be a non-negative integer")

        agent_count = len(observations)
        route_logits = _diagnostic_tensor_rows(action_output, "raw_logits", agent_count)
        route_masks = _diagnostic_tensor_rows(action_output, "action_masks", agent_count)
        route_probabilities = _diagnostic_tensor_rows(action_output, "probabilities", agent_count)
        route_indices = _diagnostic_agent_tensor(action_output, "action_indices", agent_count)
        route_active = _diagnostic_agent_tensor(action_output, "active_branches", agent_count)
        route_entropy = _diagnostic_agent_tensor(action_output, "branch_entropies", agent_count)
        records: list[dict[str, Any]] = []
        for agent_index, (observation, proposal) in enumerate(zip(observations, proposals)):
            expected_mask = [bool(item) for item in observation.action_masks.route_mask]
            output_mask = [bool(item) for item in route_masks[agent_index]]
            if expected_mask != output_mask:
                raise RouteTelemetryError("route diagnostic action mask differs from ActorObservation")
            if bool(route_active[agent_index]) != bool(observation.action_masks.route_branch_active):
                raise RouteTelemetryError("route diagnostic route-active state differs from ActorObservation")
            if not bool(observation.action_masks.route_branch_active):
                continue
            task = self._source_queue(observation)
            domain = tuple(observation.action_masks.route_domain)
            candidates = [
                self._candidate(observation, int(value), action_index, bool(legal))
                for action_index, (value, legal) in enumerate(zip(domain, expected_mask))
                if isinstance(value, int) and not isinstance(value, bool)
            ]
            route_entries = []
            for action_index, (value, legal, probability, logit) in enumerate(
                zip(domain, expected_mask, route_probabilities[agent_index], route_logits[agent_index])
            ):
                probability_value = float(probability)
                raw_logit = float(logit)
                route_entries.append({
                    "action": value,
                    "action_index": action_index,
                    "legal": bool(legal),
                    "probability": probability_value,
                    "raw_logit": raw_logit if math.isfinite(raw_logit) else None,
                })
            route_value = proposal.route
            record = {
                "record_type": "route_diagnostic_sample",
                "schema_version": ROUTE_DIAGNOSTIC_SCHEMA_VERSION,
                "utility_label": ROUTE_DIAGNOSTIC_UTILITY_LABEL,
                "run_id": self.config.run_id,
                "config_hash": self.config.config_hash,
                "git_commit": self.config.git_commit,
                "episode_id": int(episode_id),
                "global_environment_step": int(global_environment_step),
                "route_slot": int(observation.slot),
                "policy_version": int(policy_version),
                "source_uav": int(observation.uav_id),
                "head_task_id": int(task["task_id"]),
                "primary_key": [
                    self.config.run_id,
                    int(episode_id),
                    int(global_environment_step),
                    int(observation.uav_id),
                    int(task["task_id"]),
                ],
                "task": task,
                "source": self._source_primitives(observation),
                "candidates": candidates,
                "policy": {
                    "actual_route": route_value,
                    "chosen_remote": (
                        int(route_value)
                        if isinstance(route_value, int) and not isinstance(route_value, bool)
                        else None
                    ),
                    "route_action_index": int(route_indices[agent_index]),
                    "route_entries": route_entries,
                    "local_probability": next(
                        item["probability"] for item in route_entries if item["action"] == "local"
                    ),
                    "defer_probability": next(
                        item["probability"] for item in route_entries if item["action"] == "defer"
                    ),
                    "remote_probability_by_candidate": [
                        {
                            "candidate_uav": item["candidate_uav"],
                            "route_action_index": item["route_action_index"],
                            "legal": item["legal_mask"],
                            "probability": next(
                                entry["probability"]
                                for entry in route_entries
                                if entry["action_index"] == item["route_action_index"]
                            ),
                        }
                        for item in candidates
                    ],
                    "route_entropy": float(route_entropy[agent_index]),
                    "provenance": {
                        "actual_route": "existing_action_output.proposals",
                        "chosen_remote": "existing_action_output.proposals",
                        "route_action_index": "existing_action_output.action_indices['route']",
                        "route_entries.probability": "existing_action_output.probabilities['route']",
                        "route_entries.raw_logit": "existing_action_output.raw_logits['route']",
                        "route_entropy": "existing_action_output.branch_entropies['route']",
                    },
                },
                "join": {
                    "route_decision_occurrence": None,
                    "route_outcome_sequence_source": "derived_offline_from_route_outcome",
                    "outcome_status": "not_joined_at_capture",
                },
            }
            records.append(record)
        return tuple(records)


def _diagnostic_contract_value(contract: Mapping[str, Any], name: str) -> Any:
    try:
        return contract["diagnostic_contract"][name]
    except (KeyError, TypeError) as exc:
        raise RouteTelemetryError(f"diagnostic contract is missing {name!r}") from exc


def reconstruct_route_diagnostic_utility(
    sample: Mapping[str, Any],
    header: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure offline reconstruction from actor-safe sidecar primitives."""

    if not isinstance(sample, Mapping) or sample.get("record_type") != "route_diagnostic_sample":
        raise RouteTelemetryError("utility reconstruction requires a route diagnostic sample")
    if not isinstance(header, Mapping) or header.get("record_type") != "schema":
        raise RouteTelemetryError("utility reconstruction requires the sidecar schema header")
    task = sample.get("task", {})
    source = sample.get("source", {})
    if not isinstance(task, Mapping) or not isinstance(source, Mapping):
        raise RouteTelemetryError("route diagnostic sample has invalid task/source sections")
    contract = header
    slot_duration = float(_diagnostic_contract_value(contract, "slot_duration_s"))
    ru_count = int(_diagnostic_contract_value(contract, "ru_count"))
    ru_bandwidth = float(_diagnostic_contract_value(contract, "ru_bandwidth_hz"))
    reference_power = float(_diagnostic_contract_value(contract, "reference_transmit_power_w"))
    outage_threshold = float(_diagnostic_contract_value(contract, "outage_threshold_linear"))
    tolerance = float(_diagnostic_contract_value(contract, "energy_tolerance_j"))
    groups = [
        tuple(int(item) for item in group)
        for group in _diagnostic_contract_value(contract, "resource_groups")
    ]
    widths = tuple(int(item) for item in _diagnostic_contract_value(contract, "resource_width_options"))
    power_levels = tuple(float(item) for item in _diagnostic_contract_value(contract, "power_levels"))

    def positive(value: Any) -> float | None:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return None
        return converted if math.isfinite(converted) and converted > 0.0 else None

    def nonnegative(value: Any) -> float | None:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return None
        return converted if math.isfinite(converted) and converted >= 0.0 else None

    source_frequency = positive(source.get("max_cpu_frequency_hz"))
    source_kappa = positive(source.get("cpu_coefficient"))
    source_power = positive(source.get("max_transmit_power_w"))
    source_energy = nonnegative(source.get("residual_energy_j"))
    task_cycles = positive(task.get("remaining_cycles"))
    task_bits = positive(task.get("remaining_bits"))
    slack = positive(task.get("remaining_deadline_slots"))
    local_backlog = source.get("local_cpu_backlog", {})
    backlog_cycles = nonnegative(local_backlog.get("remaining_cycles")) if isinstance(local_backlog, Mapping) else None

    local: dict[str, Any] = {"status": "invalid_source_primitives"}
    if source_frequency is not None and task_cycles is not None and backlog_cycles is not None:
        local_time = slot_duration + (backlog_cycles + task_cycles) / source_frequency
        local_energy = (
            source_kappa * source_frequency**2 * task_cycles
            if source_kappa is not None else None
        )
        local = {
            "status": "valid" if local_energy is not None else "energy_unavailable",
            "T_local_est_s": local_time,
            "local_deadline_margin_s": (
                slack * slot_duration - local_time if slack is not None else None
            ),
            "E_local_est_j": local_energy,
            "local_energy_margin_j": (
                source_energy - local_energy
                if local_energy is not None and source_energy is not None else None
            ),
        }

    remote_results: list[dict[str, Any]] = []
    for candidate in sample.get("candidates", ()):
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = candidate.get("candidate_uav")
        result: dict[str, Any] = {
            "candidate_uav": candidate_id,
            "status": (
                "illegal_candidate"
                if not bool(candidate.get("legal_mask"))
                else "invalid_primitives"
            ),
            "primary_combination": None,
        }
        if not bool(candidate.get("legal_mask")):
            remote_results.append(result)
            continue
        helper = candidate.get("helper_public", {})
        link = candidate.get("link_history", {})
        tx_queue = candidate.get("source_tx_queue", {})
        if not isinstance(helper, Mapping) or not isinstance(link, Mapping) or not isinstance(tx_queue, Mapping):
            remote_results.append(result)
            continue
        helper_frequency = positive(helper.get("max_cpu_frequency_hz"))
        helper_kappa = positive(helper.get("cpu_coefficient"))
        helper_energy = nonnegative(helper.get("residual_energy_j"))
        queue_bits = nonnegative(tx_queue.get("remaining_bits"))
        quality = link.get("historical_quality_by_ru", ())
        quality_mask = link.get("historical_quality_valid_mask", ())
        if (
            source_power is None
            or source_energy is None
            or task_bits is None
            or task_cycles is None
            or queue_bits is None
            or len(quality) != ru_count
            or len(quality_mask) != ru_count
        ):
            remote_results.append(result)
            continue
        best: tuple[tuple[float, float, int, int], dict[str, Any]] | None = None
        for group_index in range(1, len(groups) + 1):
            for width in widths:
                if width <= 0 or group_index + width - 1 > len(groups):
                    continue
                rus = tuple(
                    ru
                    for group in groups[group_index - 1:group_index - 1 + width]
                    for ru in group
                )
                if not all(bool(quality_mask[ru - 1]) and quality[ru - 1] is not None for ru in rus):
                    continue
                for power_level in power_levels:
                    if power_level <= 0.0:
                        continue
                    total_power = power_level * source_power
                    if total_power * slot_duration > source_energy + tolerance:
                        continue
                    scale = (total_power / reference_power) * (ru_count / len(rus))
                    rate = 0.0
                    for ru in rus:
                        sinr = float(quality[ru - 1]) * scale
                        if not math.isfinite(sinr) or sinr < 0.0:
                            rate = 0.0
                            break
                        if sinr >= outage_threshold:
                            rate += ru_bandwidth * math.log2(1.0 + sinr)
                    if not math.isfinite(rate) or rate <= 0.0:
                        continue
                    selection = {
                        "resource_group": group_index,
                        "resource_width": width,
                        "power_level": power_level,
                        "ru_ids": list(rus),
                        "primary_rate_proxy_bps": rate,
                    }
                    key = (-rate, power_level, group_index, width)
                    if best is None or key < best[0]:
                        best = (key, selection)
        if best is None:
            result["status"] = "no_valid_rate_or_energy_feasible_combination"
            remote_results.append(result)
            continue
        selection = best[1]
        rate = float(selection["primary_rate_proxy_bps"])
        tx_time = (queue_bits + task_bits) / rate
        helper_wait = (
            float(helper.get("cpu_load_remaining_cycles")) / helper_frequency
            if helper_frequency is not None
            and helper.get("cpu_load_remaining_cycles") is not None
            else None
        )
        cpu_time = task_cycles / helper_frequency if helper_frequency is not None else None
        total_time = (
            slot_duration + tx_time + helper_wait + cpu_time
            if helper_wait is not None and cpu_time is not None else None
        )
        tx_power = float(selection["power_level"]) * source_power
        tx_energy = tx_power * task_bits / rate
        cpu_energy = (
            helper_kappa * helper_frequency**2 * task_cycles
            if helper_kappa is not None and helper_frequency is not None else None
        )
        total_energy = tx_energy + cpu_energy if cpu_energy is not None else None
        result.update({
            "status": "valid" if total_time is not None else "time_component_unavailable",
            "primary_combination": selection,
            "T_tx_est_s": tx_time,
            "T_helper_wait_est_s": helper_wait,
            "T_remote_cpu_est_s": cpu_time,
            "T_remote_total_est_s": total_time,
            "remote_deadline_margin_s": (
                slack * slot_duration - total_time
                if slack is not None and total_time is not None else None
            ),
            "E_tx_est_j": tx_energy,
            "E_remote_cpu_est_j": cpu_energy,
            "E_remote_total_est_j": total_energy,
            "remote_source_tx_energy_margin_j": source_energy - tx_energy,
            "remote_helper_cpu_energy_margin_j": (
                helper_energy - cpu_energy
                if helper_energy is not None and cpu_energy is not None else None
            ),
        })
        if total_time is not None and "T_local_est_s" in local:
            result["relative_delay_margin_s"] = local["T_local_est_s"] - total_time
        if total_energy is not None and local.get("E_local_est_j") is not None:
            result["relative_energy_margin_j"] = local["E_local_est_j"] - total_energy
        remote_results.append(result)
    return {
        "utility_label": ROUTE_DIAGNOSTIC_UTILITY_LABEL,
        "local": local,
        "remote_by_candidate": remote_results,
        "estimator_scope": "offline_pure_function_only",
    }


@dataclass(frozen=True)
class TrajectoryPreStepCapture:
    """Actor-safe route context captured before one environment transition."""

    episode_id: int
    route_step: int
    rollout_index: int
    route_candidates: Mapping[int, Mapping[str, Any]]
    workload_snapshots: Mapping[int, TaskWorkloadSnapshot]

    def __post_init__(self) -> None:
        for name in ("episode_id", "route_step", "rollout_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RouteTelemetryError(f"{name} must be a non-negative integer")


@dataclass
class _TrajectoryTaskLedger:
    episode_id: int
    task_id: int
    source_uav: int
    initial_task_snapshot: dict[str, Any]
    latest_task_snapshot: dict[str, Any]
    route_event_key: tuple[int, int, int] | None = None
    route_event: dict[str, Any] | None = None
    cumulative_workload_raw: float = 0.0
    cumulative_workload_penalty: float = 0.0
    workload_residual_raw: float = 0.0
    workload_residual_penalty: float = 0.0
    completion_component: float = 0.0
    expiration_penalty: float = 0.0
    cumulative_tx_bits: float = 0.0
    cumulative_cpu_cycles: float = 0.0
    allocated_task_tx_energy: float = 0.0
    exact_task_cpu_energy: float = 0.0
    first_tx_step: int | None = None
    last_tx_step: int | None = None
    first_cpu_step: int | None = None
    last_cpu_step: int | None = None
    terminal_emitted: bool = False


class TrajectoryCreditTracker:
    """Fresh-run-only task/route/reward/PPO sidecar with no control influence."""

    def __init__(
        self,
        config: RunConfig,
        emit: Callable[[Mapping[str, Any]], None],
    ) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        if not config.training.mappo.trajectory_credit_telemetry_enabled:
            raise RouteTelemetryError("trajectory credit tracker requires enabled telemetry")
        if not callable(emit):
            raise TypeError("emit must be callable")
        self.config = config
        self._emit = emit
        self._tasks: dict[tuple[int, int], _TrajectoryTaskLedger] = {}
        self._route_events: dict[tuple[int, int, int], dict[str, Any]] = {}
        self._route_by_transition_agent: dict[tuple[int, int], tuple[int, int, int]] = {}
        self._terminal_task_keys: set[tuple[int, int]] = set()
        self._credit_route_keys: set[tuple[int, int, int]] = set()
        self._event_counts = {name: 0 for name in TRAJECTORY_CREDIT_EVENT_TYPES}
        self._team_workload_raw = 0.0
        self._team_workload_penalty = 0.0
        self._team_completion_component = 0.0
        self._team_expiration_penalty = 0.0
        self._workload_numeric_residual_raw = 0.0
        self._workload_numeric_residual_penalty = 0.0
        self._measured_tx_energy = 0.0
        self._allocated_tx_energy = 0.0
        self._unattributed_tx_energy = 0.0
        self._measured_cpu_energy = 0.0
        self._attributed_cpu_energy = 0.0
        self._unattributed_cpu_energy = 0.0
        self._production_measured_energy = 0.0
        self._energy_numeric_residual = 0.0

    def _common(
        self,
        event_type: str,
        episode_id: int,
        task_id: int,
        route_event_key: tuple[int, int, int] | None,
    ) -> dict[str, Any]:
        return {
            "schema_version": TRAJECTORY_CREDIT_SCHEMA_VERSION,
            "event_type": event_type,
            "run_id": self.config.run_id,
            "config_hash": self.config.config_hash,
            "git_commit": self.config.git_commit,
            "episode_id": episode_id,
            "task_id": task_id,
            "route_event_key": None if route_event_key is None else list(route_event_key),
        }

    def _write(self, record: Mapping[str, Any]) -> None:
        event_type = record.get("event_type")
        if event_type not in TRAJECTORY_CREDIT_EVENT_TYPES:
            raise RouteTelemetryError("unknown trajectory-credit event type")
        self._emit(record)
        self._event_counts[str(event_type)] += 1

    def _ledger_for_task(self, episode_id: int, task: Task) -> _TrajectoryTaskLedger:
        key = (episode_id, task.task_id)
        snapshot = task.snapshot()
        ledger = self._tasks.get(key)
        if ledger is None:
            ledger = _TrajectoryTaskLedger(
                episode_id=episode_id,
                task_id=task.task_id,
                source_uav=task.source_uav,
                initial_task_snapshot=dict(snapshot),
                latest_task_snapshot=dict(snapshot),
            )
            self._tasks[key] = ledger
        else:
            ledger.latest_task_snapshot = dict(snapshot)
        return ledger

    @staticmethod
    def _selected_link_proxy(observation: Any, destination: int) -> dict[str, Any]:
        edge = observation.edge_history
        quality_values = edge.historical_quality[destination]
        quality_mask = edge.quality_valid_mask[destination]
        valid_quality = [
            float(value)
            for value, valid in zip(quality_values, quality_mask)
            if bool(valid)
        ]
        return {
            "candidate_neighbor": bool(observation.candidate_neighbor_mask[destination]),
            "visible": bool(edge.visible_mask[destination]),
            "estimated_distance_m": float(edge.estimated_distance_m[destination]),
            "historical_quality_mean": (
                math.fsum(valid_quality) / len(valid_quality) if valid_quality else None
            ),
            "historical_quality_valid_count": len(valid_quality),
            "csi_valid": bool(edge.csi_valid_mask[destination]),
            "csi_aoi_slots": (
                int(edge.csi_aoi_slots[destination])
                if bool(edge.csi_valid_mask[destination]) else None
            ),
            "message_aoi_slots": (
                int(edge.message_aoi_slots[destination])
                if bool(edge.message_aoi_valid_mask[destination]) else None
            ),
            "message_aoi_valid": bool(edge.message_aoi_valid_mask[destination]),
            "last_effective_rate_bps": (
                float(edge.last_effective_rate_bps[destination])
                if bool(edge.last_rate_valid_mask[destination]) else None
            ),
            "last_rate_valid": bool(edge.last_rate_valid_mask[destination]),
            "outage_rate": (
                float(edge.outage_rate[destination])
                if bool(edge.outage_valid_mask[destination]) else None
            ),
            "outage_valid": bool(edge.outage_valid_mask[destination]),
        }

    @staticmethod
    def _helper_queue_proxy(observation: Any, destination: int) -> dict[str, Any]:
        public = observation.neighbor_public
        return {
            "valid": bool(public.valid_mask[destination]),
            "task_count": int(public.cpu_load_task_count[destination]),
            "remaining_cycles": float(public.cpu_load_remaining_cycles[destination]),
            "message_aoi_slots": (
                int(public.message_aoi_slots[destination])
                if bool(public.message_aoi_valid_mask[destination]) else None
            ),
            "message_aoi_valid": bool(public.message_aoi_valid_mask[destination]),
        }

    def capture_pre_step(
        self,
        *,
        episode_id: int,
        rollout_index: int,
        observations: Sequence[Any],
        proposals: Sequence[ActionProposal],
        route_action_indices: Mapping[int, int],
        environment: Any,
    ) -> TrajectoryPreStepCapture:
        if not observations:
            raise RouteTelemetryError("trajectory capture requires observations")
        route_step = int(observations[0].slot)
        lifecycle = getattr(environment, "lifecycle", None)
        tasks = getattr(lifecycle, "tasks", None)
        if not isinstance(tasks, Mapping):
            raise RouteTelemetryError("telemetry requires environment lifecycle tasks")
        for task in tasks.values():
            self._ledger_for_task(episode_id, task)
        proposal_by_uav = {proposal.uav_id: proposal for proposal in proposals}
        candidates: dict[int, Mapping[str, Any]] = {}
        workload_snapshots: dict[int, TaskWorkloadSnapshot] = {}
        for observation in observations:
            proposal = proposal_by_uav.get(observation.uav_id)
            if proposal is None:
                raise RouteTelemetryError("missing proposal for trajectory capture")
            destination = proposal.route
            is_remote = isinstance(destination, int) and not isinstance(destination, bool)
            if destination != "local" and not is_remote:
                continue
            queue = observation.private_queues.unbound
            if not queue.head_valid_mask:
                continue
            task_id = int(queue.head_task_id)
            task = tasks.get(task_id)
            if not isinstance(task, Task):
                raise RouteTelemetryError("route head task is absent from lifecycle")
            route_domain = tuple(observation.action_masks.route_domain)
            route_mask = tuple(bool(value) for value in observation.action_masks.route_mask)
            legal_remote = [
                value
                for value, legal in zip(route_domain, route_mask)
                if legal and isinstance(value, int) and not isinstance(value, bool)
            ]
            selected_destination = int(destination) if is_remote else observation.uav_id
            candidates[observation.uav_id] = {
                "task_id": task_id,
                "source_uav": observation.uav_id,
                "route_category": "remote" if is_remote else "local",
                "selected_destination_uav": selected_destination,
                "route_action_index": int(route_action_indices[observation.uav_id]),
                "task_snapshot": task.snapshot(),
                "source_queue_proxy": queue.snapshot(),
                "legal_remote_destinations": list(legal_remote),
                "legal_route_mask": list(route_mask),
                "selected_link_proxy_status": "observed" if is_remote else "not_applicable_local",
                "selected_link_proxy": (
                    self._selected_link_proxy(observation, selected_destination)
                    if is_remote else None
                ),
                "helper_queue_proxy_status": "observed" if is_remote else "not_applicable_local",
                "helper_queue_proxy": (
                    self._helper_queue_proxy(observation, selected_destination)
                    if is_remote else None
                ),
            }
            workload_snapshots[task_id] = TaskWorkloadSnapshot.from_task(task)
        return TrajectoryPreStepCapture(
            episode_id=episode_id,
            route_step=route_step,
            rollout_index=rollout_index,
            route_candidates=candidates,
            workload_snapshots=workload_snapshots,
        )

    @staticmethod
    def _mapping_items(value: Any) -> tuple[Mapping[str, Any], ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return ()
        return tuple(item for item in value if isinstance(item, Mapping))

    def _observe_routes(
        self,
        capture: TrajectoryPreStepCapture,
        tasks: Mapping[int, Task],
        service: Mapping[str, Any],
    ) -> None:
        for route in self._mapping_items(service.get("routing")):
            if route.get("applied") is not True:
                continue
            source_uav = route.get("uav_id", route.get("source_uav"))
            if isinstance(source_uav, bool) or not isinstance(source_uav, int):
                raise RouteTelemetryError("applied route lacks source_uav")
            candidate = capture.route_candidates.get(source_uav)
            if candidate is None:
                raise RouteTelemetryError("applied route lacks pre-step candidate")
            task_id = route.get("task_id")
            if task_id != candidate["task_id"]:
                raise RouteTelemetryError("applied route task differs from pre-step head")
            task = tasks.get(int(task_id))
            if not isinstance(task, Task):
                raise RouteTelemetryError("applied route task is absent after step")
            key = (capture.episode_id, int(task_id), capture.route_step)
            ledger = self._ledger_for_task(capture.episode_id, task)
            if ledger.route_event_key is not None or key in self._route_events:
                raise RouteTelemetryError("task produced more than one formal route event")
            task_before = candidate["task_snapshot"]
            task_after = task.snapshot()
            event = self._common("route", capture.episode_id, int(task_id), key)
            event.update({
                "route_step": capture.route_step,
                "rollout_index": capture.rollout_index,
                "source_uav": source_uav,
                "route_category": candidate["route_category"],
                "selected_destination_uav": candidate["selected_destination_uav"],
                "route_action_index": candidate["route_action_index"],
                "route_status": "applied",
                "task_arrival_slot": task_before["arrival_slot"],
                "task_deadline_slot": task_before["deadline_slot"],
                "task_data_bits_initial": task_before["data_bits"],
                "task_cpu_cycles_initial": task_before["cpu_cycles"],
                "remaining_bits_at_route": task_before["remaining_bits"],
                "remaining_cycles_at_route": task_before["remaining_cycles"],
                "deadline_slack_slots_at_route": (
                    int(task_before["deadline_slot"]) - capture.route_step + 1
                ),
                "binding_slot": task_after["binding_slot"],
                "service_eligible_slot": task_after["service_eligible_slot"],
                "task_status_after_route": task_after["status"],
                "source_queue_proxy": candidate["source_queue_proxy"],
                "legal_remote_destinations": candidate["legal_remote_destinations"],
                "legal_route_mask": candidate["legal_route_mask"],
                "selected_link_proxy_status": candidate["selected_link_proxy_status"],
                "selected_link_proxy": candidate["selected_link_proxy"],
                "helper_queue_proxy_status": candidate["helper_queue_proxy_status"],
                "helper_queue_proxy": candidate["helper_queue_proxy"],
                "capture_source": "pre_step_observation_plus_post_step_routing",
            })
            ledger.route_event_key = key
            ledger.route_event = dict(event)
            self._route_events[key] = dict(event)
            transition_agent = (capture.rollout_index, source_uav)
            if transition_agent in self._route_by_transition_agent:
                raise RouteTelemetryError("transition agent produced duplicate route identity")
            self._route_by_transition_agent[transition_agent] = key
            self._write(event)

    def _observe_workload(
        self,
        capture: TrajectoryPreStepCapture,
        environment: Any,
        info: Mapping[str, Any],
        tasks: Mapping[int, Task],
        service: Mapping[str, Any],
    ) -> None:
        calculator = getattr(environment, "reward_calculator", None)
        primitive = getattr(calculator, "_urgent_workload_for_task", None)
        if not callable(primitive):
            raise RouteTelemetryError("production workload primitive is unavailable")
        reward = info.get("reward")
        if not isinstance(reward, Mapping):
            raise RouteTelemetryError("trajectory ledger requires reward terms")
        team_raw = _safe_float(reward.get("urgent_workload_s"))
        team_penalty = _safe_float(reward.get("workload_penalty"))
        team_completion = _safe_float(reward.get("completion_component"))
        team_expiration = _safe_float(reward.get("expiration_penalty"))
        if any(
            value is None
            for value in (
                team_raw,
                team_penalty,
                team_completion,
                team_expiration,
            )
        ):
            raise RouteTelemetryError("production workload terms are unavailable")
        arrival = info.get("arrival")
        arrival_items = (
            self._mapping_items(arrival.get("tasks"))
            if isinstance(arrival, Mapping) else ()
        )
        arrival_ids = {item.get("task_id") for item in arrival_items}
        truncated_ids = {
            item.get("task_id")
            for item in self._mapping_items(service.get("truncated_tasks"))
        }
        route_pre_mode = (
            WorkloadTimingMode(self.config.environment.workload_timing_mode)
            is WorkloadTimingMode.ROUTE_SLOT_PRE_ROUTE
        )
        coefficient = (
            calculator.weights.workload
            / calculator.references.workload_reference_s
        )
        task_raw_parts: list[float] = []
        task_penalty_parts: list[float] = []
        for task_id in sorted(tasks):
            task = tasks[task_id]
            if task_id in arrival_ids:
                continue
            if task.outcome is not TaskOutcome.NONE and task_id not in truncated_ids:
                continue
            ledger = self._ledger_for_task(capture.episode_id, task)
            snapshot = (
                capture.workload_snapshots.get(task_id)
                if route_pre_mode
                and ledger.route_event_key
                == (capture.episode_id, task_id, capture.route_step)
                else None
            )
            raw = float(primitive(task, capture.route_step, snapshot))
            penalty = coefficient * raw
            task_raw_parts.append(raw)
            task_penalty_parts.append(penalty)
            if ledger.route_event_key is None:
                ledger.workload_residual_raw += raw
                ledger.workload_residual_penalty += penalty
            else:
                ledger.cumulative_workload_raw += raw
                ledger.cumulative_workload_penalty += penalty
        task_raw = math.fsum(task_raw_parts)
        task_penalty = math.fsum(task_penalty_parts)
        self._team_workload_raw += team_raw
        self._team_workload_penalty += team_penalty
        self._team_completion_component += team_completion
        self._team_expiration_penalty += team_expiration
        self._workload_numeric_residual_raw += team_raw - task_raw
        self._workload_numeric_residual_penalty += team_penalty - task_penalty

    def _observe_energy_and_service(
        self,
        capture: TrajectoryPreStepCapture,
        info: Mapping[str, Any],
        tasks: Mapping[int, Task],
        service: Mapping[str, Any],
    ) -> None:
        slot_tx = 0.0
        slot_cpu = 0.0
        for link in self._mapping_items(service.get("links")):
            energy = _safe_float(link.get("transmit_energy_j"))
            if energy is None or energy < 0.0:
                raise RouteTelemetryError("link TX energy must be finite and non-negative")
            slot_tx += energy
            positive: list[tuple[int, float]] = []
            for item in self._mapping_items(link.get("task_services")):
                amount = _safe_float(item.get("amount"))
                task_id = item.get("task_id")
                if amount is None or amount < 0.0:
                    raise RouteTelemetryError("TX task service must be non-negative")
                if amount <= 0.0:
                    continue
                task = tasks.get(int(task_id))
                if not isinstance(task, Task):
                    raise RouteTelemetryError("TX service task is absent")
                positive.append((int(task_id), amount))
                ledger = self._ledger_for_task(capture.episode_id, task)
                ledger.cumulative_tx_bits += amount
                if ledger.first_tx_step is None:
                    ledger.first_tx_step = capture.route_step
                ledger.last_tx_step = capture.route_step
            total_bits = math.fsum(amount for _task_id, amount in positive)
            allocated_parts: list[float] = []
            if total_bits > 0.0:
                for task_id, amount in positive:
                    allocation = energy * amount / total_bits
                    allocated_parts.append(allocation)
                    self._tasks[(capture.episode_id, task_id)].allocated_task_tx_energy += allocation
            allocated = math.fsum(allocated_parts)
            self._measured_tx_energy += energy
            self._allocated_tx_energy += allocated
            self._unattributed_tx_energy += energy - allocated
        for item in self._mapping_items(service.get("cpu")):
            energy = _safe_float(item.get("cpu_energy_j"))
            cycles = _safe_float(item.get("service_cycles"))
            if energy is None or energy < 0.0 or cycles is None or cycles < 0.0:
                raise RouteTelemetryError("CPU service telemetry must be non-negative")
            slot_cpu += energy
            task_id = item.get("task_id")
            self._measured_cpu_energy += energy
            if task_id is None:
                self._unattributed_cpu_energy += energy
                continue
            task = tasks.get(int(task_id))
            if not isinstance(task, Task):
                raise RouteTelemetryError("CPU service task is absent")
            ledger = self._ledger_for_task(capture.episode_id, task)
            ledger.exact_task_cpu_energy += energy
            ledger.cumulative_cpu_cycles += cycles
            self._attributed_cpu_energy += energy
            if cycles > 0.0:
                if ledger.first_cpu_step is None:
                    ledger.first_cpu_step = capture.route_step
                ledger.last_cpu_step = capture.route_step
        reward = info.get("reward")
        production_energy = (
            _safe_float(reward.get("actual_energy_j"))
            if isinstance(reward, Mapping) else None
        )
        if production_energy is None:
            raise RouteTelemetryError("production energy term is unavailable")
        self._production_measured_energy += production_energy
        self._energy_numeric_residual += production_energy - slot_tx - slot_cpu

    def _workload_ledger_snapshot(self) -> dict[str, Any]:
        routed_raw = math.fsum(
            item.cumulative_workload_raw for item in self._tasks.values()
        )
        routed_penalty = math.fsum(
            item.cumulative_workload_penalty for item in self._tasks.values()
        )
        residual_raw = math.fsum(
            item.workload_residual_raw for item in self._tasks.values()
        )
        residual_penalty = math.fsum(
            item.workload_residual_penalty for item in self._tasks.values()
        )
        raw_conservation = self._team_workload_raw - math.fsum((
            routed_raw,
            residual_raw,
            self._workload_numeric_residual_raw,
        ))
        penalty_conservation = self._team_workload_penalty - math.fsum((
            routed_penalty,
            residual_penalty,
            self._workload_numeric_residual_penalty,
        ))
        return {
            "cumulative_team_workload_raw": self._team_workload_raw,
            "cumulative_routed_task_workload_raw": routed_raw,
            "cumulative_workload_residual_raw": residual_raw,
            "cumulative_numeric_residual_raw": self._workload_numeric_residual_raw,
            "raw_conservation_residual": raw_conservation,
            "cumulative_team_workload_penalty": self._team_workload_penalty,
            "cumulative_routed_task_workload_penalty": routed_penalty,
            "cumulative_workload_residual_penalty": residual_penalty,
            "cumulative_numeric_residual_penalty": self._workload_numeric_residual_penalty,
            "penalty_conservation_residual": penalty_conservation,
            "status": (
                "pass"
                if abs(raw_conservation) <= TRAJECTORY_LEDGER_TOLERANCE
                and abs(penalty_conservation) <= TRAJECTORY_LEDGER_TOLERANCE
                else "residual"
            ),
        }

    def _energy_ledger_snapshot(self) -> dict[str, Any]:
        tx_residual = self._measured_tx_energy - math.fsum((
            self._allocated_tx_energy,
            self._unattributed_tx_energy,
        ))
        cpu_residual = self._measured_cpu_energy - math.fsum((
            self._attributed_cpu_energy,
            self._unattributed_cpu_energy,
        ))
        production_residual = self._production_measured_energy - math.fsum((
            self._measured_tx_energy,
            self._measured_cpu_energy,
            self._energy_numeric_residual,
        ))
        return {
            "tx_energy_attribution_method": "bits_pro_rata",
            "tx_energy_attribution_status": "diagnostic_allocation",
            "production_measured_tx_energy": self._measured_tx_energy,
            "allocated_task_tx_energy": self._allocated_tx_energy,
            "unattributed_tx_energy": self._unattributed_tx_energy,
            "tx_conservation_residual": tx_residual,
            "cpu_energy_attribution_status": "exact_task_id",
            "production_measured_cpu_energy": self._measured_cpu_energy,
            "attributed_task_cpu_energy": self._attributed_cpu_energy,
            "unattributed_cpu_energy": self._unattributed_cpu_energy,
            "cpu_conservation_residual": cpu_residual,
            "production_measured_total_energy": self._production_measured_energy,
            "production_energy_numeric_residual": self._energy_numeric_residual,
            "total_conservation_residual": production_residual,
            "status": (
                "pass"
                if max(abs(tx_residual), abs(cpu_residual), abs(production_residual))
                <= TRAJECTORY_LEDGER_TOLERANCE
                else "residual"
            ),
        }

    def _reward_ledger_snapshot(self) -> dict[str, Any]:
        task_completion = math.fsum(
            item.completion_component for item in self._tasks.values()
        )
        task_expiration = math.fsum(
            item.expiration_penalty for item in self._tasks.values()
        )
        completion_residual = self._team_completion_component - task_completion
        expiration_residual = self._team_expiration_penalty - task_expiration
        return {
            "cumulative_team_completion_component": self._team_completion_component,
            "cumulative_task_completion_component": task_completion,
            "completion_conservation_residual": completion_residual,
            "cumulative_team_expiration_penalty": self._team_expiration_penalty,
            "cumulative_task_expiration_penalty": task_expiration,
            "expiration_conservation_residual": expiration_residual,
            "status": (
                "pass"
                if max(abs(completion_residual), abs(expiration_residual))
                <= TRAJECTORY_LEDGER_TOLERANCE
                else "residual"
            ),
        }

    def _emit_terminal(
        self,
        *,
        ledger: _TrajectoryTaskLedger,
        task_snapshot: Mapping[str, Any],
        observed_transition_step: int,
        terminal_kind: str,
        terminal_step: int | None,
        outcome: str | None,
        calculator: Any,
    ) -> None:
        task_key = (ledger.episode_id, ledger.task_id)
        if ledger.terminal_emitted or task_key in self._terminal_task_keys:
            raise RouteTelemetryError("task produced more than one terminal event")
        completion_component = (
            calculator.weights.completion / calculator.references.task_count_reference
            if outcome == TaskOutcome.DONE.value else 0.0
        )
        expiration_penalty = (
            calculator.weights.expiration / calculator.references.task_count_reference
            if outcome == TaskOutcome.EXPIRED.value else 0.0
        )
        energy_scale = (
            calculator.weights.energy / calculator.references.active_energy_reference_j
        )
        route = ledger.route_event or {}
        ledger.completion_component = completion_component
        ledger.expiration_penalty = expiration_penalty
        completion_slot = task_snapshot.get("completion_slot")
        latency_slots = (
            int(completion_slot) - int(task_snapshot["arrival_slot"])
            if outcome == TaskOutcome.DONE.value and completion_slot is not None
            else None
        )
        event = self._common(
            "terminal", ledger.episode_id, ledger.task_id, ledger.route_event_key
        )
        event.update({
            "route_status": "routed" if ledger.route_event_key is not None else "never_routed",
            "terminal_kind": terminal_kind,
            "outcome": outcome,
            "observed_transition_step": observed_transition_step,
            "terminal_step": terminal_step,
            "source_uav": ledger.source_uav,
            "destination_uav": task_snapshot.get("destination"),
            "route_category": route.get("route_category"),
            "route_step": route.get("route_step"),
            "task_arrival_slot": task_snapshot.get("arrival_slot"),
            "task_deadline_slot": task_snapshot.get("deadline_slot"),
            "binding_slot": task_snapshot.get("binding_slot"),
            "service_eligible_slot": task_snapshot.get("service_eligible_slot"),
            "cpu_entry_slot": task_snapshot.get("cpu_entry_slot"),
            "completion_slot": completion_slot,
            "e2e_delay_slots": latency_slots,
            "e2e_delay_s": (
                latency_slots * self.config.environment.slot_duration_s
                if latency_slots is not None else None
            ),
            "first_tx_step": ledger.first_tx_step,
            "last_tx_step": ledger.last_tx_step,
            "tx_complete_step": (
                task_snapshot.get("cpu_entry_slot")
                if route.get("route_category") == "remote" else None
            ),
            "first_cpu_step": ledger.first_cpu_step,
            "last_cpu_step": ledger.last_cpu_step,
            "final_remaining_bits": task_snapshot.get("remaining_bits"),
            "final_remaining_cycles": task_snapshot.get("remaining_cycles"),
            "cumulative_tx_bits": ledger.cumulative_tx_bits,
            "cumulative_cpu_cycles": ledger.cumulative_cpu_cycles,
            "reward_decomposition": {
                "completion_component": completion_component,
                "completion_component_signed": completion_component,
                "expiration_penalty": expiration_penalty,
                "expiration_component_signed": -expiration_penalty,
                "cumulative_workload_raw": ledger.cumulative_workload_raw,
                "cumulative_workload_penalty": ledger.cumulative_workload_penalty,
                "workload_residual_raw": ledger.workload_residual_raw,
                "workload_residual_penalty": ledger.workload_residual_penalty,
                "workload_attribution_status": (
                    "task_decomposed_with_pre_route_residual"
                    if ledger.route_event_key is not None else "unrouted_residual"
                ),
                "exact_task_cpu_energy": ledger.exact_task_cpu_energy,
                "cpu_energy_penalty": energy_scale * ledger.exact_task_cpu_energy,
                "allocated_task_tx_energy": ledger.allocated_task_tx_energy,
                "tx_energy_penalty_diagnostic": (
                    energy_scale * ledger.allocated_task_tx_energy
                ),
                "tx_energy_attribution_method": "bits_pro_rata",
                "tx_energy_attribution_status": "diagnostic_allocation",
            },
            "workload_ledger": self._workload_ledger_snapshot(),
            "energy_ledger": self._energy_ledger_snapshot(),
            "reward_ledger": self._reward_ledger_snapshot(),
        })
        ledger.terminal_emitted = True
        ledger.latest_task_snapshot = dict(task_snapshot)
        self._terminal_task_keys.add(task_key)
        self._write(event)

    def _observe_terminals(
        self,
        capture: TrajectoryPreStepCapture,
        environment: Any,
        info: Mapping[str, Any],
        tasks: Mapping[int, Task],
        service: Mapping[str, Any],
    ) -> None:
        calculator = environment.reward_calculator
        for section in ("settled_tasks", "truncated_tasks"):
            for task_snapshot in self._mapping_items(service.get(section)):
                task_id = int(task_snapshot["task_id"])
                task = tasks.get(task_id)
                if not isinstance(task, Task):
                    raise RouteTelemetryError("terminal task is absent from lifecycle")
                ledger = self._ledger_for_task(capture.episode_id, task)
                outcome = str(task_snapshot.get("outcome"))
                terminal_step = (
                    int(info.get("boundary_slot"))
                    if outcome == TaskOutcome.TRUNCATED.value
                    else int(task_snapshot["completion_slot"])
                    if outcome == TaskOutcome.DONE.value
                    and task_snapshot.get("completion_slot") is not None
                    else capture.route_step
                )
                self._emit_terminal(
                    ledger=ledger,
                    task_snapshot=task_snapshot,
                    observed_transition_step=capture.route_step,
                    terminal_kind="lifecycle",
                    terminal_step=terminal_step,
                    outcome=outcome,
                    calculator=calculator,
                )

    def observe_step(
        self,
        capture: TrajectoryPreStepCapture,
        environment: Any,
        info: Mapping[str, Any],
    ) -> None:
        if not isinstance(capture, TrajectoryPreStepCapture):
            raise TypeError("capture must be TrajectoryPreStepCapture")
        if not isinstance(info, Mapping) or info.get("slot") != capture.route_step:
            raise RouteTelemetryError("trajectory step info differs from pre-step capture")
        lifecycle = getattr(environment, "lifecycle", None)
        tasks = getattr(lifecycle, "tasks", None)
        if not isinstance(tasks, Mapping):
            raise RouteTelemetryError("telemetry requires environment lifecycle tasks")
        typed_tasks = {
            int(task_id): task
            for task_id, task in tasks.items()
            if isinstance(task, Task)
        }
        for task in typed_tasks.values():
            self._ledger_for_task(capture.episode_id, task)
        service = info.get("service")
        if not isinstance(service, Mapping):
            raise RouteTelemetryError("trajectory step requires service telemetry")
        self._observe_routes(capture, typed_tasks, service)
        self._observe_workload(capture, environment, info, typed_tasks, service)
        self._observe_energy_and_service(capture, info, typed_tasks, service)
        self._observe_terminals(capture, environment, info, typed_tasks, service)

    def observe_ppo_update(
        self,
        output: Any,
        *,
        update_index: int,
        policy_version_before: int,
        policy_version_after: int,
        rollout_start_index: int,
        rollout_length: int,
    ) -> None:
        epochs = getattr(output, "epoch_diagnostics", ())
        if not epochs or getattr(epochs[0], "epoch_index", None) != 0:
            raise RouteTelemetryError("canonical PPO epoch 0 telemetry is unavailable")
        route_telemetry = getattr(epochs[0], "route_telemetry", None)
        samples = (
            getattr(route_telemetry, "samples", ())
            if route_telemetry is not None else ()
        )
        expected = {
            key
            for key, event in self._route_events.items()
            if rollout_start_index
            <= int(event["rollout_index"])
            < rollout_start_index + rollout_length
        }
        emitted: set[tuple[int, int, int]] = set()
        chunk_length = self.config.training.mappo.recurrent_chunk_length_slots
        for sample in samples:
            local_index = (
                int(sample.batch_index) * chunk_length + int(sample.time_index)
            )
            global_index = rollout_start_index + local_index
            key = self._route_by_transition_agent.get(
                (global_index, int(sample.agent_index))
            )
            if key is None:
                continue
            route = self._route_events[key]
            if (
                sample.category != route["route_category"]
                or sample.selected_destination_uav
                != route["selected_destination_uav"]
                or sample.selected_route_action_index != route["route_action_index"]
            ):
                raise RouteTelemetryError("PPO route sample differs from route identity")
            if sample.task_id is not None and sample.task_id != key[1]:
                raise RouteTelemetryError(
                    "PPO N-step sample task differs from formal route identity"
                )
            if (
                sample.formal_route_event_key is not None
                and tuple(sample.formal_route_event_key) != key
            ):
                raise RouteTelemetryError(
                    "PPO N-step formal route-event key is inconsistent"
                )
            if key in self._credit_route_keys or key in emitted:
                raise RouteTelemetryError("route transition produced duplicate PPO credit")
            event = self._common("credit", key[0], key[1], key)
            event.update({
                "route_step": key[2],
                "rollout_index": route["rollout_index"],
                "source_uav": route["source_uav"],
                "route_category": route["route_category"],
                "selected_destination_uav": route["selected_destination_uav"],
                "route_action_index": route["route_action_index"],
                "credit_status": "linked",
                "ppo_update_index": update_index,
                "ppo_epoch_index": 0,
                "policy_version_before": policy_version_before,
                "policy_version_after": policy_version_after,
                "advantage": sample.advantage,
                "base_route_advantage": sample.base_route_advantage,
                "new_route_advantage": sample.new_route_advantage,
                "route_advantage_delta": sample.route_advantage_delta,
                "effective_route_gae_lambda": sample.effective_route_gae_lambda,
                "credit_trace_length_slots": sample.credit_trace_length_slots,
                "trace_end_kind": sample.trace_end_kind,
                "bootstrap_source": sample.bootstrap_source,
                # ``sample.task_id`` is optional N-step provenance.  Shared
                # GAE intentionally leaves it as None, while the artifact
                # credit event always has the formal identity in ``key``.
                "task_id": key[1],
                "branch_event_key": (
                    None
                    if sample.branch_event_key is None
                    else list(sample.branch_event_key)
                ),
                "branch_credit_event_key": (
                    None
                    if sample.branch_event_key is None
                    else list(sample.branch_event_key)
                ),
                "formal_route_event_key": (
                    None
                    if sample.formal_route_event_key is None
                    else list(sample.formal_route_event_key)
                ),
                "nstep_route_advantage": sample.nstep_route_advantage,
                "nstep_horizon": sample.nstep_horizon,
                "nstep_stop_kind": sample.nstep_stop_kind,
                "terminal_before_rollout": (
                    sample.terminal_before_rollout
                ),
                "bootstrap_used": sample.bootstrap_used,
                "credit_global_rollout_index": (
                    sample.global_rollout_index
                ),
                "credit_policy_version": sample.policy_version,
                "td_residual": sample.td_residual,
                "return_target": sample.return_target,
                "old_route_log_prob": sample.old_route_log_prob,
                "new_route_log_prob": sample.new_route_log_prob,
                "credit_source": "epoch0_route_telemetry",
                "na_reason": None,
            })
            self._write(event)
            emitted.add(key)
            self._credit_route_keys.add(key)
        if emitted != expected:
            missing = sorted(expected - emitted)
            raise RouteTelemetryError(
                f"canonical PPO credit linkage is incomplete; missing={missing}"
            )

    def finalize_collection(
        self,
        *,
        environment: Any,
        episode_id: int,
        observed_transition_step: int,
        optimized_transitions: int,
    ) -> Mapping[str, Any]:
        lifecycle = getattr(environment, "lifecycle", None)
        tasks = getattr(lifecycle, "tasks", {})
        calculator = getattr(environment, "reward_calculator", None)
        if calculator is None:
            raise RouteTelemetryError("reward calculator is unavailable at finalization")
        for task in tasks.values():
            if not isinstance(task, Task):
                continue
            ledger = self._ledger_for_task(episode_id, task)
            if not ledger.terminal_emitted:
                if task.is_terminal:
                    raise RouteTelemetryError(
                        "lifecycle-terminal task lacks its terminal event"
                    )
                self._emit_terminal(
                    ledger=ledger,
                    task_snapshot=task.snapshot(),
                    observed_transition_step=observed_transition_step,
                    terminal_kind="collection_censored",
                    terminal_step=None,
                    outcome=None,
                    calculator=calculator,
                )
        for key, route in sorted(
            self._route_events.items(), key=lambda item: item[1]["rollout_index"]
        ):
            if key in self._credit_route_keys:
                continue
            if int(route["rollout_index"]) < optimized_transitions:
                raise RouteTelemetryError("optimized route lacks canonical PPO credit")
            event = self._common("credit", key[0], key[1], key)
            event.update({
                "route_step": key[2],
                "rollout_index": route["rollout_index"],
                "source_uav": route["source_uav"],
                "route_category": route["route_category"],
                "selected_destination_uav": route["selected_destination_uav"],
                "route_action_index": route["route_action_index"],
                "credit_status": "tail_not_optimized",
                "ppo_update_index": None,
                "ppo_epoch_index": None,
                "policy_version_before": None,
                "policy_version_after": None,
                "advantage": None,
                "base_route_advantage": None,
                "new_route_advantage": None,
                "route_advantage_delta": None,
                "effective_route_gae_lambda": None,
                "credit_trace_length_slots": None,
                "trace_end_kind": None,
                "bootstrap_source": None,
                "task_id": route.get("task_id"),
                "branch_event_key": None,
                "branch_credit_event_key": None,
                "formal_route_event_key": None,
                "nstep_route_advantage": None,
                "nstep_horizon": None,
                "nstep_stop_kind": None,
                "terminal_before_rollout": None,
                "bootstrap_used": None,
                "credit_global_rollout_index": None,
                "credit_policy_version": None,
                "td_residual": None,
                "return_target": None,
                "old_route_log_prob": None,
                "new_route_log_prob": None,
                "credit_source": None,
                "na_reason": "tail_not_optimized",
            })
            self._write(event)
            self._credit_route_keys.add(key)
        if set(self._route_events) != self._credit_route_keys:
            raise RouteTelemetryError("route/credit cardinality differs at finalization")
        return self.summary()

    def summary(self) -> Mapping[str, Any]:
        return {
            "schema_version": TRAJECTORY_CREDIT_SCHEMA_VERSION,
            "event_counts": dict(self._event_counts),
            "task_count": len(self._tasks),
            "route_event_count": len(self._route_events),
            "terminal_event_count": len(self._terminal_task_keys),
            "credit_event_count": len(self._credit_route_keys),
            "workload_ledger": self._workload_ledger_snapshot(),
            "energy_ledger": self._energy_ledger_snapshot(),
            "reward_ledger": self._reward_ledger_snapshot(),
        }


def aggregate_route_outcomes(
    outcomes: Iterable[RouteOutcomeAssociation],
) -> dict[str, object]:
    items = list(outcomes)
    result: dict[str, object] = {}
    for category in ROUTE_ACTION_CATEGORIES:
        subset = [item for item in items if item.category == category]
        completed = [item for item in subset if item.outcome == "completed"]
        latencies = [
            float(item.end_to_end_latency_slots)
            for item in completed if item.end_to_end_latency_slots is not None
        ]
        reward_means = {}
        for name in (
            "immediate_reward", "workload_penalty_component",
            "completion_component", "expiration_component", "energy_component"
        ):
            values = [
                float(getattr(item, name)) for item in subset
                if getattr(item, name) is not None
            ]
            reward_means[name] = sum(values) / len(values) if values else None
        result[category] = {
            "count": len(subset),
            "completed": sum(item.outcome == "completed" for item in subset),
            "expired": sum(item.outcome == "expired" for item in subset),
            "truncated": sum(item.outcome == "truncated" for item in subset),
            "completion_rate": len(completed) / len(subset) if subset else None,
            "mean_latency_slots_completed": (
                sum(latencies) / len(latencies) if latencies else None
            ),
            "valid_latency_count": len(latencies),
            "team_level_reward_means": reward_means,
        }
    return result

__all__ = [
    "BRANCH_PPO_RECORD_FIELDS",
    "BranchPPODynamics",
    "BranchPPODynamicsGroup",
    "GradientMeasurement",
    "ROUTE_GRADIENT_STATUSES",
    "ROUTE_TELEMETRY_SCHEMA_VERSION",
    "ROUTE_DIAGNOSTIC_SCHEMA_VERSION",
    "ROUTE_DIAGNOSTIC_UTILITY_LABEL",
    "TRAJECTORY_CREDIT_EVENT_TYPES",
    "TRAJECTORY_CREDIT_SCHEMA_VERSION",
    "TRAJECTORY_LEDGER_TOLERANCE",
    "RouteHeadGradientTelemetry",
    "RouteTelemetry",
    "RouteTelemetryError",
    "collect_route_telemetry",
    "measure_route_head_gradients",
    "ROUTE_ACTION_CATEGORIES",
    "ROUTE_DISTRIBUTION_GROUPS",
    "ROUTE_PPO_RECORD_FIELDS",
    "RouteDistributionSummary",
    "RouteHeadGradientRowTelemetry",
    "RouteOutcomeAssociation",
    "RouteOutcomeTracker",
    "RouteDiagnosticSampleCollector",
    "RoutePPODynamics",
    "RoutePPODynamicsGroup",
    "RouteSampleTelemetry",
    "TrajectoryCreditTracker",
    "TrajectoryPreStepCapture",
    "reconstruct_route_diagnostic_utility",
    "route_diagnostic_schema_header",
    "aggregate_route_outcomes",
    "compute_route_only_ppo_dynamics",
    "compute_branch_ppo_dynamics",
    "measure_route_head_gradient_rows",
    "measure_shared_trunk_gradients",
    "summarize_distribution",
]
