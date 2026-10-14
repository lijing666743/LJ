"""Deterministic actor observations and conditional seven-branch masks.

This module is the actor-safe boundary of the environment.  It consumes only
slot-start records and deliberately has no RNG, no access to the current true
channel, and no proposal-dependent executor scalar.  Python UAV identifiers
are zero based; every public record also exposes the corresponding one-based
paper identifier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import ActionConfig, EnvironmentConfig, RunConfig
from .actions import ActionProposal
from .action_history import PreviousActionFeatures, PreviousActionSnapshot
from .energy import UavEnergyState, cpu_reservation_j
from .history import ActorChannelFeatures
from .lifecycle import LifecycleManager
from .mobility import MobilityState
from .queues import TaskQueue
from .public_history import PublicMessageSnapshot
from .topology import TopologySnapshot, candidate_neighbor_mask, pairwise_distances


BRANCH_ORDER = (
    "route",
    "tx_select",
    "resource_group",
    "resource_width",
    "power_level",
    "cpu_queue",
    "cpu_frequency",
)

ActionValue = str | int | float


class ObservationError(ValueError):
    """Raised when an actor snapshot would violate shape or causality rules."""


def _readonly(values: Any, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _finite_nonnegative(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObservationError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ObservationError(f"{name} must be finite")
    if positive and converted <= 0.0:
        raise ObservationError(f"{name} must be positive")
    if not positive and converted < 0.0:
        raise ObservationError(f"{name} must be non-negative")
    return converted


def _paired_vector(
    values: Sequence[float] | np.ndarray | None,
    mask: Sequence[bool] | np.ndarray | None,
    count: int,
    name: str,
    *,
    upper_bound: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a value/mask pair and canonicalize unavailable entries to zero."""

    if values is None and mask is None:
        return np.zeros(count, dtype=np.float64), np.zeros(count, dtype=np.bool_)
    if values is None or mask is None:
        raise ObservationError(f"{name} values and validity mask must be supplied together")
    converted = np.asarray(values, dtype=np.float64)
    valid = np.asarray(mask, dtype=np.bool_)
    if converted.shape != (count,) or valid.shape != (count,):
        raise ObservationError(f"{name} values and mask must have shape ({count},)")
    if not np.all(np.isfinite(converted)) or np.any(converted < 0.0):
        raise ObservationError(f"{name} values must be finite and non-negative")
    if upper_bound is not None and np.any(converted[valid] > upper_bound):
        raise ObservationError(f"valid {name} values must not exceed {upper_bound}")
    converted = converted.copy()
    converted[~valid] = 0.0
    return converted, valid.copy()


def _paired_matrix(
    values: np.ndarray | Sequence[Sequence[float]] | None,
    mask: np.ndarray | Sequence[Sequence[bool]] | None,
    count: int,
    name: str,
    *,
    upper_bound: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if values is None and mask is None:
        return (
            np.zeros((count, count), dtype=np.float64),
            np.zeros((count, count), dtype=np.bool_),
        )
    if values is None or mask is None:
        raise ObservationError(f"{name} values and validity mask must be supplied together")
    converted = np.asarray(values, dtype=np.float64)
    valid = np.asarray(mask, dtype=np.bool_)
    expected = (count, count)
    if converted.shape != expected or valid.shape != expected:
        raise ObservationError(f"{name} values and mask must have shape {expected}")
    if not np.all(np.isfinite(converted)) or np.any(converted < 0.0):
        raise ObservationError(f"{name} values must be finite and non-negative")
    if upper_bound is not None and np.any(converted[valid] > upper_bound):
        raise ObservationError(f"valid {name} values must not exceed {upper_bound}")
    converted = converted.copy()
    converted[~valid] = 0.0
    return converted, valid.copy()


@dataclass(frozen=True)
class QueueSummary:
    """A queue aggregate with an explicit mask for its EDF head fields."""

    task_count: int
    remaining_bits: float
    remaining_cycles: float
    head_task_id: int
    head_remaining_bits: float
    head_remaining_cycles: float
    head_slack_slots: int
    head_valid_mask: bool

    def __post_init__(self) -> None:
        if isinstance(self.task_count, bool) or not isinstance(self.task_count, int) or self.task_count < 0:
            raise ObservationError("task_count must be a non-negative integer")
        for name in (
            "remaining_bits",
            "remaining_cycles",
            "head_remaining_bits",
            "head_remaining_cycles",
        ):
            _finite_nonnegative(getattr(self, name), name)
        if self.head_valid_mask:
            if self.task_count == 0 or self.head_task_id < 0:
                raise ObservationError("a valid queue head requires a task and non-negative task ID")
        elif (
            self.head_task_id != -1
            or self.head_remaining_bits != 0.0
            or self.head_remaining_cycles != 0.0
            or self.head_slack_slots != 0
        ):
            raise ObservationError("an unavailable queue head must use canonical placeholders")

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_count": self.task_count,
            "remaining_bits": self.remaining_bits,
            "remaining_cycles": self.remaining_cycles,
            "head_task_id": self.head_task_id,
            "head_remaining_bits": self.head_remaining_bits,
            "head_remaining_cycles": self.head_remaining_cycles,
            "head_slack_slots": self.head_slack_slots,
            "head_valid_mask": self.head_valid_mask,
        }


@dataclass(frozen=True)
class IndexedQueueFeatures:
    """Fixed-UAV-index queue summaries for TX destinations or CPU sources."""

    task_count: np.ndarray
    remaining_bits: np.ndarray
    remaining_cycles: np.ndarray
    head_task_id: np.ndarray
    head_remaining_bits: np.ndarray
    head_remaining_cycles: np.ndarray
    head_slack_slots: np.ndarray
    head_valid_mask: np.ndarray

    def __post_init__(self) -> None:
        count = np.asarray(self.task_count).shape
        if len(count) != 1:
            raise ObservationError("indexed queue features must be one-dimensional")
        for name in (
            "remaining_bits",
            "remaining_cycles",
            "head_task_id",
            "head_remaining_bits",
            "head_remaining_cycles",
            "head_slack_slots",
            "head_valid_mask",
        ):
            if np.asarray(getattr(self, name)).shape != count:
                raise ObservationError("all indexed queue feature arrays must share one shape")
        task_count = np.asarray(self.task_count)
        if np.any(task_count < 0):
            raise ObservationError("indexed task counts must be non-negative")
        for name in ("remaining_bits", "remaining_cycles", "head_remaining_bits", "head_remaining_cycles"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ObservationError(f"{name} must be finite and non-negative")
        dtypes = {
            "task_count": np.int64,
            "remaining_bits": np.float64,
            "remaining_cycles": np.float64,
            "head_task_id": np.int64,
            "head_remaining_bits": np.float64,
            "head_remaining_cycles": np.float64,
            "head_slack_slots": np.int64,
            "head_valid_mask": np.bool_,
        }
        for name, dtype in dtypes.items():
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))

    @property
    def count(self) -> int:
        return int(self.task_count.shape[0])

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_count": self.task_count.tolist(),
            "remaining_bits": self.remaining_bits.tolist(),
            "remaining_cycles": self.remaining_cycles.tolist(),
            "head_task_id": self.head_task_id.tolist(),
            "head_remaining_bits": self.head_remaining_bits.tolist(),
            "head_remaining_cycles": self.head_remaining_cycles.tolist(),
            "head_slack_slots": self.head_slack_slots.tolist(),
            "head_valid_mask": self.head_valid_mask.tolist(),
        }


@dataclass(frozen=True)
class PrivateQueueFeatures:
    """Only the current actor's private queue aggregates."""

    unbound: QueueSummary
    local_cpu: QueueSummary
    tx_by_destination: IndexedQueueFeatures
    cpu_by_source: IndexedQueueFeatures

    def snapshot(self) -> dict[str, Any]:
        return {
            "unbound": self.unbound.snapshot(),
            "local_cpu": self.local_cpu.snapshot(),
            "tx_by_destination": self.tx_by_destination.snapshot(),
            "cpu_by_source": self.cpu_by_source.snapshot(),
        }


@dataclass(frozen=True)
class SelfResourceFeatures:
    """The actor's own current capability and residual-energy state."""

    residual_energy_j: float
    initial_energy_j: float
    residual_energy_ratio: float
    max_transmit_power_w: float
    max_transmit_power_ratio: float
    max_cpu_frequency_hz: float
    max_cpu_frequency_ratio: float
    cpu_coefficient: float
    cpu_coefficient_ratio: float

    def __post_init__(self) -> None:
        for name in (
            "residual_energy_j",
            "residual_energy_ratio",
            "max_transmit_power_ratio",
            "max_cpu_frequency_ratio",
            "cpu_coefficient_ratio",
        ):
            _finite_nonnegative(getattr(self, name), name)
        for name in (
            "initial_energy_j",
            "max_transmit_power_w",
            "max_cpu_frequency_hz",
            "cpu_coefficient",
        ):
            _finite_nonnegative(getattr(self, name), name, positive=True)

    def snapshot(self) -> dict[str, float]:
        return {
            "residual_energy_j": self.residual_energy_j,
            "initial_energy_j": self.initial_energy_j,
            "residual_energy_ratio": self.residual_energy_ratio,
            "max_transmit_power_w": self.max_transmit_power_w,
            "max_transmit_power_ratio": self.max_transmit_power_ratio,
            "max_cpu_frequency_hz": self.max_cpu_frequency_hz,
            "max_cpu_frequency_ratio": self.max_cpu_frequency_ratio,
            "cpu_coefficient": self.cpu_coefficient,
            "cpu_coefficient_ratio": self.cpu_coefficient_ratio,
        }


@dataclass(frozen=True)
class NeighborPublicFeatures:
    """Causal sanitized payloads received from current candidate neighbors."""

    valid_mask: np.ndarray
    uav_ids: np.ndarray
    paper_uav_ids: np.ndarray
    relative_position_m: np.ndarray
    relative_velocity_mps: np.ndarray
    residual_energy_j: np.ndarray
    residual_energy_ratio: np.ndarray
    max_transmit_power_ratio: np.ndarray
    max_cpu_frequency_ratio: np.ndarray
    cpu_coefficient_ratio: np.ndarray
    cpu_load_task_count: np.ndarray
    cpu_load_remaining_cycles: np.ndarray
    message_source_slots: np.ndarray
    message_aoi_slots: np.ndarray
    message_aoi_valid_mask: np.ndarray

    def __post_init__(self) -> None:
        valid = np.asarray(self.valid_mask)
        if valid.ndim != 1:
            raise ObservationError("neighbor valid_mask must be one-dimensional")
        count = valid.shape[0]
        vector_fields = (
            "uav_ids",
            "paper_uav_ids",
            "residual_energy_j",
            "residual_energy_ratio",
            "max_transmit_power_ratio",
            "max_cpu_frequency_ratio",
            "cpu_coefficient_ratio",
            "cpu_load_task_count",
            "cpu_load_remaining_cycles",
            "message_source_slots",
            "message_aoi_slots",
            "message_aoi_valid_mask",
        )
        for name in vector_fields:
            if np.asarray(getattr(self, name)).shape != (count,):
                raise ObservationError(f"{name} must have shape ({count},)")
        for name in ("relative_position_m", "relative_velocity_mps"):
            if np.asarray(getattr(self, name)).shape != (count, 3):
                raise ObservationError(f"{name} must have shape ({count}, 3)")
        dtypes = {
            "valid_mask": np.bool_,
            "uav_ids": np.int64,
            "paper_uav_ids": np.int64,
            "relative_position_m": np.float64,
            "relative_velocity_mps": np.float64,
            "residual_energy_j": np.float64,
            "residual_energy_ratio": np.float64,
            "max_transmit_power_ratio": np.float64,
            "max_cpu_frequency_ratio": np.float64,
            "cpu_coefficient_ratio": np.float64,
            "cpu_load_task_count": np.int64,
            "cpu_load_remaining_cycles": np.float64,
            "message_source_slots": np.int64,
            "message_aoi_slots": np.int64,
            "message_aoi_valid_mask": np.bool_,
        }
        for name, dtype in dtypes.items():
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))

    def snapshot(self) -> dict[str, Any]:
        return {name: getattr(self, name).tolist() for name in (
            "valid_mask",
            "uav_ids",
            "paper_uav_ids",
            "relative_position_m",
            "relative_velocity_mps",
            "residual_energy_j",
            "residual_energy_ratio",
            "max_transmit_power_ratio",
            "max_cpu_frequency_ratio",
            "cpu_coefficient_ratio",
            "cpu_load_task_count",
            "cpu_load_remaining_cycles",
            "message_source_slots",
            "message_aoi_slots",
            "message_aoi_valid_mask",
        )}


@dataclass(frozen=True)
class ActorEdgeFeatures:
    """Actor-legal directed edge history for candidates and bound TX edges."""

    visible_mask: np.ndarray
    estimated_distance_m: np.ndarray
    stale_csi: np.ndarray
    csi_valid_mask: np.ndarray
    csi_aoi_slots: np.ndarray
    interference_history_w: np.ndarray
    interference_valid_mask: np.ndarray
    message_aoi_slots: np.ndarray
    message_aoi_valid_mask: np.ndarray
    historical_quality: np.ndarray
    quality_valid_mask: np.ndarray
    last_effective_rate_bps: np.ndarray
    last_rate_valid_mask: np.ndarray
    outage_rate: np.ndarray
    outage_valid_mask: np.ndarray

    def __post_init__(self) -> None:
        visible = np.asarray(self.visible_mask)
        if visible.ndim != 1:
            raise ObservationError("edge visible_mask must be one-dimensional")
        count = visible.shape[0]
        stale = np.asarray(self.stale_csi)
        if stale.ndim != 2 or stale.shape[0] != count or not np.iscomplexobj(stale):
            raise ObservationError("stale_csi must have shape (uav_count, ru_count) and be complex")
        ru_count = stale.shape[1]
        for name in (
            "estimated_distance_m",
            "csi_valid_mask",
            "csi_aoi_slots",
            "message_aoi_slots",
            "message_aoi_valid_mask",
            "last_effective_rate_bps",
            "last_rate_valid_mask",
            "outage_rate",
            "outage_valid_mask",
        ):
            if np.asarray(getattr(self, name)).shape != (count,):
                raise ObservationError(f"{name} must have shape ({count},)")
        for name in (
            "interference_history_w",
            "interference_valid_mask",
            "historical_quality",
            "quality_valid_mask",
        ):
            if np.asarray(getattr(self, name)).shape != (count, ru_count):
                raise ObservationError(f"{name} must have shape ({count}, {ru_count})")
        if not np.all(np.isfinite(stale)):
            raise ObservationError("stale CSI placeholders must be finite")
        dtypes = {
            "visible_mask": np.bool_,
            "estimated_distance_m": np.float64,
            "stale_csi": np.complex128,
            "csi_valid_mask": np.bool_,
            "csi_aoi_slots": np.int64,
            "interference_history_w": np.float64,
            "interference_valid_mask": np.bool_,
            "message_aoi_slots": np.int64,
            "message_aoi_valid_mask": np.bool_,
            "historical_quality": np.float64,
            "quality_valid_mask": np.bool_,
            "last_effective_rate_bps": np.float64,
            "last_rate_valid_mask": np.bool_,
            "outage_rate": np.float64,
            "outage_valid_mask": np.bool_,
        }
        for name, dtype in dtypes.items():
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))

    def snapshot(self) -> dict[str, Any]:
        return {
            "visible_mask": self.visible_mask.tolist(),
            "estimated_distance_m": self.estimated_distance_m.tolist(),
            "stale_csi_real": self.stale_csi.real.tolist(),
            "stale_csi_imag": self.stale_csi.imag.tolist(),
            "csi_valid_mask": self.csi_valid_mask.tolist(),
            "csi_aoi_slots": self.csi_aoi_slots.tolist(),
            "interference_history_w": self.interference_history_w.tolist(),
            "interference_valid_mask": self.interference_valid_mask.tolist(),
            "message_aoi_slots": self.message_aoi_slots.tolist(),
            "message_aoi_valid_mask": self.message_aoi_valid_mask.tolist(),
            "historical_quality": self.historical_quality.tolist(),
            "quality_valid_mask": self.quality_valid_mask.tolist(),
            "last_effective_rate_bps": self.last_effective_rate_bps.tolist(),
            "last_rate_valid_mask": self.last_rate_valid_mask.tolist(),
            "outage_rate": self.outage_rate.tolist(),
            "outage_valid_mask": self.outage_valid_mask.tolist(),
        }


@dataclass(frozen=True)
class ActionMasks:
    """Immutable slot-start context for all conditional categorical masks."""

    uav_id: int
    paper_uav_id: int
    sampling_order: tuple[str, ...]
    route_domain: tuple[ActionValue, ...]
    tx_select_domain: tuple[ActionValue, ...]
    resource_group_domain: tuple[ActionValue, ...]
    resource_width_domain: tuple[ActionValue, ...]
    power_level_domain: tuple[ActionValue, ...]
    cpu_queue_domain: tuple[ActionValue, ...]
    cpu_frequency_domain: tuple[ActionValue, ...]
    route_mask: np.ndarray
    tx_select_mask: np.ndarray
    cpu_queue_mask: np.ndarray
    power_energy_mask: np.ndarray
    cpu_frequency_energy_mask: np.ndarray
    route_branch_active: bool
    tx_branch_active: bool
    cpu_queue_branch_active: bool
    resource_group_count: int
    tx_idle_action: str
    cpu_idle_action: str
    canonical_width: int
    canonical_power: float
    canonical_cpu_frequency: float
    # Optional exact slot-start metadata, never a neural candidate feature.
    # Source order matches cpu_queue_domain[1:]; entries are cycles, inclusive
    # slack, fmax. Absent on historical/off observations and checkpoints.
    cpu_deadline_context: tuple[tuple[float, int, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.paper_uav_id != self.uav_id + 1:
            raise ObservationError("paper_uav_id must equal zero-based uav_id + 1")
        if self.sampling_order != BRANCH_ORDER:
            raise ObservationError("sampling_order must match the frozen seven-branch order")
        expected = {
            "route_mask": len(self.route_domain),
            "tx_select_mask": len(self.tx_select_domain),
            "cpu_queue_mask": len(self.cpu_queue_domain),
            "power_energy_mask": len(self.power_level_domain),
        }
        for name, size in expected.items():
            values = np.asarray(getattr(self, name), dtype=np.bool_)
            if values.shape != (size,) or not np.any(values):
                raise ObservationError(f"{name} must have shape ({size},) with a valid fallback")
            object.__setattr__(self, name, _readonly(values, np.bool_))
        cpu_energy = np.asarray(self.cpu_frequency_energy_mask, dtype=np.bool_)
        if cpu_energy.shape != (len(self.cpu_queue_domain) - 1, len(self.cpu_frequency_domain)):
            raise ObservationError("cpu_frequency_energy_mask has an invalid shape")
        if np.any(~np.any(cpu_energy, axis=1)):
            raise ObservationError("each CPU source must retain a frequency fallback")
        object.__setattr__(self, "cpu_frequency_energy_mask", _readonly(cpu_energy, np.bool_))
        if self.cpu_deadline_context is not None:
            context = tuple(tuple(entry) for entry in self.cpu_deadline_context)
            if len(context) != len(self.cpu_queue_domain) - 1:
                raise ObservationError("CPU deadline context must cover every source")
            for entry in context:
                if (len(entry) != 3 or not np.isfinite(entry[0]) or entry[0] < 0
                        or isinstance(entry[1], bool) or not isinstance(entry[1], int)
                        or not np.isfinite(entry[2]) or entry[2] <= 0):
                    raise ObservationError("invalid exact CPU deadline context")
            object.__setattr__(self, "cpu_deadline_context", context)

    @property
    def domains(self) -> dict[str, tuple[ActionValue, ...]]:
        return {
            "route": self.route_domain,
            "tx_select": self.tx_select_domain,
            "resource_group": self.resource_group_domain,
            "resource_width": self.resource_width_domain,
            "power_level": self.power_level_domain,
            "cpu_queue": self.cpu_queue_domain,
            "cpu_frequency": self.cpu_frequency_domain,
        }

    def domain_for(self, branch: str) -> tuple[ActionValue, ...]:
        try:
            return self.domains[branch]
        except KeyError as exc:
            raise ObservationError(f"unknown action branch {branch!r}") from exc

    def mask_for(
        self,
        branch: str,
        previous: Mapping[str, ActionValue] | ActionProposal | None = None,
    ) -> np.ndarray:
        """Return one read-only mask conditioned only on legal earlier branches."""

        context = self._context(previous)
        if branch == "route":
            return _readonly(self.route_mask, np.bool_)
        if branch == "tx_select":
            return _readonly(self.tx_select_mask, np.bool_)
        if branch == "resource_group":
            mask = np.zeros(len(self.resource_group_domain), dtype=np.bool_)
            tx = self._normalise("tx_select", context.get("tx_select", self.tx_idle_action))
            if tx == self.tx_idle_action or not self._masked_value_is_valid(
                self.tx_select_domain, self.tx_select_mask, tx
            ):
                mask[self.resource_group_domain.index("idle")] = True
            else:
                mask[1:] = True
            return _readonly(mask, np.bool_)
        if branch == "resource_width":
            mask = np.zeros(len(self.resource_width_domain), dtype=np.bool_)
            tx = self._normalise("tx_select", context.get("tx_select", self.tx_idle_action))
            group = self._normalise("resource_group", context.get("resource_group", "idle"))
            if tx == self.tx_idle_action or group == "idle" or not isinstance(group, int):
                mask[self.resource_width_domain.index(self.canonical_width)] = True
            else:
                for index, width in enumerate(self.resource_width_domain):
                    mask[index] = isinstance(width, int) and group + width - 1 <= self.resource_group_count
            return _readonly(mask, np.bool_)
        if branch == "power_level":
            mask = np.zeros(len(self.power_level_domain), dtype=np.bool_)
            tx = self._normalise("tx_select", context.get("tx_select", self.tx_idle_action))
            group = self._normalise("resource_group", context.get("resource_group", "idle"))
            if tx == self.tx_idle_action or group == "idle":
                mask[self.power_level_domain.index(self.canonical_power)] = True
            else:
                mask[:] = self.power_energy_mask
                positive_power_available = any(
                    bool(mask[index]) and float(level) > 0.0
                    for index, level in enumerate(self.power_level_domain)
                )
                if positive_power_available:
                    zero_power_index = self._domain_index(
                        self.power_level_domain,
                        0.0,
                    )
                    if zero_power_index is None:
                        raise ObservationError(
                            "power_level_domain must contain the zero-power fail-safe"
                        )
                    mask[zero_power_index] = False
            return _readonly(mask, np.bool_)
        if branch == "cpu_queue":
            return _readonly(self.cpu_queue_mask, np.bool_)
        if branch == "cpu_frequency":
            mask = np.zeros(len(self.cpu_frequency_domain), dtype=np.bool_)
            queue = self._normalise("cpu_queue", context.get("cpu_queue", self.cpu_idle_action))
            if queue == self.cpu_idle_action:
                mask[self.cpu_frequency_domain.index(self.canonical_cpu_frequency)] = True
            else:
                queue_index = self._domain_index(self.cpu_queue_domain, queue)
                if queue_index is None or queue_index == 0 or not self.cpu_queue_mask[queue_index]:
                    mask[self.cpu_frequency_domain.index(self.canonical_cpu_frequency)] = True
                else:
                    mask[:] = self.cpu_frequency_energy_mask[queue_index - 1]
            return _readonly(mask, np.bool_)
        raise ObservationError(f"unknown action branch {branch!r}")

    def branch_masks(self, proposal: ActionProposal) -> tuple[np.ndarray, ...]:
        """Return all seven conditional masks for a complete proposal."""

        return tuple(self.mask_for(branch, proposal) for branch in self.sampling_order)

    def active_indicators(self, proposal: ActionProposal) -> tuple[bool, ...]:
        """Return the frozen branch-activity indicators for one proposal."""

        tx = self._normalise("tx_select", proposal.tx_select)
        group = self._normalise("resource_group", proposal.resource_group)
        cpu_queue = self._normalise("cpu_queue", proposal.cpu_queue)
        communication_parent = tx != self.tx_idle_action
        communication_resources = communication_parent and group != "idle"
        return (
            bool(self.route_branch_active),
            bool(self.tx_branch_active),
            bool(communication_parent),
            bool(communication_resources),
            bool(communication_resources),
            bool(self.cpu_queue_branch_active),
            bool(cpu_queue != self.cpu_idle_action),
        )

    def is_legal(self, proposal: ActionProposal) -> bool:
        """Check a full proposal against domains and sequential masks."""

        if not isinstance(proposal, ActionProposal):
            return False
        if isinstance(proposal.uav_id, bool) or not isinstance(proposal.uav_id, int):
            return False
        if proposal.uav_id != self.uav_id:
            return False
        context: dict[str, ActionValue] = {}
        for branch, raw_value in zip(self.sampling_order, proposal.branches):
            try:
                value = self._normalise(branch, raw_value)
                domain = self.domain_for(branch)
                index = self._domain_index(domain, value)
                if index is None or not bool(self.mask_for(branch, context)[index]):
                    return False
                context[branch] = value
            except (ObservationError, TypeError, ValueError):
                return False
        return True

    def snapshot(self, proposal: ActionProposal | None = None) -> dict[str, Any]:
        context: Mapping[str, ActionValue] | ActionProposal | None = proposal
        result: dict[str, Any] = {
            "uav_id": self.uav_id,
            "paper_uav_id": self.paper_uav_id,
            "sampling_order": list(self.sampling_order),
            "domains": {name: list(domain) for name, domain in self.domains.items()},
            "masks": {
                branch: self.mask_for(branch, context).tolist()
                for branch in self.sampling_order
            },
        }
        if proposal is not None:
            result["active_indicators"] = list(self.active_indicators(proposal))
            result["proposal_is_legal"] = self.is_legal(proposal)
        return result

    def _context(
        self,
        previous: Mapping[str, ActionValue] | ActionProposal | None,
    ) -> dict[str, ActionValue]:
        if previous is None:
            return {}
        if isinstance(previous, ActionProposal):
            return dict(zip(self.sampling_order, previous.branches))
        return dict(previous)

    def _normalise(self, branch: str, value: ActionValue) -> ActionValue:
        if isinstance(value, bool):
            raise ObservationError(f"boolean is not a valid {branch} action")
        if branch == "cpu_queue" and value == "local":
            return self.uav_id
        if branch in {"power_level", "cpu_frequency"}:
            if not isinstance(value, (int, float)):
                raise ObservationError(f"{branch} must be numeric")
            return float(value)
        if branch == "resource_width":
            if not isinstance(value, int):
                raise ObservationError("resource_width must be an integer")
            return value
        return value

    @staticmethod
    def _domain_index(domain: tuple[ActionValue, ...], value: ActionValue) -> int | None:
        for index, candidate in enumerate(domain):
            if isinstance(candidate, float) and isinstance(value, (int, float)) and not isinstance(value, bool):
                if candidate == float(value):
                    return index
            elif type(candidate) is type(value) and candidate == value:
                return index
        return None

    @classmethod
    def _masked_value_is_valid(
        cls,
        domain: tuple[ActionValue, ...],
        mask: np.ndarray,
        value: ActionValue,
    ) -> bool:
        index = cls._domain_index(domain, value)
        return index is not None and bool(mask[index])


@dataclass(frozen=True)
class ActorObservation:
    """One UAV's complete slot-start actor-safe observation snapshot."""

    slot: int
    uav_id: int
    paper_uav_id: int
    self_resources: SelfResourceFeatures
    private_queues: PrivateQueueFeatures
    arrival_rate_estimate: float
    arrival_rate_valid_mask: bool
    history_source_slot: int
    candidate_neighbor_mask: np.ndarray
    service_queue_available: np.ndarray
    neighbor_public: NeighborPublicFeatures
    edge_history: ActorEdgeFeatures
    previous_action: PreviousActionFeatures
    action_masks: ActionMasks

    def __post_init__(self) -> None:
        if isinstance(self.slot, bool) or not isinstance(self.slot, int) or self.slot < 0:
            raise ObservationError("observation slot must be a non-negative integer")
        if self.paper_uav_id != self.uav_id + 1:
            raise ObservationError("paper_uav_id must equal zero-based uav_id + 1")
        candidate = np.asarray(self.candidate_neighbor_mask, dtype=np.bool_)
        service = np.asarray(self.service_queue_available, dtype=np.bool_)
        if candidate.ndim != 1 or service.shape != candidate.shape:
            raise ObservationError("candidate and service masks must be equal-length vectors")
        if not 0 <= self.uav_id < candidate.shape[0]:
            raise ObservationError("uav_id is outside the observation")
        if candidate[self.uav_id] or service[self.uav_id]:
            raise ObservationError("self cannot be a candidate neighbor or TX service destination")
        if self.private_queues.tx_by_destination.count != candidate.shape[0]:
            raise ObservationError("private queue feature count must match the UAV count")
        if self.action_masks.uav_id != self.uav_id:
            raise ObservationError("action masks belong to a different UAV")
        if (
            self.previous_action.uav_id != self.uav_id
            or self.previous_action.source_slot != self.slot - 1
            or self.previous_action.valid != (self.slot > 0)
        ):
            raise ObservationError("previous-action features are not causal for this actor slot")
        _finite_nonnegative(self.arrival_rate_estimate, "arrival_rate_estimate")
        if not self.arrival_rate_valid_mask and self.arrival_rate_estimate != 0.0:
            raise ObservationError("unavailable arrival estimate must use the zero placeholder")
        if (
            isinstance(self.history_source_slot, bool)
            or not isinstance(self.history_source_slot, int)
            or self.history_source_slot != self.slot - 1
        ):
            raise ObservationError("historical actor features require source_slot=slot-1")
        object.__setattr__(self, "candidate_neighbor_mask", _readonly(candidate, np.bool_))
        object.__setattr__(self, "service_queue_available", _readonly(service, np.bool_))

    def snapshot(self, proposal: ActionProposal | None = None) -> dict[str, Any]:
        """Return a deterministic JSON-safe snapshot for Gate 0 comparison."""

        return {
            "slot": self.slot,
            "uav_id": self.uav_id,
            "paper_uav_id": self.paper_uav_id,
            "self_resources": self.self_resources.snapshot(),
            "private_queues": self.private_queues.snapshot(),
            "arrival_rate_estimate": self.arrival_rate_estimate,
            "history_source_slot": self.history_source_slot,
            "arrival_rate_valid_mask": self.arrival_rate_valid_mask,
            "candidate_neighbor_mask": self.candidate_neighbor_mask.tolist(),
            "service_queue_available": self.service_queue_available.tolist(),
            "neighbor_public": self.neighbor_public.snapshot(),
            "edge_history": self.edge_history.snapshot(),
            "previous_action": self.previous_action.snapshot(),
            "action_masks": self.action_masks.snapshot(proposal),
        }


class ObservationBuilder:
    """Build deterministic actor observations from already-created slot state."""

    def __init__(self, environment: EnvironmentConfig, action: ActionConfig,
                 *, cpu_deadline_context_enabled: bool = False) -> None:
        self.environment = environment
        self.action = action
        self.cpu_deadline_context_enabled = cpu_deadline_context_enabled
        if environment.uav_count <= 0 or environment.ru_count <= 0:
            raise ObservationError("configured UAV and RU counts must be positive")
        if action.sampling_order != BRANCH_ORDER:
            raise ObservationError("ActionConfig sampling order differs from the frozen order")
        if action.resource_group_count != environment.resource_group_count:
            raise ObservationError("action/environment resource-group counts differ")

    @classmethod
    def from_run_config(cls, config: RunConfig) -> "ObservationBuilder":
        return cls(config.environment, config.action, cpu_deadline_context_enabled=(
            config.training.mappo.cpu_anti_starvation_mode == "deadline_feasibility_v1"))

    def build_all(
        self,
        *,
        slot: int,
        mobility: MobilityState,
        topology: TopologySnapshot,
        lifecycle: LifecycleManager,
        resource_states: Mapping[int, UavEnergyState],
        initial_energy_j: Mapping[int, float],
        channel_features: ActorChannelFeatures,
        public_messages: PublicMessageSnapshot,
        previous_actions: PreviousActionSnapshot,
        history_source_slot: int,
        arrival_rate_estimates: Sequence[float] | np.ndarray | None = None,
        arrival_rate_valid_mask: Sequence[bool] | np.ndarray | None = None,
        last_effective_rate_bps: np.ndarray | None = None,
        last_rate_valid_mask: np.ndarray | None = None,
        outage_rate: np.ndarray | None = None,
        outage_valid_mask: np.ndarray | None = None,
    ) -> tuple[ActorObservation, ...]:
        """Build observations in stable zero-based UAV-ID order."""

        return tuple(
            self.build(
                slot=slot,
                uav_id=uav_id,
                mobility=mobility,
                topology=topology,
                lifecycle=lifecycle,
                resource_states=resource_states,
                initial_energy_j=initial_energy_j,
                channel_features=channel_features,
                public_messages=public_messages,
                previous_actions=previous_actions,
                history_source_slot=history_source_slot,
                arrival_rate_estimates=arrival_rate_estimates,
                arrival_rate_valid_mask=arrival_rate_valid_mask,
                last_effective_rate_bps=last_effective_rate_bps,
                last_rate_valid_mask=last_rate_valid_mask,
                outage_rate=outage_rate,
                outage_valid_mask=outage_valid_mask,
            )
            for uav_id in range(self.environment.uav_count)
        )

    def build(
        self,
        *,
        slot: int,
        uav_id: int,
        mobility: MobilityState,
        topology: TopologySnapshot,
        lifecycle: LifecycleManager,
        resource_states: Mapping[int, UavEnergyState],
        initial_energy_j: Mapping[int, float],
        channel_features: ActorChannelFeatures,
        public_messages: PublicMessageSnapshot,
        previous_actions: PreviousActionSnapshot,
        history_source_slot: int,
        arrival_rate_estimates: Sequence[float] | np.ndarray | None = None,
        arrival_rate_valid_mask: Sequence[bool] | np.ndarray | None = None,
        last_effective_rate_bps: np.ndarray | None = None,
        last_rate_valid_mask: np.ndarray | None = None,
        outage_rate: np.ndarray | None = None,
        outage_valid_mask: np.ndarray | None = None,
    ) -> ActorObservation:
        self._validate_slot_inputs(
            slot,
            uav_id,
            mobility,
            topology,
            lifecycle,
            resource_states,
            initial_energy_j,
            channel_features,
            public_messages,
            previous_actions,
        )
        count = self.environment.uav_count
        arrival_values, arrival_valid = _paired_vector(
            arrival_rate_estimates,
            arrival_rate_valid_mask,
            count,
            "arrival rate",
        )
        rate_values, rate_valid = _paired_matrix(
            last_effective_rate_bps,
            last_rate_valid_mask,
            count,
            "last effective rate",
        )
        outage_values, outage_valid = _paired_matrix(
            outage_rate,
            outage_valid_mask,
            count,
            "outage rate",
            upper_bound=1.0,
        )
        if (
            isinstance(history_source_slot, bool)
            or not isinstance(history_source_slot, int)
            or history_source_slot != slot - 1
        ):
            raise ObservationError("arrival/rate/outage history must come from slot-1")
        if slot == 0 and (np.any(arrival_valid) or np.any(rate_valid) or np.any(outage_valid)):
            raise ObservationError("slot 0 cannot contain ended-slot arrival/rate/outage samples")

        private_queues, service_available, cpu_queue_available, cpu_head_cycles = self._private_queues(
            lifecycle, uav_id, slot
        )
        candidate = np.array(topology.candidate_neighbors[uav_id], dtype=np.bool_, copy=True)
        candidate[uav_id] = False
        edge_visible = candidate | service_available
        self_features = self._self_resources(
            resource_states[uav_id], float(initial_energy_j[uav_id])
        )
        neighbor_public = self._neighbor_public(
            uav_id,
            candidate,
            mobility,
            public_messages,
        )
        edge_history = self._edge_history(
            uav_id,
            edge_visible,
            mobility,
            public_messages,
            channel_features,
            rate_values[uav_id],
            rate_valid[uav_id],
            outage_values[uav_id],
            outage_valid[uav_id],
        )
        route_queue = lifecycle.queues.unbound.get(uav_id)
        route_available = bool(
            route_queue is not None
            and len(route_queue) > 0
            and route_queue.peek() is not None
            and route_queue.peek().can_route(slot)
        )
        action_masks = self._action_masks(
            uav_id=uav_id,
            candidate_neighbors=candidate,
            route_available=route_available,
            service_queue_available=service_available,
            cpu_queue_available=cpu_queue_available,
            cpu_head_remaining_cycles=cpu_head_cycles,
            resources=resource_states[uav_id],
        )
        if self.cpu_deadline_context_enabled:
            # Both local and remote EDF heads are already in the actor's exact
            # private observation. Do not reconstruct them from float32 tensors.
            from dataclasses import replace
            summaries = private_queues.cpu_by_source
            action_masks = replace(action_masks, cpu_deadline_context=tuple(
                (float(summaries.head_remaining_cycles[source]),
                 int(summaries.head_slack_slots[source]),
                 float(resource_states[uav_id].max_cpu_frequency_hz))
                for source in range(self.environment.uav_count)))
        return ActorObservation(
            slot=slot,
            uav_id=uav_id,
            paper_uav_id=uav_id + 1,
            self_resources=self_features,
            private_queues=private_queues,
            arrival_rate_estimate=float(arrival_values[uav_id]),
            arrival_rate_valid_mask=bool(arrival_valid[uav_id]),
            history_source_slot=history_source_slot,
            candidate_neighbor_mask=candidate,
            service_queue_available=service_available,
            neighbor_public=neighbor_public,
            edge_history=edge_history,
            previous_action=previous_actions.for_uav(uav_id),
            action_masks=action_masks,
        )

    def _validate_slot_inputs(
        self,
        slot: int,
        uav_id: int,
        mobility: MobilityState,
        topology: TopologySnapshot,
        lifecycle: LifecycleManager,
        resource_states: Mapping[int, UavEnergyState],
        initial_energy_j: Mapping[int, float],
        channel_features: ActorChannelFeatures,
        public_messages: PublicMessageSnapshot,
        previous_actions: PreviousActionSnapshot,
    ) -> None:
        count = self.environment.uav_count
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise ObservationError("slot must be a non-negative integer")
        if slot >= self.environment.episode_horizon:
            raise ObservationError("actor observation cannot be built at or beyond the horizon")
        if isinstance(uav_id, bool) or not isinstance(uav_id, int) or not 0 <= uav_id < count:
            raise ObservationError("uav_id is outside the configured zero-based range")
        if mobility.slot != slot or topology.slot != slot or channel_features.slot != slot:
            raise ObservationError("mobility, topology and channel history must match the actor slot")
        if previous_actions.actor_slot != slot or len(previous_actions.features) != count:
            raise ObservationError(
                "previous-action snapshot must match the actor slot and UAV count"
            )
        if public_messages.slot != slot:
            raise ObservationError("public messages must match the actor slot")
        if public_messages.valid_mask.shape != (count,):
            raise ObservationError("public-message UAV count differs from the configuration")
        if not np.array_equal(
            public_messages.valid_mask,
            channel_features.message_aoi_valid_mask,
        ) or not np.array_equal(
            public_messages.aoi_slots,
            channel_features.message_aoi_slots,
        ):
            raise ObservationError("public payload validity/AoI differs from causal message history")
        if mobility.positions_m.shape != (count, 3) or topology.distances_m.shape != (count, count):
            raise ObservationError("mobility/topology shapes do not match the configured UAV count")
        expected_distances = pairwise_distances(mobility.positions_m)
        if not np.allclose(
            topology.distances_m,
            expected_distances,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ObservationError("topology distances do not match actor-slot mobility")
        expected_candidates = candidate_neighbor_mask(
            mobility.positions_m,
            self.environment.candidate_neighbor_radius_m,
        )
        if not np.array_equal(topology.candidate_neighbors, expected_candidates):
            raise ObservationError("candidate graph does not match actor-slot mobility")
        if channel_features.stale_csi.shape != (count, count, self.environment.ru_count):
            raise ObservationError("actor channel history shape does not match the configuration")
        expected_ids = set(range(count))
        if set(resource_states) != expected_ids or set(initial_energy_j) != expected_ids:
            raise ObservationError("resource and initial-energy mappings must contain every UAV exactly once")
        for identifier in range(count):
            state = resource_states[identifier]
            if state.uav_id != identifier:
                raise ObservationError("resource-state mapping key differs from state.uav_id")
            initial = _finite_nonnegative(initial_energy_j[identifier], "initial_energy_j", positive=True)
            if state.residual_energy_j > initial + self.environment.energy_tolerance_j:
                raise ObservationError("residual energy cannot exceed the episode initial energy")
        lifecycle.validate_invariants()

    def _self_resources(self, state: UavEnergyState, initial_energy_j: float) -> SelfResourceFeatures:
        return SelfResourceFeatures(
            residual_energy_j=float(state.residual_energy_j),
            initial_energy_j=initial_energy_j,
            residual_energy_ratio=float(state.residual_energy_j / initial_energy_j),
            max_transmit_power_w=float(state.max_transmit_power_w),
            max_transmit_power_ratio=float(
                state.max_transmit_power_w / self.environment.reference_transmit_power_w
            ),
            max_cpu_frequency_hz=float(state.max_cpu_frequency_hz),
            max_cpu_frequency_ratio=float(
                state.max_cpu_frequency_hz / self.environment.reference_cpu_frequency_hz
            ),
            cpu_coefficient=float(state.cpu_coefficient),
            cpu_coefficient_ratio=float(
                state.cpu_coefficient / self.environment.reference_cpu_coefficient
            ),
        )

    def _private_queues(
        self,
        lifecycle: LifecycleManager,
        uav_id: int,
        slot: int,
    ) -> tuple[PrivateQueueFeatures, np.ndarray, np.ndarray, np.ndarray]:
        count = self.environment.uav_count
        unbound = self._queue_summary(lifecycle.queues.unbound.get(uav_id), slot)
        local = self._queue_summary(lifecycle.queues.local.get(uav_id), slot)
        tx_summaries: list[QueueSummary] = []
        cpu_summaries: list[QueueSummary] = []
        service_available = np.zeros(count, dtype=np.bool_)
        cpu_available = np.zeros(count, dtype=np.bool_)
        cpu_head_cycles = np.zeros(count, dtype=np.float64)
        for other in range(count):
            tx_queue = None if other == uav_id else lifecycle.queues.tx.get((uav_id, other))
            tx_summary = self._queue_summary(tx_queue, slot)
            tx_summaries.append(tx_summary)
            if tx_queue is not None and len(tx_queue) > 0:
                head = tx_queue.peek()
                service_available[other] = bool(head is not None and head.can_service(slot))

            cpu_queue = (
                lifecycle.queues.local.get(uav_id)
                if other == uav_id
                else lifecycle.queues.cpu.get((other, uav_id))
            )
            cpu_summary = self._queue_summary(cpu_queue, slot)
            cpu_summaries.append(cpu_summary)
            if cpu_queue is not None and len(cpu_queue) > 0:
                head = cpu_queue.peek()
                cpu_available[other] = bool(head is not None and head.can_service(slot))
                if cpu_available[other] and head is not None:
                    cpu_head_cycles[other] = float(head.remaining_cycles)
        return (
            PrivateQueueFeatures(
                unbound=unbound,
                local_cpu=local,
                tx_by_destination=self._indexed_queue_features(tx_summaries),
                cpu_by_source=self._indexed_queue_features(cpu_summaries),
            ),
            service_available,
            cpu_available,
            cpu_head_cycles,
        )

    @staticmethod
    def _queue_summary(queue: TaskQueue | None, slot: int) -> QueueSummary:
        if queue is None or len(queue) == 0:
            return QueueSummary(0, 0.0, 0.0, -1, 0.0, 0.0, 0, False)
        tasks = queue.items
        head = tasks[0]
        return QueueSummary(
            task_count=len(tasks),
            remaining_bits=float(sum(float(task.remaining_bits) for task in tasks)),
            remaining_cycles=float(sum(float(task.remaining_cycles) for task in tasks)),
            head_task_id=head.task_id,
            head_remaining_bits=float(head.remaining_bits),
            head_remaining_cycles=float(head.remaining_cycles),
            head_slack_slots=head.deadline_slack(slot),
            head_valid_mask=True,
        )

    @staticmethod
    def _indexed_queue_features(summaries: Sequence[QueueSummary]) -> IndexedQueueFeatures:
        return IndexedQueueFeatures(
            task_count=[item.task_count for item in summaries],
            remaining_bits=[item.remaining_bits for item in summaries],
            remaining_cycles=[item.remaining_cycles for item in summaries],
            head_task_id=[item.head_task_id for item in summaries],
            head_remaining_bits=[item.head_remaining_bits for item in summaries],
            head_remaining_cycles=[item.head_remaining_cycles for item in summaries],
            head_slack_slots=[item.head_slack_slots for item in summaries],
            head_valid_mask=[item.head_valid_mask for item in summaries],
        )

    def _neighbor_public(
        self,
        uav_id: int,
        candidate: np.ndarray,
        mobility: MobilityState,
        public_messages: PublicMessageSnapshot,
    ) -> NeighborPublicFeatures:
        count = self.environment.uav_count
        positions = np.zeros((count, 3), dtype=np.float64)
        velocities = np.zeros((count, 3), dtype=np.float64)
        uav_ids = np.full(count, -1, dtype=np.int64)
        paper_ids = np.zeros(count, dtype=np.int64)
        residual = np.zeros(count, dtype=np.float64)
        energy_ratio = np.zeros(count, dtype=np.float64)
        power_ratio = np.zeros(count, dtype=np.float64)
        cpu_ratio = np.zeros(count, dtype=np.float64)
        coefficient_ratio = np.zeros(count, dtype=np.float64)
        cpu_count = np.zeros(count, dtype=np.int64)
        cpu_cycles = np.zeros(count, dtype=np.float64)
        message_source = np.full(count, -1, dtype=np.int64)
        message_aoi = np.full(count, -1, dtype=np.int64)
        message_valid = candidate & public_messages.valid_mask
        for neighbor in np.flatnonzero(message_valid):
            neighbor_id = int(neighbor)
            uav_ids[neighbor_id] = neighbor_id
            paper_ids[neighbor_id] = neighbor_id + 1
            positions[neighbor_id] = public_messages.positions_m[neighbor_id] - mobility.positions_m[uav_id]
            velocities[neighbor_id] = public_messages.velocities_mps[neighbor_id] - mobility.velocities_mps[uav_id]
            residual[neighbor_id] = public_messages.residual_energy_j[neighbor_id]
            energy_ratio[neighbor_id] = public_messages.residual_energy_ratio[neighbor_id]
            power_ratio[neighbor_id] = public_messages.max_transmit_power_ratio[neighbor_id]
            cpu_ratio[neighbor_id] = public_messages.max_cpu_frequency_ratio[neighbor_id]
            coefficient_ratio[neighbor_id] = public_messages.cpu_coefficient_ratio[neighbor_id]
            cpu_count[neighbor_id] = public_messages.cpu_load_task_count[neighbor_id]
            cpu_cycles[neighbor_id] = public_messages.cpu_load_remaining_cycles[neighbor_id]
            message_source[neighbor_id] = public_messages.source_slots[neighbor_id]
            message_aoi[neighbor_id] = public_messages.aoi_slots[neighbor_id]
        return NeighborPublicFeatures(
            valid_mask=message_valid,
            uav_ids=uav_ids,
            paper_uav_ids=paper_ids,
            relative_position_m=positions,
            relative_velocity_mps=velocities,
            residual_energy_j=residual,
            residual_energy_ratio=energy_ratio,
            max_transmit_power_ratio=power_ratio,
            max_cpu_frequency_ratio=cpu_ratio,
            cpu_coefficient_ratio=coefficient_ratio,
            cpu_load_task_count=cpu_count,
            cpu_load_remaining_cycles=cpu_cycles,
            message_source_slots=message_source,
            message_aoi_slots=message_aoi,
            message_aoi_valid_mask=message_valid,
        )

    def _edge_history(
        self,
        uav_id: int,
        visible: np.ndarray,
        mobility: MobilityState,
        public_messages: PublicMessageSnapshot,
        channel_features: ActorChannelFeatures,
        last_rate: np.ndarray,
        last_rate_valid: np.ndarray,
        outage_rate: np.ndarray,
        outage_valid: np.ndarray,
    ) -> ActorEdgeFeatures:
        distances = np.zeros(self.environment.uav_count, dtype=np.float64)
        distance_valid = visible & public_messages.valid_mask
        relative_position = public_messages.positions_m - mobility.positions_m[uav_id]
        distances[distance_valid] = np.linalg.norm(relative_position[distance_valid], axis=1)
        stale = np.array(channel_features.stale_csi[uav_id], dtype=np.complex128, copy=True)
        csi_valid = np.array(channel_features.csi_valid_mask[uav_id], dtype=np.bool_, copy=True) & visible
        stale[~csi_valid, :] = 0.0
        csi_aoi = np.array(channel_features.csi_aoi_slots[uav_id], dtype=np.int64, copy=True)
        csi_aoi[~visible] = 0
        interference = np.array(channel_features.interference_history_w, dtype=np.float64, copy=True)
        interference_valid = np.array(channel_features.interference_valid_mask, dtype=np.bool_, copy=True)
        interference_valid &= visible[:, None]
        interference[~interference_valid] = 0.0
        message_aoi = np.array(public_messages.aoi_slots, dtype=np.int64, copy=True)
        message_valid = np.array(public_messages.valid_mask, dtype=np.bool_, copy=True) & visible
        message_aoi[~message_valid] = -1
        quality = np.array(channel_features.historical_quality[uav_id], dtype=np.float64, copy=True)
        quality_valid = np.array(channel_features.quality_valid_mask[uav_id], dtype=np.bool_, copy=True)
        quality_valid &= visible[:, None]
        quality[~quality_valid] = 0.0
        rate_valid = np.array(last_rate_valid, dtype=np.bool_, copy=True) & visible
        rate = np.array(last_rate, dtype=np.float64, copy=True)
        rate[~rate_valid] = 0.0
        out_valid = np.array(outage_valid, dtype=np.bool_, copy=True) & visible
        out_rate = np.array(outage_rate, dtype=np.float64, copy=True)
        out_rate[~out_valid] = 0.0
        return ActorEdgeFeatures(
            visible_mask=visible,
            estimated_distance_m=distances,
            stale_csi=stale,
            csi_valid_mask=csi_valid,
            csi_aoi_slots=csi_aoi,
            interference_history_w=interference,
            interference_valid_mask=interference_valid,
            message_aoi_slots=message_aoi,
            message_aoi_valid_mask=message_valid,
            historical_quality=quality,
            quality_valid_mask=quality_valid,
            last_effective_rate_bps=rate,
            last_rate_valid_mask=rate_valid,
            outage_rate=out_rate,
            outage_valid_mask=out_valid,
        )

    def _action_masks(
        self,
        *,
        uav_id: int,
        candidate_neighbors: np.ndarray,
        route_available: bool,
        service_queue_available: np.ndarray,
        cpu_queue_available: np.ndarray,
        cpu_head_remaining_cycles: np.ndarray,
        resources: UavEnergyState,
    ) -> ActionMasks:
        count = self.environment.uav_count
        route_domain: tuple[ActionValue, ...] = (
            *self.action.route_fixed_actions,
            *(identifier for identifier in range(count) if identifier != uav_id),
        )
        tx_domain: tuple[ActionValue, ...] = (
            self.action.tx_select_idle_action,
            *(identifier for identifier in range(count) if identifier != uav_id),
        )
        group_domain: tuple[ActionValue, ...] = (
            "idle",
            *range(1, self.environment.resource_group_count + 1),
        )
        width_domain: tuple[ActionValue, ...] = tuple(self.action.resource_width_options)
        power_domain: tuple[ActionValue, ...] = tuple(float(item) for item in self.action.power_levels)
        cpu_queue_domain: tuple[ActionValue, ...] = (self.action.cpu_queue_idle_action, *range(count))
        cpu_frequency_domain: tuple[ActionValue, ...] = tuple(
            float(item) for item in self.action.cpu_frequency_levels
        )

        route_mask = np.zeros(len(route_domain), dtype=np.bool_)
        if not route_available:
            route_mask[route_domain.index(self.action.canonical_inactive_values["route"])] = True
        else:
            route_mask[route_domain.index("local")] = True
            route_mask[route_domain.index("defer")] = True
            for destination in np.flatnonzero(candidate_neighbors):
                index = ActionMasks._domain_index(route_domain, int(destination))
                if index is not None:
                    route_mask[index] = True

        tx_mask = np.zeros(len(tx_domain), dtype=np.bool_)
        tx_mask[0] = True
        for destination in np.flatnonzero(service_queue_available):
            index = ActionMasks._domain_index(tx_domain, int(destination))
            if index is not None:
                tx_mask[index] = True

        cpu_queue_mask = np.zeros(len(cpu_queue_domain), dtype=np.bool_)
        cpu_queue_mask[0] = True
        for source in np.flatnonzero(cpu_queue_available):
            cpu_queue_mask[int(source) + 1] = True

        power_energy_mask = np.zeros(len(power_domain), dtype=np.bool_)
        for index, level in enumerate(power_domain):
            reservation = float(level) * resources.max_transmit_power_w * self.environment.slot_duration_s
            power_energy_mask[index] = reservation <= (
                resources.residual_energy_j + self.environment.energy_tolerance_j
            )
        power_energy_mask[power_domain.index(float(self.action.canonical_inactive_values["power_level"]))] = True

        cpu_frequency_energy_mask = np.zeros(
            (count, len(cpu_frequency_domain)), dtype=np.bool_
        )
        for source in range(count):
            for index, level in enumerate(cpu_frequency_domain):
                frequency = float(level) * resources.max_cpu_frequency_hz
                reserved = cpu_reservation_j(
                    resources,
                    frequency,
                    float(cpu_head_remaining_cycles[source]),
                    self.environment.slot_duration_s,
                    active_candidate=bool(cpu_queue_available[source]),
                )
                cpu_frequency_energy_mask[source, index] = reserved <= (
                    resources.residual_energy_j + self.environment.energy_tolerance_j
                )
            zero_index = cpu_frequency_domain.index(
                float(self.action.canonical_inactive_values["cpu_frequency"])
            )
            cpu_frequency_energy_mask[source, zero_index] = True

        return ActionMasks(
            uav_id=uav_id,
            paper_uav_id=uav_id + 1,
            sampling_order=BRANCH_ORDER,
            route_domain=route_domain,
            tx_select_domain=tx_domain,
            resource_group_domain=group_domain,
            resource_width_domain=width_domain,
            power_level_domain=power_domain,
            cpu_queue_domain=cpu_queue_domain,
            cpu_frequency_domain=cpu_frequency_domain,
            route_mask=route_mask,
            tx_select_mask=tx_mask,
            cpu_queue_mask=cpu_queue_mask,
            power_energy_mask=power_energy_mask,
            cpu_frequency_energy_mask=cpu_frequency_energy_mask,
            route_branch_active=route_available,
            tx_branch_active=bool(np.any(service_queue_available)),
            cpu_queue_branch_active=bool(np.any(cpu_queue_available)),
            resource_group_count=self.environment.resource_group_count,
            tx_idle_action=self.action.tx_select_idle_action,
            cpu_idle_action=self.action.cpu_queue_idle_action,
            canonical_width=int(self.action.canonical_inactive_values["resource_width"]),
            canonical_power=float(self.action.canonical_inactive_values["power_level"]),
            canonical_cpu_frequency=float(self.action.canonical_inactive_values["cpu_frequency"]),
        )


ActorObservationBuilder = ObservationBuilder


__all__ = [
    "ActionMasks",
    "ActorEdgeFeatures",
    "ActorObservation",
    "ActorObservationBuilder",
    "BRANCH_ORDER",
    "IndexedQueueFeatures",
    "NeighborPublicFeatures",
    "ObservationBuilder",
    "ObservationError",
    "PrivateQueueFeatures",
    "QueueSummary",
    "SelfResourceFeatures",
]
