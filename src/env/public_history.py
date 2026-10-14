"""Causal owner of sanitized UAV public-message payload history.

Messages are created only from a completed environment slot.  A payload whose
``source_slot`` is ``t`` can therefore first appear in the actor snapshot for
slot ``t + 1`` with AoI equal to one.  This module owns no RNG and deliberately
stores neither task identifiers, queue heads, complete queues, nor any current
actor-slot physical outcome.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import EnvironmentConfig, RunConfig
from .energy import UavEnergyState
from .lifecycle import LifecycleManager
from .mobility import MobilityState


class PublicHistoryError(ValueError):
    """Raised when public-message state violates shape or causal boundaries."""


def _readonly(values: Any, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _positive_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublicHistoryError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise PublicHistoryError(f"{name} must be finite and positive")
    return converted


def _initial_energy_vector(
    values: Mapping[int, float] | Sequence[float] | np.ndarray,
    uav_count: int,
) -> np.ndarray:
    if isinstance(values, Mapping):
        expected = set(range(uav_count))
        if set(values) != expected:
            raise PublicHistoryError(
                "initial_energy_j mapping must contain every zero-based UAV ID exactly once"
            )
        converted = np.array([values[uav_id] for uav_id in range(uav_count)], dtype=np.float64)
    else:
        converted = np.asarray(values, dtype=np.float64)
        if converted.shape != (uav_count,):
            raise PublicHistoryError(
                f"initial_energy_j must have shape ({uav_count},)"
            )
        converted = np.array(converted, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(converted)) or np.any(converted <= 0.0):
        raise PublicHistoryError("initial energies must be finite and positive")
    return converted


@dataclass(frozen=True)
class PublicMessageSnapshot:
    """Actor-slot view of the latest sanitized message from every UAV.

    Python ``uav_ids`` are zero based.  ``paper_uav_ids`` expose the explicit
    one-based mapping used by the manuscript.  Invalid payloads use all-zero
    value placeholders, ``source_slots=-1`` and ``aoi_slots=-1``.
    """

    slot: int
    uav_ids: np.ndarray
    paper_uav_ids: np.ndarray
    source_slots: np.ndarray
    aoi_slots: np.ndarray
    valid_mask: np.ndarray
    positions_m: np.ndarray
    velocities_mps: np.ndarray
    residual_energy_j: np.ndarray
    residual_energy_ratio: np.ndarray
    max_transmit_power_ratio: np.ndarray
    max_cpu_frequency_ratio: np.ndarray
    cpu_coefficient_ratio: np.ndarray
    cpu_load_task_count: np.ndarray
    cpu_load_remaining_cycles: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.slot, bool) or not isinstance(self.slot, int) or self.slot < 0:
            raise PublicHistoryError("snapshot slot must be a non-negative integer")
        ids = np.asarray(self.uav_ids)
        if ids.ndim != 1 or ids.size == 0:
            raise PublicHistoryError("uav_ids must be a non-empty vector")
        count = ids.shape[0]
        expected_ids = np.arange(count, dtype=np.int64)
        if not np.array_equal(ids, expected_ids):
            raise PublicHistoryError("uav_ids must be in stable zero-based order")
        if not np.array_equal(np.asarray(self.paper_uav_ids), expected_ids + 1):
            raise PublicHistoryError("paper_uav_ids must equal uav_ids + 1")
        vector_names = (
            "source_slots",
            "aoi_slots",
            "valid_mask",
            "residual_energy_j",
            "residual_energy_ratio",
            "max_transmit_power_ratio",
            "max_cpu_frequency_ratio",
            "cpu_coefficient_ratio",
            "cpu_load_task_count",
            "cpu_load_remaining_cycles",
        )
        for name in vector_names:
            if np.asarray(getattr(self, name)).shape != (count,):
                raise PublicHistoryError(f"{name} must have shape ({count},)")
        for name in ("positions_m", "velocities_mps"):
            if np.asarray(getattr(self, name)).shape != (count, 3):
                raise PublicHistoryError(f"{name} must have shape ({count}, 3)")

        valid = np.asarray(self.valid_mask, dtype=np.bool_)
        source_slots = np.asarray(self.source_slots, dtype=np.int64)
        aoi_slots = np.asarray(self.aoi_slots, dtype=np.int64)
        if np.any(source_slots[valid] < 0) or np.any(source_slots[valid] >= self.slot):
            raise PublicHistoryError(
                "every valid public payload must originate before the actor slot"
            )
        if np.any(aoi_slots[valid] != self.slot - source_slots[valid]):
            raise PublicHistoryError("valid public-message AoI must equal slot - source_slot")
        if np.any(aoi_slots[valid] < 1):
            raise PublicHistoryError("the first visible public message must have AoI one")
        if np.any(source_slots[~valid] != -1) or np.any(aoi_slots[~valid] != -1):
            raise PublicHistoryError(
                "invalid public payloads require source_slot=-1 and aoi=-1"
            )

        float_names = (
            "positions_m",
            "velocities_mps",
            "residual_energy_j",
            "residual_energy_ratio",
            "max_transmit_power_ratio",
            "max_cpu_frequency_ratio",
            "cpu_coefficient_ratio",
            "cpu_load_remaining_cycles",
        )
        for name in float_names:
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise PublicHistoryError(f"{name} must contain only finite values")
        for name in (
            "residual_energy_j",
            "residual_energy_ratio",
            "max_transmit_power_ratio",
            "max_cpu_frequency_ratio",
            "cpu_coefficient_ratio",
            "cpu_load_remaining_cycles",
        ):
            if np.any(np.asarray(getattr(self, name), dtype=np.float64) < 0.0):
                raise PublicHistoryError(f"{name} must be non-negative")
        counts = np.asarray(self.cpu_load_task_count)
        if np.any(counts < 0):
            raise PublicHistoryError("CPU-load task counts must be non-negative")

        invalid = ~valid
        for name in (
            "positions_m",
            "velocities_mps",
            "residual_energy_j",
            "residual_energy_ratio",
            "max_transmit_power_ratio",
            "max_cpu_frequency_ratio",
            "cpu_coefficient_ratio",
            "cpu_load_task_count",
            "cpu_load_remaining_cycles",
        ):
            values = np.asarray(getattr(self, name))
            if np.any(values[invalid] != 0):
                raise PublicHistoryError(
                    f"invalid public payloads require canonical zero placeholders in {name}"
                )

        dtypes = {
            "uav_ids": np.int64,
            "paper_uav_ids": np.int64,
            "source_slots": np.int64,
            "aoi_slots": np.int64,
            "valid_mask": np.bool_,
            "positions_m": np.float64,
            "velocities_mps": np.float64,
            "residual_energy_j": np.float64,
            "residual_energy_ratio": np.float64,
            "max_transmit_power_ratio": np.float64,
            "max_cpu_frequency_ratio": np.float64,
            "cpu_coefficient_ratio": np.float64,
            "cpu_load_task_count": np.int64,
            "cpu_load_remaining_cycles": np.float64,
        }
        for name, dtype in dtypes.items():
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))

    @property
    def actor_slot(self) -> int:
        """Alias that makes the source-slot/actor-slot boundary explicit."""

        return self.slot

    @property
    def validity_mask(self) -> np.ndarray:
        """Compatibility alias for the payload-level validity mask."""

        return self.valid_mask

    def snapshot(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe representation for Gate 0 audits."""

        return {
            "slot": self.slot,
            "uav_ids": self.uav_ids.tolist(),
            "paper_uav_ids": self.paper_uav_ids.tolist(),
            "source_slots": self.source_slots.tolist(),
            "aoi_slots": self.aoi_slots.tolist(),
            "valid_mask": self.valid_mask.tolist(),
            "positions_m": self.positions_m.tolist(),
            "velocities_mps": self.velocities_mps.tolist(),
            "residual_energy_j": self.residual_energy_j.tolist(),
            "residual_energy_ratio": self.residual_energy_ratio.tolist(),
            "max_transmit_power_ratio": self.max_transmit_power_ratio.tolist(),
            "max_cpu_frequency_ratio": self.max_cpu_frequency_ratio.tolist(),
            "cpu_coefficient_ratio": self.cpu_coefficient_ratio.tolist(),
            "cpu_load_task_count": self.cpu_load_task_count.tolist(),
            "cpu_load_remaining_cycles": self.cpu_load_remaining_cycles.tolist(),
        }


class PublicMessageHistory:
    """Environment-owned causal public-message payload history.

    ``record_from_completed_slot`` is the only mutating slot operation.  It
    updates refreshed UAVs atomically and returns the first legal actor view,
    namely the snapshot for ``measurement_slot + 1``.
    """

    def __init__(
        self,
        environment: EnvironmentConfig,
        initial_energy_j: Mapping[int, float] | Sequence[float] | np.ndarray,
    ) -> None:
        self.environment = environment
        if isinstance(environment.uav_count, bool) or environment.uav_count <= 0:
            raise PublicHistoryError("configured UAV count must be positive")
        for name in (
            "reference_transmit_power_w",
            "reference_cpu_frequency_hz",
            "reference_cpu_coefficient",
        ):
            _positive_float(getattr(environment, name), name)
        self._initial_energy_j = _initial_energy_vector(
            initial_energy_j,
            environment.uav_count,
        )
        self.reset()

    @classmethod
    def from_run_config(
        cls,
        config: RunConfig,
        initial_energy_j: Mapping[int, float] | Sequence[float] | np.ndarray,
    ) -> "PublicMessageHistory":
        return cls(config.environment, initial_energy_j)

    @property
    def last_completed_slot(self) -> int:
        return self._last_completed_slot

    @property
    def initial_energy_j(self) -> np.ndarray:
        return _readonly(self._initial_energy_j, np.float64)

    def reset(
        self,
        initial_energy_j: Mapping[int, float] | Sequence[float] | np.ndarray | None = None,
    ) -> PublicMessageSnapshot:
        """Reset to canonical unavailable payloads and return actor slot 0."""

        count = self.environment.uav_count
        if initial_energy_j is not None:
            self._initial_energy_j = _initial_energy_vector(initial_energy_j, count)
        self._last_completed_slot = -1
        self._source_slots = np.full(count, -1, dtype=np.int64)
        self._valid_mask = np.zeros(count, dtype=np.bool_)
        self._positions_m = np.zeros((count, 3), dtype=np.float64)
        self._velocities_mps = np.zeros((count, 3), dtype=np.float64)
        self._residual_energy_j = np.zeros(count, dtype=np.float64)
        self._residual_energy_ratio = np.zeros(count, dtype=np.float64)
        self._max_transmit_power_ratio = np.zeros(count, dtype=np.float64)
        self._max_cpu_frequency_ratio = np.zeros(count, dtype=np.float64)
        self._cpu_coefficient_ratio = np.zeros(count, dtype=np.float64)
        self._cpu_load_task_count = np.zeros(count, dtype=np.int64)
        self._cpu_load_remaining_cycles = np.zeros(count, dtype=np.float64)
        return self.snapshot(0)

    def record_from_completed_slot(
        self,
        measurement_slot: int,
        refresh_mask: Sequence[bool] | np.ndarray,
        mobility: MobilityState,
        resource_states: Mapping[int, UavEnergyState],
        lifecycle: LifecycleManager,
    ) -> PublicMessageSnapshot:
        """Refresh selected UAV payloads from completed slot ``t``.

        ``resource_states`` and ``lifecycle`` must already be post-service and
        post-settlement for ``measurement_slot``.  Only aggregate CPU load is
        retained; task IDs, per-source remote queues and EDF heads are omitted.
        """

        count = self.environment.uav_count
        if (
            isinstance(measurement_slot, bool)
            or not isinstance(measurement_slot, int)
            or measurement_slot < 0
        ):
            raise PublicHistoryError("measurement_slot must be a non-negative integer")
        expected_slot = self._last_completed_slot + 1
        if measurement_slot != expected_slot:
            raise PublicHistoryError(
                f"completed slots must be recorded sequentially; expected {expected_slot}"
            )
        if mobility.slot != measurement_slot:
            raise PublicHistoryError(
                "mobility must describe the completed measurement slot"
            )
        if mobility.positions_m.shape != (count, 3):
            raise PublicHistoryError("mobility UAV count differs from the configuration")
        refresh = np.asarray(refresh_mask, dtype=np.bool_)
        if refresh.shape != (count,):
            raise PublicHistoryError(f"refresh_mask must have shape ({count},)")
        expected_ids = set(range(count))
        if set(resource_states) != expected_ids:
            raise PublicHistoryError(
                "resource_states must contain every zero-based UAV ID exactly once"
            )
        for uav_id in range(count):
            state = resource_states[uav_id]
            if state.uav_id != uav_id:
                raise PublicHistoryError(
                    "resource-state mapping key differs from state.uav_id"
                )
            if state.residual_energy_j > (
                self._initial_energy_j[uav_id] + self.environment.energy_tolerance_j
            ):
                raise PublicHistoryError(
                    "post-service residual energy exceeds episode initial energy"
                )
        lifecycle.validate_invariants()

        next_source_slots = self._source_slots.copy()
        next_valid = self._valid_mask.copy()
        next_positions = self._positions_m.copy()
        next_velocities = self._velocities_mps.copy()
        next_residual = self._residual_energy_j.copy()
        next_energy_ratio = self._residual_energy_ratio.copy()
        next_power_ratio = self._max_transmit_power_ratio.copy()
        next_cpu_ratio = self._max_cpu_frequency_ratio.copy()
        next_coefficient_ratio = self._cpu_coefficient_ratio.copy()
        next_cpu_count = self._cpu_load_task_count.copy()
        next_cpu_cycles = self._cpu_load_remaining_cycles.copy()

        for raw_uav_id in np.flatnonzero(refresh):
            uav_id = int(raw_uav_id)
            state = resource_states[uav_id]
            cpu_count, cpu_cycles = self._aggregate_cpu_load(lifecycle, uav_id)
            next_source_slots[uav_id] = measurement_slot
            next_valid[uav_id] = True
            next_positions[uav_id] = mobility.positions_m[uav_id]
            next_velocities[uav_id] = mobility.velocities_mps[uav_id]
            next_residual[uav_id] = state.residual_energy_j
            next_energy_ratio[uav_id] = (
                state.residual_energy_j / self._initial_energy_j[uav_id]
            )
            next_power_ratio[uav_id] = (
                state.max_transmit_power_w
                / self.environment.reference_transmit_power_w
            )
            next_cpu_ratio[uav_id] = (
                state.max_cpu_frequency_hz
                / self.environment.reference_cpu_frequency_hz
            )
            next_coefficient_ratio[uav_id] = (
                state.cpu_coefficient
                / self.environment.reference_cpu_coefficient
            )
            next_cpu_count[uav_id] = cpu_count
            next_cpu_cycles[uav_id] = cpu_cycles

        self._source_slots = next_source_slots
        self._valid_mask = next_valid
        self._positions_m = next_positions
        self._velocities_mps = next_velocities
        self._residual_energy_j = next_residual
        self._residual_energy_ratio = next_energy_ratio
        self._max_transmit_power_ratio = next_power_ratio
        self._max_cpu_frequency_ratio = next_cpu_ratio
        self._cpu_coefficient_ratio = next_coefficient_ratio
        self._cpu_load_task_count = next_cpu_count
        self._cpu_load_remaining_cycles = next_cpu_cycles
        self._last_completed_slot = measurement_slot
        return self.snapshot(measurement_slot + 1)

    def snapshot(self, actor_slot: int) -> PublicMessageSnapshot:
        """Return the only current causal actor view of the payload history."""

        if isinstance(actor_slot, bool) or not isinstance(actor_slot, int) or actor_slot < 0:
            raise PublicHistoryError("actor_slot must be a non-negative integer")
        expected_actor_slot = self._last_completed_slot + 1
        if actor_slot != expected_actor_slot:
            raise PublicHistoryError(
                f"public history currently belongs to actor slot {expected_actor_slot}, not {actor_slot}"
            )
        if np.any(self._source_slots[self._valid_mask] >= actor_slot):
            raise PublicHistoryError(
                "valid public payload source slots must be strictly earlier than actor_slot"
            )
        aoi = np.full(self.environment.uav_count, -1, dtype=np.int64)
        aoi[self._valid_mask] = actor_slot - self._source_slots[self._valid_mask]
        return PublicMessageSnapshot(
            slot=actor_slot,
            uav_ids=np.arange(self.environment.uav_count, dtype=np.int64),
            paper_uav_ids=np.arange(1, self.environment.uav_count + 1, dtype=np.int64),
            source_slots=self._source_slots,
            aoi_slots=aoi,
            valid_mask=self._valid_mask,
            positions_m=self._positions_m,
            velocities_mps=self._velocities_mps,
            residual_energy_j=self._residual_energy_j,
            residual_energy_ratio=self._residual_energy_ratio,
            max_transmit_power_ratio=self._max_transmit_power_ratio,
            max_cpu_frequency_ratio=self._max_cpu_frequency_ratio,
            cpu_coefficient_ratio=self._cpu_coefficient_ratio,
            cpu_load_task_count=self._cpu_load_task_count,
            cpu_load_remaining_cycles=self._cpu_load_remaining_cycles,
        )

    def _aggregate_cpu_load(
        self,
        lifecycle: LifecycleManager,
        executor_uav: int,
    ) -> tuple[int, float]:
        queues = []
        local = lifecycle.queues.local.get(executor_uav)
        if local is not None:
            queues.append(local)
        for source_uav in range(self.environment.uav_count):
            if source_uav == executor_uav:
                continue
            remote = lifecycle.queues.cpu.get((source_uav, executor_uav))
            if remote is not None:
                queues.append(remote)
        task_count = sum(len(queue) for queue in queues)
        remaining_cycles = sum(
            float(task.remaining_cycles)
            for queue in queues
            for task in queue.items
        )
        if not math.isfinite(remaining_cycles) or remaining_cycles < 0.0:
            raise PublicHistoryError("aggregate CPU load must be finite and non-negative")
        return task_count, remaining_cycles


__all__ = [
    "PublicHistoryError",
    "PublicMessageHistory",
    "PublicMessageSnapshot",
]
