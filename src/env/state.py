"""Immutable centralized pre-action state snapshots for CTDE training.

The builder in this module may read decision-time environment truth, including
the current physical channel and true arrival parameters.  Its API accepts no
future state and no post-action service, measurement, reward, or executor
result, which keeps the centralized state on the same pre-action boundary as
the actor observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..config import BuildingConfig, EnvironmentConfig, RunConfig
from .channel import ChannelSnapshot
from .energy import UavEnergyState
from .history import ActorChannelFeatures
from .lifecycle import LifecycleManager
from .mobility import MobilityState
from .queues import QueueKind, QueueLocation, QueueState
from .tasks import Task
from .topology import TopologySnapshot, candidate_neighbor_mask, pairwise_distances


class StateError(ValueError):
    """Raised when a centralized decision-time snapshot is inconsistent."""


def _readonly(values: Any, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _finite_nonnegative(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StateError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise StateError(f"{name} must be finite")
    if positive and converted <= 0.0:
        raise StateError(f"{name} must be positive")
    if not positive and converted < 0.0:
        raise StateError(f"{name} must be non-negative")
    return converted


@dataclass(frozen=True)
class TaskStateRecord:
    """A copied active-task record at the decision boundary."""

    task_id: int
    source_uav: int
    source_paper_uav_id: int
    data_bits: float
    cpu_cycles: float
    arrival_slot: int
    deadline_slot: int
    deadline_budget_slots: int
    remaining_bits: float
    remaining_cycles: float
    destination_uav: int | None
    destination_paper_uav_id: int | None
    status: str
    binding_slot: int | None
    service_eligible_slot: int | None
    cpu_entry_slot: int | None

    @classmethod
    def from_task(cls, task: Task) -> "TaskStateRecord":
        if task.is_terminal:
            raise StateError("centralized active-task state cannot contain a terminal task")
        return cls(
            task_id=task.task_id,
            source_uav=task.source_uav,
            source_paper_uav_id=task.source_uav + 1,
            data_bits=float(task.data_bits),
            cpu_cycles=float(task.cpu_cycles),
            arrival_slot=task.arrival_slot,
            deadline_slot=task.deadline_slot,
            deadline_budget_slots=int(task.deadline_budget_slots),
            remaining_bits=float(task.remaining_bits),
            remaining_cycles=float(task.remaining_cycles),
            destination_uav=task.destination,
            destination_paper_uav_id=(
                None if task.destination is None else task.destination + 1
            ),
            status=task.status.value,
            binding_slot=task.binding_slot,
            service_eligible_slot=task.service_eligible_slot,
            cpu_entry_slot=task.cpu_entry_slot,
        )

    def __post_init__(self) -> None:
        if self.source_paper_uav_id != self.source_uav + 1:
            raise StateError("source paper UAV ID must equal zero-based ID + 1")
        if self.destination_uav is None:
            if self.destination_paper_uav_id is not None:
                raise StateError("unbound destination cannot have a paper UAV ID")
        elif self.destination_paper_uav_id != self.destination_uav + 1:
            raise StateError("destination paper UAV ID must equal zero-based ID + 1")
        for name in ("data_bits", "cpu_cycles"):
            _finite_nonnegative(getattr(self, name), name, positive=True)
        for name in ("remaining_bits", "remaining_cycles"):
            _finite_nonnegative(getattr(self, name), name)

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source_uav": self.source_uav,
            "source_paper_uav_id": self.source_paper_uav_id,
            "data_bits": self.data_bits,
            "cpu_cycles": self.cpu_cycles,
            "arrival_slot": self.arrival_slot,
            "deadline_slot": self.deadline_slot,
            "deadline_budget_slots": self.deadline_budget_slots,
            "remaining_bits": self.remaining_bits,
            "remaining_cycles": self.remaining_cycles,
            "destination_uav": self.destination_uav,
            "destination_paper_uav_id": self.destination_paper_uav_id,
            "status": self.status,
            "binding_slot": self.binding_slot,
            "service_eligible_slot": self.service_eligible_slot,
            "cpu_entry_slot": self.cpu_entry_slot,
        }


@dataclass(frozen=True)
class QueueStateRecord:
    """One deterministic active queue and its ordered task identifiers."""

    kind: str
    source_uav: int
    source_paper_uav_id: int
    destination_uav: int | None
    destination_paper_uav_id: int | None
    task_ids: tuple[int, ...]

    @classmethod
    def from_queue(cls, location: QueueLocation, task_ids: tuple[int, ...]) -> "QueueStateRecord":
        destination = location.destination_uav
        return cls(
            kind=location.kind.value,
            source_uav=location.source_uav,
            source_paper_uav_id=location.source_uav + 1,
            destination_uav=destination,
            destination_paper_uav_id=None if destination is None else destination + 1,
            task_ids=tuple(task_ids),
        )

    def __post_init__(self) -> None:
        if self.kind not in {item.value for item in QueueKind}:
            raise StateError(f"unknown queue kind {self.kind!r}")
        if self.source_paper_uav_id != self.source_uav + 1:
            raise StateError("source paper UAV ID must equal zero-based ID + 1")
        if self.destination_uav is None:
            if self.destination_paper_uav_id is not None:
                raise StateError("queue without destination cannot have a paper destination ID")
        elif self.destination_paper_uav_id != self.destination_uav + 1:
            raise StateError("destination paper UAV ID must equal zero-based ID + 1")
        if any(isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0 for task_id in self.task_ids):
            raise StateError("queue task IDs must be non-negative integers")

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source_uav": self.source_uav,
            "source_paper_uav_id": self.source_paper_uav_id,
            "destination_uav": self.destination_uav,
            "destination_paper_uav_id": self.destination_paper_uav_id,
            "task_ids": list(self.task_ids),
        }


@dataclass(frozen=True)
class BuildingStateRecord:
    """Static building truth copied from the canonical environment config."""

    name: str
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    height_m: float

    @classmethod
    def from_config(cls, building: BuildingConfig) -> "BuildingStateRecord":
        return cls(
            name=building.name,
            x_min_m=float(building.x_min_m),
            x_max_m=float(building.x_max_m),
            y_min_m=float(building.y_min_m),
            y_max_m=float(building.y_max_m),
            height_m=float(building.height_m),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "x_min_m": self.x_min_m,
            "x_max_m": self.x_max_m,
            "y_min_m": self.y_min_m,
            "y_max_m": self.y_max_m,
            "height_m": self.height_m,
        }


@dataclass(frozen=True)
class CentralizedState:
    """One immutable decision-time global state for a centralized critic."""

    slot: int
    uav_ids: np.ndarray
    paper_uav_ids: np.ndarray
    positions_m: np.ndarray
    velocities_mps: np.ndarray
    max_transmit_power_w: np.ndarray
    max_cpu_frequency_hz: np.ndarray
    cpu_coefficient: np.ndarray
    residual_energy_j: np.ndarray
    true_arrival_probabilities: np.ndarray
    candidate_neighbor_mask: np.ndarray
    distances_m: np.ndarray
    true_channel: np.ndarray
    blocked_links: np.ndarray
    shadowing_db: np.ndarray
    path_loss_db: np.ndarray
    stale_csi: np.ndarray
    csi_valid_mask: np.ndarray
    csi_aoi_slots: np.ndarray
    interference_history_w: np.ndarray
    interference_valid_mask: np.ndarray
    message_aoi_slots: np.ndarray
    message_aoi_valid_mask: np.ndarray
    historical_denominator_w: np.ndarray
    historical_quality: np.ndarray
    quality_valid_mask: np.ndarray
    active_tasks: tuple[TaskStateRecord, ...]
    queues: tuple[QueueStateRecord, ...]
    buildings: tuple[BuildingStateRecord, ...]

    def __post_init__(self) -> None:
        if isinstance(self.slot, bool) or not isinstance(self.slot, int) or self.slot < 0:
            raise StateError("centralized state slot must be a non-negative integer")
        ids = np.asarray(self.uav_ids)
        if ids.ndim != 1 or ids.size == 0:
            raise StateError("uav_ids must be a non-empty vector")
        count = ids.shape[0]
        expected_ids = np.arange(count, dtype=np.int64)
        if not np.array_equal(ids, expected_ids):
            raise StateError("centralized state requires stable zero-based UAV ID order")
        if not np.array_equal(np.asarray(self.paper_uav_ids), expected_ids + 1):
            raise StateError("paper UAV IDs must equal zero-based IDs + 1")
        for name in ("positions_m", "velocities_mps"):
            if np.asarray(getattr(self, name)).shape != (count, 3):
                raise StateError(f"{name} must have shape ({count}, 3)")
        for name in (
            "max_transmit_power_w",
            "max_cpu_frequency_hz",
            "cpu_coefficient",
            "residual_energy_j",
            "true_arrival_probabilities",
        ):
            values = np.asarray(getattr(self, name))
            if values.shape != (count,) or not np.all(np.isfinite(values)):
                raise StateError(f"{name} must be a finite vector with shape ({count},)")
        link_shape = (count, count)
        for name in ("candidate_neighbor_mask", "distances_m", "blocked_links", "shadowing_db", "path_loss_db"):
            if np.asarray(getattr(self, name)).shape != link_shape:
                raise StateError(f"{name} must have shape {link_shape}")
        channel = np.asarray(self.true_channel)
        if channel.ndim != 3 or channel.shape[:2] != link_shape or not np.iscomplexobj(channel):
            raise StateError("true_channel must have shape (uav_count, uav_count, ru_count)")
        if not np.all(np.isfinite(channel)):
            raise StateError("true_channel must contain only finite values")
        stale_csi = np.asarray(self.stale_csi)
        if stale_csi.shape != channel.shape or not np.iscomplexobj(stale_csi):
            raise StateError("stale_csi must match the true-channel tensor shape")
        if not np.all(np.isfinite(stale_csi)):
            raise StateError("stale_csi must contain only finite values")
        ru_count = channel.shape[2]
        history_shapes = {
            "csi_valid_mask": link_shape,
            "csi_aoi_slots": link_shape,
            "interference_history_w": (count, ru_count),
            "interference_valid_mask": (count, ru_count),
            "message_aoi_slots": (count,),
            "message_aoi_valid_mask": (count,),
            "historical_denominator_w": (count, ru_count),
            "historical_quality": channel.shape,
            "quality_valid_mask": channel.shape,
        }
        for name, shape in history_shapes.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise StateError(f"{name} must have shape {shape}")
        for name in ("interference_history_w", "historical_denominator_w", "historical_quality"):
            if not np.all(np.isfinite(np.asarray(getattr(self, name)))):
                raise StateError(f"{name} must contain only finite values")
        if np.any(np.asarray(self.csi_aoi_slots) < 0):
            raise StateError("CSI AoI values must be non-negative")
        if np.any(np.asarray(self.message_aoi_slots) < -1):
            raise StateError("message AoI values must be -1 or non-negative")
        dtypes = {
            "uav_ids": np.int64,
            "paper_uav_ids": np.int64,
            "positions_m": np.float64,
            "velocities_mps": np.float64,
            "max_transmit_power_w": np.float64,
            "max_cpu_frequency_hz": np.float64,
            "cpu_coefficient": np.float64,
            "residual_energy_j": np.float64,
            "true_arrival_probabilities": np.float64,
            "candidate_neighbor_mask": np.bool_,
            "distances_m": np.float64,
            "true_channel": np.complex128,
            "blocked_links": np.bool_,
            "shadowing_db": np.float64,
            "path_loss_db": np.float64,
            "stale_csi": np.complex128,
            "csi_valid_mask": np.bool_,
            "csi_aoi_slots": np.int64,
            "interference_history_w": np.float64,
            "interference_valid_mask": np.bool_,
            "message_aoi_slots": np.int64,
            "message_aoi_valid_mask": np.bool_,
            "historical_denominator_w": np.float64,
            "historical_quality": np.float64,
            "quality_valid_mask": np.bool_,
        }
        for name, dtype in dtypes.items():
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))
        object.__setattr__(self, "active_tasks", tuple(self.active_tasks))
        object.__setattr__(self, "queues", tuple(self.queues))
        object.__setattr__(self, "buildings", tuple(self.buildings))

    def snapshot(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe pre-action state snapshot."""

        return {
            "slot": self.slot,
            "uav_ids": self.uav_ids.tolist(),
            "paper_uav_ids": self.paper_uav_ids.tolist(),
            "positions_m": self.positions_m.tolist(),
            "velocities_mps": self.velocities_mps.tolist(),
            "resources": {
                "max_transmit_power_w": self.max_transmit_power_w.tolist(),
                "max_cpu_frequency_hz": self.max_cpu_frequency_hz.tolist(),
                "cpu_coefficient": self.cpu_coefficient.tolist(),
                "residual_energy_j": self.residual_energy_j.tolist(),
            },
            "true_arrival_probabilities": self.true_arrival_probabilities.tolist(),
            "topology": {
                "candidate_neighbor_mask": self.candidate_neighbor_mask.tolist(),
                "distances_m": self.distances_m.tolist(),
            },
            "physical_channel": {
                "true_channel_real": self.true_channel.real.tolist(),
                "true_channel_imag": self.true_channel.imag.tolist(),
                "blocked_links": self.blocked_links.tolist(),
                "shadowing_db": self.shadowing_db.tolist(),
                "path_loss_db": self.path_loss_db.tolist(),
            },
            "channel_history": {
                "stale_csi_real": self.stale_csi.real.tolist(),
                "stale_csi_imag": self.stale_csi.imag.tolist(),
                "csi_valid_mask": self.csi_valid_mask.tolist(),
                "csi_aoi_slots": self.csi_aoi_slots.tolist(),
                "interference_history_w": self.interference_history_w.tolist(),
                "interference_valid_mask": self.interference_valid_mask.tolist(),
                "message_aoi_slots": self.message_aoi_slots.tolist(),
                "message_aoi_valid_mask": self.message_aoi_valid_mask.tolist(),
                "historical_denominator_w": self.historical_denominator_w.tolist(),
                "historical_quality": self.historical_quality.tolist(),
                "quality_valid_mask": self.quality_valid_mask.tolist(),
            },
            "active_tasks": [task.snapshot() for task in self.active_tasks],
            "queues": [queue.snapshot() for queue in self.queues],
            "buildings": [building.snapshot() for building in self.buildings],
        }


class CentralizedStateBuilder:
    """Build centralized state without accepting future or post-action inputs."""

    def __init__(self, environment: EnvironmentConfig) -> None:
        self.environment = environment
        if environment.uav_count <= 0 or environment.ru_count <= 0:
            raise StateError("configured UAV and RU counts must be positive")

    @classmethod
    def from_run_config(cls, config: RunConfig) -> "CentralizedStateBuilder":
        return cls(config.environment)

    def build(
        self,
        *,
        slot: int,
        mobility: MobilityState,
        topology: TopologySnapshot,
        channel: ChannelSnapshot,
        channel_features: ActorChannelFeatures,
        lifecycle: LifecycleManager,
        resource_states: Mapping[int, UavEnergyState],
    ) -> CentralizedState:
        """Copy the complete decision-time truth in deterministic ID order."""

        self._validate_inputs(
            slot,
            mobility,
            topology,
            channel,
            channel_features,
            lifecycle,
            resource_states,
        )
        count = self.environment.uav_count
        active_tasks = tuple(
            TaskStateRecord.from_task(task)
            for task in sorted(lifecycle.active_tasks(), key=lambda item: item.task_id)
        )
        queues = self._queue_records(lifecycle.queues)
        resources = [resource_states[uav_id] for uav_id in range(count)]
        return CentralizedState(
            slot=slot,
            uav_ids=np.arange(count, dtype=np.int64),
            paper_uav_ids=np.arange(1, count + 1, dtype=np.int64),
            positions_m=mobility.positions_m,
            velocities_mps=mobility.velocities_mps,
            max_transmit_power_w=[state.max_transmit_power_w for state in resources],
            max_cpu_frequency_hz=[state.max_cpu_frequency_hz for state in resources],
            cpu_coefficient=[state.cpu_coefficient for state in resources],
            residual_energy_j=[state.residual_energy_j for state in resources],
            true_arrival_probabilities=self.environment.arrival_probabilities,
            candidate_neighbor_mask=topology.candidate_neighbors,
            distances_m=topology.distances_m,
            true_channel=channel.channel,
            blocked_links=channel.blocked_links,
            shadowing_db=channel.shadowing_db,
            path_loss_db=channel.path_loss_db,
            stale_csi=channel_features.stale_csi,
            csi_valid_mask=channel_features.csi_valid_mask,
            csi_aoi_slots=channel_features.csi_aoi_slots,
            interference_history_w=channel_features.interference_history_w,
            interference_valid_mask=channel_features.interference_valid_mask,
            message_aoi_slots=channel_features.message_aoi_slots,
            message_aoi_valid_mask=channel_features.message_aoi_valid_mask,
            historical_denominator_w=channel_features.historical_denominator_w,
            historical_quality=channel_features.historical_quality,
            quality_valid_mask=channel_features.quality_valid_mask,
            active_tasks=active_tasks,
            queues=queues,
            buildings=tuple(
                BuildingStateRecord.from_config(building)
                for building in self.environment.building_layout
            ),
        )

    def _validate_inputs(
        self,
        slot: int,
        mobility: MobilityState,
        topology: TopologySnapshot,
        channel: ChannelSnapshot,
        channel_features: ActorChannelFeatures,
        lifecycle: LifecycleManager,
        resource_states: Mapping[int, UavEnergyState],
    ) -> None:
        count = self.environment.uav_count
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise StateError("slot must be a non-negative integer")
        if slot >= self.environment.episode_horizon:
            raise StateError("centralized state cannot be built at or beyond the horizon")
        if (
            mobility.slot != slot
            or topology.slot != slot
            or channel.slot != slot
            or channel_features.slot != slot
        ):
            raise StateError("mobility, topology, true channel and history must match the decision slot")
        if mobility.positions_m.shape != (count, 3):
            raise StateError("mobility state does not match the configured UAV count")
        if topology.distances_m.shape != (count, count):
            raise StateError("topology state does not match the configured UAV count")
        expected_distances = pairwise_distances(mobility.positions_m)
        if not np.allclose(
            topology.distances_m,
            expected_distances,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise StateError("topology distances do not match decision-time mobility")
        expected_candidates = candidate_neighbor_mask(
            mobility.positions_m,
            self.environment.candidate_neighbor_radius_m,
        )
        if not np.array_equal(topology.candidate_neighbors, expected_candidates):
            raise StateError("candidate graph does not match decision-time mobility")
        expected_channel_shape = (count, count, self.environment.ru_count)
        if channel.channel.shape != expected_channel_shape:
            raise StateError(f"true channel must have shape {expected_channel_shape}")
        if not np.allclose(
            topology.distances_m,
            channel.distances_m,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise StateError("topology and channel distances describe different decision states")
        expected_ids = set(range(count))
        if set(resource_states) != expected_ids:
            raise StateError("resource-state mapping must contain every UAV exactly once")
        for uav_id in range(count):
            state = resource_states[uav_id]
            if state.uav_id != uav_id:
                raise StateError("resource-state mapping key differs from state.uav_id")
            _finite_nonnegative(state.max_transmit_power_w, "max_transmit_power_w", positive=True)
            _finite_nonnegative(state.max_cpu_frequency_hz, "max_cpu_frequency_hz", positive=True)
            _finite_nonnegative(state.cpu_coefficient, "cpu_coefficient", positive=True)
            _finite_nonnegative(state.residual_energy_j, "residual_energy_j")
        lifecycle.validate_invariants()

    @staticmethod
    def _queue_records(queues: QueueState) -> tuple[QueueStateRecord, ...]:
        records: list[QueueStateRecord] = []
        for location, queue in queues.iter_queues():
            if len(queue) == 0:
                continue
            records.append(
                QueueStateRecord.from_queue(
                    location,
                    tuple(task.task_id for task in queue.items),
                )
            )
        return tuple(records)


__all__ = [
    "BuildingStateRecord",
    "CentralizedState",
    "CentralizedStateBuilder",
    "QueueStateRecord",
    "StateError",
    "TaskStateRecord",
]
