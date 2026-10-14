"""Pure current-slot distance and candidate-neighbor topology derivation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class TopologyError(ValueError):
    """Raised when positions or candidate-neighbor settings are invalid."""


def _positions_array(positions_m: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise TopologyError("positions_m must have shape (uav_count, 3)")
    if positions.shape[0] <= 0 or not np.all(np.isfinite(positions)):
        raise TopologyError("positions_m must contain finite positions for at least one UAV")
    return positions


def pairwise_distances(positions_m: np.ndarray) -> np.ndarray:
    """Return the full three-dimensional Euclidean distance matrix in metres."""

    positions = _positions_array(positions_m)
    deltas = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(deltas, axis=-1)


def candidate_neighbor_mask(positions_m: np.ndarray, communication_radius_m: float) -> np.ndarray:
    """Return the directed candidate graph mask recomputed from current positions."""

    if not math.isfinite(float(communication_radius_m)) or communication_radius_m <= 0:
        raise TopologyError("communication_radius_m must be finite and positive")
    distances = pairwise_distances(positions_m)
    mask = distances <= float(communication_radius_m)
    np.fill_diagonal(mask, False)
    return mask


@dataclass(frozen=True)
class TopologySnapshot:
    """Current-slot distances and the no-self candidate-neighbor mask."""

    slot: int
    distances_m: np.ndarray
    candidate_neighbors: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.slot, bool) or not isinstance(self.slot, int) or self.slot < 0:
            raise TopologyError("topology slot must be a non-negative integer")
        distances = np.array(self.distances_m, dtype=np.float64, copy=True)
        neighbors = np.array(self.candidate_neighbors, dtype=np.bool_, copy=True)
        if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
            raise TopologyError("distances_m must be a square matrix")
        if neighbors.shape != distances.shape:
            raise TopologyError("candidate neighbor mask must match the distance matrix")
        if not np.all(np.isfinite(distances)) or np.any(distances < 0):
            raise TopologyError("distances_m must be finite and non-negative")
        if np.any(np.diag(neighbors)):
            raise TopologyError("candidate topology must exclude self-links")
        distances.setflags(write=False)
        neighbors.setflags(write=False)
        object.__setattr__(self, "distances_m", distances)
        object.__setattr__(self, "candidate_neighbors", neighbors)

    def neighbors_of(self, uav_id: int) -> tuple[int, ...]:
        if isinstance(uav_id, bool) or not isinstance(uav_id, int) or not 0 <= uav_id < self.distances_m.shape[0]:
            raise TopologyError("uav_id is outside the topology")
        return tuple(int(index) for index in np.flatnonzero(self.candidate_neighbors[uav_id]))


class DynamicTopology:
    """Stateless topology builder; no neighbor hysteresis is retained."""

    def __init__(self, communication_radius_m: float) -> None:
        if not math.isfinite(float(communication_radius_m)) or communication_radius_m <= 0:
            raise TopologyError("communication_radius_m must be finite and positive")
        self.communication_radius_m = float(communication_radius_m)

    def compute(self, slot: int, positions_m: np.ndarray) -> TopologySnapshot:
        distances = pairwise_distances(positions_m)
        neighbors = distances <= self.communication_radius_m
        np.fill_diagonal(neighbors, False)
        return TopologySnapshot(slot, distances, neighbors)


__all__ = [
    "DynamicTopology",
    "TopologyError",
    "TopologySnapshot",
    "candidate_neighbor_mask",
    "pairwise_distances",
]
