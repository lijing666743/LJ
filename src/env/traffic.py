"""Causal Bernoulli task arrivals and historical arrival-rate estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import EnvironmentConfig
from .lifecycle import LifecycleManager
from .tasks import Task


class TrafficError(ValueError):
    """Raised when traffic generation is called out of slot order."""


def _readonly(values: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ArrivalEstimate:
    """Slot-start empirical arrival rate with an explicit availability mask."""

    slot: int
    values: np.ndarray
    valid_mask: np.ndarray
    sample_count: int

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        mask = np.asarray(self.valid_mask, dtype=np.bool_)
        if values.ndim != 1 or mask.shape != values.shape:
            raise TrafficError("arrival estimate values/mask must be aligned vectors")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise TrafficError("arrival estimates must be finite and non-negative")
        object.__setattr__(self, "values", _readonly(values, np.float64))
        object.__setattr__(self, "valid_mask", _readonly(mask, np.bool_))


@dataclass(frozen=True)
class ArrivalBatch:
    """Actual tasks generated at one slot end in stable UAV-ID order."""

    slot: int
    indicators: np.ndarray
    tasks: tuple[Task, ...]

    def __post_init__(self) -> None:
        indicators = np.asarray(self.indicators, dtype=np.int64)
        if indicators.ndim != 1 or np.any((indicators != 0) & (indicators != 1)):
            raise TrafficError("arrival indicators must be a binary vector")
        object.__setattr__(self, "indicators", _readonly(indicators, np.int64))


class TrafficProcess:
    """Own streams 20/30 and the completed-slot arrival history."""

    def __init__(
        self,
        config: EnvironmentConfig,
        arrival_rng: np.random.Generator,
        workload_rng: np.random.Generator,
    ) -> None:
        if not isinstance(arrival_rng, np.random.Generator):
            raise TypeError("arrival_rng must be an explicit NumPy Generator")
        if not isinstance(workload_rng, np.random.Generator):
            raise TypeError("workload_rng must be an explicit NumPy Generator")
        self.config = config
        self.arrival_rng = arrival_rng
        self.workload_rng = workload_rng
        self._history: list[np.ndarray] = []

    @property
    def completed_slot_count(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        """Clear causal history; RNG replacement is owned by environment reset."""

        self._history.clear()

    def estimate(self, slot: int) -> ArrivalEstimate:
        """Use only arrivals from slots ``0..slot-1``."""

        self._require_slot(slot)
        count = min(self.config.arrival_history_window_slots, slot)
        if count == 0:
            values = np.zeros(self.config.uav_count, dtype=np.float64)
            mask = np.zeros(self.config.uav_count, dtype=np.bool_)
        else:
            values = np.mean(np.stack(self._history[-count:], axis=0), axis=0)
            mask = np.ones(self.config.uav_count, dtype=np.bool_)
        return ArrivalEstimate(slot, values, mask, count)

    def generate(self, slot: int, lifecycle: LifecycleManager) -> ArrivalBatch:
        """Sample slot-end arrivals, then create workloads through lifecycle."""

        self._require_slot(slot)
        indicators = np.zeros(self.config.uav_count, dtype=np.int64)
        specs: list[dict[str, float | int]] = []
        for source_uav in range(self.config.uav_count):
            arrival_draw = self.arrival_rng.random()
            arrived = bool(arrival_draw < self.config.arrival_probabilities[source_uav])
            indicators[source_uav] = int(arrived)
            reference = self.config.workload_reference_arrival_probabilities
            workload_arrived = arrived if not reference else arrival_draw < reference[source_uav]
            if not workload_arrived:
                continue
            data_bits = float(
                self.workload_rng.uniform(
                    self.config.task_data_bits_min,
                    self.config.task_data_bits_max,
                )
            )
            cycles_per_bit = float(
                self.workload_rng.uniform(
                    self.config.task_cycles_per_bit_min,
                    self.config.task_cycles_per_bit_max,
                )
            )
            cpu_cycles = data_bits * cycles_per_bit
            deadline_factor = float(
                self.workload_rng.uniform(
                    self.config.task_deadline_factor_min,
                    self.config.task_deadline_factor_max,
                )
            )
            raw_budget = deadline_factor * (
                data_bits / self.config.reference_rate_bps
                + cpu_cycles / self.config.reference_cpu_frequency_hz
            ) / self.config.slot_duration_s
            budget = int(
                np.clip(
                    math.ceil(raw_budget),
                    self.config.minimum_task_slack_slots,
                    self.config.maximum_task_slack_slots,
                )
            )
            if not arrived:
                continue
            specs.append(
                {
                    "source_uav": source_uav,
                    "data_bits": data_bits,
                    "cpu_cycles": cpu_cycles,
                    "deadline_budget_slots": budget,
                }
            )
        tasks = lifecycle.create_arrivals(specs, arrival_slot=slot)
        self._history.append(indicators.copy())
        return ArrivalBatch(slot, indicators, tasks)

    def history_snapshot(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(value) for value in row) for row in self._history)

    def _require_slot(self, slot: int) -> None:
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise TrafficError("slot must be a non-negative integer")
        if slot != len(self._history):
            raise TrafficError(
                f"traffic history is ready for slot {len(self._history)}, not {slot}"
            )


__all__ = [
    "ArrivalBatch",
    "ArrivalEstimate",
    "TrafficError",
    "TrafficProcess",
]
