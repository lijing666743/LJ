"""Energy reservation, discrete downgrade, and actual active-energy debit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from ..config import ActionConfig


class EnergyError(ValueError):
    """Raised when energy inputs or accounting violate hard feasibility."""


def _finite_nonnegative(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise EnergyError(f"{name} must be finite")
    if positive and converted <= 0.0:
        raise EnergyError(f"{name} must be positive")
    if not positive and converted < 0.0:
        raise EnergyError(f"{name} must be non-negative")
    return converted


@dataclass
class UavEnergyState:
    """Per-UAV physical capability and mutable residual active-energy budget."""

    uav_id: int
    max_transmit_power_w: float
    max_cpu_frequency_hz: float
    cpu_coefficient: float
    residual_energy_j: float

    def __post_init__(self) -> None:
        if isinstance(self.uav_id, bool) or not isinstance(self.uav_id, int) or self.uav_id < 0:
            raise EnergyError("uav_id must be a non-negative integer")
        self.max_transmit_power_w = _finite_nonnegative(
            self.max_transmit_power_w,
            "max_transmit_power_w",
            positive=True,
        )
        self.max_cpu_frequency_hz = _finite_nonnegative(
            self.max_cpu_frequency_hz,
            "max_cpu_frequency_hz",
            positive=True,
        )
        self.cpu_coefficient = _finite_nonnegative(
            self.cpu_coefficient,
            "cpu_coefficient",
            positive=True,
        )
        self.residual_energy_j = _finite_nonnegative(
            self.residual_energy_j,
            "residual_energy_j",
        )


@dataclass(frozen=True)
class EnergyReservation:
    """Executor feasibility bound for one final candidate combination."""

    transmit_energy_j: float
    cpu_energy_j: float
    total_energy_j: float
    candidate_power_w: float
    candidate_cpu_frequency_hz: float


@dataclass(frozen=True)
class DowngradeSelection:
    """Highest feasible discrete power/frequency combination."""

    candidate_power_w: float
    executed_cpu_frequency_hz: float
    reservation: EnergyReservation
    power_downgraded: bool
    cpu_downgraded: bool


@dataclass(frozen=True)
class ActualEnergyDebit:
    """Auditable actual debit, distinct from the feasibility reservation."""

    uav_id: int
    transmit_energy_j: float
    cpu_energy_j: float
    total_energy_j: float
    reserved_energy_j: float
    residual_before_j: float
    residual_after_j: float


def communication_reservation_j(
    candidate_power_w: float,
    slot_duration_s: float,
    *,
    active_candidate: bool,
) -> float:
    power = _finite_nonnegative(candidate_power_w, "candidate_power_w")
    duration = _finite_nonnegative(slot_duration_s, "slot_duration_s", positive=True)
    return power * duration if active_candidate and power > 0.0 else 0.0


def cpu_active_duration_s(
    frequency_hz: float,
    remaining_cycles: float,
    slot_duration_s: float,
) -> float:
    frequency = _finite_nonnegative(frequency_hz, "frequency_hz")
    cycles = _finite_nonnegative(remaining_cycles, "remaining_cycles")
    duration = _finite_nonnegative(slot_duration_s, "slot_duration_s", positive=True)
    if frequency == 0.0 or cycles == 0.0:
        return 0.0
    return min(duration, cycles / frequency)


def cpu_dynamic_energy_j(
    cpu_coefficient: float,
    frequency_hz: float,
    active_duration_s: float,
) -> float:
    coefficient = _finite_nonnegative(cpu_coefficient, "cpu_coefficient", positive=True)
    frequency = _finite_nonnegative(frequency_hz, "frequency_hz")
    duration = _finite_nonnegative(active_duration_s, "active_duration_s")
    return coefficient * frequency**3 * duration


def cpu_reservation_j(
    state: UavEnergyState,
    candidate_frequency_hz: float,
    remaining_cycles: float,
    slot_duration_s: float,
    *,
    active_candidate: bool,
) -> float:
    if not active_candidate:
        return 0.0
    duration = cpu_active_duration_s(
        candidate_frequency_hz,
        remaining_cycles,
        slot_duration_s,
    )
    return cpu_dynamic_energy_j(state.cpu_coefficient, candidate_frequency_hz, duration)


def select_highest_feasible_levels(
    state: UavEnergyState,
    action: ActionConfig,
    *,
    proposed_power_w: float,
    proposed_cpu_frequency_hz: float,
    communication_candidate: bool,
    cpu_candidate: bool,
    cpu_head_remaining_cycles: float,
    slot_duration_s: float,
    tolerance_j: float,
) -> DowngradeSelection:
    """Apply power-first then CPU-frequency downgrade without continuous values."""

    tolerance = _finite_nonnegative(tolerance_j, "tolerance_j", positive=True)
    proposed_power = _finite_nonnegative(proposed_power_w, "proposed_power_w")
    proposed_frequency = _finite_nonnegative(
        proposed_cpu_frequency_hz,
        "proposed_cpu_frequency_hz",
    )
    if proposed_power > state.max_transmit_power_w + tolerance:
        raise EnergyError("proposed power exceeds the UAV maximum")
    if proposed_frequency > state.max_cpu_frequency_hz + tolerance:
        raise EnergyError("proposed CPU frequency exceeds the UAV maximum")

    power_levels = (
        tuple(
            level * state.max_transmit_power_w
            for level in sorted(action.power_levels, reverse=True)
            if level * state.max_transmit_power_w <= proposed_power + tolerance
        )
        if communication_candidate
        else (0.0,)
    )
    cpu_levels = (
        tuple(
            level * state.max_cpu_frequency_hz
            for level in sorted(action.cpu_frequency_levels, reverse=True)
            if level * state.max_cpu_frequency_hz <= proposed_frequency + tolerance
        )
        if cpu_candidate
        else (0.0,)
    )
    if not power_levels or power_levels[-1] != 0.0:
        raise EnergyError("power domain must retain the zero fallback")
    if not cpu_levels or cpu_levels[-1] != 0.0:
        raise EnergyError("CPU-frequency domain must retain the zero fallback")

    for candidate_frequency in cpu_levels:
        cpu_reserved = cpu_reservation_j(
            state,
            candidate_frequency,
            cpu_head_remaining_cycles,
            slot_duration_s,
            active_candidate=cpu_candidate,
        )
        for candidate_power in power_levels:
            tx_reserved = communication_reservation_j(
                candidate_power,
                slot_duration_s,
                active_candidate=communication_candidate,
            )
            total = tx_reserved + cpu_reserved
            if total <= state.residual_energy_j + tolerance:
                reservation = EnergyReservation(
                    transmit_energy_j=tx_reserved,
                    cpu_energy_j=cpu_reserved,
                    total_energy_j=total,
                    candidate_power_w=candidate_power,
                    candidate_cpu_frequency_hz=candidate_frequency,
                )
                return DowngradeSelection(
                    candidate_power_w=candidate_power,
                    executed_cpu_frequency_hz=candidate_frequency,
                    reservation=reservation,
                    power_downgraded=candidate_power + tolerance < proposed_power,
                    cpu_downgraded=candidate_frequency + tolerance < proposed_frequency,
                )
    raise EnergyError("zero-power/zero-frequency safe idle is unexpectedly infeasible")


def actual_transmit_energy_j(executed_power_w: float, active_duration_s: float) -> float:
    power = _finite_nonnegative(executed_power_w, "executed_power_w")
    duration = _finite_nonnegative(active_duration_s, "active_duration_s")
    return power * duration


def debit_actual_energy(
    states: Mapping[int, UavEnergyState],
    transmit_energy_j: Mapping[int, float],
    cpu_energy_j: Mapping[int, float],
    reservations: Mapping[int, EnergyReservation],
    *,
    tolerance_j: float,
) -> tuple[ActualEnergyDebit, ...]:
    """Atomically debit actual energy and reject material feasibility defects."""

    tolerance = _finite_nonnegative(tolerance_j, "tolerance_j", positive=True)
    pending: list[tuple[UavEnergyState, ActualEnergyDebit]] = []
    for uav_id in sorted(states):
        state = states[uav_id]
        if state.uav_id != uav_id:
            raise EnergyError("energy-state mapping key does not match state.uav_id")
        tx = _finite_nonnegative(transmit_energy_j.get(uav_id, 0.0), "transmit_energy_j")
        cpu = _finite_nonnegative(cpu_energy_j.get(uav_id, 0.0), "cpu_energy_j")
        reservation = reservations.get(uav_id)
        if reservation is None:
            raise EnergyError(f"missing energy reservation for UAV {uav_id}")
        actual = tx + cpu
        if actual > reservation.total_energy_j + tolerance:
            raise EnergyError("actual energy exceeds the executor reservation")
        remaining = state.residual_energy_j - actual
        if remaining < -tolerance:
            raise EnergyError("actual energy debit would make residual energy negative")
        if remaining < 0.0:
            remaining = 0.0
        pending.append(
            (
                state,
                ActualEnergyDebit(
                    uav_id=uav_id,
                    transmit_energy_j=tx,
                    cpu_energy_j=cpu,
                    total_energy_j=actual,
                    reserved_energy_j=reservation.total_energy_j,
                    residual_before_j=state.residual_energy_j,
                    residual_after_j=remaining,
                ),
            )
        )
    for state, record in pending:
        state.residual_energy_j = record.residual_after_j
    return tuple(record for _, record in pending)


__all__ = [
    "ActualEnergyDebit",
    "DowngradeSelection",
    "EnergyError",
    "EnergyReservation",
    "UavEnergyState",
    "actual_transmit_energy_j",
    "communication_reservation_j",
    "cpu_active_duration_s",
    "cpu_dynamic_energy_j",
    "cpu_reservation_j",
    "debit_actual_energy",
    "select_highest_feasible_levels",
]
