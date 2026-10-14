"""Actual-attempt outage sampling with explicit NA semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


class OutageError(ValueError):
    """Raised when outage inputs are outside the frozen physical domain."""


@dataclass(frozen=True)
class OutageResult:
    """One link/slot outage record; sample is None when there was no attempt."""

    attempted: bool
    sample: int | None
    rb_outage_ratio: float | None
    na_reason: str | None


def actual_attempt_outage(
    *,
    accepted: bool,
    executed_power_w: float,
    executed_ru_indices: tuple[int, ...],
    slot_start_queue_bits: float,
    effective_rate_bps: float,
    sinr_linear: Iterable[float],
    threshold_linear: float,
) -> OutageResult:
    """Apply the accepted/nonzero/resource/data attempt gate and rate outcome."""

    numeric = {
        "executed_power_w": executed_power_w,
        "slot_start_queue_bits": slot_start_queue_bits,
        "effective_rate_bps": effective_rate_bps,
        "threshold_linear": threshold_linear,
    }
    for name, value in numeric.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise OutageError(f"{name} must be finite and non-negative")

    if not accepted:
        return OutageResult(False, None, None, "not_accepted")
    if executed_power_w <= 0.0:
        return OutageResult(False, None, None, "zero_power")
    if not executed_ru_indices:
        return OutageResult(False, None, None, "empty_executed_resources")
    if slot_start_queue_bits <= 0.0:
        return OutageResult(False, None, None, "no_slot_start_data")

    values = tuple(float(value) for value in sinr_linear)
    if len(values) != len(executed_ru_indices):
        raise OutageError("SINR samples must align with executed RU indices")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise OutageError("SINR samples must be finite and non-negative")
    failed = sum(value < threshold_linear for value in values)
    rb_ratio = failed / len(values)
    return OutageResult(
        attempted=True,
        sample=1 if effective_rate_bps == 0.0 else 0,
        rb_outage_ratio=rb_ratio,
        na_reason=None,
    )


__all__ = ["OutageError", "OutageResult", "actual_attempt_outage"]
