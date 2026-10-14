"""Typed task records and deterministic task identifiers.

The task model in this module follows the frozen Section 2 contract.  In
particular, ``truncated`` is an episode-accounting outcome and is deliberately
kept separate from the physical task state machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """Physical task lifecycle states defined by Section 2."""

    UNBOUND = "unbound"
    LOCAL = "local"
    TX = "tx"
    CPU = "cpu"
    DONE = "done"
    EXPIRED = "expired"


class TaskOutcome(str, Enum):
    """Terminal accounting outcome, including the horizon-only outcome."""

    NONE = "none"
    DONE = "done"
    EXPIRED = "expired"
    TRUNCATED = "truncated"


class TaskIdGenerator:
    """Episode-local monotonically increasing task-ID generator."""

    def __init__(self, start: int = 0) -> None:
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError("task ID generator start must be a non-negative integer")
        self._next = start

    def reset(self, start: int = 0) -> None:
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError("task ID generator start must be a non-negative integer")
        self._next = start

    def next_id(self) -> int:
        task_id = self._next
        self._next += 1
        return task_id

    def __iter__(self):
        return self

    def __next__(self) -> int:
        return self.next_id()


def _finite_nonnegative(value: float, name: str, *, strictly_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and converted <= 0:
        raise ValueError(f"{name} must be positive")
    if not strictly_positive and converted < 0:
        raise ValueError(f"{name} must be non-negative")
    return converted


@dataclass
class Task:
    """A mutable task record shared by queue, service and future environment code.

    ``arrival_slot`` is the slot whose end creates the task.  Consequently the
    first legal route slot is ``arrival_slot + 1``.  A route binding records its
    slot and makes service legal starting in the following slot.
    """

    task_id: int
    source_uav: int
    data_bits: float
    cpu_cycles: float
    arrival_slot: int
    deadline_slot: int
    deadline_budget_slots: int | None = None
    remaining_bits: float | None = None
    remaining_cycles: float | None = None
    destination: int | None = None
    status: TaskStatus = TaskStatus.UNBOUND
    completion_slot: int | None = None
    terminal_reason: str | None = None
    binding_slot: int | None = None
    service_eligible_slot: int | None = None
    cpu_entry_slot: int | None = None
    outcome: TaskOutcome = TaskOutcome.NONE

    def __post_init__(self) -> None:
        if isinstance(self.task_id, bool) or not isinstance(self.task_id, int) or self.task_id < 0:
            raise ValueError("task_id must be a non-negative integer")
        if isinstance(self.source_uav, bool) or not isinstance(self.source_uav, int) or self.source_uav < 0:
            raise ValueError("source_uav must be a non-negative integer")
        if isinstance(self.arrival_slot, bool) or not isinstance(self.arrival_slot, int) or self.arrival_slot < 0:
            raise ValueError("arrival_slot must be a non-negative integer")
        if isinstance(self.deadline_slot, bool) or not isinstance(self.deadline_slot, int):
            raise ValueError("deadline_slot must be an integer")
        if self.deadline_slot < self.arrival_slot:
            raise ValueError("deadline_slot cannot precede arrival_slot")

        self.data_bits = _finite_nonnegative(self.data_bits, "data_bits", strictly_positive=True)
        self.cpu_cycles = _finite_nonnegative(self.cpu_cycles, "cpu_cycles", strictly_positive=True)
        if self.deadline_budget_slots is None:
            self.deadline_budget_slots = self.deadline_slot - self.arrival_slot
        if (
            isinstance(self.deadline_budget_slots, bool)
            or not isinstance(self.deadline_budget_slots, int)
            or self.deadline_budget_slots < 0
        ):
            raise ValueError("deadline_budget_slots must be a non-negative integer")
        if self.deadline_budget_slots != self.deadline_slot - self.arrival_slot:
            raise ValueError("deadline_budget_slots must equal deadline_slot - arrival_slot")

        if self.remaining_bits is None:
            self.remaining_bits = self.data_bits
        else:
            self.remaining_bits = _finite_nonnegative(self.remaining_bits, "remaining_bits")
        if self.remaining_cycles is None:
            self.remaining_cycles = self.cpu_cycles
        else:
            self.remaining_cycles = _finite_nonnegative(self.remaining_cycles, "remaining_cycles")

        if not isinstance(self.status, TaskStatus):
            try:
                self.status = TaskStatus(self.status)
            except ValueError as exc:
                raise ValueError(f"unknown task status: {self.status!r}") from exc
        if not isinstance(self.outcome, TaskOutcome):
            try:
                self.outcome = TaskOutcome(self.outcome)
            except ValueError as exc:
                raise ValueError(f"unknown task outcome: {self.outcome!r}") from exc
        if self.status is TaskStatus.UNBOUND:
            if self.destination is not None:
                raise ValueError("unbound task must not have a destination")
            if self.binding_slot is not None or self.service_eligible_slot is not None:
                raise ValueError("unbound task must not have binding timing")
        elif self.status is TaskStatus.LOCAL:
            self._validate_bound_destination(self.source_uav)
            if self.remaining_bits != 0:
                raise ValueError("local task must have remaining_bits equal to zero")
        elif self.status in {TaskStatus.TX, TaskStatus.CPU}:
            if self.destination is None:
                raise ValueError("bound task must have a destination")
            self._validate_bound_destination(self.destination)
        elif self.status is TaskStatus.DONE:
            self.outcome = TaskOutcome.DONE
        elif self.status is TaskStatus.EXPIRED:
            self.outcome = TaskOutcome.EXPIRED

        if self.status is TaskStatus.EXPIRED and self.completion_slot is not None:
            raise ValueError("expired task must not have a completion_slot")
        if self.outcome is TaskOutcome.TRUNCATED and self.completion_slot is not None:
            raise ValueError("truncated task must not have a completion_slot")

    def _validate_bound_destination(self, destination: int) -> None:
        if isinstance(destination, bool) or not isinstance(destination, int) or destination < 0:
            raise ValueError("destination must be a non-negative integer")
        if self.status is TaskStatus.LOCAL and destination != self.source_uav:
            raise ValueError("local task destination must equal source_uav")
        if self.status in {TaskStatus.TX, TaskStatus.CPU} and destination == self.source_uav:
            raise ValueError("remote task destination must differ from source_uav")

    @classmethod
    def create(
        cls,
        *,
        task_id: int,
        source_uav: int,
        data_bits: float,
        cpu_cycles: float,
        arrival_slot: int,
        deadline_slot: int | None = None,
        deadline_budget_slots: int | None = None,
    ) -> "Task":
        """Construct a new unbound task with deterministic initial workload."""

        if deadline_slot is None:
            if deadline_budget_slots is None:
                raise ValueError("deadline_slot or deadline_budget_slots is required")
            deadline_slot = arrival_slot + deadline_budget_slots
        return cls(
            task_id=task_id,
            source_uav=source_uav,
            data_bits=data_bits,
            cpu_cycles=cpu_cycles,
            arrival_slot=arrival_slot,
            deadline_slot=deadline_slot,
            deadline_budget_slots=deadline_budget_slots,
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in {TaskStatus.DONE, TaskStatus.EXPIRED} or self.outcome is TaskOutcome.TRUNCATED

    @property
    def truncated(self) -> bool:
        return self.outcome is TaskOutcome.TRUNCATED

    @property
    def is_work_complete(self) -> bool:
        return self.remaining_bits == 0 and self.remaining_cycles == 0

    @property
    def task_id_key(self) -> tuple[int, int]:
        return (self.deadline_slot, self.task_id)

    def deadline_slack(self, slot: int) -> int:
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise TypeError("slot must be an integer")
        return self.deadline_slot - slot + 1

    def can_route(self, slot: int) -> bool:
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise TypeError("slot must be an integer")
        return not self.is_terminal and self.status is TaskStatus.UNBOUND and self.arrival_slot < slot <= self.deadline_slot

    def can_service(self, slot: int) -> bool:
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise TypeError("slot must be an integer")
        return (
            not self.is_terminal
            and self.status in {TaskStatus.LOCAL, TaskStatus.TX, TaskStatus.CPU}
            and self.service_eligible_slot is not None
            and self.service_eligible_slot <= slot <= self.deadline_slot
        )

    def bind_local(self, binding_slot: int | None = None) -> None:
        self._require_unbound("bind local")
        slot = self._normalise_binding_slot(binding_slot)
        self.destination = self.source_uav
        self.remaining_bits = 0.0
        self.status = TaskStatus.LOCAL
        self.binding_slot = slot
        self.service_eligible_slot = slot + 1

    def bind_remote(self, destination: int, binding_slot: int | None = None) -> None:
        self._require_unbound("bind remote")
        if isinstance(destination, bool) or not isinstance(destination, int) or destination < 0:
            raise ValueError("destination must be a non-negative integer")
        if destination == self.source_uav:
            raise ValueError("remote destination must differ from source_uav")
        slot = self._normalise_binding_slot(binding_slot)
        self.destination = destination
        self.status = TaskStatus.TX
        self.binding_slot = slot
        self.service_eligible_slot = slot + 1

    def enter_cpu(self, transition_slot: int | None = None, *, service_eligible_slot: int | None = None) -> None:
        if self.is_terminal:
            raise RuntimeError("terminal task cannot enter CPU")
        if self.status not in {TaskStatus.LOCAL, TaskStatus.TX}:
            raise ValueError(f"cannot transition {self.status.value} to cpu")
        if self.destination is None:
            raise ValueError("CPU transition requires a locked destination")
        self.status = TaskStatus.CPU
        if transition_slot is not None:
            if isinstance(transition_slot, bool) or not isinstance(transition_slot, int):
                raise TypeError("transition_slot must be an integer")
            self.cpu_entry_slot = transition_slot
        if service_eligible_slot is not None:
            if isinstance(service_eligible_slot, bool) or not isinstance(service_eligible_slot, int):
                raise TypeError("service_eligible_slot must be an integer")
            self.service_eligible_slot = service_eligible_slot

    def apply_transmitted_bits(self, bits: float) -> None:
        self._require_serviceable(TaskStatus.TX, "transmission")
        amount = _finite_nonnegative(bits, "bits")
        if amount > self.remaining_bits:
            raise ValueError("transmission service cannot exceed remaining_bits")
        self.remaining_bits = max(0.0, self.remaining_bits - amount)

    def apply_cpu_cycles(self, cycles: float) -> None:
        self._require_serviceable(TaskStatus.CPU, "CPU")
        amount = _finite_nonnegative(cycles, "cycles")
        if amount > self.remaining_cycles:
            raise ValueError("CPU service cannot exceed remaining_cycles")
        self.remaining_cycles = max(0.0, self.remaining_cycles - amount)

    def complete(self, completion_slot: int) -> None:
        if self.is_terminal:
            raise RuntimeError("terminal task cannot complete")
        if self.status is not TaskStatus.CPU:
            raise ValueError("only a CPU task can complete")
        if not self.is_work_complete:
            raise ValueError("task cannot complete while workload remains")
        if completion_slot > self.deadline_slot:
            raise ValueError("task cannot complete after its deadline")
        self.status = TaskStatus.DONE
        self.completion_slot = completion_slot
        self.terminal_reason = TaskOutcome.DONE.value
        self.outcome = TaskOutcome.DONE

    def expire(self, settlement_slot: int) -> None:
        if self.is_terminal:
            raise RuntimeError("terminal task cannot expire or change outcome")
        if settlement_slot < self.deadline_slot:
            raise ValueError("task cannot expire before its deadline slot")
        self.status = TaskStatus.EXPIRED
        self.completion_slot = None
        self.terminal_reason = TaskOutcome.EXPIRED.value
        self.outcome = TaskOutcome.EXPIRED

    def mark_truncated(self) -> None:
        if self.is_terminal:
            raise RuntimeError("terminal task cannot become truncated")
        self.completion_slot = None
        self.terminal_reason = TaskOutcome.TRUNCATED.value
        self.outcome = TaskOutcome.TRUNCATED

    def e2e_delay(self, slot_duration_s: float) -> float:
        if self.status is not TaskStatus.DONE or self.completion_slot is None:
            raise ValueError("E2E delay is defined only for completed tasks")
        duration = _finite_nonnegative(slot_duration_s, "slot_duration_s", strictly_positive=True)
        return (self.completion_slot - self.arrival_slot) * duration

    def snapshot(self) -> dict[str, Any]:
        """Return a deterministic, serialization-friendly task snapshot."""

        return {
            "task_id": self.task_id,
            "source_uav": self.source_uav,
            "data_bits": self.data_bits,
            "cpu_cycles": self.cpu_cycles,
            "arrival_slot": self.arrival_slot,
            "deadline_slot": self.deadline_slot,
            "deadline_budget_slots": self.deadline_budget_slots,
            "remaining_bits": self.remaining_bits,
            "remaining_cycles": self.remaining_cycles,
            "destination": self.destination,
            "status": self.status.value,
            "completion_slot": self.completion_slot,
            "terminal_reason": self.terminal_reason,
            "binding_slot": self.binding_slot,
            "service_eligible_slot": self.service_eligible_slot,
            "cpu_entry_slot": self.cpu_entry_slot,
            "outcome": self.outcome.value,
        }

    def _normalise_binding_slot(self, binding_slot: int | None) -> int:
        slot = self.arrival_slot + 1 if binding_slot is None else binding_slot
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise TypeError("binding_slot must be an integer")
        if slot <= self.arrival_slot:
            raise ValueError("route binding cannot become effective in the arrival slot")
        if slot > self.deadline_slot:
            raise ValueError("route binding cannot become effective after the deadline")
        return slot

    def _require_unbound(self, operation: str) -> None:
        if self.is_terminal:
            raise RuntimeError(f"terminal task cannot {operation}")
        if self.status is not TaskStatus.UNBOUND or self.destination is not None:
            raise ValueError("destination is already bound; rerouting is forbidden")

    def _require_serviceable(self, expected_status: TaskStatus, operation: str) -> None:
        if self.is_terminal:
            raise RuntimeError(f"terminal task cannot receive {operation} service")
        if self.status is not expected_status:
            raise ValueError(f"{operation} service requires status {expected_status.value}")


__all__ = ["Task", "TaskIdGenerator", "TaskOutcome", "TaskStatus"]
