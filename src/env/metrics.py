"""Authentic episode metrics and workload-conservation accounting.

This module accumulates only events produced by the real lifecycle and physical
service implementations.  It writes no files and creates no synthetic samples.
Undefined ratios and means are represented by ``None`` rather than zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .lifecycle import LifecycleManager
from .queues import QueueKind
from .service import SlotPhysicalResult
from .tasks import Task, TaskOutcome, TaskStatus


class MetricsError(ValueError):
    """Raised when a metric event is duplicated or numerically inconsistent."""


def _finite_nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise MetricsError(f"{name} must be finite and non-negative")
    return converted


def _rounding_tolerance(*values: float) -> float:
    """Return a strict ULP-scale allowance, never a model-error tolerance."""

    scale = max((abs(float(value)) for value in values), default=1.0)
    return max(1.0e-12, 64.0 * math.ulp(max(1.0, scale)))


@dataclass(frozen=True)
class QueueBacklogRecord:
    """One slot-start snapshot of the four active queue families."""

    slot: int
    task_count: int
    unbound_task_count: int
    local_task_count: int
    tx_task_count: int
    cpu_task_count: int
    remaining_bits: float
    remaining_cycles: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "slot": self.slot,
            "task_count": self.task_count,
            "unbound_task_count": self.unbound_task_count,
            "local_task_count": self.local_task_count,
            "tx_task_count": self.tx_task_count,
            "cpu_task_count": self.cpu_task_count,
            "remaining_bits": self.remaining_bits,
            "remaining_cycles": self.remaining_cycles,
        }


@dataclass(frozen=True)
class SlotMetricsRecord:
    """Auditable actual service, attempt, and energy totals for one slot."""

    slot: int
    actual_served_bits: float
    actual_served_cycles: float
    actual_attempt_count: int
    outage_count: int
    tx_energy_j: float
    cpu_energy_j: float
    settled_task_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "slot": self.slot,
            "actual_served_bits": self.actual_served_bits,
            "actual_served_cycles": self.actual_served_cycles,
            "actual_attempt_count": self.actual_attempt_count,
            "outage_count": self.outage_count,
            "tx_energy_j": self.tx_energy_j,
            "cpu_energy_j": self.cpu_energy_j,
            "settled_task_ids": list(self.settled_task_ids),
        }


@dataclass(frozen=True)
class ConservationSnapshot:
    """Bit/cycle balance with active and terminal unfinished workload separated."""

    generated_bits: float
    actual_served_bits: float
    local_binding_cleared_bits: float
    active_remaining_bits: float
    terminal_unfinished_bits: float
    bit_accounted_total: float
    bit_balance_error: float
    bit_rounding_tolerance: float
    bit_conserved: bool
    generated_cycles: float
    actual_served_cycles: float
    active_remaining_cycles: float
    terminal_unfinished_cycles: float
    cycle_accounted_total: float
    cycle_balance_error: float
    cycle_rounding_tolerance: float
    cycle_conserved: bool

    @property
    def is_conserved(self) -> bool:
        return self.bit_conserved and self.cycle_conserved

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "generated_bits": self.generated_bits,
            "actual_served_bits": self.actual_served_bits,
            "local_binding_cleared_bits": self.local_binding_cleared_bits,
            "active_remaining_bits": self.active_remaining_bits,
            "terminal_unfinished_bits": self.terminal_unfinished_bits,
            "bit_accounted_total": self.bit_accounted_total,
            "bit_balance_error": self.bit_balance_error,
            "bit_rounding_tolerance": self.bit_rounding_tolerance,
            "bit_conserved": self.bit_conserved,
            "generated_cycles": self.generated_cycles,
            "actual_served_cycles": self.actual_served_cycles,
            "active_remaining_cycles": self.active_remaining_cycles,
            "terminal_unfinished_cycles": self.terminal_unfinished_cycles,
            "cycle_accounted_total": self.cycle_accounted_total,
            "cycle_balance_error": self.cycle_balance_error,
            "cycle_rounding_tolerance": self.cycle_rounding_tolerance,
            "cycle_conserved": self.cycle_conserved,
            "is_conserved": self.is_conserved,
        }


class EpisodeMetrics:
    """Mutable, resettable metrics foundation for one real environment episode."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Restore the empty-episode state, including explicit NA semantics."""

        self.generated_task_count = 0
        self.completed_task_count = 0
        self.expired_task_count = 0
        self.truncated_task_count = 0
        self.actual_attempt_count = 0
        self.outage_count = 0
        self.total_tx_energy_j = 0.0
        self.total_cpu_energy_j = 0.0
        self.generated_bits = 0.0
        self.generated_cycles = 0.0
        self.actual_served_bits = 0.0
        self.actual_served_cycles = 0.0
        self.local_binding_cleared_bits = 0.0
        self.completed_e2e_latency_samples_s: list[float] = []
        self.queue_backlog_records: list[QueueBacklogRecord] = []
        self.slot_records: list[SlotMetricsRecord] = []
        self._tasks: dict[int, Task] = {}
        self._local_binding_task_ids: set[int] = set()
        self._terminal_recorded_task_ids: set[int] = set()
        self._backlog_slots: set[int] = set()
        self._physical_slots: set[int] = set()

    @property
    def queue_backlog(self) -> int:
        """Return the most recently recorded slot-start task backlog."""

        return self.queue_backlog_records[-1].task_count if self.queue_backlog_records else 0

    @property
    def outage_ratio(self) -> float | None:
        return (
            self.outage_count / self.actual_attempt_count
            if self.actual_attempt_count > 0
            else None
        )

    @property
    def completion_rate(self) -> float | None:
        return (
            self.completed_task_count / self.generated_task_count
            if self.generated_task_count > 0
            else None
        )

    @property
    def expiration_rate(self) -> float | None:
        return (
            self.expired_task_count / self.generated_task_count
            if self.generated_task_count > 0
            else None
        )

    @property
    def truncation_rate(self) -> float | None:
        return (
            self.truncated_task_count / self.generated_task_count
            if self.generated_task_count > 0
            else None
        )

    @property
    def mean_completed_e2e_latency_s(self) -> float | None:
        samples = self.completed_e2e_latency_samples_s
        return math.fsum(samples) / len(samples) if samples else None

    def record_generated(self, tasks: Iterable[Task]) -> None:
        """Record real slot-end arrivals before any route or service mutation."""

        for task in self._unique_tasks(tasks, "generated tasks"):
            if task.task_id in self._tasks:
                raise MetricsError(f"task {task.task_id} was already recorded as generated")
            if task.status is not TaskStatus.UNBOUND or task.outcome is not TaskOutcome.NONE:
                raise MetricsError("generated tasks must be new unbound, nonterminal tasks")
            self._tasks[task.task_id] = task
            self.generated_task_count += 1
            self.generated_bits += _finite_nonnegative(task.data_bits, "task.data_bits")
            self.generated_cycles += _finite_nonnegative(task.cpu_cycles, "task.cpu_cycles")

    def record_local_bindings(self, tasks: Iterable[Task]) -> None:
        """Record the frozen local-route bit-clearing semantic exactly once."""

        for task in self._unique_tasks(tasks, "local bindings"):
            self._require_tracked_task(task)
            if task.task_id in self._local_binding_task_ids:
                raise MetricsError(f"local binding for task {task.task_id} was already recorded")
            if (
                task.destination != task.source_uav
                or task.binding_slot is None
                or task.status not in {TaskStatus.LOCAL, TaskStatus.CPU, TaskStatus.DONE, TaskStatus.EXPIRED}
                or task.remaining_bits != 0.0
            ):
                raise MetricsError("local binding record requires a bound local task with zero remaining bits")
            self._local_binding_task_ids.add(task.task_id)
            self.local_binding_cleared_bits += _finite_nonnegative(
                task.data_bits,
                "task.data_bits",
            )

    def record_slot_start_backlog(self, slot: int, lifecycle: LifecycleManager) -> QueueBacklogRecord:
        """Capture queue-owned backlog before proposal execution in ``slot``."""

        self._validate_slot(slot)
        if not isinstance(lifecycle, LifecycleManager):
            raise TypeError("lifecycle must be a LifecycleManager")
        if slot in self._backlog_slots:
            raise MetricsError(f"slot-start backlog for slot {slot} was already recorded")
        if self.queue_backlog_records and slot <= self.queue_backlog_records[-1].slot:
            raise MetricsError("slot-start backlog records must use increasing slots")
        lifecycle.validate_invariants()
        active_tasks = lifecycle.active_tasks()
        for task in active_tasks:
            self._require_tracked_task(task)

        counts = {kind: 0 for kind in QueueKind}
        for location, queue in lifecycle.queues.iter_queues():
            counts[location.kind] += len(queue)
        record = QueueBacklogRecord(
            slot=slot,
            task_count=len(active_tasks),
            unbound_task_count=counts[QueueKind.UNBOUND],
            local_task_count=counts[QueueKind.LOCAL],
            tx_task_count=counts[QueueKind.TX],
            cpu_task_count=counts[QueueKind.CPU],
            remaining_bits=math.fsum(
                _finite_nonnegative(task.remaining_bits, "task.remaining_bits")
                for task in active_tasks
            ),
            remaining_cycles=math.fsum(
                _finite_nonnegative(task.remaining_cycles, "task.remaining_cycles")
                for task in active_tasks
            ),
        )
        if sum(counts.values()) != record.task_count:
            raise MetricsError("slot-start queue counts do not match active task count")
        self._backlog_slots.add(slot)
        self.queue_backlog_records.append(record)
        return record

    def record_physical_result(self, result: SlotPhysicalResult) -> SlotMetricsRecord:
        """Accumulate actual service, attempt/outage, and debited energy once."""

        if not isinstance(result, SlotPhysicalResult):
            raise TypeError("result must be a SlotPhysicalResult")
        self._validate_slot(result.slot)
        if result.slot in self._physical_slots:
            raise MetricsError(f"physical result for slot {result.slot} was already recorded")
        if self.slot_records and result.slot <= self.slot_records[-1].slot:
            raise MetricsError("physical results must use increasing slots")

        served_bits: list[float] = []
        attempt_count = 0
        outage_count = 0
        for link in result.links:
            link_amounts: list[float] = []
            for service in link.task_services:
                if service.task_id not in self._tasks:
                    raise MetricsError(f"TX service references unrecorded task {service.task_id}")
                amount = _finite_nonnegative(service.amount, "task TX service amount")
                link_amounts.append(amount)
                served_bits.append(amount)
            link_total = math.fsum(link_amounts)
            service_bits = _finite_nonnegative(link.service_bits, "link.service_bits")
            tolerance = _rounding_tolerance(link_total, service_bits)
            if abs(link_total - service_bits) > tolerance:
                raise MetricsError("link service_bits does not equal its per-task service records")
            if link.outage.attempted:
                if link.outage.sample not in {0, 1}:
                    raise MetricsError("attempted outage must have a binary sample")
                attempt_count += 1
                outage_count += int(link.outage.sample)
            elif link.outage.sample is not None:
                raise MetricsError("non-attempt outage sample must be None")

        served_cycles: list[float] = []
        for service in result.cpu_services:
            cycles = _finite_nonnegative(service.service_cycles, "CPU service cycles")
            if cycles > 0.0:
                if service.task_id is None or service.task_id not in self._tasks:
                    raise MetricsError("positive CPU service references an unrecorded task")
            served_cycles.append(cycles)

        tx_energy_values: list[float] = []
        cpu_energy_values: list[float] = []
        for debit in result.energy_debits:
            tx_energy = _finite_nonnegative(debit.transmit_energy_j, "transmit energy")
            cpu_energy = _finite_nonnegative(debit.cpu_energy_j, "CPU energy")
            total_energy = _finite_nonnegative(debit.total_energy_j, "total energy")
            tolerance = _rounding_tolerance(tx_energy, cpu_energy, total_energy)
            if abs(total_energy - (tx_energy + cpu_energy)) > tolerance:
                raise MetricsError("energy debit total does not equal TX plus CPU energy")
            tx_energy_values.append(tx_energy)
            cpu_energy_values.append(cpu_energy)

        tx_energy_total = math.fsum(tx_energy_values)
        cpu_energy_total = math.fsum(cpu_energy_values)
        link_energy_total = math.fsum(
            _finite_nonnegative(link.transmit_energy_j, "link transmit energy")
            for link in result.links
        )
        cpu_record_energy_total = math.fsum(
            _finite_nonnegative(service.cpu_energy_j, "CPU service energy")
            for service in result.cpu_services
        )
        if abs(tx_energy_total - link_energy_total) > _rounding_tolerance(
            tx_energy_total,
            link_energy_total,
        ):
            raise MetricsError("debited TX energy does not match physical link energy")
        if abs(cpu_energy_total - cpu_record_energy_total) > _rounding_tolerance(
            cpu_energy_total,
            cpu_record_energy_total,
        ):
            raise MetricsError("debited CPU energy does not match CPU service energy")

        for task_id in result.settled_task_ids:
            if task_id not in self._tasks:
                raise MetricsError(f"settlement references unrecorded task {task_id}")
        record = SlotMetricsRecord(
            slot=result.slot,
            actual_served_bits=math.fsum(served_bits),
            actual_served_cycles=math.fsum(served_cycles),
            actual_attempt_count=attempt_count,
            outage_count=outage_count,
            tx_energy_j=tx_energy_total,
            cpu_energy_j=cpu_energy_total,
            settled_task_ids=tuple(result.settled_task_ids),
        )
        self.actual_served_bits += record.actual_served_bits
        self.actual_served_cycles += record.actual_served_cycles
        self.actual_attempt_count += record.actual_attempt_count
        self.outage_count += record.outage_count
        self.total_tx_energy_j += record.tx_energy_j
        self.total_cpu_energy_j += record.cpu_energy_j
        self._physical_slots.add(result.slot)
        self.slot_records.append(record)
        return record

    def record_settled_tasks(self, tasks: Iterable[Task], *, slot_duration_s: float) -> None:
        """Record this slot's mutually exclusive done/expired settlements."""

        duration = _finite_nonnegative(slot_duration_s, "slot_duration_s")
        if duration <= 0.0:
            raise MetricsError("slot_duration_s must be positive")
        materialized = self._unique_tasks(tasks, "settled tasks")
        if any(task.outcome not in {TaskOutcome.DONE, TaskOutcome.EXPIRED} for task in materialized):
            raise MetricsError("settled tasks may contain only done or expired outcomes")
        self._record_terminal_tasks(materialized, slot_duration_s=duration)

    def record_truncated_tasks(self, tasks: Iterable[Task]) -> None:
        """Record horizon-only truncation without fabricating completion latency."""

        materialized = self._unique_tasks(tasks, "truncated tasks")
        if any(task.outcome is not TaskOutcome.TRUNCATED for task in materialized):
            raise MetricsError("truncated task records require the truncated outcome")
        self._record_terminal_tasks(materialized, slot_duration_s=None)

    def conservation_snapshot(self) -> ConservationSnapshot:
        """Compute current bit/cycle balances from tracked live task records."""

        active_remaining_bits: list[float] = []
        terminal_unfinished_bits: list[float] = []
        active_remaining_cycles: list[float] = []
        terminal_unfinished_cycles: list[float] = []
        for task in (self._tasks[task_id] for task_id in sorted(self._tasks)):
            remaining_bits = _finite_nonnegative(task.remaining_bits, "task.remaining_bits")
            remaining_cycles = _finite_nonnegative(task.remaining_cycles, "task.remaining_cycles")
            if not task.is_terminal:
                active_remaining_bits.append(remaining_bits)
                active_remaining_cycles.append(remaining_cycles)
            elif task.outcome in {TaskOutcome.EXPIRED, TaskOutcome.TRUNCATED}:
                terminal_unfinished_bits.append(remaining_bits)
                terminal_unfinished_cycles.append(remaining_cycles)
            elif task.outcome is TaskOutcome.DONE:
                tolerance = _rounding_tolerance(remaining_bits, remaining_cycles)
                if remaining_bits > tolerance or remaining_cycles > tolerance:
                    raise MetricsError("done task retains unfinished workload")
            else:
                raise MetricsError("terminal task has no recognized terminal outcome")

        active_bits = math.fsum(active_remaining_bits)
        unfinished_bits = math.fsum(terminal_unfinished_bits)
        bit_accounted = math.fsum(
            (
                self.actual_served_bits,
                self.local_binding_cleared_bits,
                active_bits,
                unfinished_bits,
            )
        )
        bit_error = self.generated_bits - bit_accounted
        bit_tolerance = _rounding_tolerance(
            self.generated_bits,
            self.actual_served_bits,
            self.local_binding_cleared_bits,
            active_bits,
            unfinished_bits,
        )

        active_cycles = math.fsum(active_remaining_cycles)
        unfinished_cycles = math.fsum(terminal_unfinished_cycles)
        cycle_accounted = math.fsum(
            (self.actual_served_cycles, active_cycles, unfinished_cycles)
        )
        cycle_error = self.generated_cycles - cycle_accounted
        cycle_tolerance = _rounding_tolerance(
            self.generated_cycles,
            self.actual_served_cycles,
            active_cycles,
            unfinished_cycles,
        )
        return ConservationSnapshot(
            generated_bits=self.generated_bits,
            actual_served_bits=self.actual_served_bits,
            local_binding_cleared_bits=self.local_binding_cleared_bits,
            active_remaining_bits=active_bits,
            terminal_unfinished_bits=unfinished_bits,
            bit_accounted_total=bit_accounted,
            bit_balance_error=bit_error,
            bit_rounding_tolerance=bit_tolerance,
            bit_conserved=abs(bit_error) <= bit_tolerance,
            generated_cycles=self.generated_cycles,
            actual_served_cycles=self.actual_served_cycles,
            active_remaining_cycles=active_cycles,
            terminal_unfinished_cycles=unfinished_cycles,
            cycle_accounted_total=cycle_accounted,
            cycle_balance_error=cycle_error,
            cycle_rounding_tolerance=cycle_tolerance,
            cycle_conserved=abs(cycle_error) <= cycle_tolerance,
        )

    def assert_conservation(self) -> ConservationSnapshot:
        """Return the balance or raise when more than rounding error is missing."""

        snapshot = self.conservation_snapshot()
        if not snapshot.is_conserved:
            raise MetricsError(
                "workload conservation failed: "
                f"bit_error={snapshot.bit_balance_error}, "
                f"cycle_error={snapshot.cycle_balance_error}"
            )
        return snapshot

    def snapshot(self) -> dict[str, object]:
        """Return a deterministic JSON-safe episode metrics snapshot."""

        conservation = self.conservation_snapshot()
        return {
            "generated_task_count": self.generated_task_count,
            "completed_task_count": self.completed_task_count,
            "expired_task_count": self.expired_task_count,
            "truncated_task_count": self.truncated_task_count,
            "actual_attempt_count": self.actual_attempt_count,
            "outage_count": self.outage_count,
            "outage_ratio": self.outage_ratio,
            "total_tx_energy_j": self.total_tx_energy_j,
            "total_cpu_energy_j": self.total_cpu_energy_j,
            "total_active_energy_j": self.total_tx_energy_j + self.total_cpu_energy_j,
            "queue_backlog": self.queue_backlog,
            "completion_rate": self.completion_rate,
            "expiration_rate": self.expiration_rate,
            "truncation_rate": self.truncation_rate,
            "completed_e2e_latency_samples_s": list(self.completed_e2e_latency_samples_s),
            "mean_completed_e2e_latency_s": self.mean_completed_e2e_latency_s,
            "generated_bits": self.generated_bits,
            "generated_cycles": self.generated_cycles,
            "actual_served_bits": self.actual_served_bits,
            "actual_served_cycles": self.actual_served_cycles,
            "local_binding_cleared_bits": self.local_binding_cleared_bits,
            "conservation": conservation.to_dict(),
            "slot_start_backlog": [record.to_dict() for record in self.queue_backlog_records],
            "slot_records": [record.to_dict() for record in self.slot_records],
        }

    def to_dict(self) -> dict[str, object]:
        return self.snapshot()

    def _record_terminal_tasks(
        self,
        tasks: tuple[Task, ...],
        *,
        slot_duration_s: float | None,
    ) -> None:
        for task in tasks:
            self._require_tracked_task(task)
            if task.task_id in self._terminal_recorded_task_ids:
                raise MetricsError(f"terminal outcome for task {task.task_id} was already recorded")
            if task.outcome is TaskOutcome.DONE:
                if slot_duration_s is None:
                    raise MetricsError("done task accounting requires slot_duration_s")
                latency = _finite_nonnegative(
                    task.e2e_delay(slot_duration_s),
                    "completed E2E latency",
                )
                self.completed_task_count += 1
                self.completed_e2e_latency_samples_s.append(latency)
            elif task.outcome is TaskOutcome.EXPIRED:
                self.expired_task_count += 1
            elif task.outcome is TaskOutcome.TRUNCATED:
                self.truncated_task_count += 1
            else:
                raise MetricsError("task has no terminal outcome to record")
            self._terminal_recorded_task_ids.add(task.task_id)

    def _require_tracked_task(self, task: Task) -> None:
        if self._tasks.get(task.task_id) is not task:
            raise MetricsError(f"task {task.task_id} is not the recorded generated task object")

    @staticmethod
    def _unique_tasks(tasks: Iterable[Task], name: str) -> tuple[Task, ...]:
        materialized = tuple(tasks)
        if any(not isinstance(task, Task) for task in materialized):
            raise TypeError(f"{name} must contain only Task instances")
        ids = [task.task_id for task in materialized]
        if len(ids) != len(set(ids)):
            raise MetricsError(f"{name} contains duplicate task IDs")
        return tuple(sorted(materialized, key=lambda task: task.task_id))

    @staticmethod
    def _validate_slot(slot: int) -> None:
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise MetricsError("slot must be a non-negative integer")


EpisodeMetricsTracker = EpisodeMetrics


__all__ = [
    "ConservationSnapshot",
    "EpisodeMetrics",
    "EpisodeMetricsTracker",
    "MetricsError",
    "QueueBacklogRecord",
    "SlotMetricsRecord",
]
