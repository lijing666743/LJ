"""Atomic task lifecycle and queue transitions for the environment backend."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .queues import QueueState, TaskQueue
from .tasks import Task, TaskIdGenerator, TaskOutcome, TaskStatus


class LifecycleError(ValueError):
    """Raised when a task lifecycle operation violates the frozen contract."""


@dataclass(frozen=True)
class TruncationRecord:
    task_id: int
    horizon_slot: int
    outcome: TaskOutcome = TaskOutcome.TRUNCATED


class LifecycleManager:
    """Single owner of task records and active queue transitions.

    This class deliberately contains no wireless, CPU-frequency, energy or
    arrival-randomness model.  Service modules provide already-authorized bit
    or cycle amounts through ``service_transmission``/``service_cpu`` and this
    manager enforces timing, workload conservation, transitions and settlement.
    """

    def __init__(self, queue_state: QueueState | None = None, task_id_generator: TaskIdGenerator | None = None) -> None:
        self.queues = queue_state or QueueState()
        self.task_id_generator = task_id_generator or TaskIdGenerator()
        self.tasks: dict[int, Task] = {}

    @property
    def queue_state(self) -> QueueState:
        return self.queues

    def reset(self) -> None:
        self.tasks.clear()
        self.queues.clear()
        self.task_id_generator.reset()

    def create_task(
        self,
        *,
        source_uav: int,
        data_bits: float,
        cpu_cycles: float,
        arrival_slot: int,
        deadline_slot: int | None = None,
        deadline_budget_slots: int | None = None,
    ) -> Task:
        task = Task.create(
            task_id=self.task_id_generator.next_id(),
            source_uav=source_uav,
            data_bits=data_bits,
            cpu_cycles=cpu_cycles,
            arrival_slot=arrival_slot,
            deadline_slot=deadline_slot,
            deadline_budget_slots=deadline_budget_slots,
        )
        self.register_arrival(task)
        return task

    def create_arrivals(self, specs: Iterable[Mapping[str, Any]], *, arrival_slot: int | None = None) -> tuple[Task, ...]:
        """Create same-slot arrivals in source-UAV then stable-input order."""

        materialised = list(specs)
        indexed = list(enumerate(materialised))
        indexed.sort(key=lambda item: (int(item[1]["source_uav"]), item[0]))
        created: list[Task] = []
        for _, spec in indexed:
            values = dict(spec)
            if arrival_slot is not None:
                values["arrival_slot"] = arrival_slot
            created.append(self.create_task(**values))
        return tuple(created)

    def register_arrival(self, task: Task) -> Task:
        if not isinstance(task, Task):
            raise TypeError("register_arrival expects a Task")
        if task.task_id in self.tasks:
            raise LifecycleError(f"task ID {task.task_id} is already registered")
        if task.status is not TaskStatus.UNBOUND or task.destination is not None:
            raise LifecycleError("arrivals must start in the unbound state")
        self.tasks[task.task_id] = task
        try:
            self.queues.enqueue_unbound(task)
            self.validate_invariants()
        except Exception:
            self.tasks.pop(task.task_id, None)
            self.queues.unbound[task.source_uav].discard(task.task_id)
            raise
        return task

    def register_arrivals(self, tasks: Iterable[Task]) -> tuple[Task, ...]:
        """Register already-created arrivals in deterministic source/order order."""

        materialised = list(tasks)
        indexed = list(enumerate(materialised))
        indexed.sort(key=lambda item: (item[1].source_uav, item[0]))
        for _, task in indexed:
            self.register_arrival(task)
        return tuple(task for _, task in indexed)

    def get_task(self, task_or_id: Task | int) -> Task:
        task_id = task_or_id.task_id if isinstance(task_or_id, Task) else task_or_id
        try:
            task = self.tasks[task_id]
        except KeyError as exc:
            raise LifecycleError(f"unknown task ID {task_id}") from exc
        if isinstance(task_or_id, Task) and task_or_id is not task:
            raise LifecycleError(f"task ID {task_id} refers to a different task object")
        return task

    def can_route(self, task_or_id: Task | int, slot: int) -> bool:
        return self.get_task(task_or_id).can_route(slot)

    def can_service(self, task_or_id: Task | int, slot: int) -> bool:
        return self.get_task(task_or_id).can_service(slot)

    def route_task(
        self,
        task_or_id: Task | int,
        *,
        slot: int,
        destination: int | str | None,
        valid_destinations: Iterable[int] | None = None,
    ) -> Task:
        """Apply one route decision at slot end without performing service."""

        task = self.get_task(task_or_id)
        if not task.can_route(slot):
            raise LifecycleError("task cannot route in this slot")
        if not self._is_in_unbound_queue(task):
            raise LifecycleError("route requires active ownership by the unbound queue")
        self._require_queue_head(task, self.queues.unbound[task.source_uav], "route")

        if destination is None or destination in {"defer", "idle"}:
            return task
        if destination == "local":
            self.queues.remove(task)
            task.bind_local(binding_slot=slot)
            try:
                self.queues.enqueue_local(task)
            except Exception:
                self._restore_unbound(task)
                raise
            self.validate_invariants()
            return task
        if isinstance(destination, bool) or not isinstance(destination, int):
            raise LifecycleError("destination must be 'local', 'defer', or a remote UAV integer")
        if destination == task.source_uav:
            raise LifecycleError("remote destination must differ from source_uav")
        if valid_destinations is not None and destination not in set(valid_destinations):
            raise LifecycleError("destination is not a legal route candidate")

        self.queues.remove(task)
        task.bind_remote(destination, binding_slot=slot)
        try:
            self.queues.enqueue_tx(task)
        except Exception:
            self._restore_unbound(task)
            raise
        self.validate_invariants()
        return task

    def route(self, task_or_id: Task | int, slot: int, destination: int | str | None) -> Task:
        return self.route_task(task_or_id, slot=slot, destination=destination)

    def route_next(self, source_uav: int, *, slot: int, destination: int | str | None) -> Task:
        task = self.queues.peek_unbound(source_uav)
        if task is None:
            raise LifecycleError(f"UAV {source_uav} has no unbound task to route")
        return self.route_task(task, slot=slot, destination=destination)

    def service_transmission(self, task_or_id: Task | int, *, slot: int, bits: float) -> Task:
        """Apply actual TX service and move a completed transfer to CPU at slot end."""

        task = self.get_task(task_or_id)
        self._require_service_window(task, slot, TaskStatus.TX)
        if task.destination is None:
            raise LifecycleError("transmission service requires a locked destination")
        self._require_queue_head(
            task,
            self.queues.tx[(task.source_uav, task.destination)],
            "transmission service",
        )
        try:
            task.apply_transmitted_bits(bits)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise LifecycleError(str(exc)) from exc

        if task.remaining_bits == 0:
            self.queues.remove(task)
            task.enter_cpu(transition_slot=slot, service_eligible_slot=slot + 1)
            try:
                self.queues.enqueue_cpu(task)
            except Exception:
                task.status = TaskStatus.TX
                task.service_eligible_slot = slot
                self.queues.enqueue_tx(task)
                raise
        self.validate_invariants()
        return task

    def service_tx(self, task_or_id: Task | int, slot: int, bits: float) -> Task:
        return self.service_transmission(task_or_id, slot=slot, bits=bits)

    def service_cpu(self, task_or_id: Task | int, *, slot: int, cycles: float) -> Task:
        """Apply actual CPU service; settlement records completion at slot end."""

        task = self.get_task(task_or_id)
        self._require_service_window(task, slot, TaskStatus.CPU, allow_local=True)
        if task.destination is None:
            raise LifecycleError("CPU service requires a locked destination")
        if task.destination == task.source_uav:
            queue = self.queues.local[task.source_uav]
        else:
            queue = self.queues.cpu[(task.source_uav, task.destination)]
        self._require_queue_head(task, queue, "CPU service")
        if task.status is TaskStatus.LOCAL:
            task.enter_cpu(transition_slot=slot, service_eligible_slot=slot)
        try:
            task.apply_cpu_cycles(cycles)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise LifecycleError(str(exc)) from exc
        self.validate_invariants()
        return task

    def service_cpu_cycles(self, task_or_id: Task | int, slot: int, cycles: float) -> Task:
        return self.service_cpu(task_or_id, slot=slot, cycles=cycles)

    def transition_transfer_completion(self, task_or_id: Task | int, *, slot: int) -> Task:
        """Move a zero-bit TX task to the remote CPU queue at slot end."""

        task = self.get_task(task_or_id)
        if task.status is not TaskStatus.TX or task.remaining_bits != 0:
            raise LifecycleError("transfer completion requires a TX task with zero remaining bits")
        self._require_service_window(task, slot, TaskStatus.TX)
        if task.destination is None:
            raise LifecycleError("transfer completion requires a locked destination")
        self._require_queue_head(
            task,
            self.queues.tx[(task.source_uav, task.destination)],
            "transfer completion",
        )
        self.queues.remove(task)
        task.enter_cpu(transition_slot=slot, service_eligible_slot=slot + 1)
        self.queues.enqueue_cpu(task)
        self.validate_invariants()
        return task

    def settle_slot(self, slot: int) -> tuple[Task, ...]:
        """Run post-service/post-transition done/expired settlement for a slot."""

        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise LifecycleError("settlement slot must be a non-negative integer")
        settled: list[Task] = []
        for task in sorted(self.tasks.values(), key=lambda item: item.task_id):
            if task.is_terminal:
                continue
            if task.is_work_complete and task.status is TaskStatus.CPU and slot <= task.deadline_slot:
                task.complete(slot)
                self.queues.remove(task)
                settled.append(task)
                continue
            if slot >= task.deadline_slot:
                task.expire(slot)
                self.queues.remove(task)
                settled.append(task)
        self.validate_invariants()
        return tuple(settled)

    def settle(self, slot: int) -> tuple[Task, ...]:
        return self.settle_slot(slot)

    def truncate_horizon(self, horizon_slot: int) -> tuple[TruncationRecord, ...]:
        """Mark unresolved tasks at the episode boundary without fake completion."""

        if isinstance(horizon_slot, bool) or not isinstance(horizon_slot, int) or horizon_slot < 0:
            raise LifecycleError("horizon_slot must be a non-negative integer")
        records: list[TruncationRecord] = []
        for task in sorted(self.tasks.values(), key=lambda item: item.task_id):
            if task.is_terminal:
                continue
            if task.deadline_slot < horizon_slot:
                task.expire(horizon_slot)
                self.queues.remove(task)
                continue
            task.mark_truncated()
            self.queues.remove(task)
            records.append(TruncationRecord(task.task_id, horizon_slot))
        self.validate_invariants()
        return tuple(records)

    def mark_truncated(self, horizon_slot: int) -> tuple[TruncationRecord, ...]:
        return self.truncate_horizon(horizon_slot)

    def validate_invariants(self) -> None:
        self.queues.validate_invariants(self.tasks)

    def active_tasks(self) -> tuple[Task, ...]:
        return tuple(task for task in sorted(self.tasks.values(), key=lambda item: item.task_id) if not task.is_terminal)

    def _is_in_unbound_queue(self, task: Task) -> bool:
        location = self.queues.locate(task)
        return location is not None and location.kind.value == "unbound"

    @staticmethod
    def _require_queue_head(task: Task, queue: TaskQueue, operation: str) -> None:
        if not queue.contains(task):
            raise LifecycleError(f"{operation} requires task membership in its active queue")
        if queue.peek() is not task:
            raise LifecycleError(f"{operation} requires the current EDF queue head")

    def _restore_unbound(self, task: Task) -> None:
        task.destination = None
        task.status = TaskStatus.UNBOUND
        task.binding_slot = None
        task.service_eligible_slot = None
        task.outcome = TaskOutcome.NONE
        task.terminal_reason = None
        self.queues.enqueue_unbound(task)

    def _require_service_window(
        self,
        task: Task,
        slot: int,
        expected_status: TaskStatus,
        *,
        allow_local: bool = False,
    ) -> None:
        if not task.can_service(slot):
            raise LifecycleError("task is not eligible for service in this slot")
        if task.status is not expected_status and not (allow_local and task.status is TaskStatus.LOCAL):
            raise LifecycleError(f"service requires {expected_status.value} task status")
        location = self.queues.locate(task)
        if location is None:
            raise LifecycleError("service requires active queue ownership")
        if expected_status is TaskStatus.TX and location.kind.value != "tx":
            raise LifecycleError("transmission service requires tx queue ownership")
        if expected_status is TaskStatus.CPU and location.kind.value not in {"local", "cpu"}:
            raise LifecycleError("CPU service requires local or CPU queue ownership")


TaskLifecycle = LifecycleManager
TaskLifecycleManager = LifecycleManager


__all__ = [
    "LifecycleError",
    "LifecycleManager",
    "TaskLifecycle",
    "TaskLifecycleManager",
    "TruncationRecord",
]
