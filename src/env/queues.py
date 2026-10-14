"""Deterministic EDF queues and single-ownership queue state."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import DefaultDict, Iterable, Iterator, Mapping

from .tasks import Task, TaskStatus


class QueueInvariantError(ValueError):
    """Raised when queue ownership or task/queue consistency is violated."""


class QueueKind(str, Enum):
    UNBOUND = "unbound"
    LOCAL = "local"
    TX = "tx"
    CPU = "cpu"


@dataclass(frozen=True)
class QueueLocation:
    kind: QueueKind
    source_uav: int
    destination_uav: int | None = None


class TaskQueue:
    """A small sorted task queue with EDF then deterministic task-ID order."""

    def __init__(self, name: str = "queue") -> None:
        self.name = name
        self._items: list[Task] = []

    def enqueue(self, task: Task) -> None:
        if task.is_terminal:
            raise QueueInvariantError("terminal task cannot be enqueued")
        if self.contains(task.task_id):
            raise QueueInvariantError(f"task {task.task_id} is already in {self.name}")
        self._items.append(task)
        self._items.sort(key=lambda item: item.task_id_key)

    def peek(self) -> Task | None:
        return self._items[0] if self._items else None

    def dequeue(self) -> Task | None:
        return self._items.pop(0) if self._items else None

    def remove(self, task_or_id: Task | int) -> Task:
        task_id = task_or_id.task_id if isinstance(task_or_id, Task) else task_or_id
        for index, task in enumerate(self._items):
            if task.task_id == task_id:
                return self._items.pop(index)
        raise KeyError(f"task {task_id} is not in {self.name}")

    def discard(self, task_or_id: Task | int) -> bool:
        try:
            self.remove(task_or_id)
        except KeyError:
            return False
        return True

    def contains(self, task_or_id: Task | int) -> bool:
        task_id = task_or_id.task_id if isinstance(task_or_id, Task) else task_or_id
        return any(task.task_id == task_id for task in self._items)

    def validate_order(self) -> None:
        expected = sorted(self._items, key=lambda item: item.task_id_key)
        if expected != self._items:
            raise QueueInvariantError(f"{self.name} is not in deterministic EDF order")
        ids = [task.task_id for task in self._items]
        if len(ids) != len(set(ids)):
            raise QueueInvariantError(f"{self.name} contains a duplicate task ID")

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Task]:
        return iter(tuple(self._items))

    def __getitem__(self, index: int) -> Task:
        return self._items[index]

    @property
    def items(self) -> tuple[Task, ...]:
        return tuple(self._items)


class QueueState:
    """Own the four Section 2 active queue families."""

    def __init__(self) -> None:
        self.unbound: DefaultDict[int, TaskQueue] = defaultdict(lambda: TaskQueue("unbound"))
        self.local: DefaultDict[int, TaskQueue] = defaultdict(lambda: TaskQueue("local"))
        self.tx: DefaultDict[tuple[int, int], TaskQueue] = defaultdict(lambda: TaskQueue("tx"))
        self.cpu: DefaultDict[tuple[int, int], TaskQueue] = defaultdict(lambda: TaskQueue("cpu"))

        self.unbound_queues = self.unbound
        self.local_queues = self.local
        self.tx_queues = self.tx
        self.cpu_queues = self.cpu
        self.routing = self.unbound

    def clear(self) -> None:
        self.unbound.clear()
        self.local.clear()
        self.tx.clear()
        self.cpu.clear()

    def enqueue_unbound(self, task: Task) -> None:
        self._ensure_unowned(task)
        if task.status is not TaskStatus.UNBOUND or task.destination is not None:
            raise QueueInvariantError("unbound queue requires an unbound task")
        self.unbound[task.source_uav].enqueue(task)

    def enqueue_local(self, task: Task, executor_uav: int | None = None) -> None:
        self._ensure_unowned(task)
        executor = task.source_uav if executor_uav is None else executor_uav
        if task.status not in {TaskStatus.LOCAL, TaskStatus.CPU}:
            raise QueueInvariantError("local queue requires local or CPU task status")
        if task.destination != executor or task.source_uav != executor:
            raise QueueInvariantError("local queue destination/source ownership mismatch")
        if task.remaining_bits != 0:
            raise QueueInvariantError("local queue task must have zero remaining bits")
        self.local[executor].enqueue(task)

    def enqueue_tx(self, task: Task, source_uav: int | None = None, destination_uav: int | None = None) -> None:
        self._ensure_unowned(task)
        source = task.source_uav if source_uav is None else source_uav
        destination = task.destination if destination_uav is None else destination_uav
        if task.status is not TaskStatus.TX:
            raise QueueInvariantError("tx queue requires tx task status")
        if source != task.source_uav or destination != task.destination or destination == source:
            raise QueueInvariantError("tx queue ownership mismatch")
        self.tx[(source, destination)].enqueue(task)

    def enqueue_cpu(self, task: Task, source_uav: int | None = None, destination_uav: int | None = None) -> None:
        self._ensure_unowned(task)
        source = task.source_uav if source_uav is None else source_uav
        destination = task.destination if destination_uav is None else destination_uav
        if task.status is not TaskStatus.CPU:
            raise QueueInvariantError("CPU queue requires CPU task status")
        if source != task.source_uav or destination != task.destination or destination == source:
            raise QueueInvariantError("remote CPU queue ownership mismatch")
        self.cpu[(source, destination)].enqueue(task)

    def enqueue(self, task: Task, kind: QueueKind | str, *, destination_uav: int | None = None) -> None:
        queue_kind = QueueKind(kind)
        if queue_kind is QueueKind.UNBOUND:
            self.enqueue_unbound(task)
        elif queue_kind is QueueKind.LOCAL:
            self.enqueue_local(task)
        elif queue_kind is QueueKind.TX:
            self.enqueue_tx(task, destination_uav=destination_uav)
        else:
            self.enqueue_cpu(task, destination_uav=destination_uav)

    def peek_unbound(self, source_uav: int) -> Task | None:
        return self.unbound[source_uav].peek()

    def peek_local(self, executor_uav: int) -> Task | None:
        return self.local[executor_uav].peek()

    def peek_tx(self, source_uav: int, destination_uav: int) -> Task | None:
        return self.tx[(source_uav, destination_uav)].peek()

    def peek_cpu(self, source_uav: int, destination_uav: int) -> Task | None:
        return self.cpu[(source_uav, destination_uav)].peek()

    def remove(self, task_or_id: Task | int) -> Task:
        task_id = task_or_id.task_id if isinstance(task_or_id, Task) else task_or_id
        for _, queue in self.iter_queues():
            if queue.contains(task_id):
                return queue.remove(task_id)
        raise KeyError(f"task {task_id} is not owned by an active queue")

    def contains(self, task_or_id: Task | int) -> bool:
        task_id = task_or_id.task_id if isinstance(task_or_id, Task) else task_or_id
        return any(queue.contains(task_id) for _, queue in self.iter_queues())

    def locate(self, task_or_id: Task | int) -> QueueLocation | None:
        task_id = task_or_id.task_id if isinstance(task_or_id, Task) else task_or_id
        for location, queue in self.iter_queues():
            if queue.contains(task_id):
                return location
        return None

    def iter_queues(self) -> Iterator[tuple[QueueLocation, TaskQueue]]:
        for source in sorted(self.unbound):
            yield QueueLocation(QueueKind.UNBOUND, source), self.unbound[source]
        for executor in sorted(self.local):
            yield QueueLocation(QueueKind.LOCAL, executor), self.local[executor]
        for source, destination in sorted(self.tx):
            yield QueueLocation(QueueKind.TX, source, destination), self.tx[(source, destination)]
        for source, destination in sorted(self.cpu):
            yield QueueLocation(QueueKind.CPU, source, destination), self.cpu[(source, destination)]

    def active_tasks(self) -> tuple[Task, ...]:
        tasks: dict[int, Task] = {}
        for _, queue in self.iter_queues():
            for task in queue:
                tasks[task.task_id] = task
        return tuple(tasks[task_id] for task_id in sorted(tasks))

    def validate_invariants(self, tasks: Mapping[int, Task] | Iterable[Task] | None = None) -> None:
        """Validate EDF order, task ownership, and queue/status compatibility."""

        expected_tasks: dict[int, Task] = {}
        if tasks is not None:
            expected_tasks = dict(tasks) if isinstance(tasks, Mapping) else {task.task_id: task for task in tasks}

        seen: dict[int, QueueLocation] = {}
        for location, queue in self.iter_queues():
            queue.validate_order()
            for task in queue:
                if task.task_id in seen:
                    raise QueueInvariantError(
                        f"task {task.task_id} has multiple active queue owners: {seen[task.task_id]} and {location}"
                    )
                seen[task.task_id] = location
                if task.is_terminal:
                    raise QueueInvariantError(f"terminal task {task.task_id} remains in {location}")
                self._validate_task_for_location(task, location)
                if expected_tasks and expected_tasks.get(task.task_id) is not task:
                    raise QueueInvariantError(f"queue task {task.task_id} is not the registered task object")

        for task_id, task in expected_tasks.items():
            if task.is_terminal and task_id in seen:
                raise QueueInvariantError(f"terminal task {task_id} remains in an active queue")
            if not task.is_terminal and task_id not in seen:
                raise QueueInvariantError(f"active task {task_id} has no active queue owner")

    def _ensure_unowned(self, task: Task) -> None:
        location = self.locate(task.task_id)
        if location is not None:
            raise QueueInvariantError(f"task {task.task_id} is already owned by {location}")

    @staticmethod
    def _validate_task_for_location(task: Task, location: QueueLocation) -> None:
        if location.kind is QueueKind.UNBOUND:
            if task.status is not TaskStatus.UNBOUND or task.destination is not None:
                raise QueueInvariantError("unbound queue contains a bound task")
            if task.source_uav != location.source_uav:
                raise QueueInvariantError("unbound queue source mismatch")
        elif location.kind is QueueKind.LOCAL:
            if task.status not in {TaskStatus.LOCAL, TaskStatus.CPU}:
                raise QueueInvariantError("local queue contains a non-local task")
            if task.source_uav != location.source_uav or task.destination != location.source_uav:
                raise QueueInvariantError("local queue source/destination mismatch")
            if task.remaining_bits != 0:
                raise QueueInvariantError("local queue contains a task with remaining bits")
        elif location.kind is QueueKind.TX:
            if task.status is not TaskStatus.TX:
                raise QueueInvariantError("tx queue contains a non-tx task")
            if (task.source_uav, task.destination) != (location.source_uav, location.destination_uav):
                raise QueueInvariantError("tx queue source/destination mismatch")
            if task.destination == task.source_uav:
                raise QueueInvariantError("tx queue cannot contain a local destination")
        else:
            if task.status is not TaskStatus.CPU:
                raise QueueInvariantError("CPU queue contains a non-CPU task")
            if (task.source_uav, task.destination) != (location.source_uav, location.destination_uav):
                raise QueueInvariantError("CPU queue source/destination mismatch")
            if task.destination == task.source_uav:
                raise QueueInvariantError("remote CPU queue cannot contain a local destination")


__all__ = ["QueueInvariantError", "QueueKind", "QueueLocation", "QueueState", "TaskQueue"]
