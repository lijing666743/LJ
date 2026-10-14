"""Gate 0 Task/Queue/Lifecycle subset tests for Implementation 02."""

from __future__ import annotations

import unittest

from src.env.lifecycle import LifecycleError, LifecycleManager
from src.env.queues import QueueInvariantError, QueueState, TaskQueue
from src.env.tasks import Task, TaskIdGenerator, TaskOutcome, TaskStatus


class TaskQueueLifecycleTests(unittest.TestCase):
    def make_manager(self, *, data_bits: float = 10.0, cpu_cycles: float = 20.0, deadline: int = 10) -> LifecycleManager:
        manager = LifecycleManager()
        manager.create_task(
            source_uav=0,
            data_bits=data_bits,
            cpu_cycles=cpu_cycles,
            arrival_slot=0,
            deadline_slot=deadline,
        )
        return manager

    def test_deterministic_task_id_generator_and_reset(self) -> None:
        first = TaskIdGenerator()
        self.assertEqual([first.next_id(), first.next_id(), first.next_id()], [0, 1, 2])
        first.reset()
        self.assertEqual(first.next_id(), 0)

        manager = LifecycleManager()
        tasks = manager.create_arrivals(
            [
                {"source_uav": 1, "data_bits": 1, "cpu_cycles": 1, "arrival_slot": 0, "deadline_slot": 10},
                {"source_uav": 0, "data_bits": 1, "cpu_cycles": 1, "arrival_slot": 0, "deadline_slot": 10},
            ]
        )
        self.assertEqual([(task.task_id, task.source_uav) for task in tasks], [(0, 0), (1, 1)])

    def test_arrival_cannot_route_in_same_slot_but_can_route_next_slot(self) -> None:
        manager = self.make_manager()
        task = manager.get_task(0)
        self.assertFalse(task.can_route(0))
        with self.assertRaises(LifecycleError):
            manager.route_task(task, slot=0, destination="local")
        self.assertTrue(task.can_route(1))
        manager.route_task(task, slot=1, destination="local")
        self.assertEqual(task.status, TaskStatus.LOCAL)

    def test_route_binds_and_locks_destination(self) -> None:
        manager = self.make_manager()
        task = manager.get_task(0)
        manager.route_task(task, slot=1, destination=2)
        self.assertEqual(task.destination, 2)
        self.assertEqual(manager.queues.locate(task).kind.value, "tx")
        with self.assertRaises((LifecycleError, ValueError, RuntimeError)):
            manager.route_task(task, slot=2, destination=3)
        self.assertEqual(task.destination, 2)

    def test_local_route_owns_local_queue_and_clears_remaining_bits(self) -> None:
        manager = self.make_manager()
        task = manager.route_task(0, slot=1, destination="local")
        self.assertEqual(task.destination, task.source_uav)
        self.assertEqual(task.remaining_bits, 0)
        self.assertEqual(manager.queues.locate(task).kind.value, "local")
        self.assertFalse(manager.queues.contains(task.task_id) and manager.queues.tx)

    def test_remote_route_enters_tx_queue(self) -> None:
        manager = self.make_manager()
        task = manager.route_task(0, slot=1, destination=1)
        self.assertEqual(task.status, TaskStatus.TX)
        self.assertTrue(manager.queues.tx[(0, 1)].contains(task))

    def test_queue_is_edf_then_task_id(self) -> None:
        queue = TaskQueue("edf")
        task_late = Task.create(task_id=8, source_uav=0, data_bits=1, cpu_cycles=1, arrival_slot=0, deadline_slot=9)
        task_early = Task.create(task_id=2, source_uav=0, data_bits=1, cpu_cycles=1, arrival_slot=0, deadline_slot=4)
        task_tie = Task.create(task_id=1, source_uav=0, data_bits=1, cpu_cycles=1, arrival_slot=0, deadline_slot=4)
        for task in (task_late, task_early, task_tie):
            queue.enqueue(task)
        self.assertEqual([task.task_id for task in queue], [1, 2, 8])
        self.assertEqual(queue.peek().task_id, 1)
        self.assertEqual(queue.dequeue().task_id, 1)

    def test_route_requires_unbound_edf_head_and_rejection_is_atomic(self) -> None:
        manager = LifecycleManager()
        non_head = manager.create_task(
            source_uav=0, data_bits=10, cpu_cycles=10, arrival_slot=0, deadline_slot=9
        )
        head = manager.create_task(
            source_uav=0, data_bits=10, cpu_cycles=10, arrival_slot=0, deadline_slot=5
        )
        order_before = tuple(task.task_id for task in manager.queues.unbound[0])
        snapshot_before = non_head.snapshot()

        with self.assertRaisesRegex(LifecycleError, "EDF queue head"):
            manager.route_task(non_head, slot=1, destination=1)

        self.assertEqual(non_head.snapshot(), snapshot_before)
        self.assertEqual(tuple(task.task_id for task in manager.queues.unbound[0]), order_before)
        self.assertEqual(manager.queues.locate(non_head).kind.value, "unbound")
        manager.validate_invariants()

        manager.route_task(head, slot=1, destination=1)
        self.assertEqual(head.status, TaskStatus.TX)
        self.assertEqual(manager.queues.locate(head).kind.value, "tx")

    def test_tx_service_requires_edf_head_and_rejection_is_atomic(self) -> None:
        manager = LifecycleManager()
        non_head = manager.create_task(
            source_uav=0, data_bits=10, cpu_cycles=10, arrival_slot=0, deadline_slot=9
        )
        head = manager.create_task(
            source_uav=0, data_bits=10, cpu_cycles=10, arrival_slot=0, deadline_slot=5
        )
        manager.route_task(head, slot=1, destination=1)
        manager.route_task(non_head, slot=1, destination=1)
        order_before = tuple(task.task_id for task in manager.queues.tx[(0, 1)])
        snapshot_before = non_head.snapshot()

        with self.assertRaisesRegex(LifecycleError, "EDF queue head"):
            manager.service_transmission(non_head, slot=2, bits=1)

        self.assertEqual(non_head.snapshot(), snapshot_before)
        self.assertEqual(tuple(task.task_id for task in manager.queues.tx[(0, 1)]), order_before)
        self.assertEqual(manager.queues.locate(non_head).kind.value, "tx")
        manager.validate_invariants()

        manager.service_transmission(head, slot=2, bits=1)
        self.assertEqual(head.remaining_bits, 9)
        self.assertEqual(head.status, TaskStatus.TX)

    def test_local_cpu_service_requires_edf_head_and_rejection_is_atomic(self) -> None:
        manager = LifecycleManager()
        non_head = manager.create_task(
            source_uav=0, data_bits=1, cpu_cycles=10, arrival_slot=0, deadline_slot=9
        )
        head = manager.create_task(
            source_uav=0, data_bits=1, cpu_cycles=10, arrival_slot=0, deadline_slot=5
        )
        manager.route_task(head, slot=1, destination="local")
        manager.route_task(non_head, slot=1, destination="local")
        order_before = tuple(task.task_id for task in manager.queues.local[0])
        snapshot_before = non_head.snapshot()

        with self.assertRaisesRegex(LifecycleError, "EDF queue head"):
            manager.service_cpu(non_head, slot=2, cycles=1)

        self.assertEqual(non_head.snapshot(), snapshot_before)
        self.assertEqual(tuple(task.task_id for task in manager.queues.local[0]), order_before)
        self.assertEqual(manager.queues.locate(non_head).kind.value, "local")
        manager.validate_invariants()

        manager.service_cpu(head, slot=2, cycles=1)
        self.assertEqual(head.remaining_cycles, 9)
        self.assertEqual(head.status, TaskStatus.CPU)

    def test_remote_cpu_service_requires_edf_head_and_rejection_is_atomic(self) -> None:
        manager = LifecycleManager()
        non_head = manager.create_task(
            source_uav=0, data_bits=1, cpu_cycles=10, arrival_slot=0, deadline_slot=9
        )
        head = manager.create_task(
            source_uav=0, data_bits=1, cpu_cycles=10, arrival_slot=0, deadline_slot=5
        )
        manager.route_task(head, slot=1, destination=1)
        manager.route_task(non_head, slot=1, destination=1)
        manager.service_transmission(head, slot=2, bits=1)
        manager.service_transmission(non_head, slot=2, bits=1)
        order_before = tuple(task.task_id for task in manager.queues.cpu[(0, 1)])
        snapshot_before = non_head.snapshot()

        with self.assertRaisesRegex(LifecycleError, "EDF queue head"):
            manager.service_cpu(non_head, slot=3, cycles=1)

        self.assertEqual(non_head.snapshot(), snapshot_before)
        self.assertEqual(tuple(task.task_id for task in manager.queues.cpu[(0, 1)]), order_before)
        self.assertEqual(manager.queues.locate(non_head).kind.value, "cpu")
        manager.validate_invariants()

        manager.service_cpu(head, slot=3, cycles=1)
        self.assertEqual(head.remaining_cycles, 9)
        self.assertEqual(head.status, TaskStatus.CPU)

    def test_single_active_queue_ownership_and_invariant_detection(self) -> None:
        manager = self.make_manager()
        task = manager.route_task(0, slot=1, destination=1)
        with self.assertRaises(QueueInvariantError):
            manager.queues.enqueue_tx(task)
        manager.queues.local[0]._items.append(task)
        with self.assertRaises(QueueInvariantError):
            manager.validate_invariants()

    def test_service_is_rejected_before_first_legal_service_slot(self) -> None:
        manager = self.make_manager()
        task = manager.route_task(0, slot=1, destination=1)
        self.assertFalse(task.can_service(1))
        with self.assertRaises(LifecycleError):
            manager.service_transmission(task, slot=1, bits=1)
        self.assertTrue(task.can_service(2))

    def test_tx_service_and_transfer_completion_are_slot_end_transitions(self) -> None:
        manager = self.make_manager(data_bits=5, cpu_cycles=20)
        task = manager.route_task(0, slot=1, destination=1)
        manager.service_transmission(task, slot=2, bits=5)
        self.assertEqual(task.status, TaskStatus.CPU)
        self.assertEqual(manager.queues.locate(task).kind.value, "cpu")
        self.assertFalse(task.can_service(2))
        self.assertTrue(task.can_service(3))

    def test_workload_bookkeeping_rejects_negative_or_over_service(self) -> None:
        manager = self.make_manager(data_bits=5, cpu_cycles=20)
        task = manager.route_task(0, slot=1, destination=1)
        with self.assertRaises(LifecycleError):
            manager.service_transmission(task, slot=2, bits=-1)
        with self.assertRaises(LifecycleError):
            manager.service_transmission(task, slot=2, bits=6)
        self.assertGreaterEqual(task.remaining_bits, 0)
        manager = self.make_manager(data_bits=1, cpu_cycles=5)
        local = manager.route_task(0, slot=1, destination="local")
        with self.assertRaises(LifecycleError):
            manager.service_cpu(local, slot=2, cycles=-1)
        with self.assertRaises(LifecycleError):
            manager.service_cpu(local, slot=2, cycles=6)
        self.assertGreaterEqual(local.remaining_cycles, 0)

    def test_local_cpu_completion_records_done_and_completion_slot(self) -> None:
        manager = self.make_manager(data_bits=1, cpu_cycles=5)
        task = manager.route_task(0, slot=1, destination="local")
        manager.service_cpu(task, slot=2, cycles=5)
        self.assertEqual(task.status, TaskStatus.CPU)
        manager.settle_slot(2)
        self.assertEqual(task.status, TaskStatus.DONE)
        self.assertEqual(task.completion_slot, 2)
        self.assertEqual(task.e2e_delay(0.02), 0.04)
        self.assertFalse(manager.queues.contains(task))

    def test_terminal_task_cannot_receive_service(self) -> None:
        manager = self.make_manager(data_bits=1, cpu_cycles=1)
        task = manager.route_task(0, slot=1, destination="local")
        manager.service_cpu(task, slot=2, cycles=1)
        manager.settle_slot(2)
        with self.assertRaises(LifecycleError):
            manager.service_cpu(task, slot=3, cycles=1)

    def test_deadline_slot_service_has_completion_priority(self) -> None:
        manager = self.make_manager(data_bits=1, cpu_cycles=5, deadline=2)
        task = manager.route_task(0, slot=1, destination="local")
        self.assertTrue(task.can_service(2))
        manager.service_cpu(task, slot=2, cycles=5)
        manager.settle_slot(2)
        self.assertEqual(task.outcome, TaskOutcome.DONE)
        self.assertNotEqual(task.outcome, TaskOutcome.EXPIRED)

    def test_deadline_slot_unfinished_task_expires_and_cannot_recover(self) -> None:
        manager = self.make_manager(data_bits=1, cpu_cycles=5, deadline=2)
        task = manager.route_task(0, slot=1, destination="local")
        manager.service_cpu(task, slot=2, cycles=4)
        manager.settle_slot(2)
        self.assertEqual(task.status, TaskStatus.EXPIRED)
        self.assertEqual(task.outcome, TaskOutcome.EXPIRED)
        self.assertIsNone(task.completion_slot)
        with self.assertRaises((LifecycleError, ValueError, RuntimeError)):
            task.complete(2)

    def test_done_expired_and_truncated_are_mutually_exclusive(self) -> None:
        manager = self.make_manager(data_bits=1, cpu_cycles=1)
        done = manager.route_task(0, slot=1, destination="local")
        manager.service_cpu(done, slot=2, cycles=1)
        manager.settle_slot(2)
        self.assertEqual(sum(value == TaskOutcome.DONE for value in [done.outcome]), 1)

        expired_manager = self.make_manager(data_bits=1, cpu_cycles=2, deadline=2)
        expired = expired_manager.route_task(0, slot=1, destination="local")
        expired_manager.settle_slot(2)
        self.assertEqual(expired.outcome, TaskOutcome.EXPIRED)

        truncated_manager = self.make_manager(data_bits=1, cpu_cycles=2, deadline=10)
        truncated = truncated_manager.route_task(0, slot=1, destination="local")
        records = truncated_manager.truncate_horizon(3)
        self.assertEqual(records[0].outcome, TaskOutcome.TRUNCATED)
        self.assertTrue(truncated.truncated)
        self.assertNotEqual(truncated.outcome, TaskOutcome.EXPIRED)
        self.assertIsNone(truncated.completion_slot)

    def test_horizon_task_with_deadline_inside_episode_is_expired(self) -> None:
        manager = self.make_manager(data_bits=1, cpu_cycles=2, deadline=2)
        task = manager.get_task(0)
        manager.truncate_horizon(3)
        self.assertEqual(task.status, TaskStatus.EXPIRED)
        self.assertFalse(task.truncated)

    def test_queue_invariant_detects_status_and_ownership_mismatch(self) -> None:
        queue_state = QueueState()
        task = Task.create(task_id=0, source_uav=0, data_bits=1, cpu_cycles=1, arrival_slot=0, deadline_slot=5)
        queue_state.unbound[1].enqueue(task)
        with self.assertRaises(QueueInvariantError):
            queue_state.validate_invariants({0: task})

    def test_same_inputs_reproduce_lifecycle_snapshot(self) -> None:
        def run() -> list[dict[str, object]]:
            manager = LifecycleManager()
            task = manager.create_task(source_uav=0, data_bits=3, cpu_cycles=4, arrival_slot=0, deadline_slot=6)
            manager.route_task(task, slot=1, destination=1)
            manager.service_transmission(task, slot=2, bits=3)
            manager.service_cpu(task, slot=3, cycles=4)
            manager.settle_slot(3)
            return [task.snapshot()]

        self.assertEqual(run(), run())

    def test_defer_keeps_task_unbound_without_service_side_effect(self) -> None:
        manager = self.make_manager()
        task = manager.route_task(0, slot=1, destination="defer")
        self.assertEqual(task.status, TaskStatus.UNBOUND)
        self.assertIsNone(task.destination)
        self.assertEqual(task.remaining_bits, task.data_bits)
        self.assertFalse(task.can_service(2))


if __name__ == "__main__":
    unittest.main()
