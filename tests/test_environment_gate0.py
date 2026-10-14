"""Formal Gate 0 tests for the integrated U2U MEC environment.

Every frozen acceptance item G0-01 through G0-21 has one explicitly named
``unittest`` method.  End-to-end environment transitions are preferred for
episode boundaries and actor-facing contracts; focused physical modules are
used where they provide a stronger, faster invariant check.
"""

from __future__ import annotations

import inspect
import json
import math
import unittest
from dataclasses import replace
from typing import Any

import numpy as np

from src.config import EnvironmentConfig, RunConfig
from src.env.actions import ActionProposal
from src.env.energy import UavEnergyState
from src.env.environment import EnvironmentError, U2UMECEnvironment
from src.env.executor import DeterministicExecutor
from src.env.history import ChannelHistory, HistoryError
from src.env.lifecycle import LifecycleError, LifecycleManager
from src.env.metrics import EpisodeMetrics
from src.env.mobility import MobilityState
from src.env.observation import NeighborPublicFeatures, ObservationBuilder
from src.env.outage import actual_attempt_outage
from src.env.randomness import make_rng, rng_from_run_config
from src.env.service import PhysicalService
from src.env.tasks import TaskOutcome, TaskStatus
from src.env.topology import DynamicTopology


def make_environment_config(
    *,
    horizon: int = 4,
    seed: int = 42,
    arrival_probabilities: tuple[float, ...] | None = None,
    **overrides: Any,
) -> RunConfig:
    """Build a fast deterministic four-UAV config accepted by RunConfig."""

    base = RunConfig()
    arrivals = (
        tuple(0.0 for _ in range(base.environment.uav_count))
        if arrival_probabilities is None
        else arrival_probabilities
    )
    environment_values: dict[str, Any] = {
        "episode_horizon": horizon,
        "arrival_probabilities": arrivals,
        "building_layout": (),
        "candidate_neighbor_radius_m": 2_000.0,
        "velocity_std_mps": (0.0, 0.0),
        "shadowing_std_db": 0.0,
        "csi_error_std_db": 0.0,
        "fixed_csi_aoi_slots": 1,
    }
    environment_values.update(overrides)
    environment = replace(base.environment, **environment_values)
    config = replace(base, seed=seed, environment=environment)
    config.validate()
    return config


def make_physical_config(**overrides: Any) -> RunConfig:
    """Build a validation-clean config with transparent hand-checkable units."""

    return make_environment_config(
        horizon=8,
        slot_duration_s=1.0,
        total_bandwidth_hz=20.0,
        ru_count=20,
        resource_group_count=5,
        noise_psd_dbm_hz=0.0,
        receiver_noise_figure_db=0.0,
        **overrides,
    )


def make_states(
    uav_ids: tuple[int, ...],
    *,
    residual_energy_j: float = 100.0,
    residual_overrides: dict[int, float] | None = None,
) -> dict[int, UavEnergyState]:
    overrides = residual_overrides or {}
    return {
        uav_id: UavEnergyState(
            uav_id=uav_id,
            max_transmit_power_w=1.0,
            max_cpu_frequency_hz=10.0,
            cpu_coefficient=1.0e-3,
            residual_energy_j=overrides.get(uav_id, residual_energy_j),
        )
        for uav_id in uav_ids
    }


def idle_proposal(uav_id: int, **overrides: Any) -> ActionProposal:
    values: dict[str, Any] = {
        "uav_id": uav_id,
        "route": "idle",
        "tx_select": "idle",
        "resource_group": "idle",
        "resource_width": 1,
        "power_level": 0.0,
        "cpu_queue": "idle",
        "cpu_frequency": 0.0,
    }
    values.update(overrides)
    return ActionProposal(**values)


def tx_proposal(
    source: int,
    destination: int,
    *,
    group: int = 1,
    width: int = 1,
    power: float = 1.0,
    **overrides: Any,
) -> ActionProposal:
    return idle_proposal(
        source,
        tx_select=destination,
        resource_group=group,
        resource_width=width,
        power_level=power,
        **overrides,
    )


def true_channel(config: RunConfig, value: complex = 1.0 + 0.0j) -> np.ndarray:
    count = config.environment.uav_count
    channel = np.zeros(
        (count, count, config.environment.ru_count),
        dtype=np.complex128,
    )
    for source in range(count):
        for destination in range(count):
            if source != destination:
                channel[source, destination, :] = value
    return channel


def add_tx_task(
    lifecycle: LifecycleManager,
    source: int,
    destination: int,
    *,
    bits: float = 100.0,
    cycles: float = 100.0,
    deadline: int = 10,
):
    task = lifecycle.create_task(
        source_uav=source,
        data_bits=bits,
        cpu_cycles=cycles,
        arrival_slot=0,
        deadline_slot=deadline,
    )
    lifecycle.route_task(task, slot=1, destination=destination)
    return task


def add_local_task(
    lifecycle: LifecycleManager,
    uav_id: int,
    *,
    bits: float = 1.0,
    cycles: float = 100.0,
    deadline: int = 10,
):
    task = lifecycle.create_task(
        source_uav=uav_id,
        data_bits=bits,
        cpu_cycles=cycles,
        arrival_slot=0,
        deadline_slot=deadline,
    )
    lifecycle.route_task(task, slot=1, destination="local")
    return task


def run_partial_parallel_service():
    """Run authentic simultaneous TX/CPU service and conservation accounting."""

    config = make_physical_config()
    lifecycle = LifecycleManager()
    local = lifecycle.create_task(
        source_uav=0,
        data_bits=1.0,
        cpu_cycles=100.0,
        arrival_slot=0,
        deadline_slot=5,
    )
    remote = lifecycle.create_task(
        source_uav=0,
        data_bits=100.0,
        cpu_cycles=100.0,
        arrival_slot=0,
        deadline_slot=6,
    )
    metrics = EpisodeMetrics()
    metrics.record_generated((local, remote))
    lifecycle.route_task(local, slot=1, destination="local")
    lifecycle.route_task(remote, slot=1, destination=1)
    metrics.record_local_bindings((local,))
    states = make_states((0,))
    proposal = tx_proposal(
        0,
        1,
        cpu_queue="local",
        cpu_frequency=1.0,
    )
    execution = DeterministicExecutor(config, lifecycle, states).execute(2, (proposal,))
    physical = PhysicalService(config, lifecycle, states).execute(
        execution,
        true_channel(config),
        settle_deadlines=False,
    )
    metrics.record_physical_result(physical)
    settled = lifecycle.settle_slot(2)
    metrics.record_settled_tasks(
        settled,
        slot_duration_s=config.environment.slot_duration_s,
    )
    return config, lifecycle, states, execution, physical, metrics, local, remote


def run_single_transmission(power: float):
    """Run one real proposed transmission and record its authentic metrics."""

    config = make_physical_config()
    lifecycle = LifecycleManager()
    task = lifecycle.create_task(
        source_uav=0,
        data_bits=100.0,
        cpu_cycles=10.0,
        arrival_slot=0,
        deadline_slot=10,
    )
    metrics = EpisodeMetrics()
    metrics.record_generated((task,))
    lifecycle.route_task(task, slot=1, destination=1)
    states = make_states((0,))
    proposal = tx_proposal(0, 1, power=power)
    execution = DeterministicExecutor(config, lifecycle, states).execute(2, (proposal,))
    physical = PhysicalService(config, lifecycle, states).execute(
        execution,
        true_channel(config),
        settle_deadlines=False,
    )
    metrics.record_physical_result(physical)
    return task, execution, physical, metrics


def nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(nested_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            keys.update(nested_keys(child))
    return keys


class EnvironmentGate0Tests(unittest.TestCase):
    def test_g0_01_reset_reproducibility(self) -> None:
        environment = U2UMECEnvironment(make_environment_config(seed=73))
        first = environment.reset()
        first_snapshot = environment.snapshot()
        first_public = environment.public_messages.snapshot()
        first_previous = environment.previous_actions.snapshot()

        second = environment.reset()
        second_snapshot = environment.snapshot()

        self.assertEqual(first_snapshot, second_snapshot)
        self.assertEqual(first_public, environment.public_messages.snapshot())
        self.assertEqual(first_previous, environment.previous_actions.snapshot())
        self.assertEqual(
            [item.snapshot() for item in first.observations],
            [item.snapshot() for item in second.observations],
        )
        self.assertEqual(first.centralized_state.snapshot(), second.centralized_state.snapshot())
        json.dumps(second_snapshot, allow_nan=False, sort_keys=True)

    def test_g0_02_deterministic_seed_ownership(self) -> None:
        config = make_environment_config(seed=91)
        workload = rng_from_run_config(config, "task_workload")
        prefix = workload.random(8)
        rng_from_run_config(config, "task_arrival").random(10_000)
        suffix = workload.random(8)
        uninterrupted = rng_from_run_config(config, "task_workload").random(16)
        np.testing.assert_array_equal(np.concatenate((prefix, suffix)), uninterrupted)

        for stream_name in U2UMECEnvironment._STREAM_NAMES:
            with self.subTest(stream=stream_name):
                first = rng_from_run_config(config, stream_name).normal(size=12)
                second = rng_from_run_config(config, stream_name).normal(size=12)
                np.testing.assert_array_equal(first, second)
        self.assertFalse(
            np.array_equal(
                rng_from_run_config(config, "reset_mobility").random(12),
                rng_from_run_config(config, "channel_fading").random(12),
            )
        )

    def test_g0_03_task_lifecycle(self) -> None:
        lifecycle = LifecycleManager()
        task = lifecycle.create_task(
            source_uav=0,
            data_bits=4.0,
            cpu_cycles=5.0,
            arrival_slot=0,
            deadline_slot=6,
        )
        self.assertEqual(task.status, TaskStatus.UNBOUND)
        lifecycle.route_task(task, slot=1, destination=1)
        self.assertEqual(task.status, TaskStatus.TX)
        lifecycle.service_transmission(task, slot=2, bits=4.0)
        self.assertEqual(task.status, TaskStatus.CPU)
        lifecycle.service_cpu(task, slot=3, cycles=5.0)
        lifecycle.settle_slot(3)
        self.assertEqual(task.status, TaskStatus.DONE)
        self.assertEqual(task.outcome, TaskOutcome.DONE)
        self.assertFalse(lifecycle.queues.contains(task.task_id))
        with self.assertRaises((LifecycleError, RuntimeError, ValueError)):
            lifecycle.route_task(task, slot=4, destination=2)
        with self.assertRaises((LifecycleError, RuntimeError, ValueError)):
            lifecycle.service_cpu(task, slot=4, cycles=0.0)
        with self.assertRaises(RuntimeError):
            task.expire(4)

    def test_g0_04_arrival_route_service_timing(self) -> None:
        arrivals = (1.0, 0.0, 0.0, 0.0)
        environment = U2UMECEnvironment(
            make_environment_config(horizon=4, arrival_probabilities=arrivals)
        )
        environment.reset()
        environment.step(environment.canonical_proposals())
        task = environment.lifecycle.get_task(0)
        initial_cycles = task.remaining_cycles
        self.assertEqual(task.arrival_slot, 0)
        self.assertEqual(task.status, TaskStatus.UNBOUND)
        self.assertTrue(task.can_route(1))
        self.assertFalse(task.can_service(1))

        route = list(environment.canonical_proposals())
        route[0] = replace(route[0], route="local")
        environment.step(route)
        self.assertEqual(task.status, TaskStatus.LOCAL)
        self.assertEqual(task.binding_slot, 1)
        self.assertEqual(task.service_eligible_slot, 2)
        self.assertEqual(task.remaining_cycles, initial_cycles)

        service = list(environment.canonical_proposals())
        service[0] = replace(service[0], cpu_queue="local", cpu_frequency=0.25)
        environment.step(service)
        self.assertLess(task.remaining_cycles, initial_cycles)

    def test_g0_05_hard_deadline(self) -> None:
        completed_manager = LifecycleManager()
        completed = completed_manager.create_task(
            source_uav=0,
            data_bits=1.0,
            cpu_cycles=5.0,
            arrival_slot=0,
            deadline_slot=2,
        )
        completed_manager.route_task(completed, slot=1, destination="local")
        self.assertTrue(completed.can_service(2))
        completed_manager.service_cpu(completed, slot=2, cycles=5.0)
        completed_manager.settle_slot(2)
        self.assertEqual(completed.outcome, TaskOutcome.DONE)
        self.assertEqual(completed.completion_slot, 2)

        expired_manager = LifecycleManager()
        expired = expired_manager.create_task(
            source_uav=0,
            data_bits=1.0,
            cpu_cycles=5.0,
            arrival_slot=0,
            deadline_slot=2,
        )
        expired_manager.route_task(expired, slot=1, destination="local")
        expired_manager.service_cpu(expired, slot=2, cycles=4.0)
        expired_manager.settle_slot(2)
        self.assertEqual(expired.outcome, TaskOutcome.EXPIRED)
        self.assertIsNone(expired.completion_slot)
        with self.assertRaises(LifecycleError):
            expired_manager.service_cpu(expired, slot=3, cycles=1.0)

    def test_g0_06_done_expired_mutual_exclusivity(self) -> None:
        done_manager = LifecycleManager()
        done = done_manager.create_task(
            source_uav=0,
            data_bits=1.0,
            cpu_cycles=1.0,
            arrival_slot=0,
            deadline_slot=2,
        )
        done_manager.route_task(done, slot=1, destination="local")
        done_manager.service_cpu(done, slot=2, cycles=1.0)
        done_manager.settle_slot(2)

        expired_manager = LifecycleManager()
        expired = expired_manager.create_task(
            source_uav=0,
            data_bits=1.0,
            cpu_cycles=2.0,
            arrival_slot=0,
            deadline_slot=2,
        )
        expired_manager.route_task(expired, slot=1, destination="local")
        expired_manager.settle_slot(2)

        self.assertEqual(done.outcome, TaskOutcome.DONE)
        self.assertNotEqual(done.outcome, TaskOutcome.EXPIRED)
        self.assertEqual(expired.outcome, TaskOutcome.EXPIRED)
        self.assertNotEqual(expired.outcome, TaskOutcome.DONE)
        with self.assertRaises(RuntimeError):
            done.expire(2)
        with self.assertRaises(RuntimeError):
            expired.complete(2)

    def test_g0_07_truncated_semantics(self) -> None:
        arrivals = (1.0, 0.0, 0.0, 0.0)
        environment = U2UMECEnvironment(
            make_environment_config(horizon=1, arrival_probabilities=arrivals)
        )
        environment.reset()
        result = environment.step(environment.canonical_proposals())
        metrics = result.info["metrics"]

        self.assertFalse(result.terminated)
        self.assertTrue(result.truncated)
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.info["reward"]["completed_task_count"], 0)
        self.assertEqual(result.info["reward"]["expired_task_count"], 0)
        self.assertEqual(metrics["generated_task_count"], 1)
        self.assertEqual(metrics["completed_task_count"], 0)
        self.assertEqual(metrics["expired_task_count"], 0)
        self.assertEqual(metrics["truncated_task_count"], 1)
        task = environment.lifecycle.get_task(0)
        self.assertEqual(task.arrival_slot, 0)
        self.assertEqual(task.outcome, TaskOutcome.TRUNCATED)
        self.assertIsNone(task.completion_slot)

    def test_g0_08_no_fake_terminal_state(self) -> None:
        environment = U2UMECEnvironment(make_environment_config(horizon=1))
        environment.reset()
        result = environment.step(environment.canonical_proposals())

        self.assertTrue(result.truncated)
        self.assertIsNone(result.observations)
        self.assertIsNone(result.centralized_state)
        self.assertIsNone(environment.current_observations)
        self.assertIsNone(environment.current_state)
        self.assertIsNone(environment.channel_features)
        self.assertEqual(environment.slot, 1)
        self.assertEqual(len(environment.metrics.slot_records), 1)
        self.assertEqual(result.info["slot"], 0)
        self.assertIsNone(result.info["next_slot"])
        self.assertIsNone(result.info["next_decision_slot"])
        self.assertFalse(result.info["bootstrap_allowed"])
        self.assertTrue(result.info["truncated"])
        with self.assertRaises(EnvironmentError):
            environment.step(environment.canonical_proposals())

    def test_g0_09_destination_locking_beyond_neighbor_range(self) -> None:
        arrivals = (1.0, 0.0, 0.0, 0.0)
        environment = U2UMECEnvironment(
            make_environment_config(horizon=4, arrival_probabilities=arrivals)
        )
        environment.reset()
        environment.step(environment.canonical_proposals())
        route = list(environment.canonical_proposals())
        route[0] = replace(route[0], route=1)
        environment.step(route)
        task = environment.lifecycle.get_task(0)
        self.assertEqual(task.destination, 1)
        self.assertEqual(task.status, TaskStatus.TX)

        positions = environment.mobility.positions_m.copy()
        positions[1, 0] = positions[0, 0] + 3_000.0
        positions[1, 1] = positions[0, 1]
        far_mobility = MobilityState(
            slot=environment.slot,
            positions_m=positions,
            velocities_mps=environment.mobility.velocities_mps,
        )
        far_topology = DynamicTopology(
            environment.config.environment.candidate_neighbor_radius_m
        ).compute(environment.slot, positions)
        self.assertFalse(far_topology.candidate_neighbors[0, 1])

        estimate = environment.traffic.estimate(environment.slot)
        far_observation = environment.observation_builder.build(
            slot=environment.slot,
            uav_id=0,
            mobility=far_mobility,
            topology=far_topology,
            lifecycle=environment.lifecycle,
            resource_states=environment.resource_states,
            initial_energy_j=environment.initial_energy_j,
            channel_features=environment.channel_features,
            public_messages=environment.public_messages,
            previous_actions=environment.previous_actions,
            history_source_slot=environment.slot - 1,
            arrival_rate_estimates=estimate.values,
            arrival_rate_valid_mask=estimate.valid_mask,
            last_effective_rate_bps=environment.last_effective_rate_bps,
            last_rate_valid_mask=environment.last_rate_valid_mask,
            outage_rate=environment.outage_rate,
            outage_valid_mask=environment.outage_valid_mask,
        )
        self.assertTrue(far_observation.service_queue_available[1])
        self.assertTrue(far_observation.edge_history.visible_mask[1])
        self.assertEqual(far_observation.edge_history.estimated_distance_m[1], 0.0)
        proposal = replace(
            environment.canonical_proposals()[0],
            tx_select=1,
            resource_group=1,
            resource_width=1,
            power_level=1.0,
        )
        self.assertTrue(far_observation.action_masks.is_legal(proposal))
        execution = DeterministicExecutor(
            environment.config,
            environment.lifecycle,
            environment.resource_states,
        ).execute(environment.slot, (proposal,), actor_features=environment.channel_features)
        self.assertTrue(execution.for_uav(0).communication.accepted)
        self.assertEqual(task.destination, 1)
        with self.assertRaises(LifecycleError):
            environment.lifecycle.route_task(task, slot=environment.slot, destination=2)

    def test_g0_10_queue_bookkeeping(self) -> None:
        lifecycle = LifecycleManager()
        task = add_tx_task(lifecycle, 0, 1, bits=5.0, cycles=7.0)
        self.assertEqual(lifecycle.queues.locate(task).kind.value, "tx")
        lifecycle.service_transmission(task, slot=2, bits=5.0)
        self.assertEqual(task.status, TaskStatus.CPU)
        self.assertEqual(lifecycle.queues.locate(task).kind.value, "cpu")
        self.assertFalse(task.can_service(2))
        self.assertTrue(task.can_service(3))
        self.assertEqual(task.remaining_cycles, 7.0)
        lifecycle.service_cpu(task, slot=3, cycles=7.0)
        lifecycle.settle_slot(3)
        self.assertEqual(task.outcome, TaskOutcome.DONE)
        self.assertFalse(lifecycle.queues.contains(task.task_id))
        lifecycle.validate_invariants()

    def test_g0_11_bit_conservation(self) -> None:
        *_, metrics, local, remote = run_partial_parallel_service()[3:]
        conservation = metrics.assert_conservation()
        self.assertTrue(conservation.bit_conserved)
        self.assertGreater(metrics.actual_served_bits, 0.0)
        self.assertEqual(metrics.local_binding_cleared_bits, local.data_bits)
        self.assertGreater(remote.remaining_bits, 0.0)
        self.assertAlmostEqual(
            conservation.generated_bits,
            conservation.actual_served_bits
            + conservation.local_binding_cleared_bits
            + conservation.active_remaining_bits
            + conservation.terminal_unfinished_bits,
        )

    def test_g0_12_cpu_cycle_conservation(self) -> None:
        fixture = run_partial_parallel_service()
        metrics = fixture[5]
        conservation = metrics.assert_conservation()
        self.assertTrue(conservation.cycle_conserved)
        self.assertGreater(metrics.actual_served_cycles, 0.0)
        self.assertAlmostEqual(
            conservation.generated_cycles,
            conservation.actual_served_cycles
            + conservation.active_remaining_cycles
            + conservation.terminal_unfinished_cycles,
        )

    def test_g0_13_tx_energy_uses_executed_power_and_actual_duration(self) -> None:
        config = make_physical_config()
        lifecycle = LifecycleManager()
        add_tx_task(lifecycle, 0, 1, bits=1.0, cycles=10.0)
        states = make_states((0,), residual_energy_j=2.0)
        before = states[0].residual_energy_j
        proposal = tx_proposal(0, 1, power=1.0)
        execution = DeterministicExecutor(config, lifecycle, states).execute(2, (proposal,))
        physical = PhysicalService(config, lifecycle, states).execute(
            execution,
            true_channel(config),
            settle_deadlines=False,
        )
        link = physical.links[0]
        debit = physical.energy_debits[0]
        executed_power = execution.for_uav(0).communication.executed_power_w

        self.assertLess(link.active_duration_s, config.environment.slot_duration_s)
        self.assertAlmostEqual(link.transmit_energy_j, executed_power * link.active_duration_s)
        self.assertAlmostEqual(debit.transmit_energy_j, link.transmit_energy_j)
        self.assertLess(debit.transmit_energy_j, debit.reserved_energy_j)
        self.assertAlmostEqual(before - states[0].residual_energy_j, debit.total_energy_j)

    def test_g0_14_cpu_energy_uses_executed_frequency_and_actual_duration(self) -> None:
        config = make_physical_config()
        lifecycle = LifecycleManager()
        add_local_task(lifecycle, 0, cycles=5.0)
        states = make_states((0,), residual_energy_j=2.0)
        before = states[0].residual_energy_j
        proposal = idle_proposal(0, cpu_queue="local", cpu_frequency=1.0)
        execution = DeterministicExecutor(config, lifecycle, states).execute(2, (proposal,))
        physical = PhysicalService(config, lifecycle, states).execute(
            execution,
            true_channel(config),
        )
        service = physical.cpu_services[0]
        debit = physical.energy_debits[0]
        expected = (
            states[0].cpu_coefficient
            * service.executed_frequency_hz**3
            * service.active_duration_s
        )

        self.assertEqual(service.executed_frequency_hz, 10.0)
        self.assertEqual(service.active_duration_s, 0.5)
        self.assertAlmostEqual(service.cpu_energy_j, expected)
        self.assertAlmostEqual(debit.cpu_energy_j, expected)
        self.assertAlmostEqual(before - states[0].residual_energy_j, debit.total_energy_j)

    def test_g0_15_executor_deterministic_ordering(self) -> None:
        config = make_physical_config()
        lifecycle = LifecycleManager()
        add_tx_task(lifecycle, 0, 1, deadline=3)
        add_tx_task(lifecycle, 2, 1, deadline=5)
        states = make_states((0, 2))
        proposals = (tx_proposal(2, 1), tx_proposal(0, 1))
        executor = DeterministicExecutor(config, lifecycle, states)
        first = executor.execute(2, proposals)
        second = executor.execute(2, reversed(proposals))

        self.assertEqual(first, second)
        self.assertTrue(first.for_uav(0).communication.accepted)
        self.assertEqual(
            first.for_uav(2).communication.rejection_reason,
            "half_duplex_conflict",
        )

    def test_g0_16_proposal_and_executed_action_are_separate(self) -> None:
        config = make_physical_config()
        lifecycle = LifecycleManager()
        add_tx_task(lifecycle, 0, 1)
        states = make_states((0,), residual_overrides={0: 0.6})
        proposal = tx_proposal(0, 1, power=1.0)
        before = proposal.branches
        action = DeterministicExecutor(config, lifecycle, states).execute(
            2,
            (proposal,),
        ).for_uav(0)

        self.assertIs(action.proposal, proposal)
        self.assertEqual(proposal.branches, before)
        self.assertEqual(proposal.power_level, 1.0)
        self.assertEqual(action.communication.proposed_power_w, 1.0)
        self.assertEqual(action.communication.executed_power_w, 0.5)
        self.assertEqual(
            action.communication.downgrade_reason,
            "energy_power_downgrade",
        )

    def test_g0_17_zero_power_is_physical_idle(self) -> None:
        task, execution, physical, metrics = run_single_transmission(0.0)
        communication = execution.for_uav(0).communication

        self.assertEqual(communication.executed_power_w, 0.0)
        self.assertEqual(communication.executed_ru_indices, ())
        self.assertFalse(communication.accepted)
        self.assertEqual(communication.canonicalization_reason, "raw_zero_power")
        self.assertEqual(physical.links, ())
        self.assertFalse(np.any(physical.interference_measurement_mask))
        self.assertTrue(np.all(physical.interference_measurement_w == 0.0))
        self.assertEqual(physical.energy_debits[0].transmit_energy_j, 0.0)
        self.assertEqual(task.remaining_bits, task.data_bits)
        self.assertEqual(metrics.actual_attempt_count, 0)
        self.assertIsNone(metrics.outage_ratio)

    def test_g0_18_half_duplex_and_cpu_radio_parallelism(self) -> None:
        fixture = run_partial_parallel_service()
        execution = fixture[3]
        physical = fixture[4]
        senders = {link.sender_uav for link in execution.executed_links}
        receivers = {link.receiver_uav for link in execution.executed_links}

        self.assertTrue(senders.isdisjoint(receivers))
        self.assertEqual(senders, {0})
        self.assertEqual(receivers, {1})
        self.assertGreater(physical.links[0].service_bits, 0.0)
        cpu = next(item for item in physical.cpu_services if item.executor_uav == 0)
        self.assertGreater(cpu.service_cycles, 0.0)
        self.assertGreater(cpu.cpu_energy_j, 0.0)

    def test_g0_19_actor_observation_has_no_current_truth_leakage(self) -> None:
        arrivals = (1.0, 0.0, 0.0, 0.0)
        environment = U2UMECEnvironment(
            make_environment_config(horizon=5, arrival_probabilities=arrivals)
        )
        reset = environment.reset()
        for observation in reset.observations:
            self.assertFalse(np.any(observation.neighbor_public.valid_mask))
            self.assertTrue(np.all(observation.neighbor_public.message_source_slots == -1))
            self.assertTrue(np.all(observation.neighbor_public.message_aoi_slots == -1))
            self.assertTrue(np.all(observation.neighbor_public.relative_position_m == 0.0))

        environment.step(environment.canonical_proposals())
        route = list(environment.canonical_proposals())
        route[0] = replace(route[0], route=1)
        environment.step(route)
        transmit = list(environment.canonical_proposals())
        transmit[0] = replace(
            transmit[0],
            tx_select=1,
            resource_group=1,
            resource_width=1,
            power_level=1.0,
        )
        step = environment.step(transmit)
        assert step.observations is not None
        public = environment.public_messages
        self.assertEqual(public.valid_mask.tolist(), [False, True, False, False])
        self.assertEqual(public.source_slots.tolist(), [-1, 2, -1, -1])
        self.assertEqual(public.aoi_slots.tolist(), [-1, 1, -1, -1])
        actor = step.observations[0]
        self.assertTrue(actor.neighbor_public.valid_mask[1])
        self.assertEqual(actor.neighbor_public.message_source_slots[1], 2)
        self.assertEqual(actor.neighbor_public.message_aoi_slots[1], 1)
        self.assertEqual(actor.neighbor_public.uav_ids[2], -1)
        self.assertEqual(actor.neighbor_public.cpu_load_task_count[2], 0)

        expected_neighbor_fields = {
            "valid_mask",
            "uav_ids",
            "paper_uav_ids",
            "relative_position_m",
            "relative_velocity_mps",
            "residual_energy_j",
            "residual_energy_ratio",
            "max_transmit_power_ratio",
            "max_cpu_frequency_ratio",
            "cpu_coefficient_ratio",
            "cpu_load_task_count",
            "cpu_load_remaining_cycles",
            "message_source_slots",
            "message_aoi_slots",
            "message_aoi_valid_mask",
        }
        self.assertEqual(
            set(NeighborPublicFeatures.__dataclass_fields__),
            expected_neighbor_fields,
        )
        forbidden_keys = {
            "true_channel",
            "true_channel_real",
            "true_channel_imag",
            "sinr",
            "sinr_linear",
            "interference_measurement_w",
            "interference_measurement_mask",
            "historical_denominator_w",
            "executor_historical_quality",
            "true_arrival_probabilities",
        }
        self.assertTrue(forbidden_keys.isdisjoint(nested_keys(actor.snapshot())))
        signature = inspect.signature(ObservationBuilder.build).parameters
        self.assertTrue(
            {
                "channel",
                "sinr",
                "interference_measurement",
                "execution_result",
                "future_state",
            }.isdisjoint(signature)
        )

        actor_before = actor.snapshot()
        state_before = step.centralized_state
        environment.resource_states[1].residual_energy_j -= 0.5
        changed_channel = environment.channel.channel.copy()
        changed_channel *= 3.0
        environment.channel = replace(environment.channel, channel=changed_channel)
        rebuilt_observations, rebuilt_state = environment._build_decision_records(
            environment.traffic.estimate(environment.slot)
        )
        self.assertEqual(rebuilt_observations[0].snapshot(), actor_before)
        self.assertFalse(np.array_equal(rebuilt_state.true_channel, state_before.true_channel))

    def test_g0_20_historical_csi_and_interference_are_causal(self) -> None:
        config = make_physical_config(
            fixed_csi_aoi_slots=1,
            csi_error_std_db=0.0,
            interference_ema_beta=0.0,
        )
        history = ChannelHistory(config.environment, make_rng(config.seed, 50))
        channel_0 = true_channel(config, 2.0 + 0.0j)
        channel_1 = true_channel(config, 9.0 + 0.0j)
        history.record_physical_channel(0, channel_0)
        slot_0 = history.actor_features(0)
        self.assertFalse(np.any(slot_0.csi_valid_mask))
        self.assertTrue(np.all(slot_0.stale_csi == 0.0))

        measurement = np.zeros(
            (config.environment.uav_count, config.environment.ru_count),
            dtype=np.float64,
        )
        mask = np.zeros_like(measurement, dtype=np.bool_)
        measurement[1, 0] = 3.0
        mask[1, 0] = True
        history.update_with_measurement(0, measurement, mask)
        with self.assertRaises(HistoryError):
            history.actor_features(0)

        history.record_physical_channel(1, channel_1)
        slot_1 = history.actor_features(1)
        np.testing.assert_array_equal(slot_1.stale_csi[0, 1], channel_0[0, 1])
        self.assertFalse(np.array_equal(slot_1.stale_csi[0, 1], channel_1[0, 1]))
        self.assertEqual(slot_1.interference_history_w[1, 0], 3.0)
        self.assertTrue(slot_1.interference_valid_mask[1, 0])
        self.assertEqual(slot_1.message_aoi_slots[1], 1)

    def test_g0_21_outage_denominator_and_na_semantics(self) -> None:
        empty_metrics = EpisodeMetrics()
        self.assertIsNone(empty_metrics.outage_ratio)

        base = {
            "accepted": True,
            "executed_power_w": 1.0,
            "executed_ru_indices": (1,),
            "slot_start_queue_bits": 1.0,
            "effective_rate_bps": 1.0,
            "sinr_linear": (1.0,),
            "threshold_linear": 0.5,
        }
        for override in (
            {"accepted": False},
            {"executed_power_w": 0.0},
            {"executed_ru_indices": ()},
            {"slot_start_queue_bits": 0.0},
        ):
            values = dict(base)
            values.update(override)
            outage = actual_attempt_outage(**values)
            self.assertFalse(outage.attempted)
            self.assertIsNone(outage.sample)

        _, _, zero_physical, zero_metrics = run_single_transmission(0.0)
        self.assertEqual(zero_physical.links, ())
        self.assertEqual(zero_metrics.actual_attempt_count, 0)
        self.assertIsNone(zero_metrics.outage_ratio)

        _, _, attempted_physical, attempted_metrics = run_single_transmission(1.0)
        self.assertEqual(len(attempted_physical.links), 1)
        self.assertTrue(attempted_physical.links[0].outage.attempted)
        self.assertEqual(attempted_metrics.actual_attempt_count, 1)
        self.assertEqual(
            attempted_metrics.outage_ratio,
            float(attempted_physical.links[0].outage.sample),
        )


if __name__ == "__main__":
    unittest.main()
