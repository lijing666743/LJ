"""Gate 0 Executor/Service/Energy/Outage subset tests for Implementation 04."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np

from src.config import ChannelAblationMode, EnvironmentConfig, RunConfig
from src.env.actions import ActionError, ActionProposal, ProposalAdapter
from src.env.energy import UavEnergyState
from src.env.executor import DeterministicExecutor
from src.env.history import ActorChannelFeatures
from src.env.lifecycle import LifecycleManager
from src.env.outage import actual_attempt_outage
from src.env.service import PhysicalService
from src.env.tasks import TaskOutcome, TaskStatus


def make_config(uav_count: int = 4, **environment_overrides: object) -> RunConfig:
    environment = replace(
        EnvironmentConfig(),
        uav_count=uav_count,
        arrival_probabilities=tuple(0.0 for _ in range(uav_count)),
        profile_assignment=tuple("Balanced" for _ in range(uav_count)),
        slot_duration_s=1.0,
        total_bandwidth_hz=20.0,
        ru_count=20,
        resource_group_count=5,
        noise_psd_dbm_hz=0.0,
        receiver_noise_figure_db=0.0,
        **environment_overrides,
    )
    return replace(RunConfig(), environment=environment)


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


def idle_proposal(uav_id: int, **overrides: object) -> ActionProposal:
    values: dict[str, object] = {
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


def add_tx_task(
    manager: LifecycleManager,
    source: int,
    destination: int,
    *,
    bits: float = 20.0,
    cycles: float = 10.0,
    deadline: int = 10,
):
    task = manager.create_task(
        source_uav=source,
        data_bits=bits,
        cpu_cycles=cycles,
        arrival_slot=0,
        deadline_slot=deadline,
    )
    manager.route_task(task, slot=1, destination=destination)
    return task


def add_local_task(
    manager: LifecycleManager,
    uav_id: int,
    *,
    cycles: float = 10.0,
    deadline: int = 10,
):
    task = manager.create_task(
        source_uav=uav_id,
        data_bits=1.0,
        cpu_cycles=cycles,
        arrival_slot=0,
        deadline_slot=deadline,
    )
    manager.route_task(task, slot=1, destination="local")
    return task


def tx_proposal(
    source: int,
    destination: int,
    *,
    group: int = 1,
    width: int = 1,
    power: float = 1.0,
    **overrides: object,
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
    tensor = np.zeros((count, count, config.environment.ru_count), dtype=np.complex128)
    for source in range(count):
        for destination in range(count):
            if source != destination:
                tensor[source, destination, :] = value
    return tensor


def actor_features(config: RunConfig, qualities: dict[tuple[int, int], float]) -> ActorChannelFeatures:
    count = config.environment.uav_count
    ru_count = config.environment.ru_count
    stale = np.ones((count, count, ru_count), dtype=np.complex128)
    for uav_id in range(count):
        stale[uav_id, uav_id, :] = 0.0
    link_mask = ~np.eye(count, dtype=bool)
    quality = np.zeros((count, count, ru_count), dtype=np.float64)
    for (source, destination), value in qualities.items():
        quality[source, destination, :] = value
    quality_mask = np.broadcast_to(link_mask[:, :, None], quality.shape).copy()
    return ActorChannelFeatures(
        slot=2,
        stale_csi=stale,
        csi_valid_mask=link_mask,
        csi_aoi_slots=np.zeros((count, count), dtype=np.int64),
        interference_history_w=np.zeros((count, ru_count)),
        interference_valid_mask=np.ones((count, ru_count), dtype=bool),
        message_aoi_slots=np.ones(count, dtype=np.int64),
        message_aoi_valid_mask=np.ones(count, dtype=bool),
        historical_denominator_w=np.ones((count, ru_count)),
        historical_quality=quality,
        quality_valid_mask=quality_mask,
    )


class ProposalAdapterTests(unittest.TestCase):
    def test_resource_mapping_power_stages_and_proposal_preservation(self) -> None:
        config = make_config()
        manager = LifecycleManager()
        add_tx_task(manager, 0, 1)
        states = make_states((0,))
        proposal = tx_proposal(0, 1, group=2, width=2, power=0.5)
        result = DeterministicExecutor(config, manager, states).execute(2, [proposal])
        action = result.for_uav(0)

        self.assertIs(action.proposal, proposal)
        self.assertEqual(action.communication.proposed_ru_indices, tuple(range(5, 13)))
        self.assertEqual(action.communication.executed_ru_indices, tuple(range(5, 13)))
        self.assertEqual(action.communication.proposed_power_w, 0.5)
        self.assertEqual(action.communication.candidate_power_w, 0.5)
        self.assertEqual(action.communication.executed_power_w, 0.5)

    def test_malformed_resource_and_inactive_combinations_are_rejected(self) -> None:
        config = make_config()
        manager = LifecycleManager()
        add_tx_task(manager, 0, 1)
        states = make_states((0,))
        executor = DeterministicExecutor(config, manager, states)
        invalid = (
            tx_proposal(0, 1, group=0),
            tx_proposal(0, 1, group=5, width=2),
            tx_proposal(0, 1, width=3),
            tx_proposal(0, 1, power=0.3),
            idle_proposal(0, resource_group=1),
            idle_proposal(0, cpu_frequency=0.5),
        )
        for proposal in invalid:
            with self.subTest(proposal=proposal):
                with self.assertRaises(ActionError):
                    executor.execute(2, [proposal])

    def test_raw_zero_power_is_physical_idle_and_does_not_occupy_half_duplex(self) -> None:
        config = make_config(3)
        manager = LifecycleManager()
        add_tx_task(manager, 0, 1, deadline=3)
        add_tx_task(manager, 1, 2, deadline=4)
        states = make_states((0, 1))
        proposals = [tx_proposal(0, 1, power=0.0), tx_proposal(1, 2)]
        result = DeterministicExecutor(config, manager, states).execute(2, proposals)
        zero = result.for_uav(0).communication
        other = result.for_uav(1).communication

        self.assertEqual(zero.proposed_ru_indices, (1, 2, 3, 4))
        self.assertEqual(zero.executed_ru_indices, ())
        self.assertEqual(zero.y, 0)
        self.assertEqual(zero.canonicalization_reason, "raw_zero_power")
        self.assertTrue(other.accepted)
        self.assertEqual(result.for_uav(0).energy_reservation.transmit_energy_j, 0.0)


class DeterministicArbitrationTests(unittest.TestCase):
    def test_smaller_slack_has_priority(self) -> None:
        config = make_config(3)
        manager = LifecycleManager()
        add_tx_task(manager, 0, 1, deadline=3)
        add_tx_task(manager, 2, 1, deadline=5)
        states = make_states((0, 2))
        result = DeterministicExecutor(config, manager, states).execute(
            2,
            [tx_proposal(2, 1), tx_proposal(0, 1)],
        )
        self.assertTrue(result.for_uav(0).communication.accepted)
        self.assertEqual(result.for_uav(2).communication.rejection_reason, "half_duplex_conflict")

    def test_higher_historical_quality_breaks_slack_tie(self) -> None:
        config = make_config(3)
        manager = LifecycleManager()
        add_tx_task(manager, 0, 1, deadline=5)
        add_tx_task(manager, 2, 1, deadline=5)
        states = make_states((0, 2))
        features = actor_features(config, {(0, 1): 1.0, (2, 1): 5.0})
        result = DeterministicExecutor(config, manager, states).execute(
            2,
            [tx_proposal(0, 1), tx_proposal(2, 1)],
            actor_features=features,
        )
        self.assertFalse(result.for_uav(0).communication.accepted)
        self.assertTrue(result.for_uav(2).communication.accepted)

    def test_sender_and_receiver_id_tie_breaks_are_explicit(self) -> None:
        config = make_config(3)
        manager = LifecycleManager()
        add_tx_task(manager, 0, 2, deadline=5)
        add_tx_task(manager, 1, 2, deadline=5)
        states = make_states((0, 1))
        result = DeterministicExecutor(config, manager, states).execute(
            2,
            [tx_proposal(1, 2), tx_proposal(0, 2)],
        )
        self.assertTrue(result.for_uav(0).communication.accepted)

        second_manager = LifecycleManager()
        add_tx_task(second_manager, 0, 1, deadline=5)
        add_tx_task(second_manager, 0, 2, deadline=5)
        adapter = ProposalAdapter(config.environment, config.action)
        state = make_states((0,))[0]
        first = adapter.adapt(tx_proposal(0, 1), state, second_manager, slot=2)
        second = adapter.adapt(tx_proposal(0, 2), state, second_manager, slot=2)
        self.assertLess(
            DeterministicExecutor._priority_key(first),
            DeterministicExecutor._priority_key(second),
        )

    def test_no_backtracking_after_accepted_link_downgrades_to_zero(self) -> None:
        config = make_config(3)
        manager = LifecycleManager()
        add_tx_task(manager, 0, 1, deadline=3)
        add_tx_task(manager, 1, 2, deadline=5)
        states = make_states((0, 1), residual_overrides={0: 0.1, 1: 10.0})
        result = DeterministicExecutor(config, manager, states).execute(
            2,
            [tx_proposal(0, 1), tx_proposal(1, 2)],
        )
        high = result.for_uav(0).communication
        low = result.for_uav(1).communication
        self.assertTrue(high.tentatively_accepted)
        self.assertEqual(high.executed_power_w, 0.0)
        self.assertEqual(high.canonicalization_reason, "energy_zero_power")
        self.assertEqual(low.rejection_reason, "half_duplex_conflict")
        self.assertFalse(low.accepted)

    def test_fixed_inputs_and_input_order_produce_identical_result(self) -> None:
        config = make_config(4)
        manager = LifecycleManager()
        add_tx_task(manager, 0, 1)
        add_tx_task(manager, 2, 3)
        states = make_states((0, 2))
        executor = DeterministicExecutor(config, manager, states)
        proposals = [tx_proposal(0, 1), tx_proposal(2, 3)]
        self.assertEqual(
            executor.execute(2, proposals),
            executor.execute(2, reversed(proposals)),
        )


class DowngradeTests(unittest.TestCase):
    def test_highest_feasible_power_level_is_selected(self) -> None:
        config = make_config()
        manager = LifecycleManager()
        add_tx_task(manager, 0, 1)
        states = make_states((0,), residual_overrides={0: 0.6})
        action = DeterministicExecutor(config, manager, states).execute(
            2, [tx_proposal(0, 1)]
        ).for_uav(0)
        self.assertEqual(action.communication.executed_power_w, 0.5)
        self.assertEqual(action.energy_reservation.total_energy_j, 0.5)
        self.assertEqual(action.communication.downgrade_reason, "energy_power_downgrade")

    def test_power_zero_fallback_is_atomic_physical_idle(self) -> None:
        config = make_config()
        manager = LifecycleManager()
        add_tx_task(manager, 0, 1)
        states = make_states((0,), residual_overrides={0: 0.1})
        communication = DeterministicExecutor(config, manager, states).execute(
            2, [tx_proposal(0, 1)]
        ).for_uav(0).communication
        self.assertEqual(communication.candidate_power_w, 0.0)
        self.assertEqual(communication.executed_power_w, 0.0)
        self.assertEqual(communication.executed_ru_indices, ())
        self.assertEqual(communication.y, 0)

    def test_cpu_frequency_downgrades_only_after_all_power_levels_fail(self) -> None:
        config = make_config()
        manager = LifecycleManager()
        add_tx_task(manager, 0, 1)
        add_local_task(manager, 0, cycles=10.0)
        states = make_states((0,), residual_overrides={0: 0.8})
        proposal = tx_proposal(0, 1, cpu_queue="local", cpu_frequency=1.0)
        action = DeterministicExecutor(config, manager, states).execute(
            2, [proposal]
        ).for_uav(0)
        self.assertEqual(action.cpu.executed_frequency_hz, 5.0)
        self.assertEqual(action.communication.executed_power_w, 0.5)
        self.assertEqual(action.cpu.downgrade_reason, "energy_cpu_frequency_downgrade")


class PhysicalLayerAndServiceTests(unittest.TestCase):
    def test_legal_same_ru_reuse_creates_actual_interference_and_noise_once(self) -> None:
        config = make_config(4)
        manager = LifecycleManager()
        add_tx_task(manager, 0, 1, bits=100.0)
        add_tx_task(manager, 2, 3, bits=100.0)
        states = make_states((0, 2))
        execution = DeterministicExecutor(config, manager, states).execute(
            2,
            [tx_proposal(0, 1), tx_proposal(2, 3)],
        )
        self.assertEqual(len(execution.executed_links), 2)
        result = PhysicalService(config, manager, states).execute(
            execution,
            true_channel(config),
            settle_deadlines=False,
        )
        link = next(item for item in result.links if item.sender_uav == 0)
        expected_interference = 0.25
        expected_sinr = 0.25 / (0.001 + expected_interference)
        self.assertAlmostEqual(link.interference_w[0], expected_interference)
        self.assertAlmostEqual(link.sinr_linear[0], expected_sinr)
        self.assertTrue(result.interference_measurement_mask[1, 0])
        self.assertAlmostEqual(result.interference_measurement_w[1, 0], expected_interference)

    def test_simplified_same_ru_reuse_zeroes_only_interference_and_keeps_attempt_mask(self) -> None:
        def run(mode: ChannelAblationMode):
            config = make_config(4, channel_ablation_mode=mode)
            manager = LifecycleManager()
            add_tx_task(manager, 0, 1, bits=100.0)
            add_tx_task(manager, 2, 3, bits=100.0)
            states = make_states((0, 2))
            execution = DeterministicExecutor(config, manager, states).execute(
                2,
                [tx_proposal(0, 1), tx_proposal(2, 3)],
            )
            result = PhysicalService(config, manager, states).execute(
                execution,
                true_channel(config),
                settle_deadlines=False,
            )
            return execution, result

        full_execution, full = run(ChannelAblationMode.FULL)
        simplified_execution, simplified = run(
            ChannelAblationMode.SIMPLIFIED_DETERMINISTIC
        )
        self.assertEqual(full_execution.executed_links, simplified_execution.executed_links)
        self.assertTrue(any(value > 0.0 for link in full.links for value in link.interference_w))
        self.assertTrue(
            all(value == 0.0 for link in simplified.links for value in link.interference_w)
        )
        np.testing.assert_array_equal(
            simplified.interference_measurement_w,
            0.0,
        )
        np.testing.assert_array_equal(
            full.interference_measurement_mask,
            simplified.interference_measurement_mask,
        )
        self.assertEqual(
            int(np.count_nonzero(simplified.interference_measurement_mask)),
            sum(len(link.ru_indices) for link in simplified.links),
        )

    def test_simplified_mode_retains_remote_tx_cpu_terminal_lifecycle(self) -> None:
        config = make_config(
            2,
            channel_ablation_mode=ChannelAblationMode.SIMPLIFIED_DETERMINISTIC,
        )
        manager = LifecycleManager()
        task = add_tx_task(manager, 0, 1, bits=1.0, cycles=1.0, deadline=10)
        states = make_states((0, 1))
        tx_execution = DeterministicExecutor(config, manager, states).execute(
            2,
            [tx_proposal(0, 1), idle_proposal(1)],
        )
        PhysicalService(config, manager, states).execute(
            tx_execution,
            true_channel(config),
            settle_deadlines=False,
        )
        self.assertEqual(task.status, TaskStatus.CPU)

        cpu_execution = DeterministicExecutor(config, manager, states).execute(
            3,
            [idle_proposal(0), idle_proposal(1, cpu_queue=0, cpu_frequency=1.0)],
        )
        PhysicalService(config, manager, states).execute(
            cpu_execution,
            true_channel(config),
        )
        self.assertEqual(task.status, TaskStatus.DONE)
        self.assertEqual(task.outcome, TaskOutcome.DONE)

    def test_rejected_transmitter_produces_no_interference_service_energy_or_attempt(self) -> None:
        config = make_config(3)
        manager = LifecycleManager()
        rejected_task = add_tx_task(manager, 1, 2, bits=10.0, deadline=5)
        add_tx_task(manager, 0, 1, bits=100.0, deadline=3)
        states = make_states((0, 1))
        execution = DeterministicExecutor(config, manager, states).execute(
            2,
            [tx_proposal(0, 1), tx_proposal(1, 2)],
        )
        result = PhysicalService(config, manager, states).execute(
            execution,
            true_channel(config),
            settle_deadlines=False,
        )
        self.assertEqual(len(result.links), 1)
        self.assertTrue(all(value == 0.0 for value in result.links[0].interference_w))
        self.assertEqual(rejected_task.remaining_bits, 10.0)
        rejected_debit = next(item for item in result.energy_debits if item.uav_id == 1)
        self.assertEqual(rejected_debit.transmit_energy_j, 0.0)

    def test_rate_uses_current_true_channel_and_executed_power(self) -> None:
        config = make_config()
        manager = LifecycleManager()
        add_tx_task(manager, 0, 1, bits=100.0)
        states = make_states((0,))
        execution = DeterministicExecutor(config, manager, states).execute(
            2, [tx_proposal(0, 1, power=0.5)]
        )
        result = PhysicalService(config, manager, states).execute(
            execution,
            true_channel(config),
            settle_deadlines=False,
        )
        link = result.links[0]
        expected_sinr = (0.5 / 4.0) / 0.001
        expected_rate = 4.0 * math.log2(1.0 + expected_sinr)
        self.assertTrue(all(abs(value - expected_sinr) < 1.0e-12 for value in link.sinr_linear))
        self.assertAlmostEqual(link.effective_rate_bps, expected_rate)

    def test_tx_partial_service_uses_full_slot_and_edf_head(self) -> None:
        config = make_config()
        manager = LifecycleManager()
        task = add_tx_task(manager, 0, 1, bits=100.0)
        states = make_states((0,))
        execution = DeterministicExecutor(config, manager, states).execute(
            2, [tx_proposal(0, 1)]
        )
        result = PhysicalService(config, manager, states).execute(
            execution,
            true_channel(config),
            settle_deadlines=False,
        )
        link = result.links[0]
        self.assertEqual(link.active_duration_s, 1.0)
        self.assertEqual(link.task_services[0].task_id, task.task_id)
        self.assertAlmostEqual(task.remaining_bits, 100.0 - link.service_bits)

    def test_tx_early_completion_can_serve_multiple_edf_tasks_and_stops_early(self) -> None:
        config = make_config()
        manager = LifecycleManager()
        first = add_tx_task(manager, 0, 1, bits=5.0, deadline=5)
        second = add_tx_task(manager, 0, 1, bits=5.0, deadline=6)
        states = make_states((0,))
        execution = DeterministicExecutor(config, manager, states).execute(
            2, [tx_proposal(0, 1)]
        )
        result = PhysicalService(config, manager, states).execute(
            execution,
            true_channel(config),
            settle_deadlines=False,
        )
        link = result.links[0]
        self.assertEqual([record.task_id for record in link.task_services], [first.task_id, second.task_id])
        self.assertLess(link.active_duration_s, 1.0)
        self.assertAlmostEqual(link.active_duration_s, 10.0 / link.effective_rate_bps)
        self.assertEqual(first.status, TaskStatus.CPU)
        self.assertEqual(second.status, TaskStatus.CPU)

    def test_zero_rate_nonzero_attempt_uses_full_slot_energy_and_outage_one(self) -> None:
        config = make_config()
        manager = LifecycleManager()
        task = add_tx_task(manager, 0, 1, bits=10.0)
        states = make_states((0,))
        execution = DeterministicExecutor(config, manager, states).execute(
            2, [tx_proposal(0, 1)]
        )
        zero_channel = true_channel(config)
        zero_channel[0, 1, :] = 0.0
        result = PhysicalService(config, manager, states).execute(
            execution,
            zero_channel,
            settle_deadlines=False,
        )
        link = result.links[0]
        self.assertEqual(link.effective_rate_bps, 0.0)
        self.assertEqual(link.service_bits, 0.0)
        self.assertEqual(link.active_duration_s, 1.0)
        self.assertEqual(link.transmit_energy_j, 1.0)
        self.assertTrue(link.outage.attempted)
        self.assertEqual(link.outage.sample, 1)
        self.assertEqual(task.remaining_bits, 10.0)

    def test_transfer_completion_enters_cpu_but_cannot_compute_same_slot(self) -> None:
        config = make_config()
        manager = LifecycleManager()
        task = add_tx_task(manager, 0, 1, bits=1.0, cycles=7.0)
        states = make_states((0,))
        execution = DeterministicExecutor(config, manager, states).execute(
            2, [tx_proposal(0, 1)]
        )
        PhysicalService(config, manager, states).execute(
            execution,
            true_channel(config),
            settle_deadlines=False,
        )
        self.assertEqual(task.status, TaskStatus.CPU)
        self.assertEqual(task.remaining_cycles, 7.0)
        self.assertFalse(task.can_service(2))
        self.assertTrue(task.can_service(3))


class CpuAndEnergyTests(unittest.TestCase):
    def test_cpu_services_one_head_only_and_uses_actual_duration_energy(self) -> None:
        config = make_config()
        manager = LifecycleManager()
        first = add_local_task(manager, 0, cycles=5.0, deadline=5)
        second = add_local_task(manager, 0, cycles=5.0, deadline=6)
        states = make_states((0,))
        proposal = idle_proposal(0, cpu_queue="local", cpu_frequency=1.0)
        execution = DeterministicExecutor(config, manager, states).execute(2, [proposal])
        result = PhysicalService(config, manager, states).execute(
            execution,
            true_channel(config),
        )
        cpu = result.cpu_services[0]
        self.assertEqual(cpu.task_id, first.task_id)
        self.assertEqual(cpu.service_cycles, 5.0)
        self.assertEqual(cpu.active_duration_s, 0.5)
        self.assertEqual(cpu.cpu_energy_j, 0.5)
        self.assertEqual(first.outcome, TaskOutcome.DONE)
        self.assertEqual(first.completion_slot, 2)
        self.assertEqual(second.remaining_cycles, 5.0)

    def test_deadline_slot_unfinished_cpu_task_expires_after_service(self) -> None:
        config = make_config()
        manager = LifecycleManager()
        task = add_local_task(manager, 0, cycles=20.0, deadline=2)
        states = make_states((0,))
        proposal = idle_proposal(0, cpu_queue="local", cpu_frequency=1.0)
        execution = DeterministicExecutor(config, manager, states).execute(2, [proposal])
        PhysicalService(config, manager, states).execute(execution, true_channel(config))
        self.assertEqual(task.remaining_cycles, 10.0)
        self.assertEqual(task.outcome, TaskOutcome.EXPIRED)
        self.assertIsNone(task.completion_slot)

    def test_actual_energy_debits_active_duration_not_reservation(self) -> None:
        config = make_config()
        manager = LifecycleManager()
        add_tx_task(manager, 0, 1, bits=1.0)
        states = make_states((0,), residual_energy_j=2.0)
        execution = DeterministicExecutor(config, manager, states).execute(
            2, [tx_proposal(0, 1)]
        )
        result = PhysicalService(config, manager, states).execute(
            execution,
            true_channel(config),
            settle_deadlines=False,
        )
        debit = result.energy_debits[0]
        self.assertEqual(debit.reserved_energy_j, 1.0)
        self.assertLess(debit.transmit_energy_j, debit.reserved_energy_j)
        self.assertAlmostEqual(
            debit.residual_after_j,
            debit.residual_before_j - debit.total_energy_j,
        )
        self.assertGreaterEqual(states[0].residual_energy_j, 0.0)


class OutageSemanticsTests(unittest.TestCase):
    def test_no_attempt_cases_are_none_not_zero(self) -> None:
        base = {
            "accepted": True,
            "executed_power_w": 1.0,
            "executed_ru_indices": (1,),
            "slot_start_queue_bits": 1.0,
            "effective_rate_bps": 1.0,
            "sinr_linear": (1.0,),
            "threshold_linear": 0.5,
        }
        cases = (
            {"accepted": False},
            {"executed_power_w": 0.0},
            {"executed_ru_indices": ()},
            {"slot_start_queue_bits": 0.0},
        )
        for override in cases:
            values = dict(base)
            values.update(override)
            result = actual_attempt_outage(**values)
            self.assertFalse(result.attempted)
            self.assertIsNone(result.sample)

    def test_successful_attempt_is_zero_and_threshold_domain_is_linear(self) -> None:
        result = actual_attempt_outage(
            accepted=True,
            executed_power_w=1.0,
            executed_ru_indices=(1, 2),
            slot_start_queue_bits=1.0,
            effective_rate_bps=1.0,
            sinr_linear=(0.5, 0.6),
            threshold_linear=10.0 ** (-3.0 / 10.0),
        )
        self.assertTrue(result.attempted)
        self.assertEqual(result.sample, 0)
        self.assertEqual(result.rb_outage_ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
