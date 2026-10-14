"""Actual SINR, communication/CPU service, and slot energy accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from ..config import ChannelAblationMode, RunConfig
from .channel import receiver_noise_power_w
from .energy import (
    ActualEnergyDebit,
    UavEnergyState,
    actual_transmit_energy_j,
    cpu_active_duration_s,
    cpu_dynamic_energy_j,
    debit_actual_energy,
)
from .executor import CommunicationExecution, JointExecutionResult
from .lifecycle import LifecycleManager
from .outage import OutageResult, actual_attempt_outage


class ServiceError(ValueError):
    """Raised when physical execution inputs violate shape or slot contracts."""


@dataclass(frozen=True)
class TaskServiceRecord:
    task_id: int
    amount: float


@dataclass(frozen=True)
class LinkServiceResult:
    """Actual physical and queue outcome for one executed wireless link."""

    sender_uav: int
    receiver_uav: int
    ru_indices: tuple[int, ...]
    per_ru_power_w: float
    interference_w: tuple[float, ...]
    sinr_linear: tuple[float, ...]
    effective_rate_bps: float
    active_duration_s: float
    service_bits: float
    task_services: tuple[TaskServiceRecord, ...]
    transmit_energy_j: float
    outage: OutageResult


@dataclass(frozen=True)
class CpuServiceResult:
    """One selected slot-start CPU head and its actual dynamic activity."""

    executor_uav: int
    queue_source_uav: int | None
    task_id: int | None
    executed_frequency_hz: float
    service_cycles: float
    active_duration_s: float
    cpu_energy_j: float


@dataclass(frozen=True)
class SlotPhysicalResult:
    """Auditable Implementation-04 output without a full environment step."""

    slot: int
    execution: JointExecutionResult
    links: tuple[LinkServiceResult, ...]
    cpu_services: tuple[CpuServiceResult, ...]
    energy_debits: tuple[ActualEnergyDebit, ...]
    interference_measurement_w: np.ndarray
    interference_measurement_mask: np.ndarray
    settled_task_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        measurement = np.array(self.interference_measurement_w, dtype=np.float64, copy=True)
        mask = np.array(self.interference_measurement_mask, dtype=np.bool_, copy=True)
        if measurement.ndim != 2 or mask.shape != measurement.shape:
            raise ServiceError("interference measurement and mask must be receiver/RU matrices")
        if not np.all(np.isfinite(measurement)) or np.any(measurement < 0.0):
            raise ServiceError("interference measurement must be finite and non-negative")
        measurement.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "interference_measurement_w", measurement)
        object.__setattr__(self, "interference_measurement_mask", mask)


class PhysicalService:
    """Consume one final joint action using the current true channel tensor."""

    def __init__(
        self,
        config: RunConfig,
        lifecycle: LifecycleManager,
        resource_states: Mapping[int, UavEnergyState],
    ) -> None:
        config.validate()
        self.config = config
        self.lifecycle = lifecycle
        self.resource_states = dict(resource_states)
        self.noise_power_w = receiver_noise_power_w(config.environment)
        self._simplified_deterministic = (
            config.environment.channel_ablation_mode
            == ChannelAblationMode.SIMPLIFIED_DETERMINISTIC
        )

    def execute(
        self,
        execution: JointExecutionResult,
        true_channel: np.ndarray,
        *,
        settle_deadlines: bool = True,
    ) -> SlotPhysicalResult:
        channel = self._validate_channel(execution, true_channel)
        link_results = tuple(
            self._evaluate_link(link, execution.executed_links, channel)
            for link in execution.executed_links
        )
        cpu_results = tuple(
            self._evaluate_cpu(action.cpu)
            for action in execution.actions
            if action.cpu.queue_source_uav is not None
        )

        # CPU uses the slot-start selected EDF head. Applying it before TX
        # transitions prevents a newly transferred task from changing that head.
        for result in cpu_results:
            if result.service_cycles > 0.0 and result.task_id is not None:
                self.lifecycle.service_cpu(
                    result.task_id,
                    slot=execution.slot,
                    cycles=result.service_cycles,
                )

        applied_links: list[LinkServiceResult] = []
        for result in link_results:
            task_services = self._apply_link_service(execution.slot, result)
            applied_links.append(replace(result, task_services=task_services))

        tx_energy = {
            result.sender_uav: result.transmit_energy_j
            for result in applied_links
        }
        cpu_energy = {
            result.executor_uav: result.cpu_energy_j
            for result in cpu_results
        }
        slot_uav_ids = tuple(action.proposal.uav_id for action in execution.actions)
        states = {uav_id: self.resource_states[uav_id] for uav_id in slot_uav_ids}
        debits = debit_actual_energy(
            states,
            tx_energy,
            cpu_energy,
            execution.reservations,
            tolerance_j=self.config.environment.energy_tolerance_j,
        )
        settled = (
            self.lifecycle.settle_slot(execution.slot)
            if settle_deadlines
            else ()
        )
        measurement, measurement_mask = self._measurement_matrices(
            tuple(applied_links)
        )
        return SlotPhysicalResult(
            slot=execution.slot,
            execution=execution,
            links=tuple(applied_links),
            cpu_services=cpu_results,
            energy_debits=debits,
            interference_measurement_w=measurement,
            interference_measurement_mask=measurement_mask,
            settled_task_ids=tuple(task.task_id for task in settled),
        )

    def _validate_channel(
        self,
        execution: JointExecutionResult,
        true_channel: np.ndarray,
    ) -> np.ndarray:
        if execution.slot < 0:
            raise ServiceError("execution slot must be non-negative")
        channel = np.asarray(true_channel)
        expected = (
            self.config.environment.uav_count,
            self.config.environment.uav_count,
            self.config.environment.ru_count,
        )
        if channel.shape != expected or not np.iscomplexobj(channel):
            raise ServiceError(f"true_channel must be a complex tensor with shape {expected}")
        if not np.all(np.isfinite(channel)):
            raise ServiceError("true_channel must contain only finite values")
        for action in execution.actions:
            if action.proposal.uav_id not in self.resource_states:
                raise ServiceError("execution references a UAV without an energy state")
        return channel

    def _evaluate_link(
        self,
        communication: CommunicationExecution,
        all_links: tuple[CommunicationExecution, ...],
        channel: np.ndarray,
    ) -> LinkServiceResult:
        if not communication.accepted or communication.receiver_uav is None:
            raise ServiceError("physical link evaluation requires an executed link")
        sender = communication.sender_uav
        receiver = communication.receiver_uav
        per_ru_power = communication.per_ru_power_w
        interference_values: list[float] = []
        sinr_values: list[float] = []
        rate = 0.0
        threshold = self.config.environment.outage_threshold_linear
        for ru_one_based in communication.executed_ru_indices:
            ru = ru_one_based - 1
            interference = 0.0
            if not (self._simplified_deterministic or self.config.environment.cochannel_interference_disabled):
                for other in all_links:
                    if (
                        other.sender_uav == sender
                        and other.receiver_uav == receiver
                    ):
                        continue
                    if ru_one_based not in other.executed_ru_indices:
                        continue
                    interference += (
                        other.per_ru_power_w
                        * abs(channel[other.sender_uav, receiver, ru]) ** 2
                    )
            desired = per_ru_power * abs(channel[sender, receiver, ru]) ** 2
            sinr = desired / (self.noise_power_w + interference)
            if not math.isfinite(sinr) or sinr < 0.0:
                raise ServiceError("actual SINR must be finite and non-negative")
            interference_values.append(float(interference))
            sinr_values.append(float(sinr))
            if sinr >= threshold:
                rate += self.config.environment.ru_bandwidth_hz * math.log2(1.0 + sinr)

        outage = actual_attempt_outage(
            accepted=True,
            executed_power_w=communication.executed_power_w,
            executed_ru_indices=communication.executed_ru_indices,
            slot_start_queue_bits=communication.slot_start_queue_bits,
            effective_rate_bps=rate,
            sinr_linear=sinr_values,
            threshold_linear=threshold,
        )
        duration = (
            self.config.environment.slot_duration_s
            if rate == 0.0
            else min(
                self.config.environment.slot_duration_s,
                communication.slot_start_queue_bits / rate,
            )
        )
        service_bits = min(
            communication.slot_start_queue_bits,
            rate * self.config.environment.slot_duration_s,
        )
        transmit_energy = actual_transmit_energy_j(
            communication.executed_power_w,
            duration,
        )
        return LinkServiceResult(
            sender_uav=sender,
            receiver_uav=receiver,
            ru_indices=communication.executed_ru_indices,
            per_ru_power_w=per_ru_power,
            interference_w=tuple(interference_values),
            sinr_linear=tuple(sinr_values),
            effective_rate_bps=rate,
            active_duration_s=duration,
            service_bits=service_bits,
            task_services=(),
            transmit_energy_j=transmit_energy,
            outage=outage,
        )

    def _evaluate_cpu(self, cpu) -> CpuServiceResult:
        frequency = cpu.executed_frequency_hz
        cycles = (
            min(
                cpu.head_remaining_cycles,
                frequency * self.config.environment.slot_duration_s,
            )
            if cpu.active
            else 0.0
        )
        duration = cpu_active_duration_s(
            frequency,
            cycles,
            self.config.environment.slot_duration_s,
        )
        energy = cpu_dynamic_energy_j(
            self.resource_states[cpu.executor_uav].cpu_coefficient,
            frequency,
            duration,
        )
        return CpuServiceResult(
            executor_uav=cpu.executor_uav,
            queue_source_uav=cpu.queue_source_uav,
            task_id=cpu.head_task_id,
            executed_frequency_hz=frequency,
            service_cycles=cycles,
            active_duration_s=duration,
            cpu_energy_j=energy,
        )

    def _apply_link_service(
        self,
        slot: int,
        result: LinkServiceResult,
    ) -> tuple[TaskServiceRecord, ...]:
        remaining_budget = result.service_bits
        records: list[TaskServiceRecord] = []
        while remaining_budget > 0.0:
            queue = self.lifecycle.queues.tx.get(
                (result.sender_uav, result.receiver_uav)
            )
            if queue is None or len(queue) == 0:
                break
            task = queue.peek()
            assert task is not None
            amount = min(float(task.remaining_bits), remaining_budget)
            if amount <= 0.0:
                break
            self.lifecycle.service_transmission(task, slot=slot, bits=amount)
            records.append(TaskServiceRecord(task.task_id, amount))
            remaining_budget -= amount
            if remaining_budget < 0.0 and abs(remaining_budget) <= math.ulp(max(1.0, result.service_bits)):
                remaining_budget = 0.0
        return tuple(records)

    def _measurement_matrices(
        self,
        links: tuple[LinkServiceResult, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        shape = (
            self.config.environment.uav_count,
            self.config.environment.ru_count,
        )
        measurement = np.zeros(shape, dtype=np.float64)
        mask = np.zeros(shape, dtype=np.bool_)
        for link in links:
            if not link.outage.attempted:
                continue
            for ru_one_based, interference in zip(
                link.ru_indices,
                link.interference_w,
            ):
                ru = ru_one_based - 1
                measurement[link.receiver_uav, ru] = interference
                mask[link.receiver_uav, ru] = True
        return measurement, mask


__all__ = [
    "CpuServiceResult",
    "LinkServiceResult",
    "PhysicalService",
    "ServiceError",
    "SlotPhysicalResult",
    "TaskServiceRecord",
]
