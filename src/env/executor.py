"""Deterministic joint half-duplex executor with discrete energy downgrade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from ..config import RunConfig
from .actions import ActionError, ActionProposal, PreparedAction, ProposalAdapter
from .energy import (
    EnergyReservation,
    UavEnergyState,
    select_highest_feasible_levels,
)
from .history import ActorChannelFeatures
from .lifecycle import LifecycleManager


class ExecutorError(ValueError):
    """Raised when joint executor inputs are incomplete or inconsistent."""


@dataclass(frozen=True)
class CommunicationExecution:
    """Proposal, candidate, and final communication stages for one UAV."""

    sender_uav: int
    receiver_uav: int | None
    proposed_ru_indices: tuple[int, ...]
    executed_ru_indices: tuple[int, ...]
    proposed_power_w: float
    candidate_power_w: float
    executed_power_w: float
    tentatively_accepted: bool
    historical_quality: float
    priority_key: tuple[float, float, int, int] | None
    rejection_reason: str | None
    downgrade_reason: str | None
    canonicalization_reason: str | None
    slot_start_queue_bits: float
    head_task_id: int | None

    @property
    def accepted(self) -> bool:
        return (
            self.receiver_uav is not None
            and self.executed_power_w > 0.0
            and bool(self.executed_ru_indices)
        )

    @property
    def y(self) -> int:
        return int(self.accepted)

    @property
    def per_ru_power_w(self) -> float:
        return (
            self.executed_power_w / len(self.executed_ru_indices)
            if self.accepted
            else 0.0
        )


@dataclass(frozen=True)
class CpuExecution:
    """Final CPU queue/frequency decision for one computing UAV."""

    executor_uav: int
    queue_source_uav: int | None
    head_task_id: int | None
    proposed_frequency_hz: float
    executed_frequency_hz: float
    head_remaining_cycles: float
    downgrade_reason: str | None

    @property
    def active(self) -> bool:
        return (
            self.queue_source_uav is not None
            and self.head_task_id is not None
            and self.executed_frequency_hz > 0.0
        )


@dataclass(frozen=True)
class ExecutedAction:
    """Proposal-preserving final result and executor audit metadata."""

    proposal: ActionProposal
    communication: CommunicationExecution
    cpu: CpuExecution
    energy_reservation: EnergyReservation


@dataclass(frozen=True)
class JointExecutionResult:
    """Deterministic final joint action, sorted by UAV identifier."""

    slot: int
    actions: tuple[ExecutedAction, ...]

    def for_uav(self, uav_id: int) -> ExecutedAction:
        for action in self.actions:
            if action.proposal.uav_id == uav_id:
                return action
        raise KeyError(f"no executed action for UAV {uav_id}")

    @property
    def executed_links(self) -> tuple[CommunicationExecution, ...]:
        return tuple(
            action.communication
            for action in self.actions
            if action.communication.accepted
        )
    def x(self, sender_uav: int, receiver_uav: int, ru_index: int) -> int:
        """Return the final one-based RU occupancy indicator x_ijr."""

        try:
            communication = self.for_uav(sender_uav).communication
        except KeyError:
            return 0
        return int(
            communication.accepted
            and communication.receiver_uav == receiver_uav
            and ru_index in communication.executed_ru_indices
        )

    @property
    def executed_resource_sets(self) -> dict[int, tuple[int, ...]]:
        return {
            action.proposal.uav_id: action.communication.executed_ru_indices
            for action in self.actions
        }

    @property
    def reservations(self) -> dict[int, EnergyReservation]:
        return {
            action.proposal.uav_id: action.energy_reservation
            for action in self.actions
        }


class DeterministicExecutor:
    """Apply frozen priority, half-duplex, downgrade, and zero-power rules."""

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
        self.adapter = ProposalAdapter(config.environment, config.action)
        for uav_id, state in self.resource_states.items():
            if uav_id != state.uav_id:
                raise ExecutorError("resource-state mapping key must equal state.uav_id")
            if not 0 <= uav_id < config.environment.uav_count:
                raise ExecutorError("resource state contains an out-of-range UAV identifier")

    def execute(
        self,
        slot: int,
        proposals: Iterable[ActionProposal],
        *,
        actor_features: ActorChannelFeatures | None = None,
    ) -> JointExecutionResult:
        materialised = tuple(proposals)
        ids = [proposal.uav_id for proposal in materialised]
        if len(ids) != len(set(ids)):
            raise ExecutorError("joint proposal contains duplicate UAV identifiers")
        missing_resources = sorted(set(ids) - set(self.resource_states))
        if missing_resources:
            raise ExecutorError(f"missing resource states for UAVs {missing_resources}")

        prepared = tuple(
            self.adapter.adapt(
                proposal,
                self.resource_states[proposal.uav_id],
                self.lifecycle,
                slot=slot,
                actor_features=actor_features,
            )
            for proposal in sorted(materialised, key=lambda item: item.uav_id)
        )
        candidates = tuple(
            item for item in prepared if item.communication.active_candidate
        )
        ordered = sorted(candidates, key=self._priority_key)
        occupied_wireless_uavs: set[int] = set()
        tentative_acceptance: set[int] = set()
        rejection_reasons: dict[int, str] = {}
        for item in ordered:
            sender = item.proposal.uav_id
            receiver = item.communication.receiver_uav
            assert receiver is not None
            if sender in occupied_wireless_uavs or receiver in occupied_wireless_uavs:
                rejection_reasons[sender] = "half_duplex_conflict"
                continue
            tentative_acceptance.add(sender)
            occupied_wireless_uavs.add(sender)
            occupied_wireless_uavs.add(receiver)

        executed = tuple(
            self._finalize(
                item,
                tentatively_accepted=item.proposal.uav_id in tentative_acceptance,
                rejection_reason=rejection_reasons.get(item.proposal.uav_id),
            )
            for item in prepared
        )
        return JointExecutionResult(slot=slot, actions=executed)

    @staticmethod
    def _priority_key(item: PreparedAction) -> tuple[float, float, int, int]:
        communication = item.communication
        if communication.head_slack_slots is None or communication.receiver_uav is None:
            raise ExecutorError("active communication candidate lacks a priority key")
        return (
            float(communication.head_slack_slots),
            -float(communication.historical_quality),
            item.proposal.uav_id,
            communication.receiver_uav,
        )

    def _finalize(
        self,
        item: PreparedAction,
        *,
        tentatively_accepted: bool,
        rejection_reason: str | None,
    ) -> ExecutedAction:
        uav_id = item.proposal.uav_id
        state = self.resource_states[uav_id]
        communication = item.communication
        cpu = item.cpu
        communication_candidate = tentatively_accepted and communication.active_candidate
        selection = select_highest_feasible_levels(
            state,
            self.config.action,
            proposed_power_w=(
                communication.proposed_power_w if communication_candidate else 0.0
            ),
            proposed_cpu_frequency_hz=cpu.proposed_frequency_hz,
            communication_candidate=communication_candidate,
            cpu_candidate=cpu.active_candidate,
            cpu_head_remaining_cycles=cpu.head_remaining_cycles,
            slot_duration_s=self.config.environment.slot_duration_s,
            tolerance_j=self.config.environment.energy_tolerance_j,
        )

        canonicalization_reason: str | None = None
        if communication.receiver_uav is None:
            canonicalization_reason = "inactive"
        elif communication.raw_zero_power:
            canonicalization_reason = "raw_zero_power"
        elif tentatively_accepted and selection.candidate_power_w == 0.0:
            canonicalization_reason = "energy_zero_power"

        executed_power = (
            selection.candidate_power_w
            if tentatively_accepted and selection.candidate_power_w > 0.0
            else 0.0
        )
        executed_ru_indices = (
            communication.proposed_ru_indices if executed_power > 0.0 else ()
        )
        priority_key = (
            self._priority_key(item)
            if communication.active_candidate
            else None
        )
        power_downgrade = (
            "energy_power_downgrade"
            if communication_candidate and selection.power_downgraded
            else None
        )
        cpu_downgrade = (
            "energy_cpu_frequency_downgrade"
            if cpu.active_candidate and selection.cpu_downgraded
            else None
        )
        communication_result = CommunicationExecution(
            sender_uav=uav_id,
            receiver_uav=communication.receiver_uav,
            proposed_ru_indices=communication.proposed_ru_indices,
            executed_ru_indices=executed_ru_indices,
            proposed_power_w=communication.proposed_power_w,
            candidate_power_w=selection.candidate_power_w,
            executed_power_w=executed_power,
            tentatively_accepted=tentatively_accepted,
            historical_quality=communication.historical_quality,
            priority_key=priority_key,
            rejection_reason=rejection_reason,
            downgrade_reason=power_downgrade,
            canonicalization_reason=canonicalization_reason,
            slot_start_queue_bits=communication.slot_start_queue_bits,
            head_task_id=communication.head_task_id,
        )
        cpu_result = CpuExecution(
            executor_uav=uav_id,
            queue_source_uav=cpu.queue_source_uav,
            head_task_id=cpu.head_task_id,
            proposed_frequency_hz=cpu.proposed_frequency_hz,
            executed_frequency_hz=selection.executed_cpu_frequency_hz,
            head_remaining_cycles=cpu.head_remaining_cycles,
            downgrade_reason=cpu_downgrade,
        )
        return ExecutedAction(
            proposal=item.proposal,
            communication=communication_result,
            cpu=cpu_result,
            energy_reservation=selection.reservation,
        )


__all__ = [
    "CommunicationExecution",
    "CpuExecution",
    "DeterministicExecutor",
    "ExecutedAction",
    "ExecutorError",
    "JointExecutionResult",
]
