"""Seven-branch proposal records and the physical-action adapter.

The adapter validates actor output, constructs S_prop and maps normalized
power/frequency levels to physical units. It never samples a replacement action
and never mutates the proposal supplied by the policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import ActionConfig, EnvironmentConfig
from .energy import UavEnergyState
from .history import ActorChannelFeatures, executor_historical_quality_mean
from .lifecycle import LifecycleManager
from .tasks import Task


class ActionError(ValueError):
    """Raised when a seven-branch proposal is malformed or slot-ineligible."""


@dataclass(frozen=True)
class ActionProposal:
    """One UAV's seven fixed actor branches plus its stable UAV identifier."""

    uav_id: int
    route: str | int
    tx_select: str | int
    resource_group: str | int
    resource_width: int
    power_level: float
    cpu_queue: str | int
    cpu_frequency: float

    @property
    def branches(self) -> tuple[object, ...]:
        return (
            self.route,
            self.tx_select,
            self.resource_group,
            self.resource_width,
            self.power_level,
            self.cpu_queue,
            self.cpu_frequency,
        )


@dataclass(frozen=True)
class PreparedCommunication:
    """Validated communication proposal before joint arbitration."""

    receiver_uav: int | None
    proposed_ru_indices: tuple[int, ...]
    proposed_power_w: float
    slot_start_queue_bits: float
    head_task_id: int | None
    head_slack_slots: int | None
    historical_quality: float
    raw_zero_power: bool

    @property
    def active_candidate(self) -> bool:
        return (
            self.receiver_uav is not None
            and bool(self.proposed_ru_indices)
            and self.proposed_power_w > 0.0
            and self.slot_start_queue_bits > 0.0
        )


@dataclass(frozen=True)
class PreparedCpu:
    """Validated slot-start CPU queue selection before energy downgrade."""

    queue_source_uav: int | None
    proposed_frequency_hz: float
    head_task_id: int | None
    head_remaining_cycles: float

    @property
    def active_candidate(self) -> bool:
        return (
            self.queue_source_uav is not None
            and self.head_task_id is not None
            and self.proposed_frequency_hz > 0.0
            and self.head_remaining_cycles > 0.0
        )


@dataclass(frozen=True)
class PreparedAction:
    """Proposal-preserving adapter result consumed by the executor."""

    proposal: ActionProposal
    communication: PreparedCommunication
    cpu: PreparedCpu


class ProposalAdapter:
    """Map validated seven-branch proposals to executor-ready physical values."""

    def __init__(self, environment: EnvironmentConfig, action: ActionConfig) -> None:
        self.environment = environment
        self.action = action

    def adapt(
        self,
        proposal: ActionProposal,
        resources: UavEnergyState,
        lifecycle: LifecycleManager,
        *,
        slot: int,
        actor_features: ActorChannelFeatures | None = None,
    ) -> PreparedAction:
        self._validate_identity(proposal, resources, slot)
        self._validate_route(proposal)
        communication = self._prepare_communication(
            proposal,
            resources,
            lifecycle,
            slot,
            actor_features,
        )
        cpu = self._prepare_cpu(proposal, resources, lifecycle, slot)
        return PreparedAction(proposal=proposal, communication=communication, cpu=cpu)

    def _validate_identity(
        self,
        proposal: ActionProposal,
        resources: UavEnergyState,
        slot: int,
    ) -> None:
        if not isinstance(proposal, ActionProposal):
            raise TypeError("proposal must be an ActionProposal")
        if (
            isinstance(proposal.uav_id, bool)
            or not isinstance(proposal.uav_id, int)
            or not 0 <= proposal.uav_id < self.environment.uav_count
        ):
            raise ActionError("proposal uav_id is outside the configured UAV range")
        if proposal.uav_id != resources.uav_id:
            raise ActionError("proposal and resource-state UAV identifiers differ")
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise ActionError("slot must be a non-negative integer")

    def _validate_route(self, proposal: ActionProposal) -> None:
        route = proposal.route
        if route in self.action.route_fixed_actions:
            return
        if (
            isinstance(route, bool)
            or not isinstance(route, int)
            or not 0 <= route < self.environment.uav_count
            or route == proposal.uav_id
        ):
            raise ActionError("route must be idle/local/defer or a valid remote UAV")

    def _prepare_communication(
        self,
        proposal: ActionProposal,
        resources: UavEnergyState,
        lifecycle: LifecycleManager,
        slot: int,
        actor_features: ActorChannelFeatures | None,
    ) -> PreparedCommunication:
        idle = self.action.tx_select_idle_action
        if proposal.tx_select == idle:
            if (
                proposal.resource_group != self.action.canonical_inactive_values["resource_group"]
                or proposal.resource_width != self.action.canonical_inactive_values["resource_width"]
                or proposal.power_level != self.action.canonical_inactive_values["power_level"]
            ):
                raise ActionError("inactive communication branches are not canonical")
            return PreparedCommunication(None, (), 0.0, 0.0, None, None, 0.0, False)

        receiver = proposal.tx_select
        if (
            isinstance(receiver, bool)
            or not isinstance(receiver, int)
            or not 0 <= receiver < self.environment.uav_count
            or receiver == proposal.uav_id
        ):
            raise ActionError("tx_select must be idle or a valid remote UAV")
        if isinstance(proposal.resource_group, bool) or not isinstance(proposal.resource_group, int):
            raise ActionError("active communication requires an integer resource_group")
        group = proposal.resource_group
        if not 1 <= group <= self.environment.resource_group_count:
            raise ActionError("resource_group is outside the configured group domain")
        if (
            isinstance(proposal.resource_width, bool)
            or not isinstance(proposal.resource_width, int)
            or proposal.resource_width not in self.action.resource_width_options
        ):
            raise ActionError("resource_width is outside the frozen domain")
        if group + proposal.resource_width - 1 > self.environment.resource_group_count:
            raise ActionError("resource_group/resource_width combination crosses the group boundary")
        level = self._normalised_level(
            proposal.power_level,
            self.action.power_levels,
            "power_level",
        )
        groups = self.environment.resource_groups[
            group - 1 : group - 1 + proposal.resource_width
        ]
        proposed_ru_indices = tuple(ru for resource_group in groups for ru in resource_group)
        queue = lifecycle.queues.tx.get((proposal.uav_id, receiver))
        if queue is None or len(queue) == 0:
            raise ActionError("tx_select references an empty or missing slot-start TX queue")
        head = queue.peek()
        assert head is not None
        self._require_serviceable_head(head, slot, "TX")
        queue_bits = float(sum(float(task.remaining_bits) for task in queue))
        if not math.isfinite(queue_bits) or queue_bits <= 0.0:
            raise ActionError("active TX queue must contain positive remaining bits")
        historical_quality = 0.0
        if actor_features is not None:
            if actor_features.slot != slot:
                raise ActionError("actor historical features do not match the executor slot")
            historical_quality = executor_historical_quality_mean(
                actor_features,
                proposal.uav_id,
                receiver,
                tuple(ru - 1 for ru in proposed_ru_indices),
            )
        proposed_power_w = level * resources.max_transmit_power_w
        return PreparedCommunication(
            receiver_uav=receiver,
            proposed_ru_indices=proposed_ru_indices,
            proposed_power_w=proposed_power_w,
            slot_start_queue_bits=queue_bits,
            head_task_id=head.task_id,
            head_slack_slots=head.deadline_slack(slot),
            historical_quality=historical_quality,
            raw_zero_power=proposed_power_w == 0.0,
        )

    def _prepare_cpu(
        self,
        proposal: ActionProposal,
        resources: UavEnergyState,
        lifecycle: LifecycleManager,
        slot: int,
    ) -> PreparedCpu:
        idle = self.action.cpu_queue_idle_action
        if proposal.cpu_queue == idle:
            if proposal.cpu_frequency != self.action.canonical_inactive_values["cpu_frequency"]:
                raise ActionError("inactive CPU frequency is not canonical")
            return PreparedCpu(None, 0.0, None, 0.0)

        source = proposal.uav_id if proposal.cpu_queue == "local" else proposal.cpu_queue
        if isinstance(source, bool) or not isinstance(source, int) or not 0 <= source < self.environment.uav_count:
            raise ActionError("cpu_queue must be idle/local or a valid source UAV")
        level = self._normalised_level(
            proposal.cpu_frequency,
            self.action.cpu_frequency_levels,
            "cpu_frequency",
        )
        if source == proposal.uav_id:
            queue = lifecycle.queues.local.get(proposal.uav_id)
        else:
            queue = lifecycle.queues.cpu.get((source, proposal.uav_id))
        if queue is None or len(queue) == 0:
            raise ActionError("cpu_queue references an empty or missing slot-start CPU queue")
        head = queue.peek()
        assert head is not None
        self._require_serviceable_head(head, slot, "CPU")
        return PreparedCpu(
            queue_source_uav=source,
            proposed_frequency_hz=level * resources.max_cpu_frequency_hz,
            head_task_id=head.task_id,
            head_remaining_cycles=float(head.remaining_cycles),
        )

    @staticmethod
    def _normalised_level(value: float, domain: tuple[float, ...], name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ActionError(f"{name} must be numeric")
        converted = float(value)
        if not math.isfinite(converted) or converted not in domain:
            raise ActionError(f"{name} is outside the frozen discrete domain")
        return converted

    @staticmethod
    def _require_serviceable_head(task: Task, slot: int, label: str) -> None:
        if not task.can_service(slot):
            raise ActionError(f"{label} queue head is not serviceable in this slot")


__all__ = [
    "ActionError",
    "ActionProposal",
    "PreparedAction",
    "PreparedCommunication",
    "PreparedCpu",
    "ProposalAdapter",
]
