"""Fixed-reference post-settlement team reward accounting.

The calculator in this module is intentionally stateless across slots.  Its
normalization references and weights are supplied once at construction time;
no running maximum, batch statistic, or episode observation can change them.
Callers must invoke it after physical service and task settlement, but before
the current slot's arrivals are registered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from ..config import CreditAssignmentMode, RewardCreditConfig
from .tasks import Task, TaskOutcome


AGENT_REWARD_CONSERVATION_TOLERANCE = 1.0e-9


class RewardError(ValueError):
    """Raised when reward inputs violate the frozen timing or numeric contract."""


def _finite(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise RewardError(f"{name} must be finite")
    if positive and converted <= 0.0:
        raise RewardError(f"{name} must be positive")
    return converted


def _finite_nonnegative(value: float, name: str) -> float:
    converted = _finite(value, name)
    if converted < 0.0:
        raise RewardError(f"{name} must be non-negative")
    return converted


@dataclass(frozen=True)
class TaskWorkloadSnapshot:
    """Minimal immutable workload state captured immediately before routing."""

    task_id: int
    remaining_bits: float
    remaining_cycles: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.task_id, bool)
            or not isinstance(self.task_id, int)
            or self.task_id < 0
        ):
            raise RewardError("snapshot task_id must be a non-negative integer")
        object.__setattr__(
            self,
            "remaining_bits",
            _finite_nonnegative(self.remaining_bits, "snapshot.remaining_bits"),
        )
        object.__setattr__(
            self,
            "remaining_cycles",
            _finite_nonnegative(self.remaining_cycles, "snapshot.remaining_cycles"),
        )

    @classmethod
    def from_task(cls, task: Task) -> "TaskWorkloadSnapshot":
        if not isinstance(task, Task):
            raise TypeError("workload snapshot source must be a Task")
        return cls(task.task_id, task.remaining_bits, task.remaining_cycles)


@dataclass(frozen=True)
class RewardReferences:
    """Episode-invariant physical references used by the frozen reward."""

    reference_rate_bps: float
    reference_cpu_frequency_hz: float
    workload_reference_s: float
    task_count_reference: float
    active_energy_reference_j: float

    def __post_init__(self) -> None:
        for name in (
            "reference_rate_bps",
            "reference_cpu_frequency_hz",
            "workload_reference_s",
            "task_count_reference",
            "active_energy_reference_j",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name, positive=True))


@dataclass(frozen=True)
class RewardWeights:
    """Fixed non-negative coefficients for completion and three penalties."""

    completion: float
    expiration: float
    workload: float
    energy: float

    def __post_init__(self) -> None:
        for name in ("completion", "expiration", "workload", "energy"):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name))


@dataclass(frozen=True)
class RewardTerms:
    """Auditable scalar terms from one post-settlement/pre-arrival boundary."""

    slot: int
    completed_task_count: int
    expired_task_count: int
    urgent_workload_s: float
    actual_energy_j: float
    normalized_completed: float
    normalized_expired: float
    normalized_workload: float
    normalized_energy: float
    completion_component: float
    expiration_penalty: float
    workload_penalty: float
    energy_penalty: float
    reward: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a deterministic JSON-safe audit record."""

        return {
            "slot": self.slot,
            "completed_task_count": self.completed_task_count,
            "expired_task_count": self.expired_task_count,
            "urgent_workload_s": self.urgent_workload_s,
            "actual_energy_j": self.actual_energy_j,
            "normalized_completed": self.normalized_completed,
            "normalized_expired": self.normalized_expired,
            "normalized_workload": self.normalized_workload,
            "normalized_energy": self.normalized_energy,
            "completion_component": self.completion_component,
            "expiration_penalty": self.expiration_penalty,
            "workload_penalty": self.workload_penalty,
            "energy_penalty": self.energy_penalty,
            "reward": self.reward,
        }


@dataclass(frozen=True)
class RemoteCompletionAttribution:
    """Intrinsic-work split for one remotely completed task."""

    task_id: int
    source_uav: int
    destination_uav: int
    source_share: float
    destination_share: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "task_id": self.task_id,
            "source_uav": self.source_uav,
            "destination_uav": self.destination_uav,
            "source_share": self.source_share,
            "destination_share": self.destination_share,
        }


@dataclass(frozen=True)
class TaskTerminalAttribution:
    """One settled task's participant shares and lifecycle responsibility."""

    task_id: int
    source_uav: int
    destination_uav: int | None
    terminal_state: str
    completion_source_share: float
    completion_destination_share: float
    expiration_source_share: float
    expiration_destination_share: float
    completion_reward: float
    expiration_penalty: float
    cpu_entry_slot: int | None
    service_eligible_slot: int | None
    deadline_slot: int
    cpu_service_window: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "source_uav": self.source_uav,
            "destination_uav": self.destination_uav,
            "terminal_state": self.terminal_state,
            "completion_allocation": {
                "source_share": self.completion_source_share,
                "destination_share": self.completion_destination_share,
                "source_reward": self.completion_source_share * self.completion_reward,
                "destination_reward": (
                    self.completion_destination_share * self.completion_reward
                ),
            },
            "expiration_allocation": {
                "source_share": self.expiration_source_share,
                "destination_share": self.expiration_destination_share,
                "source_penalty": self.expiration_source_share * self.expiration_penalty,
                "destination_penalty": (
                    self.expiration_destination_share * self.expiration_penalty
                ),
            },
            "cpu_entry_slot": self.cpu_entry_slot,
            "service_eligible_slot": self.service_eligible_slot,
            "deadline_slot": self.deadline_slot,
            "cpu_service_window": self.cpu_service_window,
        }


@dataclass(frozen=True)
class AgentRewardTerms:
    """Role-based per-agent accounting that exactly conserves team reward."""

    slot: int
    uav_count: int
    team_reward: float
    agent_reward: tuple[float, ...]
    completion_credit: tuple[float, ...]
    expiration_count: tuple[float, ...]
    communication_workload_s: tuple[float, ...]
    computation_workload_s: tuple[float, ...]
    urgent_workload_s: tuple[float, ...]
    transmit_energy_j: tuple[float, ...]
    cpu_energy_j: tuple[float, ...]
    actual_energy_j: tuple[float, ...]
    completion_component: tuple[float, ...]
    expiration_penalty: tuple[float, ...]
    workload_penalty: tuple[float, ...]
    energy_penalty: tuple[float, ...]
    conservation_residual: float
    remote_completions: tuple[RemoteCompletionAttribution, ...]
    terminal_attributions: tuple[TaskTerminalAttribution, ...] = ()
    credit_assignment_mode: str = CreditAssignmentMode.LEGACY.value
    beta_completion_credit: float = 0.5
    mu_expiration_credit: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.uav_count, bool) or not isinstance(self.uav_count, int):
            raise RewardError("uav_count must be an integer")
        if self.uav_count <= 0:
            raise RewardError("uav_count must be positive")
        vector_names = (
            "agent_reward",
            "completion_credit",
            "expiration_count",
            "communication_workload_s",
            "computation_workload_s",
            "urgent_workload_s",
            "transmit_energy_j",
            "cpu_energy_j",
            "actual_energy_j",
            "completion_component",
            "expiration_penalty",
            "workload_penalty",
            "energy_penalty",
        )
        for name in vector_names:
            values = getattr(self, name)
            if len(values) != self.uav_count:
                raise RewardError(f"{name} must have shape [A]")
            if not all(math.isfinite(value) for value in values):
                raise RewardError(f"{name} contains NaN or Inf")
        if not math.isfinite(self.team_reward):
            raise RewardError("team_reward must be finite")
        if not math.isfinite(self.conservation_residual):
            raise RewardError("conservation_residual must be finite")
        if abs(self.conservation_residual) > AGENT_REWARD_CONSERVATION_TOLERANCE:
            raise RewardError("per-agent reward violates team-reward conservation")

    def to_dict(self) -> dict[str, object]:
        """Return additive JSON-safe telemetry without replacing scalar reward."""

        destination = [
            {
                "uav_id": uav_id,
                "cpu_energy_j": self.cpu_energy_j[uav_id],
                "agent_reward": self.agent_reward[uav_id],
                "completion_credit": self.completion_credit[uav_id],
            }
            for uav_id in range(self.uav_count)
        ]
        return {
            "team_reward": self.team_reward,
            "agent_reward": list(self.agent_reward),
            "components": {
                "completion": list(self.completion_credit),
                "expiration": list(self.expiration_count),
                "workload": list(self.urgent_workload_s),
                "energy": list(self.actual_energy_j),
                "completion_component": list(self.completion_component),
                "expiration_penalty": list(self.expiration_penalty),
                "workload_penalty": list(self.workload_penalty),
                "energy_penalty": list(self.energy_penalty),
            },
            "workload_attribution": {
                "communication_s": list(self.communication_workload_s),
                "computation_s": list(self.computation_workload_s),
            },
            "energy_attribution": {
                "transmit_j": list(self.transmit_energy_j),
                "cpu_j": list(self.cpu_energy_j),
            },
            "conservation_residual": self.conservation_residual,
            "conservation_tolerance": AGENT_REWARD_CONSERVATION_TOLERANCE,
            "remote_completions": [item.to_dict() for item in self.remote_completions],
            "terminal_attributions": [item.to_dict() for item in self.terminal_attributions],
            "credit_assignment_mode": self.credit_assignment_mode,
            "beta_completion_credit": self.beta_completion_credit,
            "mu_expiration_credit": self.mu_expiration_credit,
            "destination": destination,
        }


class RewardCalculator:
    """Evaluate the unique frozen reward with immutable normalization values."""

    def __init__(self, references: RewardReferences, weights: RewardWeights) -> None:
        if not isinstance(references, RewardReferences):
            raise TypeError("references must be a RewardReferences instance")
        if not isinstance(weights, RewardWeights):
            raise TypeError("weights must be a RewardWeights instance")
        self.references = references
        self.weights = weights

    def calculate(
        self,
        *,
        slot: int,
        all_tasks: Iterable[Task],
        settled_tasks: Iterable[Task],
        actual_energy_j: float,
        workload_snapshots: Mapping[int, TaskWorkloadSnapshot] | None = None,
    ) -> RewardTerms:
        """Calculate reward after settlement and before slot-end arrivals.

        ``all_tasks`` is the lifecycle's full task collection at that boundary.
        ``settled_tasks`` contains only tasks completed or expired in ``slot``.
        Current-slot arrivals must not yet be present in either collection.
        Optional snapshots replace workload fields only for tasks newly bound
        at the end of this slot; no lifecycle state is mutated or reconstructed.
        """

        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise RewardError("slot must be a non-negative integer")
        energy = _finite_nonnegative(actual_energy_j, "actual_energy_j")
        tasks = self._unique_tasks(all_tasks, "all_tasks")
        settled = self._unique_tasks(settled_tasks, "settled_tasks")
        by_id = {task.task_id: task for task in tasks}
        snapshots = self._workload_snapshot_by_id(workload_snapshots, by_id)
        for task in settled:
            if by_id.get(task.task_id) is not task:
                raise RewardError("settled_tasks must reference objects in all_tasks")
            if task.outcome not in {TaskOutcome.DONE, TaskOutcome.EXPIRED}:
                raise RewardError("settled_tasks may contain only done or expired tasks")

        active_tasks = tuple(task for task in tasks if not task.is_terminal)
        urgent_workload = math.fsum(
            self._urgent_workload_for_task(task, slot, snapshots.get(task.task_id))
            for task in active_tasks
        )
        completed = sum(task.outcome is TaskOutcome.DONE for task in settled)
        expired = sum(task.outcome is TaskOutcome.EXPIRED for task in settled)

        refs = self.references
        normalized_completed = completed / refs.task_count_reference
        normalized_expired = expired / refs.task_count_reference
        normalized_workload = urgent_workload / refs.workload_reference_s
        normalized_energy = energy / refs.active_energy_reference_j

        completion_component = self.weights.completion * normalized_completed
        expiration_penalty = self.weights.expiration * normalized_expired
        workload_penalty = self.weights.workload * normalized_workload
        energy_penalty = self.weights.energy * normalized_energy
        reward = (
            completion_component
            - expiration_penalty
            - workload_penalty
            - energy_penalty
        )
        if not math.isfinite(reward):
            raise RewardError("reward must be finite")
        return RewardTerms(
            slot=slot,
            completed_task_count=completed,
            expired_task_count=expired,
            urgent_workload_s=urgent_workload,
            actual_energy_j=energy,
            normalized_completed=normalized_completed,
            normalized_expired=normalized_expired,
            normalized_workload=normalized_workload,
            normalized_energy=normalized_energy,
            completion_component=completion_component,
            expiration_penalty=expiration_penalty,
            workload_penalty=workload_penalty,
            energy_penalty=energy_penalty,
            reward=reward,
        )

    def decompose_agent_credit(
        self,
        *,
        slot: int,
        uav_count: int,
        all_tasks: Iterable[Task],
        settled_tasks: Iterable[Task],
        team_terms: RewardTerms,
        agent_transmit_energy_j: Mapping[int, float],
        agent_cpu_energy_j: Mapping[int, float],
        workload_snapshots: Mapping[int, TaskWorkloadSnapshot] | None = None,
        credit_config: RewardCreditConfig | None = None,
    ) -> AgentRewardTerms:
        """Decompose existing team accounting without redefining its formula."""

        credit = RewardCreditConfig() if credit_config is None else credit_config
        if not isinstance(credit, RewardCreditConfig):
            raise TypeError("credit_config must be a RewardCreditConfig")
        responsibility = (
            credit.credit_assignment_mode is CreditAssignmentMode.RESPONSIBILITY_TERMINAL
        )
        if isinstance(uav_count, bool) or not isinstance(uav_count, int) or uav_count <= 0:
            raise RewardError("uav_count must be a positive integer")
        if not isinstance(team_terms, RewardTerms) or team_terms.slot != slot:
            raise RewardError("team_terms must describe the requested slot")
        tasks = self._unique_tasks(all_tasks, "all_tasks")
        settled = self._unique_tasks(settled_tasks, "settled_tasks")
        by_id = {task.task_id: task for task in tasks}
        snapshots = self._workload_snapshot_by_id(workload_snapshots, by_id)
        for task in settled:
            if by_id.get(task.task_id) is not task:
                raise RewardError("settled_tasks must reference objects in all_tasks")
            if task.outcome not in {TaskOutcome.DONE, TaskOutcome.EXPIRED}:
                raise RewardError("settled_tasks may contain only done or expired tasks")

        completion_parts: list[list[float]] = [[] for _ in range(uav_count)]
        expiration_parts: list[list[float]] = [[] for _ in range(uav_count)]
        communication_parts: list[list[float]] = [[] for _ in range(uav_count)]
        computation_parts: list[list[float]] = [[] for _ in range(uav_count)]
        remote_completions: list[RemoteCompletionAttribution] = []
        terminal_attributions: list[TaskTerminalAttribution] = []
        refs = self.references

        for task in settled:
            source = self._agent_id(task.source_uav, uav_count, "task.source_uav")
            destination = self._agent_id(
                task.source_uav if task.destination is None else task.destination,
                uav_count,
                "task.destination",
            )
            remote = destination != source
            # A lifecycle service window is not a claim of sufficient capacity
            # or actual service. Never gate responsibility on received cycles:
            # otherwise an executor could avoid it by idling until expiration.
            cpu_service_window = (
                remote
                and task.cpu_entry_slot is not None
                and task.service_eligible_slot is not None
                and task.service_eligible_slot <= task.deadline_slot
            )
            completion_source = completion_destination = 0.0
            expiration_source = expiration_destination = 0.0
            if task.outcome is TaskOutcome.EXPIRED:
                expiration_destination = (
                    credit.mu_expiration_credit
                    if responsibility and cpu_service_window
                    else 0.0
                )
                expiration_source = 1.0 - expiration_destination
                expiration_parts[source].append(expiration_source)
                if remote:
                    expiration_parts[destination].append(expiration_destination)
            elif not remote:
                completion_source = 1.0
                completion_parts[source].append(completion_source)
            else:
                communication_reference = task.data_bits / refs.reference_rate_bps
                computation_reference = task.cpu_cycles / refs.reference_cpu_frequency_hz
                eta = communication_reference / (
                    communication_reference + computation_reference
                )
                completion_source = eta
                if responsibility:
                    beta = credit.beta_completion_credit
                    completion_source = (1.0 - beta) * eta + beta / 2.0
                completion_destination = 1.0 - completion_source
                completion_parts[source].append(completion_source)
                completion_parts[destination].append(completion_destination)
                remote_completions.append(
                    RemoteCompletionAttribution(
                        task_id=task.task_id,
                        source_uav=source,
                        destination_uav=destination,
                        source_share=completion_source,
                        destination_share=completion_destination,
                    )
                )
            terminal_attributions.append(
                TaskTerminalAttribution(
                    task_id=task.task_id,
                    source_uav=source,
                    destination_uav=task.destination,
                    terminal_state=task.outcome.name,
                    completion_source_share=completion_source,
                    completion_destination_share=completion_destination,
                    expiration_source_share=expiration_source,
                    expiration_destination_share=expiration_destination,
                    completion_reward=self.weights.completion / refs.task_count_reference,
                    expiration_penalty=self.weights.expiration / refs.task_count_reference,
                    cpu_entry_slot=task.cpu_entry_slot,
                    service_eligible_slot=task.service_eligible_slot,
                    deadline_slot=task.deadline_slot,
                    cpu_service_window=cpu_service_window,
                )
            )

        for task in tasks:
            if task.is_terminal:
                continue
            source = self._agent_id(task.source_uav, uav_count, "task.source_uav")
            owner = self._agent_id(
                task.source_uav if task.destination is None else task.destination,
                uav_count,
                "task.destination",
            )
            workload_source = snapshots.get(task.task_id, task)
            remaining_bits = _finite_nonnegative(
                workload_source.remaining_bits, "task.remaining_bits"
            )
            remaining_cycles = _finite_nonnegative(
                workload_source.remaining_cycles, "task.remaining_cycles"
            )
            denominator = max(1, task.deadline_slot - slot + 1)
            communication_parts[source].append(
                remaining_bits / refs.reference_rate_bps / denominator
            )
            computation_parts[owner].append(
                remaining_cycles / refs.reference_cpu_frequency_hz / denominator
            )

        completion = tuple(math.fsum(parts) for parts in completion_parts)
        expiration = tuple(math.fsum(parts) for parts in expiration_parts)
        communication = tuple(math.fsum(parts) for parts in communication_parts)
        computation = tuple(math.fsum(parts) for parts in computation_parts)
        workload = tuple(
            math.fsum((communication[index], computation[index]))
            for index in range(uav_count)
        )
        tx_energy = self._agent_vector(
            agent_transmit_energy_j, uav_count, "agent_transmit_energy_j"
        )
        cpu_energy = self._agent_vector(
            agent_cpu_energy_j, uav_count, "agent_cpu_energy_j"
        )
        energy = tuple(
            math.fsum((tx_energy[index], cpu_energy[index]))
            for index in range(uav_count)
        )

        self._assert_conserved(completion, team_terms.completed_task_count, "completion")
        self._assert_conserved(expiration, team_terms.expired_task_count, "expiration")
        self._assert_conserved(workload, team_terms.urgent_workload_s, "workload")
        self._assert_conserved(energy, team_terms.actual_energy_j, "energy")

        weights = self.weights
        completion_component = tuple(
            weights.completion * value / refs.task_count_reference
            for value in completion
        )
        expiration_penalty = tuple(
            weights.expiration * value / refs.task_count_reference
            for value in expiration
        )
        workload_penalty = tuple(
            weights.workload * value / refs.workload_reference_s
            for value in workload
        )
        energy_penalty = tuple(
            weights.energy * value / refs.active_energy_reference_j
            for value in energy
        )
        agent_reward = tuple(
            completion_component[index]
            - expiration_penalty[index]
            - workload_penalty[index]
            - energy_penalty[index]
            for index in range(uav_count)
        )
        self._assert_conserved(
            completion_component, team_terms.completion_component, "completion component"
        )
        self._assert_conserved(
            expiration_penalty, team_terms.expiration_penalty, "expiration penalty"
        )
        self._assert_conserved(
            workload_penalty, team_terms.workload_penalty, "workload penalty"
        )
        self._assert_conserved(
            energy_penalty, team_terms.energy_penalty, "energy penalty"
        )
        residual = math.fsum(agent_reward) - team_terms.reward
        if abs(residual) > AGENT_REWARD_CONSERVATION_TOLERANCE:
            raise RewardError("per-agent reward violates team-reward conservation")
        return AgentRewardTerms(
            slot=slot,
            uav_count=uav_count,
            team_reward=team_terms.reward,
            agent_reward=agent_reward,
            completion_credit=completion,
            expiration_count=expiration,
            communication_workload_s=communication,
            computation_workload_s=computation,
            urgent_workload_s=workload,
            transmit_energy_j=tx_energy,
            cpu_energy_j=cpu_energy,
            actual_energy_j=energy,
            completion_component=completion_component,
            expiration_penalty=expiration_penalty,
            workload_penalty=workload_penalty,
            energy_penalty=energy_penalty,
            conservation_residual=residual,
            remote_completions=tuple(remote_completions),
            terminal_attributions=tuple(terminal_attributions),
            credit_assignment_mode=credit.credit_assignment_mode.value,
            beta_completion_credit=credit.beta_completion_credit,
            mu_expiration_credit=credit.mu_expiration_credit,
        )

    @staticmethod
    def _agent_id(value: int, uav_count: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise RewardError(f"{name} must be an integer")
        if value < 0 or value >= uav_count:
            raise RewardError(f"{name} lies outside the agent axis")
        return value

    @classmethod
    def _agent_vector(
        cls,
        values: Mapping[int, float],
        uav_count: int,
        name: str,
    ) -> tuple[float, ...]:
        if not isinstance(values, Mapping):
            raise TypeError(f"{name} must be a mapping")
        result = [0.0] * uav_count
        for raw_id, raw_value in values.items():
            agent_id = cls._agent_id(raw_id, uav_count, f"{name} key")
            result[agent_id] = _finite_nonnegative(raw_value, f"{name}[{agent_id}]")
        return tuple(result)

    @staticmethod
    def _assert_conserved(
        values: tuple[float, ...],
        team_value: float,
        name: str,
    ) -> None:
        if abs(math.fsum(values) - float(team_value)) > AGENT_REWARD_CONSERVATION_TOLERANCE:
            raise RewardError(f"per-agent {name} attribution is not conservative")

    def _urgent_workload_for_task(
        self,
        task: Task,
        slot: int,
        snapshot: TaskWorkloadSnapshot | None = None,
    ) -> float:
        source = task if snapshot is None else snapshot
        remaining_bits = _finite_nonnegative(source.remaining_bits, "task.remaining_bits")
        remaining_cycles = _finite_nonnegative(
            source.remaining_cycles,
            "task.remaining_cycles",
        )
        denominator = max(1, task.deadline_slot - slot + 1)
        return (
            remaining_bits / self.references.reference_rate_bps
            + remaining_cycles / self.references.reference_cpu_frequency_hz
        ) / denominator

    @staticmethod
    def _workload_snapshot_by_id(
        snapshots: Mapping[int, TaskWorkloadSnapshot] | None,
        tasks_by_id: Mapping[int, Task],
    ) -> dict[int, TaskWorkloadSnapshot]:
        if snapshots is None:
            return {}
        if not isinstance(snapshots, Mapping):
            raise TypeError("workload_snapshots must be a mapping")
        validated: dict[int, TaskWorkloadSnapshot] = {}
        for task_id, snapshot in snapshots.items():
            if isinstance(task_id, bool) or not isinstance(task_id, int):
                raise RewardError("workload snapshot keys must be task IDs")
            if not isinstance(snapshot, TaskWorkloadSnapshot):
                raise TypeError("workload snapshots must contain TaskWorkloadSnapshot values")
            if snapshot.task_id != task_id:
                raise RewardError("workload snapshot key and task_id disagree")
            if task_id not in tasks_by_id:
                raise RewardError("workload snapshot references an unknown task")
            validated[task_id] = snapshot
        return validated

    @staticmethod
    def _unique_tasks(tasks: Iterable[Task], name: str) -> tuple[Task, ...]:
        materialized = tuple(tasks)
        if any(not isinstance(task, Task) for task in materialized):
            raise TypeError(f"{name} must contain only Task instances")
        ids = [task.task_id for task in materialized]
        if len(ids) != len(set(ids)):
            raise RewardError(f"{name} contains duplicate task IDs")
        return tuple(sorted(materialized, key=lambda task: task.task_id))


def compute_reward(
    *,
    slot: int,
    all_tasks: Iterable[Task],
    settled_tasks: Iterable[Task],
    actual_energy_j: float,
    references: RewardReferences,
    weights: RewardWeights,
    workload_snapshots: Mapping[int, TaskWorkloadSnapshot] | None = None,
) -> RewardTerms:
    """Functional facade for callers that do not retain a calculator object."""

    return RewardCalculator(references, weights).calculate(
        slot=slot,
        all_tasks=all_tasks,
        settled_tasks=settled_tasks,
        actual_energy_j=actual_energy_j,
        workload_snapshots=workload_snapshots,
    )


__all__ = [
    "AGENT_REWARD_CONSERVATION_TOLERANCE",
    "AgentRewardTerms",
    "RemoteCompletionAttribution",
    "RewardCalculator",
    "RewardError",
    "RewardReferences",
    "RewardTerms",
    "RewardWeights",
    "TaskTerminalAttribution",
    "TaskWorkloadSnapshot",
    "compute_reward",
]
