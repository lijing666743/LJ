"""Immutable actor-safe features for the immediately preceding action.

The records in this module deliberately retain the policy proposal separately
from the executor outcome.  They expose only the previous slot's own proposal
and normalized resource use; rejection reasons, energy budgets, channel truth,
and other physical-state details are outside the actor feature boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from ..config import ActionConfig, EnvironmentConfig
from .actions import ActionProposal
from .energy import UavEnergyState
from .executor import JointExecutionResult


class PreviousActionError(ValueError):
    """Raised when a previous-action snapshot is temporally inconsistent."""


def _require_slot(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PreviousActionError(f"{name} must be an integer >= {minimum}")
    return value


def _require_identifier(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PreviousActionError(f"{name} must be a non-negative integer or None")
    return value


def _require_ratio(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreviousActionError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise PreviousActionError(f"{name} must be finite and lie in [0, 1]")
    return converted


def _require_branch_value(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise PreviousActionError(f"proposal branch {name} is not JSON-safe")
    if isinstance(value, float) and not math.isfinite(value):
        raise PreviousActionError(f"proposal branch {name} must be finite")


def _validate_environment(environment: EnvironmentConfig) -> None:
    if not isinstance(environment, EnvironmentConfig):
        raise TypeError("environment must be an EnvironmentConfig")
    if environment.uav_count <= 0 or environment.ru_count <= 0:
        raise PreviousActionError("configured UAV and RU counts must be positive")


def _normalised_execution_ratio(value: float, maximum: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreviousActionError(f"{name} must be numeric")
    if isinstance(maximum, bool) or not isinstance(maximum, (int, float)):
        raise PreviousActionError(f"maximum for {name} must be numeric")
    converted = float(value)
    upper = float(maximum)
    if not math.isfinite(converted) or converted < 0.0:
        raise PreviousActionError(f"{name} must be finite and non-negative")
    if not math.isfinite(upper) or upper <= 0.0:
        raise PreviousActionError(f"maximum for {name} must be finite and positive")
    tolerance = 1.0e-12 * max(1.0, upper)
    if converted > upper + tolerance:
        raise PreviousActionError(f"{name} exceeds the configured UAV maximum")
    return min(1.0, converted / upper)


@dataclass(frozen=True)
class PreviousActionFeatures:
    """One UAV's immutable previous-proposal and executed-utilization features."""

    uav_id: int
    source_slot: int
    valid: bool
    proposal: ActionProposal
    communication_accepted: bool
    communication_receiver_uav: int | None
    executed_ru_count: int
    ru_utilization_ratio: float
    executed_power_ratio: float
    cpu_source_uav: int | None
    cpu_active: bool
    executed_cpu_frequency_ratio: float

    def __post_init__(self) -> None:
        _require_identifier(self.uav_id, "uav_id")
        if isinstance(self.source_slot, bool) or not isinstance(self.source_slot, int):
            raise PreviousActionError("source_slot must be an integer")
        if self.source_slot < -1:
            raise PreviousActionError("source_slot must be -1 or non-negative")
        if not isinstance(self.valid, bool):
            raise PreviousActionError("valid must be boolean")
        if self.valid != (self.source_slot >= 0):
            raise PreviousActionError("valid must be true exactly when source_slot is non-negative")
        if not isinstance(self.proposal, ActionProposal):
            raise TypeError("proposal must be an ActionProposal")
        if self.proposal.uav_id != self.uav_id:
            raise PreviousActionError("proposal UAV identifier differs from feature UAV identifier")
        for name, value in zip(
            (
                "route",
                "tx_select",
                "resource_group",
                "resource_width",
                "power_level",
                "cpu_queue",
                "cpu_frequency",
            ),
            self.proposal.branches,
        ):
            _require_branch_value(value, name)
        if not isinstance(self.communication_accepted, bool):
            raise PreviousActionError("communication_accepted must be boolean")
        receiver = _require_identifier(
            self.communication_receiver_uav,
            "communication_receiver_uav",
        )
        if receiver == self.uav_id:
            raise PreviousActionError("a UAV cannot be its own executed receiver")
        if (
            isinstance(self.executed_ru_count, bool)
            or not isinstance(self.executed_ru_count, int)
            or self.executed_ru_count < 0
        ):
            raise PreviousActionError("executed_ru_count must be a non-negative integer")
        ru_ratio = _require_ratio(self.ru_utilization_ratio, "ru_utilization_ratio")
        power_ratio = _require_ratio(self.executed_power_ratio, "executed_power_ratio")
        cpu_source = _require_identifier(self.cpu_source_uav, "cpu_source_uav")
        if not isinstance(self.cpu_active, bool):
            raise PreviousActionError("cpu_active must be boolean")
        frequency_ratio = _require_ratio(
            self.executed_cpu_frequency_ratio,
            "executed_cpu_frequency_ratio",
        )
        if self.communication_accepted:
            if receiver is None or self.executed_ru_count == 0 or ru_ratio <= 0.0 or power_ratio <= 0.0:
                raise PreviousActionError("accepted communication requires receiver, RUs and positive power")
        elif receiver is not None or self.executed_ru_count != 0 or ru_ratio != 0.0 or power_ratio != 0.0:
            raise PreviousActionError("non-accepted communication features must be canonical zero/None")
        if self.cpu_active:
            if cpu_source is None or frequency_ratio <= 0.0:
                raise PreviousActionError("active CPU execution requires a source and positive frequency")
        elif cpu_source is not None or frequency_ratio != 0.0:
            raise PreviousActionError("inactive CPU features must be canonical zero/None")
        if not self.valid and (self.communication_accepted or self.cpu_active):
            raise PreviousActionError("invalid reset features cannot contain executed activity")

    @property
    def proposal_branches(self) -> tuple[object, ...]:
        """Return the seven policy branches in their frozen sampling order."""

        return self.proposal.branches

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe scalar-only representation."""

        return {
            "uav_id": self.uav_id,
            "source_slot": self.source_slot,
            "valid": self.valid,
            "proposal": {
                "route": self.proposal.route,
                "tx_select": self.proposal.tx_select,
                "resource_group": self.proposal.resource_group,
                "resource_width": self.proposal.resource_width,
                "power_level": self.proposal.power_level,
                "cpu_queue": self.proposal.cpu_queue,
                "cpu_frequency": self.proposal.cpu_frequency,
            },
            "executed_communication": {
                "accepted": self.communication_accepted,
                "receiver_uav": self.communication_receiver_uav,
                "ru_count": self.executed_ru_count,
                "ru_utilization_ratio": self.ru_utilization_ratio,
                "power_ratio": self.executed_power_ratio,
            },
            "executed_cpu": {
                "source_uav": self.cpu_source_uav,
                "active": self.cpu_active,
                "frequency_ratio": self.executed_cpu_frequency_ratio,
            },
        }


@dataclass(frozen=True)
class PreviousActionSnapshot:
    """Immutable all-UAV features aligned to one actor decision slot."""

    actor_slot: int
    source_slot: int
    features: tuple[PreviousActionFeatures, ...]

    def __post_init__(self) -> None:
        _require_slot(self.actor_slot, "actor_slot")
        if isinstance(self.source_slot, bool) or not isinstance(self.source_slot, int):
            raise PreviousActionError("source_slot must be an integer")
        if self.source_slot < -1 or self.source_slot >= self.actor_slot:
            raise PreviousActionError("source_slot must be -1 or strictly earlier than actor_slot")
        copied = tuple(self.features)
        if not copied:
            raise PreviousActionError("previous-action snapshot must contain at least one UAV")
        object.__setattr__(self, "features", copied)
        expected_ids = tuple(range(len(copied)))
        if tuple(item.uav_id for item in copied) != expected_ids:
            raise PreviousActionError("features must use stable zero-based UAV order")
        if any(item.source_slot != self.source_slot for item in copied):
            raise PreviousActionError("all feature source slots must match the snapshot source slot")
        if self.source_slot == -1:
            if self.actor_slot != 0 or any(item.valid for item in copied):
                raise PreviousActionError("only actor slot 0 may contain the invalid reset snapshot")
        else:
            if self.source_slot + 1 != self.actor_slot:
                raise PreviousActionError("previous action must come from the immediately preceding slot")
            if not all(item.valid for item in copied):
                raise PreviousActionError("execution-derived features must all be valid")
        count = len(copied)
        for item in copied:
            for name, identifier in (
                ("communication receiver", item.communication_receiver_uav),
                ("CPU source", item.cpu_source_uav),
            ):
                if identifier is not None and identifier >= count:
                    raise PreviousActionError(f"{name} lies outside the snapshot UAV range")

    @property
    def valid_mask(self) -> tuple[bool, ...]:
        """Return the immutable per-UAV validity mask."""

        return tuple(item.valid for item in self.features)

    @property
    def proposals(self) -> tuple[ActionProposal, ...]:
        """Return the preserved policy proposals in stable UAV order."""

        return tuple(item.proposal for item in self.features)

    def for_uav(self, uav_id: int) -> PreviousActionFeatures:
        """Return one UAV's features without exposing a mutable container."""

        if isinstance(uav_id, bool) or not isinstance(uav_id, int):
            raise KeyError(f"invalid UAV identifier {uav_id!r}")
        if not 0 <= uav_id < len(self.features):
            raise KeyError(f"no previous-action features for UAV {uav_id}")
        return self.features[uav_id]

    @classmethod
    def reset(
        cls,
        actor_slot: int = 0,
        *,
        environment: EnvironmentConfig,
        action: ActionConfig | None = None,
    ) -> "PreviousActionSnapshot":
        """Return canonical invalid history at the initial actor slot."""

        _validate_environment(environment)
        if actor_slot != 0:
            raise PreviousActionError("reset previous-action history is defined only at actor slot 0")
        action_config = ActionConfig() if action is None else action
        if not isinstance(action_config, ActionConfig):
            raise TypeError("action must be an ActionConfig or None")
        required = (
            "route",
            "tx_select",
            "resource_group",
            "resource_width",
            "power_level",
            "cpu_queue",
            "cpu_frequency",
        )
        missing = [name for name in required if name not in action_config.canonical_inactive_values]
        if missing:
            raise PreviousActionError(f"canonical inactive action is missing branches {missing}")
        canonical = action_config.canonical_inactive_values
        features = tuple(
            PreviousActionFeatures(
                uav_id=uav_id,
                source_slot=-1,
                valid=False,
                proposal=ActionProposal(
                    uav_id=uav_id,
                    route=canonical["route"],
                    tx_select=canonical["tx_select"],
                    resource_group=canonical["resource_group"],
                    resource_width=canonical["resource_width"],
                    power_level=canonical["power_level"],
                    cpu_queue=canonical["cpu_queue"],
                    cpu_frequency=canonical["cpu_frequency"],
                ),
                communication_accepted=False,
                communication_receiver_uav=None,
                executed_ru_count=0,
                ru_utilization_ratio=0.0,
                executed_power_ratio=0.0,
                cpu_source_uav=None,
                cpu_active=False,
                executed_cpu_frequency_ratio=0.0,
            )
            for uav_id in range(environment.uav_count)
        )
        return cls(actor_slot=actor_slot, source_slot=-1, features=features)

    @classmethod
    def from_execution(
        cls,
        *,
        actor_slot: int,
        result: JointExecutionResult,
        environment: EnvironmentConfig,
        resource_states: Mapping[int, UavEnergyState],
    ) -> "PreviousActionSnapshot":
        """Build actor features from exactly the immediately preceding result."""

        _validate_environment(environment)
        _require_slot(actor_slot, "actor_slot", minimum=1)
        if not isinstance(result, JointExecutionResult):
            raise TypeError("result must be a JointExecutionResult")
        _require_slot(result.slot, "result.slot")
        if result.slot >= actor_slot:
            raise PreviousActionError("execution source_slot must be earlier than actor_slot")
        if result.slot + 1 != actor_slot:
            raise PreviousActionError("execution must come from the immediately preceding slot")

        expected_ids = set(range(environment.uav_count))
        if set(resource_states) != expected_ids:
            raise PreviousActionError("resource_states must contain every configured UAV exactly once")
        for uav_id in range(environment.uav_count):
            state = resource_states[uav_id]
            if not isinstance(state, UavEnergyState) or state.uav_id != uav_id:
                raise PreviousActionError("resource-state mapping key must equal state.uav_id")

        by_uav: dict[int, object] = {}
        for action_result in result.actions:
            uav_id = action_result.proposal.uav_id
            if uav_id in by_uav:
                raise PreviousActionError("joint execution contains duplicate UAV identifiers")
            by_uav[uav_id] = action_result
        if set(by_uav) != expected_ids:
            raise PreviousActionError("joint execution must contain every configured UAV exactly once")

        features: list[PreviousActionFeatures] = []
        for uav_id in range(environment.uav_count):
            action_result = by_uav[uav_id]
            proposal = action_result.proposal
            communication = action_result.communication
            cpu = action_result.cpu
            if communication.sender_uav != uav_id or cpu.executor_uav != uav_id:
                raise PreviousActionError("nested execution UAV identifiers are inconsistent")
            ru_indices = tuple(communication.executed_ru_indices)
            if len(ru_indices) != len(set(ru_indices)):
                raise PreviousActionError("executed RU indices must be unique")
            if any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 1 <= index <= environment.ru_count
                for index in ru_indices
            ):
                raise PreviousActionError("executed RU indices are outside the configured domain")
            accepted = bool(communication.accepted)
            if not accepted and (ru_indices or communication.executed_power_w != 0.0):
                raise PreviousActionError("non-accepted communication execution is not canonical")
            receiver = communication.receiver_uav if accepted else None
            if receiver is not None and not 0 <= receiver < environment.uav_count:
                raise PreviousActionError("executed communication receiver is out of range")
            state = resource_states[uav_id]
            power_ratio = _normalised_execution_ratio(
                communication.executed_power_w,
                state.max_transmit_power_w,
                "executed_power_w",
            )
            cpu_active = bool(cpu.active)
            if not cpu_active and cpu.executed_frequency_hz != 0.0:
                raise PreviousActionError("inactive CPU execution must have zero frequency")
            cpu_source = cpu.queue_source_uav if cpu_active else None
            if cpu_source is not None and not 0 <= cpu_source < environment.uav_count:
                raise PreviousActionError("executed CPU source is out of range")
            frequency_ratio = _normalised_execution_ratio(
                cpu.executed_frequency_hz,
                state.max_cpu_frequency_hz,
                "executed_frequency_hz",
            )
            features.append(
                PreviousActionFeatures(
                    uav_id=uav_id,
                    source_slot=result.slot,
                    valid=True,
                    proposal=proposal,
                    communication_accepted=accepted,
                    communication_receiver_uav=receiver,
                    executed_ru_count=len(ru_indices),
                    ru_utilization_ratio=len(ru_indices) / environment.ru_count,
                    executed_power_ratio=power_ratio,
                    cpu_source_uav=cpu_source,
                    cpu_active=cpu_active,
                    executed_cpu_frequency_ratio=frequency_ratio,
                )
            )
        return cls(
            actor_slot=actor_slot,
            source_slot=result.slot,
            features=tuple(features),
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe snapshot without executor metadata."""

        return {
            "actor_slot": self.actor_slot,
            "source_slot": self.source_slot,
            "valid_mask": list(self.valid_mask),
            "features": [item.snapshot() for item in self.features],
        }


__all__ = [
    "PreviousActionError",
    "PreviousActionFeatures",
    "PreviousActionSnapshot",
]
