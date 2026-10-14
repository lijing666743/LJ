"""Full, deterministic orchestration of the U2U MEC environment.

This module owns episode reset and slot ordering.  Physical calculations stay
in their dedicated mobility, channel, executor, service, energy, observation,
reward, and metrics modules; the environment only connects those components at
their frozen causal boundaries.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
import copy
import hashlib
import json
import math
from typing import Any

import numpy as np

from ..config import AgentCreditMode, RunConfig, WorkloadTimingMode
from .actions import ActionProposal
from .action_history import PreviousActionSnapshot
from .channel import ChannelSnapshot, PhysicalChannelModel
from .energy import UavEnergyState
from .executor import DeterministicExecutor, JointExecutionResult
from .history import ActorChannelFeatures, ChannelHistory
from .lifecycle import LifecycleManager
from .metrics import ConservationSnapshot, EpisodeMetrics
from .mobility import MobilityModel, MobilityState
from .observation import ActorObservation, ObservationBuilder
from .public_history import PublicMessageHistory, PublicMessageSnapshot
from .randomness import rng_from_run_config
from .reward import (
    AgentRewardTerms,
    RewardCalculator,
    RewardReferences,
    RewardTerms,
    RewardWeights,
    TaskWorkloadSnapshot,
)
from .service import PhysicalService, SlotPhysicalResult
from .state import CentralizedState, CentralizedStateBuilder
from .tasks import Task, TaskOutcome
from .topology import DynamicTopology, TopologySnapshot
from .traffic import ArrivalBatch, ArrivalEstimate, TrafficProcess


class EnvironmentError(ValueError):
    """Raised when reset/step ordering or a joint proposal is invalid."""


class EnvironmentFingerprintError(EnvironmentError):
    """Raised when a state contains a value without stable fingerprint support."""


def _canonical_fingerprint_value(
    value: Any,
    *,
    active: list[Any] | None = None,
) -> Any:
    """Convert supported environment state to deterministic JSON data.

    The serializer deliberately has no generic representation fallback.  A
    project-state value must either expose an explicitly supported stable
    representation or fail before a diagnostic can claim replay fidelity.
    """

    active_values = [] if active is None else active

    def enter(container: Any) -> None:
        if any(container is item for item in active_values):
            raise EnvironmentFingerprintError(
                f"cyclic state fingerprint value is unsupported: {type(container).__name__}"
            )
        active_values.append(container)

    def leave() -> None:
        active_values.pop()

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EnvironmentFingerprintError(
                "state fingerprint encountered a non-finite float"
            )
        return value
    if isinstance(value, np.generic):
        return {
            "kind": "numpy_scalar",
            "dtype": value.dtype.str,
            "value": _canonical_fingerprint_value(value.item(), active=active_values),
        }
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise EnvironmentFingerprintError(
                "object-dtype ndarray is unsupported for state fingerprinting"
            )
        if np.issubdtype(value.dtype, np.inexact) and not np.all(np.isfinite(value)):
            raise EnvironmentFingerprintError(
                "state fingerprint encountered a non-finite array"
            )
        array = np.ascontiguousarray(value)
        return {
            "kind": "ndarray",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, np.random.Generator):
        return {
            "kind": "numpy_generator",
            "bit_generator": type(value.bit_generator).__name__,
            "state": _canonical_fingerprint_value(
                copy.deepcopy(value.bit_generator.state), active=active_values
            ),
        }
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {
            "kind": "bytes",
            "sha256": hashlib.sha256(bytes(value)).hexdigest(),
            "length": len(value),
        }
    if isinstance(value, defaultdict):
        enter(value)
        try:
            factory = value.default_factory
            if factory is not None:
                factory_module = getattr(factory, "__module__", None)
                factory_qualname = getattr(factory, "__qualname__", None)
                if not isinstance(factory_module, str) or not isinstance(
                    factory_qualname, str
                ):
                    raise EnvironmentFingerprintError(
                        "defaultdict factory lacks stable callable identity"
                    )
                canonical_factory: Any = {
                    "module": factory_module,
                    "qualname": factory_qualname,
                }
            else:
                canonical_factory = None
            items = [
                (
                    _canonical_fingerprint_value(key, active=active_values),
                    _canonical_fingerprint_value(item, active=active_values),
                )
                for key, item in value.items()
            ]
            items.sort(
                key=lambda item: json.dumps(
                    item[0], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            )
            return {
                "kind": "defaultdict",
                "default_factory": canonical_factory,
                "items": items,
            }
        finally:
            leave()
    if isinstance(value, Mapping):
        enter(value)
        try:
            items = [
                (
                    _canonical_fingerprint_value(key, active=active_values),
                    _canonical_fingerprint_value(item, active=active_values),
                )
                for key, item in value.items()
            ]
            items.sort(
                key=lambda item: json.dumps(
                    item[0], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            )
            return {"kind": "mapping", "items": items}
        finally:
            leave()
    if isinstance(value, (list, tuple)):
        enter(value)
        try:
            return {
                "kind": "sequence",
                "type": type(value).__name__,
                "items": [
                    _canonical_fingerprint_value(item, active=active_values)
                    for item in value
                ],
            }
        finally:
            leave()
    if isinstance(value, deque):
        enter(value)
        try:
            return {
                "kind": "deque",
                "maxlen": value.maxlen,
                "items": [
                    _canonical_fingerprint_value(item, active=active_values)
                    for item in value
                ],
            }
        finally:
            leave()
    if isinstance(value, (set, frozenset)):
        enter(value)
        try:
            items = [
                _canonical_fingerprint_value(item, active=active_values)
                for item in value
            ]
            items.sort(
                key=lambda item: json.dumps(
                    item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            )
            return {"kind": "set", "items": items}
        finally:
            leave()
    if isinstance(value, Enum):
        return {
            "kind": "enum",
            "type": type(value).__qualname__,
            "value": _canonical_fingerprint_value(value.value, active=active_values),
        }

    if is_dataclass(value):
        enter(value)
        try:
            members = {
                item.name: _canonical_fingerprint_value(
                    getattr(value, item.name), active=active_values
                )
                for item in fields(value)
            }
            if hasattr(value, "__dict__"):
                declared = {item.name for item in fields(value)}
                extras = {
                    name: item
                    for name, item in vars(value).items()
                    if name not in declared
                }
                for name, item in sorted(extras.items()):
                    members[f"__extra__.{name}"] = _canonical_fingerprint_value(
                        item, active=active_values
                    )
            return {
                "kind": "dataclass",
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "members": members,
            }
        finally:
            leave()
    elif hasattr(value, "__dict__") and type(value).__module__.startswith("src.env."):
        enter(value)
        try:
            members = {
                name: _canonical_fingerprint_value(item, active=active_values)
                for name, item in sorted(vars(value).items())
            }
            return {
                "kind": "object",
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "members": members,
            }
        finally:
            leave()
    raise EnvironmentFingerprintError(
        "unsupported state fingerprint type: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _fingerprint_digest(value: Any) -> str:
    encoded = json.dumps(
        _canonical_fingerprint_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ResetResult:
    """Decision-time records returned by a successful episode reset."""

    observations: tuple[ActorObservation, ...]
    centralized_state: CentralizedState
    info: dict[str, Any]

    @property
    def observation(self) -> tuple[ActorObservation, ...]:
        """Singular-name compatibility alias for multi-agent callers."""

        return self.observations

    @property
    def actor_observations(self) -> tuple[ActorObservation, ...]:
        return self.observations

    @property
    def state(self) -> CentralizedState:
        return self.centralized_state


@dataclass(frozen=True)
class StepResult:
    """One environment transition and its next decision-time records."""

    observations: tuple[ActorObservation, ...] | None
    centralized_state: CentralizedState | None
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def observation(self) -> tuple[ActorObservation, ...] | None:
        return self.observations

    @property
    def actor_observations(self) -> tuple[ActorObservation, ...] | None:
        return self.observations

    @property
    def next_observations(self) -> tuple[ActorObservation, ...] | None:
        return self.observations

    @property
    def next_actor_observations(self) -> tuple[ActorObservation, ...] | None:
        return self.observations

    @property
    def state(self) -> CentralizedState | None:
        return self.centralized_state

    @property
    def next_state(self) -> CentralizedState | None:
        return self.centralized_state


class U2UMECEnvironment:
    """Full finite-horizon environment with explicit causal slot boundaries."""

    _STREAM_NAMES = (
        "reset_mobility",
        "task_arrival",
        "task_workload",
        "channel_fading",
        "csi_error",
    )

    def __init__(self, config: RunConfig) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        config.validate()
        self.config = config
        self.slot = 0
        self._has_reset = False
        self._done = False

        self.lifecycle: LifecycleManager | None = None
        self.mobility_model: MobilityModel | None = None
        self.mobility: MobilityState | None = None
        self.topology_model: DynamicTopology | None = None
        self.topology: TopologySnapshot | None = None
        self.channel_model: PhysicalChannelModel | None = None
        self.channel: ChannelSnapshot | None = None
        self.channel_history: ChannelHistory | None = None
        self.channel_features: ActorChannelFeatures | None = None
        self.public_history: PublicMessageHistory | None = None
        self.public_messages: PublicMessageSnapshot | None = None
        self.previous_actions: PreviousActionSnapshot | None = None
        self.traffic: TrafficProcess | None = None
        self.resource_states: dict[int, UavEnergyState] = {}
        self.initial_energy_j: dict[int, float] = {}
        self.executor: DeterministicExecutor | None = None
        self.physical_service: PhysicalService | None = None
        self.reward_calculator: RewardCalculator | None = None
        self.metrics: EpisodeMetrics | None = None
        self.observation_builder: ObservationBuilder | None = None
        self.state_builder: CentralizedStateBuilder | None = None
        self.current_observations: tuple[ActorObservation, ...] | None = None
        self.current_state: CentralizedState | None = None

        count = config.environment.uav_count
        self._last_effective_rate_bps = np.zeros((count, count), dtype=np.float64)
        self._last_rate_valid_mask = np.zeros((count, count), dtype=np.bool_)
        self._actual_attempt_counts = np.zeros((count, count), dtype=np.int64)
        self._outage_counts = np.zeros((count, count), dtype=np.int64)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def last_effective_rate_bps(self) -> np.ndarray:
        return self._readonly(self._last_effective_rate_bps)

    @property
    def last_rate_valid_mask(self) -> np.ndarray:
        return self._readonly(self._last_rate_valid_mask, dtype=np.bool_)

    @property
    def outage_rate(self) -> np.ndarray:
        values, _ = self._cumulative_outage()
        return self._readonly(values)

    @property
    def outage_valid_mask(self) -> np.ndarray:
        _, mask = self._cumulative_outage()
        return self._readonly(mask, dtype=np.bool_)

    def reset(self) -> ResetResult:
        """Rebuild all episode-owned components from the configured master seed."""

        self.config.validate()
        env = self.config.environment
        streams = {
            name: rng_from_run_config(self.config, name)
            for name in self._STREAM_NAMES
        }

        self.lifecycle = LifecycleManager()
        self.lifecycle.reset()
        self.mobility_model = MobilityModel(env, streams["reset_mobility"])
        self.topology_model = DynamicTopology(env.candidate_neighbor_radius_m)
        self.channel_model = PhysicalChannelModel(env, streams["channel_fading"])
        self.channel_history = ChannelHistory(env, streams["csi_error"])
        self.traffic = TrafficProcess(
            env,
            streams["task_arrival"],
            streams["task_workload"],
        )
        self.traffic.reset()

        self.resource_states, self.initial_energy_j = self._build_resource_states()
        self.executor = DeterministicExecutor(
            self.config,
            self.lifecycle,
            self.resource_states,
        )
        self.public_history = PublicMessageHistory.from_run_config(
            self.config,
            self.initial_energy_j,
        )
        self.public_messages = self.public_history.reset()
        self.previous_actions = PreviousActionSnapshot.reset(
            environment=env, action=self.config.action
        )
        self.physical_service = PhysicalService(
            self.config,
            self.lifecycle,
            self.resource_states,
        )
        references = RewardReferences(
            reference_rate_bps=env.reference_rate_bps,
            reference_cpu_frequency_hz=env.reference_cpu_frequency_hz,
            workload_reference_s=env.reward_workload_reference_s,
            task_count_reference=env.reward_task_count_reference,
            active_energy_reference_j=math.fsum(self.initial_energy_j.values()),
        )
        weights = RewardWeights(
            completion=env.reward_completion_weight,
            expiration=env.reward_expiration_weight,
            workload=env.reward_workload_weight,
            energy=env.reward_energy_weight,
        )
        self.reward_calculator = RewardCalculator(references, weights)
        self.metrics = EpisodeMetrics()
        self.observation_builder = ObservationBuilder.from_run_config(self.config)
        self.state_builder = CentralizedStateBuilder.from_run_config(self.config)

        count = env.uav_count
        self._last_effective_rate_bps = np.zeros((count, count), dtype=np.float64)
        self._last_rate_valid_mask = np.zeros((count, count), dtype=np.bool_)
        self._actual_attempt_counts = np.zeros((count, count), dtype=np.int64)
        self._outage_counts = np.zeros((count, count), dtype=np.int64)

        self.slot = 0
        self._done = False
        self.mobility = self.mobility_model.reset()
        self.topology = self.topology_model.compute(0, self.mobility.positions_m)
        self.channel = self.channel_model.reset(self.mobility.positions_m)
        self.channel_history.record_physical_channel(self.channel)
        self.channel_features = self.channel_history.actor_features(0)
        estimate = self.traffic.estimate(0)
        self.metrics.record_slot_start_backlog(0, self.lifecycle)
        self.current_observations, self.current_state = self._build_decision_records(estimate)
        self._has_reset = True
        conservation = self.assert_invariants()
        info = {
            "event": "reset",
            "run_id": self.config.run_id,
            "config_hash": self.config.config_hash,
            "slot": 0,
            "seed": self.config.seed,
            "stream_ids": {
                name: int(self.config.reproducibility.stream_ids[name])
                for name in self._STREAM_NAMES
            },
            "resources": self._resource_snapshot(),
            "metrics": self.metrics.snapshot(),
            "conservation": conservation.to_dict(),
        }
        return ResetResult(self.current_observations, self.current_state, info)

    def canonical_proposals(self) -> tuple[ActionProposal, ...]:
        """Return the deterministic all-inactive/defer legal joint proposal."""

        self._require_active()
        assert self.current_observations is not None
        canonical = self.config.action.canonical_inactive_values
        proposals: list[ActionProposal] = []
        for observation in self.current_observations:
            route = "defer" if observation.action_masks.route_branch_active else canonical["route"]
            proposal = ActionProposal(
                uav_id=observation.uav_id,
                route=route,
                tx_select=canonical["tx_select"],
                resource_group=canonical["resource_group"],
                resource_width=int(canonical["resource_width"]),
                power_level=float(canonical["power_level"]),
                cpu_queue=canonical["cpu_queue"],
                cpu_frequency=float(canonical["cpu_frequency"]),
            )
            if not observation.action_masks.is_legal(proposal):
                raise EnvironmentError("configured canonical proposal is not legal")
            proposals.append(proposal)
        return tuple(proposals)

    def step(self, proposals: Iterable[ActionProposal]) -> StepResult:
        """Execute one causally ordered slot and expose no horizon state."""

        self._require_active()
        joint = self._validate_proposals(proposals)
        assert self.lifecycle is not None
        assert self.executor is not None
        assert self.physical_service is not None
        assert self.reward_calculator is not None
        assert self.metrics is not None
        assert self.topology is not None
        assert self.channel is not None
        assert self.channel_features is not None
        assert self.channel_history is not None
        assert self.traffic is not None

        assert self.public_history is not None
        assert self.mobility is not None
        slot = self.slot
        slot_start_heads = {
            uav_id: (
                self.lifecycle.queues.unbound[uav_id].peek()
                if uav_id in self.lifecycle.queues.unbound
                else None
            )
            for uav_id in range(self.config.environment.uav_count)
        }
        execution = self.executor.execute(
            slot,
            joint,
            actor_features=self.channel_features,
        )
        physical = self.physical_service.execute(
            execution,
            self.channel.channel,
            settle_deadlines=False,
        )

        (
            routing_records,
            local_bindings,
            route_slot_workload_snapshots,
        ) = self._apply_slot_end_routes(
            joint,
            slot_start_heads,
        )
        if local_bindings:
            self.metrics.record_local_bindings(local_bindings)
        settled = self.lifecycle.settle_slot(slot)
        physical = replace(
            physical,
            settled_task_ids=tuple(task.task_id for task in settled),
        )
        self.metrics.record_physical_result(physical)
        self.metrics.record_settled_tasks(
            settled,
            slot_duration_s=self.config.environment.slot_duration_s,
        )

        actual_energy_j = math.fsum(debit.total_energy_j for debit in physical.energy_debits)
        reward_workload_snapshots = (
            route_slot_workload_snapshots
            if WorkloadTimingMode(self.config.environment.workload_timing_mode)
            is WorkloadTimingMode.ROUTE_SLOT_PRE_ROUTE
            else None
        )
        reward_terms = self.reward_calculator.calculate(
            slot=slot,
            all_tasks=self.lifecycle.tasks.values(),
            settled_tasks=settled,
            actual_energy_j=actual_energy_j,
            workload_snapshots=reward_workload_snapshots,
        )
        agent_reward_terms = None
        if (
            AgentCreditMode(self.config.training.mappo.agent_credit_mode)
            is AgentCreditMode.ROLE_DECOMPOSED
        ):
            agent_reward_terms = self.reward_calculator.decompose_agent_credit(
                slot=slot,
                uav_count=self.config.environment.uav_count,
                all_tasks=self.lifecycle.tasks.values(),
                settled_tasks=settled,
                team_terms=reward_terms,
                agent_transmit_energy_j={
                    debit.uav_id: debit.transmit_energy_j
                    for debit in physical.energy_debits
                },
                agent_cpu_energy_j={
                    debit.uav_id: debit.cpu_energy_j
                    for debit in physical.energy_debits
                },
                workload_snapshots=reward_workload_snapshots,
                credit_config=self.config.reward,
            )
        slot_outage = self._update_rate_and_outage_history(physical)
        self.channel_history.update_with_measurement(
            slot,
            physical.interference_measurement_w,
            physical.interference_measurement_mask,
        )

        refresh_mask = np.any(
            physical.interference_measurement_mask,
            axis=1,
        )
        self.public_messages = self.public_history.record_from_completed_slot(
            slot,
            refresh_mask,
            self.mobility,
            self.resource_states,
            self.lifecycle,
        )
        self.previous_actions = PreviousActionSnapshot.from_execution(
            actor_slot=slot + 1,
            result=execution,
            environment=self.config.environment,
            resource_states=self.resource_states,
        )
        arrival = self.traffic.generate(slot, self.lifecycle)
        arrival_snapshot = self._arrival_snapshot(arrival)
        self.metrics.record_generated(arrival.tasks)

        horizon = self.config.environment.episode_horizon
        if slot == horizon - 1:
            active_before = {task.task_id for task in self.lifecycle.active_tasks()}
            truncation_records = self.lifecycle.truncate_horizon(horizon)
            truncated_tasks = tuple(
                self.lifecycle.tasks[record.task_id]
                for record in truncation_records
            )
            unexpected_expired = tuple(
                task
                for task_id, task in sorted(self.lifecycle.tasks.items())
                if task_id in active_before and task.outcome is TaskOutcome.EXPIRED
            )
            if unexpected_expired:
                raise EnvironmentError(
                    "horizon truncation found a task whose deadline should have settled earlier"
                )
            self.metrics.record_truncated_tasks(truncated_tasks)
            self.slot = horizon
            self._done = True
            self.current_observations = None
            self.current_state = None
            self.channel_features = None
            conservation = self.assert_invariants()
            info = self._step_info(
                joint,
                execution,
                physical,
                routing_records,
                settled,
                reward_terms,
                agent_reward_terms,
                arrival_snapshot,
                slot_outage,
                conservation,
                truncated_tasks=truncated_tasks,
            )
            return StepResult(
                observations=None,
                centralized_state=None,
                reward=reward_terms.reward,
                terminated=False,
                truncated=True,
                info=info,
            )

        self._advance_decision_state(slot + 1)
        conservation = self.assert_invariants()
        info = self._step_info(
            joint,
            execution,
            physical,
            routing_records,
            settled,
            reward_terms,
            agent_reward_terms,
            arrival_snapshot,
            slot_outage,
            conservation,
            truncated_tasks=(),
        )
        return StepResult(
            observations=self.current_observations,
            centralized_state=self.current_state,
            reward=reward_terms.reward,
            terminated=False,
            truncated=False,
            info=info,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe audit snapshot without side effects."""

        if not self._has_reset:
            return {
                "initialized": False,
                "seed": self.config.seed,
                "config_hash": self.config.config_hash,
            }
        self._require_initialized()
        assert self.lifecycle is not None
        assert self.metrics is not None
        assert self.traffic is not None
        assert self.mobility is not None
        assert self.topology is not None
        assert self.channel is not None
        outage_rate, outage_mask = self._cumulative_outage()
        return {
            "initialized": True,
            "slot": self.slot,
            "done": self._done,
            "seed": self.config.seed,
            "config_hash": self.config.config_hash,
            "resources": self._resource_snapshot(),
            "tasks": [
                task.snapshot()
                for task in sorted(self.lifecycle.tasks.values(), key=lambda item: item.task_id)
            ],
            "physical_context": {
                "slot": self.channel.slot,
                "positions_m": self.mobility.positions_m.tolist(),
                "velocities_mps": self.mobility.velocities_mps.tolist(),
                "candidate_neighbors": self.topology.candidate_neighbors.tolist(),
                "distances_m": self.topology.distances_m.tolist(),
                "true_channel_real": self.channel.channel.real.tolist(),
                "true_channel_imag": self.channel.channel.imag.tolist(),
                "blocked_links": self.channel.blocked_links.tolist(),
            },
            "observations": (
                [observation.snapshot() for observation in self.current_observations]
                if self.current_observations is not None
                else None
            ),
            "centralized_state": (
                self.current_state.snapshot()
                if self.current_state is not None
                else None
            ),
            "history": {
                "traffic": [list(row) for row in self.traffic.history_snapshot()],
                "public_messages": self.public_messages.snapshot(),
                "previous_actions": self.previous_actions.snapshot(),
                "last_effective_rate_bps": self._last_effective_rate_bps.tolist(),
                "last_rate_valid_mask": self._last_rate_valid_mask.tolist(),
                "actual_attempt_counts": self._actual_attempt_counts.tolist(),
                "outage_counts": self._outage_counts.tolist(),
                "outage_rate": outage_rate.tolist(),
                "outage_valid_mask": outage_mask.tolist(),
            },
            "metrics": self.metrics.snapshot(),
        }

    def state_fingerprint(self) -> str:
        """Return a digest of the complete active environment object graph."""

        self._require_initialized()
        return _fingerprint_digest(self.__dict__)

    def clone_for_shadow(self) -> "U2UMECEnvironment":
        """Return an isolated copy for a counterfactual shadow rollout."""

        self._require_active()
        clone = copy.deepcopy(self)
        if clone.state_fingerprint() != self.state_fingerprint():
            raise EnvironmentError("shadow clone does not preserve the initial state")
        root_pairs = (
            ("lifecycle", self.lifecycle, clone.lifecycle),
            ("mobility_model", self.mobility_model, clone.mobility_model),
            ("mobility", self.mobility, clone.mobility),
            ("topology_model", self.topology_model, clone.topology_model),
            ("topology", self.topology, clone.topology),
            ("channel_model", self.channel_model, clone.channel_model),
            ("channel", self.channel, clone.channel),
            ("channel_history", self.channel_history, clone.channel_history),
            ("public_history", self.public_history, clone.public_history),
            ("public_messages", self.public_messages, clone.public_messages),
            ("previous_actions", self.previous_actions, clone.previous_actions),
            ("traffic", self.traffic, clone.traffic),
            ("executor", self.executor, clone.executor),
            ("physical_service", self.physical_service, clone.physical_service),
            ("reward_calculator", self.reward_calculator, clone.reward_calculator),
            ("metrics", self.metrics, clone.metrics),
            ("observation_builder", self.observation_builder, clone.observation_builder),
            ("state_builder", self.state_builder, clone.state_builder),
        )
        if any(original is copied for _, original, copied in root_pairs):
            raise EnvironmentError("shadow clone shares a mutable root object")
        if clone.resource_states is self.resource_states:
            raise EnvironmentError("shadow clone shares the resource mapping")
        if any(
            clone.resource_states[uav_id] is self.resource_states[uav_id]
            for uav_id in self.resource_states
        ):
            raise EnvironmentError("shadow clone shares a mutable resource state")
        if clone.executor is None or clone.physical_service is None:
            raise EnvironmentError("shadow clone lost its service graph")
        if clone.executor.lifecycle is not clone.lifecycle:
            raise EnvironmentError("shadow executor lifecycle reference is not local")
        if clone.physical_service.lifecycle is not clone.lifecycle:
            raise EnvironmentError("shadow service lifecycle reference is not local")
        return clone

    def randomness_snapshot(self) -> dict[str, Any]:
        """Return JSON-safe digests for all environment-owned RNG streams."""

        self._require_initialized()
        streams = self._environment_rngs()
        return {
            "streams": {
                name: {
                    "bit_generator": type(rng.bit_generator).__name__,
                    "state": _canonical_fingerprint_value(
                        copy.deepcopy(rng.bit_generator.state)
                    ),
                    "state_digest": _fingerprint_digest(
                        copy.deepcopy(rng.bit_generator.state)
                    ),
                }
                for name, rng in streams.items()
            },
            "unused_streams": ["interference_measurement"],
        }

    def rng_fingerprint(self) -> str:
        """Return a stable digest of the environment RNG snapshot."""

        return _fingerprint_digest(self.randomness_snapshot())

    def exogenous_fingerprint(self) -> str:
        """Return a digest of state expected to be action-independent."""

        self._require_initialized()
        assert self.mobility is not None
        assert self.mobility_model is not None
        assert self.topology is not None
        assert self.channel is not None
        assert self.channel_model is not None
        assert self.channel_history is not None
        assert self.traffic is not None
        payload = {
            "slot": self.slot,
            "mobility": self.mobility,
            "mobility_trajectory": getattr(self.mobility_model, "_trajectory", None),
            "topology": self.topology,
            "channel": self.channel,
            "channel_internal": {
                "last_slot": getattr(self.channel_model, "_last_slot", None),
                "previous_positions_m": getattr(
                    self.channel_model, "_previous_positions_m", None
                ),
                "shadowing_db": getattr(self.channel_model, "_shadowing_db", None),
            },
            "true_channel_history": getattr(self.channel_history, "_true_channels", None),
            "csi_features": {
                "slot": None if self.channel_features is None else self.channel_features.slot,
                "stale_csi": None if self.channel_features is None else self.channel_features.stale_csi,
                "csi_valid_mask": None if self.channel_features is None else self.channel_features.csi_valid_mask,
                "csi_aoi_slots": None if self.channel_features is None else self.channel_features.csi_aoi_slots,
            },
            "traffic_history": self.traffic.history_snapshot(),
            "rng": self.randomness_snapshot(),
        }
        return _fingerprint_digest(payload)

    def _environment_rngs(self) -> dict[str, np.random.Generator]:
        assert self.mobility_model is not None
        assert self.channel_model is not None
        assert self.channel_history is not None
        assert self.traffic is not None
        return {
            "reset_mobility": self.mobility_model.rng,
            "task_arrival": self.traffic.arrival_rng,
            "task_workload": self.traffic.workload_rng,
            "channel_fading": self.channel_model.rng,
            "csi_error": self.channel_history.csi_error_rng,
        }

    def assert_invariants(self) -> ConservationSnapshot:
        """Validate lifecycle, resources, histories, outcomes, and conservation."""

        self._require_initialized(allow_reset_in_progress=True)
        assert self.lifecycle is not None
        assert self.metrics is not None
        assert self.traffic is not None
        assert self.channel_history is not None
        assert self.public_history is not None
        assert self.public_messages is not None
        assert self.previous_actions is not None
        self.lifecycle.validate_invariants()

        count = self.config.environment.uav_count
        expected_ids = set(range(count))
        if set(self.resource_states) != expected_ids or set(self.initial_energy_j) != expected_ids:
            raise EnvironmentError("resource mappings must contain every UAV exactly once")
        residual_total = 0.0
        initial_total = 0.0
        for uav_id in range(count):
            resource = self.resource_states[uav_id]
            initial = self.initial_energy_j[uav_id]
            if resource.uav_id != uav_id:
                raise EnvironmentError("resource mapping key differs from resource.uav_id")
            values = (
                resource.max_transmit_power_w,
                resource.max_cpu_frequency_hz,
                resource.cpu_coefficient,
                resource.residual_energy_j,
                initial,
            )
            if not all(math.isfinite(value) for value in values):
                raise EnvironmentError("resource state contains a non-finite value")
            if min(values[:3]) <= 0.0 or resource.residual_energy_j < 0.0 or initial <= 0.0:
                raise EnvironmentError("resource capabilities/references must be positive")
            if resource.residual_energy_j > initial + self.config.environment.energy_tolerance_j:
                raise EnvironmentError("residual energy exceeds its episode initial value")
            residual_total += resource.residual_energy_j
            initial_total += initial

        energy_used = initial_total - residual_total
        metric_energy = self.metrics.total_tx_energy_j + self.metrics.total_cpu_energy_j
        energy_tolerance = (
            self.config.environment.energy_tolerance_j
            + 32.0 * math.ulp(max(1.0, initial_total, metric_energy))
        )
        if abs(energy_used - metric_energy) > energy_tolerance:
            raise EnvironmentError(
                "resource debit and episode energy metrics disagree: "
                f"energy_used={energy_used:.17g}, "
                f"metric_energy={metric_energy:.17g}, "
                f"diff={abs(energy_used - metric_energy):.17g}, "
                f"tolerance={energy_tolerance:.17g}"
            )

        tasks = tuple(self.lifecycle.tasks.values())
        if self.metrics.generated_task_count != len(tasks):
            raise EnvironmentError("generated-task metrics do not match lifecycle ownership")
        outcome_counts = {
            TaskOutcome.DONE: sum(task.outcome is TaskOutcome.DONE for task in tasks),
            TaskOutcome.EXPIRED: sum(task.outcome is TaskOutcome.EXPIRED for task in tasks),
            TaskOutcome.TRUNCATED: sum(task.outcome is TaskOutcome.TRUNCATED for task in tasks),
        }
        if outcome_counts[TaskOutcome.DONE] != self.metrics.completed_task_count:
            raise EnvironmentError("completed-task metrics do not match lifecycle outcomes")
        if outcome_counts[TaskOutcome.EXPIRED] != self.metrics.expired_task_count:
            raise EnvironmentError("expired-task metrics do not match lifecycle outcomes")
        if outcome_counts[TaskOutcome.TRUNCATED] != self.metrics.truncated_task_count:
            raise EnvironmentError("truncated-task metrics do not match lifecycle outcomes")

        if np.any(self._actual_attempt_counts < 0) or np.any(self._outage_counts < 0):
            raise EnvironmentError("outage counters must be non-negative")
        if np.any(self._outage_counts > self._actual_attempt_counts):
            raise EnvironmentError("outage counts cannot exceed actual-attempt counts")
        if int(np.sum(self._actual_attempt_counts)) != self.metrics.actual_attempt_count:
            raise EnvironmentError("actual-attempt history disagrees with episode metrics")
        if int(np.sum(self._outage_counts)) != self.metrics.outage_count:
            raise EnvironmentError("outage history disagrees with episode metrics")
        if not np.all(np.isfinite(self._last_effective_rate_bps)):
            raise EnvironmentError("last-slot effective rates must be finite")
        if np.any(self._last_effective_rate_bps < 0.0):
            raise EnvironmentError("last-slot effective rates must be non-negative")
        if np.any(self._last_effective_rate_bps[~self._last_rate_valid_mask] != 0.0):
            raise EnvironmentError("invalid last-slot rates must use the zero placeholder")

        horizon = self.config.environment.episode_horizon
        expected_completed_slots = horizon if self._done else self.slot
        if self.traffic.completed_slot_count != expected_completed_slots:
            raise EnvironmentError("traffic history is not aligned with the environment slot")
        expected_actor_slot = horizon if self._done else self.slot
        if self.public_history.last_completed_slot != expected_completed_slots - 1:
            raise EnvironmentError("public-message history is not aligned with completed slots")
        if self.public_messages.slot != expected_actor_slot:
            raise EnvironmentError("public-message snapshot is not aligned with the boundary")
        if self.previous_actions.actor_slot != expected_actor_slot:
            raise EnvironmentError("previous-action snapshot is not aligned with the boundary")
        if self.channel_history.interference.current_slot != expected_actor_slot:
            raise EnvironmentError("interference history is not aligned with the boundary")
        expected_physical_slots = horizon if self._done else self.slot
        if len(self.metrics.slot_records) != expected_physical_slots:
            raise EnvironmentError("physical metric records are not aligned with the slot")
        expected_backlog_records = horizon if self._done else self.slot + 1
        if len(self.metrics.queue_backlog_records) != expected_backlog_records:
            raise EnvironmentError("slot-start backlog history is not aligned with the slot")

        if self._done:
            if self.slot != horizon:
                raise EnvironmentError("truncated episode must stop exactly at the horizon")
            if self.current_observations is not None or self.current_state is not None:
                raise EnvironmentError("horizon must not expose an actor observation or state")
            if any(not task.is_terminal for task in tasks):
                raise EnvironmentError("every task must be terminal at the episode boundary")
        else:
            self._validate_current_decision_records()

        return self.metrics.assert_conservation()

    def _build_resource_states(self) -> tuple[dict[int, UavEnergyState], dict[int, float]]:
        env = self.config.environment
        states: dict[int, UavEnergyState] = {}
        initial: dict[int, float] = {}
        for uav_id in range(env.uav_count):
            profile = env.profile_assignment[uav_id]
            perturbation = env.profile_perturbations[uav_id]
            frequency_ratio, power_ratio, energy_ratio, coefficient_ratio = env.profile_ratios[profile]
            frequency_delta, power_delta, energy_delta, coefficient_delta = perturbation
            max_frequency = env.reference_cpu_frequency_hz * frequency_ratio * (1.0 + frequency_delta)
            max_power = env.reference_transmit_power_w * power_ratio * (1.0 + power_delta)
            initial_energy = env.reference_initial_energy_j * energy_ratio * (1.0 + energy_delta)
            coefficient = env.reference_cpu_coefficient * coefficient_ratio * (1.0 + coefficient_delta)
            state = UavEnergyState(
                uav_id=uav_id,
                max_transmit_power_w=max_power,
                max_cpu_frequency_hz=max_frequency,
                cpu_coefficient=coefficient,
                residual_energy_j=initial_energy,
            )
            states[uav_id] = state
            initial[uav_id] = initial_energy
        return states, initial

    def _build_decision_records(
        self,
        estimate: ArrivalEstimate,
    ) -> tuple[tuple[ActorObservation, ...], CentralizedState]:
        assert self.observation_builder is not None
        assert self.state_builder is not None
        assert self.mobility is not None
        assert self.topology is not None
        assert self.lifecycle is not None
        assert self.channel_features is not None
        assert self.channel is not None
        assert self.public_messages is not None
        assert self.previous_actions is not None
        outage_rate, outage_mask = self._cumulative_outage()
        observations = self.observation_builder.build_all(
            slot=self.slot,
            mobility=self.mobility,
            topology=self.topology,
            lifecycle=self.lifecycle,
            resource_states=self.resource_states,
            initial_energy_j=self.initial_energy_j,
            channel_features=self.channel_features,
            arrival_rate_estimates=estimate.values,
            public_messages=self.public_messages,
            previous_actions=self.previous_actions,
            history_source_slot=self.slot - 1,
            arrival_rate_valid_mask=estimate.valid_mask,
            last_effective_rate_bps=self._last_effective_rate_bps,
            last_rate_valid_mask=self._last_rate_valid_mask,
            outage_rate=outage_rate,
            outage_valid_mask=outage_mask,
        )
        state = self.state_builder.build(
            slot=self.slot,
            mobility=self.mobility,
            topology=self.topology,
            channel=self.channel,
            channel_features=self.channel_features,
            lifecycle=self.lifecycle,
            resource_states=self.resource_states,
        )
        return observations, state

    def _advance_decision_state(self, next_slot: int) -> None:
        assert self.mobility_model is not None
        assert self.mobility is not None
        assert self.topology_model is not None
        assert self.channel_model is not None
        assert self.channel_history is not None
        assert self.traffic is not None
        assert self.metrics is not None
        assert self.lifecycle is not None
        self.mobility = self.mobility_model.step(self.mobility)
        if self.mobility.slot != next_slot:
            raise EnvironmentError("mobility model returned an unexpected next slot")
        self.topology = self.topology_model.compute(next_slot, self.mobility.positions_m)
        self.channel = self.channel_model.generate(next_slot, self.mobility.positions_m)
        self.channel_history.record_physical_channel(self.channel)
        self.channel_features = self.channel_history.actor_features(next_slot)
        estimate = self.traffic.estimate(next_slot)
        self.slot = next_slot
        self.metrics.record_slot_start_backlog(next_slot, self.lifecycle)
        self.current_observations, self.current_state = self._build_decision_records(estimate)

    def _validate_proposals(
        self,
        proposals: Iterable[ActionProposal],
    ) -> tuple[ActionProposal, ...]:
        try:
            materialized = tuple(proposals)
        except TypeError as exc:
            raise EnvironmentError("proposals must be an iterable of ActionProposal") from exc
        count = self.config.environment.uav_count
        if len(materialized) != count:
            raise EnvironmentError(f"joint proposal must contain exactly {count} UAV actions")
        if any(not isinstance(proposal, ActionProposal) for proposal in materialized):
            raise TypeError("joint proposal must contain only ActionProposal instances")
        ids = [proposal.uav_id for proposal in materialized]
        if any(isinstance(uav_id, bool) or not isinstance(uav_id, int) for uav_id in ids):
            raise EnvironmentError("joint proposal UAV identifiers must be non-boolean integers")
        if set(ids) != set(range(count)) or len(ids) != len(set(ids)):
            raise EnvironmentError("joint proposal must contain each UAV identifier exactly once")
        ordered = tuple(sorted(materialized, key=lambda proposal: proposal.uav_id))
        assert self.current_observations is not None
        for observation, proposal in zip(self.current_observations, ordered):
            if observation.uav_id != proposal.uav_id:
                raise EnvironmentError("observation/proposal UAV ordering is inconsistent")
            if not observation.action_masks.is_legal(proposal):
                raise EnvironmentError(
                    f"proposal for UAV {proposal.uav_id} violates its slot-start sequential masks"
                )
        return ordered

    def _apply_slot_end_routes(
        self,
        proposals: tuple[ActionProposal, ...],
        slot_start_heads: dict[int, Task | None],
    ) -> tuple[
        list[dict[str, Any]],
        tuple[Task, ...],
        dict[int, TaskWorkloadSnapshot],
    ]:
        assert self.lifecycle is not None
        assert self.topology is not None
        records: list[dict[str, Any]] = []
        local_bindings: list[Task] = []
        workload_snapshots: dict[int, TaskWorkloadSnapshot] = {}
        for proposal in proposals:
            task = slot_start_heads[proposal.uav_id]
            if task is None:
                records.append(
                    {
                        "uav_id": proposal.uav_id,
                        "task_id": None,
                        "proposal": proposal.route,
                        "applied": False,
                        "reason": "no_slot_start_unbound_head",
                    }
                )
                continue
            before_bits = float(task.remaining_bits)
            workload_snapshot = None
            if proposal.route not in {"idle", "defer"}:
                workload_snapshot = TaskWorkloadSnapshot.from_task(task)
            self.lifecycle.route_task(
                task,
                slot=self.slot,
                destination=proposal.route,
                valid_destinations=self.topology.neighbors_of(proposal.uav_id),
            )
            if proposal.route == "local":
                local_bindings.append(task)
            if proposal.route not in {"idle", "defer"}:
                assert workload_snapshot is not None
                workload_snapshots[task.task_id] = workload_snapshot
            records.append(
                {
                    "uav_id": proposal.uav_id,
                    "task_id": task.task_id,
                    "proposal": proposal.route,
                    "applied": proposal.route not in {"idle", "defer"},
                    "reason": None if proposal.route not in {"idle", "defer"} else "deferred",
                    "remaining_bits_before": before_bits,
                    "remaining_bits_after": float(task.remaining_bits),
                    "destination": task.destination,
                    "status": task.status.value,
                }
            )
        return records, tuple(local_bindings), workload_snapshots

    def _update_rate_and_outage_history(
        self,
        physical: SlotPhysicalResult,
    ) -> dict[str, Any]:
        self._last_effective_rate_bps.fill(0.0)
        self._last_rate_valid_mask.fill(False)
        records: list[dict[str, Any]] = []
        attempted = 0
        outages = 0
        for link in physical.links:
            outage = link.outage
            if outage.attempted:
                sender = link.sender_uav
                receiver = link.receiver_uav
                assert outage.sample in {0, 1}
                self._last_effective_rate_bps[sender, receiver] = link.effective_rate_bps
                self._last_rate_valid_mask[sender, receiver] = True
                self._actual_attempt_counts[sender, receiver] += 1
                self._outage_counts[sender, receiver] += int(outage.sample)
                attempted += 1
                outages += int(outage.sample)
            records.append(
                {
                    "sender_uav": link.sender_uav,
                    "receiver_uav": link.receiver_uav,
                    "attempted": outage.attempted,
                    "sample": outage.sample,
                    "rb_outage_ratio": outage.rb_outage_ratio,
                    "na_reason": outage.na_reason,
                    "effective_rate_bps": link.effective_rate_bps,
                }
            )
        cumulative_rate, cumulative_mask = self._cumulative_outage()
        return {
            "actual_attempt_count": attempted,
            "outage_count": outages,
            "outage_ratio": outages / attempted if attempted else None,
            "links": records,
            "cumulative_actual_attempt_counts": self._actual_attempt_counts.tolist(),
            "cumulative_outage_counts": self._outage_counts.tolist(),
            "cumulative_outage_rate": cumulative_rate.tolist(),
            "cumulative_outage_valid_mask": cumulative_mask.tolist(),
        }

    def _cumulative_outage(self) -> tuple[np.ndarray, np.ndarray]:
        mask = self._actual_attempt_counts > 0
        values = np.zeros(self._actual_attempt_counts.shape, dtype=np.float64)
        np.divide(
            self._outage_counts,
            self._actual_attempt_counts,
            out=values,
            where=mask,
        )
        return values, mask

    def _step_info(
        self,
        proposals: tuple[ActionProposal, ...],
        execution: JointExecutionResult,
        physical: SlotPhysicalResult,
        routing_records: list[dict[str, Any]],
        settled: tuple[Task, ...],
        reward_terms: RewardTerms,
        agent_reward_terms: AgentRewardTerms | None,
        arrival_snapshot: dict[str, Any],
        outage_snapshot: dict[str, Any],
        conservation: ConservationSnapshot,
        *,
        truncated_tasks: tuple[Task, ...],
    ) -> dict[str, Any]:
        assert self.metrics is not None
        executed = [self._executed_action_snapshot(action) for action in execution.actions]
        rejection = [
            {
                "uav_id": action.proposal.uav_id,
                "reason": action.communication.rejection_reason,
            }
            for action in execution.actions
            if action.communication.rejection_reason is not None
        ]
        downgrade = [
            {
                "uav_id": action.proposal.uav_id,
                "communication": action.communication.downgrade_reason,
                "cpu": action.cpu.downgrade_reason,
            }
            for action in execution.actions
            if action.communication.downgrade_reason is not None
            or action.cpu.downgrade_reason is not None
        ]
        canonicalization = [
            {
                "uav_id": action.proposal.uav_id,
                "reason": action.communication.canonicalization_reason,
            }
            for action in execution.actions
            if action.communication.canonicalization_reason is not None
        ]
        info = {
            "slot": execution.slot,
            "run_id": self.config.run_id,
            "config_hash": self.config.config_hash,
            "boundary_slot": self.slot,
            "next_slot": None if self._done else self.slot,
            "next_decision_slot": None if self._done else self.slot,
            "bootstrap_allowed": not self._done,
            "terminated": False,
            "truncated": self._done,
            "proposal": [self._proposal_snapshot(proposal) for proposal in proposals],
            "executed": executed,
            "rejection": rejection,
            "downgrade": downgrade,
            "canonicalization": canonicalization,
            "service": {
                "links": [self._link_snapshot(link) for link in physical.links],
                "cpu": [self._cpu_service_snapshot(item) for item in physical.cpu_services],
                "routing": routing_records,
                "settled_tasks": [task.snapshot() for task in settled],
                "truncated_tasks": [task.snapshot() for task in truncated_tasks],
                "interference_measurement_w": physical.interference_measurement_w.tolist(),
                "interference_measurement_mask": physical.interference_measurement_mask.tolist(),
            },
            "energy": {
                "actual_total_j": reward_terms.actual_energy_j,
                "debits": [
                    {
                        "uav_id": debit.uav_id,
                        "transmit_energy_j": debit.transmit_energy_j,
                        "cpu_energy_j": debit.cpu_energy_j,
                        "total_energy_j": debit.total_energy_j,
                        "reserved_energy_j": debit.reserved_energy_j,
                        "residual_before_j": debit.residual_before_j,
                        "residual_after_j": debit.residual_after_j,
                    }
                    for debit in physical.energy_debits
                ],
            },
            "outage": outage_snapshot,
            "reward": reward_terms.to_dict(),
            "arrival": arrival_snapshot,
            "metrics": self.metrics.snapshot(),
            "conservation": conservation.to_dict(),
        }
        if agent_reward_terms is not None:
            agent_credit = agent_reward_terms.to_dict()
            info["agent_reward"] = list(agent_reward_terms.agent_reward)
            info["agent_credit"] = agent_credit
        return info

    @staticmethod
    def _proposal_snapshot(proposal: ActionProposal) -> dict[str, Any]:
        return {
            "uav_id": proposal.uav_id,
            "route": proposal.route,
            "tx_select": proposal.tx_select,
            "resource_group": proposal.resource_group,
            "resource_width": proposal.resource_width,
            "power_level": proposal.power_level,
            "cpu_queue": proposal.cpu_queue,
            "cpu_frequency": proposal.cpu_frequency,
        }

    @classmethod
    def _executed_action_snapshot(cls, action: Any) -> dict[str, Any]:
        communication = action.communication
        cpu = action.cpu
        reservation = action.energy_reservation
        return {
            "uav_id": action.proposal.uav_id,
            "communication": {
                "sender_uav": communication.sender_uav,
                "receiver_uav": communication.receiver_uav,
                "proposed_ru_indices": list(communication.proposed_ru_indices),
                "executed_ru_indices": list(communication.executed_ru_indices),
                "proposed_power_w": communication.proposed_power_w,
                "candidate_power_w": communication.candidate_power_w,
                "executed_power_w": communication.executed_power_w,
                "tentatively_accepted": communication.tentatively_accepted,
                "accepted": communication.accepted,
                "historical_quality": communication.historical_quality,
                "priority_key": (
                    list(communication.priority_key)
                    if communication.priority_key is not None
                    else None
                ),
                "head_task_id": communication.head_task_id,
                "slot_start_queue_bits": communication.slot_start_queue_bits,
            },
            "cpu": {
                "executor_uav": cpu.executor_uav,
                "queue_source_uav": cpu.queue_source_uav,
                "head_task_id": cpu.head_task_id,
                "proposed_frequency_hz": cpu.proposed_frequency_hz,
                "executed_frequency_hz": cpu.executed_frequency_hz,
                "head_remaining_cycles": cpu.head_remaining_cycles,
                "active": cpu.active,
            },
            "energy_reservation": {
                "transmit_energy_j": reservation.transmit_energy_j,
                "cpu_energy_j": reservation.cpu_energy_j,
                "total_energy_j": reservation.total_energy_j,
                "candidate_power_w": reservation.candidate_power_w,
                "candidate_cpu_frequency_hz": reservation.candidate_cpu_frequency_hz,
            },
        }

    @staticmethod
    def _link_snapshot(link: Any) -> dict[str, Any]:
        return {
            "sender_uav": link.sender_uav,
            "receiver_uav": link.receiver_uav,
            "ru_indices": list(link.ru_indices),
            "per_ru_power_w": link.per_ru_power_w,
            "interference_w": list(link.interference_w),
            "sinr_linear": list(link.sinr_linear),
            "effective_rate_bps": link.effective_rate_bps,
            "active_duration_s": link.active_duration_s,
            "service_bits": link.service_bits,
            "task_services": [
                {"task_id": service.task_id, "amount": service.amount}
                for service in link.task_services
            ],
            "transmit_energy_j": link.transmit_energy_j,
        }

    @staticmethod
    def _cpu_service_snapshot(service: Any) -> dict[str, Any]:
        return {
            "executor_uav": service.executor_uav,
            "queue_source_uav": service.queue_source_uav,
            "task_id": service.task_id,
            "executed_frequency_hz": service.executed_frequency_hz,
            "service_cycles": service.service_cycles,
            "active_duration_s": service.active_duration_s,
            "cpu_energy_j": service.cpu_energy_j,
        }

    @staticmethod
    def _arrival_snapshot(arrival: ArrivalBatch) -> dict[str, Any]:
        return {
            "slot": arrival.slot,
            "indicators": arrival.indicators.tolist(),
            "tasks": [task.snapshot() for task in arrival.tasks],
        }

    def _resource_snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "uav_id": uav_id,
                "paper_uav_id": uav_id + 1,
                "profile": self.config.environment.profile_assignment[uav_id],
                "perturbation": list(self.config.environment.profile_perturbations[uav_id]),
                "max_transmit_power_w": self.resource_states[uav_id].max_transmit_power_w,
                "max_cpu_frequency_hz": self.resource_states[uav_id].max_cpu_frequency_hz,
                "cpu_coefficient": self.resource_states[uav_id].cpu_coefficient,
                "initial_energy_j": self.initial_energy_j[uav_id],
                "residual_energy_j": self.resource_states[uav_id].residual_energy_j,
            }
            for uav_id in range(self.config.environment.uav_count)
        ]

    def _validate_current_decision_records(self) -> None:
        assert self.mobility is not None
        assert self.topology is not None
        assert self.channel is not None
        assert self.channel_features is not None
        assert self.public_messages is not None
        assert self.previous_actions is not None
        if self.public_messages.slot != self.slot:
            raise EnvironmentError("public messages are not aligned with the current slot")
        if self.previous_actions.actor_slot != self.slot:
            raise EnvironmentError("previous actions are not aligned with the current slot")
        if not (
            self.mobility.slot
            == self.topology.slot
            == self.channel.slot
            == self.channel_features.slot
            == self.slot
        ):
            raise EnvironmentError("decision-time mobility/topology/channel/history slots disagree")
        if self.current_observations is None or self.current_state is None:
            raise EnvironmentError("active episode must expose observations and centralized state")
        count = self.config.environment.uav_count
        if len(self.current_observations) != count:
            raise EnvironmentError("actor observation tuple does not match the UAV count")
        for uav_id, observation in enumerate(self.current_observations):
            if observation.uav_id != uav_id or observation.slot != self.slot:
                raise EnvironmentError("actor observations are not in stable current-slot order")
        if self.current_state.slot != self.slot:
            raise EnvironmentError("centralized state is not aligned with the current slot")

    def _require_active(self) -> None:
        self._require_initialized()
        if self._done:
            raise EnvironmentError("episode is already truncated; call reset before another step")

    def _require_initialized(self, *, allow_reset_in_progress: bool = False) -> None:
        if not self._has_reset and not allow_reset_in_progress:
            raise EnvironmentError("environment must be reset before use")
        components = (
            self.lifecycle,
            self.mobility_model,
            self.mobility,
            self.topology_model,
            self.topology,
            self.channel_model,
            self.channel,
            self.public_history,
            self.public_messages,
            self.previous_actions,
            self.channel_history,
            self.traffic,
            self.executor,
            self.physical_service,
            self.reward_calculator,
            self.metrics,
            self.observation_builder,
            self.state_builder,
        )
        if any(component is None for component in components):
            raise EnvironmentError("environment component graph is incomplete")

    @staticmethod
    def _readonly(values: np.ndarray, *, dtype: Any = np.float64) -> np.ndarray:
        result = np.array(values, dtype=dtype, copy=True)
        result.setflags(write=False)
        return result


FullEnvironment = U2UMECEnvironment


__all__ = [
    "EnvironmentError",
    "FullEnvironment",
    "ResetResult",
    "StepResult",
    "U2UMECEnvironment",
]
