"""Post-Validation-V1 V2 heuristic over actor-visible observations.

The historical Validation V1 baseline remains frozen in its original
artifacts. This repair keeps the seven-branch proposal boundary while adding
actor-visible coarse route feasibility and discrete-slot CPU deadline checks.
It has no access to centralized state, current channel truth, executor
metadata, or environment RNG state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..config import RunConfig
from ..env.actions import ActionProposal
from ..env.channel import receiver_noise_power_w
from ..env.observation import ActorObservation


class HeuristicPolicyError(ValueError):
    """Raised when an actor-visible heuristic input violates the frozen contract."""


@dataclass(frozen=True)
class HistoricalQualityAggregate:
    """Masked historical-quality result that keeps missing distinct from zero."""

    history_valid: bool
    mean: float | None


@dataclass(frozen=True)
class CpuFrequencyTelemetry:
    """Observable V2 deadline-feasibility result for one actor decision."""

    uav_id: int
    selected_queue: str | int
    selected_frequency_level: float
    selected_frequency_hz: float
    pending_task_count: int
    deadline_feasible: bool | None
    infeasible_fallback: bool


@dataclass(frozen=True)
class _CpuFrequencyDecision:
    level: float
    frequency_hz: float
    pending_task_count: int
    deadline_feasible: bool | None
    infeasible_fallback: bool


@dataclass(frozen=True)
class _CpuQueueDemand:
    source: int
    task_count: int
    total_remaining_cycles: float
    head_task_id: int
    head_remaining_cycles: float
    head_slack_slots: int


def masked_historical_quality(
    quality: Sequence[float] | np.ndarray,
    valid_mask: Sequence[bool] | np.ndarray,
    ru_indices: Iterable[int] | None = None,
) -> HistoricalQualityAggregate:
    """Return the arithmetic mean over valid candidate RUs only."""

    values = np.asarray(quality, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=np.bool_)
    if values.ndim != 1 or mask.shape != values.shape:
        raise HeuristicPolicyError(
            "historical quality and mask must be equal-length vectors"
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise HeuristicPolicyError(
            "historical quality must be finite and non-negative"
        )

    if ru_indices is None:
        indices = tuple(range(values.shape[0]))
    else:
        indices = tuple(ru_indices)
        if any(
            isinstance(index, bool)
            or not isinstance(index, (int, np.integer))
            or not 0 <= int(index) < values.shape[0]
            for index in indices
        ):
            raise HeuristicPolicyError(
                "candidate RU index is outside the historical-quality vector"
            )
        indices = tuple(int(index) for index in indices)

    valid_values = [
        float(values[index]) for index in indices if bool(mask[index])
    ]
    if not valid_values:
        return HistoricalQualityAggregate(history_valid=False, mean=None)
    return HistoricalQualityAggregate(
        history_valid=True,
        mean=math.fsum(valid_values) / len(valid_values),
    )


class HeuristicPolicy:
    """Post-Validation-V1 deterministic V2 proposal policy."""

    method_id = "heuristic"
    policy_stream_id = None

    def __init__(self, config: RunConfig, *, policy_seed: int | None = None) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        config.validate()
        resolved_seed = config.seed if policy_seed is None else policy_seed
        if (
            isinstance(resolved_seed, bool)
            or not isinstance(resolved_seed, int)
            or resolved_seed < 0
        ):
            raise HeuristicPolicyError(
                "policy_seed must be a non-negative integer"
            )
        self.policy_seed = resolved_seed
        self._slot_duration_s = float(config.environment.slot_duration_s)
        self._reference_cpu_frequency_hz = float(
            config.environment.reference_cpu_frequency_hz
        )
        self._ru_bandwidth_hz = float(
            config.environment.ru_bandwidth_hz
        )
        self._reference_rate_bps = float(
            config.environment.reference_rate_bps
        )
        self._cold_start_sinr_floor = float(
            config.environment.outage_threshold_linear
        )
        self._reference_power_per_ru_w = float(
            config.environment.reference_transmit_power_w
            / config.environment.ru_count
        )
        self._noise_power_per_ru_w = receiver_noise_power_w(
            config.environment
        )
        self._cold_start_width = max(
            int(value) for value in config.action.resource_width_options
        )
        self._sampling_order = tuple(config.action.sampling_order)
        self._canonical = dict(config.action.canonical_inactive_values)
        self._resource_groups = tuple(
            tuple(int(ru) - 1 for ru in group)
            for group in config.environment.resource_groups
        )
        self._cpu_frequency_telemetry: dict[int, CpuFrequencyTelemetry] = {}

    @property
    def seed_material(self) -> dict[str, int | None]:
        """Record interface-compatible seed metadata without owning an RNG."""

        return {
            "policy_seed": self.policy_seed,
            "policy_stream_id": self.policy_stream_id,
        }

    @property
    def cpu_frequency_telemetry(self) -> tuple[CpuFrequencyTelemetry, ...]:
        """Return immutable latest-per-UAV V2 feasibility telemetry."""

        return tuple(
            self._cpu_frequency_telemetry[uav_id]
            for uav_id in sorted(self._cpu_frequency_telemetry)
        )

    def act(self, observation: ActorObservation) -> ActionProposal:
        """Choose one legal proposal in the unchanged seven-branch order."""

        masks = observation.action_masks
        if tuple(masks.sampling_order) != self._sampling_order:
            raise HeuristicPolicyError(
                "observation sampling order differs from the frozen config"
            )

        context: dict[str, str | int | float] = {}
        route = self._select_route(observation, context)
        context["route"] = route
        tx_select = self._select_tx(observation, context)
        context["tx_select"] = tx_select
        resource_group = self._select_resource_group(observation, context)
        context["resource_group"] = resource_group
        resource_width = self._select_resource_width(observation, context)
        context["resource_width"] = resource_width
        power_level = self._select_power(observation, context)
        context["power_level"] = power_level
        cpu_queue = self._select_cpu_queue(observation, context)
        context["cpu_queue"] = cpu_queue
        cpu_decision = self._select_cpu_frequency(observation, context)
        cpu_frequency = cpu_decision.level

        proposal = ActionProposal(
            uav_id=observation.uav_id,
            route=route,
            tx_select=tx_select,
            resource_group=resource_group,
            resource_width=int(resource_width),
            power_level=float(power_level),
            cpu_queue=cpu_queue,
            cpu_frequency=float(cpu_frequency),
        )
        if not masks.is_legal(proposal):
            raise HeuristicPolicyError(
                "V2 heuristic produced an illegal proposal"
            )
        self._cpu_frequency_telemetry[observation.uav_id] = (
            CpuFrequencyTelemetry(
                uav_id=observation.uav_id,
                selected_queue=cpu_queue,
                selected_frequency_level=cpu_decision.level,
                selected_frequency_hz=cpu_decision.frequency_hz,
                pending_task_count=cpu_decision.pending_task_count,
                deadline_feasible=cpu_decision.deadline_feasible,
                infeasible_fallback=cpu_decision.infeasible_fallback,
            )
        )
        return proposal

    def _select_route(
        self,
        observation: ActorObservation,
        context: dict[str, str | int | float],
    ) -> str | int:
        legal = self._legal_values(observation, "route", context)
        head = observation.private_queues.unbound
        if not bool(head.head_valid_mask):
            return self._require_value(
                "route", self._canonical["route"], legal
            )
        slack = int(head.head_slack_slots)
        if slack <= 1:
            return self._require_value("route", "defer", legal)
        local_legal = "local" in legal
        local_total_cycles = (
            float(observation.private_queues.local_cpu.remaining_cycles)
            + float(head.head_remaining_cycles)
        )
        local_capacity_cycles = (
            float(observation.self_resources.max_cpu_frequency_hz)
            * self._slot_duration_s
            * (slack - 1)
        )
        if local_legal and local_total_cycles <= local_capacity_cycles:
            return "local"
        remote_candidates = self._integer_actions(
            legal, exclude=observation.uav_id
        )
        feasible_remote_candidates = [
            destination
            for destination in remote_candidates
            if self._remote_route_is_coarsely_feasible(
                observation, destination
            )
        ]
        if feasible_remote_candidates:
            return min(
                feasible_remote_candidates,
                key=lambda destination: self._historical_key(
                    self._destination_quality(observation, destination),
                    destination,
                ),
            )
        if local_legal:
            return "local"
        return self._require_value("route", "defer", legal)

    def _remote_route_is_coarsely_feasible(
        self,
        observation: ActorObservation,
        destination: int,
    ) -> bool:
        """Exclude clearly infeasible destinations using actor-visible proxies."""

        head = observation.private_queues.unbound
        public = observation.neighbor_public
        if bool(public.valid_mask[destination]):
            maximum_hz = (
                float(public.max_cpu_frequency_ratio[destination])
                * self._reference_cpu_frequency_hz
            )
            queued_cycles = float(
                public.cpu_load_remaining_cycles[destination]
            )
        else:
            source_maximum_hz = float(
                observation.self_resources.max_cpu_frequency_hz
            )
            tolerance = math.ulp(
                max(1.0, self._reference_cpu_frequency_hz)
            )
            if (
                source_maximum_hz
                > self._reference_cpu_frequency_hz + tolerance
            ):
                return False
            maximum_hz = source_maximum_hz
            queued_cycles = 0.0
        expected_rate_bps = self._estimated_route_rate_bps(
            observation,
            destination,
        )
        remaining_bits = float(head.head_remaining_bits)
        remaining_cycles = float(head.head_remaining_cycles)
        values = (
            maximum_hz,
            queued_cycles,
            expected_rate_bps,
            remaining_bits,
            remaining_cycles,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise HeuristicPolicyError(
                "route-feasibility inputs must be finite and non-negative"
            )
        if maximum_hz <= 0.0 or expected_rate_bps <= 0.0:
            return False

        transmission_slots = self._required_service_slots(
            remaining_bits,
            expected_rate_bps,
        )
        available_cpu_slots = (
            int(head.head_slack_slots) - 1 - transmission_slots
        )
        if available_cpu_slots <= 0:
            return False
        capacity_cycles = (
            maximum_hz
            * self._slot_duration_s
            * available_cpu_slots
        )
        required_cycles = queued_cycles + remaining_cycles
        tolerance = math.ulp(max(1.0, capacity_cycles))
        return required_cycles <= capacity_cycles + tolerance

    def _estimated_route_rate_bps(
        self,
        observation: ActorObservation,
        destination: int,
    ) -> float:
        """Estimate a causal link rate without requiring a prior TX."""

        edge = observation.edge_history
        last_rates = np.asarray(
            edge.last_effective_rate_bps,
            dtype=np.float64,
        )
        last_valid = np.asarray(
            edge.last_rate_valid_mask,
            dtype=np.bool_,
        )
        if (
            last_rates.ndim != 1
            or last_valid.shape != last_rates.shape
            or not 0 <= destination < last_rates.shape[0]
        ):
            raise HeuristicPolicyError(
                "last-rate values and masks must share the UAV index"
            )
        rate = float(last_rates[destination])
        if not math.isfinite(rate) or rate < 0.0:
            raise HeuristicPolicyError(
                "last effective rates must be finite and non-negative"
            )
        if bool(last_valid[destination]):
            return rate

        quality = np.asarray(
            edge.historical_quality[destination],
            dtype=np.float64,
        )
        quality_valid = np.asarray(
            edge.quality_valid_mask[destination],
            dtype=np.bool_,
        )
        if quality.ndim != 1 or quality_valid.shape != quality.shape:
            raise HeuristicPolicyError(
                "historical quality and mask must share the RU index"
            )
        if not np.all(np.isfinite(quality)) or np.any(quality < 0.0):
            raise HeuristicPolicyError(
                "historical quality must be finite and non-negative"
            )
        valid_quality = quality[quality_valid]
        if valid_quality.size:
            sinr_proxy = float(np.min(valid_quality))
            usable_width = min(
                self._cold_start_width,
                int(valid_quality.size),
            )
        else:
            csi_valid = np.asarray(
                edge.csi_valid_mask,
                dtype=np.bool_,
            )
            stale_csi = np.asarray(edge.stale_csi)
            if (
                csi_valid.shape != last_rates.shape
                or stale_csi.ndim != 2
                or stale_csi.shape[0] != last_rates.shape[0]
                or not np.iscomplexobj(stale_csi)
                or not np.all(np.isfinite(stale_csi))
            ):
                raise HeuristicPolicyError(
                    "stale CSI values and masks must share the edge index"
                )
            if bool(csi_valid[destination]):
                noise_only_quality = (
                    self._reference_power_per_ru_w
                    * np.abs(stale_csi[destination]) ** 2
                    / self._noise_power_per_ru_w
                )
                sinr_proxy = float(np.min(noise_only_quality))
            else:
                sinr_proxy = self._cold_start_sinr_floor
            usable_width = self._cold_start_width
        estimated = (
            usable_width
            * self._ru_bandwidth_hz
            * math.log2(1.0 + sinr_proxy)
        )
        return min(estimated, self._reference_rate_bps)

    def _select_tx(
        self,
        observation: ActorObservation,
        context: dict[str, str | int | float],
    ) -> str | int:
        legal = self._legal_values(observation, "tx_select", context)
        candidates = self._integer_actions(
            legal, exclude=observation.uav_id
        )
        if not candidates:
            return self._require_value(
                "tx_select", self._canonical["tx_select"], legal
            )
        queues = observation.private_queues.tx_by_destination

        def key(destination: int) -> tuple[float, int, float, int]:
            if not bool(queues.head_valid_mask[destination]):
                raise HeuristicPolicyError(
                    "legal TX candidate lacks an actor-visible queue head"
                )
            aggregate = self._destination_quality(
                observation, destination
            )
            invalid_flag, negative_mean, identifier = (
                self._historical_key(aggregate, destination)
            )
            return (
                float(queues.head_slack_slots[destination]),
                invalid_flag,
                negative_mean,
                identifier,
            )

        return min(candidates, key=key)

    def _select_resource_group(
        self,
        observation: ActorObservation,
        context: dict[str, str | int | float],
    ) -> str | int:
        legal = self._legal_values(
            observation, "resource_group", context
        )
        if context["tx_select"] == self._canonical["tx_select"]:
            return self._require_value(
                "resource_group",
                self._canonical["resource_group"],
                legal,
            )
        destination = context["tx_select"]
        if isinstance(destination, bool) or not isinstance(destination, int):
            raise HeuristicPolicyError(
                "active communication requires an integer destination"
            )
        groups = self._integer_actions(legal)
        if not groups:
            raise HeuristicPolicyError(
                "active communication has no legal resource group"
            )

        def key(group: int) -> tuple[int, float, int]:
            if not 1 <= group <= len(self._resource_groups):
                raise HeuristicPolicyError(
                    "legal resource group is outside the frozen config"
                )
            aggregate = self._destination_quality(
                observation,
                destination,
                self._resource_groups[group - 1],
            )
            return self._historical_key(aggregate, group)

        return min(groups, key=key)

    def _select_resource_width(
        self,
        observation: ActorObservation,
        context: dict[str, str | int | float],
    ) -> int:
        legal = self._legal_values(
            observation, "resource_width", context
        )
        widths = self._integer_actions(legal)
        if not widths:
            raise HeuristicPolicyError(
                "resource width branch has no legal integer action"
            )
        if context["tx_select"] == self._canonical["tx_select"]:
            return int(
                self._require_value(
                    "resource_width",
                    self._canonical["resource_width"],
                    legal,
                )
            )
        return max(widths)

    def _select_power(
        self,
        observation: ActorObservation,
        context: dict[str, str | int | float],
    ) -> float:
        legal = self._legal_values(
            observation, "power_level", context
        )
        if context["tx_select"] == self._canonical["tx_select"]:
            return float(
                self._require_value(
                    "power_level",
                    self._canonical["power_level"],
                    legal,
                )
            )
        positive = [
            float(value)
            for value in legal
            if self._is_number(value) and float(value) > 0.0
        ]
        if positive:
            return max(positive)
        return float(
            self._require_value(
                "power_level", self._canonical["power_level"], legal
            )
        )

    def _select_cpu_queue(
        self,
        observation: ActorObservation,
        context: dict[str, str | int | float],
    ) -> str | int:
        legal = self._legal_values(observation, "cpu_queue", context)
        candidates = self._integer_actions(legal)
        if not candidates:
            return self._require_value(
                "cpu_queue", self._canonical["cpu_queue"], legal
            )
        queues = observation.private_queues.cpu_by_source

        def key(source: int) -> tuple[int, int, int]:
            if not bool(queues.head_valid_mask[source]):
                raise HeuristicPolicyError(
                    "legal CPU candidate lacks an actor-visible queue head"
                )
            return (
                int(queues.head_slack_slots[source]),
                int(queues.head_task_id[source]),
                source,
            )

        return min(candidates, key=key)

    def _select_cpu_frequency(
        self,
        observation: ActorObservation,
        context: dict[str, str | int | float],
    ) -> _CpuFrequencyDecision:
        legal = self._legal_values(
            observation, "cpu_frequency", context
        )
        selected_queue = context["cpu_queue"]
        if selected_queue == self._canonical["cpu_queue"]:
            level = float(
                self._require_value(
                    "cpu_frequency",
                    self._canonical["cpu_frequency"],
                    legal,
                )
            )
            return _CpuFrequencyDecision(
                level=level,
                frequency_hz=0.0,
                pending_task_count=0,
                deadline_feasible=None,
                infeasible_fallback=False,
            )
        source = (
            observation.uav_id
            if selected_queue == "local"
            else selected_queue
        )
        if isinstance(source, bool) or not isinstance(source, int):
            raise HeuristicPolicyError(
                "active CPU queue must identify an integer source UAV"
            )
        queues = observation.private_queues.cpu_by_source
        if not bool(queues.head_valid_mask[source]):
            raise HeuristicPolicyError(
                "selected CPU queue lacks an actor-visible head"
            )
        demands = self._visible_cpu_demands(observation)
        if all(demand.source != source for demand in demands):
            raise HeuristicPolicyError(
                "selected CPU head is absent from visible queue aggregates"
            )
        maximum_hz = float(
            observation.self_resources.max_cpu_frequency_hz
        )
        positive = sorted(
            (
                float(value) * maximum_hz,
                float(value),
                index,
            )
            for index, value in enumerate(legal)
            if self._is_number(value) and float(value) > 0.0
        )
        if not positive:
            level = float(
                self._require_value(
                    "cpu_frequency",
                    self._canonical["cpu_frequency"],
                    legal,
                )
            )
            return _CpuFrequencyDecision(
                level=level,
                frequency_hz=0.0,
                pending_task_count=sum(
                    demand.task_count for demand in demands
                ),
                deadline_feasible=False,
                infeasible_fallback=True,
            )
        for frequency_hz, level, _ in positive:
            if self._frequency_meets_deadline_prefixes(
                demands,
                frequency_hz,
            ):
                return _CpuFrequencyDecision(
                    level=level,
                    frequency_hz=frequency_hz,
                    pending_task_count=sum(
                        demand.task_count for demand in demands
                    ),
                    deadline_feasible=True,
                    infeasible_fallback=False,
                )
        frequency_hz, level, _ = max(
            positive, key=lambda item: (item[0], -item[2])
        )
        return _CpuFrequencyDecision(
            level=level,
            frequency_hz=frequency_hz,
            pending_task_count=sum(
                demand.task_count for demand in demands
            ),
            deadline_feasible=False,
            infeasible_fallback=True,
        )

    def _frequency_meets_deadline_prefixes(
        self,
        demands: Sequence[_CpuQueueDemand],
        frequency_hz: float,
    ) -> bool:
        """Check conservative head-deadline prefixes in discrete slots."""

        if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
            raise HeuristicPolicyError(
                "candidate CPU frequency must be finite and positive"
            )
        cumulative_slots = 0
        previous_key: tuple[int, int, int] | None = None
        for demand in demands:
            key = (
                demand.head_slack_slots,
                demand.head_task_id,
                demand.source,
            )
            if previous_key is not None and key < previous_key:
                raise HeuristicPolicyError(
                    "CPU queue demands are not in deterministic EDF order"
                )
            previous_key = key
            cumulative_slots += self._queue_service_slot_upper_bound(
                demand,
                frequency_hz,
            )
            if cumulative_slots > demand.head_slack_slots:
                return False
        return True

    def _visible_cpu_demands(
        self,
        observation: ActorObservation,
    ) -> tuple[_CpuQueueDemand, ...]:
        """Build deterministic demand records from actor-tensor queue fields."""

        queues = observation.private_queues.cpu_by_source
        demands: list[_CpuQueueDemand] = []
        for source, head_valid in enumerate(queues.head_valid_mask):
            if not bool(head_valid):
                continue
            task_count = int(queues.task_count[source])
            total_cycles = float(queues.remaining_cycles[source])
            head_cycles = float(queues.head_remaining_cycles[source])
            head_slack = int(queues.head_slack_slots[source])
            head_task_id = int(queues.head_task_id[source])
            if (
                task_count <= 0
                or head_slack <= 0
                or head_task_id < 0
                or not math.isfinite(total_cycles)
                or not math.isfinite(head_cycles)
                or total_cycles < 0.0
                or head_cycles < 0.0
            ):
                raise HeuristicPolicyError(
                    "visible CPU queue aggregates are internally inconsistent"
                )
            tolerance = math.ulp(max(1.0, total_cycles))
            if head_cycles > total_cycles + tolerance:
                raise HeuristicPolicyError(
                    "CPU queue head cycles exceed aggregate remaining cycles"
                )
            demands.append(
                _CpuQueueDemand(
                    source=source,
                    task_count=task_count,
                    total_remaining_cycles=total_cycles,
                    head_task_id=head_task_id,
                    head_remaining_cycles=head_cycles,
                    head_slack_slots=head_slack,
                )
            )
        return tuple(
            sorted(
                demands,
                key=lambda item: (
                    item.head_slack_slots,
                    item.head_task_id,
                    item.source,
                ),
            )
        )

    def _queue_service_slot_upper_bound(
        self,
        demand: _CpuQueueDemand,
        frequency_hz: float,
    ) -> int:
        """Upper-bound per-task ceil demand when only a queue aggregate is visible."""

        if demand.task_count == 1:
            return self._required_service_slots(
                demand.total_remaining_cycles,
                frequency_hz,
            )
        head_slots = self._required_service_slots(
            demand.head_remaining_cycles,
            frequency_hz,
        )
        tail_cycles = max(
            0.0,
            demand.total_remaining_cycles - demand.head_remaining_cycles,
        )
        tail_slots_from_cycles = self._required_service_slots(
            tail_cycles,
            frequency_hz,
        )
        tail_task_count = demand.task_count - 1
        tail_slots = max(
            tail_task_count,
            tail_slots_from_cycles + tail_task_count - 1,
        )
        return head_slots + tail_slots

    def _required_service_slots(
        self,
        remaining_work: float,
        service_rate_per_s: float,
    ) -> int:
        """Return ceil(work / per-slot service) with boundary-safe rounding."""

        if (
            not math.isfinite(remaining_work)
            or remaining_work < 0.0
            or not math.isfinite(service_rate_per_s)
            or service_rate_per_s <= 0.0
        ):
            raise HeuristicPolicyError(
                "service-slot inputs must be finite with a positive rate"
            )
        if remaining_work == 0.0:
            return 0
        ratio = remaining_work / (
            service_rate_per_s * self._slot_duration_s
        )
        nearest_integer = round(ratio)
        if math.isclose(
            ratio,
            nearest_integer,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            ratio = float(nearest_integer)
        return int(math.ceil(ratio))

    @staticmethod
    def _historical_key(
        aggregate: HistoricalQualityAggregate,
        identifier: int,
    ) -> tuple[int, float, int]:
        if aggregate.history_valid:
            assert aggregate.mean is not None
            return (0, -float(aggregate.mean), identifier)
        return (1, 0.0, identifier)

    @staticmethod
    def _integer_actions(
        values: Sequence[object],
        *,
        exclude: int | None = None,
    ) -> list[int]:
        return [
            int(value)
            for value in values
            if not isinstance(value, bool)
            and isinstance(value, (int, np.integer))
            and (exclude is None or int(value) != exclude)
        ]

    @staticmethod
    def _is_number(value: object) -> bool:
        return not isinstance(value, bool) and isinstance(
            value, (int, float, np.number)
        )

    def _destination_quality(
        self,
        observation: ActorObservation,
        destination: int,
        ru_indices: Iterable[int] | None = None,
    ) -> HistoricalQualityAggregate:
        quality = observation.edge_history.historical_quality
        valid = observation.edge_history.quality_valid_mask
        if (
            isinstance(destination, bool)
            or not isinstance(destination, int)
            or not 0 <= destination < quality.shape[0]
        ):
            raise HeuristicPolicyError(
                "destination is outside the actor edge-history tensor"
            )
        return masked_historical_quality(
            quality[destination],
            valid[destination],
            ru_indices,
        )

    @staticmethod
    def _legal_values(
        observation: ActorObservation,
        branch: str,
        context: dict[str, str | int | float],
    ) -> list[object]:
        masks = observation.action_masks
        domain = masks.domain_for(branch)
        mask = np.asarray(
            masks.mask_for(branch, context), dtype=np.bool_
        )
        if mask.shape != (len(domain),) or not np.any(mask):
            raise HeuristicPolicyError(
                f"branch {branch!r} has no legal masked action"
            )
        return [
            value
            for value, allowed in zip(domain, mask)
            if bool(allowed)
        ]

    @staticmethod
    def _require_value(
        branch: str,
        value: object,
        legal: Sequence[object],
    ):
        if value not in legal:
            raise HeuristicPolicyError(
                f"frozen {branch} fallback {value!r} "
                "is not legal in the current mask"
            )
        return value


__all__ = [
    "CpuFrequencyTelemetry",
    "HeuristicPolicy",
    "HeuristicPolicyError",
    "HistoricalQualityAggregate",
    "masked_historical_quality",
]
