"""Read-only route telemetry derived from existing evaluation decisions.

The collector observes the already-produced deterministic action-distribution
output and the environment's existing step info.  It never calls the actor or
the action distribution and therefore cannot alter action selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..config import RunConfig
from ..env.actions import ActionProposal
from ..models.ca_gat_mappo_actions import SequentialActionDistributionOutput


EVALUATION_ROUTE_TELEMETRY_SCHEMA_VERSION = "evaluation_route_telemetry.v1"


class EvaluationRouteTelemetryError(RuntimeError):
    """Raised when evaluation telemetry disagrees with the factual action path."""


@dataclass(frozen=True)
class EvaluationRouteTelemetryCollector:
    """Convert one existing deterministic actor output into JSON-ready records."""

    config: RunConfig

    def collect_step(
        self,
        *,
        evaluation_run_id: str,
        episode_id: int,
        evaluation_seed: int,
        observations: Sequence[Any],
        proposals: Sequence[ActionProposal],
        action_output: SequentialActionDistributionOutput,
        step_info: Mapping[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        """Capture route-active decisions without re-running any policy code."""

        if not observations or len(observations) != len(proposals):
            raise EvaluationRouteTelemetryError(
                "evaluation route telemetry requires aligned observations and proposals"
            )
        if action_output.mode != "deterministic":
            raise EvaluationRouteTelemetryError(
                "evaluation route telemetry requires the deterministic action output"
            )
        agent_count = len(observations)
        probabilities = self._agent_rows(
            action_output.probabilities["route"], agent_count, "route probabilities"
        )
        masks = self._agent_rows(
            action_output.action_masks["route"], agent_count, "route masks"
        )
        indices = self._agent_values(
            action_output.action_indices["route"], agent_count, "route indices"
        )
        active = self._agent_values(
            action_output.active_branches["route"], agent_count, "route activity"
        )

        executed_by_uav = {
            int(item["uav_id"]): item for item in step_info.get("executed", ())
        }
        debits_by_uav = {
            int(item["uav_id"]): item
            for item in step_info.get("energy", {}).get("debits", ())
        }
        outage_links = tuple(step_info.get("outage", {}).get("links", ()))

        records: list[dict[str, Any]] = []
        for agent_index, (observation, proposal) in enumerate(
            zip(observations, proposals)
        ):
            contract = observation.action_masks
            if bool(active[agent_index]) != bool(contract.route_branch_active):
                raise EvaluationRouteTelemetryError(
                    "action output route activity differs from the observation contract"
                )
            if not bool(contract.route_branch_active):
                continue
            if int(proposal.uav_id) != int(observation.uav_id):
                raise EvaluationRouteTelemetryError(
                    "proposal UAV differs from the observation UAV"
                )

            domain = tuple(contract.route_domain)
            route_mask = tuple(bool(value) for value in masks[agent_index])
            expected_mask = tuple(bool(value) for value in contract.route_mask)
            if route_mask != expected_mask or len(domain) != len(route_mask):
                raise EvaluationRouteTelemetryError(
                    "action output route mask differs from the observation contract"
                )
            route_probabilities = tuple(
                float(value) for value in probabilities[agent_index]
            )
            if len(route_probabilities) != len(domain):
                raise EvaluationRouteTelemetryError(
                    "route probabilities differ from the route action domain"
                )
            if any(not math.isfinite(value) or value < 0.0 for value in route_probabilities):
                raise EvaluationRouteTelemetryError("route probabilities are invalid")

            local_probability = self._probability_for(
                domain, route_probabilities, "local"
            )
            defer_probability = self._probability_for(
                domain, route_probabilities, "defer"
            )
            remote_entries = tuple(
                (index, int(value), route_mask[index], route_probabilities[index])
                for index, value in enumerate(domain)
                if isinstance(value, int) and not isinstance(value, bool)
            )
            remote_probability = math.fsum(
                probability for _index, _uav, _legal, probability in remote_entries
            )
            normalizer = remote_probability if remote_probability > 0.0 else None
            remote_destination_probabilities = {
                f"remote_{ordinal}": (
                    probability / normalizer if normalizer is not None else 0.0
                )
                for ordinal, (_index, _uav, legal, probability) in enumerate(
                    remote_entries, start=1
                )
                if legal
            }
            remote_destination_uav_ids = {
                f"remote_{ordinal}": uav_id
                for ordinal, (_index, uav_id, legal, _probability) in enumerate(
                    remote_entries, start=1
                )
                if legal
            }
            selected_top_level, selected_remote_id = self._selected_route(
                proposal.route
            )
            selected_index = int(indices[agent_index])
            if not 0 <= selected_index < len(domain):
                raise EvaluationRouteTelemetryError(
                    "selected route index is outside the route domain"
                )
            if domain[selected_index] != proposal.route:
                raise EvaluationRouteTelemetryError(
                    "selected route index differs from the factual proposal"
                )

            uav_id = int(proposal.uav_id)
            executed = executed_by_uav.get(uav_id)
            debit = debits_by_uav.get(uav_id)
            records.append(
                {
                    "schema_version": EVALUATION_ROUTE_TELEMETRY_SCHEMA_VERSION,
                    "evaluation_run_id": evaluation_run_id,
                    "decoder_mode": self.config.training.mappo.route_decoder_mode.value,
                    "episode_id": int(episode_id),
                    "evaluation_seed": int(evaluation_seed),
                    "step": int(observation.slot),
                    "uav_id": uav_id,
                    "local_probability": local_probability,
                    "remote_probability": remote_probability,
                    "defer_probability": defer_probability,
                    "remote_destination_probabilities": (
                        remote_destination_probabilities
                    ),
                    "remote_destination_uav_ids": remote_destination_uav_ids,
                    "selected_top_level": selected_top_level,
                    "selected_remote_id": selected_remote_id,
                    "final_route_action_index": selected_index,
                    "route_action_domain": [
                        self._route_label(value) for value in domain
                    ],
                    "route_action_mask": list(route_mask),
                    "legal_route_candidates": [
                        self._route_label(value)
                        for value, legal in zip(domain, route_mask)
                        if legal
                    ],
                    "tx_action": proposal.tx_select,
                    "resource_group": proposal.resource_group,
                    "power_action": float(proposal.power_level),
                    "executed_communication": (
                        None if executed is None else executed.get("communication")
                    ),
                    "tx_energy_j": (
                        None
                        if debit is None
                        else float(debit["transmit_energy_j"])
                    ),
                    "outage_links": [
                        item
                        for item in outage_links
                        if int(item.get("sender_uav", -1)) == uav_id
                    ],
                }
            )
            if self.config.training.mappo.route_decoder_mode == "unified_node_v1":
                from ..models.route_decision import unified_route_decision_record
                records[-1].update(unified_route_decision_record(
                    action_output, agent_index, int(observation.private_queues.unbound.head_task_id)))
                records[-1]["schema_version"] = "evaluation_route_telemetry.v2"
        return tuple(records)

    @staticmethod
    def _agent_rows(tensor: Any, agent_count: int, label: str) -> list[list[Any]]:
        value = tensor.detach().cpu()
        if value.ndim != 4 or tuple(value.shape[:3]) != (1, 1, agent_count):
            raise EvaluationRouteTelemetryError(
                f"{label} must have [1,1,A,D] axes"
            )
        return value[0, 0].tolist()

    @staticmethod
    def _agent_values(tensor: Any, agent_count: int, label: str) -> list[Any]:
        value = tensor.detach().cpu()
        if value.ndim != 3 or tuple(value.shape) != (1, 1, agent_count):
            raise EvaluationRouteTelemetryError(f"{label} must have [1,1,A] axes")
        return value[0, 0].tolist()

    @staticmethod
    def _probability_for(
        domain: tuple[Any, ...], probabilities: tuple[float, ...], target: str
    ) -> float:
        try:
            return probabilities[domain.index(target)]
        except ValueError as exc:
            raise EvaluationRouteTelemetryError(
                f"route domain omits required action {target!r}"
            ) from exc

    @staticmethod
    def _selected_route(route: Any) -> tuple[str, int | None]:
        if route == "local":
            return "Local", None
        if route == "defer":
            return "Defer", None
        if route == "idle":
            return "Idle", None
        if isinstance(route, int) and not isinstance(route, bool):
            return "Remote", int(route)
        raise EvaluationRouteTelemetryError(f"unknown factual route {route!r}")

    @staticmethod
    def _route_label(value: Any) -> str:
        if isinstance(value, int) and not isinstance(value, bool):
            return f"remote_uav_{value}"
        return str(value)


def build_evaluation_route_summary(
    *,
    evaluation_run_id: str,
    decoder_mode: str,
    records: Sequence[Mapping[str, Any]],
    actor_episodes: Sequence[Any],
) -> dict[str, Any]:
    """Build a compact summary from trace records and existing episode metrics."""

    counts = {name: 0 for name in ("Local", "Remote", "Defer", "Idle")}
    for record in records:
        selected = str(record["selected_top_level"])
        if selected not in counts:
            raise EvaluationRouteTelemetryError(
                f"unknown selected top-level action {selected!r}"
            )
        counts[selected] += 1
    decision_count = len(records)
    return {
        "schema_version": ("evaluation_route_telemetry.v2" if decoder_mode == "unified_node_v1"
                           else EVALUATION_ROUTE_TELEMETRY_SCHEMA_VERSION),
        **({"decision_mode": "flat_leaf_v1"} if decoder_mode == "unified_node_v1" else {}),
        "evaluation_run_id": evaluation_run_id,
        "decoder_mode": decoder_mode,
        "route_decision_count": decision_count,
        "local_count": counts["Local"],
        "remote_count": counts["Remote"],
        "defer_count": counts["Defer"],
        "idle_count": counts["Idle"],
        "remote_ratio": (
            counts["Remote"] / decision_count if decision_count else 0.0
        ),
        "tx_energy_total_j": math.fsum(
            float(item.metrics["tx_energy_j"]) for item in actor_episodes
        ),
        "outage_denominator": sum(
            int(item.metrics["outage_denominator"]) for item in actor_episodes
        ),
        "action_trace_sha256_by_episode": [
            {
                "episode_id": index,
                "evaluation_seed": int(item.evaluation_seed),
                "action_trace_sha256": item.action_trace_sha256,
            }
            for index, item in enumerate(actor_episodes)
        ],
    }


__all__ = [
    "EVALUATION_ROUTE_TELEMETRY_SCHEMA_VERSION",
    "EvaluationRouteTelemetryCollector",
    "EvaluationRouteTelemetryError",
    "build_evaluation_route_summary",
]
