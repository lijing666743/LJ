"""Deterministic local-computation-only baseline over actor-visible masks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..config import RunConfig
from ..env.actions import ActionProposal
from ..env.observation import ActorObservation


class LocalOnlyPolicyError(ValueError):
    """Raised when the frozen local-only rule is not mask-representable."""


class LocalOnlyPolicy:
    """Keep every task on its originating UAV and request no communication."""

    method_id = "local_only"
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
            raise LocalOnlyPolicyError(
                "policy_seed must be a non-negative integer"
            )
        self.policy_seed = resolved_seed
        self._sampling_order = tuple(config.action.sampling_order)
        self._canonical = dict(config.action.canonical_inactive_values)

    @property
    def seed_material(self) -> dict[str, int | None]:
        """Record interface-compatible metadata without owning an RNG."""

        return {
            "policy_seed": self.policy_seed,
            "policy_stream_id": self.policy_stream_id,
        }

    def act(self, observation: ActorObservation) -> ActionProposal:
        """Choose the frozen seven local-only branches in sequential order."""

        masks = observation.action_masks
        if tuple(masks.sampling_order) != self._sampling_order:
            raise LocalOnlyPolicyError(
                "observation sampling order differs from the frozen config"
            )

        context: dict[str, str | int | float] = {}

        route_legal = self._legal_values(masks, "route", context)
        route = self._require_value(
            "route",
            "local" if masks.route_branch_active else self._canonical["route"],
            route_legal,
        )
        context["route"] = route

        tx_legal = self._legal_values(masks, "tx_select", context)
        tx_select = self._require_value(
            "tx_select", self._canonical["tx_select"], tx_legal
        )
        context["tx_select"] = tx_select

        group_legal = self._legal_values(masks, "resource_group", context)
        resource_group = self._require_value(
            "resource_group",
            self._canonical["resource_group"],
            group_legal,
        )
        context["resource_group"] = resource_group

        width_legal = self._legal_values(masks, "resource_width", context)
        resource_width = self._require_value(
            "resource_width",
            self._canonical["resource_width"],
            width_legal,
        )
        context["resource_width"] = resource_width

        power_legal = self._legal_values(masks, "power_level", context)
        power_level = self._require_value(
            "power_level", self._canonical["power_level"], power_legal
        )
        context["power_level"] = power_level

        cpu_queue_legal = self._legal_values(masks, "cpu_queue", context)
        cpu_queue = (
            observation.uav_id
            if observation.uav_id in cpu_queue_legal
            else self._require_value(
                "cpu_queue", self._canonical["cpu_queue"], cpu_queue_legal
            )
        )
        context["cpu_queue"] = cpu_queue

        frequency_legal = self._legal_values(masks, "cpu_frequency", context)
        if cpu_queue == self._canonical["cpu_queue"]:
            cpu_frequency = self._require_value(
                "cpu_frequency",
                self._canonical["cpu_frequency"],
                frequency_legal,
            )
        else:
            positive = [
                float(value)
                for value in frequency_legal
                if self._is_number(value) and float(value) > 0.0
            ]
            cpu_frequency = (
                max(positive)
                if positive
                else self._require_value(
                    "cpu_frequency",
                    self._canonical["cpu_frequency"],
                    frequency_legal,
                )
            )

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
            raise LocalOnlyPolicyError(
                "frozen local-only rule produced an illegal proposal"
            )
        return proposal

    @staticmethod
    def _legal_values(
        masks: object,
        branch: str,
        context: Mapping[str, str | int | float],
    ) -> list[object]:
        domain = masks.domain_for(branch)
        mask = masks.mask_for(branch, context)
        if len(mask) != len(domain) or not any(bool(item) for item in mask):
            raise LocalOnlyPolicyError(
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
            raise LocalOnlyPolicyError(
                f"frozen local-only {branch} action {value!r} "
                "is not legal in the current mask"
            )
        return value

    @staticmethod
    def _is_number(value: object) -> bool:
        return not isinstance(value, bool) and isinstance(value, (int, float))


__all__ = ["LocalOnlyPolicy", "LocalOnlyPolicyError"]
