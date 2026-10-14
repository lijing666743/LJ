"""Masked random baseline with policy-owned reproducible randomness."""

from __future__ import annotations

import numpy as np

from ..config import STREAM_IDS
from ..env.actions import ActionProposal
from ..env.observation import ActorObservation


class RandomPolicyError(ValueError):
    """Raised when the frozen masked-sampling contract cannot be satisfied."""


class RandomPolicy:
    """Sample every branch uniformly from its current sequential legal mask.

    The policy owns a NumPy generator derived from ``[policy_seed, 110]``. It
    never receives or consumes an environment generator and only reads the
    actor-visible ``ActionMasks`` attached to one ``ActorObservation``.
    """

    method_id = "random"

    def __init__(
        self,
        policy_seed: int,
        *,
        policy_stream_id: int = STREAM_IDS["torch_policy_sampling"],
    ) -> None:
        if isinstance(policy_seed, bool) or not isinstance(policy_seed, int) or policy_seed < 0:
            raise RandomPolicyError("policy_seed must be a non-negative integer")
        if (
            isinstance(policy_stream_id, bool)
            or not isinstance(policy_stream_id, int)
            or policy_stream_id < 0
        ):
            raise RandomPolicyError("policy_stream_id must be a non-negative integer")
        self.policy_seed = policy_seed
        self.policy_stream_id = policy_stream_id
        self._rng = np.random.default_rng(
            np.random.SeedSequence([policy_seed, policy_stream_id])
        )

    @property
    def seed_material(self) -> dict[str, int]:
        """Return the complete reproducibility material for artifact headers."""

        return {
            "policy_seed": self.policy_seed,
            "policy_stream_id": self.policy_stream_id,
        }

    def act(self, observation: ActorObservation) -> ActionProposal:
        """Sample one legal proposal in the frozen seven-branch order."""

        masks = observation.action_masks
        context: dict[str, str | int | float] = {}
        selected: dict[str, str | int | float] = {}
        for branch in masks.sampling_order:
            domain = masks.domain_for(branch)
            mask = masks.mask_for(branch, context)
            valid_indices = np.flatnonzero(mask)
            if valid_indices.size == 0:
                raise RandomPolicyError(f"branch {branch!r} has no legal masked action")
            domain_index = int(self._rng.choice(valid_indices))
            value = domain[domain_index]
            selected[branch] = value
            context[branch] = value

        proposal = ActionProposal(
            uav_id=observation.uav_id,
            route=selected["route"],
            tx_select=selected["tx_select"],
            resource_group=selected["resource_group"],
            resource_width=int(selected["resource_width"]),
            power_level=float(selected["power_level"]),
            cpu_queue=selected["cpu_queue"],
            cpu_frequency=float(selected["cpu_frequency"]),
        )
        if not masks.is_legal(proposal):
            raise RandomPolicyError("sequential masked sampling produced an illegal proposal")
        return proposal


__all__ = ["RandomPolicy", "RandomPolicyError"]
