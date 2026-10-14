"""Minimal actor-safe policy interface for environment rollouts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..env.actions import ActionProposal
from ..env.observation import ActorObservation


@runtime_checkable
class Policy(Protocol):
    """A policy that maps one actor-visible observation to one proposal."""

    method_id: str
    policy_seed: int | None
    policy_stream_id: int | None

    def act(self, observation: ActorObservation) -> ActionProposal:
        """Return exactly seven proposal branches for one UAV."""


__all__ = ["Policy"]
