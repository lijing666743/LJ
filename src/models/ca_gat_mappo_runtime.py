"""Training-neutral factual actor execution for CA-GAT-MAPPO.

This module owns only observation tensorization, sequential action masks,
the persistent factual policy sampling stream, and recurrent actor output.
It deliberately has no critic, rollout, GAE, PPO, optimizer, or checkpoint
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from ..config import RunConfig
from ..env.observation import ActorObservation
from .ca_gat_mappo import ActorObservationTensorizer, ActorTensorBatch, CAGATMAPPOActor
from .ca_gat_mappo_actions import (
    CAGATMAPPOActionDistribution,
    SequentialActionDistributionOutput,
    SequentialActionMaskBatch,
)


class CAGATMAPPOFactualRuntimeError(RuntimeError):
    """Raised when one factual actor step violates the shared contract."""


@dataclass(frozen=True)
class CAGATMAPPOFactualActorStep:
    """All actor-side values produced for one real environment decision."""

    actor_batch: ActorTensorBatch
    action_masks: SequentialActionMaskBatch
    action_output: SequentialActionDistributionOutput


class CAGATMAPPOFactualActorRuntime:
    """Shared stochastic factual-policy path used by Trainer and diagnostics."""

    def __init__(
        self,
        actor: CAGATMAPPOActor,
        config: RunConfig,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(actor, CAGATMAPPOActor):
            raise TypeError("actor must be a CAGATMAPPOActor")
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        config.validate()
        self.actor = actor
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.actor_tensorizer = ActorObservationTensorizer(config)
        self.action_distribution = CAGATMAPPOActionDistribution(actor, config)
        self.policy_generator = torch.Generator(device=self.device.type)
        self.policy_generator.manual_seed(self.action_distribution.policy_seed)

    def sample_step(
        self,
        observations: Sequence[ActorObservation],
        hidden_in: Tensor,
        *,
        episode_start: bool,
    ) -> CAGATMAPPOFactualActorStep:
        """Run the production stochastic actor path without enabling gradients."""

        if not observations:
            raise CAGATMAPPOFactualRuntimeError(
                "factual actor step requires observations"
            )
        if hidden_in.requires_grad:
            raise CAGATMAPPOFactualRuntimeError(
                "factual actor hidden input must be detached"
            )
        with torch.no_grad():
            actor_batch = self.actor_tensorizer.encode_step(
                observations,
                device=self.device,
                dtype=self.dtype,
                episode_start=episode_start,
            )
            action_masks = SequentialActionMaskBatch.from_observations(
                observations
            )
            action_output = self.action_distribution.sample_actions(
                actor_batch,
                action_masks,
                hidden_in,
                generator=self.policy_generator,
            )
        if action_output.mode != "stochastic":
            raise CAGATMAPPOFactualRuntimeError(
                "factual actor did not use stochastic sampling"
            )
        return CAGATMAPPOFactualActorStep(
            actor_batch=actor_batch,
            action_masks=action_masks,
            action_output=action_output,
        )


__all__ = [
    "CAGATMAPPOFactualActorRuntime",
    "CAGATMAPPOFactualActorStep",
    "CAGATMAPPOFactualRuntimeError",
]
