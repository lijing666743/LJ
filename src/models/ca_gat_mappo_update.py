"""Deterministic recurrent PPO parameter updates for CA-GAT-MAPPO.

This module is deliberately limited to adapting one already-finalized full
rollout into the frozen 8-by-32 recurrent minibatch and applying four PPO
epochs.  It does not collect rollouts, run a trainer, save checkpoints, expose
a CLI, or start an experiment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.optim import Adam

from ..config import ActorRatioMode, AgentCreditMode, RouteCreditMode, RunConfig
from ..env.actions import ActionProposal
from .ca_gat_mappo import (
    ACTION_BRANCH_ORDER,
    ActorTensorBatch,
    CAGATMAPPOActor,
    CentralizedStateTensorBatch,
    MAPPOCentralizedCritic,
    MAPPOTensorSpec,
)
from .ca_gat_mappo_actions import (
    CAGATMAPPOActionDistribution,
    SequentialActionDistributionOutput,
    SequentialActionMaskBatch,
)
from .ca_gat_mappo_gae import (
    compute_rollout_gae,
    compute_route_specific_advantage,
)
from .ca_gat_mappo_ppo import (
    CAGATMAPPOLossOutput,
    compute_configured_ppo_objective_and_loss,
)
from .ca_gat_mappo_route_telemetry import (
    RouteTelemetry,
    collect_route_telemetry,
    measure_route_head_gradient_rows,
)
from .ca_gat_mappo_route_nstep import (
    RouteNstepSample,
    compute_rollout_capped_route_event_nstep,
)
from .ca_gat_mappo_rollout import CAGATMAPPORolloutBuffer


class RecurrentPPOUpdateError(ValueError):
    """Raised when the frozen recurrent PPO update contract is violated."""


_FROZEN_GEOMETRY = (256, 32, 8, 4)


@dataclass(frozen=True)
class RouteChoiceStabilityLossOutput:
    """Differentiable Candidate-A loss plus detached scalar summaries."""

    unscaled_loss: Tensor
    scaled_loss: Tensor
    conditional_remote_mass: float | None
    remote_local_binary_entropy: float | None
    floor_violation_fraction: float | None
    valid_sample_count: int
    local_sample_count: int
    remote_sample_count: int

    def __post_init__(self) -> None:
        for name, value in (
            ("unscaled_loss", self.unscaled_loss),
            ("scaled_loss", self.scaled_loss),
        ):
            if not isinstance(value, Tensor) or value.ndim != 0:
                raise RecurrentPPOUpdateError(f"{name} must be a scalar tensor")
            if not torch.isfinite(value.detach()):
                raise RecurrentPPOUpdateError(f"{name} must be finite")
            if float(value.detach().cpu().item()) < -1.0e-12:
                raise RecurrentPPOUpdateError(f"{name} must be non-negative")
        for name, value in (
            ("valid_sample_count", self.valid_sample_count),
            ("local_sample_count", self.local_sample_count),
            ("remote_sample_count", self.remote_sample_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RecurrentPPOUpdateError(f"{name} must be a non-negative integer")
        summaries = (
            self.conditional_remote_mass,
            self.remote_local_binary_entropy,
            self.floor_violation_fraction,
        )
        if self.valid_sample_count == 0:
            if any(value is not None for value in summaries):
                raise RecurrentPPOUpdateError(
                    "stability summaries must be NA when no eligible sample exists"
                )
        elif any(value is None for value in summaries):
            raise RecurrentPPOUpdateError(
                "stability summaries must be present for eligible samples"
            )
        for name, value, upper in (
            ("conditional_remote_mass", self.conditional_remote_mass, 1.0),
            (
                "remote_local_binary_entropy",
                self.remote_local_binary_entropy,
                math.log(2.0),
            ),
            ("floor_violation_fraction", self.floor_violation_fraction, 1.0),
        ):
            if value is not None and (
                not math.isfinite(value) or not -1.0e-8 <= value <= upper + 1.0e-8
            ):
                raise RecurrentPPOUpdateError(f"{name} lies outside its valid range")


@dataclass(frozen=True)
class RouteChoiceStabilityTelemetry:
    """Detached runtime telemetry for Candidate-A route stabilization."""

    conditional_remote_mass: float | None
    remote_local_binary_entropy: float | None
    floor_violation_fraction: float | None
    unscaled_stability_loss: float
    scaled_stability_loss: float
    local_sample_count: int
    remote_sample_count: int
    remote_scorer_gradient_norm: float | None
    local_route_row_gradient_norm: float | None
    gradient_norm_ratio: float | None
    valid_sample_count: int

    def __post_init__(self) -> None:
        scalar_values = (
            self.unscaled_stability_loss,
            self.scaled_stability_loss,
        )
        if not all(math.isfinite(value) and value >= -1.0e-12 for value in scalar_values):
            raise RecurrentPPOUpdateError(
                "route stability loss telemetry must be finite and non-negative"
            )
        for value in (
            self.remote_scorer_gradient_norm,
            self.local_route_row_gradient_norm,
            self.gradient_norm_ratio,
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise RecurrentPPOUpdateError(
                    "route stability gradient telemetry must be finite and non-negative"
                )

    def record(self) -> dict[str, int | float | None]:
        """Expose stable field names without changing artifact/checkpoint schemas."""

        return {
            "conditional_remote_mass": self.conditional_remote_mass,
            "remote_local_binary_entropy": self.remote_local_binary_entropy,
            "floor_violation_fraction": self.floor_violation_fraction,
            "unscaled_stability_loss": self.unscaled_stability_loss,
            "scaled_stability_loss": self.scaled_stability_loss,
            "local_sample_count": self.local_sample_count,
            "remote_sample_count": self.remote_sample_count,
            "remote_scorer_gradient_norm": self.remote_scorer_gradient_norm,
            "local_route_row_gradient_norm": self.local_route_row_gradient_norm,
            "gradient_norm_ratio": self.gradient_norm_ratio,
            "valid_sample_count": self.valid_sample_count,
        }


def compute_route_choice_stability_loss(
    *,
    route_logits: Tensor,
    route_action_masks: Tensor,
    route_active: Tensor,
    sequence_valid_mask: Tensor,
    local_domain_mask: Tensor,
    remote_domain_mask: Tensor,
    selected_route_indices: Tensor,
    entropy_floor_nats: float,
    coefficient: float,
) -> RouteChoiceStabilityLossOutput:
    """Compute the masked conditional Remote-Local entropy-floor penalty."""

    if not isinstance(route_logits, Tensor) or route_logits.ndim != 4:
        raise RecurrentPPOUpdateError("route_logits must have shape [B,T,A,D]")
    vector_shape = tuple(route_logits.shape)
    scalar_shape = vector_shape[:-1]
    for name, tensor in (
        ("route_action_masks", route_action_masks),
        ("local_domain_mask", local_domain_mask),
        ("remote_domain_mask", remote_domain_mask),
    ):
        if not isinstance(tensor, Tensor) or tuple(tensor.shape) != vector_shape:
            raise RecurrentPPOUpdateError(f"{name} must match route_logits")
        if tensor.dtype != torch.bool:
            raise RecurrentPPOUpdateError(f"{name} must be boolean")
    if (
        not isinstance(route_active, Tensor)
        or tuple(route_active.shape) != scalar_shape
        or route_active.dtype != torch.bool
    ):
        raise RecurrentPPOUpdateError("route_active must be boolean [B,T,A]")
    if (
        not isinstance(sequence_valid_mask, Tensor)
        or tuple(sequence_valid_mask.shape) != scalar_shape[:2]
        or sequence_valid_mask.dtype != torch.bool
    ):
        raise RecurrentPPOUpdateError("sequence_valid_mask must be boolean [B,T]")
    if (
        not isinstance(selected_route_indices, Tensor)
        or tuple(selected_route_indices.shape) != scalar_shape
        or selected_route_indices.dtype != torch.long
    ):
        raise RecurrentPPOUpdateError(
            "selected_route_indices must be int64 [B,T,A]"
        )
    if not torch.isfinite(route_logits).all():
        raise RecurrentPPOUpdateError("route_logits contain NaN or Inf")
    if torch.any(local_domain_mask & remote_domain_mask):
        raise RecurrentPPOUpdateError("local and Remote route domains overlap")
    if torch.any(local_domain_mask.sum(dim=-1) != 1):
        raise RecurrentPPOUpdateError(
            "each route domain must contain exactly one Local action"
        )
    if torch.any(selected_route_indices < 0) or torch.any(
        selected_route_indices >= vector_shape[-1]
    ):
        raise RecurrentPPOUpdateError("selected route index lies outside its domain")
    for name, value in (
        ("entropy_floor_nats", entropy_floor_nats),
        ("coefficient", coefficient),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise RecurrentPPOUpdateError(f"{name} must be finite and non-negative")
    if float(entropy_floor_nats) > math.log(2.0):
        raise RecurrentPPOUpdateError("entropy_floor_nats must not exceed log(2)")

    valid_actor = sequence_valid_mask.to(route_logits.device).unsqueeze(-1).expand(
        scalar_shape
    )
    active = route_active & valid_actor
    selected = selected_route_indices.unsqueeze(-1)
    selected_local = torch.gather(local_domain_mask, -1, selected).squeeze(-1)
    selected_remote = torch.gather(remote_domain_mask, -1, selected).squeeze(-1)
    local_sample_count = int((active & selected_local).sum().detach().cpu().item())
    remote_sample_count = int((active & selected_remote).sum().detach().cpu().item())

    legal_local = route_action_masks & local_domain_mask
    legal_remote = route_action_masks & remote_domain_mask
    eligible = active & legal_local.any(dim=-1) & legal_remote.any(dim=-1)
    valid_sample_count = int(eligible.sum().detach().cpu().item())
    if valid_sample_count == 0:
        zero = route_logits.reshape(-1)[0] * 0.0
        return RouteChoiceStabilityLossOutput(
            unscaled_loss=zero,
            scaled_loss=zero,
            conditional_remote_mass=None,
            remote_local_binary_entropy=None,
            floor_violation_fraction=None,
            valid_sample_count=0,
            local_sample_count=local_sample_count,
            remote_sample_count=remote_sample_count,
        )

    eligible_logits = route_logits[eligible]
    eligible_remote = legal_remote[eligible]
    eligible_local_domain = local_domain_mask[eligible]
    remote_logit = torch.logsumexp(
        eligible_logits.masked_fill(~eligible_remote, -torch.inf),
        dim=-1,
    )
    local_indices = eligible_local_domain.to(torch.long).argmax(dim=-1)
    local_logit = torch.gather(
        eligible_logits, -1, local_indices.unsqueeze(-1)
    ).squeeze(-1)
    delta = remote_logit - local_logit
    conditional_remote_mass = torch.sigmoid(delta)
    absolute_delta = delta.abs()
    minority_mass = torch.sigmoid(-absolute_delta)
    binary_entropy = (
        torch.log1p(torch.exp(-absolute_delta))
        + minority_mass * absolute_delta
    )
    violation = torch.relu(
        binary_entropy.new_tensor(float(entropy_floor_nats)) - binary_entropy
    )
    unscaled_loss = violation.square().mean()
    scaled_loss = unscaled_loss * float(coefficient)
    if not torch.isfinite(unscaled_loss) or not torch.isfinite(scaled_loss):
        raise RecurrentPPOUpdateError("route stability loss is NaN or Inf")
    return RouteChoiceStabilityLossOutput(
        unscaled_loss=unscaled_loss,
        scaled_loss=scaled_loss,
        conditional_remote_mass=float(
            conditional_remote_mass.detach().mean().cpu().item()
        ),
        remote_local_binary_entropy=float(binary_entropy.detach().mean().cpu().item()),
        floor_violation_fraction=float(
            (violation.detach() > 0.0).to(route_logits.dtype).mean().cpu().item()
        ),
        valid_sample_count=valid_sample_count,
        local_sample_count=local_sample_count,
        remote_sample_count=remote_sample_count,
    )


def _route_semantic_domain_masks(
    action_mask_batch: SequentialActionMaskBatch,
    route_logits: Tensor,
) -> tuple[Tensor, Tensor]:
    contracts = action_mask_batch.flattened()
    expected_rows = math.prod(route_logits.shape[:-1])
    if len(contracts) != expected_rows:
        raise RecurrentPPOUpdateError(
            "route contracts differ from evaluated policy axes"
        )
    local_rows: list[list[bool]] = []
    remote_rows: list[list[bool]] = []
    for contract in contracts:
        domain = tuple(contract.route_domain)
        if len(domain) != route_logits.shape[-1]:
            raise RecurrentPPOUpdateError(
                "route contract domain differs from evaluated logits"
            )
        local_rows.append(
            [type(value) is str and value == "local" for value in domain]
        )
        remote_rows.append(
            [isinstance(value, int) and not isinstance(value, bool) for value in domain]
        )
    shape = tuple(route_logits.shape)
    local = torch.tensor(
        local_rows,
        dtype=torch.bool,
        device=route_logits.device,
    ).reshape(shape)
    remote = torch.tensor(
        remote_rows,
        dtype=torch.bool,
        device=route_logits.device,
    ).reshape(shape)
    return local, remote


def _route_choice_stability_telemetry(
    loss: RouteChoiceStabilityLossOutput,
    route_head: nn.Module,
    action_mask_batch: SequentialActionMaskBatch,
) -> RouteChoiceStabilityTelemetry:
    route_domains = tuple(
        tuple(contract.route_domain) for contract in action_mask_batch.flattened()
    )
    gradient_rows = measure_route_head_gradient_rows(route_head, route_domains)
    local_rows = tuple(
        row for row in gradient_rows if row.semantic_role == "local"
    )
    remote_rows = tuple(
        row for row in gradient_rows if row.semantic_role == "remote"
    )
    if len(local_rows) != 1 or len(remote_rows) != 1:
        raise RecurrentPPOUpdateError(
            "candidate-aware route gradients require one Local and one Remote row"
        )
    local_norm = local_rows[0].grad_norm
    remote_norm = remote_rows[0].grad_norm
    gradient_ratio = (
        None
        if local_norm is None or remote_norm is None or local_norm <= 0.0
        else remote_norm / local_norm
    )
    return RouteChoiceStabilityTelemetry(
        conditional_remote_mass=loss.conditional_remote_mass,
        remote_local_binary_entropy=loss.remote_local_binary_entropy,
        floor_violation_fraction=loss.floor_violation_fraction,
        unscaled_stability_loss=float(loss.unscaled_loss.detach().cpu().item()),
        scaled_stability_loss=float(loss.scaled_loss.detach().cpu().item()),
        local_sample_count=loss.local_sample_count,
        remote_sample_count=loss.remote_sample_count,
        remote_scorer_gradient_norm=remote_norm,
        local_route_row_gradient_norm=local_norm,
        gradient_norm_ratio=gradient_ratio,
        valid_sample_count=loss.valid_sample_count,
    )


def _validate_frozen_contract(config: RunConfig) -> None:
    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    config.validate()
    mappo = config.training.mappo
    geometry = (
        mappo.rollout_length_slots,
        mappo.recurrent_chunk_length_slots,
        mappo.sequence_minibatch_size,
        mappo.update_epochs,
    )
    if geometry != _FROZEN_GEOMETRY:
        raise RecurrentPPOUpdateError(
            "recurrent PPO geometry must remain rollout=256, chunk=32, "
            "minibatch=8, epochs=4"
        )


def _cat_actor_batches(batches: tuple[ActorTensorBatch, ...]) -> ActorTensorBatch:
    return ActorTensorBatch(
        self_features=torch.cat([item.self_features for item in batches], dim=0),
        neighbor_public_features=torch.cat(
            [item.neighbor_public_features for item in batches], dim=0
        ),
        edge_features=torch.cat([item.edge_features for item in batches], dim=0),
        neighbor_mask=torch.cat([item.neighbor_mask for item in batches], dim=0),
        action_masks={
            branch: torch.cat([item.action_masks[branch] for item in batches], dim=0)
            for branch in ACTION_BRANCH_ORDER
        },
        action_indices={
            branch: torch.cat([item.action_indices[branch] for item in batches], dim=0)
            for branch in ACTION_BRANCH_ORDER
        },
        episode_starts=torch.cat([item.episode_starts for item in batches], dim=0),
    )


@dataclass(frozen=True)
class RecurrentPPOMinibatch:
    """One deterministic, padding-free view of a finalized 256-slot rollout."""

    spec: MAPPOTensorSpec
    rollout_indices: Tensor
    actor_batch: ActorTensorBatch
    action_mask_batch: SequentialActionMaskBatch
    centralized_batch: CentralizedStateTensorBatch
    proposals: tuple[tuple[tuple[ActionProposal, ...], ...], ...]
    initial_hidden: Tensor
    stored_hidden_in: Tensor
    active_branch_indicators: Tensor
    old_joint_log_prob: Tensor
    old_value: Tensor
    advantage: Tensor
    return_target: Tensor
    sequence_valid_mask: Tensor
    agent_credit_mode: str
    old_branch_log_probs: Tensor | None = None
    td_residual: Tensor | None = None
    route_advantage: Tensor | None = None
    bootstrap_mask: Tensor | None = None
    route_nstep_samples: tuple[
        tuple[tuple[RouteNstepSample | None, ...], ...], ...
    ] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, MAPPOTensorSpec):
            raise TypeError("spec must be a MAPPOTensorSpec")
        shape = self.actor_batch.validate(self.spec)
        if shape != (8, 32, self.spec.uav_count):
            raise RecurrentPPOUpdateError("actor minibatch must have shape [8,32,A]")
        self.action_mask_batch.validate(shape)
        if self.centralized_batch.validate(self.spec) != (8, 32):
            raise RecurrentPPOUpdateError("critic minibatch must have shape [8,32,F]")
        try:
            credit_mode = AgentCreditMode(self.agent_credit_mode)
        except (TypeError, ValueError) as exc:
            raise RecurrentPPOUpdateError("agent_credit_mode is invalid") from exc
        credit_shape = (
            (8, 32)
            if credit_mode is AgentCreditMode.TEAM
            else (8, 32, self.spec.uav_count)
        )
        expected = {
            "rollout_indices": (8, 32),
            "initial_hidden": (8, self.spec.uav_count, self.spec.recurrent_state_dimension),
            "stored_hidden_in": (
                8,
                32,
                self.spec.uav_count,
                self.spec.recurrent_state_dimension,
            ),
            "active_branch_indicators": (
                8,
                32,
                self.spec.uav_count,
                len(ACTION_BRANCH_ORDER),
            ),
            "old_joint_log_prob": (8, 32, self.spec.uav_count),
            "old_value": credit_shape,
            "advantage": credit_shape,
            "return_target": credit_shape,
            "sequence_valid_mask": (8, 32),
        }
        for name, expected_shape in expected.items():
            tensor = getattr(self, name)
            if not isinstance(tensor, Tensor) or tuple(tensor.shape) != expected_shape:
                raise RecurrentPPOUpdateError(f"{name} has an invalid shape")
            if tensor.device.type != "cpu":
                raise RecurrentPPOUpdateError(f"{name} must remain a CPU snapshot")
        optional = {
            "old_branch_log_probs": (8, 32, self.spec.uav_count, len(ACTION_BRANCH_ORDER)),
            "td_residual": credit_shape,
            "route_advantage": (8, 32, self.spec.uav_count),
        }
        for name, expected_shape in optional.items():
            tensor = getattr(self, name)
            if tensor is not None and (not isinstance(tensor, Tensor) or tuple(tensor.shape) != expected_shape or tensor.device.type != "cpu"):
                raise RecurrentPPOUpdateError(f"{name} has an invalid optional snapshot shape")
            if tensor is not None and (tensor.dtype != torch.float32 or tensor.requires_grad or not torch.isfinite(tensor).all()):
                raise RecurrentPPOUpdateError(f"{name} must be a detached finite CPU float32 snapshot")
        if self.bootstrap_mask is not None and (
            not isinstance(self.bootstrap_mask, Tensor)
            or tuple(self.bootstrap_mask.shape) != (8, 32)
            or self.bootstrap_mask.device.type != "cpu"
            or self.bootstrap_mask.dtype != torch.bool
        ):
            raise RecurrentPPOUpdateError(
                "bootstrap_mask must be a detached boolean [8,32] CPU snapshot"
            )
        if self.rollout_indices.dtype != torch.long:
            raise RecurrentPPOUpdateError("rollout_indices must use torch.long")
        expected_indices = torch.arange(256, dtype=torch.long).reshape(8, 32)
        if not torch.equal(self.rollout_indices, expected_indices):
            raise RecurrentPPOUpdateError(
                "chunks must preserve deterministic contiguous rollout order"
            )
        if self.sequence_valid_mask.dtype != torch.bool or not torch.all(
            self.sequence_valid_mask
        ):
            raise RecurrentPPOUpdateError("the frozen minibatch has no padding")
        if self.active_branch_indicators.dtype != torch.bool:
            raise RecurrentPPOUpdateError("active_branch_indicators must be boolean")
        floating = (
            self.initial_hidden,
            self.stored_hidden_in,
            self.old_joint_log_prob,
            self.old_value,
            self.advantage,
            self.return_target,
        )
        if any(
            tensor.dtype != torch.float32
            or tensor.requires_grad
            or not torch.isfinite(tensor).all()
            for tensor in floating
        ):
            raise RecurrentPPOUpdateError(
                "rollout and target statistics must be detached finite CPU float32"
            )
        if not torch.equal(self.initial_hidden, self.stored_hidden_in[:, 0]):
            raise RecurrentPPOUpdateError(
                "each chunk must start from stored hidden_in[t_start]"
            )
        episode_starts = self.actor_batch.episode_starts
        reset_hidden = self.stored_hidden_in.masked_select(
            episode_starts.unsqueeze(-1).expand_as(self.stored_hidden_in)
        )
        if torch.count_nonzero(reset_hidden).item() != 0:
            raise RecurrentPPOUpdateError(
                "episode-start hidden snapshots must be exactly zero"
            )
        if len(self.proposals) != 8 or any(len(chunk) != 32 for chunk in self.proposals):
            raise RecurrentPPOUpdateError("proposal snapshots must have shape [8][32][A]")
        if any(
            len(time_step) != self.spec.uav_count
            for chunk in self.proposals
            for time_step in chunk
        ):
            raise RecurrentPPOUpdateError("proposal snapshots have a ragged agent axis")
        if self.route_nstep_samples is not None and (
            len(self.route_nstep_samples) != 8
            or any(
                len(chunk) != 32
                for chunk in self.route_nstep_samples
            )
            or any(
                len(time_step) != self.spec.uav_count
                for chunk in self.route_nstep_samples
                for time_step in chunk
            )
        ):
            raise RecurrentPPOUpdateError(
                "route_nstep_samples must have shape [8][32][A]"
            )


def build_recurrent_ppo_minibatch(
    buffer: CAGATMAPPORolloutBuffer,
    config: RunConfig,
) -> RecurrentPPOMinibatch:
    """Build the sole frozen minibatch without shuffle, padding, or burn-in."""

    _validate_frozen_contract(config)
    if not isinstance(buffer, CAGATMAPPORolloutBuffer):
        raise TypeError("buffer must be a CAGATMAPPORolloutBuffer")
    expected_spec = MAPPOTensorSpec.from_config(config)
    if buffer.spec != expected_spec:
        raise RecurrentPPOUpdateError("rollout buffer and config tensor specs differ")
    if not buffer.finalized:
        raise RecurrentPPOUpdateError("rollout must be finalized before PPO update")
    if len(buffer) != 256:
        raise RecurrentPPOUpdateError("finalized rollout must contain exactly 256 slots")

    chunks = tuple(buffer.get_sequence(index * 32, 32) for index in range(8))
    full_rollout = buffer.get_sequence(0, 256)
    gae = compute_rollout_gae(full_rollout, config)
    route_advantage = None
    route_nstep_samples = None
    mappo = config.training.mappo
    route_mode = RouteCreditMode(mappo.route_credit_mode)
    if route_mode is RouteCreditMode.ROUTE_SPECIFIC_GAE:
        route_advantage = compute_route_specific_advantage(
            base_td_residual=gae.td_residual,
            bootstrap_mask=gae.bootstrap_mask,
            sequence_mask=gae.sequence_mask,
            gamma=mappo.gamma,
            route_gae_lambda=mappo.route_gae_lambda,
        )
    elif route_mode is RouteCreditMode.ROLLOUT_CAPPED_ROUTE_EVENT_NSTEP:
        nstep = compute_rollout_capped_route_event_nstep(
            reward=full_rollout.reward,
            old_value=full_rollout.old_value,
            bootstrap_value=full_rollout.bootstrap_values,
            bootstrap_mask=gae.bootstrap_mask,
            episode_boundary=full_rollout.episode_boundary,
            route_active=full_rollout.active_branch_indicators[..., 0],
            base_advantage=gae.advantage,
            transition_metadata=(
                full_rollout.rejection_or_downgrade_summaries
            ),
            gamma=mappo.gamma,
        )
        route_advantage = nstep.advantage
        route_nstep_samples = tuple(
            tuple(nstep.samples[index * 32 : (index + 1) * 32])
            for index in range(8)
        )
    actor_batches = tuple(chunk.actor_batch for chunk in chunks)
    actor_batch = _cat_actor_batches(actor_batches)
    centralized_batch = CentralizedStateTensorBatch(
        torch.cat([chunk.centralized_batch.features for chunk in chunks], dim=0)
    )
    action_mask_batch = SequentialActionMaskBatch(
        tuple(chunk.action_mask_batch.contracts[0] for chunk in chunks)
    )
    proposals = tuple(chunk.proposals[0] for chunk in chunks)
    result = RecurrentPPOMinibatch(
        spec=expected_spec,
        rollout_indices=torch.arange(256, dtype=torch.long).reshape(8, 32),
        actor_batch=actor_batch,
        action_mask_batch=action_mask_batch,
        centralized_batch=centralized_batch,
        proposals=proposals,
        initial_hidden=torch.cat([chunk.initial_hidden for chunk in chunks], dim=0),
        stored_hidden_in=torch.stack([chunk.hidden_in for chunk in chunks], dim=0),
        active_branch_indicators=torch.stack(
            [chunk.active_branch_indicators for chunk in chunks], dim=0
        ),
        old_joint_log_prob=torch.stack(
            [chunk.old_joint_log_prob for chunk in chunks], dim=0
        ),
        old_value=torch.stack([chunk.old_value for chunk in chunks], dim=0),
        advantage=gae.advantage.reshape(
            (8, 32) + tuple(gae.advantage.shape[1:])
        ).clone(),
        return_target=gae.return_target.reshape(
            (8, 32) + tuple(gae.return_target.shape[1:])
        ).clone(),
        sequence_valid_mask=gae.sequence_mask.reshape(8, 32).clone(),
        agent_credit_mode=AgentCreditMode(
            config.training.mappo.agent_credit_mode
        ).value,
        old_branch_log_probs=torch.stack([chunk.old_branch_log_probs for chunk in chunks], dim=0),
        td_residual=gae.td_residual.reshape(
            (8, 32) + tuple(gae.td_residual.shape[1:])
        ).clone(),
        route_advantage=(
            None
            if route_advantage is None
            else route_advantage.reshape(8, 32, expected_spec.uav_count).clone()
        ),
        bootstrap_mask=gae.bootstrap_mask.reshape(8, 32).clone(),
        route_nstep_samples=route_nstep_samples,
    )
    return result


@dataclass(frozen=True)
class BatchedCAGATMAPPOLossOutput:
    """The existing PPO output with explicit recurrent [B,L,A] views."""

    flattened: CAGATMAPPOLossOutput
    batch_size: int
    sequence_length: int
    agent_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.flattened, CAGATMAPPOLossOutput):
            raise TypeError("flattened must be a CAGATMAPPOLossOutput")
        if tuple(self.flattened.ratio.shape) != (
            self.batch_size * self.sequence_length,
            self.agent_count,
        ):
            raise RecurrentPPOUpdateError("flattened PPO output has incompatible axes")

    def _actor_view(self, tensor: Tensor) -> Tensor:
        return tensor.reshape(self.batch_size, self.sequence_length, self.agent_count)

    def _critic_view(self, tensor: Tensor) -> Tensor:
        return tensor.reshape(
            (self.batch_size, self.sequence_length) + tuple(tensor.shape[1:])
        )

    def _branch_view(self, tensor: Tensor) -> Tensor:
        return tensor.reshape(
            self.batch_size,
            self.sequence_length,
            self.agent_count,
            len(ACTION_BRANCH_ORDER),
        )

    @property
    def log_ratio(self) -> Tensor:
        return self._actor_view(self.flattened.log_ratio)

    @property
    def ratio(self) -> Tensor:
        return self._actor_view(self.flattened.ratio)

    @property
    def clipped_ratio(self) -> Tensor:
        return self._actor_view(self.flattened.clipped_ratio)

    @property
    def expanded_advantage(self) -> Tensor:
        return self._actor_view(self.flattened.expanded_advantage)

    @property
    def surrogate(self) -> Tensor:
        return self._actor_view(self.flattened.surrogate)

    @property
    def branch_ratio(self) -> Tensor | None:
        value = self.flattened.branch_ratio
        return None if value is None else self._branch_view(value)

    @property
    def branch_surrogate(self) -> Tensor | None:
        value = self.flattened.branch_surrogate
        return None if value is None else self._branch_view(value)

    @property
    def branch_advantage(self) -> Tensor | None:
        value = self.flattened.branch_advantage
        return None if value is None else self._branch_view(value)

    @property
    def active_branch_indicators(self) -> Tensor | None:
        value = self.flattened.active_branch_indicators
        return None if value is None else self._branch_view(value)

    @property
    def active_branch_count(self) -> Tensor | None:
        value = self.flattened.active_branch_count
        return None if value is None else self._actor_view(value)

    @property
    def actor_valid_mask(self) -> Tensor:
        return self._actor_view(self.flattened.actor_valid_mask)

    @property
    def critic_valid_mask(self) -> Tensor:
        return self._critic_view(self.flattened.critic_valid_mask)

    @property
    def clipped_objective(self) -> Tensor:
        return self.flattened.clipped_objective

    @property
    def actor_loss(self) -> Tensor:
        return self.flattened.actor_loss

    @property
    def critic_loss(self) -> Tensor:
        return self.flattened.critic_loss

    @property
    def entropy_mean(self) -> Tensor:
        return self.flattened.entropy_mean

    @property
    def total_loss(self) -> Tensor:
        return self.flattened.total_loss

    @property
    def diagnostics(self):
        return self.flattened.diagnostics

    @property
    def route_entropy_coefficient(self) -> float:
        return self.flattened.route_entropy_coefficient

    @property
    def route_entropy_schedule_progress(self) -> float:
        return self.flattened.route_entropy_schedule_progress

    @property
    def route_entropy_loss_contribution(self) -> Tensor:
        return self.flattened.route_entropy_loss_contribution

    @property
    def other_branch_entropy_loss_contribution(self) -> Tensor:
        return self.flattened.other_branch_entropy_loss_contribution

    @property
    def global_entropy_loss_contribution(self) -> Tensor:
        return self.flattened.global_entropy_loss_contribution

    @property
    def collected_environment_steps(self) -> int | None:
        return self.flattened.collected_environment_steps


def compute_configured_batched_ppo_objective_and_loss(
    *,
    new_joint_log_prob: Tensor,
    old_joint_log_prob: Tensor,
    advantage: Tensor,
    current_value: Tensor,
    return_target: Tensor,
    entropy: Tensor,
    sequence_valid_mask: Tensor,
    config: RunConfig,
    new_branch_log_probs: Tensor | None = None,
    old_branch_log_probs: Tensor | None = None,
    active_branch_indicators: Tensor | None = None,
    route_entropy: Tensor | None = None,
    collected_environment_steps: int | None = None,
    route_advantage: Tensor | None = None,
    tx_upper_logit_gap: Tensor | None = None,
    tx_upper_transmitted: Tensor | None = None,
    tx_upper_entropy: Tensor | None = None,
    entropy_without_tx_upper: Tensor | None = None,
    tx_upper_advantage: Tensor | None = None,
    tx_upper_paired_mask: Tensor | None = None,
) -> BatchedCAGATMAPPOLossOutput:
    """Apply the passed PPO objective uniformly over all B-by-L positions."""

    _validate_frozen_contract(config)
    if not isinstance(new_joint_log_prob, Tensor) or new_joint_log_prob.ndim != 3:
        raise RecurrentPPOUpdateError("new_joint_log_prob must have shape [B,L,A]")
    batch_size, sequence_length, agent_count = new_joint_log_prob.shape
    if (batch_size, sequence_length) != (8, 32) or agent_count <= 0:
        raise RecurrentPPOUpdateError("batched PPO policy axes must be [8,32,A]")
    for name, tensor in (
        ("old_joint_log_prob", old_joint_log_prob),
        ("entropy", entropy),
    ):
        if not isinstance(tensor, Tensor) or tensor.shape != new_joint_log_prob.shape:
            raise RecurrentPPOUpdateError(f"{name} must have shape [8,32,A]")
    if route_entropy is not None and (
        not isinstance(route_entropy, Tensor)
        or route_entropy.shape != new_joint_log_prob.shape
    ):
        raise RecurrentPPOUpdateError(
            "route_entropy must have shape [8,32,A] or be NA"
        )
    credit_mode = AgentCreditMode(config.training.mappo.agent_credit_mode)
    expected_credit_shape = (
        (8, 32)
        if credit_mode is AgentCreditMode.TEAM
        else (8, 32, agent_count)
    )
    for name, tensor in (
        ("advantage", advantage),
        ("current_value", current_value),
        ("return_target", return_target),
    ):
        if not isinstance(tensor, Tensor) or tensor.shape != expected_credit_shape:
            raise RecurrentPPOUpdateError(
                f"{name} shape differs from agent_credit_mode"
            )
    if route_advantage is not None and (
        not isinstance(route_advantage, Tensor)
        or tuple(route_advantage.shape) != (8, 32, agent_count)
    ):
        raise RecurrentPPOUpdateError(
            "route_advantage must have shape [8,32,A] or be NA"
        )
    if (
        not isinstance(sequence_valid_mask, Tensor)
        or sequence_valid_mask.shape != (8, 32)
        or sequence_valid_mask.dtype != torch.bool
        or not torch.all(sequence_valid_mask)
    ):
        raise RecurrentPPOUpdateError("all 8x32 positions must be real and valid")

    from .integrated_scheduler import policy_component_names
    component_count = len(policy_component_names(config))
    ratio_mode = ActorRatioMode(config.training.mappo.actor_ratio_mode)
    flattened_new_branches = None
    flattened_old_branches = None
    flattened_active_branches = None
    if ratio_mode is ActorRatioMode.BRANCH_SPECIFIC:
        expected_branch_shape = (
            8,
            32,
            agent_count,
            component_count,
        )
        for name, tensor in (
            ("new_branch_log_probs", new_branch_log_probs),
            ("old_branch_log_probs", old_branch_log_probs),
        ):
            if not isinstance(tensor, Tensor) or tuple(tensor.shape) != expected_branch_shape:
                raise RecurrentPPOUpdateError(
                    f"{name} must have shape [8,32,A,7]"
                )
        if (
            not isinstance(active_branch_indicators, Tensor)
            or tuple(active_branch_indicators.shape) != expected_branch_shape
            or active_branch_indicators.dtype != torch.bool
        ):
            raise RecurrentPPOUpdateError(
                "active_branch_indicators must be boolean [8,32,A,7]"
            )
        assert new_branch_log_probs is not None
        assert old_branch_log_probs is not None
        flattened_new_branches = new_branch_log_probs.reshape(
            256, agent_count, component_count
        )
        flattened_old_branches = old_branch_log_probs.reshape(
            256, agent_count, component_count
        )
        flattened_active_branches = active_branch_indicators.reshape(
            256, agent_count, component_count
        )

    flattened = compute_configured_ppo_objective_and_loss(
        new_joint_log_prob=new_joint_log_prob.reshape(256, agent_count),
        old_joint_log_prob=old_joint_log_prob.reshape(256, agent_count),
        advantage=advantage.reshape(
            (256,) if credit_mode is AgentCreditMode.TEAM else (256, agent_count)
        ),
        current_value=current_value.reshape(
            (256,) if credit_mode is AgentCreditMode.TEAM else (256, agent_count)
        ),
        return_target=return_target.reshape(
            (256,) if credit_mode is AgentCreditMode.TEAM else (256, agent_count)
        ),
        entropy=entropy.reshape(256, agent_count),
        sequence_valid_mask=sequence_valid_mask.reshape(256),
        config=config,
        new_branch_log_probs=flattened_new_branches,
        old_branch_log_probs=flattened_old_branches,
        active_branch_indicators=flattened_active_branches,
        route_entropy=(
            None
            if route_entropy is None
            else route_entropy.reshape(256, agent_count)
        ),
        collected_environment_steps=collected_environment_steps,
        route_advantage=(
            None
            if route_advantage is None
            else route_advantage.reshape(256, agent_count)
        ),
        tx_upper_logit_gap=(None if tx_upper_logit_gap is None else tx_upper_logit_gap.reshape(256, agent_count)),
        tx_upper_transmitted=(None if tx_upper_transmitted is None else tx_upper_transmitted.reshape(256, agent_count)),
        tx_upper_entropy=(None if tx_upper_entropy is None else tx_upper_entropy.reshape(256, agent_count)),
        entropy_without_tx_upper=(None if entropy_without_tx_upper is None else entropy_without_tx_upper.reshape(256, agent_count)),
        tx_upper_advantage=(None if tx_upper_advantage is None else tx_upper_advantage.reshape(256, agent_count)),
        tx_upper_paired_mask=(None if tx_upper_paired_mask is None else tx_upper_paired_mask.reshape(256, agent_count)),
    )
    return BatchedCAGATMAPPOLossOutput(
        flattened=flattened,
        batch_size=batch_size,
        sequence_length=sequence_length,
        agent_count=agent_count,
    )


def _module_device_dtype(module: nn.Module, name: str) -> tuple[torch.device, torch.dtype]:
    from .tx_upper_counterfactual import model_precision_tensors
    tensors = tuple(module.parameters()) + tuple(module.buffers())
    if not tensors:
        raise RecurrentPPOUpdateError(f"{name} has no tensors")
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise RecurrentPPOUpdateError(f"{name} tensors must use one device")
    floating_dtypes = {tensor.dtype for tensor in model_precision_tensors(module) if tensor.is_floating_point()}
    if len(floating_dtypes) != 1:
        raise RecurrentPPOUpdateError(f"{name} floating tensors must use one dtype")
    dtype = next(iter(floating_dtypes))
    if dtype != torch.float32:
        raise RecurrentPPOUpdateError("the frozen network update dtype is torch.float32")
    return next(iter(devices)), dtype


def _actor_optimizer_parameter_groups(
    actor: CAGATMAPPOActor, config: RunConfig,
) -> list[dict[str, object]]:
    """Partition the complete Route subsystem from every other actor parameter.

    E7 includes the four private backbone modules in the existing slow group;
    its LR reuses route_head_lr_scale without introducing another tuning knob.

    Keep model registration order inside each group; clipping and loss/gradient
    construction continue to use the unchanged actor_parameters tuple.
    """
    named = tuple((name, parameter) for name, parameter in actor.named_parameters()
                  if parameter.requires_grad)
    route = tuple(parameter for name, parameter in named
                  if name.startswith(actor.route_parameter_prefixes))
    other = tuple(parameter for name, parameter in named
                  if not name.startswith(actor.route_parameter_prefixes))
    if not route or not other:
        raise RecurrentPPOUpdateError("actor needs nonempty Route and other parameter groups")
    mappo = config.training.mappo
    return [
        {"group_name": "route_head", "params": route,
         "lr": mappo.actor_learning_rate * mappo.route_head_lr_scale},
        {"group_name": "actor_other", "params": other,
         "lr": mappo.actor_learning_rate},
    ]


@dataclass(frozen=True)
class CAGATMAPPOOptimizerBundle:
    """Separate Adam optimizers and their disjoint trainable parameter sets."""

    actor_optimizer: Adam
    critic_optimizer: Adam
    actor_parameters: tuple[nn.Parameter, ...]
    critic_parameters: tuple[nn.Parameter, ...]

    def validate(
        self,
        actor: CAGATMAPPOActor,
        critic: MAPPOCentralizedCritic,
        config: RunConfig,
    ) -> None:
        _validate_frozen_contract(config)
        if not isinstance(self.actor_optimizer, Adam) or not isinstance(
            self.critic_optimizer, Adam
        ):
            raise RecurrentPPOUpdateError("actor and critic optimizers must be Adam")
        expected_actor = tuple(parameter for parameter in actor.parameters() if parameter.requires_grad)
        expected_critic = tuple(parameter for parameter in critic.parameters() if parameter.requires_grad)
        if tuple(map(id, self.actor_parameters)) != tuple(map(id, expected_actor)):
            raise RecurrentPPOUpdateError("actor optimizer parameter set is incomplete")
        if tuple(map(id, self.critic_parameters)) != tuple(map(id, expected_critic)):
            raise RecurrentPPOUpdateError("critic optimizer parameter set is incomplete")
        if set(map(id, self.actor_parameters)) & set(map(id, self.critic_parameters)):
            raise RecurrentPPOUpdateError("actor and critic parameters must not overlap")
        actor_optimized = tuple(
            parameter
            for group in self.actor_optimizer.param_groups
            for parameter in group["params"]
        )
        critic_optimized = tuple(
            parameter
            for group in self.critic_optimizer.param_groups
            for parameter in group["params"]
        )
        actor_ids = tuple(map(id, actor_optimized))
        if (len(actor_ids) != len(set(actor_ids))
                or set(actor_ids) != set(map(id, self.actor_parameters))):
            raise RecurrentPPOUpdateError("actor Adam parameter IDs differ from actor set")
        if tuple(map(id, critic_optimized)) != tuple(map(id, self.critic_parameters)):
            raise RecurrentPPOUpdateError("critic Adam parameter IDs differ from critic set")
        mappo = config.training.mappo
        expected_actor_groups = _actor_optimizer_parameter_groups(actor, config)
        if len(self.actor_optimizer.param_groups) != 2:
            raise RecurrentPPOUpdateError("actor Adam must use two parameter groups")
        for group, expected_group in zip(self.actor_optimizer.param_groups, expected_actor_groups):
            if (group.get("group_name") != expected_group["group_name"]
                    or tuple(map(id, group["params"])) != tuple(map(id, expected_group["params"]))):
                raise RecurrentPPOUpdateError("actor Adam Route/other parameter group partition mismatch")
        if len(self.critic_optimizer.param_groups) != 1:
            raise RecurrentPPOUpdateError("critic Adam must use one parameter group")
        for name, group, learning_rate in (
            ("actor Route", self.actor_optimizer.param_groups[0],
             mappo.actor_learning_rate * mappo.route_head_lr_scale),
            ("actor other", self.actor_optimizer.param_groups[1], mappo.actor_learning_rate),
            ("critic", self.critic_optimizer.param_groups[0], mappo.critic_learning_rate),
        ):
            expected = {
                "lr": learning_rate,
                "betas": (mappo.adam_beta1, mappo.adam_beta2),
                "eps": mappo.adam_eps,
                "weight_decay": mappo.weight_decay,
            }
            if any(group[key] != value for key, value in expected.items()):
                raise RecurrentPPOUpdateError(
                    f"{name} Adam hyperparameters differ from RunConfig"
                )


def build_ca_gat_mappo_optimizers(
    actor: CAGATMAPPOActor,
    critic: MAPPOCentralizedCritic,
    config: RunConfig,
) -> CAGATMAPPOOptimizerBundle:
    """Build the two independent Adam optimizers from ``RunConfig`` only."""

    _validate_frozen_contract(config)
    if not isinstance(actor, CAGATMAPPOActor):
        raise TypeError("actor must be a CAGATMAPPOActor")
    if not isinstance(critic, MAPPOCentralizedCritic):
        raise TypeError("critic must be a MAPPOCentralizedCritic")
    expected_spec = MAPPOTensorSpec.from_config(config)
    if actor.spec != expected_spec or critic.spec != expected_spec:
        raise RecurrentPPOUpdateError("actor, critic, and config tensor specs must agree")
    actor_device, actor_dtype = _module_device_dtype(actor, "actor")
    critic_device, critic_dtype = _module_device_dtype(critic, "critic")
    if actor_device != critic_device or actor_dtype != critic_dtype:
        raise RecurrentPPOUpdateError("actor and critic must use the same device and dtype")
    actor_parameters = tuple(parameter for parameter in actor.parameters() if parameter.requires_grad)
    critic_parameters = tuple(parameter for parameter in critic.parameters() if parameter.requires_grad)
    if not actor_parameters or not critic_parameters:
        raise RecurrentPPOUpdateError("actor and critic need trainable parameters")
    if set(map(id, actor_parameters)) & set(map(id, critic_parameters)):
        raise RecurrentPPOUpdateError("actor and critic trainable parameters overlap")
    mappo = config.training.mappo
    common = {
        "betas": (mappo.adam_beta1, mappo.adam_beta2),
        "eps": mappo.adam_eps,
        "weight_decay": mappo.weight_decay,
    }
    bundle = CAGATMAPPOOptimizerBundle(
        actor_optimizer=Adam(_actor_optimizer_parameter_groups(actor, config),
                             lr=mappo.actor_learning_rate, **common),
        critic_optimizer=Adam(critic_parameters, lr=mappo.critic_learning_rate, **common),
        actor_parameters=actor_parameters,
        critic_parameters=critic_parameters,
    )
    bundle.validate(actor, critic, config)
    return bundle


@dataclass(frozen=True)
class RecurrentPPOEvaluation:
    """Current-policy proposal statistics and centralized values for one minibatch."""

    policy: SequentialActionDistributionOutput
    current_value: Tensor
    agent_credit_mode: str

    def __post_init__(self) -> None:
        if self.policy.mode != "evaluation":
            raise RecurrentPPOUpdateError("PPO update must re-evaluate stored proposals")
        if tuple(self.policy.joint_log_prob.shape[:2]) != (8, 32):
            raise RecurrentPPOUpdateError("policy evaluation must retain [8,32,A] axes")
        try:
            credit_mode = AgentCreditMode(self.agent_credit_mode)
        except (TypeError, ValueError) as exc:
            raise RecurrentPPOUpdateError("agent_credit_mode is invalid") from exc
        agent_count = self.policy.joint_log_prob.shape[-1]
        expected = (
            (8, 32)
            if credit_mode is AgentCreditMode.TEAM
            else (8, 32, agent_count)
        )
        if tuple(self.current_value.shape) != expected:
            raise RecurrentPPOUpdateError(
                "current critic value shape differs from agent_credit_mode"
            )


@dataclass(frozen=True)
class _DeviceRecurrentPPOMinibatch:
    actor_batch: ActorTensorBatch
    action_mask_batch: SequentialActionMaskBatch
    centralized_batch: CentralizedStateTensorBatch
    proposals: tuple[tuple[tuple[ActionProposal, ...], ...], ...]
    initial_hidden: Tensor
    active_branch_indicators: Tensor
    old_joint_log_prob: Tensor
    old_value: Tensor
    advantage: Tensor
    return_target: Tensor
    sequence_valid_mask: Tensor
    old_branch_log_probs: Tensor | None = None
    td_residual: Tensor | None = None
    route_advantage: Tensor | None = None
    bootstrap_mask: Tensor | None = None
    route_nstep_samples: tuple[
        tuple[tuple[RouteNstepSample | None, ...], ...], ...
    ] | None = None


def _prepare_device_minibatch(
    minibatch: RecurrentPPOMinibatch,
    device: torch.device,
    dtype: torch.dtype,
) -> _DeviceRecurrentPPOMinibatch:
    return _DeviceRecurrentPPOMinibatch(
        actor_batch=minibatch.actor_batch.to(device, dtype=dtype),
        action_mask_batch=minibatch.action_mask_batch,
        centralized_batch=minibatch.centralized_batch.to(device, dtype=dtype),
        proposals=minibatch.proposals,
        initial_hidden=minibatch.initial_hidden.to(device=device, dtype=dtype),
        active_branch_indicators=minibatch.active_branch_indicators.to(device=device),
        old_joint_log_prob=minibatch.old_joint_log_prob.to(device=device, dtype=dtype),
        old_value=minibatch.old_value.to(device=device, dtype=dtype),
        advantage=minibatch.advantage.to(device=device, dtype=dtype),
        return_target=minibatch.return_target.to(device=device, dtype=dtype),
        sequence_valid_mask=minibatch.sequence_valid_mask.to(device=device),
        old_branch_log_probs=(minibatch.old_branch_log_probs.to(device=device, dtype=dtype)
                              if minibatch.old_branch_log_probs is not None else None),
        td_residual=(minibatch.td_residual.to(device=device, dtype=dtype)
                     if minibatch.td_residual is not None else None),
        route_advantage=(minibatch.route_advantage.to(device=device, dtype=dtype)
                         if minibatch.route_advantage is not None else None),
        bootstrap_mask=(minibatch.bootstrap_mask.to(device=device)
                        if minibatch.bootstrap_mask is not None else None),
        route_nstep_samples=minibatch.route_nstep_samples,
    )


@dataclass(frozen=True)
class AgentCreditPPOEpochTelemetry:
    """Detached Fix-5 statistics derived from an existing PPO evaluation."""

    per_agent_value_mean: tuple[float, ...]
    per_agent_td_residual_mean: tuple[float, ...]
    per_agent_advantage_mean: tuple[float, ...]
    per_agent_return_mean: tuple[float, ...]
    same_timestep_advantage_equality_rate: float
    route_category_agent_advantage_mean: Mapping[
        str, tuple[float | None, ...]
    ]
    per_head_critic_loss: tuple[float, ...]

    def __post_init__(self) -> None:
        agent_count = len(self.per_agent_value_mean)
        if agent_count == 0:
            raise RecurrentPPOUpdateError("agent credit telemetry cannot be empty")
        vectors = (
            self.per_agent_value_mean,
            self.per_agent_td_residual_mean,
            self.per_agent_advantage_mean,
            self.per_agent_return_mean,
            self.per_head_critic_loss,
        )
        if any(len(values) != agent_count for values in vectors):
            raise RecurrentPPOUpdateError("agent credit telemetry axes disagree")
        if any(not math.isfinite(value) for values in vectors for value in values):
            raise RecurrentPPOUpdateError("agent credit telemetry contains NaN or Inf")
        if not 0.0 <= self.same_timestep_advantage_equality_rate <= 1.0:
            raise RecurrentPPOUpdateError(
                "same-timestep advantage equality rate must lie in [0, 1]"
            )
        if set(self.route_category_agent_advantage_mean) != {
            "local", "remote", "defer"
        }:
            raise RecurrentPPOUpdateError("route category telemetry is incomplete")
        for values in self.route_category_agent_advantage_mean.values():
            if len(values) != agent_count or any(
                value is not None and not math.isfinite(value) for value in values
            ):
                raise RecurrentPPOUpdateError(
                    "route category per-agent advantage telemetry is invalid"
                )


def _collect_agent_credit_epoch_telemetry(
    *,
    current_value: Tensor,
    td_residual: Tensor,
    advantage: Tensor,
    return_target: Tensor,
    sequence_valid_mask: Tensor,
    route_telemetry: RouteTelemetry | None,
    per_head_critic_loss: tuple[float, ...],
    same_timestep_advantage_equality_rate: float | None,
) -> AgentCreditPPOEpochTelemetry:
    tensors = (current_value, td_residual, advantage, return_target)
    if any(not isinstance(tensor, Tensor) or tensor.ndim != 3 for tensor in tensors):
        raise RecurrentPPOUpdateError(
            "role-decomposed telemetry requires [B,L,A] values"
        )
    shape = current_value.shape
    if any(tensor.shape != shape for tensor in tensors[1:]):
        raise RecurrentPPOUpdateError("agent credit telemetry value axes disagree")
    if sequence_valid_mask.shape != shape[:2] or sequence_valid_mask.dtype != torch.bool:
        raise RecurrentPPOUpdateError("agent credit telemetry mask is invalid")
    agent_count = shape[-1]

    def per_agent_mean(tensor: Tensor) -> tuple[float, ...]:
        detached = tensor.detach()
        return tuple(
            float(
                detached[..., agent_index]
                .masked_select(sequence_valid_mask)
                .mean()
                .cpu()
                .item()
            )
            for agent_index in range(agent_count)
        )

    category_values: dict[str, list[list[float]]] = {
        category: [[] for _ in range(agent_count)]
        for category in ("local", "remote", "defer")
    }
    if route_telemetry is not None:
        for sample in route_telemetry.samples:
            category_values[sample.category][sample.agent_index].append(
                sample.advantage
            )
    category_means = {
        category: tuple(
            None if not values else math.fsum(values) / len(values)
            for values in rows
        )
        for category, rows in category_values.items()
    }
    if same_timestep_advantage_equality_rate is None:
        raise RecurrentPPOUpdateError(
            "role-decomposed objective did not report advantage equality"
        )
    return AgentCreditPPOEpochTelemetry(
        per_agent_value_mean=per_agent_mean(current_value),
        per_agent_td_residual_mean=per_agent_mean(td_residual),
        per_agent_advantage_mean=per_agent_mean(advantage),
        per_agent_return_mean=per_agent_mean(return_target),
        same_timestep_advantage_equality_rate=(
            same_timestep_advantage_equality_rate
        ),
        route_category_agent_advantage_mean=category_means,
        per_head_critic_loss=per_head_critic_loss,
    )


@dataclass(frozen=True)
class RecurrentPPOEpochDiagnostics:
    """Detached diagnostics for one of the four fixed PPO epochs."""

    epoch_index: int
    actor_loss: float
    critic_loss: float
    entropy_mean: float
    total_loss: float
    ratio_mean: float
    actor_grad_norm_before_clip: float
    critic_grad_norm_before_clip: float
    clip_max_norm: float
    approx_kl: float | None = None
    clip_fraction: float | None = None
    route_entropy_coefficient: float | None = None
    route_entropy_schedule_progress: float | None = None
    route_entropy_loss_contribution: float | None = None
    other_branch_entropy_loss_contribution: float | None = None
    global_entropy_loss_contribution: float | None = None
    collected_environment_steps: int | None = None
    route_telemetry: RouteTelemetry | None = None
    agent_credit_telemetry: AgentCreditPPOEpochTelemetry | None = None
    route_choice_stability: RouteChoiceStabilityTelemetry | None = None

    def __post_init__(self) -> None:
        if self.epoch_index not in range(4):
            raise RecurrentPPOUpdateError("epoch_index must lie in [0, 3]")
        values = (
            self.actor_loss,
            self.critic_loss,
            self.entropy_mean,
            self.total_loss,
            self.ratio_mean,
            self.actor_grad_norm_before_clip,
            self.critic_grad_norm_before_clip,
            self.clip_max_norm,
        )
        if not all(math.isfinite(value) for value in values):
            raise RecurrentPPOUpdateError("update diagnostics contain NaN or Inf")
        if self.approx_kl is not None and not math.isfinite(self.approx_kl):
            raise RecurrentPPOUpdateError("approx_kl must be finite or NA")
        if self.clip_fraction is not None and not (
            math.isfinite(self.clip_fraction) and 0.0 <= self.clip_fraction <= 1.0
        ):
            raise RecurrentPPOUpdateError("clip_fraction must lie in [0, 1] or be NA")
        entropy_schedule_values = (
            self.route_entropy_coefficient,
            self.route_entropy_schedule_progress,
            self.route_entropy_loss_contribution,
            self.other_branch_entropy_loss_contribution,
            self.global_entropy_loss_contribution,
            self.collected_environment_steps,
        )
        present_count = sum(value is not None for value in entropy_schedule_values)
        if present_count not in (0, len(entropy_schedule_values)):
            raise RecurrentPPOUpdateError(
                "route entropy schedule telemetry must be all present or all NA"
            )
        if present_count:
            assert self.route_entropy_coefficient is not None
            assert self.route_entropy_schedule_progress is not None
            assert self.route_entropy_loss_contribution is not None
            assert self.other_branch_entropy_loss_contribution is not None
            assert self.global_entropy_loss_contribution is not None
            assert self.collected_environment_steps is not None
            scalar_values = (
                self.route_entropy_coefficient,
                self.route_entropy_schedule_progress,
                self.route_entropy_loss_contribution,
                self.other_branch_entropy_loss_contribution,
                self.global_entropy_loss_contribution,
            )
            if not all(math.isfinite(value) for value in scalar_values):
                raise RecurrentPPOUpdateError(
                    "route entropy schedule telemetry contains NaN or Inf"
                )
            if (
                self.route_entropy_coefficient < 0.0
                or self.route_entropy_loss_contribution < -1.0e-8
                or self.other_branch_entropy_loss_contribution < -1.0e-8
                or self.global_entropy_loss_contribution < -1.0e-8
            ):
                raise RecurrentPPOUpdateError(
                    "route entropy coefficients and contributions must be non-negative"
                )
            if not 0.0 <= self.route_entropy_schedule_progress <= 1.0:
                raise RecurrentPPOUpdateError(
                    "route entropy schedule progress must lie in [0, 1]"
                )
            if (
                isinstance(self.collected_environment_steps, bool)
                or not isinstance(self.collected_environment_steps, int)
                or self.collected_environment_steps < 0
            ):
                raise RecurrentPPOUpdateError(
                    "collected environment steps must be a non-negative integer"
                )
            if not math.isclose(
                self.global_entropy_loss_contribution,
                self.route_entropy_loss_contribution
                + self.other_branch_entropy_loss_contribution,
                rel_tol=1.0e-7,
                abs_tol=1.0e-9,
            ):
                raise RecurrentPPOUpdateError(
                    "global entropy contribution must equal route plus other branches"
                )
        if self.route_telemetry is not None and not isinstance(
            self.route_telemetry, RouteTelemetry
        ):
            raise TypeError("route_telemetry must be RouteTelemetry or None")
        if self.agent_credit_telemetry is not None and not isinstance(
            self.agent_credit_telemetry, AgentCreditPPOEpochTelemetry
        ):
            raise TypeError(
                "agent_credit_telemetry must be AgentCreditPPOEpochTelemetry or None"
            )
        if self.route_choice_stability is not None and not isinstance(
            self.route_choice_stability, RouteChoiceStabilityTelemetry
        ):
            raise TypeError(
                "route_choice_stability must be RouteChoiceStabilityTelemetry or None"
            )


@dataclass(frozen=True)
class RecurrentPPOUpdateOutput:
    """Detached summary of one completed four-epoch full-rollout update."""

    epoch_diagnostics: tuple[RecurrentPPOEpochDiagnostics, ...]
    chunk_count: int
    chunk_length: int
    valid_transition_count: int
    old_policy_snapshot_preserved: bool

    def __post_init__(self) -> None:
        if len(self.epoch_diagnostics) != 4:
            raise RecurrentPPOUpdateError("exactly four epoch diagnostics are required")
        if (self.chunk_count, self.chunk_length, self.valid_transition_count) != (
            8,
            32,
            256,
        ):
            raise RecurrentPPOUpdateError("update output geometry differs from 8x32")
        if not self.old_policy_snapshot_preserved:
            raise RecurrentPPOUpdateError("old policy snapshot changed during update")


class CAGATMAPPORecurrentPPOUpdater:
    """Apply the frozen recurrent PPO update to one finalized full rollout."""

    def __init__(
        self,
        actor: CAGATMAPPOActor,
        critic: MAPPOCentralizedCritic,
        config: RunConfig,
        *,
        optimizers: CAGATMAPPOOptimizerBundle | None = None,
        action_distribution: CAGATMAPPOActionDistribution | None = None,
        route_telemetry_enabled: bool = True,
    ) -> None:
        _validate_frozen_contract(config)
        if not isinstance(actor, CAGATMAPPOActor):
            raise TypeError("actor must be a CAGATMAPPOActor")
        if not isinstance(critic, MAPPOCentralizedCritic):
            raise TypeError("critic must be a MAPPOCentralizedCritic")
        expected_spec = MAPPOTensorSpec.from_config(config)
        if actor.spec != expected_spec or critic.spec != expected_spec:
            raise RecurrentPPOUpdateError("actor, critic, and config specs differ")
        if (hasattr(critic, "tx_upper_q")
                != (config.training.mappo.tx_upper_credit_mode == "counterfactual_q_v1")):
            raise RecurrentPPOUpdateError("TX Q critic presence differs from configured credit mode")
        if (hasattr(getattr(critic, "tx_upper_q", None), "tx_q_target_norm")
                != (config.training.mappo.tx_q_target_normalization == "popart_v1")):
            raise RecurrentPPOUpdateError("TX Q normalizer differs from configured mode")
        actor_device, actor_dtype = _module_device_dtype(actor, "actor")
        critic_device, critic_dtype = _module_device_dtype(critic, "critic")
        if actor_device != critic_device or actor_dtype != critic_dtype:
            raise RecurrentPPOUpdateError("actor and critic must share one training device")
        self.actor = actor
        self.critic = critic
        self.config = config
        self.device = actor_device
        self.dtype = actor_dtype
        self.optimizers = (
            build_ca_gat_mappo_optimizers(actor, critic, config)
            if optimizers is None
            else optimizers
        )
        if not isinstance(self.optimizers, CAGATMAPPOOptimizerBundle):
            raise TypeError("optimizers must be a CAGATMAPPOOptimizerBundle")
        self.optimizers.validate(actor, critic, config)
        self.action_distribution = (
            CAGATMAPPOActionDistribution(actor, config)
            if action_distribution is None
            else action_distribution
        )
        if not isinstance(self.action_distribution, CAGATMAPPOActionDistribution):
            raise TypeError("action_distribution must be CAGATMAPPOActionDistribution")
        if self.action_distribution.actor is not actor:
            raise RecurrentPPOUpdateError("action distribution must own the updated actor")
        if not isinstance(route_telemetry_enabled, bool):
            raise TypeError("route_telemetry_enabled must be boolean")
        self.route_telemetry_enabled = route_telemetry_enabled
        self.conditional_credit_sink = None

    def _evaluate_prepared(
        self,
        minibatch: _DeviceRecurrentPPOMinibatch,
        *,
        return_route_candidates: bool = False,
    ) -> RecurrentPPOEvaluation:
        policy = self.action_distribution.evaluate_actions(
            minibatch.actor_batch,
            minibatch.action_mask_batch,
            minibatch.proposals,
            minibatch.initial_hidden,
            **({"return_route_candidates": True} if return_route_candidates else {}),
        )
        for branch in ACTION_BRANCH_ORDER:
            if not torch.equal(
                policy.action_masks[branch], minibatch.actor_batch.action_masks[branch]
            ):
                raise RecurrentPPOUpdateError(
                    f"re-evaluated {branch} mask differs from rollout snapshot"
                )
        active = torch.stack(
            [policy.active_branches[branch] for branch in ACTION_BRANCH_ORDER], dim=-1
        )
        if not torch.equal(active, minibatch.active_branch_indicators):
            raise RecurrentPPOUpdateError(
                "re-evaluated active branches differ from rollout snapshot"
            )
        if policy.proposals != minibatch.proposals:
            raise RecurrentPPOUpdateError("proposal re-evaluation changed stored proposals")
        raw_value = self.critic(minibatch.centralized_batch)
        credit_mode = AgentCreditMode(
            self.config.training.mappo.agent_credit_mode
        )
        current_value = (
            raw_value.squeeze(-1)
            if credit_mode is AgentCreditMode.TEAM
            else raw_value
        )
        return RecurrentPPOEvaluation(
            policy=policy,
            current_value=current_value,
            agent_credit_mode=credit_mode.value,
        )

    def evaluate_minibatch(
        self,
        minibatch: RecurrentPPOMinibatch,
    ) -> RecurrentPPOEvaluation:
        """Synchronously transfer and deterministically re-evaluate one minibatch."""

        if not isinstance(minibatch, RecurrentPPOMinibatch):
            raise TypeError("minibatch must be a RecurrentPPOMinibatch")
        prepared = _prepare_device_minibatch(minibatch, self.device, self.dtype)
        return self._evaluate_prepared(prepared)

    @staticmethod
    def _finite_grad_norm(
        parameters: tuple[nn.Parameter, ...],
        max_norm: float,
        name: str,
    ) -> float:
        try:
            norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=max_norm,
                error_if_nonfinite=True,
            )
        except RuntimeError as error:
            raise RecurrentPPOUpdateError(
                f"{name} gradient norm is NaN or Inf"
            ) from error
        value = float(norm.detach().cpu().item())
        if not math.isfinite(value):
            raise RecurrentPPOUpdateError(f"{name} gradient norm is NaN or Inf")
        return value

    def update(
        self,
        buffer: CAGATMAPPORolloutBuffer,
        *,
        collected_environment_steps: int | None = None,
    ) -> RecurrentPPOUpdateOutput:
        """Run exactly four epochs over the sole deterministic 8-by-32 minibatch."""
        if collected_environment_steps is not None and (
            isinstance(collected_environment_steps, bool)
            or not isinstance(collected_environment_steps, int)
            or collected_environment_steps < 0
        ):
            raise RecurrentPPOUpdateError(
                "collected_environment_steps must be a non-negative integer"
            )
        if (
            (self.config.training.mappo.route_entropy_step_schedule_enabled
             or self.config.training.mappo.route_exploration_mode != "off"
             or self.config.training.mappo.tx_upper_optimization_mode == "deterministic_stability_v1")
            and collected_environment_steps is None
        ):
            raise RecurrentPPOUpdateError(
                "enabled entropy schedule requires collected environment steps"
            )

        minibatch = build_recurrent_ppo_minibatch(buffer, self.config)
        prepared = _prepare_device_minibatch(minibatch, self.device, self.dtype)
        fixed_targets: dict[str, Tensor] = {
            "old_joint_log_prob": prepared.old_joint_log_prob.detach().clone(),
            "old_value": prepared.old_value.detach().clone(),
            "advantage": prepared.advantage.detach().clone(),
            "return_target": prepared.return_target.detach().clone(),
        }
        if prepared.route_advantage is not None:
            fixed_targets["route_advantage"] = (
                prepared.route_advantage.detach().clone()
            )
        mappo = self.config.training.mappo
        from .integrated_scheduler import policy_component_names, rollout_policy_components, compact_scheduler_summary
        integrated = mappo.tx_upper_decoder_mode != "legacy" or mappo.cpu_scheduler_mode != "legacy_hierarchical"
        logical_old = logical_active = None
        if integrated:
            logical_old, logical_active = rollout_policy_components(buffer, self.config, self.device)
        task_credit = None
        route_advantage = prepared.route_advantage
        self.last_route_task_records = []
        self.last_route_task_value_losses = []
        route_q = None
        self.last_route_option_q_records = []
        self.last_route_option_q_losses = []
        self.last_destination_records = []
        self.last_destination_gradient_probes = []
        if mappo.route_credit_mode == "task_event_value_v1":
            from .route_task_credit import prepare_task_credit, task_value_loss, epoch_task_records
            task_credit = prepare_task_credit(buffer, prepared.advantage, self.config)
            if mappo.route_value_mode == "candidate_counterfactual_q_v1":
                from .route_option_q import prepare_route_q, route_q_loss, epoch_q_records
                with torch.no_grad():
                    frozen_policy = self._evaluate_prepared(
                        prepared, return_route_candidates=True).policy
                    route_q = prepare_route_q(
                        self.critic, frozen_policy, prepared.centralized_batch.features,
                        task_credit, mappo.route_counterfactual_credit_coefficient,
                        actor_credit_source=mappo.route_actor_credit_source)
                # Selector changes only actor credit; retain both auxiliary losses.
                task_credit.advantage = route_q.advantage
                task_credit.samples = route_q.credit_samples
            route_advantage = task_credit.advantage.reshape_as(prepared.advantage).to(prepared.advantage)
            fixed_targets["task_route_advantage"] = route_advantage.detach().clone()
        summaries = []
        self.last_route_exploration_summary = []
        tx_credit = None
        q_target = None
        counterfactual = mappo.tx_upper_credit_mode == "counterfactual_q_v1"
        factual = mappo.tx_upper_credit_mode == "factual_team_gae_v1"
        paired = mappo.tx_upper_credit_mode == "paired_shadow_v1"
        paired_samples = None
        if paired:
            from .tx_paired_shadow import credit_from_buffer
            tx_credit, paired_samples = credit_from_buffer(
                buffer, self.config, self.device,
                logical_active[..., policy_component_names(self.config).index("tx_upper")],
                prepared.sequence_valid_mask)
            paired_snapshot = (tx_credit.normalized_advantage.clone(), tx_credit.active.clone())
        factual_snapshot = None
        if factual:
            from .tx_upper_factual_credit import factual_team_credit
            tx_credit = factual_team_credit(
                prepared.return_target, prepared.old_value, mappo.agent_credit_mode,
                logical_active[..., policy_component_names(self.config).index("tx_upper")],
                prepared.sequence_valid_mask,
            )
            factual_snapshot = tx_credit.normalized_advantage.clone()
        if counterfactual:
            from .tx_upper_counterfactual import (
                counterfactual_credit, team_return_target,
                selected_action_q_loss, q_regression_telemetry,
            )
            q_target = team_return_target(prepared.return_target, self.actor.spec.uav_count).clone()
        normalized_q_target = None
        tx_norm_summary = None
        if mappo.tx_q_target_normalization == "popart_v1":
            tx_active = logical_active[..., policy_component_names(self.config).index("tx_upper")] & prepared.sequence_valid_mask.unsqueeze(-1)
            self.critic.tx_upper_q.update_target_statistics(q_target, tx_active)
            norm = self.critic.tx_upper_q.tx_q_target_norm
            normalized_q_target = norm.normalize(q_target).detach()
            tx_norm_summary = norm.target_telemetry(q_target, tx_active)
        epoch_diagnostics: list[RecurrentPPOEpochDiagnostics] = []
        for epoch_index in range(mappo.update_epochs):
            self.optimizers.actor_optimizer.zero_grad(
                set_to_none=mappo.zero_grad_set_to_none
            )
            self.optimizers.critic_optimizer.zero_grad(
                set_to_none=mappo.zero_grad_set_to_none
            )
            evaluation = self._evaluate_prepared(prepared)
            new_branch_log_probs = None
            if ActorRatioMode(mappo.actor_ratio_mode) is ActorRatioMode.BRANCH_SPECIFIC:
                new_branch_log_probs = torch.stack(
                    [
                        evaluation.policy.branch_log_probs[branch]
                        for branch in ACTION_BRANCH_ORDER
                    ],
                    dim=-1,
                )
            if integrated:
                new_branch_log_probs = torch.stack(list(evaluation.policy.component_log_probs.values()), -1)
                current_active = torch.stack(list(evaluation.policy.component_active.values()), -1)
                if not torch.equal(current_active, logical_active):
                    raise RecurrentPPOUpdateError("logical component activity changed on recurrent replay")
            tx_stability_inputs = {}
            tx_q_loss = None
            current_q = None
            if factual:
                tx_stability_inputs["tx_upper_advantage"] = tx_credit.normalized_advantage
            if paired:
                tx_stability_inputs.update(tx_upper_advantage=tx_credit.normalized_advantage,
                                           tx_upper_paired_mask=tx_credit.active)
            if counterfactual:
                if normalized_q_target is None:
                    current_q = self.critic.tx_upper_q(prepared.centralized_batch.features)
                else:
                    normalized_q = self.critic.tx_upper_q.forward_normalized(prepared.centralized_batch.features)
                    current_q = norm.denormalize(normalized_q)
                if tx_credit is None:
                    # Epoch 0 is the old on-policy actor before any Adam step.
                    # Q, baseline probabilities and A stay fixed for all epochs.
                    p_transmit = evaluation.policy.scheduler_metrics["tx_p_transmit"]
                    upper_p = torch.stack((1 - p_transmit, p_transmit), -1)
                    tx_credit = counterfactual_credit(
                        current_q, upper_p,
                        evaluation.policy.scheduler_metrics["tx_sampled_transmit"].bool(),
                        evaluation.policy.component_active["tx_upper"] & prepared.sequence_valid_mask.unsqueeze(-1),
                    )
                tx_stability_inputs["tx_upper_advantage"] = tx_credit.normalized_advantage
                tx_q_loss = selected_action_q_loss(
                    current_q if normalized_q_target is None else normalized_q,
                    tx_credit.transmitted,
                    q_target if normalized_q_target is None else normalized_q_target,
                    tx_credit.active
                )
            if mappo.tx_upper_optimization_mode == "deterministic_stability_v1":
                metrics = evaluation.policy.scheduler_metrics
                tx_stability_inputs = {
                    "tx_upper_logit_gap": metrics["tx_upper_logit_gap"],
                    "tx_upper_transmitted": metrics["tx_sampled_transmit"],
                    "tx_upper_entropy": evaluation.policy.component_entropies["tx_upper"],
                    "entropy_without_tx_upper": metrics["entropy_without_tx_upper"],
                }
            loss = compute_configured_batched_ppo_objective_and_loss(
                new_joint_log_prob=evaluation.policy.joint_log_prob,
                old_joint_log_prob=prepared.old_joint_log_prob,
                advantage=prepared.advantage,
                current_value=evaluation.current_value,
                return_target=prepared.return_target,
                entropy=evaluation.policy.entropy,
                sequence_valid_mask=prepared.sequence_valid_mask,
                config=self.config,
                new_branch_log_probs=new_branch_log_probs,
                old_branch_log_probs=(logical_old if integrated else prepared.old_branch_log_probs),
                active_branch_indicators=(logical_active if integrated else prepared.active_branch_indicators),
                route_entropy=evaluation.policy.branch_entropies["route"],
                collected_environment_steps=collected_environment_steps,
                route_advantage=route_advantage,
                **tx_stability_inputs,
            )
            if integrated:
                summaries.append(compact_scheduler_summary(evaluation.policy, loss, self.config))
                if paired:
                    tx_summary = summaries[-1]["tx_upper"]
                    tx_summary.update(tx_credit.telemetry())
                    active = tx_credit.active
                    tx_index = policy_component_names(self.config).index("tx_upper")
                    paired_surrogate = loss.flattened.branch_surrogate.reshape(
                        *active.shape, len(policy_component_names(self.config)))[..., tx_index].detach()[active]
                    tx_summary["ppo_loss"] = float(-paired_surrogate.mean()) if paired_surrogate.numel() else None
                    tx_summary["surrogate_active_samples"] = int(active.sum())
                    tx_summary["ppo_loss_cohort"] = "paired_only"
                    lr = (new_branch_log_probs[..., tx_index] - logical_old[..., tx_index]).detach()[active]
                    ratio = lr.exp()
                    tx_summary["approx_kl"] = float(((ratio - 1) - lr).mean()) if lr.numel() else None
                    tx_summary["clip_fraction"] = float(((ratio - 1).abs() > mappo.ppo_clip_epsilon).float().mean()) if lr.numel() else None
                    all_active = logical_active[..., tx_index] & prepared.sequence_valid_mask.unsqueeze(-1)
                    ps = evaluation.policy.scheduler_metrics["tx_p_transmit"].detach()[all_active].float()
                    tx_summary["p_transmit_p10_p50_p90"] = torch.quantile(ps, ps.new_tensor([.1,.5,.9])).cpu().tolist() if ps.numel() else None
                    tx_summary["policy_statistics_timing"] = "pre_step"
                if factual:
                    tx_summary = summaries[-1]["tx_upper"]
                    tx_summary.update(tx_credit.telemetry(
                        evaluation.policy.scheduler_metrics["tx_sampled_transmit"].bool()
                    ))
                    active = tx_credit.active
                    tx_index = policy_component_names(self.config).index("tx_upper")
                    log_ratio = (new_branch_log_probs[..., tx_index]
                                 - logical_old[..., tx_index]).detach()[active]
                    ratio = log_ratio.exp()
                    tx_summary["approx_kl"] = float(((ratio - 1) - log_ratio).mean()) if ratio.numel() else None
                    tx_summary["clip_fraction"] = float(((ratio - 1).abs() > mappo.ppo_clip_epsilon).float().mean()) if ratio.numel() else None
                    tx_summary["policy_statistics_timing"] = "pre_step"
                if counterfactual:
                    summaries[-1]["tx_upper"].update(tx_credit.telemetry())
                    summaries[-1]["tx_upper"].update(q_regression_telemetry(
                        current_q, tx_credit.transmitted, q_target, tx_credit.active, tx_q_loss
                    ))
                    if tx_norm_summary is not None:
                        selected = normalized_q.detach()[tx_credit.active].gather(
                            -1, tx_credit.transmitted[tx_credit.active].long().unsqueeze(-1)
                        ).squeeze(-1)
                        summaries[-1]["tx_upper"].update(tx_norm_summary)
                        summaries[-1]["tx_upper"].update({
                            "normalized_selected_q_mean": float(selected.mean()) if selected.numel() else None,
                            "normalized_selected_q_std": float(selected.std(unbiased=False)) if selected.numel() else None,
                        })
            stability_loss = None
            optimized_actor_loss = loss.actor_loss
            optimized_total_loss = loss.total_loss
            from .destination_pretraining import online_destination_enabled
            if online_destination_enabled(self.config):
                from .destination_ranking import ranking_batch_loss
                route_index = policy_component_names(self.config).index('route')
                divisor = int((logical_active.sum((0,1,2)) > 0).sum())
                rank_loss, rank_records = ranking_batch_loss(
                    self.actor, evaluation.policy, buffer, epoch_index,
                    loss.branch_surrogate[...,route_index], logical_active[...,route_index],
                    divisor, mappo.destination_oracle_rank_coefficient)
                weighted_rank = mappo.destination_oracle_rank_coefficient * rank_loss
                optimized_actor_loss = optimized_actor_loss + weighted_rank
                optimized_total_loss = optimized_total_loss + weighted_rank
                self.last_destination_records.extend(rank_records)
            if task_credit is not None:
                task_loss = task_value_loss(self.critic, prepared.centralized_batch.features, task_credit)
                optimized_total_loss = optimized_total_loss + mappo.route_task_value_coefficient * task_loss
                self.last_route_task_value_losses.append(float(task_loss.detach()))
                self.last_route_task_records.extend(epoch_task_records(
                    task_credit, evaluation.policy,
                    (logical_old if integrated else prepared.old_branch_log_probs)[..., 0], epoch_index,
                    mappo.ppo_clip_epsilon))
            if route_q is not None:
                q_loss, current_q = route_q_loss(self.critic, route_q)
                optimized_total_loss = optimized_total_loss + mappo.route_q_value_loss_coefficient * q_loss
                self.last_route_option_q_losses.append(float(q_loss.detach()))
                self.last_route_option_q_records.extend(epoch_q_records(
                    route_q, current_q, self.last_route_task_records[-len(task_credit.samples):]
                    if task_credit.samples else []))
            if mappo.route_exploration_mode != "off":
                from .route_exploration import route_exploration_loss
                route_regularizer = route_exploration_loss(
                    evaluation.policy, mappo, collected_environment_steps,
                    prepared.sequence_valid_mask)
                self.last_route_exploration_summary.append({
                    "epoch_index": epoch_index, **route_regularizer.record()})
                if route_regularizer.coefficient != 0.0:
                    optimized_actor_loss = optimized_actor_loss + route_regularizer.scaled_loss
                    optimized_total_loss = optimized_total_loss + route_regularizer.scaled_loss
            if tx_q_loss is not None:
                optimized_total_loss = optimized_total_loss + mappo.value_coefficient * tx_q_loss
            if loss.flattened.tx_upper_stability is not None:
                from .tx_upper_stability import TXUpperDeterministicStabilityPPOV1
                optimized_actor_loss = optimized_actor_loss + (
                    TXUpperDeterministicStabilityPPOV1.MARGIN_WEIGHT
                    * loss.flattened.tx_upper_stability.margin_loss
                    / loss.flattened.tx_upper_objective_divisor)
            if mappo.route_choice_stability_enabled:
                local_domain_mask, remote_domain_mask = (
                    _route_semantic_domain_masks(
                        prepared.action_mask_batch,
                        evaluation.policy.raw_logits["route"],
                    )
                )
                stability_loss = compute_route_choice_stability_loss(
                    route_logits=evaluation.policy.raw_logits["route"],
                    route_action_masks=evaluation.policy.action_masks["route"],
                    route_active=evaluation.policy.active_branches["route"],
                    sequence_valid_mask=prepared.sequence_valid_mask,
                    local_domain_mask=local_domain_mask,
                    remote_domain_mask=remote_domain_mask,
                    selected_route_indices=evaluation.policy.action_indices["route"],
                    entropy_floor_nats=mappo.route_choice_entropy_floor_nats,
                    coefficient=mappo.route_choice_stability_coef,
                )
                if mappo.route_choice_stability_coef != 0.0:
                    optimized_actor_loss = optimized_actor_loss + stability_loss.scaled_loss
                    optimized_total_loss = optimized_total_loss + stability_loss.scaled_loss
            if online_destination_enabled(self.config):
                from .destination_ranking import destination_ppo_contribution, destination_gradient_probe
                dest_ppo = destination_ppo_contribution(
                    self.actor, evaluation.policy, logical_old[...,route_index],
                    route_advantage if route_advantage is not None else prepared.advantage,
                    logical_active[...,route_index], divisor, mappo.ppo_clip_epsilon)
                probe = destination_gradient_probe(
                    self.actor, dest_ppo, rank_loss, mappo.destination_oracle_rank_coefficient,
                    mappo.gradient_clip_norm, base_total_loss=optimized_total_loss-weighted_rank)
                self.last_destination_gradient_probes.append({'epoch_index':epoch_index,**probe})
                for row in rank_records:
                    row['gradient_probe']=probe
            optimized_total_loss.backward()
            route_telemetry = None
            if self.route_telemetry_enabled:
                route_telemetry = collect_route_telemetry(
                    policy=evaluation.policy,
                    action_mask_batch=prepared.action_mask_batch,
                    proposals=prepared.proposals,
                    advantage=(
                        route_advantage
                        if route_advantage is not None
                        else prepared.advantage
                    ),
                    return_target=prepared.return_target,
                    sequence_valid_mask=prepared.sequence_valid_mask,
                    route_head=self.actor.action_heads["route"],
                    old_branch_log_probs=prepared.old_branch_log_probs,
                    td_residual=prepared.td_residual,
                    shared_trunk=self.actor,
                    epsilon_clip=self.config.training.mappo.ppo_clip_epsilon,
                    actor_ratio_mode=mappo.actor_ratio_mode,
                    base_advantage=prepared.advantage,
                    route_advantage=route_advantage,
                    branch_advantage=loss.branch_advantage,
                    bootstrap_mask=prepared.bootstrap_mask,
                    effective_route_gae_lambda=(
                        mappo.route_gae_lambda
                        if RouteCreditMode(mappo.route_credit_mode)
                        is RouteCreditMode.ROUTE_SPECIFIC_GAE
                        else (
                            None
                            if RouteCreditMode(mappo.route_credit_mode)
                            is RouteCreditMode.ROLLOUT_CAPPED_ROUTE_EVENT_NSTEP
                            else mappo.gae_lambda
                        )
                    ),
                    route_nstep_samples=prepared.route_nstep_samples,
                )
            route_choice_stability = None
            if stability_loss is not None:
                route_choice_stability = _route_choice_stability_telemetry(
                    stability_loss,
                    self.actor.action_heads["route"],
                    prepared.action_mask_batch,
                )
            agent_credit_telemetry = None
            if (
                AgentCreditMode(mappo.agent_credit_mode)
                is AgentCreditMode.ROLE_DECOMPOSED
            ):
                if prepared.td_residual is None:
                    raise RecurrentPPOUpdateError(
                        "role-decomposed update requires per-agent TD residual"
                    )
                agent_credit_telemetry = _collect_agent_credit_epoch_telemetry(
                    current_value=evaluation.current_value,
                    td_residual=prepared.td_residual,
                    advantage=prepared.advantage,
                    return_target=prepared.return_target,
                    sequence_valid_mask=prepared.sequence_valid_mask,
                    route_telemetry=route_telemetry,
                    per_head_critic_loss=loss.diagnostics.per_head_critic_loss,
                    same_timestep_advantage_equality_rate=(
                        loss.diagnostics.same_timestep_advantage_equality_rate
                    ),
                )
            if factual or paired:
                tx_parameters = tuple(self.actor.tx_upper_private.parameters())
                tx_before = tuple(p.detach().clone() for p in tx_parameters)
                tx_grad_squared = sum(
                    (p.grad.detach().square().sum() for p in tx_parameters if p.grad is not None),
                    prepared.return_target.new_zeros(()),
                )
                summaries[-1]["tx_upper"]["tx_grad_norm_before_clip"] = float(tx_grad_squared.sqrt())
            actor_grad_norm = self._finite_grad_norm(
                self.optimizers.actor_parameters,
                mappo.gradient_clip_norm,
                "actor",
            )
            critic_grad_norm = self._finite_grad_norm(
                (tuple(self.critic.value_network.parameters())
                 if counterfactual or task_credit is not None else self.optimizers.critic_parameters),
                mappo.gradient_clip_norm,
                "critic",
            )
            if counterfactual:
                # Independent clipping preserves the original V critic update.
                q_grad_norm = self._finite_grad_norm(
                    tuple(self.critic.tx_upper_q.parameters()), mappo.gradient_clip_norm, "TX Q critic"
                )
                summaries[-1]["tx_upper"]["q_critic_grad_norm_before_clip"] = q_grad_norm
            if task_credit is not None:
                self._finite_grad_norm(tuple(self.critic.route_task_value.parameters()),
                                       mappo.gradient_clip_norm, "Route task value")
            if route_q is not None:
                self._finite_grad_norm(tuple(self.critic.route_option_q.parameters()),
                                       mappo.gradient_clip_norm, "Route option Q")
            self.optimizers.actor_optimizer.step()
            self.optimizers.critic_optimizer.step()
            if factual or paired:
                update_squared = sum(
                    ((p.detach() - old).square().sum() for p, old in zip(tx_parameters, tx_before)),
                    prepared.return_target.new_zeros(()),
                )
                summaries[-1]["tx_upper"]["tx_update_norm"] = float(update_squared.sqrt())
                if factual and not torch.equal(tx_credit.normalized_advantage, factual_snapshot):
                    raise RecurrentPPOUpdateError("factual TX credit changed during PPO epochs")
                if paired and (not torch.equal(tx_credit.active, paired_snapshot[1]) or not torch.equal(
                        tx_credit.normalized_advantage[tx_credit.active], paired_snapshot[0][tx_credit.active])):
                    raise RecurrentPPOUpdateError("paired TX credit/mask changed during PPO epochs")
            if self.conditional_credit_sink is not None and not integrated:
                self.conditional_credit_sink.record_epoch(
                    epoch_index=epoch_index,
                    prepared=prepared,
                    evaluation=evaluation,
                    loss=loss,
                    actor=self.actor,
                    actor_optimizer=self.optimizers.actor_optimizer,
                )
            for name, expected in fixed_targets.items():
                actual = route_advantage if name == "task_route_advantage" else getattr(prepared, name)
                if not torch.equal(actual, expected):
                    raise RecurrentPPOUpdateError(
                        f"fixed rollout target {name} changed during PPO epochs"
                    )
            epoch_diagnostics.append(
                RecurrentPPOEpochDiagnostics(
                    epoch_index=epoch_index,
                    actor_loss=float(optimized_actor_loss.detach().cpu().item()),
                    critic_loss=float(loss.critic_loss.detach().cpu().item()),
                    entropy_mean=float(loss.entropy_mean.detach().cpu().item()),
                    total_loss=float(optimized_total_loss.detach().cpu().item()),
                    ratio_mean=loss.diagnostics.ratio_mean,
                    actor_grad_norm_before_clip=actor_grad_norm,
                    critic_grad_norm_before_clip=critic_grad_norm,
                    clip_max_norm=mappo.gradient_clip_norm,
                    approx_kl=loss.diagnostics.approx_kl,
                    clip_fraction=loss.diagnostics.clipped_fraction,
                    route_entropy_coefficient=(
                        None
                        if collected_environment_steps is None
                        else loss.route_entropy_coefficient
                    ),
                    route_entropy_schedule_progress=(
                        None
                        if collected_environment_steps is None
                        else loss.route_entropy_schedule_progress
                    ),
                    route_entropy_loss_contribution=(
                        None
                        if collected_environment_steps is None
                        else float(
                            loss.route_entropy_loss_contribution.detach().cpu().item()
                        )
                    ),
                    other_branch_entropy_loss_contribution=(
                        None
                        if collected_environment_steps is None
                        else float(
                            loss.other_branch_entropy_loss_contribution.detach()
                            .cpu()
                            .item()
                        )
                    ),
                    global_entropy_loss_contribution=(
                        None
                        if collected_environment_steps is None
                        else float(
                            loss.global_entropy_loss_contribution.detach().cpu().item()
                        )
                    ),
                    collected_environment_steps=collected_environment_steps,
                    route_telemetry=route_telemetry,
                    agent_credit_telemetry=agent_credit_telemetry,
                    route_choice_stability=route_choice_stability,
                )
            )
        self.last_integrated_summary = ({"schema": "integrated_scheduler_update.v1", "ppo_epochs": summaries}
                                        if integrated else None)
        if paired:
            self.last_integrated_summary["paired_samples"] = paired_samples
            self.last_integrated_summary["paired_shadow_steps"] = sum(row["shadow_steps"] for row in paired_samples)
            self.last_integrated_summary["paired_shadow_wall_seconds"] = sum(row.get("shadow_wall_seconds", 0.) for row in paired_samples)
        return RecurrentPPOUpdateOutput(
            epoch_diagnostics=tuple(epoch_diagnostics),
            chunk_count=8,
            chunk_length=32,
            valid_transition_count=256,
            old_policy_snapshot_preserved=True,
        )


__all__ = [
    "AgentCreditPPOEpochTelemetry",
    "BatchedCAGATMAPPOLossOutput",
    "CAGATMAPPOOptimizerBundle",
    "CAGATMAPPORecurrentPPOUpdater",
    "RecurrentPPOEpochDiagnostics",
    "RecurrentPPOMinibatch",
    "RecurrentPPOEvaluation",
    "RecurrentPPOUpdateError",
    "RecurrentPPOUpdateOutput",
    "RouteChoiceStabilityLossOutput",
    "RouteChoiceStabilityTelemetry",
    "build_ca_gat_mappo_optimizers",
    "build_recurrent_ppo_minibatch",
    "compute_configured_batched_ppo_objective_and_loss",
    "compute_route_choice_stability_loss",
]
