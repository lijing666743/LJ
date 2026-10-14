"""Explicit recurrent rollout storage for CA-GAT-MAPPO.

This module stops at validated storage and contiguous retrieval.  It does not
compute advantages, returns, PPO objectives, optimizer steps, or training.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from ..config import AgentCreditMode, RunConfig
from ..env.actions import ActionProposal
from ..env.observation import ActionMasks
from .ca_gat_mappo import (
    ACTION_BRANCH_ORDER,
    ActorTensorBatch,
    CentralizedStateTensorBatch,
    MAPPOTensorSpec,
)
from .ca_gat_mappo_actions import (
    SequentialActionDistributionOutput,
    SequentialActionMaskBatch,
)


class RolloutStorageError(ValueError):
    """Raised when a rollout snapshot violates the frozen storage contract."""


def _cpu_float_tensor(tensor: Tensor, name: str) -> Tensor:
    if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
        raise RolloutStorageError(f"{name} must be a floating tensor")
    if not torch.isfinite(tensor).all():
        raise RolloutStorageError(f"{name} contains NaN or Inf")
    return tensor.detach().to(device="cpu", dtype=torch.float32).clone().contiguous()


def _cpu_bool_tensor(tensor: Tensor, name: str) -> Tensor:
    if not isinstance(tensor, Tensor) or tensor.dtype != torch.bool:
        raise RolloutStorageError(f"{name} must be a boolean tensor")
    return tensor.detach().to(device="cpu").clone().contiguous()


def _cpu_long_tensor(tensor: Tensor, name: str) -> Tensor:
    if not isinstance(tensor, Tensor) or tensor.dtype != torch.long:
        raise RolloutStorageError(f"{name} must be a torch.long tensor")
    return tensor.detach().to(device="cpu").clone().contiguous()


def _cpu_scalar(value: Tensor | float | int, name: str) -> Tensor:
    if isinstance(value, bool):
        raise RolloutStorageError(f"{name} must be a finite scalar")
    if isinstance(value, Tensor):
        if value.numel() != 1 or not value.is_floating_point():
            raise RolloutStorageError(f"{name} must contain one floating value")
        return _cpu_float_tensor(value.reshape(()), name)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RolloutStorageError(f"{name} must be a finite scalar")
    return torch.tensor(float(value), dtype=torch.float32)


def _cpu_credit_tensor(
    value: Tensor | float | int,
    name: str,
    agent_count: int,
) -> Tensor:
    """Store either one legacy scalar or one explicit [A] credit vector."""

    if isinstance(value, Tensor) and tuple(value.shape) == (agent_count,):
        return _cpu_float_tensor(value, name)
    return _cpu_scalar(value, name)


def _validate_metadata(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RolloutStorageError(f"{path} contains NaN or Inf")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RolloutStorageError(f"{path} metadata keys must be strings")
            _validate_metadata(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_metadata(item, f"{path}[{index}]")
        return
    raise RolloutStorageError(
        f"{path} must contain only JSON-safe diagnostic metadata"
    )


def _copy_metadata(value: Any, path: str) -> Any:
    copied = copy.deepcopy(value)
    _validate_metadata(copied, path)
    return copied


def _proposal_branch_index(
    contract: ActionMasks,
    branch: str,
    proposal: ActionProposal,
) -> int:
    value: object = getattr(proposal, branch)
    if branch == "cpu_queue" and value == "local":
        value = proposal.uav_id
    if branch in {"power_level", "cpu_frequency"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RolloutStorageError(f"proposal {branch} must be numeric")
        value = float(value)
    domain = contract.domain_for(branch)
    for index, candidate in enumerate(domain):
        if isinstance(candidate, float) and isinstance(value, (int, float)):
            if not isinstance(value, bool) and candidate == float(value):
                return index
        elif type(candidate) is type(value) and candidate == value:
            return index
    raise RolloutStorageError(f"proposal {branch} lies outside its stored domain")


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise RolloutStorageError(f"{name} must be boolean")
    return value


@dataclass(frozen=True)
class CAGATMAPPORolloutTransition:
    """One explicit slot snapshot with logical [agent, ...] storage axes."""

    spec: MAPPOTensorSpec
    slot: int
    episode_start: bool
    self_features: Tensor
    neighbor_public_features: Tensor
    edge_features: Tensor
    neighbor_mask: Tensor
    action_mask_contracts: tuple[ActionMasks, ...]
    branch_masks: Mapping[str, Tensor]
    proposal_actions: tuple[ActionProposal, ...]
    action_indices: Tensor
    active_branch_indicators: Tensor
    old_branch_log_probs: Tensor
    old_joint_log_prob: Tensor
    hidden_in: Tensor
    centralized_state: Tensor
    old_value: Tensor
    reward: Tensor
    terminated: bool
    truncated: bool
    episode_boundary: bool
    bootstrap_allowed: bool
    bootstrap_value: Tensor | None
    executed_action_summary: tuple[Mapping[str, Any], ...]
    rejection_or_downgrade_summary: Mapping[str, Any]

    @property
    def shared_hidden_in(self) -> Tensor:
        return self.hidden_in[..., :self.spec.gru_hidden_dimension]

    @property
    def route_hidden_in(self) -> Tensor | None:
        return (self.hidden_in[..., self.spec.gru_hidden_dimension:]
                if self.spec.route_private_backbone else None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_mask_contracts", tuple(self.action_mask_contracts))
        object.__setattr__(self, "branch_masks", dict(self.branch_masks))
        object.__setattr__(self, "proposal_actions", tuple(self.proposal_actions))
        executed = tuple(
            _copy_metadata(item, f"executed_action_summary[{index}]")
            for index, item in enumerate(self.executed_action_summary)
        )
        object.__setattr__(self, "executed_action_summary", executed)
        object.__setattr__(
            self,
            "rejection_or_downgrade_summary",
            _copy_metadata(
                self.rejection_or_downgrade_summary,
                "rejection_or_downgrade_summary",
            ),
        )
        self.validate()

    @classmethod
    def from_step(
        cls,
        *,
        spec: MAPPOTensorSpec,
        slot: int,
        actor_batch: ActorTensorBatch,
        action_mask_batch: SequentialActionMaskBatch,
        action_output: SequentialActionDistributionOutput,
        hidden_in: Tensor,
        centralized_state: CentralizedStateTensorBatch,
        old_value: Tensor | float,
        reward: Tensor | float,
        terminated: bool,
        truncated: bool,
        episode_boundary: bool,
        bootstrap_allowed: bool,
        bootstrap_value: Tensor | float | None,
        executed_action_summary: Sequence[Mapping[str, Any]],
        rejection_or_downgrade_summary: Mapping[str, Any],
    ) -> "CAGATMAPPORolloutTransition":
        """Snapshot one already-collected stochastic policy transition on CPU."""

        if not isinstance(spec, MAPPOTensorSpec):
            raise TypeError("spec must be a MAPPOTensorSpec")
        if not isinstance(actor_batch, ActorTensorBatch):
            raise TypeError("actor_batch must be an ActorTensorBatch")
        if not isinstance(action_mask_batch, SequentialActionMaskBatch):
            raise TypeError("action_mask_batch must be a SequentialActionMaskBatch")
        if not isinstance(action_output, SequentialActionDistributionOutput):
            raise TypeError("action_output must be a SequentialActionDistributionOutput")
        if not isinstance(centralized_state, CentralizedStateTensorBatch):
            raise TypeError("centralized_state must be a CentralizedStateTensorBatch")
        shape = actor_batch.validate(spec)
        if shape != (1, 1, spec.uav_count):
            raise RolloutStorageError("one transition requires actor shape [1,1,A]")
        action_mask_batch.validate(shape)
        if action_output.validate(spec) != shape:
            raise RolloutStorageError("action output and actor snapshot axes differ")
        if action_output.mode != "stochastic":
            raise RolloutStorageError(
                "on-policy rollout storage requires stochastic proposal sampling"
            )
        if centralized_state.validate(spec) != (1, 1):
            raise RolloutStorageError("one transition requires critic shape [1,1,F]")
        expected_hidden = (1, spec.uav_count, spec.recurrent_state_dimension)
        if tuple(hidden_in.shape) != expected_hidden:
            raise RolloutStorageError("hidden_in must match configured recurrent state lanes for one transition")
        starts = actor_batch.episode_starts[0, 0]
        if not torch.all(starts == starts[0]):
            raise RolloutStorageError("team episode_start must agree for every agent")
        episode_start = bool(starts[0].item())

        branch_masks = {
            branch: _cpu_bool_tensor(
                action_output.action_masks[branch][0, 0],
                f"branch_masks[{branch}]",
            )
            for branch in ACTION_BRANCH_ORDER
        }
        action_indices = torch.stack(
            [action_output.action_indices[branch][0, 0] for branch in ACTION_BRANCH_ORDER],
            dim=-1,
        )
        active = torch.stack(
            [action_output.active_branches[branch][0, 0] for branch in ACTION_BRANCH_ORDER],
            dim=-1,
        )
        branch_log_probs = torch.stack(
            [action_output.branch_log_probs[branch][0, 0] for branch in ACTION_BRANCH_ORDER],
            dim=-1,
        )
        contracts = copy.deepcopy(action_mask_batch.contracts[0][0])
        # E14a narrows only the CPU Idle atom at the policy boundary. Persist
        # that actual sampling support so storage validation and PPO replay
        # see precisely the same mask, without changing environment legality.
        actual_cpu_masks = branch_masks["cpu_queue"]
        contracts = tuple(
            replace(contract, cpu_queue_mask=actual_cpu_masks[agent].numpy())
            if (contract.cpu_deadline_context is not None
                and not bool(actual_cpu_masks[agent, 0])
                and actual_cpu_masks[agent, 1:].tolist() == contract.cpu_queue_mask[1:].tolist())
            else contract
            for agent, contract in enumerate(contracts))
        proposals = copy.deepcopy(action_output.proposals[0][0])
        resolved_old_value = _cpu_credit_tensor(
            old_value, "old_value", spec.uav_count
        )
        resolved_reward = _cpu_credit_tensor(
            reward, "reward", spec.uav_count
        )
        if resolved_old_value.shape != resolved_reward.shape:
            raise RolloutStorageError(
                "old_value and reward must share scalar or [A] credit shape"
            )
        resolved_bootstrap = (
            None
            if bootstrap_value is None
            else _cpu_credit_tensor(
                bootstrap_value, "bootstrap_value", spec.uav_count
            )
        )
        if action_output.component_log_probs is not None:
            rejection_or_downgrade_summary = dict(rejection_or_downgrade_summary)
            rejection_or_downgrade_summary["policy_components_v1"] = {
                "names": list(action_output.component_log_probs),
                "log_probs": torch.stack(list(action_output.component_log_probs.values()), -1)[0,0].detach().cpu().tolist(),
                "active": torch.stack(list(action_output.component_active.values()), -1)[0,0].detach().cpu().tolist(),
            }
        return cls(
            spec=spec,
            slot=slot,
            episode_start=episode_start,
            self_features=_cpu_float_tensor(
                actor_batch.self_features[0, 0], "self_features"
            ),
            neighbor_public_features=_cpu_float_tensor(
                actor_batch.neighbor_public_features[0, 0],
                "neighbor_public_features",
            ),
            edge_features=_cpu_float_tensor(
                actor_batch.edge_features[0, 0], "edge_features"
            ),
            neighbor_mask=_cpu_bool_tensor(
                actor_batch.neighbor_mask[0, 0], "neighbor_mask"
            ),
            action_mask_contracts=contracts,
            branch_masks=branch_masks,
            proposal_actions=proposals,
            action_indices=_cpu_long_tensor(action_indices, "action_indices"),
            active_branch_indicators=_cpu_bool_tensor(
                active, "active_branch_indicators"
            ),
            old_branch_log_probs=_cpu_float_tensor(
                branch_log_probs, "old_branch_log_probs"
            ),
            old_joint_log_prob=_cpu_float_tensor(
                action_output.joint_log_prob[0, 0], "old_joint_log_prob"
            ),
            hidden_in=_cpu_float_tensor(hidden_in[0], "hidden_in"),
            centralized_state=_cpu_float_tensor(
                centralized_state.features[0, 0], "centralized_state"
            ),
            old_value=resolved_old_value,
            reward=resolved_reward,
            terminated=_require_bool(terminated, "terminated"),
            truncated=_require_bool(truncated, "truncated"),
            episode_boundary=_require_bool(episode_boundary, "episode_boundary"),
            bootstrap_allowed=_require_bool(bootstrap_allowed, "bootstrap_allowed"),
            bootstrap_value=resolved_bootstrap,
            executed_action_summary=tuple(executed_action_summary),
            rejection_or_downgrade_summary=rejection_or_downgrade_summary,
        )

    def validate(self) -> None:
        if not isinstance(self.spec, MAPPOTensorSpec):
            raise TypeError("spec must be a MAPPOTensorSpec")
        if isinstance(self.slot, bool) or not isinstance(self.slot, int) or self.slot < 0:
            raise RolloutStorageError("slot must be a non-negative integer")
        for name in (
            "episode_start",
            "terminated",
            "truncated",
            "episode_boundary",
            "bootstrap_allowed",
        ):
            _require_bool(getattr(self, name), name)
        if self.episode_start != (self.slot == 0):
            raise RolloutStorageError(
                "episode_start must identify exactly the slot-zero transition"
            )
        agents = self.spec.uav_count
        credit_shape = tuple(self.reward.shape)
        if credit_shape not in {(), (agents,)}:
            raise RolloutStorageError("reward must be scalar or have shape [A]")
        if tuple(self.old_value.shape) != credit_shape:
            raise RolloutStorageError("old_value and reward credit shapes differ")
        expected_shapes = {
            "self_features": (agents, self.spec.self_feature_dim),
            "neighbor_public_features": (
                agents,
                agents,
                self.spec.neighbor_public_feature_dim,
            ),
            "edge_features": (agents, agents, self.spec.edge_feature_dim),
            "hidden_in": (agents, self.spec.recurrent_state_dimension),
            "centralized_state": (self.spec.centralized_state_dim,),
            "old_joint_log_prob": (agents,),
            "action_indices": (agents, len(ACTION_BRANCH_ORDER)),
            "active_branch_indicators": (agents, len(ACTION_BRANCH_ORDER)),
            "old_branch_log_probs": (agents, len(ACTION_BRANCH_ORDER)),
            "old_value": credit_shape,
            "reward": credit_shape,
        }
        float_names = (
            "self_features",
            "neighbor_public_features",
            "edge_features",
            "hidden_in",
            "centralized_state",
            "old_branch_log_probs",
            "old_joint_log_prob",
            "old_value",
            "reward",
        )
        for name in float_names:
            tensor = getattr(self, name)
            expected = expected_shapes.get(name, ())
            if not isinstance(tensor, Tensor) or tuple(tensor.shape) != expected:
                raise RolloutStorageError(f"{name} has an invalid shape")
            if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
                raise RolloutStorageError(f"{name} must use CPU float32 storage")
            if not torch.isfinite(tensor).all():
                raise RolloutStorageError(f"{name} contains NaN or Inf")
        if tuple(self.neighbor_mask.shape) != (agents, agents):
            raise RolloutStorageError("neighbor_mask must be [A,A]")
        if self.neighbor_mask.device.type != "cpu" or self.neighbor_mask.dtype != torch.bool:
            raise RolloutStorageError("neighbor_mask must use CPU bool storage")
        if self.action_indices.device.type != "cpu" or self.action_indices.dtype != torch.long:
            raise RolloutStorageError("action_indices must use CPU torch.long storage")
        if (
            self.active_branch_indicators.device.type != "cpu"
            or self.active_branch_indicators.dtype != torch.bool
        ):
            raise RolloutStorageError(
                "active_branch_indicators must use CPU bool storage"
            )
        if tuple(self.branch_masks) != ACTION_BRANCH_ORDER:
            raise RolloutStorageError("branch_masks must contain seven ordered branches")
        if len(self.action_mask_contracts) != agents:
            raise RolloutStorageError("one ActionMasks snapshot is required per agent")
        if len(self.proposal_actions) != agents:
            raise RolloutStorageError("one proposal action is required per agent")
        if len(self.executed_action_summary) != agents:
            raise RolloutStorageError("one executed-action summary is required per agent")
        if tuple(self.action_indices.shape) != expected_shapes["action_indices"]:
            raise RolloutStorageError("action_indices must be [A,7]")
        if (
            tuple(self.active_branch_indicators.shape)
            != expected_shapes["active_branch_indicators"]
        ):
            raise RolloutStorageError(
                "active_branch_indicators must be [A,7]"
            )

        for branch_index, branch in enumerate(ACTION_BRANCH_ORDER):
            dimension = self.spec.action_dimensions[branch]
            mask = self.branch_masks[branch]
            if tuple(mask.shape) != (agents, dimension):
                raise RolloutStorageError(f"{branch} branch mask has an invalid shape")
            if mask.device.type != "cpu" or mask.dtype != torch.bool:
                raise RolloutStorageError(f"{branch} mask must use CPU bool storage")
            if not torch.all(torch.any(mask, dim=-1)):
                raise RolloutStorageError(f"{branch} mask contains an all-invalid row")
            indices = self.action_indices[:, branch_index]
            if torch.any(indices < 0) or torch.any(indices >= dimension):
                raise RolloutStorageError(f"{branch} action index is outside its domain")
            selected_legal = torch.gather(mask, -1, indices.unsqueeze(-1)).squeeze(-1)
            if not torch.all(selected_legal):
                raise RolloutStorageError(f"{branch} selected action is masked out")

        for agent, (contract, proposal, executed) in enumerate(
            zip(
                self.action_mask_contracts,
                self.proposal_actions,
                self.executed_action_summary,
            )
        ):
            if not isinstance(contract, ActionMasks) or contract.uav_id != agent:
                raise RolloutStorageError(
                    "action-mask snapshots must use stable zero-based agent order"
                )
            if not isinstance(proposal, ActionProposal) or proposal.uav_id != agent:
                raise RolloutStorageError(
                    "proposal actions must use stable zero-based agent order"
                )
            if not contract.is_legal(proposal):
                raise RolloutStorageError("stored proposal violates its sampling masks")
            if not isinstance(executed, Mapping) or executed.get("uav_id") != agent:
                raise RolloutStorageError(
                    "executed summaries must use stable zero-based agent order"
                )
            expected_active = torch.tensor(
                contract.active_indicators(proposal), dtype=torch.bool
            )
            if not torch.equal(self.active_branch_indicators[agent], expected_active):
                raise RolloutStorageError("stored active-branch indicators are inconsistent")
            for branch_index, branch in enumerate(ACTION_BRANCH_ORDER):
                expected_index = _proposal_branch_index(contract, branch, proposal)
                if int(self.action_indices[agent, branch_index]) != expected_index:
                    raise RolloutStorageError(
                        f"stored {branch} index does not encode the proposal"
                    )
                expected_mask = torch.tensor(
                    contract.mask_for(branch, proposal).copy(), dtype=torch.bool
                )
                if not torch.equal(self.branch_masks[branch][agent], expected_mask):
                    raise RolloutStorageError(
                        f"stored {branch} mask differs from the sampling snapshot"
                    )

        inactive = ~self.active_branch_indicators
        if torch.any(self.old_branch_log_probs.masked_select(inactive) != 0.0):
            raise RolloutStorageError("inactive branches must store zero old log-prob")
        expected_joint = self.old_branch_log_probs.sum(dim=-1)
        if not torch.allclose(self.old_joint_log_prob, expected_joint):
            raise RolloutStorageError(
                "old_joint_log_prob must equal the active branch-log-prob sum"
            )
        if self.episode_start and torch.count_nonzero(self.hidden_in).item() != 0:
            raise RolloutStorageError("episode-start hidden_in must be exactly zero")
        if self.terminated and self.truncated:
            raise RolloutStorageError("terminated and truncated must remain distinct")
        if self.episode_boundary != (self.terminated or self.truncated):
            raise RolloutStorageError(
                "episode_boundary must equal terminated or truncated"
            )
        if self.bootstrap_allowed == self.episode_boundary:
            raise RolloutStorageError(
                "bootstrap is allowed exactly on non-boundary transitions"
            )
        if self.bootstrap_allowed:
            if self.bootstrap_value is None:
                raise RolloutStorageError(
                    "non-boundary transition requires V(s_t+1)"
                )
            if (
                tuple(self.bootstrap_value.shape) != credit_shape
                or self.bootstrap_value.device.type != "cpu"
                or self.bootstrap_value.dtype != torch.float32
                or not torch.isfinite(self.bootstrap_value).all()
            ):
                raise RolloutStorageError(
                    "bootstrap_value must match the scalar or [A] credit shape"
                )
        elif self.bootstrap_value is not None:
            raise RolloutStorageError(
                "boundary transition must not store a fabricated next value"
            )
        required_audit_keys = {"rejection", "downgrade", "canonicalization"}
        if (
            not isinstance(self.rejection_or_downgrade_summary, Mapping)
            or not required_audit_keys.issubset(
                self.rejection_or_downgrade_summary
            )
        ):
            raise RolloutStorageError(
                "rejection/downgrade summary lacks required executor audit fields"
            )

    def detached_clone(self) -> "CAGATMAPPORolloutTransition":
        """Return an isolated CPU copy so callers cannot mutate buffer storage."""

        return CAGATMAPPORolloutTransition(
            spec=self.spec,
            slot=self.slot,
            episode_start=self.episode_start,
            self_features=self.self_features.clone(),
            neighbor_public_features=self.neighbor_public_features.clone(),
            edge_features=self.edge_features.clone(),
            neighbor_mask=self.neighbor_mask.clone(),
            action_mask_contracts=copy.deepcopy(self.action_mask_contracts),
            branch_masks={key: value.clone() for key, value in self.branch_masks.items()},
            proposal_actions=copy.deepcopy(self.proposal_actions),
            action_indices=self.action_indices.clone(),
            active_branch_indicators=self.active_branch_indicators.clone(),
            old_branch_log_probs=self.old_branch_log_probs.clone(),
            old_joint_log_prob=self.old_joint_log_prob.clone(),
            hidden_in=self.hidden_in.clone(),
            centralized_state=self.centralized_state.clone(),
            old_value=self.old_value.clone(),
            reward=self.reward.clone(),
            terminated=self.terminated,
            truncated=self.truncated,
            episode_boundary=self.episode_boundary,
            bootstrap_allowed=self.bootstrap_allowed,
            bootstrap_value=(
                None if self.bootstrap_value is None else self.bootstrap_value.clone()
            ),
            executed_action_summary=copy.deepcopy(self.executed_action_summary),
            rejection_or_downgrade_summary=copy.deepcopy(
                self.rejection_or_downgrade_summary
            ),
        )


@dataclass(frozen=True)
class CAGATMAPPORolloutChunk:
    """One contiguous transition sequence ready for recurrent re-evaluation."""

    transitions: tuple[CAGATMAPPORolloutTransition, ...]

    def __post_init__(self) -> None:
        copied = tuple(self.transitions)
        object.__setattr__(self, "transitions", copied)
        if not copied:
            raise RolloutStorageError("rollout chunk cannot be empty")
        spec = copied[0].spec
        for transition in copied:
            if not isinstance(transition, CAGATMAPPORolloutTransition):
                raise TypeError("chunk entries must be rollout transitions")
            if transition.spec != spec:
                raise RolloutStorageError("chunk transitions use different tensor specs")
            transition.validate()
        _validate_transition_order(copied)

    @property
    def spec(self) -> MAPPOTensorSpec:
        return self.transitions[0].spec

    @property
    def length(self) -> int:
        return len(self.transitions)

    @property
    def initial_hidden(self) -> Tensor:
        """Return both owned hidden lanes at t_start; legacy has only shared H."""

        return self.transitions[0].hidden_in.unsqueeze(0).clone()

    @property
    def actor_batch(self) -> ActorTensorBatch:
        action_indices = self.action_indices
        starts = torch.tensor(
            [item.episode_start for item in self.transitions], dtype=torch.bool
        ).view(1, self.length, 1).expand(1, self.length, self.spec.uav_count).clone()
        result = ActorTensorBatch(
            self_features=torch.stack(
                [item.self_features for item in self.transitions], dim=0
            ).unsqueeze(0),
            neighbor_public_features=torch.stack(
                [item.neighbor_public_features for item in self.transitions], dim=0
            ).unsqueeze(0),
            edge_features=torch.stack(
                [item.edge_features for item in self.transitions], dim=0
            ).unsqueeze(0),
            neighbor_mask=torch.stack(
                [item.neighbor_mask for item in self.transitions], dim=0
            ).unsqueeze(0),
            action_masks={
                branch: torch.stack(
                    [item.branch_masks[branch] for item in self.transitions], dim=0
                ).unsqueeze(0)
                for branch in ACTION_BRANCH_ORDER
            },
            action_indices={
                branch: action_indices[..., branch_index].unsqueeze(0)
                for branch_index, branch in enumerate(ACTION_BRANCH_ORDER)
            },
            episode_starts=starts,
        )
        result.validate(self.spec)
        return result

    @property
    def action_mask_batch(self) -> SequentialActionMaskBatch:
        return SequentialActionMaskBatch(
            ((tuple(item.action_mask_contracts for item in self.transitions)),)
        )

    @property
    def centralized_batch(self) -> CentralizedStateTensorBatch:
        result = CentralizedStateTensorBatch(
            torch.stack(
                [item.centralized_state for item in self.transitions], dim=0
            ).unsqueeze(0)
        )
        result.validate(self.spec)
        return result

    @property
    def proposals(self) -> tuple[tuple[tuple[ActionProposal, ...], ...], ...]:
        return (tuple(item.proposal_actions for item in self.transitions),)

    @property
    def action_indices(self) -> Tensor:
        return torch.stack([item.action_indices for item in self.transitions], dim=0)

    @property
    def branch_masks(self) -> dict[str, Tensor]:
        return {
            branch: torch.stack(
                [item.branch_masks[branch] for item in self.transitions], dim=0
            )
            for branch in ACTION_BRANCH_ORDER
        }

    @property
    def active_branch_indicators(self) -> Tensor:
        return torch.stack(
            [item.active_branch_indicators for item in self.transitions], dim=0
        )

    @property
    def old_branch_log_probs(self) -> Tensor:
        return torch.stack(
            [item.old_branch_log_probs for item in self.transitions], dim=0
        )

    @property
    def old_joint_log_prob(self) -> Tensor:
        return torch.stack(
            [item.old_joint_log_prob for item in self.transitions], dim=0
        )

    @property
    def hidden_in(self) -> Tensor:
        return torch.stack([item.hidden_in for item in self.transitions], dim=0)

    @property
    def old_value(self) -> Tensor:
        return torch.stack([item.old_value for item in self.transitions], dim=0)

    @property
    def reward(self) -> Tensor:
        return torch.stack([item.reward for item in self.transitions], dim=0)

    @property
    def terminated(self) -> Tensor:
        return torch.tensor(
            [item.terminated for item in self.transitions], dtype=torch.bool
        )

    @property
    def truncated(self) -> Tensor:
        return torch.tensor(
            [item.truncated for item in self.transitions], dtype=torch.bool
        )

    @property
    def episode_boundary(self) -> Tensor:
        return torch.tensor(
            [item.episode_boundary for item in self.transitions], dtype=torch.bool
        )

    @property
    def episode_starts(self) -> Tensor:
        return torch.tensor(
            [item.episode_start for item in self.transitions], dtype=torch.bool
        )

    @property
    def bootstrap_allowed(self) -> Tensor:
        return torch.tensor(
            [item.bootstrap_allowed for item in self.transitions], dtype=torch.bool
        )

    @property
    def bootstrap_values(self) -> Tensor:
        """Return [T] or [T,A] values with shape-preserving zero placeholders."""

        return torch.stack(
            [
                item.bootstrap_value
                if item.bootstrap_value is not None
                else torch.zeros_like(item.old_value)
                for item in self.transitions
            ],
            dim=0,
        )

    @property
    def sequence_mask(self) -> Tensor:
        """All retrieved positions are real; padding is not implemented here."""

        return torch.ones(self.length, dtype=torch.bool)

    @property
    def executed_action_summaries(self) -> tuple[tuple[Mapping[str, Any], ...], ...]:
        return tuple(
            copy.deepcopy(item.executed_action_summary) for item in self.transitions
        )

    @property
    def rejection_or_downgrade_summaries(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            copy.deepcopy(item.rejection_or_downgrade_summary)
            for item in self.transitions
        )


def _validate_transition_order(
    transitions: Sequence[CAGATMAPPORolloutTransition],
) -> None:
    for previous, current in zip(transitions, transitions[1:]):
        if previous.episode_boundary:
            if not current.episode_start or current.slot != 0:
                raise RolloutStorageError(
                    "transition after an episode boundary must start at slot zero"
                )
        else:
            if current.episode_start or current.slot != previous.slot + 1:
                raise RolloutStorageError(
                    "non-boundary transitions must remain temporally contiguous"
                )


class CAGATMAPPORolloutBuffer:
    """Bounded CPU buffer with explicit finalize, clear, and sequence retrieval."""

    def __init__(self, config: RunConfig, capacity: int | None = None) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        config.validate()
        resolved = (
            config.training.mappo.rollout_length_slots
            if capacity is None
            else capacity
        )
        if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved <= 0:
            raise RolloutStorageError("capacity must be a positive integer")
        self.spec = MAPPOTensorSpec.from_config(config)
        self.agent_credit_mode = AgentCreditMode(
            config.training.mappo.agent_credit_mode
        )
        self.capacity = resolved
        self._transitions: list[CAGATMAPPORolloutTransition] = []
        self._finalized = False

    def __len__(self) -> int:
        return len(self._transitions)

    @property
    def full(self) -> bool:
        return len(self) == self.capacity

    @property
    def finalized(self) -> bool:
        return self._finalized

    def append(self, transition: CAGATMAPPORolloutTransition) -> None:
        if self._finalized:
            raise RolloutStorageError("cannot append after buffer finalization")
        if self.full:
            raise RolloutStorageError("rollout buffer capacity exceeded")
        if not isinstance(transition, CAGATMAPPORolloutTransition):
            raise TypeError("transition must be a CAGATMAPPORolloutTransition")
        if transition.spec != self.spec:
            raise RolloutStorageError("transition tensor spec differs from the buffer")
        transition.validate()
        expected_credit_shape = (
            ()
            if self.agent_credit_mode is AgentCreditMode.TEAM
            else (self.spec.uav_count,)
        )
        if tuple(transition.reward.shape) != expected_credit_shape:
            raise RolloutStorageError(
                "transition reward shape differs from agent_credit_mode"
            )
        if tuple(transition.old_value.shape) != expected_credit_shape:
            raise RolloutStorageError(
                "transition value shape differs from agent_credit_mode"
            )
        if self._transitions:
            _validate_transition_order((self._transitions[-1], transition))
        elif transition.episode_start and transition.slot != 0:
            raise RolloutStorageError("episode-start transition must use slot zero")
        self._transitions.append(transition.detached_clone())

    def append_step(self, **kwargs: Any) -> CAGATMAPPORolloutTransition:
        """Build and append one transition using the current network interfaces."""

        if "spec" in kwargs:
            raise RolloutStorageError("append_step derives spec from the buffer")
        transition = CAGATMAPPORolloutTransition.from_step(spec=self.spec, **kwargs)
        self.append(transition)
        return transition.detached_clone()

    def transition_at(self, index: int) -> CAGATMAPPORolloutTransition:
        if isinstance(index, bool) or not isinstance(index, int):
            raise RolloutStorageError("transition index must be an integer")
        if index < 0 or index >= len(self):
            raise RolloutStorageError("transition index is outside written storage")
        return self._transitions[index].detached_clone()

    def finalize(self) -> CAGATMAPPORolloutChunk:
        if not self._transitions:
            raise RolloutStorageError("cannot finalize an empty rollout buffer")
        self._finalized = True
        return CAGATMAPPORolloutChunk(
            tuple(item.detached_clone() for item in self._transitions)
        )

    def get_sequence(self, start: int, length: int) -> CAGATMAPPORolloutChunk:
        if not self._finalized:
            raise RolloutStorageError("finalize the buffer before sequence retrieval")
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise RolloutStorageError("sequence start must be a non-negative integer")
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise RolloutStorageError("sequence length must be a positive integer")
        stop = start + length
        if stop > len(self):
            raise RolloutStorageError("sequence reaches unwritten buffer positions")
        return CAGATMAPPORolloutChunk(
            tuple(item.detached_clone() for item in self._transitions[start:stop])
        )

    def clear(self) -> None:
        self._transitions.clear()
        self._finalized = False


__all__ = [
    "CAGATMAPPORolloutBuffer",
    "CAGATMAPPORolloutChunk",
    "CAGATMAPPORolloutTransition",
    "RolloutStorageError",
]
