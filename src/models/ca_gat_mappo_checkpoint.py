"""Structured Checkpoint V1 support for CA-GAT-MAPPO exact resume.

The schema is intentionally independent from mutable Trainer, environment, and
rollout-buffer objects.  All persisted tensors are detached CPU snapshots.
"""

from __future__ import annotations

from dataclasses import fields
import copy
import math
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from ..config import (
    CHECKPOINT_KIND_FINAL_COMPLETED,
    CHECKPOINT_KIND_PERIODIC_RESUME,
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_V1_ACTIVE_ROLLOUT_FIELDS,
    CHECKPOINT_V1_DIAGNOSTICS_STATE_FIELDS,
    CHECKPOINT_V1_KINDS,
    CHECKPOINT_V1_MODEL_STATE_FIELDS,
    CHECKPOINT_V1_OPTIMIZER_STATE_FIELDS,
    CHECKPOINT_V1_POLICY_RNG_STATE_FIELDS,
    CHECKPOINT_V1_RUNTIME_PROVENANCE_FIELDS,
    CHECKPOINT_V1_TOP_LEVEL_FIELDS,
    CHECKPOINT_V1_TRAINING_STATE_FIELDS,
    CHECKPOINT_V1_TRANSITION_FIELDS,
    MAPPO_INITIAL_POLICY_VERSION,
    ConfigError,
    RunConfig,
    canonicalize_config_identity,
    compute_mappo_checkpoint_active_rollout_length,
    mappo_checkpoint_kind_at,
    validate_mappo_checkpoint_resume_compatibility,
    validate_mappo_checkpoint_contract,
)
from ..env.actions import ActionProposal
from ..env.observation import ActionMasks
from .ca_gat_mappo import ACTION_BRANCH_ORDER, MAPPOTensorSpec
from .ca_gat_mappo_rollout import (
    CAGATMAPPORolloutBuffer,
    CAGATMAPPORolloutTransition,
)
from .ca_gat_mappo_update import CAGATMAPPOOptimizerBundle


class CheckpointError(RuntimeError):
    """Raised when Checkpoint V1 state is invalid, corrupt, or incompatible."""


_ACTION_MASK_ARRAY_FIELDS = (
    "route_mask",
    "tx_select_mask",
    "cpu_queue_mask",
    "power_energy_mask",
    "cpu_frequency_energy_mask",
)
_ACTION_MASK_TUPLE_FIELDS = (
    "sampling_order",
    "route_domain",
    "tx_select_domain",
    "resource_group_domain",
    "resource_width_domain",
    "power_level_domain",
    "cpu_queue_domain",
    "cpu_frequency_domain",
)
_ACTION_MASK_FIELDS = tuple(item.name for item in fields(ActionMasks))
_ACTION_PROPOSAL_FIELDS = tuple(item.name for item in fields(ActionProposal))


def _mapping(value: Any, path: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointError(f"{path} must be a mapping")
    return value


def _exact_keys(value: Mapping[Any, Any], expected: tuple[str, ...], path: str) -> None:
    if tuple(value) != expected:
        missing = tuple(key for key in expected if key not in value)
        extra = tuple(key for key in value if key not in expected)
        raise CheckpointError(
            f"{path} fields differ from Checkpoint V1; missing={missing}, extra={extra}"
        )


def _non_negative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointError(f"{path} must be a non-negative integer")
    return value


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointError(f"{path} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise CheckpointError(f"{path} must be finite")
    return converted


def _cpu_tensor(value: Tensor, path: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise CheckpointError(f"{path} must be a tensor")
    if value.layout != torch.strided:
        raise CheckpointError(f"{path} must use strided tensor storage")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise CheckpointError(f"{path} contains NaN or Inf")
    return value.detach().to(device="cpu").clone().contiguous()


def _clone_structured(value: Any, path: str = "payload") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CheckpointError(f"{path} contains NaN or Inf")
        return value
    if isinstance(value, Tensor):
        return _cpu_tensor(value, path)
    if isinstance(value, list):
        return [
            _clone_structured(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _clone_structured(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        result: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                raise CheckpointError(f"{path} has an unsupported mapping key")
            result[key] = _clone_structured(item, f"{path}.{key}")
        return result
    raise CheckpointError(
        f"{path} contains unsupported long-term schema value {type(value).__name__}"
    )


def _validate_structured(value: Any, path: str = "payload") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CheckpointError(f"{path} contains NaN or Inf")
        return
    if isinstance(value, Tensor):
        if value.device.type != "cpu" or value.requires_grad:
            raise CheckpointError(
                f"{path} tensor must be a detached CPU snapshot"
            )
        if value.layout != torch.strided:
            raise CheckpointError(f"{path} must use strided tensor storage")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise CheckpointError(f"{path} contains NaN or Inf")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_structured(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                raise CheckpointError(f"{path} has an unsupported mapping key")
            _validate_structured(item, f"{path}.{key}")
        return
    raise CheckpointError(
        f"{path} contains unsupported long-term schema value "
        f"{type(value).__name__}"
    )


def _serialize_action_masks(contract: ActionMasks) -> dict[str, Any]:
    if not isinstance(contract, ActionMasks):
        raise CheckpointError("action-mask snapshot must be ActionMasks")
    result: dict[str, Any] = {}
    for name in _ACTION_MASK_FIELDS:
        value = getattr(contract, name)
        if name == "cpu_deadline_context" and value is None:
            continue  # Preserve the exact legacy Checkpoint V1 representation.
        if name in _ACTION_MASK_ARRAY_FIELDS:
            result[name] = torch.from_numpy(np.asarray(value).copy()).to(dtype=torch.bool)
        elif name in _ACTION_MASK_TUPLE_FIELDS:
            result[name] = tuple(value)
        else:
            result[name] = value
    return _clone_structured(result, "action_mask")


def _deserialize_action_masks(state: Any) -> ActionMasks:
    mapping = _mapping(state, "action_mask")
    expected = tuple(name for name in _ACTION_MASK_FIELDS
                     if name != "cpu_deadline_context" or name in mapping)
    _exact_keys(mapping, expected, "action_mask")
    kwargs: dict[str, Any] = {}
    for name in _ACTION_MASK_FIELDS:
        if name == "cpu_deadline_context" and name not in mapping:
            continue
        value = mapping[name]
        if name in _ACTION_MASK_ARRAY_FIELDS:
            tensor = _cpu_tensor(value, f"action_mask.{name}")
            if tensor.dtype != torch.bool:
                raise CheckpointError(f"action_mask.{name} must be bool")
            kwargs[name] = tensor.numpy().copy()
        elif name in _ACTION_MASK_TUPLE_FIELDS:
            if not isinstance(value, (list, tuple)):
                raise CheckpointError(f"action_mask.{name} must be a sequence")
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    try:
        return ActionMasks(**kwargs)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"invalid ActionMasks snapshot: {exc}") from exc


def _serialize_proposal(proposal: ActionProposal) -> dict[str, Any]:
    if not isinstance(proposal, ActionProposal):
        raise CheckpointError("proposal snapshot must be ActionProposal")
    return {
        name: _clone_structured(getattr(proposal, name), f"proposal.{name}")
        for name in _ACTION_PROPOSAL_FIELDS
    }


def _deserialize_proposal(state: Any) -> ActionProposal:
    mapping = _mapping(state, "proposal")
    _exact_keys(mapping, _ACTION_PROPOSAL_FIELDS, "proposal")
    try:
        return ActionProposal(**dict(mapping))
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"invalid ActionProposal snapshot: {exc}") from exc


def serialize_rollout_transition(
    transition: CAGATMAPPORolloutTransition,
) -> dict[str, Any]:
    """Convert one validated rollout transition to stable structured state."""

    if not isinstance(transition, CAGATMAPPORolloutTransition):
        raise TypeError("transition must be a CAGATMAPPORolloutTransition")
    transition.validate()
    state = {
        "slot": transition.slot,
        "episode_start": transition.episode_start,
        "self_features": transition.self_features,
        "neighbor_public_features": transition.neighbor_public_features,
        "edge_features": transition.edge_features,
        "neighbor_mask": transition.neighbor_mask,
        "action_mask_contracts": tuple(
            _serialize_action_masks(item) for item in transition.action_mask_contracts
        ),
        "branch_masks": {
            branch: transition.branch_masks[branch] for branch in ACTION_BRANCH_ORDER
        },
        "proposal_actions": tuple(
            _serialize_proposal(item) for item in transition.proposal_actions
        ),
        "action_indices": transition.action_indices,
        "active_branch_indicators": transition.active_branch_indicators,
        "old_branch_log_probs": transition.old_branch_log_probs,
        "old_joint_log_prob": transition.old_joint_log_prob,
        "hidden_in": transition.hidden_in,
        "centralized_state": transition.centralized_state,
        "old_value": transition.old_value,
        "reward": transition.reward,
        "terminated": transition.terminated,
        "truncated": transition.truncated,
        "episode_boundary": transition.episode_boundary,
        "bootstrap_allowed": transition.bootstrap_allowed,
        "bootstrap_value": transition.bootstrap_value,
        "executed_action_summary": transition.executed_action_summary,
        "rejection_or_downgrade_summary": transition.rejection_or_downgrade_summary,
    }
    _exact_keys(state, CHECKPOINT_V1_TRANSITION_FIELDS, "transition")
    return _clone_structured(state, "transition")


def restore_rollout_transition(
    config: RunConfig,
    state: Any,
) -> CAGATMAPPORolloutTransition:
    """Reconstruct and revalidate one stored transition without regeneration."""

    mapping = _mapping(state, "transition")
    _exact_keys(mapping, CHECKPOINT_V1_TRANSITION_FIELDS, "transition")
    spec = MAPPOTensorSpec.from_config(config)
    try:
        transition = CAGATMAPPORolloutTransition(
            spec=spec,
            slot=mapping["slot"],
            episode_start=mapping["episode_start"],
            self_features=_cpu_tensor(mapping["self_features"], "self_features"),
            neighbor_public_features=_cpu_tensor(
                mapping["neighbor_public_features"], "neighbor_public_features"
            ),
            edge_features=_cpu_tensor(mapping["edge_features"], "edge_features"),
            neighbor_mask=_cpu_tensor(mapping["neighbor_mask"], "neighbor_mask"),
            action_mask_contracts=tuple(
                _deserialize_action_masks(item)
                for item in mapping["action_mask_contracts"]
            ),
            branch_masks={
                branch: _cpu_tensor(mapping["branch_masks"][branch], f"branch_masks.{branch}")
                for branch in ACTION_BRANCH_ORDER
            },
            proposal_actions=tuple(
                _deserialize_proposal(item) for item in mapping["proposal_actions"]
            ),
            action_indices=_cpu_tensor(mapping["action_indices"], "action_indices"),
            active_branch_indicators=_cpu_tensor(
                mapping["active_branch_indicators"], "active_branch_indicators"
            ),
            old_branch_log_probs=_cpu_tensor(
                mapping["old_branch_log_probs"], "old_branch_log_probs"
            ),
            old_joint_log_prob=_cpu_tensor(
                mapping["old_joint_log_prob"], "old_joint_log_prob"
            ),
            hidden_in=_cpu_tensor(mapping["hidden_in"], "hidden_in"),
            centralized_state=_cpu_tensor(
                mapping["centralized_state"], "centralized_state"
            ),
            old_value=_cpu_tensor(mapping["old_value"], "old_value"),
            reward=_cpu_tensor(mapping["reward"], "reward"),
            terminated=mapping["terminated"],
            truncated=mapping["truncated"],
            episode_boundary=mapping["episode_boundary"],
            bootstrap_allowed=mapping["bootstrap_allowed"],
            bootstrap_value=(
                None
                if mapping["bootstrap_value"] is None
                else _cpu_tensor(mapping["bootstrap_value"], "bootstrap_value")
            ),
            executed_action_summary=tuple(mapping["executed_action_summary"]),
            rejection_or_downgrade_summary=mapping[
                "rejection_or_downgrade_summary"
            ],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError(f"invalid rollout transition: {exc}") from exc
    return transition


def serialize_active_rollout(
    buffer: CAGATMAPPORolloutBuffer,
    rollout_policy_version: int | None,
    checkpoint_kind: str,
) -> dict[str, Any]:
    """Snapshot an unfinalized active rollout in original temporal order."""

    if not isinstance(buffer, CAGATMAPPORolloutBuffer):
        raise TypeError("buffer must be a CAGATMAPPORolloutBuffer")
    if buffer.finalized:
        raise CheckpointError("active rollout must not be finalized")
    if len(buffer) >= buffer.capacity:
        raise CheckpointError("checkpoint cannot contain a pending full rollout")
    if len(buffer) == 0:
        if rollout_policy_version is not None:
            raise CheckpointError("empty rollout must not carry a policy version")
    else:
        _non_negative_int(rollout_policy_version, "rollout_policy_version")
    if checkpoint_kind == CHECKPOINT_KIND_FINAL_COMPLETED and len(buffer) != 0:
        raise CheckpointError("FINAL_COMPLETED must have an empty active rollout")
    if checkpoint_kind not in CHECKPOINT_V1_KINDS:
        raise CheckpointError("unknown checkpoint kind")
    state = {
        "rollout_length": len(buffer),
        "rollout_policy_version": rollout_policy_version,
        "ordered_transitions": [
            serialize_rollout_transition(buffer.transition_at(index))
            for index in range(len(buffer))
        ],
    }
    return _clone_structured(state, "active_rollout_state")


def restore_active_rollout(
    config: RunConfig,
    state: Any,
    *,
    checkpoint_kind: str,
    policy_version: int,
) -> tuple[CAGATMAPPORolloutBuffer, int | None]:
    """Restore one partial rollout and rerun all normal buffer validations."""

    mapping = _mapping(state, "active_rollout_state")
    _exact_keys(mapping, CHECKPOINT_V1_ACTIVE_ROLLOUT_FIELDS, "active_rollout_state")
    length = _non_negative_int(mapping["rollout_length"], "rollout_length")
    transitions = mapping["ordered_transitions"]
    if not isinstance(transitions, (list, tuple)) or len(transitions) != length:
        raise CheckpointError("ordered transition count differs from rollout_length")
    capacity = config.training.mappo.rollout_length_slots
    if length >= capacity:
        raise CheckpointError("restored active rollout length must be less than 256")
    version = mapping["rollout_policy_version"]
    if length == 0:
        if version is not None:
            raise CheckpointError("empty active rollout must not carry a policy version")
    else:
        version = _non_negative_int(version, "rollout_policy_version")
        if version != policy_version:
            raise CheckpointError("active rollout policy version mismatch")
    if checkpoint_kind == CHECKPOINT_KIND_FINAL_COMPLETED and length != 0:
        raise CheckpointError("FINAL_COMPLETED must restore an empty rollout")
    buffer = CAGATMAPPORolloutBuffer(config)
    try:
        for item in transitions:
            buffer.append(restore_rollout_transition(config, item))
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"invalid active rollout ordering: {exc}") from exc
    if len(buffer) != length or buffer.finalized:
        raise CheckpointError("active rollout restore did not preserve buffer state")
    if config.training.mappo.tx_upper_credit_mode == "paired_shadow_v1" and length:
        from .tx_paired_shadow import selection_state
        try:
            selection_state(buffer, config, version)
        except (TypeError, ValueError) as exc:
            raise CheckpointError(f"invalid E13 paired selector/credit state: {exc}") from exc
    return buffer, version


def serialize_policy_rng(generator: torch.Generator) -> dict[str, Any]:
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be torch.Generator")
    state = generator.get_state().detach().to(device="cpu").clone().contiguous()
    return {"generator_state": state, "device_type": generator.device.type}


def restore_policy_rng(state: Any, expected_device_type: str) -> torch.Generator:
    mapping = _mapping(state, "policy_rng_state")
    _exact_keys(mapping, CHECKPOINT_V1_POLICY_RNG_STATE_FIELDS, "policy_rng_state")
    device_type = mapping["device_type"]
    if device_type != expected_device_type:
        raise CheckpointError("policy generator device type mismatch")
    if device_type not in {"cpu", "cuda"}:
        raise CheckpointError("unsupported policy generator device type")
    generator_state = _cpu_tensor(mapping["generator_state"], "generator_state")
    if generator_state.dtype != torch.uint8 or generator_state.ndim != 1:
        raise CheckpointError("policy generator state must be one CPU uint8 vector")
    try:
        generator = torch.Generator(device=device_type)
        generator.set_state(generator_state)
    except (RuntimeError, TypeError) as exc:
        raise CheckpointError(f"invalid policy generator state: {exc}") from exc
    return generator


def runtime_provenance(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    """Return audit-only same-runtime metadata without changing torch settings."""

    device_name = None
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "device_type": device.type,
        "device_name": device_name,
        "dtype": str(dtype),
    }


def _module_device_dtype(module: nn.Module, name: str) -> tuple[torch.device, torch.dtype]:
    from .tx_upper_counterfactual import model_precision_tensors
    tensors = tuple(module.parameters()) + tuple(module.buffers())
    if not tensors:
        raise CheckpointError(f"{name} has no tensors")
    devices = {tensor.device for tensor in tensors}
    dtypes = {tensor.dtype for tensor in model_precision_tensors(module) if tensor.is_floating_point()}
    if len(devices) != 1 or len(dtypes) != 1:
        raise CheckpointError(f"{name} must use one device and floating dtype")
    return next(iter(devices)), next(iter(dtypes))


def _validate_module_state(module: nn.Module, state: Any, name: str) -> None:
    mapping = _mapping(state, f"model_state.{name}")
    current = module.state_dict()
    if tuple(mapping) != tuple(current):
        raise CheckpointError(f"{name} state_dict keys mismatch")
    for key, target in current.items():
        saved = mapping[key]
        if not isinstance(saved, Tensor):
            raise CheckpointError(f"{name}.{key} must be a tensor")
        if saved.device.type != "cpu" or saved.requires_grad:
            raise CheckpointError(f"{name}.{key} must be detached CPU state")
        if saved.shape != target.shape or saved.dtype != target.dtype:
            raise CheckpointError(f"{name}.{key} shape or dtype mismatch")


def _normalize_legacy_actor_optimizer_state(
    optimizer: torch.optim.Optimizer,
    state: Any,
    parameters: tuple[nn.Parameter, ...],
    config: RunConfig,
) -> Any:
    """Map historical single-group Adam IDs to the two groups at scale=1 only.

    The old group followed actor.parameters() registration order. Preserve
    every moment and step by parameter identity; never change the checkpoint.
    """
    mapping = _mapping(state, "optimizer_state.actor_adam")
    groups = mapping.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1 or len(optimizer.param_groups) != 2:
        return state
    if config.training.mappo.route_head_lr_scale != 1.0:
        raise CheckpointError("legacy single-group actor Adam requires route_head_lr_scale=1.0")
    old_group = groups[0]
    if not isinstance(old_group, Mapping):
        raise CheckpointError("legacy actor Adam group is invalid")
    old_ids = old_group.get("params")
    if (not isinstance(old_ids, list) or len(old_ids) != len(parameters)
            or any(isinstance(i, bool) or not isinstance(i, int) for i in old_ids)
            or len(set(old_ids)) != len(old_ids)):
        raise CheckpointError("legacy actor Adam parameter IDs are invalid")
    by_parameter = dict(zip(map(id, parameters), old_ids))
    normalized = copy.deepcopy(dict(mapping))
    normalized["param_groups"] = [
        {**copy.deepcopy(dict(old_group)), "group_name": group["group_name"],
         "params": [by_parameter[id(parameter)] for parameter in group["params"]]}
        for group in optimizer.param_groups
    ]
    return normalized


def _validate_optimizer_state(
    optimizer: torch.optim.Optimizer,
    state: Any,
    parameters: tuple[nn.Parameter, ...],
    name: str,
) -> None:
    mapping = _mapping(state, f"optimizer_state.{name}")
    if set(mapping) != {"state", "param_groups"}:
        raise CheckpointError(f"{name} optimizer state_dict fields mismatch")
    groups = mapping["param_groups"]
    states = mapping["state"]
    if not isinstance(groups, list) or len(groups) != len(optimizer.param_groups):
        raise CheckpointError(f"{name} optimizer parameter groups mismatch")
    if not isinstance(states, Mapping):
        raise CheckpointError(f"{name} optimizer state must be a mapping")
    saved_ids: list[int] = []
    for group, current_group in zip(groups, optimizer.param_groups):
        if not isinstance(group, Mapping) or "params" not in group:
            raise CheckpointError(f"{name} optimizer group is invalid")
        ids = group["params"]
        if not isinstance(ids, list) or len(ids) != len(current_group["params"]):
            raise CheckpointError(f"{name} optimizer parameter count mismatch")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in ids):
            raise CheckpointError(f"{name} optimizer parameter IDs are invalid")
        saved_ids.extend(ids)
        for key in ("lr", "betas", "eps", "weight_decay"):
            if group.get(key) != current_group[key]:
                raise CheckpointError(f"{name} optimizer hyperparameters mismatch")
        if group.get("group_name") != current_group.get("group_name"):
            raise CheckpointError(f"{name} optimizer group name mismatch")
    if (len(saved_ids) != len(parameters) or len(set(saved_ids)) != len(saved_ids)
            or set(states) - set(saved_ids)):
        raise CheckpointError(f"{name} optimizer state refers to unknown parameters")
    # Optimizer serialization follows group order, not model registration order.
    grouped_parameters = tuple(parameter for group in optimizer.param_groups for parameter in group["params"])
    parameter_by_id = dict(zip(saved_ids, grouped_parameters))
    for parameter_id, parameter_state in states.items():
        if parameter_id not in parameter_by_id or not isinstance(parameter_state, Mapping):
            raise CheckpointError(f"{name} optimizer parameter state is invalid")
        parameter = parameter_by_id[parameter_id]
        for key, value in parameter_state.items():
            if not isinstance(key, str) or not isinstance(value, Tensor):
                raise CheckpointError(f"{name} optimizer tensor state is invalid")
            if value.device.type != "cpu" or value.requires_grad:
                raise CheckpointError(f"{name} optimizer state must be detached CPU tensors")
            if key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}:
                if value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise CheckpointError(f"{name} optimizer moment shape or dtype mismatch")
            elif key == "step" and value.numel() != 1:
                raise CheckpointError(f"{name} optimizer step must be scalar")


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Move all restored optimizer tensors to the already-validated device."""

    for parameter_state in optimizer.state.values():
        for key, value in tuple(parameter_state.items()):
            if isinstance(value, Tensor):
                parameter_state[key] = value.to(device=device)


def _validate_optimizer_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    name: str,
) -> None:
    for parameter_state in optimizer.state.values():
        for value in parameter_state.values():
            if isinstance(value, Tensor) and value.device != device:
                raise CheckpointError(f"{name} optimizer state is on the wrong device")


def build_checkpoint_payload(
    *,
    config: RunConfig,
    checkpoint_kind: str,
    actor: nn.Module,
    critic: nn.Module,
    optimizers: CAGATMAPPOOptimizerBundle,
    policy_generator: torch.Generator,
    training_state: Mapping[str, Any],
    active_rollout_state: Mapping[str, Any],
    diagnostics_state: Mapping[str, Any],
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Build and fully validate one immutable structured Checkpoint V1 payload."""

    validate_mappo_checkpoint_contract(config)
    if checkpoint_kind not in CHECKPOINT_V1_KINDS:
        raise CheckpointError("unknown checkpoint kind")
    if not isinstance(optimizers, CAGATMAPPOOptimizerBundle):
        raise CheckpointError("checkpointing requires the two formal Adam optimizers")
    optimizers.validate(actor, critic, config)
    if actor.route_decoder_mode != config.training.mappo.route_decoder_mode:
        raise CheckpointError("actor and checkpoint Route decoder modes differ")
    if actor.route_feature_mode != config.training.mappo.route_feature_mode:
        raise CheckpointError("actor and checkpoint Route feature modes differ")
    actor_device, actor_dtype = _module_device_dtype(actor, "actor")
    critic_device, critic_dtype = _module_device_dtype(critic, "critic")
    if actor_device != critic_device or actor_dtype != critic_dtype or actor_dtype != dtype:
        raise CheckpointError("actor, critic, and checkpoint dtype/device differ")
    if actor_device.type != config.training.mappo.training_device:
        raise CheckpointError("model device differs from explicit training_device")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": checkpoint_kind,
        "method_id": config.method_id,
        "git_commit": config.git_commit,
        "config_hash": config.config_hash,
        "config_snapshot": config.snapshot_dict(),
        "runtime_provenance": runtime_provenance(actor_device, dtype),
        "training_state": dict(training_state),
        "model_state": {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        },
        "optimizer_state": {
            "actor_adam": optimizers.actor_optimizer.state_dict(),
            "critic_adam": optimizers.critic_optimizer.state_dict(),
        },
        "policy_rng_state": serialize_policy_rng(policy_generator),
        "active_rollout_state": dict(active_rollout_state),
        "diagnostics_state": dict(diagnostics_state),
    }
    if config.training.mappo.route_decoder_mode == "unified_node_v1":
        from .unified_route import unified_route_checkpoint_identity
        payload["route_decoder_identity"] = unified_route_checkpoint_identity(
            config.seed, config.training.mappo.route_feature_mode)
    cloned = _clone_structured(payload)
    validate_checkpoint_payload(cloned, config=config)
    return cloned


def validate_checkpoint_payload(
    payload: Any,
    *,
    config: RunConfig | None = None,
) -> Mapping[str, Any]:
    """Validate schema, structure, accounting, and optional config geometry."""

    mapping = _mapping(payload, "checkpoint")
    expected_fields = CHECKPOINT_V1_TOP_LEVEL_FIELDS
    source_snapshot = _mapping(mapping.get("config_snapshot"), "config_snapshot")
    identity_training = source_snapshot.get("training")
    identity_mappo = identity_training.get("mappo") if isinstance(identity_training, Mapping) else None
    source_mode = identity_mappo.get("route_decoder_mode") if isinstance(identity_mappo, Mapping) else None
    if source_mode == "unified_node_v1":
        from .unified_route import unified_route_checkpoint_identity
        expected_fields += ("route_decoder_identity",)
        try:
            expected_identity = unified_route_checkpoint_identity(
                source_snapshot.get("seed"),
                identity_mappo.get("route_feature_mode", "original_option_features_v1"))
        except (ValueError, TypeError) as exc:
            raise CheckpointError("invalid unified Route seed identity") from exc
        if mapping.get("route_decoder_identity") != expected_identity:
            raise CheckpointError("unified Route decoder/feature identity mismatch")
    if 'e16_provenance' in mapping:
        if 'training_state' not in mapping:
            raise CheckpointError('E16 warm-start is not a resumable/final training checkpoint')
        expected_fields += ('e16_provenance',)
        p = mapping['e16_provenance']
        if (mapping['config_snapshot']['training']['mappo'].get('destination_supervision_mode') != 'offline_pretrain'
                or p.get('config_hash') != mapping['config_hash']
                or p.get('pretrain_gate', {}).get('status') != 'PASS'
                or p.get('data_gate', {}).get('status') != 'PASS'
                or p.get('factual_transitions_since_warmstart') != mapping['training_state']['collected_environment_transitions']
                or any(p.get(k) != 0 for k in ('destination_oracle_calls', 'destination_shadow_transitions',
                                             'online_destination_ce', 'offline_replay_ce'))):
            raise CheckpointError('invalid E16 checkpoint provenance')
        for k in ('base_checkpoint_sha256', 'dataset_hash', 'scorer_pretrain_hash'):
            if not isinstance(p.get(k), str) or len(p[k]) != 64:
                raise CheckpointError('invalid E16 provenance digest')
    _exact_keys(mapping, expected_fields, "checkpoint")
    schema = mapping["schema_version"]
    if isinstance(schema, bool) or schema != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError("checkpoint schema version mismatch")
    kind = mapping["checkpoint_kind"]
    if kind not in CHECKPOINT_V1_KINDS:
        raise CheckpointError("unknown checkpoint kind")
    if mapping["method_id"] != "ca_gat_mappo":
        raise CheckpointError("checkpoint method_id mismatch")
    if not isinstance(mapping["git_commit"], str) or not mapping["git_commit"]:
        raise CheckpointError("checkpoint git_commit is invalid")
    if not isinstance(mapping["config_hash"], str) or len(mapping["config_hash"]) != 64:
        raise CheckpointError("checkpoint config_hash is invalid")
    snapshot = _mapping(mapping["config_snapshot"], "config_snapshot")
    metadata = _mapping(
        snapshot.get("_metadata"), "config_snapshot._metadata"
    )
    if metadata.get("config_hash") != mapping["config_hash"]:
        raise CheckpointError("config snapshot hash metadata mismatch")
    if metadata.get("git_commit") != mapping["git_commit"]:
        raise CheckpointError("config snapshot git metadata mismatch")

    runtime = _mapping(mapping["runtime_provenance"], "runtime_provenance")
    _exact_keys(runtime, CHECKPOINT_V1_RUNTIME_PROVENANCE_FIELDS, "runtime_provenance")
    if runtime["device_type"] not in {"cpu", "cuda"}:
        raise CheckpointError("runtime device type is invalid")
    if runtime["dtype"] != "torch.float32":
        raise CheckpointError("checkpoint model dtype mismatch")

    training = _mapping(mapping["training_state"], "training_state")
    _exact_keys(training, CHECKPOINT_V1_TRAINING_STATE_FIELDS, "training_state")
    integer_fields = CHECKPOINT_V1_TRAINING_STATE_FIELDS[:8]
    for name in integer_fields:
        if name == "active_rollout_policy_version":
            continue
        _non_negative_int(training[name], f"training_state.{name}")
    if not isinstance(training["training_complete"], bool):
        raise CheckpointError("training_complete must be boolean")
    active_version = training["active_rollout_policy_version"]
    if active_version is not None:
        _non_negative_int(active_version, "active_rollout_policy_version")
    if training["policy_version"] != MAPPO_INITIAL_POLICY_VERSION + training["ppo_update_count"]:
        raise CheckpointError("policy version and PPO update count differ")
    if training["optimized_transitions"] != (
        training["ppo_update_count"] * 256
    ):
        raise CheckpointError("optimized transition accounting is invalid")
    if training["started_episodes"] != training["completed_episodes"]:
        raise CheckpointError("checkpoint must be at a completed episode boundary")
    if training["next_episode_index"] != training["started_episodes"]:
        raise CheckpointError("next episode index differs from started episode count")

    active = _mapping(mapping["active_rollout_state"], "active_rollout_state")
    _exact_keys(active, CHECKPOINT_V1_ACTIVE_ROLLOUT_FIELDS, "active_rollout_state")
    active_length = _non_negative_int(active["rollout_length"], "rollout_length")
    if active_length >= 256:
        raise CheckpointError("active rollout length must be less than 256")
    if active_length != training["active_rollout_length"]:
        raise CheckpointError("active rollout length differs from Trainer state")
    if active["rollout_policy_version"] != active_version:
        raise CheckpointError("active rollout policy metadata differs")
    ordered = active["ordered_transitions"]
    if not isinstance(ordered, (list, tuple)) or len(ordered) != active_length:
        raise CheckpointError("ordered transition count differs from active rollout length")
    if active_length and active_version != training["policy_version"]:
        raise CheckpointError("partial rollout does not use the current policy version")
    if not active_length and active_version is not None:
        raise CheckpointError("empty rollout must not carry a policy version")
    for index, transition in enumerate(ordered):
        transition_mapping = _mapping(transition, f"ordered_transitions[{index}]")
        _exact_keys(
            transition_mapping,
            CHECKPOINT_V1_TRANSITION_FIELDS,
            f"ordered_transitions[{index}]",
        )

    model = _mapping(mapping["model_state"], "model_state")
    optimizer = _mapping(mapping["optimizer_state"], "optimizer_state")
    policy_rng = _mapping(mapping["policy_rng_state"], "policy_rng_state")
    diagnostics = _mapping(mapping["diagnostics_state"], "diagnostics_state")
    _exact_keys(model, CHECKPOINT_V1_MODEL_STATE_FIELDS, "model_state")
    source_training = snapshot.get("training")
    source_mappo = source_training.get("mappo") if isinstance(source_training, Mapping) else None
    if (isinstance(source_mappo, Mapping)
            and source_mappo.get("tx_upper_credit_mode") == "factual_team_gae_v1"):
        state = _mapping(model["critic"], "model_state.critic")
        if (source_mappo.get("tx_q_target_normalization", "none") != "none"
                or any(name.startswith("tx_upper_q.") for name in state)):
            raise CheckpointError("factual TX checkpoint must not contain Q/PopArt state")
    if (isinstance(source_mappo, Mapping)
            and source_mappo.get("tx_q_target_normalization", "none") == "popart_v1"):
        state = _mapping(model["critic"], "model_state.critic")
        prefix = "tx_upper_q.tx_q_target_norm."
        for key, expected_dtype in (("count", torch.int64), ("mean", torch.float64), ("M2", torch.float64)):
            value = state.get(prefix + key)
            if (not isinstance(value, Tensor) or value.shape != torch.Size([])
                    or value.dtype != expected_dtype or not torch.isfinite(value)):
                raise CheckpointError("E11 checkpoint requires valid TX Q normalizer state")
        count, mean, M2 = (state[prefix + k] for k in ("count", "mean", "M2"))
        if count < 0 or M2 < 0 or (count == 0 and (mean != 0 or M2 != 0)):
            raise CheckpointError("invalid TX Q normalizer moments")
    _exact_keys(optimizer, CHECKPOINT_V1_OPTIMIZER_STATE_FIELDS, "optimizer_state")
    _exact_keys(policy_rng, CHECKPOINT_V1_POLICY_RNG_STATE_FIELDS, "policy_rng_state")
    _exact_keys(
        diagnostics, CHECKPOINT_V1_DIAGNOSTICS_STATE_FIELDS, "diagnostics_state"
    )
    if policy_rng["device_type"] != runtime["device_type"]:
        raise CheckpointError("policy RNG and runtime devices differ")
    rng_tensor = policy_rng["generator_state"]
    if (
        not isinstance(rng_tensor, Tensor)
        or rng_tensor.device.type != "cpu"
        or rng_tensor.dtype != torch.uint8
        or rng_tensor.ndim != 1
        or rng_tensor.requires_grad
    ):
        raise CheckpointError("policy generator state is invalid")

    collected = training["collected_environment_transitions"]
    optimized = training["optimized_transitions"]
    unused = training["unused_final_tail_transitions"]
    if kind == CHECKPOINT_KIND_PERIODIC_RESUME:
        if training["training_complete"] or unused != 0:
            raise CheckpointError("PERIODIC_RESUME cannot be complete or have unused tail")
        if collected != optimized + active_length:
            raise CheckpointError("periodic transition accounting is invalid")
    else:
        if not training["training_complete"] or active_length != 0:
            raise CheckpointError("FINAL_COMPLETED must be complete with empty rollout")
        if collected != optimized + unused:
            raise CheckpointError("final transition accounting is invalid")

    _validate_structured(mapping)
    if config is not None:
        expected_kind = mappo_checkpoint_kind_at(config, collected)
        if expected_kind != kind:
            raise CheckpointError("checkpoint kind differs from configured schedule")
        expected_length = compute_mappo_checkpoint_active_rollout_length(
            config, collected
        )
        if expected_length != active_length:
            raise CheckpointError("active rollout length differs from configured geometry")
    if "continuation" in metadata:
        from .ca_gat_mappo_continuation import validate_continuation_metadata

        validate_continuation_metadata(mapping)
    return mapping


def validate_periodic_checkpoint_compatibility(
    config: RunConfig,
    payload: Any,
    *,
    dtype: torch.dtype,
    cuda_available: bool,
    extend_training_budget: bool = False,
) -> Mapping[str, Any]:
    """Run every strict compatibility check before model construction/mutation."""

    mapping = validate_checkpoint_payload(payload)
    validation_config = config
    if extend_training_budget:
        from .ca_gat_mappo_continuation import budget_extension_source_config

        validation_config = budget_extension_source_config(config, mapping)
    runtime = mapping["runtime_provenance"]
    source_snapshot = _mapping(mapping["config_snapshot"], "config_snapshot")
    source_training = source_snapshot.get("training")
    source_mappo = (
        source_training.get("mappo")
        if isinstance(source_training, Mapping)
        else None
    )
    source_actor_ratio_mode = (
        source_mappo.get("actor_ratio_mode")
        if isinstance(source_mappo, Mapping)
        else None
    )
    source_agent_credit_mode = (
        source_mappo.get("agent_credit_mode")
        if isinstance(source_mappo, Mapping)
        else None
    )
    source_route_decoder_mode = (
        source_mappo.get("route_decoder_mode")
        if isinstance(source_mappo, Mapping)
        else None
    )
    source_tx_decoder_mode = (
        source_mappo.get("tx_decoder_mode")
        if isinstance(source_mappo, Mapping)
        else None
    )
    source_route_credit_mode = (
        source_mappo.get("route_credit_mode")
        if isinstance(source_mappo, Mapping)
        else None
    )
    try:
        validate_mappo_checkpoint_resume_compatibility(
            validation_config,
            schema_version=mapping["schema_version"],
            checkpoint_kind=mapping["checkpoint_kind"],
            method_id=mapping["method_id"],
            git_commit=mapping["git_commit"],
            config_hash=mapping["config_hash"],
            training_device=runtime["device_type"],
            cuda_available=cuda_available,
            checkpoint_actor_ratio_mode=source_actor_ratio_mode,
            checkpoint_agent_credit_mode=source_agent_credit_mode,
            checkpoint_route_decoder_mode=source_route_decoder_mode,
            checkpoint_tx_decoder_mode=source_tx_decoder_mode,
            checkpoint_cpu_frequency_decoder_mode=(
                source_mappo.get("cpu_frequency_decoder_mode")
                if isinstance(source_mappo, Mapping) else None
            ),
            checkpoint_route_credit_mode=source_route_credit_mode,
        )
    except ConfigError as exc:
        raise CheckpointError(str(exc)) from exc
    validate_checkpoint_payload(mapping, config=config)
    resolved_snapshot = canonicalize_config_identity(mapping["config_snapshot"])
    if resolved_snapshot != validation_config.resolved_dict():
        raise CheckpointError("checkpoint config snapshot differs from RunConfig")
    if runtime["dtype"] != str(dtype):
        raise CheckpointError("checkpoint model dtype mismatch")
    return mapping


def restore_model_optimizer_and_rng(
    *,
    config: RunConfig,
    payload: Mapping[str, Any],
    actor: nn.Module,
    critic: nn.Module,
    optimizers: CAGATMAPPOOptimizerBundle,
    dtype: torch.dtype,
    extend_training_budget: bool = False,
) -> torch.Generator:
    """Restore model, two Adam states, and the exact policy generator state."""

    mapping = validate_periodic_checkpoint_compatibility(
        config,
        payload,
        dtype=dtype,
        cuda_available=torch.cuda.is_available(),
        extend_training_budget=extend_training_budget,
    )
    if not isinstance(optimizers, CAGATMAPPOOptimizerBundle):
        raise CheckpointError("resume requires the two formal Adam optimizers")
    optimizers.validate(actor, critic, config)
    actor_device, actor_dtype = _module_device_dtype(actor, "actor")
    critic_device, critic_dtype = _module_device_dtype(critic, "critic")
    if (
        actor_device != critic_device
        or actor_device.type != config.training.mappo.training_device
        or actor_dtype != dtype
        or critic_dtype != dtype
    ):
        raise CheckpointError("resume target model device or dtype mismatch")
    device = actor_device
    model_state = mapping["model_state"]
    optimizer_state = mapping["optimizer_state"]
    actor_optimizer_state = _normalize_legacy_actor_optimizer_state(
        optimizers.actor_optimizer, optimizer_state["actor_adam"],
        optimizers.actor_parameters, config,
    )
    _validate_module_state(actor, model_state["actor"], "actor")
    _validate_module_state(critic, model_state["critic"], "critic")
    _validate_optimizer_state(
        optimizers.actor_optimizer,
        actor_optimizer_state,
        optimizers.actor_parameters,
        "actor_adam",
    )
    _validate_optimizer_state(
        optimizers.critic_optimizer,
        optimizer_state["critic_adam"],
        optimizers.critic_parameters,
        "critic_adam",
    )
    generator = restore_policy_rng(mapping["policy_rng_state"], device.type)
    try:
        actor.load_state_dict(model_state["actor"], strict=True)
        critic.load_state_dict(model_state["critic"], strict=True)
        optimizers.actor_optimizer.load_state_dict(actor_optimizer_state)
        optimizers.critic_optimizer.load_state_dict(optimizer_state["critic_adam"])
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint state restore failed: {exc}") from exc
    move_optimizer_state_to_device(optimizers.actor_optimizer, device)
    move_optimizer_state_to_device(optimizers.critic_optimizer, device)
    _validate_optimizer_device(optimizers.actor_optimizer, device, "actor_adam")
    _validate_optimizer_device(optimizers.critic_optimizer, device, "critic_adam")
    optimizers.validate(actor, critic, config)
    actor_device, actor_dtype = _module_device_dtype(actor, "actor")
    critic_device, critic_dtype = _module_device_dtype(critic, "critic")
    if (
        actor_device != device
        or critic_device != device
        or actor_dtype != dtype
        or critic_dtype != dtype
    ):
        raise CheckpointError("restored model device or dtype mismatch")
    return generator


def atomic_save_checkpoint(payload: Any, path: str | Path) -> Path:
    """Atomically save one immutable payload using the frozen fsync sequence."""

    validate_checkpoint_payload(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise CheckpointError(f"checkpoint already exists: {target}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            raise CheckpointError(f"checkpoint already exists: {target}")
        os.replace(temporary_path, target)
        temporary_path = None
        if not target.is_file():
            raise CheckpointError("atomic checkpoint save did not create final file")
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(f"atomic checkpoint save failed: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return target


def load_checkpoint_payload(path: str | Path) -> Mapping[str, Any]:
    """Load exactly one explicit path; corruption never triggers fallback."""

    target = Path(path)
    if not target.is_file():
        raise CheckpointError(f"checkpoint file does not exist: {target}")
    try:
        with target.open("rb") as handle:
            try:
                payload = torch.load(handle, map_location="cpu", weights_only=True)
            except TypeError:
                handle.seek(0)
                payload = torch.load(handle, map_location="cpu")
    except Exception as exc:
        raise CheckpointError(f"checkpoint load failed: {exc}") from exc
    return validate_checkpoint_payload(payload)


__all__ = [
    "CheckpointError",
    "atomic_save_checkpoint",
    "build_checkpoint_payload",
    "load_checkpoint_payload",
    "move_optimizer_state_to_device",
    "restore_active_rollout",
    "restore_model_optimizer_and_rng",
    "restore_policy_rng",
    "restore_rollout_transition",
    "runtime_provenance",
    "serialize_active_rollout",
    "serialize_policy_rng",
    "serialize_rollout_transition",
    "validate_checkpoint_payload",
    "validate_periodic_checkpoint_compatibility",
]
