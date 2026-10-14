"""Strict actor-only loading from immutable FINAL_COMPLETED checkpoints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ..config import (
    ActorRatioMode,
    AgentCreditMode,
    CHECKPOINT_KIND_FINAL_COMPLETED,
    CHECKPOINT_KIND_PERIODIC_RESUME,
    RouteDecoderMode,
    TxDecoderMode,
    RunConfig,
    canonical_config_identity_hash,
    canonicalize_config_identity,
)
from ..models.ca_gat_mappo import CAGATMAPPOActor, MAPPOTensorSpec
from ..models.ca_gat_mappo_checkpoint import CheckpointError, load_checkpoint_payload


EVALUATION_DTYPE = torch.float32
_ARCHITECTURE_FIELDS = (
    "encoder_hidden_dimension",
    "gat_layer_count",
    "attention_head_count",
    "gru_hidden_dimension",
    "route_decoder_mode",
    "route_feature_mode",
    "tx_decoder_mode",
    "cpu_frequency_decoder_mode",
    "route_private_backbone",
    "tx_upper_decoder_mode",
    "cpu_scheduler_mode",
    "cpu_anti_starvation_mode",
    "destination_decoder_mode",
)


class EvaluationCheckpointError(RuntimeError):
    """Raised when a checkpoint is not safe for formal actor-only evaluation."""


@dataclass(frozen=True)
class LoadedEvaluationActor:
    """Frozen actor plus the validated source identity needed by evaluation."""

    actor: CAGATMAPPOActor
    source_checkpoint_path: Path
    source_checkpoint_sha256: str
    actor_state_digest: str
    source_checkpoint_kind: str
    source_method_id: str
    source_training_config_hash: str
    source_training_config_original_hash: str
    source_training_config_raw_snapshot_hash: str
    source_training_config_canonical_hash: str
    source_checkpoint_metadata: Mapping[str, Any]
    source_training_actor_ratio_mode: str
    source_training_agent_credit_mode: str
    source_training_git_commit: str
    source_training_device: str
    evaluation_device: str
    dtype: str
    actor_architecture_identity: Mapping[str, Any]
    action_domain_identity: Mapping[str, Any]


@dataclass(frozen=True)
class _SourceConfigIdentity:
    """Auditable raw and canonical views of checkpoint config identity."""

    canonical_snapshot: Mapping[str, Any]
    checkpoint_metadata: Mapping[str, Any]
    original_hash: str
    raw_snapshot_hash: str
    canonical_hash: str


def checkpoint_sha256(path: str | Path) -> str:
    """Hash one explicit checkpoint path without globbing or fallback."""

    target = Path(path)
    if not target.is_file():
        raise EvaluationCheckpointError(
            f"evaluation checkpoint is not a regular file: {target}"
        )
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise EvaluationCheckpointError(
            f"evaluation checkpoint cannot be hashed: {exc}"
        ) from exc
    return digest.hexdigest()


def load_final_actor_for_evaluation(
    config: RunConfig,
    checkpoint_path: str | Path,
    *,
    evaluation_device: str,
) -> LoadedEvaluationActor:
    """Load an evaluation actor without changing the established freeze state."""

    return _load_final_actor_only(
        config,
        checkpoint_path,
        execution_device=evaluation_device,
        required_mode="evaluation",
        freeze_parameters=False,
    )


def load_final_actor_for_oracle_diagnostic(
    config: RunConfig,
    checkpoint_path: str | Path,
    *,
    execution_device: str,
    expected_checkpoint_sha256: str,
    expected_actor_digest: str,
) -> LoadedEvaluationActor:
    """Strictly load and freeze one FINAL_COMPLETED actor for diagnostics."""

    return _load_final_actor_only(
        config,
        checkpoint_path,
        execution_device=execution_device,
        required_mode="rl",
        freeze_parameters=True,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_actor_digest=expected_actor_digest,
    )


def load_actor_for_development_diagnostic(
    config: RunConfig, checkpoint_path: str | Path, *, evaluation_device: str,
) -> LoadedEvaluationActor:
    """Explicit opt-in actor-only load of periodic/final development snapshots."""
    return _load_final_actor_only(
        config, checkpoint_path, execution_device=evaluation_device,
        required_mode="evaluation", freeze_parameters=True,
        allow_periodic_diagnostic=True,
    )


def _load_final_actor_only(
    config: RunConfig,
    checkpoint_path: str | Path,
    *,
    execution_device: str,
    required_mode: str,
    freeze_parameters: bool,
    expected_checkpoint_sha256: str | None = None,
    expected_actor_digest: str | None = None,
    allow_periodic_diagnostic: bool = False,
) -> LoadedEvaluationActor:
    """Load only the actor state from one strict FINAL_COMPLETED payload.

    The full structured payload is schema-validated because Checkpoint V1 is a
    single immutable file.  Critic, optimizer, policy RNG, partial rollout,
    Trainer counters, and diagnostics are never reconstructed or restored.
    """

    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    config.validate()
    if config.mode != required_mode or config.method_id != "ca_gat_mappo":
        raise EvaluationCheckpointError(
            f"actor-only loading requires {required_mode}/ca_gat_mappo"
        )
    device = _validate_evaluation_device(execution_device)
    target = Path(checkpoint_path)
    source_hash_before = checkpoint_sha256(target)
    if expected_checkpoint_sha256 is not None:
        _validate_sha256(expected_checkpoint_sha256, "expected checkpoint SHA256")
        if source_hash_before != expected_checkpoint_sha256:
            raise EvaluationCheckpointError(
                "checkpoint SHA256 does not match the expected digest"
            )
    try:
        if allow_periodic_diagnostic and config.training.mappo.destination_supervision_mode == 'offline_pretrain':
            from ..models.destination_pretraining import load_warmstart, WARMSTART
            raw = torch.load(target, map_location='cpu', weights_only=True)
            payload = (load_warmstart(target, config) if raw.get('checkpoint_kind') == WARMSTART
                       else load_checkpoint_payload(target))
            if 'e16_provenance' not in payload:
                raise EvaluationCheckpointError('E16 checkpoint requires pretraining provenance')
        else:
            payload = load_checkpoint_payload(target)
    except (CheckpointError, ValueError, KeyError, TypeError) as exc:
        raise EvaluationCheckpointError(str(exc)) from exc
    source_hash_after = checkpoint_sha256(target)
    if source_hash_before != source_hash_after:
        raise EvaluationCheckpointError(
            "evaluation checkpoint changed while it was being loaded"
        )

    allowed_kinds = {CHECKPOINT_KIND_FINAL_COMPLETED}
    if allow_periodic_diagnostic:
        allowed_kinds.add(CHECKPOINT_KIND_PERIODIC_RESUME)
        if config.training.mappo.destination_supervision_mode == 'offline_pretrain':
            allowed_kinds.add('E16_PRETRAINED_WARMSTART')
    if payload["checkpoint_kind"] not in allowed_kinds:
        raise EvaluationCheckpointError(
            "formal evaluation requires checkpoint kind FINAL_COMPLETED; "
            f"got {payload['checkpoint_kind']!r}"
        )
    if payload["method_id"] != "ca_gat_mappo":
        raise EvaluationCheckpointError("checkpoint method_id mismatch")

    source_identity = _validate_source_config_identity(payload)
    resolved_source = source_identity.canonical_snapshot
    _validate_source_compatibility(config, resolved_source)
    source_training = resolved_source.get("training")
    source_mappo = (
        source_training.get("mappo")
        if isinstance(source_training, Mapping)
        else None
    )
    raw_ratio_mode = (
        source_mappo.get("actor_ratio_mode", ActorRatioMode.JOINT.value)
        if isinstance(source_mappo, Mapping)
        else ActorRatioMode.JOINT.value
    )
    try:
        source_actor_ratio_mode = ActorRatioMode(raw_ratio_mode).value
    except (TypeError, ValueError) as exc:
        raise EvaluationCheckpointError(
            "checkpoint training actor_ratio_mode is invalid"
        ) from exc
    raw_credit_mode = (
        source_mappo.get("agent_credit_mode", AgentCreditMode.TEAM.value)
        if isinstance(source_mappo, Mapping)
        else AgentCreditMode.TEAM.value
    )
    try:
        source_agent_credit_mode = AgentCreditMode(raw_credit_mode).value
    except (TypeError, ValueError) as exc:
        raise EvaluationCheckpointError(
            "checkpoint training agent_credit_mode is invalid"
        ) from exc
    runtime = payload["runtime_provenance"]
    source_device = str(runtime["device_type"])
    if source_device not in {"cpu", "cuda"}:
        raise EvaluationCheckpointError("checkpoint training device is invalid")
    if runtime["dtype"] != str(EVALUATION_DTYPE):
        raise EvaluationCheckpointError("checkpoint actor dtype is not torch.float32")

    source_git = str(payload["git_commit"])
    evaluator_git = str(config.git_commit)
    if not source_git or source_git == "unknown":
        raise EvaluationCheckpointError("checkpoint training git provenance is unavailable")
    if not evaluator_git or evaluator_git == "unknown":
        raise EvaluationCheckpointError("evaluator git provenance is unavailable")

    actor = CAGATMAPPOActor(config).to(device=device, dtype=EVALUATION_DTYPE)
    actor_state = payload["model_state"]["actor"]
    try:
        actor.load_state_dict(actor_state, strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise EvaluationCheckpointError(
            f"actor architecture/state is incompatible: {exc}"
        ) from exc
    actor.eval()
    if actor.training:
        raise EvaluationCheckpointError("actor did not enter evaluation mode")
    if freeze_parameters:
        actor.requires_grad_(False)

    from .route_oracle import actor_state_digest

    loaded_actor_digest = actor_state_digest(actor)
    if expected_actor_digest is not None:
        _validate_sha256(expected_actor_digest, "expected actor digest")
        if loaded_actor_digest != expected_actor_digest:
            raise EvaluationCheckpointError(
                "loaded actor digest does not match the expected digest"
            )
    if freeze_parameters and any(
        parameter.requires_grad for parameter in actor.parameters()
    ):
        raise EvaluationCheckpointError("diagnostic actor parameters are not frozen")

    tensor_spec = MAPPOTensorSpec.from_config(config)
    tensor_identity = asdict(tensor_spec)
    if not tensor_spec.route_private_backbone:
        tensor_identity.pop("route_private_backbone")
    architecture_identity = {
        "tensor_spec": _jsonable(tensor_identity),
        "network_config": {
            name: getattr(config.training.mappo, name)
            for name in _ARCHITECTURE_FIELDS
            if (name != "route_private_backbone" or tensor_spec.route_private_backbone)
            and (name not in {"tx_upper_decoder_mode", "cpu_scheduler_mode"} or getattr(config.training.mappo, name) not in {"legacy", "legacy_hierarchical"})
            and (name != "cpu_anti_starvation_mode" or getattr(config.training.mappo, name) != "off")
            and (name != "route_feature_mode" or getattr(config.training.mappo, name) != "original_option_features_v1")
        },
    }
    action_identity = _action_domain_identity(config)
    return LoadedEvaluationActor(
        actor=actor,
        source_checkpoint_path=target.resolve(),
        source_checkpoint_sha256=source_hash_before,
        actor_state_digest=loaded_actor_digest,
        source_checkpoint_kind=str(payload["checkpoint_kind"]),
        source_method_id=str(payload["method_id"]),
        source_training_config_hash=str(payload["config_hash"]),
        source_training_config_original_hash=source_identity.original_hash,
        source_training_config_raw_snapshot_hash=source_identity.raw_snapshot_hash,
        source_training_config_canonical_hash=source_identity.canonical_hash,
        source_checkpoint_metadata=source_identity.checkpoint_metadata,
        source_training_actor_ratio_mode=source_actor_ratio_mode,
        source_training_agent_credit_mode=source_agent_credit_mode,
        source_training_git_commit=source_git,
        source_training_device=source_device,
        evaluation_device=device.type,
        dtype=str(EVALUATION_DTYPE),
        actor_architecture_identity=architecture_identity,
        action_domain_identity=action_identity,
    )


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise EvaluationCheckpointError(f"{name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise EvaluationCheckpointError(
            f"{name} must be a SHA-256 hex digest"
        ) from exc


def _validate_evaluation_device(value: str) -> torch.device:
    if value not in {"cpu", "cuda"}:
        raise EvaluationCheckpointError(
            "evaluation device must be explicitly specified as 'cpu' or 'cuda'"
        )
    if value == "cuda" and not torch.cuda.is_available():
        raise EvaluationCheckpointError(
            "evaluation device 'cuda' was requested but CUDA is unavailable; "
            "silent CPU fallback is forbidden"
        )
    return torch.device(value)


def _validate_source_config_identity(
    payload: Mapping[str, Any],
) -> _SourceConfigIdentity:
    snapshot = payload["config_snapshot"]
    if not isinstance(snapshot, Mapping):
        raise EvaluationCheckpointError("checkpoint config snapshot is invalid")
    metadata = snapshot.get("_metadata")
    if not isinstance(metadata, Mapping):
        raise EvaluationCheckpointError("checkpoint config metadata is invalid")
    original_hash = str(payload["config_hash"])
    _validate_sha256(original_hash, "checkpoint training config hash")
    raw_snapshot = {
        key: value for key, value in snapshot.items() if key != "_metadata"
    }
    raw_snapshot_hash = _config_mapping_hash(raw_snapshot)
    canonical_snapshot = canonicalize_config_identity(raw_snapshot)
    canonical_hash = canonical_config_identity_hash(canonical_snapshot)
    if canonical_hash != original_hash:
        raise EvaluationCheckpointError(
            "checkpoint training config hash does not match its canonical snapshot"
        )
    if metadata.get("config_hash") != original_hash:
        raise EvaluationCheckpointError(
            "checkpoint training config hash metadata is inconsistent"
        )
    if metadata.get("git_commit") != payload["git_commit"]:
        raise EvaluationCheckpointError(
            "checkpoint git provenance differs from its config snapshot"
        )
    if (
        canonical_snapshot.get("mode") != "rl"
        or canonical_snapshot.get("method_id") != "ca_gat_mappo"
    ):
        raise EvaluationCheckpointError(
            "FINAL_COMPLETED source config is not an rl/ca_gat_mappo run"
        )
    return _SourceConfigIdentity(
        canonical_snapshot=canonical_snapshot,
        checkpoint_metadata=_jsonable(metadata),
        original_hash=original_hash,
        raw_snapshot_hash=raw_snapshot_hash,
        canonical_hash=canonical_hash,
    )


def _config_mapping_hash(config: Mapping[str, Any]) -> str:
    """Hash one mapping exactly as stored, without identity normalization."""

    serialized = json.dumps(
        _jsonable(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_source_compatibility(
    config: RunConfig,
    source: Mapping[str, Any],
) -> None:
    if source.get("scenario_id") != config.scenario_id:
        raise EvaluationCheckpointError("checkpoint scenario_id mismatch")
    current_environment = config.resolved_dict()["environment"]
    current_action = _jsonable(asdict(config.action))
    if source.get("environment") != current_environment:
        raise EvaluationCheckpointError(
            "checkpoint environment/observation domain is incompatible"
        )
    if source.get("action") != current_action:
        raise EvaluationCheckpointError("checkpoint action domain is incompatible")
    current_reward = config.resolved_dict().get("reward")
    if source.get("reward") != current_reward:
        raise EvaluationCheckpointError("checkpoint reward provenance mismatch")
    source_training = source.get("training")
    if not isinstance(source_training, Mapping):
        raise EvaluationCheckpointError("checkpoint training config is invalid")
    source_mappo = source_training.get("mappo")
    if not isinstance(source_mappo, Mapping):
        raise EvaluationCheckpointError("checkpoint MAPPO config is invalid")
    for name in _ARCHITECTURE_FIELDS:
        source_value = source_mappo.get(name)
        if name == "route_decoder_mode" and source_value is None:
            source_value = RouteDecoderMode.LEGACY.value
        if name == "route_feature_mode" and source_value is None:
            source_value = "original_option_features_v1"
        if name == "tx_decoder_mode" and source_value is None:
            source_value = TxDecoderMode.LEGACY.value
        if name == "cpu_frequency_decoder_mode" and source_value is None:
            source_value = "legacy"
        if name == "destination_decoder_mode" and source_value is None:
            source_value = "legacy"
        if name in {"tx_upper_decoder_mode", "cpu_scheduler_mode"} and source_value is None:
            source_value = "legacy" if name == "tx_upper_decoder_mode" else "legacy_hierarchical"
        if name == "cpu_anti_starvation_mode" and source_value is None:
            source_value = "off"
        if name == "route_private_backbone" and source_value is None:
            source_value = False
        target_value = getattr(config.training.mappo, name)
        if source_value != target_value:
            raise EvaluationCheckpointError(
                f"checkpoint actor architecture mismatch: {name}"
            )


def _action_domain_identity(config: RunConfig) -> Mapping[str, Any]:
    return {
        "uav_count": config.environment.uav_count,
        "resource_group_count": config.environment.resource_group_count,
        "sampling_order": list(config.action.sampling_order),
        "route_fixed_actions": list(config.action.route_fixed_actions),
        "tx_select_idle_action": config.action.tx_select_idle_action,
        "resource_width_options": list(config.action.resource_width_options),
        "power_levels": list(config.action.power_levels),
        "cpu_queue_idle_action": config.action.cpu_queue_idle_action,
        "cpu_frequency_levels": list(config.action.cpu_frequency_levels),
        "canonical_inactive_values": _jsonable(
            dict(config.action.canonical_inactive_values)
        ),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "EVALUATION_DTYPE",
    "EvaluationCheckpointError",
    "LoadedEvaluationActor",
    "checkpoint_sha256",
    "load_final_actor_for_evaluation",
    "load_final_actor_for_oracle_diagnostic",
]
