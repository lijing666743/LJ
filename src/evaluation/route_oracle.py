"""Oracle-1 route counterfactual infrastructure.

This module is deliberately separate from actor-safe telemetry. It owns only
detached shadow rollouts, frozen-policy inference, route classification, and
the dedicated Oracle-1 JSONL sidecar.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from ..config import RunConfig
from ..env.actions import ActionProposal
from ..env.environment import StepResult, U2UMECEnvironment
from ..env.tasks import TaskOutcome
from ..models.ca_gat_mappo import ActorObservationTensorizer, CAGATMAPPOActor
from ..models.ca_gat_mappo_actions import (
    CAGATMAPPOActionDistribution,
    SequentialActionMaskBatch,
)


ORACLE_SCHEMA_VERSION = 1
ORACLE_SIDECAR = "route_oracle_counterfactual_v1"
ORACLE_SEMANTICS = "ROUTE_COUNTERFACTUAL_UNDER_FROZEN_POLICY"
ORACLE1_BEST_ROUTE_SCOPE = "EVALUATED_ORACLE1_BRANCHES_ONLY"
COMMITTED_INTERVENTION_SET = "Local + all legal Remote-j"
VALID_BRANCH_STATUS = "VALID"
INVALID_BRANCH_STATUS = "INVALID"
UNRESOLVED_BRANCH_STATUS = "UNRESOLVED"
YES = "YES"
NO = "NO"
UNRESOLVED = "UNRESOLVED"
ATTRIBUTION_EXACT = "EXACT"
ATTRIBUTION_ALLOCATED_PROXY = "ALLOCATED_PROXY"
ATTRIBUTION_NOT_AVAILABLE = "NOT_AVAILABLE"


class OracleDiagnosticError(RuntimeError):
    """Base class for an Oracle-1 reliability or artifact failure."""


class OracleReliabilityError(OracleDiagnosticError):
    """Raised when a fidelity, CRN, legality, or terminal join gate fails."""


class OracleArtifactError(OracleDiagnosticError):
    """Raised when the dedicated Oracle sidecar cannot be written or read."""


class OracleBudgetExhausted(OracleDiagnosticError):
    """Raised when a bounded feasibility collection cannot reserve work."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OracleDiagnosticError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise OracleDiagnosticError(f"{name} must be finite")
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def tensor_digest(value: Tensor) -> str:
    """Digest a detached tensor without exposing its contents in the sidecar."""

    tensor = value.detach().cpu().contiguous()
    if torch.is_floating_point(tensor) or torch.is_complex(tensor):
        if not torch.isfinite(tensor).all():
            raise OracleReliabilityError("tensor contains a non-finite value")
    payload = tensor.numpy().tobytes()
    header = {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "nbytes": len(payload)}
    return hashlib.sha256(_canonical_json(header).encode("utf-8") + payload).hexdigest()


def actor_state_digest(actor: CAGATMAPPOActor) -> str:
    """Return a stable digest over actor parameters and persistent buffers."""

    if not isinstance(actor, CAGATMAPPOActor):
        raise TypeError("actor must be a CAGATMAPPOActor")
    digest = hashlib.sha256()
    for kind, named in (("parameter", actor.named_parameters()), ("buffer", actor.named_buffers())):
        for name, value in sorted(named):
            tensor = value.detach().cpu().contiguous()
            if kind == "parameter" and not torch.isfinite(tensor).all():
                raise OracleReliabilityError(f"actor parameter {name!r} is non-finite")
            digest.update(_canonical_json({"kind": kind, "name": name}).encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(_canonical_json(list(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenRunnerIdentity:
    """Mathematical identity used to decide frozen-runner reuse."""

    policy_version: int
    actor_state_digest: str

    def __post_init__(self) -> None:
        if isinstance(self.policy_version, bool) or not isinstance(self.policy_version, int) or self.policy_version < 0:
            raise ValueError("policy_version must be a non-negative integer")
        if not isinstance(self.actor_state_digest, str) or len(self.actor_state_digest) != 64:
            raise ValueError("actor_state_digest must be a SHA-256 hex digest")
        int(self.actor_state_digest, 16)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "actor_state_digest": self.actor_state_digest,
        }


@dataclass(frozen=True)
class FrozenRunnerProvenance:
    """Non-mathematical source metadata attached to a frozen policy record."""

    checkpoint_sha256: str | None = None
    source: str = "trainer_policy_snapshot"
    git_commit: str | None = None
    config_hash: str | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("checkpoint_sha256", "git_commit", "config_hash"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{name} must be a non-empty string or None")
                if name != "git_commit" and len(value) != 64:
                    raise ValueError(f"{name} must be a SHA-256 hex digest")
                if name != "git_commit":
                    int(value, 16)
        if self.run_id is not None and (not isinstance(self.run_id, str) or not self.run_id):
            raise ValueError("run_id must be a non-empty string or None")
        if not self.source:
            raise ValueError("policy identity source cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "source": self.source,
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "run_id": self.run_id,
        }


@dataclass(frozen=True)
class FrozenPolicyIdentity:
    """Policy identity plus provenance; only runner_identity controls reuse."""

    policy_version: int
    actor_state_digest: str
    checkpoint_sha256: str | None = None
    source: str = "trainer_policy_snapshot"
    git_commit: str | None = None
    config_hash: str | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.policy_version, bool) or not isinstance(self.policy_version, int) or self.policy_version < 0:
            raise ValueError("policy_version must be a non-negative integer")
        if not isinstance(self.actor_state_digest, str) or len(self.actor_state_digest) != 64:
            raise ValueError("actor_state_digest must be a SHA-256 hex digest")
        int(self.actor_state_digest, 16)
        if self.checkpoint_sha256 is not None:
            if len(self.checkpoint_sha256) != 64:
                raise ValueError("checkpoint_sha256 must be a SHA-256 hex digest")
            int(self.checkpoint_sha256, 16)
        FrozenRunnerProvenance(
            checkpoint_sha256=self.checkpoint_sha256,
            source=self.source,
            git_commit=self.git_commit,
            config_hash=self.config_hash,
            run_id=self.run_id,
        )

    @property
    def runner_identity(self) -> FrozenRunnerIdentity:
        return FrozenRunnerIdentity(
            policy_version=self.policy_version,
            actor_state_digest=self.actor_state_digest,
        )

    @property
    def runner_provenance(self) -> FrozenRunnerProvenance:
        return FrozenRunnerProvenance(
            checkpoint_sha256=self.checkpoint_sha256,
            source=self.source,
            git_commit=self.git_commit,
            config_hash=self.config_hash,
            run_id=self.run_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "actor_state_digest": self.actor_state_digest,
            "checkpoint_sha256": self.checkpoint_sha256,
            "source": self.source,
            "runner_identity": self.runner_identity.to_dict(),
            "runner_provenance": self.runner_provenance.to_dict(),
        }


@dataclass(frozen=True)
class FrozenPolicyStep:
    """Detached deterministic proposals and hidden output for one policy step."""

    proposals: tuple[ActionProposal, ...]
    hidden_out: Tensor = field(repr=False, compare=False)
    hidden_digest: str = ""

    def __post_init__(self) -> None:
        if not self.proposals or any(not isinstance(item, ActionProposal) for item in self.proposals):
            raise OracleReliabilityError("frozen policy produced malformed proposals")
        hidden = self.hidden_out.detach().clone()
        if hidden.ndim != 3 or not torch.is_floating_point(hidden):
            raise OracleReliabilityError("frozen policy hidden output has invalid shape")
        hidden.requires_grad_(False)
        object.__setattr__(self, "hidden_out", hidden)
        digest = tensor_digest(hidden)
        if self.hidden_digest and self.hidden_digest != digest:
            raise OracleReliabilityError("hidden digest does not match hidden output")
        object.__setattr__(self, "hidden_digest", digest)


class FrozenOraclePolicyRunner:
    """Immutable, deterministic downstream policy runner for shadow branches."""

    def __init__(
        self,
        actor: CAGATMAPPOActor,
        config: RunConfig,
        policy_identity: FrozenPolicyIdentity,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(actor, CAGATMAPPOActor):
            raise TypeError("actor must be a CAGATMAPPOActor")
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        config.validate()
        if not isinstance(policy_identity, FrozenPolicyIdentity):
            raise TypeError("policy_identity must be FrozenPolicyIdentity")
        source_digest = actor_state_digest(actor)
        if source_digest != policy_identity.actor_state_digest:
            raise OracleReliabilityError("policy identity digest does not match actor")
        self.config = config
        self.identity = policy_identity
        self.device = torch.device(device)
        self.dtype = dtype
        frozen = copy.deepcopy(actor)
        frozen.to(device=self.device, dtype=self.dtype)
        frozen.eval()
        frozen.requires_grad_(False)
        self._actor = frozen
        self._tensorizer = ActorObservationTensorizer(config)
        self._distribution = CAGATMAPPOActionDistribution(frozen, config)
        self._frozen_digest = actor_state_digest(frozen)
        self._preprocessing_cache: dict[
            str,
            tuple[str, Any, SequentialActionMaskBatch],
        ] | None = None
        self._preprocessing_cache_limit = 0
        self._preprocessing_cache_hits = 0
        self._preprocessing_cache_misses = 0
        self._act_call_count = 0
        if self._frozen_digest != policy_identity.actor_state_digest:
            raise OracleReliabilityError("frozen actor digest changed during cloning")

    @property
    def actor(self) -> CAGATMAPPOActor:
        return self._actor

    @property
    def policy_version(self) -> int:
        return self.identity.policy_version

    @property
    def runner_identity(self) -> FrozenRunnerIdentity:
        return self.identity.runner_identity

    @property
    def runner_provenance(self) -> FrozenRunnerProvenance:
        return self.identity.runner_provenance

    @property
    def actor_state_digest(self) -> str:
        return self.identity.actor_state_digest

    @property
    def act_call_count(self) -> int:
        return self._act_call_count

    @property
    def preprocessing_cache_stats(self) -> Mapping[str, int]:
        cache = self._preprocessing_cache
        return {
            "hits": self._preprocessing_cache_hits,
            "misses": self._preprocessing_cache_misses,
            "entries": 0 if cache is None else len(cache),
        }

    def start_preprocessing_cache(self, *, max_entries: int = 4096) -> None:
        """Start a bounded observation-preprocessing cache for one diagnostic scope."""

        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("preprocessing cache max_entries must be a positive integer")
        self._preprocessing_cache = {}
        self._preprocessing_cache_limit = max_entries
        self._preprocessing_cache_hits = 0
        self._preprocessing_cache_misses = 0

    def stop_preprocessing_cache(self) -> Mapping[str, int]:
        """Return cache counters and release every cached tensor/mask object."""

        stats = dict(self.preprocessing_cache_stats)
        self._preprocessing_cache = None
        self._preprocessing_cache_limit = 0
        return stats

    def _prepared_inputs(
        self,
        observations: Sequence[Any],
        *,
        episode_start: bool,
    ) -> tuple[Any, SequentialActionMaskBatch]:
        cache = self._preprocessing_cache
        if cache is None:
            return (
                self._tensorizer.encode_step(
                    observations,
                    device=self.device,
                    dtype=self.dtype,
                    episode_start=episode_start,
                ),
                SequentialActionMaskBatch.from_observations(observations),
            )
        snapshot_json = _canonical_json(
            {
                "episode_start": bool(episode_start),
                "observations": [item.snapshot() for item in observations],
            }
        )
        cache_key = hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest()
        cached = cache.get(cache_key)
        if cached is not None:
            cached_snapshot, batch, masks = cached
            if cached_snapshot != snapshot_json:
                raise OracleReliabilityError(
                    "observation preprocessing cache digest collision"
                )
            self._preprocessing_cache_hits += 1
            return batch, masks
        self._preprocessing_cache_misses += 1
        batch = self._tensorizer.encode_step(
            observations,
            device=self.device,
            dtype=self.dtype,
            episode_start=episode_start,
        )
        masks = SequentialActionMaskBatch.from_observations(observations)
        if len(cache) >= self._preprocessing_cache_limit:
            del cache[next(iter(cache))]
        cache[cache_key] = (snapshot_json, batch, masks)
        return batch, masks

    def act(
        self,
        observations: Sequence[Any],
        hidden_in: Tensor,
        *,
        episode_start: bool = False,
    ) -> FrozenPolicyStep:
        """Run masked argmax inference without sampling or training RNG."""

        if hidden_in.requires_grad:
            raise OracleReliabilityError("frozen policy hidden input must be detached")
        self._act_call_count += 1
        hidden = hidden_in.detach().clone().to(device=self.device, dtype=self.dtype)
        batch, masks = self._prepared_inputs(
            observations,
            episode_start=episode_start,
        )
        with torch.inference_mode():
            output = self._distribution.deterministic_actions(batch, masks, hidden)
        result = FrozenPolicyStep(
            proposals=tuple(output.proposals[0][0]),
            hidden_out=output.hidden_out.detach().clone(),
        )
        if any(proposal.uav_id != index for index, proposal in enumerate(result.proposals)):
            raise OracleReliabilityError("frozen policy proposals are not in UAV order")
        if self._frozen_digest != self.identity.actor_state_digest:
            raise OracleReliabilityError("frozen policy parameters were mutated")
        return result


class FrozenOraclePolicyRunnerCache:
    """Reuse one immutable runner for each policy-version/actor-digest pair."""

    def __init__(self, config: RunConfig, *, device: torch.device | str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self._runner: FrozenOraclePolicyRunner | None = None
        self._current_identity: FrozenPolicyIdentity | None = None

    @property
    def runner(self) -> FrozenOraclePolicyRunner | None:
        return self._runner

    @property
    def current_identity(self) -> FrozenPolicyIdentity:
        if self._current_identity is None:
            raise OracleReliabilityError("frozen runner cache has no current identity")
        return self._current_identity

    def get(
        self,
        actor: CAGATMAPPOActor,
        policy_version: int,
        *,
        checkpoint_sha256: str | None = None,
        source: str = "trainer_policy_snapshot",
        git_commit: str | None = None,
        config_hash: str | None = None,
        run_id: str | None = None,
    ) -> FrozenOraclePolicyRunner:
        identity = FrozenPolicyIdentity(
            policy_version=policy_version,
            actor_state_digest=actor_state_digest(actor),
            checkpoint_sha256=checkpoint_sha256,
            source=source,
            git_commit=git_commit,
            config_hash=config_hash,
            run_id=run_id,
        )
        if self._runner is None:
            self._runner = FrozenOraclePolicyRunner(
                actor,
                self.config,
                identity,
                device=self.device,
            )
        elif self._runner.runner_identity != identity.runner_identity:
            if self._runner.runner_identity.policy_version == identity.runner_identity.policy_version:
                raise OracleReliabilityError(
                    "actor state digest changed without a policy-version update"
                )
            self._runner = FrozenOraclePolicyRunner(
                actor,
                self.config,
                identity,
                device=self.device,
            )
        self._current_identity = identity
        return self._runner


def selector_input(
    *,
    run_id: str,
    episode_id: int,
    global_environment_step: int,
    source_uav: int,
    task_id: int,
) -> dict[str, Any]:
    return {
        "schema_version": ORACLE_SCHEMA_VERSION,
        "run_id": run_id,
        "episode_id": episode_id,
        "global_environment_step": global_environment_step,
        "source_uav": source_uav,
        "task_id": task_id,
    }


def hash_select_oracle_decision(
    *,
    run_id: str,
    episode_id: int,
    global_environment_step: int,
    source_uav: int,
    task_id: int,
    selection_rate_ppm: int,
) -> bool:
    """Select without modulo bias and without consuming any training RNG."""

    if isinstance(selection_rate_ppm, bool) or not isinstance(selection_rate_ppm, int):
        raise ValueError("selection_rate_ppm must be an integer")
    if not 0 <= selection_rate_ppm <= 1_000_000:
        raise ValueError("selection_rate_ppm must be in [0, 1000000]")
    if selection_rate_ppm == 0:
        return False
    digest = hashlib.sha256(
        _canonical_json(
            selector_input(
                run_id=run_id,
                episode_id=episode_id,
                global_environment_step=global_environment_step,
                source_uav=source_uav,
                task_id=task_id,
            )
        ).encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big", signed=False)
    threshold = (1 << 64) * selection_rate_ppm // 1_000_000
    return value < threshold


@dataclass
class OracleCollectionBudget:
    """Three-way feasibility budget; no production numeric default is imposed."""

    target_selected_decisions: int | None = None
    max_factual_environment_transitions: int | None = None
    max_shadow_transitions: int | None = None
    selected_decisions: int = 0
    factual_environment_transitions: int = 0
    shadow_transitions: int = 0
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "target_selected_decisions",
            "max_factual_environment_transitions",
            "max_shadow_transitions",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        for name in (
            "selected_decisions",
            "factual_environment_transitions",
            "shadow_transitions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def observe_factual_transitions(self, count: int = 1) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("factual transition increment must be non-negative")
        self.factual_environment_transitions += count
        self._refresh_stop_reason()

    def record_selected_decision(self) -> None:
        if not self.can_start_decision():
            raise OracleBudgetExhausted(self.stop_reason or "Oracle budget exhausted")
        self.complete_selected_decision()

    def can_start_decision(
        self, *, factual_transitions: int | None = None
    ) -> bool:
        """Return whether a new complete decision may be started."""

        return not self.should_stop(factual_transitions=factual_transitions)

    def complete_selected_decision(self) -> None:
        """Count one decision only after all of its branches have completed."""

        if (
            self.target_selected_decisions is not None
            and self.selected_decisions >= self.target_selected_decisions
        ):
            self.stop_reason = "TARGET_SELECTED_DECISIONS"
            raise OracleBudgetExhausted(self.stop_reason)
        self.selected_decisions += 1
        self._refresh_stop_reason()

    def reserve_shadow_transition(self) -> None:
        if (
            self.max_shadow_transitions is not None
            and self.shadow_transitions >= self.max_shadow_transitions
        ):
            self.stop_reason = "MAX_SHADOW_TRANSITIONS"
            raise OracleBudgetExhausted(self.stop_reason)
        self.shadow_transitions += 1
        self._refresh_stop_reason()

    def should_stop(self, *, factual_transitions: int | None = None) -> bool:
        if factual_transitions is not None:
            self.factual_environment_transitions = factual_transitions
        self._refresh_stop_reason()
        return self.stop_reason is not None

    def _refresh_stop_reason(self) -> None:
        if (
            self.target_selected_decisions is not None
            and self.selected_decisions >= self.target_selected_decisions
        ):
            self.stop_reason = "TARGET_SELECTED_DECISIONS"
        elif (
            self.max_factual_environment_transitions is not None
            and self.factual_environment_transitions >= self.max_factual_environment_transitions
        ):
            self.stop_reason = "MAX_FACTUAL_ENVIRONMENT_TRANSITIONS"
        elif (
            self.max_shadow_transitions is not None
            and self.shadow_transitions >= self.max_shadow_transitions
        ):
            self.stop_reason = "MAX_SHADOW_TRANSITIONS"
        else:
            self.stop_reason = None


TinyOracleFeasibilityBudget = OracleCollectionBudget


@dataclass(frozen=True)
class OracleBranchResult:
    """Compact outcome of one normal environment shadow rollout."""

    route: str | int
    role: str = "COMMITTED"
    valid: bool = True
    status: str = VALID_BRANCH_STATUS
    terminal_class: str | None = None
    terminal_slot: int | None = None
    route_slot: int | None = None
    route_to_terminal_slots: int | None = None
    completion_delay_s: float | None = None
    source_tx_energy_j: float | None = None
    focal_cpu_energy_j: float | None = None
    helper_cpu_energy_j: float | None = None
    team_active_energy_j: float | None = None
    source_tx_energy_attribution: str = ATTRIBUTION_NOT_AVAILABLE
    focal_cpu_energy_attribution: str = ATTRIBUTION_NOT_AVAILABLE
    initial_state_fingerprint: str | None = None
    initial_rng_fingerprint: str | None = None
    initial_exogenous_fingerprint: str | None = None
    initial_hidden_digest: str | None = None
    final_rng_fingerprint: str | None = None
    trace: tuple[Mapping[str, Any], ...] = ()
    exogenous_trace: tuple[str, ...] = ()
    terminal_reason: str | None = None
    focal_task_id: int | None = None

    def __post_init__(self) -> None:
        if self.status not in {
            VALID_BRANCH_STATUS,
            INVALID_BRANCH_STATUS,
            UNRESOLVED_BRANCH_STATUS,
        }:
            raise ValueError("unknown Oracle branch status")
        if not self.valid and self.status == VALID_BRANCH_STATUS:
            raise ValueError("invalid branch cannot carry VALID status")
        for name in (
            "completion_delay_s",
            "source_tx_energy_j",
            "focal_cpu_energy_j",
            "helper_cpu_energy_j",
            "team_active_energy_j",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.terminal_slot is not None and self.route_slot is not None:
            if self.route_to_terminal_slots != self.terminal_slot - self.route_slot:
                raise ValueError("route-to-terminal slot count is inconsistent")

    @property
    def is_completed(self) -> bool:
        return self.valid and self.terminal_class == TaskOutcome.DONE.value

    def to_record(self, decision: "OracleDecisionResult") -> dict[str, Any]:
        return {
            "record_type": "oracle_branch",
            "schema_version": ORACLE_SCHEMA_VERSION,
            "decision_key": decision.decision_key,
            "source_uav": decision.source_uav,
            "task_id": decision.task_id,
            "route_slot": decision.route_slot,
            "factual_route": decision.factual_route,
            "branch_route": self.route,
            "branch_role": self.role,
            "legal_committed_route_set": list(decision.committed_routes),
            "evaluated_route_set": list(decision.evaluated_route_set),
            "initial_state_fingerprint": self.initial_state_fingerprint,
            "initial_rng_fingerprint": self.initial_rng_fingerprint,
            "initial_exogenous_fingerprint": self.initial_exogenous_fingerprint,
            "policy_identity": decision.policy_identity.to_dict(),
            "initial_hidden_digest": self.initial_hidden_digest,
            "branch_valid": self.valid,
            "branch_validity_status": self.status,
            "terminal_class": self.terminal_class,
            "terminal_slot": self.terminal_slot,
            "route_to_terminal_slots": self.route_to_terminal_slots,
            "completion_delay_s": self.completion_delay_s,
            "source_tx_energy_j": self.source_tx_energy_j,
            "source_tx_energy_attribution": self.source_tx_energy_attribution,
            "focal_cpu_energy_j": self.focal_cpu_energy_j,
            "focal_cpu_energy_attribution": self.focal_cpu_energy_attribution,
            "helper_cpu_energy_j": self.helper_cpu_energy_j,
            "team_active_energy_j": self.team_active_energy_j,
            "terminal_reason": self.terminal_reason,
            "final_rng_fingerprint": self.final_rng_fingerprint,
            "trace_step_count": len(self.trace),
            "exogenous_trace_count": len(self.exogenous_trace),
            "focal_task_id": self.focal_task_id,
        }


def _factual_branch_metadata(
    branches: Sequence[OracleBranchResult],
    factual_route: str | int,
    committed_routes: Sequence[str | int],
) -> tuple[str, bool]:
    branch = next((item for item in branches if item.route == factual_route), None)
    if branch is None or branch.status == UNRESOLVED_BRANCH_STATUS:
        validity = UNRESOLVED_BRANCH_STATUS
    elif branch.valid and branch.status == VALID_BRANCH_STATUS:
        validity = VALID_BRANCH_STATUS
    else:
        validity = INVALID_BRANCH_STATUS
    return validity, factual_route in committed_routes


@dataclass(frozen=True)
class OracleDecisionResult:
    """One selected eligible decision and all evaluated Oracle-1 branches."""

    decision_key: str
    episode_id: int
    global_environment_step: int
    source_uav: int
    task_id: int
    route_slot: int
    factual_route: str | int
    committed_routes: tuple[str | int, ...]
    evaluated_route_set: tuple[str | int, ...]
    policy_identity: FrozenPolicyIdentity
    initial_state_fingerprint: str
    initial_rng_fingerprint: str
    initial_exogenous_fingerprint: str
    initial_hidden_digest: str
    branches: tuple[OracleBranchResult, ...]
    classification: Mapping[str, Any]
    selector_input: Mapping[str, Any]

    def to_decision_record(self) -> dict[str, Any]:
        record = {
            "record_type": "oracle_decision",
            "schema_version": ORACLE_SCHEMA_VERSION,
            "decision_key": self.decision_key,
            "episode_id": self.episode_id,
            "global_environment_step": self.global_environment_step,
            "source_uav": self.source_uav,
            "task_id": self.task_id,
            "route_slot": self.route_slot,
            "factual_route": self.factual_route,
            "legal_committed_route_set": list(self.committed_routes),
            "evaluated_route_set": list(self.evaluated_route_set),
            "oracle1_best_route_scope": ORACLE1_BEST_ROUTE_SCOPE,
            "initial_state_fingerprint": self.initial_state_fingerprint,
            "initial_rng_fingerprint": self.initial_rng_fingerprint,
            "initial_exogenous_fingerprint": self.initial_exogenous_fingerprint,
            "initial_hidden_digest": self.initial_hidden_digest,
            "policy_identity": self.policy_identity.to_dict(),
            "selector_input": dict(self.selector_input),
        }
        factual_validity, factual_reused = _factual_branch_metadata(
            self.branches,
            self.factual_route,
            self.committed_routes,
        )
        record["factual_route_branch_validity"] = factual_validity
        record["factual_branch_reused"] = factual_reused
        record.update(dict(self.classification))
        record["branch_count"] = len(self.branches)
        return record

    def to_records(self) -> tuple[dict[str, Any], ...]:
        return (self.to_decision_record(),) + tuple(
            branch.to_record(self) for branch in self.branches
        )


def _terminal_name(value: str | TaskOutcome | None) -> str | None:
    if isinstance(value, TaskOutcome):
        return value.value
    return value


def compare_branch_outcomes(
    left: OracleBranchResult,
    right: OracleBranchResult,
    *,
    slot_duration_s: float,
    energy_tolerance_j: float,
) -> int | None:
    """Return 1 when left is better, -1 when right is better, or None."""

    if not left.valid or not right.valid:
        return None
    left_terminal = _terminal_name(left.terminal_class)
    right_terminal = _terminal_name(right.terminal_class)
    left_done = left_terminal == TaskOutcome.DONE.value
    right_done = right_terminal == TaskOutcome.DONE.value
    if left_done != right_done:
        return 1 if left_done else -1
    if not left_done:
        return 0 if left_terminal == right_terminal else None
    if left.route_to_terminal_slots is None or right.route_to_terminal_slots is None:
        return None
    delay_delta_slots = left.route_to_terminal_slots - right.route_to_terminal_slots
    if abs(delay_delta_slots) > 1:
        return 1 if delay_delta_slots < 0 else -1
    if left.team_active_energy_j is None or right.team_active_energy_j is None:
        return None
    energy_delta = left.team_active_energy_j - right.team_active_energy_j
    if abs(energy_delta) > energy_tolerance_j:
        return 1 if energy_delta < 0.0 else -1
    if left.focal_cpu_energy_j is None or right.focal_cpu_energy_j is None:
        return 0
    focal_delta = left.focal_cpu_energy_j - right.focal_cpu_energy_j
    if abs(focal_delta) > energy_tolerance_j:
        return 1 if focal_delta < 0.0 else -1
    _ = slot_duration_s
    return 0


def _best_unique(
    branches: Sequence[OracleBranchResult],
    *,
    slot_duration_s: float,
    energy_tolerance_j: float,
) -> tuple[OracleBranchResult | None, tuple[OracleBranchResult, ...], bool]:
    valid = tuple(item for item in branches if item.valid)
    if not valid:
        return None, (), False
    unresolved = False
    non_dominated: list[OracleBranchResult] = []
    for candidate in valid:
        dominated = False
        for other in valid:
            if other is candidate:
                continue
            comparison = compare_branch_outcomes(
                other,
                candidate,
                slot_duration_s=slot_duration_s,
                energy_tolerance_j=energy_tolerance_j,
            )
            if comparison is None:
                unresolved = True
            elif comparison > 0:
                dominated = True
        if not dominated:
            non_dominated.append(candidate)
    unresolved = unresolved or len(non_dominated) != 1
    if unresolved:
        return None, tuple(non_dominated), True
    return non_dominated[0], tuple(non_dominated), False


def classify_oracle_routes(
    branches: Sequence[OracleBranchResult],
    *,
    factual_route: str | int,
    committed_routes: Sequence[str | int],
    evaluated_route_set: Sequence[str | int] | None = None,
    slot_duration_s: float = 1.0,
    energy_tolerance_j: float = 1.0e-9,
) -> dict[str, Any]:
    """Classify committed-route and factual-route comparisons separately."""

    by_route = {item.route: item for item in branches}
    committed = tuple(by_route[route] for route in committed_routes if route in by_route)
    evaluated_routes = tuple(evaluated_route_set) if evaluated_route_set is not None else tuple(by_route)
    evaluated = tuple(by_route[route] for route in evaluated_routes if route in by_route)
    local = by_route.get("local")
    remote = tuple(item for item in committed if isinstance(item.route, int) and not isinstance(item.route, bool))
    best_committed, committed_ties, committed_unresolved = _best_unique(
        committed, slot_duration_s=slot_duration_s, energy_tolerance_j=energy_tolerance_j
    )
    best_remote, remote_ties, remote_unresolved = _best_unique(
        remote, slot_duration_s=slot_duration_s, energy_tolerance_j=energy_tolerance_j
    )
    best_evaluated, evaluated_ties, evaluated_unresolved = _best_unique(
        evaluated, slot_duration_s=slot_duration_s, energy_tolerance_j=energy_tolerance_j
    )

    local_remote_comparison: int | None = None
    if local is not None and best_remote is not None:
        local_remote_comparison = compare_branch_outcomes(
            best_remote,
            local,
            slot_duration_s=slot_duration_s,
            energy_tolerance_j=energy_tolerance_j,
        )
    elif local is None and best_remote is not None:
        local_remote_comparison = 1

    factual = by_route.get(factual_route)
    strictly_better_committed = False
    factual_comparison_unresolved = False
    if factual is not None and factual.valid:
        for candidate in committed:
            if candidate.route == factual_route or not candidate.valid:
                continue
            comparison = compare_branch_outcomes(
                candidate,
                factual,
                slot_duration_s=slot_duration_s,
                energy_tolerance_j=energy_tolerance_j,
            )
            if comparison is None:
                factual_comparison_unresolved = True
            elif comparison > 0:
                strictly_better_committed = True

    if factual is None or not factual.valid or factual_comparison_unresolved:
        factual_is_best = UNRESOLVED
    else:
        factual_is_best = NO if strictly_better_committed else YES
    if factual is None or not factual.valid or factual_comparison_unresolved:
        route_policy_miss = UNRESOLVED
    elif strictly_better_committed:
        route_policy_miss = YES
    else:
        route_policy_miss = NO

    if local_remote_comparison is None:
        remote_better = local_better = tie = UNRESOLVED
    else:
        remote_better = YES if local_remote_comparison > 0 else NO
        local_better = YES if local_remote_comparison < 0 else NO
        tie = YES if local_remote_comparison == 0 else NO
    oracle_unresolved = (
        YES
        if committed_unresolved or remote_unresolved or evaluated_unresolved or local_remote_comparison is None
        else NO
    )
    return {
        "DECISION_LEVEL_BEST_ROUTE": None if best_committed is None else best_committed.route,
        "DECISION_LEVEL_BEST_ROUTE_SET": [item.route for item in committed_ties],
        "DECISION_LEVEL_BEST_REMOTE": None if best_remote is None else best_remote.route,
        "DECISION_LEVEL_BEST_REMOTE_SET": [item.route for item in remote_ties],
        "ORACLE_REMOTE_BETTER": remote_better,
        "ORACLE_LOCAL_BETTER": local_better,
        "ORACLE_TIE": tie,
        "ORACLE_UNRESOLVED": oracle_unresolved,
        "FACTUAL_ROUTE": factual_route,
        "BEST_COMMITTED_ROUTE": None if best_committed is None else best_committed.route,
        "BEST_COMMITTED_ROUTE_SET": [item.route for item in committed_ties],
        "ORACLE1_BEST_ROUTE": None if best_evaluated is None else best_evaluated.route,
        "ORACLE1_BEST_ROUTE_SET": [item.route for item in evaluated_ties],
        "FACTUAL_ROUTE_IS_BEST": factual_is_best,
        "ORACLE1_ROUTE_POLICY_MISS": route_policy_miss,
        "FACTUAL_REMOTE_ID": factual_route if isinstance(factual_route, int) and not isinstance(factual_route, bool) else None,
        "BEST_REMOTE_ID": None if best_committed is None or not isinstance(best_committed.route, int) or isinstance(best_committed.route, bool) else best_committed.route,
        "ORACLE1_BEST_ROUTE_SCOPE": ORACLE1_BEST_ROUTE_SCOPE,
        "evaluated_route_set": list(evaluated_routes),
        "committed_route_set": list(committed_routes),
    }


def _focal_task_id(observation: Any) -> int | None:
    queue = observation.private_queues.unbound
    if not bool(queue.head_valid_mask):
        return None
    if queue.head_task_id is None:
        raise OracleReliabilityError("valid focal queue head has no task ID")
    return int(queue.head_task_id)


def _route_sets(observation: Any, factual_route: str | int) -> tuple[tuple[str | int, ...], tuple[str | int, ...]]:
    masks = observation.action_masks
    if not masks.route_branch_active:
        raise OracleReliabilityError("Oracle selection requires an active route branch")
    legal_values = tuple(value for value, available in zip(masks.route_domain, masks.route_mask) if bool(available))
    if factual_route not in legal_values:
        raise OracleReliabilityError("factual route is not legal under the current route mask")
    if factual_route == "idle":
        raise OracleReliabilityError("route-active factual Idle is a reliability failure")
    remote = tuple(sorted(int(value) for value in legal_values if isinstance(value, int) and not isinstance(value, bool)))
    if "local" not in legal_values or not remote:
        raise ValueError("Oracle decision is ineligible without Local and at least one legal Remote")
    committed = ("local",) + remote
    evaluated = committed if factual_route in committed else committed + (factual_route,)
    return committed, evaluated


def _proposal_digest(proposals: Sequence[ActionProposal]) -> str:
    return _sha256_json([{"uav_id": item.uav_id, "branches": list(item.branches)} for item in proposals])


def oracle_schema_header(
    config: RunConfig,
    policy_identity: FrozenPolicyIdentity,
) -> dict[str, Any]:
    """Return the auditable first record of the isolated Oracle sidecar."""

    return {
        "record_type": "schema",
        "schema_version": ORACLE_SCHEMA_VERSION,
        "sidecar": ORACLE_SIDECAR,
        "run_id": config.run_id,
        "config_hash": config.config_hash,
        "git_branch": config.git_branch,
        "git_commit": config.git_commit,
        "oracle_semantics": ORACLE_SEMANTICS,
        "oracle1_best_route_scope": ORACLE1_BEST_ROUTE_SCOPE,
        "committed_intervention_set": COMMITTED_INTERVENTION_SET,
        "policy_identity_scope": (
            "initial_header_provenance; record_level_policy_identity"
        ),
        "policy_identity": policy_identity.to_dict(),
        "downstream_policy_contract": {
            "mode": "frozen_policy",
            "evaluation": "eval",
            "selection": "masked_argmax",
            "inference": "torch.inference_mode",
            "gradients": "none",
            "critic": "not_called",
            "sampling": "not_used",
            "training_rng": "not_consumed",
            "hidden_state": "branch_local",
        },
        "sampling_contract": {
            "selector": "sha256_canonical_input_without_training_rng",
            "selection_rate_ppm": config.training.mappo.route_oracle_selection_rate_ppm,
            "selector_inputs": [
                "schema_version",
                "run_id",
                "episode_id",
                "global_environment_step",
                "source_uav",
                "task_id",
            ],
        },
        "classification_contract": {
            "factual_route_branch_validity_values": [
                VALID_BRANCH_STATUS,
                INVALID_BRANCH_STATUS,
                UNRESOLVED_BRANCH_STATUS,
            ],
            "factual_branch_reused": (
                "true_for_factual_local_or_remote; false_for_factual_defer"
            ),
            "deadline_success": "completed_beats_expired_or_truncated",
            "delay_tolerance": "one_environment_slot",
            "energy_tolerance": "config.environment.energy_tolerance_j",
            "unresolved_is_policy_miss": False,
            "best_route_scope": ORACLE1_BEST_ROUTE_SCOPE,
        },
        "common_randomness_contract": {
            "required": True,
            "comparison": "matched_shadow_step_exogenous_fingerprint",
            "mismatch": "fail_fast_no_approximate_fallback",
            "event_tape": "not_implemented_in_v1",
        },
        "sidecar_isolation": {
            "actor_safe_sidecar": "route_diagnostic_samples_v1.jsonl",
            "trajectory_credit": "excluded",
            "raw_metrics": "excluded",
            "rollout_buffer": "excluded",
            "checkpoint_payload": "excluded",
        },
    }


class RouteOracleArtifactWriter:
    """Fresh-only, duplicate-safe writer for the Oracle-1 JSONL sidecar."""

    def __init__(
        self,
        config: RunConfig,
        header: Mapping[str, Any],
        *,
        allow_existing: bool = False,
        path: str | Path | None = None,
    ) -> None:
        if not config.training.mappo.route_oracle_counterfactual_enabled:
            raise OracleArtifactError("Oracle sidecar writer requires the Oracle flag")
        self.config = config
        self.path = Path(
            config.artifact_paths()["route_oracle_counterfactual"]
            if path is None
            else path
        )
        self.header = dict(header)
        self._decision_keys: set[str] = set()
        self._branch_keys: set[tuple[str, str]] = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            if not allow_existing:
                raise OracleArtifactError(f"refusing to overwrite existing Oracle sidecar: {self.path}")
            self._load_existing()
        else:
            self._append(self.header)

    def write(self, decision: OracleDecisionResult) -> None:
        if not isinstance(decision, OracleDecisionResult):
            raise TypeError("writer accepts OracleDecisionResult")
        if decision.decision_key in self._decision_keys:
            raise OracleArtifactError("duplicate Oracle decision key")
        records = decision.to_records()
        for record in records:
            if record["record_type"] == "oracle_branch":
                key = (str(record["decision_key"]), str(record["branch_route"]))
                if key in self._branch_keys:
                    raise OracleArtifactError("duplicate Oracle branch key")
            elif record["record_type"] == "oracle_decision":
                _validate_oracle_decision_record(record)
        self._append_many(records)
        self._decision_keys.add(decision.decision_key)
        self._branch_keys.update((decision.decision_key, str(branch.route)) for branch in decision.branches)

    def _append(self, record: Mapping[str, Any]) -> None:
        self._append_many((record,))

    def _append_many(self, records: Sequence[Mapping[str, Any]]) -> None:
        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(_canonical_json(dict(record)) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            raise OracleArtifactError(f"failed to write Oracle sidecar {self.path}") from exc

    def _load_existing(self) -> None:
        try:
            records = [
                json.loads(line)
                for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise OracleArtifactError("existing Oracle sidecar is not valid JSONL") from exc
        if not records or records[0].get("record_type") != "schema":
            raise OracleArtifactError("existing Oracle sidecar is missing its schema header")
        expected_header = dict(self.header)
        existing_header = dict(records[0])
        expected_header.pop("policy_identity", None)
        existing_header.pop("policy_identity", None)
        if existing_header != expected_header:
            raise OracleArtifactError("existing Oracle sidecar header does not match")
        # The first header is the initial provenance snapshot.  Each decision
        # record carries its own policy identity so resume may advance policy
        # version without changing the meaning of prior records.
        self.header = dict(records[0])
        for record in records[1:]:
            if record.get("record_type") == "oracle_decision":
                _validate_oracle_decision_record(record)
                key = str(record.get("decision_key"))
                if key in self._decision_keys:
                    raise OracleArtifactError("existing sidecar contains duplicate decision key")
                self._decision_keys.add(key)
            elif record.get("record_type") == "oracle_branch":
                key = (str(record.get("decision_key")), str(record.get("branch_route")))
                if key in self._branch_keys:
                    raise OracleArtifactError("existing sidecar contains duplicate branch key")
                self._branch_keys.add(key)
            else:
                raise OracleArtifactError("existing Oracle sidecar contains an unknown record type")


def inspect_route_oracle_artifact(
    config: RunConfig,
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the isolated sidecar without modifying it."""

    target = Path(
        config.artifact_paths()["route_oracle_counterfactual"]
        if path is None
        else path
    )
    if not target.exists():
        raise OracleArtifactError(f"Oracle sidecar does not exist: {target}")
    try:
        records = [
            json.loads(line)
            for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleArtifactError("Oracle sidecar is not valid JSONL") from exc
    if not records or records[0].get("record_type") != "schema":
        raise OracleArtifactError("Oracle sidecar is missing its schema header")
    if records[0].get("schema_version") != ORACLE_SCHEMA_VERSION:
        raise OracleArtifactError("Oracle sidecar schema version is invalid")
    if records[0].get("oracle1_best_route_scope") != ORACLE1_BEST_ROUTE_SCOPE:
        raise OracleArtifactError("Oracle sidecar best-route scope is invalid")
    if records[0].get("policy_identity_scope") != (
        "initial_header_provenance; record_level_policy_identity"
    ):
        raise OracleArtifactError("Oracle sidecar policy identity scope is invalid")
    decisions: set[str] = set()
    branches: set[tuple[str, str]] = set()
    decision_records: dict[str, Mapping[str, Any]] = {}
    branch_records: dict[str, dict[str, Mapping[str, Any]]] = {}
    decision_count = branch_count = 0
    for record in records[1:]:
        _assert_finite_json(record)
        record_type = record.get("record_type")
        if record_type == "oracle_decision":
            _validate_oracle_decision_record(record)
            key = str(record.get("decision_key"))
            if key in decisions:
                raise OracleArtifactError("Oracle sidecar has duplicate decisions")
            decisions.add(key)
            decision_records[key] = record
            decision_count += 1
        elif record_type == "oracle_branch":
            key = (str(record.get("decision_key")), str(record.get("branch_route")))
            if key in branches:
                raise OracleArtifactError("Oracle sidecar has duplicate branches")
            branches.add(key)
            branch_records.setdefault(key[0], {})[key[1]] = record
            branch_count += 1
        else:
            raise OracleArtifactError("Oracle sidecar contains an unknown record type")
    if set(branch_records).difference(decision_records):
        raise OracleArtifactError("Oracle sidecar branch references an unknown decision")
    valid_decision_count = 0
    invalid_branch_count = 0
    for decision_key, decision in decision_records.items():
        expected = tuple(str(route) for route in decision.get("evaluated_route_set", ()))
        observed_records = branch_records.get(decision_key, {})
        observed = tuple(observed_records)
        if len(expected) != len(set(expected)):
            raise OracleArtifactError("Oracle decision has duplicate evaluated routes")
        if set(observed) != set(expected):
            raise OracleArtifactError(
                "Oracle decision and branch records do not form a complete join"
            )
        if decision.get("branch_count") != len(expected):
            raise OracleArtifactError(
                "Oracle decision branch_count disagrees with evaluated routes"
            )
        for route, branch in observed_records.items():
            if tuple(str(item) for item in branch.get("evaluated_route_set", ())) != expected:
                raise OracleArtifactError(
                    "Oracle branch evaluated routes disagree with its decision"
                )
            if str(branch.get("branch_route")) != route:
                raise OracleArtifactError("Oracle branch route identity is inconsistent")
        invalid = tuple(
            branch
            for branch in observed_records.values()
            if not bool(branch.get("branch_valid"))
            or branch.get("branch_validity_status") != VALID_BRANCH_STATUS
        )
        invalid_branch_count += len(invalid)
        if not invalid:
            valid_decision_count += 1
    return {
        "path": str(target),
        "schema_version": ORACLE_SCHEMA_VERSION,
        "decision_count": decision_count,
        "valid_decision_count": valid_decision_count,
        "branch_count": branch_count,
        "invalid_branch_count": invalid_branch_count,
        "oracle1_best_route_scope": ORACLE1_BEST_ROUTE_SCOPE,
    }


def _validate_oracle_decision_record(record: Mapping[str, Any]) -> None:
    """Validate explicit factual-route fields without reverse inference."""

    required = {
        "factual_route_branch_validity",
        "factual_branch_reused",
    }
    if not required.issubset(record):
        missing = ", ".join(sorted(required.difference(record)))
        raise OracleArtifactError(
            f"Oracle decision is missing explicit factual-route fields: {missing}"
        )
    if record["factual_route_branch_validity"] not in {
        VALID_BRANCH_STATUS,
        INVALID_BRANCH_STATUS,
        UNRESOLVED_BRANCH_STATUS,
    }:
        raise OracleArtifactError("factual route branch validity is invalid")
    if not isinstance(record["factual_branch_reused"], bool):
        raise OracleArtifactError("factual_branch_reused must be boolean")
    committed = record.get("legal_committed_route_set", ())
    factual_route = record.get("factual_route")
    expected_reuse = factual_route in committed if isinstance(committed, (list, tuple)) else False
    if record["factual_branch_reused"] != expected_reuse:
        raise OracleArtifactError("factual_branch_reused disagrees with factual route set")


def _assert_finite_json(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise OracleArtifactError("Oracle sidecar contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_finite_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_finite_json(item)


__all__ = [
    "ATTRIBUTION_ALLOCATED_PROXY",
    "ATTRIBUTION_EXACT",
    "ATTRIBUTION_NOT_AVAILABLE",
    "COMMITTED_INTERVENTION_SET",
    "FrozenOraclePolicyRunner",
    "FrozenOraclePolicyRunnerCache",
    "FrozenPolicyIdentity",
    "FrozenPolicyStep",
    "FrozenRunnerIdentity",
    "FrozenRunnerProvenance",
    "INVALID_BRANCH_STATUS",
    "NO",
    "ORACLE1_BEST_ROUTE_SCOPE",
    "ORACLE_SCHEMA_VERSION",
    "ORACLE_SEMANTICS",
    "Oracle1Collector",
    "OracleArtifactError",
    "OracleBranchResult",
    "OracleBudgetExhausted",
    "OracleCollectionBudget",
    "OracleDecisionResult",
    "OracleDiagnosticError",
    "OracleReliabilityError",
    "RouteOracleArtifactWriter",
    "TinyOracleFeasibilityBudget",
    "UNRESOLVED",
    "UNRESOLVED_BRANCH_STATUS",
    "VALID_BRANCH_STATUS",
    "YES",
    "actor_state_digest",
    "classify_oracle_routes",
    "compare_branch_outcomes",
    "hash_select_oracle_decision",
    "inspect_route_oracle_artifact",
    "oracle_schema_header",
    "selector_input",
    "tensor_digest",
]


def _energy_from_step(
    info: Mapping[str, Any],
    *,
    focal_task_id: int,
    source_uav: int,
) -> dict[str, Any]:
    service = info.get("service", {})
    links = service.get("links", ()) if isinstance(service, Mapping) else ()
    cpu_items = service.get("cpu", ()) if isinstance(service, Mapping) else ()
    energy = info.get("energy", {})
    debits = energy.get("debits", ()) if isinstance(energy, Mapping) else ()
    result = {
        "source_tx": 0.0,
        "source_tx_status": ATTRIBUTION_NOT_AVAILABLE,
        "focal_cpu": 0.0,
        "focal_cpu_status": ATTRIBUTION_NOT_AVAILABLE,
        "helper_cpu": 0.0,
        "team": 0.0,
    }
    for debit in debits if isinstance(debits, (list, tuple)) else ():
        if isinstance(debit, Mapping):
            result["team"] += _finite(debit.get("total_energy_j", 0.0), "team energy")
    for link in links if isinstance(links, (list, tuple)) else ():
        if not isinstance(link, Mapping):
            continue
        services = link.get("task_services", ())
        if not isinstance(services, (list, tuple)):
            continue
        matches = [item for item in services if isinstance(item, Mapping) and item.get("task_id") == focal_task_id]
        if not matches:
            continue
        link_energy = _finite(link.get("transmit_energy_j", 0.0), "transmit energy")
        amounts = [_finite(item.get("amount", 0.0), "task service amount") for item in services if isinstance(item, Mapping)]
        total_amount = math.fsum(amounts)
        focal_amount = math.fsum(_finite(item.get("amount", 0.0), "focal service amount") for item in matches)
        if len(services) == len(matches):
            fraction = 1.0
            status = ATTRIBUTION_EXACT
        elif total_amount > 0.0:
            fraction = focal_amount / total_amount
            status = ATTRIBUTION_ALLOCATED_PROXY
        else:
            fraction = 0.0
            status = ATTRIBUTION_NOT_AVAILABLE
        result["source_tx"] += link_energy * fraction
        result["source_tx_status"] = _merge_attribution(result["source_tx_status"], status)
    for item in cpu_items if isinstance(cpu_items, (list, tuple)) else ():
        if not isinstance(item, Mapping) or item.get("task_id") != focal_task_id:
            continue
        cpu_energy = _finite(item.get("cpu_energy_j", 0.0), "CPU energy")
        result["focal_cpu"] += cpu_energy
        result["focal_cpu_status"] = ATTRIBUTION_EXACT
        if item.get("executor_uav") != source_uav:
            result["helper_cpu"] += cpu_energy
    return result


def _task_terminal_payload(
    environment: U2UMECEnvironment,
    task_id: int,
    route_slot: int,
) -> dict[str, Any] | None:
    if environment.lifecycle is None:
        raise OracleReliabilityError("shadow lifecycle is unavailable")
    task = environment.lifecycle.tasks.get(task_id)
    if task is None or not task.is_terminal:
        return None
    completion_delay = None
    if task.outcome is TaskOutcome.DONE:
        completion_delay = _finite(
            task.e2e_delay(environment.config.environment.slot_duration_s),
            "completion delay",
        )
    terminal_slot = int(task.completion_slot) if task.completion_slot is not None else int(environment.slot)
    return {
        "terminal_class": task.outcome.value,
        "terminal_slot": terminal_slot,
        "route_to_terminal_slots": terminal_slot - route_slot,
        "completion_delay_s": completion_delay,
        "terminal_reason": task.terminal_reason,
    }


class Oracle1Collector:
    """Collect selected eligible decisions without mutating the factual episode."""

    def __init__(
        self,
        config: RunConfig,
        runner: FrozenOraclePolicyRunner,
        *,
        budget: OracleCollectionBudget | None = None,
        policy_identity: FrozenPolicyIdentity | None = None,
        selector_run_id: str | None = None,
        trace_fingerprint_mode: str = "step",
        branch_runtime_observer: Callable[[float], None] | None = None,
    ) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        if not isinstance(runner, FrozenOraclePolicyRunner):
            raise TypeError("runner must be FrozenOraclePolicyRunner")
        self.config = config
        self.runner = runner
        self.budget = budget
        if trace_fingerprint_mode not in {"step", "boundary"}:
            raise ValueError("trace_fingerprint_mode must be 'step' or 'boundary'")
        if branch_runtime_observer is not None and not callable(branch_runtime_observer):
            raise TypeError("branch_runtime_observer must be callable")
        self.trace_fingerprint_mode = trace_fingerprint_mode
        self.branch_runtime_observer = branch_runtime_observer
        self.selector_run_id = (
            config.run_id if selector_run_id is None else selector_run_id
        )
        if (
            not isinstance(self.selector_run_id, str)
            or not self.selector_run_id
        ):
            raise ValueError("selector_run_id must be a non-empty string")
        self.policy_identity = runner.identity if policy_identity is None else policy_identity
        if self.policy_identity.runner_identity != runner.runner_identity:
            raise OracleReliabilityError(
                "collector policy identity does not match the frozen runner"
            )

    def collect_pre_step(
        self,
        environment: U2UMECEnvironment,
        observations: Sequence[Any],
        factual_proposals: Sequence[ActionProposal],
        current_hidden_out: Tensor,
        *,
        episode_id: int,
        global_environment_step: int,
        policy_version: int | None = None,
        factual_environment_transitions: int | None = None,
    ) -> tuple[OracleDecisionResult, ...]:
        """Collect every selected focal decision in the current joint step."""

        if len(observations) != len(factual_proposals):
            raise OracleReliabilityError("observations and factual proposals are misaligned")
        selected_sources: list[int] = []
        for source, (observation, proposal) in enumerate(zip(observations, factual_proposals)):
            task_id = _focal_task_id(observation)
            if task_id is None or not observation.action_masks.route_branch_active:
                continue
            try:
                _route_sets(observation, proposal.route)
            except ValueError:
                continue
            if hash_select_oracle_decision(
                run_id=self.selector_run_id,
                episode_id=episode_id,
                global_environment_step=global_environment_step,
                source_uav=source,
                task_id=task_id,
                selection_rate_ppm=self.config.training.mappo.route_oracle_selection_rate_ppm,
            ):
                selected_sources.append(source)
        results: list[OracleDecisionResult] = []
        for source in selected_sources:
            if self.budget is not None and not self.budget.can_start_decision(
                factual_transitions=factual_environment_transitions
            ):
                break
            result = self._collect_one_selected(
                environment,
                observations,
                factual_proposals,
                current_hidden_out,
                episode_id=episode_id,
                global_environment_step=global_environment_step,
                policy_version=policy_version,
                factual_environment_transitions=factual_environment_transitions,
                _source_uav=source,
            )
            if result is None:
                raise OracleReliabilityError("selected Oracle decision disappeared during collection")
            results.append(result)
        return tuple(results)

    def _collect_one_selected(
        self,
        environment: U2UMECEnvironment,
        observations: Sequence[Any],
        factual_proposals: Sequence[ActionProposal],
        current_hidden_out: Tensor,
        *,
        episode_id: int,
        global_environment_step: int,
        policy_version: int | None = None,
        factual_environment_transitions: int | None = None,
        _source_uav: int | None = None,
    ) -> OracleDecisionResult | None:
        if not isinstance(environment, U2UMECEnvironment):
            raise TypeError("environment must be U2UMECEnvironment")
        if policy_version is not None and policy_version != self.runner.policy_version:
            raise OracleReliabilityError("collector runner policy version is stale")
        if self.budget is not None and not self.budget.can_start_decision(
            factual_transitions=factual_environment_transitions
        ):
            return None
        if len(observations) != len(factual_proposals):
            raise OracleReliabilityError("observations and factual proposals are misaligned")
        selected: tuple[int, int, tuple[str | int, ...], tuple[str | int, ...]] | None = None
        for source, (observation, proposal) in enumerate(zip(observations, factual_proposals)):
            if _source_uav is not None and source != _source_uav:
                continue
            task_id = _focal_task_id(observation)
            if task_id is None or not observation.action_masks.route_branch_active:
                continue
            try:
                committed, evaluated = _route_sets(observation, proposal.route)
            except ValueError:
                continue
            if hash_select_oracle_decision(
                run_id=self.selector_run_id,
                episode_id=episode_id,
                global_environment_step=global_environment_step,
                source_uav=source,
                task_id=task_id,
                selection_rate_ppm=self.config.training.mappo.route_oracle_selection_rate_ppm,
            ):
                if selected is not None:
                    raise OracleReliabilityError("more than one focal route decision selected in one call")
                selected = (source, task_id, committed, evaluated)
        if selected is None:
            return None
        source, task_id, committed, evaluated = selected
        factual_route = factual_proposals[source].route
        if factual_route == "idle":
            raise OracleReliabilityError("route-active factual Idle is a reliability failure")
        if current_hidden_out.requires_grad:
            raise OracleReliabilityError("current hidden output must be detached")
        initial_hidden = current_hidden_out.detach().clone()
        initial_hidden_digest = tensor_digest(initial_hidden)
        initial_state = environment.state_fingerprint()
        initial_rng = environment.rng_fingerprint()
        initial_exogenous = environment.exogenous_fingerprint()
        route_slot = int(environment.slot)
        selector = selector_input(
            run_id=self.selector_run_id,
            episode_id=episode_id,
            global_environment_step=global_environment_step,
            source_uav=source,
            task_id=task_id,
        )
        decision_key = _sha256_json(selector)
        branches: list[OracleBranchResult] = []
        for route in evaluated:
            if route == factual_route and route in committed:
                role = "FACTUAL_ROUTE_REUSE"
            elif route == factual_route:
                role = "FACTUAL_DEFER_SHADOW"
            else:
                role = "COMMITTED"
            shadow = environment.clone_for_shadow()
            if shadow.state_fingerprint() != initial_state or shadow.rng_fingerprint() != initial_rng:
                raise OracleReliabilityError("shadow initial state/RNG does not match factual environment")
            branch_started = (
                time.perf_counter()
                if self.branch_runtime_observer is not None
                else None
            )
            branch = self._rollout_branch(
                shadow,
                observations,
                factual_proposals,
                source_uav=source,
                focal_task_id=task_id,
                route=route,
                route_slot=route_slot,
                initial_hidden=initial_hidden,
                initial_state=initial_state,
                initial_rng=initial_rng,
                initial_exogenous=initial_exogenous,
                initial_hidden_digest=initial_hidden_digest,
                role=role,
            )
            branches.append(branch)
            if branch_started is not None:
                assert self.branch_runtime_observer is not None
                self.branch_runtime_observer(time.perf_counter() - branch_started)
        self._check_common_randomness(branches)
        if environment.state_fingerprint() != initial_state or environment.rng_fingerprint() != initial_rng:
            raise OracleReliabilityError("Oracle shadow collection mutated factual environment")
        classification = classify_oracle_routes(
            branches,
            factual_route=factual_route,
            committed_routes=committed,
            evaluated_route_set=evaluated,
            slot_duration_s=self.config.environment.slot_duration_s,
            energy_tolerance_j=self.config.environment.energy_tolerance_j,
        )
        decision = OracleDecisionResult(
            decision_key=decision_key,
            episode_id=episode_id,
            global_environment_step=global_environment_step,
            source_uav=source,
            task_id=task_id,
            route_slot=route_slot,
            factual_route=factual_route,
            committed_routes=committed,
            evaluated_route_set=evaluated,
            policy_identity=self.policy_identity,
            initial_state_fingerprint=initial_state,
            initial_rng_fingerprint=initial_rng,
            initial_exogenous_fingerprint=initial_exogenous,
            initial_hidden_digest=initial_hidden_digest,
            branches=tuple(branches),
            classification=classification,
            selector_input=selector,
        )
        if self.budget is not None:
            self.budget.complete_selected_decision()
        return decision

    def _rollout_branch(
        self,
        shadow: U2UMECEnvironment,
        observations: Sequence[Any],
        factual_proposals: Sequence[ActionProposal],
        *,
        source_uav: int,
        focal_task_id: int,
        route: str | int,
        route_slot: int,
        initial_hidden: Tensor,
        initial_state: str,
        initial_rng: str,
        initial_exogenous: str,
        initial_hidden_digest: str,
        role: str,
    ) -> OracleBranchResult:
        proposals = list(factual_proposals)
        proposals[source_uav] = replace(proposals[source_uav], route=route)
        if len(proposals) != len(observations) or any(
            not observation.action_masks.is_legal(proposal)
            for observation, proposal in zip(observations, proposals)
        ):
            raise OracleReliabilityError(f"illegal Oracle route branch {route!r}")
        hidden = initial_hidden.detach().clone()
        trace: list[Mapping[str, Any]] = []
        exogenous_trace: list[str] = []
        source_tx = focal_cpu = helper_cpu = team_energy = 0.0
        source_tx_status = focal_cpu_status = ATTRIBUTION_NOT_AVAILABLE
        while True:
            if self.budget is not None:
                self.budget.reserve_shadow_transition()
            step_slot = int(shadow.slot)
            result = shadow.step(tuple(proposals))
            if not isinstance(result, StepResult):
                raise OracleReliabilityError("shadow step did not return StepResult")
            energy = _energy_from_step(
                result.info,
                focal_task_id=focal_task_id,
                source_uav=source_uav,
            )
            source_tx += energy["source_tx"]
            focal_cpu += energy["focal_cpu"]
            helper_cpu += energy["helper_cpu"]
            team_energy += energy["team"]
            source_tx_status = _merge_attribution(source_tx_status, energy["source_tx_status"])
            focal_cpu_status = _merge_attribution(focal_cpu_status, energy["focal_cpu_status"])
            exogenous_digest = shadow.exogenous_fingerprint()
            terminal = _task_terminal_payload(shadow, focal_task_id, route_slot)
            state_digest = (
                shadow.state_fingerprint()
                if self.trace_fingerprint_mode == "step" or terminal is not None
                else None
            )
            rng_digest = (
                shadow.rng_fingerprint()
                if self.trace_fingerprint_mode == "step" or terminal is not None
                else None
            )
            exogenous_trace.append(exogenous_digest)
            trace.append({
                "slot": step_slot,
                "proposal_digest": _proposal_digest(proposals),
                "state_fingerprint": state_digest,
                "rng_fingerprint": rng_digest,
                "exogenous_fingerprint": exogenous_digest,
                "reward": _finite(result.reward, "shadow reward"),
                "terminated": bool(result.terminated),
                "truncated": bool(result.truncated),
                "terminal_class": None if terminal is None else terminal["terminal_class"],
            })
            if terminal is not None:
                assert state_digest is not None and rng_digest is not None
                return OracleBranchResult(
                    route=route,
                    role=role,
                    valid=True,
                    status=VALID_BRANCH_STATUS,
                    terminal_class=terminal["terminal_class"],
                    terminal_slot=terminal["terminal_slot"],
                    route_slot=route_slot,
                    route_to_terminal_slots=terminal["route_to_terminal_slots"],
                    completion_delay_s=terminal["completion_delay_s"],
                    source_tx_energy_j=source_tx,
                    focal_cpu_energy_j=focal_cpu,
                    helper_cpu_energy_j=helper_cpu,
                    team_active_energy_j=team_energy,
                    source_tx_energy_attribution=source_tx_status,
                    focal_cpu_energy_attribution=focal_cpu_status,
                    initial_state_fingerprint=initial_state,
                    initial_rng_fingerprint=initial_rng,
                    initial_exogenous_fingerprint=initial_exogenous,
                    initial_hidden_digest=initial_hidden_digest,
                    final_rng_fingerprint=rng_digest,
                    trace=tuple(trace),
                    exogenous_trace=tuple(exogenous_trace),
                    terminal_reason=terminal["terminal_reason"],
                    focal_task_id=focal_task_id,
                )
            if result.observations is None:
                raise OracleReliabilityError("focal task did not reach terminal before shadow boundary")
            next_step = self.runner.act(result.observations, hidden, episode_start=False)
            hidden = next_step.hidden_out.detach().clone()
            proposals = list(next_step.proposals)

    @staticmethod
    def _check_common_randomness(branches: Sequence[OracleBranchResult]) -> None:
        if not branches:
            raise OracleReliabilityError("Oracle decision has no branches")
        minimum = min(len(branch.exogenous_trace) for branch in branches)
        for step_index in range(minimum):
            expected = branches[0].exogenous_trace[step_index]
            if any(branch.exogenous_trace[step_index] != expected for branch in branches[1:]):
                raise OracleReliabilityError(
                    "COMMON_RANDOMNESS_GATE = FAIL; action-dependent draw order requires a future event tape"
                )


def _merge_attribution(current: str, observed: str) -> str:
    if observed == ATTRIBUTION_NOT_AVAILABLE:
        return current
    if current == ATTRIBUTION_NOT_AVAILABLE or current == observed:
        return observed
    return ATTRIBUTION_ALLOCATED_PROXY
