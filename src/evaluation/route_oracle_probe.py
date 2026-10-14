"""Minimal same-state Local-vs-Remote counterfactual probe.

The probe is intentionally diagnostic-only.  It reuses the existing Oracle-1
shadow rollout and records factual branch outcomes plus four infrastructure
gates.  It does not classify the outcomes or recommend algorithm changes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from ..artifacts import atomic_write_text, require_artifact_targets_absent
from ..config import CHECKPOINT_KIND_FINAL_COMPLETED, RunConfig
from ..env.actions import ActionProposal
from ..env.environment import U2UMECEnvironment
from ..models.ca_gat_mappo import ActorObservationTensorizer
from ..models.ca_gat_mappo_actions import (
    CAGATMAPPOActionDistribution,
    SequentialActionDistributionOutput,
    SequentialActionMaskBatch,
)
from ..models.ca_gat_mappo_runtime import CAGATMAPPOFactualActorRuntime
from .actor_loader import LoadedEvaluationActor, load_final_actor_for_evaluation
from .route_oracle import (
    FrozenOraclePolicyRunnerCache,
    Oracle1Collector,
    OracleBranchResult,
    OracleDecisionResult,
    actor_state_digest,
    tensor_digest,
)


ROUTE_ORACLE_PROBE_SCHEMA_VERSION = 1
ROUTE_ORACLE_PROBE_FILENAME = "route_oracle_probe_report.json"
FORMAL_ROUTE_ORACLE_PROBE_SCHEMA_VERSION = "formal_route_oracle_probe.v1"
FORMAL_ROUTE_ORACLE_PROBE_FILENAME = "formal_route_oracle_report.json"
FORMAL_ROUTE_ORACLE_PERFORMANCE_SCHEMA_VERSION = (
    "formal_route_oracle_performance.v1"
)
FORMAL_ROUTE_ORACLE_PERFORMANCE_FILENAME = (
    "formal_route_oracle_performance.json"
)
FORMAL_ROUTE_ORACLE_PROGRESS_INTERVAL = 10
FORMAL_ROUTE_ORACLE_BASELINE_TAG = "route-oracle-probe-v1-ready"
FORMAL_ROUTE_ORACLE_BASELINE_COMMIT = (
    "ae293a95d37e63f833283b78061c3acd40a3700b"
)
FORMAL_ROUTE_ORACLE_SCENARIO = "remote_stress_v2"
FORMAL_ROUTE_ORACLE_EVALUATION_SEEDS = (1042, 1043, 1044, 1045, 1046)
FORMAL_ROUTE_ORACLE_EPISODE_HORIZON = 500
FORMAL_ROUTE_ORACLE_MIN_STATES = 100
FORMAL_ROUTE_ORACLE_MAX_STATES = 300
FORMAL_ROUTE_ORACLE_DEFAULT_STATES = 200
FORMAL_ROUTE_ORACLE_RETURN_TOLERANCE = 1.0e-9
_POLICY_VERSION = 0
_ACTION_FIELDS = (
    "route",
    "tx_select",
    "resource_group",
    "resource_width",
    "power_level",
    "cpu_queue",
    "cpu_frequency",
)
_RUN_CONFIG_PROVENANCE_FIELDS = (
    "source_config_path",
    "cli_overrides",
    "interactive_overrides",
    "git_branch",
    "git_commit",
    "git_dirty",
)


class RouteOracleProbeError(RuntimeError):
    """Raised when the minimal probe cannot complete reliably."""


@dataclass(frozen=True)
class FormalRouteStateCandidate:
    """One reproducible route-decision candidate discovered in Stage A."""

    episode_id: int
    evaluation_seed: int
    slot: int
    uav_id: int
    task_id: int
    observation_hash: str
    joint_observation_hash: str
    hidden_in_hash: str
    hidden_hash: str
    policy_action: str | int
    legal_routes: tuple[str | int, ...]
    selection_key: str

    @property
    def root_key(self) -> tuple[int, int, int]:
        return (self.evaluation_seed, self.episode_id, self.slot)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode_id,
            "evaluation_seed": self.evaluation_seed,
            "slot": self.slot,
            "uav_id": self.uav_id,
            "paper_uav_id": self.uav_id + 1,
            "task_id": self.task_id,
            "observation_hash": self.observation_hash,
            "joint_observation_hash": self.joint_observation_hash,
            "hidden_in_hash": self.hidden_in_hash,
            "hidden_hash": self.hidden_hash,
            "policy_action": self.policy_action,
            "legal_routes": list(self.legal_routes),
            "selection_key": self.selection_key,
        }


@dataclass(frozen=True)
class RouteOracleProbeResult:
    """Published result of one bounded 1--3 state probe."""

    probe_id: str
    report_path: str
    requested_state_count: int
    collected_state_count: int
    factual_environment_transitions: int
    passed: bool
    checkpoint_sha256: str
    actor_digest: str
    actor_weights_unchanged: bool
    diagnostic_artifacts: tuple[str, ...] = ()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: Any, *, pretty: bool = False) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validated_checkpoint_identity(
    loaded_actor: LoadedEvaluationActor,
    config: RunConfig,
) -> dict[str, Any]:
    """Return the actual FINAL identity shared by smoke and formal probes."""

    if not isinstance(loaded_actor, LoadedEvaluationActor):
        raise TypeError("loaded_actor must be a LoadedEvaluationActor")
    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    digests = {
        "checkpoint_sha256": loaded_actor.source_checkpoint_sha256,
        "actor_digest": loaded_actor.actor_state_digest,
        "source_training_config_hash": loaded_actor.source_training_config_hash,
        "resolved_config_hash": config.config_hash,
    }
    for name, value in digests.items():
        if not isinstance(value, str) or len(value) != 64:
            raise RouteOracleProbeError(
                f"probe checkpoint identity is missing or invalid: {name}"
            )
        try:
            int(value, 16)
        except ValueError as exc:
            raise RouteOracleProbeError(
                f"probe checkpoint identity is missing or invalid: {name}"
            ) from exc
    if loaded_actor.source_checkpoint_kind != CHECKPOINT_KIND_FINAL_COMPLETED:
        raise RouteOracleProbeError(
            "probe checkpoint identity requires FINAL_COMPLETED"
        )
    checkpoint_path = loaded_actor.source_checkpoint_path
    if not isinstance(checkpoint_path, Path) or not checkpoint_path.is_file():
        raise RouteOracleProbeError(
            "probe checkpoint identity is missing its checkpoint path"
        )
    evaluator_commit = config.git_commit
    if not isinstance(evaluator_commit, str) or evaluator_commit in {"", "unknown"}:
        raise RouteOracleProbeError(
            "probe checkpoint identity is missing its evaluator commit"
        )
    return {
        "checkpoint_path": str(checkpoint_path),
        **digests,
        "checkpoint_kind": loaded_actor.source_checkpoint_kind,
        "evaluator_git_commit": evaluator_commit,
    }


def _route_label(route: str | int) -> str:
    return f"remote_uav_{route}" if isinstance(route, int) and not isinstance(route, bool) else str(route)


def _proposal_record(proposal: ActionProposal) -> dict[str, Any]:
    return {"uav_id": proposal.uav_id, **{name: getattr(proposal, name) for name in _ACTION_FIELDS}}


def _joint_action_record(proposals: Sequence[ActionProposal]) -> list[dict[str, Any]]:
    return [_proposal_record(item) for item in proposals]


def build_counterfactual_actions(
    factual_proposals: Sequence[ActionProposal],
    *,
    focal_uav: int,
    routes: Sequence[str | int],
) -> dict[str, tuple[ActionProposal, ...]]:
    """Create route-only root interventions from one policy action."""

    if not 0 <= focal_uav < len(factual_proposals):
        raise RouteOracleProbeError("focal_uav is outside the joint action")
    result: dict[str, tuple[ActionProposal, ...]] = {}
    for route in routes:
        branch = list(factual_proposals)
        branch[focal_uav] = replace(branch[focal_uav], route=route)
        result[_route_label(route)] = tuple(branch)
    return result


def action_diff_report(
    actions: Mapping[str, Sequence[ActionProposal]],
    *,
    focal_uav: int,
) -> tuple[bool, dict[str, list[dict[str, Any]]]]:
    """Return whether every branch differs only at the focal route field."""

    local = actions.get("local")
    if local is None:
        return False, {"error": [{"reason": "local branch is missing"}]}
    reports: dict[str, list[dict[str, Any]]] = {}
    passed = True
    for label, branch in actions.items():
        if label == "local":
            continue
        differences: list[dict[str, Any]] = []
        if len(local) != len(branch):
            differences.append({"field": "joint_length", "local": len(local), "branch": len(branch)})
        else:
            for left, right in zip(local, branch):
                if left.uav_id != right.uav_id:
                    differences.append({"uav_id": left.uav_id, "field": "uav_id", "local": left.uav_id, "branch": right.uav_id})
                for field in _ACTION_FIELDS:
                    left_value = getattr(left, field)
                    right_value = getattr(right, field)
                    if left_value != right_value:
                        differences.append({"uav_id": left.uav_id, "field": field, "local": left_value, "branch": right_value})
        expected = len(differences) == 1 and differences[0].get("uav_id") == focal_uav and differences[0].get("field") == "route"
        passed = passed and expected
        reports[f"local_vs_{label}"] = differences
    return passed and bool(reports), reports


def same_root_gate(
    decision: OracleDecisionResult,
    *,
    factual_state_after: str,
    factual_rng_after: str,
) -> bool:
    """Check that every branch used the same root and left factual state intact."""

    return bool(decision.branches) and all(
        branch.initial_state_fingerprint == decision.initial_state_fingerprint
        and branch.initial_rng_fingerprint == decision.initial_rng_fingerprint
        for branch in decision.branches
    ) and factual_state_after == decision.initial_state_fingerprint and factual_rng_after == decision.initial_rng_fingerprint


def same_hidden_gate(branches: Sequence[OracleBranchResult], expected_digest: str) -> bool:
    return bool(branches) and all(branch.initial_hidden_digest == expected_digest for branch in branches)


def common_random_numbers_gate(branches: Sequence[OracleBranchResult]) -> bool:
    """Compare the action-independent trace over the common branch lifetime."""

    if not branches:
        return False
    common_steps = min(len(branch.exogenous_trace) for branch in branches)
    if common_steps <= 0:
        return False
    for index in range(common_steps):
        expected = branches[0].exogenous_trace[index]
        if any(branch.exogenous_trace[index] != expected for branch in branches[1:]):
            return False
    return True


def _branch_digest(branch: OracleBranchResult) -> str:
    return _sha256_json(asdict(branch))


def repeatability_gate(
    first: Sequence[OracleBranchResult],
    second: Sequence[OracleBranchResult],
) -> tuple[bool, dict[str, dict[str, str | None]]]:
    """Require exact canonical branch equality across two root replays."""

    first_by_route = {_route_label(item.route): item for item in first}
    second_by_route = {_route_label(item.route): item for item in second}
    labels = sorted(set(first_by_route) | set(second_by_route))
    evidence: dict[str, dict[str, str | None]] = {}
    passed = set(first_by_route) == set(second_by_route)
    for label in labels:
        left = first_by_route.get(label)
        right = second_by_route.get(label)
        left_digest = None if left is None else _branch_digest(left)
        right_digest = None if right is None else _branch_digest(right)
        evidence[label] = {"first_digest": left_digest, "second_digest": right_digest}
        passed = passed and left_digest is not None and left_digest == right_digest
    return passed and bool(labels), evidence


def _branch_outcome(branch: OracleBranchResult) -> dict[str, Any]:
    reward = math.fsum(float(item["reward"]) for item in branch.trace)
    return {
        "reward": reward,
        "completion": branch.terminal_class == "done",
        "expiration": branch.terminal_class == "expired",
        "delay": branch.completion_delay_s,
        "energy": branch.team_active_energy_j,
        "terminal_class": branch.terminal_class,
        "terminal_slot": branch.terminal_slot,
    }


def _eligible_sources(observations: Sequence[Any]) -> tuple[int, ...]:
    sources: list[int] = []
    for source, observation in enumerate(observations):
        masks = observation.action_masks
        if not masks.route_branch_active or not observation.private_queues.unbound.head_valid_mask:
            continue
        legal = tuple(value for value, enabled in zip(masks.route_domain, masks.route_mask) if bool(enabled))
        remotes = tuple(value for value in legal if isinstance(value, int) and not isinstance(value, bool))
        if "local" in legal and remotes:
            sources.append(source)
    return tuple(sources)


def _is_remote(route: Any) -> bool:
    return isinstance(route, int) and not isinstance(route, bool)


def _route_sort_key(route: str | int) -> tuple[int, int | str]:
    if route == "local":
        return (0, "")
    if _is_remote(route):
        return (1, int(route))
    return (2, str(route))


def _formal_selection_key(payload: Mapping[str, Any]) -> str:
    return _sha256_json(
        {
            "schema_version": FORMAL_ROUTE_ORACLE_PROBE_SCHEMA_VERSION,
            **dict(payload),
        }
    )


def select_formal_route_states(
    candidates: Sequence[FormalRouteStateCandidate],
    *,
    target_count: int,
    evaluation_seeds: Sequence[int],
) -> tuple[tuple[FormalRouteStateCandidate, ...], dict[str, Any]]:
    """Select a deterministic seed-balanced cohort with one UAV per root."""

    if (
        isinstance(target_count, bool)
        or not isinstance(target_count, int)
        or target_count <= 0
    ):
        raise ValueError("target_count must be a positive integer")
    seeds = tuple(int(seed) for seed in evaluation_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("evaluation_seeds must be non-empty and unique")

    root_winners: dict[tuple[int, int, int], FormalRouteStateCandidate] = {}
    for candidate in candidates:
        if candidate.evaluation_seed not in seeds:
            raise ValueError("candidate evaluation seed is outside the seed set")
        current = root_winners.get(candidate.root_key)
        if current is None or candidate.selection_key < current.selection_key:
            root_winners[candidate.root_key] = candidate

    deduplicated = tuple(root_winners.values())
    by_seed = {
        seed: tuple(
            sorted(
                (
                    candidate
                    for candidate in deduplicated
                    if candidate.evaluation_seed == seed
                ),
                key=lambda item: item.selection_key,
            )
        )
        for seed in seeds
    }
    base, remainder = divmod(target_count, len(seeds))
    quotas = {
        seed: base + (1 if index < remainder else 0)
        for index, seed in enumerate(seeds)
    }
    selected: list[FormalRouteStateCandidate] = []
    selected_keys: set[str] = set()
    for seed in seeds:
        for candidate in by_seed[seed][: quotas[seed]]:
            selected.append(candidate)
            selected_keys.add(candidate.selection_key)

    shortfall = target_count - len(selected)
    if shortfall > 0:
        remaining = sorted(
            (
                candidate
                for candidate in deduplicated
                if candidate.selection_key not in selected_keys
            ),
            key=lambda item: item.selection_key,
        )
        selected.extend(remaining[:shortfall])

    selected_tuple = tuple(
        sorted(
            selected,
            key=lambda item: (
                item.episode_id,
                item.evaluation_seed,
                item.slot,
                item.uav_id,
            ),
        )
    )
    selected_by_seed = {
        str(seed): sum(
            candidate.evaluation_seed == seed for candidate in selected_tuple
        )
        for seed in seeds
    }
    metadata = {
        "target_state_count": target_count,
        "eligible_decision_count": len(candidates),
        "eligible_root_count": len(deduplicated),
        "same_root_candidates_removed": len(candidates) - len(deduplicated),
        "selected_state_count": len(selected_tuple),
        "selection_complete": len(selected_tuple) == target_count,
        "selection_rule": (
            "lowest_sha256_per_(seed,episode,slot), then per-seed quota, "
            "then global lowest-sha256 deficit fill"
        ),
        "per_seed": {
            str(seed): {
                "quota": quotas[seed],
                "eligible_root_count": len(by_seed[seed]),
                "selected_count": selected_by_seed[str(seed)],
            }
            for seed in seeds
        },
        "selected_state_ids": [item.selection_key for item in selected_tuple],
    }
    return selected_tuple, metadata


def calculate_formal_oracle(
    route_returns: Mapping[str | int, float],
    *,
    policy_action: str | int,
    tolerance: float = FORMAL_ROUTE_ORACLE_RETURN_TOLERANCE,
) -> dict[str, Any]:
    """Compute return-defined oracle facts without outcome interpretation."""

    if not route_returns:
        raise ValueError("route_returns must not be empty")
    converted: dict[str | int, float] = {}
    for route, value in route_returns.items():
        if route != "local" and not _is_remote(route):
            raise ValueError("formal oracle routes must be Local or Remote")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("route returns must be finite")
        converted[route] = numeric
    if "local" not in converted or not any(_is_remote(route) for route in converted):
        raise ValueError("formal oracle requires Local and at least one Remote")
    if policy_action not in converted:
        raise ValueError("policy action is absent from route returns")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    oracle_return = max(converted.values())
    best_set = tuple(
        sorted(
            (
                route
                for route, value in converted.items()
                if math.isclose(
                    value,
                    oracle_return,
                    rel_tol=tolerance,
                    abs_tol=tolerance,
                )
            ),
            key=_route_sort_key,
        )
    )
    policy_return = converted[policy_action]
    local_return = converted["local"]
    best_remote_return = max(
        value for route, value in converted.items() if _is_remote(route)
    )
    local_remote_tie = math.isclose(
        local_return,
        best_remote_return,
        rel_tol=tolerance,
        abs_tol=tolerance,
    )
    remote_better = (
        not local_remote_tie and best_remote_return > local_return
    )
    local_better = not local_remote_tie and local_return > best_remote_return
    return {
        "oracle_best_action": best_set[0] if len(best_set) == 1 else None,
        "oracle_best_action_set": list(best_set),
        "oracle_return": oracle_return,
        "policy_action": policy_action,
        "policy_return": policy_return,
        "regret": oracle_return - policy_return,
        "tie": len(best_set) > 1,
        "local_remote_tie": local_remote_tie,
        "local_win": local_better,
        "remote_win": remote_better,
        "oracle_remote_better": remote_better,
        "policy_miss": policy_action not in best_set,
        "return_tolerance": tolerance,
    }


def aggregate_formal_gate_status(
    per_state_results: Sequence[Mapping[str, Any]],
    *,
    requested_state_count: int,
    actor_weights_unchanged: bool,
    sampling_replay_match: bool,
) -> dict[str, Any]:
    """Aggregate required reliability gates with a fail-closed overall status."""

    required = (
        "same_root",
        "only_route_diff",
        "same_hidden",
        "crn",
        "repeatability",
    )
    counts = {
        name: sum(bool(item.get("gates", {}).get(name)) for item in per_state_results)
        for name in required
    }
    collected = len(per_state_results)
    sample_complete = collected == requested_state_count
    all_state_gates_pass = all(counts[name] == collected for name in required)
    overall = (
        sample_complete
        and actor_weights_unchanged
        and sampling_replay_match
        and all_state_gates_pass
    )
    return {
        "overall": "PASS" if overall else "FAIL",
        "sample_complete": sample_complete,
        "actor_weights_unchanged": actor_weights_unchanged,
        "sampling_replay_match": sampling_replay_match,
        "required_per_state_gates": list(required),
        "per_state_gate_pass_count": counts,
        "per_state_gate_total": collected,
    }


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile values must not be empty")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _rate(items: Sequence[Mapping[str, Any]], field: str) -> float | None:
    if not items:
        return None
    return math.fsum(1.0 if bool(item[field]) else 0.0 for item in items) / len(items)


def build_conditional_statistics(
    per_state_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build outcome-blind quartile bins followed by factual per-bin rates."""

    valid = tuple(
        item for item in per_state_results if bool(item.get("statistics_valid"))
    )
    dimensions: dict[str, Any] = {}
    for feature in (
        "local_backlog",
        "deadline_slack",
        "helper_queue",
        "channel_quality",
    ):
        available = [
            float(item["condition_features"][feature])
            for item in valid
            if item.get("condition_features", {}).get(feature) is not None
        ]
        edges = (
            sorted(
                set(
                    _linear_quantile(available, probability)
                    for probability in (0.25, 0.5, 0.75)
                )
            )
            if available
            else []
        )
        bins: list[dict[str, Any]] = []
        lower: float | None = None
        for index in range(len(edges) + 1):
            upper = edges[index] if index < len(edges) else None
            members = [
                item
                for item in valid
                if item.get("condition_features", {}).get(feature) is not None
                and (lower is None or float(item["condition_features"][feature]) > lower)
                and (upper is None or float(item["condition_features"][feature]) <= upper)
            ]
            bins.append(
                {
                    "bin": index,
                    "lower_exclusive": lower,
                    "upper_inclusive": upper,
                    "sample_count": len(members),
                    "policy_remote_ratio": _rate(members, "policy_is_remote"),
                    "oracle_remote_better_ratio": _rate(
                        members, "oracle_remote_better"
                    ),
                    "miss_rate": _rate(members, "policy_miss"),
                }
            )
            lower = upper
        unavailable = [
            item
            for item in valid
            if item.get("condition_features", {}).get(feature) is None
        ]
        if unavailable:
            bins.append(
                {
                    "bin": "unavailable",
                    "lower_exclusive": None,
                    "upper_inclusive": None,
                    "sample_count": len(unavailable),
                    "policy_remote_ratio": _rate(
                        unavailable, "policy_is_remote"
                    ),
                    "oracle_remote_better_ratio": _rate(
                        unavailable, "oracle_remote_better"
                    ),
                    "miss_rate": _rate(unavailable, "policy_miss"),
                }
            )
        dimensions[feature] = {
            "binning": "Q25/Q50/Q75 computed before reading counterfactual outcomes; duplicate edges merged",
            "edges": edges,
            "bins": bins,
        }
    return {
        "valid_state_count": len(valid),
        "feature_definitions": {
            "local_backlog": "source local_cpu remaining_cycles",
            "deadline_slack": "focal unbound head_slack_slots",
            "helper_queue": (
                "remaining_cycles of highest-policy-probability legal Remote"
            ),
            "channel_quality": (
                "mean actor-visible historical_quality over valid RUs for the "
                "highest-policy-probability legal Remote"
            ),
        },
        "dimensions": dimensions,
    }


def _summary_rate(count: int, total: int) -> float | None:
    return None if total == 0 else count / total


def summarize_formal_route_states(
    per_state_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    valid = tuple(
        item for item in per_state_results if bool(item.get("statistics_valid"))
    )
    total = len(valid)
    local_wins = sum(bool(item["local_win"]) for item in valid)
    remote_wins = sum(bool(item["remote_win"]) for item in valid)
    ties = sum(bool(item["local_remote_tie"]) for item in valid)
    policy_remote = sum(bool(item["policy_is_remote"]) for item in valid)
    misses = sum(bool(item["policy_miss"]) for item in valid)
    policy_local = tuple(item for item in valid if not bool(item["policy_is_remote"]))
    local_remote_better = sum(
        bool(item["oracle_remote_better"]) for item in policy_local
    )
    regrets = [float(item["regret"]) for item in valid]
    return {
        "collected_state_count": len(per_state_results),
        "valid_state_count": total,
        "invalid_state_count": len(per_state_results) - total,
        "local_win_count": local_wins,
        "local_win_rate": _summary_rate(local_wins, total),
        "remote_win_count": remote_wins,
        "remote_win_rate": _summary_rate(remote_wins, total),
        "local_remote_tie_count": ties,
        "local_remote_tie_rate": _summary_rate(ties, total),
        "policy_remote_count": policy_remote,
        "policy_remote_ratio": _summary_rate(policy_remote, total),
        "policy_miss_count": misses,
        "policy_miss_rate": _summary_rate(misses, total),
        "regret": {
            "sample_count": len(regrets),
            "mean": None if not regrets else math.fsum(regrets) / len(regrets),
            "maximum": None if not regrets else max(regrets),
        },
        "policy_local_subset": {
            "sample_count": len(policy_local),
            "oracle_remote_better_count": local_remote_better,
            "oracle_remote_better_rate": _summary_rate(
                local_remote_better, len(policy_local)
            ),
        },
    }


class RouteOracleProbeRunner:
    """Run a bounded diagnostic probe without constructing training objects."""

    def __init__(
        self,
        config: RunConfig,
        checkpoint_path: str | Path,
        *,
        evaluation_device: str,
        state_count: int = 3,
        environment_factory: Callable[[RunConfig], U2UMECEnvironment] | None = None,
    ) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        config.validate()
        if config.mode != "evaluation" or config.method_id != "ca_gat_mappo":
            raise RouteOracleProbeError("probe requires evaluation/ca_gat_mappo config")
        if isinstance(state_count, bool) or not isinstance(state_count, int) or not 1 <= state_count <= 3:
            raise RouteOracleProbeError("state_count must be an integer in [1, 3]")
        probe_mappo = replace(
            config.training.mappo,
            route_oracle_counterfactual_enabled=True,
            route_oracle_selection_rate_ppm=1_000_000,
        )
        self.config = replace(config, training=replace(config.training, mappo=probe_mappo))
        for name in _RUN_CONFIG_PROVENANCE_FIELDS:
            object.__setattr__(self.config, name, getattr(config, name))
        self.config.validate()
        self.state_count = state_count
        self.evaluation_device = evaluation_device
        self.environment_factory = environment_factory or U2UMECEnvironment
        self.loaded_actor: LoadedEvaluationActor = load_final_actor_for_evaluation(
            self.config,
            checkpoint_path,
            evaluation_device=evaluation_device,
        )
        self.loaded_actor.actor.requires_grad_(False)
        self.loaded_actor.actor.eval()
        checkpoint_identity = _validated_checkpoint_identity(
            self.loaded_actor,
            self.config,
        )
        self.factual_runtime = CAGATMAPPOFactualActorRuntime(
            self.loaded_actor.actor,
            self.config,
            device=evaluation_device,
            dtype=torch.float32,
        )
        cache = FrozenOraclePolicyRunnerCache(self.config, device=evaluation_device)
        frozen_runner = cache.get(
            self.loaded_actor.actor,
            _POLICY_VERSION,
            checkpoint_sha256=self.loaded_actor.source_checkpoint_sha256,
            source="route_oracle_probe_v1",
            git_commit=self.config.git_commit,
            config_hash=self.config.config_hash,
            run_id=self.config.run_id,
        )
        self.collector = Oracle1Collector(
            self.config,
            frozen_runner,
            policy_identity=cache.current_identity,
            selector_run_id=self.config.run_id,
        )
        self.identity = {
            "schema_version": ROUTE_ORACLE_PROBE_SCHEMA_VERSION,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "config_hash": self.config.config_hash,
            "run_id": self.config.run_id,
            "scenario_id": self.config.scenario_id,
            "seed": self.config.seed,
            **checkpoint_identity,
            "evaluation_device": evaluation_device,
            "requested_state_count": state_count,
        }
        self.probe_id = f"route-oracle-probe-v1__{_sha256_json(self.identity)[:24]}"
        self.run_directory = Path(self.config.output.logs_dir) / "diagnostics" / self.probe_id
        self.report_path = self.run_directory / ROUTE_ORACLE_PROBE_FILENAME
        self._has_run = False

    def run(self) -> RouteOracleProbeResult:
        if self._has_run:
            raise RouteOracleProbeError("probe runner instances are single-use")
        self._has_run = True
        require_artifact_targets_absent((self.run_directory,), group_name="route oracle probe")
        actor_digest_before = actor_state_digest(self.loaded_actor.actor)
        environment = self.environment_factory(self.config)
        reset = environment.reset()
        observations = tuple(reset.observations)
        hidden = self.loaded_actor.actor.initial_hidden(
            1,
            device=self.evaluation_device,
            dtype=torch.float32,
        )
        states: list[Mapping[str, Any]] = []
        transitions = 0
        while len(states) < self.state_count:
            slot = int(observations[0].slot)
            factual_step = self.factual_runtime.sample_step(
                observations,
                hidden.detach(),
                episode_start=slot == 0,
            )
            output = factual_step.action_output
            factual_proposals = tuple(output.proposals[0][0])
            sources = _eligible_sources(observations)
            if sources:
                states.append(
                    self._probe_state(
                        environment,
                        observations,
                        factual_proposals,
                        output.hidden_out.detach().clone(),
                        output.probabilities["route"],
                        source=sources[0],
                        global_environment_step=transitions,
                    )
                )
                if len(states) >= self.state_count:
                    break
            step = environment.step(factual_proposals)
            transitions += 1
            hidden = output.hidden_out.detach()
            if step.observations is None:
                break
            observations = tuple(step.observations)

        actor_digest_after = actor_state_digest(self.loaded_actor.actor)
        actor_unchanged = actor_digest_before == actor_digest_after
        passed = len(states) == self.state_count and actor_unchanged and all(
            all(bool(value) for value in state["gates"].values()) for state in states
        )
        report = {
            "schema_version": ROUTE_ORACLE_PROBE_SCHEMA_VERSION,
            "probe_id": self.probe_id,
            "status": "passed" if passed else "failed",
            "purpose": "counterfactual_infrastructure_smoke_only",
            "claim_boundary": "no_large_scale_statistics_no_algorithm_interpretation",
            "identity": self.identity,
            "requested_state_count": self.state_count,
            "collected_state_count": len(states),
            "factual_environment_transitions": transitions,
            "actor_weights_unchanged": actor_unchanged,
            "states": states,
        }
        atomic_write_text(self.report_path, _canonical_json(report, pretty=True) + "\n")
        return RouteOracleProbeResult(
            probe_id=self.probe_id,
            report_path=str(self.report_path),
            requested_state_count=self.state_count,
            collected_state_count=len(states),
            factual_environment_transitions=transitions,
            passed=passed,
            checkpoint_sha256=self.loaded_actor.source_checkpoint_sha256,
            actor_digest=actor_digest_after,
            actor_weights_unchanged=actor_unchanged,
        )

    def _probe_state(
        self,
        environment: U2UMECEnvironment,
        observations: Sequence[Any],
        factual_proposals: Sequence[ActionProposal],
        hidden_out: torch.Tensor,
        route_probabilities: torch.Tensor,
        *,
        source: int,
        global_environment_step: int,
    ) -> Mapping[str, Any]:
        observation = observations[source]
        task_id = int(observation.private_queues.unbound.head_task_id)
        legal_routes = tuple(
            value
            for value, enabled in zip(observation.action_masks.route_domain, observation.action_masks.route_mask)
            if bool(enabled) and (value == "local" or isinstance(value, int) and not isinstance(value, bool))
        )
        counterfactuals = build_counterfactual_actions(
            factual_proposals,
            focal_uav=source,
            routes=legal_routes,
        )
        only_route, diff = action_diff_report(counterfactuals, focal_uav=source)
        state_before = environment.state_fingerprint()
        rng_snapshot = environment.randomness_snapshot()
        rng_before = environment.rng_fingerprint()
        hidden_digest = tensor_digest(hidden_out)
        first = self.collector._collect_one_selected(
            environment,
            observations,
            factual_proposals,
            hidden_out,
            episode_id=0,
            global_environment_step=global_environment_step,
            policy_version=_POLICY_VERSION,
            _source_uav=source,
        )
        second = self.collector._collect_one_selected(
            environment,
            observations,
            factual_proposals,
            hidden_out,
            episode_id=0,
            global_environment_step=global_environment_step,
            policy_version=_POLICY_VERSION,
            _source_uav=source,
        )
        if first is None or second is None:
            raise RouteOracleProbeError("eligible focal state was not collected")
        committed = tuple(branch for branch in first.branches if branch.route in first.committed_routes)
        repeated = tuple(branch for branch in second.branches if branch.route in second.committed_routes)
        same_root = same_root_gate(
            first,
            factual_state_after=environment.state_fingerprint(),
            factual_rng_after=environment.rng_fingerprint(),
        ) and state_before == first.initial_state_fingerprint and rng_before == first.initial_rng_fingerprint
        same_hidden = same_hidden_gate(committed, hidden_digest)
        crn = common_random_numbers_gate(committed)
        repeatable, repeat_evidence = repeatability_gate(committed, repeated)
        probability_values = route_probabilities[0, 0, source].detach().cpu().tolist()
        route_ranking = [
            {"route": route, "probability": float(probability_values[index])}
            for index, route in enumerate(observation.action_masks.route_domain)
            if bool(observation.action_masks.route_mask[index])
        ]
        route_ranking.sort(key=lambda item: (-item["probability"], str(item["route"])))
        by_route = {_route_label(branch.route): branch for branch in committed}
        local_result = _branch_outcome(by_route["local"])
        remote_results = {
            label: _branch_outcome(branch)
            for label, branch in sorted(by_route.items())
            if label.startswith("remote_uav_")
        }
        state_id = _sha256_json(
            {
                "root_hash": state_before,
                "focal_uav": source,
                "task_id": task_id,
                "global_environment_step": global_environment_step,
            }
        )
        return {
            "state_id": state_id,
            "root_hash": state_before,
            "root_rng_hash": rng_before,
            "root_rng_state": rng_snapshot,
            "focal_uav": source,
            "task_id": task_id,
            "policy_action": _joint_action_record(factual_proposals),
            "route_ranking": route_ranking,
            "counterfactual_actions": {
                label: _joint_action_record(actions)
                for label, actions in counterfactuals.items()
            },
            "action_diff_report": diff,
            "local_result": local_result,
            "remote_results": remote_results,
            "gate_evidence": {
                "hidden_digest": hidden_digest,
                "repeatability": repeat_evidence,
            },
            "gates": {
                "same_root": same_root,
                "only_route_diff": only_route,
                "same_hidden": same_hidden,
                "crn": crn,
                "repeatability": repeatable,
            },
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _formal_branch_outcome(branch: OracleBranchResult) -> dict[str, Any]:
    return {
        "route": branch.route,
        "route_return": math.fsum(float(item["reward"]) for item in branch.trace),
        "completion": branch.terminal_class == "done",
        "expiration": branch.terminal_class == "expired",
        "delay": branch.completion_delay_s,
        "energy": branch.team_active_energy_j,
        "terminal_class": branch.terminal_class,
        "terminal_slot": branch.terminal_slot,
        "valid": branch.valid,
        "validity_status": branch.status,
    }


class FormalRouteOracleProbeRunner:
    """Two-stage deterministic Formal Same-State Route Oracle Probe V1."""

    def __init__(
        self,
        config: RunConfig,
        checkpoint_path: str | Path,
        *,
        evaluation_device: str,
        state_count: int = FORMAL_ROUTE_ORACLE_DEFAULT_STATES,
        environment_factory: Callable[[RunConfig], U2UMECEnvironment] | None = None,
    ) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        config.validate()
        if config.mode != "evaluation" or config.method_id != "ca_gat_mappo":
            raise RouteOracleProbeError(
                "formal probe requires evaluation/ca_gat_mappo config"
            )
        if config.scenario_id != FORMAL_ROUTE_ORACLE_SCENARIO:
            raise RouteOracleProbeError(
                "formal probe requires scenario_id='remote_stress_v2'"
            )
        if (
            tuple(config.evaluation.evaluation_seeds)
            != FORMAL_ROUTE_ORACLE_EVALUATION_SEEDS
        ):
            raise RouteOracleProbeError(
                "formal probe requires evaluation seeds 1042-1046"
            )
        if (
            config.environment.episode_horizon
            != FORMAL_ROUTE_ORACLE_EPISODE_HORIZON
        ):
            raise RouteOracleProbeError(
                "formal probe requires the fixed 500-slot evaluation horizon"
            )
        if evaluation_device != "cpu":
            raise RouteOracleProbeError("formal probe requires CPU evaluation")
        if (
            isinstance(state_count, bool)
            or not isinstance(state_count, int)
            or not FORMAL_ROUTE_ORACLE_MIN_STATES
            <= state_count
            <= FORMAL_ROUTE_ORACLE_MAX_STATES
        ):
            raise RouteOracleProbeError(
                "formal state_count must be an integer in [100, 300]"
            )

        probe_mappo = replace(
            config.training.mappo,
            route_oracle_counterfactual_enabled=True,
            route_oracle_selection_rate_ppm=1_000_000,
        )
        self.config = replace(
            config,
            training=replace(config.training, mappo=probe_mappo),
        )
        for name in _RUN_CONFIG_PROVENANCE_FIELDS:
            object.__setattr__(self.config, name, getattr(config, name))
        self.config.validate()
        self.state_count = state_count
        self.evaluation_device = evaluation_device
        self.environment_factory = environment_factory or U2UMECEnvironment
        self.loaded_actor = load_final_actor_for_evaluation(
            self.config,
            checkpoint_path,
            evaluation_device=evaluation_device,
        )
        self.loaded_actor.actor.requires_grad_(False)
        self.loaded_actor.actor.eval()
        checkpoint_identity = _validated_checkpoint_identity(
            self.loaded_actor,
            self.config,
        )
        self.actor_tensorizer = ActorObservationTensorizer(self.config)
        self.action_distribution = CAGATMAPPOActionDistribution(
            self.loaded_actor.actor,
            self.config,
        )
        cache = FrozenOraclePolicyRunnerCache(self.config, device=evaluation_device)
        self.frozen_runner = cache.get(
            self.loaded_actor.actor,
            _POLICY_VERSION,
            checkpoint_sha256=self.loaded_actor.source_checkpoint_sha256,
            source="formal_route_oracle_probe_v1",
            git_commit=self.config.git_commit,
            config_hash=self.config.config_hash,
            run_id=self.config.run_id,
        )
        self.collector = Oracle1Collector(
            self.config,
            self.frozen_runner,
            policy_identity=cache.current_identity,
            selector_run_id=self.config.run_id,
            trace_fingerprint_mode="boundary",
            branch_runtime_observer=self._record_branch_runtime,
        )

        source_config_path = (
            None
            if self.config.source_config_path is None
            else Path(self.config.source_config_path)
        )
        source_config_sha256 = (
            _file_sha256(source_config_path)
            if source_config_path is not None and source_config_path.is_file()
            else None
        )
        self.identity = {
            "schema_version": FORMAL_ROUTE_ORACLE_PROBE_SCHEMA_VERSION,
            "baseline_tag": FORMAL_ROUTE_ORACLE_BASELINE_TAG,
            "baseline_commit": FORMAL_ROUTE_ORACLE_BASELINE_COMMIT,
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "evaluator_git_branch": self.config.git_branch,
            "evaluator_git_dirty": self.config.git_dirty,
            "scenario_id": self.config.scenario_id,
            "evaluation_seeds": list(self.config.evaluation.evaluation_seeds),
            "episode_horizon": self.config.environment.episode_horizon,
            "evaluation_device": evaluation_device,
            **checkpoint_identity,
            "source_training_git_commit": (
                self.loaded_actor.source_training_git_commit
            ),
            "source_training_config_original_hash": (
                self.loaded_actor.source_training_config_original_hash
            ),
            "source_training_config_raw_snapshot_hash": (
                self.loaded_actor.source_training_config_raw_snapshot_hash
            ),
            "source_training_config_canonical_hash": (
                self.loaded_actor.source_training_config_canonical_hash
            ),
            "source_config_path": (
                None if source_config_path is None else str(source_config_path)
            ),
            "source_config_sha256": source_config_sha256,
            "requested_state_count": state_count,
            "inference_rule": "deterministic_masked_argmax",
            "training_updates": False,
        }
        digest = _sha256_json(self.identity)[:24]
        self.probe_id = f"formal-route-oracle-v1__{digest}"
        self.run_directory = (
            Path(self.config.output.logs_dir) / "diagnostics" / self.probe_id
        )
        self.report_path = self.run_directory / FORMAL_ROUTE_ORACLE_PROBE_FILENAME
        self.performance_path = (
            self.run_directory / FORMAL_ROUTE_ORACLE_PERFORMANCE_FILENAME
        )
        self._branch_rollout_times: list[float] = []
        self._stage_a_actor_call_count = 0
        self._stage_b_factual_actor_call_count = 0
        self._shadow_actor_call_count = 0
        self._preprocessing_cache_hits = 0
        self._preprocessing_cache_misses = 0
        self._stage_b_started_at: float | None = None
        self._has_run = False

    def _record_branch_runtime(self, elapsed_seconds: float) -> None:
        self._branch_rollout_times.append(float(elapsed_seconds))

    def _log_state_progress(self, completed: int) -> None:
        if self._stage_b_started_at is None or completed <= 0:
            return
        if (
            completed % FORMAL_ROUTE_ORACLE_PROGRESS_INTERVAL != 0
            and completed != self.state_count
        ):
            return
        elapsed = time.perf_counter() - self._stage_b_started_at
        average = elapsed / completed
        remaining = average * max(0, self.state_count - completed)
        print(
            "Formal Oracle Probe progress: "
            f"{completed}/{self.state_count} completed | "
            f"elapsed={elapsed:.1f}s | "
            f"avg={average:.3f}s/state | "
            f"eta={remaining:.1f}s",
            flush=True,
        )

    def _performance_report(
        self,
        *,
        run_started_at: float,
        stage_a_runtime_sec: float,
        stage_b_runtime_sec: float,
        states_completed: int,
    ) -> dict[str, Any]:
        branch_count = len(self._branch_rollout_times)
        branch_total = math.fsum(self._branch_rollout_times)
        cache_total = self._preprocessing_cache_hits + self._preprocessing_cache_misses
        return {
            "schema_version": FORMAL_ROUTE_ORACLE_PERFORMANCE_SCHEMA_VERSION,
            "probe_id": self.probe_id,
            "total_runtime_sec": time.perf_counter() - run_started_at,
            "stage_a_runtime_sec": stage_a_runtime_sec,
            "stage_b_runtime_sec": stage_b_runtime_sec,
            "states_completed": states_completed,
            "avg_seconds_per_state": (
                None
                if states_completed <= 0
                else stage_b_runtime_sec / states_completed
            ),
            "branch_rollout_count": branch_count,
            "avg_branch_rollout_time_sec": (
                None if branch_count <= 0 else branch_total / branch_count
            ),
            "stage_a_actor_call_count": self._stage_a_actor_call_count,
            "stage_b_factual_actor_call_count": (
                self._stage_b_factual_actor_call_count
            ),
            "shadow_actor_call_count": self._shadow_actor_call_count,
            "avg_actor_call_count": (
                None
                if states_completed <= 0
                else self._shadow_actor_call_count / states_completed
            ),
            "preprocessing_cache_hits": self._preprocessing_cache_hits,
            "preprocessing_cache_misses": self._preprocessing_cache_misses,
            "preprocessing_cache_hit_rate": (
                None
                if cache_total <= 0
                else self._preprocessing_cache_hits / cache_total
            ),
        }

    def run(self) -> RouteOracleProbeResult:
        if self._has_run:
            raise RouteOracleProbeError("formal probe runner instances are single-use")
        self._has_run = True
        require_artifact_targets_absent(
            (self.run_directory,), group_name="formal route oracle probe"
        )
        run_started_at = time.perf_counter()
        actor_digest_before = actor_state_digest(self.loaded_actor.actor)
        print("Formal Oracle Probe Stage A: scanning evaluation trajectories", flush=True)
        stage_a_started_at = time.perf_counter()
        candidates, scan_metadata = self._scan_evaluation_trajectories()
        stage_a_runtime_sec = time.perf_counter() - stage_a_started_at
        selected, selection_metadata = select_formal_route_states(
            candidates,
            target_count=self.state_count,
            evaluation_seeds=self.config.evaluation.evaluation_seeds,
        )
        print(
            "Formal Oracle Probe Stage A completed: "
            f"candidates={len(candidates)} selected={len(selected)}/{self.state_count} "
            f"elapsed={stage_a_runtime_sec:.1f}s",
            flush=True,
        )
        if len(selected) == self.state_count:
            print("Formal Oracle Probe Stage B: replaying selected states", flush=True)
            self._stage_b_started_at = time.perf_counter()
            states, replay_match = self._replay_selected_states(selected)
            stage_b_runtime_sec = time.perf_counter() - self._stage_b_started_at
        else:
            states, replay_match = (), False
            stage_b_runtime_sec = 0.0
        actor_digest_after = actor_state_digest(self.loaded_actor.actor)
        actor_unchanged = actor_digest_before == actor_digest_after
        gate_status = aggregate_formal_gate_status(
            states,
            requested_state_count=self.state_count,
            actor_weights_unchanged=actor_unchanged,
            sampling_replay_match=replay_match,
        )
        gate_status = {
            "checkpoint_identity": "PASS",
            "config_identity": "PASS",
            **gate_status,
        }
        passed = gate_status["overall"] == "PASS"
        sampling = {
            "stage_a": scan_metadata,
            "stage_b": selection_metadata,
            "selected_states": [item.to_dict() for item in selected],
        }
        report = {
            "schema_version": FORMAL_ROUTE_ORACLE_PROBE_SCHEMA_VERSION,
            "probe_id": self.probe_id,
            "status": "passed" if passed else "failed",
            "purpose": "evaluation_only_same_state_route_diagnostic",
            "claim_boundary": (
                "descriptive_counterfactual_facts_only_no_algorithm_interpretation"
            ),
            "identity": self.identity,
            "sampling": sampling,
            "summary": summarize_formal_route_states(states),
            "per_state_results": list(states),
            "conditional_statistics": build_conditional_statistics(states),
            "gate_status": gate_status,
        }
        atomic_write_text(self.report_path, _canonical_json(report, pretty=True) + "\n")
        performance = self._performance_report(
            run_started_at=run_started_at,
            stage_a_runtime_sec=stage_a_runtime_sec,
            stage_b_runtime_sec=stage_b_runtime_sec,
            states_completed=len(states),
        )
        atomic_write_text(
            self.performance_path,
            _canonical_json(performance, pretty=True) + "\n",
        )
        return RouteOracleProbeResult(
            probe_id=self.probe_id,
            report_path=str(self.report_path),
            requested_state_count=self.state_count,
            collected_state_count=len(states),
            factual_environment_transitions=int(
                scan_metadata["factual_environment_transitions"]
            ),
            passed=passed,
            checkpoint_sha256=self.loaded_actor.source_checkpoint_sha256,
            actor_digest=actor_digest_after,
            actor_weights_unchanged=actor_unchanged,
            diagnostic_artifacts=(str(self.performance_path),),
        )

    def _episode_config(self, evaluation_seed: int) -> RunConfig:
        episode_config = replace(
            self.config,
            mode="evaluation",
            method_id="ca_gat_mappo",
            launch_profile=None,
            seed=int(evaluation_seed),
        )
        episode_config.validate()
        return episode_config

    def _deterministic_step(
        self,
        observations: Sequence[Any],
        hidden: torch.Tensor,
    ) -> SequentialActionDistributionOutput:
        if self.loaded_actor.actor.training:
            raise RouteOracleProbeError("formal probe actor left eval mode")
        batch = self.actor_tensorizer.encode_step(
            observations,
            device=self.evaluation_device,
            dtype=torch.float32,
            episode_start=int(observations[0].slot) == 0,
        )
        masks = SequentialActionMaskBatch.from_observations(observations)
        with torch.inference_mode():
            output = self.action_distribution.deterministic_actions(
                batch,
                masks,
                hidden,
            )
        if output.mode != "deterministic":
            raise RouteOracleProbeError(
                "formal probe actor did not use deterministic selection"
            )
        return output

    def _candidate_for_source(
        self,
        *,
        episode_id: int,
        evaluation_seed: int,
        observations: Sequence[Any],
        proposals: Sequence[ActionProposal],
        hidden_in_hash: str,
        hidden_hash: str,
        joint_observation_hash: str,
        source: int,
    ) -> FormalRouteStateCandidate | None:
        observation = observations[source]
        policy_action = proposals[source].route
        if policy_action != "local" and not _is_remote(policy_action):
            return None
        legal_routes = tuple(
            value
            for value, enabled in zip(
                observation.action_masks.route_domain,
                observation.action_masks.route_mask,
            )
            if bool(enabled) and (value == "local" or _is_remote(value))
        )
        if policy_action not in legal_routes:
            raise RouteOracleProbeError("formal factual route is not legal")
        task_id = int(observation.private_queues.unbound.head_task_id)
        observation_hash = _sha256_json(observation.snapshot())
        payload = {
            "checkpoint_sha256": self.loaded_actor.source_checkpoint_sha256,
            "resolved_config_hash": self.config.config_hash,
            "episode": episode_id,
            "evaluation_seed": evaluation_seed,
            "slot": int(observation.slot),
            "uav_id": source,
            "task_id": task_id,
            "observation_hash": observation_hash,
            "hidden_hash": hidden_hash,
        }
        return FormalRouteStateCandidate(
            episode_id=episode_id,
            evaluation_seed=evaluation_seed,
            slot=int(observation.slot),
            uav_id=source,
            task_id=task_id,
            observation_hash=observation_hash,
            joint_observation_hash=joint_observation_hash,
            hidden_in_hash=hidden_in_hash,
            hidden_hash=hidden_hash,
            policy_action=policy_action,
            legal_routes=legal_routes,
            selection_key=_formal_selection_key(payload),
        )

    def _scan_evaluation_trajectories(
        self,
    ) -> tuple[tuple[FormalRouteStateCandidate, ...], dict[str, Any]]:
        candidates: list[FormalRouteStateCandidate] = []
        factual_transitions = 0
        route_active_decisions = 0
        policy_noncommitted_exclusions = 0
        per_seed: dict[str, Any] = {}
        for episode_id, evaluation_seed in enumerate(
            self.config.evaluation.evaluation_seeds
        ):
            environment = self.environment_factory(
                self._episode_config(int(evaluation_seed))
            )
            reset = environment.reset()
            observations = tuple(reset.observations)
            hidden = self.loaded_actor.actor.initial_hidden(
                1,
                device=self.evaluation_device,
                dtype=torch.float32,
            )
            seed_start = len(candidates)
            seed_transitions = 0
            while True:
                hidden_in_hash = tensor_digest(hidden)
                output = self._deterministic_step(observations, hidden)
                self._stage_a_actor_call_count += 1
                hidden_out = output.hidden_out.detach().clone()
                hidden_hash = tensor_digest(hidden_out)
                proposals = tuple(output.proposals[0][0])
                joint_observation_hash = _sha256_json(
                    [item.snapshot() for item in observations]
                )
                sources = _eligible_sources(observations)
                route_active_decisions += len(sources)
                for source in sources:
                    candidate = self._candidate_for_source(
                        episode_id=episode_id,
                        evaluation_seed=int(evaluation_seed),
                        observations=observations,
                        proposals=proposals,
                        hidden_in_hash=hidden_in_hash,
                        hidden_hash=hidden_hash,
                        joint_observation_hash=joint_observation_hash,
                        source=source,
                    )
                    if candidate is None:
                        policy_noncommitted_exclusions += 1
                    else:
                        candidates.append(candidate)
                step = environment.step(proposals)
                factual_transitions += 1
                seed_transitions += 1
                hidden = hidden_out
                if step.terminated or step.truncated:
                    break
                if step.observations is None:
                    raise RouteOracleProbeError(
                        "formal Stage A nonterminal step omitted observations"
                    )
                observations = tuple(step.observations)
            environment.assert_invariants()
            per_seed[str(evaluation_seed)] = {
                "episode": episode_id,
                "factual_environment_transitions": seed_transitions,
                "eligible_decision_count": len(candidates) - seed_start,
            }
            print(
                "Formal Oracle Probe Stage A seed completed: "
                f"{episode_id + 1}/{len(self.config.evaluation.evaluation_seeds)} "
                f"seed={evaluation_seed} "
                f"candidates={len(candidates) - seed_start}",
                flush=True,
            )
        return tuple(candidates), {
            "trajectory_policy": "frozen deterministic masked_argmax",
            "evaluation_seeds": list(self.config.evaluation.evaluation_seeds),
            "episode_count": len(self.config.evaluation.evaluation_seeds),
            "factual_environment_transitions": factual_transitions,
            "route_active_local_and_remote_legal_count": route_active_decisions,
            "policy_defer_or_idle_exclusion_count": (
                policy_noncommitted_exclusions
            ),
            "eligible_decision_count": len(candidates),
            "per_seed": per_seed,
        }

    @staticmethod
    def _candidate_replay_matches(
        candidate: FormalRouteStateCandidate,
        *,
        observations: Sequence[Any],
        proposals: Sequence[ActionProposal],
        hidden_in_hash: str,
        hidden_hash: str,
    ) -> bool:
        observation = observations[candidate.uav_id]
        return (
            int(observation.slot) == candidate.slot
            and int(observation.private_queues.unbound.head_task_id)
            == candidate.task_id
            and _sha256_json(observation.snapshot()) == candidate.observation_hash
            and _sha256_json([item.snapshot() for item in observations])
            == candidate.joint_observation_hash
            and hidden_in_hash == candidate.hidden_in_hash
            and hidden_hash == candidate.hidden_hash
            and proposals[candidate.uav_id].route == candidate.policy_action
        )

    def _replay_selected_states(
        self,
        selected: Sequence[FormalRouteStateCandidate],
    ) -> tuple[tuple[Mapping[str, Any], ...], bool]:
        selected_by_root = {
            (item.episode_id, item.evaluation_seed, item.slot): item
            for item in selected
        }
        states: list[Mapping[str, Any]] = []
        replay_match = True
        seen: set[str] = set()
        for episode_id, evaluation_seed in enumerate(
            self.config.evaluation.evaluation_seeds
        ):
            environment = self.environment_factory(
                self._episode_config(int(evaluation_seed))
            )
            reset = environment.reset()
            observations = tuple(reset.observations)
            hidden = self.loaded_actor.actor.initial_hidden(
                1,
                device=self.evaluation_device,
                dtype=torch.float32,
            )
            while True:
                hidden_in_hash = tensor_digest(hidden)
                output = self._deterministic_step(observations, hidden)
                self._stage_b_factual_actor_call_count += 1
                hidden_out = output.hidden_out.detach().clone()
                hidden_hash = tensor_digest(hidden_out)
                proposals = tuple(output.proposals[0][0])
                slot = int(observations[0].slot)
                candidate = selected_by_root.get(
                    (episode_id, int(evaluation_seed), slot)
                )
                if candidate is not None:
                    matched = self._candidate_replay_matches(
                        candidate,
                        observations=observations,
                        proposals=proposals,
                        hidden_in_hash=hidden_in_hash,
                        hidden_hash=hidden_hash,
                    )
                    replay_match = replay_match and matched
                    if matched:
                        states.append(
                            self._probe_formal_state(
                                environment,
                                observations,
                                proposals,
                                hidden_out,
                                output.probabilities["route"],
                                candidate=candidate,
                            )
                        )
                        seen.add(candidate.selection_key)
                        self._log_state_progress(len(states))
                step = environment.step(proposals)
                hidden = hidden_out
                if step.terminated or step.truncated:
                    break
                if step.observations is None:
                    raise RouteOracleProbeError(
                        "formal Stage B nonterminal step omitted observations"
                    )
                observations = tuple(step.observations)
            environment.assert_invariants()
        replay_match = replay_match and len(seen) == len(selected)
        return tuple(states), replay_match

    @staticmethod
    def _condition_features(
        observation: Any,
        route_probabilities: torch.Tensor,
        *,
        source: int,
    ) -> dict[str, Any]:
        legal_remote: list[tuple[int, int]] = []
        for index, (route, enabled) in enumerate(
            zip(
                observation.action_masks.route_domain,
                observation.action_masks.route_mask,
            )
        ):
            if bool(enabled) and _is_remote(route):
                legal_remote.append((index, int(route)))
        if not legal_remote:
            raise RouteOracleProbeError("condition features require a legal Remote")
        probabilities = route_probabilities[0, 0, source].detach().cpu().tolist()
        _, helper = min(
            legal_remote,
            key=lambda item: (-float(probabilities[item[0]]), item[1]),
        )
        quality = observation.edge_history.historical_quality[helper]
        quality_valid = observation.edge_history.quality_valid_mask[helper]
        valid_quality = [
            float(value)
            for value, valid in zip(quality, quality_valid)
            if bool(valid)
        ]
        local_cpu = observation.private_queues.local_cpu
        public = observation.neighbor_public
        return {
            "local_backlog": float(local_cpu.remaining_cycles),
            "local_backlog_task_count": int(local_cpu.task_count),
            "deadline_slack": int(
                observation.private_queues.unbound.head_slack_slots
            ),
            "helper_reference_remote": helper,
            "helper_reference_paper_uav": helper + 1,
            "helper_queue": float(public.cpu_load_remaining_cycles[helper]),
            "helper_queue_task_count": int(public.cpu_load_task_count[helper]),
            "channel_quality": (
                None
                if not valid_quality
                else math.fsum(valid_quality) / len(valid_quality)
            ),
            "channel_quality_valid_ru_count": len(valid_quality),
        }

    def _probe_formal_state(
        self,
        environment: U2UMECEnvironment,
        observations: Sequence[Any],
        factual_proposals: Sequence[ActionProposal],
        hidden_out: torch.Tensor,
        route_probabilities: torch.Tensor,
        *,
        candidate: FormalRouteStateCandidate,
    ) -> Mapping[str, Any]:
        source = candidate.uav_id
        observation = observations[source]
        legal_routes = candidate.legal_routes
        counterfactuals = build_counterfactual_actions(
            factual_proposals,
            focal_uav=source,
            routes=legal_routes,
        )
        only_route, diff = action_diff_report(counterfactuals, focal_uav=source)
        state_before = environment.state_fingerprint()
        rng_before = environment.rng_fingerprint()
        hidden_digest = tensor_digest(hidden_out)
        actor_calls_before = self.frozen_runner.act_call_count
        self.frozen_runner.start_preprocessing_cache()
        try:
            first = self.collector._collect_one_selected(
                environment,
                observations,
                factual_proposals,
                hidden_out,
                episode_id=candidate.episode_id,
                global_environment_step=candidate.slot,
                policy_version=_POLICY_VERSION,
                _source_uav=source,
            )
            second = self.collector._collect_one_selected(
                environment,
                observations,
                factual_proposals,
                hidden_out,
                episode_id=candidate.episode_id,
                global_environment_step=candidate.slot,
                policy_version=_POLICY_VERSION,
                _source_uav=source,
            )
        finally:
            cache_stats = self.frozen_runner.stop_preprocessing_cache()
            self._preprocessing_cache_hits += int(cache_stats["hits"])
            self._preprocessing_cache_misses += int(cache_stats["misses"])
            self._shadow_actor_call_count += (
                self.frozen_runner.act_call_count - actor_calls_before
            )
        if first is None or second is None:
            raise RouteOracleProbeError("selected formal state was not collected")
        committed = tuple(
            branch for branch in first.branches if branch.route in first.committed_routes
        )
        repeated = tuple(
            branch for branch in second.branches if branch.route in second.committed_routes
        )
        same_root = same_root_gate(
            first,
            factual_state_after=environment.state_fingerprint(),
            factual_rng_after=environment.rng_fingerprint(),
        ) and state_before == first.initial_state_fingerprint and rng_before == first.initial_rng_fingerprint
        same_hidden = same_hidden_gate(committed, hidden_digest)
        crn = common_random_numbers_gate(committed)
        repeatable, repeat_evidence = repeatability_gate(committed, repeated)
        gates = {
            "same_root": same_root,
            "only_route_diff": only_route,
            "same_hidden": same_hidden,
            "crn": crn,
            "repeatability": repeatable,
        }
        outcomes = tuple(
            _formal_branch_outcome(branch)
            for branch in sorted(committed, key=lambda item: _route_sort_key(item.route))
        )
        route_returns = {
            item["route"]: float(item["route_return"]) for item in outcomes
        }
        policy_route = factual_proposals[source].route
        oracle = calculate_formal_oracle(
            route_returns,
            policy_action=policy_route,
        )
        condition_features = self._condition_features(
            observation,
            route_probabilities,
            source=source,
        )
        probability_values = route_probabilities[0, 0, source].detach().cpu().tolist()
        route_ranking = [
            {"route": route, "probability": float(probability_values[index])}
            for index, route in enumerate(observation.action_masks.route_domain)
            if bool(observation.action_masks.route_mask[index])
        ]
        route_ranking.sort(
            key=lambda item: (-item["probability"], _route_sort_key(item["route"]))
        )
        all_branches_valid = all(bool(item["valid"]) for item in outcomes)
        result = {
            "state_id": candidate.selection_key,
            "episode": candidate.episode_id,
            "evaluation_seed": candidate.evaluation_seed,
            "slot": candidate.slot,
            "uav_id": source,
            "paper_uav_id": source + 1,
            "task_id": candidate.task_id,
            "observation_hash": candidate.observation_hash,
            "joint_observation_hash": candidate.joint_observation_hash,
            "hidden_in_hash": candidate.hidden_in_hash,
            "hidden_hash": candidate.hidden_hash,
            "root_hash": state_before,
            "root_rng_hash": rng_before,
            "policy_action": policy_route,
            "policy_action_category": (
                "Remote" if _is_remote(policy_route) else "Local"
            ),
            "policy_is_remote": _is_remote(policy_route),
            "legal_routes": list(legal_routes),
            "factual_joint_action": _joint_action_record(factual_proposals),
            "route_ranking": route_ranking,
            "route_outcomes": list(outcomes),
            "condition_features": condition_features,
            "action_diff_report": diff,
            "gate_evidence": {
                "branch_initial_hidden_hash": hidden_digest,
                "repeatability": repeat_evidence,
            },
            "gates": gates,
            "statistics_valid": all(gates.values()) and all_branches_valid,
            **oracle,
        }
        return result


__all__ = [
    "FORMAL_ROUTE_ORACLE_PERFORMANCE_FILENAME",
    "FORMAL_ROUTE_ORACLE_PERFORMANCE_SCHEMA_VERSION",
    "FORMAL_ROUTE_ORACLE_PROBE_FILENAME",
    "FORMAL_ROUTE_ORACLE_PROBE_SCHEMA_VERSION",
    "FormalRouteOracleProbeRunner",
    "FormalRouteStateCandidate",
    "ROUTE_ORACLE_PROBE_FILENAME",
    "ROUTE_ORACLE_PROBE_SCHEMA_VERSION",
    "RouteOracleProbeError",
    "RouteOracleProbeResult",
    "RouteOracleProbeRunner",
    "action_diff_report",
    "aggregate_formal_gate_status",
    "build_counterfactual_actions",
    "build_conditional_statistics",
    "calculate_formal_oracle",
    "common_random_numbers_gate",
    "repeatability_gate",
    "same_hidden_gate",
    "same_root_gate",
    "select_formal_route_states",
    "summarize_formal_route_states",
]
