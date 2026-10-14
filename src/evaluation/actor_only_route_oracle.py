"""Actor-only Oracle-1 diagnostic carrier with no training lifecycle."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from ..artifacts import require_artifact_targets_absent
from ..config import RunConfig
from ..env.environment import ResetResult, StepResult, U2UMECEnvironment
from ..env.randomness import derive_training_episode_seed
from ..models.ca_gat_mappo_runtime import CAGATMAPPOFactualActorRuntime
from .actor_loader import (
    LoadedEvaluationActor,
    load_final_actor_for_oracle_diagnostic,
)
from .route_oracle import (
    FrozenOraclePolicyRunnerCache,
    Oracle1Collector,
    OracleCollectionBudget,
    RouteOracleArtifactWriter,
    actor_state_digest,
    inspect_route_oracle_artifact,
    oracle_schema_header,
)


ACTOR_ONLY_ROUTE_ORACLE_SCHEMA_VERSION = 1
_POLICY_VERSION = 0
_PROTECTED_PATHS = (
    "knowledge/papers/1.pdf",
    "knowledge/papers/7121.pdf",
)


class ActorOnlyRouteOracleError(RuntimeError):
    """Raised when the diagnostic carrier violates its public contract."""


@dataclass(frozen=True)
class _TrackedGitIdentity:
    commit: str
    tracked_diff_sha256: str


@dataclass(frozen=True)
class ActorOnlyRouteOracleResult:
    """Auditable result from one completed actor-only diagnostic carrier."""

    oracle_diagnostic_id: str
    run_directory: str
    total_environment_transitions: int
    selected_oracle_decisions: int
    valid_oracle_decisions: int
    branch_count: int
    shadow_transitions: int
    stop_reason: str
    checkpoint_source_path: str
    checkpoint_sha256: str
    actor_digest: str
    sidecar_path: str
    started_episode_count: int
    completed_episode_count: int
    actor_frozen: bool
    actor_weights_unchanged: bool
    ppo_update_count: int = 0

    def __post_init__(self) -> None:
        if self.ppo_update_count != 0:
            raise ActorOnlyRouteOracleError(
                "actor-only diagnostic result cannot report PPO updates"
            )
        if self.total_environment_transitions < 0:
            raise ActorOnlyRouteOracleError(
                "diagnostic transition count cannot be negative"
            )


class ActorOnlyRouteOracleRunner:
    """Run a frozen learned actor through Oracle-1 without a training path."""

    def __init__(
        self,
        config: RunConfig,
        checkpoint_path: str | Path,
        *,
        expected_checkpoint_sha256: str,
        expected_actor_digest: str,
        oracle_budget: OracleCollectionBudget,
        max_factual_environment_transitions: int,
        execution_device: str,
        output_root: str | Path,
        environment_factory: Callable[[RunConfig], U2UMECEnvironment] | None = None,
    ) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        config.validate()
        if config.mode != "rl" or config.method_id != "ca_gat_mappo":
            raise ActorOnlyRouteOracleError(
                "diagnostic carrier requires rl/ca_gat_mappo RunConfig"
            )
        if not config.training.formal_rl_enabled:
            raise ActorOnlyRouteOracleError(
                "diagnostic carrier requires formal_rl_enabled"
            )
        if not config.training.mappo.route_oracle_counterfactual_enabled:
            raise ActorOnlyRouteOracleError(
                "diagnostic carrier requires Oracle-1 to be enabled"
            )
        if not isinstance(oracle_budget, OracleCollectionBudget):
            raise TypeError("oracle_budget must be OracleCollectionBudget")
        if (
            isinstance(max_factual_environment_transitions, bool)
            or not isinstance(max_factual_environment_transitions, int)
            or max_factual_environment_transitions <= 0
        ):
            raise ActorOnlyRouteOracleError(
                "max_factual_environment_transitions must be a positive integer"
            )
        if (
            oracle_budget.max_factual_environment_transitions
            != max_factual_environment_transitions
        ):
            raise ActorOnlyRouteOracleError(
                "runner factual cap must equal OracleCollectionBudget factual cap"
            )
        if (
            oracle_budget.selected_decisions != 0
            or oracle_budget.factual_environment_transitions != 0
            or oracle_budget.shadow_transitions != 0
            or oracle_budget.stop_reason is not None
        ):
            raise ActorOnlyRouteOracleError(
                "oracle_budget must be fresh and unconsumed"
            )
        if execution_device not in {"cpu", "cuda"}:
            raise ActorOnlyRouteOracleError(
                "execution_device must be explicitly 'cpu' or 'cuda'"
            )
        if not callable(environment_factory or U2UMECEnvironment):
            raise TypeError("environment_factory must be callable")

        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self.oracle_budget = oracle_budget
        self.max_factual_environment_transitions = (
            max_factual_environment_transitions
        )
        self.execution_device = execution_device
        self.output_root = Path(output_root)
        self.environment_factory = environment_factory or U2UMECEnvironment
        self.loaded_actor: LoadedEvaluationActor = (
            load_final_actor_for_oracle_diagnostic(
                config,
                self.checkpoint_path,
                execution_device=execution_device,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
                expected_actor_digest=expected_actor_digest,
            )
        )
        self.git_identity = _tracked_git_identity()
        self.diagnostic_identity = self._diagnostic_identity()
        self.oracle_diagnostic_id = self._diagnostic_id(
            self.diagnostic_identity
        )
        self.run_directory = self.output_root / self.oracle_diagnostic_id
        self.sidecar_path = (
            self.run_directory
            / self.config.output.route_oracle_counterfactual_filename
        )
        self.factual_actor_runtime = CAGATMAPPOFactualActorRuntime(
            self.loaded_actor.actor,
            config,
            device=execution_device,
            dtype=torch.float32,
        )
        self._has_run = False

    def run(self) -> ActorOnlyRouteOracleResult:
        """Execute one bounded diagnostic run and publish only its sidecar."""

        if self._has_run:
            raise ActorOnlyRouteOracleError(
                "one runner instance represents exactly one diagnostic run"
            )
        require_artifact_targets_absent(
            (self.run_directory,),
            group_name="actor-only Oracle diagnostic run",
        )
        self._has_run = True

        actor = self.loaded_actor.actor
        if actor.training or any(
            parameter.requires_grad for parameter in actor.parameters()
        ):
            raise ActorOnlyRouteOracleError(
                "diagnostic actor must remain eval-mode and frozen"
            )
        initial_actor_digest = actor_state_digest(actor)
        cache = FrozenOraclePolicyRunnerCache(
            self.config, device=self.execution_device
        )
        oracle_runner = cache.get(
            actor,
            _POLICY_VERSION,
            checkpoint_sha256=self.loaded_actor.source_checkpoint_sha256,
            source="actor_only_oracle_diagnostic",
            git_commit=self.git_identity.commit,
            config_hash=self.config.config_hash,
            run_id=self.oracle_diagnostic_id,
        )
        collector = Oracle1Collector(
            self.config,
            oracle_runner,
            budget=self.oracle_budget,
            policy_identity=cache.current_identity,
            selector_run_id=self.config.run_id,
        )
        header = oracle_schema_header(self.config, cache.current_identity)
        header.update(
            {
                "run_id": self.oracle_diagnostic_id,
                "base_run_id": self.config.run_id,
                "selector_run_id": self.config.run_id,
                "execution_kind": "actor_only_oracle_diagnostic",
                "oracle_diagnostic_identity": dict(self.diagnostic_identity),
            }
        )
        writer = RouteOracleArtifactWriter(
            self.config,
            header,
            path=self.sidecar_path,
        )

        transitions = 0
        started_episodes = 0
        completed_episodes = 0
        episode_index = 0
        stop = False
        while not stop:
            episode_config = replace(
                self.config,
                seed=derive_training_episode_seed(
                    self.config.seed, episode_index
                ),
            )
            episode_config.validate()
            environment = self.environment_factory(episode_config)
            if not isinstance(environment, U2UMECEnvironment):
                raise ActorOnlyRouteOracleError(
                    "environment_factory must return U2UMECEnvironment"
                )
            reset = environment.reset()
            self._validate_reset(reset)
            started_episodes += 1
            observations = tuple(reset.observations)
            hidden = actor.initial_hidden(
                1,
                device=self.execution_device,
                dtype=torch.float32,
            )
            if torch.count_nonzero(hidden).item() != 0:
                raise ActorOnlyRouteOracleError(
                    "actor initial recurrent state must be zero"
                )

            while True:
                slot = int(observations[0].slot)
                if any(int(item.slot) != slot for item in observations):
                    raise ActorOnlyRouteOracleError(
                        "actor observations are not slot-aligned"
                    )
                hidden_in = hidden.detach()
                factual_step = self.factual_actor_runtime.sample_step(
                    observations,
                    hidden_in,
                    episode_start=slot == 0,
                )
                action_output = factual_step.action_output
                proposals = tuple(action_output.proposals[0][0])
                decisions = collector.collect_pre_step(
                    environment,
                    observations,
                    proposals,
                    action_output.hidden_out.detach().clone(),
                    episode_id=episode_index,
                    global_environment_step=transitions,
                    policy_version=_POLICY_VERSION,
                    factual_environment_transitions=transitions,
                )
                for decision in decisions:
                    writer.write(decision)

                step = environment.step(proposals)
                boundary = self._validate_step(slot, step)
                transitions += 1
                self.oracle_budget.observe_factual_transitions()
                hidden = action_output.hidden_out.detach()
                stop = self.oracle_budget.should_stop(
                    factual_transitions=transitions
                )
                if transitions > self.max_factual_environment_transitions:
                    raise ActorOnlyRouteOracleError(
                        "diagnostic carrier exceeded its factual cap"
                    )
                if boundary:
                    completed_episodes += 1
                    episode_index += 1
                    break
                assert step.observations is not None
                observations = tuple(step.observations)
                if stop:
                    break
            if stop:
                break

        final_actor_digest = actor_state_digest(actor)
        actor_frozen = not actor.training and all(
            not parameter.requires_grad for parameter in actor.parameters()
        )
        actor_unchanged = final_actor_digest == initial_actor_digest
        if not actor_frozen or not actor_unchanged:
            raise ActorOnlyRouteOracleError(
                "diagnostic actor state changed during execution"
            )
        summary = inspect_route_oracle_artifact(
            self.config, path=self.sidecar_path
        )
        if summary["decision_count"] != self.oracle_budget.selected_decisions:
            raise ActorOnlyRouteOracleError(
                "sidecar decisions disagree with completed decision accounting"
            )
        return ActorOnlyRouteOracleResult(
            oracle_diagnostic_id=self.oracle_diagnostic_id,
            run_directory=str(self.run_directory),
            total_environment_transitions=transitions,
            selected_oracle_decisions=self.oracle_budget.selected_decisions,
            valid_oracle_decisions=int(summary["valid_decision_count"]),
            branch_count=int(summary["branch_count"]),
            shadow_transitions=self.oracle_budget.shadow_transitions,
            stop_reason=self.oracle_budget.stop_reason
            or "MAX_FACTUAL_ENVIRONMENT_TRANSITIONS",
            checkpoint_source_path=str(
                self.loaded_actor.source_checkpoint_path
            ),
            checkpoint_sha256=self.loaded_actor.source_checkpoint_sha256,
            actor_digest=final_actor_digest,
            sidecar_path=str(self.sidecar_path),
            started_episode_count=started_episodes,
            completed_episode_count=completed_episodes,
            actor_frozen=actor_frozen,
            actor_weights_unchanged=actor_unchanged,
        )

    def _diagnostic_identity(self) -> Mapping[str, Any]:
        return {
            "runner_schema_version": ACTOR_ONLY_ROUTE_ORACLE_SCHEMA_VERSION,
            "base_config_hash": self.config.config_hash,
            "base_run_id": self.config.run_id,
            "current_git_commit": self.git_identity.commit,
            "tracked_worktree_diff_sha256": (
                self.git_identity.tracked_diff_sha256
            ),
            "checkpoint_sha256": (
                self.loaded_actor.source_checkpoint_sha256
            ),
            "actor_digest": self.loaded_actor.actor_state_digest,
            "execution_device": self.execution_device,
            "selection_rate_ppm": (
                self.config.training.mappo.route_oracle_selection_rate_ppm
            ),
            "max_factual_environment_transitions": (
                self.max_factual_environment_transitions
            ),
            "target_selected_decisions": (
                self.oracle_budget.target_selected_decisions
            ),
            "max_shadow_transitions": (
                self.oracle_budget.max_shadow_transitions
            ),
        }

    @staticmethod
    def _diagnostic_id(identity: Mapping[str, Any]) -> str:
        payload = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return f"actor-only-oracle-v1__{digest[:24]}"

    @staticmethod
    def _validate_reset(reset: ResetResult) -> None:
        if not isinstance(reset, ResetResult):
            raise ActorOnlyRouteOracleError(
                "environment.reset() must return ResetResult"
            )
        if not reset.observations or any(
            int(item.slot) != 0 for item in reset.observations
        ):
            raise ActorOnlyRouteOracleError(
                "each diagnostic episode must reset at slot zero"
            )
        if int(reset.centralized_state.slot) != 0:
            raise ActorOnlyRouteOracleError(
                "reset centralized state must use slot zero"
            )

    @staticmethod
    def _validate_step(slot: int, step: StepResult) -> bool:
        if not isinstance(step, StepResult):
            raise ActorOnlyRouteOracleError(
                "environment.step() must return StepResult"
            )
        boundary = bool(step.terminated or step.truncated)
        if step.info.get("slot") != slot:
            raise ActorOnlyRouteOracleError(
                "StepResult.info slot differs from the factual slot"
            )
        bootstrap_allowed = step.info.get("bootstrap_allowed")
        if (
            not isinstance(bootstrap_allowed, bool)
            or bootstrap_allowed != (not boundary)
        ):
            raise ActorOnlyRouteOracleError(
                "bootstrap permission differs from the episode boundary"
            )
        if boundary and (
            step.observations is not None
            or step.centralized_state is not None
        ):
            raise ActorOnlyRouteOracleError(
                "episode boundary exposed an artificial next state"
            )
        if not boundary and (
            step.observations is None
            or step.centralized_state is None
        ):
            raise ActorOnlyRouteOracleError(
                "non-boundary step omitted its real next state"
            )
        return boundary


def _tracked_git_identity(cwd: str | Path | None = None) -> _TrackedGitIdentity:
    """Hash only HEAD and tracked changes; never inspect untracked PDFs."""

    base = Path.cwd() if cwd is None else Path(cwd)
    root = _git_output(
        ("rev-parse", "--show-toplevel"), cwd=base
    ).decode("utf-8").strip()
    commit = _git_output(
        ("rev-parse", "HEAD"), cwd=Path(root)
    ).decode("ascii").strip()
    if len(commit) != 40:
        raise ActorOnlyRouteOracleError("current Git commit is invalid")
    diff = _git_output(
        (
            "diff",
            "--no-ext-diff",
            "--binary",
            "HEAD",
            "--",
            ".",
            *tuple(f":(exclude){path}" for path in _PROTECTED_PATHS),
        ),
        cwd=Path(root),
    )
    return _TrackedGitIdentity(
        commit=commit,
        tracked_diff_sha256=hashlib.sha256(diff).hexdigest(),
    )


def _git_output(arguments: tuple[str, ...], *, cwd: Path) -> bytes:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=cwd,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise ActorOnlyRouteOracleError(
            "Git identity cannot be inspected"
        ) from exc
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ActorOnlyRouteOracleError(
            f"Git identity inspection failed: {message}"
        )
    return completed.stdout


__all__ = [
    "ACTOR_ONLY_ROUTE_ORACLE_SCHEMA_VERSION",
    "ActorOnlyRouteOracleError",
    "ActorOnlyRouteOracleResult",
    "ActorOnlyRouteOracleRunner",
]
