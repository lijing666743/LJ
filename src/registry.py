"""Execution registry shared by interactive and direct launch paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .artifacts import preflight_formal_training_artifacts
from .config import RunConfig
from .execution import ExecutionContext, validate_execution_context


Handler = Callable[..., "RunResult"]


@dataclass(frozen=True)
class RunResult:
    """Artifact-aware lifecycle result returned by every registered handler."""

    status: str
    run_id: str
    mode: str
    method_id: str
    message: str
    artifacts: tuple[str, ...] = ()

    @property
    def is_success(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "run_id": self.run_id,
            "mode": self.mode,
            "method_id": self.method_id,
            "message": self.message,
            "artifacts": list(self.artifacts),
        }


@dataclass(frozen=True)
class RegistryEntry:
    mode: str
    method_id: str
    handler: Handler


class Registry:
    """Map a validated ``(mode, method_id)`` pair to one backend handler."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], RegistryEntry] = {}

    def register(self, mode: str, method_id: str, handler: Handler) -> None:
        key = (mode, method_id)
        if key in self._entries:
            raise ValueError(f"registry entry already exists: {mode}/{method_id}")
        self._entries[key] = RegistryEntry(mode, method_id, handler)

    def resolve(self, mode: str, method_id: str) -> Handler:
        try:
            return self._entries[(mode, method_id)].handler
        except KeyError as exc:
            available = ", ".join(f"{m}/{i}" for m, i in sorted(self._entries))
            raise KeyError(f"no runner registered for {mode}/{method_id}; available: {available}") from exc

    def entries(self) -> tuple[RegistryEntry, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    def pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._entries))


def unavailable_handler(config: RunConfig) -> RunResult:
    """Return an explicit blocked state without fabricating run artifacts."""

    return RunResult(
        status="unavailable",
        run_id=config.run_id,
        mode=config.mode,
        method_id=config.method_id,
        message=(
            "The environment backend for this mode is not implemented yet. "
            "No metrics, dashboard, checkpoint, or synthetic result was generated."
        ),
    )


def environment_sanity_handler(config: RunConfig) -> RunResult:
    """Run the real environment reset/step sanity validation."""

    from .env.validation import run_environment_sanity

    outcome = run_environment_sanity(config)
    return RunResult(
        status=outcome.status,
        run_id=config.run_id,
        mode=config.mode,
        method_id=config.method_id,
        message=outcome.message,
    )


def gate0_handler(config: RunConfig) -> RunResult:
    """Run the explicit end-to-end G0-01..G0-21 test module."""

    from .env.validation import run_gate0_tests

    outcome = run_gate0_tests(config)
    return RunResult(
        status=outcome.status,
        run_id=config.run_id,
        mode=config.mode,
        method_id=config.method_id,
        message=outcome.message,
    )


def random_rollout_handler(config: RunConfig) -> RunResult:
    """Run one real masked-random episode through the unified backend."""

    from .policies.random_policy import RandomPolicy
    from .rollout import RolloutRunner

    try:
        outcome = RolloutRunner(config, RandomPolicy(config.seed)).run()
    except Exception as exc:
        return RunResult(
            status="failed",
            run_id=config.run_id,
            mode=config.mode,
            method_id=config.method_id,
            message=f"random policy rollout failed: {exc}",
        )
    tasks = outcome.summary["tasks"]
    return RunResult(
        status="completed",
        run_id=config.run_id,
        mode=config.mode,
        method_id=config.method_id,
        message=(
            "random policy rollout completed through U2UMECEnvironment: "
            f"generated={tasks['generated']}, completed={tasks['completed']}, "
            f"expired={tasks['expired']}, truncated={tasks['truncated']}"
        ),
        artifacts=outcome.artifacts,
    )


def heuristic_rollout_handler(config: RunConfig) -> RunResult:
    """Run one real deterministic heuristic episode through the unified backend."""

    from .policies.heuristic_policy import HeuristicPolicy
    from .rollout import RolloutRunner

    try:
        outcome = RolloutRunner(config, HeuristicPolicy(config)).run()
    except Exception as exc:
        return RunResult(
            status="failed",
            run_id=config.run_id,
            mode=config.mode,
            method_id=config.method_id,
            message=f"heuristic policy rollout failed: {exc}",
        )
    tasks = outcome.summary["tasks"]
    return RunResult(
        status="completed",
        run_id=config.run_id,
        mode=config.mode,
        method_id=config.method_id,
        message=(
            "heuristic policy rollout completed through U2UMECEnvironment: "
            f"generated={tasks['generated']}, completed={tasks['completed']}, "
            f"expired={tasks['expired']}, truncated={tasks['truncated']}"
        ),
        artifacts=outcome.artifacts,
    )


def local_only_rollout_handler(config: RunConfig) -> RunResult:
    """Run one deterministic local-only episode through the unified backend."""

    from .policies.local_only_policy import LocalOnlyPolicy
    from .rollout import RolloutRunner

    try:
        outcome = RolloutRunner(config, LocalOnlyPolicy(config)).run()
    except Exception as exc:
        return RunResult(
            status="failed",
            run_id=config.run_id,
            mode=config.mode,
            method_id=config.method_id,
            message=f"local-only policy rollout failed: {exc}",
        )
    tasks = outcome.summary["tasks"]
    return RunResult(
        status="completed",
        run_id=config.run_id,
        mode=config.mode,
        method_id=config.method_id,
        message=(
            "local-only policy rollout completed through U2UMECEnvironment: "
            f"generated={tasks['generated']}, completed={tasks['completed']}, "
            f"expired={tasks['expired']}, truncated={tasks['truncated']}"
        ),
        artifacts=outcome.artifacts,
    )


def ca_gat_mappo_training_handler(
    config: RunConfig,
    *,
    execution_context: ExecutionContext | None = None,
) -> RunResult:
    """Run the real production CA-GAT-MAPPO Trainer through its public API."""

    context = validate_execution_context(config, execution_context)
    live_diagnostic_path: str | None = None
    try:
        if config.launch_profile == "rl-formal":
            preflight_formal_training_artifacts(
                config.artifact_paths(),
                resume=context.resume_from is not None and not context.extend_training_budget,
            )
        from .models.ca_gat_mappo_trainer import CAGATMAPPOTrainer

        if context.warm_start_from is not None:
            trainer = CAGATMAPPOTrainer.warm_start_from_checkpoint(config, context.warm_start_from)
            if context.pause_at_periodic:
                checkpoint = trainer.train_until_periodic_checkpoint()
                return RunResult(status='paused', run_id=config.run_id, mode=config.mode,
                    method_id=config.method_id,
                    message='E16 stopped at periodic boundary; continuation requires explicit --resume-from',
                    artifacts=(str(checkpoint),))
            training = trainer.train_with_checkpoints()
        elif context.resume_from is not None:
            continuation_options = {}
            if context.extend_training_budget:
                continuation_options["extend_training_budget"] = True
            if context.verify_replay_against is not None:
                continuation_options["verify_replay_against"] = context.verify_replay_against
            trainer = CAGATMAPPOTrainer.resume_from_checkpoint(
                config,
                context.resume_from,
                **continuation_options,
            )
            if context.check_resume:
                raise RuntimeError("resume check must stop in the CLI before training dispatch")
            training = trainer.train_with_checkpoints()
        else:
            matched_writer = None
            if config.training.mappo.route_choice_stability_enabled:
                from .matched_short_diagnostics import (
                    LiveTrainingDiagnosticsWriter,
                    live_diagnostics_path,
                    matched_group_from_config,
                )

                matched_writer = LiveTrainingDiagnosticsWriter(
                    config,
                    group=matched_group_from_config(config),
                )
                live_diagnostic_path = str(live_diagnostics_path(config))
            trainer = (
                CAGATMAPPOTrainer(config)
                if matched_writer is None
                else CAGATMAPPOTrainer(
                    config,
                    matched_diagnostic_writer=matched_writer,
                )
            )
            if config.launch_profile == "rl-formal":
                training = trainer.train_with_checkpoints()
            else:
                training = trainer.train()
    except Exception as exc:
        return RunResult(
            status="failed",
            run_id=config.run_id,
            mode=config.mode,
            method_id=config.method_id,
            message=f"CA-GAT-MAPPO training failed before completion: {exc}",
            artifacts=(
                () if live_diagnostic_path is None else (live_diagnostic_path,)
            ),
        )
    try:
        from .training_artifacts import write_cagat_mappo_training_artifacts

        diagnostics = write_cagat_mappo_training_artifacts(config, training)
    except Exception as exc:
        return RunResult(
            status="failed",
            run_id=config.run_id,
            mode=config.mode,
            method_id=config.method_id,
            message=(
                "CA-GAT-MAPPO training completed but diagnostic artifact generation "
                f"failed: {exc}"
            ),
            artifacts=(
                () if live_diagnostic_path is None else (live_diagnostic_path,)
            ),
        )
    status = "completed" if diagnostics.smoke_gate_status == "pass" else "failed"
    return RunResult(
        status=status,
        run_id=config.run_id,
        mode=config.mode,
        method_id=config.method_id,
        message=(
            "CA-GAT-MAPPO training completed: "
            f"transitions={training.total_environment_transitions}, "
            f"episodes={training.completed_episode_count}, "
            f"ppo_updates={training.ppo_update_count}, "
            f"reward_mean={diagnostics.reward_mean:.6g}, "
            f"actor_loss={diagnostics.actor_loss:.6g}, "
            f"critic_loss={diagnostics.critic_loss:.6g}, "
            f"entropy={diagnostics.entropy:.6g}, "
            f"smoke_gate={diagnostics.smoke_gate_status}, "
            f"signal_gate={diagnostics.signal_gate_status}"
        ),
        artifacts=(
            diagnostics.artifacts
            if live_diagnostic_path is None
            else (*diagnostics.artifacts, live_diagnostic_path)
        ),
    )


def build_default_registry() -> Registry:
    """Register the frozen menu surface with honest early-stage handlers."""

    registry = Registry()
    from .evaluation.handler import formal_evaluation_handler

    pairs: Iterable[tuple[str, str]] = (
        ("environment_sanity", "environment"),
        ("gate0", "environment"),
        ("random", "random"),
        ("heuristic", "heuristic"),
        ("baseline", "local_only"),
        ("baseline", "factorized_action_gat_qmix"),
        ("rl", "ca_gat_mappo"),
        ("rl", "factorized_action_gat_qmix"),
        ("evaluation", "environment"),
        ("evaluation", "random"),
        ("evaluation", "heuristic"),
        ("evaluation", "ca_gat_mappo"),
        ("evaluation", "factorized_action_gat_qmix"),
        ("ablation", "ca_gat_mappo"),
        ("plot", "environment"),
    )
    implemented_handlers: dict[tuple[str, str], Handler] = {
        ("environment_sanity", "environment"): environment_sanity_handler,
        ("gate0", "environment"): gate0_handler,
        ("random", "random"): random_rollout_handler,
        ("heuristic", "heuristic"): heuristic_rollout_handler,
        ("baseline", "local_only"): local_only_rollout_handler,
        ("rl", "ca_gat_mappo"): ca_gat_mappo_training_handler,
        ("evaluation", "ca_gat_mappo"): formal_evaluation_handler,
    }
    for mode, method_id in pairs:
        handler = implemented_handlers.get((mode, method_id), unavailable_handler)
        registry.register(mode, method_id, handler)
    return registry


__all__ = [
    "Handler",
    "Registry",
    "RegistryEntry",
    "RunResult",
    "build_default_registry",
    "ca_gat_mappo_training_handler",
    "environment_sanity_handler",
    "gate0_handler",
    "heuristic_rollout_handler",
    "local_only_rollout_handler",
    "random_rollout_handler",
    "unavailable_handler",
]
