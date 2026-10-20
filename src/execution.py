"""Explicit launch-time execution instructions kept outside ``RunConfig``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import ConfigError

if TYPE_CHECKING:
    from .config import RunConfig


@dataclass(frozen=True)
class ExecutionContext:
    """Non-scientific instructions that control one launcher invocation."""

    resume_from: Path | None = None
    evaluate_from: Path | None = None
    evaluation_device: str | None = None
    oracle_probe_states: int | None = None
    oracle_probe_mode: str | None = None
    warm_start_from: Path | None = None
    pause_at_periodic: bool = False
    extend_training_budget: bool = False
    verify_replay_against: Path | None = None
    check_resume: bool = False

    def __post_init__(self) -> None:
        if self.verify_replay_against is not None:
            object.__setattr__(self, "verify_replay_against", Path(self.verify_replay_against))
        if self.warm_start_from is not None:
            object.__setattr__(self, "warm_start_from", Path(self.warm_start_from))
        if self.resume_from is not None and not isinstance(self.resume_from, Path):
            object.__setattr__(self, "resume_from", Path(self.resume_from))
        if self.evaluate_from is not None and not isinstance(self.evaluate_from, Path):
            object.__setattr__(self, "evaluate_from", Path(self.evaluate_from))
        if self.oracle_probe_states is not None and self.oracle_probe_mode is None:
            object.__setattr__(self, "oracle_probe_mode", "smoke")

    @classmethod
    def from_resume_path(cls, resume_from: str | Path | None) -> "ExecutionContext":
        return cls(None if resume_from is None else Path(resume_from))

    @classmethod
    def from_paths(
        cls,
        *,
        resume_from: str | Path | None = None,
        evaluate_from: str | Path | None = None,
        evaluation_device: str | None = None,
        oracle_probe_states: int | None = None,
        oracle_probe_mode: str | None = None,
    ) -> "ExecutionContext":
        return cls(
            resume_from=None if resume_from is None else Path(resume_from),
            evaluate_from=None if evaluate_from is None else Path(evaluate_from),
            evaluation_device=evaluation_device,
            oracle_probe_states=oracle_probe_states,
            oracle_probe_mode=oracle_probe_mode,
        )

    @classmethod
    def from_evaluation_path(
        cls,
        evaluate_from: str | Path | None,
        evaluation_device: str | None,
        oracle_probe_states: int | None = None,
        oracle_probe_mode: str | None = None,
    ) -> "ExecutionContext":
        return cls.from_paths(
            evaluate_from=evaluate_from,
            evaluation_device=evaluation_device,
            oracle_probe_states=oracle_probe_states,
            oracle_probe_mode=oracle_probe_mode,
        )


def validate_execution_context(
    config: "RunConfig",
    context: ExecutionContext | None,
) -> ExecutionContext:
    """Fail fast on incompatible training/evaluation launch instructions."""

    resolved = context or ExecutionContext()
    if config.method_id in {"factorized_action_gat_qmix", "c_feature_credit_qmix"} and any((
            resolved.verify_replay_against is not None, resolved.warm_start_from is not None,
            resolved.pause_at_periodic, resolved.oracle_probe_states is not None,
            resolved.oracle_probe_mode is not None)):
        raise ConfigError("QMIX supports full-state resume/budget extension; MAPPO-specific replay/oracle instructions are unsupported")
    if (resolved.extend_training_budget or resolved.check_resume
            or resolved.verify_replay_against is not None) and resolved.resume_from is None:
        raise ConfigError("budget extension, replay verification and resume check require --resume-from")
    if resolved.verify_replay_against is not None:
        if not resolved.extend_training_budget:
            raise ConfigError("--verify-replay-against requires --extend-training-budget")
        if not resolved.verify_replay_against.is_file():
            raise ConfigError("replay reference path must exist")
    if resolved.pause_at_periodic and (
            config.training.mappo.destination_supervision_mode != 'offline_pretrain'
            or resolved.warm_start_from is None or config.mode != 'rl'):
        raise ConfigError('E16 checkpoint pause requires explicit warm-start RL launch')
    if resolved.warm_start_from is not None:
        if resolved.resume_from is not None or resolved.evaluate_from is not None:
            raise ConfigError("warm-start is mutually exclusive with resume/evaluation")
        if (config.mode != 'rl' or config.method_id != 'ca_gat_mappo'
                or config.training.mappo.destination_supervision_mode != 'offline_pretrain'):
            raise ConfigError("warm-start requires E16 offline_pretrain RL mode")
        if not resolved.warm_start_from.is_file():
            raise ConfigError("warm-start path must exist")
    if (config.mode == 'rl' and config.training.mappo.destination_supervision_mode == 'offline_pretrain'
            and resolved.warm_start_from is None and resolved.resume_from is None):
        raise ConfigError("E16 RL requires explicit --warm-start-from; random initialization forbidden")
    if resolved.oracle_probe_mode not in {None, "smoke", "formal"}:
        raise ConfigError("oracle probe mode must be 'smoke' or 'formal'")
    if resolved.oracle_probe_states is not None:
        if isinstance(resolved.oracle_probe_states, bool) or not isinstance(
            resolved.oracle_probe_states, int
        ):
            raise ConfigError("oracle probe state count must be an integer")
        lower, upper = (
            (100, 300) if resolved.oracle_probe_mode == "formal" else (1, 3)
        )
        if not lower <= resolved.oracle_probe_states <= upper:
            raise ConfigError(
                f"{resolved.oracle_probe_mode} oracle probe state count must be "
                f"an integer in [{lower}, {upper}]"
            )
    elif resolved.oracle_probe_mode is not None:
        raise ConfigError("oracle probe mode requires a state count")
    if resolved.evaluate_from is not None:
        if resolved.resume_from is not None:
            raise ConfigError("--resume-from and --evaluate-from are mutually exclusive")
        if config.mode != "evaluation":
            raise ConfigError("--evaluate-from requires mode='evaluation'")
        if config.method_id not in {"ca_gat_mappo", "factorized_action_gat_qmix", "c_feature_credit_qmix"}:
            raise ConfigError(
                "--evaluate-from requires a supported learned method_id"
            )
        if resolved.evaluation_device not in {"cpu", "cuda"}:
            raise ConfigError(
                "--evaluate-from requires explicit --evaluation-device cpu or cuda"
            )
        if (
            resolved.oracle_probe_mode == "formal"
            and resolved.evaluation_device != "cpu"
        ):
            raise ConfigError("formal oracle probe requires --evaluation-device cpu")
        path = resolved.evaluate_from
        if not path.exists():
            raise ConfigError(f"evaluation checkpoint path does not exist: {path}")
        if not path.is_file():
            raise ConfigError(
                f"evaluation checkpoint path is not a regular file: {path}"
            )
        return resolved
    if config.mode == "evaluation":
        raise ConfigError("mode='evaluation' requires --evaluate-from")
    if resolved.oracle_probe_states is not None:
        raise ConfigError("oracle probe requires --evaluate-from")
    if resolved.oracle_probe_mode is not None:
        raise ConfigError("oracle probe mode requires --evaluate-from")
    if resolved.evaluation_device is not None:
        raise ConfigError("--evaluation-device requires --evaluate-from")
    if resolved.resume_from is None:
        return resolved
    if config.mode != "rl":
        raise ConfigError("--resume-from requires mode='rl'")
    if config.method_id not in {"ca_gat_mappo", "factorized_action_gat_qmix", "c_feature_credit_qmix"}:
        raise ConfigError("--resume-from requires a supported learned method_id")
    if config.method_id == "ca_gat_mappo" and config.launch_profile != "rl-formal":
        raise ConfigError("--resume-from requires launch_profile='rl-formal'")

    path = resolved.resume_from
    if not path.exists():
        raise ConfigError(f"checkpoint path does not exist: {path}")
    if not path.is_file():
        raise ConfigError(f"checkpoint path is not a regular file: {path}")
    return resolved


__all__ = ["ExecutionContext", "validate_execution_context"]
