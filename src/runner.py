"""Common runner entry point for all configuration execution paths."""

from __future__ import annotations

from dataclasses import dataclass

from .config import RunConfig, write_config_snapshot
from .execution import ExecutionContext, validate_execution_context
from .registry import Registry, RunResult, build_default_registry


@dataclass
class Runner:
    """Dispatch a canonical ``RunConfig`` through one registry.

    Environment sanity, Gate 0, and masked-random rollout use the real unified
    environment backend. Under-specified or future training handlers remain
    explicitly unavailable and therefore create no fake artifacts.
    """

    registry: Registry

    def __init__(self, registry: Registry | None = None) -> None:
        self.registry = registry or build_default_registry()

    def run(
        self,
        config: RunConfig,
        *,
        execution_context: ExecutionContext | None = None,
    ) -> RunResult:
        config.validate()
        context = validate_execution_context(config, execution_context)
        handler = self.registry.resolve(config.mode, config.method_id)
        if (
            context.resume_from is None
            and context.evaluate_from is None
            and context.warm_start_from is None
        ):
            result = handler(config)
        else:
            result = handler(config, execution_context=context)
        if not isinstance(result, RunResult):
            raise TypeError("registered runner handlers must return RunResult")
        return result

    def write_snapshot_after_success(self, config: RunConfig, result: RunResult):
        """Persist a config snapshot only for a genuinely completed run."""

        if not result.is_success:
            raise RuntimeError("cannot write a successful-run snapshot for a non-completed result")
        return write_config_snapshot(config)


def run_config(
    config: RunConfig,
    registry: Registry | None = None,
    *,
    execution_context: ExecutionContext | None = None,
) -> RunResult:
    """Functional façade used by tests and future batch orchestration."""

    return Runner(registry=registry).run(
        config,
        execution_context=execution_context,
    )


__all__ = ["ExecutionContext", "Runner", "run_config"]
