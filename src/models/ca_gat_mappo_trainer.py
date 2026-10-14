"""Environment-driven production trainer for the frozen CA-GAT-MAPPO contract.

Collection lifecycle and accounting live here; action sampling, rollout
storage, GAE, PPO losses, and optimization remain delegated to gated modules.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass, field
import math
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..config import (
    AgentCreditMode,
    CHECKPOINT_KIND_FINAL_COMPLETED,
    CHECKPOINT_KIND_PERIODIC_RESUME,
    CHECKPOINT_V1_DIAGNOSTICS_STATE_FIELDS,
    CHECKPOINT_V1_TRAINING_STATE_FIELDS,
    MAPPO_INITIAL_POLICY_VERSION,
    RouteCreditMode,
    RunConfig,
    canonicalize_config_identity,
    mappo_checkpoint_kind_at,
    mappo_final_checkpoint_path,
    mappo_periodic_checkpoint_path,
    require_formal_rl_enabled,
    validate_mappo_checkpoint_contract,
    validate_mappo_training_device,
)
from ..evaluation.route_oracle import (
    FrozenOraclePolicyRunnerCache,
    Oracle1Collector,
    OracleCollectionBudget,
    RouteOracleArtifactWriter,
    oracle_schema_header,
)
from ..env.environment import ResetResult, StepResult, U2UMECEnvironment
from ..env.randomness import derive_training_episode_seed
from ..env.reward import AGENT_REWARD_CONSERVATION_TOLERANCE
from .ca_gat_mappo import (
    CAGATMAPPOActor,
    CentralizedStateTensorizer,
    MAPPOCentralizedCritic,
)
from .ca_gat_mappo_actions import CAGATMAPPOActionDistribution
from .ca_gat_mappo_runtime import CAGATMAPPOFactualActorRuntime
from .ca_gat_mappo_route_telemetry import (
    RouteDiagnosticSampleCollector,
    RouteOutcomeTracker,
    TrajectoryCreditTracker,
    TrajectoryPreStepCapture,
)
from .ca_gat_mappo_route_nstep import build_route_credit_step_metadata
from .ca_gat_mappo_checkpoint import (
    CheckpointError,
    atomic_save_checkpoint,
    build_checkpoint_payload,
    load_checkpoint_payload,
    restore_active_rollout,
    restore_model_optimizer_and_rng,
    serialize_active_rollout,
    validate_periodic_checkpoint_compatibility,
)
from .ca_gat_mappo_credit_telemetry import ConditionalCreditTraceWriter
from .ca_gat_mappo_rollout import CAGATMAPPORolloutBuffer
from .ca_gat_mappo_update import (
    CAGATMAPPOOptimizerBundle,
    CAGATMAPPORecurrentPPOUpdater,
    RecurrentPPOEpochDiagnostics,
    RecurrentPPOUpdateOutput,
)


class CAGATMAPPOTrainerError(RuntimeError):
    """Raised when production collection violates the frozen lifecycle."""


_REWARD_FIELDS = (
    "completed_task_count",
    "expired_task_count",
    "urgent_workload_s",
    "actual_energy_j",
    "normalized_completed",
    "normalized_expired",
    "normalized_workload",
    "normalized_energy",
    "completion_component",
    "expiration_penalty",
    "workload_penalty",
    "energy_penalty",
)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CAGATMAPPOTrainerError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise CAGATMAPPOTrainerError(f"{name} must be finite")
    return converted


def _console_progress_logger(message: str) -> None:
    """Write one immediately visible training-progress line to stdout."""

    print(message, flush=True)


@dataclass(frozen=True)
class ScalarTrainingDiagnostics:
    """Immutable count/sum/mean for one scalar training diagnostic."""

    count: int
    total: float
    mean: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count < 0
        ):
            raise CAGATMAPPOTrainerError("diagnostic count must be non-negative")
        if not math.isfinite(self.total) or not math.isfinite(self.mean):
            raise CAGATMAPPOTrainerError("diagnostic values must be finite")
        expected = 0.0 if self.count == 0 else self.total / self.count
        if not math.isclose(
            self.mean, expected, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise CAGATMAPPOTrainerError(
                "diagnostic mean does not match count and total"
            )


@dataclass(frozen=True)
class CAGATMAPPORewardDiagnostics:
    """Reward and component aggregates sourced from real StepResult records."""

    reward: ScalarTrainingDiagnostics
    completed_task_count: ScalarTrainingDiagnostics
    expired_task_count: ScalarTrainingDiagnostics
    urgent_workload_s: ScalarTrainingDiagnostics
    actual_energy_j: ScalarTrainingDiagnostics
    normalized_completed: ScalarTrainingDiagnostics
    normalized_expired: ScalarTrainingDiagnostics
    normalized_workload: ScalarTrainingDiagnostics
    normalized_energy: ScalarTrainingDiagnostics
    completion_component: ScalarTrainingDiagnostics
    expiration_penalty: ScalarTrainingDiagnostics
    workload_penalty: ScalarTrainingDiagnostics
    energy_penalty: ScalarTrainingDiagnostics

    def __post_init__(self) -> None:
        counts = {
            getattr(self, name).count for name in ("reward", *_REWARD_FIELDS)
        }
        if len(counts) != 1:
            raise CAGATMAPPOTrainerError("reward diagnostic counts must agree")

    @property
    def transition_count(self) -> int:
        return self.reward.count


@dataclass(frozen=True)
class CAGATMAPPOEpisodeDiagnostics:
    """Collection accounting for one started training episode."""

    episode_index: int
    environment_seed: int
    transition_count: int
    completed_boundary: bool
    reward: CAGATMAPPORewardDiagnostics

    def __post_init__(self) -> None:
        if self.episode_index < 0 or self.environment_seed < 0:
            raise CAGATMAPPOTrainerError(
                "episode index and seed must be non-negative"
            )
        if (
            self.transition_count <= 0
            or self.reward.transition_count != self.transition_count
        ):
            raise CAGATMAPPOTrainerError(
                "episode transition accounting is inconsistent"
            )
        if not isinstance(self.completed_boundary, bool):
            raise CAGATMAPPOTrainerError("completed_boundary must be boolean")


@dataclass(frozen=True)
class CAGATMAPPOUpdateDiagnostics:
    """One successful full-rollout update and its policy-version barrier."""

    update_index: int
    rollout_policy_version: int
    policy_version_after_update: int
    output: RecurrentPPOUpdateOutput

    def __post_init__(self) -> None:
        if self.update_index < 0 or self.rollout_policy_version < 0:
            raise CAGATMAPPOTrainerError(
                "update index and policy version must be non-negative"
            )
        if self.policy_version_after_update != self.rollout_policy_version + 1:
            raise CAGATMAPPOTrainerError(
                "one successful update must increment policy version once"
            )
        if not isinstance(self.output, RecurrentPPOUpdateOutput):
            raise TypeError("output must be a RecurrentPPOUpdateOutput")


@dataclass(frozen=True)
class CAGATMAPPOLossDiagnostics:
    """Aggregate PPO diagnostics across all four-epoch update records."""

    actor_loss: ScalarTrainingDiagnostics
    critic_loss: ScalarTrainingDiagnostics
    entropy: ScalarTrainingDiagnostics
    total_loss: ScalarTrainingDiagnostics
    ratio: ScalarTrainingDiagnostics
    actor_grad_norm_before_clip: ScalarTrainingDiagnostics
    critic_grad_norm_before_clip: ScalarTrainingDiagnostics


@dataclass(frozen=True)
class CAGATMAPPOTrainingResult:
    """Immutable production collection/update result without artifact writes."""

    total_environment_transitions: int
    optimized_transitions: int
    unused_final_tail_transitions: int
    started_episode_count: int
    completed_episode_count: int
    ppo_update_count: int
    initial_policy_version: int
    final_policy_version: int
    episode_seeds: tuple[int, ...]
    reward: CAGATMAPPORewardDiagnostics
    ppo: CAGATMAPPOLossDiagnostics
    episodes: tuple[CAGATMAPPOEpisodeDiagnostics, ...]
    updates: tuple[CAGATMAPPOUpdateDiagnostics, ...]
    route_outcomes: tuple[object, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        counts = (
            self.total_environment_transitions,
            self.optimized_transitions,
            self.unused_final_tail_transitions,
            self.started_episode_count,
            self.completed_episode_count,
            self.ppo_update_count,
            self.initial_policy_version,
            self.final_policy_version,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in counts
        ):
            raise CAGATMAPPOTrainerError(
                "training result counts must be non-negative integers"
            )
        if (
            self.optimized_transitions + self.unused_final_tail_transitions
            != self.total_environment_transitions
        ):
            raise CAGATMAPPOTrainerError(
                "optimized plus unused transitions must equal collected transitions"
            )
        if (
            self.started_episode_count != len(self.episodes)
            or self.started_episode_count != len(self.episode_seeds)
        ):
            raise CAGATMAPPOTrainerError(
                "started episode accounting is inconsistent"
            )
        if self.completed_episode_count != sum(
            item.completed_boundary for item in self.episodes
        ):
            raise CAGATMAPPOTrainerError(
                "completed episode accounting is inconsistent"
            )
        if self.ppo_update_count != len(self.updates):
            raise CAGATMAPPOTrainerError("PPO update accounting is inconsistent")
        if (
            self.final_policy_version
            != self.initial_policy_version + self.ppo_update_count
        ):
            raise CAGATMAPPOTrainerError(
                "final policy version differs from successful update count"
            )
        if self.reward.transition_count != self.total_environment_transitions:
            raise CAGATMAPPOTrainerError(
                "global reward count differs from collected transitions"
            )

    @property
    def completed_episodes(self) -> int:
        """Return the exact count of episodes that reached a real boundary."""

        return self.completed_episode_count

    @property
    def collected_environment_transitions(self) -> int:
        """Return the exact number of real environment steps collected."""

        return self.total_environment_transitions


class _ScalarAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0

    def add(self, value: Any, name: str) -> None:
        self.total = math.fsum((self.total, _finite(value, name)))
        self.count += 1

    def snapshot(self) -> ScalarTrainingDiagnostics:
        mean = 0.0 if self.count == 0 else self.total / self.count
        return ScalarTrainingDiagnostics(self.count, self.total, mean)

    def checkpoint_state(self) -> dict[str, Any]:
        return {"count": self.count, "total": self.total}

    @classmethod
    def from_checkpoint_state(
        cls, state: Mapping[str, Any], name: str
    ) -> "_ScalarAccumulator":
        if not isinstance(state, Mapping) or tuple(state) != ("count", "total"):
            raise CheckpointError(f"{name} accumulator fields are invalid")
        count = state["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise CheckpointError(f"{name} accumulator count is invalid")
        total = _finite(state["total"], f"{name}.total")
        result = cls()
        result.count = count
        result.total = total
        return result


class _RewardAccumulator:
    def __init__(self) -> None:
        self._values = {
            name: _ScalarAccumulator() for name in ("reward", *_REWARD_FIELDS)
        }

    def add(self, step_result: StepResult) -> None:
        reward = _finite(step_result.reward, "StepResult.reward")
        info_reward = step_result.info.get("reward")
        if not isinstance(info_reward, Mapping):
            raise CAGATMAPPOTrainerError(
                "StepResult.info['reward'] must be a mapping"
            )
        diagnostic_reward = _finite(
            info_reward.get("reward"), "StepResult.info.reward.reward"
        )
        if not math.isclose(
            reward, diagnostic_reward, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise CAGATMAPPOTrainerError(
                "StepResult reward and diagnostic reward differ"
            )
        self._values["reward"].add(reward, "StepResult.reward")
        for name in _REWARD_FIELDS:
            self._values[name].add(
                info_reward.get(name), f"StepResult.info.reward.{name}"
            )

    def snapshot(self) -> CAGATMAPPORewardDiagnostics:
        return CAGATMAPPORewardDiagnostics(
            **{
                name: accumulator.snapshot()
                for name, accumulator in self._values.items()
            }
        )

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            name: accumulator.checkpoint_state()
            for name, accumulator in self._values.items()
        }

    @classmethod
    def from_checkpoint_state(
        cls, state: Mapping[str, Any], name: str
    ) -> "_RewardAccumulator":
        expected = ("reward", *_REWARD_FIELDS)
        if not isinstance(state, Mapping) or tuple(state) != expected:
            raise CheckpointError(f"{name} reward accumulator fields are invalid")
        result = cls()
        result._values = {
            field: _ScalarAccumulator.from_checkpoint_state(
                state[field], f"{name}.{field}"
            )
            for field in expected
        }
        result.snapshot()
        return result


_REWARD_DIAGNOSTIC_NAMES = ("reward", *_REWARD_FIELDS)
_PPO_EPOCH_DIAGNOSTIC_FIELDS = (
    "epoch_index",
    "actor_loss",
    "critic_loss",
    "entropy_mean",
    "total_loss",
    "ratio_mean",
    "actor_grad_norm_before_clip",
    "critic_grad_norm_before_clip",
    "clip_max_norm",
)


def _reward_diagnostics_state(
    diagnostics: CAGATMAPPORewardDiagnostics,
) -> dict[str, Any]:
    if not isinstance(diagnostics, CAGATMAPPORewardDiagnostics):
        raise TypeError("diagnostics must be CAGATMAPPORewardDiagnostics")
    return {
        name: {
            "count": getattr(diagnostics, name).count,
            "total": getattr(diagnostics, name).total,
        }
        for name in _REWARD_DIAGNOSTIC_NAMES
    }


def _episode_checkpoint_state(
    item: CAGATMAPPOEpisodeDiagnostics,
) -> dict[str, Any]:
    return {
        "episode_index": item.episode_index,
        "environment_seed": item.environment_seed,
        "transition_count": item.transition_count,
        "completed_boundary": item.completed_boundary,
        "reward": _reward_diagnostics_state(item.reward),
    }


def _restore_episode_diagnostics(
    state: Mapping[str, Any],
) -> CAGATMAPPOEpisodeDiagnostics:
    expected = (
        "episode_index",
        "environment_seed",
        "transition_count",
        "completed_boundary",
        "reward",
    )
    if not isinstance(state, Mapping) or tuple(state) != expected:
        raise CheckpointError("episode diagnostic fields are invalid")
    reward = _RewardAccumulator.from_checkpoint_state(
        state["reward"], "episode.reward"
    ).snapshot()
    try:
        return CAGATMAPPOEpisodeDiagnostics(
            episode_index=state["episode_index"],
            environment_seed=state["environment_seed"],
            transition_count=state["transition_count"],
            completed_boundary=state["completed_boundary"],
            reward=reward,
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"invalid episode diagnostics: {exc}") from exc


def _update_checkpoint_state(
    item: CAGATMAPPOUpdateDiagnostics,
) -> dict[str, Any]:
    output = item.output
    return {
        "update_index": item.update_index,
        "rollout_policy_version": item.rollout_policy_version,
        "policy_version_after_update": item.policy_version_after_update,
        "output": {
            "epoch_diagnostics": [
                {
                    name: getattr(epoch, name)
                    for name in _PPO_EPOCH_DIAGNOSTIC_FIELDS
                }
                for epoch in output.epoch_diagnostics
            ],
            "chunk_count": output.chunk_count,
            "chunk_length": output.chunk_length,
            "valid_transition_count": output.valid_transition_count,
            "old_policy_snapshot_preserved": output.old_policy_snapshot_preserved,
        },
    }


def _restore_update_diagnostics(
    state: Mapping[str, Any],
) -> CAGATMAPPOUpdateDiagnostics:
    expected = (
        "update_index",
        "rollout_policy_version",
        "policy_version_after_update",
        "output",
    )
    if not isinstance(state, Mapping) or tuple(state) != expected:
        raise CheckpointError("PPO update diagnostic fields are invalid")
    output_state = state["output"]
    output_fields = (
        "epoch_diagnostics",
        "chunk_count",
        "chunk_length",
        "valid_transition_count",
        "old_policy_snapshot_preserved",
    )
    if not isinstance(output_state, Mapping) or tuple(output_state) != output_fields:
        raise CheckpointError("PPO output diagnostic fields are invalid")
    epoch_states = output_state["epoch_diagnostics"]
    if not isinstance(epoch_states, (list, tuple)):
        raise CheckpointError("PPO epoch diagnostics must be a sequence")
    epochs = []
    for epoch_state in epoch_states:
        if (
            not isinstance(epoch_state, Mapping)
            or tuple(epoch_state) != _PPO_EPOCH_DIAGNOSTIC_FIELDS
        ):
            raise CheckpointError("PPO epoch diagnostic fields are invalid")
        try:
            epochs.append(RecurrentPPOEpochDiagnostics(**dict(epoch_state)))
        except (TypeError, ValueError) as exc:
            raise CheckpointError(f"invalid PPO epoch diagnostics: {exc}") from exc
    try:
        output = RecurrentPPOUpdateOutput(
            epoch_diagnostics=tuple(epochs),
            chunk_count=output_state["chunk_count"],
            chunk_length=output_state["chunk_length"],
            valid_transition_count=output_state["valid_transition_count"],
            old_policy_snapshot_preserved=output_state[
                "old_policy_snapshot_preserved"
            ],
        )
        return CAGATMAPPOUpdateDiagnostics(
            update_index=state["update_index"],
            rollout_policy_version=state["rollout_policy_version"],
            policy_version_after_update=state["policy_version_after_update"],
            output=output,
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"invalid PPO update diagnostics: {exc}") from exc


def _restore_diagnostics_state(
    state: Mapping[str, Any],
) -> tuple[
    _RewardAccumulator,
    list[CAGATMAPPOEpisodeDiagnostics],
    list[CAGATMAPPOUpdateDiagnostics],
]:
    if (
        not isinstance(state, Mapping)
        or tuple(state) != CHECKPOINT_V1_DIAGNOSTICS_STATE_FIELDS
    ):
        raise CheckpointError("diagnostics_state fields are invalid")
    reward = _RewardAccumulator.from_checkpoint_state(
        state["reward_accumulators"], "global_reward"
    )
    episode_states = state["episode_history"]
    update_states = state["ppo_update_history"]
    if not isinstance(episode_states, (list, tuple)):
        raise CheckpointError("episode_history must be a sequence")
    if not isinstance(update_states, (list, tuple)):
        raise CheckpointError("ppo_update_history must be a sequence")
    episodes = [_restore_episode_diagnostics(item) for item in episode_states]
    updates = [_restore_update_diagnostics(item) for item in update_states]
    return reward, episodes, updates


def _validated_training_state(state: Mapping[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(state, Mapping)
        or tuple(state) != CHECKPOINT_V1_TRAINING_STATE_FIELDS
    ):
        raise CheckpointError("training_state fields are invalid")
    return dict(state)


def _aggregate_updates(
    updates: Sequence[CAGATMAPPOUpdateDiagnostics],
) -> CAGATMAPPOLossDiagnostics:
    names = (
        "actor_loss",
        "critic_loss",
        "entropy_mean",
        "total_loss",
        "ratio_mean",
        "actor_grad_norm_before_clip",
        "critic_grad_norm_before_clip",
    )
    values = {name: _ScalarAccumulator() for name in names}
    for update in updates:
        for epoch in update.output.epoch_diagnostics:
            for name in names:
                values[name].add(getattr(epoch, name), f"PPO.{name}")
    return CAGATMAPPOLossDiagnostics(
        actor_loss=values["actor_loss"].snapshot(),
        critic_loss=values["critic_loss"].snapshot(),
        entropy=values["entropy_mean"].snapshot(),
        total_loss=values["total_loss"].snapshot(),
        ratio=values["ratio_mean"].snapshot(),
        actor_grad_norm_before_clip=values[
            "actor_grad_norm_before_clip"
        ].snapshot(),
        critic_grad_norm_before_clip=values[
            "critic_grad_norm_before_clip"
        ].snapshot(),
    )


def build_training_episode_config(
    base_config: RunConfig,
    episode_index: int,
) -> RunConfig:
    """Return an episode-local config whose only resolved change is seed."""

    if not isinstance(base_config, RunConfig):
        raise TypeError("base_config must be a RunConfig")
    if (
        isinstance(episode_index, bool)
        or not isinstance(episode_index, int)
        or episode_index < 0
    ):
        raise CAGATMAPPOTrainerError(
            "episode_index must be a non-negative integer"
        )
    base_config.validate()
    episode_seed = derive_training_episode_seed(
        base_config.seed, episode_index
    )
    episode_config = copy.copy(base_config)
    object.__setattr__(episode_config, "seed", episode_seed)
    episode_config.validate()
    return episode_config


class CAGATMAPPOTrainer:
    """Collect real environment transitions and invoke the gated updater."""

    def __init__(
        self,
        config: RunConfig,
        *,
        actor: CAGATMAPPOActor | None = None,
        critic: MAPPOCentralizedCritic | None = None,
        environment_factory: Callable[[RunConfig], Any] | None = None,
        updater_factory: Callable[
            [
                CAGATMAPPOActor,
                MAPPOCentralizedCritic,
                RunConfig,
                CAGATMAPPOActionDistribution,
            ],
            Any,
        ]
        | None = None,
        progress_logger: Callable[[str], None] | None = None,
        oracle_budget: OracleCollectionBudget | None = None,
        matched_diagnostic_writer: Any | None = None,
    ) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        require_formal_rl_enabled(config)
        device_name = validate_mappo_training_device(
            config, cuda_available=torch.cuda.is_available()
        )
        self.config = config
        self.device = torch.device(device_name)
        self.dtype = torch.float32
        self.actor = CAGATMAPPOActor(config) if actor is None else actor
        self.critic = (
            MAPPOCentralizedCritic(config) if critic is None else critic
        )
        if not isinstance(self.actor, CAGATMAPPOActor):
            raise TypeError("actor must be a CAGATMAPPOActor")
        if not isinstance(self.critic, MAPPOCentralizedCritic):
            raise TypeError("critic must be a MAPPOCentralizedCritic")
        self.actor.to(device=self.device, dtype=self.dtype)
        self.critic.to(device=self.device, dtype=self.dtype)
        self._validate_model_placement()

        self.factual_actor_runtime = CAGATMAPPOFactualActorRuntime(
            self.actor,
            config,
            device=self.device,
            dtype=self.dtype,
        )
        self.actor_tensorizer = self.factual_actor_runtime.actor_tensorizer
        self.state_tensorizer = CentralizedStateTensorizer(config)
        self.action_distribution = (
            self.factual_actor_runtime.action_distribution
        )
        self.policy_generator = self.factual_actor_runtime.policy_generator
        self.rollout_buffer = CAGATMAPPORolloutBuffer(config)
        self.environment_factory = (
            U2UMECEnvironment
            if environment_factory is None
            else environment_factory
        )
        if not callable(self.environment_factory):
            raise TypeError("environment_factory must be callable")
        self.updater = (
            CAGATMAPPORecurrentPPOUpdater(
                self.actor,
                self.critic,
                config,
                action_distribution=self.action_distribution,
            )
            if updater_factory is None
            else updater_factory(
                self.actor,
                self.critic,
                config,
                self.action_distribution,
            )
        )
        if not callable(getattr(self.updater, "update", None)):
            raise TypeError("updater must expose update(buffer)")
        self._progress_logger = (
            _console_progress_logger
            if progress_logger is None
            else progress_logger
        )
        if not callable(self._progress_logger):
            raise TypeError("progress_logger must be callable")
        if oracle_budget is not None and not isinstance(
            oracle_budget, OracleCollectionBudget
        ):
            raise TypeError("oracle_budget must be OracleCollectionBudget or None")
        self._oracle_budget = oracle_budget
        if matched_diagnostic_writer is not None:
            for method in ("write_update", "write_episode"):
                if not callable(getattr(matched_diagnostic_writer, method, None)):
                    raise TypeError(
                        "matched_diagnostic_writer must expose write_update and write_episode"
                    )
        self._matched_diagnostic_writer = matched_diagnostic_writer

        self.policy_version = MAPPO_INITIAL_POLICY_VERSION
        self._rollout_policy_version: int | None = None
        self._has_run = False
        self._global_reward = _RewardAccumulator()
        self._episodes: list[CAGATMAPPOEpisodeDiagnostics] = []
        self._episode_seeds: list[int] = []
        self._updates: list[CAGATMAPPOUpdateDiagnostics] = []
        self._transitions = 0
        self._optimized = 0
        self._completed_episodes = 0
        self._next_episode_index = 0
        self._unused_final_tail = 0
        self._training_complete = False
        self._prepared_episode: tuple[Any, ResetResult, int] | None = None
        self._destination_oracle_calls = 0
        self._destination_shadow_transitions = 0
        self._oracle_runner_cache = FrozenOraclePolicyRunnerCache(
            config, device=self.device
        )

    def _validate_model_placement(self) -> None:
        placements: list[torch.device] = []
        for name, module in (("actor", self.actor), ("critic", self.critic)):
            parameters = tuple(module.parameters())
            if not parameters:
                raise CAGATMAPPOTrainerError(f"{name} has no parameters")
            devices = {parameter.device for parameter in parameters}
            if len(devices) != 1:
                raise CAGATMAPPOTrainerError(
                    f"{name} parameters span more than one device"
                )
            actual_device = next(iter(devices))
            if actual_device.type != self.device.type:
                raise CAGATMAPPOTrainerError(
                    f"{name} is not placed on explicit training_device"
                )
            if any(
                parameter.dtype != self.dtype for parameter in parameters
            ):
                raise CAGATMAPPOTrainerError(
                    f"{name} must use float32 parameters"
                )
            placements.append(actual_device)
        if placements[0] != placements[1]:
            raise CAGATMAPPOTrainerError(
                "actor and critic must share one actual training device"
            )
        self.device = placements[0]

    def _should_continue(
        self, completed_episodes: int, transitions: int
    ) -> bool:
        mappo = self.config.training.mappo
        return (
            completed_episodes < mappo.max_training_episodes
            and transitions < mappo.max_environment_transitions
        )

    def _start_episode(
        self, episode_index: int
    ) -> tuple[Any, ResetResult, int]:
        episode_config = build_training_episode_config(
            self.config, episode_index
        )
        environment = self.environment_factory(episode_config)
        reset_result = environment.reset()
        if not isinstance(reset_result, ResetResult):
            raise CAGATMAPPOTrainerError(
                "environment.reset() must return ResetResult"
            )
        observations = tuple(reset_result.observations)
        if len(observations) != self.actor.spec.uav_count:
            raise CAGATMAPPOTrainerError(
                "reset must return one observation per UAV"
            )
        if any(observation.slot != 0 for observation in observations):
            raise CAGATMAPPOTrainerError(
                "every real episode must reset at slot zero"
            )
        if reset_result.centralized_state.slot != 0:
            raise CAGATMAPPOTrainerError(
                "reset centralized state must use slot zero"
            )
        return environment, reset_result, episode_config.seed

    def _validate_step_result(
        self, slot: int, result: StepResult
    ) -> bool:
        if not isinstance(result, StepResult):
            raise CAGATMAPPOTrainerError(
                "environment.step() must return StepResult"
            )
        boundary = result.terminated or result.truncated
        if result.info.get("slot") != slot:
            raise CAGATMAPPOTrainerError(
                "StepResult.info slot differs from collected slot"
            )
        allowed = result.info.get("bootstrap_allowed")
        if not isinstance(allowed, bool) or allowed != (not boundary):
            raise CAGATMAPPOTrainerError(
                "bootstrap permission differs from episode boundary"
            )
        if boundary:
            if (
                result.observations is not None
                or result.centralized_state is not None
            ):
                raise CAGATMAPPOTrainerError(
                    "episode boundary must expose no fake next state"
                )
        elif (
            result.observations is None
            or result.centralized_state is None
        ):
            raise CAGATMAPPOTrainerError(
                "non-boundary transition requires the real next state"
            )
        if (
            AgentCreditMode(self.config.training.mappo.agent_credit_mode)
            is AgentCreditMode.ROLE_DECOMPOSED
        ):
            self._agent_reward_from_step(result)
        return boundary

    def _agent_reward_from_step(self, result: StepResult) -> Tensor:
        raw = result.info.get("agent_reward")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise CAGATMAPPOTrainerError(
                "role_decomposed StepResult.info requires agent_reward[A]"
            )
        if len(raw) != self.actor.spec.uav_count:
            raise CAGATMAPPOTrainerError(
                "StepResult agent_reward shape differs from the agent count"
            )
        values = tuple(
            _finite(value, f"StepResult.info.agent_reward[{index}]")
            for index, value in enumerate(raw)
        )
        residual = math.fsum(values) - _finite(
            result.reward, "StepResult.reward"
        )
        if abs(residual) > AGENT_REWARD_CONSERVATION_TOLERANCE:
            raise CAGATMAPPOTrainerError(
                "StepResult agent_reward violates team-reward conservation"
            )
        credit = result.info.get("agent_credit")
        if not isinstance(credit, Mapping):
            raise CAGATMAPPOTrainerError(
                "role_decomposed StepResult.info requires agent_credit telemetry"
            )
        declared_residual = _finite(
            credit.get("conservation_residual"),
            "StepResult.info.agent_credit.conservation_residual",
        )
        if abs(declared_residual) > AGENT_REWARD_CONSERVATION_TOLERANCE:
            raise CAGATMAPPOTrainerError(
                "agent credit telemetry reports a conservation violation"
            )
        return torch.tensor(values, dtype=self.dtype, device=self.device)

    @staticmethod
    def _execution_metadata(
        info: Mapping[str, Any],
        *,
        episode_id: int,
        global_rollout_index: int,
        policy_version: int,
        route_credit_enabled: bool,
    ) -> tuple[
        Sequence[Mapping[str, Any]],
        Mapping[str, Any],
    ]:
        executed = info.get("executed")
        if not isinstance(executed, Sequence) or isinstance(
            executed, (str, bytes)
        ):
            raise CAGATMAPPOTrainerError(
                "StepResult.info['executed'] must be a sequence"
            )
        if any(not isinstance(item, Mapping) for item in executed):
            raise CAGATMAPPOTrainerError(
                "executed summaries must be mappings"
            )
        summary = {
            "rejection": info.get("rejection", ()),
            "downgrade": info.get("downgrade", ()),
            "canonicalization": info.get("canonicalization", ()),
        }
        if route_credit_enabled:
            summary["route_credit_step"] = build_route_credit_step_metadata(
                info,
                episode_id=episode_id,
                global_rollout_index=global_rollout_index,
                policy_version=policy_version,
            )
        return executed, summary

    def _collect_paired_tx(self, environment, hidden_in, action_output, proposals):
        """Atomic bounded shadows; only JSON labels accompany factual samples."""
        import numpy as np
        import random
        from .tx_paired_shadow import (KEY, SCHEMA, FrozenShadowSampler, selection_state, select_sources, run_pair)
        from ..evaluation.tx_paired_shadow_pilot import LifecycleLedger
        from ..evaluation.route_oracle import actor_state_digest
        if environment.slot == 0 or getattr(self, "_tx_shadow_ledger", None) is None:
            self._tx_shadow_ledger = LifecycleLedger()
        key = (id(self.rollout_buffer), self.policy_version, len(self.rollout_buffer))
        selector_cache = getattr(self, "_tx_shadow_selector_cache", None)
        if selector_cache is None or selector_cache[:3] != key:
            selected, spent = selection_state(self.rollout_buffer, self.config, self.policy_version)
        else:
            selected, spent = selector_cache[3:]
        sources = select_sources(self.rollout_buffer, self.config, self.policy_version, environment,
                                 state=(frozenset(selected), spent))
        if (self._transitions - len(self.rollout_buffer) + self.config.training.mappo.rollout_length_slots
                > self.config.training.mappo.max_environment_transitions):
            # Deterministically skip the unoptimizable final partial rollout.
            sources = ()
        record = {"schema": SCHEMA, "policy_version": self.policy_version,
                  "rollout_offset": len(self.rollout_buffer),
                  "paired_mask": [False] * self.actor.spec.uav_count, "samples": []}
        def cache_next_state():
            # Valid only after this factual row is appended. An interrupted or
            # rejected append changes the expected length and forces a rebuild.
            new_sources = frozenset(selected) | frozenset(row["source"] for row in record["samples"])
            new_spent = spent + sum(row["shadow_steps"] for row in record["samples"])
            self._tx_shadow_selector_cache = (*key[:2], key[2] + 1, new_sources, new_spent)
        if not sources:
            cache_next_state()
            return record
        policy_rng = self.policy_generator.get_state().clone()
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []
        numpy_rng, python_rng = copy.deepcopy(np.random.get_state()), random.getstate()
        actor_before = actor_state_digest(self.actor)
        cached = getattr(self, "_tx_shadow_sampler", None)
        if cached is None or cached.version != self.policy_version:
            cached = FrozenShadowSampler(self.actor, self.config, self.policy_version)
            self._tx_shadow_sampler = cached
        for source in sources:
            old_p = float(action_output.scheduler_metrics["tx_p_transmit"][0, 0, source])
            row = run_pair(environment, cached, hidden_in, action_output.hidden_out, proposals,
                           source, old_p, self._tx_shadow_ledger)
            record["samples"].append(row)
            record["paired_mask"][source] = True
        actual_numpy = np.random.get_state()
        if (not torch.equal(policy_rng, self.policy_generator.get_state())
                or not torch.equal(cpu_rng, torch.get_rng_state())
                or any(not torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all() if cuda_rng else []))
                or numpy_rng[0] != actual_numpy[0] or not np.array_equal(numpy_rng[1], actual_numpy[1])
                or numpy_rng[2:] != actual_numpy[2:] or python_rng != random.getstate()
                or actor_before != actor_state_digest(self.actor)):
            raise CAGATMAPPOTrainerError("E13 shadow contaminated factual RNG/actor")
        cache_next_state()
        return record

    def _perform_update(
        self,
        updates: list[CAGATMAPPOUpdateDiagnostics],
        trajectory_tracker: TrajectoryCreditTracker | None = None,
    ) -> None:
        if not self.rollout_buffer.full:
            raise CAGATMAPPOTrainerError(
                "PPO update requires one full rollout"
            )
        if self._rollout_policy_version != self.policy_version:
            raise CAGATMAPPOTrainerError(
                "rollout contains more than one policy version"
            )
        frozen_version = self.policy_version
        self.rollout_buffer.finalize()
        if self.policy_version != frozen_version:
            raise CAGATMAPPOTrainerError(
                "policy version changed before PPO update"
            )
        generator_state = self.policy_generator.get_state().clone()
        credit_writer = getattr(self, "_conditional_credit_writer", None)
        if credit_writer is not None:
            credit_writer.begin_update(len(updates))
        mappo = self.config.training.mappo
        if (mappo.route_entropy_step_schedule_enabled
                or mappo.route_exploration_mode != "off"
                or mappo.tx_upper_optimization_mode == "deterministic_stability_v1"):
            # Snapshot the completed collection counter once per rollout;
            # recurrent PPO replay must not advance an environment schedule.
            from .destination_pretraining import schedule_environment_steps
            output = self.updater.update(
                self.rollout_buffer,
                collected_environment_steps=schedule_environment_steps(self._transitions, getattr(self, '_e16_provenance', None)),
            )
        else:
            # Preserve the exact legacy updater call shape when disabled.
            output = self.updater.update(self.rollout_buffer)
        if not isinstance(output, RecurrentPPOUpdateOutput):
            raise CAGATMAPPOTrainerError(
                "updater must return RecurrentPPOUpdateOutput"
            )
        if not torch.equal(
            generator_state, self.policy_generator.get_state()
        ):
            raise CAGATMAPPOTrainerError(
                "PPO update consumed the policy sampling stream"
            )
        if self.policy_version != frozen_version:
            raise CAGATMAPPOTrainerError(
                "updater changed Trainer policy version"
            )
        update_index = len(updates)
        integrated_summary = getattr(self.updater, "last_integrated_summary", None)
        if integrated_summary is not None:
            summary_path = Path(self.config.artifact_paths()["raw_metrics"]).parent / "integrated_scheduler_updates_v1.jsonl"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            row = {"run_id": self.config.run_id, "config_hash": self.config.config_hash,
                   "update_index": update_index, "environment_steps": self._transitions,
                   **integrated_summary}
            with summary_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        if credit_writer is not None:
            credit_writer.end_update(update_index, len(output.epoch_diagnostics))
        matched_diagnostic_writer = getattr(
            self, "_matched_diagnostic_writer", None
        )
        if matched_diagnostic_writer is not None:
            matched_diagnostic_writer.write_update(
                output,
                update_index=update_index,
                environment_step=self._transitions,
                uav_count=self.actor.spec.uav_count,
            )
        self.policy_version += 1
        if trajectory_tracker is not None:
            rollout_length = self.config.training.mappo.rollout_length_slots
            trajectory_tracker.observe_ppo_update(
                output,
                update_index=update_index,
                policy_version_before=frozen_version,
                policy_version_after=self.policy_version,
                rollout_start_index=self._transitions - rollout_length,
                rollout_length=rollout_length,
            )
        updates.append(
            CAGATMAPPOUpdateDiagnostics(
                update_index=update_index,
                rollout_policy_version=frozen_version,
                policy_version_after_update=self.policy_version,
                output=output,
            )
        )
        if mappo.route_training_telemetry_enabled:
            from .route_exploration import write_route_rollout_telemetry
            write_route_rollout_telemetry(
                self.config, self.rollout_buffer, environment_steps=self._transitions,
                optimized=True, ppo_epochs=getattr(
                    self.updater, "last_route_exploration_summary", None))
        if mappo.route_credit_mode == "task_event_value_v1":
            from .route_task_credit import write_task_credit_telemetry
            write_task_credit_telemetry(
                self.config, self.updater.last_route_task_records, self._transitions,
                self.updater.last_route_task_value_losses, buffer=self.rollout_buffer)
        if mappo.route_value_mode == "candidate_counterfactual_q_v1":
            from .route_option_q import write_route_q_telemetry
            write_route_q_telemetry(
                self.config, self.updater.last_route_option_q_records, self._transitions,
                self.updater.last_route_option_q_losses, buffer=self.rollout_buffer)
        if mappo.destination_oracle_rank_coefficient > 0:
            from .destination_ranking import write_destination_telemetry
            write_destination_telemetry(self.config, self.updater.last_destination_records, self._transitions)
        if mappo.destination_supervision_mode == 'offline_pretrain':
            if (self._destination_oracle_calls or self._destination_shadow_transitions
                    or getattr(self.updater, 'last_destination_records', [])
                    or getattr(self.updater, 'last_destination_gradient_probes', [])):
                raise CAGATMAPPOTrainerError('offline_pretrain executed online destination supervision')
            path = Path(self.config.artifact_paths()['raw_metrics']).parent / 'destination_supervision_v1.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'schema': 'destination_supervision_v1',
                    'environment_steps_since_warmstart': self._transitions,
                    'schedule_environment_steps': self._transitions + self._e16_provenance['base_training_transitions'],
                    'destination_oracle_calls': self._destination_oracle_calls,
                    'destination_shadow_transitions': self._destination_shadow_transitions,
                    'online_destination_ce': 0, 'offline_replay_ce': 0,
                    'tx_credit_mode': mappo.tx_upper_credit_mode}, allow_nan=False) + '\n')
        self.rollout_buffer.clear()
        self._rollout_policy_version = None

    def _report_episode_progress(
        self, episode: CAGATMAPPOEpisodeDiagnostics
    ) -> None:
        """Report existing episode and latest-update diagnostics only."""

        if self._updates:
            latest = _aggregate_updates((self._updates[-1],))
            actor_loss = f"{latest.actor_loss.mean:.6g}"
            critic_loss = f"{latest.critic_loss.mean:.6g}"
            entropy = f"{latest.entropy.mean:.6g}"
        else:
            actor_loss = critic_loss = entropy = "n/a"
        mappo = self.config.training.mappo
        self._progress_logger(
            "CA-GAT-MAPPO progress: "
            f"episode={episode.episode_index + 1}/"
            f"{mappo.max_training_episodes}, "
            f"collected_transitions={self._transitions}/"
            f"{mappo.max_environment_transitions}, "
            f"ppo_updates={len(self._updates)}, "
            f"episode_reward={episode.reward.reward.total:.6g}, "
            "completion_count="
            f"{episode.reward.completed_task_count.total:.6g}, "
            "expiration_count="
            f"{episode.reward.expired_task_count.total:.6g}, "
            f"actor_loss={actor_loss}, "
            f"critic_loss={critic_loss}, "
            f"entropy={entropy}, "
            f"device={self.device}"
        )

    def _optimizer_bundle(self) -> CAGATMAPPOOptimizerBundle:
        optimizers = getattr(self.updater, "optimizers", None)
        if not isinstance(optimizers, CAGATMAPPOOptimizerBundle):
            raise CheckpointError(
                "checkpointing requires updater.optimizers with actor and critic Adam"
            )
        optimizers.validate(self.actor, self.critic, self.config)
        return optimizers

    def _diagnostics_checkpoint_state(self) -> dict[str, Any]:
        return {
            "reward_accumulators": self._global_reward.checkpoint_state(),
            "episode_history": [
                _episode_checkpoint_state(item) for item in self._episodes
            ],
            "ppo_update_history": [
                _update_checkpoint_state(item) for item in self._updates
            ],
        }

    def _training_checkpoint_state(
        self, checkpoint_kind: str
    ) -> dict[str, Any]:
        active_length = len(self.rollout_buffer)
        active_version = self._rollout_policy_version
        return {
            "policy_version": self.policy_version,
            "started_episodes": len(self._episode_seeds),
            "completed_episodes": self._completed_episodes,
            "next_episode_index": self._next_episode_index,
            "collected_environment_transitions": self._transitions,
            "optimized_transitions": self._optimized,
            "ppo_update_count": len(self._updates),
            "unused_final_tail_transitions": self._unused_final_tail,
            "training_complete": self._training_complete,
            "active_rollout_length": active_length,
            "active_rollout_policy_version": active_version,
        }

    def _save_checkpoint(self, checkpoint_kind: str) -> Path:
        if checkpoint_kind == CHECKPOINT_KIND_PERIODIC_RESUME:
            target = Path(
                mappo_periodic_checkpoint_path(
                    self.config, self._transitions
                )
            )
        elif checkpoint_kind == CHECKPOINT_KIND_FINAL_COMPLETED:
            target = Path(mappo_final_checkpoint_path(self.config))
        else:
            raise CheckpointError("unknown checkpoint kind")
        active_rollout = serialize_active_rollout(
            self.rollout_buffer,
            self._rollout_policy_version,
            checkpoint_kind,
        )
        payload = build_checkpoint_payload(
            config=self.config,
            checkpoint_kind=checkpoint_kind,
            actor=self.actor,
            critic=self.critic,
            optimizers=self._optimizer_bundle(),
            policy_generator=self.policy_generator,
            training_state=self._training_checkpoint_state(checkpoint_kind),
            active_rollout_state=active_rollout,
            diagnostics_state=self._diagnostics_checkpoint_state(),
            dtype=self.dtype,
        )
        provenance = getattr(self, "_continuation_provenance", None)
        if provenance is not None:
            reference_info = provenance.get("replay_reference")
            if reference_info is not None and reference_info["status"] == "pending":
                if self._transitions > reference_info["environment_steps"]:
                    raise CheckpointError("replay verification boundary was skipped")
                if self._transitions == reference_info["environment_steps"]:
                    from .ca_gat_mappo_continuation import checkpoint_sha256, verify_replay_boundary

                    if checkpoint_sha256(reference_info["path"]) != reference_info["sha256"]:
                        raise CheckpointError("replay reference file changed; continuation stopped")
                    verify_replay_boundary(payload, self._replay_reference_payload)
                    reference_info["status"] = "verified_exact"
                    print(f"Replay verification passed at {self._transitions}; pending tail retained")
            payload["config_snapshot"]["_metadata"]["continuation"] = copy.deepcopy(provenance)
        if hasattr(self, '_e16_provenance'):
            payload['e16_provenance'] = {**self._e16_provenance,
                'config_hash': self.config.config_hash,
                'factual_transitions_since_warmstart': self._transitions,
                'stage': f'E16-FT{self._transitions // 1000}k',
                'destination_oracle_calls': self._destination_oracle_calls,
                'destination_shadow_transitions': self._destination_shadow_transitions,
                'online_destination_ce': 0, 'offline_replay_ce': 0}
        saved = atomic_save_checkpoint(payload, target)
        credit_writer = getattr(self, "_conditional_credit_writer", None)
        if checkpoint_kind == CHECKPOINT_KIND_PERIODIC_RESUME and credit_writer is not None:
            credit_writer.record_checkpoint(
                environment_steps=self._transitions,
                optimized_steps=self._optimized,
                filename=saved.name,
            )
        return saved

    @classmethod
    def resume_from_checkpoint(
        cls,
        config: RunConfig,
        checkpoint_path: str | Path,
        *,
        environment_factory: Callable[[RunConfig], Any] | None = None,
        updater_factory: Callable[
            [
                CAGATMAPPOActor,
                MAPPOCentralizedCritic,
                RunConfig,
                CAGATMAPPOActionDistribution,
            ],
            Any,
        ]
        | None = None,
        extend_training_budget: bool = False,
        verify_replay_against: str | Path | None = None,
    ) -> "CAGATMAPPOTrainer":
        """Explicitly restore one PERIODIC_RESUME checkpoint and next episode."""

        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        if config.training.mappo.trajectory_credit_telemetry_enabled:
            raise CAGATMAPPOTrainerError(
                "trajectory-credit telemetry V1 is fresh-run-only; resume is unsupported"
            )
        payload = load_checkpoint_payload(checkpoint_path)
        if (config.training.mappo.destination_supervision_mode == 'offline_pretrain'
                and 'e16_provenance' not in payload):
            raise CheckpointError('E16 resume requires retained pretraining provenance')
        validate_periodic_checkpoint_compatibility(
            config,
            payload,
            dtype=torch.float32,
            cuda_available=torch.cuda.is_available(),
            extend_training_budget=extend_training_budget,
        )
        from .ca_gat_mappo_continuation import checkpoint_sha256

        previous = copy.deepcopy(payload["config_snapshot"]["_metadata"].get("continuation"))
        provenance = previous
        if extend_training_budget:
            if (previous is not None and previous.get("replay_reference") is not None
                    and previous["replay_reference"]["status"] == "pending"):
                raise CheckpointError("finish the pending replay verification before extending again")
            provenance = {
                "schema_version": 1,
                "source_checkpoint": str(Path(checkpoint_path).resolve()),
                "source_sha256": checkpoint_sha256(checkpoint_path),
                "source_config_hash": payload["config_hash"],
                "target_config_hash": config.config_hash,
                "restored_environment_steps": payload["training_state"]["collected_environment_transitions"],
                "restored_pending_transitions": payload["active_rollout_state"]["rollout_length"],
                "target_environment_steps": config.training.mappo.max_environment_transitions,
                "diagnostic_sidecars_start_at": payload["training_state"]["collected_environment_transitions"],
                "previous": previous,
                "replay_reference": None,
            }
        if verify_replay_against is not None:
            if not extend_training_budget:
                raise CheckpointError("explicit replay reference requires budget extension")
            reference = load_checkpoint_payload(verify_replay_against)
            if (reference["checkpoint_kind"] != CHECKPOINT_KIND_FINAL_COMPLETED
                    or reference["config_hash"] != payload["config_hash"]
                    or canonicalize_config_identity(reference["config_snapshot"])
                    != canonicalize_config_identity(payload["config_snapshot"])
                    or reference["git_commit"] != payload["git_commit"]
                    or reference["runtime_provenance"] != payload["runtime_provenance"]):
                raise CheckpointError("replay reference must be the source run's matching FINAL_COMPLETED")
            step = reference["training_state"]["collected_environment_transitions"]
            if (step <= payload["training_state"]["collected_environment_transitions"]
                    or mappo_checkpoint_kind_at(config, step) != CHECKPOINT_KIND_PERIODIC_RESUME):
                raise CheckpointError("replay reference must precede target budget at a periodic boundary")
            provenance["replay_reference"] = {
                "path": str(Path(verify_replay_against).resolve()),
                "sha256": checkpoint_sha256(verify_replay_against),
                "environment_steps": step,
                "status": "pending",
            }
        reference_payload = None
        if provenance is not None and provenance.get("replay_reference") is not None:
            info = provenance["replay_reference"]
            if info["status"] == "pending":
                if checkpoint_sha256(info["path"]) != info["sha256"]:
                    raise CheckpointError("replay reference file changed; continuation stopped")
                reference_payload = load_checkpoint_payload(info["path"])
        training = _validated_training_state(payload["training_state"])
        global_reward, episodes, updates = _restore_diagnostics_state(
            payload["diagnostics_state"]
        )
        buffer, rollout_version = restore_active_rollout(
            config,
            payload["active_rollout_state"],
            checkpoint_kind=payload["checkpoint_kind"],
            policy_version=training["policy_version"],
        )
        if global_reward.snapshot().transition_count != training[
            "collected_environment_transitions"
        ]:
            raise CheckpointError("restored reward count differs from Trainer count")
        if (
            len(episodes) != training["started_episodes"]
            or len(episodes) != training["completed_episodes"]
            or len(updates) != training["ppo_update_count"]
        ):
            raise CheckpointError("restored diagnostic history counts are inconsistent")
        for index, episode in enumerate(episodes):
            expected_seed = derive_training_episode_seed(config.seed, index)
            if (
                episode.episode_index != index
                or not episode.completed_boundary
                or episode.environment_seed != expected_seed
            ):
                raise CheckpointError("episode history or seed sequence is invalid")
        for index, update in enumerate(updates):
            if (
                update.update_index != index
                or update.rollout_policy_version != index
                or update.policy_version_after_update != index + 1
            ):
                raise CheckpointError("PPO update history version sequence is invalid")

        trainer = cls(
            config,
            environment_factory=environment_factory,
            updater_factory=updater_factory,
        )
        trainer.policy_generator = restore_model_optimizer_and_rng(
            config=config,
            payload=payload,
            actor=trainer.actor,
            critic=trainer.critic,
            optimizers=trainer._optimizer_bundle(),
            dtype=trainer.dtype,
            extend_training_budget=extend_training_budget,
        )
        trainer.factual_actor_runtime.policy_generator = (
            trainer.policy_generator
        )
        trainer.rollout_buffer = buffer
        trainer.policy_version = training["policy_version"]
        trainer._rollout_policy_version = rollout_version
        trainer._global_reward = global_reward
        trainer._episodes = episodes
        trainer._episode_seeds = [
            item.environment_seed for item in episodes
        ]
        trainer._updates = updates
        trainer._transitions = training["collected_environment_transitions"]
        trainer._optimized = training["optimized_transitions"]
        trainer._completed_episodes = training["completed_episodes"]
        trainer._next_episode_index = training["next_episode_index"]
        trainer._unused_final_tail = 0
        trainer._training_complete = False
        if provenance is not None:
            trainer._continuation_provenance = provenance
            trainer._replay_reference_payload = reference_payload
        if 'e16_provenance' in payload:
            trainer._e16_provenance = dict(payload['e16_provenance'])
        trainer._prepared_episode = trainer._start_episode(
            trainer._next_episode_index
        )
        if config.training.mappo.conditional_credit_telemetry_enabled:
            trainer._conditional_credit_resume = (
                trainer._transitions, trainer._optimized, Path(checkpoint_path).name,
            )
        return trainer


    @classmethod
    def warm_start_from_checkpoint(cls, config, checkpoint_path):
        """Load weights only; all PPO counters, buffers and optimizers are new."""
        from .destination_pretraining import load_warmstart
        payload = load_warmstart(checkpoint_path, config)
        trainer = cls(config)
        trainer.actor.load_state_dict(payload['model_state']['actor'], strict=True)
        trainer.critic.load_state_dict(payload['model_state']['critic'], strict=True)
        trainer._e16_provenance = dict(payload['e16_provenance'])
        if any(opt.state for opt in (trainer._optimizer_bundle().actor_optimizer,
                                     trainer._optimizer_bundle().critic_optimizer)):
            raise CheckpointError('warm-start must have empty optimizer state')
        return trainer

    def train(self) -> CAGATMAPPOTrainingResult:
        """Execute the existing finite lifecycle without checkpoint artifacts."""

        result = self._run(checkpointing=False, pause_at_periodic=False)
        if not isinstance(result, CAGATMAPPOTrainingResult):
            raise CAGATMAPPOTrainerError("training ended without a final result")
        return result

    def train_with_checkpoints(self) -> CAGATMAPPOTrainingResult:
        """Run to completion while writing due periodic and final checkpoints."""

        result = self._run(checkpointing=True, pause_at_periodic=False)
        if not isinstance(result, CAGATMAPPOTrainingResult):
            raise CAGATMAPPOTrainerError("checkpointed training ended without a result")
        return result

    def train_until_periodic_checkpoint(self) -> Path:
        """Pause only after atomically saving the next safe periodic checkpoint."""

        result = self._run(checkpointing=True, pause_at_periodic=True)
        if not isinstance(result, Path):
            raise CAGATMAPPOTrainerError(
                "training completed before a periodic checkpoint was reached"
            )
        return result

    def _run(
        self,
        *,
        checkpointing: bool,
        pause_at_periodic: bool,
    ) -> CAGATMAPPOTrainingResult | Path:
        if (self.config.training.mappo.destination_supervision_mode == 'offline_pretrain'
                and not hasattr(self, '_e16_provenance')):
            raise CAGATMAPPOTrainerError('E16 requires validated warm-start weights and provenance')
        if self._has_run:
            raise CAGATMAPPOTrainerError(
                "one Trainer instance represents exactly one training run"
            )
        if self._training_complete:
            raise CAGATMAPPOTrainerError("training already complete")
        if (
            pause_at_periodic
            and self.config.training.mappo.trajectory_credit_telemetry_enabled
        ):
            raise CAGATMAPPOTrainerError(
                "trajectory-credit telemetry V1 cannot pause for later resume"
            )
        if checkpointing:
            validate_mappo_checkpoint_contract(self.config)
        elif pause_at_periodic:
            raise CAGATMAPPOTrainerError(
                "periodic pause requires checkpointing"
            )
        if getattr(self, "_continuation_provenance", None) is not None:
            if not checkpointing:
                raise CAGATMAPPOTrainerError("budget continuation requires checkpointing for replay verification")
            from ..artifacts import atomic_write_bytes_group

            manifest_path = Path(mappo_final_checkpoint_path(self.config)).parent / (
                f"continuation_from_{self._transitions}.json"
            )
            atomic_write_bytes_group(
                [(manifest_path, json.dumps(self._continuation_provenance, ensure_ascii=False,
                                            indent=2, sort_keys=True).encode("utf-8"))],
                group_name="continuation provenance",
            )
        self._has_run = True
        self.actor.train()
        self.critic.train()

        route_outcome_tracker = RouteOutcomeTracker(self.config.environment.slot_duration_s)
        diagnostic_collector: RouteDiagnosticSampleCollector | None = None
        diagnostic_writer: Any | None = None
        if self.config.training.mappo.route_diagnostic_samples_enabled:
            # The diagnostic sidecar is independent from route-credit telemetry
            # and is populated only from the already-created actor-safe step.
            from ..training_artifacts import RouteDiagnosticArtifactWriter

            diagnostic_collector = RouteDiagnosticSampleCollector(self.config)
            diagnostic_writer = RouteDiagnosticArtifactWriter(
                self.config,
                diagnostic_collector.schema_header(),
                allow_existing=(self._transitions > 0 or self._next_episode_index > 0),
            )
        oracle_collector: Oracle1Collector | None = None
        oracle_writer: RouteOracleArtifactWriter | None = None
        if self.config.training.mappo.route_oracle_counterfactual_enabled:
            oracle_runner = self._oracle_runner_cache.get(
                self.actor,
                self.policy_version,
                git_commit=self.config.git_commit,
                config_hash=self.config.config_hash,
                run_id=self.config.run_id,
            )
            oracle_collector = Oracle1Collector(
                self.config,
                oracle_runner,
                budget=self._oracle_budget,
                policy_identity=self._oracle_runner_cache.current_identity,
            )
            oracle_writer = RouteOracleArtifactWriter(
                self.config,
                oracle_schema_header(self.config, oracle_runner.identity),
                allow_existing=(self._transitions > 0 or self._next_episode_index > 0),
            )
        trajectory_tracker: TrajectoryCreditTracker | None = None
        self._conditional_credit_writer: ConditionalCreditTraceWriter | None = None
        if (self.config.training.mappo.conditional_credit_telemetry_enabled
                and self.config.training.mappo.tx_upper_decoder_mode == "legacy"
                and self.config.training.mappo.cpu_scheduler_mode == "legacy_hierarchical"):
            if not isinstance(self.updater, CAGATMAPPORecurrentPPOUpdater):
                raise CAGATMAPPOTrainerError(
                    "conditional-credit telemetry requires the production PPO updater"
                )
            resume_identity = getattr(self, "_conditional_credit_resume", None)
            self._conditional_credit_writer = (
                ConditionalCreditTraceWriter(self.config)
                if resume_identity is None else
                ConditionalCreditTraceWriter.reopen_at_checkpoint(
                    self.config, environment_steps=resume_identity[0],
                    optimized_steps=resume_identity[1], filename=resume_identity[2],
                )
            )
            self.updater.conditional_credit_sink = self._conditional_credit_writer
        if self.config.training.mappo.trajectory_credit_telemetry_enabled:
            # Keep training_artifacts import-order neutral: that module also
            # reads Trainer result types lazily when publishing final outputs.
            from ..training_artifacts import TrajectoryCreditArtifactWriter

            trajectory_writer = TrajectoryCreditArtifactWriter(self.config)
            trajectory_tracker = TrajectoryCreditTracker(
                self.config, trajectory_writer.write
            )
        episode_reward = _RewardAccumulator()
        episode_transitions = 0
        episode_index = self._next_episode_index
        if len(self._episode_seeds) != episode_index:
            raise CAGATMAPPOTrainerError(
                "next episode index differs from restored history"
            )
        if self._prepared_episode is None:
            environment, reset_result, episode_seed = self._start_episode(
                episode_index
            )
        else:
            environment, reset_result, episode_seed = self._prepared_episode
            self._prepared_episode = None
        expected_seed = derive_training_episode_seed(
            self.config.seed, episode_index
        )
        if episode_seed != expected_seed:
            raise CAGATMAPPOTrainerError(
                "prepared episode seed differs from frozen derivation"
            )
        self._episode_seeds.append(episode_seed)
        observations = reset_result.observations
        state = reset_result.centralized_state
        hidden = self.actor.initial_hidden(
            1, device=self.device, dtype=self.dtype
        )
        episode_active = True
        rollout_length = self.config.training.mappo.rollout_length_slots
        last_observed_transition_step: int | None = None

        while self._should_continue(
            self._completed_episodes, self._transitions
        ):
            slot = observations[0].slot
            if any(observation.slot != slot for observation in observations):
                raise CAGATMAPPOTrainerError(
                    "actor observations must describe one decision slot"
                )
            if state.slot != slot:
                raise CAGATMAPPOTrainerError(
                    "actor observations and critic state must share a slot"
                )
            episode_start = slot == 0
            if episode_start and episode_transitions != 0:
                raise CAGATMAPPOTrainerError(
                    "slot zero appeared without an episode boundary"
                )

            state_batch = self.state_tensorizer.encode_step(
                state, device=self.device, dtype=self.dtype
            )
            hidden_in = hidden.detach()
            step_policy_version = self.policy_version
            if len(self.rollout_buffer) == 0:
                self._rollout_policy_version = step_policy_version
            elif self._rollout_policy_version != step_policy_version:
                raise CAGATMAPPOTrainerError(
                    "policy changed during one rollout"
                )

            factual_actor_step = self.factual_actor_runtime.sample_step(
                observations,
                hidden_in,
                episode_start=episode_start,
            )
            actor_batch = factual_actor_step.actor_batch
            action_masks = factual_actor_step.action_masks
            action_output = factual_actor_step.action_output
            with torch.no_grad():
                raw_old_value = self.critic(state_batch)
                credit_mode = AgentCreditMode(
                    self.config.training.mappo.agent_credit_mode
                )
                old_value: Tensor = (
                    raw_old_value[0, 0, 0]
                    if credit_mode is AgentCreditMode.TEAM
                    else raw_old_value[0, 0]
                )

            proposals = tuple(action_output.proposals[0][0])
            destination_labels = []
            from .destination_pretraining import online_destination_enabled
            if online_destination_enabled(self.config):
                from ..evaluation.destination_system_oracle import collect_destination_labels, write_labels
                if any(isinstance(p.route, int) and not isinstance(p.route, bool) for p in proposals):
                    destination_runner = self._oracle_runner_cache.get(
                        self.actor, step_policy_version, git_commit=self.config.git_commit,
                        config_hash=self.config.config_hash, run_id=self.config.run_id)
                    destination_labels = collect_destination_labels(
                        environment, observations, proposals, action_output.hidden_out,
                        destination_runner, episode_id=episode_index, environment_step=self._transitions)
                    self._destination_oracle_calls += 1
                    self._destination_shadow_transitions += sum(row['shadow_transitions'] for row in destination_labels)
                    from .destination_ranking import masked_log_probs
                    scorer = self.actor.route_decoder.system_destination_scorer
                    with torch.no_grad():
                        scores = scorer(action_output.destination_features)
                        probabilities = masked_log_probs(scores, action_output.action_masks['route'][...,3:]).exp()
                    for row in destination_labels:
                        u = row['source_uav']
                        row['candidate_ids'] = self.actor.route_decoder.remote_candidate_ids[u].tolist()
                        row['factual_destination_logits'] = scores[0,0,u].cpu().tolist()
                        row['factual_destination_probabilities'] = probabilities[0,0,u].cpu().tolist()
                    write_labels(self.config, destination_labels)
            paired_metadata = None
            if self.config.training.mappo.tx_upper_credit_mode == "paired_shadow_v1":
                paired_metadata = self._collect_paired_tx(environment, hidden_in, action_output, proposals)
            if diagnostic_collector is not None:
                assert diagnostic_writer is not None
                diagnostic_records = diagnostic_collector.collect(
                    episode_id=episode_index,
                    global_environment_step=self._transitions,
                    policy_version=step_policy_version,
                    observations=observations,
                    proposals=proposals,
                    action_output=action_output,
                )
                for diagnostic_record in diagnostic_records:
                    diagnostic_writer.write(diagnostic_record)
            if oracle_collector is not None:
                assert oracle_writer is not None
                oracle_runner = self._oracle_runner_cache.get(
                    self.actor,
                    step_policy_version,
                    git_commit=self.config.git_commit,
                    config_hash=self.config.config_hash,
                    run_id=self.config.run_id,
                )
                if oracle_collector.runner is not oracle_runner:
                    oracle_collector = Oracle1Collector(
                        self.config,
                        oracle_runner,
                        budget=self._oracle_budget,
                        policy_identity=self._oracle_runner_cache.current_identity,
                    )
                oracle_decisions = oracle_collector.collect_pre_step(
                    environment,
                    observations,
                    proposals,
                    action_output.hidden_out.detach().clone(),
                    episode_id=episode_index,
                    global_environment_step=self._transitions,
                    policy_version=step_policy_version,
                    factual_environment_transitions=self._transitions,
                )
                for oracle_decision in oracle_decisions:
                    oracle_writer.write(oracle_decision)
            trajectory_capture: TrajectoryPreStepCapture | None = None
            if trajectory_tracker is not None:
                route_action_indices = {
                    proposal.uav_id: int(
                        action_output.action_indices["route"][
                            0, 0, proposal.uav_id
                        ].item()
                    )
                    for proposal in proposals
                }
                trajectory_capture = trajectory_tracker.capture_pre_step(
                    episode_id=episode_index,
                    rollout_index=self._transitions,
                    observations=observations,
                    proposals=proposals,
                    route_action_indices=route_action_indices,
                    environment=environment,
                )
            task_values_before = None
            if self.config.training.mappo.route_credit_mode == "task_event_value_v1":
                from .route_task_credit import capture_task_values
                task_values_before = capture_task_values(
                    environment, state_batch.features, self.critic, self.config)
            step_result = environment.step(proposals)
            if paired_metadata is not None:
                from ..evaluation.tx_paired_shadow_pilot import digest
                actual_digest = digest(step_result.info)
                if any(row["factual_first_step_digest"] != actual_digest for row in paired_metadata["samples"]):
                    raise CAGATMAPPOTrainerError("E13 factual first-step replay identity failed")
                self._tx_shadow_ledger.update(environment, step_result)
            if self._conditional_credit_writer is not None:
                self._conditional_credit_writer.record_step(
                    global_index=self._transitions, episode=episode_index,
                    slot=slot, policy_version=step_policy_version,
                    proposals=proposals, action_output=action_output,
                    info=step_result.info, environment=environment,
                )
            if trajectory_tracker is None:
                # Preserve the exact legacy post-step extraction order when disabled.
                route_action_indices = {
                    proposal.uav_id: int(
                        action_output.action_indices["route"][
                            0, 0, proposal.uav_id
                        ].item()
                    )
                    for proposal in proposals
                }
            route_outcome_tracker.observe_step(
                episode_index,
                step_result.info,
                route_action_indices=route_action_indices,
            )
            if trajectory_tracker is not None:
                assert trajectory_capture is not None
                trajectory_tracker.observe_step(
                    trajectory_capture, environment, step_result.info
                )
            boundary = self._validate_step_result(slot, step_result)
            if self.policy_version != step_policy_version:
                raise CAGATMAPPOTrainerError(
                    "policy version changed during transition collection"
                )

            bootstrap_value: Tensor | None = None
            next_state_batch = None
            if not boundary:
                assert step_result.centralized_state is not None
                next_state_batch = self.state_tensorizer.encode_step(
                    step_result.centralized_state,
                    device=self.device,
                    dtype=self.dtype,
                )
                with torch.no_grad():
                    raw_bootstrap = self.critic(next_state_batch)
                    bootstrap_value = (
                        raw_bootstrap[0, 0, 0]
                        if credit_mode is AgentCreditMode.TEAM
                        else raw_bootstrap[0, 0]
                    )

            executed, rejections = self._execution_metadata(
                step_result.info,
                episode_id=episode_index,
                global_rollout_index=self._transitions,
                policy_version=step_policy_version,
                route_credit_enabled=(
                    RouteCreditMode(
                        self.config.training.mappo.route_credit_mode
                    )
                    is RouteCreditMode.ROLLOUT_CAPPED_ROUTE_EVENT_NSTEP
                ),
            )
            rollout_reward: Tensor | float = (
                step_result.reward
                if credit_mode is AgentCreditMode.TEAM
                else self._agent_reward_from_step(step_result)
            )
            if paired_metadata is not None:
                rejections["tx_paired_shadow"] = paired_metadata
            if destination_labels:
                from .destination_ranking import KEY as DESTINATION_KEY
                rejections[DESTINATION_KEY] = destination_labels
            if self.config.training.mappo.route_training_telemetry_enabled:
                from .route_exploration import METADATA_KEY, collect_route_training_records
                rejections[METADATA_KEY] = collect_route_training_records(
                    action_output, environment_step=self._transitions,
                    episode_id=episode_index, slot=slot,
                    task_ids={o.uav_id: int(o.private_queues.unbound.head_task_id) for o in observations})
            if task_values_before is not None:
                from .route_task_credit import KEY, make_step_metadata
                task_values_after = ([] if boundary else capture_task_values(
                    environment, next_state_batch.features, self.critic, self.config))
                rejections[KEY] = make_step_metadata(
                    before=task_values_before, after=task_values_after, info=step_result.info,
                    policy=action_output, episode_id=episode_index,
                    environment_step=self._transitions, policy_version=step_policy_version,
                    capture_route_leaves=self.config.training.mappo.route_value_mode == "candidate_counterfactual_q_v1")
            self.rollout_buffer.append_step(
                slot=slot,
                actor_batch=actor_batch,
                action_mask_batch=action_masks,
                action_output=action_output,
                hidden_in=hidden_in,
                centralized_state=state_batch,
                old_value=old_value,
                reward=rollout_reward,
                terminated=step_result.terminated,
                truncated=step_result.truncated,
                episode_boundary=boundary,
                bootstrap_allowed=not boundary,
                bootstrap_value=bootstrap_value,
                executed_action_summary=executed,
                rejection_or_downgrade_summary=rejections,
            )
            self._global_reward.add(step_result)
            episode_reward.add(step_result)
            self._transitions += 1
            episode_transitions += 1
            last_observed_transition_step = slot
            hidden = action_output.hidden_out.detach()

            if boundary:
                episode_diagnostics = CAGATMAPPOEpisodeDiagnostics(
                    episode_index=episode_index,
                    environment_seed=episode_seed,
                    transition_count=episode_transitions,
                    completed_boundary=True,
                    reward=episode_reward.snapshot(),
                )
                self._episodes.append(episode_diagnostics)
                self._completed_episodes += 1
                self._next_episode_index = episode_index + 1
                episode_active = False
                matched_diagnostic_writer = getattr(
                    self, "_matched_diagnostic_writer", None
                )
                if matched_diagnostic_writer is not None:
                    metrics = step_result.info.get("metrics")
                    if not isinstance(metrics, Mapping):
                        raise CAGATMAPPOTrainerError(
                            "matched episode diagnostics require StepResult.info metrics"
                        )
                    matched_diagnostic_writer.write_episode(
                        episode_index=episode_index,
                        environment_step=self._transitions,
                        transition_count=episode_transitions,
                        episode_return=episode_diagnostics.reward.reward.total,
                        metrics=metrics,
                        route_outcomes=route_outcome_tracker.finalize(),
                    )
                if self.rollout_buffer.full:
                    self._perform_update(self._updates, trajectory_tracker)
                    self._optimized += rollout_length
                self._report_episode_progress(episode_diagnostics)

                should_continue = self._should_continue(
                    self._completed_episodes, self._transitions
                )
                if checkpointing:
                    kind = mappo_checkpoint_kind_at(
                        self.config, self._transitions
                    )
                    if (
                        kind == CHECKPOINT_KIND_PERIODIC_RESUME
                        and should_continue
                    ):
                        checkpoint_path = self._save_checkpoint(kind)
                        if pause_at_periodic:
                            if self._conditional_credit_writer is not None:
                                self._conditional_credit_writer.pause()
                            return checkpoint_path
                if not should_continue:
                    break

                episode_index = self._next_episode_index
                (
                    environment,
                    reset_result,
                    episode_seed,
                ) = self._start_episode(episode_index)
                self._episode_seeds.append(episode_seed)
                observations = reset_result.observations
                state = reset_result.centralized_state
                hidden = self.actor.initial_hidden(
                    1, device=self.device, dtype=self.dtype
                )
                episode_transitions = 0
                episode_reward = _RewardAccumulator()
                episode_active = True
            else:
                assert step_result.observations is not None
                assert step_result.centralized_state is not None
                observations = step_result.observations
                state = step_result.centralized_state
                if self.rollout_buffer.full:
                    self._perform_update(self._updates, trajectory_tracker)
                    self._optimized += rollout_length
                if checkpointing and mappo_checkpoint_kind_at(
                    self.config, self._transitions
                ) is not None:
                    raise CheckpointError(
                        "checkpoint boundary is not a true episode boundary"
                    )

        if episode_active and episode_transitions:
            self._episodes.append(
                CAGATMAPPOEpisodeDiagnostics(
                    episode_index=episode_index,
                    environment_seed=episode_seed,
                    transition_count=episode_transitions,
                    completed_boundary=False,
                    reward=episode_reward.snapshot(),
                )
            )

        self._unused_final_tail = len(self.rollout_buffer)
        if trajectory_tracker is not None:
            if last_observed_transition_step is None:
                raise CAGATMAPPOTrainerError(
                    "trajectory-credit finalization requires one observed transition"
                )
            trajectory_tracker.finalize_collection(
                environment=environment,
                episode_id=episode_index,
                observed_transition_step=last_observed_transition_step,
                optimized_transitions=self._optimized,
            )
        if self._unused_final_tail:
            if self.rollout_buffer.finalized:
                raise CAGATMAPPOTrainerError(
                    "final partial rollout must not be finalized"
                )
            if self.config.training.mappo.route_training_telemetry_enabled:
                from .route_exploration import write_route_rollout_telemetry
                write_route_rollout_telemetry(
                    self.config, self.rollout_buffer, environment_steps=self._transitions,
                    optimized=False)
            if self.config.training.mappo.route_credit_mode == "task_event_value_v1":
                from .route_task_credit import write_task_credit_telemetry
                write_task_credit_telemetry(
                    self.config, [], self._transitions, [], buffer=self.rollout_buffer, optimized=False)
            if self.config.training.mappo.route_value_mode == "candidate_counterfactual_q_v1":
                from .route_option_q import write_route_q_telemetry
                write_route_q_telemetry(
                    self.config, [], self._transitions, [], buffer=self.rollout_buffer, optimized=False)
            self.rollout_buffer.clear()
            self._rollout_policy_version = None
        if self._optimized != len(self._updates) * rollout_length:
            raise CAGATMAPPOTrainerError(
                "optimized transition accounting differs from updates"
            )
        if self._optimized + self._unused_final_tail != self._transitions:
            raise CAGATMAPPOTrainerError(
                "final transition accounting is inconsistent"
            )
        if self._conditional_credit_writer is not None:
            self._conditional_credit_writer.record_censored(
                environment=environment, episode=episode_index,
                global_index=self._transitions,
            )
            self._conditional_credit_writer.close(
                environment_steps=self._transitions,
                optimized_steps=self._optimized,
            )
        self._training_complete = True

        if checkpointing:
            if episode_active and episode_transitions:
                raise CheckpointError(
                    "Checkpoint V1 cannot finalize a mid-episode training stop"
                )
            kind = mappo_checkpoint_kind_at(
                self.config, self._transitions
            )
            if kind != CHECKPOINT_KIND_FINAL_COMPLETED:
                raise CheckpointError(
                    "checkpointed training did not reach FINAL_COMPLETED"
                )
            self._save_checkpoint(CHECKPOINT_KIND_FINAL_COMPLETED)

        return CAGATMAPPOTrainingResult(
            total_environment_transitions=self._transitions,
            optimized_transitions=self._optimized,
            unused_final_tail_transitions=self._unused_final_tail,
            started_episode_count=len(self._episodes),
            completed_episode_count=self._completed_episodes,
            ppo_update_count=len(self._updates),
            initial_policy_version=MAPPO_INITIAL_POLICY_VERSION,
            final_policy_version=self.policy_version,
            episode_seeds=tuple(self._episode_seeds),
            reward=self._global_reward.snapshot(),
            ppo=_aggregate_updates(self._updates),
            route_outcomes=route_outcome_tracker.finalize(),
            episodes=tuple(self._episodes),
            updates=tuple(self._updates),
        )


__all__ = [
    "CAGATMAPPOLossDiagnostics",
    "CAGATMAPPOEpisodeDiagnostics",
    "CAGATMAPPORewardDiagnostics",
    "CAGATMAPPOTrainer",
    "CAGATMAPPOTrainerError",
    "CAGATMAPPOTrainingResult",
    "CAGATMAPPOUpdateDiagnostics",
    "ScalarTrainingDiagnostics",
    "build_training_episode_config",
]
