"""Evaluation-only Route-Service Factorial Diagnosis.

This module composes the route branch and the six service branches from the
frozen learned actor and the existing actor-visible heuristic.  It deliberately
does not register a CLI entry, alter the formal evaluation method suite, or
touch training state.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from ..artifacts import atomic_write_text_group, require_artifact_targets_absent
from ..config import RunConfig
from ..env.actions import ActionProposal
from ..env.environment import U2UMECEnvironment
from ..policies.heuristic_policy import HeuristicPolicy
from .actor_loader import checkpoint_sha256
from .metrics import (
    EvaluationEpisodeResult,
    aggregate_episode_metrics,
    build_episode_metrics,
)
from .runner import (
    FormalEvaluationError,
    FormalEvaluationRunner,
    _canonical_sha256,
    _compact_json,
    _pretty_json,
    _proposal_record,
    _state_dict_snapshot,
    _tensor_sha256,
)


ROUTE_SERVICE_FACTORIAL_SCHEMA_VERSION = "route_service_factorial.v1"
ROUTE_SERVICE_FACTORIAL_ARTIFACT_FILENAMES = (
    "manifest.json",
    "episode_metrics.jsonl",
    "aggregate_metrics.json",
)
SERVICE_BRANCHES = (
    "tx_select",
    "resource_group",
    "resource_width",
    "power_level",
    "cpu_queue",
    "cpu_frequency",
)


class RouteServiceFactorialError(FormalEvaluationError):
    """Raised when factorial composition or telemetry fails closed."""


class RouteServiceMode(str, Enum):
    """The four cells of the Route x Service factorial diagnosis."""

    FULL_LEARNED = "full_learned"
    LEARNED_ROUTE_HEURISTIC_SERVICE = "learned_route_heuristic_service"
    HEURISTIC_ROUTE_LEARNED_SERVICE = "heuristic_route_learned_service"
    FULL_HEURISTIC = "full_heuristic"

    @property
    def route_source(self) -> str:
        return (
            "learned"
            if self in {
                RouteServiceMode.FULL_LEARNED,
                RouteServiceMode.LEARNED_ROUTE_HEURISTIC_SERVICE,
            }
            else "heuristic"
        )

    @property
    def service_source(self) -> str:
        return (
            "learned"
            if self in {
                RouteServiceMode.FULL_LEARNED,
                RouteServiceMode.HEURISTIC_ROUTE_LEARNED_SERVICE,
            }
            else "heuristic"
        )

    @property
    def uses_learned_actor(self) -> bool:
        return self.route_source == "learned" or self.service_source == "learned"

    @property
    def uses_heuristic(self) -> bool:
        return self.route_source == "heuristic" or self.service_source == "heuristic"


ROUTE_SERVICE_FACTORIAL_MODES = tuple(RouteServiceMode)
ROUTE_SERVICE_FACTORIAL_METHOD_SUITE = tuple(
    mode.value for mode in ROUTE_SERVICE_FACTORIAL_MODES
)


class RouteServiceProposalAdapter:
    """Compose one final proposal without repairing or resampling any branch."""

    @staticmethod
    def compose(
        mode: RouteServiceMode,
        *,
        action_masks: Any,
        learned_proposal: ActionProposal | None,
        heuristic_proposal: ActionProposal | None,
    ) -> ActionProposal:
        if not isinstance(mode, RouteServiceMode):
            try:
                mode = RouteServiceMode(mode)
            except (TypeError, ValueError) as exc:
                raise RouteServiceFactorialError(
                    f"unknown Route-Service mode {mode!r}"
                ) from exc

        required = []
        if mode.uses_learned_actor:
            required.append(("learned", learned_proposal))
        if mode.uses_heuristic:
            required.append(("heuristic", heuristic_proposal))
        for label, proposal in required:
            if not isinstance(proposal, ActionProposal):
                raise RouteServiceFactorialError(
                    f"{label} proposal is required for mode {mode.value}"
                )

        proposals = [proposal for _label, proposal in required]
        if len({proposal.uav_id for proposal in proposals}) != 1:
            raise RouteServiceFactorialError(
                "learned and heuristic proposals have different UAV identifiers"
            )

        route_proposal = (
            learned_proposal if mode.route_source == "learned" else heuristic_proposal
        )
        service_proposal = (
            learned_proposal
            if mode.service_source == "learned"
            else heuristic_proposal
        )
        assert route_proposal is not None and service_proposal is not None
        proposal = ActionProposal(
            uav_id=route_proposal.uav_id,
            route=route_proposal.route,
            tx_select=service_proposal.tx_select,
            resource_group=service_proposal.resource_group,
            resource_width=service_proposal.resource_width,
            power_level=service_proposal.power_level,
            cpu_queue=service_proposal.cpu_queue,
            cpu_frequency=service_proposal.cpu_frequency,
        )
        if not action_masks.is_legal(proposal):
            raise RouteServiceFactorialError(
                f"composed proposal for UAV {proposal.uav_id} violates action masks"
            )
        return proposal

    @classmethod
    def compose_joint(
        cls,
        mode: RouteServiceMode,
        *,
        observations: Sequence[Any],
        learned_proposals: Sequence[ActionProposal] | None,
        heuristic_proposals: Sequence[ActionProposal] | None,
    ) -> tuple[ActionProposal, ...]:
        count = len(observations)
        if count == 0:
            raise RouteServiceFactorialError("factorial composition requires observations")
        if learned_proposals is not None and len(learned_proposals) != count:
            raise RouteServiceFactorialError(
                "learned proposals are not aligned with observations"
            )
        if heuristic_proposals is not None and len(heuristic_proposals) != count:
            raise RouteServiceFactorialError(
                "heuristic proposals are not aligned with observations"
            )

        composed: list[ActionProposal] = []
        for index, observation in enumerate(observations):
            learned = (
                None if learned_proposals is None else learned_proposals[index]
            )
            heuristic = (
                None if heuristic_proposals is None else heuristic_proposals[index]
            )
            proposal = cls.compose(
                mode,
                action_masks=observation.action_masks,
                learned_proposal=learned,
                heuristic_proposal=heuristic,
            )
            if proposal.uav_id != observation.uav_id:
                raise RouteServiceFactorialError(
                    "composed proposal is not aligned with its observation"
                )
            composed.append(proposal)
        return tuple(composed)


@dataclass
class FactorialPolicyWrapper:
    """Generate both source proposals and return one legal factorial proposal."""

    mode: RouteServiceMode
    heuristic_policy: HeuristicPolicy
    learned_action_fn: Callable[
        [Sequence[Any], torch.Tensor],
        tuple[tuple[ActionProposal, ...], torch.Tensor, Any],
    ]

    def act(
        self,
        observations: Sequence[Any],
        hidden: torch.Tensor | None,
    ) -> tuple[tuple[ActionProposal, ...], torch.Tensor | None, Any | None]:
        learned: tuple[ActionProposal, ...] | None = None
        heuristic: tuple[ActionProposal, ...] | None = None
        action_output: Any | None = None
        next_hidden = hidden

        if self.mode.uses_learned_actor:
            if hidden is None:
                raise RouteServiceFactorialError(
                    f"mode {self.mode.value} requires learned actor hidden state"
                )
            learned, next_hidden, action_output = self.learned_action_fn(
                observations, hidden
            )
        if self.mode.uses_heuristic:
            heuristic = tuple(
                self.heuristic_policy.act(observation)
                for observation in observations
            )

        proposals = RouteServiceProposalAdapter.compose_joint(
            self.mode,
            observations=observations,
            learned_proposals=learned,
            heuristic_proposals=heuristic,
        )
        return proposals, next_hidden, action_output


@dataclass(frozen=True)
class RouteServiceEpisodeTelemetry:
    """Task-grounded route and downstream service statistics for one episode."""

    remote_selection_count: int
    local_selection_count: int
    destination_distribution: Mapping[str, int]
    remote_binding_task_count: int
    ever_tx_service_count: int
    ever_tx_service_ratio: float | None
    tx_complete_count: int
    tx_complete_ratio: float | None
    reach_destination_cpu_count: int
    reach_destination_cpu_ratio: float | None
    done_on_time_count: int
    done_on_time_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RouteServiceTelemetryCollector:
    """Collect factual task events without reading centralized state or the future."""

    def __init__(self) -> None:
        self.remote_selection_count = 0
        self.local_selection_count = 0
        self.destination_distribution: dict[str, int] = {}
        self.remote_bindings: dict[int, tuple[int, int]] = {}
        self.tasks_with_tx_service: set[int] = set()

    def observe_step(
        self,
        proposals: Sequence[ActionProposal],
        step_info: Mapping[str, Any],
    ) -> None:
        factual = tuple(step_info.get("proposal", ()))
        expected = tuple(_proposal_record(proposal) for proposal in proposals)
        if factual != expected:
            raise RouteServiceFactorialError(
                "environment step proposal trace differs from factorial proposals"
            )

        service = step_info.get("service", {})
        for record in service.get("routing", ()):
            if not bool(record.get("applied", False)):
                continue
            route = record.get("proposal")
            task_id = record.get("task_id")
            source = record.get("uav_id")
            destination = record.get("destination")
            if route == "local":
                self.local_selection_count += 1
                continue
            if isinstance(route, int) and not isinstance(route, bool):
                if not all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in (task_id, source, destination)
                ):
                    raise RouteServiceFactorialError(
                        "successful Remote binding omitted integer task metadata"
                    )
                if destination != route or destination == source:
                    raise RouteServiceFactorialError(
                        "successful Remote binding has inconsistent destination"
                    )
                if task_id in self.remote_bindings:
                    raise RouteServiceFactorialError(
                        f"Remote task {task_id} was bound more than once"
                    )
                self.remote_selection_count += 1
                key = str(destination)
                self.destination_distribution[key] = (
                    self.destination_distribution.get(key, 0) + 1
                )
                self.remote_bindings[task_id] = (source, destination)
                continue
            raise RouteServiceFactorialError(
                f"applied routing record has unsupported route {route!r}"
            )

        for link in service.get("links", ()):
            for task_service in link.get("task_services", ()):
                task_id = task_service.get("task_id")
                amount = task_service.get("amount")
                if (
                    isinstance(task_id, bool)
                    or not isinstance(task_id, int)
                    or isinstance(amount, bool)
                    or not isinstance(amount, (int, float))
                    or not math.isfinite(float(amount))
                    or float(amount) < 0.0
                ):
                    raise RouteServiceFactorialError(
                        "TX task service record is malformed"
                    )
                if float(amount) > 0.0:
                    self.tasks_with_tx_service.add(task_id)

    def finalize(
        self,
        environment: U2UMECEnvironment,
    ) -> RouteServiceEpisodeTelemetry:
        if environment.lifecycle is None:
            raise RouteServiceFactorialError(
                "environment lifecycle is unavailable for factorial telemetry"
            )
        denominator = len(self.remote_bindings)
        ever_tx = 0
        tx_complete = 0
        reach_cpu = 0
        done_on_time = 0
        for task_id, (source, destination) in self.remote_bindings.items():
            task = environment.lifecycle.tasks.get(task_id)
            if task is None:
                raise RouteServiceFactorialError(
                    f"Remote binding task {task_id} is missing from lifecycle"
                )
            if task.source_uav != source or task.destination != destination:
                raise RouteServiceFactorialError(
                    f"Remote binding task {task_id} changed locked identity"
                )
            ever_tx += int(task_id in self.tasks_with_tx_service)
            tx_complete += int(float(task.remaining_bits) == 0.0)
            reach_cpu += int(task.cpu_entry_slot is not None)
            done_on_time += int(
                task.outcome.value == "done"
                and task.completion_slot is not None
                and task.completion_slot <= task.deadline_slot
            )
        if tx_complete != reach_cpu:
            raise RouteServiceFactorialError(
                "TX completion and destination CPU entry diverged"
            )

        def ratio(numerator: int) -> float | None:
            return numerator / denominator if denominator else None

        return RouteServiceEpisodeTelemetry(
            remote_selection_count=self.remote_selection_count,
            local_selection_count=self.local_selection_count,
            destination_distribution=dict(
                sorted(self.destination_distribution.items(), key=lambda item: int(item[0]))
            ),
            remote_binding_task_count=denominator,
            ever_tx_service_count=ever_tx,
            ever_tx_service_ratio=ratio(ever_tx),
            tx_complete_count=tx_complete,
            tx_complete_ratio=ratio(tx_complete),
            reach_destination_cpu_count=reach_cpu,
            reach_destination_cpu_ratio=ratio(reach_cpu),
            done_on_time_count=done_on_time,
            done_on_time_ratio=ratio(done_on_time),
        )


@dataclass(frozen=True)
class RouteServiceFactorialEpisodeResult:
    """One mode/seed episode with base formal metrics plus decomposition."""

    mode: RouteServiceMode
    evaluation: EvaluationEpisodeResult
    telemetry: RouteServiceEpisodeTelemetry

    def artifact_record(self, factorial_run_id: str) -> dict[str, Any]:
        return {
            "schema_version": ROUTE_SERVICE_FACTORIAL_SCHEMA_VERSION,
            "factorial_run_id": factorial_run_id,
            "mode": self.mode.value,
            "route_source": self.mode.route_source,
            "service_source": self.mode.service_source,
            "evaluation_seed": self.evaluation.evaluation_seed,
            "episode_horizon": self.evaluation.episode_horizon,
            "metrics": dict(self.evaluation.metrics),
            "route_service": self.telemetry.to_dict(),
            "action_trace_sha256": self.evaluation.action_trace_sha256,
            "external_trace_sha256": self.evaluation.external_trace_sha256,
            "hidden_trace_sha256": list(self.evaluation.hidden_trace_sha256),
            "invariants": dict(self.evaluation.invariants),
        }


@dataclass(frozen=True)
class RouteServiceFactorialResult:
    """Complete in-memory diagnosis and optional immutable artifacts."""

    factorial_run_id: str
    manifest: Mapping[str, Any]
    episodes: tuple[RouteServiceFactorialEpisodeResult, ...]
    aggregate_metrics: Mapping[str, Any]
    artifacts: tuple[str, ...]


class RouteServiceFactorialRunner(FormalEvaluationRunner):
    """Run the four factorial cells through the existing formal backend."""

    def __init__(
        self,
        config: RunConfig,
        checkpoint_path: str | Path,
        *,
        evaluation_device: str,
        environment_factory: Callable[[RunConfig], U2UMECEnvironment] | None = None,
    ) -> None:
        super().__init__(
            config,
            checkpoint_path,
            evaluation_device=evaluation_device,
            environment_factory=environment_factory,
            collect_route_telemetry=False,
        )
        identity = {
            "schema_version": ROUTE_SERVICE_FACTORIAL_SCHEMA_VERSION,
            "source_checkpoint_sha256": self.loaded_actor.source_checkpoint_sha256,
            "evaluation_config_sha256": self.evaluation_config_sha256,
            "mode_suite": list(ROUTE_SERVICE_FACTORIAL_METHOD_SUITE),
            "evaluator_git_commit": self.config.git_commit,
            "evaluator_git_dirty": self.config.git_dirty,
            "evaluation_device": self.loaded_actor.evaluation_device,
        }
        digest = _canonical_sha256(identity)
        self.factorial_run_id = (
            f"route-service-factorial__{self.config.scenario_id}__"
            f"chk-{self.loaded_actor.source_checkpoint_sha256[:12]}__"
            f"diag-{digest[:12]}"
        )

    def artifact_paths(self) -> Mapping[str, Path]:
        root = (
            Path(self.config.output.logs_dir)
            / "diagnostics"
            / self.factorial_run_id
        )
        return {
            name: root / filename
            for name, filename in zip(
                ("manifest", "episode_metrics", "aggregate_metrics"),
                ROUTE_SERVICE_FACTORIAL_ARTIFACT_FILENAMES,
            )
        }

    def run(self, *, write_artifacts: bool = True) -> RouteServiceFactorialResult:
        paths = self.artifact_paths()
        if write_artifacts:
            require_artifact_targets_absent(
                paths.values(),
                group_name="Route-Service factorial diagnostic artifact group",
            )

        actor = self.loaded_actor.actor
        actor.eval()
        before_state = _state_dict_snapshot(actor)
        before_requires_grad = tuple(
            (name, parameter.requires_grad)
            for name, parameter in actor.named_parameters()
        )
        checkpoint_before = checkpoint_sha256(
            self.loaded_actor.source_checkpoint_path
        )

        episodes: list[RouteServiceFactorialEpisodeResult] = []
        for episode_id, evaluation_seed in enumerate(
            self.config.evaluation.evaluation_seeds
        ):
            seed_episodes = [
                self._run_factorial_episode(
                    mode,
                    int(evaluation_seed),
                    episode_id=episode_id,
                )
                for mode in ROUTE_SERVICE_FACTORIAL_MODES
            ]
            trace_hashes = {
                item.mode.value: item.evaluation.external_trace_sha256
                for item in seed_episodes
            }
            if len(set(trace_hashes.values())) != 1:
                raise RouteServiceFactorialError(
                    "external environment realization diverged across factorial "
                    f"modes for evaluation seed {evaluation_seed}: {trace_hashes}"
                )
            episodes.extend(seed_episodes)

        if actor.training:
            raise RouteServiceFactorialError("frozen actor left evaluation mode")
        after_state = _state_dict_snapshot(actor)
        if tuple(before_state) != tuple(after_state) or any(
            not torch.equal(before_state[name], after_state[name])
            for name in before_state
        ):
            raise RouteServiceFactorialError(
                "frozen actor state changed during factorial evaluation"
            )
        after_requires_grad = tuple(
            (name, parameter.requires_grad)
            for name, parameter in actor.named_parameters()
        )
        if before_requires_grad != after_requires_grad:
            raise RouteServiceFactorialError(
                "actor parameter requires_grad state changed during factorial evaluation"
            )
        checkpoint_after = checkpoint_sha256(
            self.loaded_actor.source_checkpoint_path
        )
        if checkpoint_before != checkpoint_after:
            raise RouteServiceFactorialError(
                "source checkpoint changed during factorial evaluation"
            )

        episode_tuple = tuple(episodes)
        aggregate = self._aggregate(episode_tuple)
        manifest = self._factorial_manifest(
            episode_tuple,
            before_requires_grad,
            checkpoint_before,
        )
        artifacts = (
            self._write_factorial_artifacts(manifest, episode_tuple, aggregate)
            if write_artifacts
            else ()
        )
        return RouteServiceFactorialResult(
            factorial_run_id=self.factorial_run_id,
            manifest=manifest,
            episodes=episode_tuple,
            aggregate_metrics=aggregate,
            artifacts=artifacts,
        )

    def _run_factorial_episode(
        self,
        mode: RouteServiceMode,
        evaluation_seed: int,
        *,
        episode_id: int,
    ) -> RouteServiceFactorialEpisodeResult:
        del episode_id
        episode_config = replace(
            self.config,
            mode="evaluation",
            method_id="ca_gat_mappo",
            launch_profile=None,
            seed=evaluation_seed,
        )
        episode_config.validate()
        environment = self.environment_factory(episode_config)
        reset = environment.reset()
        observations = reset.observations
        external_trace: list[Mapping[str, Any]] = [
            self._external_trace_record(environment, phase="reset", arrival=None)
        ]
        slot_infos: list[Mapping[str, Any]] = []
        action_trace: list[tuple[Mapping[str, Any], ...]] = []
        hidden_trace_sha256: list[str] = []

        heuristic = HeuristicPolicy(
            episode_config,
            policy_seed=evaluation_seed,
        )
        wrapper = FactorialPolicyWrapper(
            mode=mode,
            heuristic_policy=heuristic,
            learned_action_fn=self._actor_actions,
        )
        hidden: torch.Tensor | None = None
        if mode.uses_learned_actor:
            hidden = self.loaded_actor.actor.initial_hidden(
                1,
                device=self.loaded_actor.evaluation_device,
                dtype=torch.float32,
            )
            if torch.count_nonzero(hidden).item() != 0:
                raise RouteServiceFactorialError(
                    "actor initial hidden state is not zero"
                )
            hidden_trace_sha256.append(_tensor_sha256(hidden))

        telemetry = RouteServiceTelemetryCollector()
        while True:
            proposals, hidden, _action_output = wrapper.act(observations, hidden)
            if mode.uses_learned_actor:
                assert hidden is not None
                hidden_trace_sha256.append(_tensor_sha256(hidden))
            action_trace.append(
                tuple(_proposal_record(proposal) for proposal in proposals)
            )
            step = environment.step(proposals)
            telemetry.observe_step(proposals, step.info)
            slot_infos.append(step.info)
            external_trace.append(
                self._external_trace_record(
                    environment,
                    phase=f"slot_{int(step.info['slot'])}_complete",
                    arrival=step.info["arrival"],
                )
            )
            if step.terminated or step.truncated:
                if step.observations is not None or step.centralized_state is not None:
                    raise RouteServiceFactorialError(
                        "terminal factorial step exposed a fake next state"
                    )
                break
            if step.observations is None:
                raise RouteServiceFactorialError(
                    "nonterminal factorial step omitted actor observations"
                )
            observations = step.observations

        if len(slot_infos) != episode_config.environment.episode_horizon:
            raise RouteServiceFactorialError(
                "factorial episode did not execute the configured horizon"
            )
        environment.assert_invariants()
        action_trace_tuple = tuple(action_trace)
        external_trace_tuple = tuple(external_trace)
        evaluation = build_episode_metrics(
            method_id=mode.value,
            evaluation_seed=evaluation_seed,
            environment=environment,
            slot_infos=slot_infos,
            action_trace=action_trace_tuple,
            action_trace_sha256=_canonical_sha256(action_trace_tuple),
            external_trace=external_trace_tuple,
            external_trace_sha256=_canonical_sha256(external_trace_tuple),
            hidden_trace_sha256=tuple(hidden_trace_sha256),
        )
        return RouteServiceFactorialEpisodeResult(
            mode=mode,
            evaluation=evaluation,
            telemetry=telemetry.finalize(environment),
        )

    def _aggregate(
        self,
        episodes: Sequence[RouteServiceFactorialEpisodeResult],
    ) -> dict[str, Any]:
        base = aggregate_episode_metrics(
            tuple(item.evaluation for item in episodes),
            ROUTE_SERVICE_FACTORIAL_METHOD_SUITE,
        )
        modes: dict[str, Any] = {}
        for mode in ROUTE_SERVICE_FACTORIAL_MODES:
            selected = tuple(item for item in episodes if item.mode is mode)
            denominator = sum(
                item.telemetry.remote_binding_task_count for item in selected
            )
            destinations: dict[str, int] = {}
            for item in selected:
                for destination, count in item.telemetry.destination_distribution.items():
                    destinations[destination] = destinations.get(destination, 0) + count

            def pooled_count(name: str) -> int:
                return sum(int(getattr(item.telemetry, name)) for item in selected)

            def pooled_ratio(name: str) -> float | None:
                numerator = pooled_count(name)
                return numerator / denominator if denominator else None

            modes[mode.value] = {
                "route_source": mode.route_source,
                "service_source": mode.service_source,
                "remote_selection_count": pooled_count("remote_selection_count"),
                "local_selection_count": pooled_count("local_selection_count"),
                "destination_distribution": dict(
                    sorted(destinations.items(), key=lambda item: int(item[0]))
                ),
                "remote_binding_task_count": denominator,
                "ever_tx_service_count": pooled_count("ever_tx_service_count"),
                "ever_tx_service_ratio": pooled_ratio("ever_tx_service_count"),
                "tx_complete_count": pooled_count("tx_complete_count"),
                "tx_complete_ratio": pooled_ratio("tx_complete_count"),
                "reach_destination_cpu_count": pooled_count(
                    "reach_destination_cpu_count"
                ),
                "reach_destination_cpu_ratio": pooled_ratio(
                    "reach_destination_cpu_count"
                ),
                "done_on_time_count": pooled_count("done_on_time_count"),
                "done_on_time_ratio": pooled_ratio("done_on_time_count"),
                "route_oracle_accuracy": None,
                "route_oracle_status": "not_computed_for_factorial_composition",
            }
        return {
            "schema_version": ROUTE_SERVICE_FACTORIAL_SCHEMA_VERSION,
            "factorial_run_id": self.factorial_run_id,
            "base_metrics": base,
            "route_service": modes,
        }

    def _factorial_manifest(
        self,
        episodes: Sequence[RouteServiceFactorialEpisodeResult],
        requires_grad: tuple[tuple[str, bool], ...],
        checkpoint_sha256_value: str,
    ) -> dict[str, Any]:
        traces_by_seed: dict[str, Any] = {}
        for seed in self.config.evaluation.evaluation_seeds:
            selected = tuple(
                item for item in episodes if item.evaluation.evaluation_seed == seed
            )
            traces_by_seed[str(seed)] = {
                "shared_external_trace_sha256": (
                    selected[0].evaluation.external_trace_sha256
                ),
                "mode_trace_sha256": {
                    item.mode.value: item.evaluation.external_trace_sha256
                    for item in selected
                },
            }
        return {
            "schema_version": ROUTE_SERVICE_FACTORIAL_SCHEMA_VERSION,
            "factorial_run_id": self.factorial_run_id,
            "formal_evaluation_parent_id": self.evaluation_run_id,
            "evaluation_config_sha256": self.evaluation_config_sha256,
            "mode_suite": list(ROUTE_SERVICE_FACTORIAL_METHOD_SUITE),
            "mode_composition": {
                mode.value: {
                    "route": mode.route_source,
                    "service": mode.service_source,
                }
                for mode in ROUTE_SERVICE_FACTORIAL_MODES
            },
            "source_checkpoint_path": str(
                self.loaded_actor.source_checkpoint_path
            ),
            "source_checkpoint_sha256": checkpoint_sha256_value,
            "source_checkpoint_kind": self.loaded_actor.source_checkpoint_kind,
            "source_method_id": self.loaded_actor.source_method_id,
            "source_training_config_hash": (
                self.loaded_actor.source_training_config_hash
            ),
            "source_checkpoint_metadata": dict(
                self.loaded_actor.source_checkpoint_metadata
            ),
            "scenario_id": self.config.scenario_id,
            "evaluation_seeds": list(self.config.evaluation.evaluation_seeds),
            "episode_horizon": self.config.environment.episode_horizon,
            "evaluation_device": self.loaded_actor.evaluation_device,
            "dtype": self.loaded_actor.dtype,
            "deterministic_action_mode": "deterministic",
            "masked_argmax": True,
            "actor_eval_mode": True,
            "torch_no_grad": True,
            "actor_requires_grad_state": dict(requires_grad),
            "checkpoint_immutable": True,
            "training_invoked": False,
            "route_oracle_accuracy": None,
            "route_oracle_status": "not_computed_for_factorial_composition",
            "service_metric_denominator": (
                "actual successful Remote binding task"
            ),
            "shared_external_trace_by_seed": traces_by_seed,
            "artifact_filenames": list(
                ROUTE_SERVICE_FACTORIAL_ARTIFACT_FILENAMES
            ),
        }

    def _write_factorial_artifacts(
        self,
        manifest: Mapping[str, Any],
        episodes: Sequence[RouteServiceFactorialEpisodeResult],
        aggregate: Mapping[str, Any],
    ) -> tuple[str, ...]:
        paths = self.artifact_paths()
        episode_text = "".join(
            _compact_json(item.artifact_record(self.factorial_run_id)) + "\n"
            for item in episodes
        )
        ordered = tuple(paths.values())
        atomic_write_text_group(
            zip(
                ordered,
                (
                    _pretty_json(manifest),
                    episode_text,
                    _pretty_json(aggregate),
                ),
            ),
            group_name="Route-Service factorial diagnostic artifact group",
        )
        return tuple(str(path) for path in ordered)


__all__ = [
    "ROUTE_SERVICE_FACTORIAL_ARTIFACT_FILENAMES",
    "ROUTE_SERVICE_FACTORIAL_METHOD_SUITE",
    "ROUTE_SERVICE_FACTORIAL_MODES",
    "ROUTE_SERVICE_FACTORIAL_SCHEMA_VERSION",
    "SERVICE_BRANCHES",
    "FactorialPolicyWrapper",
    "RouteServiceEpisodeTelemetry",
    "RouteServiceFactorialEpisodeResult",
    "RouteServiceFactorialError",
    "RouteServiceFactorialResult",
    "RouteServiceFactorialRunner",
    "RouteServiceMode",
    "RouteServiceProposalAdapter",
    "RouteServiceTelemetryCollector",
]
