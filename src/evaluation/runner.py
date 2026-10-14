"""Independent deterministic four-method Formal Evaluation Runner."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from ..artifacts import (
    atomic_write_text_group,
    prepare_artifact_parents,
    require_artifact_targets_absent,
)
from ..config import RunConfig
from ..env.actions import ActionProposal
from ..env.environment import U2UMECEnvironment
from ..models.ca_gat_mappo import ActorObservationTensorizer
from ..models.ca_gat_mappo_actions import (
    CAGATMAPPOActionDistribution,
    SequentialActionDistributionOutput,
    SequentialActionMaskBatch,
)
from ..policies.heuristic_policy import HeuristicPolicy
from ..policies.local_only_policy import LocalOnlyPolicy
from ..policies.random_policy import RandomPolicy
from .actor_loader import (
    LoadedEvaluationActor, load_final_actor_for_evaluation,
    load_actor_for_development_diagnostic,
)
from .metrics import (
    FORMAL_EVALUATION_SCHEMA_VERSION,
    FORMAL_METRIC_FIELDS,
    EvaluationEpisodeResult,
    aggregate_episode_metrics,
    build_episode_metrics,
)
from .route_telemetry import (
    EvaluationRouteTelemetryCollector,
    build_evaluation_route_summary,
)
from .service_telemetry import (
    SERVICE_TELEMETRY_SCHEMA,
    EvaluationServiceTelemetryCollector,
    snapshot_tasks,
)


FORMAL_METHOD_SUITE = (
    "ca_gat_mappo",
    "local_only",
    "heuristic",
    "random",
)
EVALUATION_ARTIFACT_FILENAMES = (
    "evaluation_manifest.json",
    "episode_metrics.jsonl",
    "aggregate_metrics.json",
    "evaluation_metrics.csv",
)
EVALUATION_ROUTE_DIAGNOSTIC_FILENAMES = (
    "summary.json",
    "route_decision_trace.jsonl",
)
_CONSUMED_ENVIRONMENT_STREAMS = (
    "reset_mobility",
    "task_arrival",
    "task_workload",
    "channel_fading",
    "csi_error",
)


class FormalEvaluationError(RuntimeError):
    """Raised when the Formal Evaluation Gate cannot complete honestly."""


@dataclass(frozen=True)
class FormalEvaluationResult:
    """Complete in-memory evaluation plus optional immutable artifacts."""

    evaluation_run_id: str
    manifest: Mapping[str, Any]
    episodes: tuple[EvaluationEpisodeResult, ...]
    aggregate_metrics: Mapping[str, Any]
    artifacts: tuple[str, ...]
    route_decision_trace: tuple[Mapping[str, Any], ...] = ()
    route_summary: Mapping[str, Any] | None = None
    diagnostic_artifacts: tuple[str, ...] = ()
    task_lifecycle_trace: tuple[Mapping[str, Any], ...] = ()
    cpu_service_trace: tuple[Mapping[str, Any], ...] = ()
    tx_service_trace: tuple[Mapping[str, Any], ...] = ()
    service_diagnostic_artifacts: tuple[str, ...] = ()


class FormalEvaluationRunner:
    """Evaluate a frozen actor and three baselines on aligned seed realizations."""

    def __init__(
        self,
        config: RunConfig,
        checkpoint_path: str | Path,
        *,
        evaluation_device: str,
        environment_factory: Callable[[RunConfig], U2UMECEnvironment] | None = None,
        collect_route_telemetry: bool = True,
        collect_service_telemetry: bool = False,
        development_checkpoint: bool = False,
    ) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        config.validate()
        self._validate_config(config)
        self.config = config
        self.environment_factory = environment_factory or U2UMECEnvironment
        if not isinstance(collect_route_telemetry, bool):
            raise TypeError("collect_route_telemetry must be bool")
        self.collect_route_telemetry = collect_route_telemetry
        if not isinstance(collect_service_telemetry, bool):
            raise TypeError("collect_service_telemetry must be bool")
        self.collect_service_telemetry = collect_service_telemetry
        if not isinstance(development_checkpoint, bool):
            raise TypeError("development_checkpoint must be bool")
        if development_checkpoint and not collect_service_telemetry:
            raise FormalEvaluationError("development checkpoints require service diagnostics")
        self.development_checkpoint = development_checkpoint
        loader = (load_actor_for_development_diagnostic if development_checkpoint
                  else load_final_actor_for_evaluation)
        self.loaded_actor = loader(
            config,
            checkpoint_path,
            evaluation_device=evaluation_device,
        )
        self.actor_tensorizer = ActorObservationTensorizer(config)
        self.action_distribution = CAGATMAPPOActionDistribution(
            self.loaded_actor.actor,
            config,
        )
        self.evaluation_config_identity = self._evaluation_config_identity()
        self.evaluation_config_sha256 = _canonical_sha256(
            self.evaluation_config_identity
        )
        self.evaluation_run_id = self._evaluation_run_id()
        self.route_telemetry = EvaluationRouteTelemetryCollector(config)

    def run(self, *, write_artifacts: bool = True) -> FormalEvaluationResult:
        """Run all configured seeds; a no-write path supports repeatability tests."""

        artifact_paths = self.artifact_paths()
        if write_artifacts:
            all_targets = list(artifact_paths.values())
            require_artifact_targets_absent(
                artifact_paths.values(),
                group_name="formal evaluation artifact group",
            )
            if self.collect_route_telemetry:
                all_targets.extend(self.diagnostic_artifact_paths().values())
                require_artifact_targets_absent(
                    self.diagnostic_artifact_paths().values(),
                    group_name="evaluation route diagnostic artifact group",
                )
            if self.collect_service_telemetry:
                all_targets.extend(self.service_diagnostic_artifact_paths().values())
                require_artifact_targets_absent(
                    self.service_diagnostic_artifact_paths().values(),
                    group_name="evaluation service diagnostic artifact group",
                )
            prepare_artifact_parents(
                all_targets,
                group_name="evaluation artifact parents",
                check_writable=True,
            )

        actor = self.loaded_actor.actor
        actor.eval()
        before_state = _state_dict_snapshot(actor)
        before_requires_grad = tuple(
            (name, parameter.requires_grad)
            for name, parameter in actor.named_parameters()
        )

        episodes: list[EvaluationEpisodeResult] = []
        route_decision_trace: list[Mapping[str, Any]] = []
        service_collectors: list[EvaluationServiceTelemetryCollector] | None = (
            [] if self.collect_service_telemetry else None
        )
        for episode_id, evaluation_seed in enumerate(
            self.config.evaluation.evaluation_seeds
        ):
            seed_episodes: list[EvaluationEpisodeResult] = []
            for method_id in FORMAL_METHOD_SUITE:
                seed_episodes.append(
                    self._run_episode(
                        method_id,
                        int(evaluation_seed),
                        episode_id=episode_id,
                        route_decision_trace=route_decision_trace,
                        service_collectors=service_collectors,
                    )
                )
            trace_hashes = {
                item.method_id: item.external_trace_sha256
                for item in seed_episodes
            }
            if len(set(trace_hashes.values())) != 1:
                raise FormalEvaluationError(
                    "external environment realization diverged across methods for "
                    f"evaluation seed {evaluation_seed}: {trace_hashes}"
                )
            episodes.extend(seed_episodes)

        if actor.training:
            raise FormalEvaluationError("frozen actor left evaluation mode")
        after_state = _state_dict_snapshot(actor)
        if tuple(before_state) != tuple(after_state) or any(
            not torch.equal(before_state[name], after_state[name])
            for name in before_state
        ):
            raise FormalEvaluationError("frozen actor state changed during evaluation")
        after_requires_grad = tuple(
            (name, parameter.requires_grad)
            for name, parameter in actor.named_parameters()
        )
        if before_requires_grad != after_requires_grad:
            raise FormalEvaluationError(
                "actor parameter requires_grad state changed during evaluation"
            )

        aggregate = aggregate_episode_metrics(episodes, FORMAL_METHOD_SUITE)
        aggregate["evaluation_run_id"] = self.evaluation_run_id
        manifest = self._manifest(tuple(episodes), before_requires_grad)
        actor_episodes = tuple(
            item for item in episodes if item.method_id == "ca_gat_mappo"
        )
        route_summary = (
            build_evaluation_route_summary(
                evaluation_run_id=self.evaluation_run_id,
                decoder_mode=self.config.training.mappo.route_decoder_mode.value,
                records=route_decision_trace,
                actor_episodes=actor_episodes,
            )
            if self.collect_route_telemetry
            else None
        )
        artifacts = (
            self._write_artifacts(manifest, tuple(episodes), aggregate)
            if write_artifacts
            else ()
        )
        diagnostic_artifacts = (
            self._write_diagnostic_artifacts(
                tuple(route_decision_trace), route_summary
            )
            if write_artifacts and self.collect_route_telemetry
            else ()
        )
        task_trace = tuple(record for collector in service_collectors or ()
                           for record in collector.task_records())
        cpu_trace = tuple(record for collector in service_collectors or ()
                          for record in collector.cpu_slots)
        tx_trace = tuple(record for collector in service_collectors or ()
                         for record in collector.tx_slots)
        service_artifacts = (
            self._write_service_diagnostic_artifacts(task_trace, cpu_trace, tx_trace)
            if write_artifacts and self.collect_service_telemetry else ()
        )
        return FormalEvaluationResult(
            evaluation_run_id=self.evaluation_run_id,
            manifest=manifest,
            episodes=tuple(episodes),
            aggregate_metrics=aggregate,
            artifacts=artifacts,
            route_decision_trace=tuple(route_decision_trace),
            route_summary=route_summary,
            diagnostic_artifacts=diagnostic_artifacts,
            task_lifecycle_trace=task_trace,
            cpu_service_trace=cpu_trace,
            tx_service_trace=tx_trace,
            service_diagnostic_artifacts=service_artifacts,
        )

    def artifact_paths(self) -> Mapping[str, Path]:
        """Return the isolated identity directory without creating it."""

        root = (
            Path(self.config.output.logs_dir)
            / "evaluations"
            / self.evaluation_run_id
        )
        return {
            name: root / filename
            for name, filename in zip(
                (
                    "manifest",
                    "episode_metrics",
                    "aggregate_metrics",
                    "evaluation_metrics_csv",
                ),
                EVALUATION_ARTIFACT_FILENAMES,
            )
        }

    def diagnostic_artifact_paths(self) -> Mapping[str, Path]:
        """Return isolated additive diagnostics paths without creating them."""

        root = (
            Path(self.config.output.logs_dir)
            / "diagnostics"
            / f"hierarchical-route-eval__{self.evaluation_run_id}"
        )
        return {
            "summary": root / EVALUATION_ROUTE_DIAGNOSTIC_FILENAMES[0],
            "route_decision_trace": (
                root / EVALUATION_ROUTE_DIAGNOSTIC_FILENAMES[1]
            ),
        }

    def service_diagnostic_artifact_paths(self) -> Mapping[str, Path]:
        root = (Path(self.config.output.logs_dir) / "diagnostics"
                / f"task-service-eval__{self.evaluation_run_id}")
        return {name: root / filename for name, filename in (
            ("manifest", "service_diagnostic_manifest.json"),
            ("tasks", "task_lifecycle.jsonl"),
            ("cpu", "cpu_service_trace.jsonl"),
            ("tx", "tx_service_trace.jsonl"),
        )}

    def _run_episode(
        self,
        method_id: str,
        evaluation_seed: int,
        *,
        episode_id: int,
        route_decision_trace: list[Mapping[str, Any]],
        service_collectors: list[EvaluationServiceTelemetryCollector] | None = None,
    ) -> EvaluationEpisodeResult:
        episode_config = replace(
            self.config,
            mode="evaluation",
            method_id=method_id,
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
        service_collector = (
            EvaluationServiceTelemetryCollector(
                evaluation_run_id=self.evaluation_run_id, method_id=method_id,
                evaluation_seed=evaluation_seed, episode_id=episode_id,
            ) if service_collectors is not None else None
        )

        policy: RandomPolicy | HeuristicPolicy | LocalOnlyPolicy | None = None
        hidden: torch.Tensor | None = None
        if method_id == "ca_gat_mappo":
            actor = self.loaded_actor.actor
            actor.eval()
            hidden = actor.initial_hidden(
                1,
                device=self.loaded_actor.evaluation_device,
                dtype=torch.float32,
            )
            if torch.count_nonzero(hidden).item() != 0:
                raise FormalEvaluationError("actor initial hidden state is not zero")
            hidden_trace_sha256.append(_tensor_sha256(hidden))
        elif method_id == "local_only":
            policy = LocalOnlyPolicy(episode_config, policy_seed=evaluation_seed)
        elif method_id == "heuristic":
            policy = HeuristicPolicy(episode_config, policy_seed=evaluation_seed)
        elif method_id == "random":
            policy = RandomPolicy(evaluation_seed)
        else:
            raise FormalEvaluationError(f"unknown formal method {method_id!r}")

        while True:
            action_output: SequentialActionDistributionOutput | None = None
            if method_id == "ca_gat_mappo":
                assert hidden is not None
                proposals, hidden, action_output = self._actor_actions(
                    observations, hidden
                )
                hidden_trace_sha256.append(_tensor_sha256(hidden))
            else:
                assert policy is not None
                proposals = tuple(policy.act(item) for item in observations)
            action_trace.append(tuple(_proposal_record(item) for item in proposals))
            before_tasks = (
                snapshot_tasks(environment.lifecycle.tasks.values())
                if service_collector is not None else None
            )
            step = environment.step(proposals)
            if service_collector is not None:
                assert before_tasks is not None
                service_collector.observe_step(
                    before=before_tasks,
                    after=snapshot_tasks(environment.lifecycle.tasks.values()),
                    step_info=step.info,
                )
            if self.collect_route_telemetry and action_output is not None:
                route_decision_trace.extend(
                    self.route_telemetry.collect_step(
                        evaluation_run_id=self.evaluation_run_id,
                        episode_id=episode_id,
                        evaluation_seed=evaluation_seed,
                        observations=observations,
                        proposals=proposals,
                        action_output=action_output,
                        step_info=step.info,
                    )
                )
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
                    raise FormalEvaluationError(
                        "terminal evaluation step exposed a fake next state"
                    )
                break
            if step.observations is None:
                raise FormalEvaluationError(
                    "nonterminal evaluation step omitted actor observations"
                )
            observations = step.observations

        if len(slot_infos) != episode_config.environment.episode_horizon:
            raise FormalEvaluationError(
                "evaluation episode did not execute the configured horizon"
            )
        environment.assert_invariants()
        if service_collector is not None:
            assert service_collectors is not None
            service_collectors.append(service_collector)
        action_trace_tuple = tuple(action_trace)
        external_trace_tuple = tuple(external_trace)
        return build_episode_metrics(
            method_id=method_id,
            evaluation_seed=evaluation_seed,
            environment=environment,
            slot_infos=slot_infos,
            action_trace=action_trace_tuple,
            action_trace_sha256=_canonical_sha256(action_trace_tuple),
            external_trace=external_trace_tuple,
            external_trace_sha256=_canonical_sha256(external_trace_tuple),
            hidden_trace_sha256=tuple(hidden_trace_sha256),
        )

    def _actor_actions(
        self,
        observations: Sequence[Any],
        hidden: torch.Tensor,
    ) -> tuple[
        tuple[ActionProposal, ...],
        torch.Tensor,
        SequentialActionDistributionOutput,
    ]:
        actor = self.loaded_actor.actor
        if actor.training:
            raise FormalEvaluationError("actor must remain in eval mode")
        with torch.no_grad():
            batch = self.actor_tensorizer.encode_step(
                observations,
                device=self.loaded_actor.evaluation_device,
                dtype=torch.float32,
                episode_start=int(observations[0].slot) == 0,
            )
            masks = SequentialActionMaskBatch.from_observations(observations)
            output = self.action_distribution.deterministic_actions(
                batch,
                masks,
                hidden,
            )
        if output.mode != "deterministic":
            raise FormalEvaluationError("actor did not use deterministic action mode")
        proposals = tuple(output.proposals[0][0])
        return proposals, output.hidden_out.detach(), output

    def _external_trace_record(
        self,
        environment: U2UMECEnvironment,
        *,
        phase: str,
        arrival: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        if (
            environment.mobility_model is None
            or environment.traffic is None
            or environment.channel_model is None
            or environment.channel_history is None
            or environment.mobility is None
            or environment.channel is None
        ):
            raise FormalEvaluationError("environment trace components are unavailable")
        rngs = {
            "reset_mobility": environment.mobility_model.rng,
            "task_arrival": environment.traffic.arrival_rng,
            "task_workload": environment.traffic.workload_rng,
            "channel_fading": environment.channel_model.rng,
            "csi_error": environment.channel_history.csi_error_rng,
        }
        record: dict[str, Any] = {
            "phase": phase,
            "environment_seed": environment.config.seed,
            "rng_state_sha256": {
                name: _canonical_sha256(generator.bit_generator.state)
                for name, generator in rngs.items()
            },
            "mobility": {
                "slot": environment.mobility.slot,
                "positions_sha256": _array_sha256(
                    environment.mobility.positions_m
                ),
                "velocities_sha256": _array_sha256(
                    environment.mobility.velocities_mps
                ),
            },
            "channel": {
                "slot": environment.channel.slot,
                "true_channel_sha256": _array_sha256(environment.channel.channel),
                "shadowing_sha256": _array_sha256(
                    environment.channel.shadowing_db
                ),
            },
            "csi": (
                {
                    "slot": environment.channel_features.slot,
                    "stale_csi_sha256": _array_sha256(
                        environment.channel_features.stale_csi
                    ),
                    "valid_mask_sha256": _array_sha256(
                        environment.channel_features.csi_valid_mask
                    ),
                }
                if environment.channel_features is not None
                else {"terminal_state": "not_exposed"}
            ),
            "arrival": arrival,
            "interference_measurement_rng": {
                "stream_id": int(
                    environment.config.reproducibility.stream_ids[
                        "interference_measurement"
                    ]
                ),
                "status": "reserved_not_consumed_by_current_backend",
            },
        }
        return record

    def _evaluation_config_identity(self) -> Mapping[str, Any]:
        return {
            "scenario_id": self.config.scenario_id,
            "environment": _jsonable(asdict(self.config.environment)),
            "action": _jsonable(asdict(self.config.action)),
            "evaluation": _jsonable(asdict(self.config.evaluation)),
            "evaluation_seeds": list(self.config.evaluation.evaluation_seeds),
            "episode_horizon": self.config.environment.episode_horizon,
            "method_suite": list(FORMAL_METHOD_SUITE),
            "evaluation_device": self.loaded_actor.evaluation_device,
            "dtype": self.loaded_actor.dtype,
        }

    def _evaluation_run_id(self) -> str:
        identity = {
            "source_checkpoint_sha256": (
                self.loaded_actor.source_checkpoint_sha256
            ),
            "evaluation_config_sha256": self.evaluation_config_sha256,
            "evaluation_seeds": list(self.config.evaluation.evaluation_seeds),
            "method_suite": list(FORMAL_METHOD_SUITE),
            "evaluator_git_commit": self.config.git_commit,
            "evaluator_git_dirty": self.config.git_dirty,
            "evaluation_device": self.loaded_actor.evaluation_device,
            "dtype": self.loaded_actor.dtype,
        }
        digest = _canonical_sha256(identity)
        return (
            f"{'development-service-eval' if self.development_checkpoint else 'formal-eval'}__{self.config.scenario_id}__"
            f"chk-{self.loaded_actor.source_checkpoint_sha256[:12]}__"
            f"cfg-{self.evaluation_config_sha256[:12]}__"
            f"eval-{digest[:12]}"
        )

    def _manifest(
        self,
        episodes: tuple[EvaluationEpisodeResult, ...],
        requires_grad: tuple[tuple[str, bool], ...],
    ) -> Mapping[str, Any]:
        traces_by_seed: dict[str, Any] = {}
        for seed in self.config.evaluation.evaluation_seeds:
            selected = tuple(
                item for item in episodes if item.evaluation_seed == seed
            )
            traces_by_seed[str(seed)] = {
                "shared_external_trace_sha256": selected[0].external_trace_sha256,
                "method_trace_sha256": {
                    item.method_id: item.external_trace_sha256
                    for item in selected
                },
            }
        stream_ids = self.config.reproducibility.stream_ids
        return {
            "evaluation_schema_version": FORMAL_EVALUATION_SCHEMA_VERSION,
            **({"evaluation_purpose": "development_service_diagnostic",
                "paper_formal_result": False} if self.development_checkpoint else {}),
            "evaluation_run_id": self.evaluation_run_id,
            "evaluation_config_sha256": self.evaluation_config_sha256,
            "method_suite": list(FORMAL_METHOD_SUITE),
            "source_checkpoint_path": str(
                self.loaded_actor.source_checkpoint_path
            ),
            "source_checkpoint_sha256": (
                self.loaded_actor.source_checkpoint_sha256
            ),
            "source_checkpoint_kind": self.loaded_actor.source_checkpoint_kind,
            "source_method_id": self.loaded_actor.source_method_id,
            "source_training_config_hash": (
                self.loaded_actor.source_training_config_hash
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
            "source_checkpoint_metadata": dict(
                self.loaded_actor.source_checkpoint_metadata
            ),
            "source_training_actor_ratio_mode": (
                self.loaded_actor.source_training_actor_ratio_mode
            ),
            "source_training_agent_credit_mode": (
                self.loaded_actor.source_training_agent_credit_mode
            ),
            "source_training_git_commit": (
                self.loaded_actor.source_training_git_commit
            ),
            "evaluator_git_commit": self.config.git_commit,
            "evaluator_git_branch": self.config.git_branch,
            "evaluator_git_dirty": self.config.git_dirty,
            "scenario_id": self.config.scenario_id,
            "evaluation_seeds": list(self.config.evaluation.evaluation_seeds),
            "episode_horizon": self.config.environment.episode_horizon,
            "actor_architecture_identity": dict(
                self.loaded_actor.actor_architecture_identity
            ),
            "action_domain_identity": dict(
                self.loaded_actor.action_domain_identity
            ),
            "training_device": self.loaded_actor.source_training_device,
            "evaluation_device": self.loaded_actor.evaluation_device,
            "dtype": self.loaded_actor.dtype,
            "checkpoint_map_location_contract": (
                "structured payload validated on CPU; actor state only copied to "
                "the explicit evaluation device"
            ),
            "deterministic_action_mode": "deterministic",
            "masked_argmax": True,
            "actor_eval_mode": True,
            "torch_no_grad": True,
            "actor_requires_grad_state": {
                name: value for name, value in requires_grad
            },
            "cpu_bitwise_repeatability_required": (
                self.loaded_actor.evaluation_device == "cpu"
            ),
            "cuda_bitwise_repeatability_claimed": False,
            "environment_rng_stream_contract": {
                name: {
                    "stream_id": int(stream_ids[name]),
                    "status": "consumed_and_traced",
                }
                for name in _CONSUMED_ENVIRONMENT_STREAMS
            }
            | {
                "interference_measurement": {
                    "stream_id": int(stream_ids["interference_measurement"]),
                    "status": "reserved_not_consumed_by_current_backend",
                    "action_dependent_measurement_not_forced_equal": True,
                }
            },
            "shared_external_trace_by_seed": traces_by_seed,
            "artifact_filenames": list(EVALUATION_ARTIFACT_FILENAMES),
        }

    def _write_artifacts(
        self,
        manifest: Mapping[str, Any],
        episodes: tuple[EvaluationEpisodeResult, ...],
        aggregate: Mapping[str, Any],
    ) -> tuple[str, ...]:
        paths = self.artifact_paths()
        manifest_text = _pretty_json(manifest)
        episode_text = "".join(
            _compact_json(item.artifact_record(self.evaluation_run_id)) + "\n"
            for item in episodes
        )
        aggregate_text = _pretty_json(aggregate)
        csv_text = self._evaluation_csv(episodes)
        ordered_paths = tuple(paths.values())
        atomic_write_text_group(
            zip(
                ordered_paths,
                (manifest_text, episode_text, aggregate_text, csv_text),
            ),
            group_name="formal evaluation artifact group",
        )
        return tuple(str(path) for path in ordered_paths)

    def _write_diagnostic_artifacts(
        self,
        records: tuple[Mapping[str, Any], ...],
        summary: Mapping[str, Any] | None,
    ) -> tuple[str, ...]:
        if summary is None:
            raise FormalEvaluationError("evaluation route summary is missing")
        paths = self.diagnostic_artifact_paths()
        trace_text = "".join(_compact_json(record) + "\n" for record in records)
        ordered_paths = tuple(paths.values())
        atomic_write_text_group(
            zip(ordered_paths, (_pretty_json(summary), trace_text)),
            group_name="evaluation route diagnostic artifact group",
        )
        return tuple(str(path) for path in ordered_paths)

    def _write_service_diagnostic_artifacts(
        self, tasks: tuple[Mapping[str, Any], ...],
        cpu: tuple[Mapping[str, Any], ...], tx: tuple[Mapping[str, Any], ...],
    ) -> tuple[str, ...]:
        paths = self.service_diagnostic_artifact_paths()
        manifest = {
            "schema_version": SERVICE_TELEMETRY_SCHEMA,
            **({"evaluation_purpose": "development_service_diagnostic",
                "paper_formal_result": False} if self.development_checkpoint else {}),
            "evaluation_run_id": self.evaluation_run_id,
            "source_checkpoint_sha256": self.loaded_actor.source_checkpoint_sha256,
            "source_training_config_hash": self.loaded_actor.source_training_config_hash,
            "evaluation_seeds": list(self.config.evaluation.evaluation_seeds),
            "method_suite": list(FORMAL_METHOD_SUITE),
            "queue_timing": "slot_start; CPU includes local and remote queues",
            "remaining_bits_timing": "post_service_before_or_after_settlement (unchanged by settlement)",
            "frequency_unit": "Hz",
            "tx_row_definition": "one row per selected head or serviced task; sum transmitted_bits; count links/energy only where link_summary=true",
            "stage_definition": "WAIT: never positive service in stage; SERVICE: positive service at least once, including expiry slot",
            "local_expiry": "CPU_WAIT/CPU_SERVICE; destination_uav equals source_uav",
            "task_count": len(tasks), "cpu_record_count": len(cpu),
            "tx_record_count": len(tx),
            "artifact_filenames": [path.name for path in paths.values()],
        }
        contents = (_pretty_json(manifest), *(
            "".join(_compact_json(record) + "\n" for record in records)
            for records in (tasks, cpu, tx)
        ))
        atomic_write_text_group(
            zip(paths.values(), contents),
            group_name="evaluation service diagnostic artifact group",
        )
        return tuple(str(path) for path in paths.values())

    def _evaluation_csv(
        self,
        episodes: Sequence[EvaluationEpisodeResult],
    ) -> str:
        fields = (
            "evaluation_run_id",
            "method_id",
            "evaluation_seed",
            "episode_horizon",
            *FORMAL_METRIC_FIELDS,
            "completed_task_e2e_delay_valid_sample_count",
            "outage_rate_valid_sample_count",
            "action_trace_sha256",
            "external_trace_sha256",
        )
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in episodes:
            row: dict[str, Any] = {
                "evaluation_run_id": self.evaluation_run_id,
                "method_id": item.method_id,
                "evaluation_seed": item.evaluation_seed,
                "episode_horizon": item.episode_horizon,
                **dict(item.metrics),
                "completed_task_e2e_delay_valid_sample_count": len(
                    item.completed_delay_samples_s
                ),
                "outage_rate_valid_sample_count": int(
                    item.metrics["outage_denominator"]
                ),
                "action_trace_sha256": item.action_trace_sha256,
                "external_trace_sha256": item.external_trace_sha256,
            }
            writer.writerow(
                {key: "NA" if value is None else value for key, value in row.items()}
            )
        return stream.getvalue()

    @staticmethod
    def _validate_config(config: RunConfig) -> None:
        if config.mode != "evaluation" or config.method_id != "ca_gat_mappo":
            raise FormalEvaluationError(
                "Formal Evaluation Runner requires evaluation/ca_gat_mappo"
            )
        evaluation = config.evaluation
        seeds = tuple(evaluation.evaluation_seeds)
        if not seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in seeds
        ):
            raise FormalEvaluationError(
                "evaluation seeds must be a non-empty tuple of non-negative integers"
            )
        if len(set(seeds)) != len(seeds):
            raise FormalEvaluationError("evaluation seeds must be unique")
        if set(seeds).intersection(evaluation.train_seeds):
            raise FormalEvaluationError("train and evaluation seeds must be disjoint")
        if evaluation.inference_rule_mappo != "masked_argmax":
            raise FormalEvaluationError(
                "MAPPO evaluation inference rule must remain masked_argmax"
            )
        if evaluation.update_network:
            raise FormalEvaluationError("evaluation network updates are forbidden")
        required_statistics = {"mean", "std", "valid_sample_count"}
        if not required_statistics.issubset(evaluation.metric_statistics):
            raise FormalEvaluationError(
                "evaluation statistics must include mean, std, and valid_sample_count"
            )


def _proposal_record(proposal: ActionProposal) -> Mapping[str, Any]:
    return {
        "uav_id": proposal.uav_id,
        "route": proposal.route,
        "tx_select": proposal.tx_select,
        "resource_group": proposal.resource_group,
        "resource_width": proposal.resource_width,
        "power_level": proposal.power_level,
        "cpu_queue": proposal.cpu_queue,
        "cpu_frequency": proposal.cpu_frequency,
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _state_dict_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_compact_json(value).encode("utf-8")).hexdigest()


def _compact_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _pretty_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "EVALUATION_ARTIFACT_FILENAMES",
    "EVALUATION_ROUTE_DIAGNOSTIC_FILENAMES",
    "FORMAL_METHOD_SUITE",
    "FormalEvaluationError",
    "FormalEvaluationResult",
    "FormalEvaluationRunner",
]
