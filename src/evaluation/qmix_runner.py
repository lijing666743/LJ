"""Frozen QMIX evaluation with the existing truth metrics and external traces."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from ..artifacts import atomic_write_text_group, require_artifact_targets_absent
from ..config import _construct_run_config
from ..env.environment import U2UMECEnvironment
from ..models.ca_gat_mappo import ActorObservationTensorizer
from ..models.factorized_gat_qmix import FactorizedGATQNetwork, METHOD_ID,QMIX_METHOD_IDS,build_qmix_agent
from ..models.qmix_trainer import load_checkpoint
from ..policies.heuristic_policy import HeuristicPolicy
from ..policies.local_only_policy import LocalOnlyPolicy
from ..policies.random_policy import RandomPolicy
from .metrics import build_episode_metrics, aggregate_episode_metrics
from .runner import FormalEvaluationRunner, _canonical_sha256, _tensor_sha256, _proposal_record
from .actor_loader import checkpoint_sha256


SUITE = (METHOD_ID, "local_only", "heuristic", "random")


def method_suite(config):
    return (config.method_id,*SUITE[1:])


def evaluation_fingerprints():
    root = Path(__file__).resolve().parents[2]
    paths = [root / "src/evaluation" / name for name in (
        "qmix_runner.py", "qmix_checkpoint_progress.py", "metrics.py", "runner.py")]
    paths += list((root / "src/policies").glob("*.py"))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


class QMIXEvaluationRunner:
    def __init__(self, config, checkpoint_path, *, device="cpu",
                 checkpoint_purpose="primary_final", expected_training_steps=None):
        if config.mode != "evaluation" or config.method_id not in QMIX_METHOD_IDS:
            raise ValueError("QMIX evaluation requires its own evaluation method identity")
        config.validate()
        data = load_checkpoint(checkpoint_path)
        if checkpoint_purpose not in {"primary_final", "development_checkpoint"}:
            raise ValueError("unknown QMIX checkpoint evaluation purpose")
        if checkpoint_purpose == "primary_final" and data["kind"] != "final":
            raise ValueError("primary evaluation requires the budget-end final checkpoint")
        snapshot = {k: v for k, v in data["config_snapshot"].items() if k != "_metadata"}
        source = _construct_run_config(snapshot)
        source.validate()
        if source.config_hash != data["config_hash"]:
            raise ValueError("checkpoint source configuration hash is inconsistent")
        budget = source.training.qmix.max_training_environment_steps
        steps = data["steps"]
        if type(steps) is not int or not 0 < steps <= budget:
            raise ValueError("checkpoint training step is outside its declared budget")
        if data["kind"] == "final":
            if steps != budget:
                raise ValueError("checkpoint did not finish its declared budget")
        elif data["kind"] == "periodic":
            if (steps >= budget or steps % source.training.qmix.checkpoint_interval_steps
                    or Path(checkpoint_path).name != f"step_{steps:06d}.pt"):
                raise ValueError("periodic checkpoint filename/step/interval is inconsistent")
        else:
            raise ValueError("unknown QMIX checkpoint kind")
        if steps != data["next_episode"] * source.environment.episode_horizon:
            raise ValueError("checkpoint is not at a complete episode boundary")
        if expected_training_steps is not None and (
                type(expected_training_steps) is not int or steps != expected_training_steps):
            raise ValueError("checkpoint step differs from the requested milestone")
        for name in ("scenario_id", "environment", "action", "reward", "reproducibility"):
            if getattr(source, name) != getattr(config, name):
                raise ValueError(f"evaluation differs from training contract: {name}")
        self.config, self.device = config, torch.device(device)
        self.checkpoint_hash = checkpoint_sha256(checkpoint_path)
        # Build from the checkpoint's training architecture, never eval defaults.
        self.agent = build_qmix_agent(source).to(self.device).eval().requires_grad_(False)
        agent_state = {key[len("agent."):]: value for key, value in data["online"].items() if key.startswith("agent.")}
        self.agent.load_state_dict(agent_state, strict=True)
        self.adapter = ActorObservationTensorizer(source)
        self.source_seed = source.seed
        self.training_steps = data["steps"]
        self.training_budget = budget
        self.checkpoint_kind = data["kind"]
        self.checkpoint_purpose = checkpoint_purpose
        self.source_config_hash = data["config_hash"]
        identity = {"checkpoint": self.checkpoint_hash, "config": config.config_hash, "device": str(self.device)}
        if checkpoint_purpose == "development_checkpoint":
            identity["purpose"] = checkpoint_purpose
        self.run_id = "qmix-eval__" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]

    def episode(self, method, seed):
        config = replace(self.config, method_id=method, seed=seed, launch_profile=None)
        env = U2UMECEnvironment(config)
        reset = env.reset()
        observations = reset.observations
        external = [FormalEvaluationRunner._external_trace_record(self, env, phase="reset", arrival=None)]
        actions, infos, hidden_hashes = [], [], []
        hidden = self.agent.initial_hidden()
        if method in QMIX_METHOD_IDS:
            policy = None
            hidden_hashes.append(_tensor_sha256(hidden))
        elif method == "local_only":
            policy = LocalOnlyPolicy(config, policy_seed=seed)
        elif method == "heuristic":
            policy = HeuristicPolicy(config, policy_seed=seed)
        elif method == "random":
            policy = RandomPolicy(seed)
        else:
            raise ValueError("unknown evaluation policy")
        while True:
            if policy is None:
                with torch.no_grad():
                    batch = self.adapter.encode_step(observations, device=self.device, episode_start=not infos)
                    contracts = ((tuple(obs.action_masks for obs in observations),),)
                    output = self.agent.actions(batch, hidden, contracts)
                proposals, hidden = output.proposals[0][0], output.hidden_out
                hidden_hashes.append(_tensor_sha256(hidden))
            else:
                proposals = tuple(policy.act(obs) for obs in observations)
            actions.append(tuple(_proposal_record(p) for p in proposals))
            step = env.step(proposals)
            infos.append(step.info)
            external.append(FormalEvaluationRunner._external_trace_record(self, env,
                phase=f"slot_{int(step.info['slot'])}_complete", arrival=step.info["arrival"]))
            if step.terminated or step.truncated:
                if step.observations is not None or step.centralized_state is not None:
                    raise ValueError("terminal evaluation exposed a fake next state")
                break
            observations = step.observations
        if len(infos) != config.environment.episode_horizon:
            raise ValueError("evaluation did not execute the whole horizon")
        env.assert_invariants()
        return build_episode_metrics(method_id=method, evaluation_seed=seed, environment=env,
            slot_infos=infos, action_trace=tuple(actions), action_trace_sha256=_canonical_sha256(actions),
            external_trace=tuple(external), external_trace_sha256=_canonical_sha256(external),
            hidden_trace_sha256=tuple(hidden_hashes))

    def artifact_paths(self):
        root = Path(self.config.output.logs_dir) / self.run_id
        return {key: root / filename for key, filename in (
            ("manifest", "evaluation_manifest.json"), ("episodes", "episode_metrics.jsonl"),
            ("aggregate", "aggregate_metrics.json"))}

    def run(self, *, write_artifacts=True, progress=None):
        suite=method_suite(self.config)
        paths = self.artifact_paths()
        if write_artifacts:
            require_artifact_targets_absent(paths.values(), group_name="QMIX frozen evaluation")
        before = {k: v.clone() for k, v in self.agent.state_dict().items()}
        episodes = []
        for seed in self.config.evaluation.evaluation_seeds:
            group = []
            for method in suite:
                started = time.perf_counter()
                episode = self.episode(method, int(seed))
                group.append(episode)
                if progress is not None:
                    progress({"training_steps": self.training_steps, "method_id": method,
                              "evaluation_seed": int(seed), "generated": episode.metrics["generated"],
                              "completed": episode.metrics["completed"], "seconds": time.perf_counter()-started})
            if len({item.external_trace_sha256 for item in group}) != 1:
                raise ValueError("external environment realizations diverged across methods")
            episodes.extend(group)
        if self.agent.training or any(not torch.equal(before[k], v) for k, v in self.agent.state_dict().items()):
            raise ValueError("frozen QMIX network changed during evaluation")
        aggregate = aggregate_episode_metrics(episodes, suite)
        manifest = {"schema": "qmix_frozen_evaluation.v1", "evaluation_run_id": self.run_id,
            "method_suite": suite, "training_seed": self.source_seed, "training_steps": self.training_steps,
            "evaluation_seeds": self.config.evaluation.evaluation_seeds,
            "checkpoint_sha256": self.checkpoint_hash, "config_hash": self.config.config_hash,
            "device": str(self.device), "dtype": "float32", "network_updates": 0,
            "checkpoint_kind": self.checkpoint_kind, "evaluation_purpose": self.checkpoint_purpose,
            "source_training_budget_steps": self.training_budget,
            "source_training_config_hash": self.source_config_hash,
            "evaluation_source_sha256": evaluation_fingerprints(),
            "exploration": False, "actor_inputs": "existing ActorObservation only",
            "source_manifest": "Checkpoint config and independent QMIX architecture",
            "shared_external_trace_by_seed": {str(seed): {item.method_id: item.external_trace_sha256
                for item in episodes if item.evaluation_seed == seed} for seed in self.config.evaluation.evaluation_seeds}}
        artifacts = ()
        if write_artifacts:
            artifacts = atomic_write_text_group([
                (paths["manifest"], json.dumps(manifest, ensure_ascii=False, indent=2)),
                (paths["aggregate"], json.dumps(aggregate, ensure_ascii=False, indent=2)),
                (paths["episodes"], "\n".join(json.dumps(item.artifact_record(self.run_id), ensure_ascii=False) for item in episodes)+"\n"),
            ], group_name="QMIX frozen evaluation")
        return {"manifest": manifest, "episodes": tuple(episodes), "aggregate": aggregate,
                "artifacts": tuple(str(p) for p in artifacts)}
