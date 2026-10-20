"""Real environment collection, recurrent QMIX updates, and boundary checkpoints.

One optimizer update follows each completed real replay chunk after warmup.
Checkpoints are at complete episode boundaries, including the short last
chunk. No environment or pending chunk is silently dropped on resume.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from ..artifacts import atomic_write_bytes_group
from ..config import RunConfig, require_formal_rl_enabled
from ..env.environment import U2UMECEnvironment
from ..env.randomness import derive_training_episode_seed, make_rng
from .ca_gat_mappo import ActorObservationTensorizer, CentralizedStateTensorizer, stack_actor_time
from .factorized_gat_qmix import (
    ACTOR_EDGE_TRANSFORM, CPU_UTILITY_CONTRACT, EXPLORATION_CONTRACT, METHOD_ID, validate_qmix_config,
)
from .qmix_replay import QMIXLearner, ReplaySequence, SequenceReplay
from .qmix_targets import LEARNING_TARGET_CONTRACT, TD_LAMBDA
from .qmix_recurrent import RECURRENT_CONTEXT_CONTRACT


CHECKPOINT_SCHEMA = "factorized_gat_qmix.checkpoint.v8"


def implementation_fingerprints():
    root = Path(__file__).resolve().parents[2]
    sources = [root / "src/config.py", *(root / "src/env").glob("*.py")]
    sources += [root / "src/models" / name for name in (
        "factorized_gat_qmix.py", "qmix_replay.py", "qmix_targets.py", "qmix_recurrent.py", "qmix_trainer.py", "ca_gat_mappo.py")]
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources)}


def episode_config(config, episode):
    result = copy.copy(config)
    object.__setattr__(result, "seed", derive_training_episode_seed(config.seed, episode))
    return result


def epsilon_at(config, steps):
    q = config.training.qmix
    fraction = min(1.0, steps / q.epsilon_decay_steps)
    return q.epsilon_start + fraction * (q.epsilon_end - q.epsilon_start)


def checkpoint_bytes(payload):
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def load_checkpoint(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("schema") != CHECKPOINT_SCHEMA or data.get("method_id") != METHOD_ID:
        raise ValueError("checkpoint is not a Factorized-Action GAT-QMIX checkpoint")
    if data.get("implementation_source_sha256") != implementation_fingerprints():
        raise ValueError("QMIX checkpoint implementation source differs from this workspace")
    if data.get("implementation_contract", {}).get("actor_edge_transform") != ACTOR_EDGE_TRANSFORM:
        raise ValueError("QMIX checkpoint actor edge transform differs from this workspace")
    if data.get("implementation_contract", {}).get("learning_target") != LEARNING_TARGET_CONTRACT:
        raise ValueError("QMIX checkpoint learning target differs from this workspace")
    if data.get("implementation_contract", {}).get("recurrent_context") != RECURRENT_CONTEXT_CONTRACT:
        raise ValueError("QMIX checkpoint recurrent context differs from this workspace")
    if data.get("implementation_contract", {}).get("exploration") != EXPLORATION_CONTRACT:
        raise ValueError("QMIX checkpoint exploration differs from this workspace")
    if data.get("implementation_contract", {}).get("cpu_utility") != CPU_UTILITY_CONTRACT:
        raise ValueError("QMIX checkpoint CPU utility differs from this workspace")
    return data


class QMIXTrainer:
    def __init__(self, config: RunConfig, *, device=None):
        validate_qmix_config(config)
        require_formal_rl_enabled(config)
        q = config.training.qmix
        horizon = config.environment.episode_horizon
        if q.max_training_environment_steps != q.max_training_episodes * horizon:
            raise ValueError("QMIX requires an exact whole-episode budget")
        if q.checkpoint_interval_steps % horizon:
            raise ValueError("QMIX checkpoints require complete episode boundaries")
        self.config = config
        self.device = torch.device(device or config.training.mappo.training_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("requested CUDA is unavailable; no silent CPU fallback")
        self.learner = QMIXLearner(config, self.device)
        self.actor_adapter = ActorObservationTensorizer(config)
        self.state_adapter = CentralizedStateTensorizer(config)
        self.epsilon_rng = make_rng(config.seed, config.reproducibility.stream_ids["qmix_epsilon"])
        self.replay = SequenceReplay(q.replay_capacity_sequences, config.seed)
        self.steps = 0
        self.next_episode = 0
        self.episodes: list[dict] = []
        self.updates: list[dict] = []

    def payload(self, kind):
        return {
            "schema": CHECKPOINT_SCHEMA, "method_id": METHOD_ID,
            "kind": kind, "config_hash": self.config.config_hash,
            "implementation_source_sha256": implementation_fingerprints(),
            "config_snapshot": self.config.snapshot_dict(),
            "implementation_contract": {
                "cpu_utility": copy.deepcopy(CPU_UTILITY_CONTRACT),
                "update_frequency": "one_per_completed_chunk_after_sequence_batch_warmup",
                "no_burn_in": False, "hidden": "current_weights_full_episode_prefix_detached",
                "recurrent_context": RECURRENT_CONTEXT_CONTRACT,
                "mixer_hidden": 32, "hypernetwork_hidden": 128,
                "loss": "real_transition_mean_half_MSE", "gradient_clip_norm": 10.0,
                "target_action": "online_exact_masked_greedy_evaluated_by_target",
                "learning_target": LEARNING_TARGET_CONTRACT,
                "td_lambda": TD_LAMBDA,
                "trace_boundary": "real_chunk_end_bootstrap_no_padding",
                "agent_utility": "independent_branch_scores_active_sum_v1",
                "decoder": "exact_route_tx_cpu_suffix_max_v1",
                "target_update_units": "optimizer_updates", "dtype": "float32",
                "mixer_state_transform": "signed_log1p_then_stateless_layer_norm",
                "actor_edge_transform": ACTOR_EDGE_TRANSFORM,
                "exploration": copy.deepcopy(EXPLORATION_CONTRACT),
            },
            "runtime": {"torch": str(torch.__version__), "device": str(self.device)},
            "online": self.learner.online.state_dict(),
            "target": self.learner.target.state_dict(),
            "optimizer": self.learner.optimizer.state_dict(),
            "replay": self.replay.to_state(),
            "epsilon_rng": copy.deepcopy(self.epsilon_rng.bit_generator.state),
            "steps": self.steps, "next_episode": self.next_episode,
            "update_count": self.learner.updates,
            "episodes": copy.deepcopy(self.episodes), "updates": copy.deepcopy(self.updates),
        }

    def restore(self, path):
        data = load_checkpoint(path)
        if data["kind"] != "periodic" or data["config_hash"] != self.config.config_hash:
            raise ValueError("resume requires a periodic checkpoint of the identical configuration")
        if data["runtime"]["device"] != str(self.device) or data["runtime"]["torch"] != str(torch.__version__):
            raise ValueError("exact resume requires matching device and torch runtime")
        if data["steps"] != data["next_episode"] * self.config.environment.episode_horizon:
            raise ValueError("checkpoint is not at a complete episode boundary")
        self.learner.online.load_state_dict(data["online"], strict=True)
        self.learner.target.load_state_dict(data["target"], strict=True)
        self.learner.optimizer.load_state_dict(data["optimizer"])
        self.replay = SequenceReplay.from_state(data["replay"])
        self.epsilon_rng.bit_generator.state = copy.deepcopy(data["epsilon_rng"])
        self.steps, self.next_episode = data["steps"], data["next_episode"]
        self.learner.updates = data["update_count"]
        self.episodes, self.updates = data["episodes"], data["updates"]

    def train(self, *, write_checkpoints=False, stop_after_episodes=None):
        q = self.config.training.qmix
        limit = q.max_training_episodes if stop_after_episodes is None else stop_after_episodes
        if not self.next_episode < limit <= q.max_training_episodes:
            raise ValueError("training episode limit must advance within the configured budget")
        checkpoint_dir = Path(self.config.artifact_paths()["checkpoint_directory"])
        if write_checkpoints:
            targets = [checkpoint_dir / ("final.pt" if step == q.max_training_environment_steps else f"step_{step:06d}.pt")
                       for step in range(self.steps + self.config.environment.episode_horizon,
                                         limit*self.config.environment.episode_horizon+1,
                                         self.config.environment.episode_horizon)
                       if step % q.checkpoint_interval_steps == 0 or step == q.max_training_environment_steps]
            if any(p.exists() for p in targets):
                raise FileExistsError("QMIX checkpoint exists; refusing overwrite before collection")
        for episode in range(self.next_episode, limit):
            environment = U2UMECEnvironment(episode_config(self.config, episode))
            reset = environment.reset()
            observations, state = reset.observations, reset.centralized_state
            hidden = self.learner.online.agent.initial_hidden()
            target_hidden = self.learner.target.agent.initial_hidden()
            pending = []
            total_reward = positive = penalties = 0.0
            exploration_agent_decisions = greedy_agent_decisions = 0
            first_update = len(self.updates)
            for slot in range(self.config.environment.episode_horizon):
                with torch.no_grad():
                    actor = self.actor_adapter.encode_step(observations, device=self.device, episode_start=slot == 0)
                    contracts = tuple(obs.action_masks for obs in observations)
                    actions = self.learner.online.agent.actions(actor, hidden, ((contracts,),),
                                epsilon=epsilon_at(self.config, self.steps), rng=self.epsilon_rng)
                    _, next_target_hidden = self.learner.target.agent.encode(actor, target_hidden)
                proposals = actions.proposals[0][0]
                explored = int(actions.exploration_flags.sum().item())
                exploration_agent_decisions += explored
                greedy_agent_decisions += actions.exploration_flags.numel() - explored
                actual_actor = self.actor_adapter.encode_step(observations, proposals, episode_start=slot == 0)
                state_tensor = self.state_adapter.encode_step(state).features
                result = environment.step(proposals)
                done = result.terminated or result.truncated or slot == self.config.environment.episode_horizon - 1
                if done and (result.observations is not None or result.centralized_state is not None):
                    raise ValueError("last slot must expose no fake next decision")
                if not done and (result.observations is None or result.centralized_state is None):
                    raise ValueError("nonterminal transition omitted its next decision")
                pending.append({"actor": actual_actor, "contracts": contracts, "proposals": proposals,
                    "state": state_tensor, "reward": result.reward, "bootstrap": not done,
                    "hidden": hidden.detach().cpu(), "target_hidden": target_hidden.detach().cpu(),
                    "executed": copy.deepcopy(result.info["executed"]), "active": actions.active.detach().cpu()})
                hidden, target_hidden = actions.hidden_out, next_target_hidden
                self.steps += 1
                total_reward += result.reward
                reward_record = result.info["reward"]
                positive += float(reward_record["completion_component"])
                penalties += sum(float(reward_record[key]) for key in ("expiration_penalty", "workload_penalty", "energy_penalty"))
                if len(pending) == q.recurrent_sequence_length_slots or done:
                    next_actor = None if done else self.actor_adapter.encode_step(result.observations, episode_start=False)
                    sequence = ReplaySequence(
                        actor=stack_actor_time([item["actor"] for item in pending]),
                        contracts=(tuple(item["contracts"] for item in pending),),
                        proposals=(tuple(item["proposals"] for item in pending),),
                        state=torch.cat([item["state"] for item in pending], dim=1),
                        reward=torch.tensor([[item["reward"] for item in pending]], dtype=torch.float32),
                        bootstrap=torch.tensor([[item["bootstrap"] for item in pending]], dtype=torch.bool),
                        hidden=pending[0]["hidden"], target_hidden=pending[0]["target_hidden"],
                        next_actor=next_actor,
                        next_contracts=None if done else tuple(obs.action_masks for obs in result.observations),
                        next_state=None if done else self.state_adapter.encode_step(result.centralized_state).features,
                        episode=episode, first_slot=slot-len(pending)+1,
                        executed=tuple(item["executed"] for item in pending),
                        active=torch.cat([item["active"] for item in pending], dim=1),
                    )
                    self.replay.add(sequence)
                    pending = []
                    if len(self.replay.sequences) >= q.sequence_batch_size:
                        update = self.learner.update(self.replay.sample(q.sequence_batch_size), replay=self.replay)
                        self.updates.append({**update, "environment_steps": self.steps})
                if done:
                    break
                observations, state = result.observations, result.centralized_state
            conservation = environment.assert_invariants()
            if not conservation.is_conserved:
                raise ValueError("QMIX episode violates environment conservation")
            metrics = environment.metrics.snapshot()
            episode_updates = self.updates[first_update:]
            loss = None if not episode_updates else sum(row["loss"] for row in episode_updates)/len(episode_updates)
            row = {"run_id": self.config.run_id, "epoch": episode+1, "episode": episode,
                   "seed": self.config.seed, "environment_seed": environment.config.seed,
                   "environment_steps": self.steps, "alpha": q.learning_rate,
                   "epsilon": epsilon_at(self.config, self.steps), "loss": loss,
                   "loss_type": "half_MSE", "reward": total_reward, "reward_positive": positive,
                   "penalty_total": penalties, "signal_gate_status": "insufficient-horizon" if episode < 9 else "signal-watch",
                   "completed": metrics["completed_task_count"], "generated": metrics["generated_task_count"],
                   "completion_rate": metrics["completion_rate"], "total_energy_j": metrics["total_active_energy_j"],
                   "optimizer_updates": self.learner.updates,
                   "exploration_agent_decisions": exploration_agent_decisions,
                   "greedy_agent_decisions": greedy_agent_decisions}
            self.episodes.append(row)
            self.next_episode = episode+1
            print(f"epoch={episode+1}/{q.max_training_episodes} steps={self.steps} lr={q.learning_rate:g} eps={row['epsilon']:.4f} loss={loss} reward={total_reward:.5g} done={row['completed']}/{row['generated']}", flush=True)
            if write_checkpoints and (self.steps % q.checkpoint_interval_steps == 0 or self.steps == q.max_training_environment_steps):
                kind = "final" if self.steps == q.max_training_environment_steps else "periodic"
                path = checkpoint_dir / ("final.pt" if kind == "final" else f"step_{self.steps:06d}.pt")
                atomic_write_bytes_group([(path, checkpoint_bytes(self.payload(kind)))], group_name="QMIX checkpoint")
        return tuple(self.episodes)


def write_training_artifacts(trainer: QMIXTrainer):
    """Persist real records only; short runs remain engineering diagnostics."""
    config = trainer.config
    paths = config.artifact_paths()
    csv_stream = io.StringIO(newline="")
    writer = csv.DictWriter(csv_stream, fieldnames=tuple(trainer.episodes[0]))
    writer.writeheader()
    writer.writerows(trainer.episodes)
    payloads = [
        (Path(paths["config_snapshot"]), json.dumps(config.snapshot_dict(), ensure_ascii=False, indent=2).encode("utf-8")),
        (Path(paths["raw_metrics"]), ("\n".join(json.dumps(row, ensure_ascii=False) for row in trainer.episodes)+"\n").encode("utf-8")),
        (Path(paths["aggregate_metrics"]), json.dumps({"schema": "qmix_training.v1", "method_id": METHOD_ID,
            "environment_steps": trainer.steps, "optimizer_updates": trainer.learner.updates,
            "episodes": trainer.episodes, "updates": trainer.updates,
            "signal_gate_status": trainer.episodes[-1]["signal_gate_status"],
            "claim_boundary": "Training diagnostics; not evidence of superiority or convergence"}, ensure_ascii=False, indent=2).encode("utf-8")),
        (Path(paths["dashboard_csv"]), csv_stream.getvalue().encode("utf-8")),
    ]
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    epochs = [row["epoch"] for row in trainer.episodes]
    for axis, key in zip(axes.flat, ("loss", "reward", "completion_rate", "epsilon")):
        axis.plot(epochs, [float("nan") if row[key] is None else row[key] for row in trainer.episodes])
        axis.set(xlabel="Epoch (one episode)", ylabel=key)
    fig.suptitle(f"QMIX training diagnostics | seed {config.seed} | {trainer.steps} steps")
    fig.tight_layout()
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=140)
    plt.close(fig)
    payloads.append((Path(paths["dashboard_png"]), png.getvalue()))
    atomic_write_bytes_group(payloads, group_name="QMIX training diagnostics")
    return tuple(str(path) for path, _ in payloads)
