"""Opt-in, bounded observation of factual service and real PPO epochs."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from ..config import PolicyLossAggregationMode, RunConfig
from .ca_gat_mappo_actions import ACTION_BRANCH_ORDER


class ConditionalCreditTraceError(RuntimeError):
    """A diagnostic trace is incomplete or exceeds its explicit budget."""


class ConditionalCreditTraceWriter:
    """Fresh-run-only sidecar. One action and one PPO row per agent transition."""

    BRANCHES = ("tx_select", "cpu_queue", "cpu_frequency")

    def __init__(self, config: RunConfig) -> None:
        if not config.training.mappo.conditional_credit_telemetry_enabled:
            raise ConditionalCreditTraceError("conditional credit telemetry is disabled")
        self.config = config
        self.path = Path(config.artifact_paths()["conditional_credit_trace"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8", newline="\n")
        self._bytes = 0
        self._counts: Counter[str] = Counter()
        self._actions: set[tuple[int, int]] = set()
        self._task_terminals: set[tuple[int, int]] = set()
        self._task_censored: set[tuple[int, int]] = set()
        self._task_routes: set[tuple[int, int, int]] = set()
        self._epochs: set[tuple[int, int]] = set()
        self._ppo_rows: set[tuple[int, int, int, int]] = set()
        self._update_index: int | None = None
        self._next_update_index = 0
        self._closed = False
        self.write("header", {
            "schema_version": "conditional_credit_trace.v1",
            "seed": config.seed,
            "git_dirty": config.git_dirty,
            "rollout_length": config.training.mappo.rollout_length_slots,
            "update_epochs": config.training.mappo.update_epochs,
            "uav_count": config.environment.uav_count,
            "policy_loss_aggregation_mode": config.training.mappo.policy_loss_aggregation_mode,
            "max_bytes": config.training.mappo.conditional_credit_telemetry_max_bytes,
        })

    def write(self, event_type: str, fields: Mapping[str, Any]) -> None:
        if self._closed:
            raise ConditionalCreditTraceError("trace is closed")
        row = {"event_type": event_type, "run_id": self.config.run_id,
               "config_hash": self.config.config_hash,
               "git_commit": self.config.git_commit, **fields}
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False,
                             separators=(",", ":")) + "\n"
        length = len(encoded.encode("utf-8"))
        if self._bytes + length > self.config.training.mappo.conditional_credit_telemetry_max_bytes:
            raise ConditionalCreditTraceError("conditional credit trace byte budget exhausted")
        self._stream.write(encoded)
        self._bytes += length
        self._counts[event_type] += 1

    def record_step(self, *, global_index: int, episode: int, slot: int,
                    policy_version: int, proposals: tuple[Any, ...],
                    action_output: Any, info: Mapping[str, Any],
                    environment: Any) -> None:
        service = info["service"]
        executed = info["executed"]
        if len(executed) != len(proposals):
            raise ConditionalCreditTraceError("executed action count differs from proposals")
        rollout_index = global_index // self.config.training.mappo.rollout_length_slots
        for agent, proposal in enumerate(proposals):
            key = (global_index, agent)
            if key in self._actions:
                raise ConditionalCreditTraceError("duplicate agent action")
            self._actions.add(key)
            actual = executed[agent]
            self.write("agent_action", {
                "action_key": list(key), "episode": episode, "slot": slot,
                "rollout_index": rollout_index, "policy_version": policy_version,
                "agent": agent, "tx_upper": proposal.tx_select,
                "tx_active": bool(action_output.active_branches["tx_select"][0, 0, agent].item()),
                "selected_tx_destination": proposal.tx_select if proposal.tx_select != "idle" else None,
                "cpu_queue": proposal.cpu_queue, "cpu_frequency": proposal.cpu_frequency,
                "executed_tx": actual["communication"], "executed_cpu": actual["cpu"],
            })
        for route in service["routing"]:
            task_id = route.get("task_id")
            if task_id is not None:
                route_key = (episode, int(task_id), global_index)
                if route_key in self._task_routes:
                    raise ConditionalCreditTraceError("duplicate task route identity")
                self._task_routes.add(route_key)
                self.write("task_route", {"task_key": [episode, int(task_id)],
                    "route_key": list(route_key),
                    "global_index": global_index, "route": route})
        for link_index, link in enumerate(service["links"]):
            agent = int(link["sender_uav"])
            for task_index, item in enumerate(link["task_services"]):
                task_id = int(item["task_id"])
                task = environment.lifecycle.tasks[task_id]
                self.write("task_service", {
                    "service_key": [global_index, "tx", link_index, task_index],
                    "action_key": [global_index, agent],
                    "task_key": [episode, task_id], "stage": "tx",
                    "amount": float(item["amount"]),
                    "destination_uav": int(link["receiver_uav"]),
                    "remaining_bits_after_slot": float(task.remaining_bits),
                    "rollout_index": rollout_index,
                })
        for cpu_index, item in enumerate(service["cpu"]):
            task_id = item.get("task_id")
            if task_id is None:
                continue
            agent = int(item["executor_uav"])
            self.write("task_service", {
                "service_key": [global_index, "cpu", cpu_index],
                "action_key": [global_index, agent],
                "task_key": [episode, int(task_id)], "stage": "cpu",
                "amount": float(item["service_cycles"]),
                "executed_frequency_hz": float(item["executed_frequency_hz"]),
                "rollout_index": rollout_index,
            })
        for item in (*service["settled_tasks"], *service["truncated_tasks"]):
            key = (episode, int(item["task_id"]))
            if key in self._task_terminals:
                raise ConditionalCreditTraceError("duplicate task terminal")
            self._task_terminals.add(key)
            self.write("task_terminal", {
                "task_key": list(key), "global_index": global_index,
                "rollout_index": rollout_index, "outcome": item["outcome"],
                "source_uav": item["source_uav"], "destination_uav": item["destination"],
                "deadline_slot": item["deadline_slot"],
                "task_snapshot": item,
            })

    def begin_update(self, update_index: int) -> None:
        if self._update_index is not None or update_index != self._next_update_index:
            raise ConditionalCreditTraceError("nested or out-of-order PPO update")
        self._update_index = update_index

    def record_epoch(self, *, epoch_index: int, prepared: Any,
                     evaluation: Any, loss: Any, actor: Any,
                     actor_optimizer: Any) -> None:
        update = self._update_index
        if update is None or (update, epoch_index) in self._epochs:
            raise ConditionalCreditTraceError("missing or duplicate PPO epoch")
        self._epochs.add((update, epoch_index))
        active = prepared.active_branch_indicators.detach().cpu()
        old = prepared.old_branch_log_probs.detach().cpu()
        if loss.branch_surrogate is None or loss.branch_ratio is None:
            raise ConditionalCreditTraceError("branch-specific PPO statistics are required")
        new = {name: evaluation.policy.branch_log_probs[name].detach().cpu()
               for name in self.BRANCHES}
        advantage = loss.branch_advantage.detach().cpu()
        ratio = loss.branch_ratio.detach().cpu()
        clipped_ratio = loss.flattened.branch_clipped_ratio.detach().cpu().reshape_as(ratio)
        surrogate = loss.branch_surrogate.detach().cpu()
        unclipped = loss.flattened.branch_unclipped_surrogate.detach().cpu().reshape_as(ratio)
        clipped = loss.flattened.branch_clipped_surrogate.detach().cpu().reshape_as(ratio)
        target = prepared.return_target.detach().cpu()
        joint_adv = prepared.advantage.detach().cpu()
        counts = active.sum(dim=(0, 1, 2))
        nonempty = int((counts > 0).sum().item())
        actor_count = int(active.shape[0] * active.shape[1] * active.shape[2])
        all_surrogate = loss.flattened.branch_surrogate.detach().cpu().reshape_as(active)
        if self.config.training.mappo.policy_loss_aggregation_mode == PolicyLossAggregationMode.BRANCHWISE_ACTIVE_MEAN_V1.value:
            reconstructed = float((all_surrogate.sum(dim=(0, 1, 2))[counts > 0] / counts[counts > 0]).mean())
        else:
            reconstructed = float((all_surrogate.sum(dim=-1) / active.sum(dim=-1).clamp_min(1)).mean())
        actual_objective = float(loss.clipped_objective.detach().cpu().item())
        if abs(reconstructed - actual_objective) > 1e-5:
            raise ConditionalCreditTraceError("logged aggregation weights do not reconstruct PPO objective")
        head_grad: dict[str, float | None] = {}
        for name in self.BRANCHES:
            squared = [float(torch.sum(parameter.grad.detach().square()).cpu().item())
                       for parameter in actor.action_heads[name].parameters()
                       if parameter.grad is not None]
            head_grad[name] = math.sqrt(math.fsum(squared)) if squared else None
        optimizer_steps = [state.get("step") for state in actor_optimizer.state.values()
                           if "step" in state]
        actual_optimizer_step = (int(optimizer_steps[0].item()) if optimizer_steps
                                 and isinstance(optimizer_steps[0], torch.Tensor)
                                 else int(optimizer_steps[0]) if optimizer_steps else None)
        self.write("ppo_epoch", {
            "update_index": update, "epoch_index": epoch_index,
            "optimizer_step": actual_optimizer_step,
            "branch_active_counts": counts.tolist(),
            "nonempty_branch_count": nonempty,
            "head_gradient_l2_after_clip": head_grad,
            "head_gradient_scope": "actual_full_loss_head_after_clipping_not_per_action",
            "actor_loss": float(loss.actor_loss.detach().cpu().item()),
            "clipped_objective": actual_objective,
            "reconstructed_objective": reconstructed,
        })
        branch_indices = {name: ACTION_BRANCH_ORDER.index(name) for name in self.BRANCHES}
        for chunk in range(active.shape[0]):
            for position in range(active.shape[1]):
                global_index = update * self.config.training.mappo.rollout_length_slots + chunk * active.shape[1] + position
                for agent in range(active.shape[2]):
                    row_key = (update, epoch_index, global_index, agent)
                    if row_key in self._ppo_rows or (global_index, agent) not in self._actions:
                        raise ConditionalCreditTraceError("PPO action linkage missing or duplicate")
                    self._ppo_rows.add(row_key)
                    branch_rows = {}
                    for name, branch in branch_indices.items():
                        if not bool(active[chunk, position, agent, branch]):
                            continue
                        if self.config.training.mappo.policy_loss_aggregation_mode == PolicyLossAggregationMode.BRANCHWISE_ACTIVE_MEAN_V1.value:
                            weight = 1.0 / (nonempty * int(counts[branch]))
                        else:
                            branch_count = int(active[chunk, position, agent].sum())
                            weight = 1.0 / (actor_count * branch_count)
                        branch_rows[name] = {
                            "old_log_prob": float(old[chunk, position, agent, branch]),
                            "new_log_prob": float(new[name][chunk, position, agent]),
                            "advantage": float(advantage[chunk, position, agent, branch]),
                            "ratio": float(ratio[chunk, position, agent, branch]),
                            "unclipped_surrogate": float(unclipped[chunk, position, agent, branch]),
                            "clipped_surrogate": float(clipped[chunk, position, agent, branch]),
                            "surrogate": float(surrogate[chunk, position, agent, branch]),
                            "ratio_clipped": bool(ratio[chunk, position, agent, branch] != clipped_ratio[chunk, position, agent, branch]),
                            "aggregation_weight": weight,
                            "weighted_surrogate": float(surrogate[chunk, position, agent, branch]) * weight,
                        }
                    scalar_index = (chunk, position) if target.ndim == 2 else (chunk, position, agent)
                    self.write("ppo_agent_epoch", {
                        "update_index": update, "epoch_index": epoch_index,
                        "optimizer_step": actual_optimizer_step,
                        "action_key": [global_index, agent],
                        "agent_transition_key": [global_index, agent],
                        "return_target": float(target[scalar_index]),
                        "shared_advantage": float(joint_adv[scalar_index]),
                        "branches": branch_rows,
                    })

    def end_update(self, update_index: int, epochs: int) -> None:
        if self._update_index != update_index:
            raise ConditionalCreditTraceError("PPO update identity changed")
        expected = self.config.training.mappo.rollout_length_slots * self.config.environment.uav_count * epochs
        actual = sum(key[0] == update_index for key in self._ppo_rows)
        if actual != expected:
            raise ConditionalCreditTraceError("PPO update has missing agent-epoch rows")
        self.write("update_end", {"update_index": update_index, "epoch_count": epochs,
                                  "agent_epoch_rows": actual})
        self._stream.flush()
        self._ppo_rows.clear()
        self._update_index = None
        self._next_update_index += 1

    def record_checkpoint(self, *, environment_steps: int,
                          optimized_steps: int, filename: str) -> None:
        if self._update_index is not None:
            raise ConditionalCreditTraceError("checkpoint marker during PPO update")
        self.write("checkpoint_marker", {
            "environment_steps": environment_steps,
            "optimized_steps": optimized_steps,
            "checkpoint_filename": filename,
            "event_counts_before_marker": dict(self._counts),
            "bytes_before_marker": self._bytes,
        })
        self._stream.flush()

    @classmethod
    def reopen_at_checkpoint(cls, config: RunConfig, *,
                             environment_steps: int,
                             optimized_steps: int,
                             filename: str) -> "ConditionalCreditTraceWriter":
        """Discard only post-checkpoint trace bytes; reject a missing marker."""
        path = Path(config.artifact_paths()["conditional_credit_trace"])
        if not path.is_file():
            raise ConditionalCreditTraceError("resume trace is absent")
        writer = cls.__new__(cls)
        writer.config = config
        writer.path = path
        writer._bytes = 0
        writer._counts = Counter()
        writer._actions = set()
        writer._task_terminals = set()
        writer._task_censored = set()
        writer._task_routes = set()
        writer._epochs = set()
        writer._ppo_rows = set()
        writer._update_index = None
        writer._next_update_index = 0
        writer._closed = False
        identity = (config.run_id, config.config_hash, config.git_commit)
        marker_offset = None
        ppo_keys: set[tuple[int, int, int, int]] = set()
        service_keys: set[tuple[Any, ...]] = set()
        with path.open("rb") as stream:
            while True:
                raw = stream.readline()
                if not raw:
                    break
                if not raw.endswith(b"\n"):
                    break
                try:
                    row = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ConditionalCreditTraceError("invalid trace before checkpoint marker") from exc
                if (row.get("run_id"), row.get("config_hash"), row.get("git_commit")) != identity:
                    raise ConditionalCreditTraceError("resume trace provenance differs")
                kind = row["event_type"]
                if kind == "checkpoint_marker":
                    if row["event_counts_before_marker"] != dict(writer._counts) or row["bytes_before_marker"] != writer._bytes:
                        raise ConditionalCreditTraceError("checkpoint marker counters differ")
                elif kind == "agent_action":
                    key = tuple(row["action_key"])
                    if key in writer._actions:
                        raise ConditionalCreditTraceError("duplicate resumed action")
                    writer._actions.add(key)
                elif kind == "task_route":
                    key = tuple(row.get("route_key", [*row["task_key"], row["global_index"]]))
                    if key in writer._task_routes:
                        raise ConditionalCreditTraceError("duplicate resumed route")
                    writer._task_routes.add(key)
                elif kind == "task_terminal":
                    key = tuple(row["task_key"])
                    if key in writer._task_terminals:
                        raise ConditionalCreditTraceError("duplicate resumed terminal")
                    writer._task_terminals.add(key)
                elif kind == "task_censored":
                    raise ConditionalCreditTraceError("periodic prefix contains final censoring")
                elif kind == "task_service":
                    key = tuple(row["service_key"])
                    if key in service_keys or tuple(row["action_key"]) not in writer._actions:
                        raise ConditionalCreditTraceError("duplicate or unlinked resumed task service")
                    service_keys.add(key)
                elif kind == "ppo_epoch":
                    key = (row["update_index"], row["epoch_index"])
                    if key in writer._epochs:
                        raise ConditionalCreditTraceError("duplicate resumed PPO epoch")
                    writer._epochs.add(key)
                elif kind == "ppo_agent_epoch":
                    key = (row["update_index"], row["epoch_index"], *row["action_key"])
                    if key in ppo_keys or tuple(row["action_key"]) not in writer._actions:
                        raise ConditionalCreditTraceError("duplicate or unlinked resumed PPO row")
                    ppo_keys.add(key)
                elif kind == "update_end":
                    if row["update_index"] != writer._next_update_index:
                        raise ConditionalCreditTraceError("out-of-order resumed PPO update")
                    writer._next_update_index += 1
                elif kind == "run_end":
                    raise ConditionalCreditTraceError("cannot resume a completed trace")
                writer._counts[kind] += 1
                writer._bytes += len(raw)
                if (kind == "checkpoint_marker"
                    and row["environment_steps"] == environment_steps
                    and row["optimized_steps"] == optimized_steps
                    and row["checkpoint_filename"] == filename):
                    marker_offset = writer._bytes
                    break
        if marker_offset is None:
            raise ConditionalCreditTraceError("matching complete checkpoint marker is absent")
        agents = config.environment.uav_count
        rollout = config.training.mappo.rollout_length_slots
        epochs = config.training.mappo.update_epochs
        if (writer._actions != {(step, agent) for step in range(environment_steps) for agent in range(agents)}
            or optimized_steps % rollout
            or writer._next_update_index != optimized_steps // rollout
            or len(writer._epochs) != writer._next_update_index * epochs
            or len(ppo_keys) != optimized_steps * agents * epochs):
            raise ConditionalCreditTraceError("checkpoint trace prefix has missing actions or PPO rows")
        with path.open("r+b") as stream:
            stream.truncate(marker_offset)
        writer._stream = path.open("a", encoding="utf-8", newline="\n")
        return writer

    def pause(self) -> None:
        if self._update_index is not None:
            raise ConditionalCreditTraceError("cannot pause inside PPO update")
        self._stream.flush()
        self._stream.close()
        self._closed = True

    def record_censored(self, *, environment: Any, episode: int,
                        global_index: int) -> None:
        for task in environment.lifecycle.tasks.values():
            if task.is_terminal:
                continue
            key = (episode, task.task_id)
            if key in self._task_terminals or key in self._task_censored:
                raise ConditionalCreditTraceError("duplicate censored task identity")
            self._task_censored.add(key)
            self.write("task_censored", {"task_key": list(key),
                "global_index": global_index,
                "rollout_index": global_index // self.config.training.mappo.rollout_length_slots,
                "task_snapshot": task.snapshot()})

    def close(self, *, environment_steps: int, optimized_steps: int) -> None:
        if self._update_index is not None:
            raise ConditionalCreditTraceError("cannot close an incomplete PPO update")
        self.write("run_end", {"environment_steps": environment_steps,
                "optimized_steps": optimized_steps, "event_counts_before_end": dict(self._counts),
                "bytes_before_end": self._bytes})
        self._stream.flush()
        self._stream.close()
        self._closed = True


__all__ = ["ConditionalCreditTraceError", "ConditionalCreditTraceWriter"]
