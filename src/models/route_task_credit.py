"""E14c task-event Route returns with a training-only task value baseline.

This is an auxiliary completion objective, not reward-equivalent RUDDER or
COMA. Environment/team rewards remain intact. Targets use only the current
frozen-policy rollout; no delayed actor replay across a PPO policy update.
Defer stops at a later binding's *post-step* value, never at that binding's
eventual DONE payoff. Unbound expiry remains a terminal event for early Defer.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch import Tensor, nn

from .ca_gat_mappo_route_nstep import build_route_credit_step_metadata
from .route_exploration import collect_route_training_records, route_top_statistics


MODE = "task_event_value_v1"
KEY = "route_task_credit_v1"


class RouteTaskValue(nn.Module):
    """Independent centralized task value; never called by the actor/evaluator."""

    def __init__(self, state_dim: int, hidden: int, uav_count: int, seed: int):
        super().__init__()
        self.task_dim = 10 + 2 * uav_count
        with torch.random.fork_rng(devices=[]):
            # Seed only CPU initialization: torch.manual_seed also reseeds CUDA
            # generators, which a CPU-only fork would not restore.
            torch.default_generator.manual_seed(seed + 140003)
            self.network = nn.Sequential(
                nn.Linear(state_dim + self.task_dim, hidden), nn.LeakyReLU(),
                nn.LayerNorm(hidden), nn.Linear(hidden, 1))
            # Cold baseline is neutral, rather than a random preference.
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def forward(self, state: Tensor, task: Tensor) -> Tensor:
        if task.shape[-1] != self.task_dim or state.shape[:-1] != task.shape[:-1]:
            raise ValueError("task value inputs have incompatible axes")
        return self.network(torch.cat((state, task), -1)).squeeze(-1)


def task_features(task: dict, slot: int, config) -> list[float]:
    """Known current task state, for the critic only; no terminal labels/IDs."""
    e = config.environment
    phases = ("unbound", "local", "tx", "cpu")
    if task["status"] not in phases:
        raise ValueError("task value requires a live task snapshot")
    bits = e.task_data_bits_max
    cycles = bits * e.task_cycles_per_bit_max
    horizon = e.maximum_task_slack_slots
    result = [task["remaining_bits"] / bits, task["data_bits"] / bits,
              task["remaining_cycles"] / cycles, task["cpu_cycles"] / cycles,
              (task["deadline_slot"] - slot) / horizon,
              (slot - task["arrival_slot"]) / horizon]
    result += [float(task["status"] == phase) for phase in phases]
    result += [float(task["source_uav"] == u) for u in range(e.uav_count)]
    result += [float(task["destination"] == u) for u in range(e.uav_count)]
    return result


def capture_task_values(environment, centralized_features: Tensor, critic, config) -> list[dict]:
    """Observe all live tasks under the frozen critic, without RNG or mutation."""
    tasks = [t.snapshot() for t in environment.lifecycle.tasks.values() if not t.is_terminal]
    if not tasks:
        return []
    state = centralized_features.reshape(-1, centralized_features.shape[-1])
    if state.shape[0] != 1:
        raise ValueError("factual task value capture expects one slot")
    features = [task_features(t, environment.slot, config) for t in tasks]
    with torch.no_grad():
        values = critic.route_task_value(
            state.expand(len(tasks), -1), state.new_tensor(features)).cpu().tolist()
    return [{"task_id": t["task_id"], "source_uav": t["source_uav"],
             "status": t["status"], "destination": t["destination"],
             "features": f, "old_value": v}
            for t, f, v in zip(tasks, features, values)]


def make_step_metadata(*, before, after, info, policy, episode_id, environment_step,
                       policy_version, capture_route_leaves=False) -> dict:
    """Reuse settled TaskTerminalAttribution and existing route event identities."""
    events = build_route_credit_step_metadata(
        info, episode_id=episode_id, global_rollout_index=environment_step,
        policy_version=policy_version)
    attribution = {t["task_id"]: t for t in
                   info.get("agent_credit", {}).get("terminal_attributions", ())}
    for terminal in events["terminals"]:
        kind = terminal["kind"]
        if kind == "truncated":
            payoff = 0.0  # Censored at episode horizon, never relabel as EXPIRED.
        else:
            item = attribution.get(terminal["task_id"])
            if item is None:
                raise ValueError("E14c requires existing task terminal attribution")
            field = "completion_allocation" if kind == "completed" else "expiration_allocation"
            suffix = "reward" if kind == "completed" else "penalty"
            allocation = item[field]
            payoff = sum(float(allocation[f"{role}_{suffix}"]) for role in ("source", "destination"))
            if kind == "expired":
                payoff = -payoff
        terminal["payoff"] = payoff
        terminal["slot"] = int(info["slot"])
    records = collect_route_training_records(
        policy, environment_step=environment_step, episode_id=episode_id, slot=int(info["slot"]),
        task_ids={d["source_uav"]: d["task_id"] for d in events["decisions"]})
    by_uav = {r["uav_id"]: r for r in records}
    by_task = {t["task_id"]: t for t in before}
    for d in events["decisions"]:
        if d["task_id"] not in by_task or d["source_uav"] not in by_uav:
            raise ValueError("Route decision lacks a current task/value/probability snapshot")
        d.update(by_uav[d["source_uav"]])
        if capture_route_leaves:
            u = d["source_uav"]
            d["actor_leaf_probability"] = policy.probabilities["route"][0, 0, u].detach().cpu().tolist()
            d["route_legal_mask"] = policy.action_masks["route"][0, 0, u].detach().cpu().tolist()
            remote_mask = policy.action_masks["route"][0, 0, u, 3:]
            remote_logits = policy.raw_logits["route"][0, 0, u, 3:].detach()
            conditional = (remote_logits.masked_fill(~remote_mask, -torch.inf).softmax(-1)
                           if remote_mask.any() else torch.zeros_like(remote_logits))
            d["actor_remote_conditional_probability"] = conditional.cpu().tolist()
        # At deadline/horizon a newly bound task can settle immediately.
        if d["category"] == "local":
            d["destination"] = d["source_uav"]
        elif d["category"] == "remote":
            routing = next(r for r in info["service"]["routing"]
                           if r["task_id"] == d["task_id"])
            d["destination"] = int(routing["proposal"])
        else:
            d["destination"] = None
    events.update(slot=int(info["slot"]), before=before, after=after)
    return events


@dataclass
class TaskCreditBatch:
    advantage: Tensor
    task_advantage: Tensor
    samples: list[dict]
    state_indices: Tensor
    task_features: Tensor
    value_targets: Tensor


def construct_task_credit(metadata: list[dict], shared_advantage: Tensor,
                          route_active: Tensor, gamma: float, coefficient: float) -> TaskCreditBatch:
    """Freeze task targets/advantages once before all PPO epochs.

    Terminal at end of slot d has exponent d-t. A post-slot value describes
    state d+1 and has exponent d+1-t. Episode keys prevent task-ID reuse leaks.
    All current live-task states train the baseline, including bound tasks.
    """
    if len(metadata) != shared_advantage.shape[0] or route_active.shape != shared_advantage.shape:
        raise ValueError("task credit axes differ from the recurrent rollout")
    states = {}; terminals = {}; binds = {}; frontier = {}
    versions = set()
    for i, m in enumerate(metadata):
        ep = m["episode_id"]; versions.add(m["policy_version"])
        for s in m["before"]:
            key = (ep, s["source_uav"], s["task_id"])
            if (i, key) in states:
                raise ValueError("duplicate task state")
            states[i, key] = s
        for s in m["after"]:
            frontier[ep, s["source_uav"], s["task_id"]] = (i, m["slot"] + 1, s["old_value"])
        for t in m["terminals"]:
            key = (ep, t["source_uav"], t["task_id"])
            if key in terminals:
                raise ValueError("duplicate task terminal attribution")
            terminals[key] = t
        for d in m["decisions"]:
            key = (ep, d["source_uav"], d["task_id"])
            if d["category"] != "defer":
                if key in binds:
                    raise ValueError("task was bound twice")
                post = next((s for s in m["after"] if s["task_id"] == d["task_id"]), None)
                if post is None:
                    if key not in terminals:
                        raise ValueError("binding has neither post-step value nor terminal")
                    # Same-slot expiry/truncation is already observed at the
                    # binding event, rather than a later successful DONE label.
                    binds[key] = (i, m["slot"], terminals[key]["payoff"], True)
                else:
                    binds[key] = (i, m["slot"] + 1, post["old_value"], False)
    if len(versions) != 1:
        raise ValueError("E14c actor credit cannot cross a policy-version boundary")

    def task_return(key, slot, index):
        terminal = terminals.get(key)
        if terminal is not None:
            delay = terminal["slot"] - slot
            if delay < 0:
                raise ValueError("task state is after its terminal")
            return gamma ** delay * terminal["payoff"], terminal["kind"], delay
        if key not in frontier:
            raise ValueError("live task lacks rollout frontier bootstrap")
        end_index, end_slot, value = frontier[key]
        if end_index != len(metadata) - 1:
            raise ValueError("live task bootstrap is stale or crossed an episode boundary")
        delay = end_slot - slot
        return gamma ** delay * value, "rollout_bootstrap", delay

    combined = shared_advantage.detach().clone()
    task_adv = torch.zeros_like(combined)
    samples = []; indices = []; features = []; targets = []
    seen = torch.zeros_like(route_active)
    for i, m in enumerate(metadata):
        ep, slot = m["episode_id"], m["slot"]
        for s in m["before"]:
            key = (ep, s["source_uav"], s["task_id"])
            target, _, _ = task_return(key, slot, i)
            indices.append(i); features.append(s["features"]); targets.append(target)
        for d in m["decisions"]:
            u = d["source_uav"]; key = (ep, u, d["task_id"])
            if not bool(route_active[i, u]) or bool(seen[i, u]):
                raise ValueError("task decision disagrees with active Route mask")
            seen[i, u] = True
            s = states[i, key]
            full_return, stop, delay = task_return(key, slot, i)
            target = full_return
            if d["category"] == "defer" and key in binds:
                bind_index, next_slot, next_value, settled = binds[key]
                if bind_index <= i:
                    raise ValueError("Defer cannot occur after task binding")
                target = gamma ** (next_slot - slot) * next_value
                stop = "binding_same_slot_terminal" if settled else "binding_value_bootstrap"
            a = target - s["old_value"]
            task_adv[i, u] = a
            combined[i, u] += coefficient * a
            terminal = terminals.get(key)
            samples.append({**d, "route_slot": slot, "rollout_offset": i,
                            "shared_advantage": float(shared_advantage[i, u]),
                            "route_specific_advantage": a,
                            "coefficient": coefficient,
                            "combined_advantage": float(combined[i, u]),
                            "return": target, "task_mc_return": full_return,
                            "value": s["old_value"], "credit_stop": stop,
                            "terminal_status": None if terminal is None else terminal["kind"],
                            "terminal_slot": None if terminal is None else terminal["slot"],
                            "route_terminal_delay": None if terminal is None else delay,
                            "censored": terminal is None or terminal["kind"] == "truncated"})
    if not torch.equal(seen, route_active):
        raise ValueError("active Route samples are missing task credit metadata")
    dim = len(features[0]) if features else 0
    return TaskCreditBatch(combined, task_adv, samples,
                           torch.tensor(indices, dtype=torch.long),
                           torch.tensor(features, dtype=torch.float32).reshape(len(features), dim),
                           torch.tensor(targets, dtype=torch.float32))


def prepare_task_credit(buffer, shared_advantage: Tensor, config) -> TaskCreditBatch:
    metadata = []
    for i in range(len(buffer)):
        m = buffer.transition_at(i).rejection_or_downgrade_summary.get(KEY)
        if m is None:
            raise ValueError("enabled E14c task credit metadata is missing")
        metadata.append(m)
    active = torch.stack([buffer.transition_at(i).active_branch_indicators[:, 0]
                          for i in range(len(buffer))])
    return construct_task_credit(metadata, shared_advantage.reshape(len(buffer), -1).cpu(),
                                 active, config.training.mappo.gamma,
                                 config.training.mappo.route_task_credit_coefficient)


def task_value_loss(critic, centralized_features: Tensor, credit: TaskCreditBatch) -> Tensor:
    if not credit.state_indices.numel():
        return next(critic.route_task_value.parameters()).sum() * 0.0
    state = centralized_features.reshape(-1, centralized_features.shape[-1])
    idx = credit.state_indices.to(state.device)
    current = critic.route_task_value(state[idx].detach(), credit.task_features.to(state))
    return 0.5 * (current - credit.value_targets.to(state)).square().mean()


def epoch_task_records(credit, policy, old_route_logs, epoch_index, epsilon) -> list[dict]:
    """Actual Route leaf ratio, including conditional Remote destination."""
    logs = policy.branch_log_probs["route"].detach().cpu().reshape(-1, old_route_logs.shape[-1])
    old = old_route_logs.detach().cpu().reshape_as(logs)
    top_logs, support = route_top_statistics(policy.raw_logits["route"], policy.action_masks["route"])
    top = torch.where(support, top_logs.exp(), torch.zeros_like(top_logs)).detach().cpu().reshape(-1, logs.shape[-1], 3)
    rows = []
    for s in credit.samples:
        i, u = s["rollout_offset"], s["source_uav"]
        ratio = float((logs[i, u] - old[i, u]).exp())
        clipped = min(1 + epsilon, max(1 - epsilon, ratio))
        a = s["combined_advantage"]
        rows.append({**s, "epoch_index": epoch_index,
                     "new_actor_top_probability": dict(zip(("Local", "Remote", "Defer"), top[i, u].tolist())),
                     "old_log_prob": float(old[i, u]), "new_log_prob": float(logs[i, u]),
                     "ppo_ratio": ratio, "unclipped_surrogate": ratio * a,
                     "clipped_surrogate": clipped * a, "surrogate": min(ratio * a, clipped * a)})
    return rows


def summarize_credit_records(rows) -> dict:
    fields = ("shared_advantage", "route_specific_advantage", "combined_advantage",
              "return", "value", "ppo_ratio", "surrogate")
    groups = {}
    for phase in range(4):
        for action in ("local", "remote", "defer"):
            rr = [r for r in rows if r["category"] == action and
                  phase * 2000 <= r["global_rollout_index"] < (phase + 1) * 2000]
            groups[f"{phase * 2000}-{(phase + 1) * 2000}/{action}"] = {
                "count": len(rr), "unique_factual_count": len({
                    (r["episode_id"], r["global_rollout_index"], r["source_uav"]) for r in rr}),
                **{f"mean_{f}": sum(r[f] for r in rr) / len(rr) if rr else None for f in fields},
                "positive_task_advantage_count": sum(r["route_specific_advantage"] > 0 for r in rr),
                "negative_task_advantage_count": sum(r["route_specific_advantage"] < 0 for r in rr)}
    return groups


def summarize_route_credit_components(rows) -> dict:
    """Per-epoch/action credit diagnostics; missing CF is NA, never invented."""
    groups = {}
    for epoch in sorted({r["epoch_index"] for r in rows}):
        for category in ("local", "remote", "defer"):
            rr = [r for r in rows if r["epoch_index"] == epoch and r["category"] == category]
            components = {}
            for name, fallback in (("shared_advantage", None),
                                   ("task_advantage", "route_specific_advantage"),
                                   ("counterfactual_advantage", None),
                                   ("final_route_advantage", "combined_advantage")):
                values = [r.get(name, r.get(fallback)) for r in rr]
                values = [float(v) for v in values if v is not None]
                if not all(math.isfinite(v) for v in values):
                    raise ValueError("non-finite Route credit telemetry")
                n = len(values)
                mean = sum(values) / n if n else None
                components[name] = {
                    "count": n, "mean": mean,
                    "std": math.sqrt(sum((v-mean)**2 for v in values)/n) if n else None,
                    "rms": math.sqrt(sum(v*v for v in values)/n) if n else None,
                    "min": min(values) if n else None, "max": max(values) if n else None,
                    "positive_count": sum(v > 0 for v in values),
                    "negative_count": sum(v < 0 for v in values),
                    "zero_count": sum(v == 0 for v in values)}
            groups[f"epoch{epoch}/{category}"] = components
    return groups


def write_task_credit_telemetry(config, rows, environment_steps, value_losses, *, buffer=None, optimized=True):
    path = Path(config.artifact_paths()["raw_metrics"]).parent / "route_task_credit_updates_v1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"schema": "route_task_credit_update.v1", "run_id": config.run_id,
           "config_hash": config.config_hash, "environment_step_end": environment_steps,
           "credit_mode": MODE,
           "actor_advantage_mode": (
               MODE if (config.training.mappo.route_value_mode == "off"
                        or config.training.mappo.route_actor_credit_source == "task_event")
               else config.training.mappo.route_value_mode),
           "actor_credit_source": config.training.mappo.route_actor_credit_source,
           "task_credit_coefficient": config.training.mappo.route_task_credit_coefficient,
           "effective_actor_task_credit_coefficient": (
               config.training.mappo.route_task_credit_coefficient
               if (config.training.mappo.route_value_mode == "off"
                   or config.training.mappo.route_actor_credit_source == "task_event") else 0.0),
           "effective_actor_counterfactual_coefficient": (
               config.training.mappo.route_counterfactual_credit_coefficient
               if (config.training.mappo.route_value_mode != "off"
                   and config.training.mappo.route_actor_credit_source == "counterfactual_q") else 0.0),
           "credit_components_by_epoch": summarize_route_credit_components(rows),
           "task_value_coefficient": config.training.mappo.route_task_value_coefficient,
           "policy_statistics_timing": "pre_optimizer_step", "samples": rows,
           "optimized": optimized,
           "groups": summarize_credit_records(rows), "task_value_loss_by_epoch": value_losses}
    if config.training.mappo.route_decoder_mode == "unified_node_v1":
        row.update(schema="route_task_credit_update.v2", decision_mode="flat_leaf_v1")
    if buffer is not None:
        # Keep later terminal facts joinable for diagnostics, including tasks
        # whose binding was in an earlier rollout. Never replay their actor loss.
        events = [buffer.transition_at(i).rejection_or_downgrade_summary[KEY]
                  for i in range(len(buffer))]
        row["task_terminals"] = [{**t, "episode_id": m["episode_id"]}
                                 for m in events for t in m["terminals"]]
        row["route_decisions"] = [d for m in events for d in m["decisions"]]
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    return row
