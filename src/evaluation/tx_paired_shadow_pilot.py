"""Bounded, frozen-policy paired TX reliability diagnostic; never a trainer.

Only the focal initial TX subtree changes. Subsequent actions use the frozen
policy on each arm's own observations and recurrent state. Random numbers are
addressed by absolute slot, so divergent actions cannot shift future draws.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import load_run_config
from ..env.environment import U2UMECEnvironment
from ..models.ca_gat_mappo import ActorObservationTensorizer
from ..models.ca_gat_mappo_actions import (
    CAGATMAPPOActionDistribution, SequentialActionMaskBatch,
)
from .actor_loader import load_final_actor_for_oracle_diagnostic
from .route_oracle import actor_state_digest, tensor_digest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "logs/gate_m/ca_gat_mappo__remote_stress_v2__seed-43__cfg-94adc863939f__git-21624c6cd274"
CHECKPOINT_SHA = "d21270613770994759190334164f52ddb134103434ae258de840ee246c3079c8"
ACTOR_SHA = "a00b84cc08d8d5ce59e736568c19f1ab800940417965bb7163f3975ae920c16f"
GAMMA = 0.99
MAX_SHADOW_STEPS = 24000
PAIR_COUNT = 24
NEAR_ZERO = 1e-8
COMPONENTS = ("completion_component", "expiration_penalty", "workload_penalty", "energy_penalty")
SIGNS = (1.0, -1.0, -1.0, -1.0)


class ReliabilityError(RuntimeError):
    """A failed invariant stops the diagnostic instead of accepting a label."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReliabilityError(message)


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def intrinsic_tasks(env: U2UMECEnvironment) -> list[dict[str, Any]]:
    fields = ("task_id", "source_uav", "arrival_slot", "deadline_slot", "data_bits", "cpu_cycles")
    return [{name: getattr(task, name) for name in fields}
            for _, task in sorted(env.lifecycle.tasks.items())]


def exogenous_digest(env: U2UMECEnvironment) -> str:
    """Hash actual exogenous realizations as well as the five RNG states.

    Historical CSI interference/public messages are action-dependent and are
    deliberately excluded. Current true channels, stale CSI noise, mobility,
    arrivals and channel internal shadowing remain subject to equality checks.
    """
    hasher = hashlib.sha256()
    hasher.update(str(env.slot).encode())
    hasher.update(env.rng_fingerprint().encode())
    hasher.update(digest(intrinsic_tasks(env)).encode())
    arrays = [env.mobility.positions_m, env.mobility.velocities_mps,
              env.topology.candidate_neighbors, env.topology.distances_m,
              env.channel.channel, env.channel.blocked_links,
              getattr(env.channel_model, "_previous_positions_m", None),
              getattr(env.channel_model, "_shadowing_db", None)]
    if env.channel_features is not None:
        arrays.extend([env.channel_features.stale_csi,
                       env.channel_features.csi_valid_mask,
                       env.channel_features.csi_aoi_slots])
    for array in arrays:
        if array is None:
            hasher.update(b"None")
        else:
            data = np.asarray(array)
            hasher.update(str((data.shape, data.dtype.str)).encode())
            hasher.update(data.tobytes())
    hasher.update(digest(env.traffic.history_snapshot()).encode())
    return hasher.hexdigest()


def policy_seed(slot: int, namespace: str = "continuation", index: int = 0) -> int:
    raw = hashlib.sha256(f"E13-C:43:{namespace}:{slot}:{index}".encode()).digest()
    return int.from_bytes(raw[:8], "little") % (2**63 - 1)


class FrozenSampler:
    def __init__(self, actor: Any, config: Any) -> None:
        self.actor = actor
        self.tensorizer = ActorObservationTensorizer(config)
        self.distribution = CAGATMAPPOActionDistribution(actor, config)
        self.calls = 0

    def sample(self, env: U2UMECEnvironment, hidden: torch.Tensor,
               *, seed: int | None = None) -> tuple[tuple[Any, ...], torch.Tensor, dict[str, Any]]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(policy_seed(env.slot) if seed is None else seed)
        before = tensor_digest(generator.get_state())
        batch = self.tensorizer.encode_step(env.current_observations, device="cpu",
                                            dtype=torch.float32, episode_start=env.slot == 0)
        masks = SequentialActionMaskBatch.from_observations(env.current_observations)
        with torch.inference_mode():
            output = self.distribution.sample_actions(batch, masks, hidden.detach().clone(), generator=generator)
        self.calls += 1
        return tuple(output.proposals[0][0]), output.hidden_out.detach().clone(), {
            "seed": generator.initial_seed(), "before": before,
            "after": tensor_digest(generator.get_state()),
        }

    def complete_transmit(self, env: U2UMECEnvironment, hidden: torch.Tensor,
                          factual: tuple[Any, ...], source: int, pair_id: int) -> tuple[Any, int]:
        if factual[source].tx_select != "idle":
            return factual[source], 0
        # Rejection on upper T alone is an exact conditional policy sample.
        # Never reject a destination/resource/power for its execution outcome.
        for attempt in range(1, 4097):
            actions, _, _ = self.sample(env, hidden, seed=policy_seed(env.slot, f"completion:{pair_id}", attempt))
            proposal = actions[source]
            if proposal.tx_select != "idle":
                return replace(factual[source], tx_select=proposal.tx_select,
                               resource_group=proposal.resource_group,
                               resource_width=proposal.resource_width,
                               power_level=proposal.power_level), attempt
        raise ReliabilityError("old policy conditional Transmit rejection budget exhausted")


def make_arms(env: U2UMECEnvironment, factual: tuple[Any, ...], source: int,
              transmit: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    mask = env.current_observations[source].action_masks
    require(mask.tx_branch_active, "focal slot is not TX-active")
    idle = replace(factual[source], tx_select=mask.tx_idle_action, resource_group="idle",
                   resource_width=mask.canonical_width, power_level=mask.canonical_power)
    require(transmit.tx_select != mask.tx_idle_action, "T arm did not choose Transmit")
    arm_i, arm_t = list(factual), list(factual)
    arm_i[source], arm_t[source] = idle, transmit
    for proposals in (arm_i, arm_t):
        require(all(obs.action_masks.is_legal(proposal)
                    for obs, proposal in zip(env.current_observations, proposals)), "illegal arm action")
        for agent, proposal in enumerate(proposals):
            if agent != source:
                require(proposal == factual[agent], "other agent's initial action changed")
            else:
                require((proposal.route, proposal.cpu_queue, proposal.cpu_frequency) ==
                        (factual[agent].route, factual[agent].cpu_queue, factual[agent].cpu_frequency),
                        "focal Route/CPU initial action changed")
    return tuple(arm_i), tuple(arm_t)


def reward_check(result: Any) -> dict[str, float]:
    terms = result.info["reward"]
    components = {name: float(terms[name]) for name in COMPONENTS}
    reconstructed = math.fsum(sign * components[name] for sign, name in zip(SIGNS, COMPONENTS))
    require(math.isclose(result.reward, reconstructed, abs_tol=1e-10, rel_tol=1e-10), "reward component nonconservation")
    require(math.isclose(float(terms["reward"]), result.reward, abs_tol=1e-12), "reward telemetry mismatch")
    debits = result.info["energy"]["debits"]
    energy = math.fsum(float(row["transmit_energy_j"]) + float(row["cpu_energy_j"]) for row in debits)
    require(math.isclose(energy, float(terms["actual_energy_j"]), abs_tol=1e-10), "energy nonconservation")
    if "agent_reward" in result.info:
        require(math.isclose(math.fsum(result.info["agent_reward"]), result.reward, abs_tol=1e-9),
                "agent/team reward nonconservation")
    return components


class LifecycleLedger:
    """Full factual prefix plus arm suffix, with episode-local task-ID joins."""
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.keys: set[tuple[Any, ...]] = set()
        self.tasks: dict[int, dict[str, Any]] = {}

    def add(self, env: U2UMECEnvironment, slot: int, kind: str, task_id: int,
            **details: Any) -> None:
        require(task_id in env.lifecycle.tasks, f"missing task join: {task_id}")
        task = env.lifecycle.tasks[task_id]
        if "source" in details:
            require(details["source"] == task.source_uav, "task source join mismatch")
        if "destination" in details:
            require(details["destination"] == task.destination, "task destination join mismatch")
        key = (slot, kind, task_id, details.get("source"), details.get("destination"))
        require(key not in self.keys, f"duplicate lifecycle event: {key}")
        self.keys.add(key)
        self.events.append({"slot": slot, "kind": kind, "task_id": task_id, **details})

    def update(self, env: U2UMECEnvironment, result: Any) -> None:
        info, slot = result.info, result.info["slot"]
        for executed in info["executed"]:
            comm = executed["communication"]
            task_id = comm["head_task_id"]
            if task_id is not None and comm["receiver_uav"] is not None:
                self.add(env, slot, "tx_attempt", task_id,
                         source=comm["sender_uav"], destination=comm["receiver_uav"])
                if comm["accepted"]:
                    self.add(env, slot, "tx_accepted", task_id,
                             source=comm["sender_uav"], destination=comm["receiver_uav"])
        for link in info["service"]["links"]:
            for service in link["task_services"]:
                if service["amount"] > 0:
                    self.add(env, slot, "effective_tx", service["task_id"],
                             source=link["sender_uav"], destination=link["receiver_uav"], amount=service["amount"])
        for cpu in info["service"]["cpu"]:
            if cpu["task_id"] is not None and cpu["service_cycles"] > 0:
                self.add(env, slot, "cpu_service", cpu["task_id"], source=cpu["queue_source_uav"],
                         destination=cpu["executor_uav"], amount=cpu["service_cycles"])
        for task_id, task in env.lifecycle.tasks.items():
            previous = self.tasks.get(task_id)
            if task.cpu_entry_slot is not None and (previous is None or previous["cpu_entry_slot"] is None):
                if task.destination != task.source_uav:
                    self.add(env, task.cpu_entry_slot, "tx_complete", task_id,
                             source=task.source_uav, destination=task.destination)
            self.tasks[task_id] = task.snapshot()
        for task in info["service"]["settled_tasks"] + info["service"]["truncated_tasks"]:
            self.add(env, slot, task["outcome"], task["task_id"])

    def validate(self, env: U2UMECEnvironment) -> list[dict[str, Any]]:
        require(set(self.tasks) == set(env.lifecycle.tasks), "task ledger coverage missing")
        by_task: dict[int, list[dict[str, Any]]] = {task_id: [] for task_id in self.tasks}
        for event in self.events:
            by_task[event["task_id"]].append(event)
        joined = []
        for task_id, task in sorted(self.tasks.items()):
            events = by_task[task_id]
            terminal = [event for event in events if event["kind"] in ("done", "expired", "truncated")]
            require(len(terminal) == 1, "missing/duplicate terminal task join")
            require(terminal[0]["kind"] == task["outcome"], "terminal outcome mismatch")
            effective = [event["slot"] for event in events if event["kind"] == "effective_tx"]
            complete = [event["slot"] for event in events if event["kind"] == "tx_complete"]
            cpu = [event["slot"] for event in events if event["kind"] == "cpu_service"]
            require(len(complete) <= 1, "duplicate TX completion")
            if complete:
                require(bool(effective) and min(effective) <= complete[0], "TX complete without effective TX")
            if cpu and task["destination"] != task["source_uav"]:
                require(bool(complete) and complete[0] < min(cpu), "remote CPU before TX complete")
            joined.append({**task, "first_effective_tx": min(effective) if effective else None,
                           "tx_complete": complete[0] if complete else None,
                           "cpu_start": min(cpu) if cpu else None,
                           "tx_attempt_count": sum(e["kind"] == "tx_attempt" for e in events),
                           "tx_accepted_count": sum(e["kind"] == "tx_accepted" for e in events)})
        return joined


def candidates(env: U2UMECEnvironment, actions: tuple[Any, ...]) -> list[dict[str, Any]]:
    rows = []
    for source, observation in enumerate(env.current_observations):
        masks = observation.action_masks
        if not masks.tx_branch_active:
            continue
        destinations = [value for value, valid in zip(masks.tx_select_domain, masks.tx_select_mask)
                        if valid and value != masks.tx_idle_action]
        queues = [env.lifecycle.queues.tx.get((source, destination)) for destination in destinations]
        heads = [queue.peek() for queue in queues if queue is not None and len(queue)]
        require(bool(heads), "active TX without queued head")
        count = sum(len(queue) for queue in queues if queue is not None)
        slack = min(task.deadline_slack(env.slot) for task in heads)
        rows.append({"slot": env.slot, "source": source, "queue_count": count,
                     "slack": slack, "queue_bin": "1" if count == 1 else "2+",
                     "slack_bin": "<=15" if slack <= 15 else "16-40" if slack <= 40 else ">40",
                     "destinations": destinations, "head_ids": [task.task_id for task in heads],
                     "factual_upper": "I" if actions[source].tx_select == "idle" else "T"})
    return rows


def select_pairs(rows: list[dict[str, Any]], sources: int) -> list[dict[str, Any]]:
    require(sources == 4, "pilot source count must remain four")
    selected = []
    # Coverage-only selection, with no shadow return or oracle outcome access.
    for source in range(sources):
        pool = [row for row in rows if row["source"] == source]
        require(len(pool) >= 6, f"fewer than six legal decisions for source {source}")
        seen: set[tuple[str, Any]] = set()
        slots: list[int] = []
        for _ in range(6):
            def priority(row: dict[str, Any]) -> tuple[Any, ...]:
                novelty = sum((key, row[key]) not in seen for key in ("queue_bin", "slack_bin", "factual_execution"))
                separation = min((abs(row["slot"] - slot) for slot in slots), default=0)
                return novelty, separation, -row["slot"]
            best = max(pool, key=priority)
            pool.remove(best)
            selected.append(dict(best))
            seen.update((key, best[key]) for key in ("queue_bin", "slack_bin", "factual_execution"))
            slots.append(best["slot"])
    selected.sort(key=lambda row: (row["slot"], row["source"]))
    for index, row in enumerate(selected):
        row["pair_id"] = index
    return selected


def signature(env: U2UMECEnvironment, result: Any, hidden: torch.Tensor, rng: dict[str, Any]) -> dict[str, Any]:
    return {"info": digest(result.info), "hidden": tensor_digest(hidden),
            "exogenous": exogenous_digest(env), "environment_rng": env.rng_fingerprint(), "policy_rng": rng}


def run_pilot() -> dict[str, Any]:
    started = time.perf_counter()
    outdir = ROOT / "logs/diagnostics/e13-c-paired-shadow-seed43"
    require(not outdir.exists(), "pilot output exists; do not overwrite or silently rerun")
    report_path = ROOT / "E13_PAIRED_SHADOW_RELIABILITY_PILOT_REPORT.md"
    require(not report_path.exists(), "report exists; read and review before replacing")
    outdir.mkdir(parents=True)
    input_paths = [SOURCE / "config_snapshot.yaml", SOURCE / "checkpoints/final.pt"]
    input_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
    config = load_run_config(input_paths[0])
    raw_config = json.loads(input_paths[0].read_text(encoding="utf-8"))
    object.__setattr__(config, "git_commit", raw_config["_metadata"]["git_commit"])
    require(config.seed == 43 and config.environment.episode_horizon == 500, "source seed/horizon mismatch")
    require(config.training.mappo.gamma == GAMMA, "original gamma differs from 0.99")
    # Actor construction consumes global RNG. Restore it immediately; all rollout
    # draws below use owned generators rather than process-global random state.
    global_torch = torch.get_rng_state().clone()
    loaded = load_final_actor_for_oracle_diagnostic(config, input_paths[1], execution_device="cpu",
                expected_checkpoint_sha256=CHECKPOINT_SHA, expected_actor_digest=ACTOR_SHA)
    torch.set_rng_state(global_torch)
    actor = loaded.actor
    sampler = FrozenSampler(actor, config)
    global_before = (tensor_digest(torch.get_rng_state()), digest(np.random.get_state()[1].tolist()), repr(random.getstate()))
    actor_before = actor_state_digest(actor)
    traces, all_candidates = [], []
    env = U2UMECEnvironment(config)
    env.reset()
    hidden = actor.initial_hidden(1)
    initial_state = env.state_fingerprint()
    ledger = LifecycleLedger()
    for slot in range(500):
        proposals, hidden_out, rng = sampler.sample(env, hidden)
        choices = candidates(env, proposals)
        result = env.step(proposals)
        reward_check(result)
        ledger.update(env, result)
        traces.append(signature(env, result, hidden_out, rng))
        for row in choices:
            comm = result.info["executed"][row["source"]]["communication"]
            link = next((link for link in result.info["service"]["links"] if link["sender_uav"] == row["source"]), None)
            row["factual_execution"] = ("Idle" if row["factual_upper"] == "I" else
                "effective" if link and link["service_bits"] > 0 else "accepted-zero" if comm["accepted"] else "rejected")
            all_candidates.append(row)
        hidden = hidden_out
        if slot % 100 == 99:
            print(f"factual carrier {slot + 1}/500", flush=True)
    ledger.validate(env)
    selected = select_pairs(all_candidates, config.environment.uav_count)
    predicted_budget = sum(2 * (500 - row["slot"]) for row in selected)
    require(predicted_budget <= MAX_SHADOW_STEPS, "selected shadows exceed budget")
    (outdir / "selection.json").write_text(json.dumps({"available_candidates": len(all_candidates),
         "available_strata": {str(s): {key: sorted({r[key] for r in all_candidates if r["source"] == s})
             for key in ("queue_bin", "slack_bin", "factual_execution")} for s in range(4)},
         "selected": selected}, indent=2), encoding="utf-8")
    (outdir / "factual_trace.jsonl").write_text("".join(json.dumps({"slot": i, **row}) + "\n" for i, row in enumerate(traces)), encoding="utf-8")
    # Second factual pass proves identity and retains only selected in-memory
    # clones. These are not a persisted state bank.
    carrier = U2UMECEnvironment(config)
    carrier.reset()
    require(carrier.state_fingerprint() == initial_state, "reset replay initial state mismatch")
    hidden = actor.initial_hidden(1)
    ledger = LifecycleLedger()
    snapshots: dict[int, tuple[Any, ...]] = {}
    for slot in range(500):
        proposals, hidden_out, rng = sampler.sample(carrier, hidden)
        for row in selected:
            if row["slot"] == slot:
                snapshots[row["pair_id"]] = (carrier.clone_for_shadow(), hidden.clone(), hidden_out.clone(),
                                              proposals, copy.deepcopy(ledger), rng)
        result = carrier.step(proposals)
        reward_check(result)
        ledger.update(carrier, result)
        require(signature(carrier, result, hidden_out, rng) == traces[slot], f"factual replay mismatch at {slot}")
        hidden = hidden_out
        if slot % 100 == 99:
            print(f"factual identity {slot + 1}/500", flush=True)
    real_before = (carrier.state_fingerprint(), carrier.rng_fingerprint(), tensor_digest(hidden))
    pairs, shadow_steps, step_checks = [], 0, 0
    with (outdir / "shadow_trace.jsonl").open("w", encoding="utf-8") as trace_file, \
         (outdir / "lifecycle_events.jsonl").open("w", encoding="utf-8") as event_file, \
         (outdir / "task_join.jsonl").open("w", encoding="utf-8") as join_file:
        for row in selected:
            pair_id, slot_start, source = row["pair_id"], row["slot"], row["source"]
            base, hidden_in, hidden_out, factual, prefix, initial_rng = snapshots.pop(pair_id)
            base_before = (base.state_fingerprint(), base.rng_fingerprint(), tensor_digest(hidden_in))
            transmit, attempts = sampler.complete_transmit(base, hidden_in, factual, source, pair_id)
            initial_i, initial_t = make_arms(base, factual, source, transmit)
            arms = {"I": base.clone_for_shadow(), "T": base.clone_for_shadow()}
            require(arms["I"].state_fingerprint() == arms["T"].state_fingerprint() == base_before[0], "initial shadow state mismatch")
            require(arms["I"].lifecycle is not arms["T"].lifecycle, "shared arm lifecycle")
            arm_hidden = {name: hidden_out.clone() for name in arms}
            require(arm_hidden["I"].data_ptr() != arm_hidden["T"].data_ptr(), "shared hidden storage")
            arm_ledgers = {name: copy.deepcopy(prefix) for name in arms}
            reward_lists = {name: [] for name in arms}
            component_lists = {name: {key: [] for key in COMPONENTS} for name in arms}
            initial_execution: dict[str, Any] = {}
            matched = row["factual_upper"]
            for slot in range(slot_start, 500):
                require(exogenous_digest(arms["I"]) == exogenous_digest(arms["T"]), "prestep exogenous mismatch")
                rngs = {}
                for name, shadow in arms.items():
                    if slot == slot_start:
                        actions = initial_i if name == "I" else initial_t
                        rng = dict(initial_rng)
                        next_hidden = arm_hidden[name]
                    else:
                        actions, next_hidden, rng = sampler.sample(shadow, arm_hidden[name])
                    require(all(obs.action_masks.is_legal(action) for obs, action in zip(shadow.current_observations, actions)), "illegal continuation action")
                    require(shadow_steps < MAX_SHADOW_STEPS, "hard shadow step budget exhausted")
                    result = shadow.step(actions)
                    shadow_steps += 1
                    components = reward_check(result)
                    arm_ledgers[name].update(shadow, result)
                    discount = GAMMA ** (slot - slot_start)
                    reward_lists[name].append(discount * result.reward)
                    for key in COMPONENTS:
                        component_lists[name][key].append(discount * components[key])
                    sig = signature(shadow, result, next_hidden, rng)
                    if name == matched:
                        require(sig == traces[slot], f"factual matching arm mismatch pair {pair_id} slot {slot}")
                    trace_file.write(json.dumps({"pair_id": pair_id, "arm": name, "slot": slot,
                        "shadow_id": f"pair-{pair_id:02d}-{name}", "reward": result.reward,
                        "reward_components": components, "discount": discount, **sig}) + "\n")
                    if slot == slot_start:
                        initial_execution[name] = result.info["executed"][source]["communication"]
                    arm_hidden[name] = next_hidden
                    rngs[name] = rng
                require(rngs["I"] == rngs["T"], "policy RNG draw consumption differs across arms")
                require(exogenous_digest(arms["I"]) == exogenous_digest(arms["T"]), "poststep exogenous mismatch")
                step_checks += 1
            returns = {name: math.fsum(values) for name, values in reward_lists.items()}
            component_totals = {name: {key: math.fsum(values) for key, values in totals.items()}
                                for name, totals in component_lists.items()}
            delta = returns["T"] - returns["I"]
            delta_components = {key: sign * (component_totals["T"][key] - component_totals["I"][key])
                                for key, sign in zip(COMPONENTS, SIGNS)}
            require(math.isclose(delta, math.fsum(delta_components.values()), abs_tol=1e-9), "discounted delta nonconservation")
            lifecycle_summary = {}
            for name, shadow in arms.items():
                require(shadow.done and shadow.slot == 500, "shadow did not reach original endpoint")
                joined = arm_ledgers[name].validate(shadow)
                for task in joined:
                    join_file.write(json.dumps({"pair_id": pair_id, "arm": name, **task}) + "\n")
                for event in arm_ledgers[name].events:
                    event_file.write(json.dumps({"pair_id": pair_id, "arm": name,
                        "scope": "prefix" if event["slot"] < slot_start else "shadow", **event}) + "\n")
                lifecycle_summary[name] = {
                    "tasks": len(joined), "outcomes": dict(Counter(task["outcome"] for task in joined)),
                    "suffix_events": dict(Counter(e["kind"] for e in arm_ledgers[name].events if e["slot"] >= slot_start)),
                }
            require(intrinsic_tasks(arms["I"]) == intrinsic_tasks(arms["T"]), "exogenous task identities differ")
            require((base.state_fingerprint(), base.rng_fingerprint(), tensor_digest(hidden_in)) == base_before, "shadow mutated slot-start source")
            require((carrier.state_fingerprint(), carrier.rng_fingerprint(), tensor_digest(hidden)) == real_before, "shadow mutated factual environment/RNG/hidden")
            require(actor_state_digest(actor) == actor_before, "shadow mutated actor")
            pairs.append({**row, "valid": True, "G_I": returns["I"], "G_T": returns["T"], "d": delta,
                "components": component_totals, "delta_components": delta_components,
                "conditional_rejection_draws": attempts, "initial_I": [asdict(a) for a in initial_i],
                "initial_T": [asdict(a) for a in initial_t], "initial_execution": initial_execution,
                "lifecycle": lifecycle_summary, "steps": 2 * (500 - slot_start),
                "initial_state_digest": base_before[0], "initial_rng_digest": base_before[1],
                "factual_matching_arm": matched})
            print(f"pair {pair_id + 1}/24 valid: slot={slot_start} source={source} d={delta:.9g} steps={shadow_steps}", flush=True)
    require(shadow_steps == predicted_budget, "shadow budget accounting mismatch")
    require(actor_state_digest(actor) == actor_before and not actor.training and
            all(not p.requires_grad for p in actor.parameters()), "frozen actor invariant failed")
    global_after = (tensor_digest(torch.get_rng_state()), digest(np.random.get_state()[1].tolist()), repr(random.getstate()))
    require(global_before == global_after, "global RNG changed during pilot")
    require(all(hashlib.sha256(path.read_bytes()).hexdigest() == input_hashes[str(path)] for path in input_paths), "source input modified")
    deltas = [pair["d"] for pair in pairs]
    near_zero_count = sum(abs(value) <= NEAR_ZERO for value in deltas)
    summary = {"pair_count": len(pairs), "valid_pairs": sum(pair["valid"] for pair in pairs),
        "near_zero_threshold": NEAR_ZERO, "near_zero_count": near_zero_count,
        "positive": sum(value > NEAR_ZERO for value in deltas), "negative": sum(value < -NEAR_ZERO for value in deltas),
        "mean_d": statistics.mean(deltas), "median_d": statistics.median(deltas),
        "mean_delta_components": {key: statistics.mean(pair["delta_components"][key] for pair in pairs) for key in COMPONENTS},
        "shadow_steps": shadow_steps, "carrier_steps": 1000, "total_environment_steps": shadow_steps + 1000,
        "aligned_step_checks": step_checks, "factual_replay_steps": 500,
        "matching_arm_identity_steps": step_checks, "actor_calls": sampler.calls,
        "wall_seconds": time.perf_counter() - started, "checkpoint_sha256": CHECKPOINT_SHA,
        "actor_digest": actor_before, "input_hashes": input_hashes,
        "lifecycle_join": "PASS", "global_rng_invariance": "PASS", "real_state_invariance": "PASS",
        "ppo_go": len(pairs) == PAIR_COUNT and near_zero_count / PAIR_COUNT <= 0.9}
    (outdir / "pairs.json").write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_path, summary, pairs, outdir)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def write_report(path: Path, summary: dict[str, Any], pairs: list[dict[str, Any]], outdir: Path) -> None:
    lines = ["# E13-C paired action-complete shadow reliability pilot", "",
        "本轮只实施、测试、运行可靠性 pilot 和报告。未接入 PPO，未训练、resume、formal evaluation 或运行 seed42/44。",
        "未改 reward/environment/action semantics、lambda/LR/entropy，也未新增 Q critic。", "",
        "## 冻结协议与证据", "",
        f"- Source：`{SOURCE}`；final checkpoint SHA256 `{CHECKPOINT_SHA}`；actor digest `{ACTOR_SHA}`。",
        "- 环境使用原 E12 配置、seed43、500-slot horizon；原 team reward，gamma=0.99，不 bootstrap，不新增 event bonus。",
        "- 两遍独立 factual carrier，各 500 步；第一遍记录候选及实际执行分层，第二遍逐步严格重放并仅保留选中状态的内存克隆。不是历史训练轨迹重放，也不是正式评估。",
        "- 每个 source 选 6 个合法 TX-active 决策，优先覆盖 queue、slack、factual execution 分层，再扩大时间间隔。选择前不访问 shadow d；不按最优执行结果筛选。候选覆盖详见 selection.json。",
        "- 初始其他 agent 全部动作与 focal Route/CPU 固定。I arm 使用原合法 Idle 子树；T arm 使用 old policy 条件 Transmit 子树。若 factual 已 T，直接复用；否则仅拒绝 upper=Idle 的独立 policy 抽样，接受首个 T，绝不按 destination/resource/power 的执行收益拒绝。",
        "- 后续各 arm 使用自己的 observation 与 hidden 执行同一个冻结随机 old policy；不复用已经分叉后的 factual future actions。",
        "- 每 slot 的 policy generator 由 seed43、绝对 slot 与 namespace 定址，两臂使用独立 generator 对象但相同 seed。逐步核验前后 RNG digest 与抽样消耗相同。条件动作补全使用独立 namespace。",
        "- 环境五个外生 RNG 状态逐步一致，并核验实际 mobility/topology/channel/CSI/arrival task identities；动作相关的 interference/public message 不强制相同。故本次 d 不能由外生流错位解释，但单次共同随机样本仍有统计方差。", "",
        "## 可靠性验收", "",
        "|检查|结果|", "|---|---|",
        f"|24 pairs 原始 slot-start 完整状态一致、对象隔离、合法初始及后续动作|{summary['valid_pairs']}/24 PASS|",
        "|factual 两遍逐步 info/hidden/RNG/外生状态重放|500/500 PASS|",
        f"|每对 factual matching arm 全 suffix replay|{summary['matching_arm_identity_steps']} steps PASS|",
        f"|common randomness 前后及 policy RNG draw consumption|{summary['aligned_step_checks']} paired slots PASS|",
        "|原 team reward 分量、energy debit、agent reward、discounted d 守恒|PASS，绝对容差 1e-9 或更严|",
        "|task ID/source/destination/event 去重、完整 prefix+suffix、唯一 terminal outcome|PASS|",
        "|真实 carrier、slot-start clone、hidden、actor、全局 RNG 不变；source 文件 hash 不变|PASS|", "",
        "## 24 对原 team return", "",
        "|pair|slot|source (0-based)|queue|slack|factual result|G_I|G_T|d|steps|",
        "|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for pair in pairs:
        lines.append(f"|{pair['pair_id']}|{pair['slot']}|{pair['source']}|{pair['queue_count']}|{pair['slack']}|{pair['factual_execution']}|{pair['G_I']:.10g}|{pair['G_T']:.10g}|{pair['d']:.10g}|{pair['steps']}|")
    lines.extend(["", f"近零定义：`|d| <= {NEAR_ZERO}`（原 team discounted return 单位）；比例 {summary['near_zero_count']}/24 = {summary['near_zero_count']/24:.2%}。",
        f"d > epsilon：{summary['positive']}；d < -epsilon：{summary['negative']}；mean {summary['mean_d']:.12g}；median {summary['median_d']:.12g}。", "",
        "## 分量与 lifecycle", "", "差值采用 reward 中的符号：completion 为正，expiration/workload/energy penalties 为负。", "",
        "|分量|mean signed (T-I)|", "|---|---:|"])
    lines.extend(f"|{key}|{value:.12g}|" for key, value in summary["mean_delta_components"].items())
    lines.extend(["", "事件表记录 TX attempt/accepted/effective service、TX complete、CPU service、DONE/EXPIRED/TRUNCATED；task_join.jsonl 给出每个 task 的 first effective TX、TX complete、CPU start 和唯一 outcome。缺少有效事件以 null 表示实际未发生，不制造事件。task ID 只在本 seed43 episode 内使用，pair+arm 为 shadow identity。",
        "完整 factual prefix 在每臂 ledger 中保留并标记 prefix；reward G 只包含分叉起点之后的 suffix，避免把历史事件或历史 reward 加入 d。", "",
        "## 成本与接入判定", "",
        f"- Shadow environment steps：{summary['shadow_steps']}/24000；carrier/replay：1000；总 environment steps：{summary['total_environment_steps']}。",
        f"- Actor forward calls（含条件拒绝抽样）：{summary['actor_calls']}；wall time：{summary['wall_seconds']:.3f} s（不含单元测试）。",
        "- 预先设置的可靠性 GO 门：24/24 合法完整 pairs、七项不变量全通过、近零比例不高于 90%。",
        f"- 接入 PPO 的可靠性门：**{'GO' if summary['ppo_go'] else 'NO-GO'}**。此结论只允许进一步审查/实施 label 接口；本轮没有接入 PPO，更没有训练授权。",
        "- 24 对属于有目的覆盖抽样，同一个 episode 的 pairs 有相关性；单个共同随机 draw 不是 E13 效果、credit 无偏性或训练稳定性的证明。若 d 主要为能耗负值或未产生完成对比，须明确其不能证明能恢复 completion。", "",
        "## 原始产物", "", f"`{outdir}`：selection.json、factual_trace.jsonl、shadow_trace.jsonl、pairs.json、summary.json、lifecycle_events.jsonl、task_join.jsonl。",
        "实现：src/evaluation/tx_paired_shadow_pilot.py；测试：tests/test_tx_paired_shadow_pilot.py。单元测试结果另见 tests.txt。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    torch.set_num_threads(1)
    run_pilot()
