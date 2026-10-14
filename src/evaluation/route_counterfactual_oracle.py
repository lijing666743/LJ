"""One-shot paired Route interventions under a frozen recurrent actor.

This evaluation-only tool never constructs a trainer or an optimizer. Branches
use the environment's verified complete clone, intervene on one Route field
once, and continue the SAME checkpoint to the horizon. It is not a globally
optimal scheduler, nor a source of training labels.
"""
from __future__ import annotations

import argparse
import copy
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import cloudpickle
import numpy as np
import torch

from ..config import load_run_config
from ..env.environment import U2UMECEnvironment
from .route_oracle import (
    FrozenOraclePolicyRunnerCache, OracleReliabilityError, actor_state_digest,
    tensor_digest, _energy_from_step,
)
from .runner import FormalEvaluationRunner, _canonical_sha256
from .service_telemetry import EvaluationServiceTelemetryCollector, snapshot_tasks


PROTOCOL = {
    "schema": "paired_route_one_shot.v1",
    "seed": 1044, "device": "cpu", "dtype": "float32",
    "continuation": "same checkpoint masked deterministic actor; all branches to episode end",
    "intervention": "one source UAV Route field at one binding slot; all other fields unchanged",
    "latency_tie_slots": 1,
    "energy_tie_abs_j": 0.001, "energy_tie_relative": 0.01,
    "energy_tie_rule": "abs(E1-E2) <= max(0.001 J, 0.01*max(E1,E2))",
    "primary": "completion, then latency outside 1-slot tie, then energy outside tolerance",
    "all_fail": "F; no best option claimed",
    "strict_completion": "both DONE => T; neither DONE => F; otherwise L/R",
    "latency_only": "completion, then latency outside 1-slot tie",
    "energy_only": "completion, then energy outside tolerance (ignore latency)",
    "system": "episode DONE max, EXPIRED min, active CPU+TX energy min outside tolerance",
    "factual_reuse": "recorded factual suffix reused for factual candidate after a separate full no-change clone replay gate",
    "decision_thresholds": {"local_rare": 0.05, "local_material": 0.20},
    "estimators": "full-frequency backlog/capacity proxy; stale CSI noise-limited all-RU max-power rate; not actor input or actual future service",
    "four_k": "20 factual Local binding states; same 4k frozen continuation; distinct from final-policy cohort",
}


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def intervene(observations, proposals, source, route):
    """Replace exactly one Route value; reject illegal complete proposals."""
    result = list(proposals)
    result[source] = replace(result[source], route=route)
    if not all(obs.action_masks.is_legal(action) for obs, action in zip(observations, result)):
        raise OracleReliabilityError("illegal intervention")
    for index, (old, new) in enumerate(zip(proposals, result)):
        if index != source and old != new:
            raise OracleReliabilityError("non-target proposal changed")
        if index == source and replace(new, route=old.route) != old:
            raise OracleReliabilityError("non-Route field changed")
    return tuple(result)


def energy_tie(a, b):
    return abs(a-b) <= max(PROTOCOL["energy_tie_abs_j"], PROTOCOL["energy_tie_relative"]*max(a, b))


def compare(a, b, mode="primary"):
    """-1 means a better; +1 b better; 0 tie. No unobserved Q labels."""
    da, db = a["final_status"] == "DONE", b["final_status"] == "DONE"
    if da != db:
        return -1 if da else 1
    if not da or mode == "strict_completion":
        return 0
    delta = a["latency_slots"] - b["latency_slots"]
    if mode in {"primary", "latency_only"} and abs(delta) > PROTOCOL["latency_tie_slots"]:
        return -1 if delta < 0 else 1
    if mode in {"primary", "energy_only"} and not energy_tie(a["task_service_energy_j"], b["task_service_energy_j"]):
        return -1 if a["task_service_energy_j"] < b["task_service_energy_j"] else 1
    return 0


def best_set(rows, mode="primary"):
    # Anchor to the minimum latency/energy, avoiding non-transitive pairwise
    # tolerance chains. Select near-optimal set against that common anchor.
    done = [x for x in rows if x["final_status"] == "DONE"]
    if not done:
        return []
    if mode in {"primary", "latency_only"}:
        fastest = min(x["latency_slots"] for x in done)
        done = [x for x in done if x["latency_slots"] <= fastest+PROTOCOL["latency_tie_slots"]]
    if mode in {"primary", "energy_only"}:
        minimum = min(x["task_service_energy_j"] for x in done)
        done = [x for x in done if energy_tie(x["task_service_energy_j"], minimum)]
    return done


def classify(rows, mode="primary"):
    local = next(x for x in rows if x["route"] == "local")
    remote = [x for x in rows if isinstance(x["route"], int)]
    choices = best_set(remote, mode)
    if not choices:
        return "L" if local["final_status"] == "DONE" else "F"
    anchor = min(choices, key=lambda x: (x["latency_slots"], x["task_service_energy_j"]))
    order = compare(local, anchor, mode)
    return "L" if order < 0 else "R" if order > 0 else "T"


def summary(collector, prefix_cpu=0., prefix_tx=0.):
    rows = collector.task_records()
    counts = Counter(x["final_status"] for x in rows)
    return {
        "generated": len(rows), **{k: counts[k] for k in ("DONE", "EXPIRED", "TRUNCATED")},
        "cpu_energy_j": prefix_cpu+sum(x["cpu_energy_j"] for x in collector.cpu_slots),
        "tx_energy_j": prefix_tx+sum(x["tx_energy_j"] for x in collector.tx_slots if x["link_summary"]),
        "expiry_stages": dict(Counter(x["expire_stage"] for x in rows if x["final_status"] == "EXPIRED")),
        "suffix_cpu_nonempty_idle": sum(x["cpu_queue_length"] > 0 and x["served_cycles"] <= 0 for x in collector.cpu_slots),
        "suffix_cpu_queue_task_slots": sum(x["cpu_queue_length"] for x in collector.cpu_slots),
        "suffix_cpu_queue_cycle_slots": sum(x["cpu_queue_cycles"] for x in collector.cpu_slots),
    }


def target_result(collector, task_id, route, cpu_energy, tx_energy, tx_attribution):
    task = dict(collector.tasks[task_id])
    done = task["done_slot"]
    terminal = done if done is not None else task["expired_slot"]
    if terminal is None:
        terminal = 499
    tx_first, tx_end = task["tx_first_service_slot"], task["tx_complete_slot"]
    cpu_first = task["cpu_first_service_slot"]
    entry = task["route_bind_slot"]+1 if route == "local" else (None if tx_end is None else tx_end+1)
    return {
        **task, "route": route, "terminal_slot": terminal,
        "latency_slots": None if done is None else done-task["generation_slot"],
        "deadline_margin_slots": None if done is None else task["deadline_slot"]-done,
        "route_to_tx_first_slots": None if tx_first is None else tx_first-task["route_bind_slot"],
        "tx_service_elapsed_slots": None if tx_end is None or tx_first is None else tx_end-tx_first+1,
        "cpu_wait_slots": None if entry is None or cpu_first is None else cpu_first-entry,
        "cpu_service_elapsed_slots": None if cpu_first is None else terminal-cpu_first+1,
        "task_cpu_energy_j": cpu_energy, "task_tx_energy_j": tx_energy,
        "task_service_energy_j": cpu_energy+tx_energy, "task_tx_energy_attribution": tx_attribution,
    }


def state_features(env, source, task_id, probabilities):
    task = env.lifecycle.tasks[task_id]
    resources = env.resource_states
    dt = env.config.environment.slot_duration_s
    active = [t for t in env.lifecycle.tasks.values() if t.outcome.value == "none"]
    queues = {}
    for uav, resource in resources.items():
        cpu = [t for t in active if t.status.value in {"local", "cpu"} and t.destination == uav]
        cycles = sum(t.remaining_cycles for t in cpu)
        queues[str(uav)] = {
            "cpu_queue_length": len(cpu), "cpu_queue_cycles": cycles,
            "residual_energy_j": resource.residual_energy_j,
            "cpu_capacity_hz": resource.max_cpu_frequency_hz,
            "queue_full_frequency_slots_proxy": cycles/resource.max_cpu_frequency_hz/dt,
        }
    remotes = {}
    observation = env.current_observations[source]
    masks = observation.action_masks
    for route, legal in zip(masks.route_domain, masks.route_mask):
        if not legal or not isinstance(route, int):
            continue
        csi = env.channel_features.stale_csi[source, route]
        valid = env.channel_features.csi_valid_mask[source, route]
        gain = np.abs(csi)**2
        bandwidth = env.config.environment.ru_bandwidth_hz
        noise = 10**((env.config.environment.noise_psd_dbm_hz-30+env.config.environment.receiver_noise_figure_db)/10)*bandwidth
        rate = float(np.sum(bandwidth*np.log2(1+resources[source].max_transmit_power_w/env.config.environment.ru_count*gain/noise))) if np.all(valid) else None
        tx_slots = None if not rate or rate <= 0 else task.data_bits/rate/dt
        remote_compute = task.remaining_cycles/resources[route].max_cpu_frequency_hz/dt
        remotes[str(route)] = {
            **queues[str(route)], "stale_gain_mean": float(np.mean(gain)),
            "stale_csi_valid": bool(np.all(valid)),
            "stale_csi_real": csi.real.tolist(), "stale_csi_imag": csi.imag.tolist(),
            "noise_limited_rate_bps_proxy": rate, "tx_time_slots_proxy": tx_slots,
            "compute_slots_proxy": remote_compute,
            "tx_queue_compute_slots_proxy": None if tx_slots is None else tx_slots+remote_compute+queues[str(route)]["queue_full_frequency_slots_proxy"],
            "distance_m": float(env.topology.distances_m[source, route]),
        }
    return {
        "task_id": task_id, "source_uav": source, "generation_slot": task.arrival_slot,
        "route_slot": env.slot, "deadline_slot": task.deadline_slot,
        "slack_slots": task.deadline_slot-env.slot,
        "required_cycles": task.cpu_cycles, "input_bits": task.data_bits,
        "local_compute_slots_proxy": task.remaining_cycles/resources[source].max_cpu_frequency_hz/dt,
        "local_queue_compute_slots_proxy": (task.remaining_cycles+queues[str(source)]["cpu_queue_cycles"])/resources[source].max_cpu_frequency_hz/dt,
        "source_resources": queues[str(source)], "uavs": queues, "remotes": remotes,
        "route_domain": list(masks.route_domain), "route_mask": list(map(bool, masks.route_mask)),
        "actor_route_probabilities": probabilities,
        "actor_observation_snapshot": observation.snapshot(),
    }


def external_digest(evaluator, env, info):
    record = evaluator._external_trace_record(env, phase=f"slot_{info['slot']}_complete", arrival=info["arrival"])
    return _canonical_sha256(record), env.rng_fingerprint(), record


def observe(collector, env, proposals):
    before = snapshot_tasks(env.lifecycle.tasks.values())
    result = env.step(proposals)
    collector.observe_step(before=before, after=snapshot_tasks(env.lifecycle.tasks.values()), step_info=result.info)
    return result


def collect_factual(evaluator, cohort, root):
    config = replace(evaluator.config, seed=1044)
    env = U2UMECEnvironment(config)
    observations = env.reset().observations
    actor = evaluator.loaded_actor.actor
    hidden = actor.initial_hidden(1, device="cpu", dtype=torch.float32)
    collector = EvaluationServiceTelemetryCollector(evaluation_run_id=cohort, method_id="ca_gat_mappo", evaluation_seed=1044, episode_id=0)
    decisions, exogenous, rngs, actions, states = [], {}, {}, [], []
    external_records = [evaluator._external_trace_record(env, phase="reset", arrival=None)]
    while True:
        proposals, next_hidden, output = evaluator._actor_actions(observations, hidden)
        actions.append(proposals)
        for source, proposal in enumerate(proposals):
            if proposal.route in {"idle", "defer"} or (cohort == "4k_local" and proposal.route != "local"):
                continue
            task_id = observations[source].private_queues.unbound.head_task_id
            if task_id is None:
                raise OracleReliabilityError("binding without task")
            state = env.state_fingerprint()
            cloned = env.clone_for_shadow()
            decisions.append({
                "env": cloned, "proposals": proposals, "hidden_out": next_hidden.detach().clone(),
                "source": source, "task_id": task_id, "slot": env.slot,
                "state_digest": state, "rng_digest": env.rng_fingerprint(),
                "hidden_in_digest": tensor_digest(hidden), "hidden_out_digest": tensor_digest(next_hidden),
                "prefix_tasks": copy.deepcopy(collector.tasks),
                "prefix_cpu_energy": sum(x["cpu_energy_j"] for x in collector.cpu_slots),
                "prefix_tx_energy": sum(x["tx_energy_j"] for x in collector.tx_slots if x["link_summary"]),
                "features": state_features(env, source, task_id, output.probabilities["route"][0, 0, source].detach().cpu().tolist()),
            })
            if env.state_fingerprint() != state:
                raise OracleReliabilityError("clone/feature observation mutated factual state")
        result = observe(collector, env, proposals)
        digest, rng, external = external_digest(evaluator, env, result.info)
        exogenous[int(result.info["slot"])], rngs[int(result.info["slot"])] = digest, rng
        external_records.append(external)
        hidden = next_hidden
        if result.truncated or result.terminated:
            break
        observations = result.observations
    env.assert_invariants()
    # Validate exact task facts/service times against the existing evaluation.
    prior_root = Path(evaluator.loaded_actor.source_checkpoint_path).parents[2] / ("eval1044_8k" if cohort == "8k" else "eval1044_4k")
    paths = list(prior_root.rglob("task_lifecycle.jsonl"))
    if len(paths) != 1:
        raise OracleReliabilityError(f"expected one existing task trace: {prior_root}")
    prior = [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines()]
    prior = {x["task_id"]: x for x in prior if x["method_id"] == "ca_gat_mappo"}
    ignored = {"evaluation_run_id"}
    for record in collector.task_records():
        if {k:v for k,v in record.items() if k not in ignored} != {k:v for k,v in prior[record["task_id"]].items() if k not in ignored}:
            raise OracleReliabilityError(f"factual task trace mismatch {record['task_id']}")
    counts = summary(collector)
    expected = (94, 19, 4, 117) if cohort == "8k" else (62, 53, 2, 20)
    observed = (counts["DONE"], counts["EXPIRED"], counts["TRUNCATED"], len(decisions))
    if observed != expected:
        raise OracleReliabilityError(f"factual baseline mismatch {observed} != {expected}")
    write_json(root/f"{cohort}_factual.json", {"summary": counts, "external_trace_sha256": _canonical_sha256(tuple(external_records)), "task_trace_matches_existing": True})
    write_jsonl(root/f"{cohort}_factual_tasks.jsonl", collector.task_records())
    write_jsonl(root/f"{cohort}_decision_features.jsonl", [x["features"] for x in decisions])
    print(json.dumps({"cohort": cohort, "factual_gate": "PASS", "counts": counts, "decisions": len(decisions)}), flush=True)
    return decisions, collector, exogenous, rngs, actions


def rollout_branch(evaluator, frozen, decision, route, factual, exogenous, rngs, expected_actions=None):
    root = decision["env"]
    state = root.state_fingerprint()
    shadow = root.clone_for_shadow()
    if shadow.state_fingerprint() != decision["state_digest"] or shadow.rng_fingerprint() != decision["rng_digest"]:
        raise OracleReliabilityError("branch initial state/RNG mismatch")
    hidden = decision["hidden_out"].detach().clone()
    proposals = intervene(shadow.current_observations, decision["proposals"], decision["source"], route)
    collector = EvaluationServiceTelemetryCollector(evaluation_run_id="oracle", method_id="ca_gat_mappo", evaluation_seed=1044, episode_id=0)
    collector.tasks = copy.deepcopy(decision["prefix_tasks"])
    cpu_energy = tx_energy = 0.
    attribution = "NOT_AVAILABLE"
    trace = []
    with torch.no_grad():
        while True:
            if expected_actions is not None and tuple(proposals) != tuple(expected_actions[int(shadow.slot)]):
                raise OracleReliabilityError("no-intervention joint actions mismatch")
            result = observe(collector, shadow, proposals)
            digest, rng, _ = external_digest(evaluator, shadow, result.info)
            slot = int(result.info["slot"])
            if digest != exogenous[slot] or rng != rngs[slot]:
                raise OracleReliabilityError(f"CRN diverged at slot {slot}")
            trace.append({"slot": slot, "external": digest, "rng": rng,
                          "proposal_digest": _canonical_sha256([list(x.branches) for x in proposals])})
            energy = _energy_from_step(result.info, focal_task_id=decision["task_id"], source_uav=decision["source"])
            cpu_energy += energy["focal_cpu"]
            tx_energy += energy["source_tx"]
            if energy["source_tx_status"] == "ALLOCATED_PROXY" or attribution == "ALLOCATED_PROXY":
                attribution = "ALLOCATED_PROXY"
            elif energy["source_tx_status"] != "NOT_AVAILABLE":
                attribution = energy["source_tx_status"]
            if result.truncated or result.terminated:
                break
            step = frozen.act(result.observations, hidden)
            hidden, proposals = step.hidden_out, step.proposals
    shadow.assert_invariants()
    if root.state_fingerprint() != state:
        raise OracleReliabilityError("branch mutated root")
    target = target_result(collector, decision["task_id"], route, cpu_energy, tx_energy, attribution)
    if target["destination_uav"] != (decision["source"] if route == "local" else route):
        raise OracleReliabilityError("intervention destination not executed")
    changed = []
    for task_id, task in collector.tasks.items():
        prior = factual.tasks[task_id]
        if task["final_status"] != prior["final_status"]:
            changed.append({"task_id": task_id, "factual": prior["final_status"], "branch": task["final_status"]})
    target.update(
        system=summary(collector, decision["prefix_cpu_energy"], decision["prefix_tx_energy"]),
        changed_outcomes=changed, other_changed_outcomes=[x for x in changed if x["task_id"] != decision["task_id"]],
        reliability={"initial_state_digest": state, "initial_rng_digest": decision["rng_digest"],
                     "hidden_in_digest": decision["hidden_in_digest"], "initial_hidden_out_digest": tensor_digest(decision["hidden_out"]),
                     "exogenous_rng_each_slot_match": True, "verified_slots": len(trace),
                     "trace_sha256": _canonical_sha256(trace), "intervention_count": 1,
                     "only_target_route_changed": True, "root_unchanged": True},
        replayed=True,
    )
    return target, collector


def factual_candidate(decision, collector):
    task_id = decision["task_id"]
    cpu = sum(x["cpu_energy_j"] for x in collector.cpu_slots if x["served_task_id"] == task_id)
    tx = 0.
    attribution = "NOT_APPLICABLE" if decision["proposals"][decision["source"]].route == "local" else "EXACT"
    for row in collector.tx_slots:
        if row["task_id"] != task_id or not row["accepted"]:
            continue
        # A zero-bit accepted attempt still consumes radio energy; the head
        # owns it when this is the only serviced task (same existing contract).
        fraction = row["transmitted_bits"]/row["link_transmitted_bits"] if row["link_transmitted_bits"] > 0 else float(row["link_summary"])
        tx += row["tx_energy_j"]*fraction
        if fraction < 1-1e-12:
            attribution = "ALLOCATED_PROXY"
    target = target_result(collector, task_id, decision["proposals"][decision["source"]].route, cpu, tx, attribution)
    system = summary(collector)
    suffix = [x for x in collector.cpu_slots if x["slot"] >= decision["slot"]]
    system["suffix_cpu_nonempty_idle"] = sum(x["cpu_queue_length"] > 0 and x["served_cycles"] <= 0 for x in suffix)
    system["suffix_cpu_queue_task_slots"] = sum(x["cpu_queue_length"] for x in suffix)
    system["suffix_cpu_queue_cycle_slots"] = sum(x["cpu_queue_cycles"] for x in suffix)
    target.update(system=system, changed_outcomes=[], other_changed_outcomes=[], replayed=False,
                  reliability={"factual_suffix_reused": True, "initial_state_digest": decision["state_digest"], "initial_rng_digest": decision["rng_digest"], "initial_hidden_out_digest": decision["hidden_out_digest"]})
    return target


_WORKER = None


def initialize_worker(config_path, checkpoint, output_root, factual_tasks, exogenous, rngs):
    """Spawned evaluation processes own independent frozen actors; no learner."""
    global _WORKER
    torch.set_num_threads(1)
    config = load_run_config(config_path, cli_overrides={"mode":"evaluation", "method_id":"ca_gat_mappo", "launch_profile":None, "evaluation.evaluation_seeds":[1044], "output.logs_dir": str(output_root)})
    evaluator = FormalEvaluationRunner(config, checkpoint, evaluation_device="cpu", collect_service_telemetry=True, development_checkpoint=True)
    frozen = FrozenOraclePolicyRunnerCache(config, device="cpu").get(evaluator.loaded_actor.actor, 0, checkpoint_sha256=sha_file(checkpoint), source="P1b_frozen_checkpoint")
    from types import SimpleNamespace
    _WORKER = (evaluator, frozen, SimpleNamespace(tasks=factual_tasks), exogenous, rngs)


def evaluate_decision(payload):
    decision, factual_row = cloudpickle.loads(payload)
    evaluator, frozen, factual, exogenous, rngs = _WORKER
    initial_actor_digest = actor_state_digest(frozen.actor)
    legal = [route for route, available in zip(decision["features"]["route_domain"], decision["features"]["route_mask"]) if available and (route == "local" or isinstance(route, int))]
    if "local" not in legal or not any(isinstance(x, int) for x in legal):
        raise OracleReliabilityError("Local/Remote comparison ineligible")
    rows = []
    for route in legal:
        if route == decision["proposals"][decision["source"]].route:
            result = factual_row
        else:
            result, _ = rollout_branch(evaluator, frozen, decision, route, factual, exogenous, rngs)
        rows.append(result)
    if actor_state_digest(frozen.actor) != initial_actor_digest or frozen.actor.training or any(p.requires_grad or p.grad is not None for p in frozen.actor.parameters()):
        raise OracleReliabilityError("worker actor not frozen")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--final-checkpoint", required=True)
    parser.add_argument("--step4000-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    root = Path(args.output_root)
    if root.exists():
        raise FileExistsError("refuse to overwrite oracle output root")
    root.mkdir(parents=True)
    write_json(root/"precommitted_protocol.json", PROTOCOL)
    repository = Path(__file__).resolve().parents[2]
    protected = [p for folder in ("src", "tests", "experiments") for p in (repository/folder).rglob("*.py" if folder != "experiments" else "*.json")]
    protected += [Path(args.final_checkpoint), Path(args.step4000_checkpoint)]
    before = {str(p): sha_file(p) for p in protected}
    write_json(root/"read_only_before_hashes.json", before)
    began = time.perf_counter()
    for cohort, checkpoint in (("8k", args.final_checkpoint), ("4k_local", args.step4000_checkpoint)):
        config = load_run_config(args.config, cli_overrides={"mode":"evaluation", "method_id":"ca_gat_mappo", "launch_profile":None, "evaluation.evaluation_seeds":[1044], "output.logs_dir": str(root)})
        evaluator = FormalEvaluationRunner(config, checkpoint, evaluation_device="cpu", collect_service_telemetry=True, development_checkpoint=True)
        actor = evaluator.loaded_actor.actor
        digest_before = actor_state_digest(actor)
        frozen = FrozenOraclePolicyRunnerCache(config, device="cpu").get(actor, 0, checkpoint_sha256=sha_file(checkpoint), source="P1b_frozen_checkpoint")
        decisions, factual, exogenous, rngs, actions = collect_factual(evaluator, cohort, root)
        # Independently replay the earliest binding suffix with NO Route change.
        # Compare the complete service lifecycle and all joint proposals.
        first = decisions[0]
        replay, replay_collector = rollout_branch(evaluator, frozen, first, first["proposals"][first["source"]].route, factual, exogenous, rngs, expected_actions=actions)
        for task_id, task in replay_collector.tasks.items():
            if {k:v for k,v in task.items() if k != "evaluation_run_id"} != {k:v for k,v in factual.tasks[task_id].items() if k != "evaluation_run_id"}:
                raise OracleReliabilityError("no-intervention full task lifecycle replay mismatch")
        write_json(root/f"{cohort}_no_intervention_replay_gate.json", {"pass":True, "result":replay, "actor_digest":digest_before, "checkpoint_sha256":sha_file(checkpoint)})
        if args.workers <= 0:
            raise ValueError("workers must be positive")
        pool = ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                    initargs=(args.config,checkpoint,str(root),factual.tasks,exogenous,rngs))
        # QueueState has local defaultdict factories. cloudpickle preserves
        # those factories and object aliases; every deserialized environment
        # is still checked against its complete pre-serialization fingerprint.
        payloads = [cloudpickle.dumps((d, factual_candidate(d, factual))) for d in decisions]
        with pool:
          for index, (decision, rows) in enumerate(zip(decisions, pool.map(evaluate_decision, payloads))):
            for result in rows:
                result.update(cohort=cohort, actor_route=decision["proposals"][decision["source"]].route, source_uav=decision["source"], decision_slot=decision["slot"])
                append_jsonl(root/"branch_results.jsonl", result)
            classes = {mode: classify(rows, mode) for mode in ("primary", "strict_completion", "latency_only", "energy_only")}
            best = best_set(rows)
            remotes = best_set([x for x in rows if isinstance(x["route"], int)])
            system_best = max(x["system"]["DONE"] for x in rows)
            system_rows = [x for x in rows if x["system"]["DONE"] == system_best]
            min_exp = min(x["system"]["EXPIRED"] for x in system_rows)
            system_rows = [x for x in system_rows if x["system"]["EXPIRED"] == min_exp]
            min_energy = min(x["system"]["cpu_energy_j"]+x["system"]["tx_energy_j"] for x in system_rows)
            system_rows = [x for x in system_rows if energy_tie(x["system"]["cpu_energy_j"]+x["system"]["tx_energy_j"], min_energy)]
            actor_route = decision["proposals"][decision["source"]].route
            record = {"cohort":cohort, "task_id":decision["task_id"], "source_uav":decision["source"], "slot":decision["slot"], "actor_route":actor_route, "classes":classes, "oracle_best_routes":[x["route"] for x in best], "best_remote_routes":[x["route"] for x in remotes], "actor_agrees":actor_route in [x["route"] for x in best], "system_best_routes":[x["route"] for x in system_rows], "system_best_DONE":system_best, "factual_DONE":summary(factual)["DONE"]}
            append_jsonl(root/"actor_vs_oracle.jsonl", record)
            print(json.dumps({"cohort":cohort, "completed_decisions":index+1, "total":len(decisions), "task_id":decision["task_id"], "classes":classes, "branches":len(rows), "elapsed_s":round(time.perf_counter()-began,1)}), flush=True)
        if actor_state_digest(actor) != digest_before or actor_state_digest(frozen.actor) != digest_before or actor.training:
            raise OracleReliabilityError("frozen actor changed")
    after = {str(p): sha_file(p) for p in protected}
    if before != after:
        raise OracleReliabilityError("source/config/test/checkpoint changed during oracle")
    write_json(root/"read_only_integrity.json", {"pass":True,"protected_file_count":len(before), "checkpoint_sha256": {str(p):after[str(p)] for p in (Path(args.final_checkpoint),Path(args.step4000_checkpoint))}, "optimizer_constructed":False, "training_started":False, "elapsed_s":time.perf_counter()-began})
    rows = [json.loads(line) for line in (root/"branch_results.jsonl").read_text(encoding="utf-8").splitlines()]
    fields = [k for k in rows[0] if k not in {"system", "changed_outcomes", "other_changed_outcomes", "reliability"}]
    with (root/"branch_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"completed":True, "output_root":str(root), "branches":len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
