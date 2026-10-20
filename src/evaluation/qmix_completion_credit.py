"""Audit actual completion service, replay draws and direct delayed-reward credit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from ..artifacts import atomic_write_text_group
from ..config import load_run_config
from ..models.qmix_continuation import load_continuation_checkpoint
from .actor_loader import checkpoint_sha256


def describe(values):
    values = np.asarray(values, dtype=float)
    return {"count": len(values), "min": float(values.min()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "mean": float(values.mean()) if len(values) else None,
            "max": float(values.max()) if len(values) else None}


def run(config_path, output_path):
    config = load_run_config(config_path)
    path = Path(config.artifact_paths()["final_checkpoint"])
    payload = load_continuation_checkpoint(path)
    raw = payload["replay"]["sequences"]
    counts = np.zeros(len(raw), dtype=int)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 120, 1]))
    steps, selections = 0, []
    for i, sequence in enumerate(raw):
        steps += sequence["reward"].shape[1]
        if i + 1 < config.training.qmix.sequence_batch_size:
            continue
        indices = rng.choice(i+1, config.training.qmix.sequence_batch_size, replace=False)
        row = payload["updates"][len(selections)]
        assert row["environment_steps"] == steps
        assert row["real_transitions"] == sum(raw[j]["reward"].shape[1] for j in indices)
        counts[indices] += 1
        selections.append([int(j) for j in indices])
    assert len(selections) == payload["update_count"]
    assert rng.bit_generator.state == payload["replay"]["rng"]

    services, routes, events = defaultdict(list), {}, []
    completed_by_episode, credit_counts = Counter(), Counter()
    scale = config.environment.uav_count * config.environment.episode_horizon
    for index, sequence in enumerate(raw):
        episode = sequence["episode"]
        for time, actions in enumerate(sequence["executed"]):
            slot = sequence["first_slot"] + time
            for agent, contract in enumerate(sequence["contracts"][0][time]):
                proposal = sequence["proposals"][0][time][agent]
                if contract["route_branch_active"] and proposal["route"] not in ("idle", "defer"):
                    # The existing whitelist tensorizer stores unbound head ID
                    # at 9 resource features + queue-vector offset 3. Verify
                    # reversible integer decoding rather than truncating it.
                    encoded = float(sequence["actor"]["self_features"][0,time,agent,12])*scale
                    task_id = round(encoded)-1
                    assert abs(encoded-round(encoded)) < 1.e-3 and task_id >= 0
                    key = (episode, task_id)
                    assert key not in routes
                    routes[key] = {"slot": slot, "sequence_index": index, "source": agent}
            for action in actions:
                cpu = action["cpu"]
                if not cpu["active"]:
                    continue
                remaining = cpu["head_remaining_cycles"]
                amount = min(remaining, cpu["executed_frequency_hz"]*config.environment.slot_duration_s)
                key = (episode, cpu["head_task_id"])
                services[key].append({"slot": slot, "sequence_index": index, "cycles": amount})
                if amount == remaining and remaining > 0:
                    completed_by_episode[episode] += 1
                    events.append({"episode": episode, "task_id": cpu["head_task_id"],
                                   "slot": slot, "sequence_index": index, "executor": cpu["executor_uav"]})
    assert len(events) == sum(row["completed"] for row in payload["episodes"])
    assert [completed_by_episode[i] for i in range(payload["next_episode"])] == [row["completed"] for row in payload["episodes"]]
    gamma = config.training.qmix.gamma
    for event in events:
        key = (event["episode"], event["task_id"])
        first = services[key][0]
        route = routes[key]
        assert route["slot"] < first["slot"] <= event["slot"]
        event.update(first_service_slot=first["slot"], route_slot=route["slot"],
                     first_service_delay=event["slot"]-first["slot"], route_delay=event["slot"]-route["slot"],
                     completion_chunk_sample_count=int(counts[event["sequence_index"]]),
                     first_service_crosses_chunk=first["sequence_index"] != event["sequence_index"],
                     route_crosses_chunk=route["sequence_index"] != event["sequence_index"])
        for name, start in (("route",route),("first_service",first)):
            delay = event["slot"]-start["slot"]
            same = start["sequence_index"] == event["sequence_index"]
            event[name+"_direct_old"] = (gamma*.6)**delay if same else 0.
            event[name+"_direct_full_suffix_lambda06"] = (gamma*.6)**delay
            event[name+"_direct_full_suffix_lambda095"] = (gamma*.95)**delay
        for service in services[key]:
            same = service["sequence_index"] == event["sequence_index"]
            coefficient = (gamma*.6)**(event["slot"]-service["slot"]) if same else 0.
            credit_counts["positive_service_actions_of_completed_tasks"] += 1
            credit_counts["service_reward_crosses_chunk"] += int(not same)
            credit_counts["direct_completion_coefficient_below_001"] += int(coefficient < .01)
    positive_chunks = {event["sequence_index"] for event in events}
    result = {"schema": "qmix_exact_completion_credit.v1", "checked_at": datetime.now().astimezone().isoformat(),
              "checkpoint_sha256": checkpoint_sha256(path), "completed_tasks": len(events),
              "optimizer_updates": len(selections), "replay_draws_exactly_reconstructed": True,
              "sampled_chunk_counts": counts.tolist(), "completion_chunks": len(positive_chunks),
              "completion_events_in_unsampled_chunks": int(sum(counts[event["sequence_index"]] == 0 for event in events)),
              "completion_chunk_sample_counts": describe([counts[i] for i in positive_chunks]),
              "route_delay_slots": describe([event["route_delay"] for event in events]),
              "first_service_delay_slots": describe([event["first_service_delay"] for event in events]),
              "route_crosses_chunk": sum(event["route_crosses_chunk"] for event in events),
              "first_service_crosses_chunk": sum(event["first_service_crosses_chunk"] for event in events),
              "direct_credit": {name: describe([event[name] for event in events]) for name in (
                  "route_direct_old", "route_direct_full_suffix_lambda06", "route_direct_full_suffix_lambda095",
                  "first_service_direct_old", "first_service_direct_full_suffix_lambda06", "first_service_direct_full_suffix_lambda095")},
              "completed_service_credit": dict(credit_counts), "events": events, "optimizer_updates_during_audit": 0,
              "claim_boundary": "Exact actual-service, tensor head-ID and replay-RNG accounting. Direct completion credit excludes learned bootstrap credit. No unique root cause or learned performance improvement is established."}
    atomic_write_text_group([(Path(output_path), json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))],
                           group_name="Actual QMIX completion credit audit")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    result = run(args.config, args.output_file)
    print(json.dumps({k:v for k,v in result.items() if k not in ("events","sampled_chunk_counts")},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
