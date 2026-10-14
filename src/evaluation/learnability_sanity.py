"""Short real environment diagnostics without a network, optimizer, or checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from ..config import load_run_config
from ..env.environment import U2UMECEnvironment
from ..policies.heuristic_policy import HeuristicPolicy


def run_sanity(config_path: str, slots: int) -> dict:
    """Preserve the configured horizon/reset while stopping a rollout early."""
    config = load_run_config(config_path)
    if not 1 <= slots <= config.environment.episode_horizon:
        raise ValueError("slots must be between 1 and the unchanged episode horizon")
    env = U2UMECEnvironment(config)
    result = env.reset()
    initial_positions = env.mobility.positions_m.copy()
    initial_graph = env.topology.candidate_neighbors.copy()
    policy = HeuristicPolicy(config)
    route_counts: Counter = Counter()
    positive_bit_links = attempted_links = outage_links = 0
    received_public_valid = graph_changes = 0
    max_interference = max_position_displacement = 0.
    for _ in range(slots):
        proposals = tuple(policy.act(o) for o in result.observations)
        for proposal, observation in zip(proposals, result.observations):
            if observation.action_masks.route_branch_active:
                route_counts["Remote" if isinstance(proposal.route, int) else str(proposal.route).capitalize()] += 1
        received_public_valid += int(np.count_nonzero(env.public_messages.valid_mask))
        graph_changes += int(np.count_nonzero(env.topology.candidate_neighbors != initial_graph))
        max_position_displacement = max(max_position_displacement, float(np.max(np.linalg.norm(env.mobility.positions_m - initial_positions, axis=1))))
        result = env.step(proposals)
        physical = result.info["service"]
        for link in physical["links"]:
            positive_bit_links += int(link["service_bits"] > 0)
            max_interference = max(max_interference, max(link["interference_w"], default=0.))
        attempted_links += int(result.info["outage"]["actual_attempt_count"])
        outage_links += int(result.info["outage"]["outage_count"])
        assert env.assert_invariants().is_conserved
    tasks = list(env.lifecycle.tasks.values())
    return {
        "kind": "short_environment_sanity_not_training_or_policy_evaluation",
        "config": str(Path(config_path).resolve()), "config_hash": config.config_hash,
        "seed": config.seed, "configured_horizon": config.environment.episode_horizon,
        "executed_slots": slots, "episode_finished": env.done,
        "policy": "existing_heuristic", "route_active_proposals": dict(route_counts),
        "generated_tasks": len(tasks),
        "task_outcomes": dict(Counter(t.outcome.value for t in tasks)),
        "current_task_statuses": dict(Counter(t.status.value for t in tasks)),
        "positive_bit_link_slots": positive_bit_links, "actual_attempts": attempted_links,
        "outage_link_slots": outage_links, "max_actual_interference_w": max_interference,
        "max_displacement_from_reset_m": max_position_displacement,
        "candidate_edge_difference_count_from_reset": graph_changes,
        "public_valid_uav_slot_count": received_public_valid,
        "residual_energy_j": {str(k): v.residual_energy_j for k, v in env.resource_states.items()},
        "conservation": env.assert_invariants().to_dict(),
        "rng": env.randomness_snapshot(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--slots", type=int, default=96)
    parser.add_argument("--output", required=True, help="new JSON path under logs/")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    logs_root = Path(__file__).resolve().parents[2] / "logs"
    if not output.is_relative_to(logs_root.resolve()):
        parser.error("sanity output must be under the project logs directory")
    if output.exists():
        parser.error("sanity output already exists; choose a new path")
    result = run_sanity(args.config, args.slots)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"saved": str(output), "slots": args.slots, "training_started": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
