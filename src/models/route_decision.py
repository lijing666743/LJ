"""Passive versioned Route identity; missing historical fields stay unknown."""
from __future__ import annotations

from .route_candidates import route_leaf_result


DECISION_FIELDS = (
    "decision_mode", "feature_mode", "task_id", "candidate_node_ids", "candidate_action_indices",
    "leaf_node_ids", "route_legal_mask", "leaf_scores", "leaf_probabilities",
    "chosen_leaf_index", "chosen_node_id", "leaf_argmax_index", "leaf_argmax_node_id",
    "group_mass", "group_mass_argmax", "actor_decision_argmax",
)


def normalize_route_decision_record(record: dict) -> dict:
    """Read v1/v2 records without inventing old task IDs, scores or decisions."""
    return {**{key: None for key in DECISION_FIELDS}, **record}


def unified_route_decision_record(policy, uav: int, task_id: int) -> dict:
    """Use the already-computed actual proposal, without a new forward or RNG."""
    if getattr(policy, "route_decoder_mode", None) != "unified_node_v1":
        raise ValueError("unified decision record requires the actual unified policy")
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
        raise ValueError("unified decision record requires the factual task ID")
    leaf = route_leaf_result(policy.raw_logits["route"], policy.action_masks["route"],
                             policy.action_indices["route"])
    chosen = int(leaf.chosen_indices[0, 0, uav])
    winner = int(leaf.leaf_argmax_indices[0, 0, uav])
    nodes = leaf.leaf_node_ids[uav].cpu().tolist()
    groups = ("Local", "Remote", "Defer")
    return {
        "schema_version": "route_decision.v2", "decoder_mode": "unified_node_v1",
        "decision_mode": "flat_leaf_v1", "task_id": task_id,
        "feature_mode": getattr(policy, "route_feature_mode", None),
        "candidate_node_ids": [nodes[1], *nodes[3:]],
        "candidate_action_indices": [1, *range(3, len(nodes))],
        "leaf_node_ids": nodes,
        "route_legal_mask": leaf.legal_mask[0, 0, uav].cpu().tolist(),
        "leaf_scores": policy.raw_logits["route"][0, 0, uav].detach().cpu().tolist(),
        "leaf_probabilities": leaf.probabilities[0, 0, uav].detach().cpu().tolist(),
        "chosen_leaf_index": chosen, "chosen_node_id": nodes[chosen] if nodes[chosen] >= 0 else None,
        "leaf_argmax_index": winner, "leaf_argmax_node_id": nodes[winner] if nodes[winner] >= 0 else None,
        "group_mass": dict(zip(groups, leaf.group_probabilities[0, 0, uav].detach().cpu().tolist())),
        "group_mass_argmax": (groups[int(leaf.group_mass_argmax[0, 0, uav])]
                              if int(leaf.group_mass_argmax[0, 0, uav]) >= 0 else None),
        "actor_decision_argmax": "Local" if winner == 1 else "Defer" if winner == 2 else "Remote" if winner >= 3 else "Idle",
        "actor_top_level_argmax": None,
    }
