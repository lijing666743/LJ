"""E14b: annealed Route top-level log barrier and passive factual telemetry.

The prior is uniform over legal Local/Remote/Defer OPTIONS, never destination
leaves. KL(prior || actor) supplies recovery gradients even for unsampled,
low-probability options. It is an actor regularizer, not a reward or a sampler.
Reference: Agarwal et al., JMLR 22(98), 2021, Section 5.2 / Remark 14.
The tabular convergence theorem is NOT a guarantee for recurrent MAPPO.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ..config import MAPPOConfig, RunConfig
    from .ca_gat_mappo_actions import SequentialActionDistributionOutput
    from .ca_gat_mappo_rollout import CAGATMAPPORolloutBuffer


OPTIONS = ("Local", "Remote", "Defer")
METADATA_KEY = "route_training_v1"


def route_top_statistics(logits: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Stable log-probabilities / support in Local, Remote, Defer order.

Reuse the unchanged leaf distribution. A safe remote reduction prevents
all-minus-infinity gradients when Remote is unavailable. No probability clamp
is used: it would erase the very low-probability recovery gradient.
"""
    if logits.shape != mask.shape or mask.dtype != torch.bool or logits.shape[-1] < 3:
        raise ValueError("Route logits/mask must have matching [...,D>=3] axes")
    if not torch.any(mask, dim=-1).all() or not torch.isfinite(logits).all():
        raise ValueError("Route requires finite logits and nonempty legal support")
    leaf_logs = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
    remote_mask = mask[..., 3:]
    remote_legal = remote_mask.any(dim=-1)
    if logits.shape[-1] > 3:
        safe_terms = torch.where(remote_legal.unsqueeze(-1), leaf_logs[..., 3:],
                                 torch.zeros_like(leaf_logs[..., 3:]))
        remote_logs = torch.logsumexp(safe_terms, dim=-1)
    else:
        remote_logs = torch.zeros_like(leaf_logs[..., 0])
    support = torch.stack((mask[..., 1], remote_legal, mask[..., 2]), dim=-1)
    logs = torch.stack((leaf_logs[..., 1], remote_logs, leaf_logs[..., 2]), dim=-1)
    # Inactive Route has only the canonical Idle leaf and no top-level options.
    return torch.where(support, logs, torch.zeros_like(logs)), support


def route_exploration_coefficient(mappo: MAPPOConfig, steps: int) -> float:
    """Absolute real-transition schedule; frozen across all PPO replay epochs."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("Route schedule requires non-negative integer environment steps")
    if mappo.route_exploration_mode == "off":
        return 0.0
    return mappo.route_exploration_initial_coefficient * max(
        0.0, 1.0 - steps / mappo.route_exploration_anneal_steps)


@dataclass(frozen=True)
class RouteExplorationLoss:
    scaled_loss: Tensor
    kl_mean: Tensor
    coefficient: float
    active_count: int

    def record(self) -> dict:
        return {"coefficient": self.coefficient, "active_count": self.active_count,
                "kl_prior_to_actor_mean": float(self.kl_mean.detach().cpu()),
                "loss_contribution": float(self.scaled_loss.detach().cpu())}


def route_exploration_loss(policy: SequentialActionDistributionOutput,
                           mappo: MAPPOConfig, steps: int,
                           sequence_valid_mask: Tensor) -> RouteExplorationLoss:
    """Positive KL penalty averaged over valid Route-active decisions only."""
    logits = policy.raw_logits["route"]
    coefficient = route_exploration_coefficient(mappo, steps)
    if sequence_valid_mask.shape != logits.shape[:2] or sequence_valid_mask.dtype != torch.bool:
        raise ValueError("Route sequence validity must have [B,T] bool axes")
    active = policy.active_branches["route"] & sequence_valid_mask.unsqueeze(-1)
    logs, support = route_top_statistics(logits, policy.action_masks["route"])
    if torch.any(active & ~support.any(-1)):
        raise ValueError("active Route has no legal top-level option")
    count = support.sum(-1).clamp_min(1).to(logits.dtype)
    prior = support.to(logits.dtype) / count.unsqueeze(-1)
    kl = (prior * (-count.log().unsqueeze(-1) - logs)).sum(-1)
    n = int(active.sum())
    mean = kl[active].mean() if n else logs.sum() * 0.0
    # The updater skips the addition completely at coefficient=0 for exact
    # post-anneal and disabled arithmetic compatibility.
    return RouteExplorationLoss(coefficient * mean, mean, coefficient, n)


def collect_route_training_records(policy: SequentialActionDistributionOutput,
                                   *, environment_step: int, episode_id: int,
                                   slot: int, task_ids: dict[int, int] | None = None) -> list[dict]:
    """Observe the factual sample once, before PPO; no forward or RNG calls."""
    with torch.no_grad():
        logs, support = route_top_statistics(policy.raw_logits["route"],
                                             policy.action_masks["route"])
        probabilities = torch.where(support, logs.exp(), torch.zeros_like(logs))
        entropy = -(probabilities * logs).sum(-1)
        active = policy.active_branches["route"]
        if active.shape[:2] != (1, 1):
            raise ValueError("factual telemetry expects exactly one real decision slot")
        result = []
        for uav in range(active.shape[-1]):
            if not bool(active[0, 0, uav]):
                continue
            index = int(policy.action_indices["route"][0, 0, uav])
            category = "Local" if index == 1 else "Defer" if index == 2 else "Remote"
            p = probabilities[0, 0, uav].cpu().tolist()
            legal = support[0, 0, uav].cpu().tolist()
            # Actual hierarchical argmax uses Local, Defer, Remote tie order.
            top_tie_order = (0, 2, 1)
            winner = max((i for i in top_tie_order if legal[i]), key=lambda i: p[i])
            result.append({"environment_step": environment_step, "episode_id": episode_id,
                           "slot": slot, "uav_id": uav,
                           "actor_top_probability": dict(zip(OPTIONS, p)),
                           "legal": dict(zip(OPTIONS, legal)),
                           "top_level_entropy": float(entropy[0, 0, uav].cpu()),
                           "sampled_top_level_action": category,
                           "actor_top_level_argmax": OPTIONS[winner],
                           "route_action_index": index,
                           "stored_behavior_log_prob": float(
                               policy.branch_log_probs["route"][0, 0, uav].detach().cpu()),
                           "behavior_equals_actor": True})
            if getattr(policy, "route_decoder_mode", None) == "unified_node_v1":
                from .route_decision import unified_route_decision_record
                if task_ids is None or uav not in task_ids:
                    raise ValueError("unified Route telemetry requires factual task IDs")
                result[-1].update(unified_route_decision_record(policy, uav, task_ids[uav]))
    return result


def summarize_route_records(records: list[dict]) -> dict:
    """Conservation-checked summaries of real samples, not PPO replay visits."""
    def group(rows: list[dict]) -> dict:
        n = len(rows)
        sampled = {option: sum(r["sampled_top_level_action"] == option for r in rows)
                   for option in OPTIONS}
        legal = {option: sum(r["legal"][option] for r in rows) for option in OPTIONS}
        for r in rows:
            if not r["legal"][r["sampled_top_level_action"]]:
                raise ValueError("Route telemetry contains an illegal factual sample")
            ps = r["actor_top_probability"]
            if not math.isclose(sum(ps.values()), 1.0, abs_tol=2e-6):
                raise ValueError("Route telemetry probability conservation failed")
            if any(ps[o] != 0.0 for o in OPTIONS if not r["legal"][o]):
                raise ValueError("illegal Route option has nonzero probability")
        assert sum(sampled.values()) == n
        return {"route_active_count": n, "legal_count": legal,
                "sampled_count": sampled,
                "visitation_rate": {o: sampled[o] / n if n else None for o in OPTIONS},
                "sampled_per_legal_rate": {o: sampled[o] / legal[o] if legal[o] else None
                                           for o in OPTIONS},
                "actor_argmax_count": {o: sum(r.get("actor_decision_argmax", r["actor_top_level_argmax"]) == o for r in rows)
                                       for o in OPTIONS},
                "mean_actor_probability": {o: sum(r["actor_top_probability"][o] for r in rows) / n
                                           if n else None for o in OPTIONS},
                "mean_top_level_entropy": sum(r["top_level_entropy"] for r in rows) / n
                                          if n else None}
    return {"all_route_active": group(records),
            "remote_legal": group([r for r in records if r["legal"]["Remote"]]),
            "remote_unavailable": group([r for r in records if not r["legal"]["Remote"]])}


def write_route_rollout_telemetry(config: RunConfig, buffer: CAGATMAPPORolloutBuffer,
                                 *, environment_steps: int, optimized: bool,
                                 ppo_epochs: list[dict] | None = None) -> dict:
    """Write one record per full rollout, plus the unoptimized final tail.

    Partial periodic checkpoints retain pending samples in existing structured
    rollout metadata; they are emitted once when that rollout finishes on resume.
    """
    records = []
    for i in range(len(buffer)):
        metadata = buffer.transition_at(i).rejection_or_downgrade_summary
        if METADATA_KEY not in metadata:
            raise ValueError("enabled Route telemetry is missing factual rollout records")
        records.extend(metadata[METADATA_KEY])
    row = {"schema": "route_training_rollout.v1", "run_id": config.run_id,
           "config_hash": config.config_hash,
           "environment_step_start": environment_steps - len(buffer),
           "environment_step_end": environment_steps,
           "transition_count": len(buffer), "optimized": optimized,
           "route_exploration_mode": config.training.mappo.route_exploration_mode,
           "groups": summarize_route_records(records),
           "ppo_epochs": ppo_epochs or [], "samples": records}
    if config.training.mappo.route_decoder_mode == "unified_node_v1":
        row.update(schema="route_training_rollout.v2", decision_mode="flat_leaf_v1")
    path = Path(config.artifact_paths()["raw_metrics"]).parent / "route_training_rollouts_v1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    return row
