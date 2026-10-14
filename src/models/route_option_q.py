"""E14d on-policy, candidate-specific Route Q and frozen relative credit.

Only the factual leaf regresses to E14c G_event. Other leaves are predictions,
never counterfactual ground truth. The pre-update critic freezes the actor's
advantage; current-rollout Q fitting then runs alongside the four PPO epochs.
No Q gradient enters an actor encoder, behavior distribution or V_task target.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch import Tensor, nn

MODE = "candidate_counterfactual_q_v1"


class RouteOptionQ(nn.Module):
    """One independent scalar head shared by every legal candidate leaf."""

    def __init__(self, state_dim: int, candidate_dim: int, hidden: int, seed: int,
                 *, input_mode: str = "legacy", state_fields=None):
        super().__init__()
        if input_mode not in {"legacy", "normalized", "dual_branch"}:
            raise ValueError("invalid Route Q input mode")
        self.input_mode = input_mode
        self.state_dim = state_dim
        self.candidate_dim = candidate_dim
        if input_mode != "legacy":
            from .route_q_inputs import RouteQInputConditioner
            self.conditioner = RouteQInputConditioner(state_dim, candidate_dim, state_fields)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed + 140004)
            if input_mode == "dual_branch":
                self.state_projection = nn.Sequential(nn.Linear(state_dim, hidden), nn.LeakyReLU(), nn.LayerNorm(hidden))
                self.candidate_projection = nn.Sequential(nn.Linear(candidate_dim, hidden), nn.LeakyReLU(), nn.LayerNorm(hidden))
            self.network = nn.Sequential(
                nn.Linear(2*hidden+3 if input_mode == "dual_branch" else state_dim + candidate_dim, hidden), nn.LeakyReLU(),
                nn.LayerNorm(hidden), nn.Linear(hidden, 1))
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def preprocess(self, state: Tensor, candidates: Tensor) -> tuple[Tensor, Tensor]:
        if self.input_mode == "legacy":
            return state, candidates
        return self.conditioner(state, candidates)

    def forward(self, state: Tensor, candidates: Tensor, legal: Tensor) -> Tensor:
        if (candidates.shape[:-1] != legal.shape or legal.dtype != torch.bool
                or state.shape[:-1] != legal.shape[:-1]
                or state.shape[-1] != self.state_dim or candidates.shape[-1] != self.candidate_dim):
            raise ValueError("Route Q state/candidate/legal axes disagree")
        # Gather before the MLP: illegal candidates do not enter normalization,
        # regression or baseline, even if their input contains invalid values.
        result = candidates.new_zeros(legal.shape)
        if legal.any():
            expanded = state.detach().unsqueeze(-2).expand(*legal.shape, state.shape[-1])
            state_input, candidate_input = self.preprocess(expanded[legal], candidates.detach()[legal])
            if self.input_mode == "dual_branch":
                inputs = torch.cat((self.state_projection(state_input),
                                    self.candidate_projection(candidate_input), candidate_input[..., -3:]), -1)
            else:
                inputs = torch.cat((state_input, candidate_input), -1)
            result[legal] = self.network(inputs).squeeze(-1)
        return result


def counterfactual_values(q: Tensor, probability: Tensor, legal: Tensor,
                          chosen: Tensor, destination_probability: Tensor | None = None
                          ) -> tuple[Tensor, Tensor, Tensor]:
    """Leaf expectation equals the hierarchical top-level expectation."""
    if q.shape != probability.shape or q.shape != legal.shape:
        raise ValueError("Route Q/probability/legal shapes disagree")
    if torch.any(probability[~legal] != 0) or not torch.isfinite(probability).all():
        raise ValueError("illegal Route probability or nonfinite distribution")
    if torch.any(probability < 0) or not torch.allclose(
            probability.sum(-1), torch.ones_like(probability[..., 0]), atol=2e-6, rtol=0):
        raise ValueError("Route leaf probabilities do not conserve mass")
    if not legal.gather(-1, chosen.unsqueeze(-1)).all():
        raise ValueError("chosen Route option is illegal")
    masked_q = q.masked_fill(~legal, 0)
    baseline = (probability * masked_q).sum(-1)
    remote_mass = probability[..., 3:].sum(-1)
    if destination_probability is None:
        destination_probability = (probability[..., 3:]
                                   / remote_mass.unsqueeze(-1).clamp_min(torch.finfo(q.dtype).tiny))
    if (destination_probability.shape != probability[..., 3:].shape
            or not torch.isfinite(destination_probability).all()
            or torch.any(destination_probability < 0)
            or torch.any(destination_probability[~legal[..., 3:]] != 0)):
        raise ValueError("invalid Route destination conditional probability")
    if not torch.allclose(destination_probability.sum(-1), legal[..., 3:].any(-1).to(q.dtype),
                          atol=2e-6, rtol=0):
        raise ValueError("Route conditional probabilities do not conserve mass")
    if not torch.allclose(probability[..., 3:], remote_mass.unsqueeze(-1)*destination_probability,
                          atol=2e-6, rtol=0):
        raise ValueError("Route leaf/conditional hierarchy disagrees")
    remote_q = (destination_probability * masked_q[..., 3:]).sum(-1)
    chosen_q = masked_q.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
    return baseline, chosen_q - baseline, remote_q


@dataclass
class RouteQBatch:
    state: Tensor
    candidates: Tensor
    legal: Tensor
    chosen: Tensor
    targets: Tensor
    frozen_q: Tensor
    baseline: Tensor
    counterfactual_advantage: Tensor
    advantage: Tensor
    credit_samples: list[dict]
    records: list[dict]


def prepare_route_q(critic, policy, centralized: Tensor, task_credit, beta: float,
                    *, actor_credit_source: str = "counterfactual_q") -> RouteQBatch:
    """Freeze current-rollout inputs, old Q, targets and actor credit once."""
    if actor_credit_source not in {"counterfactual_q", "task_event"}:
        raise ValueError("invalid Route actor credit source")
    features = policy.route_candidate_features
    if features is None:
        raise ValueError("Route candidate representation is missing")
    agents, options = policy.action_masks["route"].shape[-2:]
    features = features.detach().reshape(-1, agents, options, features.shape[-1])
    masks = policy.action_masks["route"].detach().reshape(-1, agents, options)
    choices = policy.action_indices["route"].detach().reshape(-1, agents)
    probabilities = policy.probabilities["route"].detach().reshape_as(masks)
    state = centralized.detach().reshape(-1, centralized.shape[-1])
    # Exact old factual distribution is saved at collection, including Remote
    # conditional leaves. Replay only supplies current-state embeddings.
    ii = [s["rollout_offset"] for s in task_credit.samples]
    uu = [s["source_uav"] for s in task_credit.samples]
    selected_state = state[ii].clone()
    selected_features = features[ii, uu].clone()
    legal = masks[ii, uu].clone()
    chosen = choices[ii, uu].clone()
    p = probabilities[ii, uu].clone()
    conditional = p[..., 3:].clone()
    for n, s in enumerate(task_credit.samples):
        if "actor_leaf_probability" not in s:
            raise ValueError("E14d requires factual Route leaf probabilities")
        stored = p.new_tensor(s["actor_leaf_probability"])
        if not torch.equal(legal[n].cpu(), torch.tensor(s["route_legal_mask"])):
            raise ValueError("factual and replay Route legal masks differ")
        if int(chosen[n]) != s["route_action_index"]:
            raise ValueError("factual and replay Route index differ")
        p[n] = stored
        conditional[n] = p.new_tensor(s["actor_remote_conditional_probability"])
        if legal[n, 0]:
            raise ValueError("inactive Idle cannot receive Route Q credit")
    targets = state.new_tensor([s["return"] for s in task_credit.samples])
    with torch.no_grad():
        q = critic.route_option_q(selected_state, selected_features, legal).detach().clone()
        baseline, cf, remote_q = counterfactual_values(q, p, legal, chosen, conditional)
    advantage = task_credit.advantage.detach().clone().to(state)
    records = []; credit_samples = []
    for n, s in enumerate(task_credit.samples):
        i, u = ii[n], uu[n]
        # P1a reuses E14c's already-frozen combined tensor verbatim. Q inputs,
        # targets, baseline and CF diagnostics are identical in both modes.
        effective_task = s["coefficient"] if actor_credit_source == "task_event" else 0.0
        effective_cf = beta if actor_credit_source == "counterfactual_q" else 0.0
        if actor_credit_source == "counterfactual_q":
            advantage[i, u] = s["shared_advantage"] + beta * cf[n]
        labels = ["Idle", "Local", "Defer"] + [
            f"Remote-UAV{d}" for d in range(agents) if d != u]
        support = {label: index for index, label in enumerate(labels) if bool(legal[n, index])}
        remote_mass = float(p[n, 3:].sum())
        probabilities_by_option = {label: float(p[n, index]) for label, index in support.items()}
        predicted_q = {label: float(q[n, index]) for label, index in support.items()}
        remote_values = [value for label, value in predicted_q.items() if label.startswith("Remote-")]
        updated = {**s, "coefficient": effective_task,
                   "combined_advantage": float(advantage[i, u]),
                   "actor_advantage_mode": MODE if actor_credit_source == "counterfactual_q" else "task_event_value_v1",
                   "actor_credit_source": actor_credit_source,
                   "effective_actor_task_credit_coefficient": effective_task,
                   "effective_actor_counterfactual_coefficient": effective_cf,
                   "task_advantage": s["route_specific_advantage"],
                   "counterfactual_advantage": float(cf[n]),
                   "final_route_advantage": float(advantage[i, u])}
        credit_samples.append(updated)
        records.append({**updated, "legal_options": list(support),
                        "chosen_option": labels[int(chosen[n])],
                        "chosen_destination": (u if int(chosen[n]) == 1 else
                            None if int(chosen[n]) == 2 else int(labels[int(chosen[n])].split("UAV")[1])),
                        "frozen_leaf_probability": probabilities_by_option,
                        "frozen_actor_top_probability": {
                            "Local": float(p[n, 1]), "Remote": remote_mass, "Defer": float(p[n, 2])},
                        "destination_conditional_probability": {
                            label: float(conditional[n, index-3]) for label, index in support.items()
                            if label.startswith("Remote-")},
                        "frozen_q": predicted_q, "aggregated_remote_q": float(remote_q[n]) if remote_values else None,
                        "counterfactual_baseline": float(baseline[n]),
                        "q_chosen": float(q[n, chosen[n]]), "counterfactual_advantage": float(cf[n]),
                        "counterfactual_coefficient": beta, "task_advantage": s["route_specific_advantage"],
                        "final_route_advantage": float(advantage[i, u]), "G_event": s["return"],
                        "V_task": s["value"], "supervised_option": labels[int(chosen[n])],
                        "predicted_local_minus_max_remote": (predicted_q["Local"] - max(remote_values)
                            if "Local" in predicted_q and remote_values else None)})
    return RouteQBatch(selected_state, selected_features, legal, chosen, targets,
                       q, baseline.detach().clone(), cf.detach().clone(), advantage,
                       credit_samples, records)


def route_q_loss(critic, batch: RouteQBatch) -> tuple[Tensor, Tensor]:
    q = critic.route_option_q(batch.state, batch.candidates, batch.legal)
    if not batch.chosen.numel():
        return sum(p.sum() * 0 for p in critic.route_option_q.parameters()), q
    prediction = q.gather(-1, batch.chosen.unsqueeze(-1)).squeeze(-1)
    return 0.5 * (prediction - batch.targets).square().mean(), q


def epoch_q_records(batch: RouteQBatch, current_q: Tensor, actor_rows: list[dict]) -> list[dict]:
    if len(actor_rows) != len(batch.records):
        raise ValueError("Route Q/PPO telemetry sample counts disagree")
    rows = []
    for n, (frozen, actor) in enumerate(zip(batch.records, actor_rows)):
        if any(frozen[k] != actor[k] for k in ("episode_id", "rollout_offset", "source_uav", "task_id")):
            raise ValueError("Route Q/PPO telemetry identities disagree")
        prediction = float(current_q[n, batch.chosen[n]].detach())
        rows.append({**actor, **frozen, "epoch_index": actor["epoch_index"],
                     "current_q_chosen": prediction,
                     "q_error": prediction - frozen["G_event"],
                     "q_statistics_timing": "pre_optimizer_step"})
    return rows


def calibration(rows: list[dict]) -> dict:
    groups = {}
    for category in ("local", "remote", "defer"):
        rr = [r for r in rows if r["category"] == category]
        groups[category] = {
            "epoch_sample_count": len(rr),
            "unique_factual_count": len({(r["episode_id"], r["global_rollout_index"], r["source_uav"])
                                         for r in rr}),
            "mae": sum(abs(r["q_error"]) for r in rr) / len(rr) if rr else None,
            "rmse": math.sqrt(sum(r["q_error"] ** 2 for r in rr) / len(rr)) if rr else None,
            "mean_prediction": sum(r["current_q_chosen"] for r in rr) / len(rr) if rr else None,
            "mean_target": sum(r["G_event"] for r in rr) / len(rr) if rr else None}
    return groups


def predicted_rankings(rows: list[dict]) -> dict:
    """Outcome-conditioned predictions, not unobserved action correctness."""
    groups = {}
    for category in ("local", "remote", "defer"):
        for outcome in ("completed", "expired", "truncated", None):
            rr = [r for r in rows if r["category"] == category and r["terminal_status"] == outcome
                  and r["predicted_local_minus_max_remote"] is not None]
            gaps = [r["predicted_local_minus_max_remote"] for r in rr]
            groups[f"{category}/{outcome or 'censored'}"] = {
                "epoch_sample_count": len(rr),
                "mean_predicted_local_minus_max_remote": sum(gaps) / len(gaps) if gaps else None,
                "predicted_local_above_all_remote_count": sum(g > 0 for g in gaps)}
    return groups


def write_route_q_telemetry(config, rows, steps, losses, *, buffer, optimized=True):
    from .route_task_credit import KEY, summarize_route_credit_components
    events = [buffer.transition_at(i).rejection_or_downgrade_summary[KEY] for i in range(len(buffer))]
    row = {"schema": "route_option_q_update.v1", "run_id": config.run_id,
           "config_hash": config.config_hash, "environment_step_start": steps - len(buffer),
           "environment_step_end": steps, "optimized": optimized, "mode": MODE,
           "counterfactual_coefficient": config.training.mappo.route_counterfactual_credit_coefficient,
           "q_loss_coefficient": config.training.mappo.route_q_value_loss_coefficient,
           "q_input_mode": config.training.mappo.route_q_input_mode,
           "actor_credit_source": config.training.mappo.route_actor_credit_source,
           "effective_actor_counterfactual_coefficient": (
               config.training.mappo.route_counterfactual_credit_coefficient
               if config.training.mappo.route_actor_credit_source == "counterfactual_q" else 0.0),
           "credit_components_by_epoch": summarize_route_credit_components(rows),
           "q_freeze_timing": "before_current_rollout_Q_and_PPO_updates",
           "unselected_Q_is_prediction_only": True,
           "samples": rows, "q_loss_by_epoch": losses, "calibration": calibration(rows),
           "predicted_rankings": predicted_rankings(rows),
           "calibration_by_epoch": {str(epoch): calibration([r for r in rows if r["epoch_index"] == epoch])
                                    for epoch in sorted({r["epoch_index"] for r in rows})},
           "task_terminals": [{**t, "episode_id": m["episode_id"]} for m in events for t in m["terminals"]],
           "route_decisions": [d for m in events for d in m["decisions"]]}
    if config.training.mappo.route_decoder_mode == "unified_node_v1":
        row.update(schema="route_option_q_update.v2", decision_mode="flat_leaf_v1")
    path = Path(config.artifact_paths()["raw_metrics"]).parent / "route_option_q_updates_v1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    return row
