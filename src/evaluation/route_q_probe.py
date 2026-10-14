"""Offline Route-Q probes: saved training tensors and disposable synthetic Qs.

Never creates an environment, selects an action, or updates a real checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


def distribution(x):
    t = torch.as_tensor(x, dtype=torch.float64).detach().flatten()
    return {"count": t.numel(), "mean": float(t.mean()) if t.numel() else None,
            "std": float(t.std(unbiased=False)) if t.numel() else None,
            "min": float(t.min()) if t.numel() else None,
            "max": float(t.max()) if t.numel() else None}


def norm(x):
    return float(torch.linalg.vector_norm(x.detach().double()))


def mean_row_norm(x):
    return float(x.detach().double().norm(dim=-1).mean()) if x.numel() else 0.


def training_audit(path):
    samples = []
    for line in Path(path).open(encoding="utf-8"):
        row = json.loads(line)
        samples.extend(s for s in row["samples"] if s["epoch_index"] == 0)
    groups = {}
    for action in ("local", "remote", "defer"):
        rr = [r for r in samples if r["category"] == action]
        groups[action] = {"targets": distribution([r["G_event"] for r in rr]),
            "remote_legal_count": sum(any(k.startswith("Remote-") for k in r["legal_options"]) for r in rr),
            "sources": sorted({r["source_uav"] for r in rr}),
            "episodes": sorted({r["episode_id"] for r in rr}),
            "target_types": {str(k): sum(r["terminal_status"] == k for r in rr)
                             for k in {r["terminal_status"] for r in rr}},
            "preupdate_error": distribution([r["q_chosen"] - r["G_event"] for r in rr])}
    cells = {}
    for r in samples:
        key = f"episode{r['episode_id']}/source{r['source_uav']}/remote{any(k.startswith('Remote-') for k in r['legal_options'])}"
        cells.setdefault(key, set()).add(r["category"])
    return {"scope": "unique epoch-zero training records; no unchosen target exists",
            "sample_count": len(samples), "targets": distribution([r["G_event"] for r in samples]),
            "actions": groups, "coarse_coverage_overlap": {
                "definition": "episode/source/Remote-legality cells; NOT identical-state overlap",
                "cells": len(cells), "local_remote_shared_cells": sum({"local", "remote"} <= v for v in cells.values()),
                "all_three_shared_cells": sum({"local", "remote", "defer"} <= v for v in cells.values())}}


def load_saved_tensors(config_path, checkpoint):
    from src.config import load_run_config
    from src.models.ca_gat_mappo import CAGATMAPPOActor, MAPPOCentralizedCritic
    from src.models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution
    from src.models.ca_gat_mappo_checkpoint import restore_active_rollout
    from src.models.ca_gat_mappo_rollout import CAGATMAPPORolloutChunk
    from src.models.route_task_credit import prepare_task_credit
    from src.models.route_option_q import prepare_route_q
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = load_run_config(config_path)
    if payload["config_hash"] != config.config_hash:
        raise ValueError("checkpoint/config identity differs")
    buffer, _ = restore_active_rollout(config, payload["active_rollout_state"],
        checkpoint_kind=payload["checkpoint_kind"], policy_version=payload["training_state"]["policy_version"])
    if not len(buffer):
        raise ValueError("checkpoint has no saved active training tensors")
    actor = CAGATMAPPOActor(config); critic = MAPPOCentralizedCritic(config)
    actor.load_state_dict(payload["model_state"]["actor"])
    critic.load_state_dict(payload["model_state"]["critic"])
    chunk = CAGATMAPPORolloutChunk(tuple(buffer.transition_at(i) for i in range(len(buffer))))
    policy = CAGATMAPPOActionDistribution(actor, config).evaluate_actions(
        chunk.actor_batch, chunk.action_mask_batch, chunk.proposals, chunk.initial_hidden,
        return_route_candidates=True)
    # Shared GAE is not reconstructed here: zero is ONLY a placeholder to build
    # unchanged event targets. It is never reported as the training advantage.
    task = prepare_task_credit(buffer, torch.zeros(len(buffer), actor.spec.uav_count), config)
    batch = prepare_route_q(critic, policy, chunk.centralized_batch.features, task,
                            config.training.mappo.route_counterfactual_credit_coefficient,
                            actor_credit_source=config.training.mappo.route_actor_credit_source)
    return config, payload, actor, critic, chunk, policy, batch


def layer_probe(q, state, candidates, legal, chosen, targets):
    s = state.detach().unsqueeze(-2).expand(*legal.shape, state.shape[-1])[legal]
    c = candidates.detach()[legal]
    result = {"raw_state_norm": mean_row_norm(s), "raw_candidate_norm": mean_row_norm(c)}
    if hasattr(q, "preprocess"):
        ps, pc = q.preprocess(s, c)
    else:
        ps, pc = s, c
    result.update(preprocessed_state_norm=mean_row_norm(ps), preprocessed_candidate_norm=mean_row_norm(pc))
    if getattr(q, "input_mode", "legacy") == "dual_branch":
        a, b = q.state_projection(ps), q.candidate_projection(pc)
        # Type bypass is explicit and is not part of either learned projection.
        inputs = torch.cat((a, b, pc[..., -3:]), -1)
        result.update(projected_state_norm=mean_row_norm(a), projected_candidate_norm=mean_row_norm(b))
        for name, branch, value in (("state", q.state_projection, ps), ("candidate", q.candidate_projection, pc)):
            linear = branch[0](value); activated = branch[1](linear)
            result[name+"_branch_linear_norm"] = mean_row_norm(linear)
            result[name+"_branch_activation_norm"] = mean_row_norm(activated)
        w = q.network[0].weight; width = a.shape[-1]
        a = nn.functional.linear(a, w[:, :width])
        b = nn.functional.linear(inputs[..., width:], w[:, width:])
        result.update(fusion_state_contribution_norm=mean_row_norm(a), fusion_candidate_contribution_norm=mean_row_norm(b))
    else:
        inputs = torch.cat((ps, pc), -1)
        w = q.network[0].weight
        a = nn.functional.linear(ps, w[:, :s.shape[-1]])
        b = nn.functional.linear(pc, w[:, s.shape[-1]:])
        result.update(projected_state_norm=mean_row_norm(a), projected_candidate_norm=mean_row_norm(b))
    state_only = a + q.network[0].bias
    act = q.network[1](state_only + b); base = q.network[1](state_only)
    ln = q.network[2](act); ln_base = q.network[2](base)
    result.update(activation_total_norm=mean_row_norm(act),
        activation_state_only_norm=mean_row_norm(base),
        activation_candidate_delta_norm=mean_row_norm(act-base),
        layernorm_total_norm=mean_row_norm(ln), layernorm_state_only_norm=mean_row_norm(ln_base),
        layernorm_candidate_delta_norm=mean_row_norm(ln-ln_base),
        nonlinear_decomposition_note="total and state-only counterfactual difference; nonlinear components are not independently additive")
    prediction = q(state, candidates, legal)
    rr = [prediction[i][legal[i]].max()-prediction[i][legal[i]].min()
          for i in range(len(legal)) if legal[i].any()]
    error = prediction.gather(-1, chosen[:, None]).squeeze(-1)-targets
    loss = .5*error.square().mean()
    params = list(q.named_parameters())
    grads = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True)
    result.update(q=distribution(prediction[legal]), legal_range=distribution(torch.stack(rr)),
        chosen_mae=float(error.detach().abs().mean()), chosen_rmse=float(error.detach().square().mean().sqrt()),
        loss=float(loss.detach()), parameter_grad_norms={k: norm(g) if g is not None else None
                                                      for (k, _), g in zip(params, grads)})
    factual = prediction.gather(-1, chosen[:, None]).squeeze(-1).detach()
    result["chosen_by_type"] = {}
    for category, select in (("local", chosen == 1), ("defer", chosen == 2), ("remote", chosen >= 3)):
        result["chosen_by_type"][category] = {"target": distribution(targets[select]),
            "prediction": distribution(factual[select]), "error": distribution(error[select]),
            "mae": float(error[select].detach().abs().mean()) if select.any() else None,
            "rmse": float(error[select].detach().square().mean().sqrt()) if select.any() else None}
    if getattr(q, "input_mode", "legacy") != "dual_branch":
        g = dict(zip((k for k, _ in params), grads))["network.0.weight"]
        result["first_layer_gradient_columns"] = {"state": norm(g[:, :state.shape[-1]]),
                                                    "candidate": norm(g[:, state.shape[-1]:])}
    swapped = candidates.detach().clone()
    for i in range(len(legal)):
        swapped[i, legal[i]] = candidates[i, legal[i]].detach().roll(1, dims=0)
    # Response only, NEVER interpreted as unchosen accuracy.
    result["candidate_swap_response"] = distribution((q(state, swapped, legal)-prediction)[legal])
    return result


def synthetic_fit(mode, *, steps=240, seed=42):
    """Disposable Q-only supervised problem with known candidate labels.

    Targets are shuffled per state; there is no preferred action type. Shared
    state is identical within each candidate set. No MEC optimality claim.
    """
    from src.models.route_option_q import RouteOptionQ
    from src.models.route_q_inputs import StateField
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        state = torch.randn(32, 12)
        state[:, :4] *= 50000
        candidates = torch.randn(32, 5, 11)*.1
        targets = torch.stack([torch.linspace(-1, 1, 5)[torch.randperm(5)] for _ in range(32)])
        candidates[..., 0] = targets
        candidates[..., -3:] = nn.functional.one_hot(torch.arange(5) % 3, 3).float()
        legal = torch.ones(32, 5, dtype=torch.bool)
        fields = (StateField("heavy_history", 0, 4, "ratio", signed_log=True), StateField("scaled", 4, 12, "ratio"))
        q = RouteOptionQ(12, 11, 32, seed, input_mode=mode, state_fields=fields)
        initial = {k: p.detach().clone() for k, p in q.named_parameters()}
        optimizer = torch.optim.Adam(q.parameters(), lr=.01)
        initial_q = q(state, candidates, legal).detach()
        first_gradient = None
        for step in range(steps):
            # All candidate labels exist ONLY in this synthetic dataset. Each
            # row is equivalently an independent chosen-option observation.
            prediction = q(state, candidates, legal)
            loss = .5*(prediction-targets).square().mean()
            optimizer.zero_grad(); loss.backward()
            if step == 1:
                first_gradient = {k: norm(p.grad) if p.grad is not None else None for k, p in q.named_parameters()}
            optimizer.step()
        prediction = q(state, candidates, legal).detach()
        pair = targets.unsqueeze(-1)-targets.unsqueeze(-2)
        predicted_pair = prediction.unsqueeze(-1)-prediction.unsqueeze(-2)
        valid = pair > 0
        held_state = torch.randn(16, 12); held_state[:, :4] *= 50000
        held_candidates = torch.randn(16, 5, 11)*.1
        held_targets = torch.rand(16, 5)*2-1
        held_candidates[..., 0] = held_targets
        held_candidates[..., -3:] = candidates[:16, :, -3:]
        held_prediction = q(held_state, held_candidates, legal[:16]).detach()
        hp = held_targets.unsqueeze(-1)-held_targets.unsqueeze(-2)
        pp = held_prediction.unsqueeze(-1)-held_prediction.unsqueeze(-2)
        by_type = {str(t): {"targets": distribution(targets[:, torch.arange(5)%3 == t]),
                           "error": distribution((prediction-targets)[:, torch.arange(5)%3 == t])}
                   for t in range(3)}
        return {"scope": "synthetic Q-only fit; disposable model, NOT MEC training or counterfactual validation",
            "seed": seed, "steps": steps, "initial_q_range": float(initial_q.max()-initial_q.min()),
            "rmse": float((prediction-targets).square().mean().sqrt()),
            "pairwise_accuracy": float((predicted_pair[valid] > 0).float().mean()),
            "mean_legal_range": float((prediction.max(-1).values-prediction.min(-1).values).mean()),
            "heldout_rmse": float((held_prediction-held_targets).square().mean().sqrt()),
            "heldout_pairwise_accuracy": float((pp[hp > 0] > 0).float().mean()),
            "action_types": by_type,
            "second_step_gradients": first_gradient,
            "parameter_update_norms": {k: norm(p-initial[k]) for k, p in q.named_parameters()}}


def actor_cf_probe(actor, chunk, policy, batch, config):
    """Actual autograd on saved factual-action log-probs; no actor step.

    Route surrogate normalized by Route-active samples only. Production also
    reduces over other active components; this is an explicitly isolated term.
    """
    ii = [s["rollout_offset"] for s in batch.credit_samples]
    uu = [s["source_uav"] for s in batch.credit_samples]
    new = policy.branch_log_probs["route"].reshape(-1, actor.spec.uav_count)[ii, uu]
    old = chunk.old_branch_log_probs[..., 0][ii, uu]
    ratio = (new-old).exp()
    cf = batch.counterfactual_advantage.detach()*config.training.mappo.route_counterfactual_credit_coefficient
    clipped = ratio.clamp(1-config.training.mappo.ppo_clip_epsilon, 1+config.training.mappo.ppo_clip_epsilon)
    loss = -torch.minimum(ratio*cf, clipped*cf).mean()
    params = [(k, p) for k, p in actor.named_parameters() if k.startswith(actor.route_parameter_prefixes)]
    grads = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True)
    flags = torch.tensor([t.rejection_or_downgrade_summary["policy_components_v1"]["active"]
                          for t in chunk.transitions], dtype=torch.bool)
    divisor = int((flags.sum((0, 1)) > 0).sum())
    grad_norm = sum(norm(g)**2 for g in grads if g is not None)**.5
    return {"scope": "isolated CF-only clipped Route surrogate, Route-active mean; saved 4k training tensors",
            "loss": float(loss.detach()), "gradient_norm": grad_norm,
            "partial_batch_active_component_divisor": divisor,
            "production_reduction_cf_gradient_norm_on_partial_batch": grad_norm/divisor,
            "production_reduction_note": "branch means averaged over originally active components; all replay ratios inside clip, hence CF contribution is additive here; not a historical full-update gradient",
            "parameter_grad_norms": {k: norm(g) if g is not None else None for (k, _), g in zip(params, grads)},
            "ratio": distribution(ratio), "real_actor_step": False,
            "shared_gradient_comparison": "not available: saved partial rollout has no frozen PPO GAE"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--training-log")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--synthetic-steps", type=int, default=240)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--final-checkpoint", help="optional same-run final weights for longitudinal Q parameter delta only")
    parser.add_argument("--synthetic-only", action="store_true", help="no checkpoint/config/trajectory needed")
    args = parser.parse_args(argv)
    if args.synthetic_steps <= 0 or args.synthetic_steps > 1000:
        parser.error("synthetic steps must be in [1,1000]; this is a bounded diagnostic")
    if not args.synthetic_only and not all((args.config, args.checkpoint, args.training_log)):
        parser.error("saved-tensor audit requires --config, --checkpoint and --training-log")
    torch.set_num_threads(1)
    rng = torch.get_rng_state().clone()
    if args.synthetic_only:
        result = {"scope": "synthetic tensor-only diagnostic; no environment, real checkpoint or MEC trajectory",
                  "synthetic": {m: synthetic_fit(m, steps=args.synthetic_steps) for m in ("legacy", "normalized", "dual_branch")}}
        out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
        (out/"route_q_probe.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        torch.set_rng_state(rng)
        print(json.dumps({"output": str(out/"route_q_probe.json"), "scope": result["scope"]}))
        return
    config, payload, actor, critic, chunk, policy, batch = load_saved_tensors(args.config, args.checkpoint)
    result = {"scope": "offline existing training tensors only; no environment, rollout, evaluation or real-weight update",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "saved_transition_count": chunk.length, "active_route_samples": len(batch.targets),
        "state_shape": list(batch.state.shape), "candidate_shape": list(batch.candidates.shape),
        "legal_shape": list(batch.legal.shape), "training_audit": training_audit(args.training_log),
        "legacy": layer_probe(critic.route_option_q, batch.state, batch.candidates, batch.legal, batch.chosen, batch.targets),
        "actor_cf": actor_cf_probe(actor, chunk, policy, batch, config)}
    from src.models.route_q_inputs import centralized_q_fields
    fields = centralized_q_fields(config)
    result["state_fields"] = [{**f.__dict__, "observed": distribution(batch.state[..., f.start:f.stop])} for f in fields]
    result["candidate_blocks"] = {"endpoint_task_source": [0, 204], "legal_remote_mean": [204, 408],
        "type_Local_Defer_Remote": [408, 411], "absolute_destination_id_onehot": False,
        "destination_identity": "existing destination-specific embedding and source-dependent ordered leaf mapping"}
    if not args.audit_only:
        from src.models.route_option_q import RouteOptionQ
        result["synthetic"] = {m: synthetic_fit(m, steps=args.synthetic_steps) for m in ("legacy", "normalized", "dual_branch")}
        # P0a retains the original MLP, allowing a controlled same-weight input
        # intervention. This is NOT a trained P0a checkpoint or eval result.
        normalized = RouteOptionQ(batch.state.shape[-1], batch.candidates.shape[-1],
                                  config.training.mappo.encoder_hidden_dimension, config.seed,
                                  input_mode="normalized", state_fields=fields)
        normalized.network.load_state_dict(critic.route_option_q.network.state_dict())
        result["normalized_same_weight_intervention"] = layer_probe(normalized, batch.state, batch.candidates,
                                                                    batch.legal, batch.chosen, batch.targets)
        result["normalized_same_weight_intervention"]["scope"] = "trained legacy weights under changed inputs; not refitted or calibrated"
        dual = RouteOptionQ(batch.state.shape[-1], batch.candidates.shape[-1],
                            config.training.mappo.encoder_hidden_dimension, config.seed,
                            input_mode="dual_branch", state_fields=fields)
        result["dual_untrained"] = layer_probe(dual, batch.state, batch.candidates, batch.legal, batch.chosen, batch.targets)
        result["dual_untrained"]["scope"] = "zero-output initialized new model; zero candidate spread is expected before fitting"
    if args.final_checkpoint:
        final = torch.load(args.final_checkpoint, map_location="cpu", weights_only=False)
        if final["config_hash"] != payload["config_hash"]:
            raise ValueError("longitudinal checkpoint config mismatch")
        result["historical_q_parameter_delta_4k_to_final"] = {
            k: norm(final["model_state"]["critic"]["route_option_q."+k]-v)
            for k, v in critic.route_option_q.state_dict().items()}
        result["historical_gradient_limitation"] = "per-update historical gradients unavailable; these are cumulative parameter deltas, not recovered gradients"
    torch.set_rng_state(rng)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out/"route_q_probe.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(out/"route_q_probe.json"), "samples": len(batch.targets), "legacy": result["legacy"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
