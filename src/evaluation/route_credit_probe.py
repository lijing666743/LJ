"""Read-only Route credit gradients on saved factual training tensors.

Never creates an environment, samples an action or executes optimizer.step.
Historical log advantages are required: partial-rollout GAE is not fabricated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .route_q_probe import load_saved_tensors, norm


def credit_gradient_diagnostics(new, old, shared, task, cf, parameters, epsilon=.2):
    """Isolated Route-active mean surrogate, before global gradient clipping."""
    params = tuple(parameters)
    ratio = (new-old.detach()).exp()
    clipped = ratio.clamp(1-epsilon, 1+epsilon)
    components = {"shared": shared, "task": task, "CF": cf, "combined": shared+task}
    vectors = {}; result = {}
    for name, advantage in components.items():
        advantage = advantage.detach().to(new)
        if not torch.isfinite(advantage).all():
            raise ValueError("non-finite saved credit")
        loss = -torch.minimum(ratio*advantage, clipped*advantage).mean()
        gradients = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        vector = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                            for p, g in zip(params, gradients)]).detach().double()
        vectors[name] = vector
        result[name] = {"loss": float(loss.detach()), "gradient_norm": norm(vector),
                        "advantage_rms": float(advantage.square().mean().sqrt())}
    def quotient(a, b):
        denominator = norm(vectors[b])
        return norm(vectors[a])/denominator if denominator else None
    def cosine(a, b):
        denominator = norm(vectors[a])*norm(vectors[b])
        return float(torch.dot(vectors[a], vectors[b]))/denominator if denominator else None
    result.update(shared_to_task_gradient_norm=quotient("shared", "task"),
                  task_to_shared_gradient_norm=quotient("task", "shared"),
                  CF_to_shared_gradient_norm=quotient("CF", "shared"),
                  cosines={a+"_"+b: cosine(a, b) for a, b in (
                      ("shared", "task"), ("shared", "CF"),
                      ("shared", "combined"), ("task", "combined"))},
                  combined_minus_shared_task_norm=norm(vectors["combined"]-vectors["shared"]-vectors["task"]),
                  ratio_min=float(ratio.min().detach()), ratio_max=float(ratio.max().detach()),
                  outside_clip_count=int(((ratio < 1-epsilon) | (ratio > 1+epsilon)).sum()),
                  optimizer_step=False)
    return result


def saved_credit_probe(config_path, checkpoint, training_log):
    config, payload, actor, critic, chunk, policy, _ = load_saved_tensors(config_path, checkpoint)
    start = payload["training_state"]["collected_environment_transitions"]-chunk.length
    records = []
    for line in Path(training_log).open(encoding="utf-8"):
        row = json.loads(line)
        if row["config_hash"] != payload["config_hash"]:
            raise ValueError("training log identity differs from checkpoint")
        records.extend(s for s in row["samples"] if s["epoch_index"] == 0
                       and start <= s["global_rollout_index"] < start+chunk.length)
    records.sort(key=lambda s: (s["global_rollout_index"], s["source_uav"]))
    keys = [(s["global_rollout_index"], s["source_uav"]) for s in records]
    if not records or len(set(keys)) != len(keys):
        raise ValueError("missing or duplicate saved credit records")
    if len(records) != int(chunk.active_branch_indicators[..., 0].sum()):
        raise ValueError("saved Route credit coverage differs from factual batch")
    ii = [s["global_rollout_index"]-start for s in records]
    uu = [s["source_uav"] for s in records]
    new = policy.branch_log_probs["route"].reshape(-1, actor.spec.uav_count)[ii, uu]
    old = chunk.old_branch_log_probs[..., 0][ii, uu]
    logged_old = old.new_tensor([s["old_log_prob"] for s in records])
    if not torch.allclose(new.detach(), logged_old, atol=2e-6, rtol=0) or not torch.equal(old, logged_old):
        raise ValueError("checkpoint policy differs from recorded old policy")
    parameters = [p for k, p in actor.named_parameters() if k.startswith(actor.route_parameter_prefixes)]
    alpha = config.training.mappo.route_task_credit_coefficient
    beta = config.training.mappo.route_counterfactual_credit_coefficient
    result = credit_gradient_diagnostics(new, old,
        old.new_tensor([s["shared_advantage"] for s in records]),
        old.new_tensor([alpha*s["task_advantage"] for s in records]),
        old.new_tensor([beta*s["counterfactual_advantage"] for s in records]), parameters,
        config.training.mappo.ppo_clip_epsilon)
    flags = torch.tensor([t.rejection_or_downgrade_summary["policy_components_v1"]["active"]
                          for t in chunk.transitions], dtype=torch.bool)
    divisor = int((flags.sum((0, 1)) > 0).sum())
    result.update(scope="saved factual partial rollout with actual next-update frozen credit; not full historical minibatch",
                  source_actor_credit=config.training.mappo.route_actor_credit_source,
                  combined_definition="shared + alpha * task (diagnostic P1a loss on saved actor)",
                  checkpoint=str(Path(checkpoint).resolve()),
                  checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                  transitions=chunk.length, route_samples=len(records),
                  environment_step_start=start, environment_step_end=start+chunk.length,
                  alpha=alpha, beta=beta, active_component_divisor=divisor,
                  production_partial_gradient_norms={k: result[k]["gradient_norm"]/divisor
                                                     for k in ("shared", "task", "CF", "combined")})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("use a new diagnostic output file")
    result = saved_credit_probe(args.config, args.checkpoint, args.training_log)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
