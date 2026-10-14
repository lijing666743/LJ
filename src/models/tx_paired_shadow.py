"""E13 detached paired credit, bounded deterministic selection and shadows.

No shadow transition is returned to the rollout collector. Completed labels and
selector accounting use existing JSON-safe transition metadata, so an episode-
boundary checkpoint with a partial rollout contains all necessary E13 state.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

MODE = "paired_shadow_v1"
SCHEMA = "tx_paired_shadow.v1"
KEY = "tx_paired_shadow"


def selector_sources(version: int, agents: int, maximum: int) -> tuple[int, ...]:
    return tuple((version + index) % agents for index in range(maximum))


def selection_state(buffer: Any, config: Any, version: int) -> tuple[set[int], int]:
    """Recover stateless selector counters from safely persisted factual rows."""
    sources: set[int] = set()
    spent = 0
    m = config.training.mappo
    for index in range(len(buffer)):
        transition = buffer.transition_at(index)
        record = transition.rejection_or_downgrade_summary.get(KEY)
        if not isinstance(record, dict) or record.get("schema") != SCHEMA:
            raise ValueError("E13 rollout requires explicit paired metadata, never fallback")
        if record.get("policy_version") != version or record.get("rollout_offset") != index:
            raise ValueError("paired selector version/offset mismatch")
        mask = record.get("paired_mask")
        if not isinstance(mask, list) or len(mask) != config.environment.uav_count or any(type(x) is not bool for x in mask):
            raise ValueError("explicit paired mask must be boolean [A]")
        rows = record.get("samples")
        if not isinstance(rows, list):
            raise ValueError("paired metadata samples missing")
        observed: set[int] = set()
        for row in rows:
            source = row.get("source")
            if (type(source) is not int or source in sources or source in observed
                    or source not in selector_sources(version, config.environment.uav_count, m.tx_shadow_max_pairs_per_rollout)):
                raise ValueError("invalid/duplicate paired selector source")
            if row.get("slot") != transition.slot or not mask[source] or not transition.action_mask_contracts[source].tx_branch_active:
                raise ValueError("paired row is not a legal active source")
            if type(row.get("y")) is not bool or row["y"] != (transition.proposal_actions[source].tx_select != "idle"):
                raise ValueError("paired y differs from real factual proposal")
            p, gt, gi = (row.get(key) for key in ("old_p", "G_T", "G_I"))
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (p, gt, gi)) or not 0 <= p <= 1:
                raise ValueError("invalid paired old p/returns")
            d = gt - gi
            raw = (float(row["y"]) - p) * d
            if (row.get("lifecycle_join_status") != "PASS" or row.get("paired_mask") is not True
                    or not math.isclose(d, row.get("d", math.inf), abs_tol=1e-10)
                    or not math.isclose(raw, row.get("A_TX_raw", math.inf), abs_tol=1e-10)):
                raise ValueError("paired label invariant failed")
            cost = row.get("shadow_steps")
            if type(cost) is not int or cost != 2 * (config.environment.episode_horizon - transition.slot):
                raise ValueError("paired shadow budget accounting invalid")
            observed.add(source)
            spent += cost
        if {i for i, active in enumerate(mask) if active} != observed:
            raise ValueError("paired mask/sample rows disagree")
        sources.update(observed)
    if len(sources) > m.tx_shadow_max_pairs_per_rollout or spent > m.tx_shadow_step_budget_per_rollout:
        raise ValueError("paired rollout budget exceeded")
    return sources, spent


def select_sources(buffer: Any, config: Any, version: int, env: Any, *,
                   state: tuple[frozenset[int], int] | None = None) -> tuple[int, ...]:
    # The collector can pass its validated incremental state. Checkpoint restore
    # and credit construction still validate every persisted factual row.
    selected, spent = selection_state(buffer, config, version) if state is None else state
    m = config.training.mappo
    result = []
    cost = 2 * (config.environment.episode_horizon - env.slot)
    for source in selector_sources(version, config.environment.uav_count, m.tx_shadow_max_pairs_per_rollout):
        if (source not in selected and env.current_observations[source].action_masks.tx_branch_active
                and spent + cost <= m.tx_shadow_step_budget_per_rollout):
            result.append(source)
            spent += cost
    return tuple(result)


@dataclass(frozen=True)
class PairedCredit:
    old_p: Tensor
    transmitted: Tensor
    g_t: Tensor
    g_i: Tensor
    d: Tensor
    raw_advantage: Tensor
    normalized_advantage: Tensor
    active: Tensor
    scale: float

    def telemetry(self) -> dict[str, Any]:
        active = self.active
        def summary(value, mask=active):
            values = value.detach()[mask].double()
            return {"mean": float(values.mean()) if values.numel() else None,
                    "std": float(values.std(unbiased=False)) if values.numel() else None}
        values = self.d[active].double()
        score = self.transmitted.to(self.old_p.dtype) - self.old_p
        return {"credit_mode": MODE, "paired_sample_count": int(active.sum()),
                "d_mean": float(values.mean()) if values.numel() else None,
                "d_median": float(values.quantile(.5)) if values.numel() else None,
                "d_std": float(values.std(unbiased=False)) if values.numel() else None,
                "d_positive": int((values > 0).sum()), "d_negative": int((values < 0).sum()),
                "d_zero": int((values == 0).sum()),
                "A_given_T_raw": summary(self.raw_advantage, active & self.transmitted),
                "A_given_I_raw": summary(self.raw_advantage, active & ~self.transmitted),
                "A_given_T_normalized": summary(self.normalized_advantage, active & self.transmitted),
                "A_given_I_normalized": summary(self.normalized_advantage, active & ~self.transmitted),
                "raw_signed_score_mean": summary(self.raw_advantage * score)["mean"],
                "signed_score_mean": summary(self.normalized_advantage * score)["mean"],
                "normalization": "paired_rms_v1", "normalization_scale": self.scale,
                "credit_frozen_for_update": True, "q_status": "inactive", "factual_gae_status": "inactive"}


@torch.no_grad()
def paired_credit(g_t: Tensor, g_i: Tensor, old_p: Tensor, transmitted: Tensor,
                  paired_mask: Tensor, tx_active: Tensor, valid: Tensor) -> PairedCredit:
    if (not old_p.is_floating_point() or g_t.dtype != old_p.dtype or g_i.dtype != old_p.dtype
            or any(x.shape != old_p.shape or x.device != old_p.device for x in (g_t, g_i, transmitted, paired_mask, tx_active))
            or any(x.dtype != torch.bool for x in (transmitted, paired_mask, tx_active, valid))
            or valid.shape != old_p.shape[:-1] or valid.device != old_p.device):
        raise ValueError("paired credit input axes/dtypes/devices differ")
    allowed = tx_active & valid.unsqueeze(-1)
    if (paired_mask & ~allowed).any():
        raise ValueError("paired mask contains inactive/invalid factual samples")
    active = paired_mask.detach().clone()
    if (not all(torch.isfinite(x[active]).all() for x in (g_t, g_i, old_p))
            or ((old_p[active] < 0) | (old_p[active] > 1)).any()):
        raise ValueError("invalid paired returns/probability")
    # NaN outside paired mask means absent label, not a zero-advantage sample.
    def absent(value):
        return torch.where(active, value.detach(), torch.full_like(value, float("nan"))).clone()
    gt, gi, p = absent(g_t), absent(g_i), absent(old_p)
    y = transmitted.detach().clone()
    d = gt - gi
    raw = (y.to(p.dtype) - p) * d
    # RMS-only scaling preserves action-contrast sign, including a singleton
    # or all-same-action cohort. No mean centering that can flip weak labels.
    scale = max(float(raw[active].double().square().mean().sqrt()) if active.any() else 0., 1e-8)
    normalized = raw / scale
    if not torch.isfinite(raw[active]).all() or not torch.isfinite(normalized[active]).all():
        raise ValueError("nonfinite paired advantage")
    return PairedCredit(p, y, gt, gi, d, raw, normalized, active, scale)


def credit_from_buffer(buffer: Any, config: Any, device: Any, tx_active: Tensor,
                       valid: Tensor) -> tuple[PairedCredit, list[dict[str, Any]]]:
    first = buffer.transition_at(0).rejection_or_downgrade_summary.get(KEY, {})
    selection_state(buffer, config, first.get("policy_version"))
    shape = tx_active.shape
    # Preserve small return contrasts before PPO's float32 arithmetic. Rounding
    # each large G arm to float32 first could silently erase a real small d.
    arrays = [torch.full(shape, float("nan"), device=device, dtype=torch.float64) for _ in range(3)]
    y = torch.zeros(shape, dtype=torch.bool, device=device)
    mask = torch.zeros_like(y)
    records = []
    for index in range(len(buffer)):
        transition = buffer.transition_at(index)
        for row in transition.rejection_or_downgrade_summary[KEY]["samples"]:
            source = row["source"]
            position = (index // 32, index % 32, source)
            for array, key in zip(arrays, ("G_T", "G_I", "old_p")):
                array[position] = row[key]
            y[position], mask[position] = row["y"], True
            # Check old binary p against the real selected log probability.
            metadata = transition.rejection_or_downgrade_summary["policy_components_v1"]
            old_log = metadata["log_probs"][source][metadata["names"].index("tx_upper")]
            expected = row["old_p"] if row["y"] else 1 - row["old_p"]
            if not math.isclose(math.exp(old_log), expected, rel_tol=1e-5, abs_tol=1e-7):
                raise ValueError("paired old probability differs from factual old log-prob")
            records.append({**row, "rollout_offset": index})
    credit = paired_credit(*arrays, y, mask, tx_active, valid)
    for row in records:
        position = (row["rollout_offset"] // 32, row["rollout_offset"] % 32, row["source"])
        raw = float(credit.raw_advantage[position])
        norm = float(credit.normalized_advantage[position])
        score = float(row["y"]) - row["old_p"]
        row.update(A_TX_raw=raw, A_TX_normalized=norm, binary_score=score,
                   raw_signed_score=raw * score, signed_score=norm * score)
    return credit, records


class FrozenShadowSampler:
    def __init__(self, actor: Any, config: Any, version: int) -> None:
        from .ca_gat_mappo import ActorObservationTensorizer
        from .ca_gat_mappo_actions import CAGATMAPPOActionDistribution
        self.actor = copy.deepcopy(actor).eval().requires_grad_(False)
        for module in self.actor.modules():
            if isinstance(module, torch.nn.RNNBase):
                module.flatten_parameters()
        self.device = next(self.actor.parameters()).device
        self.tensorizer = ActorObservationTensorizer(config)
        self.distribution = CAGATMAPPOActionDistribution(self.actor, config)
        self.config, self.version = config, version

    def seed(self, env: Any, namespace: str, index: int = 0) -> int:
        raw = f"E13:{self.config.training.mappo.tx_shadow_rng_seed}:{self.version}:{env.config.seed}:{env.slot}:{namespace}:{index}"
        return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "little") % (2**63 - 1)

    @torch.no_grad()
    def sample(self, env: Any, hidden: Tensor, namespace: str = "continuation", index: int = 0):
        from .ca_gat_mappo_actions import SequentialActionMaskBatch
        from ..evaluation.route_oracle import tensor_digest
        generator = torch.Generator(device=self.device).manual_seed(self.seed(env, namespace, index))
        before = tensor_digest(generator.get_state())
        batch = self.tensorizer.encode_step(env.current_observations, device=self.device,
                                            dtype=torch.float32, episode_start=env.slot == 0)
        output = self.distribution.sample_actions(batch, SequentialActionMaskBatch.from_observations(env.current_observations),
                                                  hidden.detach().clone().to(self.device), generator=generator)
        return tuple(output.proposals[0][0]), output.hidden_out.detach().clone(), (before, tensor_digest(generator.get_state()))

    def transmit(self, env: Any, hidden: Tensor, factual: tuple[Any, ...], source: int):
        from dataclasses import replace
        if factual[source].tx_select != "idle":
            return factual[source], 0
        for draw in range(1, self.config.training.mappo.tx_shadow_max_conditional_draws + 1):
            actions, _, _ = self.sample(env, hidden, f"conditional:{source}", draw)
            proposal = actions[source]
            if proposal.tx_select != "idle":
                return replace(factual[source], tx_select=proposal.tx_select, resource_group=proposal.resource_group,
                               resource_width=proposal.resource_width, power_level=proposal.power_level), draw
        raise ValueError("E13 conditional T sampling exhausted; no fallback")


def run_pair(env: Any, sampler: FrozenShadowSampler, hidden_in: Tensor, hidden_out: Tensor,
             factual: tuple[Any, ...], source: int, old_p: float, prefix: Any) -> dict[str, Any]:
    """Atomic complete pair. The caller has reserved exactly 2*(H-slot) steps."""
    from ..evaluation.tx_paired_shadow_pilot import (make_arms, reward_check, exogenous_digest, intrinsic_tasks, digest)
    from ..evaluation.route_oracle import actor_state_digest, tensor_digest
    started = time.perf_counter()
    start = env.slot
    initial = (env.state_fingerprint(), env.rng_fingerprint(), tensor_digest(hidden_in), tensor_digest(hidden_out))
    actor_digest = actor_state_digest(sampler.actor)
    transmit, draws = sampler.transmit(env, hidden_in, factual, source)
    action_i, action_t = make_arms(env, factual, source, transmit)
    arms = [env.clone_for_shadow(), env.clone_for_shadow()]
    hiddens = [hidden_out.detach().clone(), hidden_out.detach().clone()]
    ledgers = [copy.deepcopy(prefix), copy.deepcopy(prefix)]
    rewards, components = [[], []], [{}, {}]
    first_info = None
    matched = int(factual[source].tx_select != "idle")
    for slot in range(start, env.config.environment.episode_horizon):
        if exogenous_digest(arms[0]) != exogenous_digest(arms[1]):
            raise ValueError("paired prestep exogenous mismatch")
        rngs = []
        for arm, shadow in enumerate(arms):
            if slot == start:
                actions = (action_i, action_t)[arm]
                rng = None
            else:
                actions, hiddens[arm], rng = sampler.sample(shadow, hiddens[arm])
            if not all(obs.action_masks.is_legal(action) for obs, action in zip(shadow.current_observations, actions)):
                raise ValueError("illegal paired continuation")
            result = shadow.step(actions)
            terms = reward_check(result)
            ledgers[arm].update(shadow, result)
            discount = sampler.config.training.mappo.gamma ** (slot - start)
            rewards[arm].append(discount * result.reward)
            for key, value in terms.items():
                components[arm].setdefault(key, []).append(discount * value)
            if slot == start and arm == matched:
                first_info = digest(result.info)
            rngs.append(rng)
        if rngs[0] != rngs[1] or exogenous_digest(arms[0]) != exogenous_digest(arms[1]):
            raise ValueError("paired external/policy randomness misalignment")
    totals = [math.fsum(values) for values in rewards]
    signed = {key: sign * (math.fsum(components[1][key]) - math.fsum(components[0][key]))
              for key, sign in (("completion_component", 1), ("expiration_penalty", -1),
                                ("workload_penalty", -1), ("energy_penalty", -1))}
    if not math.isclose(totals[1] - totals[0], math.fsum(signed.values()), abs_tol=1e-9):
        raise ValueError("paired discounted reward nonconservation")
    # Identify the actual legal slot-start TX queue head, not a best task.
    focal_id = env.lifecycle.queues.tx[(source, transmit.tx_select)].peek().task_id
    joins = [ledger.validate(shadow) for ledger, shadow in zip(ledgers, arms)]
    if intrinsic_tasks(arms[0]) != intrinsic_tasks(arms[1]):
        raise ValueError("paired task identity mismatch")
    if initial != (env.state_fingerprint(), env.rng_fingerprint(), tensor_digest(hidden_in), tensor_digest(hidden_out)):
        raise ValueError("shadow mutated factual state/hidden/RNG")
    if actor_state_digest(sampler.actor) != actor_digest:
        raise ValueError("shadow mutated frozen actor")
    y = factual[source].tx_select != "idle"
    d = totals[1] - totals[0]
    return {"source": source, "slot": start, "y": y, "old_p": old_p,
            "G_T": totals[1], "G_I": totals[0], "d": d, "A_TX_raw": (float(y) - old_p) * d,
            "paired_mask": True, "lifecycle_join_status": "PASS", "focal_task_id": focal_id,
            "focal_lifecycle": {name: next(task for task in arm_join if task["task_id"] == focal_id)
                                for name, arm_join in zip(("I", "T"), joins)},
            "reward_delta_components": signed, "shadow_steps": 2 * (env.config.environment.episode_horizon - start),
            "conditional_draws": draws, "factual_first_step_digest": first_info,
            "initial_state_digest": initial[0], "initial_rng_digest": initial[1],
            "common_randomness_status": "PASS", "state_isolation_status": "PASS",
            "shadow_wall_seconds": time.perf_counter() - started}
