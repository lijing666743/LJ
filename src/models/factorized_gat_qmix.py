"""Recurrent factorized Route/TX and joint CPU utilities with a QMIX mixer.

Only actor-safe tensor adapters and the CA-GATv2 layer are reused from MAPPO.
No actor parameters, logits distribution, critic, PPO, GAE, or shadow credit
are used. Queue-conditioned frequency scores express arbitrary CPU pair
preferences. Execution, teacher replay and TD targets use the same utility
and exact conditional suffix maximisation over the original legal actions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import RunConfig
from ..env.actions import ActionProposal
from ..env.observation import ActionMasks, BRANCH_ORDER
from .ca_gat_mappo import ActorTensorBatch, CAGATv2Layer, MAPPOTensorSpec, _derived_torch_seed


METHOD_ID = "factorized_action_gat_qmix"
QMIX_METHOD_IDS = (METHOD_ID, "c_feature_credit_qmix")
ACTOR_EDGE_TRANSFORM = "signed_log1p_v1"
CPU_UTILITY_CONTRACT = {
    "kind": "queue_conditioned_frequency_pair_q_v1",
    "utility": "queue_q_plus_selected_queue_frequency_q",
    "idle": "queue_q_only",
    "rows": "nonidle_cpu_queue_domain_order",
    "frequencies": "original_full_domain_including_zero",
    "legality": "original_conditional_masks_unchanged",
}
EXPLORATION_CONTRACT = {
    "trigger": "one_bernoulli_per_batch_time_agent_v1",
    "random_action": "conditional_mask_uniform_sequential_v1",
    "greedy_action": "exact_route_tx_cpu_suffix_max_v1",
    "rng": "private_qmix_epsilon_stream",
}


def stabilize_edge_features(features: Tensor) -> Tensor:
    """Compress actor-safe channel ratios before graph attention, without fitting.

    Interference/noise and historical quality ratios can be orders of magnitude
    larger than private queue features. LayerNorm after graph aggregation does
    not undo their domination of attention and the local residual. This fixed,
    elementwise, sign-preserving transform is shared by collection, replay and
    target/evaluation forwards; raw observations and saved replay stay unchanged.
    """
    return features.sign() * torch.log1p(features.abs())


def validate_qmix_config(config: RunConfig) -> None:
    config.validate()
    if config.method_id not in QMIX_METHOD_IDS:
        raise ValueError("QMIX requires its own method_id")
    q = config.training.qmix
    import math
    for name in (
        "encoder_hidden_dimension", "gat_layer_count", "attention_head_count",
        "gru_hidden_dimension", "replay_capacity_sequences", "recurrent_sequence_length_slots",
        "sequence_batch_size", "epsilon_decay_steps", "target_update_interval_updates",
        "max_training_episodes", "max_training_environment_steps", "checkpoint_interval_steps",
    ):
        if type(getattr(q, name)) is not int or getattr(q, name) <= 0:
            raise ValueError(f"QMIX {name} must be a positive integer")
    for name in ("learning_rate", "gamma", "epsilon_start", "epsilon_end"):
        value = getattr(q, name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"QMIX {name} must be finite numeric")
        if name == "learning_rate" and value <= 0:
            raise ValueError("QMIX learning rate must be positive")
        if name != "learning_rate" and not 0 <= value <= 1:
            raise ValueError(f"QMIX {name} must lie in [0,1]")
    if q.epsilon_start < q.epsilon_end or q.optimizer != "Adam":
        raise ValueError("QMIX requires decreasing epsilon and Adam")
    if q.gat_layer_count != 1 or q.encoder_hidden_dimension % q.attention_head_count:
        raise ValueError("QMIX requires one GAT layer and divisible head dimension")
    if q.sequence_batch_size > q.replay_capacity_sequences:
        raise ValueError("QMIX sequence batch exceeds replay capacity")


@dataclass(frozen=True)
class QActions:
    proposals: tuple
    utilities: Tensor
    hidden_out: Tensor
    indices: dict[str, Tensor]
    masks: dict[str, Tensor]
    active: Tensor
    branch_values: dict[str, Tensor]
    exploration_flags: Tensor
    cpu_frequency_values: Tensor


class FactorizedGATQNetwork(nn.Module):
    """Shared agent parameters, local graph input, and independent GRU state."""

    def __init__(self, config: RunConfig):
        super().__init__()
        validate_qmix_config(config)
        q = config.training.qmix
        self.spec = replace(
            MAPPOTensorSpec.from_config(config),
            encoder_hidden_dimension=q.encoder_hidden_dimension,
            gru_hidden_dimension=q.gru_hidden_dimension,
            attention_head_count=q.attention_head_count,
            route_private_backbone=False,
        )
        s = self.spec
        self.cpu_queue_count = s.action_dimensions["cpu_queue"] - 1
        self.cpu_frequency_count = s.action_dimensions["cpu_frequency"]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(_derived_torch_seed(config.seed))
            self.self_encoder = nn.Sequential(nn.Linear(s.self_feature_dim, q.encoder_hidden_dimension), nn.LeakyReLU())
            self.public_encoder = nn.Sequential(nn.Linear(s.neighbor_public_feature_dim, q.encoder_hidden_dimension), nn.LeakyReLU())
            self.graph = CAGATv2Layer(q.encoder_hidden_dimension, s.edge_feature_dim, q.attention_head_count)
            self.gru = nn.GRU(q.encoder_hidden_dimension, q.gru_hidden_dimension, batch_first=True)
            heads = {}
            for branch in BRANCH_ORDER:
                output_size = s.action_dimensions[branch]
                if branch == "cpu_frequency":
                    output_size *= self.cpu_queue_count
                heads[branch] = nn.Sequential(
                    nn.Linear(q.gru_hidden_dimension, q.encoder_hidden_dimension),
                    nn.ReLU(), nn.Linear(q.encoder_hidden_dimension, output_size),
                )
            self.heads = nn.ModuleDict(heads)

    def initial_hidden(self, batch_size=1):
        p = next(self.parameters())
        return torch.zeros(batch_size, self.spec.uav_count, self.spec.gru_hidden_dimension, device=p.device, dtype=p.dtype)

    def encode(self, batch: ActorTensorBatch, hidden: Tensor) -> tuple[Tensor, Tensor]:
        b, t, a = batch.validate(self.spec)
        if tuple(hidden.shape) != (b, a, self.spec.gru_hidden_dimension):
            raise ValueError("QMIX hidden must have shape [batch,agent,GRU dimension]")
        spatial = self.graph(
            self.self_encoder(batch.self_features).reshape(b*t*a, -1),
            self.public_encoder(batch.neighbor_public_features).reshape(b*t*a, a, -1),
            stabilize_edge_features(batch.edge_features).reshape(b*t*a, a, -1),
            batch.neighbor_mask.reshape(b*t*a, a),
        ).reshape(b, t, a, -1)
        outputs = []
        recurrent = hidden.reshape(b*a, -1).unsqueeze(0)
        for time in range(t):
            start = batch.episode_starts[:, time].reshape(1, b*a, 1)
            recurrent = recurrent * (~start)
            out, recurrent = self.gru(spatial[:, time].reshape(b*a, 1, -1), recurrent)
            outputs.append(out.reshape(b, a, -1))
        return torch.stack(outputs, dim=1), recurrent.reshape(b, a, -1)

    def actions(self, batch: ActorTensorBatch, hidden: Tensor, contracts: Sequence,
                *, proposals: Sequence | None = None, epsilon=0.0,
                rng: np.random.Generator | None = None) -> QActions:
        if not 0 <= epsilon <= 1 or (epsilon > 0 and rng is None):
            raise ValueError("epsilon exploration requires a private generator")
        recurrent, hidden_out = self.encode(batch, hidden)
        shape = recurrent.shape[:3]
        b, t, a = shape
        if (len(contracts) != b or any(len(row) != t for row in contracts)
                or any(len(step) != a for row in contracts for step in row)):
            raise ValueError("mask contracts must have shape [batch,time,agent]")
        flat = [m for row in contracts for step in row for m in step]
        if any(m.uav_id != i % a for i, m in enumerate(flat)):
            raise ValueError("mask contracts must use stable agent order")
        teachers = None if proposals is None else [p for row in proposals for step in row for p in step]
        if teachers is not None and len(teachers) != b*t*a:
            raise ValueError("proposal grid has wrong size")
        contexts = [{} for _ in flat]
        # Share one exploration trigger across the agent's complete proposal.
        # Conditional uniform sampling remains the existing structured-action
        # adaptation; it is not uniform over all complete legal proposals.
        exploring = ([bool(rng.random() < epsilon) for _ in flat]
                     if teachers is None and epsilon > 0 else [False] * len(flat))
        branch_values = {branch: self.heads[branch](recurrent) for branch in BRANCH_ORDER
                         if branch != "cpu_frequency"}
        cpu_frequency_values = self.heads["cpu_frequency"](recurrent).reshape(
            *shape, self.cpu_queue_count, self.cpu_frequency_count)
        if (any(not torch.isfinite(value).all() for value in branch_values.values())
                or not torch.isfinite(cpu_frequency_values).all()):
            raise ValueError("non-finite Q values")
        # The legal action set factors into Route, TX and CPU groups. Exact
        # suffix maxima handle the dependencies inside the latter two groups.
        selection = (branch_values if teachers is not None else
                     self._selection_scores(branch_values, flat, cpu_frequency_values))
        indices, masks, values, chosen = {}, {}, {}, []
        for branch in BRANCH_ORDER:
            if branch == "cpu_frequency":
                # Select the row after the original queue branch (including
                # exploratory/teacher queues). Idle has no frequency utility.
                queue_index = indices["cpu_queue"]
                row = (queue_index - 1).clamp_min(0)
                q = cpu_frequency_values.gather(-2, row[..., None, None].expand(
                    *shape, 1, self.cpu_frequency_count)).squeeze(-2)
                q = torch.where((queue_index > 0).unsqueeze(-1), q, torch.zeros_like(q))
            else:
                q = branch_values[branch]
            mask = torch.tensor(np.stack([m.mask_for(branch, ctx) for m, ctx in zip(flat, contexts)]),
                                dtype=torch.bool, device=q.device).reshape(*shape, -1)
            if not torch.isfinite(q).all() or not mask.any(-1).all():
                raise ValueError("non-finite Q values or empty action mask")
            score = q if branch == "cpu_frequency" else selection[branch]
            idx = score.masked_fill(~mask, -torch.inf).argmax(-1).reshape(-1).tolist()
            for i, m in enumerate(flat):
                domain = m.domain_for(branch)
                if len(domain) != q.shape[-1]:
                    raise ValueError("action domain differs from network head")
                if teachers is not None:
                    idx[i] = domain.index(getattr(teachers[i], branch))
                    if teachers[i].uav_id != m.uav_id or not mask.reshape(-1, q.shape[-1])[i, idx[i]]:
                        raise ValueError("replay proposal violates its conditional mask")
                elif exploring[i]:
                    legal = np.flatnonzero(mask.reshape(-1, q.shape[-1])[i].detach().cpu().numpy())
                    if len(legal) > 1:
                        idx[i] = int(rng.choice(legal))
                contexts[i][branch] = domain[idx[i]]
            index = torch.tensor(idx, dtype=torch.long, device=q.device).reshape(shape)
            indices[branch], masks[branch], values[branch] = index, mask, q
            chosen.append(q.gather(-1, index.unsqueeze(-1)).squeeze(-1))
        selected = [ActionProposal(uav_id=m.uav_id, **ctx) for m, ctx in zip(flat, contexts)]
        if not all(m.is_legal(p) for m, p in zip(flat, selected)):
            raise ValueError("QMIX selected an illegal proposal")
        active = torch.tensor([m.active_indicators(p) for m, p in zip(flat, selected)],
                              device=recurrent.device, dtype=torch.bool).reshape(*shape, 7)
        utility = (torch.stack(chosen, -1) * active).sum(-1)
        grid = tuple(tuple(tuple(selected[(i*t+j)*a:(i*t+j+1)*a]) for j in range(t)) for i in range(b))
        exploration_flags = torch.tensor(exploring, dtype=torch.bool, device=recurrent.device).reshape(shape)
        return QActions(grid, utility, hidden_out, indices, masks, active, values, exploration_flags,
                        cpu_frequency_values)

    def _selection_scores(self, values: dict[str, Tensor], contracts: Sequence,
                          cpu_frequency_values: Tensor) -> dict[str, Tensor]:
        """Back up complete legal suffixes; raw scores still define utilities.

        Frozen masks couple TX to group/width/power and CPU queue to frequency,
        but do not couple the Route, TX and CPU groups. Inactive suffixes add
        zero. No reward, feasibility mask or service priority is changed.
        """
        scores = dict(values)
        shape = values["route"].shape[:-1]
        device = values["route"].device
        first = contracts[0]
        for mask in contracts:
            if (any(mask.domain_for(branch) != first.domain_for(branch) for branch in (
                    "resource_group", "resource_width", "power_level", "cpu_queue", "cpu_frequency"))
                    or mask.resource_group_count != first.resource_group_count):
                raise ValueError("QMIX service domains must be fixed across the batch")

        # For non-Idle TX, the existing power mask is independent of receiver,
        # group and width. Preserve its zero-power fallback exactly.
        power_mask = torch.tensor(np.stack([
            mask.mask_for("power_level", {"tx_select": mask.tx_select_domain[1], "resource_group": 1})
            for mask in contracts]), dtype=torch.bool, device=device).reshape(*shape, -1)
        power_best = values["power_level"].masked_fill(~power_mask, -torch.inf).max(-1).values
        groups = torch.tensor(first.resource_group_domain[1:], device=device)
        widths = torch.tensor(first.resource_width_domain, device=device)
        width_mask = groups[:, None] + widths[None, :] - 1 <= first.resource_group_count
        width_best = values["resource_width"].unsqueeze(-2).masked_fill(~width_mask, -torch.inf).max(-1).values
        group_scores = values["resource_group"].clone()
        group_scores[..., 1:] = group_scores[..., 1:] + width_best + power_best.unsqueeze(-1)
        scores["resource_group"] = group_scores
        tx_active = torch.tensor([mask.tx_branch_active for mask in contracts], device=device).reshape(shape)
        tx_scores = values["tx_select"] * tx_active.unsqueeze(-1)
        tx_scores[..., 1:] = tx_scores[..., 1:] + group_scores[..., 1:].max(-1).values.unsqueeze(-1)
        scores["tx_select"] = tx_scores

        frequency_masks = torch.tensor(np.stack([mask.cpu_frequency_energy_mask for mask in contracts]),
                                       dtype=torch.bool, device=device).reshape(
                                           *shape, len(first.cpu_queue_domain)-1, len(first.cpu_frequency_domain))
        frequency_best = cpu_frequency_values.masked_fill(
            ~frequency_masks, -torch.inf).max(-1).values
        queue_active = torch.tensor([mask.cpu_queue_branch_active for mask in contracts], device=device).reshape(shape)
        queue_scores = values["cpu_queue"] * queue_active.unsqueeze(-1)
        queue_scores[..., 1:] = queue_scores[..., 1:] + frequency_best
        scores["cpu_queue"] = queue_scores
        return scores


class MonotonicMixer(nn.Module):
    """State-conditioned nonnegative weights; centralized state never enters Q agents."""

    def __init__(self, agents: int, state_dim: int, hidden=32, hyper_hidden=128):
        super().__init__()
        self.agents, self.hidden = agents, hidden
        self.w1 = nn.Sequential(nn.Linear(state_dim, hyper_hidden), nn.ReLU(), nn.Linear(hyper_hidden, agents*hidden))
        self.w2 = nn.Sequential(nn.Linear(state_dim, hyper_hidden), nn.ReLU(), nn.Linear(hyper_hidden, hidden))
        self.b1 = nn.Linear(state_dim, hidden)
        self.bias = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, utilities: Tensor, state: Tensor) -> Tensor:
        if utilities.shape[:-1] != state.shape[:-1] or utilities.shape[-1] != self.agents:
            raise ValueError("mixer utility and state axes differ")
        # Actor inputs/rewards are untouched. Historical quality/denominators
        # can span many orders of magnitude in the centralized tensor. This
        # stateless training-only transform prevents multiplicative hypernet
        # weights from amplifying those physical scales into enormous TDs.
        state = torch.sign(state) * torch.log1p(state.abs())
        state = F.layer_norm(state, (state.shape[-1],))
        w1 = self.w1(state).abs().reshape(*state.shape[:-1], self.agents, self.hidden)
        intermediate = F.elu((utilities.unsqueeze(-1) * w1).sum(-2) + self.b1(state))
        return (intermediate * self.w2(state).abs()).sum(-1) + self.bias(state).squeeze(-1)


class FactorizedQMIX(nn.Module):
    def __init__(self, config: RunConfig):
        super().__init__()
        self.agent = build_qmix_agent(config)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(_derived_torch_seed(config.seed) + 120)
            self.mixer = MonotonicMixer(self.agent.spec.uav_count, self.agent.spec.centralized_state_dim)
        if config.method_id == "c_feature_credit_qmix":
            from .route_task_credit import RouteTaskValue
            self.route_task_value = RouteTaskValue(self.agent.spec.centralized_state_dim,
                config.training.qmix.encoder_hidden_dimension, self.agent.spec.uav_count, config.seed)


def build_qmix_agent(config):
    if config.method_id == "c_feature_credit_qmix":
        from .c_feature_qmix_agent import CFeatureQNetwork
        return CFeatureQNetwork(config)
    return FactorizedGATQNetwork(config)


def masked_half_mse(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    if prediction.shape != target.shape or valid.shape != prediction.shape or valid.dtype != torch.bool:
        raise ValueError("TD tensors and boolean validity mask must share shape")
    if not valid.any():
        raise ValueError("TD loss requires a real transition")
    # Select before arithmetic: invalid padding may carry NaN and must not leak.
    p, y = prediction[valid], target.detach()[valid]
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():
        raise ValueError("TD values must be finite on real transitions")
    return .5 * (p - y).square().mean()


def td_target(reward: Tensor, next_total: Tensor, bootstrap: Tensor, gamma: float) -> Tensor:
    if reward.shape != next_total.shape or bootstrap.shape != reward.shape or bootstrap.dtype != torch.bool:
        raise ValueError("TD target tensors and boolean bootstrap mask must share shape")
    return reward + gamma * torch.where(bootstrap, next_total, torch.zeros_like(next_total))
