"""Actor-visible node identities and Route leaf readouts, without a new policy.

Node candidates are Local followed by Remote nodes. Executor indices remain
Idle, Local, Defer, then ascending non-self node IDs regardless of view order.
These readouts neither change the old feature semantics nor sample actions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .ca_gat_mappo import MAPPOTensorSpec


def remote_node_ids(count: int, *, device=None) -> Tensor:
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("Route candidates require at least two UAVs")
    return torch.tensor([[j for j in range(count) if j != i]
                         for i in range(count)], dtype=torch.long, device=device)


@dataclass(frozen=True)
class VisibleLocalCpuBurden:
    """Queue aggregates in the original tensorizer's normalized units."""

    normalized_task_count: Tensor
    normalized_remaining_cycles: Tensor


def visible_local_cpu_burden(self_features: Tensor, spec: MAPPOTensorSpec) -> VisibleLocalCpuBurden:
    """Count local plus received CPU queues once; exclude TX and UNBOUND.

    ObservationBuilder aliases cpu_by_source[ego] to local_cpu. Replace that
    slot with the canonical local summary, then sum the indexed sources; never
    add local_cpu a second time. Distinct source queues own disjoint tasks.
    This is an audit/C-feature input, not a change to the A/B burden block.
    """
    from .ca_gat_mappo import RouteOptionFeatureExtractor
    ex = RouteOptionFeatureExtractor(spec)
    if (self_features.ndim != 4 or self_features.shape[-2] != spec.uav_count
            or self_features.shape[-1] != spec.self_feature_dim
            or not self_features.is_floating_point()
            or not torch.isfinite(self_features).all()):
        raise ValueError("Local CPU burden requires finite actor self features")
    queues = ex._indexed_queues(self_features, ex.cpu_start)
    local = ex._queue(self_features, ex.local_start)
    diagonal = torch.eye(spec.uav_count, device=self_features.device, dtype=torch.bool)
    queues = torch.where(diagonal.view(1, 1, spec.uav_count, spec.uav_count, 1),
                         local.unsqueeze(-2), queues)
    return VisibleLocalCpuBurden(queues[..., 0].sum(-1), queues[..., 2].sum(-1))


@dataclass(frozen=True)
class RouteCandidateBatch:
    """Aligned node view; action indices always refer to the environment."""

    node_ids: Tensor                    # [A,N], Local is source UAV
    action_indices: Tensor              # [A,N], Local=1, Remote>=3
    legal_mask: Tensor                  # [B,T,A,N], environment legality
    task_features: Tensor
    source_features: Tensor
    burden_features: Tensor             # [B,T,A,N,10], original semantics
    validity_features: Tensor           # [B,T,A,N,4], observation validity
    local_cpu_total: VisibleLocalCpuBurden
    endpoint_features: Tensor | None = None
    route_recurrent_features: Tensor | None = None
    route_hidden_out: Tensor | None = None
    uses_private_backbone: bool = False


def build_route_candidates(
    spec: MAPPOTensorSpec, self_features: Tensor, public_features: Tensor,
    edge_features: Tensor, route_mask: Tensor, *, remote_ids: Tensor | None = None,
    local_endpoint_features: Tensor | None = None,
    candidate_endpoint_features: Tensor | None = None,
    route_recurrent_features: Tensor | None = None,
    route_hidden_out: Tensor | None = None, uses_private_backbone: bool = False,
) -> RouteCandidateBatch:
    """Use only actor inputs; Remote public validity does not change legality.

    A permutation of remote_ids permutes this view, including embeddings and
    masks, while retaining the original destination's executor action index.
    No centralized state or live helper queue is accepted.
    """
    from .ca_gat_mappo import RouteOptionFeatureExtractor
    ex = RouteOptionFeatureExtractor(spec)
    count = spec.uav_count
    canonical = remote_node_ids(count, device=self_features.device)
    ids = canonical if remote_ids is None else remote_ids
    if (ids.dtype != torch.long or ids.device != self_features.device
            or ids.shape != canonical.shape
            or not torch.equal(ids.sort(-1).values, canonical)):
        raise ValueError("Remote IDs must be a permutation of every non-self UAV")
    tensors = (self_features, public_features, edge_features)
    if any(not x.is_floating_point() or x.dtype != self_features.dtype
           or x.device != self_features.device or not torch.isfinite(x).all()
           for x in tensors):
        raise ValueError("Candidate features must be finite with shared dtype/device")
    task, source, local, remote, lv, rv = ex.extract(*tensors, ids)
    shape = self_features.shape[:3]
    if (route_mask.shape != (*shape, count + 2) or route_mask.dtype != torch.bool
            or route_mask.device != self_features.device or not route_mask.any(-1).all()):
        raise ValueError("Route mask must match actor shape with nonempty support")
    ego = torch.arange(count, device=ids.device).unsqueeze(-1)
    nodes = torch.cat((ego, ids), -1)
    # Position among sorted non-self IDs, rather than position in this view.
    actions = torch.cat((torch.ones_like(ego), 3 + ids - (ids > ego).long()), -1)
    legal = route_mask.gather(-1, actions.view(1, 1, count, count).expand(*shape, count))
    endpoints = None
    if (local_endpoint_features is None) != (candidate_endpoint_features is None):
        raise ValueError("Local and Remote endpoint embeddings must be supplied together")
    for value, expected in ((route_recurrent_features, (*shape, spec.gru_hidden_dimension)),
                            (route_hidden_out, (shape[0], count, spec.gru_hidden_dimension)),
                            (local_endpoint_features, (*shape, spec.encoder_hidden_dimension)),
                            (candidate_endpoint_features, (*shape, count, spec.encoder_hidden_dimension))):
        if value is not None and (value.shape != expected or value.device != self_features.device
                or value.dtype != self_features.dtype or not torch.isfinite(value).all()):
            raise ValueError("Candidate context must preserve actor shape, device and dtype")
    if local_endpoint_features is not None:
        endpoints = torch.cat((local_endpoint_features.unsqueeze(-2),
                               ex._gather_candidates(candidate_endpoint_features, ids)), -2)
    if uses_private_backbone and (route_recurrent_features is None or route_hidden_out is None):
        raise ValueError("Private Route readout requires its recurrent features and hidden lane")
    return RouteCandidateBatch(nodes, actions, legal, task, source,
                               torch.cat((local.unsqueeze(-2), remote), -2),
                               torch.cat((lv.unsqueeze(-2), rv), -2),
                               visible_local_cpu_burden(self_features, spec), endpoints,
                               route_recurrent_features, route_hidden_out, uses_private_backbone)


@dataclass(frozen=True)
class RouteLeafResult:
    """Actual selection and separate diagnostics over canonical executor leaves.

    Group order is Local/Remote/Defer. Idle has group argmax -1. Control
    actions have chosen_node_id -1; chosen_indices retains Idle/Defer identity.
    A historical hierarchical choice need not equal the leaf argmax.
    """

    probabilities: Tensor
    log_probabilities: Tensor
    legal_mask: Tensor
    leaf_node_ids: Tensor
    chosen_indices: Tensor
    chosen_node_ids: Tensor
    chosen_log_prob: Tensor
    leaf_argmax_indices: Tensor
    group_probabilities: Tensor
    group_support: Tensor
    group_mass_argmax: Tensor
    conditional_remote_probabilities: Tensor


def route_leaf_result(logits: Tensor, mask: Tensor, chosen_indices: Tensor) -> RouteLeafResult:
    """Read a supplied actual choice; never replace it with grouped argmax."""
    if (logits.ndim != 4 or not logits.is_floating_point()
            or logits.shape[-1] != logits.shape[-2] + 2
            or logits.shape[-2] < 2 or not torch.isfinite(logits).all()
            or mask.shape != logits.shape or mask.dtype != torch.bool
            or mask.device != logits.device or not mask.any(-1).all()):
        raise ValueError("Route logits/mask must be finite [B,T,A,A+2] with legal support")
    if (chosen_indices.shape != logits.shape[:-1] or chosen_indices.dtype != torch.long
            or chosen_indices.device != logits.device or (chosen_indices < 0).any()
            or (chosen_indices >= logits.shape[-1]).any()):
        raise ValueError("Chosen Route indices must match the canonical leaf domain")
    if not mask.gather(-1, chosen_indices.unsqueeze(-1)).all():
        raise ValueError("Chosen Route action is illegal")
    masked = logits.masked_fill(~mask, -torch.inf)
    logs = masked.log_softmax(-1)
    probabilities = logs.exp()
    remote_mask = mask[..., 3:]
    remote_available = remote_mask.any(-1)
    groups = torch.stack((probabilities[..., 1], probabilities[..., 3:].sum(-1),
                          probabilities[..., 2]), -1)
    support = torch.stack((mask[..., 1], remote_available, mask[..., 2]), -1)
    group_argmax = groups.masked_fill(~support, -torch.inf).argmax(-1)
    group_argmax = torch.where(support.any(-1), group_argmax, -torch.ones_like(group_argmax))
    fallback = torch.zeros_like(remote_mask); fallback[..., 0] = True
    safe_mask = torch.where(remote_available.unsqueeze(-1), remote_mask, fallback)
    conditional = logits[..., 3:].masked_fill(~safe_mask, -torch.inf).softmax(-1)
    conditional = torch.where(remote_available.unsqueeze(-1), conditional, torch.zeros_like(conditional))
    count = logits.shape[-2]
    ego = torch.arange(count, device=logits.device).unsqueeze(-1)
    no_node = -torch.ones_like(ego)
    node_ids = torch.cat((no_node, ego, no_node, remote_node_ids(count, device=logits.device)), -1)
    node_grid = node_ids.view(1, 1, count, count + 2).expand_as(logits)
    return RouteLeafResult(probabilities, logs, mask, node_ids, chosen_indices,
                           node_grid.gather(-1, chosen_indices.unsqueeze(-1)).squeeze(-1),
                           logs.gather(-1, chosen_indices.unsqueeze(-1)).squeeze(-1),
                           masked.argmax(-1), groups, support, group_argmax, conditional)
