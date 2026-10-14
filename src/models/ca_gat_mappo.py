"""CA-GAT-MAPPO networks and tensor adapters for Stage 07B-1.

This module stops at deterministic neural-network forward interfaces. It does
not sample actions, create rollout storage, calculate GAE/PPO losses, own an
optimizer, or run training. Actor tensors are built only from the existing
ActorObservation boundary; centralized tensors are built separately from the
existing decision-time CentralizedState boundary.

Public layouts keep batch, time, and agent axes explicit:
actor features [B,T,A,F], graph features [B,T,A,N,F], logits [B,T,A,D],
and recurrent hidden input/output [B,A,H]. The critic consumes [B,T,S] and
returns [B,T,1].
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..config import AgentCreditMode, CpuFrequencyDecoderMode, RouteDecoderMode, RunConfig, STREAM_IDS, TxDecoderMode
from ..env.actions import ActionProposal
from ..env.observation import ActorObservation, BRANCH_ORDER, QueueSummary
from ..env.state import CentralizedState

if TYPE_CHECKING:
    from .route_candidates import RouteCandidateBatch


class MAPPONetworkError(ValueError):
    """Raised when a Stage 07B-1 tensor violates the frozen interface."""


ACTION_BRANCH_ORDER = tuple(BRANCH_ORDER)
ACTION_BRANCH_DEPENDENCIES: Mapping[str, tuple[str, ...]] = {
    "route": (),
    "tx_select": (),
    "resource_group": ("tx_select",),
    "resource_width": ("tx_select", "resource_group"),
    "power_level": ("tx_select", "resource_group", "resource_width"),
    "cpu_queue": (),
    "cpu_frequency": ("cpu_queue",),
}
_ACTIVE_TASK_STATUSES = ("unbound", "local", "tx", "cpu")
_QUEUE_KINDS = ("unbound", "local", "tx", "cpu")
_SELF_RESOURCE_FEATURE_DIM = 9
_QUEUE_SUMMARY_FEATURE_DIM = 8
_PUBLIC_FEATURE_DIM = 17
_TASK_OPTION_FEATURE_DIM = 5
_SOURCE_OPTION_FEATURE_DIM = 14
_BURDEN_OPTION_FEATURE_DIM = 10
_OPTION_VALIDITY_FEATURE_DIM = 4


class RouteOptionFeatureExtractor:
    """Central semantic view over the frozen actor tensor layouts."""

    _RESOURCE = {
        "residual_energy_ratio": 2,
        "max_transmit_power_ratio": 4,
        "max_cpu_frequency_ratio": 6,
        "cpu_coefficient_ratio": 8,
    }
    _QUEUE = {
        "task_count": 0,
        "remaining_bits": 1,
        "remaining_cycles": 2,
        "head_remaining_bits": 4,
        "head_remaining_cycles": 5,
        "head_slack": 6,
        "head_valid": 7,
    }
    _PUBLIC = {
        "valid": 0,
        "max_cpu_frequency_ratio": 10,
        "cpu_load_task_count": 12,
        "cpu_load_remaining_cycles": 13,
    }
    _EDGE = {
        "visible": 0,
        "last_rate": -4,
        "last_rate_valid": -3,
        "outage_rate": -2,
        "outage_valid": -1,
    }

    def __init__(self, spec: "MAPPOTensorSpec") -> None:
        self.uav_count = spec.uav_count
        self.self_feature_dim = spec.self_feature_dim
        self.public_feature_dim = spec.neighbor_public_feature_dim
        self.edge_feature_dim = spec.edge_feature_dim
        expected_self = (
            _SELF_RESOURCE_FEATURE_DIM
            + 2 * _QUEUE_SUMMARY_FEATURE_DIM
            + 2 * self.uav_count * _QUEUE_SUMMARY_FEATURE_DIM
            + 2
            + 2 * self.uav_count
            + sum(spec.action_dimensions.values())
            + 2 * self.uav_count
            + 7
        )
        if self.self_feature_dim != expected_self:
            raise MAPPONetworkError(
                "route option extractor differs from the frozen self-feature layout"
            )
        if self.public_feature_dim != _PUBLIC_FEATURE_DIM:
            raise MAPPONetworkError(
                "route option extractor differs from the frozen public-feature layout"
            )
        if self.edge_feature_dim < 4:
            raise MAPPONetworkError("route option extractor requires edge history tails")

        self.unbound_start = _SELF_RESOURCE_FEATURE_DIM
        self.local_start = self.unbound_start + _QUEUE_SUMMARY_FEATURE_DIM
        self.tx_start = self.local_start + _QUEUE_SUMMARY_FEATURE_DIM
        self.cpu_start = self.tx_start + self.uav_count * _QUEUE_SUMMARY_FEATURE_DIM
        self.arrival_start = self.cpu_start + self.uav_count * _QUEUE_SUMMARY_FEATURE_DIM

    @staticmethod
    def _safe_pressure(
        queued_cycles: Tensor,
        task_cycles: Tensor,
        cpu_ratio: Tensor,
        valid: Tensor,
    ) -> Tensor:
        epsilon = torch.finfo(cpu_ratio.dtype).eps
        pressure = (queued_cycles + task_cycles) / cpu_ratio.clamp_min(epsilon)
        return torch.where(valid > 0.5, pressure, torch.zeros_like(pressure))

    def _queue(self, self_features: Tensor, start: int) -> Tensor:
        return self_features[..., start : start + _QUEUE_SUMMARY_FEATURE_DIM]

    def _indexed_queues(self, self_features: Tensor, start: int) -> Tensor:
        end = start + self.uav_count * _QUEUE_SUMMARY_FEATURE_DIM
        return self_features[..., start:end].reshape(
            *self_features.shape[:-1],
            self.uav_count,
            _QUEUE_SUMMARY_FEATURE_DIM,
        )

    def _gather_candidates(self, values: Tensor, candidate_ids: Tensor) -> Tensor:
        batch, time, agents, _, features = values.shape
        index = candidate_ids.view(
            1, 1, agents, self.uav_count - 1, 1
        ).expand(batch, time, agents, self.uav_count - 1, features)
        return torch.gather(values, -2, index)

    def extract(
        self,
        self_features: Tensor,
        public_features: Tensor,
        edge_features: Tensor,
        candidate_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return task/source plus aligned local and remote burden blocks."""

        if self_features.ndim != 4 or self_features.shape[-1] != self.self_feature_dim:
            raise MAPPONetworkError("route option self features have an invalid shape")
        batch, time, agents, _ = self_features.shape
        expected_public = (
            batch,
            time,
            agents,
            self.uav_count,
            self.public_feature_dim,
        )
        expected_edge = (
            batch,
            time,
            agents,
            self.uav_count,
            self.edge_feature_dim,
        )
        if agents != self.uav_count or tuple(public_features.shape) != expected_public:
            raise MAPPONetworkError("route option public features have an invalid shape")
        if tuple(edge_features.shape) != expected_edge:
            raise MAPPONetworkError("route option edge features have an invalid shape")
        if tuple(candidate_ids.shape) != (self.uav_count, self.uav_count - 1):
            raise MAPPONetworkError("route option candidate IDs have an invalid shape")

        resources = self_features[..., :_SELF_RESOURCE_FEATURE_DIM]
        unbound = self._queue(self_features, self.unbound_start)
        local = self._queue(self_features, self.local_start)
        tx = self._indexed_queues(self_features, self.tx_start)
        arrival = self_features[..., self.arrival_start : self.arrival_start + 2]

        task_bits = unbound[..., self._QUEUE["head_remaining_bits"]]
        task_cycles = unbound[..., self._QUEUE["head_remaining_cycles"]]
        task_valid = unbound[..., self._QUEUE["head_valid"]]
        epsilon = torch.finfo(task_bits.dtype).eps
        cycles_per_bit = torch.where(
            (task_valid > 0.5) & (task_bits > epsilon),
            task_cycles / task_bits.clamp_min(epsilon),
            torch.zeros_like(task_bits),
        )
        task = torch.stack(
            [
                task_bits,
                task_cycles,
                cycles_per_bit,
                unbound[..., self._QUEUE["head_slack"]],
                task_valid,
            ],
            dim=-1,
        )

        source = torch.stack(
            [
                resources[..., self._RESOURCE["residual_energy_ratio"]],
                resources[..., self._RESOURCE["max_transmit_power_ratio"]],
                resources[..., self._RESOURCE["max_cpu_frequency_ratio"]],
                resources[..., self._RESOURCE["cpu_coefficient_ratio"]],
                local[..., self._QUEUE["task_count"]],
                local[..., self._QUEUE["remaining_cycles"]],
                local[..., self._QUEUE["head_remaining_cycles"]],
                local[..., self._QUEUE["head_slack"]],
                local[..., self._QUEUE["head_valid"]],
                unbound[..., self._QUEUE["task_count"]],
                unbound[..., self._QUEUE["remaining_bits"]],
                unbound[..., self._QUEUE["remaining_cycles"]],
                arrival[..., 0],
                arrival[..., 1],
            ],
            dim=-1,
        )

        local_cpu = resources[..., self._RESOURCE["max_cpu_frequency_ratio"]]
        local_pressure = self._safe_pressure(
            local[..., self._QUEUE["remaining_cycles"]],
            task_cycles,
            local_cpu,
            torch.ones_like(task_valid),
        )
        zeros = torch.zeros_like(task_cycles)
        local_burden = torch.stack(
            [
                local[..., self._QUEUE["task_count"]],
                local[..., self._QUEUE["remaining_cycles"]],
                task_cycles,
                local_cpu,
                local_pressure,
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
            ],
            dim=-1,
        )
        local_validity = torch.stack(
            [task_valid, torch.ones_like(task_valid), zeros, zeros], dim=-1
        )

        remote_public = self._gather_candidates(public_features, candidate_ids)
        remote_edge = self._gather_candidates(edge_features, candidate_ids)
        remote_tx = self._gather_candidates(tx, candidate_ids)
        remote_task_cycles = task_cycles.unsqueeze(-1).expand(
            batch, time, agents, self.uav_count - 1
        )
        endpoint_valid = remote_public[..., self._PUBLIC["valid"]]
        remote_cpu = remote_public[..., self._PUBLIC["max_cpu_frequency_ratio"]]
        remote_pressure = self._safe_pressure(
            remote_public[..., self._PUBLIC["cpu_load_remaining_cycles"]],
            remote_task_cycles,
            remote_cpu,
            endpoint_valid,
        )
        remote_burden = torch.stack(
            [
                remote_public[..., self._PUBLIC["cpu_load_task_count"]],
                remote_public[..., self._PUBLIC["cpu_load_remaining_cycles"]],
                remote_task_cycles,
                remote_cpu,
                remote_pressure,
                remote_tx[..., self._QUEUE["remaining_bits"]],
                remote_edge[..., self._EDGE["last_rate"]],
                remote_edge[..., self._EDGE["last_rate_valid"]],
                remote_edge[..., self._EDGE["outage_rate"]],
                remote_edge[..., self._EDGE["outage_valid"]],
            ],
            dim=-1,
        )
        remote_validity = torch.stack(
            [
                task_valid.unsqueeze(-1).expand_as(endpoint_valid),
                endpoint_valid,
                remote_edge[..., self._EDGE["visible"]],
                remote_edge[..., self._EDGE["last_rate_valid"]],
            ],
            dim=-1,
        )
        return (
            task,
            source,
            local_burden,
            remote_burden,
            local_validity,
            remote_validity,
        )


def _derived_torch_seed(master_seed: int) -> int:
    """Derive Torch initialization from the frozen stream ID 100."""

    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
        raise MAPPONetworkError("master seed must be a non-negative integer")
    state = np.random.SeedSequence(
        [master_seed, STREAM_IDS["torch_initialization"]]
    ).generate_state(1, dtype=np.uint64)
    return int(state[0] % np.uint64(np.iinfo(np.int64).max))


def _action_dimensions(config: RunConfig) -> dict[str, int]:
    count = config.environment.uav_count
    return {
        "route": count + 2,
        "tx_select": count,
        "resource_group": config.environment.resource_group_count + 1,
        "resource_width": len(config.action.resource_width_options),
        "power_level": len(config.action.power_levels),
        "cpu_queue": count + 1,
        "cpu_frequency": len(config.action.cpu_frequency_levels),
    }


@dataclass(frozen=True)
class MAPPOTensorSpec:
    """Dimensions derived from the environment schema and MAPPO defaults."""

    uav_count: int
    ru_count: int
    self_feature_dim: int
    neighbor_public_feature_dim: int
    edge_feature_dim: int
    centralized_state_dim: int
    action_dimensions: Mapping[str, int]
    encoder_hidden_dimension: int
    gat_layer_count: int
    attention_head_count: int
    gru_hidden_dimension: int
    route_private_backbone: bool = False

    @property
    def recurrent_state_dimension(self) -> int:
        """Persist [shared_hidden, route_hidden] lanes when Route is private."""
        return self.gru_hidden_dimension * (2 if self.route_private_backbone else 1)

    @classmethod
    def from_config(cls, config: RunConfig) -> "MAPPOTensorSpec":
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        config.validate()
        env = config.environment
        model = config.training.mappo
        if tuple(config.action.sampling_order) != ACTION_BRANCH_ORDER:
            raise MAPPONetworkError("config action order differs from the frozen order")
        if model.gat_layer_count != 1:
            raise MAPPONetworkError("Stage 07B-1 requires one CA-GATv2 layer")
        if model.encoder_hidden_dimension % model.attention_head_count != 0:
            raise MAPPONetworkError("encoder width must be divisible by attention heads")
        if model.gru_hidden_dimension != model.encoder_hidden_dimension:
            raise MAPPONetworkError("graph and GRU widths must match the frozen 128 interface")

        count = env.uav_count
        ru_count = env.ru_count
        action_dimensions = _action_dimensions(config)
        previous_dim = sum(action_dimensions.values()) + 2 * count + 7
        self_dim = 9 + 16 + 16 * count + 2 + 2 * count + previous_dim
        public_dim = 17
        edge_dim = 10 + 6 * ru_count

        fixed_state_dim = (
            1
            + 13 * count
            + 7 * count * count
            + 3 * count * ru_count
            + 6 * count * count * ru_count
        )
        aggregate_state_dim = 10 * count + 5 * count * count
        building_dim = 6 * len(env.building_layout)
        return cls(
            uav_count=count,
            ru_count=ru_count,
            self_feature_dim=self_dim,
            neighbor_public_feature_dim=public_dim,
            edge_feature_dim=edge_dim,
            centralized_state_dim=fixed_state_dim + aggregate_state_dim + building_dim,
            action_dimensions=action_dimensions,
            encoder_hidden_dimension=model.encoder_hidden_dimension,
            gat_layer_count=model.gat_layer_count,
            attention_head_count=model.attention_head_count,
            gru_hidden_dimension=model.gru_hidden_dimension,
            route_private_backbone=model.route_private_backbone,
        )


@dataclass(frozen=True)
class ActorTensorBatch:
    """One explicit recurrent actor tensor batch."""

    self_features: Tensor
    neighbor_public_features: Tensor
    edge_features: Tensor
    neighbor_mask: Tensor
    action_masks: Mapping[str, Tensor]
    action_indices: Mapping[str, Tensor]
    episode_starts: Tensor

    def validate(self, spec: MAPPOTensorSpec) -> tuple[int, int, int]:
        if self.self_features.ndim != 4:
            raise MAPPONetworkError("self_features must be [batch,time,agent,feature]")
        batch, time, agents, features = self.self_features.shape
        if agents != spec.uav_count or features != spec.self_feature_dim:
            raise MAPPONetworkError("self_features differ from the configured tensor spec")
        if tuple(self.neighbor_public_features.shape) != (
            batch,
            time,
            agents,
            spec.uav_count,
            spec.neighbor_public_feature_dim,
        ):
            raise MAPPONetworkError("neighbor_public_features have an invalid shape")
        if tuple(self.edge_features.shape) != (
            batch,
            time,
            agents,
            spec.uav_count,
            spec.edge_feature_dim,
        ):
            raise MAPPONetworkError("edge_features have an invalid shape")
        if tuple(self.neighbor_mask.shape) != (batch, time, agents, spec.uav_count):
            raise MAPPONetworkError("neighbor_mask has an invalid shape")
        if tuple(self.episode_starts.shape) != (batch, time, agents):
            raise MAPPONetworkError("episode_starts must be [batch,time,agent]")
        for name, tensor in (
            ("self_features", self.self_features),
            ("neighbor_public_features", self.neighbor_public_features),
            ("edge_features", self.edge_features),
        ):
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise MAPPONetworkError(f"{name} must be a finite floating tensor")
        if self.neighbor_mask.dtype != torch.bool or self.episode_starts.dtype != torch.bool:
            raise MAPPONetworkError("graph and episode masks must be boolean")
        if tuple(self.action_masks) != ACTION_BRANCH_ORDER:
            raise MAPPONetworkError("action_masks must contain exactly seven ordered branches")
        if tuple(self.action_indices) != ACTION_BRANCH_ORDER:
            raise MAPPONetworkError("action_indices must contain exactly seven ordered branches")
        for branch in ACTION_BRANCH_ORDER:
            dimension = spec.action_dimensions[branch]
            mask = self.action_masks[branch]
            indices = self.action_indices[branch]
            if tuple(mask.shape) != (batch, time, agents, dimension):
                raise MAPPONetworkError(f"{branch} mask has an invalid shape")
            if mask.dtype != torch.bool or not torch.all(torch.any(mask, dim=-1)):
                raise MAPPONetworkError(f"{branch} mask has no legal fallback")
            if tuple(indices.shape) != (batch, time, agents) or indices.dtype != torch.long:
                raise MAPPONetworkError(f"{branch} indices must be long [batch,time,agent]")
            if torch.any(indices < 0) or torch.any(indices >= dimension):
                raise MAPPONetworkError(f"{branch} index lies outside its domain")
            legal = torch.gather(mask, -1, indices.unsqueeze(-1)).squeeze(-1)
            if not torch.all(legal):
                raise MAPPONetworkError(f"{branch} action context violates its mask")
        return batch, time, agents

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> "ActorTensorBatch":
        target_dtype = self.self_features.dtype if dtype is None else dtype
        return ActorTensorBatch(
            self_features=self.self_features.to(device=device, dtype=target_dtype),
            neighbor_public_features=self.neighbor_public_features.to(
                device=device, dtype=target_dtype
            ),
            edge_features=self.edge_features.to(device=device, dtype=target_dtype),
            neighbor_mask=self.neighbor_mask.to(device=device),
            action_masks={key: value.to(device=device) for key, value in self.action_masks.items()},
            action_indices={
                key: value.to(device=device) for key, value in self.action_indices.items()
            },
            episode_starts=self.episode_starts.to(device=device),
        )

    def repeat_batch(self, repeats: int) -> "ActorTensorBatch":
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
            raise MAPPONetworkError("repeats must be a positive integer")

        def repeat(tensor: Tensor) -> Tensor:
            return tensor.repeat((repeats,) + (1,) * (tensor.ndim - 1))

        return ActorTensorBatch(
            self_features=repeat(self.self_features),
            neighbor_public_features=repeat(self.neighbor_public_features),
            edge_features=repeat(self.edge_features),
            neighbor_mask=repeat(self.neighbor_mask),
            action_masks={key: repeat(value) for key, value in self.action_masks.items()},
            action_indices={key: repeat(value) for key, value in self.action_indices.items()},
            episode_starts=repeat(self.episode_starts),
        )


def stack_actor_time(steps: Sequence[ActorTensorBatch]) -> ActorTensorBatch:
    """Concatenate step batches along the explicit time dimension."""

    copied = tuple(steps)
    if not copied:
        raise MAPPONetworkError("at least one actor step is required")
    return ActorTensorBatch(
        self_features=torch.cat([item.self_features for item in copied], dim=1),
        neighbor_public_features=torch.cat(
            [item.neighbor_public_features for item in copied], dim=1
        ),
        edge_features=torch.cat([item.edge_features for item in copied], dim=1),
        neighbor_mask=torch.cat([item.neighbor_mask for item in copied], dim=1),
        action_masks={
            branch: torch.cat([item.action_masks[branch] for item in copied], dim=1)
            for branch in ACTION_BRANCH_ORDER
        },
        action_indices={
            branch: torch.cat([item.action_indices[branch] for item in copied], dim=1)
            for branch in ACTION_BRANCH_ORDER
        },
        episode_starts=torch.cat([item.episode_starts for item in copied], dim=1),
    )


@dataclass(frozen=True)
class CentralizedStateTensorBatch:
    """Deterministic centralized critic tensor batch."""

    features: Tensor

    def validate(self, spec: MAPPOTensorSpec) -> tuple[int, int]:
        if self.features.ndim != 3:
            raise MAPPONetworkError("centralized features must be [batch,time,feature]")
        batch, time, dimension = self.features.shape
        if dimension != spec.centralized_state_dim:
            raise MAPPONetworkError("centralized feature dimension differs from tensor spec")
        if not self.features.is_floating_point() or not torch.isfinite(self.features).all():
            raise MAPPONetworkError("centralized features must be finite floating tensors")
        return batch, time

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> "CentralizedStateTensorBatch":
        target_dtype = self.features.dtype if dtype is None else dtype
        return CentralizedStateTensorBatch(
            self.features.to(device=device, dtype=target_dtype)
        )

    def repeat_batch(self, repeats: int) -> "CentralizedStateTensorBatch":
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
            raise MAPPONetworkError("repeats must be a positive integer")
        return CentralizedStateTensorBatch(self.features.repeat(repeats, 1, 1))


def stack_centralized_time(
    steps: Sequence[CentralizedStateTensorBatch],
) -> CentralizedStateTensorBatch:
    copied = tuple(steps)
    if not copied:
        raise MAPPONetworkError("at least one centralized state is required")
    return CentralizedStateTensorBatch(
        torch.cat([item.features for item in copied], dim=1)
    )

class ActorObservationTensorizer:
    """Whitelist-only adapter from ActorObservation to Torch tensors."""

    def __init__(self, config: RunConfig) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        config.validate()
        self.config = config
        self.spec = MAPPOTensorSpec.from_config(config)
        env = config.environment
        self._max_tasks = max(1, env.uav_count * env.episode_horizon)
        self._max_task_bits = float(env.task_data_bits_max)
        self._max_task_cycles = float(env.task_data_bits_max * env.task_cycles_per_bit_max)
        self._max_queue_bits = self._max_tasks * self._max_task_bits
        self._max_queue_cycles = self._max_tasks * self._max_task_cycles
        self._time_scale = float(max(1, env.episode_horizon))
        self._slack_scale = float(max(1, env.maximum_task_slack_slots))
        self._distance_scale = math.sqrt(
            (env.bounds.x_max_m - env.bounds.x_min_m) ** 2
            + (env.bounds.y_max_m - env.bounds.y_min_m) ** 2
            + env.height_m**2
        )
        noise_psd_w_hz = 10.0 ** ((env.noise_psd_dbm_hz - 30.0) / 10.0)
        self._noise_reference_w = max(
            np.finfo(np.float64).tiny,
            noise_psd_w_hz
            * env.ru_bandwidth_hz
            * 10.0 ** (env.receiver_noise_figure_db / 10.0),
        )

    def encode_step(
        self,
        observations: Sequence[ActorObservation],
        proposals: Sequence[ActionProposal] | None = None,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        episode_start: bool | None = None,
    ) -> ActorTensorBatch:
        copied = tuple(observations)
        count = self.spec.uav_count
        if len(copied) != count or any(
            not isinstance(item, ActorObservation) for item in copied
        ):
            raise MAPPONetworkError("one ActorObservation per configured UAV is required")
        if tuple(item.uav_id for item in copied) != tuple(range(count)):
            raise MAPPONetworkError("observations must use stable zero-based UAV order")
        if len({item.slot for item in copied}) != 1:
            raise MAPPONetworkError("all observations must describe one decision slot")
        selected = (
            tuple(proposals)
            if proposals is not None
            else tuple(self._first_legal_proposal(item) for item in copied)
        )
        if len(selected) != count:
            raise MAPPONetworkError("proposals must align with observations")
        for observation, proposal in zip(copied, selected):
            if not observation.action_masks.is_legal(proposal):
                raise MAPPONetworkError("proposal violates its observation masks")

        self_values = np.stack([self._self_vector(item) for item in copied])
        public_values = np.stack([self._public_matrix(item) for item in copied])
        edge_values = np.stack([self._edge_matrix(item) for item in copied])
        graph_masks = np.stack([item.candidate_neighbor_mask for item in copied])

        action_masks: dict[str, Tensor] = {}
        action_indices: dict[str, Tensor] = {}
        for branch in ACTION_BRANCH_ORDER:
            masks = np.stack(
                [
                    observation.action_masks.mask_for(branch, proposal)
                    for observation, proposal in zip(copied, selected)
                ]
            )
            indices = np.asarray(
                [
                    self._branch_index(
                        observation, branch, getattr(proposal, branch)
                    )
                    for observation, proposal in zip(copied, selected)
                ],
                dtype=np.int64,
            )
            action_masks[branch] = torch.as_tensor(
                masks[None, None], dtype=torch.bool, device=device
            )
            action_indices[branch] = torch.as_tensor(
                indices[None, None], dtype=torch.long, device=device
            )

        start = copied[0].slot == 0 if episode_start is None else episode_start
        if not isinstance(start, (bool, np.bool_)):
            raise MAPPONetworkError("episode_start must be boolean")
        result = ActorTensorBatch(
            self_features=torch.as_tensor(
                self_values[None, None], dtype=dtype, device=device
            ),
            neighbor_public_features=torch.as_tensor(
                public_values[None, None], dtype=dtype, device=device
            ),
            edge_features=torch.as_tensor(
                edge_values[None, None], dtype=dtype, device=device
            ),
            neighbor_mask=torch.as_tensor(
                graph_masks[None, None], dtype=torch.bool, device=device
            ),
            action_masks=action_masks,
            action_indices=action_indices,
            episode_starts=torch.full(
                (1, 1, count), bool(start), dtype=torch.bool, device=device
            ),
        )
        result.validate(self.spec)
        return result

    def _self_vector(self, observation: ActorObservation) -> np.ndarray:
        env = self.config.environment
        resources = observation.self_resources
        values: list[float] = [
            resources.residual_energy_j / env.reference_initial_energy_j,
            resources.initial_energy_j / env.reference_initial_energy_j,
            resources.residual_energy_ratio,
            resources.max_transmit_power_w / env.reference_transmit_power_w,
            resources.max_transmit_power_ratio,
            resources.max_cpu_frequency_hz / env.reference_cpu_frequency_hz,
            resources.max_cpu_frequency_ratio,
            resources.cpu_coefficient / env.reference_cpu_coefficient,
            resources.cpu_coefficient_ratio,
        ]
        queues = observation.private_queues
        values.extend(self._queue_vector(queues.unbound))
        values.extend(self._queue_vector(queues.local_cpu))
        for indexed in (queues.tx_by_destination, queues.cpu_by_source):
            for index in range(self.spec.uav_count):
                values.extend(
                    self._queue_vector(
                        QueueSummary(
                            task_count=int(indexed.task_count[index]),
                            remaining_bits=float(indexed.remaining_bits[index]),
                            remaining_cycles=float(indexed.remaining_cycles[index]),
                            head_task_id=int(indexed.head_task_id[index]),
                            head_remaining_bits=float(indexed.head_remaining_bits[index]),
                            head_remaining_cycles=float(indexed.head_remaining_cycles[index]),
                            head_slack_slots=int(indexed.head_slack_slots[index]),
                            head_valid_mask=bool(indexed.head_valid_mask[index]),
                        )
                    )
                )
        values.extend(
            [
                float(observation.arrival_rate_estimate),
                float(observation.arrival_rate_valid_mask),
            ]
        )
        values.extend(observation.candidate_neighbor_mask.astype(np.float64).tolist())
        values.extend(observation.service_queue_available.astype(np.float64).tolist())
        values.extend(self._previous_action_vector(observation))
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (self.spec.self_feature_dim,):
            raise MAPPONetworkError("self-feature encoder produced the wrong dimension")
        return result

    def _queue_vector(self, summary: QueueSummary) -> list[float]:
        return [
            summary.task_count / self._max_tasks,
            summary.remaining_bits / self._max_queue_bits,
            summary.remaining_cycles / self._max_queue_cycles,
            (summary.head_task_id + 1) / self._max_tasks
            if summary.head_valid_mask
            else 0.0,
            summary.head_remaining_bits / self._max_task_bits,
            summary.head_remaining_cycles / self._max_task_cycles,
            summary.head_slack_slots / self._slack_scale
            if summary.head_valid_mask
            else 0.0,
            float(summary.head_valid_mask),
        ]

    def _previous_action_vector(self, observation: ActorObservation) -> list[float]:
        previous = observation.previous_action
        values: list[float] = [float(previous.valid)]
        for branch in ACTION_BRANCH_ORDER:
            domain = observation.action_masks.domain_for(branch)
            index = self._branch_index(
                observation, branch, getattr(previous.proposal, branch)
            )
            one_hot = [0.0] * len(domain)
            one_hot[index] = 1.0
            values.extend(one_hot)
        values.append(float(previous.communication_accepted))
        receiver = [0.0] * self.spec.uav_count
        if previous.communication_receiver_uav is not None:
            receiver[previous.communication_receiver_uav] = 1.0
        values.extend(receiver)
        values.extend(
            [
                previous.executed_ru_count / max(1, self.spec.ru_count),
                previous.ru_utilization_ratio,
                previous.executed_power_ratio,
            ]
        )
        cpu_source = [0.0] * self.spec.uav_count
        if previous.cpu_source_uav is not None:
            cpu_source[previous.cpu_source_uav] = 1.0
        values.extend(cpu_source)
        values.extend(
            [float(previous.cpu_active), previous.executed_cpu_frequency_ratio]
        )
        return values

    def _public_matrix(self, observation: ActorObservation) -> np.ndarray:
        env = self.config.environment
        public = observation.neighbor_public
        span_x = max(1.0, env.bounds.x_max_m - env.bounds.x_min_m)
        span_y = max(1.0, env.bounds.y_max_m - env.bounds.y_min_m)
        velocity_scale = np.asarray(
            [
                max(1.0, abs(env.velocity_std_mps[0])),
                max(1.0, abs(env.velocity_std_mps[1])),
                1.0,
            ],
            dtype=np.float64,
        )
        result = np.zeros(
            (self.spec.uav_count, self.spec.neighbor_public_feature_dim),
            dtype=np.float64,
        )
        for neighbor in range(self.spec.uav_count):
            result[neighbor] = np.asarray(
                [
                    float(public.valid_mask[neighbor]),
                    public.relative_position_m[neighbor, 0] / span_x,
                    public.relative_position_m[neighbor, 1] / span_y,
                    public.relative_position_m[neighbor, 2] / max(1.0, env.height_m),
                    *(public.relative_velocity_mps[neighbor] / velocity_scale),
                    public.residual_energy_j[neighbor] / env.reference_initial_energy_j,
                    public.residual_energy_ratio[neighbor],
                    public.max_transmit_power_ratio[neighbor],
                    public.max_cpu_frequency_ratio[neighbor],
                    public.cpu_coefficient_ratio[neighbor],
                    public.cpu_load_task_count[neighbor] / self._max_tasks,
                    public.cpu_load_remaining_cycles[neighbor] / self._max_queue_cycles,
                    (
                        (public.message_source_slots[neighbor] + 1) / self._time_scale
                        if public.valid_mask[neighbor]
                        else 0.0
                    ),
                    (
                        (public.message_aoi_slots[neighbor] + 1) / self._time_scale
                        if public.message_aoi_valid_mask[neighbor]
                        else 0.0
                    ),
                    float(public.message_aoi_valid_mask[neighbor]),
                ],
                dtype=np.float64,
            )
        return result
    def _edge_matrix(self, observation: ActorObservation) -> np.ndarray:
        edge = observation.edge_history
        result = np.zeros(
            (self.spec.uav_count, self.spec.edge_feature_dim), dtype=np.float64
        )
        for neighbor in range(self.spec.uav_count):
            stale = edge.stale_csi[neighbor]
            row: list[float] = [
                float(edge.visible_mask[neighbor]),
                edge.estimated_distance_m[neighbor] / self._distance_scale,
            ]
            row.extend(stale.real.tolist())
            row.extend(stale.imag.tolist())
            row.extend(
                [
                    float(edge.csi_valid_mask[neighbor]),
                    edge.csi_aoi_slots[neighbor] / self._time_scale,
                ]
            )
            row.extend(
                (edge.interference_history_w[neighbor] / self._noise_reference_w).tolist()
            )
            row.extend(
                edge.interference_valid_mask[neighbor].astype(np.float64).tolist()
            )
            row.extend(
                [
                    (
                        (edge.message_aoi_slots[neighbor] + 1) / self._time_scale
                        if edge.message_aoi_valid_mask[neighbor]
                        else 0.0
                    ),
                    float(edge.message_aoi_valid_mask[neighbor]),
                ]
            )
            row.extend(edge.historical_quality[neighbor].tolist())
            row.extend(edge.quality_valid_mask[neighbor].astype(np.float64).tolist())
            row.extend(
                [
                    edge.last_effective_rate_bps[neighbor]
                    / self.config.environment.reference_rate_bps,
                    float(edge.last_rate_valid_mask[neighbor]),
                    edge.outage_rate[neighbor],
                    float(edge.outage_valid_mask[neighbor]),
                ]
            )
            result[neighbor] = np.asarray(row, dtype=np.float64)
        return result

    def _first_legal_proposal(self, observation: ActorObservation) -> ActionProposal:
        values: dict[str, str | int | float] = {}
        for branch in ACTION_BRANCH_ORDER:
            valid = np.flatnonzero(
                observation.action_masks.mask_for(branch, values)
            )
            if valid.size == 0:
                raise MAPPONetworkError(f"{branch} has no legal action")
            values[branch] = observation.action_masks.domain_for(branch)[int(valid[0])]
        return ActionProposal(
            uav_id=observation.uav_id,
            route=values["route"],
            tx_select=values["tx_select"],
            resource_group=values["resource_group"],
            resource_width=int(values["resource_width"]),
            power_level=float(values["power_level"]),
            cpu_queue=values["cpu_queue"],
            cpu_frequency=float(values["cpu_frequency"]),
        )

    @staticmethod
    def _branch_index(
        observation: ActorObservation,
        branch: str,
        raw_value: object,
    ) -> int:
        value = raw_value
        if branch == "cpu_queue" and value == "local":
            value = observation.uav_id
        if branch in {"power_level", "cpu_frequency"} and isinstance(value, (int, float)):
            value = float(value)
        for index, candidate in enumerate(
            observation.action_masks.domain_for(branch)
        ):
            if isinstance(candidate, float) and isinstance(value, (int, float)):
                if not isinstance(value, bool) and candidate == float(value):
                    return index
            elif type(candidate) is type(value) and candidate == value:
                return index
        raise MAPPONetworkError(f"{branch} value is outside the observation domain")


class CentralizedStateTensorizer:
    """Adapter from the official pre-action CentralizedState interface."""

    def __init__(self, config: RunConfig) -> None:
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        config.validate()
        self.config = config
        self.spec = MAPPOTensorSpec.from_config(config)
        env = config.environment
        self._max_tasks = max(1, env.uav_count * env.episode_horizon)
        self._max_task_bits = float(env.task_data_bits_max)
        self._max_task_cycles = float(env.task_data_bits_max * env.task_cycles_per_bit_max)
        self._time_scale = float(max(1, env.episode_horizon))
        self._distance_scale = math.sqrt(
            (env.bounds.x_max_m - env.bounds.x_min_m) ** 2
            + (env.bounds.y_max_m - env.bounds.y_min_m) ** 2
            + env.height_m**2
        )
        noise_psd_w_hz = 10.0 ** ((env.noise_psd_dbm_hz - 30.0) / 10.0)
        self._noise_reference_w = max(
            np.finfo(np.float64).tiny,
            noise_psd_w_hz
            * env.ru_bandwidth_hz
            * 10.0 ** (env.receiver_noise_figure_db / 10.0),
        )

    def encode_step(
        self,
        state: CentralizedState,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> CentralizedStateTensorBatch:
        vector = self._state_vector(state)
        result = CentralizedStateTensorBatch(
            torch.as_tensor(vector[None, None], dtype=dtype, device=device)
        )
        result.validate(self.spec)
        return result

    def _state_vector(self, state: CentralizedState) -> np.ndarray:
        if not isinstance(state, CentralizedState):
            raise TypeError("state must be a CentralizedState")
        count = self.spec.uav_count
        env = self.config.environment
        if state.uav_ids.shape != (count,) or state.true_channel.shape != (
            count,
            count,
            self.spec.ru_count,
        ):
            raise MAPPONetworkError("centralized state differs from configured scenario")

        span_x = max(1.0, env.bounds.x_max_m - env.bounds.x_min_m)
        span_y = max(1.0, env.bounds.y_max_m - env.bounds.y_min_m)
        positions = state.positions_m.copy()
        positions[:, 0] = (positions[:, 0] - env.bounds.x_min_m) / span_x
        positions[:, 1] = (positions[:, 1] - env.bounds.y_min_m) / span_y
        positions[:, 2] = positions[:, 2] / max(1.0, env.height_m)
        velocity_scale = np.asarray(
            [
                max(1.0, abs(env.velocity_std_mps[0])),
                max(1.0, abs(env.velocity_std_mps[1])),
                1.0,
            ]
        )

        values: list[float] = [state.slot / self._time_scale]
        values.extend(positions.reshape(-1).tolist())
        values.extend((state.velocities_mps / velocity_scale).reshape(-1).tolist())
        values.extend(
            (state.max_transmit_power_w / env.reference_transmit_power_w).tolist()
        )
        values.extend(
            (state.max_cpu_frequency_hz / env.reference_cpu_frequency_hz).tolist()
        )
        values.extend((state.cpu_coefficient / env.reference_cpu_coefficient).tolist())
        values.extend((state.residual_energy_j / env.reference_initial_energy_j).tolist())
        values.extend(state.true_arrival_probabilities.tolist())
        values.extend(state.candidate_neighbor_mask.astype(float).reshape(-1).tolist())
        values.extend((state.distances_m / self._distance_scale).reshape(-1).tolist())
        values.extend(state.true_channel.real.reshape(-1).tolist())
        values.extend(state.true_channel.imag.reshape(-1).tolist())
        values.extend(state.blocked_links.astype(float).reshape(-1).tolist())
        values.extend(
            (state.shadowing_db / max(1.0, env.shadowing_std_db)).reshape(-1).tolist()
        )
        values.extend(state.path_loss_db.reshape(-1).tolist())
        values.extend(state.stale_csi.real.reshape(-1).tolist())
        values.extend(state.stale_csi.imag.reshape(-1).tolist())
        values.extend(state.csi_valid_mask.astype(float).reshape(-1).tolist())
        values.extend((state.csi_aoi_slots / self._time_scale).reshape(-1).tolist())
        values.extend(
            (state.interference_history_w / self._noise_reference_w).reshape(-1).tolist()
        )
        values.extend(state.interference_valid_mask.astype(float).reshape(-1).tolist())
        values.extend(((state.message_aoi_slots + 1) / self._time_scale).tolist())
        values.extend(state.message_aoi_valid_mask.astype(float).tolist())
        values.extend(
            (state.historical_denominator_w / self._noise_reference_w).reshape(-1).tolist()
        )
        values.extend(state.historical_quality.reshape(-1).tolist())
        values.extend(state.quality_valid_mask.astype(float).reshape(-1).tolist())
        values.extend(self._task_aggregates(state))
        values.extend(self._queue_aggregates(state))
        if len(state.buildings) != len(env.building_layout):
            raise MAPPONetworkError("building snapshot differs from config")
        for building in state.buildings:
            values.extend(
                [
                    1.0,
                    (building.x_min_m - env.bounds.x_min_m) / span_x,
                    (building.x_max_m - env.bounds.x_min_m) / span_x,
                    (building.y_min_m - env.bounds.y_min_m) / span_y,
                    (building.y_max_m - env.bounds.y_min_m) / span_y,
                    building.height_m / max(1.0, env.height_m),
                ]
            )

        result = np.asarray(values, dtype=np.float64)
        if result.shape != (self.spec.centralized_state_dim,):
            raise MAPPONetworkError("centralized encoder produced the wrong dimension")
        if not np.all(np.isfinite(result)):
            raise MAPPONetworkError("centralized encoder produced NaN or Inf")
        return result

    def _task_aggregates(self, state: CentralizedState) -> list[float]:
        count = self.spec.uav_count
        aggregate = np.zeros((count, 10), dtype=np.float64)
        destination_counts = np.zeros((count, count), dtype=np.float64)
        slack_min = np.full(count, np.inf, dtype=np.float64)
        for task in state.active_tasks:
            source = task.source_uav
            aggregate[source, 0] += 1.0
            aggregate[source, 1] += task.data_bits / self._max_task_bits
            aggregate[source, 2] += task.cpu_cycles / self._max_task_cycles
            aggregate[source, 3] += task.remaining_bits / self._max_task_bits
            aggregate[source, 4] += task.remaining_cycles / self._max_task_cycles
            slack_min[source] = min(slack_min[source], task.deadline_slot - state.slot)
            try:
                status_index = _ACTIVE_TASK_STATUSES.index(task.status)
            except ValueError as exc:
                raise MAPPONetworkError("active task has terminal or unknown status") from exc
            aggregate[source, 6 + status_index] += 1.0
            if task.destination_uav is not None:
                destination_counts[source, task.destination_uav] += 1.0
        aggregate[:, 0] /= self._max_tasks
        aggregate[:, 1:5] /= self._max_tasks
        aggregate[:, 5] = np.where(
            np.isfinite(slack_min),
            slack_min / max(1, self.config.environment.maximum_task_slack_slots),
            0.0,
        )
        aggregate[:, 6:10] /= self._max_tasks
        destination_counts /= self._max_tasks
        return [*aggregate.reshape(-1).tolist(), *destination_counts.reshape(-1).tolist()]

    def _queue_aggregates(self, state: CentralizedState) -> list[float]:
        count = self.spec.uav_count
        values = np.zeros((len(_QUEUE_KINDS), count, count), dtype=np.float64)
        for queue in state.queues:
            kind = _QUEUE_KINDS.index(queue.kind)
            destination = (
                queue.source_uav
                if queue.destination_uav is None
                else queue.destination_uav
            )
            values[kind, queue.source_uav, destination] = (
                len(queue.task_ids) / self._max_tasks
            )
        return values.reshape(-1).tolist()

class CAGATv2Layer(nn.Module):
    """One ego-centered channel/context-aware GATv2 layer."""

    def __init__(self, hidden_dimension: int, edge_dimension: int, heads: int) -> None:
        super().__init__()
        if hidden_dimension % heads != 0:
            raise MAPPONetworkError("hidden dimension must be divisible by heads")
        self.hidden_dimension = hidden_dimension
        self.edge_dimension = edge_dimension
        self.heads = heads
        self.head_dimension = hidden_dimension // heads
        self.query = nn.ModuleList(
            [nn.Linear(hidden_dimension, self.head_dimension) for _ in range(heads)]
        )
        self.edge_projection = nn.ModuleList(
            [nn.Linear(edge_dimension, self.head_dimension) for _ in range(heads)]
        )
        pair_dimension = hidden_dimension + self.head_dimension
        self.key = nn.ModuleList(
            [nn.Linear(pair_dimension, self.head_dimension) for _ in range(heads)]
        )
        self.value = nn.ModuleList(
            [nn.Linear(pair_dimension, self.head_dimension) for _ in range(heads)]
        )
        self.attention_vector = nn.Parameter(torch.empty(heads, self.head_dimension))
        nn.init.xavier_uniform_(self.attention_vector)
        self.output_projection = nn.Linear(hidden_dimension, hidden_dimension)
        self.normalization = nn.LayerNorm(hidden_dimension)

    def forward(
        self,
        self_state: Tensor,
        neighbor_public_state: Tensor,
        edge_features: Tensor,
        neighbor_mask: Tensor,
        *,
        return_candidate_values: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        if self_state.ndim != 2:
            raise MAPPONetworkError("CA-GAT self state must be [item,hidden]")
        items, hidden = self_state.shape
        if neighbor_public_state.ndim != 3:
            raise MAPPONetworkError("CA-GAT public state must be [item,neighbor,hidden]")
        neighbors = neighbor_public_state.shape[1]
        if hidden != self.hidden_dimension:
            raise MAPPONetworkError("CA-GAT self hidden dimension mismatch")
        if tuple(neighbor_public_state.shape) != (items, neighbors, hidden):
            raise MAPPONetworkError("CA-GAT public hidden shape mismatch")
        if tuple(edge_features.shape) != (items, neighbors, self.edge_dimension):
            raise MAPPONetworkError("CA-GAT edge shape mismatch")
        if tuple(neighbor_mask.shape) != (items, neighbors) or neighbor_mask.dtype != torch.bool:
            raise MAPPONetworkError("CA-GAT neighbor mask mismatch")

        outputs: list[Tensor] = []
        self_values: list[Tensor] = []
        candidate_values: list[Tensor] = []
        for head in range(self.heads):
            edge_state = self.edge_projection[head](edge_features)
            neighbor_pair = torch.cat([neighbor_public_state, edge_state], dim=-1)
            query = self.query[head](self_state)
            key = self.key[head](neighbor_pair)
            value = self.value[head](neighbor_pair)
            self_edge = torch.zeros(
                (items, self.head_dimension),
                device=self_state.device,
                dtype=self_state.dtype,
            )
            self_pair = torch.cat([self_state, self_edge], dim=-1)
            self_key = self.key[head](self_pair)
            self_value = self.value[head](self_pair)
            if return_candidate_values:
                self_values.append(self_value)
                candidate_values.append(value)
            self_score = torch.sum(
                self.attention_vector[head]
                * torch.nn.functional.leaky_relu(query + self_key),
                dim=-1,
                keepdim=True,
            )
            neighbor_score = torch.sum(
                self.attention_vector[head]
                * torch.nn.functional.leaky_relu(query[:, None, :] + key),
                dim=-1,
            )
            scores = torch.cat([self_score, neighbor_score], dim=-1)
            valid = torch.cat(
                [
                    torch.ones((items, 1), dtype=torch.bool, device=self_state.device),
                    neighbor_mask,
                ],
                dim=-1,
            )
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            attention = torch.softmax(scores, dim=-1)
            all_values = torch.cat([self_value[:, None, :], value], dim=1)
            outputs.append(torch.sum(attention[..., None] * all_values, dim=1))
        aggregate = torch.cat(outputs, dim=-1)
        update = torch.nn.functional.leaky_relu(self.output_projection(aggregate))
        result = self.normalization(self_state + update)
        if return_candidate_values:
            return (
                result,
                torch.cat(self_values, dim=-1),
                torch.cat(candidate_values, dim=-1),
            )
        return result


class CandidateAwareRouteDecoderV1(nn.Module):
    """Shared candidate scorer with explicit destination-to-logit wiring."""

    def __init__(
        self,
        *,
        uav_count: int,
        recurrent_dimension: int,
        candidate_dimension: int,
        edge_dimension: int,
    ) -> None:
        super().__init__()
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (
                uav_count,
                recurrent_dimension,
                candidate_dimension,
                edge_dimension,
            )
        ):
            raise MAPPONetworkError(
                "candidate-aware decoder dimensions must be positive integers"
            )
        if uav_count < 2:
            raise MAPPONetworkError(
                "candidate-aware decoder requires at least two UAVs"
            )
        self.uav_count = uav_count
        self.recurrent_dimension = recurrent_dimension
        self.candidate_dimension = candidate_dimension
        self.edge_dimension = edge_dimension
        self.edge_projection = nn.Linear(edge_dimension, candidate_dimension)
        self.candidate_normalization = nn.LayerNorm(candidate_dimension)
        self.query_projection = nn.Linear(
            recurrent_dimension, candidate_dimension
        )
        self.candidate_projection = nn.Linear(
            candidate_dimension, candidate_dimension
        )
        self.fixed_head = nn.Linear(recurrent_dimension, 3)
        self.remote_scorer = nn.Linear(candidate_dimension, 1)
        self.register_buffer(
            "remote_candidate_ids",
            torch.tensor(
                [
                    [candidate for candidate in range(uav_count) if candidate != ego]
                    for ego in range(uav_count)
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )

    def forward(
        self,
        recurrent_features: Tensor,
        candidate_public_features: Tensor,
        candidate_edge_features: Tensor,
    ) -> Tensor:
        if recurrent_features.ndim != 4:
            raise MAPPONetworkError(
                "route recurrent features must be [batch,time,agent,hidden]"
            )
        batch, time, agents, recurrent_dimension = recurrent_features.shape
        expected_public = (
            batch,
            time,
            agents,
            self.uav_count,
            self.candidate_dimension,
        )
        expected_edge = (
            batch,
            time,
            agents,
            self.uav_count,
            self.edge_dimension,
        )
        if (
            agents != self.uav_count
            or recurrent_dimension != self.recurrent_dimension
        ):
            raise MAPPONetworkError("route recurrent features differ from decoder spec")
        if tuple(candidate_public_features.shape) != expected_public:
            raise MAPPONetworkError(
                "candidate public features differ from decoder spec"
            )
        if tuple(candidate_edge_features.shape) != expected_edge:
            raise MAPPONetworkError(
                "candidate edge features differ from decoder spec"
            )
        tensors = (
            recurrent_features,
            candidate_public_features,
            candidate_edge_features,
        )
        if any(not tensor.is_floating_point() for tensor in tensors):
            raise MAPPONetworkError("route decoder inputs must be floating tensors")
        if any(
            tensor.device != recurrent_features.device
            or tensor.dtype != recurrent_features.dtype
            for tensor in tensors[1:]
        ):
            raise MAPPONetworkError(
                "route decoder inputs must share device and dtype"
            )
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise MAPPONetworkError("route decoder inputs contain NaN or Inf")

        candidate_state = self.candidate_normalization(
            candidate_public_features
            + self.edge_projection(candidate_edge_features)
        )
        query = self.query_projection(recurrent_features).unsqueeze(-2)
        candidate_key = self.candidate_projection(candidate_state)
        all_candidate_logits = self.remote_scorer(
            torch.tanh(query + candidate_key)
        ).squeeze(-1)
        gather_index = self.remote_candidate_ids.view(
            1, 1, self.uav_count, self.uav_count - 1
        ).expand(batch, time, -1, -1)
        remote_logits = torch.gather(all_candidate_logits, -1, gather_index)
        logits = torch.cat(
            [self.fixed_head(recurrent_features), remote_logits], dim=-1
        )
        expected_logits = (batch, time, agents, self.uav_count + 2)
        if tuple(logits.shape) != expected_logits or not torch.isfinite(logits).all():
            raise MAPPONetworkError("candidate-aware route logits are invalid")
        return logits


class OptionAwareRouteDecoderV1(nn.Module):
    """Score Local and every Remote option with one shared final scorer."""

    _BLOCK_DIMENSION = 32
    _OPTION_TYPE_DIMENSION = 8
    _SCORER_HIDDEN_DIMENSION = 64
    CANDIDATE_VALUE_DIMENSION = 2 * (6 * 32 + 8 + _OPTION_VALIDITY_FEATURE_DIM) + 3

    def __init__(self, *, spec: MAPPOTensorSpec) -> None:
        super().__init__()
        if spec.uav_count < 2:
            raise MAPPONetworkError(
                "option-aware decoder requires at least two UAVs"
            )
        self.uav_count = spec.uav_count
        self.recurrent_dimension = spec.gru_hidden_dimension
        self.option_dimension = spec.encoder_hidden_dimension
        self.feature_extractor = RouteOptionFeatureExtractor(spec)

        def projection(input_dimension: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dimension, self._BLOCK_DIMENSION),
                nn.LayerNorm(self._BLOCK_DIMENSION),
                nn.LeakyReLU(),
            )

        self.recurrent_projection = projection(self.recurrent_dimension)
        self.cooperative_projection = projection(self.option_dimension)
        self.task_projection = projection(_TASK_OPTION_FEATURE_DIM)
        self.source_projection = projection(_SOURCE_OPTION_FEATURE_DIM)
        self.endpoint_projection = projection(self.option_dimension)
        self.burden_projection = projection(_BURDEN_OPTION_FEATURE_DIM)
        self.option_type_embedding = nn.Embedding(2, self._OPTION_TYPE_DIMENSION)
        fusion_dimension = (
            6 * self._BLOCK_DIMENSION
            + self._OPTION_TYPE_DIMENSION
            + _OPTION_VALIDITY_FEATURE_DIM
        )
        self.fusion_normalization = nn.LayerNorm(fusion_dimension)
        self.shared_option_scorer = nn.Sequential(
            nn.Linear(fusion_dimension, self._SCORER_HIDDEN_DIMENSION),
            nn.LeakyReLU(),
            nn.Linear(self._SCORER_HIDDEN_DIMENSION, 1),
        )
        self.control_head = nn.Linear(self.recurrent_dimension, 2)
        self.register_buffer(
            "remote_candidate_ids",
            torch.tensor(
                [
                    [candidate for candidate in range(self.uav_count) if candidate != ego]
                    for ego in range(self.uav_count)
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )

    def _gather_remote_endpoints(self, candidate_endpoints: Tensor) -> Tensor:
        batch, time, agents, _, features = candidate_endpoints.shape
        index = self.remote_candidate_ids.view(
            1, 1, agents, self.uav_count - 1, 1
        ).expand(batch, time, agents, self.uav_count - 1, features)
        return torch.gather(candidate_endpoints, -2, index)

    def forward(
        self,
        recurrent_features: Tensor,
        cooperative_features: Tensor,
        local_endpoint_features: Tensor,
        candidate_endpoint_features: Tensor,
        self_features: Tensor,
        candidate_public_features: Tensor,
        candidate_edge_features: Tensor,
        *,
        route_action_mask: Tensor | None = None,
        route_candidate_sink: list[Tensor] | None = None,
    ) -> Tensor:
        if recurrent_features.ndim != 4:
            raise MAPPONetworkError(
                "option-aware recurrent features must be [batch,time,agent,hidden]"
            )
        batch, time, agents, recurrent_dimension = recurrent_features.shape
        common_shape = (batch, time, agents)
        expected_shapes = (
            (cooperative_features, (*common_shape, self.option_dimension)),
            (local_endpoint_features, (*common_shape, self.option_dimension)),
            (
                candidate_endpoint_features,
                (*common_shape, self.uav_count, self.option_dimension),
            ),
            (self_features, (*common_shape, self.feature_extractor.self_feature_dim)),
            (
                candidate_public_features,
                (
                    *common_shape,
                    self.uav_count,
                    self.feature_extractor.public_feature_dim,
                ),
            ),
            (
                candidate_edge_features,
                (
                    *common_shape,
                    self.uav_count,
                    self.feature_extractor.edge_feature_dim,
                ),
            ),
        )
        if agents != self.uav_count or recurrent_dimension != self.recurrent_dimension:
            raise MAPPONetworkError(
                "option-aware recurrent features differ from decoder spec"
            )
        if any(tuple(tensor.shape) != shape for tensor, shape in expected_shapes):
            raise MAPPONetworkError("option-aware input features differ from decoder spec")
        tensors = (recurrent_features,) + tuple(
            tensor for tensor, _ in expected_shapes
        )
        if any(not tensor.is_floating_point() for tensor in tensors):
            raise MAPPONetworkError("option-aware inputs must be floating tensors")
        if any(
            tensor.device != recurrent_features.device
            or tensor.dtype != recurrent_features.dtype
            for tensor in tensors[1:]
        ):
            raise MAPPONetworkError(
                "option-aware inputs must share device and dtype"
            )
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise MAPPONetworkError("option-aware inputs contain NaN or Inf")

        (
            task_features,
            source_features,
            local_burden,
            remote_burden,
            local_validity,
            remote_validity,
        ) = self.feature_extractor.extract(
            self_features,
            candidate_public_features,
            candidate_edge_features,
            self.remote_candidate_ids,
        )
        remote_endpoints = self._gather_remote_endpoints(
            candidate_endpoint_features
        )
        option_endpoints = torch.cat(
            [local_endpoint_features.unsqueeze(-2), remote_endpoints], dim=-2
        )
        option_burdens = torch.cat(
            [local_burden.unsqueeze(-2), remote_burden], dim=-2
        )
        option_validity = torch.cat(
            [local_validity.unsqueeze(-2), remote_validity], dim=-2
        )
        option_count = self.uav_count

        def expand_options(features: Tensor) -> Tensor:
            return features.unsqueeze(-2).expand(
                batch, time, agents, option_count, features.shape[-1]
            )

        option_type_ids = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=recurrent_features.device),
                torch.ones(
                    option_count - 1,
                    dtype=torch.long,
                    device=recurrent_features.device,
                ),
            ]
        )
        option_types = self.option_type_embedding(option_type_ids).view(
            1, 1, 1, option_count, self._OPTION_TYPE_DIMENSION
        ).expand(batch, time, agents, -1, -1)
        fused = self.fusion_normalization(self.augment_option_features(
            torch.cat(
                [
                    expand_options(self.recurrent_projection(recurrent_features)),
                    expand_options(self.cooperative_projection(cooperative_features)),
                    expand_options(self.task_projection(task_features)),
                    expand_options(self.source_projection(source_features)),
                    self.endpoint_projection(option_endpoints),
                    self.burden_projection(option_burdens),
                    option_types,
                    option_validity,
                ],
                dim=-1,
            ), self_features, candidate_public_features, candidate_edge_features
        ))
        option_logits = self.shared_option_scorer(fused).squeeze(-1)
        control_logits = self.control_head(recurrent_features)
        logits = torch.cat(
            [
                control_logits[..., :1],
                option_logits[..., :1],
                control_logits[..., 1:],
                option_logits[..., 1:],
            ],
            dim=-1,
        )
        expected_logits = (batch, time, agents, self.uav_count + 2)
        if tuple(logits.shape) != expected_logits or not torch.isfinite(logits).all():
            raise MAPPONetworkError("option-aware route logits are invalid")
        if route_candidate_sink is not None:
            if (route_action_mask is None or route_action_mask.shape != logits.shape
                    or route_action_mask.dtype != torch.bool):
                raise MAPPONetworkError("unified Route Q requires the matching legal mask")
            # Same 411-dim layout as P1a: candidate, legal-Remote context,
            # Local/Defer/Remote type bits. Encoder inputs are predictions only.
            weights = route_action_mask[..., 3:].to(fused).unsqueeze(-1)
            context = (fused[..., 1:, :] * weights).sum(-2) / weights.sum(-2).clamp_min(1.)
            local = fused[..., :1, :]
            leaves = torch.cat((local, local, fused[..., 1:, :]), -2)
            types = leaves.new_zeros((*leaves.shape[:-1], 3))
            types[..., 0, 0] = 1; types[..., 1, 1] = 1; types[..., 2:, 2] = 1
            features = torch.cat((leaves, context.unsqueeze(-2).expand_as(leaves), types), -1).detach()
            idle = features.new_zeros((*features.shape[:-2], 1, features.shape[-1]))
            route_candidate_sink.append(torch.cat((idle, features), -2))
        return logits

    def augment_option_features(self, fused: Tensor, self_features: Tensor,
                                public_features: Tensor, edge_features: Tensor) -> Tensor:
        """Identity for A/B; C supplies actor-visible relations before normalization."""
        return fused


class HierarchicalOptionAwareRouteDecoderV1(nn.Module):
    """Compose top-level route and conditional remote-destination probabilities."""

    _BLOCK_DIMENSION = 32
    _OPTION_TYPE_DIMENSION = 8
    _SCORER_HIDDEN_DIMENSION = 64
    CANDIDATE_VALUE_DIMENSION = 2 * (6 * 32 + 8 + _OPTION_VALIDITY_FEATURE_DIM) + 3

    def __init__(self, *, spec: MAPPOTensorSpec) -> None:
        super().__init__()
        if spec.uav_count < 2:
            raise MAPPONetworkError(
                "hierarchical option-aware decoder requires at least two UAVs"
            )
        self.uav_count = spec.uav_count
        self.recurrent_dimension = spec.gru_hidden_dimension
        self.option_dimension = spec.encoder_hidden_dimension
        self.feature_extractor = RouteOptionFeatureExtractor(spec)

        def projection(input_dimension: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dimension, self._BLOCK_DIMENSION),
                nn.LayerNorm(self._BLOCK_DIMENSION),
                nn.LeakyReLU(),
            )

        self.recurrent_projection = projection(self.recurrent_dimension)
        self.cooperative_projection = projection(self.option_dimension)
        self.task_projection = projection(_TASK_OPTION_FEATURE_DIM)
        self.source_projection = projection(_SOURCE_OPTION_FEATURE_DIM)
        self.endpoint_projection = projection(self.option_dimension)
        self.burden_projection = projection(_BURDEN_OPTION_FEATURE_DIM)
        self.option_type_embedding = nn.Embedding(2, self._OPTION_TYPE_DIMENSION)
        fusion_dimension = (
            6 * self._BLOCK_DIMENSION
            + self._OPTION_TYPE_DIMENSION
            + _OPTION_VALIDITY_FEATURE_DIM
        )
        self.fusion_normalization = nn.LayerNorm(fusion_dimension)
        self.top_level_head = nn.Sequential(
            nn.Linear(2 * fusion_dimension, self._SCORER_HIDDEN_DIMENSION),
            nn.LeakyReLU(),
            nn.Linear(self._SCORER_HIDDEN_DIMENSION, 3),
        )
        self.remote_destination_head = nn.Sequential(
            nn.Linear(fusion_dimension, self._SCORER_HIDDEN_DIMENSION),
            nn.LeakyReLU(),
            nn.Linear(self._SCORER_HIDDEN_DIMENSION, 1),
        )
        self.system_destination_scorer = None
        self.register_buffer(
            "remote_candidate_ids",
            torch.tensor(
                [
                    [candidate for candidate in range(self.uav_count) if candidate != ego]
                    for ego in range(self.uav_count)
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )

    def _gather_remote_endpoints(self, candidate_endpoints: Tensor) -> Tensor:
        batch, time, agents, _, features = candidate_endpoints.shape
        index = self.remote_candidate_ids.view(
            1, 1, agents, self.uav_count - 1, 1
        ).expand(batch, time, agents, self.uav_count - 1, features)
        return torch.gather(candidate_endpoints, -2, index)

    def forward(
        self,
        recurrent_features: Tensor,
        cooperative_features: Tensor,
        local_endpoint_features: Tensor,
        candidate_endpoint_features: Tensor,
        self_features: Tensor,
        candidate_public_features: Tensor,
        candidate_edge_features: Tensor,
        route_action_mask: Tensor,
        *,
        route_candidate_sink: list[Tensor] | None = None,
        destination_sink: list[Tensor] | None = None,
    ) -> Tensor:
        if recurrent_features.ndim != 4:
            raise MAPPONetworkError(
                "hierarchical option-aware recurrent features must be "
                "[batch,time,agent,hidden]"
            )
        batch, time, agents, recurrent_dimension = recurrent_features.shape
        common_shape = (batch, time, agents)
        expected_shapes = (
            (cooperative_features, (*common_shape, self.option_dimension)),
            (local_endpoint_features, (*common_shape, self.option_dimension)),
            (
                candidate_endpoint_features,
                (*common_shape, self.uav_count, self.option_dimension),
            ),
            (self_features, (*common_shape, self.feature_extractor.self_feature_dim)),
            (
                candidate_public_features,
                (
                    *common_shape,
                    self.uav_count,
                    self.feature_extractor.public_feature_dim,
                ),
            ),
            (
                candidate_edge_features,
                (
                    *common_shape,
                    self.uav_count,
                    self.feature_extractor.edge_feature_dim,
                ),
            ),
        )
        if agents != self.uav_count or recurrent_dimension != self.recurrent_dimension:
            raise MAPPONetworkError(
                "hierarchical option-aware recurrent features differ from decoder spec"
            )
        if any(tuple(tensor.shape) != shape for tensor, shape in expected_shapes):
            raise MAPPONetworkError(
                "hierarchical option-aware input features differ from decoder spec"
            )
        tensors = (recurrent_features,) + tuple(
            tensor for tensor, _ in expected_shapes
        )
        if any(not tensor.is_floating_point() for tensor in tensors):
            raise MAPPONetworkError(
                "hierarchical option-aware inputs must be floating tensors"
            )
        if any(
            tensor.device != recurrent_features.device
            or tensor.dtype != recurrent_features.dtype
            for tensor in tensors[1:]
        ):
            raise MAPPONetworkError(
                "hierarchical option-aware inputs must share device and dtype"
            )
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise MAPPONetworkError(
                "hierarchical option-aware inputs contain NaN or Inf"
            )
        expected_mask = (*common_shape, self.uav_count + 2)
        if (
            tuple(route_action_mask.shape) != expected_mask
            or route_action_mask.dtype != torch.bool
            or route_action_mask.device != recurrent_features.device
        ):
            raise MAPPONetworkError(
                "hierarchical route action mask must be boolean [batch,time,agent,action]"
            )
        if not torch.all(torch.any(route_action_mask, dim=-1)):
            raise MAPPONetworkError("hierarchical route action mask has no legal action")
        idle_legal = route_action_mask[..., 0]
        non_idle_legal = torch.any(route_action_mask[..., 1:], dim=-1)
        if torch.any(idle_legal & non_idle_legal):
            raise MAPPONetworkError(
                "idle cannot coexist with active hierarchical route actions"
            )

        (
            task_features,
            source_features,
            local_burden,
            remote_burden,
            local_validity,
            remote_validity,
        ) = self.feature_extractor.extract(
            self_features,
            candidate_public_features,
            candidate_edge_features,
            self.remote_candidate_ids,
        )
        remote_endpoints = self._gather_remote_endpoints(
            candidate_endpoint_features
        )
        option_endpoints = torch.cat(
            [local_endpoint_features.unsqueeze(-2), remote_endpoints], dim=-2
        )
        option_burdens = torch.cat(
            [local_burden.unsqueeze(-2), remote_burden], dim=-2
        )
        option_validity = torch.cat(
            [local_validity.unsqueeze(-2), remote_validity], dim=-2
        )
        option_count = self.uav_count

        def expand_options(features: Tensor) -> Tensor:
            return features.unsqueeze(-2).expand(
                batch, time, agents, option_count, features.shape[-1]
            )

        option_type_ids = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=recurrent_features.device),
                torch.ones(
                    option_count - 1,
                    dtype=torch.long,
                    device=recurrent_features.device,
                ),
            ]
        )
        option_types = self.option_type_embedding(option_type_ids).view(
            1, 1, 1, option_count, self._OPTION_TYPE_DIMENSION
        ).expand(batch, time, agents, -1, -1)
        fused = self.fusion_normalization(
            torch.cat(
                [
                    expand_options(self.recurrent_projection(recurrent_features)),
                    expand_options(self.cooperative_projection(cooperative_features)),
                    expand_options(self.task_projection(task_features)),
                    expand_options(self.source_projection(source_features)),
                    self.endpoint_projection(option_endpoints),
                    self.burden_projection(option_burdens),
                    option_types,
                    option_validity,
                ],
                dim=-1,
            )
        )

        remote_mask = route_action_mask[..., 3:]
        remote_legal = torch.any(remote_mask, dim=-1)
        remote_fused = fused[..., 1:, :]
        remote_weights = remote_mask.to(dtype=fused.dtype).unsqueeze(-1)
        remote_summary = (remote_fused * remote_weights).sum(dim=-2) / remote_weights.sum(
            dim=-2
        ).clamp_min(1.0)
        top_level_logits = self.top_level_head(
            torch.cat([fused[..., 0, :], remote_summary], dim=-1)
        )
        top_level_mask = torch.stack(
            [
                route_action_mask[..., 1],
                route_action_mask[..., 2],
                remote_legal,
            ],
            dim=-1,
        )
        active = ~idle_legal
        if torch.any(active & ~torch.any(top_level_mask, dim=-1)):
            raise MAPPONetworkError(
                "active hierarchical route mask has no legal top-level action"
            )

        invalid_logit = torch.finfo(recurrent_features.dtype).min
        top_fallback = torch.zeros_like(top_level_mask)
        top_fallback[..., 0] = True
        safe_top_mask = torch.where(
            active.unsqueeze(-1), top_level_mask, top_fallback
        )
        top_level_log_probabilities = torch.log_softmax(
            top_level_logits.masked_fill(~safe_top_mask, invalid_logit), dim=-1
        )

        if self.system_destination_scorer is None:
            destination_logits = self.remote_destination_head(remote_fused).squeeze(-1)
        else:
            destination_features = self.system_destination_scorer.features(
                recurrent_features, local_endpoint_features, candidate_endpoint_features,
                self_features, candidate_public_features, candidate_edge_features,
                self.remote_candidate_ids)
            destination_logits = self.system_destination_scorer(destination_features)
            if destination_sink is not None:
                destination_sink.append(destination_features)
        destination_fallback = torch.zeros_like(remote_mask)
        destination_fallback[..., 0] = True
        safe_remote_mask = torch.where(
            remote_legal.unsqueeze(-1), remote_mask, destination_fallback
        )
        destination_log_probabilities = torch.log_softmax(
            destination_logits.masked_fill(~safe_remote_mask, invalid_logit),
            dim=-1,
        )

        invalid = torch.full_like(idle_legal, invalid_logit, dtype=recurrent_features.dtype)
        idle_logits = torch.where(idle_legal, torch.zeros_like(invalid), invalid)
        local_logits = torch.where(
            active & top_level_mask[..., 0],
            top_level_log_probabilities[..., 0],
            invalid,
        )
        defer_logits = torch.where(
            active & top_level_mask[..., 1],
            top_level_log_probabilities[..., 1],
            invalid,
        )
        remote_logits = (
            top_level_log_probabilities[..., 2:].expand_as(
                destination_log_probabilities
            )
            + destination_log_probabilities
        )
        remote_logits = torch.where(
            active.unsqueeze(-1) & remote_mask,
            remote_logits,
            torch.full_like(remote_logits, invalid_logit),
        )
        logits = torch.cat(
            [
                idle_logits.unsqueeze(-1),
                local_logits.unsqueeze(-1),
                defer_logits.unsqueeze(-1),
                remote_logits,
            ],
            dim=-1,
        )
        expected_logits = (batch, time, agents, self.uav_count + 2)
        if tuple(logits.shape) != expected_logits or not torch.isfinite(logits).all():
            raise MAPPONetworkError(
                "hierarchical option-aware route logits are invalid"
            )
        if route_candidate_sink is not None:
            # Pure, detached wiring of the existing encoder output. Defer uses
            # the same current-task/source and pooled legal-Remote context;
            # a fixed type indicator distinguishes the special options.
            local = fused[..., :1, :]
            leaves = torch.cat((local, local, remote_fused), dim=-2)
            context = remote_summary.unsqueeze(-2).expand_as(leaves)
            types = leaves.new_zeros((*leaves.shape[:-1], 3))
            types[..., 0, 0] = 1
            types[..., 1, 1] = 1
            types[..., 2:, 2] = 1
            features = torch.cat((leaves, context, types), dim=-1).detach()
            idle = features.new_zeros((*features.shape[:-2], 1, features.shape[-1]))
            route_candidate_sink.append(torch.cat((idle, features), dim=-2))
        return logits


class HierarchicalCandidateAwareTxDecoderV1(nn.Module):
    """Compose Idle/Transmit and conditional destination scores from actor-visible data."""

    _QUEUE_START = 9 + 2 * 8
    _QUEUE_WIDTH = 8
    _CANDIDATE_HIDDEN = 64

    def __init__(self, *, spec: MAPPOTensorSpec) -> None:
        super().__init__()
        if spec.uav_count < 2:
            raise MAPPONetworkError("hierarchical TX decoder requires at least two UAVs")
        self.spec = spec
        candidate_input = (
            spec.gru_hidden_dimension
            + self._QUEUE_WIDTH
            + spec.neighbor_public_feature_dim
            + spec.edge_feature_dim
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_input, self._CANDIDATE_HIDDEN),
            nn.LayerNorm(self._CANDIDATE_HIDDEN),
            nn.LeakyReLU(),
        )
        self.top_level_head = nn.Sequential(
            nn.Linear(spec.gru_hidden_dimension + self._CANDIDATE_HIDDEN, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
        )
        self.destination_head = nn.Linear(self._CANDIDATE_HIDDEN, 1)
        self.register_buffer(
            "candidate_ids",
            torch.tensor(
                [
                    [destination for destination in range(spec.uav_count) if destination != source]
                    for source in range(spec.uav_count)
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )

    def _destinations(self, features: Tensor) -> Tensor:
        batch, time, agents, _, width = features.shape
        indices = self.candidate_ids.view(1, 1, agents, agents - 1, 1)
        return torch.gather(
            features,
            -2,
            indices.expand(batch, time, agents, agents - 1, width),
        )

    def forward(self, recurrent: Tensor, batch: ActorTensorBatch, upper_decoder=None, return_components=False) -> Tensor:
        batch_size, time, agents, hidden = recurrent.shape
        if agents != self.spec.uav_count or hidden != self.spec.gru_hidden_dimension:
            raise MAPPONetworkError("hierarchical TX recurrent features differ from spec")
        if tuple(batch.self_features.shape[:3]) != (batch_size, time, agents):
            raise MAPPONetworkError("hierarchical TX actor features differ from spec")
        tx_mask = batch.action_masks["tx_select"]
        if tuple(tx_mask.shape) != (batch_size, time, agents, agents):
            raise MAPPONetworkError("hierarchical TX mask differs from action domain")
        if not torch.all(tx_mask[..., 0]):
            raise MAPPONetworkError("hierarchical TX requires legal Idle")
        queue_end = self._QUEUE_START + agents * self._QUEUE_WIDTH
        queues = batch.self_features[..., self._QUEUE_START:queue_end].reshape(
            batch_size, time, agents, agents, self._QUEUE_WIDTH
        )
        destination_features = (
            self._destinations(queues),
            self._destinations(batch.neighbor_public_features),
            self._destinations(batch.edge_features),
        )
        recurrent_candidates = recurrent.unsqueeze(-2).expand(
            batch_size, time, agents, agents - 1, hidden
        )
        candidate_state = self.candidate_encoder(
            torch.cat((recurrent_candidates, *destination_features), dim=-1)
        )
        remote_mask = tx_mask[..., 1:]
        remote_available = torch.any(remote_mask, dim=-1)
        weights = remote_mask.to(candidate_state.dtype).unsqueeze(-1)
        summary = (candidate_state * weights).sum(dim=-2) / weights.sum(
            dim=-2
        ).clamp_min(1.0)
        top_logits = (self.top_level_head(torch.cat((recurrent, summary), dim=-1))
                      if upper_decoder is None else upper_decoder(batch, remote_mask))
        invalid = torch.finfo(recurrent.dtype).min
        top_mask = torch.stack((tx_mask[..., 0], remote_available), dim=-1)
        top_log_probs = torch.log_softmax(
            top_logits.masked_fill(~top_mask, invalid), dim=-1
        )
        destination_logits = self.destination_head(candidate_state).squeeze(-1)
        fallback = torch.zeros_like(remote_mask)
        fallback[..., 0] = True
        safe_mask = torch.where(remote_available.unsqueeze(-1), remote_mask, fallback)
        destination_log_probs = torch.log_softmax(
            destination_logits.masked_fill(~safe_mask, invalid), dim=-1
        )
        remote_log_probs = top_log_probs[..., 1:] + destination_log_probs
        remote_log_probs = remote_log_probs.masked_fill(~remote_mask, invalid)
        logits = torch.cat((top_log_probs[..., :1], remote_log_probs), dim=-1)
        if tuple(logits.shape) != tuple(tx_mask.shape) or not torch.isfinite(logits).all():
            raise MAPPONetworkError("hierarchical TX logits are invalid")
        return (logits, top_log_probs, destination_log_probs) if return_components else logits


@dataclass(frozen=True)
class ActorNetworkOutput:
    """Raw seven-head logits, validated masks, and recurrent output."""

    raw_logits: Mapping[str, Tensor]
    action_masks: Mapping[str, Tensor]
    recurrent_features: Tensor
    hidden_out: Tensor
    route_recurrent_features: Tensor | None = None
    route_candidate_features: Tensor | None = None
    destination_features: Tensor | None = None
    route_candidates: RouteCandidateBatch | None = None

    @property
    def shared_hidden_out(self) -> Tensor:
        return self.hidden_out[..., :self.recurrent_features.shape[-1]]

    @property
    def route_hidden_out(self) -> Tensor | None:
        if self.route_recurrent_features is None:
            return None
        return self.hidden_out[..., self.recurrent_features.shape[-1]:]

    @property
    def logits(self) -> Mapping[str, Tensor]:
        return self.raw_logits

    def masked_logits(self) -> dict[str, Tensor]:
        """Apply masks without sampling or constructing categorical objects."""

        return {
            branch: logits.masked_fill(~self.action_masks[branch], -torch.inf)
            for branch, logits in self.raw_logits.items()
        }


class CandidateAwareCpuFrequencyDecoderV1(nn.Module):
    """Shared frequency scorer using only the selected actor-visible queue.

    Feature order: frequency level, cycles/slot divided by maximum task cycles,
    head cycles, head slack, selected queue cycles, residual energy ratio.
    The final four quantities use the existing tensorizer normalizations.
    Masks and categorical statistics remain owned by the action distribution.
    """

    FEATURE_NAMES = (
        "frequency_level", "normalized_slot_capacity", "normalized_head_cycles",
        "normalized_head_slack", "normalized_queue_cycles", "residual_energy_ratio",
    )

    def __init__(self, config: RunConfig, spec: MAPPOTensorSpec) -> None:
        super().__init__()
        self.extractor = RouteOptionFeatureExtractor(spec)
        env = config.environment
        self.capacity_scale = (
            env.reference_cpu_frequency_hz * env.slot_duration_s
            / (env.task_data_bits_max * env.task_cycles_per_bit_max)
        )
        self.register_buffer(
            "frequency_levels", torch.tensor(config.action.cpu_frequency_levels),
            persistent=False,
        )
        # Same shared scalar-scoring pattern as the candidate-aware TX head;
        # no candidate-index embedding and no extra recurrent/graph network.
        self.scorer = nn.Sequential(
            nn.Linear(spec.gru_hidden_dimension + len(self.FEATURE_NAMES), 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def candidate_features(self, batch: ActorTensorBatch, queue_indices: Tensor) -> Tensor:
        features = batch.self_features
        extractor = self.extractor
        if features.shape[-1] != extractor.self_feature_dim:
            raise MAPPONetworkError("CPU frequency features differ from actor layout")
        queues = extractor._indexed_queues(features, extractor.cpu_start)
        local = extractor._queue(features, extractor.local_start)
        # CPU source == ego uses local_cpu, exactly as ObservationBuilder does.
        queues = torch.where(
            torch.eye(extractor.uav_count, device=features.device, dtype=torch.bool)
            .view(1, 1, extractor.uav_count, extractor.uav_count, 1),
            local.unsqueeze(-2), queues,
        )
        selected = torch.gather(
            queues, -2,
            (queue_indices - 1).clamp_min(0).unsqueeze(-1).unsqueeze(-1)
            .expand(*queue_indices.shape, 1, _QUEUE_SUMMARY_FEATURE_DIM),
        ).squeeze(-2)
        selected = torch.where(
            ((queue_indices > 0) & (selected[..., 7] > 0.5)).unsqueeze(-1),
            selected, torch.zeros_like(selected),
        )
        levels = self.frequency_levels.to(dtype=features.dtype)
        levels = levels.view(1, 1, 1, -1).expand(*features.shape[:3], -1)
        capacity = levels * features[..., 5].unsqueeze(-1) * self.capacity_scale
        common = torch.stack(
            (selected[..., 5], selected[..., 6], selected[..., 2], features[..., 2]),
            dim=-1,
        ).unsqueeze(-2).expand(*levels.shape, 4)
        return torch.cat((levels.unsqueeze(-1), capacity.unsqueeze(-1), common), dim=-1)

    def score_candidates(self, recurrent: Tensor, candidates: Tensor) -> Tensor:
        context = recurrent.unsqueeze(-2).expand(*candidates.shape[:-1], recurrent.shape[-1])
        return self.scorer(torch.cat((context, candidates), dim=-1)).squeeze(-1)

    def forward(self, recurrent: Tensor, batch: ActorTensorBatch, queue_indices: Tensor) -> Tensor:
        return self.score_candidates(recurrent, self.candidate_features(batch, queue_indices))


class CAGATMAPPOActor(nn.Module):
    """CA-GATv2/GRU actor with an optional independently owned Route path.

    The persistent state is [B,A,H] in legacy mode and [B,A,2H] in E7,
    packing shared_hidden followed by route_hidden in disjoint lanes. This
    preserves existing collector/checkpoint/replay transport without ever
    feeding the shared lane to route_gru.
    """

    def __init__(self, config: RunConfig) -> None:
        super().__init__()
        self.spec = MAPPOTensorSpec.from_config(config)
        self.route_decoder_mode = RouteDecoderMode(
            config.training.mappo.route_decoder_mode
        )
        self.route_feature_mode = config.training.mappo.route_feature_mode
        self.tx_decoder_mode = TxDecoderMode(config.training.mappo.tx_decoder_mode)
        self.cpu_frequency_decoder_mode = CpuFrequencyDecoderMode(
            config.training.mappo.cpu_frequency_decoder_mode
        )
        self.tx_upper_decoder_mode = config.training.mappo.tx_upper_decoder_mode
        self.cpu_scheduler_mode = config.training.mappo.cpu_scheduler_mode
        self.route_private_backbone = config.training.mappo.route_private_backbone
        hidden = self.spec.encoder_hidden_dimension
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(
                _derived_torch_seed(config.seed)
            )
            self.self_encoder = nn.Sequential(
                nn.Linear(self.spec.self_feature_dim, hidden),
                nn.LeakyReLU(),
            )
            self.public_encoder = nn.Sequential(
                nn.Linear(self.spec.neighbor_public_feature_dim, hidden),
                nn.LeakyReLU(),
            )
            self.graph_encoder = CAGATv2Layer(
                hidden,
                self.spec.edge_feature_dim,
                self.spec.attention_head_count,
            )
            self.gru = nn.GRU(
                hidden,
                self.spec.gru_hidden_dimension,
                num_layers=1,
                batch_first=True,
            )
            dimensions = self.spec.action_dimensions
            self.branch_embeddings = nn.ModuleDict(
                {
                    branch: nn.Embedding(dimensions[branch], hidden)
                    for branch in (
                        "tx_select",
                        "resource_group",
                        "resource_width",
                        "cpu_queue",
                    )
                }
            )
            # Always consume the historical route-head RNG draw before the
            # shared non-route modules.  This preserves same-seed common-module
            # initialization across both decoder modes.
            legacy_route_head = nn.Linear(hidden, dimensions["route"])
            non_route_heads = {
                "tx_select": nn.Linear(hidden, dimensions["tx_select"]),
                "resource_group": nn.Linear(2 * hidden, dimensions["resource_group"]),
                "resource_width": nn.Linear(3 * hidden, dimensions["resource_width"]),
                "power_level": nn.Linear(4 * hidden, dimensions["power_level"]),
                "cpu_queue": nn.Linear(hidden, dimensions["cpu_queue"]),
                "cpu_frequency": nn.Linear(2 * hidden, dimensions["cpu_frequency"]),
            }
            route_head: nn.Module
            if self.route_decoder_mode is RouteDecoderMode.LEGACY:
                route_head = legacy_route_head
            elif self.route_decoder_mode is RouteDecoderMode.CANDIDATE_AWARE_V1:
                route_head = CandidateAwareRouteDecoderV1(
                    uav_count=self.spec.uav_count,
                    recurrent_dimension=self.spec.gru_hidden_dimension,
                    candidate_dimension=self.spec.encoder_hidden_dimension,
                    edge_dimension=self.spec.edge_feature_dim,
                )
            elif self.route_decoder_mode is RouteDecoderMode.OPTION_AWARE_V1:
                route_head = OptionAwareRouteDecoderV1(spec=self.spec)
            elif (
                self.route_decoder_mode
                in {RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1, RouteDecoderMode.UNIFIED_NODE_V1}
            ):
                route_head = HierarchicalOptionAwareRouteDecoderV1(spec=self.spec)
            else:
                raise MAPPONetworkError(
                    f"unsupported route decoder mode {self.route_decoder_mode.value!r}"
                )
            # Draw the treatment-only TX parameters after every E1 module so
            # same-seed shared and route weights remain identical across arms.
            if self.tx_decoder_mode is TxDecoderMode.HIERARCHICAL_CANDIDATE_AWARE_V1:
                non_route_heads["tx_select"] = HierarchicalCandidateAwareTxDecoderV1(
                    spec=self.spec
                )
            # Consume the historical CPU linear draw above, then construct the
            # treatment after all E3 modules to preserve every common weight.
            if self.cpu_frequency_decoder_mode is CpuFrequencyDecoderMode.CANDIDATE_AWARE_V1:
                non_route_heads["cpu_frequency"] = CandidateAwareCpuFrequencyDecoderV1(
                    config, self.spec
                )
            self.action_heads = nn.ModuleDict(
                {"route": route_head, **non_route_heads}
            )

        # Deep-copy only after historical initialization has finished: no new
        # RNG draws and no parameter aliases with the non-Route path.
        if self.route_private_backbone:
            for name in ("self_encoder", "public_encoder", "graph_encoder", "gru"):
                setattr(self, "route_" + name, copy.deepcopy(getattr(self, name)))

        # Treatment-only draws happen after all historical E7 initialization.
        from .integrated_scheduler import TXUpperPrivateBinaryDecoderV1, DeadlineAwareJointCPUSchedulerV1
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(_derived_torch_seed(config.seed) + 8001)
            if self.tx_upper_decoder_mode != "legacy":
                self.tx_upper_private = TXUpperPrivateBinaryDecoderV1(self.spec)
                del self.action_heads["tx_select"].top_level_head
            if self.cpu_scheduler_mode != "legacy_hierarchical":
                self.cpu_joint_scheduler = DeadlineAwareJointCPUSchedulerV1(config, self.spec)
                del self.action_heads["cpu_queue"]
                del self.action_heads["cpu_frequency"]
                del self.branch_embeddings["cpu_queue"]

        # Draw E15-only parameters after every historical module; legacy paths
        # retain bitwise initialization and checkpoint keys. No parameter alias.
        if config.training.mappo.destination_decoder_mode == "system_aware_v1":
            from .destination_ranking import SystemAwareDestinationScorer
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(_derived_torch_seed(config.seed) + 15001)
                self.route_decoder.system_destination_scorer = SystemAwareDestinationScorer(config, self.spec)
            # The unused historical head is retained as checkpoint-compatible
            # storage, but is explicitly frozen and excluded from Adam.
            self.route_decoder.remote_destination_head.requires_grad_(False)
        if self.route_decoder_mode is RouteDecoderMode.UNIFIED_NODE_V1:
            # Preserve all historical draws before replacing only the Route head.
            from .unified_route import build_unified_route_decoder
            self.action_heads["route"] = build_unified_route_decoder(
                self.spec, config.seed, config=config)

    @property
    def route_parameter_prefixes(self) -> tuple[str, ...]:
        prefixes = ("action_heads.route.",)
        if self.route_private_backbone:
            prefixes += ("route_self_encoder.", "route_public_encoder.",
                         "route_graph_encoder.", "route_gru.")
        return prefixes

    @property
    def route_decoder(self) -> nn.Module:
        """Return the selected route module without duplicate registration."""

        return self.action_heads["route"]

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise MAPPONetworkError("batch_size must be a positive integer")
        parameter = next(self.parameters())
        return torch.zeros(
            (batch_size, self.spec.uav_count, self.spec.recurrent_state_dimension),
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )

    def route_logits(
        self,
        recurrent_features: Tensor,
        candidate_public_features: Tensor | None = None,
        candidate_edge_features: Tensor | None = None,
        *,
        cooperative_features: Tensor | None = None,
        local_endpoint_features: Tensor | None = None,
        candidate_endpoint_features: Tensor | None = None,
        self_features: Tensor | None = None,
        route_action_mask: Tensor | None = None,
        route_candidate_sink: list[Tensor] | None = None,
        destination_sink: list[Tensor] | None = None,
    ) -> Tensor:
        """Return raw route logits for the configured decoder architecture."""

        if recurrent_features.ndim != 4:
            raise MAPPONetworkError(
                "recurrent_features must be [batch,time,agent,hidden]"
            )
        expected = (
            *recurrent_features.shape[:3],
            self.spec.action_dimensions["route"],
        )
        if self.route_decoder_mode is RouteDecoderMode.LEGACY:
            route_head = self.route_decoder
            if not isinstance(route_head, nn.Linear):
                raise MAPPONetworkError("legacy route decoder must be Linear")
            logits = route_head(recurrent_features)
        elif self.route_decoder_mode is RouteDecoderMode.CANDIDATE_AWARE_V1:
            route_decoder = self.route_decoder
            if not isinstance(route_decoder, CandidateAwareRouteDecoderV1):
                raise MAPPONetworkError(
                    "candidate-aware route decoder has an invalid module type"
                )
            if candidate_public_features is None or candidate_edge_features is None:
                raise MAPPONetworkError(
                    "candidate-aware route logits require candidate features"
                )
            logits = route_decoder(
                recurrent_features,
                candidate_public_features,
                candidate_edge_features,
            )
        elif self.route_decoder_mode in {
            RouteDecoderMode.OPTION_AWARE_V1, RouteDecoderMode.UNIFIED_NODE_V1,
        }:
            route_decoder = self.route_decoder
            if not isinstance(route_decoder, OptionAwareRouteDecoderV1):
                raise MAPPONetworkError(
                    "option-aware route decoder has an invalid module type"
                )
            required = (
                cooperative_features,
                local_endpoint_features,
                candidate_endpoint_features,
                self_features,
                candidate_public_features,
                candidate_edge_features,
            )
            if any(tensor is None for tensor in required):
                raise MAPPONetworkError(
                    "option-aware route logits require recurrent, graph, and semantic features"
                )
            unified_kwargs = ({"route_action_mask": route_action_mask,
                               "route_candidate_sink": route_candidate_sink}
                              if self.route_decoder_mode is RouteDecoderMode.UNIFIED_NODE_V1 else {})
            logits = route_decoder(
                recurrent_features,
                cooperative_features,
                local_endpoint_features,
                candidate_endpoint_features,
                self_features,
                candidate_public_features,
                candidate_edge_features,
                **unified_kwargs,
            )
        elif (
            self.route_decoder_mode
            is RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1
        ):
            route_decoder = self.route_decoder
            if not isinstance(route_decoder, HierarchicalOptionAwareRouteDecoderV1):
                raise MAPPONetworkError(
                    "hierarchical option-aware route decoder has an invalid module type"
                )
            required = (
                cooperative_features,
                local_endpoint_features,
                candidate_endpoint_features,
                self_features,
                candidate_public_features,
                candidate_edge_features,
                route_action_mask,
            )
            if any(tensor is None for tensor in required):
                raise MAPPONetworkError(
                    "hierarchical option-aware route logits require recurrent, graph, "
                    "semantic, and route-mask features"
                )
            logits = route_decoder(
                recurrent_features,
                cooperative_features,
                local_endpoint_features,
                candidate_endpoint_features,
                self_features,
                candidate_public_features,
                candidate_edge_features,
                route_action_mask,
                route_candidate_sink=route_candidate_sink,
                destination_sink=destination_sink,
            )
        else:
            raise MAPPONetworkError("unsupported route decoder mode")
        if tuple(logits.shape) != expected or not torch.isfinite(logits).all():
            raise MAPPONetworkError("route logits are invalid")
        return logits

    def branch_logits(
        self,
        branch: str,
        recurrent_features: Tensor,
        action_indices: Mapping[str, Tensor],
        actor_batch: ActorTensorBatch | None = None,
    ) -> Tensor:
        """Return one head's logits under the frozen predecessor dependencies."""

        if branch not in ACTION_BRANCH_DEPENDENCIES:
            raise MAPPONetworkError(f"unknown action branch {branch!r}")
        if branch == "route" and self.route_decoder_mode is not RouteDecoderMode.LEGACY:
            raise MAPPONetworkError(
                "structured route logits require explicit candidate features"
            )
        if recurrent_features.ndim != 4:
            raise MAPPONetworkError(
                "recurrent_features must be [batch,time,agent,hidden]"
            )
        batch, time, agents, hidden = recurrent_features.shape
        if agents != self.spec.uav_count or hidden != self.spec.gru_hidden_dimension:
            raise MAPPONetworkError("recurrent_features differ from the actor spec")
        if (
            not recurrent_features.is_floating_point()
            or not torch.isfinite(recurrent_features).all()
        ):
            raise MAPPONetworkError("recurrent_features must be finite floating tensors")

        if self.cpu_scheduler_mode != "legacy_hierarchical" and branch in {"cpu_queue", "cpu_frequency"}:
            # Executor transport slots only; the joint scheduler owns CPU action forward.
            return recurrent_features.new_zeros((batch, time, agents, self.spec.action_dimensions[branch]))

        context_parts = [recurrent_features]
        for dependency in ACTION_BRANCH_DEPENDENCIES[branch]:
            if dependency not in action_indices:
                raise MAPPONetworkError(
                    f"{branch} requires the selected {dependency} indices"
                )
            indices = action_indices[dependency]
            if tuple(indices.shape) != (batch, time, agents) or indices.dtype != torch.long:
                raise MAPPONetworkError(
                    f"{dependency} indices must be long [batch,time,agent]"
                )
            if indices.device != recurrent_features.device:
                raise MAPPONetworkError(
                    f"{dependency} indices and recurrent features must share a device"
                )
            dimension = self.spec.action_dimensions[dependency]
            if torch.any(indices < 0) or torch.any(indices >= dimension):
                raise MAPPONetworkError(f"{dependency} index lies outside its domain")
            context_parts.append(self.branch_embeddings[dependency](indices))

        context = (
            context_parts[0]
            if len(context_parts) == 1
            else torch.cat(context_parts, dim=-1)
        )
        if branch == "tx_select" and self.tx_decoder_mode is not TxDecoderMode.LEGACY:
            if actor_batch is None:
                raise MAPPONetworkError("hierarchical TX logits require actor features")
            tx_head = self.action_heads[branch]
            if not isinstance(tx_head, HierarchicalCandidateAwareTxDecoderV1):
                raise MAPPONetworkError("hierarchical TX decoder has an invalid module type")
            logits = tx_head(context, actor_batch,
                             upper_decoder=getattr(self, "tx_upper_private", None))
        elif branch == "cpu_frequency" and self.cpu_frequency_decoder_mode is CpuFrequencyDecoderMode.CANDIDATE_AWARE_V1:
            if actor_batch is None:
                raise MAPPONetworkError("candidate CPU frequency logits require actor features")
            logits = self.action_heads[branch](
                recurrent_features, actor_batch, action_indices["cpu_queue"]
            )
        else:
            logits = self.action_heads[branch](context)
        expected = (batch, time, agents, self.spec.action_dimensions[branch])
        if tuple(logits.shape) != expected:
            raise MAPPONetworkError(f"{branch} logits have an invalid shape")
        if not torch.isfinite(logits).all():
            raise MAPPONetworkError(f"{branch} logits contain NaN or Inf")
        return logits

    def _backbone_features(
        self, batch: ActorTensorBatch, hidden: Tensor, *, private: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None, Tensor]:
        """Encode one owned path; all learned Route contexts use private modules."""
        batch_size, time, agents = batch.self_features.shape[:3]
        prefix = "route_" if private else ""
        self_encoder = getattr(self, prefix + "self_encoder")
        public_encoder = getattr(self, prefix + "public_encoder")
        graph_encoder = getattr(self, prefix + "graph_encoder")
        gru = getattr(self, prefix + "gru")
        item_count = batch_size * time * agents
        self_state = self_encoder(
            batch.self_features.reshape(item_count, self.spec.self_feature_dim)
        )
        public_state = public_encoder(
            batch.neighbor_public_features.reshape(
                item_count,
                self.spec.uav_count,
                self.spec.neighbor_public_feature_dim,
            )
        )
        graph_arguments = (
            self_state,
            public_state,
            batch.edge_features.reshape(
                item_count, self.spec.uav_count, self.spec.edge_feature_dim
            ),
            batch.neighbor_mask.reshape(item_count, self.spec.uav_count),
        )
        local_endpoint_features: Tensor | None = None
        candidate_endpoint_features: Tensor | None = None
        if self.route_decoder_mode in {
            RouteDecoderMode.OPTION_AWARE_V1,
            RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1,
            RouteDecoderMode.UNIFIED_NODE_V1,
        }:
            graph_result = graph_encoder(
                *graph_arguments,
                return_candidate_values=True,
            )
            if not isinstance(graph_result, tuple):
                raise MAPPONetworkError(
                    "option-aware graph encoder did not expose candidate values"
                )
            spatial_flat, local_endpoint_flat, candidate_endpoint_flat = graph_result
            local_endpoint_features = local_endpoint_flat.reshape(
                batch_size,
                time,
                agents,
                self.spec.encoder_hidden_dimension,
            )
            candidate_endpoint_features = candidate_endpoint_flat.reshape(
                batch_size,
                time,
                agents,
                self.spec.uav_count,
                self.spec.encoder_hidden_dimension,
            )
        else:
            graph_result = graph_encoder(*graph_arguments)
            if not isinstance(graph_result, Tensor):
                raise MAPPONetworkError("graph encoder returned an invalid result")
            spatial_flat = graph_result
        spatial = spatial_flat.reshape(
            batch_size,
            time,
            agents,
            self.spec.encoder_hidden_dimension,
        )

        outputs: list[Tensor] = []
        for step in range(time):
            keep = (~batch.episode_starts[:, step]).to(hidden.dtype).unsqueeze(-1)
            hidden = hidden * keep
            step_output, next_hidden = gru(
                spatial[:, step].reshape(
                    batch_size * agents, 1, self.spec.encoder_hidden_dimension
                ),
                hidden.reshape(1, batch_size * agents, self.spec.gru_hidden_dimension),
            )
            outputs.append(
                step_output.reshape(batch_size, agents, self.spec.gru_hidden_dimension)
            )
            hidden = next_hidden.reshape(
                batch_size, agents, self.spec.gru_hidden_dimension
            )
        recurrent = torch.stack(outputs, dim=1)

        candidate_public_state = public_state.reshape(
            batch_size,
            time,
            agents,
            self.spec.uav_count,
            self.spec.encoder_hidden_dimension,
        )
        route_public_features = (
            batch.neighbor_public_features
            if self.route_decoder_mode
            in {
                RouteDecoderMode.OPTION_AWARE_V1,
                RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1,
                RouteDecoderMode.UNIFIED_NODE_V1,
            }
            else candidate_public_state
        )
        return (recurrent, hidden, spatial, local_endpoint_features,
                candidate_endpoint_features, route_public_features)

    def forward(
        self,
        batch: ActorTensorBatch,
        hidden_in: Tensor | None = None,
    ) -> ActorNetworkOutput:
        return self._forward_impl(batch, hidden_in)

    def forward_with_route_candidates(
        self,
        batch: ActorTensorBatch,
        hidden_in: Tensor | None = None,
    ) -> ActorNetworkOutput:
        """Training-only readout; preserve the historical execution interface."""
        return self._forward_impl(batch, hidden_in, return_route_candidates=True)

    def forward_with_route_interface(
        self,
        batch: ActorTensorBatch,
        hidden_in: Tensor | None = None,
    ) -> ActorNetworkOutput:
        """Opt-in actor-visible readout; use the actual Route backbone lane."""
        return self._forward_impl(batch, hidden_in, return_route_interface=True)

    def _forward_impl(
        self,
        batch: ActorTensorBatch,
        hidden_in: Tensor | None = None,
        *,
        return_route_candidates: bool = False,
        return_route_interface: bool = False,
    ) -> ActorNetworkOutput:
        batch_size, time, agents = batch.validate(self.spec)
        if hidden_in is None:
            hidden = self.initial_hidden(
                batch_size,
                device=batch.self_features.device,
                dtype=batch.self_features.dtype,
            )
        else:
            expected = (batch_size, agents, self.spec.recurrent_state_dimension)
            if tuple(hidden_in.shape) != expected or not hidden_in.is_floating_point():
                raise MAPPONetworkError("hidden_in must match the configured recurrent state width")
            if hidden_in.device != batch.self_features.device:
                raise MAPPONetworkError("hidden_in and actor inputs must share a device")
            if hidden_in.dtype != batch.self_features.dtype:
                raise MAPPONetworkError("hidden_in and actor inputs must share a dtype")
            if not torch.isfinite(hidden_in).all():
                raise MAPPONetworkError("hidden_in contains NaN or Inf")
            hidden = hidden_in

        shared_hidden = hidden[..., :self.spec.gru_hidden_dimension]
        recurrent, shared_out, spatial, local, candidates, public = self._backbone_features(
            batch, shared_hidden,
        )
        route_recurrent = recurrent
        route_out = None
        if self.route_private_backbone:
            # The private GRU receives only its own persisted lane, never shared_hidden.
            route_hidden = hidden[..., self.spec.gru_hidden_dimension:]
            route_recurrent, route_out, spatial, local, candidates, public = self._backbone_features(
                batch, route_hidden, private=True,
            )
        route_candidate_sink = [] if return_route_candidates else None
        destination_sink = []
        logits = {
            "route": self.route_logits(
                route_recurrent, public, batch.edge_features,
                cooperative_features=spatial,
                local_endpoint_features=local,
                candidate_endpoint_features=candidates,
                self_features=batch.self_features,
                route_action_mask=batch.action_masks["route"],
                route_candidate_sink=route_candidate_sink,
                destination_sink=destination_sink,
            )
        }
        logits.update(
            {
                branch: self.branch_logits(branch, recurrent, batch.action_indices, batch)
                for branch in ACTION_BRANCH_ORDER
                if branch != "route"
            }
        )
        for branch in ACTION_BRANCH_ORDER:
            if logits[branch].shape != batch.action_masks[branch].shape:
                raise MAPPONetworkError(f"{branch} logits and mask shapes differ")
            if not torch.isfinite(logits[branch]).all():
                raise MAPPONetworkError(f"{branch} logits contain NaN or Inf")
        candidate_view = None
        if return_route_interface:
            from .route_candidates import build_route_candidates
            candidate_view = build_route_candidates(
                self.spec, batch.self_features, batch.neighbor_public_features,
                batch.edge_features, batch.action_masks["route"],
                local_endpoint_features=local, candidate_endpoint_features=candidates,
                route_recurrent_features=route_recurrent,
                route_hidden_out=(shared_out if route_out is None else route_out),
                uses_private_backbone=self.route_private_backbone,
            )
        return ActorNetworkOutput(
            raw_logits=logits,
            action_masks=batch.action_masks,
            recurrent_features=recurrent,
            hidden_out=(shared_out if route_out is None else torch.cat((shared_out, route_out), dim=-1)),
            route_recurrent_features=(route_recurrent if self.route_private_backbone else None),
            route_candidate_features=(route_candidate_sink[0] if route_candidate_sink else None),
            destination_features=(destination_sink[0] if destination_sink else None),
            route_candidates=candidate_view,
        )


class MAPPOCentralizedCritic(nn.Module):
    """Centralized value network with a legacy scalar or per-agent head."""

    def __init__(self, config: RunConfig) -> None:
        super().__init__()
        self.spec = MAPPOTensorSpec.from_config(config)
        self.agent_credit_mode = AgentCreditMode(
            config.training.mappo.agent_credit_mode
        )
        self.output_dimension = (
            1
            if self.agent_credit_mode is AgentCreditMode.TEAM
            else self.spec.uav_count
        )
        hidden = self.spec.encoder_hidden_dimension
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_derived_torch_seed(config.seed))
            self.value_network = nn.Sequential(
                nn.Linear(self.spec.centralized_state_dim, hidden),
                nn.LeakyReLU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, self.output_dimension),
            )
        if config.training.mappo.tx_upper_credit_mode == "counterfactual_q_v1":
            from .tx_upper_counterfactual import TXUpperCounterfactualCriticV1
            self.tx_upper_q = TXUpperCounterfactualCriticV1(
                self.spec.centralized_state_dim, hidden, self.spec.uav_count, config.seed,
                normalize_targets=(config.training.mappo.tx_q_target_normalization == "popart_v1"),
            )
        if config.training.mappo.route_credit_mode == "task_event_value_v1":
            from .route_task_credit import RouteTaskValue
            self.route_task_value = RouteTaskValue(
                self.spec.centralized_state_dim, hidden, self.spec.uav_count, config.seed)
        if config.training.mappo.route_value_mode == "candidate_counterfactual_q_v1":
            from .route_option_q import RouteOptionQ
            from .route_q_inputs import centralized_q_fields
            self.route_option_q = RouteOptionQ(
                self.spec.centralized_state_dim,
                HierarchicalOptionAwareRouteDecoderV1.CANDIDATE_VALUE_DIMENSION,
                hidden, config.seed, input_mode=config.training.mappo.route_q_input_mode,
                state_fields=centralized_q_fields(config))

    def forward(self, batch: CentralizedStateTensorBatch | Tensor) -> Tensor:
        features = batch.features if isinstance(batch, CentralizedStateTensorBatch) else batch
        CentralizedStateTensorBatch(features).validate(self.spec)
        values = self.value_network(features)
        if values.shape != (*features.shape[:2], self.output_dimension):
            raise MAPPONetworkError(
                "critic output head count differs from agent_credit_mode"
            )
        if not torch.isfinite(values).all():
            raise MAPPONetworkError("critic value contains NaN or Inf")
        return values


CentralizedCritic = MAPPOCentralizedCritic


__all__ = [
    "ACTION_BRANCH_DEPENDENCIES",
    "ACTION_BRANCH_ORDER",
    "ActorNetworkOutput",
    "ActorObservationTensorizer",
    "ActorTensorBatch",
    "CAGATMAPPOActor",
    "CAGATv2Layer",
    "CandidateAwareRouteDecoderV1",
    "CandidateAwareCpuFrequencyDecoderV1",
    "HierarchicalOptionAwareRouteDecoderV1",
    "HierarchicalCandidateAwareTxDecoderV1",
    "CentralizedCritic",
    "CentralizedStateTensorBatch",
    "CentralizedStateTensorizer",
    "MAPPONetworkError",
    "MAPPOCentralizedCritic",
    "MAPPOTensorSpec",
    "OptionAwareRouteDecoderV1",
    "RouteOptionFeatureExtractor",
    "stack_actor_time",
    "stack_centralized_time",
]
