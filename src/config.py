"""Typed experiment configuration and reproducibility metadata.

This module is the single configuration entry point for the interactive launcher,
direct CLI calls, tests, and future environment backends.  It intentionally uses
only the Python standard library so that the configuration layer remains usable
before the environment dependencies are added.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import re
import subprocess
import sys
from dataclasses import InitVar, asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar, get_args, get_origin, get_type_hints

from .artifacts import atomic_write_text


CONFIG_VERSION = "section4.v1"
DEFAULT_SEED = 42
MAPPO_INITIAL_POLICY_VERSION = 0
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_KIND_PERIODIC_RESUME = "PERIODIC_RESUME"
CHECKPOINT_KIND_FINAL_COMPLETED = "FINAL_COMPLETED"
CHECKPOINT_V1_RESUME_BOUNDARY = "EPISODE_BOUNDARY"
CHECKPOINT_V1_EXACTNESS = "SAME_RUNTIME_STATE_EXACT"
CHECKPOINT_V1_KINDS = (
    CHECKPOINT_KIND_PERIODIC_RESUME,
    CHECKPOINT_KIND_FINAL_COMPLETED,
)
CHECKPOINT_V1_TOP_LEVEL_FIELDS = (
    "schema_version",
    "checkpoint_kind",
    "method_id",
    "git_commit",
    "config_hash",
    "config_snapshot",
    "runtime_provenance",
    "training_state",
    "model_state",
    "optimizer_state",
    "policy_rng_state",
    "active_rollout_state",
    "diagnostics_state",
)
CHECKPOINT_V1_RUNTIME_PROVENANCE_FIELDS = (
    "python_version",
    "torch_version",
    "cuda_runtime_version",
    "device_type",
    "device_name",
    "dtype",
)
CHECKPOINT_V1_TRAINING_STATE_FIELDS = (
    "policy_version",
    "started_episodes",
    "completed_episodes",
    "next_episode_index",
    "collected_environment_transitions",
    "optimized_transitions",
    "ppo_update_count",
    "unused_final_tail_transitions",
    "training_complete",
    "active_rollout_length",
    "active_rollout_policy_version",
)
CHECKPOINT_V1_MODEL_STATE_FIELDS = ("actor", "critic")
CHECKPOINT_V1_OPTIMIZER_STATE_FIELDS = ("actor_adam", "critic_adam")
CHECKPOINT_V1_POLICY_RNG_STATE_FIELDS = (
    "generator_state",
    "device_type",
)
CHECKPOINT_V1_ACTIVE_ROLLOUT_FIELDS = (
    "rollout_length",
    "rollout_policy_version",
    "ordered_transitions",
)
CHECKPOINT_V1_TRANSITION_FIELDS = (
    "slot",
    "episode_start",
    "self_features",
    "neighbor_public_features",
    "edge_features",
    "neighbor_mask",
    "action_mask_contracts",
    "branch_masks",
    "proposal_actions",
    "action_indices",
    "active_branch_indicators",
    "old_branch_log_probs",
    "old_joint_log_prob",
    "hidden_in",
    "centralized_state",
    "old_value",
    "reward",
    "terminated",
    "truncated",
    "episode_boundary",
    "bootstrap_allowed",
    "bootstrap_value",
    "executed_action_summary",
    "rejection_or_downgrade_summary",
)
CHECKPOINT_V1_DIAGNOSTICS_STATE_FIELDS = (
    "reward_accumulators",
    "episode_history",
    "ppo_update_history",
)
CHECKPOINT_V1_ALLOWED_PAYLOAD_VALUE_KINDS = (
    "python_scalar",
    "string",
    "bool",
    "list",
    "tuple",
    "dict",
    "detached_cpu_tensor",
)
CHECKPOINT_V1_SAFE_BOUNDARY_ORDER = (
    "terminal_environment_step_complete",
    "terminal_settlement_and_reward_complete",
    "terminal_transition_appended",
    "episode_accounting_updated",
    "due_full_rollout_ppo_update_completed",
    "training_stop_checked",
    "periodic_checkpoint_due_checked",
    "periodic_resume_snapshot_saved",
    "next_episode_reset",
)
EVALUATION_SNAPSHOT_V1_REQUIRED_FIELDS = (
    "snapshot_schema_version",
    "snapshot_kind",
    "method_id",
    "config_hash",
    "provenance",
    "actor_architecture_config",
    "action_domain_config",
    "actor_state",
)
EVALUATION_SNAPSHOT_V1_EXCLUDED_FIELDS = (
    "critic_state",
    "optimizer_state",
    "policy_rng_state",
    "active_rollout_state",
    "training_state",
    "diagnostics_state",
)
SUPPORTED_MODES = (
    "environment_sanity",
    "gate0",
    "random",
    "heuristic",
    "baseline",
    "rl",
    "evaluation",
    "ablation",
    "plot",
)
SUPPORTED_METHODS = (
    "environment",
    "random",
    "heuristic",
    "local_only",
    "ca_gat_mappo",
    "factorized_action_gat_qmix",
)
SUPPORTED_SCENARIOS = ("small", "medium", "large", "remote_stress", "remote_stress_v2")
SUPPORTED_LAUNCH_PROFILES = (
    "rl-smoke",
    "rl-long-smoke",
    "rl-formal",
)

STREAM_IDS = {
    "reset_mobility": 10,
    "task_arrival": 20,
    "task_workload": 30,
    "channel_fading": 40,
    "csi_error": 50,
    "interference_measurement": 60,
    "python_utility": 70,
    "torch_initialization": 100,
    "torch_policy_sampling": 110,
    "qmix_epsilon": 120,
}

_T = TypeVar("_T")


class ConfigError(ValueError):
    """Raised when a config file or CLI override violates the schema."""


class WorkloadTimingMode(str, Enum):
    """Select which task state supplies route-slot workload accounting."""

    LEGACY_POST_ROUTE = "legacy_post_route"
    ROUTE_SLOT_PRE_ROUTE = "route_slot_pre_route"


class ChannelAblationMode(str, Enum):
    """Select the physical-channel uncertainty profile."""

    FULL = "full"
    SIMPLIFIED_DETERMINISTIC = "simplified_deterministic"


class ActorRatioMode(str, Enum):
    """Select the PPO importance-ratio credit boundary."""

    JOINT = "joint"
    BRANCH_SPECIFIC = "branch_specific"


class PolicyLossAggregationMode(str, Enum):
    """Select how active branch PPO surrogates form one policy objective."""

    ACTIVE_BRANCH_MEAN_V1 = "active_branch_mean_v1"
    BRANCHWISE_ACTIVE_MEAN_V1 = "branchwise_active_mean_v1"


class AgentCreditMode(str, Enum):
    """Select scalar team credit or conservation-preserving role credit."""

    TEAM = "team"
    ROLE_DECOMPOSED = "role_decomposed"


class CreditAssignmentMode(str, Enum):
    """Select terminal-task credit allocation without changing team reward."""

    LEGACY = "legacy"
    RESPONSIBILITY_TERMINAL = "responsibility_terminal"


@dataclass(frozen=True)
class RewardCreditConfig:
    """Fixed participant credit parameters; legacy remains the default."""

    credit_assignment_mode: CreditAssignmentMode = CreditAssignmentMode.LEGACY
    beta_completion_credit: float = 0.5
    mu_expiration_credit: float = 0.5

    def __post_init__(self) -> None:
        try:
            mode = CreditAssignmentMode(self.credit_assignment_mode)
        except (TypeError, ValueError) as exc:
            raise ConfigError("reward.credit_assignment_mode is invalid") from exc
        object.__setattr__(self, "credit_assignment_mode", mode)
        for name in ("beta_completion_credit", "mu_expiration_credit"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ConfigError(f"reward.{name} must be finite and lie in [0, 1]")
            object.__setattr__(self, name, float(value))


class RouteDecoderMode(str, Enum):
    """Select the actor route-logit decoder architecture."""

    LEGACY = "legacy"
    CANDIDATE_AWARE_V1 = "candidate_aware_v1"
    OPTION_AWARE_V1 = "option_aware_v1"
    HIERARCHICAL_OPTION_AWARE_V1 = "hierarchical_option_aware_v1"
    UNIFIED_NODE_V1 = "unified_node_v1"


class TxDecoderMode(str, Enum):
    """Select the actor transmit-selection decoder architecture."""

    LEGACY = "legacy"
    HIERARCHICAL_CANDIDATE_AWARE_V1 = "hierarchical_candidate_aware_v1"


class CpuFrequencyDecoderMode(str, Enum):
    """Select only the conditional CPU-frequency ranking parameterization."""

    LEGACY = "legacy"
    CANDIDATE_AWARE_V1 = "candidate_aware_v1"


class RouteCreditMode(str, Enum):
    """Select shared base GAE or a route-only longer-horizon estimator."""

    SHARED_GAE = "shared_gae"
    ROUTE_SPECIFIC_GAE = "route_specific_gae"  # Provenance: REJECT_AS_PRIMARY.
    ROLLOUT_CAPPED_ROUTE_EVENT_NSTEP = "rollout_capped_route_event_nstep"
    TASK_EVENT_VALUE_V1 = "task_event_value_v1"


def _installed_distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


@lru_cache(maxsize=1)
def _detected_runtime_versions() -> dict[str, str]:
    """Detect versions without importing the heavyweight Torch runtime."""

    torch_version = _installed_distribution_version("torch")
    cuda_version = "not-available"
    try:
        torch_spec = importlib.util.find_spec("torch")
    except (ImportError, ModuleNotFoundError, ValueError):
        torch_spec = None
    if torch_spec is not None and torch_spec.origin:
        version_file = Path(torch_spec.origin).resolve().parent / "version.py"
        if version_file.is_file():
            try:
                tree = ast.parse(version_file.read_text(encoding="utf-8"))
                assignments: dict[str, Any] = {}
                for node in tree.body:
                    if isinstance(node, ast.Assign) and len(node.targets) == 1:
                        target = node.targets[0]
                        value_node = node.value
                    elif isinstance(node, ast.AnnAssign):
                        target = node.target
                        value_node = node.value
                    else:
                        continue
                    if not isinstance(target, ast.Name):
                        continue
                    if value_node is None:
                        continue
                    try:
                        assignments[target.id] = ast.literal_eval(value_node)
                    except (ValueError, TypeError):
                        continue
                if isinstance(assignments.get("__version__"), str):
                    torch_version = assignments["__version__"]
                if isinstance(assignments.get("cuda"), str):
                    cuda_version = assignments["cuda"]
            except (OSError, SyntaxError, UnicodeError):
                pass

    conda_version = _installed_distribution_version("conda")
    return {
        "python": platform.python_version(),
        "numpy": _installed_distribution_version("numpy"),
        "torch": torch_version,
        "cuda": cuda_version,
        "miniconda": (
            conda_version if conda_version != "not-installed" else "not-installed-or-not-active"
        ),
    }


@dataclass(frozen=True)
class BoundsConfig:
    x_min_m: float = 0.0
    x_max_m: float = 1000.0
    y_min_m: float = 0.0
    y_max_m: float = 1000.0


@dataclass(frozen=True)
class BuildingConfig:
    name: str
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    height_m: float


@dataclass(frozen=True)
class EnvironmentConfig:
    """Frozen Section 4 environment values used by the future backend."""

    channel_ablation_mode: ChannelAblationMode = ChannelAblationMode.FULL
    # Independent diagnostics; defaults preserve historical FULL dynamics.
    channel_stochasticity_disabled: bool = False
    cochannel_interference_disabled: bool = False
    mobility_static: bool = False
    # Draw workloads for reference arrivals before thinning actual arrivals.
    # Empty keeps the historical conditional workload consumption unchanged.
    workload_reference_arrival_probabilities: tuple[float, ...] = ()
    episode_horizon: int = 500
    slot_duration_s: float = 0.020
    height_m: float = 80.0
    bounds: BoundsConfig = field(default_factory=BoundsConfig)
    boundary_rule: str = "coordinate-wise specular reflection"
    reset_safe_distance_m: float = 50.0
    velocity_mean_mps: tuple[float, float] = (0.0, 0.0)
    velocity_std_mps: tuple[float, float] = (1.0, 1.0)
    gauss_markov_alpha: float = 0.85
    candidate_neighbor_radius_m: float = 500.0
    total_bandwidth_hz: float = 20e6
    ru_count: int = 20
    resource_group_count: int = 5
    carrier_frequency_hz: float = 3.5e9
    reference_transmit_power_w: float = 1.0
    reference_cpu_frequency_hz: float = 2.0e9
    reference_cpu_coefficient: float = 1.0e-28
    reference_initial_energy_j: float = 30.0
    reference_rate_bps: float = 20e6
    antenna_gain_dbi: float = 2.0
    noise_psd_dbm_hz: float = -174.0
    receiver_noise_figure_db: float = 7.0
    building_loss_db: float = 15.0
    los_rician_k_db: float = 6.0
    shadowing_std_db: float = 4.0
    shadowing_correlation_distance_m: float = 50.0
    fixed_csi_aoi_slots: int = 3
    csi_error_std_db: float = 2.0
    interference_ema_beta: float = 0.8
    initial_interference_w: float = 0.0
    outage_threshold_db: float = -3.0
    energy_tolerance_j: float = 1.0e-12
    minimum_task_slack_slots: int = 10
    maximum_task_slack_slots: int = 150
    task_data_bits_min: float = 0.25e6
    task_data_bits_max: float = 1.5e6
    task_cycles_per_bit_min: float = 300.0
    task_cycles_per_bit_max: float = 1000.0
    task_deadline_factor_min: float = 1.5
    task_deadline_factor_max: float = 3.0
    arrival_history_window_slots: int = 500
    workload_timing_mode: WorkloadTimingMode = WorkloadTimingMode.LEGACY_POST_ROUTE
    reward_workload_reference_s: float = 1.0
    reward_task_count_reference: float = 1.0
    reward_completion_weight: float = 1.0
    reward_expiration_weight: float = 1.0
    reward_workload_weight: float = 0.2
    reward_energy_weight: float = 0.05
    uav_count: int = 4
    arrival_probabilities: tuple[float, ...] = (0.04, 0.05, 0.06, 0.05)
    profile_assignment: tuple[str, ...] = (
        "Balanced",
        "Resource-poor",
        "Compute-rich",
        "Energy-limited",
    )
    # One explicit scenario-level realization of the frozen +/-5% hardware
    # perturbation. It is canonical config rather than episode RNG, so resource
    # capabilities and E_act^ref do not drift across train/evaluation seeds.
    profile_perturbations: tuple[tuple[float, float, float, float], ...] = (
        (-0.013814309988, 0.044593840641, 0.044559189965, 0.033901094602),
        (0.024305247334, -0.014446956216, 0.005750819819, -0.029482012276),
        (-0.000601140633, -0.015719950799, 0.049877636039, 0.034480593142),
        (-0.000524427719, 0.006763643213, -0.028161367743, -0.001054193533),
    )
    building_layout: tuple[BuildingConfig, ...] = field(
        default_factory=lambda: (
            BuildingConfig("B1", 150.0, 250.0, 150.0, 350.0, 120.0),
            BuildingConfig("B2", 400.0, 500.0, 150.0, 350.0, 120.0),
            BuildingConfig("B3", 550.0, 650.0, 650.0, 850.0, 120.0),
            BuildingConfig("B4", 750.0, 850.0, 650.0, 850.0, 120.0),
        )
    )
    profile_ratios: Mapping[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "Resource-poor": (0.7, 0.8, 0.8, 1.2),
            "Balanced": (1.0, 1.0, 1.0, 1.0),
            "Compute-rich": (1.5, 1.0, 1.1, 0.9),
            "Energy-limited": (1.0, 0.8, 0.5, 1.1),
            "Communication-rich": (1.0, 1.5, 1.1, 1.0),
        }
    )

    @property
    def ru_bandwidth_hz(self) -> float:
        return self.total_bandwidth_hz / self.ru_count

    @property
    def ru_per_group(self) -> int:
        return self.ru_count // self.resource_group_count

    @property
    def resource_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return the frozen one-based, equal-width contiguous RU groups."""

        width = self.ru_per_group
        return tuple(
            tuple(range(index * width + 1, (index + 1) * width + 1))
            for index in range(self.resource_group_count)
        )

    @property
    def outage_threshold_linear(self) -> float:
        """Return the configured dB outage threshold in the linear domain."""

        return 10.0 ** (self.outage_threshold_db / 10.0)


@dataclass(frozen=True)
class ActionConfig:
    """Seven fixed factorized action branches from Sections 3 and 4."""

    route_fixed_actions: tuple[str, ...] = ("idle", "local", "defer")
    tx_select_idle_action: str = "idle"
    resource_group_count: int = 5
    resource_width_options: tuple[int, ...] = (1, 2)
    power_levels: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    cpu_queue_idle_action: str = "idle"
    cpu_frequency_levels: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    resource_group_numbering: str = "one_based_contiguous"
    sampling_order: tuple[str, ...] = (
        "route",
        "tx_select",
        "resource_group",
        "resource_width",
        "power_level",
        "cpu_queue",
        "cpu_frequency",
    )
    canonical_inactive_values: Mapping[str, Any] = field(
        default_factory=lambda: {
            "route": "idle",
            "tx_select": "idle",
            "resource_group": "idle",
            "resource_width": 1,
            "power_level": 0.0,
            "cpu_queue": "idle",
            "cpu_frequency": 0.0,
        }
    )

    @property
    def resource_groups(self) -> tuple[tuple[int, ...], ...]:
        width = self.resource_group_count
        return tuple(
            tuple(range(index * width + 1, (index + 1) * width + 1))
            for index in range(self.resource_group_count)
        )


@dataclass(frozen=True)
class ReproducibilityConfig:
    seed_policy: str = "single_seed_default_42"
    stream_ids: Mapping[str, int] = field(default_factory=lambda: dict(STREAM_IDS))
    snapshot_enabled: bool = True
    backend_schema_version: str = CONFIG_VERSION
    python_version: str = field(
        default_factory=lambda: _detected_runtime_versions()["python"]
    )
    numpy_version: str = field(
        default_factory=lambda: _detected_runtime_versions()["numpy"]
    )
    torch_version: str = field(
        default_factory=lambda: _detected_runtime_versions()["torch"]
    )
    cuda_version: str = field(default_factory=lambda: _detected_runtime_versions()["cuda"])
    miniconda_version: str = field(
        default_factory=lambda: _detected_runtime_versions()["miniconda"]
    )


@dataclass(frozen=True)
class MAPPOConfig:
    encoder_hidden_dimension: int = 128
    gat_layer_count: int = 1
    attention_head_count: int = 4
    gru_hidden_dimension: int = 128
    actor_learning_rate: float = 3.0e-4
    route_head_lr_scale: float = 1.0
    route_private_backbone: bool = False
    tx_upper_decoder_mode: str = "legacy"
    cpu_scheduler_mode: str = "legacy_hierarchical"
    cpu_anti_starvation_mode: str = "off"
    route_exploration_mode: str = "off"
    route_exploration_initial_coefficient: float = 0.05
    route_exploration_anneal_steps: int = 6000
    route_training_telemetry_enabled: bool = False
    route_task_credit_coefficient: float = 1.0
    route_task_value_coefficient: float = 0.5
    route_value_mode: str = "off"
    route_counterfactual_credit_coefficient: float = 1.0
    route_q_value_loss_coefficient: float = 0.5
    route_q_input_mode: str = "legacy"
    route_actor_credit_source: str = "counterfactual_q"
    destination_decoder_mode: str = "legacy"
    destination_oracle_rank_coefficient: float = 0.0
    destination_supervision_mode: str = "legacy"
    tx_upper_optimization_mode: str = "ppo_standard"
    tx_upper_credit_mode: str = "shared_agent_advantage"
    tx_shadow_max_pairs_per_rollout: int = 2
    tx_shadow_step_budget_per_rollout: int = 2000
    tx_shadow_selector: str = "cyclic_source_first_active_v1"
    tx_shadow_rng_mode: str = "slot_addressed_v1"
    tx_shadow_rng_seed: int = 130043
    tx_shadow_max_conditional_draws: int = 128
    tx_shadow_advantage_normalization: str = "paired_rms_v1"
    tx_q_target_normalization: str = "none"
    critic_learning_rate: float = 3.0e-4
    optimizer: str = "Adam"
    optimizer_topology: str = "separate_actor_critic"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1.0e-8
    weight_decay: float = 0.0
    zero_grad_set_to_none: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip_epsilon: float = 0.20
    actor_ratio_mode: ActorRatioMode = ActorRatioMode.JOINT
    policy_loss_aggregation_mode: PolicyLossAggregationMode = (
        PolicyLossAggregationMode.ACTIVE_BRANCH_MEAN_V1
    )
    agent_credit_mode: AgentCreditMode = AgentCreditMode.TEAM
    route_decoder_mode: RouteDecoderMode = RouteDecoderMode.LEGACY
    route_feature_mode: str = "original_option_features_v1"
    tx_decoder_mode: TxDecoderMode = TxDecoderMode.LEGACY
    cpu_frequency_decoder_mode: CpuFrequencyDecoderMode = CpuFrequencyDecoderMode.LEGACY
    route_credit_mode: RouteCreditMode = RouteCreditMode.SHARED_GAE
    route_gae_lambda: float = 1.0
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.50
    rollout_length_slots: int = 256
    recurrent_chunk_length_slots: int = 32
    sequence_minibatch_size: int = 8
    update_epochs: int = 4
    gradient_clip_norm: float = 0.5
    gradient_clip_scope: str = "separate_actor_critic_global_norm"
    require_full_rollout: bool = True
    chunk_shuffle: bool = False
    timestep_shuffle: bool = False
    allow_incomplete_minibatch: bool = False
    sequence_padding: bool = False
    padded_actor_forward: bool = False
    advantage_normalization: bool = False
    value_clipping: bool = False
    target_kl_enabled: bool = False
    kl_early_stopping: bool = False
    learning_rate_schedule_enabled: bool = False
    entropy_coefficient_schedule_enabled: bool = False
    route_entropy_start_coefficient: float = 0.03
    route_entropy_schedule_start_step: int = 3072
    route_entropy_schedule_end_step: int = 32768
    route_only_entropy_initial_coefficient: float = 0.0
    route_only_entropy_final_coefficient: float = 0.0
    route_only_entropy_anneal_steps: int = 0
    route_choice_stability_enabled: InitVar[bool] = False
    route_choice_entropy_floor_nats: InitVar[float] = 0.20
    route_choice_stability_coef: InitVar[float] = 0.0
    gradient_accumulation: bool = False
    mixed_precision: bool = False
    training_device: str = "cuda"
    carry_hidden_across_update_boundary: bool = True
    policy_sampling_reset_each_episode: bool = False
    discard_final_partial_rollout: bool = True
    max_training_episodes: int = 1000
    max_training_environment_steps: int = 500000
    evaluation_interval_steps: int = 50000
    checkpoint_interval_steps: int = 50000
    checkpoint_schema_version: int = CHECKPOINT_SCHEMA_VERSION
    trajectory_credit_telemetry_enabled: bool = False
    conditional_credit_telemetry_enabled: bool = False
    conditional_credit_telemetry_max_bytes: int = 1073741824
    route_diagnostic_samples_enabled: bool = False
    route_oracle_counterfactual_enabled: InitVar[bool] = False
    route_oracle_selection_rate_ppm: InitVar[int] = 0

    def __post_init__(
        self,
        route_choice_stability_enabled: bool,
        route_choice_entropy_floor_nats: float,
        route_choice_stability_coef: float,
        route_oracle_counterfactual_enabled: bool,
        route_oracle_selection_rate_ppm: int,
    ) -> None:
        object.__setattr__(
            self,
            "route_choice_stability_enabled",
            route_choice_stability_enabled,
        )
        object.__setattr__(
            self,
            "route_choice_entropy_floor_nats",
            route_choice_entropy_floor_nats,
        )
        object.__setattr__(
            self,
            "route_choice_stability_coef",
            route_choice_stability_coef,
        )
        object.__setattr__(
            self,
            "route_oracle_counterfactual_enabled",
            route_oracle_counterfactual_enabled,
        )
        object.__setattr__(
            self,
            "route_oracle_selection_rate_ppm",
            route_oracle_selection_rate_ppm,
        )

    @property
    def recurrent_chunk_count(self) -> int:
        """Return the deterministic number of chunks in one full rollout."""

        return self.rollout_length_slots // self.recurrent_chunk_length_slots

    @property
    def max_environment_transitions(self) -> int:
        """Expose the canonical environment-step budget without duplicating it."""

        return self.max_training_environment_steps

    @property
    def route_only_entropy_enabled(self) -> bool:
        """Return whether the additive route-only regularizer is active."""

        return (
            self.route_only_entropy_initial_coefficient != 0.0
            or self.route_only_entropy_final_coefficient != 0.0
        )

    @property
    def route_entropy_step_schedule_enabled(self) -> bool:
        """Return whether route entropy depends on collected environment steps."""

        return (
            self.entropy_coefficient_schedule_enabled
            or self.route_only_entropy_enabled
        )


@dataclass(frozen=True)
class RouteEntropySchedulePoint:
    """Pure route-only entropy schedule value at one collected-step count."""

    collected_environment_steps: int
    coefficient: float
    progress: float
    enabled: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.collected_environment_steps, bool)
            or not isinstance(self.collected_environment_steps, int)
            or self.collected_environment_steps < 0
        ):
            raise ConfigError(
                "collected_environment_steps must be a non-negative integer"
            )
        if not math.isfinite(self.coefficient) or self.coefficient < 0.0:
            raise ConfigError(
                "route entropy coefficient must be finite and non-negative"
            )
        if not math.isfinite(self.progress) or not 0.0 <= self.progress <= 1.0:
            raise ConfigError("route entropy schedule progress must lie in [0, 1]")
        if not isinstance(self.enabled, bool):
            raise ConfigError("route entropy schedule enabled state must be boolean")


def compute_route_entropy_schedule(
    mappo: MAPPOConfig,
    collected_environment_steps: int,
) -> RouteEntropySchedulePoint:
    """Return the configured route coefficient from config and env steps.

    The treatment intentionally jumps from the base coefficient immediately
    before ``route_entropy_schedule_start_step`` to the start coefficient at
    that exact step.  This warmup-to-intervention discontinuity must not be
    smoothed or interpolated.
    """

    if not isinstance(mappo, MAPPOConfig):
        raise TypeError("mappo must be a MAPPOConfig")
    if (
        isinstance(collected_environment_steps, bool)
        or not isinstance(collected_environment_steps, int)
        or collected_environment_steps < 0
    ):
        raise ConfigError(
            "collected_environment_steps must be a non-negative integer"
        )
    base = float(mappo.entropy_coefficient)
    if not mappo.entropy_coefficient_schedule_enabled:
        return RouteEntropySchedulePoint(
            collected_environment_steps=collected_environment_steps,
            coefficient=base,
            progress=0.0,
            enabled=False,
        )

    start_step = mappo.route_entropy_schedule_start_step
    end_step = mappo.route_entropy_schedule_end_step
    if collected_environment_steps < start_step:
        coefficient = base
        progress = 0.0
    elif collected_environment_steps < end_step:
        progress = (collected_environment_steps - start_step) / (
            end_step - start_step
        )
        coefficient = base + (
            mappo.route_entropy_start_coefficient - base
        ) * (end_step - collected_environment_steps) / (end_step - start_step)
    else:
        coefficient = base
        progress = 1.0
    return RouteEntropySchedulePoint(
        collected_environment_steps=collected_environment_steps,
        coefficient=float(coefficient),
        progress=float(progress),
        enabled=True,
    )


def compute_route_only_entropy_schedule(
    mappo: MAPPOConfig,
    collected_environment_steps: int,
) -> RouteEntropySchedulePoint:
    """Return the additive route-only entropy coefficient at one env step."""

    if not isinstance(mappo, MAPPOConfig):
        raise TypeError("mappo must be a MAPPOConfig")
    if (
        isinstance(collected_environment_steps, bool)
        or not isinstance(collected_environment_steps, int)
        or collected_environment_steps < 0
    ):
        raise ConfigError(
            "collected_environment_steps must be a non-negative integer"
        )
    if not mappo.route_only_entropy_enabled:
        return RouteEntropySchedulePoint(
            collected_environment_steps=collected_environment_steps,
            coefficient=0.0,
            progress=0.0,
            enabled=False,
        )
    anneal_steps = mappo.route_only_entropy_anneal_steps
    progress = (
        1.0
        if anneal_steps == 0
        else min(collected_environment_steps / anneal_steps, 1.0)
    )
    initial = mappo.route_only_entropy_initial_coefficient
    final = mappo.route_only_entropy_final_coefficient
    coefficient = initial + (final - initial) * progress
    return RouteEntropySchedulePoint(
        collected_environment_steps=collected_environment_steps,
        coefficient=float(coefficient),
        progress=float(progress),
        enabled=True,
    )


def compute_effective_route_entropy_schedule(
    mappo: MAPPOConfig,
    collected_environment_steps: int,
) -> RouteEntropySchedulePoint:
    """Combine the legacy absolute schedule with the additive V1 treatment."""

    existing = compute_route_entropy_schedule(mappo, collected_environment_steps)
    route_only = compute_route_only_entropy_schedule(
        mappo, collected_environment_steps
    )
    if existing.enabled and route_only.enabled:
        raise ConfigError(
            "legacy and additive route entropy schedules cannot both be enabled"
        )
    return RouteEntropySchedulePoint(
        collected_environment_steps=collected_environment_steps,
        coefficient=existing.coefficient + route_only.coefficient,
        progress=route_only.progress if route_only.enabled else existing.progress,
        enabled=existing.enabled or route_only.enabled,
    )


@dataclass(frozen=True)
class QMIXConfig:
    encoder_hidden_dimension: int = 128
    gat_layer_count: int = 1
    attention_head_count: int = 4
    gru_hidden_dimension: int = 128
    learning_rate: float = 5.0e-4
    gamma: float = 0.99
    optimizer: str = "Adam"
    replay_capacity_sequences: int = 10000
    recurrent_sequence_length_slots: int = 32
    sequence_batch_size: int = 32
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100000
    target_update_interval_updates: int = 2000
    max_training_episodes: int = 1000
    max_training_environment_steps: int = 500000
    checkpoint_interval_steps: int = 50000


@dataclass(frozen=True)
class TrainingConfig:
    mappo: MAPPOConfig = field(default_factory=MAPPOConfig)
    qmix: QMIXConfig = field(default_factory=QMIXConfig)
    formal_rl_enabled: bool = False


@dataclass(frozen=True)
class MAPPOTrainingAccounting:
    """Pure accounting geometry for the frozen finite training budget."""

    collected_environment_transitions: int
    optimized_transitions: int
    unused_final_tail_transitions: int
    ppo_update_count: int


@dataclass(frozen=True)
class EvaluationConfig:
    train_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    evaluation_seeds: tuple[int, ...] = (1042, 1043, 1044, 1045, 1046)
    inference_rule_mappo: str = "masked_argmax"
    inference_rule_qmix: str = "masked_greedy"
    update_network: bool = False
    metric_statistics: tuple[str, ...] = ("mean", "std", "valid_sample_count")


@dataclass(frozen=True)
class OutputConfig:
    logs_dir: str = "logs"
    plots_dir: str = "plots"
    dashboard_logs_dir: str = "dashboard_logs"
    snapshot_filename: str = "config_snapshot.yaml"
    raw_metrics_filename: str = "raw_metrics.jsonl"
    aggregate_metrics_filename: str = "aggregate_metrics.json"
    route_oracle_counterfactual_filename: InitVar[str] = "route_oracle_counterfactual_v1.jsonl"

    def __post_init__(self, route_oracle_counterfactual_filename: str) -> None:
        object.__setattr__(
            self,
            "route_oracle_counterfactual_filename",
            route_oracle_counterfactual_filename,
        )


@dataclass(frozen=True)
class RunConfig:
    """Canonical, typed configuration shared by every execution path."""

    config_version: str = CONFIG_VERSION
    mode: str = "environment_sanity"
    method_id: str = "environment"
    launch_profile: str | None = None
    scenario_id: str = "small"
    seed: int = DEFAULT_SEED
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    reward: RewardCreditConfig = field(default_factory=RewardCreditConfig)

    # Provenance is attached after schema parsing and is deliberately excluded
    # from the resolved config hash.  It is still present in snapshots.
    source_config_path: str | None = field(default=None, init=False, repr=False, compare=False)
    cli_overrides: Mapping[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)
    interactive_overrides: Mapping[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)
    git_branch: str = field(default="unknown", init=False, repr=False, compare=False)
    git_commit: str = field(default="unknown", init=False, repr=False, compare=False)
    git_dirty: bool = field(default=False, init=False, repr=False, compare=False)

    def resolved_dict(self) -> dict[str, Any]:
        """Return only typed final values used to compute ``config_hash``."""

        resolved = _jsonable(asdict(self), exclude={
            "source_config_path",
            "cli_overrides",
            "interactive_overrides",
            "git_branch",
            "git_commit",
            "git_dirty",
        })
        mappo = resolved.get("training", {}).get("mappo")
        if isinstance(mappo, dict) and self.training.mappo.route_choice_stability_enabled:
            mappo["route_choice_stability_enabled"] = True
            mappo["route_choice_entropy_floor_nats"] = (
                self.training.mappo.route_choice_entropy_floor_nats
            )
            mappo["route_choice_stability_coef"] = (
                self.training.mappo.route_choice_stability_coef
            )
        if isinstance(mappo, dict) and self.training.mappo.route_oracle_counterfactual_enabled:
            mappo["route_oracle_counterfactual_enabled"] = True
            mappo["route_oracle_selection_rate_ppm"] = self.training.mappo.route_oracle_selection_rate_ppm
            output = resolved.get("output")
            if isinstance(output, dict):
                output["route_oracle_counterfactual_filename"] = self.output.route_oracle_counterfactual_filename
        return canonicalize_config_identity(resolved)

    def to_dict(self) -> dict[str, Any]:
        """Alias used by callers that need the canonical resolved mapping."""

        return self.resolved_dict()

    @property
    def config_hash(self) -> str:
        return canonical_config_identity_hash(self.resolved_dict())

    @property
    def run_id(self) -> str:
        return (
            f"{self.method_id}__{self.scenario_id}__seed-{self.seed}__"
            f"cfg-{self.config_hash[:12]}__git-{self.git_commit[:12]}"
        )

    @property
    def derived_stream_ids(self) -> dict[str, int]:
        return dict(self.reproducibility.stream_ids)

    def snapshot_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot with provenance and run metadata."""

        snapshot = self.resolved_dict()
        environment = snapshot.get("environment")
        if isinstance(environment, dict):
            # Record the effective mode even though historical Full omits the
            # selector from the canonical hash payload.
            environment["channel_ablation_mode"] = self.environment.channel_ablation_mode.value
        snapshot["_metadata"] = {
            "source_config_path": self.source_config_path,
            "cli_overrides": _jsonable(dict(self.cli_overrides)),
            "interactive_overrides": _jsonable(dict(self.interactive_overrides)),
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "git_branch": self.git_branch,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "derived_stream_ids": self.derived_stream_ids,
            "runtime_versions": {
                "python": self.reproducibility.python_version,
                "numpy": self.reproducibility.numpy_version,
                "torch": self.reproducibility.torch_version,
                "cuda": self.reproducibility.cuda_version,
                "miniconda": self.reproducibility.miniconda_version,
            },
        }
        return snapshot

    def artifact_paths(self) -> dict[str, str]:
        """Return contract paths without creating files or directories."""

        run_dir = Path(self.output.logs_dir) / self.run_id
        checkpoint_directory = run_dir / "checkpoints"
        return {
            "config_snapshot": str(run_dir / self.output.snapshot_filename),
            "raw_metrics": str(run_dir / self.output.raw_metrics_filename),
            "aggregate_metrics": str(run_dir / self.output.aggregate_metrics_filename),
            "trajectory_credit_events": str(run_dir / "trajectory_credit_events.jsonl"),
            "conditional_credit_trace": str(run_dir / "conditional_credit_trace_v1.jsonl"),
            "route_diagnostic_samples": str(run_dir / "route_diagnostic_samples_v1.jsonl"),
            "route_oracle_counterfactual": str(
                run_dir / self.output.route_oracle_counterfactual_filename
            ),
            "dashboard_csv": str(Path(self.output.dashboard_logs_dir) / f"{self.run_id}_metrics.csv"),
            "figure_input": str(Path(self.output.plots_dir) / f"{self.run_id}_figure_input.csv"),
            "dashboard_png": str(Path(self.output.plots_dir) / f"{self.run_id}_dashboard.png"),
            "checkpoint_directory": str(checkpoint_directory),
            "final_checkpoint": str(checkpoint_directory / "final.pt"),
        }

    def validate(self) -> None:
        if self.config_version != CONFIG_VERSION:
            raise ConfigError(f"config_version must be {CONFIG_VERSION!r}, got {self.config_version!r}")
        if self.mode not in SUPPORTED_MODES:
            raise ConfigError(f"mode must be one of {SUPPORTED_MODES}, got {self.mode!r}")
        if self.method_id not in SUPPORTED_METHODS:
            raise ConfigError(f"method_id must be one of {SUPPORTED_METHODS}, got {self.method_id!r}")
        if (
            self.launch_profile is not None
            and self.launch_profile not in SUPPORTED_LAUNCH_PROFILES
        ):
            raise ConfigError(
                "launch_profile must be None or one of "
                f"{SUPPORTED_LAUNCH_PROFILES}, got {self.launch_profile!r}"
            )
        if self.scenario_id not in SUPPORTED_SCENARIOS:
            raise ConfigError(f"scenario_id must be one of {SUPPORTED_SCENARIOS}, got {self.scenario_id!r}")
        if self.seed < 0:
            raise ConfigError("seed must be a non-negative integer")
        expected_method = _default_method_for_mode(self.mode)
        if self.mode in {"environment_sanity", "gate0"} and self.method_id != "environment":
            raise ConfigError(f"mode {self.mode!r} requires method_id 'environment'")
        if self.mode == "random" and self.method_id != "random":
            raise ConfigError("mode 'random' requires method_id 'random'")
        if self.mode == "heuristic" and self.method_id != "heuristic":
            raise ConfigError("mode 'heuristic' requires method_id 'heuristic'")
        if self.mode == "rl" and self.method_id not in {"ca_gat_mappo", "factorized_action_gat_qmix"}:
            raise ConfigError("mode 'rl' requires a registered RL method")
        if self.mode == "baseline" and self.method_id not in {
            "local_only",
            "factorized_action_gat_qmix",
        }:
            raise ConfigError(
                "mode 'baseline' requires a registered baseline method"
            )
        if expected_method and self.mode in {"evaluation", "ablation", "plot"}:
            # These modes are early-stage menu entries.  Keep a single explicit
            # default while allowing a future method-specific evaluator.
            if self.method_id not in SUPPORTED_METHODS:
                raise ConfigError(f"unsupported method for mode {self.mode!r}")
        env = self.environment
        for name in ("channel_stochasticity_disabled", "cochannel_interference_disabled", "mobility_static"):
            if type(getattr(env, name)) is not bool:
                raise ConfigError(f"environment.{name} must be a boolean")
        reference_arrivals = env.workload_reference_arrival_probabilities
        if reference_arrivals:
            if len(reference_arrivals) != env.uav_count:
                raise ConfigError("workload_reference_arrival_probabilities length must equal uav_count")
            if any(not math.isfinite(p) or p < 0.0 or p > 1.0 for p in reference_arrivals):
                raise ConfigError("workload_reference_arrival_probabilities must be finite and in [0, 1]")
            if any(p > r for p, r in zip(env.arrival_probabilities, reference_arrivals)):
                raise ConfigError("arrival_probabilities must not exceed workload reference probabilities")
        try:
            ChannelAblationMode(env.channel_ablation_mode)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "environment.channel_ablation_mode must be one of "
                f"{tuple(mode.value for mode in ChannelAblationMode)}, "
                f"got {env.channel_ablation_mode!r}"
            ) from exc
        try:
            WorkloadTimingMode(env.workload_timing_mode)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "environment.workload_timing_mode must be one of "
                f"{tuple(mode.value for mode in WorkloadTimingMode)}, "
                f"got {env.workload_timing_mode!r}"
            ) from exc
        if env.episode_horizon <= 0 or env.slot_duration_s <= 0:
            raise ConfigError("episode_horizon and slot_duration_s must be positive")
        if env.uav_count <= 0 or env.height_m <= 0:
            raise ConfigError("uav_count and height_m must be positive")
        bounds = env.bounds
        if bounds.x_max_m <= bounds.x_min_m or bounds.y_max_m <= bounds.y_min_m:
            raise ConfigError("environment bounds must have positive width and height")
        if env.boundary_rule != "coordinate-wise specular reflection":
            raise ConfigError("boundary_rule must be 'coordinate-wise specular reflection'")
        if env.reset_safe_distance_m <= 0 or env.candidate_neighbor_radius_m <= 0:
            raise ConfigError("reset safe distance and candidate neighbor radius must be positive")
        if len(env.velocity_mean_mps) != 2 or len(env.velocity_std_mps) != 2:
            raise ConfigError("velocity mean/std must each contain x and y values")
        if any(value < 0 for value in env.velocity_std_mps):
            raise ConfigError("velocity standard deviations must be non-negative")
        if not 0.0 <= env.gauss_markov_alpha <= 1.0:
            raise ConfigError("gauss_markov_alpha must be in [0, 1]")
        if env.ru_count <= 0 or env.resource_group_count <= 0 or env.ru_count % env.resource_group_count:
            raise ConfigError("ru_count must be positive and divisible by resource_group_count")
        if env.total_bandwidth_hz <= 0 or env.carrier_frequency_hz <= 0:
            raise ConfigError("bandwidth and carrier frequency must be positive")
        if env.reference_transmit_power_w <= 0:
            raise ConfigError("reference_transmit_power_w must be positive")
        if env.shadowing_std_db < 0 or env.shadowing_correlation_distance_m <= 0:
            raise ConfigError("shadowing std must be non-negative and correlation distance positive")
        if env.csi_error_std_db < 0 or env.initial_interference_w < 0:
            raise ConfigError("CSI error std and initial interference must be non-negative")
        if not math.isfinite(env.energy_tolerance_j) or env.energy_tolerance_j <= 0:
            raise ConfigError("energy_tolerance_j must be finite and positive")
        if not math.isfinite(env.outage_threshold_db):
            raise ConfigError("outage_threshold_db must be finite")
        if env.building_loss_db < 0:
            raise ConfigError("building_loss_db must be non-negative")
        for building in env.building_layout:
            if (
                building.x_max_m <= building.x_min_m
                or building.y_max_m <= building.y_min_m
                or building.height_m <= 0
            ):
                raise ConfigError(f"building {building.name!r} has invalid bounds or height")
        if len(env.arrival_probabilities) != env.uav_count:
            raise ConfigError("arrival_probabilities length must equal environment.uav_count")
        if len(env.profile_assignment) != env.uav_count:
            raise ConfigError("profile_assignment length must equal environment.uav_count")
        if any(prob < 0.0 or prob > 1.0 for prob in env.arrival_probabilities):
            raise ConfigError("arrival_probabilities must be in [0, 1]")
        if env.fixed_csi_aoi_slots < 0:
            raise ConfigError("fixed_csi_aoi_slots must be non-negative")
        if not 0.0 <= env.interference_ema_beta < 1.0:
            raise ConfigError("interference_ema_beta must be in [0, 1)")
        if env.minimum_task_slack_slots <= 0 or env.maximum_task_slack_slots < env.minimum_task_slack_slots:
            raise ConfigError("task slack clip bounds are invalid")
        positive_task_values = (
            env.task_data_bits_min,
            env.task_data_bits_max,
            env.task_cycles_per_bit_min,
            env.task_cycles_per_bit_max,
            env.task_deadline_factor_min,
            env.task_deadline_factor_max,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive_task_values):
            raise ConfigError("task workload/deadline distribution bounds must be finite and positive")
        if env.task_data_bits_max < env.task_data_bits_min:
            raise ConfigError("task_data_bits_max must be >= task_data_bits_min")
        if env.task_cycles_per_bit_max < env.task_cycles_per_bit_min:
            raise ConfigError("task_cycles_per_bit_max must be >= task_cycles_per_bit_min")
        if env.task_deadline_factor_max < env.task_deadline_factor_min:
            raise ConfigError("task_deadline_factor_max must be >= task_deadline_factor_min")
        if env.arrival_history_window_slots <= 0:
            raise ConfigError("arrival_history_window_slots must be positive")
        reward_references = (
            env.reward_workload_reference_s,
            env.reward_task_count_reference,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in reward_references):
            raise ConfigError("reward reference constants must be finite and positive")
        reward_weights = (
            env.reward_completion_weight,
            env.reward_expiration_weight,
            env.reward_workload_weight,
            env.reward_energy_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in reward_weights):
            raise ConfigError("reward weights must be finite and non-negative")
        if len(env.profile_perturbations) < env.uav_count:
            raise ConfigError("profile_perturbations must provide at least one row per UAV")
        for perturbation in env.profile_perturbations[: env.uav_count]:
            if len(perturbation) != 4:
                raise ConfigError("each profile perturbation must contain four resource values")
            if any(not math.isfinite(value) or not -0.05 <= value <= 0.05 for value in perturbation):
                raise ConfigError("profile perturbations must be finite and lie in [-0.05, 0.05]")
        unknown_profiles = sorted(set(env.profile_assignment) - set(env.profile_ratios))
        if unknown_profiles:
            raise ConfigError(f"unknown resource profiles: {unknown_profiles}")
        if self.action.resource_group_count != env.resource_group_count:
            raise ConfigError("action.resource_group_count must equal environment.resource_group_count")
        if tuple(self.action.sampling_order) != (
            "route", "tx_select", "resource_group", "resource_width", "power_level", "cpu_queue", "cpu_frequency"
        ):
            raise ConfigError("action.sampling_order must follow the frozen seven-branch order")
        if tuple(self.action.power_levels) != (0.0, 0.25, 0.5, 1.0):
            raise ConfigError("action.power_levels must be (0.0, 0.25, 0.5, 1.0)")
        if tuple(self.action.cpu_frequency_levels) != (0.0, 0.25, 0.5, 1.0):
            raise ConfigError("action.cpu_frequency_levels must be (0.0, 0.25, 0.5, 1.0)")
        mappo = self.training.mappo
        if not isinstance(self.reward, RewardCreditConfig):
            raise ConfigError("reward must be a RewardCreditConfig")
        if (
            self.reward.credit_assignment_mode
            is CreditAssignmentMode.RESPONSIBILITY_TERMINAL
            and mappo.agent_credit_mode != AgentCreditMode.ROLE_DECOMPOSED
        ):
            raise ConfigError(
                "responsibility_terminal requires "
                "training.mappo.agent_credit_mode='role_decomposed'"
            )
        try:
            ActorRatioMode(mappo.actor_ratio_mode)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "training.mappo.actor_ratio_mode must be one of "
                f"{tuple(mode.value for mode in ActorRatioMode)}, "
                f"got {mappo.actor_ratio_mode!r}"
            ) from exc
        try:
            aggregation_mode = PolicyLossAggregationMode(
                mappo.policy_loss_aggregation_mode
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "training.mappo.policy_loss_aggregation_mode must be one of "
                f"{tuple(mode.value for mode in PolicyLossAggregationMode)}"
            ) from exc
        if (
            aggregation_mode is PolicyLossAggregationMode.BRANCHWISE_ACTIVE_MEAN_V1
            and ActorRatioMode(mappo.actor_ratio_mode) is not ActorRatioMode.BRANCH_SPECIFIC
        ):
            raise ConfigError(
                "branchwise_active_mean_v1 requires actor_ratio_mode=branch_specific"
            )
        try:
            AgentCreditMode(mappo.agent_credit_mode)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "training.mappo.agent_credit_mode must be one of "
                f"{tuple(mode.value for mode in AgentCreditMode)}, "
                f"got {mappo.agent_credit_mode!r}"
            ) from exc
        try:
            route_decoder_mode = RouteDecoderMode(mappo.route_decoder_mode)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "training.mappo.route_decoder_mode must be one of "
                f"{tuple(mode.value for mode in RouteDecoderMode)}, "
                f"got {mappo.route_decoder_mode!r}"
            ) from exc
        if (not isinstance(mappo.route_feature_mode, str)
                or mappo.route_feature_mode not in {"original_option_features_v1", "physical_relations_v1"}):
            raise ConfigError("invalid training.mappo.route_feature_mode")
        if (mappo.route_feature_mode == "physical_relations_v1"
                and route_decoder_mode is not RouteDecoderMode.UNIFIED_NODE_V1):
            raise ConfigError("physical_relations_v1 requires unified_node_v1")
        if route_decoder_mode is RouteDecoderMode.UNIFIED_NODE_V1 and (
            mappo.destination_decoder_mode != "legacy"
            or mappo.destination_supervision_mode != "legacy"
            or mappo.destination_oracle_rank_coefficient != 0
        ):
            raise ConfigError("unified_node_v1 requires legacy destination without destination supervision")
        try:
            TxDecoderMode(mappo.tx_decoder_mode)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "training.mappo.tx_decoder_mode must be one of "
                f"{tuple(mode.value for mode in TxDecoderMode)}, "
                f"got {mappo.tx_decoder_mode!r}"
            ) from exc
        try:
            CpuFrequencyDecoderMode(mappo.cpu_frequency_decoder_mode)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "training.mappo.cpu_frequency_decoder_mode must be legacy or candidate_aware_v1"
            ) from exc
        try:
            route_credit_mode = RouteCreditMode(mappo.route_credit_mode)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "training.mappo.route_credit_mode must be one of "
                f"{tuple(mode.value for mode in RouteCreditMode)}, "
                f"got {mappo.route_credit_mode!r}"
            ) from exc
        if (
            route_decoder_mode
            in {
                RouteDecoderMode.CANDIDATE_AWARE_V1,
                RouteDecoderMode.OPTION_AWARE_V1,
            }
            and route_credit_mode is not RouteCreditMode.SHARED_GAE
        ):
            raise ConfigError(
                f"{route_decoder_mode.value} requires "
                "training.mappo.route_credit_mode=shared_gae"
            )
        if (
            route_decoder_mode
            in {RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1, RouteDecoderMode.UNIFIED_NODE_V1}
            and route_credit_mode
            not in {
                RouteCreditMode.SHARED_GAE,
                RouteCreditMode.ROLLOUT_CAPPED_ROUTE_EVENT_NSTEP,
                RouteCreditMode.TASK_EVENT_VALUE_V1,
            }
        ):
            raise ConfigError(
                "hierarchical_option_aware_v1 requires "
                "training.mappo.route_credit_mode to be shared_gae or "
                "rollout_capped_route_event_nstep or task_event_value_v1"
            )
        if not isinstance(mappo.route_choice_stability_enabled, bool):
            raise ConfigError(
                "training.mappo.route_choice_stability_enabled must be boolean"
            )
        for name, value in (
            (
                "route_choice_entropy_floor_nats",
                mappo.route_choice_entropy_floor_nats,
            ),
            ("route_choice_stability_coef", mappo.route_choice_stability_coef),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ConfigError(
                    f"training.mappo.{name} must be finite and non-negative"
                )
        if mappo.route_choice_entropy_floor_nats > math.log(2.0):
            raise ConfigError(
                "training.mappo.route_choice_entropy_floor_nats must not exceed log(2)"
            )
        if mappo.route_choice_stability_enabled:
            if route_decoder_mode is not RouteDecoderMode.CANDIDATE_AWARE_V1:
                raise ConfigError(
                    "route choice stability requires "
                    "training.mappo.route_decoder_mode=candidate_aware_v1"
                )
            if mappo.route_choice_entropy_floor_nats != 0.20:
                raise ConfigError(
                    "Route Preference Collapse Fix V1 requires "
                    "training.mappo.route_choice_entropy_floor_nats=0.20"
                )
        if (
            isinstance(mappo.route_gae_lambda, bool)
            or not isinstance(mappo.route_gae_lambda, (int, float))
            or not math.isfinite(float(mappo.route_gae_lambda))
            or not 0.0 <= float(mappo.route_gae_lambda) <= 1.0
        ):
            raise ConfigError(
                "training.mappo.route_gae_lambda must lie in [0, 1]"
            )
        if route_credit_mode is RouteCreditMode.ROUTE_SPECIFIC_GAE:
            required = (
                ("gamma", mappo.gamma, 0.99),
                ("gae_lambda", mappo.gae_lambda, 0.95),
                ("route_gae_lambda", mappo.route_gae_lambda, 1.0),
                (
                    "actor_ratio_mode",
                    ActorRatioMode(mappo.actor_ratio_mode),
                    ActorRatioMode.BRANCH_SPECIFIC,
                ),
                (
                    "agent_credit_mode",
                    AgentCreditMode(mappo.agent_credit_mode),
                    AgentCreditMode.ROLE_DECOMPOSED,
                ),
            )
            for name, actual, expected in required:
                if actual != expected:
                    expected_value = getattr(expected, "value", expected)
                    raise ConfigError(
                        "route_specific_gae requires "
                        f"training.mappo.{name}={expected_value}"
                    )
        if (
            route_credit_mode
            is RouteCreditMode.ROLLOUT_CAPPED_ROUTE_EVENT_NSTEP
        ):
            required = (
                ("gamma", mappo.gamma, 0.99),
                ("gae_lambda", mappo.gae_lambda, 0.95),
                (
                    "actor_ratio_mode",
                    ActorRatioMode(mappo.actor_ratio_mode),
                    ActorRatioMode.BRANCH_SPECIFIC,
                ),
                (
                    "agent_credit_mode",
                    AgentCreditMode(mappo.agent_credit_mode),
                    AgentCreditMode.ROLE_DECOMPOSED,
                ),
            )
            for name, actual, expected in required:
                if actual != expected:
                    expected_value = getattr(expected, "value", expected)
                    raise ConfigError(
                        "rollout_capped_route_event_nstep requires "
                        f"training.mappo.{name}={expected_value}"
                    )
        for name in ("route_task_credit_coefficient", "route_task_value_coefficient",
                     "route_counterfactual_credit_coefficient", "route_q_value_loss_coefficient"):
            value = getattr(mappo, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ConfigError(f"{name} must be finite and non-negative")
        if mappo.route_value_mode not in {"off", "candidate_counterfactual_q_v1"}:
            raise ConfigError("invalid training.mappo.route_value_mode")
        if mappo.route_actor_credit_source not in {"counterfactual_q", "task_event"}:
            raise ConfigError("invalid training.mappo.route_actor_credit_source")
        if mappo.destination_decoder_mode not in {"legacy", "system_aware_v1"}:
            raise ConfigError("invalid destination_decoder_mode")
        coefficient = mappo.destination_oracle_rank_coefficient
        if mappo.destination_supervision_mode not in {"legacy", "online_oracle", "offline_pretrain"}:
            raise ConfigError("invalid destination_supervision_mode")
        if mappo.destination_supervision_mode == "offline_pretrain" and (
                coefficient != 0 or mappo.destination_decoder_mode != "system_aware_v1"):
            raise ConfigError("offline_pretrain requires system-aware scorer and zero online CE")
        if mappo.destination_supervision_mode == "online_oracle" and coefficient <= 0:
            raise ConfigError("online_oracle requires positive ranking coefficient")
        if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) or not math.isfinite(coefficient) or coefficient < 0:
            raise ConfigError("destination_oracle_rank_coefficient must be finite and nonnegative")
        if mappo.destination_decoder_mode != "legacy":
            if (route_decoder_mode is not RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1
                    or mappo.route_actor_credit_source != "task_event"
                    or mappo.route_q_input_mode != "normalized"):
                raise ConfigError("E15 destination decoder requires P1a hierarchical/task-event/normalized-Q baseline")
        if coefficient > 0 and mappo.destination_decoder_mode != "system_aware_v1":
            raise ConfigError("destination ranking requires system-aware decoder")
        if (mappo.route_actor_credit_source == "task_event"
                and mappo.route_value_mode != "candidate_counterfactual_q_v1"):
            raise ConfigError("task_event actor source requires retained candidate Route Q")
        if mappo.route_q_input_mode not in {"legacy", "normalized", "dual_branch"}:
            raise ConfigError("invalid training.mappo.route_q_input_mode")
        if mappo.route_q_input_mode != "legacy" and mappo.route_value_mode == "off":
            raise ConfigError("conditioned Route Q requires enabled candidate Route Q")
        if mappo.route_value_mode != "off":
            if (route_credit_mode is not RouteCreditMode.TASK_EVENT_VALUE_V1
                    or mappo.route_q_value_loss_coefficient <= 0):
                raise ConfigError("candidate Route Q requires task_event_value_v1 and positive Q loss")
        if route_credit_mode is RouteCreditMode.TASK_EVENT_VALUE_V1:
            if (route_decoder_mode not in {RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1, RouteDecoderMode.UNIFIED_NODE_V1}
                    or not mappo.route_private_backbone
                    or ActorRatioMode(mappo.actor_ratio_mode) is not ActorRatioMode.BRANCH_SPECIFIC
                    or AgentCreditMode(mappo.agent_credit_mode) is not AgentCreditMode.ROLE_DECOMPOSED
                    or not mappo.route_training_telemetry_enabled
                    or mappo.route_task_value_coefficient <= 0):
                raise ConfigError("task_event_value_v1 requires private hierarchical or unified Route, "
                                  "branch-specific ratio, role credit, telemetry and positive task value loss")
            if self.reward.credit_assignment_mode != "responsibility_terminal":
                raise ConfigError("task_event_value_v1 requires responsibility_terminal attribution")
        if mappo.tx_upper_decoder_mode not in {"legacy", "private_candidate_binary_v1"}:
            raise ConfigError("invalid training.mappo.tx_upper_decoder_mode")
        if mappo.cpu_scheduler_mode not in {"legacy_hierarchical", "deadline_joint_v1"}:
            raise ConfigError("invalid training.mappo.cpu_scheduler_mode")
        if mappo.cpu_anti_starvation_mode not in {"off", "deadline_feasibility_v1"}:
            raise ConfigError("invalid training.mappo.cpu_anti_starvation_mode")
        if (mappo.cpu_anti_starvation_mode != "off"
                and mappo.cpu_scheduler_mode != "deadline_joint_v1"):
            raise ConfigError("CPU anti-starvation requires deadline_joint_v1")
        if mappo.route_exploration_mode not in {"off", "annealed_top_level_log_barrier_v1"}:
            raise ConfigError("invalid training.mappo.route_exploration_mode")
        if not isinstance(mappo.route_training_telemetry_enabled, bool):
            raise ConfigError("route_training_telemetry_enabled must be bool")
        route_coefficient = mappo.route_exploration_initial_coefficient
        if (isinstance(route_coefficient, bool)
                or not isinstance(route_coefficient, (int, float))
                or not math.isfinite(route_coefficient) or route_coefficient < 0):
            raise ConfigError("route_exploration_initial_coefficient must be finite and non-negative")
        route_steps = mappo.route_exploration_anneal_steps
        if isinstance(route_steps, bool) or not isinstance(route_steps, int) or route_steps <= 0:
            raise ConfigError("route_exploration_anneal_steps must be a positive integer")
        if mappo.route_exploration_mode != "off":
            if (route_decoder_mode not in {RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1, RouteDecoderMode.UNIFIED_NODE_V1}
                    or not mappo.route_private_backbone):
                raise ConfigError("Route log barrier requires hierarchical Route and its private backbone")
            if route_coefficient == 0:
                raise ConfigError("enabled Route log barrier requires a positive initial coefficient")
        if (mappo.route_training_telemetry_enabled
                and route_decoder_mode not in {RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1, RouteDecoderMode.UNIFIED_NODE_V1}):
            raise ConfigError("Route training telemetry requires hierarchical Route")
        if (not isinstance(mappo.tx_upper_optimization_mode, str)
                or mappo.tx_upper_optimization_mode not in {"ppo_standard", "deterministic_stability_v1"}):
            raise ConfigError("invalid training.mappo.tx_upper_optimization_mode")
        if (not isinstance(mappo.tx_upper_credit_mode, str)
                or mappo.tx_upper_credit_mode not in {"shared_agent_advantage", "counterfactual_q_v1", "factual_team_gae_v1", "paired_shadow_v1"}):
            raise ConfigError("invalid training.mappo.tx_upper_credit_mode")
        if mappo.tx_upper_credit_mode in {"counterfactual_q_v1", "factual_team_gae_v1", "paired_shadow_v1"} and (
                mappo.tx_upper_optimization_mode != "ppo_standard"
                or mappo.tx_upper_decoder_mode != "private_candidate_binary_v1"
                or mappo.cpu_scheduler_mode != "deadline_joint_v1"
                or not mappo.route_private_backbone):
            raise ConfigError("TX branch credit requires E8 architecture and ppo_standard; E9 stabilization cannot coexist")
        for name in ("tx_shadow_max_pairs_per_rollout", "tx_shadow_step_budget_per_rollout",
                     "tx_shadow_rng_seed", "tx_shadow_max_conditional_draws"):
            value = getattr(mappo, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigError(f"training.mappo.{name} must be a nonnegative integer")
        if (mappo.tx_shadow_selector != "cyclic_source_first_active_v1"
                or mappo.tx_shadow_rng_mode != "slot_addressed_v1"
                or mappo.tx_shadow_advantage_normalization != "paired_rms_v1"):
            raise ConfigError("unsupported paired TX selector/RNG/normalization protocol")
        if mappo.tx_upper_credit_mode == "paired_shadow_v1" and (
                not 1 <= mappo.tx_shadow_max_pairs_per_rollout <= self.environment.uav_count
                or mappo.tx_shadow_step_budget_per_rollout < 2
                or mappo.tx_shadow_max_conditional_draws < 1):
            raise ConfigError("paired TX requires bounded positive pair/step/draw budgets")
        if (not isinstance(mappo.tx_q_target_normalization, str)
                or mappo.tx_q_target_normalization not in {"none", "popart_v1"}):
            raise ConfigError("invalid training.mappo.tx_q_target_normalization")
        if (mappo.tx_q_target_normalization != "none"
                and mappo.tx_upper_credit_mode != "counterfactual_q_v1"):
            raise ConfigError("TX Q target normalization requires counterfactual_q_v1")
        if mappo.tx_upper_optimization_mode == "deterministic_stability_v1" and (
                mappo.tx_upper_decoder_mode != "private_candidate_binary_v1"
                or mappo.cpu_scheduler_mode != "deadline_joint_v1"
                or not mappo.route_private_backbone):
            raise ConfigError("TX deterministic stability requires the E8 private TX / joint CPU / private Route architecture")
        integrated = (mappo.tx_upper_decoder_mode != "legacy" or mappo.cpu_scheduler_mode != "legacy_hierarchical")
        if mappo.tx_upper_decoder_mode != "legacy" and mappo.tx_decoder_mode != "hierarchical_candidate_aware_v1":
            raise ConfigError("private TX upper requires hierarchical_candidate_aware_v1 destination")
        if integrated and (mappo.actor_ratio_mode != "branch_specific" or mappo.policy_loss_aggregation_mode != "branchwise_active_mean_v1"):
            raise ConfigError("integrated schedulers require branch_specific / branchwise_active_mean_v1 PPO")
        if mappo.cpu_scheduler_mode == "deadline_joint_v1" and mappo.cpu_frequency_decoder_mode != "legacy":
            raise ConfigError("deadline_joint_v1 requires legacy frequency marker; old frequency decoder is inactive")
        if not isinstance(mappo.route_private_backbone, bool):
            raise ConfigError("training.mappo.route_private_backbone must be boolean")
        if mappo.optimizer != "Adam":
            raise ConfigError("training.mappo.optimizer must be 'Adam'")
        if mappo.optimizer_topology != "separate_actor_critic":
            raise ConfigError(
                "training.mappo.optimizer_topology must be 'separate_actor_critic'"
            )
        for name, value in (
            ("actor_learning_rate", mappo.actor_learning_rate),
            ("route_head_lr_scale", mappo.route_head_lr_scale),
            ("critic_learning_rate", mappo.critic_learning_rate),
            ("adam_eps", mappo.adam_eps),
            ("gradient_clip_norm", mappo.gradient_clip_norm),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0.0
            ):
                raise ConfigError(f"training.mappo.{name} must be finite and positive")
        for name, value in (
            ("adam_beta1", mappo.adam_beta1),
            ("adam_beta2", mappo.adam_beta2),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 < value < 1.0
            ):
                raise ConfigError(f"training.mappo.{name} must lie in (0, 1)")
        if mappo.weight_decay != 0.0:
            raise ConfigError("training.mappo.weight_decay must remain 0.0")
        if not isinstance(mappo.entropy_coefficient_schedule_enabled, bool):
            raise ConfigError(
                "training.mappo.entropy_coefficient_schedule_enabled must be boolean"
            )
        for name, value in (
            ("entropy_coefficient", mappo.entropy_coefficient),
            (
                "route_entropy_start_coefficient",
                mappo.route_entropy_start_coefficient,
            ),
            (
                "route_only_entropy_initial_coefficient",
                mappo.route_only_entropy_initial_coefficient,
            ),
            (
                "route_only_entropy_final_coefficient",
                mappo.route_only_entropy_final_coefficient,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ConfigError(
                    f"training.mappo.{name} must be finite and non-negative"
                )
        if (
            isinstance(mappo.route_only_entropy_anneal_steps, bool)
            or not isinstance(mappo.route_only_entropy_anneal_steps, int)
            or mappo.route_only_entropy_anneal_steps < 0
        ):
            raise ConfigError(
                "training.mappo.route_only_entropy_anneal_steps must be a "
                "non-negative integer"
            )
        if (
            mappo.route_only_entropy_initial_coefficient
            != mappo.route_only_entropy_final_coefficient
            and mappo.route_only_entropy_anneal_steps == 0
        ):
            raise ConfigError(
                "training.mappo.route_only_entropy_anneal_steps must be positive "
                "when initial and final coefficients differ"
            )
        if (
            not mappo.route_only_entropy_enabled
            and mappo.route_only_entropy_anneal_steps != 0
        ):
            raise ConfigError(
                "training.mappo.route_only_entropy_anneal_steps must be 0 when "
                "the route-only coefficient is disabled"
            )
        if (
            mappo.entropy_coefficient_schedule_enabled
            and mappo.route_only_entropy_enabled
        ):
            raise ConfigError(
                "legacy and additive route entropy schedules cannot both be enabled"
            )
        for name, value in (
            (
                "route_entropy_schedule_start_step",
                mappo.route_entropy_schedule_start_step,
            ),
            (
                "route_entropy_schedule_end_step",
                mappo.route_entropy_schedule_end_step,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigError(
                    f"training.mappo.{name} must be a non-negative integer"
                )
        if (
            mappo.route_entropy_schedule_end_step
            <= mappo.route_entropy_schedule_start_step
        ):
            raise ConfigError(
                "training.mappo.route entropy schedule end must exceed start"
            )
        if mappo.entropy_coefficient_schedule_enabled:
            preregistered = (
                ("entropy_coefficient", mappo.entropy_coefficient, 0.01),
                (
                    "route_entropy_schedule_start_step",
                    mappo.route_entropy_schedule_start_step,
                    3072,
                ),
                (
                    "route_entropy_schedule_end_step",
                    mappo.route_entropy_schedule_end_step,
                    32768,
                ),
            )
            for name, actual, expected in preregistered:
                if actual != expected:
                    raise ConfigError(
                        "the preregistered route entropy treatment requires "
                        f"training.mappo.{name}={expected}"
                    )
            if mappo.route_entropy_start_coefficient not in (0.03, 0.05):
                raise ConfigError(
                    "the preregistered route entropy treatment requires "
                    "training.mappo.route_entropy_start_coefficient "
                    "to be one of (0.03, 0.05)"
                )
        for name, value in (
            ("rollout_length_slots", mappo.rollout_length_slots),
            ("recurrent_chunk_length_slots", mappo.recurrent_chunk_length_slots),
            ("sequence_minibatch_size", mappo.sequence_minibatch_size),
            ("update_epochs", mappo.update_epochs),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"training.mappo.{name} must be a positive integer")
        if mappo.rollout_length_slots % mappo.recurrent_chunk_length_slots != 0:
            raise ConfigError(
                "training.mappo.rollout_length_slots must be divisible by "
                "recurrent_chunk_length_slots"
            )
        if mappo.recurrent_chunk_count != mappo.sequence_minibatch_size:
            raise ConfigError(
                "one recurrent PPO minibatch must contain every chunk from the rollout"
            )
        if not mappo.zero_grad_set_to_none:
            raise ConfigError("training.mappo.zero_grad_set_to_none must remain true")
        if mappo.gradient_clip_scope != "separate_actor_critic_global_norm":
            raise ConfigError(
                "training.mappo.gradient_clip_scope must be "
                "'separate_actor_critic_global_norm'"
            )
        if not mappo.require_full_rollout:
            raise ConfigError("training.mappo.require_full_rollout must remain true")
        if mappo.training_device not in {"cpu", "cuda"}:
            raise ConfigError("training.mappo.training_device must be 'cpu' or 'cuda'")
        if (
            isinstance(mappo.checkpoint_schema_version, bool)
            or not isinstance(mappo.checkpoint_schema_version, int)
            or mappo.checkpoint_schema_version != CHECKPOINT_SCHEMA_VERSION
        ):
            raise ConfigError(
                "training.mappo.checkpoint_schema_version must remain "
                f"{CHECKPOINT_SCHEMA_VERSION}"
            )
        if not isinstance(mappo.trajectory_credit_telemetry_enabled, bool):
            raise ConfigError(
                "training.mappo.trajectory_credit_telemetry_enabled must be boolean"
            )
        if not isinstance(mappo.conditional_credit_telemetry_enabled, bool):
            raise ConfigError(
                "training.mappo.conditional_credit_telemetry_enabled must be boolean"
            )
        if (mappo.conditional_credit_telemetry_enabled
            and ActorRatioMode(mappo.actor_ratio_mode) is not ActorRatioMode.BRANCH_SPECIFIC):
            raise ConfigError(
                "conditional-credit telemetry requires branch_specific actor ratios"
            )
        if (isinstance(mappo.conditional_credit_telemetry_max_bytes, bool)
            or not isinstance(mappo.conditional_credit_telemetry_max_bytes, int)
            or not 1048576 <= mappo.conditional_credit_telemetry_max_bytes <= 2147483648):
            raise ConfigError(
                "training.mappo.conditional_credit_telemetry_max_bytes must be 1 MiB–2 GiB"
            )
        if not isinstance(mappo.route_diagnostic_samples_enabled, bool):
            raise ConfigError(
                "training.mappo.route_diagnostic_samples_enabled must be boolean"
            )
        if not isinstance(mappo.route_oracle_counterfactual_enabled, bool):
            raise ConfigError(
                "training.mappo.route_oracle_counterfactual_enabled must be boolean"
            )
        selection_rate = mappo.route_oracle_selection_rate_ppm
        if (
            isinstance(selection_rate, bool)
            or not isinstance(selection_rate, int)
            or not 0 <= selection_rate <= 1_000_000
        ):
            raise ConfigError(
                "training.mappo.route_oracle_selection_rate_ppm must be an integer in [0, 1000000]"
            )
        if mappo.route_oracle_counterfactual_enabled and selection_rate == 0:
            raise ConfigError(
                "enabled Oracle-1 collection requires a positive selection rate"
            )
        checkpoint_interval = mappo.checkpoint_interval_steps
        if (
            isinstance(checkpoint_interval, bool)
            or not isinstance(checkpoint_interval, int)
            or checkpoint_interval <= 0
        ):
            raise ConfigError(
                "training.mappo.checkpoint_interval_steps must be a positive integer"
            )
        if self.mode == "rl" and self.method_id == "ca_gat_mappo":
            if checkpoint_interval % env.episode_horizon != 0:
                raise ConfigError(
                    "CA-GAT-MAPPO Checkpoint V1 requires checkpoint_interval_steps "
                    "to be divisible by environment.episode_horizon"
                )
            if checkpoint_interval >= mappo.max_environment_transitions:
                raise ConfigError(
                    "CA-GAT-MAPPO Checkpoint V1 requires checkpoint_interval_steps "
                    "to be less than max_training_environment_steps"
                )
        frozen_lifecycle_flags = {
            "carry_hidden_across_update_boundary": (
                mappo.carry_hidden_across_update_boundary,
                True,
            ),
            "policy_sampling_reset_each_episode": (
                mappo.policy_sampling_reset_each_episode,
                False,
            ),
            "discard_final_partial_rollout": (
                mappo.discard_final_partial_rollout,
                True,
            ),
        }
        for name, (actual, expected) in frozen_lifecycle_flags.items():
            if actual is not expected:
                raise ConfigError(f"training.mappo.{name} must remain {expected}")
        for name, value in (
            ("max_training_episodes", mappo.max_training_episodes),
            ("max_training_environment_steps", mappo.max_training_environment_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"training.mappo.{name} must be a positive integer")
        if not isinstance(self.training.formal_rl_enabled, bool):
            raise ConfigError("training.formal_rl_enabled must be boolean")
        disabled_mechanisms = {
            "chunk_shuffle": mappo.chunk_shuffle,
            "timestep_shuffle": mappo.timestep_shuffle,
            "allow_incomplete_minibatch": mappo.allow_incomplete_minibatch,
            "sequence_padding": mappo.sequence_padding,
            "padded_actor_forward": mappo.padded_actor_forward,
            "advantage_normalization": mappo.advantage_normalization,
            "value_clipping": mappo.value_clipping,
            "target_kl_enabled": mappo.target_kl_enabled,
            "kl_early_stopping": mappo.kl_early_stopping,
            "learning_rate_schedule_enabled": mappo.learning_rate_schedule_enabled,
            "gradient_accumulation": mappo.gradient_accumulation,
            "mixed_precision": mappo.mixed_precision,
        }
        enabled = sorted(name for name, value in disabled_mechanisms.items() if value)
        if enabled:
            raise ConfigError(
                "the frozen MAPPO contract requires these mechanisms to remain disabled: "
                + ", ".join(enabled)
            )
        if self.evaluation.update_network:
            raise ConfigError("evaluation.update_network must remain false")


def compute_mappo_training_accounting(config: RunConfig) -> MAPPOTrainingAccounting:
    """Compute collection and optimization counts without running an environment."""

    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    config.validate()
    mappo = config.training.mappo
    episode_budget = mappo.max_training_episodes * config.environment.episode_horizon
    collected = min(episode_budget, mappo.max_environment_transitions)
    update_count, unused_tail = divmod(collected, mappo.rollout_length_slots)
    optimized = update_count * mappo.rollout_length_slots
    return MAPPOTrainingAccounting(
        collected_environment_transitions=collected,
        optimized_transitions=optimized,
        unused_final_tail_transitions=unused_tail,
        ppo_update_count=update_count,
    )


def validate_mappo_checkpoint_contract(config: RunConfig) -> None:
    """Validate the pure Checkpoint V1 geometry without reading or writing files."""

    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    config.validate()
    mappo = config.training.mappo
    interval = mappo.checkpoint_interval_steps
    if interval % config.environment.episode_horizon != 0:
        raise ConfigError(
            "Checkpoint V1 periodic interval must be episode-boundary aligned"
        )
    if interval >= mappo.max_environment_transitions:
        raise ConfigError(
            "Checkpoint V1 periodic interval must be less than the training budget"
        )


def compute_mappo_periodic_checkpoint_steps(config: RunConfig) -> tuple[int, ...]:
    """Return all immutable periodic-resume boundaries before final completion."""

    validate_mappo_checkpoint_contract(config)
    mappo = config.training.mappo
    return tuple(
        range(
            mappo.checkpoint_interval_steps,
            mappo.max_environment_transitions,
            mappo.checkpoint_interval_steps,
        )
    )


def mappo_checkpoint_kind_at(
    config: RunConfig,
    collected_environment_transitions: int,
) -> str | None:
    """Classify one collection count as periodic, final, or not a checkpoint."""

    validate_mappo_checkpoint_contract(config)
    if (
        isinstance(collected_environment_transitions, bool)
        or not isinstance(collected_environment_transitions, int)
        or collected_environment_transitions < 0
        or collected_environment_transitions
        > config.training.mappo.max_environment_transitions
    ):
        raise ConfigError(
            "collected_environment_transitions must lie within the training budget"
        )
    mappo = config.training.mappo
    if collected_environment_transitions == mappo.max_environment_transitions:
        return CHECKPOINT_KIND_FINAL_COMPLETED
    if (
        collected_environment_transitions > 0
        and collected_environment_transitions % mappo.checkpoint_interval_steps == 0
    ):
        return CHECKPOINT_KIND_PERIODIC_RESUME
    return None


def compute_mappo_checkpoint_active_rollout_length(
    config: RunConfig,
    collected_environment_transitions: int,
) -> int:
    """Return the saved active-rollout length at a valid Checkpoint V1 boundary."""

    kind = mappo_checkpoint_kind_at(config, collected_environment_transitions)
    if kind is None:
        raise ConfigError("transition count is not a Checkpoint V1 boundary")
    if kind == CHECKPOINT_KIND_FINAL_COMPLETED:
        return 0
    return (
        collected_environment_transitions
        % config.training.mappo.rollout_length_slots
    )


def mappo_periodic_checkpoint_path(
    config: RunConfig,
    collected_environment_transitions: int,
) -> str:
    """Return one periodic path without creating or inspecting filesystem state."""

    kind = mappo_checkpoint_kind_at(config, collected_environment_transitions)
    if kind != CHECKPOINT_KIND_PERIODIC_RESUME:
        raise ConfigError("periodic checkpoint paths require PERIODIC_RESUME boundaries")
    directory = Path(config.artifact_paths()["checkpoint_directory"])
    return str(directory / f"step_{collected_environment_transitions}.pt")


def mappo_final_checkpoint_path(config: RunConfig) -> str:
    """Return the canonical final path without creating or inspecting it."""

    validate_mappo_checkpoint_contract(config)
    return config.artifact_paths()["final_checkpoint"]


def validate_mappo_checkpoint_resume_compatibility(
    config: RunConfig,
    *,
    schema_version: int,
    checkpoint_kind: str,
    method_id: str,
    git_commit: str,
    config_hash: str,
    training_device: str,
    cuda_available: bool,
    checkpoint_actor_ratio_mode: str | None = None,
    checkpoint_agent_credit_mode: str | None = None,
    checkpoint_route_decoder_mode: str | None = None,
    checkpoint_tx_decoder_mode: str | None = None,
    checkpoint_cpu_frequency_decoder_mode: str | None = None,
    checkpoint_route_credit_mode: str | None = None,
) -> None:
    """Apply the fail-fast V1 metadata checks without loading checkpoint bytes."""

    if config.training.mappo.trajectory_credit_telemetry_enabled:
        raise ConfigError(
            "trajectory-credit telemetry V1 is fresh-run-only; resume is unsupported"
        )
    validate_mappo_checkpoint_contract(config)
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != CHECKPOINT_SCHEMA_VERSION
    ):
        raise ConfigError("checkpoint schema version mismatch")
    if checkpoint_kind == CHECKPOINT_KIND_FINAL_COMPLETED:
        raise ConfigError("training already complete")
    if checkpoint_kind != CHECKPOINT_KIND_PERIODIC_RESUME:
        raise ConfigError("checkpoint kind is not resumable")
    if config.mode != "rl":
        raise ConfigError("checkpoint resume requires mode='rl'")
    if config.method_id != "ca_gat_mappo" or method_id != config.method_id:
        raise ConfigError("checkpoint method_id mismatch")
    if git_commit != config.git_commit:
        raise ConfigError("checkpoint git_commit mismatch")
    try:
        source_ratio_mode = ActorRatioMode(
            ActorRatioMode.JOINT.value
            if checkpoint_actor_ratio_mode is None
            else checkpoint_actor_ratio_mode
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError("checkpoint actor_ratio_mode is invalid") from exc
    target_ratio_mode = ActorRatioMode(config.training.mappo.actor_ratio_mode)
    if source_ratio_mode != target_ratio_mode:
        raise ConfigError(
            "checkpoint actor_ratio_mode mismatch; changing between joint and "
            "branch_specific requires a new run"
        )
    try:
        source_credit_mode = AgentCreditMode(
            AgentCreditMode.TEAM.value
            if checkpoint_agent_credit_mode is None
            else checkpoint_agent_credit_mode
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError("checkpoint agent_credit_mode is invalid") from exc
    target_credit_mode = AgentCreditMode(config.training.mappo.agent_credit_mode)
    if source_credit_mode != target_credit_mode:
        raise ConfigError(
            "checkpoint agent_credit_mode mismatch; changing between team and "
            "role_decomposed requires a new run"
        )
    try:
        source_route_decoder_mode = RouteDecoderMode(
            RouteDecoderMode.LEGACY.value
            if checkpoint_route_decoder_mode is None
            else checkpoint_route_decoder_mode
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError("checkpoint route_decoder_mode is invalid") from exc
    target_route_decoder_mode = RouteDecoderMode(
        config.training.mappo.route_decoder_mode
    )
    if source_route_decoder_mode != target_route_decoder_mode:
        raise ConfigError(
            "checkpoint route_decoder_mode mismatch; changing route decoder "
            "requires a new run"
        )
    try:
        source_tx_decoder_mode = TxDecoderMode(
            TxDecoderMode.LEGACY.value
            if checkpoint_tx_decoder_mode is None
            else checkpoint_tx_decoder_mode
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError("checkpoint tx_decoder_mode is invalid") from exc
    if source_tx_decoder_mode != TxDecoderMode(config.training.mappo.tx_decoder_mode):
        raise ConfigError(
            "checkpoint tx_decoder_mode mismatch; changing transmit decoder "
            "requires a new run"
        )
    try:
        source_cpu_mode = CpuFrequencyDecoderMode(
            CpuFrequencyDecoderMode.LEGACY.value
            if checkpoint_cpu_frequency_decoder_mode is None
            else checkpoint_cpu_frequency_decoder_mode
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError("checkpoint cpu_frequency_decoder_mode is invalid") from exc
    if source_cpu_mode != CpuFrequencyDecoderMode(config.training.mappo.cpu_frequency_decoder_mode):
        raise ConfigError(
            "checkpoint cpu_frequency_decoder_mode mismatch; changing CPU frequency decoder requires a new run"
        )
    try:
        source_route_credit_mode = RouteCreditMode(
            RouteCreditMode.SHARED_GAE.value
            if checkpoint_route_credit_mode is None
            else checkpoint_route_credit_mode
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError("checkpoint route_credit_mode is invalid") from exc
    target_route_credit_mode = RouteCreditMode(
        config.training.mappo.route_credit_mode
    )
    if source_route_credit_mode != target_route_credit_mode:
        raise ConfigError(
            "checkpoint route_credit_mode mismatch; changing route-credit mode "
            "requires a new run"
        )
    if config_hash != config.config_hash:
        raise ConfigError("checkpoint canonical config_hash mismatch")
    if training_device != config.training.mappo.training_device:
        raise ConfigError("checkpoint training device mismatch")
    validate_mappo_training_device(config, cuda_available=cuda_available)


def validate_mappo_training_device(
    config: RunConfig,
    *,
    cuda_available: bool,
) -> str:
    """Validate the explicit runtime device without selecting a fallback."""

    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    if not isinstance(cuda_available, bool):
        raise TypeError("cuda_available must be boolean")
    config.validate()
    device = config.training.mappo.training_device
    if device == "cuda" and not cuda_available:
        raise ConfigError(
            "training_device='cuda' requires CUDA; silent CPU fallback is forbidden"
        )
    return device


def require_formal_rl_enabled(config: RunConfig) -> None:
    """Fail before real training work when the explicit execution gate is closed."""

    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    config.validate()
    if not config.training.formal_rl_enabled:
        raise ConfigError(
            "formal RL training is disabled before environment reset, collection, or update"
        )


SCENARIO_DEFAULTS: dict[str, dict[str, Any]] = {
    "small": {
        "uav_count": 4,
        "candidate_neighbor_radius_m": 525.0,
        "arrival_probabilities": [0.04, 0.05, 0.06, 0.05],
        "profile_assignment": ["Balanced", "Resource-poor", "Compute-rich", "Energy-limited"],
        "profile_perturbations": [
            [-0.013814309988, 0.044593840641, 0.044559189965, 0.033901094602],
            [0.024305247334, -0.014446956216, 0.005750819819, -0.029482012276],
            [-0.000601140633, -0.015719950799, 0.049877636039, 0.034480593142],
            [-0.000524427719, 0.006763643213, -0.028161367743, -0.001054193533],
        ],
    },
    "medium": {
        "uav_count": 6,
        "candidate_neighbor_radius_m": 500.0,
        "arrival_probabilities": [0.03, 0.04, 0.05, 0.06, 0.05, 0.07],
        "profile_assignment": [
            "Balanced", "Resource-poor", "Compute-rich", "Energy-limited", "Communication-rich", "Balanced"
        ],
        "profile_perturbations": [
            [-0.013814309988, 0.044593840641, 0.044559189965, 0.033901094602],
            [0.024305247334, -0.014446956216, 0.005750819819, -0.029482012276],
            [-0.000601140633, -0.015719950799, 0.049877636039, 0.034480593142],
            [-0.000524427719, 0.006763643213, -0.028161367743, -0.001054193533],
            [-0.048552394638, 0.010776985782, -0.02486412725, -0.0013647122],
            [-0.01912646906, -0.012256723723, 0.001229671851, 0.011639336192],
        ],
    },
    "large": {
        "uav_count": 8,
        "candidate_neighbor_radius_m": 500.0,
        "arrival_probabilities": [0.02, 0.04, 0.06, 0.08, 0.02, 0.04, 0.06, 0.08],
        "profile_assignment": [
            "Resource-poor", "Balanced", "Compute-rich", "Energy-limited", "Communication-rich",
            "Resource-poor", "Balanced", "Compute-rich",
        ],
        "profile_perturbations": [
            [-0.013814309988, 0.044593840641, 0.044559189965, 0.033901094602],
            [0.024305247334, -0.014446956216, 0.005750819819, -0.029482012276],
            [-0.000601140633, -0.015719950799, 0.049877636039, 0.034480593142],
            [-0.000524427719, 0.006763643213, -0.028161367743, -0.001054193533],
            [-0.048552394638, 0.010776985782, -0.02486412725, -0.0013647122],
            [-0.01912646906, -0.012256723723, 0.001229671851, 0.011639336192],
            [-0.005497708292, -0.026275079713, -0.016823141819, 0.017532792782],
            [0.024883910851, 0.023672284376, 0.010141524582, -0.001992215387],
        ],
    },
    "remote_stress": {
        "uav_count": 4,
        "candidate_neighbor_radius_m": 525.0,
        "minimum_task_slack_slots": 18,
        "maximum_task_slack_slots": 80,
        "task_deadline_factor_min": 1.75,
        "task_deadline_factor_max": 2.5,
        "arrival_probabilities": [0.05, 0.12, 0.02, 0.07],
        "profile_assignment": ["Balanced", "Resource-poor", "Compute-rich", "Energy-limited"],
        "profile_perturbations": [
            [-0.013814309988, 0.044593840641, 0.044559189965, 0.033901094602],
            [0.024305247334, -0.014446956216, 0.005750819819, -0.029482012276],
            [-0.000601140633, -0.015719950799, 0.049877636039, 0.034480593142],
            [-0.000524427719, 0.006763643213, -0.028161367743, -0.001054193533],
        ],
        "profile_ratios": {
            "Resource-poor": [0.7, 0.8, 0.8, 1.2],
            "Balanced": [1.0, 1.0, 1.0, 1.0],
            "Compute-rich": [2.0, 1.0, 1.1, 0.9],
            "Energy-limited": [1.0, 0.8, 0.5, 1.1],
            "Communication-rich": [1.0, 1.5, 1.1, 1.0],
        },
    },
    "remote_stress_v2": {
        "uav_count": 4,
        "candidate_neighbor_radius_m": 525.0,
        "arrival_probabilities": [0.0467, 0.11, 0.02, 0.0633],
        "profile_assignment": ["Balanced", "Resource-poor", "Compute-rich", "Energy-limited"],
        "profile_perturbations": [
            [-0.013814309988, 0.044593840641, 0.044559189965, 0.033901094602],
            [0.024305247334, -0.014446956216, 0.005750819819, -0.029482012276],
            [-0.000601140633, -0.015719950799, 0.049877636039, 0.034480593142],
            [-0.000524427719, 0.006763643213, -0.028161367743, -0.001054193533],
        ],
    },
}


DEFAULT_CONFIG_DATA: dict[str, Any] = {
    "config_version": CONFIG_VERSION,
    "mode": "environment_sanity",
    "method_id": "environment",
    "scenario_id": "small",
    "launch_profile": None,
    "seed": DEFAULT_SEED,
    "environment": asdict(EnvironmentConfig()),
    "reward": asdict(RewardCreditConfig()),
    "action": asdict(ActionConfig()),
    "reproducibility": asdict(ReproducibilityConfig()),
    "training": {
        **asdict(TrainingConfig()),
        "mappo": {
            **asdict(MAPPOConfig()),
            "route_oracle_counterfactual_enabled": False,
            "route_oracle_selection_rate_ppm": 0,
        },
    },
    "evaluation": asdict(EvaluationConfig()),
    "output": asdict(OutputConfig()),
}


def _default_method_for_mode(mode: str) -> str | None:
    return {
        "environment_sanity": "environment",
        "gate0": "environment",
        "random": "random",
        "heuristic": "heuristic",
        "baseline": "local_only",
        "rl": "ca_gat_mappo",
        "evaluation": "ca_gat_mappo",
        "ablation": "ca_gat_mappo",
        "plot": "environment",
    }.get(mode)


def load_run_config(
    config_path: str | Path | None = None,
    *,
    cli_overrides: Mapping[str, Any] | None = None,
    interactive_overrides: Mapping[str, Any] | None = None,
) -> RunConfig:
    """Load defaults, then file, CLI, and interactive overrides in that order."""

    cli = _normalize_overrides(cli_overrides or {}, source="CLI")
    interactive = _normalize_overrides(interactive_overrides or {}, source="interactive")
    merged = copy.deepcopy(DEFAULT_CONFIG_DATA)
    source_path = str(Path(config_path)) if config_path is not None else None
    file_data: Mapping[str, Any] = {}
    if config_path is not None:
        path = Path(config_path)
        if not path.exists() or not path.is_file():
            raise ConfigError(f"config file does not exist or is not a file: {path}")
        loaded_data = _load_mapping_file(path)
        # A snapshot produced by this module is also a valid input config;
        # provenance is metadata, not a second source of typed config leaves.
        file_data = {key: value for key, value in loaded_data.items() if key != "_metadata"}
        _validate_mapping_keys(file_data, RunConfig, path="config")
        merged = _deep_merge(merged, file_data)

    # Scenario values are derived from the frozen scenario table unless the
    # caller explicitly overrides the corresponding environment leaf.
    scenario_id = str(merged.get("scenario_id", "small"))
    if scenario_id not in SUPPORTED_SCENARIOS:
        raise ConfigError(f"scenario_id must be one of {SUPPORTED_SCENARIOS}, got {scenario_id!r}")
    explicit_env = set(file_data.get("environment", {}).keys()) if isinstance(file_data.get("environment", {}), Mapping) else set()
    explicit_env.update(_dotted_top_level_keys(cli, prefix="environment"))
    explicit_env.update(_dotted_top_level_keys(interactive, prefix="environment"))
    for key, value in SCENARIO_DEFAULTS[scenario_id].items():
        if key not in explicit_env:
            merged["environment"][key] = value

    merged = _apply_dotted_overrides(merged, cli)
    merged = _apply_dotted_overrides(merged, interactive)
    if "mode" in merged and "method_id" not in cli and "method_id" not in interactive:
        if "method_id" not in file_data or file_data.get("method_id") in (None, ""):
            merged["method_id"] = _default_method_for_mode(str(merged["mode"])) or merged.get("method_id")

    # Re-apply scenario defaults after mode/file merging only when scenario_id
    # itself was supplied by a higher-priority source.
    final_scenario = str(merged.get("scenario_id", "small"))
    if final_scenario not in SUPPORTED_SCENARIOS:
        raise ConfigError(f"scenario_id must be one of {SUPPORTED_SCENARIOS}, got {final_scenario!r}")
    if final_scenario != scenario_id:
        for key, value in SCENARIO_DEFAULTS[final_scenario].items():
            if key not in explicit_env:
                merged["environment"][key] = value

    config = _construct_run_config(merged)
    _attach_provenance(
        config,
        source_config_path=source_path,
        cli_overrides=cli,
        interactive_overrides=interactive,
    )
    config.validate()
    return config


def write_config_snapshot(config: RunConfig, path: str | Path | None = None) -> Path:
    """Atomically publish a canonical snapshot without replacing an old run."""

    target = Path(path) if path is not None else Path(config.artifact_paths()["config_snapshot"])
    # JSON is valid YAML 1.2 and avoids introducing an unconfirmed dependency.
    return atomic_write_text(
        target,
        json.dumps(config.snapshot_dict(), ensure_ascii=False, indent=2) + "\n",
    )


def _construct_run_config(mapping: Mapping[str, Any]) -> RunConfig:
    values = _coerce_dataclass_mapping(mapping, RunConfig, path="config")
    return RunConfig(**values)


def _attach_provenance(
    config: RunConfig,
    *,
    source_config_path: str | None,
    cli_overrides: Mapping[str, Any],
    interactive_overrides: Mapping[str, Any],
) -> None:
    branch, commit, dirty = _git_metadata()
    object.__setattr__(config, "source_config_path", source_config_path)
    object.__setattr__(config, "cli_overrides", dict(cli_overrides))
    object.__setattr__(config, "interactive_overrides", dict(interactive_overrides))
    object.__setattr__(config, "git_branch", branch)
    object.__setattr__(config, "git_commit", commit)
    object.__setattr__(config, "git_dirty", dirty)


def _validate_mapping_keys(mapping: Mapping[str, Any], cls: type[Any], *, path: str) -> None:
    if not isinstance(mapping, Mapping):
        raise ConfigError(f"{path} must be a mapping")
    hints = get_type_hints(cls)
    allowed = {
        item.name
        for item in cls.__dataclass_fields__.values()
        if item.init
    }
    for key, value in mapping.items():
        if key not in allowed:
            raise ConfigError(f"unknown field {path}.{key}")
        annotation = hints[key] if key in hints else item_type(cls, key)
        nested = _nested_dataclass_type(annotation)
        if nested is not None and isinstance(value, Mapping):
            _validate_mapping_keys(value, nested, path=f"{path}.{key}")
        elif nested is not None and value is not None:
            raise ConfigError(f"{path}.{key} must be a mapping")


def _coerce_dataclass_mapping(mapping: Mapping[str, Any], cls: type[_T], *, path: str) -> dict[str, Any]:
    _validate_mapping_keys(mapping, cls, path=path)
    hints = get_type_hints(cls)
    result: dict[str, Any] = {}
    for item in cls.__dataclass_fields__.values():
        if not item.init:
            continue
        if item.name not in mapping:
            continue
        annotation = hints.get(item.name, item.type)
        if isinstance(annotation, InitVar):
            annotation = annotation.type
        result[item.name] = _coerce_value(mapping[item.name], annotation, f"{path}.{item.name}")
    return result


def _coerce_value(value: Any, annotation: Any, path: str) -> Any:
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{path} is not a valid {annotation.__name__}") from exc
    nested = _nested_dataclass_type(annotation)
    if nested is not None:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path} must be a mapping")
        return nested(**_coerce_dataclass_mapping(value, nested, path=path))
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (tuple,):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path} must be a list or tuple")
        element_type = args[0] if args and args[-1] is not Ellipsis else (args[0] if args else Any)
        if len(args) > 1 and args[-1] is not Ellipsis:
            if len(value) != len(args):
                raise ConfigError(f"{path} must contain {len(args)} values")
            return tuple(_coerce_value(item, item_type, f"{path}[{index}]") for index, (item, item_type) in enumerate(zip(value, args)))
        return tuple(_coerce_value(item, element_type, f"{path}[{index}]") for index, item in enumerate(value))
    if origin in (list, Sequence):
        if not isinstance(value, list):
            raise ConfigError(f"{path} must be a list")
        element_type = args[0] if args else Any
        return [_coerce_value(item, element_type, f"{path}[{index}]") for index, item in enumerate(value)]
    if origin in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path} must be a mapping")
        key_type, value_type = args if len(args) == 2 else (Any, Any)
        return {
            _coerce_value(key, key_type, f"{path}.<key>"): _coerce_value(item, value_type, f"{path}.{key}")
            for key, item in value.items()
        }
    if annotation is Any or annotation is None:
        return value
    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path} must be a boolean")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path} must be an integer")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path} must be numeric")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path} must be a string")
        return value
    return value


def _nested_dataclass_type(annotation: Any) -> type[Any] | None:
    if isinstance(annotation, type) and is_dataclass(annotation):
        return annotation
    return None


def item_type(cls: type[Any], name: str) -> Any:
    return next(item.type for item in fields(cls) if item.name == name)


def _normalize_overrides(overrides: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in overrides.items():
        if value is None:
            continue
        if not isinstance(key, str) or not key:
            raise ConfigError(f"{source} override keys must be non-empty strings")
        result[key] = value
    return result


def _dotted_top_level_keys(mapping: Mapping[str, Any], *, prefix: str) -> set[str]:
    return {key.split(".", 1)[1] for key in mapping if key.startswith(prefix + ".")}


def _apply_dotted_overrides(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        if len(parts) == 1:
            if parts[0] not in result:
                raise ConfigError(f"unknown override field {dotted_key}")
            result[parts[0]] = value
            continue
        cursor: Any = result
        for part in parts[:-1]:
            if not isinstance(cursor, Mapping) or part not in cursor:
                raise ConfigError(f"unknown override field {dotted_key}")
            cursor = cursor[part]
        if not isinstance(cursor, dict) or parts[-1] not in cursor:
            raise ConfigError(f"unknown override field {dotted_key}")
        cursor[parts[-1]] = value
    return result


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_mapping_file(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ConfigError(f"config file is empty: {path}")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError:
            value = _parse_minimal_yaml(text)
        else:
            value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise ConfigError(f"config root must be a mapping: {path}")
    return value


def _parse_minimal_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset needed for project config files.

    PyYAML remains preferred when available.  This fallback handles nested
    mappings, lists, quoted strings, numeric values, booleans, and nulls.
    """

    raw_lines = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        content = line.split(" #", 1)[0].rstrip()
        indent = len(content) - len(content.lstrip(" "))
        raw_lines.append((indent, content.strip()))
    if not raw_lines:
        return {}

    root: Any = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    for index, (indent, content) in enumerate(raw_lines):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1] if stack else root
        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ConfigError("minimal YAML parser expected a list parent")
            parent.append(_parse_scalar(content[2:].strip()))
            continue
        if ":" not in content:
            raise ConfigError(f"unsupported YAML line: {content}")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not isinstance(parent, dict):
            raise ConfigError(f"minimal YAML parser expected a mapping parent for {key!r}")
        if raw_value:
            parent[key] = _parse_scalar(raw_value)
        else:
            next_content = raw_lines[index + 1][1] if index + 1 < len(raw_lines) else ""
            child: Any = [] if next_content.startswith("- ") else {}
            parent[key] = child
            stack.append((indent, child))
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError) as exc:
                raise ConfigError(f"invalid inline YAML value: {value}") from exc
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+e[-+]?\d+|\d+\.\d*e[-+]?\d+)", value, re.I):
            return float(value)
    except ValueError:
        pass
    return value


def _git_metadata() -> tuple[str, str, bool]:
    def git(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
                cwd=Path(__file__).resolve().parents[1],
            )
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
        return completed.stdout.strip() or "unknown"

    branch = git("branch", "--show-current")
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return branch, commit, status not in {"", "unknown"}


def _jsonable(value: Any, *, exclude: set[str] | None = None) -> Any:
    exclude = exclude or set()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item, exclude=exclude) for key, item in value.items() if str(key) not in exclude}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, exclude=exclude) for item in value]
    if is_dataclass(value):
        return _jsonable(asdict(value), exclude=exclude)
    return value


def canonicalize_config_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the one canonical identity mapping for configs and snapshots.

    Snapshot provenance and effective historical defaults remain available in
    the raw snapshot. Only the identity copy is normalized, so an omitted
    historical default and the same explicitly recorded default hash equally,
    while every non-default value remains identity-bearing.
    """

    if not isinstance(config, Mapping):
        raise TypeError("config identity source must be a mapping")
    canonical = _jsonable(config)
    canonical.pop("_metadata", None)
    for name in (
        "source_config_path",
        "cli_overrides",
        "interactive_overrides",
        "git_branch",
        "git_commit",
        "git_dirty",
    ):
        canonical.pop(name, None)

    if canonical.get("reward") == _jsonable(asdict(RewardCreditConfig())):
        # The default legacy reward-credit contract predates this section.
        canonical.pop("reward")

    environment = canonical.get("environment")
    if isinstance(environment, dict):
        for name in ("channel_stochasticity_disabled", "cochannel_interference_disabled", "mobility_static"):
            if environment.get(name) is False:
                environment.pop(name)
        if environment.get("workload_reference_arrival_probabilities") == []:
            environment.pop("workload_reference_arrival_probabilities", None)
        if environment.get("channel_ablation_mode") == ChannelAblationMode.FULL.value:
            # Explicit ``full`` and its historical omission are one identity.
            environment.pop("channel_ablation_mode")
        if (
            environment.get("workload_timing_mode")
            == WorkloadTimingMode.LEGACY_POST_ROUTE.value
        ):
            environment.pop("workload_timing_mode")

    training = canonical.get("training")
    mappo = training.get("mappo") if isinstance(training, dict) else None
    if isinstance(mappo, dict):
        if (
            mappo.get("policy_loss_aggregation_mode")
            == PolicyLossAggregationMode.ACTIVE_BRANCH_MEAN_V1.value
        ):
            mappo.pop("policy_loss_aggregation_mode")
        if mappo.get("actor_ratio_mode") == ActorRatioMode.JOINT.value:
            mappo.pop("actor_ratio_mode")
        if mappo.get("agent_credit_mode") == AgentCreditMode.TEAM.value:
            mappo.pop("agent_credit_mode")
        if mappo.get("route_decoder_mode") == RouteDecoderMode.LEGACY.value:
            mappo.pop("route_decoder_mode")
        if mappo.get("route_feature_mode", "original_option_features_v1") == "original_option_features_v1":
            mappo.pop("route_feature_mode", None)
        if mappo.get("tx_decoder_mode") == TxDecoderMode.LEGACY.value:
            mappo.pop("tx_decoder_mode")
        if mappo.get("cpu_frequency_decoder_mode") == CpuFrequencyDecoderMode.LEGACY.value:
            mappo.pop("cpu_frequency_decoder_mode")
        if mappo.get("tx_upper_decoder_mode") == "legacy":
            mappo.pop("tx_upper_decoder_mode")
        if mappo.get("cpu_scheduler_mode") == "legacy_hierarchical":
            mappo.pop("cpu_scheduler_mode")
        if mappo.get("cpu_anti_starvation_mode") == "off":
            mappo.pop("cpu_anti_starvation_mode")
        if mappo.get("route_exploration_mode", "off") == "off":
            for key in ("route_exploration_mode", "route_exploration_initial_coefficient",
                        "route_exploration_anneal_steps"):
                mappo.pop(key, None)
        if mappo.get("route_training_telemetry_enabled") is False:
            mappo.pop("route_training_telemetry_enabled")
        if mappo.get("route_credit_mode") != RouteCreditMode.TASK_EVENT_VALUE_V1.value:
            mappo.pop("route_task_credit_coefficient", None)
            mappo.pop("route_task_value_coefficient", None)
        if mappo.get("route_value_mode", "off") == "off":
            for key in ("route_value_mode", "route_counterfactual_credit_coefficient",
                        "route_q_value_loss_coefficient"):
                mappo.pop(key, None)
        if mappo.get("route_q_input_mode", "legacy") == "legacy":
            # Preserve canonical hashes of every E13/E14a/b/c/d checkpoint.
            mappo.pop("route_q_input_mode", None)
        if mappo.get("route_actor_credit_source", "counterfactual_q") == "counterfactual_q":
            # Inactive/default selector must not change any historical identity.
            mappo.pop("route_actor_credit_source", None)
        if mappo.get("destination_decoder_mode", "legacy") == "legacy":
            mappo.pop("destination_decoder_mode", None)
        if mappo.get("destination_oracle_rank_coefficient", 0.0) == 0.0:
            mappo.pop("destination_oracle_rank_coefficient", None)
        if mappo.get("destination_supervision_mode", "legacy") == "legacy":
            mappo.pop("destination_supervision_mode", None)
        if mappo.get("tx_upper_optimization_mode") == "ppo_standard":
            mappo.pop("tx_upper_optimization_mode")
        if mappo.get("tx_upper_credit_mode") == "shared_agent_advantage":
            mappo.pop("tx_upper_credit_mode")
        if mappo.get("tx_upper_credit_mode") != "paired_shadow_v1":
            # Inactive E13 fields must not alter any legacy checkpoint identity.
            for key in tuple(mappo):
                if key.startswith("tx_shadow_"):
                    mappo.pop(key)
        if mappo.get("tx_q_target_normalization") == "none":
            # Disabled E11 preserves historical config/checkpoint identities.
            mappo.pop("tx_q_target_normalization")
        if mappo.get("route_private_backbone") is False:
            # Disabled E7 preserves every pre-existing canonical identity.
            mappo.pop("route_private_backbone")
        if mappo.get("route_head_lr_scale") == 1.0:
            # Historical checkpoints/config identities predate the LR split.
            mappo.pop("route_head_lr_scale")
        if mappo.get("route_credit_mode") == RouteCreditMode.SHARED_GAE.value:
            mappo.pop("route_credit_mode")
            mappo.pop("route_gae_lambda", None)
        elif mappo.get("route_credit_mode") == RouteCreditMode.TASK_EVENT_VALUE_V1.value:
            mappo.pop("route_gae_lambda", None)
        elif (
            mappo.get("route_credit_mode")
            == RouteCreditMode.ROLLOUT_CAPPED_ROUTE_EVENT_NSTEP.value
        ):
            mappo.pop("route_gae_lambda", None)
        if not mappo.get("entropy_coefficient_schedule_enabled", False):
            mappo.pop("route_entropy_start_coefficient", None)
            mappo.pop("route_entropy_schedule_start_step", None)
            mappo.pop("route_entropy_schedule_end_step", None)
        if (
            mappo.get("route_only_entropy_initial_coefficient", 0.0) == 0.0
            and mappo.get("route_only_entropy_final_coefficient", 0.0) == 0.0
            and mappo.get("route_only_entropy_anneal_steps", 0) == 0
        ):
            mappo.pop("route_only_entropy_initial_coefficient", None)
            mappo.pop("route_only_entropy_final_coefficient", None)
            mappo.pop("route_only_entropy_anneal_steps", None)
        if not mappo.get("trajectory_credit_telemetry_enabled", False):
            mappo.pop("trajectory_credit_telemetry_enabled", None)
        if not mappo.get("conditional_credit_telemetry_enabled", False):
            mappo.pop("conditional_credit_telemetry_enabled", None)
            mappo.pop("conditional_credit_telemetry_max_bytes", None)
        if not mappo.get("route_diagnostic_samples_enabled", False):
            mappo.pop("route_diagnostic_samples_enabled", None)
        if not mappo.get("route_oracle_counterfactual_enabled", False):
            mappo.pop("route_oracle_counterfactual_enabled", None)
            mappo.pop("route_oracle_selection_rate_ppm", None)
    return canonical


def canonical_config_identity_hash(config: Mapping[str, Any]) -> str:
    """Hash a configuration only after applying canonical identity rules."""

    canonical = canonicalize_config_identity(config)
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "ActionConfig",
    "AgentCreditMode",
    "ActorRatioMode",
    "PolicyLossAggregationMode",
    "ChannelAblationMode",
    "CHECKPOINT_KIND_FINAL_COMPLETED",
    "CHECKPOINT_KIND_PERIODIC_RESUME",
    "CHECKPOINT_SCHEMA_VERSION",
    "CHECKPOINT_V1_ACTIVE_ROLLOUT_FIELDS",
    "CHECKPOINT_V1_ALLOWED_PAYLOAD_VALUE_KINDS",
    "CHECKPOINT_V1_DIAGNOSTICS_STATE_FIELDS",
    "CHECKPOINT_V1_EXACTNESS",
    "CHECKPOINT_V1_KINDS",
    "CHECKPOINT_V1_MODEL_STATE_FIELDS",
    "CHECKPOINT_V1_OPTIMIZER_STATE_FIELDS",
    "CHECKPOINT_V1_POLICY_RNG_STATE_FIELDS",
    "CHECKPOINT_V1_RESUME_BOUNDARY",
    "CHECKPOINT_V1_RUNTIME_PROVENANCE_FIELDS",
    "CHECKPOINT_V1_SAFE_BOUNDARY_ORDER",
    "CHECKPOINT_V1_TOP_LEVEL_FIELDS",
    "CHECKPOINT_V1_TRAINING_STATE_FIELDS",
    "CHECKPOINT_V1_TRANSITION_FIELDS",
    "ConfigError",
    "CreditAssignmentMode",
    "DEFAULT_SEED",
    "EnvironmentConfig",
    "EVALUATION_SNAPSHOT_V1_EXCLUDED_FIELDS",
    "EVALUATION_SNAPSHOT_V1_REQUIRED_FIELDS",
    "EvaluationConfig",
    "MAPPOConfig",
    "MAPPOTrainingAccounting",
    "MAPPO_INITIAL_POLICY_VERSION",
    "OutputConfig",
    "QMIXConfig",
    "RewardCreditConfig",
    "RouteDecoderMode",
    "TxDecoderMode",
    "CpuFrequencyDecoderMode",
    "RouteCreditMode",
    "RouteEntropySchedulePoint",
    "RunConfig",
    "SUPPORTED_METHODS",
    "SUPPORTED_LAUNCH_PROFILES",
    "SUPPORTED_MODES",
    "SUPPORTED_SCENARIOS",
    "TrainingConfig",
    "WorkloadTimingMode",
    "canonical_config_identity_hash",
    "canonicalize_config_identity",
    "compute_effective_route_entropy_schedule",
    "compute_mappo_checkpoint_active_rollout_length",
    "compute_mappo_periodic_checkpoint_steps",
    "compute_route_entropy_schedule",
    "compute_route_only_entropy_schedule",
    "mappo_checkpoint_kind_at",
    "mappo_final_checkpoint_path",
    "mappo_periodic_checkpoint_path",
    "validate_mappo_checkpoint_contract",
    "validate_mappo_checkpoint_resume_compatibility",
    "compute_mappo_training_accounting",
    "require_formal_rl_enabled",
    "validate_mappo_training_device",
    "load_run_config",
    "write_config_snapshot",
]
