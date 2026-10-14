"""Unified Route scorer initialization and explicit engineering fixtures.

The probe starts a fresh reference actor, replaces only its Route decoder, and
uses the existing flat categorical sampling/replay implementation. It never
loads weights or changes the reference config's credit or service settings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import RouteDecoderMode, RunConfig, STREAM_IDS
from .ca_gat_mappo import CAGATMAPPOActor, MAPPOTensorSpec, OptionAwareRouteDecoderV1
from .ca_gat_mappo_actions import CAGATMAPPOActionDistribution


UNIFIED_ROUTE_INITIALIZATION_DOMAIN = 17001
PHYSICAL_ROUTE_INITIALIZATION_DOMAIN = 17002


def physical_route_initialization_seed(master_seed: int) -> int:
    """C-only parameters do not consume B/common initialization draws."""
    unified_route_initialization_seed(master_seed)  # Validate the master seed.
    state = np.random.SeedSequence([
        master_seed, STREAM_IDS["torch_initialization"], PHYSICAL_ROUTE_INITIALIZATION_DOMAIN,
    ]).generate_state(1, dtype=np.uint64)
    return int(state[0] % np.uint64(np.iinfo(np.int64).max))


def unified_route_initialization_seed(master_seed: int) -> int:
    """Independent subdomain of stream 100; leave policy stream 110 alone."""
    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
        raise ValueError("master seed must be a non-negative integer")
    state = np.random.SeedSequence([
        master_seed, STREAM_IDS["torch_initialization"],
        UNIFIED_ROUTE_INITIALIZATION_DOMAIN,
    ]).generate_state(1, dtype=np.uint64)
    return int(state[0] % np.uint64(np.iinfo(np.int64).max))


def build_unified_route_decoder(
    spec: MAPPOTensorSpec, master_seed: int, *, config: RunConfig | None = None,
) -> OptionAwareRouteDecoderV1:
    """Reuse the old features and shared scorer in an isolated CPU RNG domain."""
    if config is not None and config.seed != master_seed:
        raise ValueError("unified Route config/master seed mismatch")
    with torch.random.fork_rng(devices=[]):
        # Seed only the CPU generator: constructing CPU parameters must not
        # reseed initialized CUDA generators in the calling process.
        torch.random.default_generator.manual_seed(
            unified_route_initialization_seed(master_seed)
        )
        if config is not None and config.training.mappo.route_feature_mode == "physical_relations_v1":
            from .route_physical_relations import PhysicalOptionAwareRouteDecoderV1
            return PhysicalOptionAwareRouteDecoderV1(spec=spec, config=config)
        return OptionAwareRouteDecoderV1(spec=spec)


@dataclass(frozen=True)
class UnifiedRouteProbe:
    """Forward/sample fixture, explicitly distinct from a runnable experiment.

    baseline_config is the validated reference identity, NOT a B run config.
    No optimizer, checkpoint, rollout or telemetry artifact is provided.
    """
    actor: CAGATMAPPOActor
    distribution: CAGATMAPPOActionDistribution
    baseline_config: RunConfig
    initialization_seed: int


def build_unified_route_probe(config: RunConfig) -> UnifiedRouteProbe:
    """Build a fresh hierarchy reference then replace its head for Step03.

    Consuming the unchanged reference constructor first protects every common
    tensor, including heads initialized after the historical Route head.
    The discarded head is neither registered nor optimized in the probe.
    This reference-config probe remains separate from production run identity.
    """
    config.validate()
    if RouteDecoderMode(config.training.mappo.route_decoder_mode) is not (
        RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1
    ):
        raise ValueError("unified probe requires the hierarchical reference config")
    if config.training.mappo.destination_decoder_mode != "legacy":
        raise ValueError("unified probe requires original features and legacy destination decoder")
    actor = CAGATMAPPOActor(config)
    distribution = CAGATMAPPOActionDistribution(actor, config)
    actor.action_heads["route"] = build_unified_route_decoder(actor.spec, config.seed)
    actor.route_decoder_mode = RouteDecoderMode.UNIFIED_NODE_V1
    distribution.route_decoder_mode = RouteDecoderMode.UNIFIED_NODE_V1
    return UnifiedRouteProbe(
        actor, distribution, config, unified_route_initialization_seed(config.seed),
    )


def unified_route_checkpoint_identity(
    master_seed: int, feature_mode: str = "original_option_features_v1",
) -> dict:
    """Versioned architecture/feature identity, independent of experiment paths."""
    if (not isinstance(feature_mode, str)
            or feature_mode not in {"original_option_features_v1", "physical_relations_v1"}):
        raise ValueError("invalid unified Route feature identity")
    identity = {
        "schema": "unified_route_identity.v1",
        "decoder_mode": "unified_node_v1",
        "decision_mode": "flat_leaf_v1",
        "feature_mode": "original_option_features_v1",
        "q_candidate_mode": "candidate_remote_context_type_411_v1",
        "initialization_domain": UNIFIED_ROUTE_INITIALIZATION_DOMAIN,
        "initialization_seed": unified_route_initialization_seed(master_seed),
    }
    if feature_mode == "physical_relations_v1":
        identity.update(schema="unified_route_identity.v2", feature_mode=feature_mode,
                        q_candidate_mode="physical_candidate_remote_context_type_411_v1",
                        physical_initialization_domain=PHYSICAL_ROUTE_INITIALIZATION_DOMAIN,
                        physical_initialization_seed=physical_route_initialization_seed(master_seed),
                        physical_transform="signed_log1p_config_scales_v1")
    return identity
