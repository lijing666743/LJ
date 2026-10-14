"""External Gauss--Markov UAV mobility with specular rectangle reflection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..config import BoundsConfig, EnvironmentConfig
from .channel import BuildingPrism
from .randomness import rng_from_run_config

if TYPE_CHECKING:
    from ..config import RunConfig


DEFAULT_RESET_PLACEMENT_ATTEMPTS = 10_000


class MobilityError(ValueError):
    """Raised when a mobility state or reset placement is invalid."""


def _readonly_float64(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MobilityState:
    """One slot's fixed-height three-dimensional UAV position and velocity."""

    slot: int
    positions_m: np.ndarray
    velocities_mps: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.slot, bool) or not isinstance(self.slot, int) or self.slot < 0:
            raise MobilityError("mobility slot must be a non-negative integer")
        positions = np.asarray(self.positions_m)
        velocities = np.asarray(self.velocities_mps)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise MobilityError("positions_m must have shape (uav_count, 3)")
        if velocities.shape != positions.shape:
            raise MobilityError("velocities_mps must have the same shape as positions_m")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise MobilityError("mobility state must contain only finite values")
        object.__setattr__(self, "positions_m", _readonly_float64(positions))
        object.__setattr__(self, "velocities_mps", _readonly_float64(velocities))


def reflect_coordinate(
    candidate_position: float,
    candidate_velocity: float,
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, float]:
    """Reflect one coordinate repeatedly until it lies in the closed interval.

    The reflection count, rather than clipping, determines the final velocity
    sign.  This preserves the Section 2 semantics even when one update crosses
    multiple boundaries.
    """

    values = (candidate_position, candidate_velocity, lower_bound, upper_bound)
    if not all(math.isfinite(float(value)) for value in values):
        raise MobilityError("reflection inputs must be finite")
    if upper_bound <= lower_bound:
        raise MobilityError("reflection upper_bound must exceed lower_bound")
    if lower_bound <= candidate_position <= upper_bound:
        return float(candidate_position), float(candidate_velocity)

    span = upper_bound - lower_bound
    if candidate_position > upper_bound:
        reflection_count = math.ceil((candidate_position - upper_bound) / span)
    else:
        reflection_count = math.ceil((lower_bound - candidate_position) / span)

    wrapped = (candidate_position - lower_bound) % (2.0 * span)
    if wrapped <= span:
        reflected_position = lower_bound + wrapped
    else:
        reflected_position = upper_bound - (wrapped - span)
    reflected_velocity = -candidate_velocity if reflection_count % 2 else candidate_velocity
    return float(reflected_position), float(reflected_velocity)


def coordinate_wise_specular_reflection(
    candidate_positions_xy: np.ndarray,
    candidate_velocities_xy: np.ndarray,
    bounds: BoundsConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply independent x/y mirror reflection to all UAVs."""

    positions = np.asarray(candidate_positions_xy, dtype=np.float64)
    velocities = np.asarray(candidate_velocities_xy, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or velocities.shape != positions.shape:
        raise MobilityError("horizontal positions and velocities must both have shape (uav_count, 2)")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise MobilityError("reflection arrays must contain only finite values")

    reflected_positions = positions.copy()
    reflected_velocities = velocities.copy()
    coordinate_bounds = (
        (float(bounds.x_min_m), float(bounds.x_max_m)),
        (float(bounds.y_min_m), float(bounds.y_max_m)),
    )
    for uav_id in range(positions.shape[0]):
        for coordinate, (lower, upper) in enumerate(coordinate_bounds):
            position, velocity = reflect_coordinate(
                positions[uav_id, coordinate],
                velocities[uav_id, coordinate],
                lower,
                upper,
            )
            reflected_positions[uav_id, coordinate] = position
            reflected_velocities[uav_id, coordinate] = velocity
    return reflected_positions, reflected_velocities


class MobilityModel:
    """Own reset placement and external Gauss--Markov mobility state."""

    def __init__(
        self,
        config: EnvironmentConfig,
        rng: np.random.Generator,
        *,
        max_placement_attempts_per_uav: int = DEFAULT_RESET_PLACEMENT_ATTEMPTS,
    ) -> None:
        self.config = config
        self.rng = rng
        if not isinstance(rng, np.random.Generator):
            raise TypeError("mobility rng must be an explicit numpy.random.Generator")
        if (
            isinstance(max_placement_attempts_per_uav, bool)
            or not isinstance(max_placement_attempts_per_uav, int)
            or max_placement_attempts_per_uav <= 0
        ):
            raise MobilityError("max_placement_attempts_per_uav must be a positive integer")
        self.max_placement_attempts_per_uav = max_placement_attempts_per_uav
        self._validate_config()
        self.buildings = tuple(BuildingPrism.from_config(item) for item in config.building_layout)
        self._state: MobilityState | None = None
        self._trajectory: tuple[MobilityState, ...] = ()

    @classmethod
    def from_run_config(
        cls,
        config: "RunConfig",
        *,
        max_placement_attempts_per_uav: int = DEFAULT_RESET_PLACEMENT_ATTEMPTS,
    ) -> "MobilityModel":
        return cls(
            config.environment,
            rng_from_run_config(config, "reset_mobility"),
            max_placement_attempts_per_uav=max_placement_attempts_per_uav,
        )

    @property
    def state(self) -> MobilityState:
        if self._state is None:
            raise MobilityError("mobility model must be reset before its state is read")
        return self._state

    def reset(self) -> MobilityState:
        """Generate and cache one complete building-valid mobility trajectory."""

        self._state = None
        self._trajectory = ()
        for _ in range(DEFAULT_RESET_PLACEMENT_ATTEMPTS):
            trajectory = self._generate_candidate_trajectory()
            if self._trajectory_is_building_free(trajectory):
                if self.config.mobility_static:
                    # Accept the same FULL trajectory first, including reset
                    # rejection and all RNG draws; then hold its initial place.
                    initial_positions = trajectory[0].positions_m
                    trajectory = tuple(
                        MobilityState(item.slot, initial_positions, np.zeros_like(item.velocities_mps))
                        for item in trajectory
                    )
                self._trajectory = trajectory
                self._state = trajectory[0]
                return self._state
        raise MobilityError(
            "unable to sample a building-valid complete mobility trajectory after "
            f"{DEFAULT_RESET_PLACEMENT_ATTEMPTS} attempts; the configured building "
            "layout may leave no feasible discrete-slot trajectory"
        )

    def _sample_initial_state(self) -> MobilityState:
        """Sample the unchanged safe reset placement and initial velocity."""

        cfg = self.config
        bounds = cfg.bounds
        horizontal = np.empty((cfg.uav_count, 2), dtype=np.float64)
        for uav_id in range(cfg.uav_count):
            placed = False
            for _ in range(self.max_placement_attempts_per_uav):
                candidate = self.rng.uniform(
                    low=(bounds.x_min_m, bounds.y_min_m),
                    high=(bounds.x_max_m, bounds.y_max_m),
                )
                if uav_id == 0:
                    placed = True
                else:
                    distances = np.linalg.norm(horizontal[:uav_id] - candidate, axis=1)
                    placed = bool(np.all(distances >= cfg.reset_safe_distance_m))
                if placed:
                    horizontal[uav_id] = candidate
                    break
            if not placed:
                raise MobilityError(
                    "unable to place UAVs with reset_safe_distance_m="
                    f"{cfg.reset_safe_distance_m} after "
                    f"{self.max_placement_attempts_per_uav} attempts for UAV {uav_id}; "
                    "the configured reset placement may be infeasible"
                )

        mean = np.asarray(cfg.velocity_mean_mps, dtype=np.float64)
        std = np.asarray(cfg.velocity_std_mps, dtype=np.float64)
        horizontal_velocity = self.rng.normal(loc=mean, scale=std, size=(cfg.uav_count, 2))
        positions = np.column_stack((horizontal, np.full(cfg.uav_count, cfg.height_m)))
        velocities = np.column_stack((horizontal_velocity, np.zeros(cfg.uav_count)))
        return MobilityState(slot=0, positions_m=positions, velocities_mps=velocities)

    def step(self, state: MobilityState | None = None) -> MobilityState:
        """Replay the next state from the accepted complete trajectory."""

        current = self.state if state is None else state
        if (
            current.slot != self.state.slot
            or not np.array_equal(current.positions_m, self.state.positions_m)
            or not np.array_equal(current.velocities_mps, self.state.velocities_mps)
        ):
            raise MobilityError("provided mobility state does not match the cached current state")
        next_slot = current.slot + 1
        if next_slot >= len(self._trajectory):
            raise MobilityError("accepted mobility trajectory is exhausted")
        self._state = self._trajectory[next_slot]
        return self._state

    def _generate_candidate_trajectory(self) -> tuple[MobilityState, ...]:
        """Consume all mobility draws for one complete candidate trajectory."""

        trajectory = [self._sample_initial_state()]
        for _ in range(1, self.config.episode_horizon):
            trajectory.append(self._advance_candidate(trajectory[-1]))
        return tuple(trajectory)

    def _advance_candidate(self, current: MobilityState) -> MobilityState:
        """Apply the unchanged Gauss--Markov update and outer reflection once."""

        cfg = self.config
        mean = np.asarray(cfg.velocity_mean_mps, dtype=np.float64)
        std = np.asarray(cfg.velocity_std_mps, dtype=np.float64)
        innovation = self.rng.normal(loc=0.0, scale=std, size=(cfg.uav_count, 2))
        alpha = cfg.gauss_markov_alpha
        candidate_velocity = (
            alpha * current.velocities_mps[:, :2]
            + (1.0 - alpha) * mean
            + math.sqrt(1.0 - alpha * alpha) * innovation
        )
        candidate_position = current.positions_m[:, :2] + cfg.slot_duration_s * candidate_velocity
        horizontal, horizontal_velocity = coordinate_wise_specular_reflection(
            candidate_position,
            candidate_velocity,
            cfg.bounds,
        )
        positions = np.column_stack((horizontal, np.full(cfg.uav_count, cfg.height_m)))
        velocities = np.column_stack((horizontal_velocity, np.zeros(cfg.uav_count)))
        return MobilityState(
            slot=current.slot + 1,
            positions_m=positions,
            velocities_mps=velocities,
        )

    def _trajectory_is_building_free(self, trajectory: tuple[MobilityState, ...]) -> bool:
        """Validate every discrete UAV position after the candidate is complete."""

        valid = True
        for state in trajectory:
            for position in state.positions_m:
                for building in self.buildings:
                    if building.contains_point(position):
                        valid = False
        return valid

    def _validate_config(self) -> None:
        cfg = self.config
        bounds = cfg.bounds
        numbers = (
            bounds.x_min_m,
            bounds.x_max_m,
            bounds.y_min_m,
            bounds.y_max_m,
            cfg.height_m,
            cfg.slot_duration_s,
            cfg.reset_safe_distance_m,
            cfg.gauss_markov_alpha,
        )
        if not all(math.isfinite(float(value)) for value in numbers):
            raise MobilityError("mobility configuration must contain finite values")
        if bounds.x_max_m <= bounds.x_min_m or bounds.y_max_m <= bounds.y_min_m:
            raise MobilityError("mobility bounds must have positive width and height")
        if cfg.uav_count <= 0:
            raise MobilityError("uav_count must be positive")
        if (
            isinstance(cfg.episode_horizon, bool)
            or not isinstance(cfg.episode_horizon, int)
            or cfg.episode_horizon <= 0
        ):
            raise MobilityError("episode_horizon must be a positive integer")
        if cfg.height_m <= 0 or cfg.slot_duration_s <= 0:
            raise MobilityError("height_m and slot_duration_s must be positive")
        if cfg.reset_safe_distance_m <= 0:
            raise MobilityError("reset_safe_distance_m must be positive")
        if not 0.0 <= cfg.gauss_markov_alpha <= 1.0:
            raise MobilityError("gauss_markov_alpha must be in [0, 1]")
        if cfg.boundary_rule != "coordinate-wise specular reflection":
            raise MobilityError("only coordinate-wise specular reflection is supported")
        if len(cfg.velocity_mean_mps) != 2 or len(cfg.velocity_std_mps) != 2:
            raise MobilityError("velocity mean/std must each contain x and y values")
        if not np.all(np.isfinite(cfg.velocity_mean_mps)) or not np.all(np.isfinite(cfg.velocity_std_mps)):
            raise MobilityError("velocity mean/std must be finite")
        if np.any(np.asarray(cfg.velocity_std_mps) < 0):
            raise MobilityError("velocity innovation standard deviations must be non-negative")

        diagonal = math.hypot(
            bounds.x_max_m - bounds.x_min_m,
            bounds.y_max_m - bounds.y_min_m,
        )
        if cfg.uav_count > 1 and cfg.reset_safe_distance_m > diagonal:
            raise MobilityError(
                "reset_safe_distance_m exceeds the region diagonal, so multi-UAV placement is impossible"
            )


__all__ = [
    "DEFAULT_RESET_PLACEMENT_ATTEMPTS",
    "MobilityError",
    "MobilityModel",
    "MobilityState",
    "coordinate_wise_specular_reflection",
    "reflect_coordinate",
]
