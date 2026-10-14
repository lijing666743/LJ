"""Current true U2U channel state for the modeled physical-layer evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np

from ..config import BuildingConfig, ChannelAblationMode, EnvironmentConfig
from .randomness import rng_from_run_config
from .topology import pairwise_distances

if TYPE_CHECKING:
    from ..config import RunConfig


SPEED_OF_LIGHT_MPS = 299_792_458.0


class ChannelError(ValueError):
    """Raised when physical channel inputs or slot ordering are invalid."""


def db_to_power_ratio(db_value: float | np.ndarray) -> float | np.ndarray:
    """Convert a dB power ratio to a linear power ratio."""

    converted = np.power(10.0, np.asarray(db_value, dtype=np.float64) / 10.0)
    return float(converted) if converted.ndim == 0 else converted


def db_to_amplitude_ratio(db_value: float | np.ndarray) -> float | np.ndarray:
    """Convert a dB power-gain perturbation to its complex-amplitude factor."""

    converted = np.power(10.0, np.asarray(db_value, dtype=np.float64) / 20.0)
    return float(converted) if converted.ndim == 0 else converted


def dbm_to_watts(dbm_value: float | np.ndarray) -> float | np.ndarray:
    """Convert dBm to linear watts."""

    converted = np.power(10.0, (np.asarray(dbm_value, dtype=np.float64) - 30.0) / 10.0)
    return float(converted) if converted.ndim == 0 else converted


def receiver_noise_power_w(config: EnvironmentConfig) -> float:
    """Compute ``N0 * B_RU * F`` with one explicit dB-to-linear chain."""

    if config.total_bandwidth_hz <= 0 or config.ru_count <= 0:
        raise ChannelError("total bandwidth and RU count must be positive")
    noise_psd_w_hz = float(dbm_to_watts(config.noise_psd_dbm_hz))
    noise_factor = float(db_to_power_ratio(config.receiver_noise_figure_db))
    noise = noise_psd_w_hz * config.ru_bandwidth_hz * noise_factor
    if not math.isfinite(noise) or noise <= 0:
        raise ChannelError("receiver noise power must be finite and positive")
    return noise


@dataclass(frozen=True)
class BuildingPrism:
    """Axis-aligned building prism occupying z in ``[0, height_m]``."""

    name: str
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    height_m: float

    @classmethod
    def from_config(cls, config: BuildingConfig) -> "BuildingPrism":
        return cls(
            config.name,
            config.x_min_m,
            config.x_max_m,
            config.y_min_m,
            config.y_max_m,
            config.height_m,
        )

    def __post_init__(self) -> None:
        values = (self.x_min_m, self.x_max_m, self.y_min_m, self.y_max_m, self.height_m)
        if not all(math.isfinite(float(value)) for value in values):
            raise ChannelError(f"building {self.name!r} contains a non-finite value")
        if self.x_max_m <= self.x_min_m or self.y_max_m <= self.y_min_m or self.height_m <= 0:
            raise ChannelError(f"building {self.name!r} must have positive width, depth, and height")

    def contains_point(self, position_m: np.ndarray) -> bool:
        """Return whether a finite 3-D point lies in this closed prism."""

        position = np.asarray(position_m, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ChannelError("building containment point must be a finite 3-D coordinate")
        return bool(
            self.x_min_m <= position[0] <= self.x_max_m
            and self.y_min_m <= position[1] <= self.y_max_m
            and 0.0 <= position[2] <= self.height_m
        )

    def intersects_segment(self, start_m: np.ndarray, end_m: np.ndarray) -> bool:
        """Test intersection between a closed 3-D segment and this closed prism."""

        start = np.asarray(start_m, dtype=np.float64)
        end = np.asarray(end_m, dtype=np.float64)
        if start.shape != (3,) or end.shape != (3,) or not np.all(np.isfinite((start, end))):
            raise ChannelError("building intersection endpoints must be finite 3-D coordinates")
        lower = np.array((self.x_min_m, self.y_min_m, 0.0), dtype=np.float64)
        upper = np.array((self.x_max_m, self.y_max_m, self.height_m), dtype=np.float64)
        direction = end - start
        t_min = 0.0
        t_max = 1.0
        for coordinate in range(3):
            if direction[coordinate] == 0.0:
                if start[coordinate] < lower[coordinate] or start[coordinate] > upper[coordinate]:
                    return False
                continue
            inverse = 1.0 / direction[coordinate]
            near = (lower[coordinate] - start[coordinate]) * inverse
            far = (upper[coordinate] - start[coordinate]) * inverse
            if near > far:
                near, far = far, near
            t_min = max(t_min, near)
            t_max = min(t_max, far)
            if t_min > t_max:
                return False
        return True


def building_blockage_mask(
    positions_m: np.ndarray,
    buildings: Iterable[BuildingPrism],
) -> np.ndarray:
    """Return ``True`` when any building intersects a directed link segment."""

    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.all(np.isfinite(positions)):
        raise ChannelError("positions_m must be a finite array with shape (uav_count, 3)")
    building_list = tuple(buildings)
    count = positions.shape[0]
    blocked = np.zeros((count, count), dtype=np.bool_)
    for source in range(count):
        for destination in range(count):
            if source == destination:
                continue
            blocked[source, destination] = any(
                building.intersects_segment(positions[source], positions[destination])
                for building in building_list
            )
    return blocked


def _readonly(values: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ChannelSnapshot:
    """Environment-internal true channel state for exactly one slot."""

    slot: int
    channel: np.ndarray
    distances_m: np.ndarray
    blocked_links: np.ndarray
    shadowing_db: np.ndarray
    path_loss_db: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.slot, bool) or not isinstance(self.slot, int) or self.slot < 0:
            raise ChannelError("channel slot must be a non-negative integer")
        channel = np.asarray(self.channel)
        if channel.ndim != 3 or channel.shape[0] != channel.shape[1]:
            raise ChannelError("channel must have shape (uav_count, uav_count, ru_count)")
        link_shape = channel.shape[:2]
        for name, values in (
            ("distances_m", self.distances_m),
            ("blocked_links", self.blocked_links),
            ("shadowing_db", self.shadowing_db),
            ("path_loss_db", self.path_loss_db),
        ):
            if np.asarray(values).shape != link_shape:
                raise ChannelError(f"{name} must have shape {link_shape}")
        if not np.iscomplexobj(channel) or not np.all(np.isfinite(channel)):
            raise ChannelError("channel tensor must be finite and complex")
        if not np.all(np.isfinite(self.distances_m)) or not np.all(np.isfinite(self.shadowing_db)):
            raise ChannelError("distance and shadowing tensors must be finite")
        if not np.all(np.isfinite(self.path_loss_db)):
            raise ChannelError("path-loss tensor must be finite")
        object.__setattr__(self, "channel", _readonly(channel, np.complex128))
        object.__setattr__(self, "distances_m", _readonly(self.distances_m, np.float64))
        object.__setattr__(self, "blocked_links", _readonly(self.blocked_links, np.bool_))
        object.__setattr__(self, "shadowing_db", _readonly(self.shadowing_db, np.float64))
        object.__setattr__(self, "path_loss_db", _readonly(self.path_loss_db, np.float64))


class PhysicalChannelModel:
    """Generate directed, per-RU true channels and correlated shadowing."""

    def __init__(self, config: EnvironmentConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self._simplified_deterministic = (
            config.channel_ablation_mode
            == ChannelAblationMode.SIMPLIFIED_DETERMINISTIC
        )
        if not isinstance(rng, np.random.Generator):
            raise TypeError("channel rng must be an explicit numpy.random.Generator")
        self.buildings = tuple(BuildingPrism.from_config(item) for item in config.building_layout)
        self._validate_config()
        self._last_slot = -1
        self._previous_positions_m: np.ndarray | None = None
        self._shadowing_db: np.ndarray | None = None

    @classmethod
    def from_run_config(cls, config: "RunConfig") -> "PhysicalChannelModel":
        return cls(config.environment, rng_from_run_config(config, "channel_fading"))

    def reset(self, positions_m: np.ndarray) -> ChannelSnapshot:
        """Clear episode state and directly generate the slot-0 true channel."""

        self._last_slot = -1
        self._previous_positions_m = None
        self._shadowing_db = None
        return self.generate(slot=0, positions_m=positions_m)

    def generate(self, slot: int, positions_m: np.ndarray) -> ChannelSnapshot:
        """Generate one complete ``(source, destination, RU)`` channel tensor."""

        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise ChannelError("channel slot must be a non-negative integer")
        if slot != self._last_slot + 1:
            raise ChannelError(f"channel slots must be generated sequentially; expected {self._last_slot + 1}")
        positions = np.asarray(positions_m, dtype=np.float64)
        if positions.shape != (self.config.uav_count, 3) or not np.all(np.isfinite(positions)):
            raise ChannelError(
                f"positions_m must be finite with shape ({self.config.uav_count}, 3)"
            )
        distances = pairwise_distances(positions)
        off_diagonal = ~np.eye(self.config.uav_count, dtype=np.bool_)
        if np.any(distances[off_diagonal] <= 0):
            raise ChannelError("every directed non-self link must have strictly positive distance")

        blocked = building_blockage_mask(positions, self.buildings)
        if self._simplified_deterministic:
            shadowing = np.zeros(
                (self.config.uav_count, self.config.uav_count),
                dtype=np.float64,
            )
        elif slot == 0:
            shadowing = self._sample_initial_shadowing()
        else:
            if self._previous_positions_m is None or self._shadowing_db is None:
                raise ChannelError("channel state was not initialized at slot 0")
            shadowing = self._advance_shadowing(positions)

        if self.config.channel_stochasticity_disabled:
            # Keep FULL's draw order/count, but remove random propagation gain.
            shadowing = np.zeros_like(shadowing)

        path_loss = np.zeros((self.config.uav_count, self.config.uav_count), dtype=np.float64)
        channels = np.zeros(
            (self.config.uav_count, self.config.uav_count, self.config.ru_count),
            dtype=np.complex128,
        )
        antenna_power_gain = float(db_to_power_ratio(self.config.antenna_gain_dbi))
        combined_antenna_gain = antenna_power_gain * antenna_power_gain
        rician_k = float(db_to_power_ratio(self.config.los_rician_k_db))
        deterministic_weight = math.sqrt(rician_k / (rician_k + 1.0))
        innovation_weight = math.sqrt(1.0 / (rician_k + 1.0))

        for source in range(self.config.uav_count):
            for destination in range(self.config.uav_count):
                if source == destination:
                    continue
                fspl_db = 20.0 * math.log10(
                    4.0
                    * math.pi
                    * self.config.carrier_frequency_hz
                    * distances[source, destination]
                    / SPEED_OF_LIGHT_MPS
                )
                link_path_loss = (
                    fspl_db
                    + (self.config.building_loss_db if blocked[source, destination] else 0.0)
                    + shadowing[source, destination]
                )
                path_loss[source, destination] = link_path_loss
                amplitude = math.sqrt(combined_antenna_gain * float(db_to_power_ratio(-link_path_loss)))
                if self._simplified_deterministic:
                    channels[source, destination, :] = complex(amplitude, 0.0)
                    continue
                for ru in range(self.config.ru_count):
                    real, imaginary = self.rng.normal(0.0, math.sqrt(0.5), size=2)
                    innovation = complex(real, imaginary)
                    if blocked[source, destination]:
                        small_scale = innovation
                    else:
                        small_scale = deterministic_weight + innovation_weight * innovation
                    if self.config.channel_stochasticity_disabled:
                        small_scale = complex(1.0, 0.0)
                    channels[source, destination, ru] = amplitude * small_scale

        if not np.all(np.isfinite(channels)) or not np.all(np.isfinite(path_loss)):
            raise ChannelError("channel generation produced a non-finite value")
        self._last_slot = slot
        self._previous_positions_m = positions.copy()
        self._shadowing_db = shadowing.copy()
        return ChannelSnapshot(
            slot=slot,
            channel=channels,
            distances_m=distances,
            blocked_links=blocked,
            shadowing_db=shadowing,
            path_loss_db=path_loss,
        )

    def _sample_initial_shadowing(self) -> np.ndarray:
        shadowing = np.zeros((self.config.uav_count, self.config.uav_count), dtype=np.float64)
        for source in range(self.config.uav_count):
            for destination in range(self.config.uav_count):
                if source != destination:
                    shadowing[source, destination] = self.rng.normal(
                        0.0,
                        self.config.shadowing_std_db,
                    )
        return shadowing

    def _advance_shadowing(self, positions_m: np.ndarray) -> np.ndarray:
        assert self._previous_positions_m is not None
        assert self._shadowing_db is not None
        horizontal_displacement = np.linalg.norm(
            positions_m[:, :2] - self._previous_positions_m[:, :2],
            axis=1,
        )
        shadowing = np.zeros_like(self._shadowing_db)
        for source in range(self.config.uav_count):
            for destination in range(self.config.uav_count):
                if source == destination:
                    continue
                average_displacement = 0.5 * (
                    horizontal_displacement[source] + horizontal_displacement[destination]
                )
                rho = math.exp(-average_displacement / self.config.shadowing_correlation_distance_m)
                innovation = self.rng.normal(0.0, 1.0)
                shadowing[source, destination] = (
                    rho * self._shadowing_db[source, destination]
                    + math.sqrt(max(0.0, 1.0 - rho * rho))
                    * self.config.shadowing_std_db
                    * innovation
                )
        return shadowing

    def _validate_config(self) -> None:
        cfg = self.config
        positive = {
            "uav_count": cfg.uav_count,
            "ru_count": cfg.ru_count,
            "carrier_frequency_hz": cfg.carrier_frequency_hz,
            "shadowing_correlation_distance_m": cfg.shadowing_correlation_distance_m,
            "total_bandwidth_hz": cfg.total_bandwidth_hz,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or value <= 0:
                raise ChannelError(f"{name} must be finite and positive")
        for name, value in {
            "antenna_gain_dbi": cfg.antenna_gain_dbi,
            "building_loss_db": cfg.building_loss_db,
            "los_rician_k_db": cfg.los_rician_k_db,
            "shadowing_std_db": cfg.shadowing_std_db,
        }.items():
            if not math.isfinite(float(value)):
                raise ChannelError(f"{name} must be finite")
        if cfg.building_loss_db < 0 or cfg.shadowing_std_db < 0:
            raise ChannelError("building loss and shadowing standard deviation must be non-negative")


ChannelModel = PhysicalChannelModel


__all__ = [
    "BuildingPrism",
    "ChannelError",
    "ChannelModel",
    "ChannelSnapshot",
    "PhysicalChannelModel",
    "SPEED_OF_LIGHT_MPS",
    "building_blockage_mask",
    "db_to_amplitude_ratio",
    "db_to_power_ratio",
    "dbm_to_watts",
    "receiver_noise_power_w",
]
