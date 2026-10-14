"""Actor-visible stale CSI and causal historical-interference state."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

from ..config import ChannelAblationMode, EnvironmentConfig
from .channel import ChannelSnapshot, db_to_amplitude_ratio, receiver_noise_power_w
from .randomness import rng_from_run_config

if TYPE_CHECKING:
    from ..config import RunConfig


class HistoryError(ValueError):
    """Raised when channel/history state violates shape, time, or unit rules."""


def _readonly(values: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class InterferenceSnapshot:
    """Receiver/RU interference-only history visible at one slot start."""

    slot: int
    interference_w: np.ndarray
    valid_mask: np.ndarray
    message_aoi_slots: np.ndarray
    message_aoi_valid_mask: np.ndarray

    def __post_init__(self) -> None:
        interference = np.asarray(self.interference_w)
        valid = np.asarray(self.valid_mask)
        if interference.ndim != 2 or valid.shape != interference.shape:
            raise HistoryError("interference values and masks must have shape (uav_count, ru_count)")
        if not np.all(np.isfinite(interference)) or np.any(interference < 0):
            raise HistoryError("interference history must be finite and non-negative")
        if np.asarray(self.message_aoi_slots).shape != (interference.shape[0],):
            raise HistoryError("message AoI must have shape (uav_count,)")
        if np.asarray(self.message_aoi_valid_mask).shape != (interference.shape[0],):
            raise HistoryError("message AoI validity must have shape (uav_count,)")
        object.__setattr__(self, "interference_w", _readonly(interference, np.float64))
        object.__setattr__(self, "valid_mask", _readonly(valid, np.bool_))
        object.__setattr__(self, "message_aoi_slots", _readonly(self.message_aoi_slots, np.int64))
        object.__setattr__(
            self,
            "message_aoi_valid_mask",
            _readonly(self.message_aoi_valid_mask, np.bool_),
        )


class InterferenceHistory:
    """Causal EMA state updated from a completed slot into the next slot."""

    def __init__(
        self,
        uav_count: int,
        ru_count: int,
        beta: float,
        initial_interference_w: float,
    ) -> None:
        if isinstance(uav_count, bool) or not isinstance(uav_count, int) or uav_count <= 0:
            raise HistoryError("uav_count must be a positive integer")
        if isinstance(ru_count, bool) or not isinstance(ru_count, int) or ru_count <= 0:
            raise HistoryError("ru_count must be a positive integer")
        if not math.isfinite(float(beta)) or not 0.0 <= beta < 1.0:
            raise HistoryError("interference EMA beta must be in [0, 1)")
        if not math.isfinite(float(initial_interference_w)) or initial_interference_w < 0:
            raise HistoryError("initial interference must be finite and non-negative")
        self.uav_count = uav_count
        self.ru_count = ru_count
        self.beta = float(beta)
        self.initial_interference_w = float(initial_interference_w)
        self.reset()

    @property
    def current_slot(self) -> int:
        return self._current_slot

    def reset(self) -> None:
        self._current_slot = 0
        self._interference_w = np.full(
            (self.uav_count, self.ru_count),
            self.initial_interference_w,
            dtype=np.float64,
        )
        self._valid_mask = np.zeros((self.uav_count, self.ru_count), dtype=np.bool_)
        self._message_aoi_slots = np.full(self.uav_count, -1, dtype=np.int64)

    def snapshot(self, slot: int | None = None) -> InterferenceSnapshot:
        requested_slot = self._current_slot if slot is None else slot
        if requested_slot != self._current_slot:
            raise HistoryError(
                f"interference history is at slot {self._current_slot}, not slot {requested_slot}"
            )
        return InterferenceSnapshot(
            slot=self._current_slot,
            interference_w=self._interference_w,
            valid_mask=self._valid_mask,
            message_aoi_slots=self._message_aoi_slots,
            message_aoi_valid_mask=self._message_aoi_slots >= 0,
        )

    def update_with_measurement(
        self,
        measurement_slot: int,
        measurement_w: np.ndarray,
        measurement_mask: np.ndarray | None = None,
    ) -> InterferenceSnapshot:
        """Apply valid executed-action measurements to the next-slot history."""

        measurements = np.asarray(measurement_w, dtype=np.float64)
        if measurements.shape != (self.uav_count, self.ru_count):
            raise HistoryError(
                f"measurement_w must have shape ({self.uav_count}, {self.ru_count})"
            )
        mask = (
            np.ones_like(measurements, dtype=np.bool_)
            if measurement_mask is None
            else np.asarray(measurement_mask, dtype=np.bool_)
        )
        if mask.shape != measurements.shape:
            raise HistoryError("measurement_mask must match measurement_w")
        if np.any(~np.isfinite(measurements[mask])) or np.any(measurements[mask] < 0):
            raise HistoryError("valid interference measurements must be finite and non-negative")
        return self._advance(measurement_slot, measurements, mask)

    def advance_without_measurement(self, measurement_slot: int) -> InterferenceSnapshot:
        """Keep the estimate, preserve masks, and increment any valid message AoI."""

        return self._advance(
            measurement_slot,
            np.zeros((self.uav_count, self.ru_count), dtype=np.float64),
            np.zeros((self.uav_count, self.ru_count), dtype=np.bool_),
        )

    def _advance(
        self,
        measurement_slot: int,
        measurements: np.ndarray,
        measurement_mask: np.ndarray,
    ) -> InterferenceSnapshot:
        if measurement_slot != self._current_slot:
            raise HistoryError(
                f"measurement for slot {measurement_slot} cannot update history at slot {self._current_slot}"
            )
        previous_receiver_valid = np.any(self._valid_mask, axis=1)
        next_interference = self._interference_w.copy()
        next_interference[measurement_mask] = (
            self.beta * self._interference_w[measurement_mask]
            + (1.0 - self.beta) * measurements[measurement_mask]
        )
        next_valid = self._valid_mask | measurement_mask
        receiver_refresh = np.any(measurement_mask, axis=1)
        next_aoi = self._message_aoi_slots.copy()
        next_aoi[receiver_refresh] = 1
        increments = ~receiver_refresh & previous_receiver_valid
        next_aoi[increments] += 1
        next_aoi[~receiver_refresh & ~previous_receiver_valid] = -1

        self._interference_w = next_interference
        self._valid_mask = next_valid
        self._message_aoi_slots = next_aoi
        self._current_slot += 1
        return self.snapshot()


@dataclass(frozen=True)
class ActorChannelFeatures:
    """Only the stale/history channel features legal for an actor to read.

    This object deliberately contains neither the current true channel nor the
    proposal-dependent executor scalar historical quality.
    """

    slot: int
    stale_csi: np.ndarray
    csi_valid_mask: np.ndarray
    csi_aoi_slots: np.ndarray
    interference_history_w: np.ndarray
    interference_valid_mask: np.ndarray
    message_aoi_slots: np.ndarray
    message_aoi_valid_mask: np.ndarray
    historical_denominator_w: np.ndarray
    historical_quality: np.ndarray
    quality_valid_mask: np.ndarray

    def __post_init__(self) -> None:
        stale = np.asarray(self.stale_csi)
        if stale.ndim != 3 or stale.shape[0] != stale.shape[1] or not np.iscomplexobj(stale):
            raise HistoryError("stale_csi must be a complex (uav_count, uav_count, ru_count) tensor")
        link_shape = stale.shape[:2]
        receiver_ru_shape = (stale.shape[1], stale.shape[2])
        if np.asarray(self.csi_valid_mask).shape != link_shape:
            raise HistoryError("csi_valid_mask must have shape (uav_count, uav_count)")
        if np.asarray(self.csi_aoi_slots).shape != link_shape:
            raise HistoryError("csi_aoi_slots must have shape (uav_count, uav_count)")
        for name in (
            "interference_history_w",
            "interference_valid_mask",
            "historical_denominator_w",
        ):
            if np.asarray(getattr(self, name)).shape != receiver_ru_shape:
                raise HistoryError(f"{name} must have shape {receiver_ru_shape}")
        if np.asarray(self.historical_quality).shape != stale.shape:
            raise HistoryError("historical_quality must match stale_csi")
        if np.asarray(self.quality_valid_mask).shape != stale.shape:
            raise HistoryError("quality_valid_mask must match stale_csi")
        if not np.all(np.isfinite(stale)) or not np.all(np.isfinite(self.historical_quality)):
            raise HistoryError("actor channel features must be finite")

        dtype_by_field = {
            "stale_csi": np.complex128,
            "csi_valid_mask": np.bool_,
            "csi_aoi_slots": np.int64,
            "interference_history_w": np.float64,
            "interference_valid_mask": np.bool_,
            "message_aoi_slots": np.int64,
            "message_aoi_valid_mask": np.bool_,
            "historical_denominator_w": np.float64,
            "historical_quality": np.float64,
            "quality_valid_mask": np.bool_,
        }
        for item in fields(self):
            if item.name in dtype_by_field:
                object.__setattr__(
                    self,
                    item.name,
                    _readonly(getattr(self, item.name), dtype_by_field[item.name]),
                )


class ChannelHistory:
    """Episode-bounded true-channel storage with an actor-safe history façade."""

    def __init__(
        self,
        config: EnvironmentConfig,
        csi_error_rng: np.random.Generator,
    ) -> None:
        if not isinstance(csi_error_rng, np.random.Generator):
            raise TypeError("CSI error rng must be an explicit numpy.random.Generator")
        if config.uav_count <= 0 or config.ru_count <= 0:
            raise HistoryError("uav_count and ru_count must be positive")
        if config.fixed_csi_aoi_slots < 0:
            raise HistoryError("fixed_csi_aoi_slots must be non-negative")
        if not math.isfinite(config.csi_error_std_db) or config.csi_error_std_db < 0:
            raise HistoryError("csi_error_std_db must be finite and non-negative")
        if not math.isfinite(config.reference_transmit_power_w) or config.reference_transmit_power_w <= 0:
            raise HistoryError("reference_transmit_power_w must be finite and positive")
        self.config = config
        self.csi_error_rng = csi_error_rng
        self._simplified_deterministic = (
            config.channel_ablation_mode
            == ChannelAblationMode.SIMPLIFIED_DETERMINISTIC
        )
        self.noise_power_w = receiver_noise_power_w(config)
        self.interference = InterferenceHistory(
            config.uav_count,
            config.ru_count,
            config.interference_ema_beta,
            config.initial_interference_w,
        )
        self._true_channels: OrderedDict[int, np.ndarray] = OrderedDict()
        self._last_physical_slot = -1
        self._cached_features: ActorChannelFeatures | None = None

    @classmethod
    def from_run_config(cls, config: "RunConfig") -> "ChannelHistory":
        return cls(config.environment, rng_from_run_config(config, "csi_error"))

    def reset(self, csi_error_rng: np.random.Generator | None = None) -> None:
        """Clear all episode history so no prior-episode sample can leak."""

        if csi_error_rng is not None:
            if not isinstance(csi_error_rng, np.random.Generator):
                raise TypeError("CSI error rng must be an explicit numpy.random.Generator")
            self.csi_error_rng = csi_error_rng
        self._true_channels.clear()
        self._last_physical_slot = -1
        self._cached_features = None
        self.interference.reset()

    def record_physical_channel(
        self,
        slot_or_snapshot: int | ChannelSnapshot,
        channel: np.ndarray | None = None,
    ) -> None:
        """Store one true channel tensor without returning it through the actor API."""

        if isinstance(slot_or_snapshot, ChannelSnapshot):
            if channel is not None:
                raise HistoryError("channel must be omitted when a ChannelSnapshot is supplied")
            slot = slot_or_snapshot.slot
            values = slot_or_snapshot.channel
        else:
            slot = slot_or_snapshot
            if channel is None:
                raise HistoryError("channel is required when recording by slot")
            values = channel
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise HistoryError("physical channel slot must be a non-negative integer")
        if slot != self._last_physical_slot + 1:
            raise HistoryError(
                f"physical channel slots must be recorded sequentially; expected {self._last_physical_slot + 1}"
            )
        tensor = np.asarray(values)
        expected_shape = (self.config.uav_count, self.config.uav_count, self.config.ru_count)
        if tensor.shape != expected_shape or not np.iscomplexobj(tensor):
            raise HistoryError(f"true channel must be a complex tensor with shape {expected_shape}")
        if not np.all(np.isfinite(tensor)):
            raise HistoryError("true channel must contain only finite values")
        self._true_channels[slot] = np.array(tensor, dtype=np.complex128, copy=True)
        effective_aoi_slots = 0 if self._simplified_deterministic else self.config.fixed_csi_aoi_slots
        capacity = effective_aoi_slots + 1
        while len(self._true_channels) > capacity:
            self._true_channels.popitem(last=False)
        self._last_physical_slot = slot
        self._cached_features = None

    def actor_features(self, slot: int) -> ActorChannelFeatures:
        """Build stale CSI/history features without exposing the current true channel."""

        if slot != self._last_physical_slot:
            raise HistoryError(
                f"actor features require the current physical channel for slot {slot} to be recorded"
            )
        if slot != self.interference.current_slot:
            raise HistoryError(
                f"interference history is at slot {self.interference.current_slot}, not actor slot {slot}"
            )
        if self._cached_features is not None and self._cached_features.slot == slot:
            return self._cached_features

        count = self.config.uav_count
        ru_count = self.config.ru_count
        effective_aoi_slots = 0 if self._simplified_deterministic else self.config.fixed_csi_aoi_slots
        stale_index = slot - effective_aoi_slots
        stale = np.zeros((count, count, ru_count), dtype=np.complex128)
        csi_valid = np.zeros((count, count), dtype=np.bool_)
        source = self._true_channels.get(stale_index) if stale_index >= 0 else None

        # Full mode samples the complete directed-link/RU error tensor in its
        # historical fixed order even when the stale lookup is unavailable.
        # Simplified mode has zero CSI error and intentionally consumes no RNG.
        error_db = np.zeros((count, count, ru_count), dtype=np.float64)
        if not self._simplified_deterministic:
            for source_id in range(count):
                for destination_id in range(count):
                    if source_id == destination_id:
                        continue
                    for ru in range(ru_count):
                        error_db[source_id, destination_id, ru] = self.csi_error_rng.normal(
                            0.0,
                            self.config.csi_error_std_db,
                        )
        if source is not None:
            csi_valid[:] = True
            np.fill_diagonal(csi_valid, False)
            stale = source * np.asarray(db_to_amplitude_ratio(error_db), dtype=np.float64)
            for uav_id in range(count):
                stale[uav_id, uav_id, :] = 0.0

        interference = self.interference.snapshot(slot)
        denominator = self.noise_power_w + interference.interference_w
        if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0):
            raise HistoryError("historical denominator must be finite and positive")
        quality_valid = (
            csi_valid[:, :, None]
            & interference.valid_mask[None, :, :]
        )
        reference_power_per_ru = self.config.reference_transmit_power_w / ru_count
        quality = np.zeros_like(stale.real, dtype=np.float64)
        raw_quality = (
            reference_power_per_ru
            * np.abs(stale) ** 2
            / denominator[None, :, :]
        )
        quality[quality_valid] = raw_quality[quality_valid]
        csi_aoi = np.full(
            (count, count),
            effective_aoi_slots,
            dtype=np.int64,
        )
        np.fill_diagonal(csi_aoi, 0)
        self._cached_features = ActorChannelFeatures(
            slot=slot,
            stale_csi=stale,
            csi_valid_mask=csi_valid,
            csi_aoi_slots=csi_aoi,
            interference_history_w=interference.interference_w,
            interference_valid_mask=interference.valid_mask,
            message_aoi_slots=interference.message_aoi_slots,
            message_aoi_valid_mask=interference.message_aoi_valid_mask,
            historical_denominator_w=denominator,
            historical_quality=quality,
            quality_valid_mask=quality_valid,
        )
        return self._cached_features

    def update_with_measurement(
        self,
        measurement_slot: int,
        measurement_w: np.ndarray,
        measurement_mask: np.ndarray | None = None,
    ) -> InterferenceSnapshot:
        """Update slot ``t+1`` history from an actual slot ``t`` measurement."""

        snapshot = self.interference.update_with_measurement(
            measurement_slot,
            measurement_w,
            measurement_mask,
        )
        self._cached_features = None
        return snapshot

    def advance_without_measurement(self, measurement_slot: int) -> InterferenceSnapshot:
        snapshot = self.interference.advance_without_measurement(measurement_slot)
        self._cached_features = None
        return snapshot


def executor_historical_quality_mean(
    features: ActorChannelFeatures,
    source_uav: int,
    destination_uav: int,
    proposed_ru_indices_zero_based: np.ndarray | list[int] | tuple[int, ...],
) -> float:
    """Derive the proposal-dependent executor scalar without mutating actor data."""

    count = features.historical_quality.shape[0]
    if not 0 <= source_uav < count or not 0 <= destination_uav < count or source_uav == destination_uav:
        raise HistoryError("executor quality requires a valid directed non-self link")
    indices = np.asarray(proposed_ru_indices_zero_based, dtype=np.int64)
    if indices.ndim != 1:
        raise HistoryError("proposed RU indices must be one-dimensional")
    if np.any(indices < 0) or np.any(indices >= features.historical_quality.shape[2]):
        raise HistoryError("proposed RU index is outside the configured RU range")
    if indices.size == 0:
        return 0.0
    valid = features.quality_valid_mask[source_uav, destination_uav, indices]
    if not np.any(valid):
        return 0.0
    values = features.historical_quality[source_uav, destination_uav, indices]
    return float(np.mean(values[valid]))


__all__ = [
    "ActorChannelFeatures",
    "ChannelHistory",
    "HistoryError",
    "InterferenceHistory",
    "InterferenceSnapshot",
    "executor_historical_quality_mean",
]
