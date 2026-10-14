"""Gate 0 Mobility/Topology/Channel-History subset tests for Implementation 03."""

from __future__ import annotations

import dataclasses
import hashlib
import unittest
from dataclasses import replace

import numpy as np

from src.config import (
    BoundsConfig,
    BuildingConfig,
    ChannelAblationMode,
    EnvironmentConfig,
)
from src.env.channel import (
    BuildingPrism,
    ChannelError,
    PhysicalChannelModel,
    building_blockage_mask,
    db_to_amplitude_ratio,
    db_to_power_ratio,
    dbm_to_watts,
    receiver_noise_power_w,
)
from src.env.history import (
    ActorChannelFeatures,
    ChannelHistory,
    HistoryError,
    InterferenceHistory,
    executor_historical_quality_mean,
)
from src.env.mobility import (
    MobilityError,
    MobilityModel,
    MobilityState,
    coordinate_wise_specular_reflection,
)
from src.env.randomness import make_rng
from src.env.topology import DynamicTopology, candidate_neighbor_mask, pairwise_distances


def environment_config(**overrides: object) -> EnvironmentConfig:
    count = int(overrides.get("uav_count", 2))
    base = replace(
        EnvironmentConfig(),
        uav_count=count,
        arrival_probabilities=tuple(0.0 for _ in range(count)),
        profile_assignment=tuple("Balanced" for _ in range(count)),
        building_layout=(),
    )
    return replace(base, **overrides)


def true_channel(config: EnvironmentConfig, value: complex) -> np.ndarray:
    tensor = np.zeros((config.uav_count, config.uav_count, config.ru_count), dtype=np.complex128)
    for source in range(config.uav_count):
        for destination in range(config.uav_count):
            if source != destination:
                tensor[source, destination, :] = value * (1.0 + 0.1 * source + 0.01 * destination)
    return tensor


class RandomStreamTests(unittest.TestCase):
    def test_streams_are_reproducible_and_component_owned(self) -> None:
        first_channel = make_rng(42, 40).normal(size=8)
        second_channel = make_rng(42, 40).normal(size=8)
        csi_error = make_rng(42, 50).normal(size=8)
        np.testing.assert_array_equal(first_channel, second_channel)
        self.assertFalse(np.array_equal(first_channel, csi_error))


class MobilityTests(unittest.TestCase):
    def test_reset_is_reproducible_inside_bounds_and_safe(self) -> None:
        config = environment_config(uav_count=6, reset_safe_distance_m=50.0)
        first = MobilityModel(config, make_rng(42, 10)).reset()
        second = MobilityModel(config, make_rng(42, 10)).reset()
        np.testing.assert_array_equal(first.positions_m, second.positions_m)
        np.testing.assert_array_equal(first.velocities_mps, second.velocities_mps)
        self.assertEqual(first.positions_m.dtype, np.float64)
        self.assertEqual(first.positions_m.shape, (6, 3))
        self.assertTrue(np.all(first.positions_m[:, 0] >= config.bounds.x_min_m))
        self.assertTrue(np.all(first.positions_m[:, 0] <= config.bounds.x_max_m))
        self.assertTrue(np.all(first.positions_m[:, 1] >= config.bounds.y_min_m))
        self.assertTrue(np.all(first.positions_m[:, 1] <= config.bounds.y_max_m))
        self.assertTrue(np.all(first.positions_m[:, 2] == config.height_m))
        distances = pairwise_distances(first.positions_m[:, :])
        safe_values = distances[~np.eye(config.uav_count, dtype=bool)]
        self.assertTrue(np.all(safe_values >= config.reset_safe_distance_m))

    def test_impossible_safe_placement_raises_clear_error(self) -> None:
        config = environment_config(
            uav_count=3,
            bounds=BoundsConfig(0.0, 1.0, 0.0, 1.0),
            reset_safe_distance_m=1.3,
        )
        model = MobilityModel(config, make_rng(42, 10), max_placement_attempts_per_uav=100)
        with self.assertRaisesRegex(MobilityError, "placement may be infeasible"):
            model.reset()

    def test_normal_movement_and_single_coordinate_reflection(self) -> None:
        config = environment_config(
            uav_count=1,
            bounds=BoundsConfig(0.0, 10.0, 0.0, 10.0),
            slot_duration_s=1.0,
            gauss_markov_alpha=1.0,
            velocity_mean_mps=(0.0, 0.0),
            velocity_std_mps=(0.0, 0.0),
            reset_safe_distance_m=1.0,
        )
        state = MobilityState(
            slot=0,
            positions_m=np.array([[9.0, 4.0, config.height_m]]),
            velocities_mps=np.array([[4.0, 2.0, 0.0]]),
        )
        updated = MobilityModel(config, make_rng(42, 10))._advance_candidate(state)
        np.testing.assert_allclose(updated.positions_m[0], [7.0, 6.0, config.height_m])
        np.testing.assert_allclose(updated.velocities_mps[0], [-4.0, 2.0, 0.0])

    def test_complete_trajectory_is_building_free_deterministic_and_replayed(self) -> None:
        config = environment_config(
            uav_count=3,
            episode_horizon=25,
            building_layout=(BuildingConfig("B", 400.0, 600.0, 400.0, 600.0, 120.0),),
        )
        models = (
            MobilityModel(config, make_rng(42, 10)),
            MobilityModel(config, make_rng(42, 10)),
        )
        trajectories = []
        for model in models:
            state = model.reset()
            trajectory = [state]
            rng_state_after_reset = model.rng.bit_generator.state
            for _ in range(1, config.episode_horizon):
                state = model.step(state)
                trajectory.append(state)
            self.assertEqual(model.rng.bit_generator.state, rng_state_after_reset)
            self.assertEqual([item.slot for item in trajectory], list(range(config.episode_horizon)))
            self.assertTrue(
                all(
                    not building.contains_point(position)
                    for item in trajectory
                    for position in item.positions_m
                    for building in model.buildings
                )
            )
            trajectories.append(trajectory)

        for first, second in zip(*trajectories):
            np.testing.assert_array_equal(first.positions_m, second.positions_m)
            np.testing.assert_array_equal(first.velocities_mps, second.velocities_mps)

    def test_reset_rejects_an_invalid_complete_candidate(self) -> None:
        config = environment_config(
            uav_count=1,
            episode_horizon=2,
            height_m=5.0,
            building_layout=(BuildingConfig("B", 0.0, 2.0, 0.0, 2.0, 10.0),),
        )
        invalid = (
            MobilityState(0, np.array([[1.0, 1.0, 5.0]]), np.zeros((1, 3))),
            MobilityState(1, np.array([[8.0, 8.0, 5.0]]), np.zeros((1, 3))),
        )
        valid = (
            MobilityState(0, np.array([[8.0, 8.0, 5.0]]), np.zeros((1, 3))),
            MobilityState(1, np.array([[9.0, 9.0, 5.0]]), np.zeros((1, 3))),
        )

        class ScriptedMobilityModel(MobilityModel):
            def __init__(self) -> None:
                super().__init__(config, make_rng(42, 10))
                self.candidates = [invalid, valid]
                self.generated_candidates = 0

            def _generate_candidate_trajectory(self) -> tuple[MobilityState, ...]:
                candidate = self.candidates[self.generated_candidates]
                self.generated_candidates += 1
                return candidate

        model = ScriptedMobilityModel()
        reset_state = model.reset()
        self.assertEqual(model.generated_candidates, 2)
        np.testing.assert_array_equal(reset_state.positions_m, valid[0].positions_m)
        np.testing.assert_array_equal(model.step(reset_state).positions_m, valid[1].positions_m)

    def test_coordinate_reflection_handles_multi_boundary_overshoot(self) -> None:
        positions, velocities = coordinate_wise_specular_reflection(
            np.array([[35.0, -25.0]]),
            np.array([[7.0, -9.0]]),
            BoundsConfig(0.0, 10.0, 0.0, 10.0),
        )
        np.testing.assert_allclose(positions, [[5.0, 5.0]])
        np.testing.assert_allclose(velocities, [[-7.0, 9.0]])


class TopologyTests(unittest.TestCase):
    def test_pairwise_distance_self_exclusion_and_boundary_equality(self) -> None:
        positions = np.array([[0.0, 0.0, 80.0], [3.0, 4.0, 80.0], [0.0, 0.0, 92.0]])
        distances = pairwise_distances(positions)
        self.assertEqual(distances[0, 1], 5.0)
        self.assertEqual(distances[0, 2], 12.0)
        mask = candidate_neighbor_mask(positions, 5.0)
        self.assertTrue(mask[0, 1])
        self.assertTrue(mask[1, 0])
        self.assertFalse(np.any(np.diag(mask)))

    def test_topology_recomputes_current_positions_without_hysteresis(self) -> None:
        topology = DynamicTopology(5.0)
        near = np.array([[0.0, 0.0, 80.0], [5.0, 0.0, 80.0]])
        far = np.array([[0.0, 0.0, 80.0], [5.01, 0.0, 80.0]])
        near_again = np.array([[0.0, 0.0, 80.0], [4.99, 0.0, 80.0]])
        self.assertTrue(topology.compute(0, near).candidate_neighbors[0, 1])
        self.assertFalse(topology.compute(1, far).candidate_neighbors[0, 1])
        self.assertTrue(topology.compute(2, near_again).candidate_neighbors[0, 1])


class PhysicalChannelTests(unittest.TestCase):
    def test_building_point_containment_uses_closed_finite_height_solid(self) -> None:
        building = BuildingPrism("B", 4.0, 6.0, -1.0, 1.0, 100.0)
        self.assertTrue(building.contains_point(np.array([5.0, 0.0, 80.0])))
        self.assertTrue(building.contains_point(np.array([4.0, 0.0, 80.0])))
        self.assertTrue(building.contains_point(np.array([5.0, 0.0, 100.0])))
        self.assertFalse(building.contains_point(np.array([5.0, 0.0, 100.01])))
        self.assertFalse(building.contains_point(np.array([6.01, 0.0, 80.0])))

    def test_building_intersection_uses_three_dimensional_closed_segment(self) -> None:
        building = BuildingPrism("B", 4.0, 6.0, -1.0, 1.0, 100.0)
        blocked = building_blockage_mask(
            np.array([[0.0, 0.0, 80.0], [10.0, 0.0, 80.0]]),
            [building],
        )
        self.assertTrue(blocked[0, 1])
        above = building_blockage_mask(
            np.array([[0.0, 0.0, 120.0], [10.0, 0.0, 120.0]]),
            [building],
        )
        self.assertFalse(above[0, 1])

    def test_channel_shape_complex_dtype_seed_and_directed_ru_innovation(self) -> None:
        config = environment_config(uav_count=3, ru_count=4, shadowing_std_db=0.0)
        positions = np.array(
            [[0.0, 0.0, 80.0], [100.0, 0.0, 80.0], [0.0, 150.0, 80.0]]
        )
        first = PhysicalChannelModel(config, make_rng(42, 40)).reset(positions)
        second = PhysicalChannelModel(config, make_rng(42, 40)).reset(positions)
        self.assertEqual(first.channel.shape, (3, 3, 4))
        self.assertEqual(first.channel.dtype, np.complex128)
        self.assertTrue(np.all(np.isfinite(first.channel)))
        np.testing.assert_array_equal(first.channel, second.channel)
        self.assertFalse(np.array_equal(first.channel[0, 1], first.channel[1, 0]))
        self.assertGreater(np.unique(first.channel[0, 1]).size, 1)
        np.testing.assert_array_equal(np.diagonal(first.channel, axis1=0, axis2=1), 0.0)

    def test_full_channel_matches_pre_ablation_golden_rng_and_numerics(self) -> None:
        config = EnvironmentConfig()
        positions = np.array(
            [
                [0.0, 0.0, 80.0],
                [100.0, 0.0, 80.0],
                [0.0, 150.0, 80.0],
                [150.0, 150.0, 80.0],
            ]
        )
        rng = make_rng(42, 40)
        model = PhysicalChannelModel(config, rng)
        slot_0 = model.reset(positions)
        slot_1 = model.generate(1, positions)

        digest = lambda values: hashlib.sha256(  # noqa: E731
            np.ascontiguousarray(values).tobytes()
        ).hexdigest()
        self.assertEqual(
            digest(slot_0.channel),
            "9f9e0eddb05233884559b259916a546893512c635a8d9d9379397ee24b4cdbe9",
        )
        self.assertEqual(
            digest(slot_0.shadowing_db),
            "5cc531701fe378461255e395081b2e82f58deae436ed80c29d19ead5e5d7fd1d",
        )
        self.assertEqual(
            digest(slot_0.path_loss_db),
            "4638b996907a2ff4e4363700e992d5265feeb7362aecc3ab32a4cbb8d7ae4faa",
        )
        self.assertEqual(
            digest(slot_1.channel),
            "50b22fe9a0633a1747f3a7577db8bbe366e149a367dba258eeda8c5d20b89061",
        )
        self.assertEqual(
            rng.bit_generator.state["state"]["state"],
            80934111541331465111921140916389443600,
        )

    def test_simplified_channel_is_seed_independent_and_retains_blockage_loss(self) -> None:
        building = BuildingConfig("B", 40.0, 60.0, -10.0, 10.0, 100.0)
        config = environment_config(
            ru_count=3,
            channel_ablation_mode=ChannelAblationMode.SIMPLIFIED_DETERMINISTIC,
            building_layout=(building,),
        )
        positions = np.array([[0.0, 0.0, 80.0], [100.0, 0.0, 80.0]])
        first_rng = make_rng(42, 40)
        second_rng = make_rng(99, 40)
        first_state = first_rng.bit_generator.state
        second_state = second_rng.bit_generator.state

        first = PhysicalChannelModel(config, first_rng).reset(positions)
        second = PhysicalChannelModel(config, second_rng).reset(positions)

        np.testing.assert_array_equal(first.channel, second.channel)
        np.testing.assert_array_equal(first.shadowing_db, 0.0)
        self.assertTrue(first.blocked_links[0, 1])
        fspl_db = 20.0 * np.log10(
            4.0 * np.pi * config.carrier_frequency_hz * 100.0 / 299_792_458.0
        )
        self.assertAlmostEqual(first.path_loss_db[0, 1], fspl_db + config.building_loss_db)
        np.testing.assert_array_equal(first.channel[0, 1], first.channel[0, 1, 0])
        self.assertEqual(first_rng.bit_generator.state, first_state)
        self.assertEqual(second_rng.bit_generator.state, second_state)

    def test_channel_rejects_invalid_distance_and_nonsequential_slot(self) -> None:
        config = environment_config(uav_count=2, ru_count=2)
        duplicate = np.array([[0.0, 0.0, 80.0], [0.0, 0.0, 80.0]])
        with self.assertRaisesRegex(ChannelError, "strictly positive distance"):
            PhysicalChannelModel(config, make_rng(42, 40)).reset(duplicate)

        invalid_config = environment_config(carrier_frequency_hz=0.0)
        with self.assertRaisesRegex(ChannelError, "carrier_frequency_hz"):
            PhysicalChannelModel(invalid_config, make_rng(42, 40))

        positions = np.array([[0.0, 0.0, 80.0], [10.0, 0.0, 80.0]])
        model = PhysicalChannelModel(config, make_rng(42, 40))
        model.reset(positions)
        with self.assertRaisesRegex(ChannelError, "sequentially"):
            model.generate(2, positions)

    def test_db_linear_helpers_and_ru_noise_chain(self) -> None:
        self.assertAlmostEqual(float(db_to_power_ratio(10.0)), 10.0)
        self.assertAlmostEqual(float(db_to_amplitude_ratio(20.0)), 10.0)
        self.assertAlmostEqual(float(dbm_to_watts(30.0)), 1.0)
        config = environment_config(total_bandwidth_hz=20e6, ru_count=20)
        self.assertEqual(config.ru_bandwidth_hz, 1e6)
        expected = float(dbm_to_watts(-174.0)) * 1e6 * float(db_to_power_ratio(7.0))
        self.assertAlmostEqual(receiver_noise_power_w(config), expected)


class StaleCsiAndHistoryTests(unittest.TestCase):
    def test_stale_lookup_uses_t_minus_aoi_and_negative_index_is_invalid(self) -> None:
        config = environment_config(ru_count=2, fixed_csi_aoi_slots=1, csi_error_std_db=0.0)
        history = ChannelHistory(config, make_rng(42, 50))
        channel_0 = true_channel(config, 2.0 + 0.0j)
        channel_1 = true_channel(config, 9.0 + 0.0j)
        history.record_physical_channel(0, channel_0)
        slot_0 = history.actor_features(0)
        self.assertFalse(np.any(slot_0.csi_valid_mask))
        self.assertTrue(np.all(slot_0.stale_csi == 0.0))
        history.advance_without_measurement(0)
        history.record_physical_channel(1, channel_1)
        slot_1 = history.actor_features(1)
        self.assertTrue(slot_1.csi_valid_mask[0, 1])
        np.testing.assert_array_equal(slot_1.stale_csi[0, 1], channel_0[0, 1])
        self.assertFalse(np.array_equal(slot_1.stale_csi[0, 1], channel_1[0, 1]))

    def test_csi_error_uses_amplitude_scaling_and_seed_is_reproducible(self) -> None:
        config = environment_config(ru_count=1, fixed_csi_aoi_slots=0, csi_error_std_db=2.0)
        seed = 99
        expected_rng = make_rng(seed, 50)
        expected_error_db = expected_rng.normal(0.0, config.csi_error_std_db)
        expected_factor = float(db_to_amplitude_ratio(expected_error_db))
        channel = true_channel(config, 1.0 + 0.0j)

        first = ChannelHistory(config, make_rng(seed, 50))
        second = ChannelHistory(config, make_rng(seed, 50))
        first.record_physical_channel(0, channel)
        second.record_physical_channel(0, channel)
        first_features = first.actor_features(0)
        second_features = second.actor_features(0)
        np.testing.assert_array_equal(first_features.stale_csi, second_features.stale_csi)
        self.assertAlmostEqual(
            abs(first_features.stale_csi[0, 1, 0]),
            abs(channel[0, 1, 0]) * expected_factor,
        )

    def test_simplified_csi_is_current_exact_and_consumes_no_rng(self) -> None:
        config = environment_config(
            ru_count=2,
            fixed_csi_aoi_slots=3,
            csi_error_std_db=2.0,
            channel_ablation_mode=ChannelAblationMode.SIMPLIFIED_DETERMINISTIC,
        )
        channel = true_channel(config, 3.0 + 0.0j)
        rng = make_rng(42, 50)
        rng_state = rng.bit_generator.state
        history = ChannelHistory(config, rng)
        history.record_physical_channel(0, channel)
        features = history.actor_features(0)

        np.testing.assert_array_equal(features.stale_csi, channel)
        self.assertTrue(np.all(features.csi_valid_mask[~np.eye(2, dtype=bool)]))
        self.assertTrue(np.all(features.csi_aoi_slots == 0))
        self.assertEqual(features.stale_csi.shape, channel.shape)
        self.assertEqual(features.stale_csi.dtype, np.complex128)
        self.assertEqual(rng.bit_generator.state, rng_state)

    def test_actor_stale_tensor_is_a_copy_and_true_channel_is_not_in_feature_contract(self) -> None:
        config = environment_config(ru_count=1, fixed_csi_aoi_slots=0, csi_error_std_db=0.0)
        channel = true_channel(config, 3.0 + 0.0j)
        history = ChannelHistory(config, make_rng(42, 50))
        history.record_physical_channel(0, channel)
        features = history.actor_features(0)
        self.assertFalse(np.shares_memory(features.stale_csi, channel))
        channel[0, 1, 0] = 99.0
        self.assertNotEqual(features.stale_csi[0, 1, 0], 99.0)
        self.assertFalse(hasattr(features, "true_channel"))
        self.assertFalse(hasattr(features, "executor_historical_quality"))
        self.assertNotIn("executor_historical_quality", {field.name for field in dataclasses.fields(features)})

    def test_zero_measurement_is_valid_missing_keeps_history_and_aoi_increments(self) -> None:
        history = InterferenceHistory(2, 2, beta=0.8, initial_interference_w=5.0)
        initial = history.snapshot(0)
        self.assertFalse(np.any(initial.valid_mask))
        self.assertTrue(np.all(initial.message_aoi_slots == -1))

        measurement = np.zeros((2, 2), dtype=np.float64)
        mask = np.zeros((2, 2), dtype=bool)
        mask[0, 0] = True
        updated = history.update_with_measurement(0, measurement, mask)
        self.assertAlmostEqual(updated.interference_w[0, 0], 4.0)
        self.assertTrue(updated.valid_mask[0, 0])
        self.assertEqual(updated.message_aoi_slots[0], 1)

        missing = history.advance_without_measurement(1)
        self.assertAlmostEqual(missing.interference_w[0, 0], 4.0)
        self.assertTrue(missing.valid_mask[0, 0])
        self.assertEqual(missing.message_aoi_slots[0], 2)
        self.assertEqual(missing.message_aoi_slots[1], -1)

    def test_history_reset_clears_values_masks_aoi_and_episode_channel_buffer(self) -> None:
        config = environment_config(ru_count=1, fixed_csi_aoi_slots=1, csi_error_std_db=0.0)
        history = ChannelHistory(config, make_rng(42, 50))
        history.record_physical_channel(0, true_channel(config, 2.0))
        history.update_with_measurement(0, np.zeros((2, 1)), np.ones((2, 1), dtype=bool))
        history.reset(make_rng(42, 50))
        history.record_physical_channel(0, true_channel(config, 8.0))
        features = history.actor_features(0)
        self.assertFalse(np.any(features.csi_valid_mask))
        self.assertFalse(np.any(features.interference_valid_mask))
        self.assertTrue(np.all(features.message_aoi_slots == -1))

    def test_historical_denominator_quality_masks_and_stale_not_current(self) -> None:
        config = environment_config(
            ru_count=2,
            total_bandwidth_hz=2e6,
            fixed_csi_aoi_slots=1,
            csi_error_std_db=0.0,
            interference_ema_beta=0.0,
            reference_transmit_power_w=2.0,
        )
        history = ChannelHistory(config, make_rng(42, 50))
        channel_0 = true_channel(config, 2.0)
        channel_1 = true_channel(config, 10.0)
        history.record_physical_channel(0, channel_0)
        invalid = history.actor_features(0)
        self.assertFalse(np.any(invalid.quality_valid_mask))

        measured_interference = np.full((2, 2), 3.0)
        history.update_with_measurement(0, measured_interference, np.ones((2, 2), dtype=bool))
        history.record_physical_channel(1, channel_1)
        features = history.actor_features(1)
        expected_denominator = history.noise_power_w + 3.0
        np.testing.assert_allclose(features.historical_denominator_w, expected_denominator)
        expected_quality = (
            (config.reference_transmit_power_w / config.ru_count)
            * abs(channel_0[0, 1, 0]) ** 2
            / expected_denominator
        )
        self.assertAlmostEqual(features.historical_quality[0, 1, 0], expected_quality)
        self.assertTrue(features.quality_valid_mask[0, 1, 0])
        scalar = executor_historical_quality_mean(features, 0, 1, [0, 1])
        self.assertAlmostEqual(scalar, np.mean(features.historical_quality[0, 1, :]))

    def test_history_rejects_same_slot_measurement_visibility(self) -> None:
        history = InterferenceHistory(2, 1, beta=0.0, initial_interference_w=0.0)
        history.update_with_measurement(0, np.ones((2, 1)), np.ones((2, 1), dtype=bool))
        with self.assertRaisesRegex(HistoryError, "not slot 0"):
            history.snapshot(0)
        self.assertEqual(history.snapshot(1).message_aoi_slots[0], 1)


if __name__ == "__main__":
    unittest.main()
