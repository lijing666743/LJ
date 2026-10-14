"""Boundary cases for offline energy-risk reconstruction."""

import pytest

from src.evaluation.checkpoint_energy_audit import cpu_energy_blocked, persistent_blockage_start


def blockage(cycles, energy):
    return cpu_energy_blocked(cycles, energy=energy, frequency=4, coefficient=1,
                             minimum_level=.25, slot_duration=1, tolerance=1e-12)


def test_partial_head_can_be_served_below_full_slot_energy():
    assert blockage([.1], .1) == (False, .1)
    assert blockage([2], .1) == (True, 1)


def test_short_head_in_another_source_keeps_a_positive_action_legal():
    assert blockage([2, .05], .05) == (False, .05)
    assert blockage([], 0) == (False, None)


@pytest.mark.parametrize("blocked,positive,backlogged,expected", [
    ([2, 3, 5], [0, 1], [0, 1, 2, 3, 5], 2),
    ([2, 3, 5], [0, 1], [0, 1, 2, 3, 4, 5], 5),
    ([2, 3], [0, 4], [0, 1, 2, 3, 4], None),
    ([], [], [], None),
])
def test_persistent_blockage_requires_a_blocked_suffix_after_last_service(blocked, positive, backlogged, expected):
    assert persistent_blockage_start(blocked, positive, backlogged) == expected
