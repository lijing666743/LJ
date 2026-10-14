"""Focused tests for the active-TX power-level mask contract."""

from __future__ import annotations

import unittest

import numpy as np

from src.env.actions import ActionProposal
from src.env.observation import BRANCH_ORDER, ActionMasks


class TxPowerMaskTests(unittest.TestCase):
    @staticmethod
    def _masks(power_energy_mask: tuple[bool, ...]) -> ActionMasks:
        return ActionMasks(
            uav_id=0,
            paper_uav_id=1,
            sampling_order=BRANCH_ORDER,
            route_domain=("idle", "local", "defer", 1),
            tx_select_domain=("idle", 1),
            resource_group_domain=("idle", 1, 2),
            resource_width_domain=(1, 2),
            power_level_domain=(0.0, 0.25, 0.5, 1.0),
            cpu_queue_domain=("idle", 0, 1),
            cpu_frequency_domain=(0.0, 1.0),
            route_mask=np.array((True, False, False, False), dtype=np.bool_),
            tx_select_mask=np.array((True, True), dtype=np.bool_),
            cpu_queue_mask=np.array((True, True, False), dtype=np.bool_),
            power_energy_mask=np.array(power_energy_mask, dtype=np.bool_),
            cpu_frequency_energy_mask=np.array(
                ((True, True), (True, False)),
                dtype=np.bool_,
            ),
            route_branch_active=False,
            tx_branch_active=True,
            cpu_queue_branch_active=True,
            resource_group_count=2,
            tx_idle_action="idle",
            cpu_idle_action="idle",
            canonical_width=1,
            canonical_power=0.0,
            canonical_cpu_frequency=0.0,
        )

    @staticmethod
    def _proposal(
        *,
        tx_select: str | int,
        resource_group: str | int,
        power_level: float,
        cpu_queue: str | int = "idle",
        cpu_frequency: float = 0.0,
    ) -> ActionProposal:
        return ActionProposal(
            uav_id=0,
            route="idle",
            tx_select=tx_select,
            resource_group=resource_group,
            resource_width=1,
            power_level=power_level,
            cpu_queue=cpu_queue,
            cpu_frequency=cpu_frequency,
        )

    def test_active_tx_excludes_zero_when_positive_power_is_feasible(self) -> None:
        masks = self._masks((True, True, False, True))

        power_mask = masks.mask_for(
            "power_level",
            {"tx_select": 1, "resource_group": 1},
        )

        np.testing.assert_array_equal(power_mask, (False, True, False, True))
        self.assertFalse(
            masks.is_legal(
                self._proposal(tx_select=1, resource_group=1, power_level=0.0)
            )
        )
        self.assertTrue(
            masks.is_legal(
                self._proposal(tx_select=1, resource_group=1, power_level=0.25)
            )
        )

    def test_active_tx_retains_zero_when_all_positive_power_is_infeasible(self) -> None:
        masks = self._masks((True, False, False, False))

        power_mask = masks.mask_for(
            "power_level",
            {"tx_select": 1, "resource_group": 1},
        )

        np.testing.assert_array_equal(power_mask, (True, False, False, False))
        self.assertTrue(
            masks.is_legal(
                self._proposal(tx_select=1, resource_group=1, power_level=0.0)
            )
        )

    def test_idle_tx_power_behavior_is_unchanged(self) -> None:
        masks = self._masks((True, True, True, True))

        power_mask = masks.mask_for(
            "power_level",
            {"tx_select": "idle", "resource_group": "idle"},
        )

        np.testing.assert_array_equal(power_mask, (True, False, False, False))
        self.assertTrue(
            masks.is_legal(
                self._proposal(
                    tx_select="idle",
                    resource_group="idle",
                    power_level=0.0,
                )
            )
        )

    def test_local_cpu_service_remains_legal(self) -> None:
        masks = self._masks((True, True, True, True))

        local_proposal = self._proposal(
            tx_select="idle",
            resource_group="idle",
            power_level=0.0,
            cpu_queue="local",
            cpu_frequency=1.0,
        )

        self.assertTrue(masks.is_legal(local_proposal))
        np.testing.assert_array_equal(
            masks.mask_for("cpu_frequency", local_proposal),
            (True, True),
        )


if __name__ == "__main__":
    unittest.main()
