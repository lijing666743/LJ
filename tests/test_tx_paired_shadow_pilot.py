"""Reliability tests only: no optimizer, backward, training or evaluation job."""
from __future__ import annotations

import copy
import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch

from src.config import RunConfig
from src.env.environment import U2UMECEnvironment
from src.evaluation.tx_paired_shadow_pilot import (
    FrozenSampler, LifecycleLedger, ReliabilityError, exogenous_digest, make_arms,
    policy_seed, reward_check, select_pairs,
)


def active_fixture() -> U2UMECEnvironment:
    config = RunConfig()
    config = replace(config, seed=43, environment=replace(config.environment,
        episode_horizon=8, arrival_probabilities=(1.0,) * config.environment.uav_count,
        building_layout=(), candidate_neighbor_radius_m=2000.0))
    env = U2UMECEnvironment(config)
    env.reset()
    env.step(env.canonical_proposals())
    proposals = list(env.canonical_proposals())
    for index, observation in enumerate(env.current_observations):
        mask = observation.action_masks
        destination = next(value for value, valid in zip(mask.route_domain, mask.route_mask)
                           if valid and isinstance(value, int))
        proposals[index] = replace(proposals[index], route=destination)
    env.step(proposals)
    return env


def legal_transmit(env: U2UMECEnvironment, source: int = 0):
    proposal = env.canonical_proposals()[source]
    mask = env.current_observations[source].action_masks
    for branch in ("tx_select", "resource_group", "resource_width", "power_level"):
        domain, valid = mask.domain_for(branch), mask.mask_for(branch, proposal)
        values = [value for value, allowed in zip(domain, valid) if allowed]
        if branch == "tx_select":
            values = [value for value in values if value != "idle"]
        proposal = replace(proposal, **{branch: values[0]})
    return proposal


class PairedShadowReliabilityTests(unittest.TestCase):
    def test_clones_have_identical_state_and_own_mutable_graphs(self):
        env = active_fixture()
        clone = env.clone_for_shadow()
        self.assertEqual(env.state_fingerprint(), clone.state_fingerprint())
        self.assertIsNot(env.lifecycle, clone.lifecycle)
        self.assertIsNot(env.executor.lifecycle, clone.executor.lifecycle)
        before = env.state_fingerprint()
        clone.step(clone.canonical_proposals())
        self.assertEqual(before, env.state_fingerprint())

    def test_cloned_rng_draws_do_not_mutate_original_rng(self):
        env = active_fixture()
        clone = env.clone_for_shadow()
        before = env.rng_fingerprint()
        clone.traffic.arrival_rng.random()
        self.assertEqual(before, env.rng_fingerprint())
        self.assertNotEqual(before, clone.rng_fingerprint())

    def test_factual_replay_matches_info_and_state(self):
        env = active_fixture()
        clone = env.clone_for_shadow()
        for _ in range(3):
            actions = env.canonical_proposals()
            a, b = env.step(actions), clone.step(actions)
            self.assertEqual(a.info, b.info)
            self.assertEqual(env.state_fingerprint(), clone.state_fingerprint())

    def test_arms_are_legal_and_change_only_tx_subtree(self):
        env = active_fixture()
        factual = env.canonical_proposals()
        arm_i, arm_t = make_arms(env, factual, 0, legal_transmit(env))
        self.assertEqual(arm_i[0].tx_select, "idle")
        self.assertIsInstance(arm_t[0].tx_select, int)
        self.assertGreater(arm_t[0].power_level, 0)
        self.assertEqual(arm_i[1:], arm_t[1:])
        self.assertEqual(arm_i[0].route, arm_t[0].route)
        self.assertEqual(arm_i[0].cpu_queue, arm_t[0].cpu_queue)

    def test_illegal_transmit_fails_closed(self):
        env = active_fixture()
        invalid = replace(legal_transmit(env), resource_width=999)
        with self.assertRaises(ReliabilityError):
            make_arms(env, env.canonical_proposals(), 0, invalid)

    def test_idle_disguised_as_transmit_fails(self):
        env = active_fixture()
        with self.assertRaises(ReliabilityError):
            make_arms(env, env.canonical_proposals(), 0, env.canonical_proposals()[0])

    def test_common_exogenous_realizations_survive_different_actions(self):
        env = active_fixture()
        a, b = env.clone_for_shadow(), env.clone_for_shadow()
        arm_i, arm_t = make_arms(env, env.canonical_proposals(), 0, legal_transmit(env))
        a.step(arm_i)
        b.step(arm_t)
        self.assertEqual(exogenous_digest(a), exogenous_digest(b))
        self.assertNotEqual(a.state_fingerprint(), b.state_fingerprint())

    def test_exogenous_misalignment_is_detected(self):
        env = active_fixture()
        clone = env.clone_for_shadow()
        clone.traffic.arrival_rng.random()
        self.assertNotEqual(exogenous_digest(env), exogenous_digest(clone))

    def test_slot_addressed_policy_rng_is_owned_and_reproducible(self):
        global_before = torch.get_rng_state().clone()
        a, b = torch.Generator(), torch.Generator()
        a.manual_seed(policy_seed(100)); b.manual_seed(policy_seed(100))
        self.assertTrue(torch.equal(torch.rand(8, generator=a), torch.rand(8, generator=b)))
        self.assertTrue(torch.equal(torch.get_rng_state(), global_before))
        self.assertNotEqual(policy_seed(100), policy_seed(101))
        self.assertNotEqual(policy_seed(100), policy_seed(100, "completion"))

    def test_actual_reward_and_energy_conservation(self):
        env = active_fixture()
        actions = list(env.canonical_proposals())
        actions[0] = legal_transmit(env)
        result = env.step(actions)
        reward_check(result)
        invalid = SimpleNamespace(reward=result.reward + 1, info=result.info)
        with self.assertRaises(ReliabilityError):
            reward_check(invalid)

    def test_reward_energy_mismatch_is_detected(self):
        env = active_fixture()
        result = env.step(env.canonical_proposals())
        info = copy.deepcopy(result.info)
        info["reward"]["actual_energy_j"] += 1
        with self.assertRaises(ReliabilityError):
            reward_check(SimpleNamespace(reward=result.reward, info=info))

    def test_task_ledger_has_complete_terminal_join(self):
        env = active_fixture()
        # Record the fixture's existing arrivals with no fabricated events.
        ledger = LifecycleLedger()
        ledger.tasks = {task_id: task.snapshot() for task_id, task in env.lifecycle.tasks.items()}
        while not env.done:
            result = env.step(env.canonical_proposals())
            ledger.update(env, result)
        joined = ledger.validate(env)
        self.assertEqual(len(joined), len(env.lifecycle.tasks))
        self.assertTrue(all(task["outcome"] in ("done", "expired", "truncated") for task in joined))

    def test_missing_and_duplicate_task_events_fail_closed(self):
        env = active_fixture()
        ledger = LifecycleLedger()
        with self.assertRaises(ReliabilityError):
            ledger.add(env, env.slot, "tx_attempt", 999999)
        task_id = next(iter(env.lifecycle.tasks))
        ledger.add(env, env.slot, "tx_attempt", task_id)
        with self.assertRaises(ReliabilityError):
            ledger.add(env, env.slot, "tx_attempt", task_id)

    def test_wrong_source_join_is_detected(self):
        env = active_fixture()
        ledger = LifecycleLedger()
        task = next(iter(env.lifecycle.tasks.values()))
        with self.assertRaises(ReliabilityError):
            ledger.add(env, env.slot, "tx_attempt", task.task_id, source=(task.source_uav + 1) % 4)

    def test_selection_balances_sources_and_coverage_without_d(self):
        rows = [{"slot": slot, "source": source, "queue_bin": "1" if slot % 2 else "2+",
                 "slack_bin": ("<=15", "16-40", ">40")[slot % 3],
                 "factual_execution": "Idle" if slot % 2 else "effective"}
                for source in range(4) for slot in range(10)]
        selected = select_pairs(rows, 4)
        self.assertEqual(len(selected), 24)
        self.assertEqual(len({(row["slot"], row["source"]) for row in selected}), 24)
        for source in range(4):
            group = [row for row in selected if row["source"] == source]
            self.assertEqual(len(group), 6)
            self.assertEqual({row["slack_bin"] for row in group}, {"<=15", "16-40", ">40"})

    def test_insufficient_source_coverage_fails_closed(self):
        with self.assertRaises(ReliabilityError):
            select_pairs([], 4)

    def test_conditional_completion_accepts_first_t_without_outcome_search(self):
        env = active_fixture()
        factual = env.canonical_proposals()
        transmit = legal_transmit(env)
        sampler = object.__new__(FrozenSampler)
        calls = []
        def sample(environment, hidden, *, seed):
            calls.append(seed)
            actions = list(factual)
            if len(calls) == 2:
                actions[0] = replace(transmit, route="defer")
            return tuple(actions), hidden.clone(), {}
        sampler.sample = sample
        before = env.state_fingerprint()
        selected, count = sampler.complete_transmit(env, torch.zeros(1), factual, 0, 7)
        self.assertEqual(count, 2)
        self.assertEqual(selected.tx_select, transmit.tx_select)
        self.assertEqual(selected.resource_group, transmit.resource_group)
        self.assertEqual(selected.power_level, transmit.power_level)
        self.assertEqual(selected.route, factual[0].route)
        self.assertEqual(env.state_fingerprint(), before)
        self.assertNotEqual(calls[0], calls[1])

    def test_conditional_completion_reuses_factual_transmit_without_draws(self):
        env = active_fixture()
        factual = list(env.canonical_proposals())
        factual[0] = legal_transmit(env)
        sampler = object.__new__(FrozenSampler)
        def forbidden(*args, **kwargs):
            self.fail("factual T must not be resampled")
        sampler.sample = forbidden
        selected, count = sampler.complete_transmit(env, torch.zeros(1), tuple(factual), 0, 0)
        self.assertEqual(count, 0)
        self.assertEqual(selected, factual[0])

    def test_conditional_completion_exhaustion_fails_closed(self):
        env = active_fixture()
        factual = env.canonical_proposals()
        sampler = object.__new__(FrozenSampler)
        calls = []
        def sample(environment, hidden, *, seed):
            calls.append(seed)
            return factual, hidden, {}
        sampler.sample = sample
        with self.assertRaises(ReliabilityError):
            sampler.complete_transmit(env, torch.zeros(1), factual, 0, 0)
        self.assertEqual(len(calls), 4096)


if __name__ == "__main__":
    unittest.main()
