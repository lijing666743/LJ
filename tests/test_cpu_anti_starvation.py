"""E14a feasibility gate contracts; no development/formal training runs."""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.config import (
    ConfigError, CHECKPOINT_KIND_PERIODIC_RESUME, load_run_config,
    mappo_periodic_checkpoint_path, mappo_final_checkpoint_path,
)
from src.env.environment import U2UMECEnvironment
from src.env.lifecycle import LifecycleManager
from src.env.tasks import TaskStatus
from src.evaluation.actor_loader import (
    EvaluationCheckpointError, load_final_actor_for_evaluation,
    load_actor_for_development_diagnostic,
)
from src.evaluation.runner import FormalEvaluationRunner, FormalEvaluationError
from src.models.ca_gat_mappo import (
    CAGATMAPPOActor, ActorObservationTensorizer, CentralizedStateTensorBatch,
)
from src.models.ca_gat_mappo_actions import (
    CAGATMAPPOActionDistribution, SequentialActionMaskBatch, ActionDistributionError,
)
from src.models.ca_gat_mappo_checkpoint import (
    _serialize_action_masks, _deserialize_action_masks,
    serialize_active_rollout, restore_active_rollout,
    atomic_save_checkpoint,
)
from src.models.ca_gat_mappo_rollout import CAGATMAPPORolloutBuffer, CAGATMAPPORolloutTransition
from src.models.integrated_scheduler import (
    cpu_deadline_feasibility, cpu_idle_deadline_critical, cpu_joint_statistics,
)
from tests.test_integrated_scheduler import fixture
from tests.test_ca_gat_mappo_checkpoint import make_payload
from tests.test_formal_evaluation import make_evaluation_fixture


ROOT = Path(__file__).resolve().parents[1]
E13 = ROOT / 'tests/fixtures/configs/e13-v1-tx-paired-shadow-seed42-32k.json'
E14A = E13.with_name('e14a-v1-cpu-feasibility-seed42-8k-dev.json')


def with_mode(config, mode):
    return replace(config, training=replace(config.training, mappo=replace(
        config.training.mappo, cpu_anti_starvation_mode=mode)))


class TestCPUAntiStarvation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.off = load_run_config(E13)
        cls.on = with_mode(cls.off, 'deadline_feasibility_v1')
        cls.seed, cls.batch, cls.original_masks = fixture(cls.off)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def masks(self, heads=(), *, frequencies=(True, True, True, True)):
        """head tuples: source, cycles, inclusive slack, fmax (exact metadata)."""
        contracts = []
        for c in self.original_masks.flattened():
            qmask = np.array([True, False, False, False, False])
            context = [(0.0, 0, 1.e9)] * 4
            for source, cycles, slack, fmax in heads:
                qmask[source + 1] = True
                context[source] = (cycles, slack, fmax)
            contracts.append(replace(c, cpu_queue_mask=qmask,
                cpu_queue_branch_active=bool(heads),
                cpu_frequency_energy_mask=np.tile(frequencies, (4, 1)),
                cpu_deadline_context=tuple(context)))
        return SequentialActionMaskBatch(((tuple(contracts),),))

    def joint(self, masks, *, config=None, external=None):
        actor = CAGATMAPPOActor(self.on if config is None else config).eval()
        with torch.no_grad():
            recurrent = actor(self.batch, actor.initial_hidden(1)).recurrent_features
            return cpu_joint_statistics(actor, recurrent, self.batch,
                masks.flattened(), external, 'deterministic', None)

    def test_empty_queue_keeps_idle(self):
        result = self.joint(self.masks())
        self.assertTrue(result['mask'][..., 0].all())
        self.assertEqual(int(result['mask'].sum()), 4)

    def test_safe_slack_keeps_idle_even_when_slow_atom_is_urgent(self):
        # Fastest requires two slots; half frequency requires four. Four slots
        # remain, so considering the slow atom alone would mask Idle wrongly.
        result = self.joint(self.masks([(0, 40.e6, 4, 1.e9)]))
        self.assertTrue(result['mask'][..., 0].all())

    def test_last_chance_masks_only_idle(self):
        masks = self.masks([(0, 40.e6, 2, 1.e9)])
        on = self.joint(masks)
        off = self.joint(masks, config=self.off)
        self.assertFalse(on['mask'][..., 0].any())
        self.assertTrue(torch.equal(on['mask'][..., 1:], off['mask'][..., 1:]))
        self.assertTrue((on['queue_index'] > 0).all())
        self.assertTrue((on['frequency_index'] > 0).all())

    def test_already_impossible_keeps_idle(self):
        self.assertTrue(self.joint(self.masks([(0, 40.e6 + 1., 2, 1.e9)]))['mask'][..., 0].all())

    def test_no_energy_feasible_positive_frequency_keeps_idle(self):
        result = self.joint(self.masks([(0, 40.e6, 2, 1.e9)],
                                      frequencies=(True, False, False, False)))
        self.assertTrue(result['mask'][..., 0].all())
        self.assertEqual(int(result['mask'].sum()), 4)

    def test_uses_fastest_legal_frequency_instead_of_unavailable_fmax(self):
        masks = self.masks([(0, 40.e6, 4, 1.e9)], frequencies=(True, True, True, False))
        self.assertFalse(self.joint(masks)['mask'][..., 0].any())

    def test_any_source_including_local_can_trigger_gate(self):
        for source in range(4):
            heads = [(q, 40.e6, 9 if q != source else 2, 1.e9) for q in range(4)]
            with self.subTest(source=source):
                self.assertFalse(self.joint(self.masks(heads))['mask'][..., 0].any())

    def test_illegal_head_cannot_trigger_gate(self):
        c = self.masks([(0, 40.e6, 2, 1.e9)]).flattened()[0]
        c = replace(c, cpu_queue_mask=np.array([True, False, False, False, False]))
        self.assertFalse(cpu_idle_deadline_critical(c, .02))

    def test_float64_boundary_and_last_deadline_slot(self):
        self.assertEqual(cpu_deadline_feasibility(20.e6, 1, 1.e9, .02), (1, 0))
        above = math.nextafter(20.e6, math.inf)
        below = math.nextafter(20.e6, -math.inf)
        self.assertTrue(math.isinf(cpu_deadline_feasibility(above, 1, 1.e9, .02)[0]))
        self.assertEqual(cpu_deadline_feasibility(below, 1, 1.e9, .02), (1, 0))
        self.assertFalse(self.joint(self.masks([(0, below, 1, 1.e9)]))['mask'][..., 0].any())
        self.assertTrue(self.joint(self.masks([(0, above, 1, 1.e9)]))['mask'][..., 0].all())
        # No epsilon: the real environment subtracts float64 service each slot
        # and tests remaining_cycles == 0. 1 - .1 ten times is still positive.
        self.assertTrue(math.isinf(cpu_deadline_feasibility(1., 10, 5., .02)[0]))
        self.assertEqual(cpu_deadline_feasibility(1., 11, 5., .02), (11, 0))

    def test_actual_service_settlement_confirms_inclusive_deadline(self):
        # Use the actual lifecycle transition, service and settlement methods.
        for cycles in (40.e6, math.nextafter(40.e6, math.inf), .6):
            for idle in (False, True):
                with self.subTest(cycles=cycles, idle=idle):
                    manager = LifecycleManager()
                    t = manager.create_task(source_uav=0, data_bits=1., cpu_cycles=cycles,
                                            arrival_slot=0, deadline_slot=3)
                    manager.route_task(t, slot=1, destination='local')
                    capacity = .2 if cycles == .6 else 20.e6
                    for slot in (2, 3):
                        if not (idle and slot == 2):
                            manager.service_cpu(t, slot=slot, cycles=min(t.remaining_cycles, capacity))
                        manager.settle_slot(slot)
                    required, _ = cpu_deadline_feasibility(cycles, 2-int(idle), capacity/.02, .02)
                    self.assertEqual(t.status == TaskStatus.DONE, math.isfinite(required))

    def test_task11_task5_start_safe_then_become_critical(self):
        for name, cycles, slack, fmax, required in (
            ('task11', 1056982879.771, 45, 1998951144.562, 27),
            ('task5', 1201733795.1456351, 62, 1434027346.2676, 42),
        ):
            with self.subTest(task=name):
                self.assertEqual(cpu_deadline_feasibility(cycles, slack, fmax, .02),
                                 (required, slack-required))
                self.assertTrue(self.joint(self.masks([(0, cycles, slack, fmax)]))['mask'][..., 0].all())
                self.assertFalse(self.joint(self.masks([(0, cycles, required, fmax)]))['mask'][..., 0].any())
                self.assertTrue(self.joint(self.masks([(0, cycles, required-1, fmax)]))['mask'][..., 0].all())

    def test_off_retains_original_joint_distribution_without_context(self):
        masks = self.masks([(0, 40.e6, 2, 1.e9)])
        legacy = SequentialActionMaskBatch(((tuple(replace(c, cpu_deadline_context=None)
                                             for c in masks.flattened()),),))
        a = self.joint(masks, config=self.off)
        b = self.joint(legacy, config=self.off)
        for key in ('mask', 'probabilities', 'queue_index', 'frequency_index'):
            self.assertTrue(torch.equal(a[key], b[key]))

    def test_train_eval_replay_use_identical_support_and_log_prob(self):
        masks = self.masks([(0, 40.e6, 2, 1.e9)])
        actor = CAGATMAPPOActor(self.on)
        distribution = CAGATMAPPOActionDistribution(actor, self.on)
        with torch.no_grad():
            actor.train()
            train = distribution.sample_actions(self.batch, masks, actor.initial_hidden(1))
            actor.eval()
            evaluation = distribution.deterministic_actions(self.batch, masks, actor.initial_hidden(1))
            replay = distribution.evaluate_actions(self.batch, masks, train.proposals, actor.initial_hidden(1))
        for out in (train, evaluation, replay):
            self.assertFalse(out.action_masks['cpu_queue'][..., 0].any())
            self.assertTrue((out.probabilities['cpu_queue'][..., 0] == 0).all())
            self.assertTrue(torch.equal(out.action_masks['cpu_queue'], evaluation.action_masks['cpu_queue']))
        self.assertTrue(torch.equal(train.joint_log_prob, replay.joint_log_prob))
        idle = replace(train.proposals[0][0][0], cpu_queue='idle', cpu_frequency=0.)
        external = (idle, *train.proposals[0][0][1:])
        with self.assertRaises(ActionDistributionError):
            self.joint(masks, external=external)

    def test_no_parameter_feature_domain_route_or_tx_changes(self):
        masks = self.masks([(0, 40.e6, 2, 1.e9)])
        off_actor, on_actor = CAGATMAPPOActor(self.off).eval(), CAGATMAPPOActor(self.on).eval()
        self.assertEqual(off_actor.state_dict().keys(), on_actor.state_dict().keys())
        for key, value in off_actor.state_dict().items():
            self.assertTrue(torch.equal(value, on_actor.state_dict()[key]), key)
        off_features, off_phys = off_actor.cpu_joint_scheduler.candidate_features(self.batch)
        on_features, on_phys = on_actor.cpu_joint_scheduler.candidate_features(self.batch)
        self.assertTrue(torch.equal(off_features, on_features))
        for key in off_phys:
            self.assertTrue(torch.equal(off_phys[key], on_phys[key]), key)
        with torch.no_grad():
            before = CAGATMAPPOActionDistribution(off_actor, self.off).deterministic_actions(
                self.batch, masks, off_actor.initial_hidden(1))
            after = CAGATMAPPOActionDistribution(on_actor, self.on).deterministic_actions(
                self.batch, masks, on_actor.initial_hidden(1))
        for branch in ('route', 'tx_select', 'resource_group', 'resource_width', 'power_level'):
            self.assertTrue(torch.equal(before.action_masks[branch], after.action_masks[branch]), branch)
            self.assertTrue(torch.equal(before.probabilities[branch], after.probabilities[branch]), branch)
            self.assertTrue(torch.equal(before.action_indices[branch], after.action_indices[branch]), branch)

    def test_critical_sampling_mask_survives_rollout_storage_and_replay(self):
        masks = self.masks([(0, 40.e6, 2, 1.e9)])
        actor = CAGATMAPPOActor(self.on).eval()
        policy = CAGATMAPPOActionDistribution(actor, self.on)
        hidden = actor.initial_hidden(1)
        with torch.no_grad():
            output = policy.sample_actions(self.batch, masks, hidden)
        transition = CAGATMAPPORolloutTransition.from_step(
            spec=actor.spec, slot=0, actor_batch=self.batch,
            action_mask_batch=masks, action_output=output, hidden_in=hidden,
            centralized_state=CentralizedStateTensorBatch(self.seed.centralized_state[None, None]),
            old_value=torch.zeros(4), reward=torch.zeros(4), terminated=False,
            truncated=False, episode_boundary=False, bootstrap_allowed=True,
            bootstrap_value=torch.zeros(4), executed_action_summary=self.seed.executed_action_summary,
            rejection_or_downgrade_summary=self.seed.rejection_or_downgrade_summary)
        transition.validate()
        for contract in transition.action_mask_contracts:
            self.assertFalse(contract.cpu_queue_mask[0])
        replay_masks = SequentialActionMaskBatch(((transition.action_mask_contracts,),))
        replay = policy.evaluate_actions(self.batch, replay_masks, output.proposals, hidden)
        self.assertTrue(torch.equal(output.joint_log_prob, replay.joint_log_prob))
        (-replay.joint_log_prob.mean()).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters()))

    def test_exact_context_and_rng_come_from_observation_only(self):
        off_env, on_env = U2UMECEnvironment(self.off), U2UMECEnvironment(self.on)
        off_reset, on_reset = off_env.reset(), on_env.reset()
        self.assertEqual(off_env.randomness_snapshot(), on_env.randomness_snapshot())
        for old, new in zip(off_reset.observations, on_reset.observations):
            self.assertIsNone(old.action_masks.cpu_deadline_context)
            for source, (cycles, slack, fmax) in enumerate(new.action_masks.cpu_deadline_context):
                self.assertEqual(cycles, new.private_queues.cpu_by_source.head_remaining_cycles[source])
                self.assertEqual(slack, new.private_queues.cpu_by_source.head_slack_slots[source])
                self.assertEqual(fmax, new.self_resources.max_cpu_frequency_hz)
            self.assertEqual(old.snapshot(), new.snapshot())

    def test_optional_checkpoint_context_roundtrip_and_legacy_serialization(self):
        old = self.original_masks.flattened()[0]
        saved = _serialize_action_masks(old)
        self.assertNotIn('cpu_deadline_context', saved)
        self.assertIsNone(_deserialize_action_masks(saved).cpu_deadline_context)
        new = self.masks([(0, 40.e6, 2, 1.e9)]).flattened()[0]
        restored = _deserialize_action_masks(_serialize_action_masks(new))
        self.assertEqual(new.cpu_deadline_context, restored.cpu_deadline_context)
        self.assertTrue(cpu_idle_deadline_critical(restored, .02))
        # Preserve exact gate inputs across an active rollout checkpoint.
        buffer = CAGATMAPPORolloutBuffer(self.on)
        metadata = dict(self.seed.rejection_or_downgrade_summary)
        metadata['tx_paired_shadow'] = {
            'schema': 'tx_paired_shadow.v1', 'policy_version': 0,
            'rollout_offset': 0, 'paired_mask': [False] * 4, 'samples': [],
        }
        buffer.append(replace(self.seed, reward=torch.ones(4),
            rejection_or_downgrade_summary=metadata,
            old_value=self.seed.old_value.expand(4).clone(),
            bootstrap_value=self.seed.bootstrap_value.expand(4).clone(), action_mask_contracts=tuple(
            replace(c, cpu_deadline_context=new.cpu_deadline_context)
            for c in self.seed.action_mask_contracts)))
        state = serialize_active_rollout(buffer, 0, CHECKPOINT_KIND_PERIODIC_RESUME)
        restored_buffer, version = restore_active_rollout(self.on, state,
            checkpoint_kind=CHECKPOINT_KIND_PERIODIC_RESUME, policy_version=0)
        self.assertEqual(version, 0)
        self.assertEqual(restored_buffer.transition_at(0).action_mask_contracts[0]
                         .cpu_deadline_context, new.cpu_deadline_context)

    def test_legacy_identity_defaults_and_switch_validation(self):
        self.assertEqual(self.off.config_hash, '16ed37e1f161f7bdadf44fef120e3cfe26f4142c001e0e3b0dee7e4dd99d5d27')
        self.assertEqual(self.off.training.mappo.cpu_anti_starvation_mode, 'off')
        self.assertNotIn('cpu_anti_starvation_mode', self.off.resolved_dict()['training']['mappo'])
        for mode in ('bad', None):
            with self.assertRaises(ConfigError):
                with_mode(self.off, mode).validate()
        wrong = replace(self.on, training=replace(self.on.training, mappo=replace(
            self.on.training.mappo, cpu_scheduler_mode='legacy_hierarchical')))
        with self.assertRaises(ConfigError):
            wrong.validate()

    def test_off_fixed_seed_actions_rewards_and_terminal_outcomes_match(self):
        explicit = load_run_config(E13, cli_overrides={
            'training.mappo.cpu_anti_starvation_mode': 'off'})
        # One complete environment unit episode, without metrics/artifacts,
        # an optimizer, a Trainer, or the formal evaluation runner.
        # Independent owned sampling streams start at the same seed.
        envs = [U2UMECEnvironment(c) for c in (self.off, explicit)]
        configs = (self.off, explicit)
        actors = [CAGATMAPPOActor(c).eval() for c in configs]
        policies = [CAGATMAPPOActionDistribution(a, c) for a, c in zip(actors, configs)]
        tensorizers = [ActorObservationTensorizer(c) for c in configs]
        observations = [e.reset().observations for e in envs]
        hidden = [a.initial_hidden(1) for a in actors]
        for slot in range(self.off.environment.episode_horizon):
            outputs, steps = [], []
            for i in range(2):
                batch = tensorizers[i].encode_step(observations[i], episode_start=slot == 0)
                with torch.no_grad():
                    out = policies[i].sample_actions(batch,
                        SequentialActionMaskBatch.from_observations(observations[i]), hidden[i])
                outputs.append(out)
                step = envs[i].step(out.proposals[0][0])
                steps.append(step)
                observations[i], hidden[i] = step.observations, out.hidden_out
            self.assertEqual(outputs[0].proposals, outputs[1].proposals)
            self.assertEqual(steps[0].reward, steps[1].reward)
            self.assertEqual((steps[0].terminated, steps[0].truncated),
                             (steps[1].terminated, steps[1].truncated))
            self.assertEqual([t.snapshot() for t in envs[0].lifecycle.tasks.values()],
                             [t.snapshot() for t in envs[1].lifecycle.tasks.values()])
            self.assertEqual(envs[0].randomness_snapshot(), envs[1].randomness_snapshot())
        self.assertTrue(steps[0].terminated or steps[0].truncated)

    def test_dev_config_only_changes_budget_gate_and_output_roots(self):
        a, b = json.loads(E13.read_text(encoding='utf-8')), json.loads(E14A.read_text(encoding='utf-8'))
        b['training']['mappo'].pop('cpu_anti_starvation_mode')
        for name in ('max_training_episodes', 'max_training_environment_steps'):
            b['training']['mappo'][name] = a['training']['mappo'][name]
        b['output'] = a['output']
        self.assertEqual(a, b)
        config = load_run_config(E14A)
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.training.mappo.max_environment_transitions, 8000)
        self.assertEqual(config.training.mappo.max_training_episodes * config.environment.episode_horizon, 8000)
        self.assertEqual(Path(mappo_periodic_checkpoint_path(config, 4000)).name, 'step_4000.pt')
        self.assertEqual(Path(mappo_final_checkpoint_path(config)).name, 'final.pt')


class TestDevelopmentCheckpointDiagnostic(unittest.TestCase):
    def test_periodic_opt_in_does_not_relax_formal_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            _, evaluation, checkpoint, _ = make_evaluation_fixture(
                Path(directory), kind=CHECKPOINT_KIND_PERIODIC_RESUME)
            with self.assertRaises(EvaluationCheckpointError):
                load_final_actor_for_evaluation(evaluation, checkpoint, evaluation_device='cpu')
            actor = load_actor_for_development_diagnostic(evaluation, checkpoint, evaluation_device='cpu')
            self.assertEqual(actor.source_checkpoint_kind, CHECKPOINT_KIND_PERIODIC_RESUME)
            self.assertFalse(any(p.requires_grad for p in actor.actor.parameters()))
            with self.assertRaises(FormalEvaluationError):
                FormalEvaluationRunner(evaluation, checkpoint, evaluation_device='cpu', development_checkpoint=True)
            runner = FormalEvaluationRunner(evaluation, checkpoint, evaluation_device='cpu',
                collect_service_telemetry=True, development_checkpoint=True)
            self.assertTrue(runner.evaluation_run_id.startswith('development-service-eval__'))
            # Construction/actor loading only: do not execute any evaluation.

    def test_e14a_checkpoint_requires_matching_evaluation_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_run_config(E14A, cli_overrides={'training.mappo.training_device': 'cpu'})
            payload, *_ = make_payload(config, 'FINAL_COMPLETED')
            checkpoint = Path(directory) / 'unit.pt'
            atomic_save_checkpoint(payload, checkpoint)
            evaluation = replace(config, mode='evaluation', launch_profile=None)
            for name in ('git_commit', 'git_branch', 'git_dirty'):
                object.__setattr__(evaluation, name, getattr(config, name))
            loaded = load_actor_for_development_diagnostic(evaluation, checkpoint, evaluation_device='cpu')
            self.assertEqual(loaded.actor.cpu_joint_scheduler.anti_starvation_mode, 'deadline_feasibility_v1')
            with self.assertRaises(EvaluationCheckpointError):
                load_actor_for_development_diagnostic(with_mode(evaluation, 'off'), checkpoint, evaluation_device='cpu')


if __name__ == '__main__':
    unittest.main()
