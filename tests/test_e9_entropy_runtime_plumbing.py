"""Bounded production-trainer integration; never launch a formal experiment."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.config import load_run_config
from src.models.ca_gat_mappo_checkpoint import load_checkpoint_payload
from src.models.ca_gat_mappo_trainer import CAGATMAPPOTrainer
from src.models.ca_gat_mappo_update import compute_configured_batched_ppo_objective_and_loss
from src.models.tx_upper_stability import TXUpperDeterministicStabilityPPOV1 as Stability
from tests.test_integrated_scheduler import CONFIGS, fixture, vector_buffer


class E9EntropyRuntimePlumbingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.e9 = load_run_config(CONFIGS / 'e9-v1-tx-deterministic-stability-seed43-32k.json')
        cls.e8 = load_run_config(CONFIGS / 'e8-v1-integrated-tx-deadline-cpu-seed43-32k.json')

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def bounded_config(self, base, directory, device='cpu', *, horizon=256,
                       episodes=1, interval=4096):
        # Keep the real E9 objective, model dimensions, seed and 32K schedule.
        # Bound collection by episode count; all artifacts are temporary fixtures.
        root = Path(directory)
        config = replace(base, environment=replace(base.environment, episode_horizon=horizon),
            training=replace(base.training, mappo=replace(base.training.mappo,
                training_device=device, max_training_episodes=episodes,
                checkpoint_interval_steps=interval)),
            output=replace(base.output, logs_dir=str(root / 'logs'),
                plots_dir=str(root / 'plots'), dashboard_logs_dir=str(root / 'dashboard')))
        config.validate()
        self.assertEqual(config.training.mappo.max_training_environment_steps, 32000)
        return config

    def capture_loss(self, records):
        def compute(**kwargs):
            loss = compute_configured_batched_ppo_objective_and_loss(**kwargs)
            stability = loss.flattened.tx_upper_stability
            records.append((kwargs.get('collected_environment_steps'),
                            None if stability is None else stability.beta))
            return loss
        return patch('src.models.ca_gat_mappo_update.compute_configured_batched_ppo_objective_and_loss',
                     side_effect=compute)

    def assert_epoch_steps(self, records, step):
        beta = Stability.entropy_coefficient(.01, step, 32000)
        self.assertEqual(len(records), 4)
        for actual_step, actual_beta in records:
            self.assertEqual(actual_step, step)
            self.assertAlmostEqual(actual_beta, beta, places=12)

    def real_collection_update(self, device):
        with tempfile.TemporaryDirectory() as directory:
            config = self.bounded_config(self.e9, directory, device)
            trainer = CAGATMAPPOTrainer(config, progress_logger=lambda _: None)
            before = {k: v.clone() for k, v in trainer.actor.tx_upper_private.state_dict().items()}
            records = []
            with self.capture_loss(records), patch.object(trainer.updater, 'update',
                    wraps=trainer.updater.update) as update, patch.object(
                    trainer.updater.optimizers.actor_optimizer, 'step',
                    wraps=trainer.updater.optimizers.actor_optimizer.step) as adam:
                result = trainer.train()
            self.assertEqual(result.total_environment_transitions, 256)
            self.assertEqual(result.ppo_update_count, 1)
            self.assertEqual(result.optimized_transitions, 256)
            self.assertEqual(update.call_args.kwargs, {'collected_environment_steps': 256})
            self.assert_epoch_steps(records, 256)
            self.assertEqual(adam.call_count, 4)
            self.assertEqual(trainer._transitions, 256)
            self.assertTrue(result.updates[0].output.old_policy_snapshot_preserved)
            self.assertTrue(any(not torch.equal(v, before[k])
                for k, v in trainer.actor.tx_upper_private.state_dict().items()))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                for p in trainer.actor.parameters()))
            self.assertTrue(all(p.device.type == device for p in trainer.actor.parameters()))
            summary_path = Path(config.artifact_paths()['raw_metrics']).parent / 'integrated_scheduler_updates_v1.jsonl'
            rows = [json.loads(s) for s in summary_path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['environment_steps'], 256)
            self.assertEqual([e['tx_upper']['beta_tx'] for e in rows[0]['ppo_epochs']], [.01]*4)

    def test_real_E9_trainer_collection_PPO_update_cpu(self):
        self.real_collection_update('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_real_E9_trainer_collection_PPO_update_cuda(self):
        self.real_collection_update('cuda')

    def test_E8_trainer_retains_no_step_legacy_call(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.bounded_config(self.e8, directory)
            trainer = CAGATMAPPOTrainer(config, progress_logger=lambda _: None)
            records = []
            with self.capture_loss(records), patch.object(trainer.updater, 'update',
                    wraps=trainer.updater.update) as update:
                result = trainer.train()
            self.assertEqual(result.ppo_update_count, 1)
            self.assertEqual(update.call_args.kwargs, {})
            self.assertEqual(records, [(None, None)]*4)

    def test_8192_trainer_counter_is_constant_across_four_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.bounded_config(self.e9, directory, horizon=500, interval=4000)
            seed, _, _ = fixture(config)
            # The reusable seed helper starts with scalar credit; E9 uses [A].
            seed = replace(seed, old_value=torch.zeros(config.environment.uav_count),
                           reward=torch.ones(config.environment.uav_count),
                           bootstrap_value=torch.zeros(config.environment.uav_count))
            trainer = CAGATMAPPOTrainer(config, progress_logger=lambda _: None)
            trainer.rollout_buffer = vector_buffer(config, seed, finalized=False)
            trainer._rollout_policy_version = trainer.policy_version
            # Synthetic late-rollout state, not a new 8K training experiment.
            trainer._transitions = 8192
            trainer._optimized = 7936
            records = []
            with self.capture_loss(records):
                trainer._perform_update([])
            self.assert_epoch_steps(records, 8192)
            self.assertAlmostEqual(records[0][1], .00988, places=12)
            self.assertEqual(trainer._transitions, 8192)
            self.assertEqual(trainer._optimized, 7936)

    def test_actual_partial_checkpoint_resume_continues_collected_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.bounded_config(self.e9, directory, horizon=128,
                                         episodes=250, interval=128)
            paused = CAGATMAPPOTrainer(config, progress_logger=lambda _: None)
            checkpoint = paused.train_until_periodic_checkpoint()
            saved = load_checkpoint_payload(checkpoint)
            self.assertEqual(saved['training_state']['collected_environment_transitions'], 128)
            self.assertEqual(saved['training_state']['optimized_transitions'], 0)
            self.assertEqual(saved['active_rollout_state']['rollout_length'], 128)
            resumed = CAGATMAPPOTrainer.resume_from_checkpoint(config, checkpoint)
            resumed._progress_logger = lambda _: None
            self.assertEqual(resumed._transitions, 128)
            self.assertEqual(resumed._optimized, 0)
            records = []
            with self.capture_loss(records):
                next_checkpoint = resumed.train_until_periodic_checkpoint()
            self.assert_epoch_steps(records, 256)
            self.assertEqual(resumed._transitions, 256)
            self.assertEqual(resumed._optimized, 256)
            self.assertEqual(next_checkpoint.name, 'step_256.pt')
            next_saved = load_checkpoint_payload(next_checkpoint)
            self.assertFalse(next_saved['training_state']['training_complete'])
            self.assertEqual(next_saved['training_state']['collected_environment_transitions'], 256)


if __name__ == '__main__':
    unittest.main()
