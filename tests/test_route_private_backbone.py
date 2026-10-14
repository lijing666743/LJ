"""E7 architecture contracts; unit fixtures only, no formal experiment."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from src.config import (
    CHECKPOINT_KIND_FINAL_COMPLETED, CHECKPOINT_KIND_PERIODIC_RESUME,
    ConfigError, RunConfig, load_run_config,
)
from src.evaluation.actor_loader import (
    EvaluationCheckpointError, load_final_actor_for_evaluation,
)
from src.models.ca_gat_mappo import (
    ACTION_BRANCH_ORDER, CAGATMAPPOActor, MAPPOCentralizedCritic, MAPPONetworkError,
)
from src.models.ca_gat_mappo_checkpoint import (
    atomic_save_checkpoint, load_checkpoint_payload, restore_active_rollout,
    restore_model_optimizer_and_rng, serialize_active_rollout,
)
from src.models.ca_gat_mappo_rollout import CAGATMAPPORolloutBuffer, RolloutStorageError
from src.models.ca_gat_mappo_update import (
    CAGATMAPPORecurrentPPOUpdater, build_ca_gat_mappo_optimizers,
    build_recurrent_ppo_minibatch,
)
from tests.test_ca_gat_mappo_checkpoint import (
    assert_nested_equal, make_checkpoint_config, make_payload,
)
from tests.test_ca_gat_mappo_update import make_seed_transition, make_synthetic_buffer
from tests.test_candidate_aware_route_decoder_v1 import differing_paths
from tests.test_route_head_lr_isolation import persisted_batch, ppo_loss


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'tests/fixtures/configs'
BACKBONES = ('self_encoder', 'public_encoder', 'graph_encoder', 'gru')


def private_config(config, enabled=True):
    result = replace(config, training=replace(config.training, mappo=replace(
        config.training.mappo, route_private_backbone=enabled)))
    for name in ('git_commit', 'git_branch', 'git_dirty'):
        object.__setattr__(result, name, getattr(config, name))
    return result


def gradient_sum(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


def extend_batch(batch, length=32):
    """Repeat persisted observations, with a real recurrent reset inside a chunk."""
    starts = batch.episode_starts.expand(1, length, -1).clone()
    starts.fill_(False)
    starts[:, 17, 1] = True
    return replace(
        batch,
        self_features=batch.self_features.expand(1, length, -1, -1).clone(),
        neighbor_public_features=batch.neighbor_public_features.expand(1, length, -1, -1, -1).clone(),
        edge_features=batch.edge_features.expand(1, length, -1, -1, -1).clone(),
        neighbor_mask=batch.neighbor_mask.expand(1, length, -1, -1).clone(),
        action_masks={k:v.expand(1, length, -1, -1).clone() for k,v in batch.action_masks.items()},
        action_indices={k:v.expand(1, length, -1).clone() for k,v in batch.action_indices.items()},
        episode_starts=starts,
    )


def one_slot(batch, slot):
    return replace(
        batch, self_features=batch.self_features[:, slot:slot+1],
        neighbor_public_features=batch.neighbor_public_features[:, slot:slot+1],
        edge_features=batch.edge_features[:, slot:slot+1],
        neighbor_mask=batch.neighbor_mask[:, slot:slot+1],
        action_masks={k:v[:, slot:slot+1] for k,v in batch.action_masks.items()},
        action_indices={k:v[:, slot:slot+1] for k,v in batch.action_indices.items()},
        episode_starts=batch.episode_starts[:, slot:slot+1],
    )


class RoutePrivateBackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.e6 = load_run_config(CONFIGS/'e6-v1-legacy-cpu-frequency-rollback-seed43-32k.json')
        cls.e7 = load_run_config(CONFIGS/'e7-v1-route-private-backbone-seed43-32k.json')
        cls.batch = persisted_batch(cls.e7)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_boolean_default_and_exact_canonical_diff(self):
        self.assertFalse(RunConfig().training.mappo.route_private_backbone)
        self.assertNotIn('route_private_backbone', self.e6.resolved_dict()['training']['mappo'])
        self.assertEqual(differing_paths(self.e6.resolved_dict(), self.e7.resolved_dict()),
                         {('training', 'mappo', 'route_private_backbone')})
        for value in (0, 1, 'true', None):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                private_config(self.e6, value).validate()
        self.assertEqual(self.e7.seed, 43)
        self.assertEqual(self.e7.training.mappo.max_training_environment_steps, 32000)
        self.assertEqual(self.e7.training.mappo.cpu_frequency_decoder_mode, 'legacy')
        self.assertEqual(self.e7.evaluation.evaluation_seeds, (1042,1043,1044,1045,1046))
        self.assertNotEqual(self.e6.run_id, self.e7.run_id)
        self.assertEqual(Path(self.e7.artifact_paths()['config_snapshot']).parent.name,self.e7.run_id)

    def test_initialization_equal_without_parameter_alias_or_rng_draw(self):
        before = torch.random.get_rng_state().clone()
        old, new = CAGATMAPPOActor(self.e6), CAGATMAPPOActor(self.e7)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
        for name in BACKBONES:
            shared, private = getattr(new, name), getattr(new, 'route_'+name)
            assert_nested_equal(self, shared.state_dict(), private.state_dict())
            for p,q in zip(shared.parameters(), private.parameters()):
                self.assertNotEqual(p.data_ptr(), q.data_ptr())
        assert_nested_equal(self, MAPPOCentralizedCritic(self.e6).state_dict(),
                            MAPPOCentralizedCritic(self.e7).state_dict())

    def test_forward_identity_at_initialization_all_seven_heads(self):
        with torch.no_grad():
            old = CAGATMAPPOActor(self.e6)(self.batch)
            new = CAGATMAPPOActor(self.e7)(self.batch)
        assert_nested_equal(self, old.raw_logits, new.raw_logits)
        torch.testing.assert_close(old.hidden_out, new.shared_hidden_out, rtol=0, atol=0)
        torch.testing.assert_close(old.hidden_out, new.route_hidden_out, rtol=0, atol=0)

    def check_route_gradient(self, device):
        actor = CAGATMAPPOActor(self.e7).to(device)
        batch = self.batch.to(device=device)
        out = actor(batch)
        logits = out.raw_logits['route'].masked_fill(~batch.action_masks['route'], -torch.inf)
        (-torch.log_softmax(logits, -1)[..., 1].mean()).backward()
        for name in BACKBONES:
            self.assertGreater(gradient_sum(getattr(actor, 'route_'+name)), 0, name)
            self.assertTrue(all(p.grad is None for p in getattr(actor, name).parameters()), name)
        self.assertGreater(gradient_sum(actor.route_decoder), 0)
        for name,p in actor.named_parameters():
            if not name.startswith(actor.route_parameter_prefixes):
                self.assertIsNone(p.grad, name)
            elif p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_route_policy_gradient_isolation_cpu(self):
        self.check_route_gradient('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_route_policy_gradient_isolation_cuda(self):
        self.check_route_gradient('cuda')

    def check_other_gradient(self, device):
        # All-option categorical unit losses exercise active TX and CPU graphs;
        # no environment legality or training result is claimed by this fixture.
        for branch in ACTION_BRANCH_ORDER[1:]:
            actor = CAGATMAPPOActor(self.e7).to(device)
            batch = self.batch.to(device=device)
            batch = replace(batch, action_masks={k:(v if k=='route' else torch.ones_like(v)) for k,v in batch.action_masks.items()},
                            action_indices={k:(torch.ones_like(v) if k=='cpu_queue' else v) for k,v in batch.action_indices.items()})
            out = actor(batch)
            (-torch.log_softmax(out.raw_logits[branch], -1)[..., 1].mean()).backward()
            for name in BACKBONES:
                self.assertGreater(gradient_sum(getattr(actor, name)), 0, (branch,name))
                self.assertTrue(all(p.grad is None for p in getattr(actor,'route_'+name).parameters()))
            self.assertTrue(all(p.grad is None for p in actor.route_decoder.parameters()))
            self.assertGreater(gradient_sum(actor.action_heads[branch]), 0)

    def test_tx_cpu_policy_gradient_isolation_cpu(self):
        self.check_other_gradient('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_tx_cpu_policy_gradient_isolation_cuda(self):
        self.check_other_gradient('cuda')

    def test_optimizer_slow_complete_route_subsystem_and_fast_other(self):
        actor,critic = CAGATMAPPOActor(self.e7), MAPPOCentralizedCritic(self.e7)
        bundle = build_ca_gat_mappo_optimizers(actor, critic, self.e7)
        route,other = bundle.actor_optimizer.param_groups
        expected = {id(p) for n,p in actor.named_parameters() if n.startswith(actor.route_parameter_prefixes)}
        self.assertEqual({id(p) for p in route['params']}, expected)
        self.assertEqual(sum(p.numel() for p in route['params']), 265228)
        self.assertAlmostEqual(route['lr'], 3e-5)
        self.assertEqual(other['lr'], 3e-4)
        self.assertEqual(bundle.critic_optimizer.param_groups[0]['lr'], 3e-4)
        for name in BACKBONES:
            self.assertTrue({id(p) for p in getattr(actor,name).parameters()} <= {id(p) for p in other['params']})
        bundle.validate(actor,critic,self.e7)

    def test_shared_weight_changes_do_not_change_route_or_private_hidden(self):
        actor = CAGATMAPPOActor(self.e7)
        baseline = actor(self.batch)
        with torch.no_grad():
            for name in BACKBONES:
                for p in getattr(actor, name).parameters(): p.add_(.1)
        changed = actor(self.batch)
        torch.testing.assert_close(baseline.raw_logits['route'],changed.raw_logits['route'],rtol=0,atol=0)
        torch.testing.assert_close(baseline.route_hidden_out,changed.route_hidden_out,rtol=0,atol=0)
        self.assertFalse(torch.equal(baseline.shared_hidden_out,changed.shared_hidden_out))

    def test_private_weight_changes_do_not_change_other_logits_or_shared_hidden(self):
        actor = CAGATMAPPOActor(self.e7)
        baseline = actor(self.batch)
        with torch.no_grad():
            for name in BACKBONES:
                for p in getattr(actor, 'route_'+name).parameters(): p.add_(.1)
        changed = actor(self.batch)
        for branch in ACTION_BRANCH_ORDER[1:]:
            torch.testing.assert_close(baseline.raw_logits[branch],changed.raw_logits[branch],rtol=0,atol=0)
        torch.testing.assert_close(baseline.shared_hidden_out,changed.shared_hidden_out,rtol=0,atol=0)
        self.assertFalse(torch.equal(baseline.raw_logits['route'],changed.raw_logits['route']))

    def test_hidden_lanes_independent_and_reset_both(self):
        actor = CAGATMAPPOActor(self.e7)
        batch = replace(self.batch,episode_starts=torch.zeros_like(self.batch.episode_starts))
        hidden = actor.initial_hidden(1)
        h = actor.spec.gru_hidden_dimension
        with torch.no_grad():
            base = actor(batch,hidden)
            shared = hidden.clone();shared[...,:h]=.7
            route = hidden.clone();route[...,h:]=-.4
            shared_out,route_out = actor(batch,shared),actor(batch,route)
            torch.testing.assert_close(base.raw_logits['route'],shared_out.raw_logits['route'],rtol=0,atol=0)
            for b in ACTION_BRANCH_ORDER[1:]:
                torch.testing.assert_close(base.raw_logits[b],route_out.raw_logits[b],rtol=0,atol=0)
            self.assertFalse(torch.equal(base.route_hidden_out,route_out.route_hidden_out))
            self.assertFalse(torch.equal(base.shared_hidden_out,shared_out.shared_hidden_out))
            reset = actor(self.batch,torch.cat((shared[...,:h],route[...,h:]),-1))
            zero = actor(self.batch,actor.initial_hidden(1))
            assert_nested_equal(self,reset.raw_logits,zero.raw_logits)
            torch.testing.assert_close(reset.hidden_out,zero.hidden_out,rtol=0,atol=0)
        with self.assertRaises(MAPPONetworkError):
            actor(batch,hidden[...,:h])

    def test_32_slot_replay_matches_stepwise_both_lanes_and_mid_chunk_reset(self):
        actor = CAGATMAPPOActor(self.e7).eval()
        batch = extend_batch(self.batch)
        hidden = actor.initial_hidden(1)
        h = actor.spec.gru_hidden_dimension
        hidden[...,:h]=.2;hidden[...,h:]=-.3
        with torch.no_grad():
            full = actor(batch,hidden)
            current = hidden.clone();parts=[]
            for t in range(32):
                part=actor(one_slot(batch,t),current);parts.append(part);current=part.hidden_out
        for branch in ACTION_BRANCH_ORDER:
            torch.testing.assert_close(full.raw_logits[branch],torch.cat([p.raw_logits[branch] for p in parts],1),rtol=2e-6,atol=2e-6)
        torch.testing.assert_close(full.hidden_out,current,rtol=2e-6,atol=2e-6)
        self.assertTrue(torch.equal(hidden[...,:h],torch.full_like(hidden[...,:h],.2)))
        self.assertTrue(torch.equal(hidden[...,h:],torch.full_like(hidden[...,h:],-.3)))

    def test_rollout_storage_checkpoint_roundtrip_and_recurrent_ppo_8_by_32(self):
        config=replace(self.e7, environment=replace(self.e7.environment,arrival_probabilities=(1.0,)*4),
                       training=replace(self.e7.training,mappo=replace(
                           self.e7.training.mappo,training_device='cpu')))
        seed=make_seed_transition(config)
        # Existing helper emits scalar credit; make a labelled vector fixture
        # to test the actual E7 responsibility_terminal / role-decomposed path.
        seed=replace(seed,old_value=seed.old_value.expand(4).clone(),
                     reward=seed.reward.expand(4).clone(),
                     bootstrap_value=seed.bootstrap_value.expand(4).clone())
        h=seed.spec.gru_hidden_dimension
        saved=seed.hidden_in.clone();saved[...,:h]=.2;saved[...,h:]=-.4
        seed=replace(seed,hidden_in=saved)
        partial=CAGATMAPPORolloutBuffer(config)
        for index in range(3):
            partial.append(replace(seed,slot=index,episode_start=index==0,
                                   hidden_in=torch.zeros_like(saved) if index==0 else saved*(index+1)))
        state=serialize_active_rollout(partial,0,CHECKPOINT_KIND_PERIODIC_RESUME)
        recovered,_=restore_active_rollout(config,state,checkpoint_kind=CHECKPOINT_KIND_PERIODIC_RESUME,policy_version=0)
        torch.testing.assert_close(partial.transition_at(1).hidden_in,recovered.transition_at(1).hidden_in,rtol=0,atol=0)
        full=CAGATMAPPORolloutBuffer(config)
        for index in range(256):
            owned=torch.full_like(saved,.001*(index+1));owned[...,h:]*=-2
            if index==0:owned.zero_()
            full.append(replace(seed,slot=index,episode_start=index==0,hidden_in=owned))
        full.finalize()
        minibatch=build_recurrent_ppo_minibatch(full,config)
        self.assertEqual(tuple(minibatch.initial_hidden.shape),(8,4,256))
        torch.testing.assert_close(minibatch.initial_hidden,minibatch.stored_hidden_in[:,0],rtol=0,atol=0)
        actor,critic=CAGATMAPPOActor(config),MAPPOCentralizedCritic(config)
        updater=CAGATMAPPORecurrentPPOUpdater(actor,critic,config)
        replay=updater.evaluate_minibatch(minibatch)
        self.assertEqual(tuple(replay.policy.hidden_out.shape),(8,4,256))
        self.assertTrue(torch.isfinite(replay.policy.joint_log_prob).all())
        changed=minibatch.initial_hidden.clone();changed[...,h:]+=.2
        other=updater.action_distribution.evaluate_actions(minibatch.actor_batch,minibatch.action_mask_batch,minibatch.proposals,changed)
        for b in ACTION_BRANCH_ORDER[1:]:
            torch.testing.assert_close(replay.policy.raw_logits[b],other.raw_logits[b],rtol=0,atol=0)
        # Chunk 1 starts mid-episode, so changing its private hidden affects Route.
        self.assertFalse(torch.equal(replay.policy.raw_logits['route'][1],other.raw_logits['route'][1]))
        with self.assertRaises(RolloutStorageError):
            replace(seed,hidden_in=seed.hidden_in[...,:h])

    def test_e7_optimizer_moments_checkpoint_and_actor_only_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            config=private_config(make_checkpoint_config(Path(directory)))
            config=replace(config,training=replace(config.training,mappo=replace(
                config.training.mappo,route_head_lr_scale=.1)))
            for name in ('git_commit','git_branch','git_dirty'):
                object.__setattr__(config,name,getattr(self.e7,name))
            payload,actor,critic,updater=make_payload(config,CHECKPOINT_KIND_PERIODIC_RESUME)
            # Controlled optimizer step in a unit fixture creates genuine Adam
            # moments; it is not an environment/PPO training run or resume.
            sum(p.square().sum() for p in actor.parameters()).backward()
            updater.optimizers.actor_optimizer.step()
            payload['model_state']['actor']={k:v.detach().clone() for k,v in actor.state_dict().items()}
            payload['optimizer_state']['actor_adam']=copy.deepcopy(updater.optimizers.actor_optimizer.state_dict())
            path=Path(directory)/'unit-periodic.pt';atomic_save_checkpoint(payload,path)
            read=load_checkpoint_payload(path)
            target,tc=CAGATMAPPOActor(config),MAPPOCentralizedCritic(config)
            optimizers=build_ca_gat_mappo_optimizers(target,tc,config)
            restore_model_optimizer_and_rng(config=config,payload=read,actor=target,critic=tc,optimizers=optimizers,dtype=torch.float32)
            assert_nested_equal(self,actor.state_dict(),target.state_dict())
            assert_nested_equal(self,updater.optimizers.actor_optimizer.state_dict(),optimizers.actor_optimizer.state_dict())
            self.assertTrue(any(k.startswith('route_gru.') for k in read['model_state']['actor']))
            for r in read['active_rollout_state']['ordered_transitions']:
                self.assertEqual(r['hidden_in'].shape[-1],16)
            final,*_=make_payload(self.e7,CHECKPOINT_KIND_FINAL_COMPLETED)
            fp=Path(directory)/'unit-final.pt';atomic_save_checkpoint(final,fp)
            evaluation=replace(self.e7,mode='evaluation',launch_profile=None)
            for name in ('git_commit','git_branch','git_dirty'):
                object.__setattr__(evaluation,name,getattr(self.e7,name))
            loaded=load_final_actor_for_evaluation(evaluation,fp,evaluation_device='cpu')
            self.assertTrue(loaded.actor.route_private_backbone)
            self.assertEqual(loaded.actor.initial_hidden(1).shape[-1],256)
            with self.assertRaisesRegex(EvaluationCheckpointError,'route_private_backbone'):
                load_final_actor_for_evaluation(private_config(evaluation,False),fp,evaluation_device='cpu')


    def test_unchanged_configured_ppo_loss_at_initialization(self):
        results=[]
        for config in (self.e6,self.e7):
            actor,critic=CAGATMAPPOActor(config),MAPPOCentralizedCritic(config)
            output,loss=ppo_loss(actor,critic,config,self.batch)
            loss.total_loss.backward()
            results.append((output,loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None))
        torch.testing.assert_close(results[0][1].total_loss,results[1][1].total_loss,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
