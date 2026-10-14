"""E5-v1 optimizer tests only: no formal training or evaluation episodes."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.config import (
    CHECKPOINT_KIND_FINAL_COMPLETED, CHECKPOINT_KIND_PERIODIC_RESUME,
    ConfigError, RunConfig, load_run_config,
)
from src.evaluation.actor_loader import load_final_actor_for_evaluation
from src.evaluation.e4_defer_localization import encode_bank
from src.models.ca_gat_mappo import (
    ACTION_BRANCH_ORDER, ActorTensorBatch, CAGATMAPPOActor,
    CentralizedStateTensorBatch, MAPPOCentralizedCritic,
)
from src.models.ca_gat_mappo_checkpoint import (
    CheckpointError, _normalize_legacy_actor_optimizer_state, _validate_optimizer_state,
    atomic_save_checkpoint, restore_model_optimizer_and_rng, validate_periodic_checkpoint_compatibility,
)
from src.models.ca_gat_mappo_ppo import compute_configured_ppo_objective_and_loss
from src.models.ca_gat_mappo_update import (
    RecurrentPPOUpdateError, build_ca_gat_mappo_optimizers,
)
from tests.test_ca_gat_mappo_checkpoint import (
    assert_nested_equal, make_checkpoint_config, make_payload,
)
from tests.test_candidate_aware_route_decoder_v1 import differing_paths

ROOT = Path(__file__).resolve().parents[1]
E4 = ROOT / 'tests/fixtures/configs/e4-v1-candidate-cpu-frequency-seed43-32k.json'
E5 = E4.with_name('e5-v1-route-head-lr-isolation-seed43-32k.json')
RUN4 = ROOT / 'logs/gate_m/ca_gat_mappo__remote_stress_v2__seed-43__cfg-a216fc6b1306__git-21624c6cd274'


def scaled(config, scale):
    result = replace(config, training=replace(config.training, mappo=replace(
        config.training.mappo, route_head_lr_scale=scale)))
    for name in ('git_commit', 'git_branch', 'git_dirty'):
        object.__setattr__(result, name, getattr(config, name))
    return result


def adam(parameters, config, lr):
    m = config.training.mappo
    return torch.optim.Adam(parameters, lr=lr, betas=(m.adam_beta1,m.adam_beta2),
                            eps=m.adam_eps, weight_decay=m.weight_decay)


def controlled_loss(actor):
    """Linear unit-test objective gives identical nonzero gradients every step."""
    return sum(parameter.sum() * (0.001 + i * .0001)
               for i, parameter in enumerate(actor.parameters()))


def persisted_batch(config):
    records = [r for line in (ROOT/'tests/fixtures/state_bank_route.jsonl').read_text().splitlines()
               if (r := json.loads(line))['branch']=='route']
    rows = [next(r for r in records if r['uav_id']==i) for i in range(4)]
    encoded = encode_bank(rows, config)
    masks = {b:torch.tensor([r['observation']['action_masks']['masks'][b] for r in rows],
                           dtype=torch.bool)[None,None] for b in ACTION_BRANCH_ORDER}
    return ActorTensorBatch(
        encoded['self'][None,None], encoded['public'][None,None],
        encoded['edge'][None,None], encoded['neighbor'][None,None], masks,
        {b:m.long().argmax(-1) for b,m in masks.items()}, torch.ones((1,1,4),dtype=torch.bool),
    )


def ppo_loss(actor, critic, config, batch):
    output = actor(batch)
    logps, entropies, active = [], [], []
    for branch in ACTION_BRANCH_ORDER:
        mask = output.action_masks[branch][0]
        logits = output.raw_logits[branch][0].masked_fill(~mask,-1.e9)
        lp = torch.log_softmax(logits,-1)
        logps.append(lp.gather(-1,batch.action_indices[branch][0].unsqueeze(-1)).squeeze(-1))
        entropies.append(-(lp.exp()*lp).sum(-1))
        active.append(mask.sum(-1)>1)
    branch_lp = torch.stack(logps,-1)
    joint = branch_lp.sum(-1)
    value = critic(CentralizedStateTensorBatch(batch.self_features.new_zeros(
        (1,1,critic.spec.centralized_state_dim))))[0]
    loss = compute_configured_ppo_objective_and_loss(
        new_joint_log_prob=joint, old_joint_log_prob=joint.detach(),
        advantage=torch.ones_like(joint), current_value=value,
        return_target=torch.ones_like(value), entropy=sum(entropies),
        sequence_valid_mask=torch.ones(1,dtype=torch.bool,device=joint.device), config=config,
        new_branch_log_probs=branch_lp, old_branch_log_probs=branch_lp.detach(),
        active_branch_indicators=torch.stack(active,-1),
    )
    return output,loss


class RouteHeadLrIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(1)
        cls.e4=load_run_config(E4)
        cls.e5=load_run_config(E5)
        cls.batch=persisted_batch(cls.e4)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_default_one_and_historical_hash_identity(self):
        self.assertEqual(RunConfig().training.mappo.route_head_lr_scale,1.)
        self.assertEqual(self.e4.config_hash,'a216fc6b130686222b3bbd1bd19bada399d5be7043e007f0df37b8dd09e880a2')
        self.assertEqual(self.e4.config_hash,scaled(self.e4,1.).config_hash)
        self.assertNotIn('route_head_lr_scale',self.e4.resolved_dict()['training']['mappo'])

    def test_invalid_scale_is_rejected(self):
        for value in (0.,-1.,float('nan'),float('inf'),True,False,'0.1',None):
            with self.subTest(scale=value),self.assertRaisesRegex(ConfigError,'route_head_lr_scale'):
                scaled(self.e4,value).validate()
        with self.assertRaises(ConfigError):
            load_run_config(E4, cli_overrides={'training.mappo.route_head_lr_scale':True})

    def test_matched_config_has_exactly_one_difference(self):
        self.assertEqual(differing_paths(self.e4.resolved_dict(),self.e5.resolved_dict()),
                         {('training','mappo','route_head_lr_scale')})
        self.assertEqual(self.e5.training.mappo.cpu_frequency_decoder_mode,'candidate_aware_v1')
        self.assertEqual(self.e5.seed,43)
        self.assertEqual(self.e5.training.mappo.max_training_environment_steps,32000)
        self.assertEqual(self.e5.evaluation.evaluation_seeds,(1042,1043,1044,1045,1046))

    def test_full_route_partition_and_unscaled_other_heads(self):
        actor=CAGATMAPPOActor(self.e5); critic=MAPPOCentralizedCritic(self.e5)
        bundle=build_ca_gat_mappo_optimizers(actor,critic,self.e5)
        groups=bundle.actor_optimizer.param_groups
        self.assertEqual([g['group_name'] for g in groups],['route_head','actor_other'])
        self.assertEqual([g['lr'] for g in groups],[self.e5.training.mappo.actor_learning_rate*.1,3.e-4])
        expected={id(p) for n,p in actor.named_parameters() if n.startswith('action_heads.route.')}
        self.assertEqual({id(p) for p in groups[0]['params']},expected)
        self.assertEqual(expected,{id(p) for p in actor.route_decoder.parameters()})
        self.assertEqual(sum(p.numel() for p in groups[0]['params']),53772)
        ids=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in actor.parameters() if p.requires_grad})
        other={id(p) for p in groups[1]['params']}
        for n,p in actor.named_parameters():
            if not n.startswith('action_heads.route.'): self.assertIn(id(p),other,n)
        self.assertEqual(bundle.critic_optimizer.param_groups[0]['lr'],self.e5.training.mappo.critic_learning_rate)

    def test_partition_corruption_and_wrong_lr_fail_closed(self):
        actor=CAGATMAPPOActor(self.e5); critic=MAPPOCentralizedCritic(self.e5)
        for mode in ('duplicate','missing','swapped','lr'):
            bundle=build_ca_gat_mappo_optimizers(actor,critic,self.e5)
            route,other=bundle.actor_optimizer.param_groups
            if mode=='duplicate': other['params'].append(route['params'][0])
            if mode=='missing': route['params'].pop()
            if mode=='swapped': route['params'][0],other['params'][0]=other['params'][0],route['params'][0]
            if mode=='lr': other['lr'] *= .1
            with self.subTest(mode=mode),self.assertRaises(RecurrentPPOUpdateError):
                bundle.validate(actor,critic,self.e5)

    def check_forward_backward(self,device):
        actors=[CAGATMAPPOActor(c).to(device) for c in (self.e4,self.e5)]
        critics=[MAPPOCentralizedCritic(c).to(device) for c in (self.e4,self.e5)]
        assert_nested_equal(self,actors[0].state_dict(),actors[1].state_dict())
        assert_nested_equal(self,critics[0].state_dict(),critics[1].state_dict())
        results=[]
        for a,c,cfg in zip(actors,critics,(self.e4,self.e5)):
            result,loss=ppo_loss(a,c,cfg,self.batch.to(device=device))
            loss.total_loss.backward()
            results.append((result,loss))
        for b in ACTION_BRANCH_ORDER:
            self.assertTrue(torch.equal(results[0][0].raw_logits[b],results[1][0].raw_logits[b]))
        self.assertTrue(torch.equal(results[0][1].total_loss,results[1][1].total_loss))
        for modules in (actors,critics):
            for (n,p),(m,q) in zip(modules[0].named_parameters(),modules[1].named_parameters()):
                self.assertEqual(n,m)
                self.assertEqual(p.grad is None,q.grad is None,n)
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all(),n)
                    self.assertTrue(torch.equal(p.grad,q.grad),n)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in actors[0].route_decoder.parameters()
                               if p.grad is not None),0)

    def test_initial_forward_ppo_loss_gradients_identical_cpu(self):
        self.check_forward_backward('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_initial_forward_ppo_loss_gradients_identical_cuda(self):
        self.check_forward_backward('cuda')

    def check_controlled_adam(self,device):
        baseline=CAGATMAPPOActor(self.e4).to(device)
        unit=copy.deepcopy(baseline); isolated=copy.deepcopy(baseline); reference=copy.deepcopy(baseline)
        critic=MAPPOCentralizedCritic(self.e4).to(device)
        legacy=adam(baseline.parameters(),self.e4,3.e-4)
        equal=build_ca_gat_mappo_optimizers(unit,critic,self.e4).actor_optimizer
        scaled_opt=build_ca_gat_mappo_optimizers(isolated,critic,self.e5).actor_optimizer
        ref_route=adam(reference.route_decoder.parameters(),self.e4,self.e4.training.mappo.actor_learning_rate*.1)
        ref_other=adam([p for n,p in reference.named_parameters() if not n.startswith('action_heads.route.')],self.e4,3.e-4)
        before={n:p.detach().clone() for n,p in baseline.named_parameters()}
        for _ in range(3):
            for actor in (baseline,unit,isolated,reference):
                actor.zero_grad(set_to_none=True)
                controlled_loss(actor).backward()
                torch.nn.utils.clip_grad_norm_(tuple(actor.parameters()),self.e4.training.mappo.gradient_clip_norm)
            for opt in (legacy,equal,scaled_opt,ref_route,ref_other): opt.step()
        for (n,p),(m,q),(j,r),(k,ref) in zip(baseline.named_parameters(),unit.named_parameters(),
                                            isolated.named_parameters(),reference.named_parameters()):
            self.assertEqual((n,m,j,k),(n,n,n,n))
            torch.testing.assert_close(p,q,rtol=0,atol=0)
            torch.testing.assert_close(r,ref,rtol=0,atol=0)
            if not n.startswith('action_heads.route.'):
                torch.testing.assert_close(p,r,rtol=0,atol=0)
        names=[n for n in before if n.startswith('action_heads.route.')]
        a=dict(baseline.named_parameters()); b=dict(isolated.named_parameters())
        delta_a=torch.cat([(a[n]-before[n]).flatten() for n in names])
        delta_b=torch.cat([(b[n]-before[n]).flatten() for n in names])
        self.assertGreater(float(delta_b.norm()),0)
        self.assertLess(float(delta_b.norm()),float(delta_a.norm()))
        self.assertGreater(float(torch.dot(delta_a,delta_b)),0)

    def test_scale_one_legacy_equivalence_and_point_one_adam_reference_cpu(self):
        self.check_controlled_adam('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_scale_one_legacy_equivalence_and_point_one_adam_reference_cuda(self):
        self.check_controlled_adam('cuda')


    def test_point_one_checkpoint_resume_with_moments_and_final_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=scaled(make_checkpoint_config(Path(directory),horizon=4,interval=4,budget=8),.1)
            # Direct RunConfig unit fixtures have no repository provenance;
            # actor-only loaders require the actual current repository identity.
            for name in ('git_commit','git_branch','git_dirty'):
                object.__setattr__(cfg,name,getattr(self.e4,name))
            payload,actor,critic,updater=make_payload(cfg,CHECKPOINT_KIND_PERIODIC_RESUME)
            controlled_loss(actor).backward(); updater.optimizers.actor_optimizer.step()
            payload['model_state']['actor']={k:v.detach().clone() for k,v in actor.state_dict().items()}
            payload['optimizer_state']['actor_adam']=copy.deepcopy(updater.optimizers.actor_optimizer.state_dict())
            # Small CPU unit fixture only; never a formal training checkpoint.
            target_actor=CAGATMAPPOActor(cfg); target_critic=MAPPOCentralizedCritic(cfg)
            target=build_ca_gat_mappo_optimizers(target_actor,target_critic,cfg)
            restore_model_optimizer_and_rng(config=cfg,payload=payload,actor=target_actor,critic=target_critic,
                                            optimizers=target,dtype=torch.float32)
            assert_nested_equal(self,target.actor_optimizer.state_dict(),payload['optimizer_state']['actor_adam'])
            assert_nested_equal(self,actor.state_dict(),target_actor.state_dict())
            final,*_=make_payload(cfg,CHECKPOINT_KIND_FINAL_COMPLETED)
            path=Path(directory)/'final.pt'; atomic_save_checkpoint(final,path)
            eval_cfg=replace(cfg,mode='evaluation')
            for n in ('git_commit','git_branch','git_dirty'): object.__setattr__(eval_cfg,n,getattr(cfg,n))
            loaded=load_final_actor_for_evaluation(eval_cfg,path,evaluation_device='cpu')
            assert_nested_equal(self,loaded.actor.state_dict(),final['model_state']['actor'])

    def test_old_single_group_adam_migration_preserves_named_moments_and_continuation(self):
        actor=CAGATMAPPOActor(self.e4); critic=MAPPOCentralizedCritic(self.e4)
        legacy=adam(actor.parameters(),self.e4,3.e-4)
        controlled_loss(actor).backward(); legacy.step(); actor.zero_grad(set_to_none=True)
        reference=copy.deepcopy(actor)
        groups=build_ca_gat_mappo_optimizers(reference,critic,self.e4)
        old=copy.deepcopy(legacy.state_dict()); preserved=copy.deepcopy(old)
        converted=_normalize_legacy_actor_optimizer_state(groups.actor_optimizer,old,groups.actor_parameters,self.e4)
        _validate_optimizer_state(groups.actor_optimizer,converted,groups.actor_parameters,'actor_adam')
        groups.actor_optimizer.load_state_dict(converted)
        assert_nested_equal(self,old,preserved)
        for p,q in zip(actor.parameters(),reference.parameters()):
            assert_nested_equal(self,legacy.state[p],groups.actor_optimizer.state[q])
        for a,o in ((actor,legacy),(reference,groups.actor_optimizer)):
            a.zero_grad(set_to_none=True); controlled_loss(a).backward(); o.step()
        assert_nested_equal(self,actor.state_dict(),reference.state_dict())
        with self.assertRaisesRegex(CheckpointError,'route_head_lr_scale=1.0'):
            _normalize_legacy_actor_optimizer_state(groups.actor_optimizer,old,groups.actor_parameters,self.e5)



if __name__=='__main__': unittest.main()
