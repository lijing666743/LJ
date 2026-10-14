"""E10 bounded synthetic credit/replay/checkpoint tests; no formal runs."""
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.config import ConfigError, load_run_config
from src.models.ca_gat_mappo import CAGATMAPPOActor, MAPPOCentralizedCritic, CentralizedStateTensorBatch
from src.models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution
from src.models.ca_gat_mappo_checkpoint import (
    atomic_save_checkpoint, load_checkpoint_payload, restore_model_optimizer_and_rng,
    serialize_active_rollout, restore_active_rollout,
)
from src.models.ca_gat_mappo_rollout import CAGATMAPPORolloutTransition
from src.models.ca_gat_mappo_update import CAGATMAPPORecurrentPPOUpdater, build_ca_gat_mappo_optimizers
from src.models.ca_gat_mappo_ppo import compute_configured_ppo_objective_and_loss, PPOObjectiveError
from src.models.tx_upper_counterfactual import (
    TXUpperCounterfactualCriticV1, counterfactual_credit, selected_action_q_loss, team_return_target,
)
from tests.test_integrated_scheduler import fixture, vector_buffer, CONFIGS, ROOT
from tests.test_route_private_backbone import gradient_sum
from tests.test_tx_upper_stability import preserve_git


class TXCounterfactualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.config = load_run_config(CONFIGS/'e10-v1-tx-counterfactual-credit-seed43-32k.json')
        cls.e9 = load_run_config(CONFIGS/'e9-v1-tx-deterministic-stability-seed43-32k.json')
        cls.e8 = load_run_config(CONFIGS/'e8-v1-integrated-tx-deadline-cpu-seed43-32k.json')
        (Path(tempfile.gettempdir()) / 'e10-v1-validation').mkdir(exist_ok=True)
        _, cls.batch, cls.masks = fixture(cls.config)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self, device='cpu'):
        return CAGATMAPPOActor(self.config).to(device), MAPPOCentralizedCritic(self.config).to(device)

    def toy(self, q=None, p=None, actions=None, active=None):
        q = torch.tensor([[0.,2.],[0.,2.]]) if q is None else q
        p = torch.full_like(q,.5) if p is None else p
        actions = torch.tensor([True,False]) if actions is None else actions
        active = torch.ones(q.shape[:-1],dtype=torch.bool) if active is None else active
        return counterfactual_credit(q,p,actions,active)

    def test_exact_counterfactual_credit_example(self):
        credit=self.toy()
        torch.testing.assert_close(credit.baseline,torch.ones(2),rtol=0,atol=0)
        torch.testing.assert_close(credit.raw_advantage,torch.tensor([1.,-1.]),rtol=0,atol=0)
        torch.testing.assert_close(credit.normalized_advantage,credit.raw_advantage,rtol=0,atol=0)
        torch.testing.assert_close(credit.selected_q,torch.tensor([2.,0.]),rtol=0,atol=0)

    def test_equal_q_both_action_advantages_zero(self):
        c=self.toy(q=torch.ones(2,2));self.assertTrue((c.raw_advantage==0).all())
        self.assertTrue((c.normalized_advantage==0).all())

    def test_probability_weighted_baseline_and_ordering(self):
        c=self.toy(p=torch.tensor([[.2,.8],[.2,.8]]))
        torch.testing.assert_close(c.baseline,torch.tensor([1.6,1.6]))
        self.assertGreater(c.raw_advantage[0],c.raw_advantage[1])
        torch.testing.assert_close((torch.tensor([.8,.2])*c.raw_advantage).sum(),torch.tensor(0.),atol=1e-6,rtol=0)

    def test_policy_and_q_detached_in_credit(self):
        q=torch.tensor([[0.,2.],[0.,2.]],requires_grad=True)
        logits=torch.zeros_like(q,requires_grad=True)
        c=self.toy(q=q,p=logits.softmax(-1))
        for x in [c.q,c.selected_q,c.baseline,c.raw_advantage,c.normalized_advantage]: self.assertFalse(x.requires_grad)

    def test_action_label_symmetry_including_normalization(self):
        q=torch.tensor([[.1,2.],[3.,-1.],[.5,1.]])
        p=torch.tensor([[.2,.8],[.7,.3],[.5,.5]])
        actions=torch.tensor([1,0,1],dtype=torch.bool)
        x=self.toy(q=q,p=p,actions=actions)
        y=self.toy(q=q.flip(-1),p=p.flip(-1),actions=~actions)
        for name in ['selected_q','baseline','raw_advantage','normalized_advantage']:
            torch.testing.assert_close(getattr(x,name),getattr(y,name),rtol=0,atol=0)

    def test_active_mask_excludes_nonfinite_inactive_values(self):
        c=self.toy(q=torch.tensor([[0.,2.],[float('nan'),float('nan')]]),
                   p=torch.tensor([[.5,.5],[float('nan'),float('nan')]]),active=torch.tensor([1,0],dtype=torch.bool))
        self.assertEqual(c.raw_advantage[0],1);self.assertEqual(c.raw_advantage[1],0)
        self.assertTrue((c.normalized_advantage==0).all())
        json.dumps(c.telemetry(),allow_nan=False)

    def test_selected_action_only_huber_gradient_and_target_detach(self):
        q=torch.tensor([[0.,2.],[0.,2.],[5.,6.]],requires_grad=True)
        target=torch.tensor([4.,4.,1000.],requires_grad=True)
        mask=torch.tensor([1,1,0],dtype=torch.bool);actions=torch.tensor([1,0,1],dtype=torch.bool)
        loss=selected_action_q_loss(q,actions,target,mask);loss.backward()
        torch.testing.assert_close(q.grad,torch.tensor([[0.,-.5],[-.5,0.],[0.,0.]]),rtol=0,atol=0)
        self.assertIsNone(target.grad)
        expected=torch.nn.functional.smooth_l1_loss(torch.tensor([2.,0.]),torch.tensor([4.,4.]))
        torch.testing.assert_close(loss,expected,rtol=0,atol=0)

    def test_empty_cohort_has_no_q_graph_or_credit(self):
        q=torch.ones(2,2,requires_grad=True);mask=torch.zeros(2,dtype=torch.bool)
        c=self.toy(q=q,active=mask)
        self.assertTrue((c.raw_advantage==0).all())
        loss=selected_action_q_loss(q,mask,torch.zeros(2),mask)
        self.assertEqual(loss,0);self.assertFalse(loss.requires_grad)

    def test_q_shape_and_original_v_unchanged(self):
        actor,critic=self.model();old=MAPPOCentralizedCritic(self.e9)
        features=torch.zeros(8,32,critic.spec.centralized_state_dim)
        self.assertEqual(critic.tx_upper_q(features).shape,(8,32,4,2))
        self.assertIsInstance(critic.tx_upper_q,TXUpperCounterfactualCriticV1)
        torch.testing.assert_close(critic(features),old(features),rtol=0,atol=0)
        for name,p in old.state_dict().items(): torch.testing.assert_close(critic.state_dict()[name],p,rtol=0,atol=0)
        self.assertEqual(set(actor.state_dict()),set(CAGATMAPPOActor(self.e9).state_dict()))

    def test_team_target_exact_conservation_and_boundary_semantics(self):
        from src.models.ca_gat_mappo_gae import compute_per_agent_gae_and_returns,compute_gae_and_returns
        rewards=torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.]])
        values=torch.tensor([[.1,.2],[.3,.4],[.5,.6],[.7,.8]])
        next_values=torch.full_like(values,20.)
        term=torch.tensor([0,1,0,0],dtype=torch.bool);trunc=torch.tensor([0,0,1,0],dtype=torch.bool)
        boundary=term|trunc
        kwargs=dict(terminated=term,truncated=trunc,episode_boundary=boundary,bootstrap_allowed=~boundary,gamma=.99,gae_lambda=.95)
        per=compute_per_agent_gae_and_returns(reward=rewards,old_value=values,bootstrap_value=next_values,**kwargs)
        team=compute_gae_and_returns(reward=rewards.sum(-1),old_value=values.sum(-1),bootstrap_value=next_values.sum(-1),**kwargs)
        y=team_return_target(per.return_target.reshape(1,4,2),2)
        torch.testing.assert_close(y[0,:,0],team.return_target,atol=1e-5,rtol=1e-6)
        self.assertEqual(y[0,1,0],7);self.assertEqual(y[0,2,0],11)
        self.assertGreater(y[0,3,0],15)  # ordinary rollout tail bootstraps

    def loss(self,policy,credit=None,config=None,active_override=None):
        config=self.config if config is None else config
        logs=torch.stack(list(policy.component_log_probs.values()),-1).reshape(1,4,-1)
        active=torch.stack(list(policy.component_active.values()),-1).reshape(1,4,-1)
        if active_override is not None: active=active_override
        return compute_configured_ppo_objective_and_loss(
            new_joint_log_prob=policy.joint_log_prob.reshape(1,4),old_joint_log_prob=policy.joint_log_prob.detach().reshape(1,4),
            advantage=torch.tensor([[-2.,-1.,1.,2.]],device=logs.device),current_value=torch.zeros(1,4,device=logs.device),
            return_target=torch.ones(1,4,device=logs.device),entropy=policy.entropy.reshape(1,4),
            sequence_valid_mask=torch.ones(1,dtype=torch.bool,device=logs.device),config=config,
            new_branch_log_probs=logs,old_branch_log_probs=logs.detach(),active_branch_indicators=active,
            route_entropy=policy.branch_entropies['route'].reshape(1,4),tx_upper_advantage=credit)

    def gradient_check(self,device):
        actor,critic=self.model(device)
        policy=CAGATMAPPOActionDistribution(actor,self.config).sample_actions(self.batch.to(device),self.masks)
        features=torch.randn(1,1,critic.spec.centralized_state_dim,device=device,requires_grad=True)
        q=critic.tx_upper_q(features);mask=policy.component_active['tx_upper']
        transmitted=policy.scheduler_metrics['tx_sampled_transmit'].bool()
        p=policy.scheduler_metrics['tx_p_transmit'];prob=torch.stack((1-p,p),-1)
        credit=counterfactual_credit(q,prob,transmitted,mask)
        loss=self.loss(policy,credit.normalized_advantage.reshape(1,4))
        upper=-loss.branch_surrogate[...,1].mean()
        upper.backward()
        self.assertGreater(gradient_sum(actor.tx_upper_private),0)
        self.assertEqual(gradient_sum(critic),0)
        self.assertEqual(gradient_sum(actor.cpu_joint_scheduler),0)
        self.assertEqual(gradient_sum(actor.action_heads['tx_select']),0)
        self.assertEqual(gradient_sum(actor.route_gru),0)
        self.assertIsNone(features.grad)
        actor.zero_grad(set_to_none=True)
        regression=selected_action_q_loss(q,transmitted,torch.ones_like(p)*3,mask)
        regression.backward()
        self.assertGreater(gradient_sum(critic.tx_upper_q),0)
        self.assertEqual(gradient_sum(critic.value_network),0)
        self.assertEqual(gradient_sum(actor),0)
        self.assertIsNone(features.grad)

    def test_actor_and_critic_gradient_isolation_cpu(self): self.gradient_check('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_actual_backward(self): self.gradient_check('cuda')

    def test_only_tx_upper_advantage_changes_destination_route_cpu_preserved(self):
        actor,_=self.model()
        policy=CAGATMAPPOActionDistribution(actor,self.config).sample_actions(self.batch,self.masks)
        x=self.loss(policy,torch.tensor([[4.,3.,2.,1.]]))
        y=self.loss(policy,config=self.e8)
        torch.testing.assert_close(x.branch_advantage[...,1],torch.tensor([[4.,3.,2.,1.]]),rtol=0,atol=0)
        for index in [0,2,3,4,5,6]:
            torch.testing.assert_close(x.branch_advantage[...,index],y.branch_advantage[...,index],rtol=0,atol=0)
            torch.testing.assert_close(x.branch_surrogate[...,index],y.branch_surrogate[...,index],rtol=0,atol=0)
        self.assertIsNone(x.tx_upper_stability)
        torch.testing.assert_close(x.global_entropy_loss_contribution,y.global_entropy_loss_contribution,rtol=0,atol=0)

    def test_inactive_tx_does_not_enter_ppo(self):
        actor,_=self.model()
        policy=CAGATMAPPOActionDistribution(actor,self.config).sample_actions(self.batch,self.masks)
        active=torch.stack(list(policy.component_active.values()),-1).reshape(1,4,-1).clone()
        active[...,1]=False
        x=self.loss(policy,torch.ones(1,4)*1e6,active_override=active)
        y=self.loss(policy,torch.ones(1,4)*-1e6,active_override=active)
        self.assertTrue((x.branch_surrogate[...,1]==0).all())
        torch.testing.assert_close(x.actor_loss,y.actor_loss,rtol=0,atol=0)

    def test_empty_q_loss_does_not_advance_existing_adam_moments(self):
        actor,critic=self.model();o=build_ca_gat_mappo_optimizers(actor,critic,self.config)
        features=torch.zeros(1,1,critic.spec.centralized_state_dim)
        q=critic.tx_upper_q(features);active=torch.ones(1,1,4,dtype=torch.bool)
        selected_action_q_loss(q,active,torch.ones(1,1,4),active).backward();o.critic_optimizer.step()
        before=copy.deepcopy(critic.tx_upper_q.state_dict())
        o.critic_optimizer.zero_grad(set_to_none=True)
        empty=selected_action_q_loss(critic.tx_upper_q(features),active,torch.zeros(1,1,4),~active)
        (critic(features).square().mean()+empty).backward();o.critic_optimizer.step()
        for k,p in critic.tx_upper_q.state_dict().items(): torch.testing.assert_close(before[k],p,rtol=0,atol=0)

    def test_config_guard_and_missing_credit_fail_closed(self):
        c=replace(self.config,training=replace(self.config.training,mappo=replace(self.config.training.mappo,tx_upper_optimization_mode='deterministic_stability_v1')))
        with self.assertRaises(ConfigError): c.validate()
        for value in ['bad',True,[],None]:
            c=replace(self.config,training=replace(self.config.training,mappo=replace(self.config.training.mappo,tx_upper_credit_mode=value)))
            with self.assertRaises(ConfigError): c.validate()
        actor,_=self.model();p=CAGATMAPPOActionDistribution(actor,self.config).sample_actions(self.batch,self.masks)
        with self.assertRaises(PPOObjectiveError): self.loss(p)
        with self.assertRaises(PPOObjectiveError): self.loss(p,torch.ones(1,4),self.e8)

    def buffer(self,actor,critic,length=256):
        with torch.no_grad(): policy=CAGATMAPPOActionDistribution(actor,self.config).sample_actions(self.batch,self.masks)
        seed=CAGATMAPPORolloutTransition.from_step(spec=actor.spec,slot=0,actor_batch=self.batch,action_mask_batch=self.masks,
            action_output=policy,hidden_in=actor.initial_hidden(1),centralized_state=CentralizedStateTensorBatch(torch.ones(1,1,critic.spec.centralized_state_dim)*.1),
            old_value=torch.zeros(4),reward=torch.ones(4),terminated=False,truncated=False,episode_boundary=False,
            bootstrap_allowed=True,bootstrap_value=torch.zeros(4),executed_action_summary=tuple({'uav_id':i} for i in range(4)),
            rejection_or_downgrade_summary={'rejection':[],'downgrade':[],'canonicalization':[]})
        return vector_buffer(self.config,seed,length=length,finalized=length==256)

    def test_recurrent_8x32_replay_q_training_and_frozen_credit(self):
        actor,critic=self.model();buffer=self.buffer(actor,critic)
        updater=CAGATMAPPORecurrentPPOUpdater(actor,critic,self.config)
        before=copy.deepcopy(critic.tx_upper_q.state_dict())
        output=updater.update(buffer)
        self.assertTrue(output.old_policy_snapshot_preserved)
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in critic.tx_upper_q.state_dict().items()))
        summaries=updater.last_integrated_summary['ppo_epochs'];json.dumps(summaries,allow_nan=False)
        tx=summaries[0]['tx_upper']
        for e in summaries:
            for k in ['mean_q_idle','mean_q_transmit','mean_counterfactual_baseline','raw_a_tx_mean','normalized_a_tx_std']:
                self.assertEqual(e['tx_upper'][k],tx[k])
            self.assertNotIn('margin_loss',e['tx_upper']);self.assertNotIn('beta_tx',e['tx_upper'])
            self.assertGreaterEqual(e['tx_upper']['q_critic_loss'],0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_recurrent_update(self):
        actor,critic=self.model();buffer=self.buffer(actor,critic)
        actor.to('cuda');critic.to('cuda')
        updater=CAGATMAPPORecurrentPPOUpdater(actor,critic,self.config)
        output=updater.update(buffer)
        self.assertTrue(output.old_policy_snapshot_preserved)
        self.assertTrue(any(p.grad is not None for p in critic.tx_upper_q.parameters()))

    def test_partial_rollout_roundtrip(self):
        actor,critic=self.model();buffer=self.buffer(actor,critic,8)
        state=serialize_active_rollout(buffer,rollout_policy_version=0,checkpoint_kind='PERIODIC_RESUME')
        restored,_=restore_active_rollout(self.config,state,checkpoint_kind='PERIODIC_RESUME',policy_version=0)
        torch.testing.assert_close(restored.transition_at(0).old_joint_log_prob,buffer.transition_at(0).old_joint_log_prob,rtol=0,atol=0)
        self.assertEqual(restored.transition_at(0).proposal_actions,buffer.transition_at(0).proposal_actions)

    def test_checkpoint_model_adam_resume_exact_update_and_actor_only_loader(self):
        from tests.test_ca_gat_mappo_checkpoint import make_payload,training_state,assert_nested_equal
        from src.evaluation.actor_loader import load_final_actor_for_evaluation
        config=preserve_git(replace(self.config,environment=replace(self.config.environment,episode_horizon=512),
            training=replace(self.config.training,mappo=replace(self.config.training.mappo,training_device='cpu',checkpoint_interval_steps=4096,max_training_environment_steps=32768))),self.config)
        actor=CAGATMAPPOActor(config);critic=MAPPOCentralizedCritic(config)
        updater=CAGATMAPPORecurrentPPOUpdater(actor,critic,config)
        buffer=self.buffer(actor,critic);updater.update(buffer)
        payload,*_=make_payload(config,'FINAL_COMPLETED',actor=actor,critic=critic,updater=updater)
        payload['checkpoint_kind']='PERIODIC_RESUME';payload['training_state']=training_state(config,'PERIODIC_RESUME',0)
        with tempfile.TemporaryDirectory(dir=Path(tempfile.gettempdir()) / 'e10-v1-validation') as directory:
            path=Path(directory)/'unit-periodic.pt';atomic_save_checkpoint(payload,path);saved=load_checkpoint_payload(path)
            a=CAGATMAPPOActor(config);c=MAPPOCentralizedCritic(config);u=CAGATMAPPORecurrentPPOUpdater(a,c,config)
            restore_model_optimizer_and_rng(config=config,payload=saved,actor=a,critic=c,optimizers=u.optimizers,dtype=torch.float32)
            assert_nested_equal(self,critic.state_dict(),c.state_dict())
            assert_nested_equal(self,updater.optimizers.critic_optimizer.state_dict(),u.optimizers.critic_optimizer.state_dict())
            updater.update(buffer);u.update(buffer)
            assert_nested_equal(self,actor.state_dict(),a.state_dict());assert_nested_equal(self,critic.state_dict(),c.state_dict())
            assert_nested_equal(self,updater.optimizers.critic_optimizer.state_dict(),u.optimizers.critic_optimizer.state_dict())
            final,*_=make_payload(config,'FINAL_COMPLETED',actor=actor,critic=critic,updater=updater)
            path=Path(directory)/'unit-final.pt';atomic_save_checkpoint(final,path)
            ev=preserve_git(replace(config,mode='evaluation',launch_profile=None),config)
            with patch('src.models.ca_gat_mappo.MAPPOCentralizedCritic',side_effect=AssertionError('formal actor loading must not construct any critic')):
                loaded=load_final_actor_for_evaluation(ev,path,evaluation_device='cpu')
            assert_nested_equal(self,actor.state_dict(),loaded.actor.state_dict())



    def test_optimizer_lr_groups_and_q_coverage(self):
        actor,critic=self.model();bundle=build_ca_gat_mappo_optimizers(actor,critic,self.config)
        bundle.validate(actor,critic,self.config)
        self.assertEqual(bundle.critic_optimizer.param_groups[0]['lr'],3e-4)
        ids={id(p) for p in bundle.critic_parameters}
        self.assertTrue(all(id(p) in ids for p in critic.tx_upper_q.parameters()))
        for name,p in actor.named_parameters():
            lr=next(g['lr'] for g in bundle.actor_optimizer.param_groups if any(v is p for v in g['params']))
            self.assertAlmostEqual(lr,3e-5 if name.startswith(actor.route_parameter_prefixes) else 3e-4)

    def test_trainer_update_writes_only_compact_q_telemetry(self):
        from src.models.ca_gat_mappo_trainer import CAGATMAPPOTrainer
        with tempfile.TemporaryDirectory(dir=Path(tempfile.gettempdir()) / 'e10-v1-validation') as directory:
            config=preserve_git(replace(self.config,
                training=replace(self.config.training,mappo=replace(self.config.training.mappo,training_device='cpu')),
                output=replace(self.config.output,logs_dir=directory)),self.config)
            trainer=CAGATMAPPOTrainer(config,progress_logger=lambda _:None,
                environment_factory=lambda _:self.fail('synthetic update test must not collect environment'))
            trainer.rollout_buffer=self.buffer(trainer.actor,trainer.critic)
            trainer._rollout_policy_version=trainer.policy_version
            trainer._transitions=256
            trainer._perform_update(trainer._updates)
            path=Path(config.artifact_paths()['raw_metrics']).parent/'integrated_scheduler_updates_v1.jsonl'
            rows=[json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['environment_steps'],256)
            self.assertEqual(trainer.policy_version,1)
            for epoch in rows[0]['ppo_epochs']:
                self.assertIn('q_critic_loss',epoch['tx_upper'])
                self.assertNotIn('margin_loss',epoch['tx_upper']);self.assertNotIn('beta_tx',epoch['tx_upper'])
            self.assertEqual(list(path.parent.glob('*.jsonl')),[path])


if __name__=='__main__': unittest.main()
