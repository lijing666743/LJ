"""E11 bounded fixtures, exact resume and CPU/CUDA correctness; no formal runs."""
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
from src.models.ca_gat_mappo import CAGATMAPPOActor, MAPPOCentralizedCritic
from src.models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution
from src.models.ca_gat_mappo_checkpoint import (
    CheckpointError, atomic_save_checkpoint, load_checkpoint_payload,
    restore_model_optimizer_and_rng, validate_checkpoint_payload,
)
from src.models.ca_gat_mappo_update import CAGATMAPPORecurrentPPOUpdater
from src.models.tx_upper_counterfactual import (
    TXQTargetNormalizerV1, TXUpperCounterfactualCriticV1,
    counterfactual_credit, selected_action_q_loss,
)
from tests.test_integrated_scheduler import fixture, ROOT, CONFIGS
from tests.test_route_private_backbone import gradient_sum
from tests.test_tx_upper_stability import preserve_git


class TXQPopArtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.config = load_run_config(CONFIGS/'e11-v1-tx-q-popart-seed42-32k.json')
        cls.base = load_run_config(CONFIGS/'e10-seed42-config-snapshot.yaml')
        cls.output = Path(tempfile.gettempdir()) / 'e11-v1-validation'
        cls.output.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def model(self, device='cpu'):
        return CAGATMAPPOActor(self.config).to(device), MAPPOCentralizedCritic(self.config).to(device)

    def buffer(self, actor, critic, tx_active=True):
        # Reuse the E10 vector fixture, without collecting an environment.
        from tests.test_tx_upper_counterfactual import TXCounterfactualTests
        helper = TXCounterfactualTests()
        helper.config = self.config
        _, helper.batch, helper.masks = fixture(self.config)
        if not tx_active:
            from src.models.ca_gat_mappo_actions import SequentialActionMaskBatch
            contracts=[]
            for contract in helper.masks.contracts[0][0]:
                mask=contract.tx_select_mask.copy();mask[1:]=False
                contracts.append(replace(contract,tx_select_mask=mask,tx_branch_active=False))
            helper.masks=SequentialActionMaskBatch(((tuple(contracts),),))
            branch_masks=dict(helper.batch.action_masks)
            branch_masks['tx_select']=branch_masks['tx_select'].clone();branch_masks['tx_select'][...,1:]=False
            helper.batch=replace(helper.batch,action_masks=branch_masks)
        return helper.buffer(actor, critic)

    def test_welford_exact_running_moments_and_mask(self):
        n=TXQTargetNormalizerV1();layer=torch.nn.Linear(3,8)
        self.assertEqual(n.count,0);self.assertEqual(n.mean,0);self.assertEqual(n.std,1)
        for samples in [torch.tensor([1.,2.,float('nan')]),torch.tensor([10.,20.,30.])]:
            mask=torch.isfinite(samples);n.update_and_compensate(samples,mask,layer)
        x=torch.tensor([1.,2.,10.,20.,30.],dtype=torch.float64)
        self.assertEqual(int(n.count),5)
        torch.testing.assert_close(n.mean,x.mean(),rtol=1e-14,atol=1e-14)
        torch.testing.assert_close(n.M2,((x-x.mean())**2).sum(),rtol=1e-14,atol=1e-14)
        torch.testing.assert_close(n.std,x.std(unbiased=False),rtol=1e-14,atol=1e-14)

    def test_empty_active_does_not_update_stats_or_weights(self):
        q=TXUpperCounterfactualCriticV1(3,4,2,42,True)
        old=copy.deepcopy(q.state_dict())
        self.assertFalse(q.update_target_statistics(torch.tensor([float('nan')]),torch.zeros(1,dtype=torch.bool)))
        for k,v in old.items():torch.testing.assert_close(v,q.state_dict()[k],rtol=0,atol=0)

    def test_epsilon_fixed_constant_targets_and_dtype_move(self):
        n=TXQTargetNormalizerV1();layer=torch.nn.Linear(3,4)
        n.update_and_compensate(torch.full((4,),5.),torch.ones(4,dtype=torch.bool),layer)
        self.assertEqual(n.std,1e-5)
        n.to(dtype=torch.float32)
        self.assertEqual(n.mean.dtype,torch.float64);self.assertEqual(n.M2.dtype,torch.float64)
        self.assertEqual(n.count.dtype,torch.int64)

    def test_popart_unnormalized_Q_identity_multiple_updates(self):
        q=TXUpperCounterfactualCriticV1(7,8,4,42,True)
        x=torch.linspace(-1,1,35).reshape(5,7)
        for target in [torch.tensor([1.,2.,3.,10.]),torch.tensor([-10.,20.,-30.])]:
            before=q(x).detach().clone()
            q.update_target_statistics(target,torch.ones_like(target,dtype=torch.bool))
            torch.testing.assert_close(q(x),before,rtol=2e-5,atol=2e-6)

    def test_normalized_target_formula_and_detached_stats(self):
        n=TXQTargetNormalizerV1();layer=torch.nn.Linear(2,4)
        y=torch.tensor([2.,4.,6.],requires_grad=True)
        n.update_and_compensate(y,torch.ones(3,dtype=torch.bool),layer)
        expected=(y-4)/(torch.tensor(8/3).sqrt())
        torch.testing.assert_close(n.normalize(y),expected)
        torch.testing.assert_close(n.denormalize(n.normalize(y)),y)
        self.assertFalse(n.mean.requires_grad);self.assertFalse(n.M2.requires_grad)
        self.assertIsNone(y.grad)

    def test_shared_normalizer_across_agents_and_action_gap_sign(self):
        q=TXUpperCounterfactualCriticV1(3,4,4,42,True)
        target=torch.arange(8.).reshape(2,4);active=torch.ones_like(target,dtype=torch.bool)
        q.update_target_statistics(target,active)
        self.assertEqual(int(q.tx_q_target_norm.count),8)
        self.assertEqual(len(list(q.tx_q_target_norm.buffers())),3)
        x=torch.randn(2,3);hat=q.forward_normalized(x);raw=q(x)
        torch.testing.assert_close(raw,q.tx_q_target_norm.std.to(hat)*hat+q.tx_q_target_norm.mean.to(hat),rtol=0,atol=0)
        self.assertTrue(torch.equal((raw[...,1]-raw[...,0]).sign(),(hat[...,1]-hat[...,0]).sign()))

    def test_action_label_symmetry_after_compensation(self):
        a=TXUpperCounterfactualCriticV1(3,4,2,42,True);b=copy.deepcopy(a)
        with torch.no_grad():
            b.network[-1].weight.copy_(a.network[-1].weight.reshape(2,2,4).flip(1).reshape(4,4))
            b.network[-1].bias.copy_(a.network[-1].bias.reshape(2,2).flip(1).reshape(4))
        target=torch.tensor([1.,3.,5.]);active=torch.ones(3,dtype=torch.bool)
        a.update_target_statistics(target,active);b.update_target_statistics(target,active)
        x=torch.randn(5,3)
        torch.testing.assert_close(a(x).flip(-1),b(x),rtol=0,atol=0)

    def test_normalized_regression_selected_action_only(self):
        n=TXQTargetNormalizerV1();layer=torch.nn.Linear(2,2)
        y=torch.tensor([1.,5.,float('nan')],requires_grad=True);active=torch.tensor([1,1,0],dtype=torch.bool)
        n.update_and_compensate(y,active,layer)
        hat=torch.tensor([[0.,0.],[0.,0.],[float('nan'),float('nan')]],requires_grad=True)
        actions=torch.tensor([0,1,0],dtype=torch.bool)
        loss=selected_action_q_loss(hat,actions,n.normalize(y),active);loss.backward()
        torch.testing.assert_close(hat.grad,torch.tensor([[.5,0.],[0.,-.5],[0.,0.]]),rtol=0,atol=0)
        self.assertIsNone(y.grad)

    def test_actor_credit_uses_original_units_and_probability_detach(self):
        n=TXQTargetNormalizerV1();layer=torch.nn.Linear(2,2)
        n.update_and_compensate(torch.tensor([10.,14.]),torch.ones(2,dtype=torch.bool),layer)
        hat=torch.tensor([[0.,2.],[1.,-1.]],requires_grad=True)
        logits=torch.tensor([[1.,2.],[2.,1.]],requires_grad=True);p=logits.softmax(-1)
        actions=torch.tensor([1,0],dtype=torch.bool);active=torch.ones(2,dtype=torch.bool)
        c=counterfactual_credit(n.denormalize(hat),p,actions,active)
        expected_q=torch.tensor([[12.,16.],[14.,10.]])
        torch.testing.assert_close(c.q,expected_q,rtol=0,atol=0)
        torch.testing.assert_close(c.baseline,(p.detach()*expected_q).sum(-1))
        self.assertFalse(c.normalized_advantage.requires_grad);self.assertIsNone(logits.grad)

    def test_recurrent_update_stats_once_fixed_four_epochs_and_original_credit(self):
        actor,critic=self.model();buffer=self.buffer(actor,critic)
        updater=CAGATMAPPORecurrentPPOUpdater(actor,critic,self.config)
        with patch.object(critic.tx_upper_q.tx_q_target_norm,'update_and_compensate',wraps=critic.tx_upper_q.tx_q_target_norm.update_and_compensate) as update:
            output=updater.update(buffer)
        self.assertEqual(update.call_count,1);self.assertTrue(output.old_policy_snapshot_preserved)
        summaries=updater.last_integrated_summary['ppo_epochs'];json.dumps(summaries,allow_nan=False)
        first=summaries[0]['tx_upper']
        self.assertEqual(int(critic.tx_upper_q.tx_q_target_norm.count),first['active_samples'])
        for e in summaries:
            for k in ['running_count','running_mean','running_std','target_raw_mean','normalized_target_mean','mean_q_idle','mean_q_transmit']:
                self.assertEqual(e['tx_upper'][k],first[k])
            self.assertIn('normalized_selected_q_std',e['tx_upper'])
        updater.update(buffer)
        self.assertEqual(int(critic.tx_upper_q.tx_q_target_norm.count),2*first['active_samples'])

    def test_no_active_update_stats_noop_and_sequence_mask(self):
        actor,critic=self.model();buffer=self.buffer(actor,critic)
        # Unit population test explicitly includes sequence-invalid atoms.
        target=torch.tensor([[1.,2.],[1000.,1000.]])
        active=torch.ones(2,2,dtype=torch.bool)&torch.tensor([True,False])[:,None]
        critic.tx_upper_q.update_target_statistics(target,active)
        self.assertEqual(int(critic.tx_upper_q.tx_q_target_norm.count),2)
        self.assertEqual(critic.tx_upper_q.tx_q_target_norm.mean,1.5)
        # Empty cohort is validated independently of changing replay masks.
        old=copy.deepcopy(critic.tx_upper_q.state_dict())
        critic.tx_upper_q.update_target_statistics(target,torch.zeros_like(active))
        for k,v in old.items():torch.testing.assert_close(v,critic.tx_upper_q.state_dict()[k],rtol=0,atol=0)

    def test_empty_TX_recurrent_update_preserves_stats_and_Q(self):
        actor,critic=self.model();buffer=self.buffer(actor,critic,tx_active=False)
        before=copy.deepcopy(critic.tx_upper_q.state_dict())
        updater=CAGATMAPPORecurrentPPOUpdater(actor,critic,self.config);updater.update(buffer)
        for k,v in before.items():torch.testing.assert_close(v,critic.tx_upper_q.state_dict()[k],rtol=0,atol=0)
        for epoch in updater.last_integrated_summary['ppo_epochs']:
            self.assertEqual(epoch['tx_upper']['running_count'],0)
            self.assertIsNone(epoch['tx_upper']['target_raw_mean'])

    def test_trainer_only_compact_popart_telemetry(self):
        from src.models.ca_gat_mappo_trainer import CAGATMAPPOTrainer
        with tempfile.TemporaryDirectory(dir=self.output) as directory:
            config=preserve_git(replace(self.config,
                training=replace(self.config.training,mappo=replace(self.config.training.mappo,training_device='cpu')),
                output=replace(self.config.output,logs_dir=directory)),self.config)
            trainer=CAGATMAPPOTrainer(config,progress_logger=lambda _:None,
                environment_factory=lambda _:self.fail('test must not collect environment'))
            trainer.rollout_buffer=self.buffer(trainer.actor,trainer.critic)
            trainer._rollout_policy_version=trainer.policy_version;trainer._transitions=256
            trainer._perform_update(trainer._updates)
            path=Path(config.artifact_paths()['raw_metrics']).parent/'integrated_scheduler_updates_v1.jsonl'
            rows=[json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual(len(rows),1)
            self.assertEqual(list(path.parent.glob('*.jsonl')),[path])
            self.assertIn('running_std',rows[0]['ppo_epochs'][0]['tx_upper'])

    def gradient_check(self,device):
        actor,critic=self.model(device);_,batch,masks=fixture(self.config)
        policy=CAGATMAPPOActionDistribution(actor,self.config).sample_actions(batch.to(device),masks)
        features=torch.randn(1,1,critic.spec.centralized_state_dim,device=device,requires_grad=True)
        active=policy.component_active['tx_upper'];actions=policy.scheduler_metrics['tx_sampled_transmit'].bool()
        target=torch.arange(4.,device=device).reshape(1,1,4)
        critic.tx_upper_q.update_target_statistics(target,active)
        hat=critic.tx_upper_q.forward_normalized(features)
        selected_action_q_loss(hat,actions,critic.tx_upper_q.tx_q_target_norm.normalize(target),active).backward()
        self.assertGreater(gradient_sum(critic.tx_upper_q),0);self.assertEqual(gradient_sum(critic.value_network),0)
        self.assertEqual(gradient_sum(actor),0);self.assertIsNone(features.grad)
        from tests.test_tx_upper_counterfactual import TXCounterfactualTests
        helper=TXCounterfactualTests();helper.config=self.config
        p=policy.scheduler_metrics['tx_p_transmit'];c=counterfactual_credit(critic.tx_upper_q(features),torch.stack((1-p,p),-1),actions,active)
        critic.zero_grad(set_to_none=True)
        helper.loss(policy,c.normalized_advantage.reshape(1,4)).actor_loss.backward()
        self.assertEqual(gradient_sum(critic),0)
        self.assertGreater(gradient_sum(actor.cpu_joint_scheduler),0)
        self.assertGreater(gradient_sum(actor.tx_upper_private),0)
        # A sampled Route can be saturated; exercise a nontrivial Route-only
        # categorical objective rather than assume that sample has a gradient.
        actor.zero_grad(set_to_none=True)
        unit_batch=batch.to(device)
        route_masks=dict(unit_batch.action_masks)
        route_masks['route']=torch.ones_like(route_masks['route']);route_masks['route'][...,0]=False
        indices=dict(unit_batch.action_indices);indices['route']=torch.ones_like(indices['route'])
        unit_batch=replace(unit_batch,action_masks=route_masks,action_indices=indices)
        logits=actor(unit_batch).raw_logits['route']
        (-torch.log_softmax(logits,-1)[...,1].mean()).backward()
        self.assertGreater(gradient_sum(actor.route_gru),0)
        self.assertEqual(gradient_sum(actor.cpu_joint_scheduler),0)
        self.assertEqual(gradient_sum(critic),0)

    def test_gradient_isolation_and_route_cpu_gradients_cpu(self):self.gradient_check('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_backward_and_recurrent_replay(self):
        self.gradient_check('cuda')
        actor,critic=self.model();buffer=self.buffer(actor,critic);actor.to('cuda');critic.to('cuda',dtype=torch.float32)
        updater=CAGATMAPPORecurrentPPOUpdater(actor,critic,self.config)
        output=updater.update(buffer)
        self.assertTrue(output.old_policy_snapshot_preserved)
        self.assertEqual(critic.tx_upper_q.tx_q_target_norm.mean.dtype,torch.float64)
        self.assertTrue(any(p.grad is not None for p in critic.tx_upper_q.parameters()))

    def test_original_V_and_actor_forward_parameter_count_unchanged(self):
        actor,critic=self.model();old_actor=CAGATMAPPOActor(self.base);old=MAPPOCentralizedCritic(self.base)
        for k,v in old.value_network.state_dict().items():torch.testing.assert_close(v,critic.value_network.state_dict()[k],rtol=0,atol=0)
        for k,v in old_actor.state_dict().items():torch.testing.assert_close(v,actor.state_dict()[k],rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in critic.parameters()),sum(p.numel() for p in old.parameters()))
        self.assertEqual(sum(p.numel() for p in critic.tx_upper_q.parameters()),sum(p.numel() for p in old.tx_upper_q.parameters()))
        features=torch.randn(1,1,critic.spec.centralized_state_dim)
        torch.testing.assert_close(critic(features),old(features),rtol=0,atol=0)
        torch.testing.assert_close(critic.tx_upper_q(features),old.tx_upper_q(features),rtol=0,atol=0)

    def test_original_V_identical_synthetic_update(self):
        actor,critic=self.model();old_actor=CAGATMAPPOActor(self.base);old=MAPPOCentralizedCritic(self.base)
        buffer=self.buffer(actor,critic)
        CAGATMAPPORecurrentPPOUpdater(actor,critic,self.config).update(buffer)
        CAGATMAPPORecurrentPPOUpdater(old_actor,old,self.base).update(buffer)
        for k,v in old.value_network.state_dict().items():torch.testing.assert_close(v,critic.value_network.state_dict()[k],rtol=0,atol=0)

    def test_checkpoint_resume_stats_adam_and_next_update_exact(self):
        from tests.test_ca_gat_mappo_checkpoint import make_payload,training_state,assert_nested_equal
        config=preserve_git(replace(self.config,environment=replace(self.config.environment,episode_horizon=512),
            training=replace(self.config.training,mappo=replace(self.config.training.mappo,training_device='cpu',checkpoint_interval_steps=4096,max_training_environment_steps=32768))),self.config)
        actor=CAGATMAPPOActor(config);critic=MAPPOCentralizedCritic(config);updater=CAGATMAPPORecurrentPPOUpdater(actor,critic,config)
        buffer=self.buffer(actor,critic);updater.update(buffer)
        payload,*_=make_payload(config,'FINAL_COMPLETED',actor=actor,critic=critic,updater=updater)
        payload['checkpoint_kind']='PERIODIC_RESUME';payload['training_state']=training_state(config,'PERIODIC_RESUME',0)
        with tempfile.TemporaryDirectory(dir=self.output) as directory:
            path=Path(directory)/'unit.pt';atomic_save_checkpoint(payload,path);saved=load_checkpoint_payload(path)
            a=CAGATMAPPOActor(config);c=MAPPOCentralizedCritic(config);u=CAGATMAPPORecurrentPPOUpdater(a,c,config)
            restore_model_optimizer_and_rng(config=config,payload=saved,actor=a,critic=c,optimizers=u.optimizers,dtype=torch.float32)
            assert_nested_equal(self,critic.state_dict(),c.state_dict())
            self.assertGreater(int(c.tx_upper_q.tx_q_target_norm.count),0)
            updater.update(buffer);u.update(buffer)
            assert_nested_equal(self,critic.state_dict(),c.state_dict());assert_nested_equal(self,actor.state_dict(),a.state_dict())
            assert_nested_equal(self,updater.optimizers.critic_optimizer.state_dict(),u.optimizers.critic_optimizer.state_dict())
            final,*_=make_payload(config,'FINAL_COMPLETED',actor=actor,critic=critic,updater=updater)
            path=Path(directory)/'unit-final.pt'
            atomic_save_checkpoint(final,path)
            from src.evaluation.actor_loader import load_final_actor_for_evaluation
            ev=preserve_git(replace(config,mode='evaluation',launch_profile=None),config)
            with patch('src.models.ca_gat_mappo.MAPPOCentralizedCritic',side_effect=AssertionError('actor-only loading constructs no critic')):
                loaded=load_final_actor_for_evaluation(ev,path,evaluation_device='cpu')
            assert_nested_equal(self,actor.state_dict(),loaded.actor.state_dict())

    def test_missing_corrupt_stats_and_E10_resume_fail_closed(self):
        from tests.test_ca_gat_mappo_checkpoint import make_payload
        config=preserve_git(replace(self.config,training=replace(self.config.training,mappo=replace(self.config.training.mappo,training_device='cpu'))),self.config)
        payload,actor,critic,updater=make_payload(config,'FINAL_COMPLETED')
        for key,value in [('count',None),('count',torch.tensor(-1)),('mean',torch.tensor(float('nan'),dtype=torch.float64)),('M2',torch.tensor(-1.,dtype=torch.float64))]:
            p=copy.deepcopy(payload);name='tx_upper_q.tx_q_target_norm.'+key
            if value is None:del p['model_state']['critic'][name]
            else:p['model_state']['critic'][name]=value
            with self.assertRaises(CheckpointError):validate_checkpoint_payload(p)

    def test_config_single_treatment_and_invalid_modes(self):
        from src.evaluation.e4_v1_matched_audit import differences
        a,b=self.base.resolved_dict(),self.config.resolved_dict()
        for c in (a,b):c['training']['mappo'].setdefault('tx_q_target_normalization','none')
        self.assertEqual(differences(a,b),[{'field':'training.mappo.tx_q_target_normalization','E3':'none','E4':'popart_v1'}])
        for value in ['bad',True,[],None]:
            config=replace(self.config,training=replace(self.config.training,mappo=replace(self.config.training.mappo,tx_q_target_normalization=value)))
            with self.assertRaises(ConfigError):config.validate()
        config=replace(self.config,training=replace(self.config.training,mappo=replace(self.config.training.mappo,tx_upper_credit_mode='shared_agent_advantage')))
        with self.assertRaises(ConfigError):config.validate()


if __name__=='__main__':unittest.main()
