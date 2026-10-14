"""E8 unit contracts; no formal training, evaluation, or new state bank."""
import copy
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.config import ConfigError, load_run_config
from src.env.actions import ActionProposal
from src.models.ca_gat_mappo import CAGATMAPPOActor, MAPPOCentralizedCritic, CentralizedStateTensorBatch
from src.models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution, SequentialActionMaskBatch, ActionDistributionError
from src.models.ca_gat_mappo_rollout import CAGATMAPPORolloutTransition, CAGATMAPPORolloutBuffer
from src.models.ca_gat_mappo_update import (CAGATMAPPORecurrentPPOUpdater, build_ca_gat_mappo_optimizers,
    build_recurrent_ppo_minibatch, compute_configured_batched_ppo_objective_and_loss)
from src.models.ca_gat_mappo_checkpoint import (serialize_active_rollout, restore_active_rollout,
    atomic_save_checkpoint, load_checkpoint_payload)
from src.models.integrated_scheduler import policy_component_names, rollout_policy_components
from tests.test_ca_gat_mappo_update import make_seed_transition, make_synthetic_buffer
from tests.test_route_head_lr_isolation import persisted_batch
from tests.test_route_private_backbone import gradient_sum

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT/'tests/fixtures/configs'


def vector_buffer(config, seed, length=256, finalized=True):
    buffer=CAGATMAPPORolloutBuffer(config)
    for index in range(length):
        buffer.append(replace(seed,slot=index,episode_start=index==0,
                              hidden_in=torch.zeros_like(seed.hidden_in),reward=torch.ones(config.environment.uav_count)))
    if finalized:buffer.finalize()
    return buffer


def fixture(config):
    seed = make_seed_transition(config)
    batch = persisted_batch(config)
    a = config.environment.uav_count
    contracts = []
    for c in seed.action_mask_contracts:
        contracts.append(replace(c, tx_select_mask=np.ones(a, dtype=bool),
            cpu_queue_mask=np.ones(a+1, dtype=bool), cpu_frequency_energy_mask=np.ones((a,len(config.action.cpu_frequency_levels)),dtype=bool),
            tx_branch_active=True, cpu_queue_branch_active=True))
    masks = SequentialActionMaskBatch(((tuple(contracts),),))
    x = batch.self_features.clone()
    for agent in range(a):
        for q in range(a):
            # Distinct source tasks; executor-local queue intentionally differs.
            row = torch.tensor([.002,.01,.0005*(q+1),.1,.02,.1*(q+1),.05*(q+1),1.])
            x[0,0,agent,25+8*a+8*q:25+8*a+8*(q+1)] = row
            x[0,0,agent,25+8*q:25+8*(q+1)] = row
        x[0,0,agent,17:25] = torch.tensor([.003,.02,.001,.2,.03,.6,.02,1.])
    branch_masks = {b:torch.tensor(np.stack([c.mask_for(b) for c in contracts]))[None,None]
                    for b in batch.action_masks}
    batch = replace(batch, self_features=x, action_masks=branch_masks,
                    action_indices={b:m.long().argmax(-1) for b,m in branch_masks.items()})
    return seed, batch, masks


class IntegratedSchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.e7 = load_run_config(CONFIGS/'e7-v1-route-private-backbone-seed43-32k.json')
        cls.e8 = load_run_config(CONFIGS/'e8-v1-integrated-tx-deadline-cpu-seed43-32k.json')
        cls.seed, cls.batch, cls.masks = fixture(cls.e8)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_exact_two_effective_config_changes(self):
        from src.evaluation.e4_v1_matched_audit import differences
        a,b = self.e7.resolved_dict(),self.e8.resolved_dict()
        for c in (a,b):
            c['training']['mappo'].setdefault('tx_upper_decoder_mode','legacy')
            c['training']['mappo'].setdefault('cpu_scheduler_mode','legacy_hierarchical')
        self.assertEqual([r['field'] for r in differences(a,b)],
                         ['training.mappo.cpu_scheduler_mode','training.mappo.tx_upper_decoder_mode'])
        self.assertEqual(self.e7.config_hash,'ab921ec032abbddb52ec807bc203faea0a57be93b7a8170b0791edfc848e0748')
        self.assertEqual(Path(self.e8.artifact_paths()['config_snapshot']).parent.name,self.e8.run_id)
        for field,value in [('tx_upper_decoder_mode','bad'),('cpu_scheduler_mode','bad'),
                            ('cpu_frequency_decoder_mode','candidate_aware_v1')]:
            c=replace(self.e8,training=replace(self.e8.training,mappo=replace(self.e8.training.mappo,**{field:value})))
            with self.assertRaises(ConfigError): c.validate()

    def test_e7_common_weights_and_route_forward_unchanged(self):
        old,new=CAGATMAPPOActor(self.e7),CAGATMAPPOActor(self.e8)
        for k,v in new.state_dict().items():
            if k in old.state_dict(): torch.testing.assert_close(v,old.state_dict()[k],rtol=0,atol=0)
        with torch.no_grad():
            out1,out2=old(self.batch),new(self.batch)
        torch.testing.assert_close(out1.raw_logits['route'],out2.raw_logits['route'],rtol=0,atol=0)
        torch.testing.assert_close(out1.hidden_out,out2.hidden_out,rtol=0,atol=0)
        self.assertNotIn('cpu_queue',new.action_heads)
        self.assertNotIn('cpu_frequency',new.action_heads)
        self.assertFalse(hasattr(new.action_heads['tx_select'],'top_level_head'))

    def test_tx_permutation_and_illegal_mask(self):
        head=CAGATMAPPOActor(self.e8).tx_upper_private
        n=head.candidate_scorer[0].in_features
        x=torch.randn(3,4,n); source=torch.randn(3,9); legal=torch.tensor([[1,1,0,1]]*3,dtype=torch.bool)
        expected=head.score(source,x,legal); order=torch.tensor([2,0,3,1])
        torch.testing.assert_close(expected,head.score(source,x[:,order],legal[:,order]),atol=1e-6,rtol=1e-6)
        changed=x.clone();changed[:,2]=1000
        torch.testing.assert_close(expected,head.score(source,changed,legal),rtol=0,atol=0)
        logits=head.score(source,x,torch.zeros_like(legal));p=logits.softmax(-1)
        self.assertTrue(torch.equal(p[:,0],torch.ones(3)));self.assertTrue(torch.equal(p[:,1],torch.zeros(3)))

    def test_cpu_candidate_task_binding_and_workload(self):
        scheduler=CAGATMAPPOActor(self.e8).cpu_joint_scheduler
        _,physical=scheduler.candidate_features(self.batch)
        for ego in range(4):
            for q in range(4):
                expected=.6 if q==ego else .1*(q+1)
                self.assertAlmostEqual(float(physical['cycles'][0,0,ego,q,0])/scheduler.max_cycles,expected,places=6)
                expected_w=.001 if q==ego else .0005*(q+1)
                self.assertAlmostEqual(float(physical['workload'][0,0,ego,q,0])/(scheduler.max_cycles*scheduler.max_tasks),expected_w,places=6)

    def test_cpu_physical_units_inclusive_slack_and_no_feasibility_mask(self):
        actor=CAGATMAPPOActor(self.e8);features,p=actor.cpu_joint_scheduler.candidate_features(self.batch)
        self.assertAlmostEqual(float(p['slack'][0,0,0,0,0]),3.,places=4)
        torch.testing.assert_close(p['required_slots'],p['cycles']/p['capacity'])
        torch.testing.assert_close(p['margin'],p['slack']-p['required_slots'])
        fmax=self.batch.self_features[...,5]*self.e8.environment.reference_cpu_frequency_hz
        torch.testing.assert_close(p['rho'],p['cycles']/(p['slack']*fmax[...,None,None]*self.e8.environment.slot_duration_s))
        d=CAGATMAPPOActionDistribution(actor,self.e8)
        out=d.deterministic_actions(self.batch,self.masks)
        self.assertTrue(torch.any(p['rho']>1));self.assertTrue(torch.any(p['margin']<0))
        # Support ignores the optimistic indicators: all 13 atoms legal here.
        from src.models.integrated_scheduler import cpu_joint_statistics
        c=cpu_joint_statistics(actor,out.recurrent_features,self.batch,self.masks.flattened(),None,'deterministic',None)
        self.assertTrue(c['mask'].all());self.assertEqual(c['mask'].shape[-1],13)

    def test_cpu_queue_permutation_equivariance(self):
        actor=CAGATMAPPOActor(self.e8);recurrent=actor(self.batch).recurrent_features
        features,_=actor.cpu_joint_scheduler.candidate_features(self.batch)
        score=actor.cpu_joint_scheduler.score_candidates(recurrent,features)
        order=torch.tensor([3,1,0,2])
        changed=actor.cpu_joint_scheduler.score_candidates(recurrent,features[...,order,:,:])
        torch.testing.assert_close(changed,score[...,order,:],rtol=0,atol=0)

    def test_joint_probability_entropy_and_logprob_replay(self):
        actor=CAGATMAPPOActor(self.e8);d=CAGATMAPPOActionDistribution(actor,self.e8)
        out=d.sample_actions(self.batch,self.masks)
        replay=d.evaluate_actions(self.batch,self.masks,out.proposals)
        torch.testing.assert_close(out.joint_log_prob,replay.joint_log_prob,rtol=0,atol=0)
        torch.testing.assert_close(out.branch_log_probs['tx_select'],out.component_log_probs['tx_upper']+out.component_log_probs['tx_destination'])
        torch.testing.assert_close(sum(out.component_log_probs.values()),out.joint_log_prob)
        self.assertTrue(torch.equal(out.branch_log_probs['cpu_frequency'],torch.zeros_like(out.joint_log_prob)))
        for b,p in out.probabilities.items(): torch.testing.assert_close(p.sum(-1),torch.ones_like(out.joint_log_prob))
        for proposal in out.proposals[0][0]:
            self.assertTrue(proposal.cpu_queue=='idle' or proposal.cpu_frequency>0)

    def test_cpu_single_idle_and_existing_energy_masks(self):
        actor=CAGATMAPPOActor(self.e8)
        contracts=tuple(replace(c,cpu_frequency_energy_mask=np.tile(np.array([True,False,False,False]),(4,1))) for c in self.masks.flattened())
        masks=SequentialActionMaskBatch(((contracts,),))
        out=CAGATMAPPOActionDistribution(actor,self.e8).deterministic_actions(self.batch,masks)
        self.assertTrue(all(p.cpu_queue=='idle' and p.cpu_frequency==0 for p in out.proposals[0][0]))
        bad=tuple(replace(p,cpu_queue=0,cpu_frequency=0.) for p in out.proposals[0][0])
        # Restore original masks: zero-frequency queue is legal to environment,
        # but absent from the intentional single-Idle E8 policy support.
        with self.assertRaises(ActionDistributionError):
            CAGATMAPPOActionDistribution(actor,self.e8).evaluate_actions(self.batch,self.masks,((bad,),))

    def gradient_check(self,device):
        actor=CAGATMAPPOActor(self.e8).to(device);batch=self.batch.to(device)
        d=CAGATMAPPOActionDistribution(actor,self.e8)
        for branch in ['tx_upper','tx_destination','cpu_joint','route']:
            actor.zero_grad(set_to_none=True)
            out=d.sample_actions(batch,self.masks)
            if branch=='tx_destination' and not out.component_active[branch].any():
                proposals=tuple(replace(p,tx_select=next(v for v in c.tx_select_domain[1:]),resource_group=1,resource_width=1,power_level=.25)
                                for p,c in zip(out.proposals[0][0],self.masks.flattened()))
                out=d.evaluate_actions(batch,self.masks,((proposals,),))
            loss=-out.component_log_probs[branch].sum();loss.backward()
            if branch!='tx_upper':self.assertEqual(gradient_sum(actor.tx_upper_private),0)
            if branch!='cpu_joint':self.assertEqual(gradient_sum(actor.cpu_joint_scheduler),0)
            if branch!='route':
                for name in ['route_self_encoder','route_public_encoder','route_graph_encoder','route_gru','route_decoder']:
                    self.assertEqual(gradient_sum(getattr(actor,name)),0)
            if branch=='tx_upper':self.assertGreater(gradient_sum(actor.tx_upper_private),0)
            if branch=='cpu_joint':self.assertGreater(gradient_sum(actor.cpu_joint_scheduler),0)
            if branch=='tx_destination':self.assertGreater(gradient_sum(actor.action_heads['tx_select'].destination_head),0)
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters()))

    def test_gradient_isolation_cpu(self): self.gradient_check('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_gradient_isolation_cuda(self): self.gradient_check('cuda')

    def test_optimizer_groups(self):
        actor=CAGATMAPPOActor(self.e8);critic=MAPPOCentralizedCritic(self.e8);o=build_ca_gat_mappo_optimizers(actor,critic,self.e8)
        lrs={id(p):g['lr'] for g in o.actor_optimizer.param_groups for p in g['params']}
        for name,p in actor.named_parameters():self.assertAlmostEqual(lrs[id(p)],3e-5 if name.startswith(actor.route_parameter_prefixes) else 3e-4)

    def test_checkpoint_metadata_roundtrip_recurrent_ppo_and_compact_summary(self):
        actor=CAGATMAPPOActor(self.e8);critic=MAPPOCentralizedCritic(self.e8);d=CAGATMAPPOActionDistribution(actor,self.e8)
        with torch.no_grad():out=d.sample_actions(self.batch,self.masks)
        seed=CAGATMAPPORolloutTransition.from_step(spec=actor.spec,slot=0,actor_batch=self.batch,action_mask_batch=self.masks,
            action_output=out,hidden_in=actor.initial_hidden(1),centralized_state=CentralizedStateTensorBatch(torch.zeros(1,1,critic.spec.centralized_state_dim)),
            old_value=torch.zeros(4),reward=torch.ones(4),terminated=False,truncated=False,episode_boundary=False,bootstrap_allowed=True,
            bootstrap_value=torch.zeros(4),executed_action_summary=tuple({'uav_id':i} for i in range(4)),rejection_or_downgrade_summary={"rejection":[],"downgrade":[],"canonicalization":[]})
        buffer=vector_buffer(self.e8,seed)
        partial=vector_buffer(self.e8,seed,length=8,finalized=False)
        state=serialize_active_rollout(partial,rollout_policy_version=0,checkpoint_kind='PERIODIC_RESUME')
        roundtrip,_=restore_active_rollout(self.e8,state,checkpoint_kind='PERIODIC_RESUME',policy_version=0)
        self.assertEqual(roundtrip.transition_at(0).rejection_or_downgrade_summary,
                         partial.transition_at(0).rejection_or_downgrade_summary)
        restored=buffer
        old,active=rollout_policy_components(restored,self.e8,'cpu')
        updater=CAGATMAPPORecurrentPPOUpdater(actor,critic,self.e8)
        minibatch=build_recurrent_ppo_minibatch(restored,self.e8);evaluation=updater.evaluate_minibatch(minibatch)
        loss=compute_configured_batched_ppo_objective_and_loss(new_joint_log_prob=evaluation.policy.joint_log_prob,
            old_joint_log_prob=minibatch.old_joint_log_prob,advantage=minibatch.advantage,current_value=evaluation.current_value,
            return_target=minibatch.return_target,entropy=evaluation.policy.entropy,sequence_valid_mask=minibatch.sequence_valid_mask,
            config=self.e8,new_branch_log_probs=torch.stack(list(evaluation.policy.component_log_probs.values()),-1),
            old_branch_log_probs=old,active_branch_indicators=active)
        expected=-(loss.flattened.branch_surrogate.sum((0,1))/active.reshape(256,4,-1).sum((0,1)).clamp_min(1))
        nonempty=active.reshape(256,4,-1).sum((0,1))>0
        torch.testing.assert_close(loss.actor_loss,expected[nonempty].mean())
        # Four bounded unit-fixture PPO epochs; no formal experiment run.
        output=updater.update(restored);self.assertTrue(output.old_policy_snapshot_preserved)
        summary=updater.last_integrated_summary
        json.dumps(summary,allow_nan=False);self.assertEqual(len(summary['ppo_epochs']),4)
        self.assertIn('tx_upper',summary['ppo_epochs'][0]);self.assertIn('cpu_joint',summary['ppo_epochs'][0])

    def test_separate_upper_destination_normalization_and_joint_cpu_ratio(self):
        from src.models.ca_gat_mappo_ppo import compute_configured_ppo_objective_and_loss
        def loss_with(destination_visits):
            t=8;n=len(policy_component_names(self.e8));upper=torch.tensor(.1,requires_grad=True)
            destination=torch.tensor(.07,requires_grad=True);cpu=torch.tensor(.09,requires_grad=True)
            active=torch.zeros(t,1,n,dtype=torch.bool);active[...,1]=True;active[:destination_visits,:,2]=True;active[...,-1]=True
            delta=torch.stack([upper*0,upper,destination,upper*0,upper*0,upper*0,cpu]).view(1,1,n).expand(t,1,n)
            old=torch.full((t,1,n),-.8,requires_grad=True);old_used=torch.where(active,old,torch.zeros_like(old))
            new=torch.where(active,old_used.detach()+delta,torch.zeros_like(delta))
            out=compute_configured_ppo_objective_and_loss(new_joint_log_prob=new.sum(-1),old_joint_log_prob=old_used.sum(-1),
                advantage=torch.ones(t,1),current_value=torch.zeros(t,1),return_target=torch.zeros(t,1),entropy=torch.zeros(t,1),
                sequence_valid_mask=torch.ones(t,dtype=torch.bool),config=self.e8,new_branch_log_probs=new,
                old_branch_log_probs=old_used,active_branch_indicators=active)
            torch.testing.assert_close(out.branch_ratio[...,-1],torch.full((t,1),torch.exp(torch.tensor(.09))))
            out.actor_loss.backward();self.assertIsNone(old.grad)
            return upper.grad.clone(),destination.grad.clone(),cpu.grad.clone()
        a,b=loss_with(1),loss_with(8)
        for ga,gb in zip(a,b):torch.testing.assert_close(ga,gb,rtol=1e-6,atol=1e-6)

    def test_destination_conditional_semantics_unchanged(self):
        old,new=CAGATMAPPOActor(self.e7),CAGATMAPPOActor(self.e8)
        recurrent=old(self.batch).recurrent_features
        old_leaf=old.branch_logits('tx_select',recurrent,{},self.batch)
        new_leaf=new.branch_logits('tx_select',recurrent,{},self.batch)
        torch.testing.assert_close(old_leaf[...,1:].softmax(-1),new_leaf[...,1:].softmax(-1),atol=1e-7,rtol=1e-6)

    def test_cpu_mask_support_exact_and_nonstandard_frequency_count(self):
        from src.models.integrated_scheduler import cpu_joint_statistics
        actor=CAGATMAPPOActor(self.e8);recurrent=actor(self.batch).recurrent_features
        contracts=[]
        for c in self.masks.flattened():
            energy=np.ones_like(c.cpu_frequency_energy_mask);energy[1,2]=False
            q=c.cpu_queue_mask.copy();q[3]=False
            contracts.append(replace(c,cpu_queue_mask=q,cpu_frequency_energy_mask=energy))
        result=cpu_joint_statistics(actor,recurrent,self.batch,tuple(contracts),None,'deterministic',None)
        self.assertFalse(result['mask'][...,1+1*3+1].any());self.assertFalse(result['mask'][...,1+2*3:1+3*3].any())
        self.assertTrue((result['probabilities'][~result['mask']]==0).all())
        config=replace(self.e8,action=replace(self.e8.action,cpu_frequency_levels=(0.,.5,1.)))
        # Feature-module sizing is domain-derived. The production environment
        # still validates its frozen four-level action schema unchanged.
        from src.models.integrated_scheduler import DeadlineAwareJointCPUSchedulerV1
        other=DeadlineAwareJointCPUSchedulerV1(config,actor.spec)
        self.assertEqual(other.positive_levels.numel(),2)
        self.assertEqual(other(recurrent,self.batch)[0].shape[-1],9)

    def test_full_checkpoint_actor_optimizer_loader_roundtrip(self):
        from tests.test_ca_gat_mappo_checkpoint import make_payload,assert_nested_equal,training_state
        from src.models.ca_gat_mappo_checkpoint import restore_model_optimizer_and_rng
        from src.evaluation.actor_loader import load_final_actor_for_evaluation,EvaluationCheckpointError
        config=replace(self.e8,training=replace(self.e8.training,mappo=replace(self.e8.training.mappo,training_device='cpu')))
        for name in ('git_commit','git_branch','git_dirty'):object.__setattr__(config,name,getattr(self.e8,name))
        config=replace(config,environment=replace(config.environment,episode_horizon=512),
                       training=replace(config.training,mappo=replace(config.training.mappo,checkpoint_interval_steps=4096,max_training_environment_steps=32768)))
        for name in ('git_commit','git_branch','git_dirty'):object.__setattr__(config,name,getattr(self.e8,name))
        payload,actor,critic,updater=make_payload(config,'FINAL_COMPLETED')
        payload['checkpoint_kind']='PERIODIC_RESUME'
        payload['training_state']=training_state(config,'PERIODIC_RESUME',0)
        # A controlled unit objective populates Adam moments only, not an
        # environment training run. Unit checkpoint is inside TemporaryDirectory.
        sum(p.square().sum() for p in actor.parameters()).backward()
        updater.optimizers.actor_optimizer.step()
        payload['model_state']['actor']={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
        payload['optimizer_state']['actor_adam']=copy.deepcopy(updater.optimizers.actor_optimizer.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'unit-e8-final.pt';atomic_save_checkpoint(payload,path)
            read=load_checkpoint_payload(path);target=CAGATMAPPOActor(config);tc=MAPPOCentralizedCritic(config)
            optimizers=build_ca_gat_mappo_optimizers(target,tc,config)
            restore_model_optimizer_and_rng(config=config,payload=read,actor=target,critic=tc,optimizers=optimizers,dtype=torch.float32)
            assert_nested_equal(self,actor.state_dict(),target.state_dict())
            assert_nested_equal(self,updater.optimizers.actor_optimizer.state_dict(),optimizers.actor_optimizer.state_dict())
            final,*_=make_payload(config,'FINAL_COMPLETED',actor=actor,critic=critic,updater=updater)
            path=Path(directory)/'unit-final-completed.pt'
            atomic_save_checkpoint(final,path)
            evaluation=replace(config,mode='evaluation',launch_profile=None)
            for name in ('git_commit','git_branch','git_dirty'):object.__setattr__(evaluation,name,getattr(config,name))
            loaded=load_final_actor_for_evaluation(evaluation,path,evaluation_device='cpu')
            assert_nested_equal(self,actor.state_dict(),loaded.actor.state_dict())
            wrong=replace(evaluation,training=replace(evaluation.training,mappo=replace(evaluation.training.mappo,tx_upper_decoder_mode='legacy')))
            with self.assertRaises(EvaluationCheckpointError):load_final_actor_for_evaluation(wrong,path,evaluation_device='cpu')



if __name__=='__main__':unittest.main()
