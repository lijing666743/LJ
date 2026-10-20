"""Factual-only credit records and bounded isolated paired-Q continuations."""
import copy

import torch

from .ca_gat_mappo import ActorObservationTensorizer
from .c_feature_qmix_credit import live_tasks,collect_credit_record
from .tx_paired_shadow import FrozenShadowSampler,selector_sources,run_pair
from ..evaluation.tx_paired_shadow_pilot import LifecycleLedger


class FrozenQShadowSampler:
    seed=FrozenShadowSampler.seed

    def __init__(self,agent,config,version):
        self.q_agent=copy.deepcopy(agent).eval().requires_grad_(False)
        # The reused pair validator accepts and hashes the actual C backbone.
        # The adapter also has a state-only Q head, so verify the complete frozen
        # agent separately around each pair. Selection uses exact Q scores.
        self.actor=self.q_agent.backbone
        for module in self.q_agent.modules():
            if isinstance(module,torch.nn.RNNBase):module.flatten_parameters()
        self.config,self.version=config,version
        self.device=next(self.actor.parameters()).device
        self.tensorizer=ActorObservationTensorizer(config)

    @torch.no_grad()
    def sample(self,env,hidden,namespace="continuation",index=0):
        batch=self.tensorizer.encode_step(env.current_observations,device=self.device,episode_start=env.slot==0)
        contracts=((tuple(obs.action_masks for obs in env.current_observations),),)
        output=self.q_agent.actions(batch,hidden.to(self.device),contracts)
        # Greedy continuation has no policy RNG. The explicit deterministic
        # address is the same in both arms; exogenous streams are checked too.
        tag=("deterministic_q_no_policy_rng",env.config.seed,env.slot)
        return tuple(output.proposals[0][0]),output.hidden_out.detach(),tag

    @torch.no_grad()
    def transmit(self,env,hidden,factual,source):
        if factual[source].tx_select!="idle":return factual[source],0
        batch=self.tensorizer.encode_step(env.current_observations,device=self.device,episode_start=env.slot==0)
        contracts=((tuple(obs.action_masks for obs in env.current_observations),),)
        output=self.q_agent.actions(batch,hidden.to(self.device),contracts,force_transmit_source=source)
        from dataclasses import replace
        candidate=output.proposals[0][0][source]
        return replace(factual[source],tx_select=candidate.tx_select,resource_group=candidate.resource_group,
                       resource_width=candidate.resource_width,power_level=candidate.power_level),0


class CFeatureCreditCollector:
    def __init__(self,trainer,episode):
        self.trainer,self.config,self.episode=trainer,trainer.config,episode
        self.ledger=LifecycleLedger()

    def before_step(self,environment,observations,proposals,hidden,output):
        before=live_tasks(environment,self.config)
        state=self.trainer.credit_collection_state
        m=self.config.training.mappo
        window=self.trainer.steps//m.rollout_length_slots
        if state["window"]!=window:
            state.update(window=window,selected=[],spent=0)
        pairs=[]
        cost=2*(self.config.environment.episode_horizon-environment.slot)
        # Match C's omission of shadow labels in its unoptimized final tail.
        if (window+1)*m.rollout_length_slots>self.config.training.qmix.max_training_environment_steps:
            return before,pairs
        sources=[]
        spent=state["spent"]
        for source in selector_sources(window,self.config.environment.uav_count,m.tx_shadow_max_pairs_per_rollout):
            if source not in state["selected"] and observations[source].action_masks.tx_branch_active and spent+cost<=m.tx_shadow_step_budget_per_rollout:
                sources.append(source);spent+=cost
        if sources:
            sampler=FrozenQShadowSampler(self.trainer.learner.online.agent,self.config,self.trainer.steps)
            for source in sources:
                from .qmix_continuation import state_digest
                frozen_before=state_digest(sampler.q_agent.state_dict())
                # The reused validator's PPO bookkeeping argument is unused by
                # this learner; remove those fields rather than calling it a
                # factual behavior probability or storing a fictitious advantage.
                row=run_pair(environment,sampler,hidden,output.hidden_out,proposals,source,.5,self.ledger)
                if state_digest(sampler.q_agent.state_dict())!=frozen_before:
                    raise ValueError("paired continuation mutated the complete frozen Q agent")
                row["frozen_q_agent_digest"]=frozen_before
                row.pop("old_p");row.pop("A_TX_raw")
                # Wall time belongs to run diagnostics, not deterministic
                # replay/training state used for exact restoration checks.
                row.pop("shadow_wall_seconds")
                row["credit_target"]="upper_Q_transmit_minus_idle_return_gap"
                pairs.append(row)
                state["selected"].append(source);state["spent"]+=row["shadow_steps"]
                state["total_shadow_steps"]+=row["shadow_steps"];state["pair_count"]+=1
        return before,pairs

    def after_step(self,environment,observations,proposals,result,before,pairs,done):
        from ..evaluation.tx_paired_shadow_pilot import digest
        actual=digest(result.info)
        if any(row["factual_first_step_digest"]!=actual for row in pairs):
            raise ValueError("paired label did not reproduce the real factual first step")
        self.ledger.update(environment,result)
        record=collect_credit_record(before=before,after=[] if done else live_tasks(environment,self.config),
            observations=observations,proposals=proposals,result=result,global_step=self.trainer.steps,pairs=pairs)
        self.trainer.replay.add_credit_record(self.episode,record)
