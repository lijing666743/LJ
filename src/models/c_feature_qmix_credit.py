"""Detached C-source responsibility/task/shadow targets for enhanced QMIX.

The mixer is trained on the unique team reward. Conserved per-agent rewards
and task/paired labels provide separately named auxiliary losses, never an
extra environment reward or actor input.
"""
import math

import torch

from .ca_gat_mappo import stack_actor_time
from .qmix_targets import join_return_context, td_lambda_targets
from .route_task_credit import task_features


CREDIT_CONTRACT = {
    "name": "c_source_responsibility_task_paired_auxiliary_q_v1",
    "team_reward": "original_scalar_once", "responsibility_td_coefficient": 1.,
    "route_event_q_coefficient": 1., "task_value_coefficient": .5,
    "tx_pair_gap_coefficient": .5, "return_context_max_slots": 128,
    "defer": "post_binding_task_value_not_eventual_binding_payoff",
    "tx_shadow_continuation": "frozen_greedy_q_no_policy_rng",
    "not_equivalent_to_ppo_advantage": True,
}


def live_tasks(environment, config):
    return [{"task_id":task.task_id,"source_uav":task.source_uav,
             "features":task_features(task.snapshot(),environment.slot,config)}
            for task in environment.lifecycle.tasks.values() if not task.is_terminal]


def collect_credit_record(*, before, after, observations, proposals, result, global_step, pairs):
    info=result.info;credit=info["agent_credit"]
    rewards=[float(v) for v in credit["agent_reward"]]
    if not math.isclose(math.fsum(rewards),result.reward,abs_tol=1e-9):
        raise ValueError("responsibility labels must conserve the original team reward")
    attributions={row["task_id"]:row for row in credit["terminal_attributions"]}
    terminals=[]
    for task in info["service"]["settled_tasks"]+info["service"]["truncated_tasks"]:
        outcome=task["outcome"]
        payoff=0.
        if outcome != "truncated":
            row=attributions[task["task_id"]]
            field,suffix=("completion_allocation","reward") if outcome=="done" else ("expiration_allocation","penalty")
            payoff=sum(float(row[field][role+"_"+suffix]) for role in ("source","destination"))
            if outcome=="expired":payoff=-payoff
        terminals.append({"task_id":task["task_id"],"source_uav":task["source_uav"],
                          "slot":int(info["slot"]),"outcome":outcome,"payoff":payoff})
    decisions=[{"task_id":int(obs.private_queues.unbound.head_task_id),"source_uav":obs.uav_id,
                "route":proposal.route,"slot":int(info["slot"])}
               for obs,proposal in zip(observations,proposals) if obs.action_masks.route_branch_active]
    return {"slot":int(info["slot"]),"global_step":global_step,"agent_reward":rewards,
            "team_reward":float(result.reward),"before":before,"after":after,
            "terminals":terminals,"decisions":decisions,"tx_pairs":pairs}


def next_components(online,target,sequence,device):
    with torch.no_grad():
        actor=sequence.actor.to(device)
        batch=actor if sequence.next_actor is None else stack_actor_time((actor,sequence.next_actor.to(device)))
        contracts=sequence.contracts if sequence.next_contracts is None else (sequence.contracts[0]+(sequence.next_contracts,),)
        selected=online.agent.actions(batch,sequence.hidden.to(device),contracts)
        evaluated=target.agent.actions(batch,sequence.target_hidden.to(device),contracts,proposals=selected.proposals)
        states=sequence.state.to(device)
        if sequence.next_state is not None:states=torch.cat((states,sequence.next_state.to(device)),1)
        totals=target.mixer(evaluated.utilities,states)
        team=torch.zeros_like(sequence.reward,device=device)
        roles=evaluated.utilities.new_zeros((*sequence.reward.shape,evaluated.utilities.shape[-1]))
        team[:,:-1]=totals[:,1:sequence.length]
        roles[:,:-1]=evaluated.utilities[:,1:sequence.length]
        if sequence.next_state is not None:
            team[:,-1]=totals[:,-1];roles[:,-1]=evaluated.utilities[:,-1]
        return team,roles


def credit_return_targets(online,target,sequences,replay,device,gamma):
    """Use at most four real 32-slot chunks; no fabricated live-episode future."""
    result=[];evaluated=extra=0
    for sequence in sequences:
        chunks=[];length=0
        for chunk in replay.future_sequences(sequence):
            if length and length+chunk.length>CREDIT_CONTRACT["return_context_max_slots"]:break
            chunks.append(chunk);length+=chunk.length
        context=join_return_context(chunks)
        team_next,role_next=next_components(online,target,context,device)
        team=td_lambda_targets(context.reward.to(device),team_next,context.bootstrap.to(device),gamma)
        records=replay.credit_records.get(sequence.episode,())
        selected=records[context.first_slot:context.first_slot+context.length]
        if len(selected)!=context.length:raise ValueError("missing responsibility records for real transitions")
        for time,row in enumerate(selected):
            if row["slot"]!=context.first_slot+time or not math.isclose(sum(row["agent_reward"]),float(context.reward[0,time]),abs_tol=2e-6):
                raise ValueError("responsibility/time labels disagree with replay team reward")
        rewards=torch.tensor([[row["agent_reward"] for row in selected]],device=device)
        a=rewards.shape[-1]
        flat=lambda x:x.permute(0,2,1).reshape(a,context.length)
        roles=td_lambda_targets(flat(rewards),flat(role_next),context.bootstrap.to(device).expand(a,-1),gamma)
        roles=roles.reshape(1,a,context.length).permute(0,2,1)
        result.append((team[:,:sequence.length],roles[:,:sequence.length]))
        evaluated+=context.length;extra+=context.length-sequence.length
    return result,{"td_return_context_slots":evaluated,"td_return_extra_slots":extra}


def replay_state_at(replay,episode,slot):
    for chunk in replay.sequences:
        if chunk.episode!=episode:continue
        offset=slot-chunk.first_slot
        if 0<=offset<chunk.length:return chunk.state[:,offset:offset+1]
        if offset==chunk.length and chunk.next_state is not None:return chunk.next_state
    raise ValueError("task-value target requested an unavailable real state")


def task_credit_labels(target,sequence,replay,gamma,device):
    records=replay.credit_records.get(sequence.episode,())
    if not records:raise ValueError("missing factual task-event credit records")
    terminals={};binds={}
    for row in records:
        for terminal in row["terminals"]:
            key=(terminal["source_uav"],terminal["task_id"])
            if key in terminals:raise ValueError("duplicate task terminal")
            terminals[key]=terminal
        for decision in row["decisions"]:
            if decision["route"]!="defer":
                key=(decision["source_uav"],decision["task_id"])
                if key in binds:raise ValueError("task was bound more than once")
                binds[key]=(row,decision)
    end=records[-1];frontier={(r["source_uav"],r["task_id"]):r for r in end["after"]}
    cache={}
    def task_value(slot,task):
        key=(slot,task["source_uav"],task["task_id"])
        if key not in cache:
            state=replay_state_at(replay,sequence.episode,slot).reshape(1,-1).to(device)
            with torch.no_grad():cache[key]=float(target.route_task_value(state,state.new_tensor([task["features"]]))[0])
        return cache[key]
    def event_return(key,slot):
        if key in terminals:
            row=terminals[key]
            if row["slot"]<slot:raise ValueError("task state follows terminal event")
            return gamma**(row["slot"]-slot)*row["payoff"]
        if key not in frontier:raise ValueError("live task has no factual frontier")
        return gamma**(end["slot"]+1-slot)*task_value(end["slot"]+1,frontier[key])
    routes=[];values=[]
    for offset,row in enumerate(records[sequence.first_slot:sequence.first_slot+sequence.length]):
        for task in row["before"]:
            key=(task["source_uav"],task["task_id"])
            values.append((offset,task["features"],event_return(key,row["slot"])))
        for decision in row["decisions"]:
            key=(decision["source_uav"],decision["task_id"]);label=event_return(key,row["slot"]);stop="event_or_live_frontier"
            if decision["route"]=="defer" and key in binds:
                bound,_=binds[key]
                if bound["slot"]<=row["slot"]:raise ValueError("Defer after binding")
                if key in terminals and terminals[key]["slot"]==bound["slot"]:
                    label=gamma**(bound["slot"]-row["slot"])*terminals[key]["payoff"]
                    stop="binding_same_slot_terminal"
                else:
                    post=next((r for r in bound["after"] if (r["source_uav"],r["task_id"])==key),None)
                    if post is None:raise ValueError("Defer binding lacks factual post-task state")
                    label=gamma**(bound["slot"]+1-row["slot"])*task_value(bound["slot"]+1,post)
                    stop="binding_value_bootstrap"
            routes.append((offset,decision["source_uav"],label,stop))
    return routes,values


def auxiliary_losses(online,target,sequence,output,replay,gamma,role_target,device):
    zero=output.utilities.sum()*0.
    role=.5*(output.utilities-role_target).square().mean()
    routes,values=task_credit_labels(target,sequence,replay,gamma,device)
    route=zero;value=zero;paired=zero
    if routes:
        ti=torch.tensor([r[0] for r in routes],device=device);ai=torch.tensor([r[1] for r in routes],device=device)
        qi=output.indices["route"][0,ti,ai]
        prediction=output.credit_values["route"][0,ti,ai,qi]
        route=.5*(prediction-prediction.new_tensor([r[2] for r in routes])).square().mean()
    if values:
        indices=torch.tensor([v[0] for v in values],device=device)
        states=sequence.state.to(device)[0,indices]
        features=states.new_tensor([v[1] for v in values])
        prediction=online.route_task_value(states,features)
        value=.5*(prediction-prediction.new_tensor([v[2] for v in values])).square().mean()
    pairs=[]
    for offset,row in enumerate(replay.credit_records[sequence.episode][sequence.first_slot:sequence.first_slot+sequence.length]):
        pairs.extend((offset,pair) for pair in row["tx_pairs"])
    if pairs:
        ti=torch.tensor([t for t,_ in pairs],device=device);ai=torch.tensor([p["source"] for _,p in pairs],device=device)
        binary=output.credit_values["tx_upper"][0,ti,ai]
        labels=binary.new_tensor([p["G_T"]-p["G_I"] for _,p in pairs])
        paired=.5*((binary[:,1]-binary[:,0])-labels).square().mean()
    stats={"responsibility_loss":float(role.detach()),"route_event_loss":float(route.detach()),
           "task_value_loss":float(value.detach()),"tx_pair_loss":float(paired.detach()),
           "route_credit_samples":len(routes),"task_value_samples":len(values),"tx_pair_samples":len(pairs)}
    return role+route+.5*value+.5*paired,stats
