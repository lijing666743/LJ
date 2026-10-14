"""Training-only all-Remote, full-episode system labels for E15.

This collector is called before a factual Remote first binding. It never steps
the factual environment, samples learner actions or owns an optimizer. Future
shadow outcomes are labels only and are not exported to actor observations.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

from ..models.destination_ranking import KEY, PROTOCOL, system_oracle_label
from .route_counterfactual_oracle import intervene
from .route_oracle import OracleReliabilityError, actor_state_digest, tensor_digest


def collect_destination_labels(env, observations, proposals, hidden_out, runner, *,
                               episode_id, environment_step):
    selected = []
    for source, proposal in enumerate(proposals):
        if isinstance(proposal.route, int) and not isinstance(proposal.route, bool):
            task_id = int(observations[source].private_queues.unbound.head_task_id)
            task = env.lifecycle.tasks[task_id]
            if task.binding_slot is not None or task.status.value != 'unbound':
                raise OracleReliabilityError('destination label is not a first binding')
            selected.append((source, task_id))
    if not selected:
        return []
    original = env.state_fingerprint()
    rng = env.rng_fingerprint()
    hidden_digest = tensor_digest(hidden_out)
    frozen_digest = actor_state_digest(runner.actor)
    records = []
    # The prefix is shared by every branch. Metrics include it in full-system
    # counts/energy; no fabricated task attribution or prefix replay is needed.
    for source, task_id in selected:
        begin = time.perf_counter()
        mask = observations[source].action_masks
        legal = [d for d,m in zip(mask.route_domain,mask.route_mask)
                 if m and isinstance(d,int) and not isinstance(d,bool)]
        branches, reference_tape = [], None
        for destination in legal:
            branch = env.clone_for_shadow()
            hidden = hidden_out.detach().clone()
            actions = intervene(branch.current_observations,proposals,source,destination)
            tape, steps = [], 0
            while True:
                result = branch.step(actions); steps += 1
                if steps == 1:
                    bound = branch.lifecycle.tasks[task_id]
                    if bound.destination != destination or bound.binding_slot != env.slot:
                        raise OracleReliabilityError('shadow did not execute the requested first Remote binding')
                tape.append((branch.exogenous_fingerprint(), branch.rng_fingerprint()))
                if result.terminated or result.truncated:
                    break
                output = runner.act(result.observations, hidden, episode_start=False)
                hidden, actions = output.hidden_out, output.proposals
            if steps != env.config.environment.episode_horizon-env.slot:
                raise OracleReliabilityError('destination shadow shortened its full horizon')
            if reference_tape is None:
                reference_tape = tape
            elif reference_tape != tape:
                raise OracleReliabilityError('destination shadow exogenous/RNG trace mismatch')
            tasks = list(branch.lifecycle.tasks.values())
            outcomes = Counter(t.outcome.value for t in tasks)
            if outcomes['none']:
                raise OracleReliabilityError('shadow has unresolved terminal tasks')
            target = branch.lifecycle.tasks[task_id]
            status = {'done':'DONE','expired':'EXPIRED','truncated':'TRUNCATED'}
            target_status = status[target.outcome.value]
            cpu_energy = float(branch.metrics.total_cpu_energy_j)
            tx_energy = float(branch.metrics.total_tx_energy_j)
            branches.append({'destination_uav':destination,
                'DONE':outcomes['done'],'EXPIRED':outcomes['expired'],'TRUNCATED':outcomes['truncated'],
                'target_outcome':target_status,'target_completion_slot':target.completion_slot,
                'target_latency_slots':None if target_status!='DONE' else target.completion_slot-target.arrival_slot,
                'cpu_energy_j':cpu_energy,'tx_energy_j':tx_energy,
                'system_service_energy_j':cpu_energy+tx_energy,'shadow_transitions':steps})
        label = system_oracle_label(branches,legal)
        chosen = proposals[source].route
        factual_branch = next(r for r in branches if r['destination_uav']==chosen)
        records.append({'schema':KEY,'episode_id':episode_id,'environment_step':environment_step,
            'slot':env.slot,'task_id':task_id,'source_uav':source,
            'legal_destinations':legal,'factual_chosen_destination':chosen,
            **label,'branches':branches,
            'recoverable_failure_proxy': factual_branch['target_outcome']!='DONE' and any(r['target_outcome']=='DONE' for r in branches),
            'shadow_transitions':sum(r['shadow_transitions'] for r in branches),
            'shadow_wall_seconds':time.perf_counter()-begin,
            'provenance':{'policy':runner.identity.to_dict(),'root_state':original,
                'environment_seed':env.config.seed,
                'root_rng':rng,'hidden_out':hidden_digest,'interventions_per_branch':1,
                'full_horizon':True,'all_legal_candidates':True,
                'crn_each_slot_match':True,'continuation':PROTOCOL['continuation']}})
    if (env.state_fingerprint()!=original or env.rng_fingerprint()!=rng
            or tensor_digest(hidden_out)!=hidden_digest or actor_state_digest(runner.actor)!=frozen_digest):
        raise OracleReliabilityError('destination oracle mutated factual state/hidden/frozen actor')
    return records


def write_labels(config, records):
    if not records:
        return
    path = Path(config.output.logs_dir)/config.run_id/'destination_system_oracle_labels_v1.jsonl'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        for row in records:
            f.write(json.dumps({'config_hash':config.config_hash,**row},allow_nan=False)+'\n')
