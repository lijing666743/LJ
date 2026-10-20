"""Bounded, read-only policy/replay diagnostics for compatible QMIX checkpoints.

Rollouts use the real environment and must reproduce saved development records.
Backward probes compute gradients only: no optimizer is created or stepped.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..artifacts import _filesystem_path, atomic_write_text_group
from ..config import load_run_config
from ..env.environment import U2UMECEnvironment
from ..models.factorized_gat_qmix import FactorizedQMIX, METHOD_ID
from ..models.qmix_continuation import load_continuation_checkpoint, source_config, state_digest
from ..models.qmix_replay import ReplaySequence, SequenceReplay
from ..models.qmix_recurrent import reconstruct_replay_sequences, RECURRENT_CONTEXT_CONTRACT
from ..models.qmix_targets import double_q_next_values, td_lambda_targets, replay_return_targets
from .metrics import build_episode_metrics
from .qmix_runner import QMIXEvaluationRunner
from .runner import FormalEvaluationRunner, _canonical_sha256, _tensor_sha256, _proposal_record


def describe(values):
    array = np.asarray(values, dtype=float).reshape(-1)
    if not len(array):
        return {"count": 0, "mean": None, "min": None, "max": None}
    if not np.isfinite(array).all():
        raise ValueError("non-finite diagnostic value")
    return {"count": len(array), "mean": float(array.mean()),
            "min": float(array.min()), "max": float(array.max())}


def terminal_stage(task):
    """Stage after actual service, before expiry; terminal status erases it."""
    if task.destination is None:
        return "unbound"
    if task.destination == task.source_uav:
        return "local"
    return "tx" if task.remaining_bits > 0 else "cpu"


class DecisionProbe:
    def __init__(self):
        self.counts = Counter()
        self.routes = Counter()
        self.frequencies = Counter()
        self.margins = defaultdict(list)

    def observe(self, masks, output):
        for i, (mask, proposal) in enumerate(zip(masks, output.proposals[0][0])):
            if mask.route_branch_active:
                self.counts["route_decisions"] += 1
                choice = proposal.route if isinstance(proposal.route, str) else "remote"
                self.routes[choice] += 1
                values = output.branch_values["route"][0, 0, i].detach().cpu()
                if mask.route_mask[1] and mask.route_mask[2]:
                    self.margins["route_local_minus_defer"].append(float(values[1]-values[2]))
            if mask.cpu_queue_branch_active:
                self.counts["cpu_backlogged_decisions"] += 1
                idle = proposal.cpu_queue == "idle"
                self.counts["cpu_backlogged_idle"] += int(idle)
                self.counts["cpu_selected_queue"] += int(not idle)
                self.counts["cpu_selected_queue_zero_frequency"] += int(not idle and proposal.cpu_frequency == 0)
                self.frequencies[str(proposal.cpu_frequency)] += 1
                positive = np.asarray(mask.cpu_frequency_domain) > 0
                feasible = any(mask.cpu_queue_mask[j+1] and np.any(row & positive)
                               for j, row in enumerate(mask.cpu_frequency_energy_mask))
                self.counts["cpu_positive_feasible_decisions"] += int(feasible)
                self.counts["cpu_positive_feasible_but_idle_or_zero"] += int(
                    feasible and (idle or proposal.cpu_frequency == 0))
                queues = output.branch_values["cpu_queue"][0, 0, i].detach().cpu()
                frequency = output.cpu_frequency_values[0, 0, i].detach().cpu()
                scores = [float(queues[j+1] + frequency[j, k])
                          for j, row in enumerate(mask.cpu_frequency_energy_mask)
                          for k in np.flatnonzero(row & positive) if mask.cpu_queue_mask[j+1]]
                if scores:
                    self.margins["cpu_best_positive_minus_idle"].append(max(scores)-float(queues[0]))

    def result(self):
        return {"counts": dict(self.counts), "route_choices": dict(self.routes),
                "backlogged_frequency_choices": dict(self.frequencies),
                "q_margins": {key: describe(value) for key, value in self.margins.items()}}


class TracedQMIXRunner(QMIXEvaluationRunner):
    def traced_episode(self, seed):
        config = replace(self.config, seed=seed, launch_profile=None)
        env = U2UMECEnvironment(config)
        reset = env.reset()
        observations = reset.observations
        external = [FormalEvaluationRunner._external_trace_record(self, env, phase="reset", arrival=None)]
        actions, infos = [], []
        hidden = self.agent.initial_hidden()
        hidden_hashes = [_tensor_sha256(hidden)]
        probe = DecisionProbe()
        for slot in range(config.environment.episode_horizon):
            with torch.no_grad():
                batch = self.adapter.encode_step(observations, device=self.device, episode_start=slot == 0)
                masks = tuple(obs.action_masks for obs in observations)
                output = self.agent.actions(batch, hidden, ((masks,),))
            probe.observe(masks, output)
            proposals, hidden = output.proposals[0][0], output.hidden_out
            hidden_hashes.append(_tensor_sha256(hidden))
            actions.append(tuple(_proposal_record(p) for p in proposals))
            step = env.step(proposals)
            infos.append(step.info)
            external.append(FormalEvaluationRunner._external_trace_record(self, env,
                phase=f"slot_{int(step.info['slot'])}_complete", arrival=step.info["arrival"]))
            if step.terminated or step.truncated:
                if slot != config.environment.episode_horizon-1:
                    raise ValueError("unexpected early diagnostic termination")
                break
            observations = step.observations
        env.assert_invariants()
        episode = build_episode_metrics(method_id=METHOD_ID, evaluation_seed=seed, environment=env,
            slot_infos=infos, action_trace=tuple(actions), action_trace_sha256=_canonical_sha256(actions),
            external_trace=tuple(external), external_trace_sha256=_canonical_sha256(external),
            hidden_trace_sha256=tuple(hidden_hashes))
        tasks = [{**task.snapshot(), "terminal_stage": terminal_stage(task)}
                 for task in env.lifecycle.tasks.values()]
        expired = Counter(task["terminal_stage"] for task in tasks if task["outcome"] == "expired")
        if sum(expired.values()) != episode.metrics["expired"]:
            raise ValueError("expiration-stage accounting differs from environment")
        return episode, {"seed": seed, "metrics": dict(episode.metrics),
            **probe.result(), "expired_by_stage": dict(expired), "tasks": tasks}


def assert_saved_episode(episode, saved):
    actual = episode.artifact_record(saved["evaluation_run_id"])
    for key in ("metrics", "action_trace_sha256", "external_trace_sha256", "hidden_trace_sha256", "invariants"):
        if actual[key] != saved[key]:
            raise ValueError(f"diagnostic replay differs from saved evaluation: {key}")


def load_system(config, payload):
    system = FactorizedQMIX(config).cpu()
    target = FactorizedQMIX(config).cpu().requires_grad_(False).eval()
    system.load_state_dict(payload["online"], strict=True)
    target.load_state_dict(payload["target"], strict=True)
    return system, target


def gradient_probe(system, target, sequence, gamma, *, replay=None):
    """Real saved-action TD loss/gradients without any parameter update."""
    context_slots = 0
    if replay is not None:
        rebuilt, context_slots = reconstruct_replay_sequences(system, target, (sequence,), replay)
        sequence = rebuilt[0]
    system.zero_grad(set_to_none=True)
    output = system.agent.actions(sequence.actor, sequence.hidden, sequence.contracts,
                                  proposals=sequence.proposals)
    prediction = system.mixer(output.utilities, sequence.state)
    labels, return_stats = replay_return_targets(system, target, (sequence,), "cpu", gamma, replay=replay)
    td_target = labels[0]
    error = prediction-td_target
    sensitivity = torch.autograd.grad(prediction.sum(), output.utilities, retain_graph=True)[0]
    loss = .5 * error.square().mean()
    loss.backward()
    norms = {}
    for name, module in {**dict(system.agent.heads.items()), "gru": system.agent.gru,
                         "graph": system.agent.graph, "self_encoder": system.agent.self_encoder,
                         "mixer": system.mixer}.items():
        norms[name] = float(sum(p.grad.detach().square().sum() for p in module.parameters()
                               if p.grad is not None).sqrt())
    with torch.no_grad():
        greedy = system.agent.actions(sequence.actor, sequence.hidden, sequence.contracts)
        target_greedy = target.agent.actions(sequence.actor, sequence.target_hidden, sequence.contracts)
        current_labels, _ = replay_return_targets(system, system,
            (replace(sequence, target_hidden=sequence.hidden),), "cpu", gamma, replay=replay)
        current_td = current_labels[0]
        target_disagreements = {}
        for branch_index, branch in enumerate(system.agent.heads):
            common = greedy.active[..., branch_index] & target_greedy.active[..., branch_index]
            target_disagreements[branch] = {
                "both_active": int(common.sum()),
                "different": int(((greedy.indices[branch] != target_greedy.indices[branch]) & common).sum()),
            }
        branch_matches = torch.stack([output.indices[b] == greedy.indices[b]
                                      for b in system.agent.heads], -1)
        joint_match = branch_matches.all(-1)
    result = {"episode": sequence.episode, "first_slot": sequence.first_slot,
        "recurrent_context": RECURRENT_CONTEXT_CONTRACT if replay is not None else "saved_collection_hidden_diagnostic_only",
        "recurrent_context_slots": context_slots, **return_stats,
        "length": sequence.length, "loss": float(loss.detach()),
        "q_taken": describe(prediction.detach().numpy()), "td_target": describe(td_target.numpy()),
        "mean_abs_td": float(error.detach().abs().mean()),
        "utility": describe(output.utilities.detach().numpy()),
        "mixer_sensitivity": describe(sensitivity.detach().numpy()),
        "mixer_sensitivity_below_1e_6_fraction": float((sensitivity < 1e-6).float().mean()),
        "saved_vs_greedy_agent_joint_match_fraction": float(joint_match.float().mean()),
        "online_target_active_branch_disagreements": target_disagreements,
        "target_lag_mean_abs_td_target_change": float((current_td-td_target).abs().mean()),
        "branch_gradient_norms": norms,
        "active_branch_counts": dict(zip(system.agent.heads, output.active.sum((0,1,2)).tolist())),
        "mean_branch_q": {b:float(v.detach().mean()) for b,v in output.branch_values.items()}}
    system.zero_grad(set_to_none=True)
    return result


def hidden_probe(system, target, sequence, prefixes, gamma):
    """Compare collection states with current prefixes; not v6 learning starts."""
    online_hidden = system.agent.initial_hidden()
    target_hidden = target.agent.initial_hidden()
    with torch.no_grad():
        for prefix in prefixes:
            _, online_hidden = system.agent.encode(prefix.actor, online_hidden)
            _, target_hidden = target.agent.encode(prefix.actor, target_hidden)
        saved = system.agent.actions(sequence.actor, sequence.hidden, sequence.contracts)
        corrected = system.agent.actions(sequence.actor, online_hidden, sequence.contracts)
        rebuilt = replace(sequence, hidden=online_hidden, target_hidden=target_hidden)
        old_td = td_lambda_targets(sequence.reward, double_q_next_values(system,target,sequence,"cpu"),
                                  sequence.bootstrap,gamma)
        new_td = td_lambda_targets(sequence.reward, double_q_next_values(system,target,rebuilt,"cpu"),
                                  sequence.bootstrap,gamma)
    matches = torch.stack([saved.indices[b] == corrected.indices[b] for b in system.agent.heads],-1).all(-1)
    return {"episode": sequence.episode,"first_slot":sequence.first_slot,
        "online_hidden_rms_change":float((sequence.hidden-online_hidden).square().mean().sqrt()),
        "target_hidden_rms_change":float((sequence.target_hidden-target_hidden).square().mean().sqrt()),
        "greedy_agent_joint_disagreement_fraction":float((~matches).float().mean()),
        "td_target_mean_absolute_change":float((old_td-new_td).abs().mean())}


def branch_gradient_comparison(system, target, sequence, prefixes, gamma):
    online_hidden=system.agent.initial_hidden(); target_hidden=target.agent.initial_hidden()
    with torch.no_grad():
        for prefix in prefixes:
            _,online_hidden=system.agent.encode(prefix.actor,online_hidden)
            _,target_hidden=target.agent.encode(prefix.actor,target_hidden)
    rebuilt=replace(sequence,hidden=online_hidden,target_hidden=target_hidden)
    gradients=[]
    for item in (sequence,rebuilt):
        system.zero_grad(set_to_none=True)
        chosen=system.agent.actions(item.actor,item.hidden,item.contracts,proposals=item.proposals)
        prediction=system.mixer(chosen.utilities,item.state)
        value=td_lambda_targets(item.reward,double_q_next_values(system,target,item,'cpu'),item.bootstrap,gamma)
        (.5*(prediction-value).square().mean()).backward()
        gradients.append({name:torch.cat([p.grad.flatten().detach().clone() for p in module.parameters()])
            for name,module in {'route':system.agent.heads['route'],'cpu_queue':system.agent.heads['cpu_queue'],
                'cpu_frequency':system.agent.heads['cpu_frequency'],'gru':system.agent.gru}.items()})
    system.zero_grad(set_to_none=True)
    return {name:{'cosine':float(torch.nn.functional.cosine_similarity(
        gradients[0][name].double(),gradients[1][name].double(),dim=0).clamp(-1,1)),
        'old_norm':float(gradients[0][name].norm()),'rebuilt_norm':float(gradients[1][name].norm())}
        for name in gradients[0]}


def group_swap_episode(base, donor, seed, groups):
    """Closed-loop frozen intervention; both actors see the same legal snapshot."""
    if set(groups)-{'route','cpu'}:
        raise ValueError('unknown swap group')
    config=replace(base.config,seed=seed,launch_profile=None)
    env=U2UMECEnvironment(config); reset=env.reset(); observations=reset.observations
    hidden=base.agent.initial_hidden(); donor_hidden=donor.agent.initial_hidden()
    hidden_hashes=[_tensor_sha256(hidden)]; actions=[]; infos=[]
    external=[FormalEvaluationRunner._external_trace_record(base,env,phase='reset',arrival=None)]
    for slot in range(config.environment.episode_horizon):
        masks=tuple(obs.action_masks for obs in observations)
        with torch.no_grad():
            batch=base.adapter.encode_step(observations,episode_start=slot==0)
            current=base.agent.actions(batch,hidden,((masks,),))
            alternative=donor.agent.actions(batch,donor_hidden,((masks,),))
        hidden,donor_hidden=current.hidden_out,alternative.hidden_out
        hidden_hashes.append(_tensor_sha256(hidden))
        proposals=[]
        for original,replacement in zip(current.proposals[0][0],alternative.proposals[0][0]):
            values={}
            if 'route' in groups: values['route']=replacement.route
            if 'cpu' in groups: values.update(cpu_queue=replacement.cpu_queue,cpu_frequency=replacement.cpu_frequency)
            proposals.append(replace(original,**values))
        if not all(mask.is_legal(p) for mask,p in zip(masks,proposals)):
            raise ValueError('swap violated the original legal mask')
        actions.append(tuple(_proposal_record(p) for p in proposals))
        step=env.step(proposals); infos.append(step.info)
        external.append(FormalEvaluationRunner._external_trace_record(base,env,
            phase=f"slot_{int(step.info['slot'])}_complete",arrival=step.info['arrival']))
        if step.terminated or step.truncated: break
        observations=step.observations
    env.assert_invariants()
    return build_episode_metrics(method_id=METHOD_ID,evaluation_seed=seed,environment=env,
        slot_infos=infos,action_trace=tuple(actions),action_trace_sha256=_canonical_sha256(actions),
        external_trace=tuple(external),external_trace_sha256=_canonical_sha256(external),
        hidden_trace_sha256=tuple(hidden_hashes))


def run_group_swaps(config_paths,output_directory,progress=print):
    output=Path(output_directory)
    if _filesystem_path(output/'group_swaps.json').exists():
        raise FileExistsError('group-swap diagnostic already exists')
    runners=[]; payloads=[]
    for path in config_paths:
        config=load_run_config(path); checkpoint=config.artifact_paths()['final_checkpoint']
        payload=load_continuation_checkpoint(checkpoint); payloads.append((config,payload))
        evaluation=replace(config,mode='evaluation',training=replace(config.training,formal_rl_enabled=False))
        runners.append(TracedQMIXRunner(evaluation,checkpoint,device='cpu',checkpoint_purpose='development_checkpoint'))
    frozen=[state_digest(r.agent.state_dict()) for r in runners]
    passive=json.loads(_filesystem_path(output/'diagnostic.json').read_text(encoding='utf-8'))
    # Read original external hashes, rather than relying on intervention outcomes.
    config=payloads[-1][0]; checkpoint=Path(config.artifact_paths()['final_checkpoint'])
    directory=checkpoint.parent.parent/'checkpoint_progress'/f"step_{payloads[-1][1]['steps']:06d}"
    files=list(_filesystem_path(directory).glob('*/episode_metrics.jsonl'))
    saved={r['evaluation_seed']:r for r in map(json.loads,files[0].read_text(encoding='utf-8').splitlines())
           if r['method_id']==METHOD_ID}
    interventions=[]
    for groups in [('route',),('cpu',),('route','cpu')]:
        scenes=[]
        for seed in config.evaluation.evaluation_seeds:
            episode=group_swap_episode(runners[1],runners[0],seed,groups)
            assert episode.external_trace_sha256==saved[seed]['external_trace_sha256']
            scenes.append({'seed':seed,'metrics':dict(episode.metrics),
                'action_trace_sha256':episode.action_trace_sha256,'external_trace_sha256':episode.external_trace_sha256})
            progress(json.dumps({'stage':'frozen_group_swap','groups':groups,'seed':seed,
                'completed':episode.metrics['completed'],'optimizer_updates':0}),flush=True)
        interventions.append({'groups':groups,'scenes':scenes})
    assert frozen==[state_digest(r.agent.state_dict()) for r in runners]
    gradient_rows=[]
    source=payloads[-1][1]['replay']['sequences']
    for config,payload in payloads:
        system,target=load_system(config,payload)
        before=state_digest(system.state_dict()); before_target=state_digest(target.state_dict())
        for probe in passive['replay']:
            if probe['model_steps']!=payload['steps']:continue
            for index in probe['indices'][:2]:
                sequence=ReplaySequence.from_state(source[index])
                prefixes=[ReplaySequence.from_state(item) for item in source
                          if item['episode']==sequence.episode and item['first_slot']<sequence.first_slot]
                prefixes.sort(key=lambda p:p.first_slot)
                gradient_rows.append({'steps':payload['steps'],'stratum':probe['replay_stratum'],
                    'episode':sequence.episode,'first_slot':sequence.first_slot,
                    'gradients':branch_gradient_comparison(system,target,sequence,prefixes,config.training.qmix.gamma)})
        assert before==state_digest(system.state_dict()) and before_target==state_digest(target.state_dict())
    result={'schema':'qmix_frozen_group_swaps.v1','interventions':interventions,
        'prefix_gradient_comparison_purpose':'Collection-state comparison only; v6 learning already reconstructs current-weight prefixes.',
        'prefix_gradient_comparison':gradient_rows,'optimizer_updates':0,'parameters_unchanged':True,
        'external_traces_match':True,'claim_boundary':'Closed-loop frozen interventions and sampled gradients; not trained baseline performance or a unique root-cause proof.'}
    atomic_write_text_group([(output/'group_swaps.json',json.dumps(result,ensure_ascii=False,indent=2))],
                           group_name='QMIX frozen group swaps')
    return result


def run_diagnostic(config_paths, output_directory, progress=print, *, include_checkpoint=None):
    output = Path(output_directory)
    if _filesystem_path(output/'diagnostic.json').exists():
        raise FileExistsError("diagnostic already exists; refusing overwrite")
    results = []
    payloads = []
    plans = [(load_run_config(path), None) for path in config_paths]
    plans = [(config, Path(config.artifact_paths()["final_checkpoint"])) for config, _ in plans]
    if include_checkpoint is not None:
        plans.insert(-1, (plans[-1][0], Path(include_checkpoint)))
    for config, path in plans:
        payload = load_continuation_checkpoint(path)
        if payload['config_hash'] != config.config_hash:
            raise ValueError('diagnostic checkpoint differs from its training configuration')
        payloads.append((config,payload))
        evaluation = replace(config, mode="evaluation", launch_profile=None,
                             training=replace(config.training,formal_rl_enabled=False))
        runner = TracedQMIXRunner(evaluation,path,device="cpu",
                                 checkpoint_purpose="development_checkpoint")
        before = state_digest(runner.agent.state_dict())
        directory = path.parent.parent/'checkpoint_progress'/f'step_{payload["steps"]:06d}'
        files = list(_filesystem_path(directory).glob('*/episode_metrics.jsonl'))
        if len(files)!=1:
            raise ValueError("requires one completed saved evaluation per checkpoint")
        saved = {row['evaluation_seed']:row for row in map(json.loads,files[0].read_text(encoding='utf-8').splitlines())
                 if row['method_id']==METHOD_ID}
        scenes = []
        for seed in config.evaluation.evaluation_seeds:
            episode, trace = runner.traced_episode(seed)
            assert_saved_episode(episode,saved[seed])
            scenes.append(trace)
            progress(json.dumps({"stage":"frozen_behavior","steps":payload['steps'],"seed":seed,
                "completed":episode.metrics['completed'],"saved_replay_exact":True},ensure_ascii=False),flush=True)
        assert before==state_digest(runner.agent.state_dict())
        results.append({"steps":payload['steps'],"scenes":scenes,"frozen_parameters_unchanged":True})
    source = payloads[-1][1]['replay']['sequences']
    source_replay = SequenceReplay.from_state(payloads[-1][1]['replay'])
    boundary = len(payloads[0][1]['replay']['sequences'])
    rng = np.random.default_rng(20261016)
    strata = {"early":sorted(rng.choice(boundary,16,replace=False).tolist()),
              "late":sorted(rng.choice(np.arange(boundary,len(source)),16,replace=False).tolist())}
    probes = []
    for config,payload in payloads:
        system,target=load_system(config,payload)
        before=state_digest(system.state_dict()); before_target=state_digest(target.state_dict())
        for stratum,indices in strata.items():
            rows=[]; hidden_rows=[]
            for index in indices:
                sequence=ReplaySequence.from_state(source[index])
                rows.append(gradient_probe(system,target,sequence,config.training.qmix.gamma,replay=source_replay))
            # Two predefined prefix probes per stratum; do not select on results.
            for index in indices[:2]:
                sequence=ReplaySequence.from_state(source[index])
                prefixes=[ReplaySequence.from_state(item) for item in source
                          if item['episode']==sequence.episode and item['first_slot']<sequence.first_slot]
                prefixes.sort(key=lambda item:item.first_slot)
                hidden_rows.append(hidden_probe(system,target,sequence,prefixes,config.training.qmix.gamma))
            probes.append({"model_steps":payload['steps'],"replay_stratum":stratum,
                           "indices":indices,"gradient_probes":rows,"hidden_probes":hidden_rows})
            progress(json.dumps({"stage":"replay_backward_only","model_steps":payload['steps'],
                "stratum":stratum,"chunks":len(rows),"optimizer_updates":0}),flush=True)
        assert before==state_digest(system.state_dict()) and before_target==state_digest(target.state_dict())
    changes=[]
    for first,last in zip(results[0]['scenes'],results[-1]['scenes']):
        assert first['seed']==last['seed']
        old={t['task_id']:t for t in first['tasks']}; new={t['task_id']:t for t in last['tasks']}
        assert old.keys()==new.keys()
        transitions=Counter()
        for tid,item in old.items():
            if item['outcome']=='done' and new[tid]['outcome']!='done':
                transitions[new[tid]['outcome']+':'+new[tid]['terminal_stage']]+=1
        changes.append({"seed":first['seed'],"previously_done_now_failed":dict(transitions)})
    result={"schema":"qmix_limited_diagnostic.v2","behavior":results,"replay":probes,
        "learning_recurrent_context":RECURRENT_CONTEXT_CONTRACT,
        "collection_hidden_comparison_purpose":"Supplementary collection-state comparison, not the hidden states used by v6 TD learning.",
        "lost_completion_stages":changes,"optimizer_updates":0,
        "claim_boundary":"Frozen replay and sampled backward diagnostics; no trained performance or unique causal proof."}
    atomic_write_text_group([(output/'diagnostic.json',json.dumps(result,ensure_ascii=False,indent=2))],
                           group_name="QMIX limited diagnostic")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',action='append',required=True)
    parser.add_argument('--output-directory',required=True)
    parser.add_argument('--phase',choices=('behavior_replay','group_swaps'),default='behavior_replay')
    parser.add_argument('--include-checkpoint',help='Optional intermediate checkpoint from the second training configuration')
    args=parser.parse_args()
    if len(args.config)!=2:
        parser.error('exactly two ordered checkpoint configurations are required')
    torch.set_num_threads(1)
    if args.phase=='behavior_replay':
        run_diagnostic(args.config,args.output_directory,include_checkpoint=args.include_checkpoint)
    else:
        if args.include_checkpoint is not None:
            parser.error('--include-checkpoint is only valid for behavior_replay')
        run_group_swaps(args.config,args.output_directory)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
