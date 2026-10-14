"""Evaluation-only one-shot Route oracle reliability; no learner updates."""
from dataclasses import replace
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import cloudpickle

from src.env.environment import U2UMECEnvironment
from src.evaluation.route_counterfactual_oracle import (
    PROTOCOL, best_set, classify, external_digest, intervene, observe,
    rollout_branch, summary,
)
from src.evaluation.route_oracle import FrozenOraclePolicyRunnerCache, tensor_digest
from src.evaluation.route_counterfactual_analysis import anchored_classification
from src.evaluation.runner import FormalEvaluationRunner
from src.evaluation.service_telemetry import EvaluationServiceTelemetryCollector
from src.models.ca_gat_mappo import CAGATMAPPOActor
from tests.test_route_oracle_diagnostic import make_oracle_config


class RouteCounterfactualOracleTests(unittest.TestCase):
    def test_global_anchor_avoids_nontransitive_tie_chain(self):
        # The second Remote is within one slot of the fastest Remote. Local
        # is within one slot of that second Remote, but not of the fastest.
        # Global near-best must not accidentally promote Local via this chain.
        rows=[dict(route=1,final_status="DONE",latency_slots=3,task_service_energy_j=.4),
              dict(route=2,final_status="DONE",latency_slots=4,task_service_energy_j=.2),
              dict(route="local",final_status="DONE",latency_slots=5,task_service_energy_j=.1)]
        before=copy.deepcopy(rows)
        self.assertEqual(anchored_classification(rows),"R")
        self.assertEqual(anchored_classification(rows,"strict_completion"),"T")
        self.assertEqual(anchored_classification(rows,"energy_only"),"L")
        self.assertEqual(rows,before)

    def test_thresholds_and_failure_not_ranked(self):
        self.assertEqual(PROTOCOL["latency_tie_slots"], 1)
        def row(route, status, latency, energy):
            return dict(route=route, final_status=status, latency_slots=latency, task_service_energy_j=energy)
        a, b = row("local", "DONE", 10, .2), row(1, "DONE", 14, .1)
        self.assertEqual(classify([a,b]), "L")
        self.assertEqual(classify([a,b], "strict_completion"), "T")
        self.assertEqual(classify([a,b], "energy_only"), "R")
        self.assertEqual(classify([replace_dict(a, final_status="EXPIRED"), replace_dict(b, final_status="TRUNCATED")]), "F")
        self.assertEqual(best_set([replace_dict(a, final_status="EXPIRED")]), [])

    def test_one_shot_isolation_frozen_continuation_crn_and_terminal(self):
        torch.set_num_threads(1)
        config = make_oracle_config(horizon=5)
        env = U2UMECEnvironment(config)
        env.reset()
        env.step(env.canonical_proposals())
        source, task_id = 0, env.current_observations[0].private_queues.unbound.head_task_id
        masks = env.current_observations[source].action_masks
        remote = next(x for x,m in zip(masks.route_domain,masks.route_mask) if m and isinstance(x,int))
        proposals = list(env.canonical_proposals())
        proposals[source] = replace(proposals[source], route=remote)
        proposals = tuple(proposals)
        for route in ("local", remote):
            changed = intervene(env.current_observations, proposals, source, route)
            self.assertEqual(changed[source].route, route)
            self.assertEqual(changed[1:], proposals[1:])
            self.assertEqual(replace(changed[source],route=remote), proposals[source])
        with self.assertRaises(Exception):
            intervene(env.current_observations, proposals, source, source)
        actor = CAGATMAPPOActor(config).eval()
        frozen = FrozenOraclePolicyRunnerCache(config).get(actor, 0)
        hidden = actor.initial_hidden(1, device="cpu", dtype=torch.float32)
        decision = dict(env=env.clone_for_shadow(), source=source, task_id=task_id,
                        slot=env.slot, proposals=proposals, hidden_out=hidden.clone(),
                        hidden_out_digest=tensor_digest(hidden), hidden_in_digest=tensor_digest(hidden),
                        state_digest=env.state_fingerprint(), rng_digest=env.rng_fingerprint(),
                        prefix_tasks={}, prefix_cpu_energy=0., prefix_tx_energy=0.)
        decision = cloudpickle.loads(cloudpickle.dumps(decision))
        self.assertEqual(decision["env"].state_fingerprint(),decision["state_digest"])
        self.assertEqual(decision["env"].rng_fingerprint(),decision["rng_digest"])
        self.assertIs(decision["env"].executor.lifecycle, decision["env"].lifecycle)
        evaluator = object.__new__(FormalEvaluationRunner)
        collector = EvaluationServiceTelemetryCollector(evaluation_run_id="test",method_id="ca_gat_mappo",evaluation_seed=1044,episode_id=0)
        exogenous, rngs, actions = {}, {}, {0: ()}
        while True:
            actions[env.slot] = proposals
            step = observe(collector,env,proposals)
            digest,rng,_ = external_digest(evaluator,env,step.info)
            exogenous[step.info["slot"]],rngs[step.info["slot"]] = digest,rng
            if step.truncated:
                break
            output = frozen.act(step.observations,hidden)
            hidden,proposals = output.hidden_out,output.proposals
        root_digest = decision["env"].state_fingerprint()
        hidden_digest = tensor_digest(decision["hidden_out"])
        calls = frozen.act_call_count
        # Constructing a learner optimizer would make this test fail.
        with patch("torch.optim.Adam.__init__",side_effect=AssertionError("optimizer forbidden")):
            factual, trace = rollout_branch(evaluator,frozen,decision,remote,collector,exogenous,rngs,expected_actions=actions)
            local, local_trace = rollout_branch(evaluator,frozen,decision,"local",collector,exogenous,rngs)
        self.assertEqual(frozen.act_call_count-calls, 2*(config.environment.episode_horizon-decision["slot"]-1))
        self.assertEqual(root_digest,decision["env"].state_fingerprint())
        self.assertEqual(hidden_digest,tensor_digest(decision["hidden_out"]))
        self.assertEqual(factual["system"]["DONE"],summary(collector)["DONE"])
        self.assertEqual(factual["destination_uav"],remote)
        self.assertEqual(local["destination_uav"],source)
        for result, observed in ((factual,trace),(local,local_trace)):
            self.assertEqual(result["reliability"]["intervention_count"],1)
            self.assertTrue(result["reliability"]["exogenous_rng_each_slot_match"])
            self.assertIn(result["final_status"], {"DONE","EXPIRED","TRUNCATED"})
            self.assertEqual(result["final_status"],observed.tasks[task_id]["final_status"])
        self.assertFalse(frozen.actor.training)
        self.assertTrue(all(not p.requires_grad for p in frozen.actor.parameters()))
        self.assertTrue(all(p.grad is None for p in frozen.actor.parameters()))


def replace_dict(value, **updates):
    return {**value, **updates}


if __name__ == "__main__":
    unittest.main()
