"""Frozen CPU capacity, value-ranking and load audit; no learning or mask edits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path

import torch

from ..artifacts import _filesystem_path, atomic_write_text_group
from ..config import load_run_config
from ..env.environment import U2UMECEnvironment
from ..models.factorized_gat_qmix import METHOD_ID
from ..models.integrated_scheduler import cpu_deadline_feasibility
from ..models.qmix_continuation import state_digest
from ..policies.heuristic_policy import HeuristicPolicy
from .actor_loader import checkpoint_sha256
from .metrics import build_episode_metrics
from .qmix_limited_diagnostic import assert_saved_episode, terminal_stage
from .qmix_runner import QMIXEvaluationRunner
from .runner import FormalEvaluationRunner, _canonical_sha256, _tensor_sha256, _proposal_record


def capacity_atoms(mask, delta_t):
    """Enumerate current legal CPU pairs with exact inclusive-deadline arithmetic.

    This optimistic single-head test repeats the current frequency until the
    deadline. It does not reserve future energy, model competition, or predict
    future policy actions. No legality or observation is changed.
    """
    if mask.cpu_deadline_context is None:
        raise ValueError("diagnostic requires exact slot-start CPU deadline context")
    atoms = []
    for qi, queue in enumerate(mask.cpu_queue_domain[1:], 1):
        if not mask.cpu_queue_mask[qi]:
            continue
        cycles, slack, fmax = mask.cpu_deadline_context[qi - 1]
        legal = mask.mask_for("cpu_frequency", {"cpu_queue": queue})
        for fi, level in enumerate(mask.cpu_frequency_domain):
            if legal[fi] and level > 0:
                required, margin = cpu_deadline_feasibility(cycles, slack, level * fmax, delta_t)
                atoms.append({"queue_index": qi, "queue": queue, "frequency_index": fi,
                              "frequency": level, "required_slots": required,
                              "completion_margin": margin, "capacity_feasible": math.isfinite(required)})
    return atoms


class CPUDeadlineProbe:
    def __init__(self, uav_count, delta_t):
        self.delta_t = delta_t
        self.counts = Counter()
        self.by_uav = [Counter() for _ in range(uav_count)]
        self.margins = defaultdict(list)
        self.examples = []
        self.task_service = Counter()
        self.task_energy = Counter()
        self.completed_by_service = Counter()
        self.outcomes = Counter()

    def decision(self, observation, proposal, queue_q, frequency_q):
        mask = observation.action_masks
        uav = observation.uav_id
        row = self.by_uav[uav]
        queues = observation.private_queues.cpu_by_source
        row["backlog_task_slots"] += int(queues.task_count.sum())
        row["backlog_cycle_slots"] += float(queues.remaining_cycles.sum())
        row["max_backlog_tasks"] = max(row["max_backlog_tasks"], int(queues.task_count.sum()))
        row["max_cpu_frequency_hz"] = observation.self_resources.max_cpu_frequency_hz
        if not mask.cpu_queue_branch_active:
            return
        local = Counter(backlogged_decisions=1)
        atoms = capacity_atoms(mask, self.delta_t)
        feasible = [atom for atom in atoms if atom["capacity_feasible"]]
        qi = mask.cpu_queue_domain.index(proposal.cpu_queue)
        fi = mask.cpu_frequency_domain.index(proposal.cpu_frequency)
        idle = qi == 0
        zero = not idle and proposal.cpu_frequency == 0
        local["idle"] = int(idle)
        local["selected_queue_zero_frequency"] = int(zero)
        local["energy_positive_available"] = int(bool(atoms))
        local["capacity_feasible_head_available"] = int(bool(feasible))
        local["energy_positive_but_no_capacity_feasible_head"] = int(bool(atoms) and not feasible)
        local["feasible_available_but_idle_or_zero"] = int(bool(feasible) and (idle or zero))
        local["no_feasible_head_idle_or_zero"] = int(not feasible and (idle or zero))
        local["positive_service_choice"] = int(not idle and not zero)
        local[f"frequency_{proposal.cpu_frequency}"] += 1
        feasible_queues = {atom["queue_index"] for atom in feasible}
        positive = not idle and not zero
        chosen_feasible = any(atom["queue_index"] == qi and atom["frequency_index"] == fi for atom in feasible)
        local["positive_chosen_pair_capacity_feasible"] = int(positive and chosen_feasible)
        local["positive_chosen_head_feasible_but_frequency_insufficient"] = int(
            positive and qi in feasible_queues and not chosen_feasible)
        local["positive_impossible_head_when_other_head_feasible"] = int(
            positive and qi not in feasible_queues and bool(feasible))
        critical = any(atom["completion_margin"] == 0 and not any(
            other["queue_index"] == atom["queue_index"] and other["completion_margin"] > 0
            for other in feasible) for atom in feasible)
        local["last_chance_head_available"] = int(critical)
        local["last_chance_available_but_idle_or_zero"] = int(critical and (idle or zero))
        chosen_q = float(queue_q[qi]) + (float(frequency_q[qi-1][fi]) if not idle else 0.0)
        if feasible:
            best_feasible_q = max(float(queue_q[atom["queue_index"]]) +
                                  float(frequency_q[atom["queue_index"]-1][atom["frequency_index"]])
                                  for atom in feasible)
            gap = best_feasible_q - chosen_q
            self.margins["best_capacity_feasible_minus_chosen"].append(gap)
            self.margins["best_capacity_feasible_minus_idle"].append(best_feasible_q-float(queue_q[0]))
            # 1e-6 is a Q comparison tolerance only, never a capacity tolerance.
            local["feasible_strictly_outranked_by_selected"] = int(gap < -1e-6)
            local["feasible_tied_with_selected"] = int(abs(gap) <= 1e-6)
            if idle or zero:
                self.margins["best_feasible_minus_chosen_when_idle_or_zero"].append(gap)
        if len(self.examples) < 12 and feasible and (idle or zero):
            self.examples.append({"slot": observation.slot, "uav": uav,
                                  "proposal_queue": proposal.cpu_queue, "proposal_frequency": proposal.cpu_frequency,
                                  "chosen_q": chosen_q, "best_feasible_q": best_feasible_q,
                                  "heads": [list(head) for head in mask.cpu_deadline_context]})
        self.counts.update(local)
        row.update(local)

    def settlement(self, info):
        completed = {task["task_id"] for task in info["service"]["settled_tasks"] if task["outcome"] == "done"}
        served_ids = set()
        for service in info["service"]["cpu"]:
            row = self.by_uav[service["executor_uav"]]
            task_id = service["task_id"]
            cycles, energy = service["service_cycles"], service["cpu_energy_j"]
            row["actual_service_cycles"] += cycles
            row["actual_cpu_energy_j"] += energy
            row["actual_positive_service_slots"] += int(cycles > 0)
            if cycles > 0:
                served_ids.add(task_id)
                self.task_service[task_id] += cycles
                self.task_energy[task_id] += energy
            if task_id in completed:
                self.completed_by_service[task_id] += 1
                row["actual_completions"] += 1
        if not completed <= served_ids:
            raise ValueError("completion without positive actual CPU service")

    def finish(self, env, metrics):
        tasks = list(env.lifecycle.tasks.values())
        for task in tasks:
            outcome = task.snapshot()["outcome"]
            self.outcomes[outcome] += 1
            if task.destination is not None:
                row = self.by_uav[task.destination]
                row["assigned_tasks"] += 1
                row["assigned_initial_cycles"] += task.cpu_cycles
                row[f"assigned_{outcome}"] += 1
                row[f"assigned_from_{task.source_uav}"] += 1
            actual = self.task_service[task.task_id]
            if not math.isclose(actual, task.cpu_cycles-task.remaining_cycles, rel_tol=1e-12, abs_tol=1e-5):
                raise ValueError("task cycle service differs from actual lifecycle")
            if outcome == "done" and self.completed_by_service[task.task_id] != 1:
                raise ValueError("completion execution credit differs from task outcome")
            if outcome == "expired":
                self.counts[f"expired_{terminal_stage(task)}"] += 1
            if task.task_id in self.task_energy:
                self.counts[f"cpu_energy_to_{outcome}_j"] += self.task_energy[task.task_id]
        if self.outcomes["done"] != metrics["completed"]:
            raise ValueError("completed task count differs from evaluation")
        energy = math.fsum(row["actual_cpu_energy_j"] for row in self.by_uav)
        if not math.isclose(energy, metrics["cpu_energy_j"], rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("actual CPU service energy differs from evaluation")
        margins = {name: {"count": len(values), "sum": math.fsum(values),
                          "mean": math.fsum(values)/len(values), "min": min(values), "max": max(values)}
                   for name, values in self.margins.items()}
        return {"counts": dict(self.counts), "by_uav": [dict(row) for row in self.by_uav],
                "q_margins": margins, "examples": self.examples, "task_outcomes": dict(self.outcomes),
                "actual_completion_cycle_energy_accounting_verified": True}


def replace_proposal_groups(original, donor, groups):
    """Diagnostic group replacement; does not install a learned baseline policy."""
    if not set(groups) <= {"route", "cpu"}:
        raise ValueError("only route and CPU diagnostic groups are supported")
    values = {}
    if "route" in groups:
        values["route"] = donor.route
    if "cpu" in groups:
        values.update(cpu_queue=donor.cpu_queue, cpu_frequency=donor.cpu_frequency)
    return replace(original, **values)


def frozen_episode(runner, seed, *, heuristic_groups=()):
    if not set(heuristic_groups) <= {"route", "cpu"}:
        raise ValueError("only route and CPU diagnostic groups are supported")
    config = replace(runner.config, seed=seed, launch_profile=None)
    env = U2UMECEnvironment(config)
    observations = env.reset().observations
    external = [FormalEvaluationRunner._external_trace_record(runner, env, phase="reset", arrival=None)]
    actions, infos = [], []
    hidden = runner.agent.initial_hidden()
    hidden_hashes = [_tensor_sha256(hidden)]
    probe = CPUDeadlineProbe(config.environment.uav_count, config.environment.slot_duration_s)
    donor = HeuristicPolicy(config, policy_seed=seed) if heuristic_groups else None
    for slot in range(config.environment.episode_horizon):
        with torch.no_grad():
            batch = runner.adapter.encode_step(observations, device=runner.device, episode_start=slot == 0)
            output = runner.agent.actions(batch, hidden, ((tuple(obs.action_masks for obs in observations),),))
        proposals, hidden = output.proposals[0][0], output.hidden_out
        if donor is not None:
            proposals = tuple(replace_proposal_groups(p, donor.act(obs), heuristic_groups)
                              for p, obs in zip(proposals, observations))
        if not all(obs.action_masks.is_legal(p) for p, obs in zip(proposals, observations)):
            raise ValueError("diagnostic group replacement violated the original masks")
        queues = output.branch_values["cpu_queue"][0, 0].detach().cpu().tolist()
        frequencies = output.cpu_frequency_values[0, 0].detach().cpu().tolist()
        for i, obs in enumerate(observations):
            probe.decision(obs, proposals[i], queues[i], frequencies[i])
        hidden_hashes.append(_tensor_sha256(hidden))
        actions.append(tuple(_proposal_record(p) for p in proposals))
        step = env.step(proposals)
        probe.settlement(step.info)
        infos.append(step.info)
        external.append(FormalEvaluationRunner._external_trace_record(runner, env,
            phase=f"slot_{int(step.info['slot'])}_complete", arrival=step.info["arrival"]))
        if step.terminated or step.truncated:
            if slot != config.environment.episode_horizon - 1:
                raise ValueError("unexpected early termination")
            break
        observations = step.observations
    env.assert_invariants()
    episode = build_episode_metrics(method_id=METHOD_ID, evaluation_seed=seed, environment=env,
        slot_infos=infos, action_trace=tuple(actions), action_trace_sha256=_canonical_sha256(actions),
        external_trace=tuple(external), external_trace_sha256=_canonical_sha256(external),
        hidden_trace_sha256=tuple(hidden_hashes))
    return episode, {"seed": seed, "metrics": dict(episode.metrics), **probe.finish(env, episode.metrics)}


def aggregate_scenes(scenes):
    counts, uavs, margins = Counter(), [Counter() for _ in scenes[0]["by_uav"]], defaultdict(list)
    for scene in scenes:
        counts.update(scene["counts"])
        for index, row in enumerate(scene["by_uav"]):
            maximum = max(uavs[index]["max_backlog_tasks"], row["max_backlog_tasks"])
            uavs[index].update(row)
            uavs[index]["max_backlog_tasks"] = maximum
            uavs[index]["max_cpu_frequency_hz"] = row["max_cpu_frequency_hz"]
        for name, values in scene["q_margins"].items():
            margins[name].append(values)
    return {"counts": dict(counts), "by_uav": [dict(row) for row in uavs],
            "q_margins": {name: {"count": sum(row["count"] for row in rows),
                                      "mean": math.fsum(row["sum"] for row in rows)/sum(row["count"] for row in rows)}
                          for name, rows in margins.items()}}


def run(config_path, output_path, progress=print):
    config = load_run_config(config_path)
    output_path = Path(output_path)
    if _filesystem_path(output_path).exists():
        raise FileExistsError("refusing to overwrite CPU diagnostic evidence")
    final = Path(config.artifact_paths()["final_checkpoint"])
    evaluation = replace(config, mode="evaluation", launch_profile=None,
                         training=replace(config.training, formal_rl_enabled=False))
    records = []
    for steps in (4000, 8000):
        checkpoint = final if steps == config.training.qmix.max_training_environment_steps else final.with_name(f"step_{steps:06d}.pt")
        runner = QMIXEvaluationRunner(evaluation, checkpoint, device="cpu", checkpoint_purpose="development_checkpoint",
                                      expected_training_steps=steps)
        before = state_digest(runner.agent.state_dict())
        before_file = checkpoint_sha256(checkpoint)
        directory = checkpoint.parent.parent / "checkpoint_progress" / f"step_{steps:06d}"
        files = list(_filesystem_path(directory).glob("*/episode_metrics.jsonl"))
        if len(files) != 1:
            raise ValueError("requires exactly one saved evaluation per checkpoint")
        saved = {row["evaluation_seed"]: row for row in map(json.loads, files[0].read_text(encoding="utf-8").splitlines())
                 if row["method_id"] == METHOD_ID}
        scenes = []
        for seed in config.evaluation.evaluation_seeds:
            episode, diagnostic = frozen_episode(runner, seed)
            assert_saved_episode(episode, saved[seed])
            scenes.append(diagnostic)
            progress(json.dumps({"stage": "cpu_deadline_frozen", "steps": steps, "seed": seed,
                                 "completed": episode.metrics["completed"], "original_replay_exact": True,
                                 "optimizer_updates": 0}), flush=True)
        if before != state_digest(runner.agent.state_dict()) or before_file != checkpoint_sha256(checkpoint):
            raise ValueError("diagnostic changed model parameters or checkpoint")
        records.append({"training_steps": steps, "checkpoint_sha256": before_file,
                        "scenes": scenes, **aggregate_scenes(scenes)})
    result = {"schema": "qmix_cpu_deadline_diagnostic.v1", "optimizer_updates": 0,
              "episodes": sum(len(record["scenes"]) for record in records),
              "checkpoint_parameters_unchanged": True, "original_metrics_actions_hidden_external_invariants_exact": True,
              "capacity_definition": "Exact float64 service on a slot-start head at a current legal positive frequency, repeated through inclusive deadline; optimistic, without future energy/competition/policy guarantees.",
              "claim_boundary": "Read-only single-seed development diagnosis; capacity availability does not prove that immediate service is optimal or that a new policy improves performance.",
              "behavior": records}
    atomic_write_text_group([(output_path, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))],
                           group_name="QMIX CPU deadline diagnostic")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.config, args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
