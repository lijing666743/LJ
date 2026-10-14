from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from src.artifacts import ArtifactConflictError
from src.config import CHECKPOINT_KIND_FINAL_COMPLETED, RouteDecoderMode
from src.env.environment import U2UMECEnvironment
from src.env.tasks import Task
from src.evaluation.runner import FormalEvaluationRunner
from src.evaluation.service_diagnostics import main
from src.evaluation.service_telemetry import EvaluationServiceTelemetryCollector, snapshot_tasks
from src.models.ca_gat_mappo_checkpoint import atomic_save_checkpoint
from tests.test_ca_gat_mappo_checkpoint import make_payload
from tests.test_formal_evaluation import make_evaluation_fixture


def collector():
    return EvaluationServiceTelemetryCollector(
        evaluation_run_id="test", method_id="ca_gat_mappo", evaluation_seed=1044, episode_id=0,
    )


def task(task_id=0, *, deadline=5):
    return Task.create(task_id=task_id, source_uav=0, data_bits=10.0,
                       cpu_cycles=20.0, arrival_slot=0, deadline_slot=deadline)


def info(slot, *, head=None, accepted=False, amount=0.0, reject=False,
         cpu_task=None, cycles=0.0, frequency=0.0, services=None):
    return {
        "slot": slot, "proposal": [{"uav_id": 0, "tx_select": 1 if head is not None else "idle"}],
        "executed": [{"uav_id": 0, "communication": {
            "head_task_id": head, "receiver_uav": 1 if head is not None else None,
            "accepted": accepted, "executed_power_w": 1.0 if accepted else 0.0,
            "executed_ru_indices": [1] if accepted else [],
        }, "cpu": {"head_task_id": cpu_task, "executed_frequency_hz": frequency}}],
        "rejection": [{"uav_id": 0, "reason": "half_duplex_conflict"}] if reject else [],
        "service": {
            "links": [{"sender_uav": 0, "service_bits": amount, "effective_rate_bps": amount,
                       "task_services": services if services is not None else (
                           [{"task_id": head, "amount": amount}] if amount > 0 else [])}] if accepted else [],
            "cpu": [{"executor_uav": 0, "task_id": cpu_task, "service_cycles": cycles}]
                   if cpu_task is not None else [],
        },
        "energy": {"debits": [{"uav_id": 0, "cpu_energy_j": 0.0,
                               "transmit_energy_j": 0.0, "residual_after_j": 10.0}]},
    }


class TestServiceStageClassification(unittest.TestCase):
    def test_unbound_tx_wait_cpu_wait_and_truncation(self):
        cases = (("UNBOUND", False, False), ("TX_WAIT", True, False),
                 ("CPU_WAIT", True, True))
        for expected, bind, enter_cpu in cases:
            with self.subTest(stage=expected):
                t = task()
                if bind:
                    t.bind_remote(1, 1)
                if enter_cpu:
                    t.apply_transmitted_bits(10.0)
                    t.enter_cpu(5, service_eligible_slot=6)
                before = snapshot_tasks([t])
                t.expire(5)
                c = collector()
                c.observe_step(before=before, after=snapshot_tasks([t]), step_info=info(5))
                r = c.task_records()[0]
                self.assertEqual(r["expire_stage"], expected)
                self.assertEqual(r["expired_slot"], 5)
                self.assertEqual(r["final_status"], "EXPIRED")
                if enter_cpu:
                    self.assertEqual(r["tx_complete_slot"], 5)
                    self.assertIsNone(r["cpu_first_service_slot"])
        t = task(deadline=20)
        t.bind_remote(1, 1)
        before = snapshot_tasks([t])
        t.mark_truncated()
        c = collector()
        c.observe_step(before=before, after=snapshot_tasks([t]), step_info=info(5))
        r = c.task_records()[0]
        self.assertEqual(r["final_status"], "TRUNCATED")
        self.assertIsNone(r["expire_stage"])
        self.assertIsNone(r["expired_slot"])
        self.assertEqual(r["terminal_stage"], "TX_WAIT")

    def test_positive_tx_and_cpu_service_are_joined_before_expiry(self):
        tx = task(0)
        tx.bind_remote(1, 1)
        cpu = task(1)
        cpu.bind_remote(1, 1)
        cpu.apply_transmitted_bits(10.0)
        cpu.enter_cpu(2, service_eligible_slot=3)
        before = snapshot_tasks([tx, cpu])
        tx.apply_transmitted_bits(3.0)
        cpu.apply_cpu_cycles(2.0)
        tx.expire(5)
        cpu.expire(5)
        c = collector()
        step = info(5, head=0, accepted=True, amount=3.0, cpu_task=1, cycles=2.0, frequency=2e9)
        after = snapshot_tasks([tx, cpu])
        original = copy.deepcopy((before, after, step))
        c.observe_step(before=before, after=after, step_info=step)
        self.assertEqual((before, after, step), original)
        r0, r1 = c.task_records()
        self.assertEqual(r0["expire_stage"], "TX_SERVICE")
        self.assertEqual(r0["tx_first_service_slot"], 5)
        self.assertEqual(r1["expire_stage"], "CPU_SERVICE")
        self.assertEqual(r1["cpu_first_service_slot"], 5)
        self.assertEqual(c.tx_slots[0]["transmitted_bits"], 3.0)
        self.assertEqual(c.tx_slots[0]["remaining_bits"], 7.0)
        self.assertEqual(c.cpu_slots[0]["served_cycles"], 2.0)

    def test_zero_service_rejection_and_multi_task_link(self):
        t = task()
        t.bind_remote(1, 1)
        c = collector()
        state = snapshot_tasks([t])
        c.observe_step(before=state, after=state, step_info=info(2, head=0, accepted=True))
        self.assertIsNone(c.task_records()[0]["tx_first_service_slot"])
        c.observe_step(before=state, after=state, step_info=info(3, head=0, reject=True))
        self.assertTrue(c.tx_slots[-1]["rejected"])
        self.assertFalse(c.tx_slots[-1]["accepted"])
        self.assertEqual(c.tx_slots[-1]["rejection_reason"], "half_duplex_conflict")
        t2 = task(1)
        t2.bind_remote(1, 1)
        before = snapshot_tasks([t, t2])
        t.apply_transmitted_bits(10.0)
        t.enter_cpu(4, service_eligible_slot=5)
        t2.apply_transmitted_bits(2.0)
        c.observe_step(before=before, after=snapshot_tasks([t, t2]), step_info=info(
            4, head=0, accepted=True, amount=12.0,
            services=[{"task_id": 0, "amount": 10.0}, {"task_id": 1, "amount": 2.0}],
        ))
        rows = c.tx_slots[-2:]
        self.assertEqual(sum(r["transmitted_bits"] for r in rows), 12.0)
        self.assertEqual(sum(r["link_summary"] for r in rows), 1)
        self.assertEqual(c.task_records()[0]["tx_complete_slot"], 4)

    def test_cpu_queue_start_timing_zero_frequency_and_done(self):
        t = task()
        t.bind_local(1)
        c = collector()
        state = snapshot_tasks([t])
        c.observe_step(before=state, after=state, step_info=info(2, cpu_task=0))
        self.assertEqual(c.cpu_slots[-1]["cpu_queue_length"], 1)
        self.assertEqual(c.cpu_slots[-1]["cpu_queue_cycles"], 20.0)
        self.assertIsNone(c.cpu_slots[-1]["served_task_id"])
        self.assertIsNone(c.task_records()[0]["cpu_first_service_slot"])
        t.enter_cpu(3, service_eligible_slot=3)
        t.apply_cpu_cycles(20.0)
        t.complete(3)
        c.observe_step(before=state, after=snapshot_tasks([t]), step_info=info(
            3, cpu_task=0, cycles=20.0, frequency=2e9))
        r = c.task_records()[0]
        self.assertEqual(r["done_slot"], 3)
        self.assertEqual(r["final_status"], "DONE")
        self.assertIsNone(r["tx_complete_slot"])
        self.assertIsNone(r["expire_stage"])


class TestServiceTelemetryIntegration(unittest.TestCase):
    def test_switch_is_action_reward_terminal_state_and_rng_neutral(self):
        with tempfile.TemporaryDirectory() as directory:
            source, config, checkpoint, _ = make_evaluation_fixture(
                Path(directory), evaluation_seeds=(1044,), multi_uav=True,
                route_decoder_mode=RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1,
            )
            # Short, congested test fixture: exercise DONE, EXPIRED and TRUNCATED.
            config = replace(config, environment=replace(
                config.environment, episode_horizon=24,
                task_data_bits_min=250_000.0, task_data_bits_max=250_000.0,
                task_cycles_per_bit_min=300.0, task_cycles_per_bit_max=300.0,
                task_deadline_factor_min=1.5, task_deadline_factor_max=1.5,
            ))
            object.__setattr__(config, "git_branch", "test-evaluator")
            object.__setattr__(config, "git_commit", "b" * 40)
            object.__setattr__(config, "git_dirty", False)
            source = replace(source, environment=config.environment, training=replace(
                source.training, mappo=replace(source.training.mappo,
                    checkpoint_interval_steps=24, max_training_environment_steps=48,
                    max_training_episodes=2,
                ),
            ))
            object.__setattr__(source, "git_branch", "test-branch")
            object.__setattr__(source, "git_commit", "a" * 40)
            object.__setattr__(source, "git_dirty", False)
            payload, *_ = make_payload(source, CHECKPOINT_KIND_FINAL_COMPLETED)
            checkpoint = Path(directory) / "long-test-final.pt"
            atomic_save_checkpoint(payload, checkpoint)
            environments = []
            class RecordedEnvironment(U2UMECEnvironment):
                def step(self, proposals):
                    result = super().step(proposals)
                    self.test_steps.append((result.reward, self.state_fingerprint(), self.rng_fingerprint()))
                    return result
            def factory(c):
                env = RecordedEnvironment(c)
                env.test_steps = []
                environments.append(env)
                return env
            results = []
            with patch.object(torch.Tensor, "backward", side_effect=AssertionError("no training")):
                for enabled in (False, True):
                    results.append(FormalEvaluationRunner(
                        config, checkpoint, evaluation_device="cpu", environment_factory=factory,
                        collect_service_telemetry=enabled,
                    ).run(write_artifacts=False))
            off, on = results
            self.assertEqual(off.evaluation_run_id, on.evaluation_run_id)
            self.assertEqual(off.aggregate_metrics, on.aggregate_metrics)
            for left, right, env_off, env_on in zip(off.episodes, on.episodes, environments[:4], environments[4:]):
                self.assertEqual(left.action_trace, right.action_trace)
                self.assertEqual(left.hidden_trace_sha256, right.hidden_trace_sha256)
                self.assertEqual(left.external_trace_sha256, right.external_trace_sha256)
                self.assertEqual(left.metrics, right.metrics)
                self.assertEqual(env_off.test_steps, env_on.test_steps)
                self.assertEqual(snapshot_tasks(env_off.lifecycle.tasks.values()), snapshot_tasks(env_on.lifecycle.tasks.values()))
                records = [r for r in on.task_lifecycle_trace if r["method_id"] == left.method_id]
                self.assertEqual(len(records), left.metrics["generated"])
                for status, key in (("DONE", "completed"), ("EXPIRED", "expired"), ("TRUNCATED", "truncated")):
                    self.assertEqual(sum(r["final_status"] == status for r in records), left.metrics[key])
                cpu = [r for r in on.cpu_service_trace if r["method_id"] == left.method_id]
                tx = [r for r in on.tx_service_trace if r["method_id"] == left.method_id]
                self.assertEqual(len(cpu), 24 * config.environment.uav_count)
                self.assertAlmostEqual(sum(r["transmitted_bits"] for r in tx), env_on.metrics.actual_served_bits, delta=1e-5)
                self.assertAlmostEqual(sum(r["served_cycles"] for r in cpu), env_on.metrics.actual_served_cycles, delta=1e-4)
            self.assertFalse(off.task_lifecycle_trace)
            self.assertTrue(on.task_lifecycle_trace)
            self.assertEqual({r["final_status"] for r in on.task_lifecycle_trace}, {"DONE", "EXPIRED", "TRUNCATED"})
            self.assertTrue(any(r["transmitted_bits"] > 0 for r in on.tx_service_trace))
            self.assertTrue(any(r["served_cycles"] > 0 for r in on.cpu_service_trace))

    def test_artifacts_are_opt_in_complete_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config, checkpoint, _ = make_evaluation_fixture(Path(directory), multi_uav=True)
            runner = FormalEvaluationRunner(config, checkpoint, evaluation_device="cpu", collect_service_telemetry=True)
            # A service-side conflict must fail before any environment is created.
            path = runner.service_diagnostic_artifact_paths()["tasks"]
            path.parent.mkdir(parents=True)
            path.write_text("existing\n", encoding="utf-8")
            with patch.object(runner, "environment_factory", side_effect=AssertionError("rollout forbidden")):
                with self.assertRaises(ArtifactConflictError):
                    runner.run()
            self.assertEqual(path.read_text(), "existing\n")
            path.unlink()
            result = runner.run()
            self.assertEqual(len(result.artifacts), 4)
            self.assertEqual(len(result.service_diagnostic_artifacts), 4)
            tasks = [json.loads(x) for x in Path(result.service_diagnostic_artifacts[1]).read_text().splitlines()]
            self.assertEqual(tasks, list(result.task_lifecycle_trace))
            self.assertTrue(all(r["final_status"] in {"DONE", "EXPIRED", "TRUNCATED"} for r in tasks))
            with self.assertRaises(ArtifactConflictError):
                runner.run()

    def test_dry_run_loads_only_and_selects_1044(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config, checkpoint, _ = make_evaluation_fixture(Path(directory))
            from src.config import write_config_snapshot
            config_path = write_config_snapshot(config)
            with patch.object(FormalEvaluationRunner, "run", side_effect=AssertionError("rollout forbidden")):
                with redirect_stdout(io.StringIO()) as output:
                    status = main([
                        "--config", str(config_path), "--evaluate-from", str(checkpoint),
                        "--evaluation-seed", "1044", "--evaluation-device", "cpu",
                        "--logs-dir", str(Path(directory) / "unused"), "--dry-run",
                    ])
            self.assertEqual(status, 0)
            record = json.loads(output.getvalue())
            self.assertFalse(record["rollout_started"])
            self.assertEqual(record["evaluation_seeds"], [1044])
            self.assertFalse((Path(directory) / "unused").exists())


if __name__ == "__main__":
    unittest.main()
