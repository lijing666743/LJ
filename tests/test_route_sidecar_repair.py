"""Synthetic JSON/checkpoint metadata tests; no Trainer, policy, or optimizer runs."""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.config import RunConfig
from src.evaluation.route_sidecar_repair import (
    ResumeContext, SidecarRepairError, apply_route_repair, load_resume_context,
    audit_conditional_resume, scan_route_sidecar, sha256,
)
from src.models.ca_gat_mappo_route_telemetry import route_diagnostic_schema_header
from src.training_artifacts import RouteDiagnosticArtifactError, RouteDiagnosticArtifactWriter


class RouteSidecarRepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = RunConfig()
        cfg = replace(base, output=replace(base.output, logs_dir=self.tmp.name),
                      training=replace(base.training, mappo=replace(base.training.mappo,
                          route_diagnostic_samples_enabled=True)))
        object.__setattr__(cfg, 'git_commit', 'a' * 40)
        self.run = Path(cfg.artifact_paths()['route_diagnostic_samples']).parent
        (self.run / 'checkpoints').mkdir(parents=True)
        cp = self.run / 'checkpoints' / 'step_4000.pt'
        cp.write_bytes(b'synthetic-checkpoint-identity-no-model-training')
        self.context = ResumeContext(self.run, cp, sha256(cp), 4000, 3840, cfg)
        self.header = route_diagnostic_schema_header(cfg)

    def row(self, step, task=7, agent=0):
        cfg = self.context.config
        candidates = [i for i in range(cfg.environment.uav_count) if i != agent]
        return {
            'record_type': 'route_diagnostic_sample',
            **{k: self.header[k] for k in ('schema_version', 'utility_label', 'run_id', 'config_hash', 'git_commit')},
            'episode_id': step // cfg.environment.episode_horizon,
            'global_environment_step': step, 'route_slot': step % cfg.environment.episode_horizon,
            'policy_version': step // cfg.training.mappo.rollout_length_slots,
            'source_uav': agent, 'head_task_id': task,
            'primary_key': [cfg.run_id, step // cfg.environment.episode_horizon, step, agent, task],
            'task': {'task_id': task, 'arrival_slot': 0}, 'source': {},
            'candidates': [{'candidate_uav': i, 'link_history': {
                name: [None] * cfg.environment.ru_count for name in (
                    'historical_quality_by_ru', 'historical_quality_valid_mask',
                    'historical_interference_w_by_ru', 'historical_interference_valid_mask')}}
                for i in candidates],
            'policy': {'actual_route': 'defer', 'route_entries': [
                {'action_index': i} for i in range(len(cfg.action.route_fixed_actions) + len(candidates))],
                'remote_probability_by_candidate': [{'candidate_uav': i} for i in candidates]},
            'join': {'outcome_status': 'not_joined_at_capture'},
        }

    def write(self, rows):
        data = b''.join((json.dumps(r, sort_keys=True) + '\n').encode() for r in [self.header, *rows])
        self.context.sidecar.write_bytes(data)
        return data

    def apply(self):
        plan = scan_route_sidecar(self.context)
        return apply_route_repair(self.context, plan, {'repair_tool_commit': 'synthetic-test'})

    def test_clean_matching_sidecar_is_noop(self):
        original = self.write([self.row(100), self.row(3999)])
        self.assertEqual(self.apply()['status'], 'no-op')
        self.assertEqual(self.context.sidecar.read_bytes(), original)
        self.assertFalse((self.run / 'sidecar_resume_repairs').exists())

    def test_4001_to_5359_tail_and_byte_exact_backup(self):
        rows = [self.row(s) for s in (100, 3999, 4001, 5000, 5359)]
        original = self.write(rows)
        plan = scan_route_sidecar(self.context)
        self.assertEqual((plan['safe_rows'], plan['orphan_rows']), (2, 3))
        result = self.apply()
        evidence = Path(result['quarantine_directory'])
        self.assertEqual((evidence / 'route_sidecar.original.jsonl').read_bytes(), original)
        tail = (evidence / 'route_sidecar.orphaned_after_step_4000.jsonl').read_bytes()
        self.assertEqual(self.context.sidecar.read_bytes() + tail, original)
        self.assertEqual(sha256(evidence / 'route_sidecar.original.jsonl'), result['sidecar_sha256_before'])
        self.assertEqual(sha256(self.context.sidecar), result['sidecar_sha256_after'])
        manifest = json.loads((evidence / 'repair_manifest.json').read_text())
        self.assertEqual(manifest['status'], 'applied')
        self.assertEqual(manifest['quarantined_row_count'], 3)

    def test_second_run_is_noop_and_has_no_duplicate_keys(self):
        self.write([self.row(3999), self.row(4001)])
        self.apply()
        self.assertEqual(self.apply()['status'], 'no-op')
        scan = scan_route_sidecar(self.context)
        self.assertEqual(scan['total_rows'], scan['unique_primary_keys'])
        self.assertEqual(scan['orphan_rows'], 0)

    def test_apply_preserves_conditional_trace_backup_and_live_bytes(self):
        original = self.write([self.row(3999), self.row(4001)])
        trace = self.run / 'conditional_credit_trace_v1.jsonl'
        trace_bytes = b'synthetic-trace-backup-only-no-policy\n'
        trace.write_bytes(trace_bytes)
        audit = {'trace_sha256': sha256(trace)}
        plan = scan_route_sidecar(self.context)
        result = apply_route_repair(self.context, plan,
                                   {'repair_tool_commit': 'synthetic-test'}, audit)
        evidence = Path(result['quarantine_directory'])
        self.assertLess(len(str(evidence / 'conditional_credit_trace.before_manual_resume.jsonl')), 260)
        self.assertEqual(Path(result['conditional_trace_backup']).read_bytes(), trace_bytes)
        self.assertEqual(trace.read_bytes(), trace_bytes)
        self.assertEqual((evidence / 'route_sidecar.original.jsonl').read_bytes(), original)

    def test_exact_checkpoint_index_is_uncommitted(self):
        self.write([self.row(3999), self.row(4000)])
        scan = scan_route_sidecar(self.context)
        self.assertEqual((scan['safe_rows'], scan['orphan_rows']), (1, 1))
        self.assertEqual(scan['orphan_min_step'], 4000)

    def test_cross_rollout_task_uses_action_time_not_task_arrival(self):
        self.write([self.row(s, task=99) for s in (3841, 3999, 4001)])
        scan = scan_route_sidecar(self.context)
        self.assertEqual(scan['safe_rows'], 2)
        self.assertEqual(scan['orphan_rows'], 1)
        self.apply()
        self.assertEqual(scan_route_sidecar(self.context)['safe_max_step'], 3999)

    def test_malformed_final_row_fails_without_changes(self):
        self.write([self.row(3999)])
        with self.context.sidecar.open('ab') as f:
            f.write(b'{"incomplete":')
        original = self.context.sidecar.read_bytes()
        with self.assertRaises(SidecarRepairError):
            self.apply()
        self.assertEqual(self.context.sidecar.read_bytes(), original)

    def test_wrong_run_identity_rejected(self):
        bad = self.row(3999)
        bad['run_id'] = 'other-run'
        original = self.write([bad])
        with self.assertRaises(RouteDiagnosticArtifactError):
            self.apply()
        self.assertEqual(self.context.sidecar.read_bytes(), original)

    def test_wrong_checkpoint_path_rejected_before_load(self):
        wrong = self.run / 'wrong.pt'
        wrong.write_bytes(b'wrong')
        with self.assertRaises(SidecarRepairError):
            load_resume_context(self.run, wrong)

    def test_wrong_checkpoint_step_filename_rejected(self):
        cp = self.context.checkpoint
        cfg = self.context.config
        payload = {'config_snapshot': cfg.snapshot_dict(), 'git_commit': cfg.git_commit,
                   'training_state': {'collected_environment_transitions': 8000,
                                      'optimized_transitions': 7936}}
        # Only the native serialization validator is mocked, never a training call.
        with patch('src.evaluation.route_sidecar_repair.load_checkpoint_payload', return_value=payload), \
                patch('src.evaluation.route_sidecar_repair.validate_periodic_checkpoint_compatibility'):
            with self.assertRaises(SidecarRepairError):
                load_resume_context(self.run, cp)

    def test_duplicate_in_safe_prefix_rejected(self):
        row = self.row(3999)
        original = self.write([row, copy.deepcopy(row), self.row(4001)])
        with self.assertRaisesRegex(SidecarRepairError, 'duplicate primary key'):
            self.apply()
        self.assertEqual(self.context.sidecar.read_bytes(), original)

    def test_stale_plan_or_changed_checkpoint_rejected(self):
        self.write([self.row(3999), self.row(4001)])
        plan = scan_route_sidecar(self.context)
        self.context.checkpoint.write_bytes(b'changed-checkpoint')
        original = self.context.sidecar.read_bytes()
        with self.assertRaises(SidecarRepairError):
            apply_route_repair(self.context, plan)
        self.assertEqual(self.context.sidecar.read_bytes(), original)

    def test_native_writer_collision_before_and_append_after_repair(self):
        tail_row = self.row(4001)
        self.write([self.row(3999), tail_row])
        writer = RouteDiagnosticArtifactWriter(self.context.config, self.header, allow_existing=True)
        with self.assertRaisesRegex(RouteDiagnosticArtifactError, 'duplicate route diagnostic primary key'):
            writer.write(tail_row)
        self.apply()
        writer = RouteDiagnosticArtifactWriter(self.context.config, self.header, allow_existing=True)
        writer.write(tail_row)
        self.assertEqual(writer.summary()['sample_count'], 2)

    def test_native_conditional_prefix_probe_is_readonly_and_linked(self):
        cfg = self.context.config
        cfg = replace(cfg, training=replace(cfg.training, mappo=replace(
            cfg.training.mappo, rollout_length_slots=8)))
        object.__setattr__(cfg, 'git_commit', 'a' * 40)
        cp = self.run / 'checkpoints' / 'step_8.pt'
        cp.write_bytes(b'synthetic-metadata-only')
        self.context = ResumeContext(self.run, cp, sha256(cp), 8, 8, cfg)
        self.header = route_diagnostic_schema_header(cfg)
        self.write([self.row(7)])
        trace = self.run / 'conditional_credit_trace_v1.jsonl'
        raw = bytearray()
        counts = Counter()
        identity = {k: self.header[k] for k in ('run_id', 'config_hash', 'git_commit')}

        def append(kind, **fields):
            raw.extend((json.dumps(dict(identity, event_type=kind, **fields)) + '\n').encode())
            counts[kind] += 1

        append('header')
        for step in range(8):
            for agent in range(cfg.environment.uav_count):
                append('agent_action', action_key=[step, agent])
        append('task_route', task_key=[0, 7], global_index=7, route_key=[0, 7, 7],
               route={'uav_id': 0, 'proposal': 'defer'})
        for epoch in range(cfg.training.mappo.update_epochs):
            append('ppo_epoch', update_index=0, epoch_index=epoch)
            for step in range(8):
                for agent in range(cfg.environment.uav_count):
                    append('ppo_agent_epoch', update_index=0, epoch_index=epoch,
                           action_key=[step, agent])
        append('update_end', update_index=0)
        append('checkpoint_marker', environment_steps=8, optimized_steps=8,
               checkpoint_filename='step_8.pt', event_counts_before_marker=dict(counts),
               bytes_before_marker=len(raw))
        append('agent_action', action_key=[8, 0])
        trace.write_bytes(raw)
        before = bytes(raw)
        result = audit_conditional_resume(self.context)
        self.assertTrue(result['native_resume_prefix_valid'])
        self.assertTrue(result['route_sample_linkage_exact'])
        self.assertEqual(result['linked_route_samples'], 1)
        self.assertGreater(result['post_checkpoint_bytes'], 0)
        self.assertEqual(trace.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
