"""Offline, evidence-preserving rollback of a Route diagnostic sidecar.

Never construct a Trainer or run a policy. Route indices are zero based,
captured BEFORE environment.step: a checkpoint with N collected transitions
commits precisely indices < N, irrespective of task arrival/terminal time.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import torch

from src.config import RunConfig, _construct_run_config
from src.models.ca_gat_mappo_checkpoint import (
    load_checkpoint_payload, validate_periodic_checkpoint_compatibility,
)
from src.models.ca_gat_mappo_credit_telemetry import ConditionalCreditTraceWriter
from src.models.ca_gat_mappo_route_telemetry import route_diagnostic_schema_header
from src.training_artifacts import (
    _validate_route_diagnostic_header, _validate_route_diagnostic_record,
)


class SidecarRepairError(ValueError):
    """Reject an inconsistent artifact without modifying it."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ResumeContext:
    run_directory: Path
    checkpoint: Path
    checkpoint_sha256: str
    step: int
    optimized_steps: int
    config: RunConfig

    @property
    def sidecar(self) -> Path:
        return self.run_directory / 'route_diagnostic_samples_v1.jsonl'


def load_resume_context(run_directory: Path, checkpoint: Path) -> ResumeContext:
    run = run_directory.resolve(strict=True)
    cp = checkpoint.resolve(strict=True)
    if cp.parent != run / 'checkpoints' or not run.is_dir():
        raise SidecarRepairError('checkpoint must belong to the explicit run/checkpoints')
    before = sha256(cp)
    payload = load_checkpoint_payload(cp)
    snapshot = payload['config_snapshot']
    config = _construct_run_config({k: v for k, v in snapshot.items() if k != '_metadata'})
    object.__setattr__(config, 'git_commit', payload['git_commit'])
    config.validate()
    # This is metadata/schema validation only; no model or optimizer is instantiated.
    validate_periodic_checkpoint_compatibility(
        config, payload, dtype=torch.float32, cuda_available=torch.cuda.is_available())
    metadata = snapshot['_metadata']
    if (metadata['run_id'] != config.run_id or metadata['config_hash'] != config.config_hash
            or metadata['git_commit'] != config.git_commit
            or run != Path(config.artifact_paths()['config_snapshot']).parent.resolve()):
        raise SidecarRepairError('checkpoint snapshot/run identity differs')
    step = payload['training_state']['collected_environment_transitions']
    if cp.name != f'step_{step}.pt' or (run / 'checkpoints' / 'final.pt').exists():
        raise SidecarRepairError('checkpoint filename/step differs or final already exists')
    if sha256(cp) != before:
        raise SidecarRepairError('checkpoint changed during preflight')
    return ResumeContext(run, cp, before, step,
                         payload['training_state']['optimized_transitions'], config)


def _integer(row: dict, name: str) -> int:
    value = row[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SidecarRepairError(f'{name} must be a nonnegative integer')
    return value


def scan_route_sidecar(context: ResumeContext) -> dict[str, Any]:
    """Read every byte, reject corruption/duplicates, identify a contiguous prefix."""
    path = context.sidecar
    if not path.is_file() or path.is_symlink():
        raise SidecarRepairError('regular non-symlink Route sidecar required')
    full, prefix, tail = (hashlib.sha256() for _ in range(3))
    keys: set[tuple] = set()
    safe = orphan = offset = prefix_bytes = 0
    safe_first = safe_last = orphan_first = orphan_last = None
    max_step = safe_max = orphan_min = None
    previous = (-1, -1)
    tail_started = False
    with path.open('rb') as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.endswith(b'\n') or not raw.strip():
                raise SidecarRepairError(f'incomplete JSONL framing at line {line_number}')
            try:
                row = json.loads(raw.decode('utf-8'))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise SidecarRepairError(f'malformed JSON at line {line_number}') from exc
            full.update(raw)
            if line_number == 1:
                header = row
                _validate_route_diagnostic_header(context.config, header)
                if header != route_diagnostic_schema_header(context.config):
                    raise SidecarRepairError('Route header differs from frozen schema contract')
                prefix.update(raw)
                prefix_bytes += len(raw)
            else:
                key = _validate_route_diagnostic_record(context.config, header, row)
                step = _integer(row, 'global_environment_step')
                agent = _integer(row, 'source_uav')
                episode = _integer(row, 'episode_id')
                slot = _integer(row, 'route_slot')
                version = _integer(row, 'policy_version')
                _integer(row, 'head_task_id')
                if key in keys:
                    raise SidecarRepairError(f'duplicate primary key at line {line_number}: {key}')
                keys.add(key)
                if (step, agent) <= previous or agent >= context.config.environment.uav_count:
                    raise SidecarRepairError('Route step/agent order is invalid')
                horizon = context.config.environment.episode_horizon
                if episode != step // horizon or slot != step % horizon:
                    raise SidecarRepairError('episode/slot and global index differ')
                if version != step // context.config.training.mappo.rollout_length_slots:
                    raise SidecarRepairError('Route policy version and rollout index differ')
                previous = (step, agent)
                max_step = step
                if step < context.step:
                    if tail_started:
                        raise SidecarRepairError('safe record follows orphan tail')
                    safe += 1
                    safe_first = safe_first or list(key)
                    safe_last = list(key)
                    safe_max = step
                    prefix.update(raw)
                    prefix_bytes += len(raw)
                else:
                    tail_started = True
                    orphan += 1
                    orphan_first = orphan_first or list(key)
                    orphan_last = list(key)
                    orphan_min = step if orphan_min is None else orphan_min
                    tail.update(raw)
            offset += len(raw)
    if offset == 0:
        raise SidecarRepairError('empty Route sidecar')
    if sha256(path) != full.hexdigest():
        raise SidecarRepairError('sidecar changed during scan')
    return {
        'source_run': context.config.run_id, 'checkpoint_path': str(context.checkpoint),
        'checkpoint_sha256': context.checkpoint_sha256, 'checkpoint_step': context.step,
        'sidecar_path': str(path), 'sidecar_identity': {
            name: header[name] for name in ('run_id', 'config_hash', 'git_commit', 'schema_version')},
        'sidecar_sha256_before': full.hexdigest(), 'sidecar_sha256_after': prefix.hexdigest(),
        'original_bytes': offset, 'prefix_bytes': prefix_bytes,
        'orphan_bytes': offset - prefix_bytes, 'total_rows': safe + orphan,
        'unique_primary_keys': len(keys), 'max_step': max_step,
        'safe_rows': safe, 'safe_max_step': safe_max,
        'orphan_rows': orphan, 'orphan_min_step': orphan_min,
        'orphan_max_step': max_step if orphan else None,
        'safe_first_key': safe_first, 'safe_last_key': safe_last,
        'orphan_first_key': orphan_first, 'orphan_last_key': orphan_last,
        'orphan_sha256': tail.hexdigest(), 'conflicts': [],
        'boundary_rule': 'zero-based pre-step global_environment_step < checkpoint transitions',
        'planned_action': 'backup_full_original_and_quarantine_tail_then_atomic_replace' if orphan else 'no-op',
    }


class _TracePathOverride:
    """Use the native log reopener on a scratch copy with unchanged provenance."""
    def __init__(self, config: RunConfig, path: Path):
        self.config, self.path = config, path

    def __getattr__(self, name: str):
        return getattr(self.config, name)

    def artifact_paths(self):
        paths = self.config.artifact_paths()
        paths['conditional_credit_trace'] = str(self.path)
        return paths


def audit_conditional_resume(context: ResumeContext) -> dict[str, Any]:
    """Validate native resume of the trace on a disposable copy, without training."""
    path = context.run_directory / 'conditional_credit_trace_v1.jsonl'
    before = sha256(path)
    with tempfile.TemporaryDirectory(prefix='gate-m-r-trace-') as directory:
        scratch = Path(directory) / path.name
        shutil.copyfile(path, scratch)
        if sha256(scratch) != before:
            raise SidecarRepairError('trace changed during copying')
        writer = ConditionalCreditTraceWriter.reopen_at_checkpoint(
            _TracePathOverride(context.config, scratch),
            environment_steps=context.step, optimized_steps=context.optimized_steps,
            filename=context.checkpoint.name)
        writer.pause()
        trace_routes = {}
        with scratch.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['event_type'] == 'task_route':
                    key = (row['task_key'][0], row['global_index'],
                           row['route']['uav_id'], row['task_key'][1])
                    trace_routes[key] = row['route']['proposal']
        sidecar_routes = {}
        with context.sidecar.open(encoding='utf-8') as stream:
            next(stream)
            for line in stream:
                row = json.loads(line)
                if row['global_environment_step'] < context.step:
                    key = tuple(row['primary_key'][1:])
                    sidecar_routes[key] = row['policy']['actual_route']
        if sidecar_routes != trace_routes:
            raise SidecarRepairError('safe Route samples differ from committed conditional task_route events')
        result = {'native_resume_prefix_valid': True,
                  'trace_sha256': before, 'bytes_on_disk': path.stat().st_size,
                  'checkpoint_prefix_bytes': scratch.stat().st_size,
                  'post_checkpoint_bytes': path.stat().st_size - scratch.stat().st_size,
                  'prefix_event_counts': dict(writer._counts),
                  'route_sample_linkage_exact': True, 'linked_route_samples': len(trace_routes),
                  'original_trace_modified': False}
    if sha256(path) != before:
        raise SidecarRepairError('live trace changed during audit')
    return result


def tool_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), root / 'tools' / 'repair_route_sidecar_for_resume.py']
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                            capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(['git', 'status', '--porcelain', '--', *map(str, files)],
                            cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    return {'repair_tool_commit': commit, 'repair_tool_has_uncommitted_changes': bool(status),
            'repair_tool_sha256': {str(p.relative_to(root)): sha256(p) for p in files}}


def _write_synced(path: Path, data: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def apply_route_repair(context: ResumeContext, expected_plan: dict,
                       provenance: dict | None = None,
                       trace_audit: dict | None = None) -> dict:
    """Preserve all evidence before atomically publishing the byte-exact prefix.

    Caller must ensure training is stopped. A lock excludes concurrent repair
    tools; repeated hashes reject changes by other writers before publication.
    Interrupted prepared repairs fail closed and retain the backup/manifest.
    """
    lock = context.run_directory / '.route_sidecar_repair.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    temporary: Path | None = None
    try:
        plan = scan_route_sidecar(context)
        if plan != expected_plan or sha256(context.checkpoint) != context.checkpoint_sha256:
            raise SidecarRepairError('artifact identity/content changed after planning')
        if not plan['orphan_rows']:
            return dict(plan, status='no-op')
        evidence = (context.run_directory / 'sidecar_resume_repairs' /
                    f"step_{context.step}__before-{plan['sidecar_sha256_before'][:16]}")
        evidence.mkdir(parents=True, exist_ok=False)
        backup = evidence / 'route_sidecar.original.jsonl'
        shutil.copyfile(context.sidecar, backup)
        with backup.open('rb') as source, backup.open('ab') as sync:
            os.fsync(sync.fileno())
            source.seek(plan['prefix_bytes'])
            orphan_data = source.read()
        orphan_path = evidence / f'route_sidecar.orphaned_after_step_{context.step}.jsonl'
        _write_synced(orphan_path, orphan_data)
        _write_synced(evidence / 'route_sidecar.original.sha256',
                      (plan['sidecar_sha256_before'] + '\n').encode('ascii'))
        if (sha256(backup) != plan['sidecar_sha256_before']
                or sha256(orphan_path) != plan['orphan_sha256']):
            raise SidecarRepairError('backup/quarantine SHA verification failed')
        if trace_audit is not None:
            trace_path = context.run_directory / 'conditional_credit_trace_v1.jsonl'
            trace_backup = evidence / 'conditional_credit_trace.before_manual_resume.jsonl'
            shutil.copyfile(trace_path, trace_backup)
            with trace_backup.open('ab') as stream:
                stream.flush()
                os.fsync(stream.fileno())
            if (sha256(trace_backup) != trace_audit['trace_sha256']
                    or sha256(trace_path) != trace_audit['trace_sha256']):
                raise SidecarRepairError('conditional trace changed before repair')
        handle, name = tempfile.mkstemp(prefix='.route-sidecar-prefix-', dir=context.run_directory)
        temporary = Path(name)
        with os.fdopen(handle, 'wb') as target, backup.open('rb') as source:
            remaining = plan['prefix_bytes']
            while remaining:
                chunk = source.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise SidecarRepairError('backup prefix is incomplete')
                target.write(chunk)
                remaining -= len(chunk)
            target.flush()
            os.fsync(target.fileno())
        if sha256(temporary) != plan['sidecar_sha256_after']:
            raise SidecarRepairError('staged prefix SHA differs')
        manifest = dict(plan, **(provenance or tool_identity()), status='prepared',
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        quarantine_directory=str(evidence),
                        quarantined_row_count=plan['orphan_rows'])
        if trace_audit is not None:
            manifest['conditional_trace_resume_audit'] = trace_audit
            manifest['conditional_trace_backup'] = str(trace_backup)
        _write_synced(evidence / 'repair_manifest.prepared.json',
                      (json.dumps(manifest, indent=2) + '\n').encode('utf-8'))
        if (sha256(context.sidecar) != plan['sidecar_sha256_before']
                or sha256(context.checkpoint) != context.checkpoint_sha256):
            raise SidecarRepairError('live artifacts changed before atomic publication')
        os.replace(temporary, context.sidecar)
        temporary = None
        if sha256(context.sidecar) != plan['sidecar_sha256_after']:
            raise SidecarRepairError('published prefix SHA differs; original backup retained')
        after = scan_route_sidecar(context)
        if after['orphan_rows'] or after['safe_rows'] != plan['safe_rows']:
            raise SidecarRepairError('post-repair scan differs; original backup retained')
        manifest['status'] = 'applied'
        _write_synced(evidence / 'repair_manifest.json',
                      (json.dumps(manifest, indent=2) + '\n').encode('utf-8'))
        return manifest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink()
