"""Engineering regressions using disposable four-slot evaluation fixtures."""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from src.artifacts import (
    ArtifactConflictError,
    ArtifactWriteError,
    atomic_write_bytes_group,
    atomic_write_text_group,
    prepare_artifact_parents,
    _filesystem_path,
)
from src.config import CHECKPOINT_KIND_PERIODIC_RESUME
from src.evaluation.runner import FormalEvaluationRunner
from tests.test_formal_evaluation import make_evaluation_fixture


def io_path(path: Path) -> Path:
    # Independently spell the native Windows path for regression assertions.
    return Path("\\\\?\\" + str(path.resolve())) if os.name == "nt" else path


@contextmanager
def long_path_test_directory():
    # shutil's ordinary-path cleanup also hits MAX_PATH; keep fixture cleanup native.
    with tempfile.TemporaryDirectory(dir=io_path(Path(tempfile.gettempdir()))) as directory:
        yield Path(directory.removeprefix("\\\\?\\"))


class TestNestedArtifactPaths(unittest.TestCase):
    def test_existing_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.bin"
            self.assertEqual(atomic_write_bytes_group(((target, b"exact\x00bytes"),)), (target,))
            self.assertEqual(target.read_bytes(), b"exact\x00bytes")
            self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_missing_multiple_parents(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "new" / "deeper" / "result.bin"
            self.assertFalse(target.parent.exists())
            atomic_write_bytes_group(((target, b"nested"),))
            self.assertEqual(target.read_bytes(), b"nested")

    def test_different_nested_parents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = (root / "a" / "b" / "first", root / "c" / "d" / "second")
            self.assertEqual(atomic_write_bytes_group(zip(targets, (b"a", b"b"))), targets)
            self.assertEqual([p.read_bytes() for p in targets], [b"a", b"b"])

    def test_existing_target_blocks_all_parent_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing, new = root / "old", root / "absent" / "new"
            existing.write_bytes(b"preserve")
            before = (existing.read_bytes(), existing.stat().st_mtime_ns)
            with self.assertRaises(ArtifactConflictError):
                atomic_write_bytes_group(((new, b"new"), (existing, b"overwrite")))
            self.assertFalse(new.parent.exists())
            self.assertEqual((existing.read_bytes(), existing.stat().st_mtime_ns), before)

    def test_second_staging_failure_cleans_every_temp_and_preserves_existing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root / "existing"
            sentinel.write_bytes(b"preserve")
            before = (sentinel.read_bytes(), sentinel.stat().st_mtime_ns)
            targets = (root / "a" / "one", root / "b" / "two")
            with patch("src.artifacts.os.fsync", side_effect=(None, OSError("second staging"))):
                with self.assertRaisesRegex(ArtifactWriteError, "second staging"):
                    atomic_write_bytes_group(zip(targets, (b"one", b"two")))
            self.assertTrue(all(not p.exists() for p in targets))
            self.assertEqual(list(root.rglob("*.tmp")), [])
            self.assertEqual((sentinel.read_bytes(), sentinel.stat().st_mtime_ns), before)

    def test_text_group_nested_parents(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "new" / "deep" / "text.json"
            atomic_write_text_group(((target, "中文\n"),))
            self.assertEqual(target.read_bytes(), "中文\n".encode("utf-8"))

    def test_parent_preparation_failure_precedes_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blocked = root / "file"
            blocked.write_bytes(b"parent is a file")
            first = root / "first" / "result"
            with self.assertRaises(ArtifactWriteError):
                atomic_write_bytes_group(((first, b"a"), (blocked / "result", b"b")))
            self.assertFalse(first.exists())
            self.assertEqual(list(root.rglob("*.tmp")), [])

    def test_writable_probe_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "result"
            prepare_artifact_parents((target,), check_writable=True)
            self.assertTrue(target.parent.is_dir())
            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.iterdir()), [])

    def test_target_created_during_staging_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first", root / "second"

            def create_conflict(_descriptor):
                second.write_bytes(b"created by another writer")

            with patch("src.artifacts.os.fsync", side_effect=create_conflict):
                with self.assertRaises(ArtifactConflictError):
                    atomic_write_bytes_group(((first, b"one"), (second, b"two")))
            self.assertFalse(first.exists())
            self.assertEqual(second.read_bytes(), b"created by another writer")
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_late_publication_failure_keeps_original_per_file_atomic_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first", root / "second"
            original_replace = os.replace

            def fail_second(source, target):
                if Path(target).name == "second":
                    raise OSError("second publication")
                return original_replace(source, target)

            with patch("src.artifacts.os.replace", side_effect=fail_second):
                with self.assertRaisesRegex(ArtifactWriteError, "second publication"):
                    atomic_write_bytes_group(((first, b"one"), (second, b"two")))
            self.assertEqual(first.read_bytes(), b"one")
            self.assertFalse(second.exists())
            self.assertEqual(list(root.glob("*.tmp")), [])

    @unittest.skipUnless(os.name == "nt", "Windows extended-path regression")
    def test_windows_long_parent_final_name_and_conflict(self):
        with long_path_test_directory() as directory:
            parent = Path(directory) / ("a" * 100) / ("b" * 100)
            target = parent / ("c" * 240)
            self.assertGreater(len(str(target)), 260)
            # Final component is valid, but the old name-derived temp exceeds 255.
            self.assertEqual(atomic_write_bytes_group(((target, b"long path"),)), (target,))
            self.assertEqual(io_path(target).read_bytes(), b"long path")
            self.assertEqual([p.name for p in io_path(parent).iterdir()], [target.name])
            before = io_path(target).stat().st_mtime_ns
            with self.assertRaises(ArtifactConflictError):
                atomic_write_bytes_group(((target, b"overwrite"),))
            self.assertEqual(io_path(target).stat().st_mtime_ns, before)

    @unittest.skipUnless(os.name == "nt", "Windows extended-path spelling")
    def test_windows_unc_and_already_extended_paths(self):
        unc = Path(r"\\server\share\directory\result.json")
        extended = Path(r"\\?\UNC\server\share\directory\result.json")
        self.assertEqual(_filesystem_path(unc), extended)
        self.assertEqual(_filesystem_path(extended), extended)
        local = Path(r"\\?\D:\directory\result.json")
        self.assertEqual(_filesystem_path(local), local)


class TestFreshEvaluationArtifactPaths(unittest.TestCase):
    def test_fresh_long_logs_route_service_and_repeat_conflict(self):
        with long_path_test_directory() as directory:
            root = Path(directory)
            _source, config, checkpoint, _actor = make_evaluation_fixture(
                root, evaluation_seeds=(2042,), kind=CHECKPOINT_KIND_PERIODIC_RESUME,
                multi_uav=True,
            )
            fresh = root / ("fresh-evaluation-" + "x" * 90)
            relocated = replace(config, output=replace(config.output, logs_dir=str(fresh)))
            for field in ("git_branch", "git_commit", "git_dirty"):
                object.__setattr__(relocated, field, getattr(config, field))
            config = relocated
            runner = FormalEvaluationRunner(
                config, checkpoint, evaluation_device="cpu",
                collect_service_telemetry=True, development_checkpoint=True,
            )
            before_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            before_config_hash = config.config_hash
            before_identity = runner.evaluation_run_id
            self.assertFalse(fresh.exists())
            no_write = runner.run(write_artifacts=False)
            self.assertFalse(fresh.exists())
            result = runner.run()
            self.assertEqual(result.evaluation_run_id, before_identity)
            self.assertEqual(config.config_hash, before_config_hash)
            self.assertEqual(result.manifest, no_write.manifest)
            self.assertEqual(result.aggregate_metrics, no_write.aggregate_metrics)
            self.assertEqual(result.route_decision_trace, no_write.route_decision_trace)
            self.assertEqual(result.task_lifecycle_trace, no_write.task_lifecycle_trace)
            self.assertEqual(result.cpu_service_trace, no_write.cpu_service_trace)
            self.assertEqual(result.tx_service_trace, no_write.tx_service_trace)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before_hash)
            self.assertEqual(len(result.artifacts), 4)
            self.assertEqual(len(result.service_diagnostic_artifacts), 4)
            self.assertEqual({Path(p).name for p in result.diagnostic_artifacts},
                             {"summary.json", "route_decision_trace.jsonl"})
            targets = [Path(p) for p in (*result.artifacts, *result.diagnostic_artifacts,
                                        *result.service_diagnostic_artifacts)]
            states = [(io_path(p).read_bytes(), io_path(p).stat().st_mtime_ns) for p in targets]
            self.assertEqual(list(io_path(fresh).rglob("*.tmp")), [])
            with patch.object(runner, "environment_factory", Mock()) as factory:
                with self.assertRaises(ArtifactConflictError):
                    runner.run()
                factory.assert_not_called()
            self.assertEqual([(io_path(p).read_bytes(), io_path(p).stat().st_mtime_ns)
                              for p in targets], states)

    def test_route_parent_probe_failure_precedes_rollout_and_main_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source, config, checkpoint, _actor = make_evaluation_fixture(root)
            runner = FormalEvaluationRunner(config, checkpoint, evaluation_device="cpu",
                                            collect_service_telemetry=True)
            original_temp = tempfile.NamedTemporaryFile

            def fail_route_probe(*args, **kwargs):
                if "hierarchical-route-eval__" in str(kwargs.get("dir")):
                    raise PermissionError("route parent not writable")
                return original_temp(*args, **kwargs)

            with patch("src.artifacts.tempfile.NamedTemporaryFile", side_effect=fail_route_probe), \
                    patch.object(runner, "environment_factory", Mock()) as factory:
                with self.assertRaisesRegex(ArtifactWriteError, "route parent not writable"):
                    runner.run()
                factory.assert_not_called()
            targets = [*runner.artifact_paths().values(),
                       *runner.diagnostic_artifact_paths().values(),
                       *runner.service_diagnostic_artifact_paths().values()]
            self.assertTrue(all(not io_path(p).exists() for p in targets))
            self.assertEqual(list(root.rglob("*.tmp")), [])
