from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from src.config import CHECKPOINT_KIND_FINAL_COMPLETED, load_run_config
from src.evaluation.actor_loader import (
    EvaluationCheckpointError,
    checkpoint_sha256,
)
from src.evaluation.route_oracle_probe import (
    FormalRouteOracleProbeRunner,
    RouteOracleProbeRunner,
)
from src.models.ca_gat_mappo_checkpoint import atomic_save_checkpoint
from tests.test_ca_gat_mappo_checkpoint import make_payload


CONFIG_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "tests" / "fixtures" / "configs"
)
BASELINE_CONFIG = CONFIG_DIRECTORY / (
    "cooperative-load-v2-hierarchical-optionaware-v1-"
    "remote-stress-v2-32k.json"
)
ENTROPY_CONFIG = CONFIG_DIRECTORY / (
    "cooperative-load-v2-hierarchical-optionaware-v1-"
    "remote-stress-v2-route-entropy-v1-32k.json"
)


def _make_final_checkpoint(root: Path, config_path: Path):
    source = load_run_config(config_path)
    source = replace(
        source,
        training=replace(
            source.training,
            mappo=replace(source.training.mappo, training_device="cpu"),
        ),
    )
    object.__setattr__(source, "git_branch", "checkpoint-source")
    object.__setattr__(source, "git_commit", "a" * 40)
    object.__setattr__(source, "git_dirty", False)
    source.validate()
    payload, _actor, _critic, _updater = make_payload(
        source, CHECKPOINT_KIND_FINAL_COMPLETED
    )
    checkpoint = root / f"{config_path.stem}.pt"
    atomic_save_checkpoint(payload, checkpoint)

    evaluation = replace(
        source,
        mode="evaluation",
        method_id="ca_gat_mappo",
        launch_profile=None,
    )
    object.__setattr__(evaluation, "git_branch", "checkpoint-evaluator")
    object.__setattr__(evaluation, "git_commit", "b" * 40)
    object.__setattr__(evaluation, "git_dirty", False)
    evaluation.validate()
    return evaluation, checkpoint


class FormalRouteOracleCheckpointIdentityTests(unittest.TestCase):
    def _assert_smoke_and_formal_accept(
        self, root: Path, config_path: Path
    ) -> tuple[str, str]:
        evaluation, checkpoint = _make_final_checkpoint(root, config_path)
        expected_sha = checkpoint_sha256(checkpoint)
        smoke = RouteOracleProbeRunner(
            evaluation,
            checkpoint,
            evaluation_device="cpu",
            state_count=1,
        )
        formal = FormalRouteOracleProbeRunner(
            evaluation,
            checkpoint,
            evaluation_device="cpu",
            state_count=100,
        )
        for runner in (smoke, formal):
            identity = runner.identity
            self.assertEqual(identity["checkpoint_sha256"], expected_sha)
            self.assertEqual(identity["checkpoint_kind"], "FINAL_COMPLETED")
            self.assertEqual(identity["resolved_config_hash"], runner.config.config_hash)
            self.assertEqual(identity["evaluator_git_commit"], "b" * 40)
            self.assertEqual(identity["checkpoint_path"], str(checkpoint.resolve()))
            self.assertEqual(len(identity["source_training_config_hash"]), 64)
        return smoke.identity["checkpoint_sha256"], formal.identity["actor_digest"]

    def test_baseline_final_checkpoint_passes_both_identity_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            smoke_sha, actor_digest = self._assert_smoke_and_formal_accept(
                Path(directory), BASELINE_CONFIG
            )
        self.assertEqual(len(smoke_sha), 64)
        self.assertEqual(len(actor_digest), 64)

    def test_entropy_final_checkpoint_passes_both_identity_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            smoke_sha, actor_digest = self._assert_smoke_and_formal_accept(
                Path(directory), ENTROPY_CONFIG
            )
        self.assertEqual(len(smoke_sha), 64)
        self.assertEqual(len(actor_digest), 64)

    def test_checkpoint_without_v1_identity_fails_smoke_and_formal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation, _checkpoint = _make_final_checkpoint(root, BASELINE_CONFIG)
            invalid = root / "missing-identity.pt"
            torch.save({"schema_version": 1}, invalid)
            for runner_type, state_count in (
                (RouteOracleProbeRunner, 1),
                (FormalRouteOracleProbeRunner, 100),
            ):
                with self.subTest(runner=runner_type.__name__):
                    with self.assertRaises(EvaluationCheckpointError):
                        runner_type(
                            evaluation,
                            invalid,
                            evaluation_device="cpu",
                            state_count=state_count,
                        )


if __name__ == "__main__":
    unittest.main()
