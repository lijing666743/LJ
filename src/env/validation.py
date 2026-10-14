"""Real environment sanity and formal Gate 0 validation entry points."""

from __future__ import annotations

import io
import json
import math
import unittest
from dataclasses import dataclass
from pathlib import Path

from ..config import RunConfig
from .environment import U2UMECEnvironment


@dataclass(frozen=True)
class ValidationOutcome:
    """Artifact-free validation result consumed by the execution registry."""

    status: str
    message: str
    test_count: int = 0


def run_environment_sanity(config: RunConfig) -> ValidationOutcome:
    """Exercise deterministic reset and one real environment transition."""

    try:
        environment = U2UMECEnvironment(config)
        environment.reset()
        first_reset = json.dumps(
            environment.snapshot(),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        environment.reset()
        second_reset = json.dumps(
            environment.snapshot(),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        if first_reset != second_reset:
            raise RuntimeError("same config/seed produced different reset snapshots")

        result = environment.step(environment.canonical_proposals())
        if not math.isfinite(result.reward):
            raise RuntimeError("environment returned a non-finite reward")
        json.dumps(result.info, ensure_ascii=False, sort_keys=True, allow_nan=False)
        json.dumps(environment.snapshot(), ensure_ascii=False, sort_keys=True, allow_nan=False)
        conservation = environment.assert_invariants()
        if not conservation.is_conserved:
            raise RuntimeError("environment conservation invariant failed")
    except Exception as exc:  # validation boundary must report, not fabricate success
        return ValidationOutcome("failed", f"environment sanity failed: {exc}")
    return ValidationOutcome(
        "completed",
        "environment sanity passed: deterministic reset, canonical step, JSON audit, and invariants",
    )


def run_gate0_tests(config: RunConfig) -> ValidationOutcome:
    """Run the explicit G0-01..G0-21 unittest module without creating artifacts."""

    del config
    project_root = Path(__file__).resolve().parents[2]
    tests_dir = project_root / "tests"
    gate_file = tests_dir / "test_environment_gate0.py"
    if not gate_file.is_file():
        return ValidationOutcome("failed", "Full Gate 0 is unavailable: test_environment_gate0.py is missing")

    stream = io.StringIO()
    try:
        suite = unittest.defaultTestLoader.discover(
            str(tests_dir),
            pattern=gate_file.name,
            top_level_dir=str(tests_dir),
        )
        test_count = suite.countTestCases()
        if test_count < 21:
            return ValidationOutcome(
                "failed",
                f"Full Gate 0 requires at least 21 mapped tests; discovered {test_count}",
                test_count,
            )
        result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    except Exception as exc:  # discovery/import errors are genuine gate failures
        return ValidationOutcome("failed", f"Full Gate 0 could not run: {exc}")

    if not result.wasSuccessful():
        tail = " | ".join(stream.getvalue().strip().splitlines()[-6:])
        return ValidationOutcome(
            "failed",
            f"Full Gate 0 failed ({test_count} tests): {tail}",
            test_count,
        )
    return ValidationOutcome(
        "completed",
        f"Full Gate 0 passed ({test_count} explicit tests covering G0-01..G0-21)",
        test_count,
    )


__all__ = [
    "ValidationOutcome",
    "run_environment_sanity",
    "run_gate0_tests",
]
