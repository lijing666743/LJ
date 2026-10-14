"""Opt-in service diagnostics using the unchanged formal evaluation loop.

The existing four-method suite is retained. No training or parameter updates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_run_config
from .runner import FormalEvaluationRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluate-from", required=True)
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--evaluation-device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--logs-dir", required=True, help="new isolated output root")
    parser.add_argument("--dry-run", action="store_true", help="validate/load only; no rollout or files")
    parser.add_argument("--development-checkpoint", action="store_true",
                        help="actor-only periodic/final development diagnostics; never paper formal results")
    args = parser.parse_args(argv)
    config = load_run_config(args.config, cli_overrides={
        "mode": "evaluation", "method_id": "ca_gat_mappo", "launch_profile": None,
        "evaluation.evaluation_seeds": [args.evaluation_seed],
        "output.logs_dir": str(Path(args.logs_dir)),
    })
    runner = FormalEvaluationRunner(
        config, args.evaluate_from, evaluation_device=args.evaluation_device,
        collect_service_telemetry=True,
        development_checkpoint=args.development_checkpoint,
    )
    if args.dry_run:
        print(json.dumps({
            "rollout_started": False,
            "training_seed": config.seed,
            "evaluation_seeds": list(config.evaluation.evaluation_seeds),
            "checkpoint_sha256": runner.loaded_actor.source_checkpoint_sha256,
            "service_diagnostic_paths": {
                key: str(value) for key, value in runner.service_diagnostic_artifact_paths().items()
            },
        }, ensure_ascii=True, indent=2))
        return 0
    result = runner.run()
    print(json.dumps({
        "evaluation_run_id": result.evaluation_run_id,
        "service_diagnostic_artifacts": result.service_diagnostic_artifacts,
    }, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
