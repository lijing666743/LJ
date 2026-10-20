"""Repair diagnostic logs offline; never execute training or evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.route_sidecar_repair import (
    apply_route_repair, audit_conditional_resume, load_resume_context, scan_route_sidecar,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='default: validate without writes')
    mode.add_argument('--apply', action='store_true', help='backup/quarantine then atomic rollback')
    args = parser.parse_args()
    context = load_resume_context(args.run_directory, args.checkpoint)
    plan = scan_route_sidecar(context)
    trace_audit = audit_conditional_resume(context)
    result = (apply_route_repair(context, plan, trace_audit=trace_audit)
              if args.apply else dict(plan, status='dry-run'))
    result['conditional_trace'] = trace_audit
    if args.apply:
        result['post_repair_conditional_trace'] = audit_conditional_resume(context)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
