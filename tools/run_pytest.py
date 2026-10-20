"""Run tests with a short, isolated external temporary root on Windows."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=Path(project_root.anchor) / "My_project_li_test_tmp",
    )
    options, pytest_args = parser.parse_known_args()
    temp_root = options.temp_root.resolve()
    if temp_root == project_root or temp_root.is_relative_to(project_root):
        parser.error("--temp-root must be outside the project")
    temp_root.mkdir(parents=True, exist_ok=True)

    # Each invocation owns a new directory; tests still create their own
    # TemporaryDirectory instances and exercise the original artifact writer.
    with tempfile.TemporaryDirectory(prefix="t-", dir=temp_root) as directory:
        run_root = Path(directory)
        child_env = os.environ.copy()
        for name in ("TEMP", "TMP", "TMPDIR"):
            child_env[name] = str(run_root)
        child_env.update(
            GIT_OPTIONAL_LOCKS="0",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONUTF8="1",
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            MPLCONFIGDIR=str(run_root / "matplotlib"),
        )
        child_env.setdefault("OMP_NUM_THREADS", "1")
        child_env.setdefault("MKL_NUM_THREADS", "1")
        print(f"Test-only TEMP/TMP/TMPDIR: {run_root}", flush=True)
        probe = subprocess.run(
            [
                sys.executable, "-B", "-X", "utf8", "-c",
                "import pathlib,tempfile,sys; "
                "actual=pathlib.Path(tempfile.gettempdir()).resolve(); "
                "assert actual==pathlib.Path(sys.argv[1]).resolve(); "
                "print('tempfile.gettempdir():', actual)",
                str(run_root),
            ],
            cwd=project_root,
            env=child_env,
            check=False,
        )
        if probe.returncode:
            return probe.returncode
        return subprocess.run(
            [
                sys.executable, "-u", "-B", "-X", "utf8", "-m", "pytest",
                "-p", "no:cacheprovider", "--basetemp", str(run_root / "pytest"),
                *(pytest_args or ["tests", "-q", "--maxfail=1"]),
            ],
            cwd=project_root,
            env=child_env,
            check=False,
        ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
