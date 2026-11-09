"""The README's numeric tables have to match committed JSON."""

import subprocess
import sys
from pathlib import Path


def test_readme_drift_check_passes():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(root / "scripts" / "check_readme_drift.py")],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
