"""Mutation gate for the policy/certificate/ledger layers.

The kernel gate (``tests/mutation_gate.py``) compares against a dense oracle
in-process. These layers are statistical, so the gate takes the other route: it
runs the **real test files** with a mutant plugin loaded (``-p tests.mutants.X``)
and requires the suite to fail. A mutant that survives means the corresponding
test is decoration.

Control: the same files with no plugin must pass. Both directions are checked for
every mutant, so a permanently-red suite cannot fake a pass.

Run:  python tests/mutation_gate_evict.py
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

# (name, mutant plugin, test file, what the mutant breaks)
MUTANTS = [
    ("NO_HAJEK_OFFSET", "tests.mutants.no_hajek_offset", "tests/test_certificate.py",
     "estimator ignores the design probabilities -> biased subsample mean"),
    ("MISSING_VARIANCE", "tests.mutants.missing_variance", "tests/test_certificate.py",
     "radius keeps only the range term -> coverage below 1-delta"),
    ("NO_TOP_SLICE", "tests.mutants.no_top_slice", "tests/test_certificate.py",
     "certainty set loses its score top slice -> variance explodes"),
    ("NO_BLOCK_TABLE", "tests.mutants.no_block_table", "tests/test_budget.py",
     "ragged storage forgets block-table overhead"),
    ("ALWAYS_ENFORCED", "tests.mutants.always_enforced", "tests/test_budget.py",
     "ledger stamps 'enforced' regardless of the arithmetic"),
]


def _run(test_file: str, plugin: str | None) -> subprocess.CompletedProcess:
    cmd = [PYTHON, "-m", "pytest", test_file, "-q", "-x", "--no-header", "-p", "no:cacheprovider"]
    if plugin:
        cmd += ["-p", plugin]
    # inherit the caller's environment (HOME, LD_LIBRARY_PATH, ... matter for the
    # torch install) and only prepend the repo to PYTHONPATH
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{ROOT}" + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)


def _failed_names(stdout: str) -> list[str]:
    return [line.split("::")[-1].split()[0]
            for line in stdout.splitlines() if line.startswith("FAILED")]


def main() -> int:
    print(f"mutation gate (policy layers) — {len(MUTANTS)} mutants\n")
    failures = 0

    control = _run("tests/test_certificate.py", None)
    control2 = _run("tests/test_budget.py", None)
    ok_control = control.returncode == 0 and control2.returncode == 0
    print(f"CONTROL  clean tree: exit={control.returncode}/{control2.returncode} "
          f"{'PASS' if ok_control else 'FAIL (suite is not green to begin with)'}")
    if not ok_control:
        print(control.stdout[-2500:])
        return 1

    for name, plugin, test_file, what in MUTANTS:
        proc = _run(test_file, plugin)
        caught = proc.returncode != 0
        names = _failed_names(proc.stdout) or _failed_names(proc.stderr)
        verdict = "CAUGHT" if caught else "SURVIVED"
        print(f"{name:16s} -> {verdict:8s} by {names if names else ['<no FAILED line>']}")
        print(f"{'':16s}    breaks: {what}")
        if not caught:
            failures += 1
            print(proc.stdout[-1500:])

    print()
    if failures:
        print(f"GATE FAILED: {failures} mutant(s) survived — the matching test is decoration")
        return 1
    print(f"GATE PASSED: all {len(MUTANTS)} mutants caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
