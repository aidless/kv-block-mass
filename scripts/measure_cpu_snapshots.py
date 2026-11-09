"""Write bench/cpu_*.json from a real CPU run.

Does not read or write the archived GPU JSON. Refuses to invent wall-clock
numbers: early-exit rows store wall_clock = UNKNOWN.

Run from the repo root:
    python scripts/measure_cpu_snapshots.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from tests.cpu_observe import (  # noqa: E402
    early_exit_rows,
    ledger_snapshot,
    mutation_rows,
    oracle_rows,
    policy_rows,
)

BENCH = ROOT / "bench"


def _meta(kind: str) -> dict:
    return {
        "kind": kind,
        "produced_by": "scripts/measure_cpu_snapshots.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": bool(torch.cuda.is_available()),
        "note": (
            "CPU snapshot from the machine that ran this script. "
            "Not a GPU bench. Do not hand-edit the numbers."
        ),
    }


def _write(name: str, payload: dict) -> None:
    path = BENCH / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)}", flush=True)


def main() -> int:
    print("ledger", flush=True)
    ledger = ledger_snapshot()
    _write("cpu_ledger.json", {"meta": _meta("cpu-ledger"), **ledger})

    print("oracle", flush=True)
    oracle = oracle_rows()
    _write("cpu_oracle.json", {"meta": _meta("cpu-oracle"), "rows": oracle})

    print("early-exit exactness (no timings)", flush=True)
    early = early_exit_rows()
    _write("cpu_early_exit.json", {"meta": _meta("cpu-early-exit"), "rows": early})

    print("mutation gate", flush=True)
    rows, caught = mutation_rows()
    if not caught:
        print("mutation gate did not catch every bug; not writing a green snapshot")
        return 1
    _write("cpu_mutation_gate.json", {"meta": _meta("cpu-mutation-gate"), "rows": rows})

    print("policy (this is the slow part)", flush=True)
    policy = policy_rows()
    _write("cpu_policy.json", {"meta": _meta("cpu-policy"), "rows": policy})
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
