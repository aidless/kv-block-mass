"""Kaggle T4 entry point: run the bench inside a notebook cell or script.

On Kaggle, enable GPU (T4 x2), turn Internet OFF, and either:
  1. attach this repo as a utility script / dataset, or
  2. paste the repo zip and unzip.

Then run:  python kaggle/run_bench_kernel.py
Outputs land in /kaggle/working (auto-saved with the notebook).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# make src/ and tests/ importable without installing
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def main():
    import torch

    print("device:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__, "| hip:", torch.version.hip)
    assert torch.cuda.is_available(), "enable a GPU accelerator in Kaggle settings"

    cmd = [sys.executable, str(ROOT / "bench" / "bench_block_mass.py"),
           "--out", "/kaggle/working/bench_t4.json"]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
