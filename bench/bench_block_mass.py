"""Benchmark: reference vs Triton kernel for block-granular attention mass.

Measures wall time (torch.cuda.Event) and peak device memory
(torch.cuda.max_memory_allocated) across sequence lengths. The dense oracle
(tests/naive.py) is included at small N only, to show WHY the naive route
(materializing [B, H, N, N]) is not viable.

Usage:  python bench/bench_block_mass.py [--out results.json]
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.naive import block_mass_naive  # noqa: E402
from kvblockmass.reference import block_mass_reference  # noqa: E402
from kvblockmass.triton_kernel import block_mass_triton  # noqa: E402

NS = [1024, 2048, 4096, 8192]
DENSE_NAIVE_NS = [1024, 2048]          # O(N^2) oracle: small N only
B, H_Q, H_KV, D, BLOCK = 1, 8, 2, 64, 64
REPEATS = 5


def _backend() -> str:
    return "rocm" if torch.version.hip else "cuda"


def _bench(fn, *args, **kw):
    """Return (median_ms, peak_mib, result)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn(*args, **kw)                     # warmup / compile
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(REPEATS):
        start.record()
        res = fn(*args, **kw)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    peak = torch.cuda.max_memory_allocated() / 2**20
    return times[len(times) // 2], peak, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "this benchmark needs a CUDA/ROCm device"
    dev = torch.device("cuda")
    torch.manual_seed(0)

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device_name": torch.cuda.get_device_name(0),
        "backend": _backend(),
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "python": platform.python_version(),
        "shape": {"B": B, "H_q": H_Q, "H_kv": H_KV, "D": D, "block_size": BLOCK},
        "repeats": REPEATS,
    }
    print(json.dumps(meta, indent=2))

    rows = []
    for N in NS:
        g = torch.Generator(device="cpu").manual_seed(N)
        q = torch.randn(B, H_Q, N, D, generator=g, dtype=torch.float32).to(dev)
        k = torch.randn(B, H_KV, N, D, generator=g, dtype=torch.float32).to(dev)

        row = {"N": N}

        t_ref, mem_ref, res_ref = _bench(block_mass_reference, q, k, causal=True, block_size=BLOCK)
        row["reference_ms"], row["reference_mib"] = t_ref, mem_ref

        t_tri, mem_tri, res_tri = _bench(block_mass_triton, q, k, causal=True, block_size=BLOCK)
        row["triton_ms"], row["triton_mib"] = t_tri, mem_tri

        rel = ((res_ref - res_tri).abs() / res_ref.abs().clamp_min(1e-6)).max().item()
        row["rel_err"] = rel
        assert rel < 1e-3, f"N={N}: triton/reference rel err {rel}"

        if N in DENSE_NAIVE_NS:
            t_nv, mem_nv, res_nv = _bench(block_mass_naive, q, k, causal=True, block_size=BLOCK)
            row["dense_naive_ms"], row["dense_naive_mib"] = t_nv, mem_nv
            assert (res_nv - res_ref).abs().max().item() < 1e-4

        rows.append(row)
        print(json.dumps(row))

    result = {"meta": meta, "rows": rows}
    out = args.out
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = str(Path(__file__).parent / f"bench_{_backend()}_{stamp}.json")
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")

    # markdown table alongside
    md = [f"# bench @ {meta['device_name']} ({meta['backend']})", "",
          f"torch {meta['torch']} / triton {meta['triton']} / shape B{B} Hq{H_Q} Hkv{H_KV} D{D} block{BLOCK}", "",
          "| N | dense O(N²) ms | dense MiB | reference ms | reference MiB | triton ms | triton MiB | rel err |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['N']} | {r.get('dense_naive_ms', float('nan')):.2f} | {r.get('dense_naive_mib', float('nan')):.0f} "
            f"| {r['reference_ms']:.2f} | {r['reference_mib']:.0f} "
            f"| {r['triton_ms']:.2f} | {r['triton_mib']:.0f} | {r['rel_err']:.2e} |"
        )
    md_path = str(Path(out).with_suffix(".md"))
    Path(md_path).write_text("\n".join(md) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
