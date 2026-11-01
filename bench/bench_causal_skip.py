"""Benchmark the causal early-exit (`causal_skip`) path vs the full path.

Reproduces the CPU timing numbers reported in README ("Causal early exit"):
N=2048 291.2 -> 123.5 ms (2.36x), N=4096 1417.2 -> 496.0 ms (2.86x), with the
skip path being *bit-identical* (max relative diff 0.0e+00).

Usage:
    python bench/bench_causal_skip.py [--repeats 3]

Numbers are median wall-clock over `--repeats` runs of the streaming reference
(`query_chunk=256`, torch CPU). The script prints a markdown table and a JSON
record to stdout; pass `--json bench/bench_causal_skip_<ts>.json` to archive it.
"""

import argparse
import json
import time

import torch

from kvblockmass.reference import block_mass_reference


def _qk(N, H_q=2, H_kv=1, D=16, seed=91, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(1, H_q, N, D, generator=g, device=device),
        torch.randn(1, H_kv, N, D, generator=g, device=device),
    )


def median_ms(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json", default=None, help="write JSON record to this path")
    args = ap.parse_args()

    print(f"# causal_skip benchmark (CPU, torch {torch.__version__}, "
          f"repeats={args.repeats}, median)")
    print("| N | full ms | skip ms | speedup | max rel diff |")
    print("|---|---|---|---|---|")

    rows = []
    for N in (2048, 4096):
        q, k = _qk(N)
        full = median_ms(
            lambda: block_mass_reference(q, k, causal=True, block_size=64,
                                         query_chunk=256, causal_skip=False),
            args.repeats,
        )
        skip = median_ms(
            lambda: block_mass_reference(q, k, causal=True, block_size=64,
                                         query_chunk=256, causal_skip=True),
            args.repeats,
        )
        full_val = block_mass_reference(q, k, causal=True, block_size=64,
                                        query_chunk=256, causal_skip=False)
        skip_val = block_mass_reference(q, k, causal=True, block_size=64,
                                        query_chunk=256, causal_skip=True)
        max_rel = float((skip_val - full_val).abs().max().item() /
                        max(full_val.abs().max().item(), 1e-30))
        speedup = full / skip
        print(f"| {N} | {full:.1f} | {skip:.1f} | {speedup:.2f}x | {max_rel:.2e} |")
        rows.append({"N": N, "full_ms": full, "skip_ms": skip,
                     "speedup": speedup, "max_rel_diff": max_rel})

    if args.json:
        record = {
            "meta": {
                "torch": torch.__version__,
                "device": "cpu",
                "repeats": args.repeats,
                "query_chunk": 256,
            },
            "rows": rows,
        }
        with open(args.json, "w") as f:
            json.dump(record, f, indent=1)
        print(f"\n# archived -> {args.json}")


if __name__ == "__main__":
    main()
