#!/usr/bin/env python3
"""Render the peak-memory and latency comparison from the committed bench JSON.

Numbers are read straight out of bench/bench_cuda_T4_20261031.json (Tesla T4);
nothing is hand-typed. Run: python3 docs/make_bench_figure.py
"""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
data = json.loads((ROOT / "bench" / "bench_cuda_T4_20261031.json").read_text())
rows = data["rows"]
N = [r["N"] for r in rows]
ref_mem = [r["reference_mib"] for r in rows]
tri_mem = [r["triton_mib"] for r in rows]
ref_ms = [r["reference_ms"] for r in rows]
tri_ms = [r["triton_ms"] for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
import numpy as np
x = np.arange(len(N)); w = 0.38
ax1.bar(x - w/2, ref_mem, w, label="streaming reference (O(N))", color="#c44e52")
ax1.bar(x + w/2, tri_mem, w, label="Triton kernel (O(1))", color="#4c72b0")
ax1.set_xticks(x); ax1.set_xticklabels([str(n) for n in N])
ax1.set_xlabel("sequence length N"); ax1.set_ylabel("peak memory (MiB)")
ax1.set_title("Peak GPU memory — Tesla T4"); ax1.legend(frameon=False)
for i, v in enumerate(tri_mem):
    ax1.text(i + w/2, v + 4, f"{v:.0f}", ha="center", fontsize=9)

ax2.bar(x - w/2, ref_ms, w, label="streaming reference", color="#c44e52")
ax2.bar(x + w/2, tri_ms, w, label="Triton kernel", color="#4c72b0")
ax2.set_xticks(x); ax2.set_xticklabels([str(n) for n in N])
ax2.set_xlabel("sequence length N"); ax2.set_ylabel("latency (ms, median of 5)")
ax2.set_title("Latency — kernel wins from N=4096"); ax2.legend(frameon=False)
for i, v in enumerate(tri_ms):
    ax2.text(i + w/2, v + 3, f"{v:.0f}", ha="center", fontsize=9)

fig.suptitle("kv-block-mass: block-mass scoring without materialising [B,H,N,N]  (B=1,H_q=8,H_kv=2,D=64,block=64,causal)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = ROOT / "docs" / "figures" / "bench-t4-mem-latency.png"
fig.savefig(out, dpi=130)
print("wrote", out)
