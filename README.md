# kv-block-mass

**Block-granular attention mass in O(N) memory — reference + Triton kernels, cross-vendor (CUDA / ROCm).**

Companion of [`aidless/paged-kv`](https://github.com/aidless/paged-kv): paged-kv manages block
*lifetimes* (refcount / COW / failure-side-effect-free bookkeeping). This repo computes
*which blocks deserve to live* — the per-block attention-mass scores used by KV-eviction
research (H2O / SnapKV style) — **without materializing the `[B, H, N, N]` softmax matrix**.

That matrix is the reason H2O-style scoring gives up FlashAttention: for N=4096, one layer,
fp32, it is 512 MiB **per head per batch**. We stream over query chunks and scatter-add column
sums into `num_blocks` buckets, so peak memory is `O(B·H·chunk·N)` for the reference and one
score tile for the Triton kernel.

## What it computes

```
mass[j] = Σ_{b,h,i, j_pos ∈ block j} softmax_j( q_i · k_j · scale )
```

- `q: [B, H_q, N, D]`, `k: [B, H_kv, N, D]` — GQA supported (`H_q % H_kv == 0`)
- causal and arbitrary `attn_mask` (`[N]` column or `[N, N]` pair mask)
- **sum rule**: every unmasked query row distributes exactly 1.0 over keys, so
  `Σ mass == B · H_q · N` — this identity is a test, not a hope
- fully-masked rows contribute exactly 0 (no NaN leakage)

## Results

Filled from measured runs — see `bench/` JSON + MD artifacts:

> TODO: paste T4 / ROCm tables after measured runs land.

## Layout

```
src/kvblockmass/reference.py      O(N) PyTorch: query-chunk streaming + bucket scatter-add
src/kvblockmass/triton_kernel.py  fused Triton: two passes over k (max+denom, then add),
                                   tl.atomic_add into block buckets
tests/naive.py                    dense [B,H,N,N] oracle — TESTS ONLY
tests/test_reference.py           sum rule, GQA, ragged blocks, masks, NaN discipline
tests/test_triton.py              reference-vs-kernel agreement (GPU runners only)
tests/mutation_gate.py            injects 3 real bugs, asserts the oracle CATCHES all
bench/bench_block_mass.py         CUDA-Event timing + peak-memory, JSON/MD output
kaggle/run_bench_kernel.py        T4 entry point
docs/reading-notes.md             what we learned from torch_npu / torch_musa / aiter
```

## Testing discipline

The gate that matters is **exact comparison against the dense oracle**, not correlation.
In a 2026-10 incident on the predecessor experiment (`kvfrag`), dropping the causal mask left
Pearson correlation > 0.999 while relative error was ~0.2 — "looks correlated" is not
correctness. `tests/mutation_gate.py` proves the oracle catches: dropped causality,
missing softmax denominator, and bucket off-by-one (measured max errors 3.5e2 / 4.9e4 /
5.8e2 respectively, control 2.3e-13).

## Usage

```python
import torch
from kvblockmass import block_mass_reference          # CPU or any device
from kvblockmass.triton_kernel import block_mass_triton  # CUDA / ROCm

q = torch.randn(1, 8, 4096, 64, device="cuda")
k = torch.randn(1, 2, 4096, 64, device="cuda")
mass = block_mass_triton(q, k, causal=True, block_size=64)   # [64]
```

## License

MIT — see [LICENSE](LICENSE).
