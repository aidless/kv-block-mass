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

All numbers measured (not projected), 2026-10-31. Shapes: B=1, H_q=8, H_kv=2, D=64,
block_size=64, causal; median of 5; memory = `torch.cuda.max_memory_allocated`.

**Tesla T4** (Kaggle, torch 2.11.0+cu128, Triton 3.6.0 — `bench/bench_cuda_T4_20261031.json`):

| N | dense O(N²) MiB | reference ms | reference MiB | triton ms | triton MiB | rel err |
|---|---|---|---|---|---|---|
| 1024 | 110 | 3.47 | 47 | 4.26 | 11 | 2.2e-6 |
| 2048 | 405 | 12.32 | 87 | 15.19 | 13 | 3.0e-6 |
| 4096 | — | 40.18 | 174 | **22.59** | 18 | 7.5e-6 |
| 8192 | — | 161.34 | 372 | **89.76** (1.8×) | **28** (13× less) | 1.5e-5 |

**AMD Radeon W7900-class (gfx1100)** (ROCm 7.2, torch 2.14.1+rocm7.2, Triton 3.5.1/HIP —
`bench/bench_rocm_20261031T172739Z.json`):

| N | dense O(N²) MiB | reference ms | reference MiB | triton ms | triton MiB | rel err |
|---|---|---|---|---|---|---|
| 1024 | 134 | 1.06 | 71 | 10.42 | 35 | 3.4e-6 |
| 2048 | 429 | 2.41 | 111 | 25.01 | 37 | 3.2e-6 |
| 4096 | — | 7.67 | 198 | 49.56 | 42 | 7.9e-6 |
| 8192 | — | 59.39 | 396 | 137.41 | **52** (7.6× less) | 1.6e-5 |

Honest reading, both directions:

- **Memory**: the Triton kernel's footprint grows O(1) in N (one 64×64 tile) while the
  streaming reference grows linearly (chunk×N score buffer) — 13× less at N=8192 on T4.
  The dense O(N²) baseline that H2O-style scoring normally pays is 429 MiB at N=2048
  *for this tiny shape* and grows quadratically from there.
- **Speed**: on T4 the kernel wins from N=4096 (1.8× at 8192). On RDNA3/HIP it does not —
  the 64×64 `input_precision="ieee"` dot compiles poorly on gfx1100, so the PyTorch
  streaming reference (rocBLAS-backed) is faster there. We report both rather than
  benchmarking only the GPU that flatters the kernel. Tuning (BLOCK sizes, fewer
  atomics, fp16 accumulation experiments) is open work.

Correctness across backends: Triton-vs-reference agreement ≤ 1.6e-5 max relative error
on both CUDA and ROCm; the fp64 dense oracle agrees to ~1e-6 on every bucket.

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
