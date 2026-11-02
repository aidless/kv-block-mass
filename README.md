# kv-block-mass

[![ci](https://github.com/aidless/kv-block-mass/actions/workflows/ci.yml/badge.svg)](https://github.com/aidless/kv-block-mass/actions/workflows/ci.yml)

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

Causal early exit (`causal_skip`, on by default): under a causal mask, key columns past a
query block's **last** row are masked for every row of that block, so they contribute exactly
0 and are never touched. Measured on CPU (torch 2.14.1, 8 threads, streaming reference,
`query_chunk=256`, median of 3): N=2048 **291.2 → 123.5 ms (2.36×)**, N=4096
**1417.2 → 496.0 ms (2.86×)**, output **bit-identical** (max relative diff 0.0e+00).
The Triton kernel applies the same bound as a dynamic loop trip count. The bound is
mutation-tested: `EARLY_EXIT_TOO_TIGHT` (using the block's first row instead of its last,
which silently drops the diagonal block) is one of the four injections the oracle must catch.

Correctness across backends: Triton-vs-reference agreement ≤ 1.6e-5 max relative error
on both CUDA and ROCm; the fp64 dense oracle agrees to ~1e-6 on every bucket.

Two independent T4 runs are archived: `bench/bench_cuda_T4_20261031.json` (worker v5,
numbers recovered from stdout after the final write hit Kaggle's read-only FS) and
`bench/bench_cuda_T4_v6_final.json` (worker v6, status COMPLETE, code at commit 278d479).
They agree within run-to-run noise — e.g. N=8192 triton 89.8 vs 91.6 ms, 28 MiB both.

## Layers 2–4: policy, certificate, byte ledger

The kernel says *what the mass is*. The next three layers answer questions that a results table
usually gets away with not answering: **which blocks to drop, how wrong that choice is, and how
many bytes were actually kept.** They came from two papers read closely; the full accounting of
what was taken, what was refused and what the measurements changed is in
[`docs/evolution-from-papers.md`](docs/evolution-from-papers.md).

```python
from kvblockmass.evict import design_from_scores
from kvblockmass.certificate import error_certificate, full_cache_output, relative_error
from kvblockmass.budget import CacheLayout, audit_layout

design = design_from_scores(token_scores, tail_budget=256, n_sinks=4, recency=32, top_slice=128)
cert = error_certificate(scores, values, design.keep, design.pi, design.certainty, delta=0.1)
print(cert.radius)                    # relative-error bound, from the retained set alone
print(audit_layout(layout, kept).as_row())   # what a "25% budget" really holds
```

Measured, CPU, 2026-11-01 (printed by the tests, so the README cannot drift from them):

| property | measured |
|---|---|
| certificate coverage at δ=0.1 | **1.000** at (N=1024, 25%), (N=4096, 25%), (N=4096, 50%) |
| certificate tightness | median radius/error **1.83–5.06** (loose, bounded, not vacuous) |
| certificate ranking | Spearman **0.926** over budgets {12.5%, 25%, 50%} |
| design unbiasedness | Hájek mean over **800 designs**: bias **0.00413**; top-k fixed error **0.2002** (48×) |
| design vs top-k, power-law scores | **0.0627 vs 0.1806** (2.88× in the design's favour, N=4096, 25%) |
| design vs top-k, near-Gaussian scores | 0.8641 vs **0.6323** — *top-k wins here* (asserted, not hidden) |
| deterministic scheme's blindness (Thm 1) | realized error **0.1485 → 1.0390 (7.0×)** with bit-identical retained statistics |

And the byte ledger, on 28 layers × 8 heads × N=4096 × D=128 × fp16 (full cache **448 MiB**):

| storage strategy | bytes | fraction of full | verdict |
|---|---|---|---|
| dense + per-head mask, claims 25% | 469,876,736 | **100.02%** | `holds_full_cache = True` |
| ragged, 1024 kept tokens/head | 117,497,856 | **25.01%** | budget physically enforced |

The mask costs **0.024%** of the cache, which is precisely why "same budget" comparisons can
compare 4× different memory. The audit reports both numbers instead of one ratio.

The regime boundary is the honest headline: the design wins on heavy-tailed score distributions
and **loses** to top-k on near-Gaussian ones, where the only thing it buys is auditability. Both
directions are tests (`test_design_wins_on_heavy_tailed_scores`,
`test_topk_wins_on_near_gaussian_scores`) so the claim cannot be generalised by a later edit.

## Layout

```
src/kvblockmass/reference.py      O(N) PyTorch: query-chunk streaming + bucket scatter-add
src/kvblockmass/triton_kernel.py  fused Triton: two passes over k (max+denom, then add),
                                   tl.atomic_add into block buckets
src/kvblockmass/evict.py          Poisson design: certainty set (top slice + sinks + recency) + tail
src/kvblockmass/certificate.py    Hájek estimate + empirical-Bernstein radius (O(|S|) per head)
src/kvblockmass/budget.py         byte ledger: what a policy holds vs the budget it claims
tests/naive.py                    dense [B,H,N,N] oracle — TESTS ONLY
tests/test_reference.py           sum rule, GQA, ragged blocks, masks, NaN discipline
tests/test_triton.py              reference-vs-kernel agreement (GPU runners only)
tests/test_evict.py               design bounds, expected retention, Hájek offset, reproducibility
tests/test_certificate.py         coverage/tightness/ranking/unbiasedness/regime/Theorem-1
tests/test_budget.py              byte arithmetic + the mask-vs-ragged verdicts
tests/mutation_gate.py            injects 4 real bugs (2 shapes), asserts the oracle CATCHES all
tests/mutation_gate_evict.py      runs the real suites under 5 mutant plugins; all must fail
.github/workflows/ci.yml          CPU test suite + mutation gate on every push
bench/bench_block_mass.py         CUDA-Event timing + peak-memory, JSON/MD output
kaggle/run_bench_kernel.py        T4 entry point
docs/reading-notes.md             what we learned from torch_npu / torch_musa / aiter
docs/evolution-from-papers.md     what two KV papers changed, and what was refused
```

## Testing discipline

Two gates, because the two halves of the repo fail differently. The kernel half is compared against
a dense fp64 oracle in-process (`tests/mutation_gate.py`). The statistical half is gated by running
**the real test files under mutant plugins** (`tests/mutation_gate_evict.py`, `-p tests.mutants.X`):
`NO_HAJEK_OFFSET`, `MISSING_VARIANCE`, `NO_TOP_SLICE`, `NO_BLOCK_TABLE`, `ALWAYS_ENFORCED` — each
must make the suite fail, with a clean-tree control that must pass. Building it paid for itself
immediately: `MISSING_VARIANCE` **survived** the first version of the suite, because coverage was
only tested at 25% budget where the range term dominates the radius; adding a 50% budget case
(variance share there is 0.56–0.70) killed it.

The gate that matters is **exact comparison against the dense oracle**, not correlation.
In a 2026-10 incident on the predecessor experiment (`kvfrag`), dropping the causal mask left
Pearson correlation > 0.999 while relative error was ~0.2 — "looks correlated" is not
correctness. `tests/mutation_gate.py` proves the oracle catches: dropped causality, missing softmax
denominator, bucket off-by-one, and an over-tight causal early-exit bound (measured max
errors 3.5e2 / 4.9e4 / 5.8e2 / 6.1e2 at N=128 and 7.5e2 / 2.6e5 / 1.0e3 / 7.0e2 at N=600;
control 2.3e-13 / 5.7e-13).

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
