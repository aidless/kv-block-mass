# From one kernel to four layers: what the papers changed (and what they didn't)

This repo started as a single question: *can per-block attention mass be computed without ever
materializing `[B, H, N, N]`?* The kernel answers yes. Everything below that line came from
reading two papers closely and asking what a *library* would have to prove, not just measure.

Every number here was produced by code in this repo (tests print them; see
`tests/test_certificate.py`, `tests/test_budget.py`, `tests/mutation_gate_evict.py`). Nothing in
this document is projected.

## The papers

| # | Source | One-line claim | What I took | What I did **not** take |
|---|---|---|---|---|
| P1 | **Trust the Mass: Forced Weights in KV-Cache Eviction** — arXiv:2608.25230 | top-weight retention is near-optimal; the real problem is that "budget" is a fiction in half the literature | the byte ledger (`budget.py`), the "mask over dense tensor" failure mode | the 14–62 benchmark-point cost of enforcing a budget on a fixed selection — needs a real model + benchmark, out of scope here |
| P2 | **Error Certificates for KV-Cache Eviction via Randomized Design** — arXiv:2607.21475 | a deterministic value-blind eviction cannot estimate its own error; a Poisson-sampled design can, with a Hájek estimator + empirical-Bernstein radius | the whole statistical layer (`evict.py`, `certificate.py`): certainty set = top slice + sinks + recency, `π = clip(m·w/Σw, ε, 1)`, Hájek ratio estimate, `r = (√(2V̂ log 1/δ) + b log 1/δ)/(‖ŷ‖+ε₀)` | their absolute error levels and coverage on 12 real architectures — those need real models; and the `1/3` in the textbook range term (see caveat below) |
| P3 | **Correct but Slow** — arXiv:2607.04454 | correctness suites do not detect kernels that cannot be swapped in | the framing behind `tests/mutation_gate.py`: an oracle comparison is not enough, a *mutation* must be caught | their 22-kernel benchmark matrix |
| P4 | PagedAttention (SOSP'23), vAttention (ASPLOS'25), ForkKV — arXiv:2604.06370 | block lifetime vs contiguous memory vs copy-on-write sharing | the block-granular framing shared with `aidless/paged-kv` | a scheduler / memory allocator — `paged-kv` owns that; this repo owns the score |

Full texts of P1 and P2 were read section by section (78,109 and 64,901 characters); the numbers
quoted above are theirs, and they are labelled as theirs — none of their absolute quality results
are reproduced in this repo's results tables.

## What the measurements changed about the design

### 1. The certainty set must contain the score top slice (a real bug of mine)

First draft: certainty = attention sinks + recency window only, tail sampled by Poisson. Result at
25% budget: **Hájek 0.2298 vs top-k 0.0952** median relative error — the design lost badly. Cause:
the heaviest tokens were being *sampled* rather than kept, so variance exploded. P2's certainty set
includes the score top slice; with `top_slice = budget//2` the same comparison at N=4096 became
**0.0627 vs 0.1806 (2.88× in the design's favour)**. This is recorded as mutant
`NO_TOP_SLICE` in the mutation gate so the regression cannot come back silently.

### 2. The design wins on heavy-tailed scores — and **loses** on near-Gaussian ones

Asserted in both directions (`test_design_wins_on_heavy_tailed_scores`,
`test_topk_wins_on_near_gaussian_scores`), N=4096, 25% budget, 40 draws:

| score distribution | design + Hájek | top-k at matched count | winner |
|---|---|---|---|
| power-law (Zipf, permuted) | **0.0627** | 0.1806 | design (2.88×) |
| near-Gaussian (`N(0,1)` logits) | 0.8641 | **0.6323** | top-k (1.37×) |

So the honest claim is **not** "this beats top-k". It is: *on heavy-tailed score distributions the
design is better at the same byte count; on flat distributions top-k wins and the design's only
purchase is auditability.* A README that claimed the first row alone would be misleading, which is
why the second row is a test that fails if top-k ever stops winning.

### 3. What the certificate buys, measured

| property | measured | how |
|---|---|---|
| coverage at δ=0.1 | **1.000** at (N=1024, 25%), (N=4096, 25%), (N=4096, 50%), 60 draws each | radius contains the realized error every time — conservative, not nominal |
| tightness | median radius/error **1.83–5.06** | a certificate 1000× loose would be decoration; bounded by test |
| ranking | Spearman **0.926** over budgets {12.5%, 25%, 50%}, N=4096 | usable as a risk signal even when loose |
| unbiasedness | Hájek mean over **800 designs**: bias **0.00413**; top-k's fixed error on the same problem: **0.2002** (48×) | the reason the design exists |
| variance term is load-bearing | variance share of the radius: **0.05** at 12.5%, **0.19–0.29** at 25%, **0.56–0.70** at 50% | range-only coverage collapses to 0.68 (N=1024) / **0.00** (N=4096) at 50% — this is why the coverage test is parametrized by budget |

That last row came from the mutation gate: `MISSING_VARIANCE` initially **survived**, because the
suite only tested 25% budget where the range term dominates. Adding the 50% budget to the coverage
test made the mutant die. A gate that cannot kill a mutant is a decoration; this one now kills 5/5.

### 4. Theorem 1 as an executable test

P2's Theorem 1 says a deterministic value-blind scheme has no consistent error estimator. Test
form: keep the retained set fixed, perturb **only the evicted values**, and check that every
retained statistic is bit-identical while the realized relative error moves
**0.1485 → 1.0390 (7.0×)**, saturating near 1.0 as the evicted mass dominates. If this ever stops
being true, the certificate has become unnecessary and the module's premise is wrong.

### 5. "Same budget" can be 4× different bytes

Byte ledger on a realistic shape (28 layers, 8 heads, N=4096, D=128, fp16 → full cache **448 MiB**):

| storage strategy | bytes | fraction of full | verdict |
|---|---|---|---|
| dense + per-head mask (claims 25%) | 469,876,736 | **100.02%** | `holds_full_cache=True` |
| ragged (1024 kept tokens/head) | 117,497,856 | **25.01%** | budget physically enforced |

The mask costs **0.024%** of the cache — which is exactly why the failure mode in P1 is cheap to
commit and invisible in a "budget" column: **a policy can claim 25% and hold 100% for 0.02% extra
memory.** The audit exposes both numbers (`as_stored_frac`, `enforced_frac`) so a results table
cannot quote the flattering one.

## Honest boundaries

- **No real model, no benchmark.** The policy layer runs on synthetic score distributions
  (power-law and near-Gaussian) with random values. Absolute error levels here are *not*
  comparable to P2's 0.0317/0.0447 on real attention; only the relative ordering and the
  certificate properties transfer.
- **The quality cost of enforcing a budget is not measured.** P1's 14–62 benchmark points on
  LongBench/RULER need a model and a harness this repo does not have.
- **The free `1/3`**: the textbook empirical-Bernstein range term carries a factor 1/3; P2 states
  the shape without it. We implement it as stated (conservative). Measured coverage, not the
  textbook, is the acceptance criterion.
- **Sampling variance estimates target the linearized error**, so they differ from the true ratio
  error by `O(B²/m²)` (P2's own caveat).

## Reproduction

```bash
PYTHONPATH=src python -m pytest tests/ -q            # 49 passed, 8 skipped (GPU cases need CUDA/HIP)
PYTHONPATH=src python tests/mutation_gate.py         # kernel: 4 injections, all caught
PYTHONPATH=src python tests/mutation_gate_evict.py   # policy layers: 5 mutants, all caught
```
