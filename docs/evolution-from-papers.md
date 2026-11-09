# From one kernel to four layers: what the papers changed (and what they didn't)

This repo started as a single question: *can per-block attention mass be computed without ever
materializing `[B, H, N, N]`?* The kernel answers yes. Everything below that line came from
reading two papers closely and asking what a *library* would have to prove, not just measure.

Checked numeric tables live in the README and are compared to committed JSON by
`scripts/check_readme_drift.py`. This note does not keep a second results table.
Where an older draft quoted a figure that is not in those files, it is marked
UNKNOWN below. Nothing here is a projected speedup.

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

First draft: certainty = attention sinks + recency window only, tail sampled by Poisson. The
heaviest tokens were being *sampled* rather than kept, and the design lost to top-k. The pair of
median errors from that draft is not in any committed JSON, so it is UNKNOWN. P2's certainty set
includes the score top slice; with `top_slice = budget//2` the comparison is the README regime
rows (Zipf favours the design). The regression is mutant `NO_TOP_SLICE`.

### 2. The design wins on heavy-tailed scores — and **loses** on near-Gaussian ones

Asserted in both directions (`test_design_wins_on_heavy_tailed_scores`,
`test_topk_wins_on_near_gaussian_scores`). The measured medians are the README regime rows.

The honest claim is not "this beats top-k". On heavy-tailed score distributions the design is
better at the same count; on near-Gaussian scores top-k wins and the design's only purchase is
auditability. The losing side is a test, so a later edit cannot keep only the win.

### 3. What the certificate buys, measured

Coverage, tightness, ranking, unbiasedness, and variance-share medians are the README policy
table, including the small-budget rows the tightness assertion does not gate. An earlier note
quoted a single tightness range and a variance-share range that hid which `N` was which. Those
collapsed ranges are withdrawn. Use the table.

`MISSING_VARIANCE` initially survived, because the suite only tested one budget where the range
term dominates. Adding the larger budget made the mutant die. The gate is
`tests/mutation_gate_evict.py`: six mutants, each must fail, plus a clean control.

### 4. Theorem 1 as an executable test

P2's Theorem 1 says a deterministic value-blind scheme has no consistent error estimator. Test
form: keep the retained set fixed, perturb **only the evicted values**, and check that every
retained statistic is bit-identical while the realized relative error moves. The archived
min, max, and spread are the README theorem rows. The error saturates as the evicted values
dominate. If the move ever stops, the certificate has become unnecessary and the module's
premise is wrong.

### 5. "Same budget" can still be the full cache

The byte ledger for the shape in the README (layers, heads, sequence, dim, fp16) is the README
ledger table. Read both boolean columns. `nominal_is_enforced` follows the ragged byte count
even when the stored layout is a mask, and `holds_full_cache` is true on that mask row. The
mask overhead is why P1's failure mode is cheap to commit and invisible in a single "budget"
column: a policy can claim a fraction of the tokens and still hold the full cache. The audit
exposes both fractions so a results table cannot quote the flattering one.

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
python scripts/check_readme_drift.py
python -m pytest tests/ -q
python tests/mutation_gate.py
python tests/mutation_gate_evict.py
```

GPU numeric tests skip without a device. A pass count is not frozen here, because it goes stale
the moment a test is added. CI is the list of commands in the README.

## What the layers compose into

`reclaim.plan_reclaim` is the seam. The papers describe a *method* (design + certificate)
and a *ledger* critique (mask stored densely is not a saving); neither hands you the
join, and the join is where a token budget can silently still be a full cache.
The plan therefore returns the card a table would quote — `kept_tokens` **and**
`bytes_as_stored` / `bytes_enforced` — so the two cannot drift apart in prose.
