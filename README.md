# kv-block-mass

[![ci](https://github.com/aidless/kv-block-mass/actions/workflows/ci.yml/badge.svg)](https://github.com/aidless/kv-block-mass/actions/workflows/ci.yml)

**STATUS: WORK-IN-PROGRESS — not A yet.**

<!-- drift:version -->
Version: 0.1.0. Not on PyPI.
<!-- /drift:version -->

## Problem

Computes which KV blocks deserve to live: per-block attention mass, without building the dense `[B, H, N, N]` softmax.
Not a full eviction system (no scheduler, no real-model quality number) and not production serving.
Companion of [`aidless/paged-kv`](https://github.com/aidless/paged-kv): that repo keeps block lifetimes; this one only scores which blocks the mass says are worth keeping.

## Gaps

1. GPU timings were not re-run. No CUDA or ROCm device was used for this change. The GPU tables are OBSERVED from the committed JSON named under each table. Those files were not edited and no new timing was written. The v6 `timestamp_utc` remains the archived string `2026-10-31T19:5xZ`, which is not valid ISO.
2. CI does not launch the Triton kernel. Numeric tests in `tests/test_triton.py` skip when `torch.cuda.is_available()` is false. The structural test only imports the module.
3. The CPU policy, oracle, and early-exit files are one snapshot. `meta.torch` in each file names the build. CI recomputes the ledger from the library and checks that this README matches the files. It does not re-compare the policy or oracle floats. The CI torch spec is a minimum, recorded in the workflow, not that snapshot's build.
4. Causal early-exit wall clock is UNKNOWN. Millisecond quotes that used to appear here were not in any committed JSON and were removed rather than repeated.
5. A fp64 dense-oracle error for the GPU kernel is UNKNOWN. The GPU `rel err` column is Triton versus the streaming reference on device. The CPU oracle table is a different comparison.
6. The certificate tightness assertion covers three budget rows. The policy table also records a smaller budget whose median radius/error is above that assertion's cap. That row is measured and shown. It is not a passing gate.
7. Scope, restated above: not a full eviction system and not production serving. Block mass is not a task-quality score. See FAILURE_MODES.

## What CI runs (no GPU)

Ubuntu latest, two Python versions. The job installs a CPU torch wheel and Triton, then runs:

<!-- drift:ci-commands -->
python scripts/check_readme_drift.py
pytest tests/ -q
python tests/mutation_gate.py
python tests/mutation_gate_evict.py
python examples/policy_card.py
<!-- /drift:ci-commands -->

Triton is there so `kvblockmass.triton_kernel` imports. The job does not call the kernel. It does not run the GPU bench. The manual GPU path, on a machine that actually has a device, is:

```bash
python bench/bench_block_mass.py
```

That writes a new stamped JSON next to the script. Do not point it at the archived files. New `bench/bench_*_20*.json` names are gitignored until someone force-adds a real run. This change did not do that.

## Offline path (no GPU)

Reference tests plus the drift check. No Triton required for this path. The sum rule is `test_sum_rule_full_mass_equals_query_count`: every unmasked query row contributes one, and a fully masked row contributes nothing.

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0"
pip install pytest
pip install -e .
pytest tests/test_reference.py -q
python scripts/check_readme_drift.py
```

## What it computes

```text
mass[j] = sum of softmax(q_i · k · scale) over batch, query head, query row,
          and key positions that fall in block j
```

`q` is `[B, H_q, N, D]` and `k` is `[B, H_kv, N, D]`. GQA requires `H_q` divisible by `H_kv` and raises `ValueError` otherwise. The head map is repeat-interleave, not modulo. Causal masks and a `[N]` or `[N, N]` keep-mask are supported. The reference streams query chunks. The Triton kernel keeps a tile and atomically adds into block buckets. Both are research code.

The reference also allocates an `N` by `N` boolean keep mask (one byte per entry). That is separate from the score tile. Peak device memory in the GPU tables is `torch.cuda.max_memory_allocated` for the whole step, including `Q` and `K`, so the Triton column is not an O(1) process footprint.

## Results (OBSERVED)

Rounding is fixed in `scripts/check_readme_drift.py`: milliseconds to two fractional digits, MiB to the nearest integer, relative error to two significant digits. A dash means that JSON row has no `dense_naive_mib` (the dense oracle was not run at that `N`). Ratios in the claims table are computed from the raw JSON, not from the rounded cells.

An empty dense cell is not a zero. Do not read a missing dense run as a memory win.

**Tesla T4, OBSERVED**, from `bench/bench_cuda_T4_20261031.json` (worker v5, recovered from stdout). Shape in that file: batch 1, eight query heads, two KV heads, head dim 64, block 64, causal, median of five.

<!-- drift:gpu-t4-v5 -->
| N | dense MiB | reference ms | reference MiB | triton ms | triton MiB | rel err |
|---|---|---|---|---|---|---|
| 1024 | 110 | 3.47 | 47 | 4.26 | 11 | 2.2e-6 |
| 2048 | 405 | 12.32 | 87 | 15.19 | 13 | 3.0e-6 |
| 4096 | — | 40.18 | 174 | 22.59 | 18 | 7.5e-6 |
| 8192 | — | 161.34 | 372 | 89.76 | 28 | 1.5e-5 |
<!-- /drift:gpu-t4-v5 -->

**Tesla T4, second run, OBSERVED**, from `bench/bench_cuda_T4_v6_final.json` (worker v6). Same shape. Not a replacement for the v5 table.

<!-- drift:gpu-t4-v6 -->
| N | dense MiB | reference ms | reference MiB | triton ms | triton MiB | rel err |
|---|---|---|---|---|---|---|
| 1024 | 110 | 3.42 | 47 | 4.25 | 11 | 2.1e-6 |
| 2048 | 405 | 12.28 | 87 | 15.79 | 13 | 2.8e-6 |
| 4096 | — | 40.21 | 174 | 23.25 | 18 | 6.9e-6 |
| 8192 | — | 160.73 | 372 | 91.61 | 28 | 1.5e-5 |
<!-- /drift:gpu-t4-v6 -->

**ROCm, OBSERVED**, from `bench/bench_rocm_20261031T172739Z.json`. The device field is `AMD Radeon Graphics`, backend `rocm`. Same shape and repeat count.

<!-- drift:gpu-rocm -->
| N | dense MiB | reference ms | reference MiB | triton ms | triton MiB | rel err |
|---|---|---|---|---|---|---|
| 1024 | 134 | 1.06 | 71 | 10.42 | 35 | 3.4e-6 |
| 2048 | 429 | 2.41 | 111 | 25.01 | 37 | 3.2e-6 |
| 4096 | — | 7.67 | 198 | 49.56 | 42 | 7.9e-6 |
| 8192 | — | 59.39 | 396 | 137.41 | 52 | 1.6e-5 |
<!-- /drift:gpu-rocm -->

Checked ratios and the max relative error across all three files:

<!-- drift:gpu-claims -->
| claim | value |
|---|---|
| t4 v5 N=8192 reference_ms/triton_ms | 1.8× |
| t4 v5 N=8192 reference_mib/triton_mib | 13.2× |
| t4 v5 N=8192 triton_ms | 89.76 |
| t4 v6 N=8192 triton_ms | 91.61 |
| t4 v5 N=8192 triton_mib | 28 |
| t4 v6 N=8192 triton_mib | 28 |
| rocm N=8192 reference_ms/triton_ms | 0.4× |
| rocm N=8192 reference_mib/triton_mib | 7.6× |
| max rel_err over t4 v5, t4 v6, rocm | 1.6e-5 |
<!-- /drift:gpu-claims -->

On the T4 v5 file the Triton millisecond cell is larger than the reference cell at the two shorter lengths and smaller at the two longer lengths. On the ROCm file the reference is faster at every listed length (the N=8192 time ratio is below one). Both memory columns still grow with `N`, because the activations grow. The ROCm kernel was not retuned here. The source sets `input_precision` to `ieee` on the 64 by 64 dots. This repository does not include a compiler log that isolates that flag as the cause of the ROCm times.

`bench/bench_rocm_20261031T172739Z.md` is the bench script's own dump (`nan` where dense did not run, a different relative-error format). It is not a second contract. The drift check covers the tables in this README.

## CPU oracle and sum rule (OBSERVED)

fp64 reference versus `tests/naive.py` on CPU. Seeds and shapes are in `bench/cpu_oracle.json`. This is not a GPU timing and not the GPU `rel err` column.

<!-- drift:cpu-oracle -->
| N | causal | dtype | max rel | max abs | sum abs err |
|---|---|---|---|---|---|
| 128 | False | float64 | 3.3e-16 | 1.705e-13 | 2.274e-13 |
| 128 | True | float64 | 5.4e-16 | 1.137e-13 | 2.274e-13 |
| 96 | True | float64 | 4.2e-16 | 2.274e-13 | 1.137e-13 |
<!-- /drift:cpu-oracle -->

## Causal early exit

Skipping key columns past a causal query chunk is exact in the reference (the bound is the chunk's last row, not its first). Wall clock is UNKNOWN. Value agreement on CPU fp64 is OBSERVED in `bench/cpu_early_exit.json`.

<!-- drift:cpu-early-exit -->
| N | dtype | max rel diff | max abs diff | wall clock |
|---|---|---|---|---|
| 2048 | float64 | 0 | 0.000e+00 | UNKNOWN |
| 4096 | float64 | 0 | 0.000e+00 | UNKNOWN |
<!-- /drift:cpu-early-exit -->

`test_causal_skip_is_exact` and `test_causal_skip_actually_skips` check values and that the key extent shrinks. They are not a timing.

## Policy, certificate, ledger (OBSERVED CPU)

Synthetic scores, not a model. The build is `meta.torch` in `bench/cpu_policy.json` and `bench/cpu_ledger.json`. Zipf is where the design's median error is lower than top-k. Gaussian is where top-k is lower. Both rows are in the table so a later edit cannot keep only the win.

The coverage tests assert three of these budgets (N=1024 at a quarter, N=4096 at a quarter and at a half). The other budget rows use the same helper and are not those assertions. The smallest-budget tightness at N=1024 is the row that sits above the test's cap.

<!-- drift:cpu-policy -->
| metric | value |
|---|---|
| coverage covered_frac N=1024 budget=0.125 trials=60 | 1.000 |
| coverage median_radius_over_err N=1024 budget=0.125 trials=60 | 14.01 |
| variance_share median N=1024 budget=0.125 trials=60 | 0.05 |
| coverage covered_frac N=1024 budget=0.25 trials=60 | 1.000 |
| coverage median_radius_over_err N=1024 budget=0.25 trials=60 | 5.06 |
| variance_share median N=1024 budget=0.25 trials=60 | 0.19 |
| coverage covered_frac N=1024 budget=0.5 trials=60 | 1.000 |
| coverage median_radius_over_err N=1024 budget=0.5 trials=60 | 2.25 |
| variance_share median N=1024 budget=0.5 trials=60 | 0.56 |
| coverage covered_frac N=4096 budget=0.125 trials=60 | 1.000 |
| coverage median_radius_over_err N=4096 budget=0.125 trials=60 | 8.52 |
| variance_share median N=4096 budget=0.125 trials=60 | 0.09 |
| coverage covered_frac N=4096 budget=0.25 trials=60 | 1.000 |
| coverage median_radius_over_err N=4096 budget=0.25 trials=60 | 3.17 |
| variance_share median N=4096 budget=0.25 trials=60 | 0.29 |
| coverage covered_frac N=4096 budget=0.5 trials=60 | 1.000 |
| coverage median_radius_over_err N=4096 budget=0.5 trials=60 | 1.83 |
| variance_share median N=4096 budget=0.5 trials=60 | 0.69 |
| ranking spearman N=4096 trials=80 | 0.926 |
| unbiased hajek_mean_bias N=1024 trials=800 | 0.00413 |
| unbiased topk_fixed_error N=1024 trials=800 | 0.2002 |
| unbiased topk_over_hajek_bias | 48.5× |
| regime zipf design_median_rel N=4096 trials=40 | 0.0627 |
| regime zipf topk_median_rel N=4096 trials=40 | 0.1806 |
| regime zipf topk_over_design | 2.88× |
| regime gauss design_median_rel N=4096 trials=40 | 0.8641 |
| regime gauss topk_median_rel N=4096 trials=40 | 0.6323 |
| regime gauss design_over_topk | 1.37× |
| theorem1 min_rel | 0.1485 |
| theorem1 max_rel | 1.0390 |
| theorem1 spread | 7.0× |
<!-- /drift:cpu-policy -->

Byte ledger for 28 layers, 8 heads, N=4096, head dim 128, fp16 items, block 16, 1024 kept tokens per head. `nominal_is_enforced` is computed from the ragged byte count even on the mask row. `holds_full_cache` is the flag that says the stored layout still holds the dense cache. On the mask row both are true: the choice is near the claimed budget and the bytes are not.

<!-- drift:cpu-ledger -->
| item | bytes | fraction of full | holds_full_cache | nominal_is_enforced |
|---|---|---|---|---|
| full cache | 469,762,048 | 100.00% |  |  |
| dense + per-head mask, claims 25% | 469,876,736 | 100.02% | True | True |
| ragged, 1024 kept tokens/head | 117,497,856 | 25.01% | False | True |
| mask overhead | 114,688 | 0.024% |  |  |
<!-- /drift:cpu-ledger -->

`reclaim.plan_reclaim` joins block mass, the design, and this ledger. `examples/policy_card.py` runs that chain on synthetic Q/K and prints the card. The card's stored fraction is the number to quote. A token budget can still be a full cache if the selection is a mask over a dense tensor. CI runs the example. This README does not copy the example's printed fractions.

## FAILURE_MODES

**Numerical drift.** The Triton kernel accumulates fp32 with atomic adds. The reference uses fp64 when the inputs are fp64 and fp32 otherwise. Fully masked rows are forced to zero so a NaN softmax does not leak. In the archived GPU tables the relative error at the longest `N` is larger than at the shortest `N`. The CPU sum rule is a test. The GPU sum-rule test does not run in CI. The CPU oracle table is roundoff against the dense reference on small shapes. It does not certify the GPU kernel.

**GQA.** Both entry points raise if `H_q` is not divisible by `H_kv` (`test_gqa_indivisible_raises` covers the reference). The map is `h // (H_q // H_kv)`, the same as `repeat_interleave`. A modulo map is a different, silent answer. The kernel comment records that the modulo form was wrong on a real GQA shape. `Q` and `K` are assumed to share `N`. Neither function checks `k`'s sequence length. The kernel has no dtype guard. The committed GPU benches pass fp32.

**ROCm and CUDA gaps.** Numeric kernel tests skip without a device, so a green CI run is not a GPU run. The archived ROCm table shows the kernel slower than the streaming reference at every listed `N`. The archived T4 table shows a crossover as `N` grows. No new device run was done here. There is no production autotune, no multi-GPU path, and no serving integration.

**When mass is not eviction quality.** Block mass sums attention weights inside a block. It is not a downstream task score. Broadcasting a block score onto tokens gives every token in the block the same weight. The policy table is Zipf and Gaussian noise with random values, not a language model. On the Gaussian row top-k wins. On the mask ledger row a quarter of the tokens still holds the full cache. Nothing here measures LongBench, RULER, or serving latency. A high-mass block is not, by itself, a decision to keep or drop it.

## Layout

```text
src/kvblockmass/reference.py       streaming reference
src/kvblockmass/triton_kernel.py   fused Triton kernel (CUDA / ROCm)
src/kvblockmass/evict.py           Poisson design on token scores
src/kvblockmass/certificate.py     Hajek estimate and radius
src/kvblockmass/budget.py          byte ledger
src/kvblockmass/reclaim.py         block mass to design to bytes
tests/test_reference.py            CPU oracle, sum rule, GQA, masks
tests/test_triton.py               GPU agreement, skipped without a device
tests/mutation_gate.py             four injected kernel bugs must be caught
tests/mutation_gate_evict.py       six policy mutants must fail the real suites
tests/cpu_observe.py               CPU snapshot procedures
scripts/check_readme_drift.py     README tables versus bench JSON
scripts/measure_cpu_snapshots.py   rewrite bench/cpu_*.json from a CPU run
bench/bench_block_mass.py          manual GPU bench, not CI
examples/policy_card.py            one synthetic chain, prints a card
```

## Testing

Two gates. The kernel gate compares injected bugs to the dense oracle (`python tests/mutation_gate.py`). Measured absolute errors from the snapshot that produced the JSON are below. A dropped causal mask can still look correlated. This gate does not use correlation.

<!-- drift:cpu-mutation -->
| N | bug | max abs err | status |
|---|---|---|---|
| 128 | control | 1.137e-13 | PASSED |
| 128 | CAUSAL_DROPPED | 3.516e+02 | CAUGHT |
| 128 | NO_DENOM | 4.934e+04 | CAUGHT |
| 128 | BUCKET_OFF_BY_ONE | 5.783e+02 | CAUGHT |
| 128 | EARLY_EXIT_TOO_TIGHT | 6.145e+02 | CAUGHT |
| 600 | control | 5.684e-13 | PASSED |
| 600 | CAUSAL_DROPPED | 7.483e+02 | CAUGHT |
| 600 | NO_DENOM | 2.552e+05 | CAUGHT |
| 600 | BUCKET_OFF_BY_ONE | 1.005e+03 | CAUGHT |
| 600 | EARLY_EXIT_TOO_TIGHT | 7.033e+02 | CAUGHT |
<!-- /drift:cpu-mutation -->

The policy gate reruns the real suites under mutant plugins (`python tests/mutation_gate_evict.py`): `NO_HAJEK_OFFSET`, `MISSING_VARIANCE`, `NO_TOP_SLICE`, `NO_BLOCK_TABLE`, `ALWAYS_ENFORCED`, `IGNORE_MASS`. Each must fail. A clean control must pass. `MISSING_VARIANCE` survived an earlier suite that only covered a quarter budget, which is why coverage is parametrized.

## Usage

```python
import torch
from kvblockmass import block_mass_reference
from kvblockmass.triton_kernel import block_mass_triton

q = torch.randn(1, 8, 4096, 64)
k = torch.randn(1, 2, 4096, 64)
mass = block_mass_reference(q, k, causal=True, block_size=64)  # CPU, [num_blocks]

# On a CUDA or ROCm device the same shapes can call block_mass_triton.
# This repository's CI does not execute that call.
```

## License

MIT. See [LICENSE](LICENSE).
