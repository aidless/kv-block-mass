"""CPU observations archived under bench/cpu_*.json.

Policy loops use the helpers in tests/test_certificate.py (same seeds and
trial counts). The GPU benches are not produced here. Wall-clock early-exit
speedup is not measured: the previous millisecond quotes were not in any
committed JSON, and this module does not invent a replacement timing.
"""

from __future__ import annotations

import math

import torch

from kvblockmass.budget import CacheLayout, audit_layout, bytes_full
from kvblockmass.certificate import full_cache_output, hajek_output, relative_error
from tests.mutation_gate import collect_rows
from tests.naive import block_mass_naive
from tests.test_certificate import _median, _median_error, _one_draw, _spearman


def _variance_share(cert, delta: float = 0.1) -> float:
    """Fraction of the certificate numerator that is the variance term.

    Same split as error_certificate: sqrt(2 V log(1/delta)) over that plus
    the scaled range term. This is a description of the radius we computed,
    not a second bound.
    """
    log1d = float(torch.log(torch.tensor(1.0 / delta, dtype=torch.float64)))
    var_term = math.sqrt(max(0.0, 2.0 * cert.variance * log1d))
    range_scaled = cert.range_term * log1d
    denom = var_term + range_scaled
    if denom == 0.0:
        return 0.0
    return var_term / denom


def policy_rows() -> list[dict]:
    rows: list[dict] = []
    # Coverage cases asserted by test_certificate_covers_realized_error, plus
    # the 12.5% budget the ranking loop also uses. 12.5% is not a separate
    # coverage test; it is the same _one_draw helper.
    for n, budget_frac in (
        (1024, 0.125),
        (1024, 0.25),
        (1024, 0.5),
        (4096, 0.125),
        (4096, 0.25),
        (4096, 0.5),
    ):
        trials, delta = 60, 0.1
        covered = 0
        radii: list[float] = []
        errs: list[float] = []
        shares: list[float] = []
        for t in range(trials):
            *_rest, cert, _yf, err = _one_draw(
                n, 100 + t, budget_frac=budget_frac, delta=delta
            )
            covered += int(err <= cert.radius)
            radii.append(cert.radius)
            errs.append(err)
            shares.append(_variance_share(cert, delta))
        rows.append({
            "kind": "coverage",
            "n": n,
            "budget_frac": budget_frac,
            "delta": delta,
            "trials": trials,
            "covered_frac": covered / trials,
            "median_radius_over_err": _median(
                [r / max(e, 1e-12) for r, e in zip(radii, errs)]
            ),
            "median_variance_share": _median(shares),
        })

    radii, errs = [], []
    for t in range(80):
        budget_frac = (0.125, 0.25, 0.5)[t % 3]
        *_rest, cert, _yf, err = _one_draw(4096, 500 + t, budget_frac=budget_frac)
        radii.append(cert.radius)
        errs.append(err)
    rows.append({
        "kind": "ranking",
        "n": 4096,
        "budgets": [0.125, 0.25, 0.5],
        "trials": 80,
        "spearman": _spearman(radii, errs),
    })

    # Unbiasedness: same problem, 800 designs. Copied from
    # test_design_estimator_is_unbiased_where_topk_is_pinned_to_its_bias.
    n, trials = 1024, 800
    from tests.test_certificate import _design, _zipf_scores

    g = torch.Generator().manual_seed(11)
    scores = _zipf_scores(n, g)
    values = torch.randn(n, 16, generator=g, dtype=torch.float64)
    y_full = full_cache_output(scores, values)
    tok = torch.softmax(scores, 0)
    acc = torch.zeros(16, dtype=torch.float64)
    d = None
    for t in range(trials):
        d = _design(tok, n // 4, seed=t)
        y_hat, _, _ = hajek_output(scores, values, d.keep, d.pi)
        acc += y_hat
    mean_bias = float((acc / trials - y_full).norm() / y_full.norm())
    k = int(d.keep.sum())
    keep = torch.zeros(n, dtype=torch.bool)
    keep[torch.topk(tok, k).indices] = True
    y_topk, _, _ = hajek_output(scores, values, keep, torch.ones_like(tok))
    topk_err = float((y_topk - y_full).norm() / y_full.norm())
    rows.append({
        "kind": "unbiased",
        "n": n,
        "trials": trials,
        "budget_frac": 0.25,
        "hajek_mean_bias": mean_bias,
        "topk_fixed_error": topk_err,
    })

    for kind in ("zipf", "gauss"):
        eh, et = _median_error(kind, 4096)
        rows.append({
            "kind": "regime",
            "distribution": kind,
            "n": 4096,
            "budget_frac": 0.25,
            "trials": 40,
            "design_median_rel": eh,
            "topk_median_rel": et,
        })

    # Theorem 1 loop, same constants as the test.
    n, d_head, k = 48, 6, 16
    g = torch.Generator().manual_seed(11)
    scores = torch.randn(n, generator=g, dtype=torch.float64) * 2.0
    values = torch.randn(n, d_head, generator=g, dtype=torch.float64)
    keep = torch.zeros(n, dtype=torch.bool)
    keep[torch.topk(scores, k).indices] = True
    pi = torch.ones(n, dtype=torch.float64)
    errs = []
    for c in (0.0, 0.5, 2.0, 8.0, 32.0, 512.0):
        v = values.clone()
        v[~keep] = v[~keep] + c
        y_hat, _, _ = hajek_output(scores, v, keep, pi)
        y_full_c = full_cache_output(scores, v)
        errs.append(relative_error(y_hat, y_full_c))
    spread = max(errs) / max(min(errs), 1e-12)
    rows.append({
        "kind": "theorem1",
        "n": n,
        "min_rel": min(errs),
        "max_rel": max(errs),
        "spread": spread,
    })
    return rows


def ledger_snapshot() -> dict:
    layout = {
        "n_layers": 28,
        "n_heads": 8,
        "seq_len": 4096,
        "head_dim": 128,
        "item_bytes": 2,
        "block_size": 16,
    }
    lay = CacheLayout(**layout)
    kept = 1024
    full = bytes_full(lay)
    mask = audit_layout(lay, kept, mask_stored=True)
    ragged = audit_layout(lay, kept, mask_stored=False)
    return {
        "layout": layout,
        "kept_per_head": kept,
        "bytes_full": full,
        "full_mib": full / (1024 ** 2),
        "mask_overhead_bytes": mask.bytes_as_stored - full,
        "mask": {
            "bytes": mask.bytes_as_stored,
            "fraction": mask.fraction_as_stored,
            "holds_full_cache": bool(mask.holds_full_cache),
            "nominal_is_enforced": bool(mask.nominal_is_enforced),
        },
        "ragged": {
            "bytes": ragged.bytes_as_stored,
            "fraction": ragged.fraction_as_stored,
            "holds_full_cache": bool(ragged.holds_full_cache),
            "nominal_is_enforced": bool(ragged.nominal_is_enforced),
        },
    }


def oracle_rows() -> list[dict]:
    """fp64 reference versus the dense oracle. Not the GPU rel_err column."""
    from kvblockmass.reference import block_mass_reference

    specs = [
        dict(B=2, H_q=4, H_kv=2, N=128, D=32, seed=7, causal=False, block_size=64),
        dict(B=2, H_q=4, H_kv=2, N=128, D=32, seed=7, causal=True, block_size=64),
        dict(B=1, H_q=8, H_kv=2, N=96, D=16, seed=3, causal=True, block_size=32),
    ]
    rows = []
    for s in specs:
        g = torch.Generator().manual_seed(s["seed"])
        q = torch.randn(s["B"], s["H_q"], s["N"], s["D"], generator=g, dtype=torch.float64)
        k = torch.randn(s["B"], s["H_kv"], s["N"], s["D"], generator=g, dtype=torch.float64)
        ref = block_mass_reference(q, k, causal=s["causal"], block_size=s["block_size"])
        nv = block_mass_naive(q, k, causal=s["causal"], block_size=s["block_size"])
        rows.append({
            **s,
            "dtype": "float64",
            "max_abs": (ref - nv).abs().max().item(),
            "max_rel": ((ref - nv).abs() / nv.abs().clamp_min(1e-12)).max().item(),
            "sum_abs_err": abs(ref.sum().item() - s["B"] * s["H_q"] * s["N"]),
        })
    return rows


def early_exit_rows() -> list[dict]:
    """Value exactness only. wall_clock is the string UNKNOWN on purpose."""
    from kvblockmass.reference import block_mass_reference

    rows = []
    for N in (2048, 4096):
        g = torch.Generator().manual_seed(0)
        q = torch.randn(1, 8, N, 64, generator=g, dtype=torch.float64)
        k = torch.randn(1, 2, N, 64, generator=g, dtype=torch.float64)
        skip = block_mass_reference(
            q, k, causal=True, block_size=64, query_chunk=256, causal_skip=True
        )
        full = block_mass_reference(
            q, k, causal=True, block_size=64, query_chunk=256, causal_skip=False
        )
        rows.append({
            "N": N,
            "B": 1,
            "H_q": 8,
            "H_kv": 2,
            "D": 64,
            "block_size": 64,
            "query_chunk": 256,
            "dtype": "float64",
            "causal": True,
            "seed": 0,
            "max_abs_diff": (skip - full).abs().max().item(),
            "max_rel_diff": (
                (skip - full).abs() / full.abs().clamp_min(1e-12)
            ).max().item(),
            "wall_clock": "UNKNOWN",
        })
    return rows


def mutation_rows() -> tuple[list[dict], bool]:
    return collect_rows()
