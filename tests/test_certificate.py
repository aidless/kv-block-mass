"""Tests for the error certificate.

What is asserted, in order of how easy each is to fake:

1. **Coverage** — the radius contains the realized error at (at least) the stated
   rate. Measured over draws, and the measured numbers are printed so the README
   cannot drift away from the test.
2. **Tightness** — a certificate that is always covered but 1000x loose is
   decoration; the radius/error ratio is bounded.
3. **Ranking** — the radius orders errors (Spearman), i.e. it is a usable risk
   signal even when loose.
4. **Unbiasedness of the design estimator, and the bias of top-k** — with the
   *same problem* and many designs, the Hájek mean lands on the truth while
   top-k's fixed error stays put and invisible. This is why the design exists.
5. **The regime boundary, asserted in both directions** — the design wins on
   heavy-tailed (power-law) score distributions and *loses* on near-Gaussian
   ones. The losing side is asserted too: a suite that only records wins would
   let a README claim a general improvement that this harness contradicts.
6. **Unidentifiability of deterministic schemes** — the source's Theorem 1 as an
   executable test: evicted values can be perturbed so that every retained
   statistic is bit-identical while the true error moves a lot.
"""

import math

import pytest
import torch

from kvblockmass.certificate import (
    error_certificate,
    full_cache_output,
    hajek_output,
    relative_error,
)
from kvblockmass.evict import design_from_scores

D_HEAD = 16


def _zipf_scores(n, g):
    """Heavy-tailed logits, permuted so the heavy hitters are not positional."""
    w = 1.0 / torch.arange(1, n + 1, dtype=torch.float64)
    perm = torch.randperm(n, generator=g)
    return torch.log(w[perm] * n)


def _gauss_scores(n, g):
    return torch.randn(n, generator=g, dtype=torch.float64)


def _median(xs):
    return float(torch.tensor(xs, dtype=torch.float64).median())


def _spearman(a, b):
    ra = torch.tensor(a).argsort().argsort().to(torch.float64)
    rb = torch.tensor(b).argsort().argsort().to(torch.float64)
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean()) / rb.std()
    return float((ra * rb).mean())


def _design(tok, budget, seed, *, n_sinks=4, recency=32, top_slice=None):
    """Half the budget in the certainty top slice, half in the Poisson tail.

    This is the source's shape (certainty = top slice + sinks + recency). Without
    the top slice the dominant tokens get sampled and the variance explodes on
    every distribution we tried.
    """
    top = budget // 2 if top_slice is None else top_slice
    tail_m = max(1, budget - top)
    return design_from_scores(
        tok, tail_m, n_sinks=n_sinks, recency=recency, top_slice=top,
        generator=torch.Generator().manual_seed(seed),
    )


def _one_draw(n, seed, *, kind="zipf", budget_frac=0.25, delta=0.1):
    g = torch.Generator().manual_seed(seed)
    scores = _zipf_scores(n, g) if kind == "zipf" else _gauss_scores(n, g)
    values = torch.randn(n, D_HEAD, generator=g, dtype=torch.float64)
    tok = torch.softmax(scores, 0)
    d = _design(tok, int(n * budget_frac), seed=9_000 + seed)
    cert = error_certificate(scores, values, d.keep, d.pi, d.certainty, delta=delta)
    y_full = full_cache_output(scores, values)
    return scores, values, tok, d, cert, y_full, relative_error(cert.y_hat, y_full)


# --------------------------------------------------------------------------- #
# 1-3: coverage, tightness, ranking
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n,budget_frac", [(1024, 0.25), (4096, 0.25), (4096, 0.5)])
def test_certificate_covers_realized_error(n, budget_frac):
    """Coverage at more than one budget: the variance term carries more of the
    radius as the budget grows. Medians are archived in bench/cpu_policy.json
    (drift-checked from the README) and are not re-typed in this comment. A
    suite that only tests one budget cannot tell whether the variance term is
    load-bearing at all."""
    trials, delta = 60, 0.1
    covered, radii, errs = 0, [], []
    for t in range(trials):
        *_rest, cert, _yf, err = _one_draw(n, 100 + t, budget_frac=budget_frac, delta=delta)
        covered += int(err <= cert.radius)
        radii.append(cert.radius)
        errs.append(err)
    frac = covered / trials
    ratio = _median([r / max(e, 1e-12) for r, e in zip(radii, errs)])
    print(f"[coverage] N={n} budget={budget_frac:.3f} delta={delta} "
          f"covered={frac:.3f} median(radius/err)={ratio:.2f}")
    assert frac >= 1 - delta, f"coverage {frac:.3f} below stated {1 - delta}"
    # a certificate that is vacuously loose is not a certificate
    assert ratio <= 10.0, f"radius is {ratio:.1f}x the realized error — too loose to be useful"


def test_certificate_ranks_realized_error():
    """Loose-but-ordered is the useful regime; unordered is not."""
    radii, errs = [], []
    for t in range(80):
        budget_frac = (0.125, 0.25, 0.5)[t % 3]
        *_rest, cert, _yf, err = _one_draw(4096, 500 + t, budget_frac=budget_frac)
        radii.append(cert.radius)
        errs.append(err)
    rho = _spearman(radii, errs)
    print(f"[ranking] spearman(radius, realized error) = {rho:.3f}")
    assert rho > 0.5


# --------------------------------------------------------------------------- #
# 4: unbiasedness (the reason the design exists)
# --------------------------------------------------------------------------- #

def test_design_estimator_is_unbiased_where_topk_is_pinned_to_its_bias():
    """Same problem, many designs: Hájek's mean lands on the truth; top-k cannot move.

    This is the claim that survives the regime-dependence in §5 — unbiasedness
    plus a computable radius is what a heuristic cannot have.
    """
    n, trials = 1024, 800
    g = torch.Generator().manual_seed(11)
    scores = _zipf_scores(n, g)
    values = torch.randn(n, D_HEAD, generator=g, dtype=torch.float64)
    y_full = full_cache_output(scores, values)
    tok = torch.softmax(scores, 0)

    acc = torch.zeros(D_HEAD, dtype=torch.float64)
    d = None
    for t in range(trials):
        d = _design(tok, n // 4, seed=t)
        y_hat, _, _ = hajek_output(scores, values, d.keep, d.pi)
        acc += y_hat
    mean_bias = float((acc / trials - y_full).norm() / y_full.norm())

    # deterministic top-k at the same retained count: one fixed answer, no design
    k = int(d.keep.sum())
    keep = torch.zeros(n, dtype=torch.bool)
    keep[torch.topk(tok, k).indices] = True
    y_topk, _, _ = hajek_output(scores, values, keep, torch.ones_like(tok))
    topk_err = float((y_topk - y_full).norm() / y_full.norm())

    print(f"[unbiased] hajek mean bias over {trials} designs = {mean_bias:.5f}; "
          f"top-k fixed error = {topk_err:.4f}")
    assert mean_bias < 0.01, f"Hájek mean over designs is biased by {mean_bias:.4f}"
    assert topk_err > 5 * mean_bias, "top-k should be farther from the truth than the design's mean"


# --------------------------------------------------------------------------- #
# 5: the regime boundary, asserted in both directions
# --------------------------------------------------------------------------- #

def _median_error(kind, n, trials=40, budget_frac=0.25):
    """Median relative error of (design+Hájek) vs (top-k at the same count)."""
    eh, et = [], []
    for t in range(trials):
        scores, values, tok, d, _cert, y_full, err = _one_draw(
            n, 300 + t, kind=kind, budget_frac=budget_frac
        )
        eh.append(err)
        k = int(d.keep.sum())
        keep = torch.zeros(n, dtype=torch.bool)
        keep[torch.topk(tok, k).indices] = True
        y_topk, _, _ = hajek_output(scores, values, keep, torch.ones_like(tok))
        et.append(relative_error(y_topk, y_full))
    return _median(eh), _median(et)


def test_design_wins_on_heavy_tailed_scores():
    n = 4096
    eh, et = _median_error("zipf", n)
    print(f"[regime] power-law N={n}: design={eh:.4f} top-k={et:.4f} ({et / eh:.2f}x)")
    assert eh < et, "on power-law scores the design must beat top-k at a matched count"


def test_topk_wins_on_near_gaussian_scores():
    """Negative control, asserted: the design's advantage is NOT universal.

    On near-Gaussian score distributions top-k's truncation bias is smaller than
    the design's sampling variance, so top-k wins — and the README says so. This
    test exists so nobody re-introduces a general "beats top-k" claim.
    """
    n = 4096
    eh, et = _median_error("gauss", n)
    print(f"[regime] near-Gaussian N={n}: design={eh:.4f} top-k={et:.4f} ({eh / et:.2f}x worse)")
    assert et < eh, "top-k is expected to win here; if it stops winning, re-measure the story"


# --------------------------------------------------------------------------- #
# 6: Theorem 1 as an executable test
# --------------------------------------------------------------------------- #

def test_deterministic_retained_set_cannot_identify_its_own_error():
    """Identical retained information, divergent true error.

    A deterministic value-blind scheme keeps a fixed set S. Perturb the *evicted*
    values only: S, the retained scores, the retained values and pi are all
    bit-identical, so no estimator computable from them can move — while the true
    error does.
    """
    n, d, k = 48, 6, 16
    g = torch.Generator().manual_seed(11)
    scores = torch.randn(n, generator=g, dtype=torch.float64) * 2.0
    values = torch.randn(n, d, generator=g, dtype=torch.float64)
    keep = torch.zeros(n, dtype=torch.bool)
    keep[torch.topk(scores, k).indices] = True
    pi = torch.ones(n, dtype=torch.float64)   # deterministic scheme: no design stored

    errs, first = [], None
    # The spread saturates: as c grows the evicted values dominate the true
    # output. The archived min, max, and spread are bench/cpu_policy.json
    # (theorem1). The claim checked here is a move of more than 5x with
    # bit-identical retained information.
    for c in (0.0, 0.5, 2.0, 8.0, 32.0, 512.0):
        v = values.clone()
        v[~keep] = v[~keep] + c                # evicted values only
        y_hat, _, _ = hajek_output(scores, v, keep, pi)
        # the true full-cache output of *this* configuration (the evicted values
        # still exist in the dense reference; the scheme just cannot see them)
        y_full_c = full_cache_output(scores, v)
        errs.append(relative_error(y_hat, y_full_c))
        fingerprint = (scores[keep].clone(), v[keep].clone(), keep.clone(), pi.clone())
        if first is None:
            first = fingerprint
        else:
            for a, b in zip(first, fingerprint):
                assert torch.equal(a, b), "retained information changed — the test is void"

    spread = max(errs) / max(min(errs), 1e-12)
    print(f"[impossibility] realized error {min(errs):.4f} -> {max(errs):.4f} (x{spread:.1f})"
          f" with bit-identical retained statistics")
    assert spread > 5.0, "perturbing evicted values should move the error by >5x"


# --------------------------------------------------------------------------- #
# 7: input contracts
# --------------------------------------------------------------------------- #

def test_empty_retained_set_raises():
    scores = torch.randn(8, dtype=torch.float64)
    values = torch.randn(8, 4, dtype=torch.float64)
    keep = torch.zeros(8, dtype=torch.bool)
    with pytest.raises(ValueError):
        error_certificate(scores, values, keep, torch.ones(8, dtype=torch.float64),
                          torch.ones(8, dtype=torch.bool))


def test_bad_delta_raises():
    scores = torch.randn(8, dtype=torch.float64)
    values = torch.randn(8, 4, dtype=torch.float64)
    keep = torch.ones(8, dtype=torch.bool)
    with pytest.raises(ValueError):
        error_certificate(scores, values, keep, torch.ones(8, dtype=torch.float64),
                          torch.zeros(8, dtype=torch.bool), delta=1.5)


def test_certainty_tokens_carry_no_variance_term():
    """pi == 1 tokens contribute zero variance and zero range by construction."""
    scores = torch.randn(64, dtype=torch.float64)
    values = torch.randn(64, 4, dtype=torch.float64)
    keep = torch.ones(64, dtype=torch.bool)
    pi = torch.ones(64, dtype=torch.float64)
    cert = error_certificate(scores, values, keep, pi, torch.ones(64, dtype=torch.bool))
    assert cert.variance == 0.0 and cert.range_term == 0.0
    assert cert.radius == 0.0
    assert math.isclose(relative_error(cert.y_hat, full_cache_output(scores, values)), 0.0, abs_tol=1e-12)
