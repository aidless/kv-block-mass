"""Per-step error certificate for an eviction design (survey-sampling variance).

What it computes
----------------
For one attention head at one step, given the retained set ``S`` and the design
probabilities ``pi``, the Hájek estimate of the full-cache output is

    y_hat = sum_{i in S} (a_i / pi_i) v_i / sum_{i in S} (a_i / pi_i)

with ``a_i = exp(s_i)`` the (offset-corrected) attention weights. The realized
error you cannot see is ``||y_hat - y_full||``; the certificate estimates it
from ``S`` alone:

    N_hat = sum_{i in S} a_i / pi_i
    V_hat = 1/N_hat^2 * sum_{i in S n T} (1 - pi_i)/pi_i * a_i^2 * ||v_i - y_hat||^2
    b     = max_{i in S n T} (1 - pi_i)/pi_i^2 * a_i * ||v_i - y_hat|| / N_hat
    r     = ( sqrt(2 * V_hat * log(1/delta)) + b * log(1/delta) ) / (||y_hat|| + eps0)

``r`` is the empirical-Bernstein shape of Xie & Alanwar (arXiv:2607.21475,
eq. 4), instantiated with the design's variance estimator (their Thm 2: under
Poisson sampling the Sen-Yates-Grundy double sum collapses to the single sum
above, so the cost is O(|S n T|) per head per step).

Two honest caveats, kept in code rather than in a footnote:

* The textbook empirical-Bernstein range term carries a ``1/3``. The source
  states the shape above; we implement the shape **as stated**, which makes the
  radius conservative by that factor when the range term dominates. Our
  coverage test measures what we actually get instead of assuming it.
* The variance estimator targets the *linearized* error; the gap to the true
  ratio error is ``O(B^2/m^2)`` per the source. Measured coverage, not theory,
  is the acceptance criterion here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "Certificate",
    "hajek_output",
    "error_certificate",
    "relative_error",
    "full_cache_output",
]


@dataclass
class Certificate:
    y_hat: torch.Tensor      # [D] Hájek estimate of the head output
    variance: float          # V_hat
    range_term: float        # b
    radius: float            # r: relative-error certificate at confidence 1-delta
    n_hat: float             # N_hat

    @property
    def relative(self) -> float:
        """Certificate as a fraction of ``||y_hat||`` (the radius already is)."""
        return self.radius


def full_cache_output(scores: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Dense reference: softmax over *all* keys (the quantity eviction estimates)."""
    w = torch.softmax(scores.to(torch.float64), dim=0)
    return w @ values.to(torch.float64)


def _weights(scores: torch.Tensor) -> torch.Tensor:
    return torch.exp(scores.to(torch.float64) - scores.to(torch.float64).max())


def hajek_output(
    scores: torch.Tensor,        # [N] logits (already offset-corrected if desired)
    values: torch.Tensor,        # [N, D]
    keep: torch.Tensor,          # [N] bool
    pi: torch.Tensor,            # [N] float
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """Hájek estimate over the retained set. Returns (y_hat, N_hat, a_keep).

    ``a_keep`` are the un-normalized retained weights; the caller reuses them so
    no second exp() pass is needed.
    """
    s = scores.to(torch.float64)
    v = values.to(torch.float64)
    p = pi.to(torch.float64)
    kept = keep.to(torch.bool)
    if not bool(kept.any()):
        raise ValueError("retained set is empty — no estimate is computable")
    a = torch.zeros_like(s)
    a[kept] = torch.exp(s[kept] - s[kept].max())
    weighted = a[kept] / p[kept]
    n_hat = float(weighted.sum())
    y_hat = (weighted[:, None] * v[kept]).sum(0) / n_hat
    return y_hat, n_hat, a


def error_certificate(
    scores: torch.Tensor,        # [N] logits
    values: torch.Tensor,        # [N, D]
    design_keep: torch.Tensor,   # [N] bool — realized retained set S
    design_pi: torch.Tensor,     # [N] float — inclusion probabilities
    certainty: torch.Tensor,     # [N] bool — A (pi == 1)
    delta: float = 0.1,
    eps0: float = 1e-12,
) -> Certificate:
    """Compute the relative-error certificate for one realized design.

    Only retained information is read: ``scores``/``values`` are indexed by the
    retained set, and ``pi`` is the design (which the mechanism stored). The
    true error needs ``y_full`` and is *not* an input.
    """
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    kept = design_keep.to(torch.bool)
    y_hat, n_hat, a = hajek_output(scores, values, kept, design_pi)

    p = design_pi.to(torch.float64)
    v = values.to(torch.float64)
    tail = (~certainty.to(torch.bool)) & kept

    if bool(tail.any()):
        ai = a[tail]
        pii = p[tail]
        dev = (v[tail] - y_hat).pow(2).sum(-1)          # ||v_i - y_hat||^2
        norms = (v[tail] - y_hat).norm(dim=-1)          # ||v_i - y_hat||
        variance = float(((1 - pii) / pii * ai.pow(2) * dev).sum() / n_hat**2)
        range_term = float(
            (((1 - pii) / pii.pow(2)) * ai * norms).max() / n_hat
        )
    else:
        variance = 0.0
        range_term = 0.0

    log1d = float(torch.log(torch.tensor(1.0 / delta, dtype=torch.float64)))
    radius = (float((2.0 * variance * log1d) ** 0.5) + range_term * log1d) / (
        float(y_hat.norm()) + eps0
    )
    return Certificate(
        y_hat=y_hat,
        variance=variance,
        range_term=range_term,
        radius=float(radius),
        n_hat=n_hat,
    )


def relative_error(y_hat: torch.Tensor, y_full: torch.Tensor, eps0: float = 1e-12) -> float:
    """Realized relative attention-output error ``||y_hat - y_full|| / ||y_full||``."""
    yf = y_full.to(torch.float64)
    return float((y_hat - yf).norm() / (yf.norm() + eps0))
