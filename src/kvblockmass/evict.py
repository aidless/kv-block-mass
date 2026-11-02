"""Eviction designs with a known sampling probability (design-based, not heuristic).

Why this module exists
----------------------
A deterministic top-k eviction deletes tokens and then *cannot say what the
deletion cost*. Formally (Xie & Alanwar, arXiv:2607.21475, Thm 1): for any
estimator computable from what a deterministic, value-blind scheme retains,
two cache configurations exist that produce **identical** retained information
and arbitrarily different true errors. Keeping an eviction *design* — the
inclusion probabilities — is what makes the error identifiable.

This module builds that design from block/token importance scores:

* certainty set ``A``: score top slice + attention sinks + recency window, kept
  for sure (``pi = 1``);
* tail ``T``: Poisson sampling with
  ``pi_i = clip(m * score_i / sum_{j in T} score_j, eps, 1)``,
  so the expected number of retained tail tokens is ``m``;
* the Hájek correction is one line at use time — add ``log(1/pi_i)`` to the
  retained logits and let the softmax denominator renormalize
  (``hajek_log_offsets``).

Nothing here touches the attention kernel: importance scores are exactly what
``block_mass_reference`` / the Triton kernel already return, which is the point
— the design consumes a score tensor, not the attention matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "EvictionDesign",
    "design_from_scores",
    "expand_block_scores",
    "hajek_log_offsets",
]


@dataclass
class EvictionDesign:
    """A realized eviction: which tokens are kept, and with what probability."""

    keep: torch.Tensor          # bool [N] — realized retained set S
    pi: torch.Tensor            # float [N] — inclusion probabilities, 1 on certainty set
    certainty: torch.Tensor     # bool [N] — A (pi == 1 by construction)
    tail_budget: int            # m: expected number of retained tail tokens
    eps: float                  # floor on tail probabilities

    @property
    def tail(self) -> torch.Tensor:
        return ~self.certainty

    def n_expected(self) -> float:
        """Expected retention count under the design (not the realized count)."""
        return float(self.pi.sum())

    def n_kept(self) -> int:
        return int(self.keep.sum())


def expand_block_scores(
    block_scores: torch.Tensor,   # [num_blocks]
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    """Broadcast per-block scores to per-token scores [N].

    Block-granular scoring is what makes an O(N)-memory pass possible; a
    token-level design needs a token-level score, so block mass is broadcast to
    its members. Ties inside a block are broken by sampling, not by position.
    """
    if block_scores.dim() != 1:
        raise ValueError(f"block_scores must be 1-D, got {tuple(block_scores.shape)}")
    num_blocks = block_scores.numel()
    if num_blocks * block_size < seq_len:
        raise ValueError(
            f"{num_blocks} blocks of {block_size} cannot cover seq_len={seq_len}"
        )
    idx = torch.arange(seq_len) // block_size
    return block_scores.to(torch.float64)[idx]


def _top_slice(scores: torch.Tensor, n: int) -> torch.Tensor:
    """Boolean mask of the ``n`` highest scores (ties broken by index)."""
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if n > 0:
        top = torch.topk(scores, k=min(n, scores.numel())).indices
        mask[top] = True
    return mask


def design_from_scores(
    scores: torch.Tensor,        # [N] token importance, non-negative
    tail_budget: int,
    *,
    n_sinks: int = 4,
    recency: int = 32,
    top_slice: int = 0,
    eps: float = 1e-6,
    generator: torch.Generator | None = None,
    uniform_tail: bool = False,
) -> EvictionDesign:
    """Build a certainty-set + Poisson-tail eviction design.

    Args:
        scores: per-token importance, typically ``expand_block_scores(mass)``.
        tail_budget: expected number of retained tail tokens (``m``).
        n_sinks: number of leading tokens kept for sure (attention sinks).
        recency: number of trailing tokens kept for sure.
        top_slice: extra highest-score tokens forced into the certainty set.
        eps: floor on tail inclusion probabilities (must be > 0).
        generator: torch RNG; pass a seeded generator for reproducible designs.
        uniform_tail: ignore ``scores`` on the tail and sample uniformly — the
            null baseline that arXiv:2609.03430 argues is hard to beat.

    The certainty set is what keeps Assumption 2 (bounded weights) satisfiable:
    it absorbs the head of the score distribution, so the tail cannot dominate
    a single token's weight.
    """
    if scores.dim() != 1:
        raise ValueError(f"scores must be 1-D, got {tuple(scores.shape)}")
    if eps <= 0:
        raise ValueError("eps must be > 0 (the design needs a positive floor)")
    if tail_budget < 0:
        raise ValueError("tail_budget must be >= 0")
    s = scores.to(torch.float64)
    if torch.any(s < 0):
        raise ValueError("scores must be non-negative")
    n = s.numel()

    certainty = torch.zeros(n, dtype=torch.bool)
    if n_sinks:
        certainty[: min(n_sinks, n)] = True
    if recency:
        certainty[max(0, n - recency) :] = True
    if top_slice:
        certainty |= _top_slice(s, top_slice)

    tail_idx = (~certainty).nonzero(as_tuple=True)[0]
    pi = torch.ones(n, dtype=torch.float64)

    if tail_idx.numel() and tail_budget > 0:
        tail_scores = s[tail_idx]
        if uniform_tail:
            weights = torch.ones_like(tail_scores)
        else:
            weights = tail_scores
        total = weights.sum()
        if total <= 0:
            # No signal at all: fall back to a uniform design rather than
            # dividing by zero. This is exactly the regime where the *score*
            # carries no information, so uniform is the honest design.
            weights = torch.ones_like(tail_scores)
            total = weights.sum()
        pi_tail = torch.clamp(tail_budget * weights / total, eps, 1.0)
        pi[tail_idx] = pi_tail
    elif tail_idx.numel():
        pi[tail_idx] = eps

    keep = certainty.clone()
    if tail_idx.numel():
        u = torch.rand(tail_idx.numel(), dtype=torch.float64, generator=generator)
        keep[tail_idx] = u < pi[tail_idx]

    return EvictionDesign(
        keep=keep,
        pi=pi,
        certainty=certainty,
        tail_budget=int(tail_budget),
        eps=float(eps),
    )


def hajek_log_offsets(pi: torch.Tensor) -> torch.Tensor:
    """``log(1/pi)`` — add this to retained logits and softmax does the rest.

    This is the Hájek correction expressed as a logit offset: the softmax
    denominator over the retained set renormalizes automatically, so no separate
    division pass is needed and the correction costs one scalar per tail token.
    """
    pi = pi.to(torch.float64)
    if torch.any(pi <= 0):
        raise ValueError("inclusion probabilities must be > 0")
    return -torch.log(pi)
