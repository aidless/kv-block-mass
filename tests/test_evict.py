"""Tests for the eviction design (design-based sampling, not heuristics).

Every assertion here is about the *design*, which is what makes a later error
certificate valid: probabilities bounded and stored, certainty set deterministic,
the Hájek offset exact when nothing is evicted, and the sampler reproducible
from a seed.
"""

import math

import pytest
import torch

from kvblockmass.evict import (
    EvictionDesign,
    design_from_scores,
    expand_block_scores,
    hajek_log_offsets,
)


def _scores(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, generator=g, dtype=torch.float64)


def test_design_bounds_and_certainty_set():
    s = _scores(64, seed=1)
    d = design_from_scores(s, tail_budget=8, n_sinks=2, recency=8, generator=torch.Generator().manual_seed(0))
    assert isinstance(d, EvictionDesign)
    # certainty tokens are kept for sure
    assert torch.equal(d.pi[d.certainty], torch.ones(int(d.certainty.sum()), dtype=torch.float64))
    assert bool(d.keep[d.certainty].all())
    # tail probabilities respect the floor and never exceed 1
    tail_pi = d.pi[d.tail]
    assert float(tail_pi.min()) >= d.eps
    assert float(tail_pi.max()) <= 1.0
    # sinks (leading) and recency (trailing) are inside the certainty set
    assert bool(d.certainty[:2].all())
    assert bool(d.certainty[-8:].all())


def test_expected_tail_retention_matches_budget():
    """The budget is an *expectation*: average realized retention must track m."""
    n, m, trials = 256, 20, 300
    kept = []
    for t in range(trials):
        s = _scores(n, seed=t)
        d = design_from_scores(
            s, tail_budget=m, n_sinks=2, recency=4,
            generator=torch.Generator().manual_seed(1000 + t),
        )
        kept.append(int(d.keep.sum()) - int(d.certainty.sum()))
    mean = sum(kept) / len(kept)
    # Poisson(m) has sd sqrt(m); allow 4 sigma on the mean of `trials` draws
    sd = math.sqrt(m / trials)
    assert abs(mean - m) <= 4 * sd + 1e-9, f"mean tail kept {mean:.2f} vs m={m} (sd {sd:.2f})"


def test_hajek_offset_is_log_inverse_probability():
    pi = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64)
    off = hajek_log_offsets(pi)
    assert torch.allclose(off, torch.tensor([0.0, math.log(2), math.log(4)], dtype=torch.float64))


def test_prefix_equal_to_capped_log_weights():
    """pi = min(1, m * w / sum w): the clip is what bounds expected retention."""
    w = torch.tensor([4.0, 1.0, 1.0], dtype=torch.float64)
    d = design_from_scores(
        w, tail_budget=2, n_sinks=0, recency=0, generator=torch.Generator().manual_seed(0)
    )
    # sum(w)=6, m=2 -> [1.333, 0.333, 0.333] clipped to [1, .333, .333]
    assert torch.allclose(d.pi, torch.tensor([1.0, 1 / 3, 1 / 3], dtype=torch.float64), atol=1e-12)


def test_uniform_tail_ignores_scores():
    s = torch.tensor([10.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    d = design_from_scores(
        s, tail_budget=1, n_sinks=0, recency=0, uniform_tail=True,
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.allclose(d.pi, torch.full((4,), 0.25, dtype=torch.float64))


def test_zero_score_tail_falls_back_to_uniform_design():
    s = torch.zeros(16, dtype=torch.float64)
    d = design_from_scores(s, tail_budget=4, n_sinks=1, recency=1)
    tail_pi = d.pi[d.tail]
    assert torch.all(tail_pi > 0)
    assert torch.all(tail_pi <= 1)


def test_design_is_reproducible_from_generator():
    s = _scores(32, seed=7)
    a = design_from_scores(s, 6, n_sinks=1, recency=2, generator=torch.Generator().manual_seed(42))
    b = design_from_scores(s, 6, n_sinks=1, recency=2, generator=torch.Generator().manual_seed(42))
    assert torch.equal(a.keep, b.keep)
    assert torch.equal(a.pi, b.pi)


def test_expand_block_scores_covers_every_token():
    bs = torch.tensor([1.0, 2.0], dtype=torch.float64)
    tok = expand_block_scores(bs, seq_len=5, block_size=3)
    assert torch.equal(tok, torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0], dtype=torch.float64))
    with pytest.raises(ValueError):
        expand_block_scores(bs, seq_len=9, block_size=3)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        design_from_scores(torch.tensor([1.0]), 1, eps=0.0)
    with pytest.raises(ValueError):
        design_from_scores(torch.tensor([-1.0, 2.0]), 1)
