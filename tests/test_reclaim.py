"""Tests for the glue: block mass -> design -> certificate preconditions -> bytes.

Each layer is tested on its own; this file tests that they compose, because the
composition is where a "policy" actually lives. The claim being pinned down:
given per-block attention mass, a **token-level sampling design at a stated byte
budget** is produced, and the ledger reports what that design holds.
"""

import json

import pytest
import torch

from kvblockmass.budget import CacheLayout, audit_layout
from kvblockmass.reclaim import ReclaimPlan, plan_reclaim
from kvblockmass.reference import block_mass_reference
from kvblockmass.evict import design_from_scores


def _layout(**kw):
    base = dict(n_layers=4, n_heads=8, seq_len=256, head_dim=32, item_bytes=2, block_size=16)
    base.update(kw)
    return CacheLayout(**base)


def _mass(n=256, block=16, seed=0, head=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, 8, n, 32, generator=g, dtype=torch.float64)
    k = torch.randn(1, 2, n, 32, generator=g, dtype=torch.float64)
    m = block_mass_reference(q, k, causal=True, block_size=block)
    return m[head] if m.dim() == 2 else m


def test_plan_uses_the_kernels_block_mass_and_states_its_bytes():
    mass = _mass()
    plan = plan_reclaim(
        mass, block_size=16, layout=_layout(), tail_budget=48,
        n_sinks=4, recency=8, top_slice=16, seed=0,
    )
    assert isinstance(plan, ReclaimPlan)
    # the design is token-level and consistent with the plan
    assert plan.design.keep.numel() == 256
    assert int(plan.design.keep.sum()) == plan.kept_tokens
    # the ledger reports the same kept count the design realized
    assert plan.audit.kept_per_head == pytest.approx(plan.kept_tokens)
    assert 0.0 < plan.audit.fraction_enforced < 1.0
    assert plan.audit.fraction_as_stored > plan.audit.fraction_enforced


def test_block_level_retention_is_monotone_in_budget():
    """More tail budget must never mean fewer kept tokens."""
    mass = _mass(seed=1)
    kept = []
    for budget in (16, 32, 64, 128):
        plan = plan_reclaim(mass, 16, _layout(), budget, n_sinks=4, recency=8,
                            top_slice=8, seed=2)
        kept.append(plan.kept_tokens)
    assert kept == sorted(kept), kept


def test_plan_drops_blocks_the_mass_says_are_empty():
    """A block whose total mass is ~0 must not be kept more often than its mass share.

    Uniform mass -> every block is equally likely; concentrated mass -> the empty
    half of the sequence is mostly dropped. This test pins the second case.
    """
    n, bs = 256, 16
    mass = torch.zeros(n // bs, dtype=torch.float64)
    mass[: n // bs // 2] = 1.0        # all mass in the first half of the sequence
    plan = plan_reclaim(mass, bs, _layout(), tail_budget=32, n_sinks=4, recency=8,
                        top_slice=8, seed=3)
    # recency window is positional and must still be kept
    assert bool(plan.design.keep[-8:].all())
    blocks_kept = plan.design.keep.reshape(-1, bs).any(-1)
    empty_half = blocks_kept[n // bs // 2:]
    assert int(empty_half.sum()) <= 1, "empty-mass blocks were kept — mass is not driving the design"


def test_card_is_json_ready_and_carries_the_verdict_words():
    plan = plan_reclaim(_mass(seed=4), 16, _layout(), 64, n_sinks=4, recency=8,
                        top_slice=16, seed=5)
    card = plan.card
    json.dumps(card)                       # numpy types would raise here
    for key in ("kept_tokens", "kept_fraction", "bytes_full", "bytes_as_stored",
                "bytes_enforced", "holds_full_cache", "nominal_is_enforced",
                "blocks_total", "blocks_kept", "tail_budget"):
        assert key in card, key
    assert card["kept_tokens"] == plan.kept_tokens


def test_plan_is_reproducible_from_seed():
    mass = _mass(seed=6)
    a = plan_reclaim(mass, 16, _layout(), 48, n_sinks=4, recency=8, top_slice=8, seed=7)
    b = plan_reclaim(mass, 16, _layout(), 48, n_sinks=4, recency=8, top_slice=8, seed=7)
    assert torch.equal(a.design.keep, b.design.keep)
    assert a.card == b.card


def test_budget_larger_than_sequence_is_clamped_not_an_error():
    plan = plan_reclaim(_mass(seed=8), 16, _layout(), 10_000, n_sinks=4, recency=8,
                        top_slice=16, seed=9)
    assert plan.kept_tokens <= 256
    assert plan.audit.fraction_enforced <= 1.01


def test_shape_mismatch_is_rejected():
    mass = torch.ones(4, dtype=torch.float64)      # 4 blocks * 16 != 256 seq_len
    with pytest.raises(ValueError):
        plan_reclaim(mass, 16, _layout(), 32, n_sinks=4, recency=8, top_slice=8)
