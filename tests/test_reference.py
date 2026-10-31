"""Reference (O(N) memory) implementation tests. RED before reference.py exists."""

import math

import pytest
import torch

from tests.naive import block_mass_naive


def _qk(B=2, H_q=4, H_kv=2, N=128, D=32, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H_q, N, D, generator=g, dtype=dtype)
    k = torch.randn(B, H_kv, N, D, generator=g, dtype=dtype)
    return q, k


def _num_blocks(N, bs):
    return (N + bs - 1) // bs


# ---------------------------------------------------------------- sum rule
@pytest.mark.parametrize("causal", [False, True])
def test_sum_rule_full_mass_equals_query_count(causal):
    """Every unmasked query position distributes exactly 1.0 of mass over keys."""
    from kvblockmass.reference import block_mass_reference

    B, H_q, N = 2, 4, 128
    q, k = _qk(B=B, H_q=H_q, N=N)
    mass = block_mass_reference(q, k, causal=causal, block_size=64)
    assert mass.shape == (_num_blocks(N, 64),)
    assert torch.isfinite(mass).all()
    assert mass.sum().item() == pytest.approx(B * H_q * N, rel=1e-9)


@pytest.mark.parametrize("causal", [False, True])
def test_matches_naive_oracle(causal):
    from kvblockmass.reference import block_mass_reference

    q, k = _qk(B=2, H_q=4, H_kv=2, N=128, D=32, seed=7)
    ref = block_mass_reference(q, k, causal=causal, block_size=64)
    nv = block_mass_naive(q, k, causal=causal, block_size=64)
    assert torch.allclose(ref, nv, rtol=1e-5, atol=1e-9)


# --------------------------------------------------------------- GQA shapes
def test_gqa_matches_naive():
    from kvblockmass.reference import block_mass_reference

    q, k = _qk(B=1, H_q=8, H_kv=2, N=96, D=16, seed=3)
    ref = block_mass_reference(q, k, causal=True, block_size=32)
    nv = block_mass_naive(q, k, causal=True, block_size=32)
    assert torch.allclose(ref, nv, rtol=1e-5, atol=1e-9)
    assert ref.shape == (_num_blocks(96, 32),)


# ------------------------------------------------------- ragged block edges
@pytest.mark.parametrize("N", [1, 5, 64, 65, 100, 129])
def test_ragged_lengths_match_naive(N):
    from kvblockmass.reference import block_mass_reference

    q, k = _qk(B=2, H_q=2, H_kv=1, N=N, D=8, seed=11)
    ref = block_mass_reference(q, k, causal=True, block_size=64)
    nv = block_mass_naive(q, k, causal=True, block_size=64)
    assert ref.shape == (_num_blocks(N, 64),)
    assert torch.allclose(ref, nv, rtol=1e-5, atol=1e-9)


def test_n_smaller_than_block_size():
    from kvblockmass.reference import block_mass_reference

    q, k = _qk(B=1, H_q=1, H_kv=1, N=16, D=8, seed=5)
    mass = block_mass_reference(q, k, causal=False, block_size=64)
    assert mass.shape == (1,)
    assert mass.sum().item() == pytest.approx(1 * 1 * 16, rel=1e-9)


# ------------------------------------------------------------------- masks
def test_attn_mask_redistributes_but_preserves_row_mass():
    """Softmax rows renormalize: forbidding key columns keeps each unmasked
    row's total mass at 1.0 (distribution changes, total does not)."""
    from kvblockmass.reference import block_mass_reference

    B, H_q, N, D = 2, 2, 128, 16
    q, k = _qk(B=B, H_q=H_q, N=N, D=D, seed=13)
    full = block_mass_reference(q, k, causal=False, block_size=64)
    assert full.sum().item() == pytest.approx(B * H_q * N, rel=1e-9)

    # forbid half the key columns
    keep = torch.ones(N, dtype=torch.bool)
    keep[64:] = False
    masked = block_mass_reference(q, k, attn_mask=keep, causal=False, block_size=64)
    # renormalization property: total is still B*H_q*N ...
    assert masked.sum().item() == pytest.approx(B * H_q * N, rel=1e-9)
    # ... but the distribution must have moved (block 1 can still receive mass
    # from later query rows; the point is it is NOT identical to `full`)
    assert not torch.allclose(masked, full)
    assert torch.isfinite(masked).all()


def test_all_masked_row_contributes_zero_and_no_nan():
    from kvblockmass.reference import block_mass_reference

    B, H_q, N, D = 1, 1, 64, 8
    q, k = _qk(B=B, H_q=H_q, H_kv=1, N=N, D=D, seed=17)
    # every row keeps ONLY column 0 → fine; now a fully-forbidden row via mask=None
    # but causal at N with query row 0 keeps only col 0. Construct: mask keeps
    # nothing for row 0 by passing a row-specific [N, N] mask.
    keep = torch.ones(N, N, dtype=torch.bool)
    keep[0, :] = False  # row 0 fully masked
    mass = block_mass_reference(q, k, attn_mask=keep, causal=False, block_size=32)
    assert torch.isfinite(mass).all()
    # total = B*H_q*(N-1): row 0 contributes nothing
    assert mass.sum().item() == pytest.approx(B * H_q * (N - 1), rel=1e-9)


def test_causal_first_row_only_first_block_gets_mass():
    """With causal=True and N=64/block=64, query row 0 attends only to key 0."""
    from kvblockmass.reference import block_mass_reference

    q, k = _qk(B=1, H_q=1, H_kv=1, N=64, D=8, seed=23)
    mass = block_mass_reference(q, k, causal=True, block_size=64)
    assert mass.sum().item() == pytest.approx(1 * 1 * 64, rel=1e-9)
    # single block here; sanity only
    assert mass.shape == (1,)


def test_dtype_and_device_arg_accepted():
    """Device argument must be accepted (CPU today, cuda/rocm on runners)."""
    from kvblockmass.reference import block_mass_reference

    q, k = _qk(B=1, H_q=2, H_kv=1, N=32, D=8, seed=29)
    mass = block_mass_reference(q, k, causal=False, block_size=16, device="cpu")
    assert mass.device.type == "cpu"
    assert mass.dtype == q.dtype
