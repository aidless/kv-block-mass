"""Reference (O(N) memory) implementation tests. RED before reference.py exists."""


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


def test_n_greater_than_query_chunk():
    """N must exceed the default query_chunk=256 to exercise chunked key reuse
    (a slice-the-keys bug only shows up here; caught on a real T4 run)."""
    from kvblockmass.reference import block_mass_reference

    q, k = _qk(B=1, H_q=2, H_kv=1, N=600, D=16, seed=59)
    ref = block_mass_reference(q, k, causal=True, block_size=64, query_chunk=256)
    nv = block_mass_naive(q, k, causal=True, block_size=64)
    assert torch.allclose(ref, nv, rtol=1e-5, atol=1e-9)
    assert ref.sum().item() == pytest.approx(1 * 2 * 600, rel=1e-9)


def test_dtype_and_device_arg_accepted():
    """Device argument must be accepted (CPU today, cuda/rocm on runners)."""
    from kvblockmass.reference import block_mass_reference

    q, k = _qk(B=1, H_q=2, H_kv=1, N=32, D=8, seed=29)
    mass = block_mass_reference(q, k, causal=False, block_size=16, device="cpu")
    assert mass.device.type == "cpu"
    assert mass.dtype == q.dtype

def _recording_matmul(seen, real):
    """Count matmul calls and, for 4-D score matmuls, the key extent per chunk.

    NOTE: torch.matmul recurses internally (per-head 3-D calls), so the call
    count alone is not a work measure — the sum of key extents is.
    """
    def inner(a, b, *args, **kwargs):
        if a.ndim == 4:
            seen.append(b.shape[-1])   # b is k.T -> last dim = key extent
        return real(a, b, *args, **kwargs)
    return inner


# ------------------------------------------------- causal early exit (exact)
def test_causal_skip_is_exact():
    """The skip must change *work*, never *values*: compare against both the
    un-skipped path and the dense oracle, incl. N > query_chunk and a 2-D mask."""
    from kvblockmass.reference import block_mass_reference

    # N=600 > default chunk 256, so several chunks get different bounds
    q, k = _qk(B=1, H_q=4, H_kv=2, N=600, D=16, seed=71)
    skip = block_mass_reference(q, k, causal=True, block_size=64, query_chunk=256)
    full = block_mass_reference(
        q, k, causal=True, block_size=64, query_chunk=256, causal_skip=False
    )
    oracle = block_mass_naive(q, k, causal=True, block_size=64)
    assert torch.allclose(skip, full, rtol=1e-9, atol=0.0)
    assert torch.allclose(skip, oracle, rtol=1e-9, atol=1e-9)

    # ragged N and a pair mask stacked on top of causality
    q2, k2 = _qk(B=2, H_q=2, H_kv=1, N=129, D=8, seed=72)
    mask2d = torch.rand(N_ := 129, 129, generator=torch.Generator().manual_seed(5)) > 0.5
    a = block_mass_reference(q2, k2, attn_mask=mask2d, causal=True, block_size=32)
    b = block_mass_reference(
        q2, k2, attn_mask=mask2d, causal=True, block_size=32, causal_skip=False
    )
    c = block_mass_naive(q2, k2, attn_mask=mask2d, causal=True, block_size=32)
    assert torch.allclose(a, b, rtol=1e-9, atol=0.0)
    assert torch.allclose(a, c, rtol=1e-9, atol=1e-9)


def test_causal_skip_actually_skips():
    """Engagement proof: a gate that only compares values cannot tell "the
    optimisation runs" from "the optimisation is a no-op". Record the key extent
    handed to torch.matmul and assert the total work really shrinks."""
    from kvblockmass import reference as ref_mod

    q, k = _qk(B=1, H_q=2, H_kv=1, N=1024, D=8, seed=73)
    seen = []
    real_matmul = torch.matmul
    recording_matmul = _recording_matmul(seen, real_matmul)

    torch.matmul = recording_matmul
    try:
        ref_mod.block_mass_reference(q, k, causal=True, block_size=64, query_chunk=256)
        with_skip = list(seen)
        seen.clear()
        ref_mod.block_mass_reference(
            q, k, causal=True, block_size=64, query_chunk=256, causal_skip=False
        )
        without_skip = list(seen)
    finally:
        torch.matmul = real_matmul

    # 4 chunks of 256 rows over N=1024: bounds are 256,512,768,1024
    assert with_skip == [256, 512, 768, 1024], with_skip   # key extent per chunk
    assert without_skip == [1024] * 4, without_skip
    assert sum(with_skip) < 0.7 * sum(without_skip)
