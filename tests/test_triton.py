"""Reference-vs-Triton agreement tests (GPU only; skipped without CUDA/ROCm).

Run on Kaggle T4 (CUDA) or AMD Radeon W7900 (ROCm/HIP). The structural test
at the bottom always runs so the file is not silently empty locally.
"""

import inspect

import pytest
import torch

from tests.naive import block_mass_naive

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA/ROCm device"
)


def _qk(B=2, H_q=4, H_kv=2, N=128, D=32, seed=0, device="cuda", dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, H_q, N, D, generator=g, dtype=torch.float64).to(dtype).to(device)
    k = torch.randn(B, H_kv, N, D, generator=g, dtype=torch.float64).to(dtype).to(device)
    return q, k


# ------------------------------------------------------- structural (local)
def test_kernel_module_structure():
    # triton is a runtime-only dep for the kernel; skip (not fail) when absent
    tk = pytest.importorskip("kvblockmass.triton_kernel")

    assert isinstance(tk._block_mass_kernel, object)
    assert "causal" in inspect.signature(tk.block_mass_triton).parameters
    assert "attn_mask" in inspect.signature(tk.block_mass_triton).parameters


# ------------------------------------------------------------ numeric (GPU)
@cuda
@pytest.mark.parametrize("causal", [False, True])
def test_triton_matches_reference(causal):
    from kvblockmass.reference import block_mass_reference
    from kvblockmass.triton_kernel import block_mass_triton

    q, k = _qk(B=2, H_q=4, H_kv=2, N=256, seed=31)
    ref = block_mass_reference(q, k, causal=causal, block_size=64)
    tri = block_mass_triton(q, k, causal=causal, block_size=64)
    rel = ((ref - tri).abs() / ref.abs().clamp_min(1e-6)).max().item()
    assert rel < 1e-4, f"max rel err {rel}"


@cuda
def test_triton_matches_naive_gqa_ragged():
    from kvblockmass.triton_kernel import block_mass_triton

    q, k = _qk(B=1, H_q=8, H_kv=2, N=100, D=16, seed=37)  # 100 not divisible by 64
    tri = block_mass_triton(q, k, causal=True, block_size=64)
    nv = block_mass_naive(q, k, causal=True, block_size=64)
    assert torch.allclose(tri, nv, rtol=1e-3, atol=1e-4)


@cuda
def test_triton_1d_mask():
    from kvblockmass.triton_kernel import block_mass_triton

    q, k = _qk(B=1, H_q=2, H_kv=1, N=128, seed=41)
    keep = torch.ones(128, dtype=torch.bool, device=q.device)
    keep[64:] = False
    tri = block_mass_triton(q, k, attn_mask=keep, causal=False, block_size=32)
    assert torch.isfinite(tri).all()
    # rows renormalize -> total preserved
    assert tri.sum().item() == pytest.approx(1 * 2 * 128, rel=1e-3)


@cuda
def test_triton_2d_mask_all_masked_row_zero():
    from kvblockmass.triton_kernel import block_mass_triton

    q, k = _qk(B=1, H_q=1, H_kv=1, N=64, seed=43)
    keep = torch.ones(64, 64, dtype=torch.bool, device=q.device)
    keep[0, :] = False
    tri = block_mass_triton(q, k, attn_mask=keep, causal=False, block_size=32)
    assert torch.isfinite(tri).all()
    assert tri.sum().item() == pytest.approx(1 * 1 * 63, rel=1e-3)


@cuda
def test_triton_n4096_memory_and_value():
    """The headline workload: N=4096 must run without O(N^2) float materialization."""
    from kvblockmass.reference import block_mass_reference
    from kvblockmass.triton_kernel import block_mass_triton

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    q, k = _qk(B=1, H_q=8, H_kv=2, N=4096, D=64, seed=47)
    tri = block_mass_triton(q, k, causal=True, block_size=64)
    peak = torch.cuda.max_memory_allocated() / 2**20

    ref = block_mass_reference(q, k, causal=True, block_size=64)
    rel = ((ref - tri).abs() / ref.abs().clamp_min(1e-6)).max().item()
    assert rel < 1e-4, f"max rel err {rel}"
    # record for the README; soft bound: must stay well under the ~2 GiB a
    # [1, 8, 4096, 4096] fp32 matrix would take (that alone is 512 MiB * 4 B)
    print(f"\nN=4096 triton peak memory: {peak:.1f} MiB")
    assert peak < 512, f"peak {peak:.1f} MiB too high"


@cuda
def test_triton_sum_rule_gpu():
    from kvblockmass.triton_kernel import block_mass_triton

    q, k = _qk(B=2, H_q=4, H_kv=2, N=512, seed=53)
    tri = block_mass_triton(q, k, causal=False, block_size=64)
    assert tri.sum().item() == pytest.approx(2 * 4 * 512, rel=1e-3)


@cuda
def test_triton_causal_skip_is_exact():
    """The kernel's causal key truncation must match the un-truncated kernel
    exactly (same bound as the reference: last row of the query block)."""
    from kvblockmass.triton_kernel import block_mass_triton

    q, k = _qk(B=1, H_q=4, H_kv=2, N=1024, D=32, seed=83)
    skip = block_mass_triton(q, k, causal=True, block_size=64)
    full = block_mass_triton(q, k, causal=True, block_size=64, causal_skip=False)
    rel = ((skip - full).abs() / full.abs().clamp_min(1e-6)).max().item()
    assert rel < 1e-6, f"early exit changed values: max rel {rel}"
    # and it must still match the reference (which skips too)
    from kvblockmass.reference import block_mass_reference

    ref = block_mass_reference(q, k, causal=True, block_size=64)
    rel_ref = ((skip - ref).abs() / ref.abs().clamp_min(1e-6)).max().item()
    assert rel_ref < 1e-4, f"triton vs reference rel {rel_ref}"
