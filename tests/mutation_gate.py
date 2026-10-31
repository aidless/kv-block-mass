"""Mutation gate: prove the test suite has teeth.

Injects three realistic bugs into block-mass implementations and asserts the
naive-oracle comparison CATCHES every one of them (plus a control: the correct
implementation passes). Run:  python tests/mutation_gate.py

Bugs (drawn from real failure modes of this operator family):
  1. CAUSAL_DROPPED  — causal mask silently ignored (kvfrag hit this in
     2026-10: Pearson > 0.999 yet ~0.2 relative error — correlation is NOT
     the gate, exact comparison is).
  2. NO_DENOM        — softmax denominator skipped (exp(s) summed, not
     normalized): totals drift by ~N per row.
  3. BUCKET_OFF_BY_ONE — key columns land in bucket (c // bs) + 1 with the
     last bucket wrapping to 0: shape-right, values wrong.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from tests.naive import block_mass_naive


def _reference_variant(q, k, causal, block_size, bug: str):
    """Reference implementation with one injected bug (or none)."""
    B, H_q, N, D = q.shape
    H_kv = k.shape[1]
    rep = H_q // H_kv
    k_full = k.repeat_interleave(rep, dim=1)
    scale = 1.0 / D ** 0.5
    chunk = 256

    keep = torch.ones(N, N, dtype=torch.bool)
    if causal and bug != "CAUSAL_DROPPED":
        keep = torch.tril(keep)

    num_blocks = (N + block_size - 1) // block_size
    mass = torch.zeros(num_blocks, dtype=torch.float64)

    col_block = torch.arange(N) // block_size
    if bug == "BUCKET_OFF_BY_ONE":
        col_block = (col_block + 1) % num_blocks

    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        rows_keep = keep[start:stop]
        row_has_any = rows_keep.any(dim=-1)
        scores = (q[:, :, start:stop] @ k_full[:, :, start:stop].transpose(-1, -2)) * scale
        scores = scores.masked_fill(~rows_keep, float("-inf"))
        if bug == "NO_DENOM":
            weights = torch.exp(scores)
            weights = torch.nan_to_num(weights, nan=0.0)  # -inf rows -> 0
        else:
            weights = torch.softmax(scores, dim=-1)
            weights = torch.where(
                row_has_any.view(1, 1, -1, 1), weights, torch.zeros_like(weights)
            )
            weights = torch.nan_to_num(weights, nan=0.0)
        contrib = weights.sum(dim=(0, 1))                 # [C, N]
        mass.index_add_(0, col_block, contrib.sum(dim=0))

    return mass


def main() -> int:
    torch.manual_seed(0)
    B, H_q, H_kv, N, D = 2, 4, 2, 128, 16
    g = torch.Generator().manual_seed(99)
    q = torch.randn(B, H_q, N, D, generator=g, dtype=torch.float64)
    k = torch.randn(B, H_kv, N, D, generator=g, dtype=torch.float64)
    block_size = 32

    nv = block_mass_naive(q, k, causal=True, block_size=block_size)
    tol = 1e-8

    # control: the honest implementation must PASS the oracle comparison
    ok = _reference_variant(q, k, causal=True, block_size=block_size, bug="none")
    ctrl_err = (ok - nv).abs().max().item()
    assert torch.allclose(ok, nv, rtol=1e-5, atol=tol), f"CONTROL FAILED: err={ctrl_err}"

    results = []
    for bug in ("CAUSAL_DROPPED", "NO_DENOM", "BUCKET_OFF_BY_ONE"):
        bad = _reference_variant(q, k, causal=True, block_size=block_size, bug=bug)
        caught = not torch.allclose(bad, nv, rtol=1e-5, atol=tol)
        max_err = (bad - nv).abs().max().item()
        results.append((bug, caught, max_err))

    print(f"control (no bug): max|err| = {ctrl_err:.3e}  -> PASSED (as required)")
    all_caught = True
    for bug, caught, err in results:
        print(f"{bug:18s} max|err| = {err:.3e}  -> {'CAUGHT' if caught else 'MISSED'}")
        all_caught &= caught

    if not all_caught:
        print("MUTATION GATE FAILED: at least one bug escaped the oracle comparison")
        return 1
    print("MUTATION GATE PASSED: all 3 injected bugs caught, control passes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
