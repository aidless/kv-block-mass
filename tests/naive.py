"""Dense (materialized) block-mass oracle. TESTS ONLY — O(N^2) memory.

Deliberately simple and obviously correct: full softmax matrix, then pool
columns into block buckets. Used to cross-check the O(N) implementations.
"""

import torch


def block_mass_naive(
    q: torch.Tensor,          # [B, H_q, N, D]
    k: torch.Tensor,          # [B, H_kv, N, D]  (GQA: H_q % H_kv == 0)
    attn_mask: torch.Tensor | None = None,  # broadcastable to [B, 1, N, N]; True = keep
    causal: bool = False,
    block_size: int = 64,
    scale: float | None = None,
) -> torch.Tensor:
    """Return mass [num_blocks] pooled over query rows/heads/batch.

    mass[j] = sum over (b, h, i, j_pos in block j) of softmax_j(s)[i, j_pos]
    where s = q @ k^T * scale and softmax is over the key axis N, with
    positions removed by mask/causal contributing exactly 0.
    """
    B, H_q, N, D = q.shape
    H_kv = k.shape[1]
    assert H_q % H_kv == 0, "GQA requires H_q divisible by H_kv"
    if scale is None:
        scale = 1.0 / D ** 0.5

    # expand KV heads to query heads (GQA)
    rep = H_q // H_kv
    k_full = k.repeat_interleave(rep, dim=1)          # [B, H_q, N, D]

    scores = torch.matmul(q, k_full.transpose(-1, -2)) * scale   # [B, H_q, N, N]

    keep = torch.ones(N, N, dtype=torch.bool, device=q.device)
    if causal:
        keep = torch.tril(keep)
    if attn_mask is not None:
        keep = keep & attn_mask.reshape(1, 1, N, N).expand(B, 1, N, N)[0, 0]
    keep = keep & True  # keep broadcasted shape [N, N]

    # masked-out positions get -inf so their softmax weight is exactly 0
    scores = scores.masked_fill(~keep, float("-inf"))

    # rows with nothing kept would be NaN; they must contribute 0, not NaN
    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)

    num_blocks = (N + block_size - 1) // block_size
    # pad N to a multiple of block_size with zeros, then reshape-pool
    pad = num_blocks * block_size - N
    if pad:
        weights = torch.nn.functional.pad(weights, (0, pad))

    # [B, H, N, num_blocks, block] -> sum over last axis, then over B/H/N
    weights = weights.reshape(B, H_q, N, num_blocks, block_size)
    mass = weights.sum(dim=-1).sum(dim=(0, 1, 2))
    return mass
