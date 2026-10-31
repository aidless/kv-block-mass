"""O(N)-memory reference implementation of block-granular attention mass.

`block_mass_reference(q, k, ...) -> mass[num_blocks]`

mass[j] = sum over (batch b, head h, query row i, key column j_pos in block j)
of softmax_j(q_i . k_j * scale), i.e. how much attention (post-softmax weight)
flows into KV block j in total.

Why this exists: KV-eviction research (H2O, SnapKV, ...) wants per-block
importance scores, but the naive route materializes softmax([B, H, N, N]) —
for N=4096, one layer, fp16, that is ~1 GB * H * B. We instead accumulate
column sums while streaming over query chunks, so peak memory is
O(B * H * chunk * N) and the per-block accumulator is O(num_blocks).

Numerics: every unmasked query row sums to exactly 1.0 over keys, so
sum(mass) == B * H_q * (number of unmasked queries). Rows that are fully
masked contribute exactly 0 (we zero their NaN softmax weights).
"""

import torch


def _expand_kv_heads(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """GQA: [B, H_kv, N, D] -> [B, H_q, N, D] via repeat_interleave."""
    B, H_q, N, D = q.shape
    H_kv = k.shape[1]
    if H_kv == H_q:
        return k
    if H_q % H_kv != 0:
        raise ValueError(f"GQA requires H_q % H_kv == 0, got H_q={H_q}, H_kv={H_kv}")
    return k.repeat_interleave(H_q // H_kv, dim=1)


@torch.no_grad()
def block_mass_reference(
    q: torch.Tensor,                        # [B, H_q, N, D]
    k: torch.Tensor,                        # [B, H_kv, N, D]
    attn_mask: torch.Tensor | None = None,  # [N] or [N, N]; True = keep
    causal: bool = False,
    block_size: int = 64,
    scale: float | None = None,
    query_chunk: int = 256,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Block-pooled column sums of the softmax attention matrix, O(N) memory.

    Args:
        q: queries [B, H_q, N, D].
        k: keys [B, H_kv, N, D]; GQA supported (H_q % H_kv == 0).
        attn_mask: bool mask over key columns ([N]) or (query, key) pairs
            ([N, N]); True keeps the position.
        causal: apply lower-triangular causality on top of attn_mask.
        block_size: KV block size for pooling (last block may be ragged).
        scale: score scale; defaults to 1/sqrt(D).
        query_chunk: rows of the query axis processed at once — the memory
            knob; peak score buffer is [B, H_q, query_chunk, N].
        device: override device for the computation (e.g. run on GPU while
            inputs are small CPU tensors).

    Returns:
        mass [num_blocks] on `q.device` (float32 accumulation, same dtype as
        inputs if that is float64).
    """
    B, H_q, N, D = q.shape
    if scale is None:
        scale = 1.0 / D ** 0.5

    # Resolve compute device / accumulation dtype (fp64 stays exact, else fp32)
    if device is not None:
        q = q.to(device)
        k = k.to(device)
    acc_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32

    k_full = _expand_kv_heads(q, k)  # [B, H_q, N, D]

    # Build the [N, N] keep-matrix once (bool). This is the only O(N^2)
    # allocation, and it is 1 byte per entry (N=8192 -> 64 MB), no H/B factor.
    keep = torch.ones(N, N, dtype=torch.bool, device=q.device)
    if causal:
        keep = torch.tril(keep)
    if attn_mask is not None:
        keep = keep & attn_mask.to(q.device).bool()

    num_blocks = (N + block_size - 1) // block_size
    mass = torch.zeros(num_blocks, dtype=acc_dtype, device=q.device)

    # Block-bucket of each key column, expanded per key column: col_block[c]
    # says which bucket column c accumulates into. Ragged tail included.
    col_block = torch.arange(N, device=q.device) // block_size  # [N]

    for start in range(0, N, query_chunk):
        stop = min(start + query_chunk, N)
        rows_keep = keep[start:stop]                          # [C, N]
        # rows that have at least one allowed key — others would softmax to NaN
        row_has_any = rows_keep.any(dim=-1)                   # [C]

        scores = torch.matmul(
            q[:, :, start:stop].to(acc_dtype),
            k_full[:, :, start:stop].to(acc_dtype).transpose(-1, -2),
        ) * scale                                             # [B, H, C, N]

        scores = scores.masked_fill(~rows_keep, float("-inf"))
        weights = torch.softmax(scores, dim=-1)               # [B, H, C, N]
        # fully-masked rows -> NaN; they must contribute exactly 0
        weights = torch.where(
            row_has_any.view(1, 1, -1, 1), weights, torch.zeros_like(weights)
        )
        weights = torch.nan_to_num(weights, nan=0.0)

        # pool columns into block buckets: sum batch/head/query rows -> [C, N],
        # then over rows -> [N], then scatter-add into block buckets
        contrib = weights.sum(dim=(0, 1))                     # [C, N]
        mass.index_add_(0, col_block, contrib.sum(dim=0).to(acc_dtype))

    return mass


def block_mass(q, k, **kwargs):
    """Default entry: reference on CPU, kernel on GPU if available."""
    if q.is_cuda:
        from .triton_kernel import block_mass_triton

        return block_mass_triton(q, k, **kwargs)
    return block_mass_reference(q, k, **kwargs)
