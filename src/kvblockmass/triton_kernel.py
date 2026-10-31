"""Triton kernel for block-granular attention mass.

Strategy (per docs/superpowers/specs/2026-10-31-kv-block-mass-design.md):
each program owns one query block of one (batch, head); it reads the key
stream TWICE — pass 1 computes the online-softmax row max and denominator,
pass 2 recomputes scores and atomically accumulates exp(s - m)/l into the
block buckets. Reading k twice is the deliberate trade for numerical
stability (a single pass with running rescaling of bucket accumulators would
need exp() per element twice anyway, plus non-atomic per-program buffers).

Memory: never materializes [B, H, N, N]; peak kernel memory is the
[BLOCK_M, BLOCK_N] score tile in registers/SMEM plus O(num_blocks) mass.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _block_mass_kernel(
    Q, K, MASK, MASS,                # ptrs
    stride_qb, stride_qm, stride_qd,  # Q [B*H_q, N, D]
    stride_kb, stride_km, stride_kd,  # K [B*H_kv, N, D]
    stride_mn,                        # MASK row stride (used when MASK_2D)
    n_q_blocks,                       # grid dim 0 size (for 1D mask indexing nothing)
    B, H_q, H_kv, N,
    scale,
    kv_block_size,                    # pooling block size (runtime; ragged tail OK)
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASK_2D: tl.constexpr,            # mask is [N, N] (else [N] column mask)
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)          # query block id
    pid_bh = tl.program_id(1)         # flat batch*head id
    b = pid_bh // H_q
    h = pid_bh % H_q
    kv_h = h // (H_q // H_kv)         # GQA: repeat_interleave semantics —
                                      # q-head h shares kv-head h // group
                                      # (NOT h % H_kv; caught by the ROCm
                                      # bench with raw H_kv=2 inputs)

    q_start = pid_m * BLOCK_M
    offs_m = q_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    row_valid = offs_m < N

    # base pointers
    q_ptr = Q + pid_bh * stride_qb
    k_ptr = K + (b * H_kv + kv_h) * stride_kb

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # ------------------------------------------------ pass 1: row max + denom
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        col_valid = offs_n < N
        k_tile = tl.load(
            k_ptr + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd,
            mask=col_valid[:, None] & (offs_d[None, :] < D),
            other=0.0,
        )                                                     # [BLOCK_N, D]
        q_tile = tl.load(
            q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=row_valid[:, None] & (offs_d[None, :] < D),
            other=0.0,
        )                                                     # [BLOCK_M, D]
        s = tl.dot(q_tile, tl.trans(k_tile), input_precision="ieee").to(tl.float32) * scale

        keep = col_valid[None, :] & row_valid[:, None]
        if CAUSAL:
            keep = keep & (offs_m[:, None] >= offs_n[None, :])
        if MASK_2D:
            mvals = tl.load(MASK + offs_m[:, None] * stride_mn + offs_n[None, :],
                            mask=row_valid[:, None] & col_valid[None, :],
                            other=0).to(tl.int1)
            keep = keep & mvals
        else:
            mvals = tl.load(MASK + offs_n, mask=col_valid, other=0).to(tl.int1)
            keep = keep & mvals[None, :]

        s = tl.where(keep, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # rows that are entirely masked keep m = -inf; guard alpha/nan
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)                          # exp(-inf-0)=0 ok
        p = tl.exp(s - m_safe[:, None])                       # masked -> 0
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # ------------------------------------------------ pass 2: bucketed add
    denom = tl.where(l_i == 0.0, 1.0, l_i)                    # dead rows -> /1
    m_final = tl.where(m_i == float("-inf"), 0.0, m_i)
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        col_valid = offs_n < N
        k_tile = tl.load(
            k_ptr + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd,
            mask=col_valid[:, None] & (offs_d[None, :] < D),
            other=0.0,
        )
        q_tile = tl.load(
            q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=row_valid[:, None] & (offs_d[None, :] < D),
            other=0.0,
        )
        s = tl.dot(q_tile, tl.trans(k_tile), input_precision="ieee").to(tl.float32) * scale

        keep = col_valid[None, :] & row_valid[:, None]
        if CAUSAL:
            keep = keep & (offs_m[:, None] >= offs_n[None, :])
        if MASK_2D:
            mvals = tl.load(MASK + offs_m[:, None] * stride_mn + offs_n[None, :],
                            mask=row_valid[:, None] & col_valid[None, :],
                            other=0).to(tl.int1)
            keep = keep & mvals
        else:
            mvals = tl.load(MASK + offs_n, mask=col_valid, other=0).to(tl.int1)
            keep = keep & mvals[None, :]

        s = tl.where(keep, s, float("-inf"))
        p = tl.exp(s - m_final[:, None])                      # masked -> 0
        p = p / denom[:, None]
        p = tl.where(keep, p, 0.0)

        bucket = offs_n // kv_block_size
        # rank-1 atomic: all query rows share the same key buckets, so fold the
        # BLOCK_M axis first (identical math, 1/64 the atomic traffic; and
        # Triton 3.5.1/HIP refuses to auto-broadcast rank-1 pointer tensors)
        tl.atomic_add(MASS + bucket, tl.sum(p, axis=0))


def block_mass_triton(
    q: torch.Tensor,                        # [B, H_q, N, D] cuda/rocm
    k: torch.Tensor,                        # [B, H_kv, N, D]
    attn_mask: torch.Tensor | None = None,  # [N] or [N, N]; True = keep
    causal: bool = False,
    block_size: int = 64,
    scale: float | None = None,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
):
    B, H_q, N, D = q.shape
    H_kv = k.shape[1]
    if H_q % H_kv != 0:
        raise ValueError(f"GQA requires H_q % H_kv == 0, got H_q={H_q}, H_kv={H_kv}")
    if scale is None:
        scale = 1.0 / D ** 0.5

    qf = q.reshape(B * H_q, N, D).contiguous()
    kf = k.reshape(B * H_kv, N, D).contiguous()

    if attn_mask is None:
        mask_flat = torch.ones(N, dtype=torch.bool, device=q.device)
        mask_2d = False
    elif attn_mask.dim() == 1:
        mask_flat = attn_mask.to(q.device).bool()
        mask_2d = False
    else:
        mask_flat = attn_mask.to(q.device).bool().contiguous()  # [N, N]
        mask_2d = True

    num_blocks = (N + block_size - 1) // block_size
    mass = torch.zeros(num_blocks, dtype=torch.float32, device=q.device)

    grid = ((N + BLOCK_M - 1) // BLOCK_M, B * H_q)
    _block_mass_kernel[grid](
        qf, kf, mask_flat, mass,
        qf.stride(0), qf.stride(1), qf.stride(2),
        kf.stride(0), kf.stride(1), kf.stride(2),
        mask_flat.stride(0) if mask_2d else 0,
        num_blocks,
        B, H_q, H_kv, N,
        scale,
        block_size,
        D=D, BLOCK_D=triton.next_power_of_2(D), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        MASK_2D=mask_2d, CAUSAL=causal,
        num_warps=4,
    )
    return mass
