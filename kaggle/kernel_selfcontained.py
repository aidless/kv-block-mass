# AUTO-GENERATED — do not edit by hand. Sources: tests/naive.py, src/kvblockmass/{reference,triton_kernel}.py, bench/bench_block_mass.py
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

        # NOTE: slice ONLY the query axis; keys stay full-length
        # (slicing k too would give [B,H,C,C] scores and break the [C,N] mask
        # whenever N > query_chunk — caught by a Kaggle T4 run, see git log)
        scores = torch.matmul(
            q[:, :, start:stop].to(acc_dtype),
            k_full.to(acc_dtype).transpose(-1, -2),
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
    if q.is_cuda:  # self-contained kernel build: use inline triton module
        from _kvbm_triton import block_mass_triton
        return block_mass_triton(q, k, **kwargs)
    return block_mass_reference(q, k, **kwargs)


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


if __name__ == '__main__':
    """Benchmark: reference vs Triton kernel for block-granular attention mass.
    
    Measures wall time (torch.cuda.Event) and peak device memory
    (torch.cuda.max_memory_allocated) across sequence lengths. The dense oracle
    (tests/naive.py) is included at small N only, to show WHY the naive route
    (materializing [B, H, N, N]) is not viable.
    
    Usage:  python bench/bench_block_mass.py [--out results.json]
    """
    
    import argparse
    import json
    import platform
    from datetime import datetime, timezone
    from pathlib import Path
    
    import torch
    
    import sys as _sys, types as _types
    
    def _register(name, **fns):
        m = _types.ModuleType(name)
        for k, v in fns.items(): setattr(m, k, v)
        _sys.modules[name] = m; return m
    
    _register('_kvbm_naive', block_mass_naive=block_mass_naive)
    _register('_kvbm_reference', block_mass_reference=block_mass_reference)
    _register('_kvbm_triton', block_mass_triton=block_mass_triton)
    
    from _kvbm_naive import block_mass_naive as _n  # noqa: E402,F401
    from _kvbm_reference import block_mass_reference as _r  # noqa: E402,F401
    from _kvbm_triton import block_mass_triton as _t  # noqa: E402,F401
    
    NS = [1024, 2048, 4096, 8192]
    DENSE_NAIVE_NS = [1024, 2048]          # O(N^2) oracle: small N only
    B, H_Q, H_KV, D, BLOCK = 1, 8, 2, 64, 64
    REPEATS = 5
    
    
    def _backend() -> str:
        return "rocm" if torch.version.hip else "cuda"
    
    
    def _bench(fn, *args, **kw):
        """Return (median_ms, peak_mib, result)."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fn(*args, **kw)                     # warmup / compile
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(REPEATS):
            start.record()
            res = fn(*args, **kw)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        times.sort()
        peak = torch.cuda.max_memory_allocated() / 2**20
        return times[len(times) // 2], peak, res
    
    
    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", default=None)
        args = ap.parse_args()
    
        assert torch.cuda.is_available(), "this benchmark needs a CUDA/ROCm device"
        dev = torch.device("cuda")
        torch.manual_seed(0)
    
        meta = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "device_name": torch.cuda.get_device_name(0),
            "backend": _backend(),
            "torch": torch.__version__,
            "triton": __import__("triton").__version__,
            "python": platform.python_version(),
            "shape": {"B": B, "H_q": H_Q, "H_kv": H_KV, "D": D, "block_size": BLOCK},
            "repeats": REPEATS,
        }
        print(json.dumps(meta, indent=2))
    
        rows = []
        for N in NS:
            g = torch.Generator(device="cpu").manual_seed(N)
            q = torch.randn(B, H_Q, N, D, generator=g, dtype=torch.float32).to(dev)
            k = torch.randn(B, H_KV, N, D, generator=g, dtype=torch.float32).to(dev)
    
            row = {"N": N}
    
            t_ref, mem_ref, res_ref = _bench(block_mass_reference, q, k, causal=True, block_size=BLOCK)
            row["reference_ms"], row["reference_mib"] = t_ref, mem_ref
    
            t_tri, mem_tri, res_tri = _bench(block_mass_triton, q, k, causal=True, block_size=BLOCK)
            row["triton_ms"], row["triton_mib"] = t_tri, mem_tri
    
            rel = ((res_ref - res_tri).abs() / res_ref.abs().clamp_min(1e-6)).max().item()
            row["rel_err"] = rel
            assert rel < 1e-3, f"N={N}: triton/reference rel err {rel}"
    
            if N in DENSE_NAIVE_NS:
                t_nv, mem_nv, res_nv = _bench(block_mass_naive, q, k, causal=True, block_size=BLOCK)
                row["dense_naive_ms"], row["dense_naive_mib"] = t_nv, mem_nv
                # fp32 accumulations differ in summation order; absolute 1e-4 on
                # O(1e3)-magnitude bucket sums is noise-level — use relative
                rel_nv = ((res_nv - res_ref).abs().max() / res_ref.abs().max()).item()
                assert rel_nv < 1e-5, f"N={N}: naive/reference rel err {rel_nv}"
    
            rows.append(row)
            print(json.dumps(row))
    
        result = {"meta": meta, "rows": rows}
        out = args.out
        if out is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out = str(Path(__file__).parent / f"bench_{_backend()}_{stamp}.json")
        try:
            Path(out).write_text(json.dumps(result, indent=2))
            print(f"\nwrote {out}")
        except OSError as e:  # e.g. read-only FS on Kaggle workers — stdout already
            print(f"\n(write {out} failed: {e}; JSON above is authoritative)")
    
        # markdown table alongside
        md = [f"# bench @ {meta['device_name']} ({meta['backend']})", "",
              f"torch {meta['torch']} / triton {meta['triton']} / shape B{B} Hq{H_Q} Hkv{H_KV} D{D} block{BLOCK}", "",
              "| N | dense O(N²) ms | dense MiB | reference ms | reference MiB | triton ms | triton MiB | rel err |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            md.append(
                f"| {r['N']} | {r.get('dense_naive_ms', float('nan')):.2f} | {r.get('dense_naive_mib', float('nan')):.0f} "
                f"| {r['reference_ms']:.2f} | {r['reference_mib']:.0f} "
                f"| {r['triton_ms']:.2f} | {r['triton_mib']:.0f} | {r['rel_err']:.2e} |"
            )
        md_path = str(Path(out).with_suffix(".md"))
        try:
            Path(md_path).write_text("\n".join(md) + "\n")
            print(f"wrote {md_path}")
        except OSError as e:
            print("\n".join(md))
    
    
    if __name__ == "__main__":
        main()