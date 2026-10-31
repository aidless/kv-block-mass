# Reading notes: how national / vendor GPGPU backends hook into PyTorch

Sources verified 2026-10-31 via raw.githubusercontent.com (HTTP-status based) and
api.github.com star counts. Facts only; every file:line below was actually fetched.

## The one mechanism all three vendor plugins share

| repo | file:line | call |
|---|---|---|
| MooreThreads/torch_musa (main) | `torch_musa/__init__.py:43` | `torch.utils.rename_privateuse1_backend("musa")` |
| Ascend/pytorch (master) | `torch_npu/_init/registry/backend.py:47` | same, `"npu"`, plus `_register_device_module("npu", torch_npu.npu)` |
| Cambricon/torch_mlu (r1.22_pt2.1.0) | `torch_mlu/__init__.py:48` | same, `"mlu"` |

So "adapting a framework to a domestic GPGPU" is, at the Python layer, the
**PrivateUse1 dispatch key** plus four registration families (documented in the
official tutorial `pytorch/tutorials → advanced_source/privateuseone.rst`;
the often-cited `docs/source/notes/privateuse1_backend.rst` does **not** exist —
raw 404):

1. kernels: `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)` + `custom_cpu_fallback`
2. Generator: `REGISTER_GENERATOR_PRIVATEUSE1`
3. DeviceGuard: `C10_REGISTER_GUARD_IMPL(PrivateUse1, CustomGuardImpl)`
4. BackendMeta serialization + AMP autocast methods
   (`get_amp_supported_dtype` / `is_autocast_enabled` / ...)

A hardware-free way to *run* the whole mechanism: `pytorch/pytorch`
`test/test_privateuseone_python_backend.py` registers a pure-Python backend
(`"npy"`) via `torch.utils._setup_privateuseone_for_python_backend`
(`torch/utils/backend_registration.py:549`).

## What each vendor repo teaches

- **torch_musa (519★, BSD-3)**: mirrors PyTorch's own directory tree
  (`core/ aten/ distributed/ profiler/ _inductor/`); ships a cuda→musa
  source-porting tool (`torch_musa/utils/mapping/*.json`). Test convention:
  `testing.OpTest(func=..., input_args=..., test_dtype=...).check_result()`.
- **TorchNPU (598★, BSD-3)**: heaviest engineering — 22 files under
  `torch_npu/_init/` with a `patches/` monkey-patch layer + registry
  stratification. Hard version matrix (their `COMPATIBILITY.en.md`:
  TorchNPU 26.2.0 ↔ PyTorch 2.9–2.14 ↔ CANN 9.2.0). Primary repo is GitCode;
  GitHub is an official mirror.
- **torch_mlu (61★)**: codegen-driven (`codegen/mlu_functions.yaml` →
  `gen_mlu_stubs.py` → templates); stale public default branch
  (`r1.22_pt2.1.0`, last push 2025-03) — link the branch, never `main`.

## Why kv-block-mass exists in this context

PagedAttention-style serving (vLLM `v1/core/block_pool.py` — the Python
sibling of our [`paged-kv`](https://github.com/aidless/paged-kv) C++ block
manager) needs per-block eviction scores. Computing them naively materializes
softmax `[B, H, N, N]`: 512 MiB fp32 per head at N=4096. H2O/SnapKV-style
implementations accept that cost and drop FlashAttention; this repo shows the
O(N) alternative and proves kernel-vs-reference agreement across CUDA (T4)
and ROCm (W7900 / gfx1100, where Triton runs via the HIP backend — see
ROCm/aiter's official gfx1100 support statement).
