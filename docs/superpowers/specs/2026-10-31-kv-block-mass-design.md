# kv-block-mass 设计文档

日期：2026-10-31
作者：刘泽文（独立研究者 / 求职中）
状态：待批准（brainstorming 门禁）

## 一句话

**决定该回收哪些 KV 块的注意力质量统计，用 Triton 算子算，不物化 N×N 注意力矩阵。**

## 为什么做这个（意图与动机）

我在找「国产 GPGPU 上 AI 框架适配 / 算子优化」这类岗位，JD 里两条硬要求我拿不出证据：

- 「熟悉 CUDA 编程模型及 GPU 执行与内存体系」——我懂概念，但没有一个**自己写的 GPU kernel** 可以拿出来讲
- 「具备 CPU/GPU 算子优化经验，TVM/MLIR/XLA/Triton 编译优化经验者优先」——完全没有

我已有的 `paged-kv`（C++17）实现了 vLLM 那套**块的生命周期管理**：分配、引用计数、fork、写时复制。但它**不碰 KV 数据本身**——README 里明确写了「数据拷贝由上层引擎执行」。

缺口就在这里：块管理器决定「哪些块可以回收」，但**谁来决定「该回收哪些块」**？

这正是 KV cache 驱逐研究（H2O / SnapKV / StreamingLLM）的核心问题：每个 key 位置累计收到多少注意力质量。算这个东西的朴素做法是 `output_attentions=True`，它物化 `[1,H,N,N]`——N=4096、16 头、fp32 单层就 ~1GB，也正因为如此，这类方法在实践中被迫回退 eager attention、放弃 FlashAttention 类融合实现。

所以这个仓库做一件事：**只输出每块的注意力质量（O(N) 显存），用 Triton 写融合算子，跨 CUDA/ROCm 两个后端跑出可复算的数字。**

它和 `paged-kv` 合成一条完整的线：
- `paged-kv`：块怎么管（引用计数、COW、失败无副作用）
- `kv-block-mass`：该回收哪些块（注意力质量 → 块级掩码 → 回收率）

## 目标

1. 一个**正确的** Triton kernel，算块级注意力质量，数值与 PyTorch 参考实现对齐（相对误差门，不是相关系数门）
2. 一个**可复算**的基准：显存峰值与延迟，对比 dense-eager / 分块参考实现 / Triton 算子，硬件名与后端名写进结果
3. 一个**有牙齿的**测试集：CPU 侧纯 PyTorch 参考实现对拍、GPU 侧 Triton 对参考、以及一条「注入已知 bug 必须变红」的变异门
4. 一份**诚实**的 README：数字来自实测并注明硬件；没跑过的不写
5. 学习笔记：精读 torch_npu / torch_musa / ROCm aiter / Triton tutorials 的要点，落到 `docs/reading-notes.md`

## 非目标

- 不做完整的推理引擎（不是第二个 vLLM）
- 不实现具体的驱逐策略论文（H2O 的重计算启发式、SnapKV 的观测窗口选择）——本仓库只提供**判据的输入**（块级质量），策略留给调用方
- 不做量化、不做多卡、不做 serving 集成
- 不声称在国产加速卡上跑过（本机无卡；AMD ROCm 是 AMD 不是国产，也不混为一谈）

## 反目标（技术上成功也算失败）

- 为了「好看」写一个没人能复算的加速比
- 只在小尺寸上跑，回避 N=4096 这类真实长度
- 把「相关系数 > 0.999」当成正确性证据（漏因果掩码时相关系数照样很高，相对误差会到 0.2 量级——这是我在 kvfrag 里踩过的坑，写进 README 的坑位清单）
- 让 Triton 与参考实现「差不多」就算过：必须有明确容差与失败信息

## 约束

- 本机**无可用 GPU**：`lspci` 有 RTX 3060 Mobile，但无驱动（无 `/dev/nvidia*`、无 nvidia 模块、无 nvidia-smi），`torch.cuda.is_available()=False`。因此本地只能跑 CPU 参考实现与语法检查
- GPU 执行通道（已验证可用）：Kaggle kernel（T4，免费）；AMD Radeon Cloud 实例（`root@36.150.116.206 -p 34677`，ROCm，gfx1100 级，端口当前 OPEN，额度剩 28 credits）
- 本机 Python 环境：本地虚拟环境（绝对路径不入库）→ torch 2.14.1+cu130 / triton 3.8.0（可导入，无 GPU）
- 依赖上限：numpy + torch + triton（+ pytest），不引第三方 kernel 库
- 许可证：MIT

## 设计（我的判断，不是用户指定的）

### 数据形状与语义

输入：`q: [B, H_q, N, D]`、`k: [B, H_kv, N, D]`（支持 GQA，`H_q % H_kv == 0`），因果掩码可选，`block_size` 整除关系不假设。

输出：`mass: [num_blocks]`，其中

```
mass[b] = Σ_{i∈queries} Σ_{h∈heads} Σ_{j∈block b} softmax_j(q_i·k_j / sqrt(D))[j]
```

即「第 b 块收到的注意力质量总和」。它天然归一：全序列非掩码位置的质量总和 = 每个 query 每头 1.0，乘 `B*H_q*N`。这个归一性质本身就是一条测试。

### 三份实现

1. `reference.py`：`block_mass_reference(q, k, block_size, causal, chunk)` —— 纯 PyTorch，query 分块 + 列累加，显存 O(chunk×N)。作为正确性基准
2. `triton_kernel.py`：`block_mass_triton(q, k, block_size, causal)` —— 每 program 负责一个 query 块：第一遍循环 key 块求行最大值与分母，第二遍循环累加 `exp(s-m)/denom` 到块桶（原子加）。第二遍是为了数值稳定，代价是读 k 两遍——这个 trade-off 写进 README
3. `naive.py`（仅测试用）：dense 物化 `[B,H,N,N]` 的朴素实现，只在小尺寸上跑，用来证明 1 和 2 都不是自说自话

### 判据

- CPU 测试：`naive` vs `reference`，`rtol=1e-5`（fp32），覆盖：因果/非因果、GQA（H_q≠H_kv）、N 不整除 block_size、N < block_size、全掩码行不产生 NaN
- GPU 测试：`reference` vs `triton`，相对误差 `max|a-b| / max|b| < 1e-4`，覆盖同上一组形状
- 变异门（`tests/mutation_gate.py`）：注入三类已知 bug —— ①去掉因果掩码 ②分母用错（漏掉 exp 归一）③块索引 off-by-one —— 断言对应测试**必须变红**；三门全红才算测试有牙齿
- 基准：`bench/bench_block_mass.py`，`torch.cuda.Event` 计时 + `torch.cuda.max_memory_allocated`，N ∈ {1024, 2048, 4096, 8192}，输出 JSON + Markdown 表格，含设备名、torch/triton 版本、后端（cuda/hip）

### 文件布局

```
kv-block-mass/
├── README.md                     # 诚实结果 + 坑位清单 + 与 paged-kv 的关系
├── LICENSE                       # MIT
├── pyproject.toml
├── src/kvblockmass/
│   ├── __init__.py
│   ├── reference.py              # 分块参考实现（O(N) 显存）
│   ├── triton_kernel.py          # Triton 融合算子
│   └── reclaim.py                # mass → 块级回收掩码 + 回收率统计
├── tests/
│   ├── naive.py                  # dense 物化，仅测试用
│   ├── test_reference.py         # CPU，必跑
│   ├── test_triton.py            # GPU，无卡则 skip
│   └── mutation_gate.py          # 注入 bug 必须变红
├── bench/
│   └── bench_block_mass.py
├── kaggle/
│   └── run_bench_kernel.py       # Kaggle kernel 入口（T4）
└── docs/
    ├── reading-notes.md          # 精读笔记（torch_npu / torch_musa / aiter / triton）
    ├── rocm-notes.md             # 在 ROCm 上跑 Triton 的实测记录
    └── superpowers/specs/2026-10-31-kv-block-mass-design.md
```

### 实施顺序（TDD）

1. `naive.py` + `test_reference.py::test_matches_naive` 先红 → 写 `reference.py` 转绿
2. 逐步加形状/掩码/GQA 测试，每次先红后绿
3. `triton_kernel.py`：本地只做 import 与 kernel 签名检查（无卡跑不了）；上 Kaggle 后跑 `test_triton.py` 对参考
4. `mutation_gate.py`：三个注入，逐个验证变红
5. `bench`：Kaggle T4 出数字；AMD 实例出 ROCm 数字（若实例仍可用）
6. README 填实测数字，写清「哪些是 T4、哪些是 gfx1100、哪些没跑」
7. 建公开仓 `aidless/kv-block-mass` 并推送

## 留给实现者（我）的自由

- 具体分块大小、`num_warps`、是否用 `tl.atomic_add` 还是 per-block 二次归约——由基准决定，写进 README
- 是否额外提供 `torch.compile` 对照——有余力就加，没余力就不加
- 学习笔记的深度——以「能对着源码讲清一个机制」为准

## 待用户确认的三件事

1. 仓库名 `kv-block-mass` 是否可接受（备选：`kv-mass`、`attn-block-mass`）
2. 是否同意「先只做块级质量算子，不做完整驱逐策略」的范围
3. 公开仓库的许可证用 MIT 是否可以
