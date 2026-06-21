# ForgeAttention: Fused 3-bit KV Dequantization inside Metal Attention Kernels

**Sabowsla (user-23xyz)**
Independent Researcher
GitHub: [@user-23xyz](https://github.com/user-23xyz)

---

## Abstract

Standard KV cache compression on Apple Silicon decompresses packed data to FP16 before attention. ForgeAttention eliminates this intermediate tensor by fusing the dequantization directly into the attention dot product via custom Metal compute kernels. The decompressed FP16 values never touch device memory.

On a 16GB M4 Mac Mini, this achieves 82% per-layer KV cache memory reduction at 0.99x baseline decode speed, enabling Gemma4-E4B (4B active params) to pass strict needle-in-a-haystack retrieval at 100K context and Gemma4-E2B at 300K context with flat memory pressure throughout.

We also implement per-head adaptive sparse attention (each attention head independently selects which tokens to attend to) with a fused two-dispatch Metal kernel that skips entire 256-token tiles when no tokens pass the top-K threshold.

---

## 1. Background

### 1.1 The FP16 Materialization Problem

PlanarQuant and TurboQuant compress the KV cache to 3-4 bits via rotation + scalar quantization. During attention, the standard approach decompresses the entire cache to FP16, computes Q·K^T and softmax·V, then discards the FP16 tensors.

At 40K context with 8 KV heads and 28 layers: **327MB of FP16 tensors written, read once, discarded** — per token generated. This is the dominant memory overhead and the cause of OOM on consumer hardware.

### 1.2 Prior Art

Fused quantized KV attention exists on CUDA:
- **fused-turboquant** (Argonaut790): Triton kernel, WHT dequant in registers
- **DEJAN blog**: Triton QK kernel from packed indices
- **TurboESM**: Streaming dequantization for protein models

On Metal/Apple Silicon: **no prior implementation.** The TurboQuant-MLX author attempted fusing dequant into attention and found Apple's native SDPA too fast to beat with a custom kernel.

We sidestep that blocker by not replacing SDPA entirely — instead fusing dequant only into the QK dot product (simpler operation) and using tiled online-softmax for SV.

---

## 2. Implementation

### 2.1 Fused QK Kernel

Grid: `(seq_len × dim, B×H, 1)` with threadgroup `(dim, 1, 1)`.

Each threadgroup handles one token. 128 threads cooperate to:
1. Load Q into threadgroup shared memory
2. Unpack K from packed uint32 (3-bit, 10 values per word)
3. Codebook lookup (Lloyd-Max 8 centroids) + norm scaling
4. Inverse Givens rotation on adjacent pairs in shared memory
5. Parallel dot product Q·K
6. Tree reduction (7 rounds: 128→1)
7. Thread 0 writes ONE float32 scalar to device memory

Total shared memory: 3KB of 32KB. Total device write per token: 4 bytes.

### 2.2 Tiled SV Kernel

Processes V in 256-token tiles. Each tile decompresses V on-the-fly in shared memory and accumulates `prob × V`. Partial sums reduced across tiles via `mx.sum()`. No FP16 V tensor ever cached in device memory.

### 2.3 Flash Decode Kernel

Single-pass QK + online-softmax + SV per tile. Outputs partial_o + tile_max + tile_sum_exp for log-sum-exp merge. No intermediate scores tensor in device memory.

### 2.4 Fused Sparse Attention (Two GPU Dispatches)

Phase 1: Score ALL tokens via fused QK kernel. Track per-tile top-4 scores.

Bridge: Compute per-head threshold from tile summaries (~800 floats, microseconds).

Phase 2: Selective V fetch with tile-level early exit. If no token in a 256-token tile passes the threshold, the entire threadgroup returns immediately — zero barriers, zero V work. At top-1024 from 50K tokens: ~188 of 196 tiles skip entirely.

### 2.5 FP16 Attention Math

All kernels use half-precision for the QK dot product and V accumulation (float32 for tree reduction and softmax accumulators). M4's GPU has 2x FP16 ALU throughput.

---

## 3. Results

### 3.1 Memory

| Phase | 20K ctx per layer | Reduction |
|-------|-------------------|-----------|
| Original (FP16 K + V cached) | 99.8 MB | baseline |
| Fused QK only (V still FP16) | 58.8 MB | 41% |
| Fully fused (K + V from packed) | 17.9 MB | **82%** |

### 3.2 Speed

| Context | FP32 kernels | FP16 kernels | vs baseline |
|---------|:---:|:---:|:---:|
| 1K | 1.61ms | 1.01ms | 0.99x |
| 5K | 2.56ms | 1.52ms | 0.99x |
| 10K | 4.07ms | 1.85ms | 0.99x |
| 20K | 5.29ms | 2.91ms | 0.99x |

ForgeAttention adds zero overhead vs standard FP16 KV cache decode. Measured on live server: FP16 path and ForgeAttention path produce identical decode tok/s.

### 3.3 NIAH (Needle-in-a-Haystack)

Following the strict protocol from the TriAttention V3 paper (Section 3.3): exact string matching, no display-prompt echo, temperature 0.

**Gemma4-E4B (4B active, 4-bit weights) + ForgeAttention:**

| Test | Score |
|------|-------|
| Single NIAH 10-100K (start/mid/end) | **12/12 PASS** |
| Multi-needle 20K (5 needles) | **5/5** |
| Multi-needle 50K | **5/5** |
| Multi-needle 100K | **5/5** |
| Varied haystack | **4/4 PASS** |
| Distractors (similar needles) | CONFUSED (7741 vs 7742) |
| Generative QA (real fact extraction) | **5/5 PASS** |
| Stress to 100K | **PASS** |

**Gemma4-E2B (2B active) + ForgeAttention:**

| Context | Middle NIAH | Time |
|---------|:-----------:|:----:|
| 50K | PASS | 27s |
| 100K | PASS | 81s |
| 200K | PASS | 262s |
| 300K (245K tokens) | **PASS** | 499s |

### 3.4 Maximum Context (Projected)

| Hardware | E4B Max Context |
|----------|:---------------:|
| M4 Mini 16GB | 1.3M tokens |
| M4 Pro 48GB | 6.8M tokens |
| M4 Ultra 192GB | 31.6M tokens |

---

## 4. Bugs Found

### 4.1 MLX Grid Semantics

`mx.fast.metal_kernel` grid parameter specifies **total threads**, not threadgroup count. `grid=(seq_len, H, 1)` with `threadgroup=(dim, 1, 1)` launches `ceil(seq_len/dim)` threadgroups, not `seq_len` threadgroups. Most tokens silently return zero. Fix: `grid=(seq_len * dim, H, 1)`.

### 4.2 Deferred K Runtime State

`_alloc()` checked `self.defer_k` (config flag) instead of `self._k_deferred` (runtime state) when extending storage after quantization. Shape mismatch crash on hot buffer flush.

### 4.3 ArraysCache.trim() for Hybrid Models

Qwen3.5's GatedDeltaNet linear attention layers use `ArraysCache` which had no `trim()` method. `can_trim_prompt_cache()` returned False, silently preventing KV eviction under `--prompt-cache-bytes`. Fix: `is_trimmable()` returns True, `trim(n)` is a no-op (linear attention state is O(1), not sequence-indexed).

---

## 5. Interaction with TriAttention V3

ForgeAttention and TriAttention V3 solve orthogonal problems:
- **TriAttention V3**: which tokens to **evict** (fewer tokens in cache)
- **ForgeAttention**: how to **store and read** remaining tokens (fewer bits, no FP16 intermediate)

They stack: TriAttention evicts 10% of tokens → ForgeAttention stores the remaining 90% at 82% less memory → combined compression multiplies.

ForgeAttention's sliding window (`attention_window`) is a simpler alternative to eviction that achieves O(1) decode but loses retrieval on hybrid architectures (same failure mode as V3 on Qwen3.5, documented in V3 Section 5).

PR #75's hybrid budget scaling formula (`effective_budget = 1 - (1 - raw_budget) * attention_fraction`) is directly applicable to ForgeAttention's window sizing for Gemma4-E4B (7/42 global layers).

---

## 6. Code

MIT licensed: [github.com/user-23xyz/forgeattention](https://github.com/user-23xyz/forgeattention)

Files:
- `kernels/planarquant_kernels.py` — 6 Metal kernel sources + Python bindings
- `kernels/planarquant_cache.py` — PlanarQuantKVCache with fused_attend()
- `kernels/calibration.py` — per-head budget calibration + redundancy-aware token selection
- `tests/` — 6-test NIAH suite

---

## References

- TheTom, **TriAttention V3**, turboquant_plus, 2026.
- Scrya, **RotorQuant**, github.com/scrya-com/rotorquant, 2026.
- ParaMind2025, **PlanarQuant/IsoQuant**, RotorQuant paper, 2026.
- Tri Dao, **Flash Decoding**, 2023.
- TCA-Attention, arXiv 2512.09238, 2025.
