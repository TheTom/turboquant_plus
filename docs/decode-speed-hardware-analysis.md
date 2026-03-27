# TurboQuant Decode Speed: Why Some Hardware Struggles

## The Problem

turbo3 decode speed varies dramatically across Apple Silicon generations:

| Hardware | Decode vs q8_0 (short) | Decode vs q8_0 (16K) | Has Tensor API |
|----------|----------------------|---------------------|---------------|
| M5 Max | 0.89x | ~0.82x | yes |
| M2 Pro | 0.73x | 0.67x (with 4-mag fix) | no |
| M1 Max | 0.84x | 0.45x (reported) | no |

M5 Max has near-parity. M2/M1 fall off a cliff at long context. This doc explains why and what we did about it.

## Root Cause: Constant Memory LUT Divergence

turbo3 dequant uses an 8-entry centroid lookup table (LUT) in Metal `constant` memory. Each element's value is determined by a 3-bit index (2-bit qs + 1-bit sign), so each thread in a 32-thread simdgroup may access a different LUT entry.

This "divergent access" pattern is where hardware generations differ:

- **M5 Max (Apple10)**: Efficient constant cache handles 8-way divergent access with minimal penalty. LUT costs ~14% of decode time.
- **M2 Pro (Apple8)**: Older constant cache serializes divergent access. LUT costs ~25% of decode time — **2x worse than M5**.
- **M1 Max (Apple7)**: Even worse, likely 30%+ LUT cost based on reported numbers.

## How We Know: Profiling Isolation

We added `TURBO_PROFILE_MODE` (0-4) to strip away dequant layers one at a time:

| Mode | What | M5 (% of ceiling) | M2 (% of ceiling) |
|------|------|-------------------|-------------------|
| 1 | No dequant at all | 78.9 (100%) | 24.5 (100%) |
| 2 | + read norm only | 75.1 (95%) | 22.1 (90%) |
| 4 | + read all bytes | 75.2 (95%) | 21.9 (89%) |
| 3 | + qs extraction + LUT | 64.9 (82%) | 16.4 (67%) |
| 0 | + signs + full LUT | 59.2 (75%) | 14.0 (57%) |

**Key finding: No-dequant turbo3 is 12% FASTER than q8_0 on M2 Pro** (24.5 vs 21.9) because the compressed cache moves less data over the memory bus. The compression FORMAT is not the problem. The dequant COMPUTATION is.

**The LUT accounts for:**
- M5: 13.7% of ceiling (Mode 4 → Mode 3)
- M2: 25.1% of ceiling — **2x worse**

Reading the bytes (norm + qs + signs) costs ~10% on both chips. That's not the bottleneck.

## What We Tried (7 Approaches)

| # | Approach | M2 8K tok/s | vs Main | Why it works/doesn't |
|---|----------|-------------|---------|---------------------|
| 1 | **4-mag LUT + XOR sign** | **15.1** | **+38%** | **Halves constant addresses (4 vs 8). Sweet spot on M2.** |
| 2 | Batched byte extract | 13.7 | +25% | Better byte reading, still 8 LUT addresses |
| 3 | Deferred norm multiply | 12.9 | +18% | Loses ILP — per-element norm hides LUT latency |
| 4 | 2-pair half2 LUT | 12.0 | +10% | Ternary overhead exceeds savings from 2 addresses |
| 5 | Select chain (zero LUT) | 11.9 | +9% | Too much ALU — branches on M2 GPU |
| 6 | Bit-arithmetic | 11.6 | +6% | Pure ALU, zero memory — but ALU cost too high |
| 7 | Non-vec FA (nl=2) | 10.2 | -7% | Non-vec kernel not designed for single-token decode |

Also tested on M5 Max (no help needed):
- float cn[8] registers: 75.2 (spills to stack on Metal)
- half cn[8] registers: 74.4 (also spills)
- Split 2×4 half LUT: 74.0 (branch overhead)

## The Fix: Auto-Detection

The build auto-detects hardware at Metal library compile time:

```
M1/M2/M3/M4 (has_tensor=false) → TURBO_USE_4MAG=1 → 4-entry magnitude LUT
M5+          (has_tensor=true)  → TURBO_USE_4MAG=0 → 8-entry full LUT
```

Each chip gets its optimal dequant path. No user configuration needed.

### How 4-mag LUT works

The 8 centroids have structure: 4 magnitudes × 2 signs. We split the 3-bit index:
- **2-bit qs** → selects magnitude from 4-entry LUT (4 possible constant addresses)
- **1-bit sign** → reverses magnitude index via XOR, then negates via ALU multiply
- **Norm** → multiplied per-element (provides ALU work that hides constant memory latency)

The XOR trick: for negative values (sign=0), the magnitude index is reversed (qs=0 → highest magnitude). `qs ^ 0x3` flips the 2-bit index without a branch.

### Critical finding: On M2, branches cost MORE than divergent constant reads

We tested 9 approaches total. The pattern is clear:

| Approach | M2 8K | Constant reads | Branches | Result |
|----------|-------|---------------|----------|--------|
| **4-mag LUT** | **15.1** | **4 divergent** | **0** | **BEST** |
| Batched extract (8-LUT) | 13.7 | 8 divergent | 0 | Good |
| Deferred norm (4-mag) | 12.9 | 4 divergent | 0 | Lose ILP |
| 2-pair half2 + ternary | 12.0 | 2 divergent | 4 | Branches hurt |
| Select chain (zero LUT) | 11.9 | 0 | 8 | Branches kill |
| Bit-arithmetic | 11.6 | 0 | 4+ | ALU too heavy |
| Named-reg + ternary select | 10.3 | 4 uniform | 8 | Worst — ternary kills |
| Main (8-entry LUT) | 10.95 | 8 divergent | 0 | Baseline |
| Inline block (FA inner loop) | 13.5 | 4 divergent | 0 | I-cache pressure |
| Non-vec FA (nl=2) | 10.2 | 8 divergent | 0 | Wrong kernel for decode |

**Any approach that replaces constant reads with branches loses on M2.** The Apple8 GPU's branch predictor/execution is worse than its constant cache. The 4-mag LUT succeeds because it reduces constant addresses (4 vs 8) WITHOUT adding branches.

### Per-element norm multiply is faster than deferred

Deferring `float4 * norm` at the end (12.9 tok/s) is slower than per-element `v * norm` (15.1 tok/s). The per-element multiply provides ALU work that hides constant memory latency via instruction-level parallelism. The GPU can overlap the next constant read while the current multiply executes.

## Why CUDA Doesn't Have This Problem

@spiritbuun's CUDA fork achieves 97.5% of q8_0 decode. The key difference:

- **CUDA has 255 registers per thread** — cn[8] fits in registers without spilling
- **Metal has a smaller register file** — cn[8] spills to thread stack memory
- **CUDA warp semantics** — better divergent access handling across 32 threads
- **Metal simdgroup semantics** — constant cache serializes more on older chips

The register LUT approach works perfectly on CUDA but is fundamentally incompatible with Metal's register file on current hardware.

## Remaining Gap

Even with 4-mag LUT, M2 Pro decode at 8K is 15.1 vs 24.5 ceiling (62%). The remaining 38% gap needs kernel-level changes:

1. **SMEM pre-dequant**: Pre-dequant entire KV blocks into threadgroup memory before dot products. Would eliminate constant memory from the inner loop entirely. Requires modifying the FA vec kernel template.

2. **Per-group device-memory cached centroid×norm**: Store 8 pre-computed centroid×norm values per 128-element group alongside the block data. Dequant reads from device memory (sequential) instead of constant memory (divergent). Changes the block format.

3. **Fused Q·centroid attention**: Compute attention scores directly from quantized indices without full dequantization. Precompute Q·centroid table (8 values per block), then each K element is a table lookup. Complex (custom FA kernel).

These are tracked in GitHub issue #39.

## Summary

| What | Result |
|------|--------|
| Root cause | Constant memory LUT divergence, 2x worse on M2 vs M5 |
| Best fix found | 4-mag LUT (+38-45% on M2, auto-detected) |
| M5 impact | Zero regression (uses separate code path) |
| Remaining gap | 38% to ceiling, needs kernel-level surgery |
| CUDA comparison | buun gets 97.5% via register LUT (not portable to Metal) |

## Complete Experiment Log (12 approaches, 2026-03-26/27)

### M2 Pro (Apple8, has_tensor=false) — 8K context decode

| # | Approach | tok/s | vs q8_0 | vs Ceiling | Key finding |
|---|----------|-------|---------|-----------|-------------|
| — | No-op ceiling | 24.5 | 1.12x | 100% | turbo3 format FASTER than q8_0 (less bandwidth) |
| **1** | **4-mag LUT + XOR sign** | **15.1** | **0.69x** | **62%** | **Sweet spot: 4 constant addrs, 0 branches** |
| 2 | Batched byte extract (8-LUT) | 13.7 | 0.63x | 56% | Better byte reading, still 8 addresses |
| 3 | Inline block in FA loop | 13.5 | 0.62x | 55% | I-cache pressure from expanded inline |
| 4 | Deferred norm (float4 * norm) | 12.9 | 0.59x | 53% | Loses ILP — norm multiply hides LUT latency |
| 5 | 2-pair half2 + ternary | 12.0 | 0.55x | 49% | Ternary overhead exceeds 2-addr savings |
| 6 | Select chain (zero LUT) | 11.9 | 0.54x | 49% | 8 ternaries = 8 branches on Apple8 |
| 7 | Bit-arithmetic (mul+add) | 11.6 | 0.53x | 47% | 7 ALU > 4 constant reads |
| 8 | FMA branchless (zero ternary) | 11.4 | 0.52x | 47% | fma doesn't help — same ALU count |
| — | Main (8-entry constant LUT) | 10.95 | 0.50x | 45% | Baseline — 8-way divergent |
| 9 | Named-reg ternary select | 10.3 | 0.47x | 42% | 4 uniform reads + 8 branches = worst |
| 10 | Non-vec FA forced (nl=2) | 10.2 | 0.47x | 42% | Non-vec kernel wrong for single-token decode |

### M5 Max (Apple10, has_tensor=true) — short context decode

All approaches: 75-77 tok/s (M5 uses 8-LUT path, unaffected by TURBO_USE_4MAG).
No regression from any experiment. q8_0 baseline: 85 tok/s.

### Additional checks
- **Qwen2 attention bias**: NOT present in Qwen2.5-7B-Instruct model (339 tensors, no attn_q/k/v.bias). The GÖKYILDIZ bug does not apply.
- **Model-independence**: Profiling pattern (LUT = 25% cost on M2) is consistent across Qwen2.5-7B (M2 Pro) and Qwen3.5-35B-A3B (M5 Max). Architecture-independent.
- **Non-vec FA (nl=2)**: Faster on M5 (+1.7%) but much worse on M2 (-7%). The non-vec kernel processes batch=1 inefficiently at long context.

### Hardware truth (Apple8/M2 Pro)
1. **1 divergent constant read < 7 ALU ops** — even with fma()
2. **Metal compiles ternaries to branches** — not predicated moves like CUDA
3. **Branches cost MORE than divergent constant reads** — the opposite of CUDA
4. **Array indexing ALWAYS spills** — Metal's register file is too small for cn[4+]
5. **4 constant addresses is the sweet spot** — fewer adds branches, more adds thrashing
6. **Per-element norm multiply provides ILP** — hides constant memory latency

### Papers reviewed for novel approaches
- AttentionPack (arxiv 2603.23914): validates kernel fusion for attention-aware decompression
- GlowQ (arxiv 2603.25385): validates group-shared factor caching (+37.4% throughput)
- Embedding Compression via Spherical Coordinates (2602.00079): same PolarQuant framework, different encoding
- MKA: Memory-Keyed Attention (2603.20587): learned memory lookup, hardware-aware
- Scaling Attention via Feature Sparsity (2603.22300): skip negligible attention weights

### Next frontier
The 4-mag LUT is the dequant-level ceiling. The remaining 38% gap requires:
1. **Block format change**: embed precomputed centroid×norm in device memory (sequential reads, zero divergence)
2. **Custom FA kernel**: fuse dequant into attention with restructured computation
3. **Different quantization scheme**: format designed for Metal's constraints from scratch

## Extended Experiments (approaches 12-13, 2026-03-27)

| # | Approach | M2 8K tok/s | vs 4-mag | Key finding |
|---|----------|-------------|---------|-------------|
| 12 | FMA branchless (zero ternary, zero memory) | 11.4 | -24% | 7 ALU ops slower than 4 constant reads even with fma() |
| 13 | simd_shuffle cross-lane magnitude select | 14.7 | -2.6% | Shuffle latency ≈ constant read latency on Apple8 |

### FMA branchless details
Fully branchless: XOR mask via `3 - 3*sign_bit` (not ternary), sign via `2*s - 1` (not ternary), magnitude via 3-chained `fma()`. Zero branches, zero memory. Still slower because 7 ALU cycles > 1 divergent constant read cycle on Apple8.

### simd_shuffle details
Threads 0-3 within each 8-thread block compute mag[i]×norm. All threads use `simd_shuffle(value, block_base + mi)` to read the correct mag×norm. This IS branchless and memory-free — the value moves via cross-lane register transfer. But shuffle latency on Apple8 is comparable to constant cache access, negating the benefit.

### Total: 13 approaches tested
The 4-mag constant LUT at 15.1 tok/s (0.69× q8_0) is the definitive dequant-level ceiling on Apple8 hardware. Every alternative — fewer constant addresses, zero constant addresses, branchless ALU, cross-lane shuffle, inline FA blocks — is equal or worse.
