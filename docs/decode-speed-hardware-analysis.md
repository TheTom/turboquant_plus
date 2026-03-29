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

## Extended Experiments (approaches 14, 2026-03-27)

| # | Approach | M2 8K tok/s | vs 4-mag | Key finding |
|---|----------|-------------|---------|-------------|
| 14 | Fused block dot (per-centroid Q accum) | 8.1 | -46% | 64 float comparisons per block — worst result of all |

### Fused block dot details
Flipped computation: instead of per-element centroid lookup, iterate over 4 centroid values and accumulate matching Q elements. Uses `float(mi == c)` as a branchless mask. But 4 centroids × 4 elements × 4 comparisons = 64 float comparison operations per dequant call. Each comparison likely compiles to a branch on Apple8, making this the most expensive approach tested.

### Final tally: 14 approaches tested

| Rank | Approach | M2 8K | vs Ceiling | Category |
|------|----------|-------|-----------|----------|
| 1 | 4-mag LUT | 15.1 | 62% | 4 constant reads |
| 2 | simd_shuffle | 14.7 | 60% | Cross-lane transfer |
| 3 | Batched extract (8-LUT) | 13.7 | 56% | 8 constant reads |
| 4 | Inline FA block | 13.5 | 55% | Inlined dequant |
| 5 | Deferred norm | 12.9 | 53% | Loses ILP |
| 6 | 2-pair half2 | 12.0 | 49% | 2 reads + ternary |
| 7 | Select chain | 11.9 | 49% | Pure branches |
| 8 | Bit-arithmetic | 11.6 | 47% | Pure ALU |
| 9 | FMA branchless | 11.4 | 47% | fma() chain |
| 10 | Main (8-LUT) | 10.95 | 45% | Baseline |
| 11 | Named-reg ternary | 10.3 | 42% | Registers + branches |
| 12 | Non-vec FA | 10.2 | 42% | Wrong kernel |
| 13 | Fused block dot | 8.1 | 33% | 64 comparisons |
| — | Ceiling (no dequant) | 24.5 | 100% | Zero overhead |

## Extended Experiments (approach 15, 2026-03-28)

| # | Approach | M2 8K tok/s | vs 4-mag | Key finding |
|---|----------|-------------|---------|-------------|
| 15 | SMEM pre-dequant (threadgroup memory tile) | 10.17 | -33% | Threadgroup store/load overhead > constant cache savings |

### SMEM pre-dequant details
Pre-dequantize entire K/V tiles (C=32 × DK=128 = 8KB half) into threadgroup memory before dot products. All 32 threads cooperatively dequant C cache positions, barrier, then compute from SMEM.

**Why it failed**: The FA vec kernel distributes work so each thread operates on unique data (DK4/NL=1 dequant per thread per cache position). Each dequanted value is used exactly once by its producer thread. Caching in SMEM adds 64 threadgroup memory ops (32 stores + 32 loads) per thread per outer iteration + barrier, for zero reduction in constant LUT reads. Additionally, separating dequant from compute destroys the ILP that makes the 4-mag LUT work (constant reads overlapped with ALU). At short context, the overhead is negligible (+1.8%), but at 8K it's catastrophic (-51.5%).

**Lesson**: SMEM only helps when data is shared between threads. Don't separate what the hardware pipelines together.

### Updated tally: 16 approaches tested

| Rank | Approach | M2 8K | vs Ceiling | Category |
|------|----------|-------|-----------|----------|
| 1 | 4-mag LUT | 15.1 | 62% | 4 constant reads |
| 2 | simd_shuffle | 14.7 | 60% | Cross-lane transfer |
| 3 | Batched extract (8-LUT) | 13.7 | 56% | 8 constant reads |
| 4 | Inline FA block | 13.5 | 55% | Inlined dequant |
| 5 | Deferred norm | 12.9 | 53% | Loses ILP |
| 6 | 2-pair half2 | 12.0 | 49% | 2 reads + ternary |
| 7 | Select chain | 11.9 | 49% | Pure branches |
| 8 | Bit-arithmetic | 11.6 | 47% | Pure ALU |
| 9 | FMA branchless | 11.4 | 47% | fma() chain |
| 10 | Main (8-LUT) | 10.95 | 45% | Baseline |
| 11 | Named-reg ternary | 10.3 | 42% | Registers + branches |
| 12 | Non-vec FA | 10.2 | 42% | Wrong kernel |
| 13 | SMEM pre-dequant | 10.17 | 41% | Threadgroup cache (ILP loss) |
| 14 | Q·centroid precompute | 10.10 | 41% | select() register LUT |
| 15 | Fused block dot | 8.1 | 33% | 64 comparisons |
| — | Ceiling (no dequant) | 24.5 | 100% | Zero overhead |

### Critical ILP insight (2026-03-28)

The 2x slowdown of SMEM pre-dequant revealed WHY the 4-mag LUT works: **interleaved constant reads and ALU provide instruction-level parallelism.** The GPU overlaps the next constant read while the current dot product + norm multiply executes. Any approach that separates memory reads from compute phases loses this overlap and runs at 2x the latency.

This explains the ranking pattern: approaches that maintain per-element `read → ALU` interleaving (4-mag, simd_shuffle, batched 8-LUT) outperform approaches that batch reads (SMEM, deferred norm, QC precompute) or eliminate reads but add more ALU (select chain, FMA, bit-arithmetic).

**The remaining 38% gap cannot be closed by rearranging the same reads and ALU.** The only path forward is reducing the total constant read count per element below 4, which requires changing the block format or computation structure.

## Extended Experiments (approaches 16h + 19, 2026-03-28 evening)

Clean benchmarks (no DINOv2 contention). Note: absolute tok/s numbers differ from earlier runs due to DINOv2 GPU contention during those measurements. Relative rankings are consistent.

### Qwen2.5-7B-Instruct-Q4_K_M (8K context, p=8192 tg128)

| # | Approach | Env Var | turbo3 t/s | turbo4 t/s | q8_0 t/s | Key finding |
|---|----------|---------|-----------|-----------|---------|-------------|
| — | Baseline (4-mag LUT) | — | 25.88 | 27.59 | 33.69 | turbo4 > turbo3 by 6.6% |
| 16h | Half register LUT (`half cn[8]`) | TURBO_HALF_REG_LUT=1 | 25.55 (-1.3%) | 27.26 (-1.2%) | — | Still spills on Apple8. Noise. |
| 19 | Threadgroup centroid cache | TURBO_TG_CENTROID=1 | 25.96 (+0.3%) | 27.42 (-0.6%) | — | Flat. Threadgroup read ≈ constant read. |

### Qwen2.5-1.5B-Instruct-Q4_K_M (8K context, p=8192 tg128)

| # | Approach | Env Var | turbo3 t/s | turbo4 t/s | q8_0 t/s | Key finding |
|---|----------|---------|-----------|-----------|---------|-------------|
| — | Baseline (4-mag LUT) | — | 51.84 | 58.74 | 113.02 | Gap worse on 1.5B (attention dominates) |
| 16h | Half register LUT | TURBO_HALF_REG_LUT=1 | 50.67 (-2.3%) | 57.76 (-1.7%) | — | Consistent regression across models |
| 19 | Threadgroup centroid cache | TURBO_TG_CENTROID=1 | 52.19 (+0.7%) | 57.98 (-1.3%) | — | Flat |

### Prompt processing (pp8192, for reference)

| Model | turbo3 | turbo4 | q8_0 |
|-------|--------|--------|------|
| 7B | 241.26 | 240.00 | 254.74 |
| 1.5B | 816.13 | 807.03 | 896.42 |

### Key takeaways from approaches 16h + 19

1. **Both approaches are noise on M2 Pro.** Half Reg LUT is consistently -1 to -2% (regression). TG Centroid is flat (within variance).

2. **turbo4 > turbo3 on M2 Pro** by 6-13% on decode. Holds across both model sizes. 16 centroids (turbo4) is faster than 8 (turbo3) — possibly better ILP with 4-bit indices.

3. **The gap to q8_0 grows with smaller models**: 30% on 7B, 118% on 1.5B. Smaller models make attention a larger fraction of decode, making the dequant overhead more visible.

4. **The centroid LUT is NOT the sole bottleneck.** Moving centroids to half-precision registers or threadgroup memory doesn't help. The bottleneck is broader — likely the full WHT transform + extraction pipeline, not just the final centroid lookup.

### Revised theory (2026-03-28)

The original theory was "constant memory LUT divergence is the bottleneck." After testing 4 approaches targeting the centroid LUT specifically (#15 SMEM pre-dequant, #16q Q·centroid precompute, #16h half reg LUT, #19 TG centroid), NONE improved decode speed. The 4-mag LUT improvement from earlier was real (+38%), but it was optimizing the full dequant pipeline (fewer reads + XOR sign trick), not just the centroid lookup.

The remaining gap is structural: turbo3/4 dequant requires WHT extraction + centroid lookup + norm multiply per element, while q8_0 is a simple `int8 * scale`. No rearrangement of the same operations will close this gap.

### Updated tally: 18 approaches tested (16h and 19 added)

| Rank | Approach | M2 8K | vs Ceiling | Category |
|------|----------|-------|-----------|----------|
| 1 | 4-mag LUT | 15.1 | 62% | 4 constant reads |
| 2 | simd_shuffle | 14.7 | 60% | Cross-lane transfer |
| 3 | Batched extract (8-LUT) | 13.7 | 56% | 8 constant reads |
| 4 | Inline FA block | 13.5 | 55% | Inlined dequant |
| 5 | Deferred norm | 12.9 | 53% | Loses ILP |
| 6 | 2-pair half2 | 12.0 | 49% | 2 reads + ternary |
| 7 | Select chain | 11.9 | 49% | Pure branches |
| 8 | Bit-arithmetic | 11.6 | 47% | Pure ALU |
| 9 | FMA branchless | 11.4 | 47% | fma() chain |
| 10 | Main (8-LUT) | 10.95 | 45% | Baseline |
| 11 | Named-reg ternary | 10.3 | 42% | Registers + branches |
| 12 | Non-vec FA | 10.2 | 42% | Wrong kernel |
| 13 | SMEM pre-dequant | 10.17 | 41% | Threadgroup cache (ILP loss) |
| 14 | Q·centroid precompute | 10.10 | 41% | select() register LUT |
| 15 | Fused block dot | 8.1 | 33% | 64 comparisons |
| 16h | Half register LUT | ~noise | ~62% | half cn[8] — still spills |
| 19 | TG centroid cache | ~noise | ~62% | Threadgroup centroid table |
| — | Ceiling (no dequant) | 24.5 | 100% | Zero overhead |

## Extended Experiments (approaches 20 + 22, 2026-03-28 night)

### Qwen2.5-1.5B-Instruct-Q4_K_M (8K context, p=8192 tg128)

| # | Approach | Env Var | turbo3 t/s | turbo4 t/s | Key finding |
|---|----------|---------|-----------|-----------|-------------|
| — | Baseline (4-mag LUT) | — | 51.80 | 58.60 | — |
| 20 | 2-bit direct encode (pure ALU, no LUT) | TURBO_DIRECT_ENCODE=1 | 52.16 (+0.7%) | 57.78 (-1.4%) | **Same speed with zero constant reads** |
| 22 | Async prefetch (batch device reads) | TURBO_ASYNC_PREFETCH=1 | 50.73 (-2.1%) | 57.02 (-2.7%) | GPU already interleaves reads optimally |

### Definitive proof: constant memory LUT is FREE on M2 Pro

Approach #20 replaced the entire centroid LUT with pure ALU (`norm * (0.25 + 0.5*idx)`) — zero constant memory reads. The result is identical speed. This means the 4 constant reads per element are hitting L1 cache and costing essentially nothing.

The bottleneck is **device memory bandwidth**: streaming 14 bytes per 32 elements (turbo3) from DRAM, strided across cache positions. q8_0 streams 34 bytes per 32 elements but with a simpler access pattern and trivial dequant (`int8 * scale`).

### Why async prefetch didn't help (#22)

Staging device memory reads (norm, qs, signs) into registers before constant reads forced a specific execution order. The GPU's out-of-order execution already interleaves these optimally. Forcing order adds register pressure without hiding latency.

### Updated tally: 20 approaches tested

| Rank | Approach | M2 8K | vs Ceiling | Category |
|------|----------|-------|-----------|----------|
| 1 | 4-mag LUT | 15.1 | 62% | 4 constant reads |
| 2 | simd_shuffle | 14.7 | 60% | Cross-lane transfer |
| 3 | Batched extract (8-LUT) | 13.7 | 56% | 8 constant reads |
| 4 | Inline FA block | 13.5 | 55% | Inlined dequant |
| 5 | Deferred norm | 12.9 | 53% | Loses ILP |
| 6 | 2-pair half2 | 12.0 | 49% | 2 reads + ternary |
| 7 | Select chain | 11.9 | 49% | Pure branches |
| 8 | Bit-arithmetic | 11.6 | 47% | Pure ALU |
| 9 | FMA branchless | 11.4 | 47% | fma() chain |
| 10 | Main (8-LUT) | 10.95 | 45% | Baseline |
| 11 | Named-reg ternary | 10.3 | 42% | Registers + branches |
| 12 | Non-vec FA | 10.2 | 42% | Wrong kernel |
| 13 | SMEM pre-dequant | 10.17 | 41% | Threadgroup cache (ILP loss) |
| 14 | Q·centroid precompute | 10.10 | 41% | select() register LUT |
| 15 | Fused block dot | 8.1 | 33% | 64 comparisons |
| 16h | Half register LUT | ~noise | ~62% | half cn[8] — still spills |
| 19 | TG centroid cache | ~noise | ~62% | Threadgroup centroid table |
| 20 | Direct encode (no LUT) | ~noise | ~62% | Pure ALU, zero constant reads |
| 22 | Async prefetch | -2% | ~61% | Forced read ordering |
| — | Ceiling (no dequant) | 24.5 | 100% | Zero overhead |

## SMEM Pre-Dequant Re-Run (2026-03-28 night, corrected)

The original -51% SMEM result was measured during DINOv2 GPU contention. Clean re-run shows it's actually a small positive:

### Qwen2.5-1.5B Q4_K_M (8K context)

| Config | SMEM t/s | Baseline t/s | Delta |
|--------|---------|-------------|-------|
| turbo3 short | 53.96 | 51.84 | **+4.1%** |
| turbo3 8K | 54.13 | 51.84 | **+4.4%** |
| turbo4 short | 60.20 | 58.74 | **+2.5%** |
| turbo4 8K | 60.37 | 58.74 | **+2.8%** |

### Qwen2.5-7B Q4_K_M (8K context)

| Config | SMEM t/s | Baseline t/s | Delta |
|--------|---------|-------------|-------|
| turbo3 8K | 26.17 | 25.88 | +1.1% |
| turbo4 8K | 26.31 | 27.59 | **-4.6%** |

### Analysis: SMEM helps small models, hurts large turbo4

The pattern is clear: SMEM pre-dequant trades constant memory pressure for threadgroup memory + barriers. This helps when:
- **Model is small** (attention is larger fraction of decode → dequant matters more)
- **KV format is turbo3** (8 centroids = more constant reads per element to avoid)

It hurts when:
- **Model is large** (register pressure from SMEM allocation → spills)
- **KV format is turbo4 on 7B** (16 centroids = larger pre-dequant tile → more SMEM overhead)

This suggests a **model-size-adaptive dispatch**: use SMEM pre-dequant for small models (≤3B), 4-mag LUT for large models (≥7B). The crossover is somewhere in between.

## SMEM Pre-Dequant on M5 Max — CATASTROPHIC (2026-03-28 night)

Tested SMEM on M5 Max (Apple10) with Qwen3.5-35B-A3B MoE. Opposite of M2 — massive regression.

### Qwen3.5-35B-A3B Q8_0, M5 Max, decode (tg128 t/s)

| Context | turbo3 baseline | turbo3 SMEM | turbo4 baseline | turbo4 SMEM | q8_0 |
|---------|----------------|-------------|-----------------|-------------|------|
| short | 78.47 | **18.39 (-77%)** | 80.40 | **16.93 (-79%)** | 85.48 |
| 16K | 78.64 | **16.47 (-79%)** | 79.45 | **15.58 (-80%)** | 78.09 |
| 32K | 78.17 | **16.19 (-79%)** | 80.64 | **16.29 (-80%)** | 86.83 |

### Why M5 is opposite of M2

- **M5's constant cache (Apple10) is already fast** — handles 8-way divergent reads with minimal penalty. SMEM replaces something that's nearly free with something that costs threadgroup stores + barriers + loads.
- **SMEM kills occupancy** — the threadgroup memory allocation reduces the number of active threadgroups per GPU core. On M5's larger kernels (35B MoE), this is devastating.
- **M2 benefited because its constant cache is slower** — the tradeoff (SMEM overhead vs slow constant reads) was marginally positive on M2 for small models. On M5, the constant cache is fast enough that the SMEM overhead dominates completely.

### Verdict: SMEM is dead for M5

Do NOT ship SMEM pre-dequant for any M5 configuration. The +4% on M2 small models is not worth the complexity or the risk of M5 regression. SMEM pre-dequant is a net negative optimization.

## Final Conclusion: M2 Decode Ceiling (2026-03-28)

**20 approaches tested + 1 corrected re-run. The 4-mag LUT is the M2 ceiling for large models. SMEM pre-dequant adds ~2-4% for small models.**

The bottleneck is NOT:
- Constant memory LUT divergence (proven by #20: zero LUT = same speed)
- Centroid lookup specifically (proven by #16h, #19: different storage = same speed)
- Read ordering (proven by #22: forced ordering = worse)
- Branch overhead (proven by #6-9: branchless = worse)

The bottleneck IS:
- **Device memory bandwidth** for streaming quantized KV blocks from DRAM
- **Dequant ALU complexity** (WHT extraction + centroid + norm vs q8_0's simple `int8 * scale`)
- These two costs are inherent to the turbo format and cannot be optimized away without changing the format itself
- For small models where attention dominates, SMEM can squeeze out ~4% by trading constant reads for threadgroup reads

### Remaining untested approaches (low priority)

| # | Approach | Category | Expected Impact | Status |
|---|----------|----------|----------------|--------|
| 17 | Device-memory centroid×norm | Block format change | Moot — centroid reads are free | NOT TESTED |
| 18 | Byte-indexed 256-entry LUT | LUT restructure | Moot — same reason | NOT TESTED |
| 21 | Hybrid 4-mag + simd_shuffle | Combined | Low — simd_shuffle was only -2.6% standalone | NOT TESTED |

## M5 Max Long-Context Discovery (2026-03-27)

### The constant cache bottleneck hits M5 Max too at long context

| Depth | Ceiling | Reads only | Full dequant | q8_0 | LUT cost | Ceiling vs q8_0 |
|-------|---------|-----------|-------------|------|----------|----------------|
| 8K | 78.9 | 75.2 | 59.2 | 78.8 | 20% | 1.00x |
| 16K | 75.9 | 74.7 | 58.7 | 72.0 | 21% | 1.05x |
| 32K | 78.3 | 71.9 | 47.1 | 61.0 | **34%** | **1.28x** |

**turbo3 with zero dequant is 28% FASTER than q8_0 at 32K on M5 Max.** The compressed cache bandwidth advantage grows with context. The LUT cost explodes from 20% to 34% as context grows.

### 4-mag vs 8-LUT on M5 Max across context depths

| Depth | q8_0 | 8-LUT | 4-mag | 4-mag vs 8-LUT |
|-------|------|-------|-------|----------------|
| short | 85.0 | 76.7 | 76.2 | -0.7% |
| 16K | 72.0 | 58.9 | **60.3** | **+2.4%** |
| 32K | 61.0 | 47.6 | 44.1 | -7.3% |

4-mag helps at 16K (+2.4%) but hurts at 32K (-7.3%) on M5. Crossover around 20K.

### Context-adaptive dispatch (planned)
Compile both 4-mag and 8-LUT FA kernel instantiations. At dispatch time, select based on KV cache size:
- Pre-M5 (no tensor API): always 4-mag
- M5+ with context < 8K: 8-LUT (minimal cache pressure)
- M5+ with 8K-20K context: 4-mag (moderate pressure, 4-mag helps)
- M5+ with context > 20K: 8-LUT (fully thrashed, ALU overhead dominates)

### The real prize
If we can reduce dequant cost from 34% to ~10% at 32K, turbo3 decode would be **FASTER than q8_0** at long context. The bandwidth advantage (28% at 32K) far exceeds the dequant overhead — we just need to close the gap. This flips the narrative from "turbo3 is slower but smaller" to "turbo3 is faster AND smaller."
