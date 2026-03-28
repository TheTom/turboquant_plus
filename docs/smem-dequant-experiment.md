# SMEM Pre-Dequant Experiment — NEGATIVE RESULT

## TL;DR

Pre-dequantizing K/V tiles into threadgroup memory (SMEM) before the FA dot product loop is **2x slower** than the baseline at 8K context on M2 Pro. The threadgroup store/load overhead exceeds any benefit from avoiding constant cache stalls.

## What Was Changed

Branch: `experiment/smem-pre-dequant`

Added a new code path in the FA vec kernel gated behind `TURBO_USE_SMEM_DEQUANT`:

1. **Pre-dequant phase (K)**: Before the Q*K^T block, all 32 threads cooperatively dequant C=32 cache positions' K vectors into a threadgroup `half4` buffer. Thread `tiisg` dequants cache position `tiisg` (all DK4=32 float4s).

2. **Compute phase (K)**: The dot product reads from threadgroup memory instead of calling `deq_k_t4`.

3. **Pre-dequant phase (V)**: Before the O accumulation block, same pattern for V data. Reuses the same SMEM buffer.

4. **Compute phase (V)**: Weighted accumulation reads from threadgroup memory instead of calling `deq_v_t4`.

Threadgroup memory budget: C * max(DK, DV) * sizeof(half) = 32 * 128 * 2 = **8,192 bytes** extra. Total SMEM ~9.5KB (well within 32KB limit).

### Files Modified

- `ggml/src/ggml-metal/ggml-metal.metal` — SMEM buffer declaration + pre-dequant + compute paths
- `ggml/src/ggml-metal/ggml-metal-device.m` — `TURBO_SMEM_DEQUANT=1` env var → compile flag
- `ggml/src/ggml-metal/ggml-metal-ops.cpp` — increased SMEM allocation when enabled

## Results (M2 Pro, Qwen2.5-7B-Instruct-Q4_K_M)

| Test | Baseline (4-mag LUT) | SMEM Pre-Dequant | Delta |
|------|---------------------|------------------|-------|
| Short decode | 25.93 ± 0.09 | 26.39 ± 0.11 | +1.8% |
| 8K decode | 20.99 ± 4.87 | 10.17 ± 0.35 | **-51.5%** |

## Why It Failed

### The kernel's parallelism pattern doesn't benefit from SMEM caching

In the FA vec kernel with turbo3 dk128 (NE=1):
- 32 threads, each handles a different part of the DK=128 K vector
- Each thread dequants only DK4/NL = 1 float4 per cache position
- The dequanted value is used EXACTLY ONCE by the same thread

Adding SMEM means each dequanted value is:
1. Written to threadgroup memory (extra store)
2. Synchronized via barrier (pipeline stall)
3. Read back from threadgroup memory (extra load)

For data that's only used once by its producer, this is pure overhead.

### Total dequant calls unchanged

| | Per thread per outer iteration | Total |
|---|---|---|
| Baseline | 32 dequants (interleaved with 32 dots) | Same |
| SMEM | 32 dequants (bunched), then 32 dots | Same |

The SMEM approach adds 64 threadgroup memory ops (32 stores + 32 loads) plus a barrier, for zero reduction in constant LUT reads.

### ILP destruction

The baseline interleaves dequant + dot product, enabling instruction-level parallelism:
```
deq_k → dot(k, q) → deq_k → dot(k, q) → ...
```
The GPU overlaps constant reads with ALU operations. The SMEM approach forces sequential phases:
```
deq_k → deq_k → ... → BARRIER → dot(k,q) → dot(k,q) → ...
```

This eliminates the ILP benefit that makes the 4-mag LUT work well.

### Short context was neutral (not beneficial)

At short context, the outer loop runs ~1 iteration, so the overhead is negligible relative to model weight loading. The +1.8% is noise.

## Predicted Occupancy vs Reality

SMEM usage went from ~1.5KB to ~9.5KB (well within 32KB). Occupancy was NOT the bottleneck — the threadgroup memory access pattern was.

## How to Build and Test

```bash
# Build (Mac Mini M2 Pro)
cd ~/dev/turbo_test/llama-cpp-turbo
cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Baseline (no SMEM)
./build/bin/llama-bench -m ~/dev/turbo_test/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  -ngl 99 -fa 1 -ctk turbo3 -ctv turbo3 -t 1 -p 8192 -n 128

# SMEM enabled
TURBO_SMEM_DEQUANT=1 ./build/bin/llama-bench -m ~/dev/turbo_test/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  -ngl 99 -fa 1 -ctk turbo3 -ctv turbo3 -t 1 -p 8192 -n 128
```

## Lessons Learned

1. **SMEM only helps when data is shared between threads.** The FA vec kernel's parallelism distributes work so each thread operates on unique data — caching in SMEM adds overhead without benefit.

2. **Don't separate what the hardware pipelines together.** The constant LUT read interleaved with ALU provides ILP that the GPU exploits. Batching all reads first destroys this.

3. **The 4-mag LUT IS the dequant-level ceiling.** After 15 approaches tested (14 from decode-speed-hardware-analysis.md + this one), the conclusion is firm: the remaining 38% gap requires kernel-structural changes, not just dequant changes.

## What Would Actually Work (Hypotheses)

The remaining viable approaches from the analysis doc:

1. **Block format change**: Embed precomputed `centroid×norm` values alongside block data in device memory. Reads become sequential (non-divergent) instead of divergent constant memory. Changes the on-disk format.

2. **Fused Q·centroid attention**: Precompute a Q·centroid table (8 or 4 values per block), then each K element is a table index — replacing per-element constant reads with per-block precomputation. Requires custom FA kernel.

3. **Different quantization scheme**: A format designed from scratch for Metal's constant memory constraints (e.g., using lookup-free encoding like bit shifts instead of centroid tables).

## Approach Count

This is approach **#15** in the M2 decode speed experiments. The 4-mag LUT at 15.1 tok/s (62% of ceiling) remains the best result.
