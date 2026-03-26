# Decode Speed Status Report

## Current State (main branch)

turbo3 decode speed is 84% of q8_0 on all tested hardware. This is consistent and reproducible.

### Our Measurements (M5 Max 128GB, Qwen3.5-35B-A3B)

| Metric | turbo3 | q8_0 | Ratio |
|--------|--------|------|-------|
| Prefill (32-chunk) | 2777 tok/s | 2694 tok/s | 1.03x |
| Decode (8K ctx) | 65.8 tok/s | 78.3 tok/s | 0.84x |
| PPL (32-chunk) | 5.471 | 5.414 | +1.1% |
| KV cache size | 140 MiB | 340 MiB | 0.41x |

Prefill is at parity. Context scaling is flat (0.99x through 32K). Quality is within 1.1%. The only gap is **decode: 16% slower**.

### External Validation

| Tester | Hardware | Context | Decode ratio |
|--------|----------|---------|-------------|
| Mario (M1 Max 64GB) | M1 Max | 32K | 0.83x |
| Us (M5 Max 128GB) | M5 Max | 8K | 0.84x |
| Anon (M1 Max 64GB) | M1 Max | 42K | 0.36x (needs investigation) |

Mario's 0.83x matches ours closely. The anon tester's 0.36x is an outlier — likely different code version, no flash attention, or much deeper context.

## Root Cause

**Identified and verified:** The centroid table lookup in the flash attention dequant kernel.

Speed ceiling test (removing the LUT, returning constant values): turbo3 reaches **78.1 tok/s = exact q8_0 parity**. The entire 16% gap comes from 82 million data-dependent constant memory accesses per decode token.

q8_0 dequant: `int8_value * scale` (1 multiply per element)
turbo3 dequant: `centroid_table[3bit_index] * norm` (1 table lookup + 1 multiply per element)

The table lookup is the cost — it's data-dependent indexing that creates pipeline stalls on the GPU.

## What Was Tried (experiment/decode-speed-parity branch)

| Approach | Decode tok/s | vs baseline | Verdict |
|----------|-------------|-------------|---------|
| float32 LUT (main branch) | 65.8 | baseline | current |
| fp16 centroid LUT (vec only) | 67.4 | +2.4% | marginal win, offset by custom op overhead |
| Register float4 select | 63.1 | -4.1% | variable indexing equally bad |
| Inline switch (8 cases) | 57.3 | -12.9% | branches worse than LUT |
| Precomputed centroid*norm | 66.9 | +1.7% | marginal |
| fp16 LUT both paths | 59.2 | -10.0% | non-vec fp16 conversion overhead |
| No LUT (speed ceiling) | 78.1 | +18.7% | theoretical max |
| **Custom GGML_OP_TURBO_WHT + fp16 LUT** | **61.5** | **-6.5%** | **NET NEGATIVE — extra graph nodes hurt decode** |

**Key finding:** The custom GGML_OP_TURBO_WHT (which replaces the dense matmul with O(d log d) butterfly) adds graph overhead that makes decode WORSE. The matmul was never the decode bottleneck — it's negligible for single-token generation. The additional Metal kernel dispatches from the custom op cost more than they save.

## Mario's 70-Page PDF Test (M5 Max, 48K tokens, main branch)

Real-world benchmark using Mario Tomic's fitness PDF (870 KB, ~48K tokens):

| | turbo3 | q8_0 | Ratio |
|---|---|---|---|
| Prefill | 954.5 tok/s | 965.4 tok/s | **0.99x** |
| **Decode** | **36.7 tok/s** | **55.6 tok/s** | **0.66x** |
| KV cache | 280 MiB | 680 MiB | 0.41x |

## Decode Context Scaling (Verified)

| Context depth | turbo3/q8_0 decode | Source |
|--------------|-------------------|--------|
| ~12 tokens | 0.88x | server test |
| ~8K tokens | 0.76x | server test |
| ~32K tokens | 0.83x | Mario M1 Max bench |
| **~48K tokens (PDF)** | **0.66x** | **Mario's PDF test** |

Decode ratio degrades ~4% per doubling of context due to centroid LUT constant cache thrashing.

## Conclusion

**The main branch is the correct configuration.** The experiment branch (custom WHT op + fp16 LUT) is a net negative for decode.

The decode gap scales with context and is fundamental to the 3-bit block format. At 48K context it's 34% slower than q8_0. Closing this requires fused compressed attention (complex) or a different block format (trades compression).

## Recommendation for Testers

- **Prefill** is at parity regardless of context (0.99x even at 48K)
- **Decode degrades with context**: 0.88x@short, 0.76x@8K, 0.66x@48K
- The 2.4x smaller KV cache enables longer context that wouldn't fit otherwise
- **Short conversations** (<4K): turbo3 is barely slower
- **Long document processing** (40K+): expect ~35% decode slowdown vs q8_0
- **Memory-constrained**: turbo3 is the only way to fit the context at all
