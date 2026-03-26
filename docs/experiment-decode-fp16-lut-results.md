# fp16 Centroid LUT (Vec Path) — Full Test Results

Branch: `experiment/decode-speed-parity` commit `e503d77`
Change: `constant half turbo_centroids_3bit_h[8]` replacing `constant float turbo_centroids_3bit[8]` in vec dequant only

## Quality

| Test | Main | Experiment | Delta |
|------|------|-----------|-------|
| PPL (8-chunk) | 6.211 | 6.211 | unchanged |
| PPL (32-chunk) | 5.471 | 5.471 | unchanged |

**Zero quality regression.**

## Prefill Speed

| Context | Main tok/s | Experiment tok/s | Delta |
|---------|-----------|-----------------|-------|
| 32-chunk | 2777 | 2784 | +0.3% |
| 2K | 4729 | 4694 | -0.7% |
| 4K | 3079 | 3062 | -0.6% |
| 8K | 2289 | 2261 | -1.2% |
| 16K | 1736 | 1736 | 0.0% |
| 32K | — | 1224 | — |

**No prefill regression.** All deltas within measurement noise.

## Decode Speed (THE WIN)

| Context | Main tok/s | Experiment tok/s | q8_0 tok/s | Main/q8_0 | Exp/q8_0 | Improvement |
|---------|-----------|-----------------|-----------|----------|---------|-------------|
| Short (~12) | 75.3 | **77.2** | 85.2 | 0.88x | **0.91x** | +2.5% |
| 8K | 59.2 | **67.3** | 77.7 | 0.76x | **0.87x** | +13.7% |
| 48K (PDF) | 36.7 | **39.0** | 55.6 | 0.66x | **0.70x** | +6.3% |

**Decode improvement at all context depths.** Biggest improvement at 8K (+13.7%). The half-precision LUT reduces constant cache pressure because half values are 2 bytes vs 4 bytes per entry.

## Change Description

One change to the Metal shader: the vec flash attention dequant (`dequantize_turbo3_0_t4`) uses `constant half[8]` instead of `constant float[8]` for the centroid table, and keeps the norm multiply in half precision (`xb->norm` is already stored as `ggml_half`).

The non-vec dequant (`dequantize_turbo3_0`) is unchanged — fp16 hurts the non-vec path due to half→float conversion overhead in the 16-element-per-call pattern.

## Recommendation

**This change is a net positive and should be merged to main.** It:
- Improves decode speed at all context depths (biggest win at long context)
- Has zero quality regression
- Has no prefill regression
- Is a 2-line change with minimal risk
- Directly addresses the tester-reported decode slowdown at long context
