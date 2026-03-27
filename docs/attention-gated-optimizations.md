# Attention-Gated Kernel Optimizations

Opportunities identified in the Metal flash attention kernel using the same principle as sparse V: use attention weights (already computed) to gate downstream computation.

## Candidates

### 1. Tile-level V skip
**Status:** CLOSED — not worth it
**Location:** ggml-metal.metal ~line 7328, outer loop
**Idea:** Before entering the per-position V loop, check if ALL positions in the tile have `ss[] < τ`. If so, skip the entire tile. Currently sparse V checks per-position, which still enters the loop and evaluates the branch for every position. A tile-level check (`simd_max(ss) < τ`) would avoid entering the inner loop entirely. At 32K with 90% sparsity, many tiles are entirely negligible.
**Expected impact:** Reduces branch overhead on dense models (no per-position check when tile is skipped), bigger wins at long context on MoE.
**Risk:** Low — same quality guarantee as sparse V, just coarser granularity.
**Result:** MoE +0.6-0.9%, dense -0.8-1.1%. Tile max scan (8 reads from ss[]) costs more than it saves when most tiles have at least one non-negligible position. Not worth the complexity. See `threshold-ablation-logs/tile_skip_experiment.txt`.

### 2. Sparse V on non-quantized (f16) V path
**Status:** TODO
**Location:** ggml-metal.metal ~line 7315, the `is_same<vd4_t, v4_t>` branch
**Idea:** Sparse V currently only applies to the quantized V path (else branch). The f16 path also does multiply-accumulate for every position. Skipping negligible positions saves FMA work even without dequant savings.
**Expected impact:** Small — no dequant to skip, just FMA. Maybe 1-3% at long context.
**Risk:** Very low.

### 3. O rescaling skip
**Status:** TODO
**Location:** ggml-metal.metal ~line 7298-7303
**Idea:** `O = diag(ms)*O` rescales running output when max changes. When `ms ≈ 1.0` (max didn't change), the multiply is wasted. Gate with `if (fabs(ms - 1.0f) > ε)`.
**Expected impact:** Small — rescaling is cheap relative to dequant. Depends on how often max changes.
**Risk:** Low, but floating point edge cases need care.

### 4. Softmax exp() skip
**Status:** TODO
**Location:** ggml-metal.metal ~line 7291
**Idea:** `exp(s - M)` is computed for every position. When `s - M < -20`, result is ~0. Skip exp() and write 0 directly. exp() is expensive on GPU.
**Expected impact:** Medium — exp() is costly, and at long context most scores are very negative relative to max.
**Risk:** Low — mathematically identical for `s - M < -20` (exp(-20) ≈ 2e-9).
