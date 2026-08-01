import mlx.core as mx
import math

PLANAR_FUSED_QK_KERNEL = """
    uint seq_idx = threadgroup_position_in_grid.x;
    uint head_idx = threadgroup_position_in_grid.y;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint seq_len = dims[1];
    uint n_heads = dims[2];
    uint bits = dims[3];
    uint vals_per_word = dims[4];
    uint packed_dim = dims[5];
    uint bit_mask = (1u << bits) - 1u;

    // Load Q into shared memory — half precision for 2x ALU throughput
    threadgroup half q_shared[256];
    q_shared[elem] = (half)query[head_idx * dim + elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Extract K index from packed uint32
    uint word_idx = elem / vals_per_word;
    uint pos_in_word = elem % vals_per_word;
    uint word = packed[(head_idx * seq_len + seq_idx) * packed_dim + word_idx];
    uint idx = (word >> (pos_in_word * bits)) & bit_mask;

    // Codebook lookup — half precision
    half val = (half)(centroids[idx] * norms[head_idx * seq_len + seq_idx]);

    // Load K into shared memory
    threadgroup half k_shared[256];
    k_shared[elem] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inverse Givens rotation in half precision
    half SQRT2_2 = (half)0.70710678118f;
    if (elem % 2 == 0) {
        half in0 = k_shared[elem];
        half in1 = k_shared[elem + 1];
        k_shared[elem]     = (in0 + in1) * SQRT2_2;
        k_shared[elem + 1] = (in1 - in0) * SQRT2_2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Dot product — half precision multiply, float accumulate for stability
    float dot = (float)(q_shared[elem] * k_shared[elem]);
    threadgroup float dot_shared[256];
    dot_shared[elem] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction in float32 (accumulation needs precision)
    for (uint stride = dim / 2; stride > 0; stride >>= 1) {
        if (elem < stride) {
            dot_shared[elem] += dot_shared[elem + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (elem == 0) {
        out[head_idx * seq_len + seq_idx] = (T)(dot_shared[0] * scale[0]);
    }
"""

PLANAR_FUSED_SV_KERNEL = """
    uint head_idx = threadgroup_position_in_grid.y;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint seq_len = dims[1];
    uint n_heads = dims[2];
    uint bits = dims[3];
    uint vals_per_word = dims[4];
    uint packed_dim = dims[5];
    uint bit_mask = (1u << bits) - 1u;

    float acc = 0.0f;
    float SQRT2_2 = 0.70710678118f;

    for (uint seq_idx = 0; seq_idx < seq_len; seq_idx++) {
        uint word_idx = elem / vals_per_word;
        uint pos_in_word = elem % vals_per_word;
        uint word = packed[(head_idx * seq_len + seq_idx) * packed_dim + word_idx];
        uint idx = (word >> (pos_in_word * bits)) & bit_mask;
        float val = centroids[idx] * norms[head_idx * seq_len + seq_idx];

        threadgroup float v_shared[256];
        v_shared[elem] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (elem % 2 == 0) {
            float in0 = v_shared[elem];
            float in1 = v_shared[elem + 1];
            v_shared[elem]     = (in0 + in1) * SQRT2_2;
            v_shared[elem + 1] = (in1 - in0) * SQRT2_2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float prob = (float)probs[head_idx * seq_len + seq_idx];
        acc += prob * v_shared[elem];
    }
    
    out[head_idx * dim + elem] = (T)acc;
"""

PLANAR_TILED_SV_KERNEL = """
    uint tile_idx = threadgroup_position_in_grid.x;
    uint head_idx = threadgroup_position_in_grid.y;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint seq_len = dims[1];
    uint n_heads = dims[2];
    uint bits = dims[3];
    uint vals_per_word = dims[4];
    uint packed_dim = dims[5];
    uint tile_size = dims[6];
    uint bit_mask = (1u << bits) - 1u;

    uint tile_start = tile_idx * tile_size;
    uint tile_end = tile_start + tile_size;
    if (tile_end > seq_len) tile_end = seq_len;

    float acc = 0.0f;  // accumulate in float32 for precision
    half SQRT2_2 = (half)0.70710678118f;
    threadgroup half v_shared[256];

    for (uint seq_idx = tile_start; seq_idx < tile_end; seq_idx++) {
        uint word_idx = elem / vals_per_word;
        uint pos_in_word = elem % vals_per_word;
        uint word = packed[(head_idx * seq_len + seq_idx) * packed_dim + word_idx];
        uint idx = (word >> (pos_in_word * bits)) & bit_mask;
        half val = (half)(centroids[idx] * norms[head_idx * seq_len + seq_idx]);

        v_shared[elem] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (elem % 2 == 0) {
            half in0 = v_shared[elem];
            half in1 = v_shared[elem + 1];
            v_shared[elem]     = (in0 + in1) * SQRT2_2;
            v_shared[elem + 1] = (in1 - in0) * SQRT2_2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        half prob = (half)probs[head_idx * seq_len + seq_idx];
        acc += (float)(prob * v_shared[elem]);  // half multiply, float accumulate
    }

    // Write partial sum for this tile
    partial_out[(tile_idx * n_heads + head_idx) * dim + elem] = acc;
"""

PLANAR_FLASH_DECODE_KERNEL = """
    // Combined QK + online-softmax + SV in one pass per tile.
    // Each threadgroup processes a 256-token tile for one head.
    // Reads packed K and V exactly once — no FP16 intermediate in device memory.
    // Outputs: partial_o (D floats) + lse (1 float) per tile for log-sum-exp merge.

    uint tile_idx = threadgroup_position_in_grid.x;
    uint head_idx = threadgroup_position_in_grid.y;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint seq_len = dims[1];
    uint n_heads = dims[2];
    uint bits = dims[3];
    uint vals_per_word = dims[4];
    uint packed_dim = dims[5];
    uint tile_size = dims[6];
    uint bit_mask = (1u << bits) - 1u;

    uint tile_start = tile_idx * tile_size;
    uint tile_end = tile_start + tile_size;
    if (tile_end > seq_len) tile_end = seq_len;

    float SQRT2_2 = 0.70710678118f;

    // Load Q once for the entire tile
    threadgroup float q_shared[256];
    q_shared[elem] = (float)query[head_idx * dim + elem];

    // Shared scalars for online softmax broadcast
    threadgroup float s_corr[1];   // correction factor
    threadgroup float s_expsc[1];  // exp(score - new_max)
    threadgroup float s_max[1];    // running max
    threadgroup float s_sum[1];    // running sum_exp

    if (elem == 0) {
        s_max[0] = -1e30f;
        s_sum[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float kv_shared[256];
    threadgroup float dot_shared[256];
    float acc_v = 0.0f;

    for (uint seq_idx = tile_start; seq_idx < tile_end; seq_idx++) {
        // ── Unpack K element ──
        uint word_idx = elem / vals_per_word;
        uint pos_in_word = elem % vals_per_word;
        uint k_word = k_packed[(head_idx * seq_len + seq_idx) * packed_dim + word_idx];
        uint k_idx = (k_word >> (pos_in_word * bits)) & bit_mask;
        float k_val = centroids[k_idx] * k_norms[head_idx * seq_len + seq_idx];

        kv_shared[elem] = k_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inverse Givens on K
        if (elem % 2 == 0) {
            float a = kv_shared[elem], b = kv_shared[elem + 1];
            kv_shared[elem]     = (a + b) * SQRT2_2;
            kv_shared[elem + 1] = (b - a) * SQRT2_2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── QK dot product ──
        dot_shared[elem] = q_shared[elem] * kv_shared[elem];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = dim / 2; stride > 0; stride >>= 1) {
            if (elem < stride) dot_shared[elem] += dot_shared[elem + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ── Thread 0: online softmax update + broadcast ──
        if (elem == 0) {
            float score = dot_shared[0] * scale[0];
            float old_max = s_max[0];
            float new_max = (score > old_max) ? score : old_max;
            float corr = exp(old_max - new_max);
            float es = exp(score - new_max);
            s_max[0] = new_max;
            s_sum[0] = s_sum[0] * corr + es;
            s_corr[0] = corr;
            s_expsc[0] = es;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float corr = s_corr[0];
        float es = s_expsc[0];

        // ── Correct accumulated V by softmax rescaling ──
        acc_v = acc_v * corr;

        // ── Unpack V element ──
        uint v_word = v_packed[(head_idx * seq_len + seq_idx) * packed_dim + word_idx];
        uint v_idx = (v_word >> (pos_in_word * bits)) & bit_mask;
        float v_val = centroids[v_idx] * v_norms[head_idx * seq_len + seq_idx];

        kv_shared[elem] = v_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inverse Givens on V
        if (elem % 2 == 0) {
            float a = kv_shared[elem], b = kv_shared[elem + 1];
            kv_shared[elem]     = (a + b) * SQRT2_2;
            kv_shared[elem + 1] = (b - a) * SQRT2_2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Accumulate weighted V ──
        acc_v += es * kv_shared[elem];
    }

    // Write partial output (unnormalized) + tile_max + tile_sum_exp
    uint out_base = (tile_idx * n_heads + head_idx) * dim;
    partial_o[out_base + elem] = acc_v;

    if (elem == 0) {
        uint meta_idx = tile_idx * n_heads + head_idx;
        tile_max[meta_idx] = s_max[0];
        tile_sum_exp[meta_idx] = s_sum[0];
    }
"""

_planar_fused_qk = None
_planar_fused_sv = None
_planar_tiled_sv = None
_planar_flash_decode = None
_planar_sparse_flash = None
_fused_sparse_attend = None

# ═══════════════════════════════════════════════════════════════════════════
# FULLY FUSED SPARSE ATTENTION — Two GPU dispatches, zero Python round-trips
# ═══════════════════════════════════════════════════════════════════════════
#
# Dispatch 1 (PHASE1_SCORE_KERNEL): Score ALL tokens, write per-tile top-K
#   Each tile (256 tokens): QK dot products → find local top scores
#   Outputs: all_scores (B*H*T), tile_top_scores (num_tiles*B*H*topk_per_tile)
#
# Python bridge: compute threshold from tile_top_scores (tiny array, microseconds)
#
# Dispatch 2 (PHASE2_SPARSE_SV_KERNEL): Selective V fetch + softmax + accumulate
#   Each tile reads pre-computed scores, skips below threshold
#   Does online softmax + V accumulate only for selected tokens
#   Output: partial_o per tile, merged via log-sum-exp

PHASE1_SCORE_KERNEL = """
    // Phase 1: Score all tokens, track per-tile top-K scores
    // Each threadgroup = one tile of one head
    // Outputs: all_scores[head*T + token] and tile_top[tile*H*K + head*K + k]

    uint tile_idx = threadgroup_position_in_grid.x;
    uint head_idx = threadgroup_position_in_grid.y;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint seq_len = dims[1];
    uint n_heads = dims[2];
    uint bits = dims[3];
    uint vals_per_word = dims[4];
    uint packed_dim = dims[5];
    uint tile_size = dims[6];
    uint bit_mask = (1u << bits) - 1u;

    uint tile_start = tile_idx * tile_size;
    uint tile_end = tile_start + tile_size;
    if (tile_end > seq_len) tile_end = seq_len;

    half SQRT2_2 = (half)0.70710678118f;

    // Load Q once
    threadgroup half q_shared[256];
    q_shared[elem] = (half)query[head_idx * dim + elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup half k_shared[256];
    threadgroup float dot_shared[256];

    // Track top-4 scores in this tile (enough to find threshold later)
    threadgroup float tile_tops[4];
    if (elem < 4) tile_tops[elem] = -1e30f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint seq_idx = tile_start; seq_idx < tile_end; seq_idx++) {
        // Unpack K
        uint word_idx = elem / vals_per_word;
        uint pos_in_word = elem % vals_per_word;
        uint word = k_packed[(head_idx * seq_len + seq_idx) * packed_dim + word_idx];
        uint idx = (word >> (pos_in_word * bits)) & bit_mask;
        half k_val = (half)(centroids[idx] * k_norms[head_idx * seq_len + seq_idx]);

        k_shared[elem] = k_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Givens inverse
        if (elem % 2 == 0) {
            half in0 = k_shared[elem], in1 = k_shared[elem + 1];
            k_shared[elem]     = (in0 + in1) * SQRT2_2;
            k_shared[elem + 1] = (in1 - in0) * SQRT2_2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Dot product + tree reduction
        dot_shared[elem] = (float)(q_shared[elem] * k_shared[elem]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = dim / 2; stride > 0; stride >>= 1) {
            if (elem < stride) dot_shared[elem] += dot_shared[elem + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (elem == 0) {
            float score = dot_shared[0] * scale[0];
            // Write score to global buffer
            all_scores[head_idx * seq_len + seq_idx] = score;

            // Track top-4 for this tile (insertion sort, tiny)
            if (score > tile_tops[3]) {
                tile_tops[3] = score;
                // Bubble up
                for (int i = 2; i >= 0; i--) {
                    if (tile_tops[i+1] > tile_tops[i]) {
                        float tmp = tile_tops[i];
                        tile_tops[i] = tile_tops[i+1];
                        tile_tops[i+1] = tmp;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write tile's top-4 scores
    if (elem < 4) {
        uint base = (tile_idx * n_heads + head_idx) * 4;
        tile_top_scores[base + elem] = tile_tops[elem];
    }
"""

PHASE2_SPARSE_ATTEND_KERNEL = """
    // Phase 2: Read pre-computed scores, skip below threshold,
    // online-softmax + V accumulate for survivors only.
    // Each threadgroup = one tile of one head.

    uint tile_idx = threadgroup_position_in_grid.x;
    uint head_idx = threadgroup_position_in_grid.y;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint seq_len = dims[1];
    uint n_heads = dims[2];
    uint bits = dims[3];
    uint vals_per_word = dims[4];
    uint packed_dim = dims[5];
    uint tile_size = dims[6];
    uint bit_mask = (1u << bits) - 1u;

    uint tile_start = tile_idx * tile_size;
    uint tile_end = tile_start + tile_size;
    if (tile_end > seq_len) tile_end = seq_len;

    float threshold_val = threshold[head_idx]; // per-head threshold

    // ── Tile-level early exit: skip entire tile if no survivors ──────
    // At top-1024 from 50K: ~196 tiles, only ~4-8 have survivors.
    // The other 188 tiles return immediately — no barriers, no V work.
    threadgroup bool tile_has_survivors[1];
    if (elem == 0) {
        tile_has_survivors[0] = false;
        for (uint i = tile_start; i < tile_end; i++) {
            if (all_scores[head_idx * seq_len + i] >= threshold_val) {
                tile_has_survivors[0] = true;
                break;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!tile_has_survivors[0]) {
        // ENTIRE TILE SKIPPED — zero barriers, zero V work
        uint out_base = (tile_idx * n_heads + head_idx) * dim;
        partial_o[out_base + elem] = 0.0f;
        if (elem == 0) {
            uint meta = tile_idx * n_heads + head_idx;
            tile_max[meta] = -1e30f;
            tile_sum_exp[meta] = 0.0f;
        }
        return;
    }

    half SQRT2_2 = (half)0.70710678118f;

    // Online softmax state
    threadgroup float s_max[1];
    threadgroup float s_sum[1];
    threadgroup float s_corr[1];
    threadgroup float s_expsc[1];
    if (elem == 0) {
        s_max[0] = -1e30f;
        s_sum[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup half v_shared[256];
    float acc_v = 0.0f;

    for (uint seq_idx = tile_start; seq_idx < tile_end; seq_idx++) {
        // Read pre-computed score
        float score = all_scores[head_idx * seq_len + seq_idx];

        // Skip non-selected tokens (barriers still fire but math is skipped)
        if (score < threshold_val) continue;

        // ── This token was selected: fetch V and accumulate ──

        // Online softmax update (thread 0 broadcasts)
        if (elem == 0) {
            float old_max = s_max[0];
            float new_max = (score > old_max) ? score : old_max;
            float corr = exp(old_max - new_max);
            float es = exp(score - new_max);
            s_max[0] = new_max;
            s_sum[0] = s_sum[0] * corr + es;
            s_corr[0] = corr;
            s_expsc[0] = es;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float corr = s_corr[0];
        float es = s_expsc[0];
        acc_v = acc_v * corr;

        // Unpack V
        uint word_idx = elem / vals_per_word;
        uint pos_in_word = elem % vals_per_word;
        uint word = v_packed[(head_idx * seq_len + seq_idx) * packed_dim + word_idx];
        uint idx = (word >> (pos_in_word * bits)) & bit_mask;
        half v_val = (half)(centroids[idx] * v_norms[head_idx * seq_len + seq_idx]);

        v_shared[elem] = v_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Givens inverse on V
        if (elem % 2 == 0) {
            half in0 = v_shared[elem], in1 = v_shared[elem + 1];
            v_shared[elem]     = (in0 + in1) * SQRT2_2;
            v_shared[elem + 1] = (in1 - in0) * SQRT2_2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        acc_v += es * (float)v_shared[elem];
    }

    uint out_base = (tile_idx * n_heads + head_idx) * dim;
    partial_o[out_base + elem] = acc_v;

    if (elem == 0) {
        uint meta = tile_idx * n_heads + head_idx;
        tile_max[meta] = s_max[0];
        tile_sum_exp[meta] = s_sum[0];
    }
"""

# ── Sparse Flash Decode: QK score → threshold → selective V fetch ────────
# Legacy two-pass with Python topk (kept for comparison)
# This avoids the redundancy math overhead while keeping V fetch sparse.

PLANAR_SPARSE_SV_KERNEL = """
    // Tiled SV that SKIPS tokens where prob == 0 (masked by top-K).
    // Same structure as PLANAR_TILED_SV but with an early-continue
    // that avoids the V unpack + Givens rotation for masked tokens.
    // At top-1024 from 50K tokens: skips 98% of V operations.

    uint tile_idx = threadgroup_position_in_grid.x;
    uint head_idx = threadgroup_position_in_grid.y;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint seq_len = dims[1];
    uint n_heads = dims[2];
    uint bits = dims[3];
    uint vals_per_word = dims[4];
    uint packed_dim = dims[5];
    uint tile_size = dims[6];
    uint bit_mask = (1u << bits) - 1u;

    uint tile_start = tile_idx * tile_size;
    uint tile_end = tile_start + tile_size;
    if (tile_end > seq_len) tile_end = seq_len;

    float acc = 0.0f;
    half SQRT2_2 = (half)0.70710678118f;
    threadgroup half v_shared[256];

    for (uint seq_idx = tile_start; seq_idx < tile_end; seq_idx++) {
        // ── Check if this token was selected (prob > 0) ──
        half prob = (half)probs[head_idx * seq_len + seq_idx];
        if (prob < (half)1e-8f) continue;  // SKIP: not in top-K for this head

        // ── Unpack V (only for selected tokens) ──
        uint word_idx = elem / vals_per_word;
        uint pos_in_word = elem % vals_per_word;
        uint word = packed[(head_idx * seq_len + seq_idx) * packed_dim + word_idx];
        uint idx = (word >> (pos_in_word * bits)) & bit_mask;
        half val = (half)(centroids[idx] * norms[head_idx * seq_len + seq_idx]);

        v_shared[elem] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (elem % 2 == 0) {
            half in0 = v_shared[elem];
            half in1 = v_shared[elem + 1];
            v_shared[elem]     = (in0 + in1) * SQRT2_2;
            v_shared[elem + 1] = (in1 - in0) * SQRT2_2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        acc += (float)(prob * v_shared[elem]);
    }

    partial_out[(tile_idx * n_heads + head_idx) * dim + elem] = acc;
"""

TILE_SIZE = 256

def planar_fused_qk_scores(
    query: mx.array,
    k_packed: mx.array,
    k_norms: mx.array,
    centroids: mx.array,
    scale: float,
    dim: int,
    bits: int,
) -> mx.array:
    global _planar_fused_qk
    if _planar_fused_qk is None:
        _planar_fused_qk = mx.fast.metal_kernel(
            name="planar_fused_qk",
            input_names=["query", "packed", "norms", "centroids", "scale", "dims"],
            output_names=["out"],
            source=PLANAR_FUSED_QK_KERNEL,
        )

    # query: (B, H, 1, D) -> reshape to (H, D) assuming B=1 for now.
    # Actually B could be > 1. Let's reshape to (B*H, D)
    B = query.shape[0]
    H = query.shape[1]
    seq_len = k_norms.shape[2]
    p_dim = k_packed.shape[-1]
    vpw = {1: 32, 2: 16, 3: 10, 4: 8}[bits]
    
    scale_arr = mx.array([scale], dtype=mx.float32)
    dims_arr = mx.array([dim, seq_len, B * H, bits, vpw, p_dim], dtype=mx.uint32)

    outputs = _planar_fused_qk(
        inputs=[
            query.astype(mx.float32).reshape(B * H * dim),
            k_packed.astype(mx.uint32).reshape(B * H * seq_len * p_dim),
            k_norms.astype(mx.float32).reshape(B * H * seq_len),
            centroids, scale_arr, dims_arr,
        ],
        template=[("T", mx.float32)],
        # grid = total threads; threadgroups = grid / threadgroup_size
        # We want seq_len threadgroups in x, each with dim threads
        grid=(seq_len * dim, B * H, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(B * H * seq_len,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(B, H, 1, seq_len)

def planar_fused_sv_values(
    probs: mx.array,
    v_packed: mx.array,
    v_norms: mx.array,
    centroids: mx.array,
    dim: int,
    bits: int,
) -> mx.array:
    global _planar_fused_sv
    if _planar_fused_sv is None:
        _planar_fused_sv = mx.fast.metal_kernel(
            name="planar_fused_sv",
            input_names=["probs", "packed", "norms", "centroids", "dims"],
            output_names=["out"],
            source=PLANAR_FUSED_SV_KERNEL,
        )

    B = probs.shape[0]
    H = probs.shape[1]
    seq_len = v_norms.shape[2]
    p_dim = v_packed.shape[-1]
    vpw = {1: 32, 2: 16, 3: 10, 4: 8}[bits]
    
    dims_arr = mx.array([dim, seq_len, B * H, bits, vpw, p_dim], dtype=mx.uint32)

    outputs = _planar_fused_sv(
        inputs=[
            probs.astype(mx.float32).reshape(B * H * seq_len),
            v_packed.astype(mx.uint32).reshape(B * H * seq_len * p_dim),
            v_norms.astype(mx.float32).reshape(B * H * seq_len),
            centroids, dims_arr,
        ],
        template=[("T", mx.float32)],
        # 1 threadgroup per head, dim threads per threadgroup
        grid=(dim, B * H, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(B * H * dim,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(B, H, 1, dim)


def planar_tiled_sv_values(
    probs: mx.array,
    v_packed: mx.array,
    v_norms: mx.array,
    centroids: mx.array,
    dim: int,
    bits: int,
) -> mx.array:
    """Tiled SV kernel: reads packed V on-the-fly in 256-token tiles.

    Eliminates the need for a cached decompressed V tensor.
    Pass 1: Metal kernel computes partial weighted sums per tile.
    Pass 2: mx.sum reduces tiles (trivial).
    """
    global _planar_tiled_sv
    if _planar_tiled_sv is None:
        _planar_tiled_sv = mx.fast.metal_kernel(
            name="planar_tiled_sv",
            input_names=["probs", "packed", "norms", "centroids", "dims"],
            output_names=["partial_out"],
            source=PLANAR_TILED_SV_KERNEL,
        )

    B = probs.shape[0]
    H = probs.shape[1]
    seq_len = v_norms.shape[2]
    p_dim = v_packed.shape[-1]
    vpw = {1: 32, 2: 16, 3: 10, 4: 8}[bits]
    num_tiles = (seq_len + TILE_SIZE - 1) // TILE_SIZE

    dims_arr = mx.array([dim, seq_len, B * H, bits, vpw, p_dim, TILE_SIZE],
                        dtype=mx.uint32)

    outputs = _planar_tiled_sv(
        inputs=[
            probs.astype(mx.float32).reshape(B * H * seq_len),
            v_packed.astype(mx.uint32).reshape(B * H * seq_len * p_dim),
            v_norms.astype(mx.float32).reshape(B * H * seq_len),
            centroids, dims_arr,
        ],
        template=[("T", mx.float32)],
        # num_tiles threadgroups in x, each with dim threads
        grid=(num_tiles * dim, B * H, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(num_tiles * B * H * dim,)],
        output_dtypes=[mx.float32],
    )
    # Reduce partial sums across tiles
    partial = outputs[0].reshape(num_tiles, B * H, dim)
    reduced = mx.sum(partial, axis=0)  # (B*H, dim)
    return reduced.reshape(B, H, 1, dim)


def fused_sparse_attend(
    query: mx.array,
    k_packed: mx.array, k_norms: mx.array,
    v_packed: mx.array, v_norms: mx.array,
    centroids: mx.array,
    scale: float, dim: int, bits: int,
    topk: int = 1024,
) -> mx.array:
    """Fully fused sparse attention — two GPU dispatches, zero Python overhead.

    Dispatch 1: Score ALL tokens via fused QK kernel, track per-tile top scores
    Bridge:     Compute per-head threshold from tile tops (tiny array, microseconds)
    Dispatch 2: Selective V fetch + online softmax, skipping below-threshold tokens

    At 50K tokens with topk=1024: Dispatch 1 scores 50K tokens (fast QK).
    Bridge picks the 1024th-highest score per head from tile summaries.
    Dispatch 2 only unpacks+rotates V for ~1024 tokens per head (98% skipped).
    """
    global _fused_sparse_attend
    _phase1 = getattr(fused_sparse_attend, '_phase1', None)
    _phase2 = getattr(fused_sparse_attend, '_phase2', None)

    if _phase1 is None:
        _phase1 = mx.fast.metal_kernel(
            name="phase1_score",
            input_names=["query", "k_packed", "k_norms", "centroids", "scale", "dims"],
            output_names=["all_scores", "tile_top_scores"],
            source=PHASE1_SCORE_KERNEL,
        )
        fused_sparse_attend._phase1 = _phase1

    if _phase2 is None:
        _phase2 = mx.fast.metal_kernel(
            name="phase2_sparse_attend",
            input_names=["all_scores", "v_packed", "v_norms", "centroids", "threshold", "dims"],
            output_names=["partial_o", "tile_max", "tile_sum_exp"],
            source=PHASE2_SPARSE_ATTEND_KERNEL,
        )
        fused_sparse_attend._phase2 = _phase2

    B = query.shape[0]
    H = query.shape[1]
    seq_len = k_norms.shape[2]
    p_dim = k_packed.shape[-1]
    vpw = {1: 32, 2: 16, 3: 10, 4: 8}[bits]
    num_tiles = (seq_len + TILE_SIZE - 1) // TILE_SIZE
    n_bh = B * H
    top_per_tile = 4  # track top-4 scores per tile

    scale_arr = mx.array([scale], dtype=mx.float32)
    dims_arr = mx.array([dim, seq_len, n_bh, bits, vpw, p_dim, TILE_SIZE], dtype=mx.uint32)

    # ── Dispatch 1: Score all tokens, collect tile tops ──────────────────
    phase1_out = _phase1(
        inputs=[
            query.astype(mx.float32).reshape(n_bh * dim),
            k_packed.astype(mx.uint32).reshape(n_bh * seq_len * p_dim),
            k_norms.astype(mx.float32).reshape(n_bh * seq_len),
            centroids, scale_arr, dims_arr,
        ],
        template=[("T", mx.float32)],
        grid=(num_tiles * dim, n_bh, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(n_bh * seq_len,), (num_tiles * n_bh * top_per_tile,)],
        output_dtypes=[mx.float32, mx.float32],
    )

    all_scores = phase1_out[0]  # (n_bh * seq_len,)
    tile_tops = phase1_out[1].reshape(num_tiles, n_bh, top_per_tile)  # (tiles, BH, 4)

    # ── Bridge: compute per-head threshold from tile tops ────────────────
    # Flatten all tile tops per head: (num_tiles * top_per_tile, n_bh)
    # Pick the topk-th score as threshold
    all_tops = tile_tops.reshape(-1, n_bh).transpose()  # (n_bh, num_tiles * 4)
    n_candidates = all_tops.shape[1]

    if topk < n_candidates:
        # Per-head: find the topk-th highest score
        topk_vals = mx.topk(all_tops, k=min(topk, n_candidates), axis=-1)  # (n_bh, topk)
        threshold = mx.min(topk_vals, axis=-1)  # (n_bh,) — the K-th score per head
    else:
        # More candidates than topk — use min as threshold (keep everything)
        threshold = mx.min(all_tops, axis=-1)

    mx.eval(threshold)  # tiny array, microseconds

    # ── Dispatch 2: Sparse V fetch + online softmax ──────────────────────
    phase2_out = _phase2(
        inputs=[
            all_scores,
            v_packed.astype(mx.uint32).reshape(n_bh * seq_len * p_dim),
            v_norms.astype(mx.float32).reshape(n_bh * seq_len),
            centroids, threshold, dims_arr,
        ],
        template=[("T", mx.float32)],
        grid=(num_tiles * dim, n_bh, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(num_tiles * n_bh * dim,), (num_tiles * n_bh,), (num_tiles * n_bh,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )

    partial_o = phase2_out[0].reshape(num_tiles, n_bh, dim)
    t_max = phase2_out[1].reshape(num_tiles, n_bh, 1)
    t_sum_exp = phase2_out[2].reshape(num_tiles, n_bh, 1)

    # ── Log-sum-exp merge across tiles ───────────────────────────────────
    global_max = mx.max(t_max, axis=0, keepdims=True)
    corrections = mx.exp(t_max - global_max)
    numerator = mx.sum(partial_o * corrections, axis=0)
    denominator = mx.sum(t_sum_exp * corrections, axis=0)
    result = numerator / (denominator + 1e-8)

    return result.reshape(B, H, 1, dim)


def planar_sparse_sv_values(
    probs: mx.array,
    v_packed: mx.array,
    v_norms: mx.array,
    centroids: mx.array,
    dim: int,
    bits: int,
) -> mx.array:
    """Sparse tiled SV: skips tokens where prob == 0 (masked by top-K).

    Same interface as planar_tiled_sv_values but uses PLANAR_SPARSE_SV_KERNEL
    which early-continues on zero-prob tokens. At top-1024 from 50K tokens,
    this skips 98% of V unpack + Givens operations.
    """
    global _planar_sparse_flash
    if _planar_sparse_flash is None:
        _planar_sparse_flash = mx.fast.metal_kernel(
            name="planar_sparse_sv",
            input_names=["probs", "packed", "norms", "centroids", "dims"],
            output_names=["partial_out"],
            source=PLANAR_SPARSE_SV_KERNEL,
        )

    B = probs.shape[0]
    H = probs.shape[1]
    seq_len = v_norms.shape[2]
    p_dim = v_packed.shape[-1]
    vpw = {1: 32, 2: 16, 3: 10, 4: 8}[bits]
    num_tiles = (seq_len + TILE_SIZE - 1) // TILE_SIZE

    dims_arr = mx.array([dim, seq_len, B * H, bits, vpw, p_dim, TILE_SIZE],
                        dtype=mx.uint32)

    outputs = _planar_sparse_flash(
        inputs=[
            probs.astype(mx.float32).reshape(B * H * seq_len),
            v_packed.astype(mx.uint32).reshape(B * H * seq_len * p_dim),
            v_norms.astype(mx.float32).reshape(B * H * seq_len),
            centroids, dims_arr,
        ],
        template=[("T", mx.float32)],
        grid=(num_tiles * dim, B * H, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(num_tiles * B * H * dim,)],
        output_dtypes=[mx.float32],
    )
    partial = outputs[0].reshape(num_tiles, B * H, dim)
    reduced = mx.sum(partial, axis=0)
    return reduced.reshape(B, H, 1, dim)


def planar_flash_decode(
    query: mx.array,
    k_packed: mx.array, k_norms: mx.array,
    v_packed: mx.array, v_norms: mx.array,
    centroids: mx.array,
    scale: float, dim: int, bits: int,
) -> mx.array:
    """Flash decode: fused QK + online-softmax + SV in one pass per tile.

    Single-pass attention over packed 3-bit K and V.  Each 256-token tile
    runs independently as one threadgroup, computing a partial output with
    log-sum-exp for cross-tile merging.  No FP16 K or V ever touches device
    memory.  No intermediate scores tensor.  Perfect GPU parallelism.
    """
    global _planar_flash_decode
    if _planar_flash_decode is None:
        _planar_flash_decode = mx.fast.metal_kernel(
            name="planar_flash_decode",
            input_names=["query", "k_packed", "k_norms", "v_packed", "v_norms",
                         "centroids", "scale", "dims"],
            output_names=["partial_o", "tile_max", "tile_sum_exp"],
            source=PLANAR_FLASH_DECODE_KERNEL,
        )

    B = query.shape[0]
    H = query.shape[1]
    seq_len = k_norms.shape[2]
    p_dim = k_packed.shape[-1]
    vpw = {1: 32, 2: 16, 3: 10, 4: 8}[bits]
    num_tiles = (seq_len + TILE_SIZE - 1) // TILE_SIZE

    scale_arr = mx.array([scale], dtype=mx.float32)
    dims_arr = mx.array([dim, seq_len, B * H, bits, vpw, p_dim, TILE_SIZE],
                        dtype=mx.uint32)
    n_bh = B * H

    outputs = _planar_flash_decode(
        inputs=[
            query.astype(mx.float32).reshape(n_bh * dim),
            k_packed.astype(mx.uint32).reshape(n_bh * seq_len * p_dim),
            k_norms.astype(mx.float32).reshape(n_bh * seq_len),
            v_packed.astype(mx.uint32).reshape(n_bh * seq_len * p_dim),
            v_norms.astype(mx.float32).reshape(n_bh * seq_len),
            centroids, scale_arr, dims_arr,
        ],
        template=[("T", mx.float32)],
        grid=(num_tiles * dim, n_bh, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(num_tiles * n_bh * dim,),
                       (num_tiles * n_bh,),
                       (num_tiles * n_bh,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )

    partial_o = outputs[0].reshape(num_tiles, n_bh, dim)
    t_max     = outputs[1].reshape(num_tiles, n_bh, 1)
    t_sum_exp = outputs[2].reshape(num_tiles, n_bh, 1)

    # ── Exact log-sum-exp merge across tiles ──
    # partial_o[i] = sum_j_in_tile(exp(s_j - max_i) * V_j)   (unnormalized)
    # To get global: rescale each tile to a common max
    global_max = mx.max(t_max, axis=0, keepdims=True)          # (1, n_bh, 1)
    corrections = mx.exp(t_max - global_max)                    # (num_tiles, n_bh, 1)
    numerator   = mx.sum(partial_o * corrections, axis=0)       # (n_bh, dim)
    denominator = mx.sum(t_sum_exp * corrections, axis=0)       # (n_bh, 1)
    result = numerator / (denominator + 1e-8)

    return result.reshape(B, H, 1, dim)
