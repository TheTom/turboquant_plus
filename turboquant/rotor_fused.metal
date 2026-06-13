#include <metal_stdlib>
using namespace metal;

/*
 * RotorQuant fused Metal compute shader for Apple Silicon.
 *
 * Full pipeline per thread: embed → rotor sandwich → quantize → inverse → extract
 * Each thread handles one (batch_item, group) pair.
 * Rotors and centroids loaded into threadgroup (shared) memory.
 *
 * Exploits rotor sparsity: only 4/8 multivector components are non-zero.
 * Sparse GP: 28 FMAs vs 64 for full product, 56 total per sandwich.
 */

struct Params {
    uint batch_size;
    uint emb_dim;
    uint n_groups;
    uint n_levels;    // centroids count (same for all grades in this version)
};

// Sparse geometric product: rotor * multivector
// Rotor has only [s, 0, 0, 0, b12, b13, b23, 0] non-zero
static void gp_rotor_mv(
    float s, float p12, float p13, float p23,
    thread float *x, thread float *r)
{
    r[0] = s*x[0] - p12*x[4] - p13*x[5] - p23*x[6];
    r[1] = s*x[1] + p12*x[2] + p13*x[3] + p23*x[7];
    r[2] = s*x[2] - p12*x[1] + p23*x[3] - p13*x[7];
    r[3] = s*x[3] - p13*x[1] - p23*x[2] + p12*x[7];
    r[4] = s*x[4] + p12*x[0];
    r[5] = s*x[5] + p13*x[0];
    r[6] = s*x[6] + p23*x[0];
    r[7] = s*x[7] - p23*x[1] + p13*x[2] - p12*x[3];
}

// Find nearest centroid (linear scan — fine for n_levels <= 16)
static float quantize_scalar(float val, threadgroup float *centroids, uint n_levels) {
    float best = centroids[0];
    float min_d = abs(val - best);
    for (uint i = 1; i < n_levels; i++) {
        float d = abs(val - centroids[i]);
        if (d < min_d) { min_d = d; best = centroids[i]; }
    }
    return best;
}

kernel void rotor_full_fused(
    device const float *input    [[buffer(0)]],   // (batch, emb_dim)
    device const float *rotors   [[buffer(1)]],   // (n_groups, 4): [s, b12, b13, b23]
    device const float *cents    [[buffer(2)]],   // (n_levels,) centroids
    device float       *output   [[buffer(3)]],   // (batch, emb_dim)
    constant Params    &params   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],        // gid.x = batch, gid.y = group
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgsize [[threads_per_threadgroup]]
)
{
    uint b = gid.x;
    uint g = gid.y;

    if (b >= params.batch_size || g >= params.n_groups) return;

    // Load centroids into threadgroup memory (cooperative)
    threadgroup float sh_cents[256];  // max 256 levels (8-bit)
    uint tid = lid.y * tgsize.x + lid.x;
    if (tid < params.n_levels) {
        sh_cents[tid] = cents[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load rotor for this group
    uint ri = g * 4;
    float s   = rotors[ri + 0];
    float p12 = rotors[ri + 1];
    float p13 = rotors[ri + 2];
    float p23 = rotors[ri + 3];

    // Embed: 3 vector dims → multivector (grade-1 only)
    uint d0 = g * 3;
    float x_mv[8] = {0.0f};
    if (d0     < params.emb_dim) x_mv[1] = input[b * params.emb_dim + d0];
    if (d0 + 1 < params.emb_dim) x_mv[2] = input[b * params.emb_dim + d0 + 1];
    if (d0 + 2 < params.emb_dim) x_mv[3] = input[b * params.emb_dim + d0 + 2];

    // Forward sandwich: temp = R * x, rotated = temp * R̃
    float temp[8], rotated[8];
    gp_rotor_mv(s, p12, p13, p23, x_mv, temp);
    gp_rotor_mv(s, -p12, -p13, -p23, temp, rotated);

    // Grade-aware quantization (all grades use same codebook for simplicity)
    float q_mv[8];
    for (int c = 0; c < 8; c++) {
        q_mv[c] = quantize_scalar(rotated[c], sh_cents, params.n_levels);
    }

    // Inverse sandwich: temp' = R̃ * q, final = temp' * R
    float temp2[8], final_mv[8];
    gp_rotor_mv(s, -p12, -p13, -p23, q_mv, temp2);
    gp_rotor_mv(s, p12, p13, p23, temp2, final_mv);

    // Extract vector grades back to output
    if (d0     < params.emb_dim) output[b * params.emb_dim + d0]     = final_mv[1];
    if (d0 + 1 < params.emb_dim) output[b * params.emb_dim + d0 + 1] = final_mv[2];
    if (d0 + 2 < params.emb_dim) output[b * params.emb_dim + d0 + 2] = final_mv[3];
}

// Standalone forward rotation only (for quantize path without dequant)
kernel void rotor_sandwich_only(
    device const float *input    [[buffer(0)]],   // (batch, emb_dim)
    device const float *rotors   [[buffer(1)]],   // (n_groups, 4)
    device float       *output   [[buffer(2)]],   // (batch, n_groups * 8)
    constant Params    &params   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
)
{
    uint b = gid.x;
    uint g = gid.y;
    if (b >= params.batch_size || g >= params.n_groups) return;

    uint ri = g * 4;
    float s   = rotors[ri]; float p12 = rotors[ri+1];
    float p13 = rotors[ri+2]; float p23 = rotors[ri+3];

    uint d0 = g * 3;
    float x_mv[8] = {0.0f};
    if (d0     < params.emb_dim) x_mv[1] = input[b * params.emb_dim + d0];
    if (d0 + 1 < params.emb_dim) x_mv[2] = input[b * params.emb_dim + d0 + 1];
    if (d0 + 2 < params.emb_dim) x_mv[3] = input[b * params.emb_dim + d0 + 2];

    float temp[8], rotated[8];
    gp_rotor_mv(s, p12, p13, p23, x_mv, temp);
    gp_rotor_mv(s, -p12, -p13, -p23, temp, rotated);

    uint base = b * params.n_groups * 8 + g * 8;
    for (int c = 0; c < 8; c++) {
        output[base + c] = rotated[c];
    }
}
