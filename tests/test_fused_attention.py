"""Tests for ForgeAttention fused Metal kernels.

Requires: Apple Silicon Mac with MLX installed.
Skip gracefully on non-Apple hardware.
"""
import pytest
import numpy as np

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available (requires Apple Silicon)")


@pytest.fixture
def planarquant_cache():
    """Create a PlanarQuantKVCache with test data."""
    from turboquant.mlx_fused_attention import (
        _planar_rotate, _planar_unrotate, _compress, _decompress,
        _CODEBOOKS, _packed_dim,
    )
    return _planar_rotate, _planar_unrotate, _compress, _decompress, _CODEBOOKS


def test_givens_rotation_roundtrip():
    """Verify Givens rotation is perfectly invertible."""
    from turboquant.mlx_fused_attention import _planar_rotate, _planar_unrotate
    x = mx.random.normal((1, 128))
    rotated = _planar_rotate(x)
    recovered = _planar_unrotate(rotated)
    mx.eval(recovered)
    diff = mx.max(mx.abs(x.astype(mx.float32) - recovered.astype(mx.float32))).item()
    assert diff < 1e-5, f"Rotation roundtrip error: {diff}"


def test_compress_decompress_roundtrip():
    """Verify compress → decompress preserves information within quantization error."""
    from turboquant.mlx_fused_attention import _compress, _decompress, _planar_rotate, _planar_unrotate
    x = mx.random.normal((100, 128)).astype(mx.float32)
    packed, norms = _compress(x, bits=3, rotate_fn=_planar_rotate)
    recovered = _decompress(packed, norms, 128, 3, _planar_unrotate, mx.float32)
    mx.eval(recovered)
    mse = mx.mean((x - recovered) ** 2).item()
    assert mse < 0.1, f"Compress/decompress MSE too high: {mse}"


def test_fused_qk_scores_match_reference():
    """Verify fused QK kernel produces same scores as decompress + matmul."""
    from turboquant.mlx_fused_attention import (
        planar_fused_qk_scores, _compress, _decompress,
        _planar_rotate, _planar_unrotate, _CODEBOOKS,
    )
    import math
    mx.random.seed(42)
    B, H, T, D = 1, 2, 50, 64
    bits = 3

    k_raw = mx.random.normal((B * H * T, D)).astype(mx.float32)
    k_packed, k_norms = _compress(k_raw, bits, _planar_rotate)
    k_packed = k_packed.reshape(B, H, T, -1)
    k_norms = k_norms.reshape(B, H, T)

    k_decompressed = _decompress(
        k_packed.reshape(-1, k_packed.shape[-1]),
        k_norms.reshape(-1), D, bits, _planar_unrotate, mx.float32
    ).reshape(B, H, T, D)

    q = mx.random.normal((B, H, 1, D)).astype(mx.float16)
    scale = 1.0 / math.sqrt(D)

    ref_scores = (q.astype(mx.float32) @ k_decompressed.swapaxes(-1, -2)) * scale
    centroids = mx.array(_CODEBOOKS[bits], dtype=mx.float32)
    fused_scores = planar_fused_qk_scores(q, k_packed, k_norms, centroids, scale, D, bits)
    mx.eval(ref_scores, fused_scores)

    diff = mx.max(mx.abs(fused_scores - ref_scores)).item()
    assert diff < 0.001, f"Fused QK error: {diff}"


def test_fused_qk_multi_head():
    """Verify fused QK works with multiple heads."""
    from turboquant.mlx_fused_attention import (
        planar_fused_qk_scores, _compress, _planar_rotate, _CODEBOOKS,
    )
    import math
    mx.random.seed(7)
    B, H, T, D = 1, 8, 100, 128
    bits = 3

    k_raw = mx.random.normal((B * H * T, D)).astype(mx.float32)
    k_packed, k_norms = _compress(k_raw, bits, _planar_rotate)
    k_packed = k_packed.reshape(B, H, T, -1)
    k_norms = k_norms.reshape(B, H, T)

    q = mx.random.normal((B, H, 1, D)).astype(mx.float16)
    centroids = mx.array(_CODEBOOKS[bits], dtype=mx.float32)
    scores = planar_fused_qk_scores(q, k_packed, k_norms, centroids, 1.0 / math.sqrt(D), D, bits)
    mx.eval(scores)

    assert scores.shape == (B, H, 1, T), f"Wrong shape: {scores.shape}"
    assert not mx.any(mx.isnan(scores)).item(), "NaN in scores"


def test_all_bit_widths():
    """Verify fused kernel works at 2, 3, and 4 bits."""
    from turboquant.mlx_fused_attention import (
        planar_fused_qk_scores, _compress, _planar_rotate, _CODEBOOKS,
    )
    import math
    mx.random.seed(0)
    B, H, T, D = 1, 2, 30, 64

    for bits in [2, 3, 4]:
        k_raw = mx.random.normal((B * H * T, D)).astype(mx.float32)
        k_packed, k_norms = _compress(k_raw, bits, _planar_rotate)
        k_packed = k_packed.reshape(B, H, T, -1)
        k_norms = k_norms.reshape(B, H, T)

        q = mx.random.normal((B, H, 1, D)).astype(mx.float16)
        centroids = mx.array(_CODEBOOKS[bits], dtype=mx.float32)
        scores = planar_fused_qk_scores(q, k_packed, k_norms, centroids, 1.0 / math.sqrt(D), D, bits)
        mx.eval(scores)
        assert not mx.any(mx.isnan(scores)).item(), f"NaN at {bits}-bit"
