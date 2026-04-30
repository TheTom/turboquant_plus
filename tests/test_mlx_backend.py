"""Parity tests for the optional MLX backend."""

from __future__ import annotations

import numpy as np
import pytest

from turboquant.kv_cache import KVCacheCompressor
from turboquant.mlx_backend import (
    MLXKVCacheCompressor,
    MLXPolarQuant,
    MLXTurboQuant,
    MLX_AVAILABLE,
    to_numpy,
)
from turboquant.polar_quant import PolarQuant
from turboquant.turboquant import TurboQuant


mlx = pytest.importorskip("mlx.core", reason="MLX backend tests require the optional mlx package")


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX is not available")
class TestMLXPolarQuant:
    def test_quantize_matches_numpy(self):
        d = 128
        rng = np.random.default_rng(7)
        x = rng.standard_normal((8, d))

        numpy_pq = PolarQuant(d=d, bit_width=3, seed=42, norm_correction=True)
        mlx_pq = MLXPolarQuant(d=d, bit_width=3, seed=42, norm_correction=True, dtype="float64")

        numpy_idx, numpy_norms = numpy_pq.quantize(x)
        mlx_idx, mlx_norms = mlx_pq.quantize(x)
        mlx_idx_np = to_numpy(mlx_idx)
        mlx_norms_np = to_numpy(mlx_norms)

        np.testing.assert_array_equal(mlx_idx_np, numpy_idx)
        np.testing.assert_allclose(mlx_norms_np, numpy_norms, atol=1e-10)

        numpy_recon = numpy_pq.dequantize(numpy_idx, numpy_norms)
        mlx_recon = to_numpy(mlx_pq.dequantize(mlx_idx, mlx_norms))
        np.testing.assert_allclose(mlx_recon, numpy_recon, atol=1e-8)

    def test_single_vector_round_trip(self):
        d = 64
        rng = np.random.default_rng(11)
        x = rng.standard_normal(d)

        mlx_pq = MLXPolarQuant(d=d, bit_width=2, seed=99, dtype="float64")
        idx, norms = mlx_pq.quantize(x)
        x_hat = to_numpy(mlx_pq.dequantize(idx, norms))

        assert x_hat.shape == x.shape
        assert np.mean((x - x_hat) ** 2) < np.mean(x ** 2)


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX is not available")
class TestMLXTurboQuant:
    def test_turboquant_matches_numpy(self):
        d = 128
        rng = np.random.default_rng(21)
        x = rng.standard_normal((6, d))

        numpy_tq = TurboQuant(d=d, bit_width=3, seed=42, norm_correction=True)
        mlx_tq = MLXTurboQuant(d=d, bit_width=3, seed=42, norm_correction=True, dtype="float64")

        numpy_compressed = numpy_tq.quantize(x)
        mlx_compressed = mlx_tq.quantize(x)

        np.testing.assert_array_equal(to_numpy(mlx_compressed.mse_indices), numpy_compressed.mse_indices)
        np.testing.assert_allclose(to_numpy(mlx_compressed.vector_norms), numpy_compressed.vector_norms, atol=1e-10)
        np.testing.assert_array_equal(to_numpy(mlx_compressed.qjl_signs), numpy_compressed.qjl_signs)
        np.testing.assert_allclose(
            to_numpy(mlx_compressed.residual_norms),
            numpy_compressed.residual_norms,
            atol=1e-10,
        )

        numpy_recon = numpy_tq.dequantize(numpy_compressed)
        mlx_recon = to_numpy(mlx_tq.dequantize(mlx_compressed))
        np.testing.assert_allclose(mlx_recon, numpy_recon, atol=1e-8)


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX is not available")
class TestMLXKVCache:
    def test_kv_cache_matches_numpy(self):
        rng = np.random.default_rng(42)
        k = rng.standard_normal((2, 3, 8, 64))
        v = rng.standard_normal((2, 3, 8, 64))

        numpy_compressor = KVCacheCompressor(head_dim=64, k_bits=3, v_bits=3)
        mlx_compressor = MLXKVCacheCompressor(head_dim=64, k_bits=3, v_bits=3, dtype="float64")

        numpy_compressed = numpy_compressor.compress(k, v)
        mlx_compressed = mlx_compressor.compress(k, v)

        numpy_k, numpy_v = numpy_compressor.decompress(numpy_compressed)
        mlx_k, mlx_v = mlx_compressor.decompress(mlx_compressed)
        mlx_k = to_numpy(mlx_k)
        mlx_v = to_numpy(mlx_v)

        np.testing.assert_allclose(mlx_k, numpy_k, atol=1e-8)
        np.testing.assert_allclose(mlx_v, numpy_v, atol=1e-8)

    def test_attention_output_remains_reasonable(self):
        head_dim = 64
        seq_len = 16
        rng = np.random.default_rng(123)

        q = rng.standard_normal((1, head_dim))
        k = rng.standard_normal((seq_len, head_dim))
        v = rng.standard_normal((seq_len, head_dim))

        compressor = MLXKVCacheCompressor(head_dim=head_dim, k_bits=3, v_bits=3, dtype="float32")
        compressed = compressor.compress(k[np.newaxis, np.newaxis, :, :], v[np.newaxis, np.newaxis, :, :])
        k_hat, v_hat = compressor.decompress(compressed)
        k_hat = to_numpy(k_hat)[0, 0]
        v_hat = to_numpy(v_hat)[0, 0]

        scores_orig = q @ k.T / np.sqrt(head_dim)
        attn_orig = _softmax(scores_orig)
        out_orig = attn_orig @ v

        scores_comp = q @ k_hat.T / np.sqrt(head_dim)
        attn_comp = _softmax(scores_comp)
        out_comp = attn_comp @ v_hat

        cosine = np.dot(out_orig.ravel(), out_comp.ravel()) / (
            np.linalg.norm(out_orig) * np.linalg.norm(out_comp)
        )
        assert cosine > 0.5


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)
