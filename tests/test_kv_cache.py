"""Tests for KV cache integration layer."""

import numpy as np
import pytest

from turboquant.kv_cache import KVCacheCompressor


class TestKVCacheCompressor:
    """Test full KV cache compress → decompress pipeline."""

    def test_round_trip_shape(self):
        """Output shape should match input shape."""
        head_dim = 64
        num_layers, num_heads, seq_len = 2, 4, 16

        compressor = KVCacheCompressor(head_dim=head_dim, k_bits=3, v_bits=3)
        rng = np.random.default_rng(42)

        k = rng.standard_normal((num_layers, num_heads, seq_len, head_dim))
        v = rng.standard_normal((num_layers, num_heads, seq_len, head_dim))

        compressed = compressor.compress(k, v)
        k_hat, v_hat = compressor.decompress(compressed)

        assert k_hat.shape == k.shape
        assert v_hat.shape == v.shape

    def test_round_trip_quality(self):
        """Decompressed cache should have bounded error."""
        head_dim = 128
        num_layers, num_heads, seq_len = 2, 4, 32

        compressor = KVCacheCompressor(head_dim=head_dim, k_bits=3, v_bits=3)
        rng = np.random.default_rng(42)

        k = rng.standard_normal((num_layers, num_heads, seq_len, head_dim))
        v = rng.standard_normal((num_layers, num_heads, seq_len, head_dim))

        # Normalize to unit vectors (paper bounds are for unit vectors)
        k_norm = np.linalg.norm(k, axis=-1, keepdims=True)
        k_norm[k_norm == 0] = 1.0
        k = k / k_norm
        v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
        v_norm[v_norm == 0] = 1.0
        v = v / v_norm

        compressed = compressor.compress(k, v)
        k_hat, v_hat = compressor.decompress(compressed)

        k_mse = np.mean((k - k_hat) ** 2)
        v_mse = np.mean((v - v_hat) ** 2)

        # 3-bit TurboQuant for K: paper MSE bound = 0.03 (3× slack)
        assert k_mse < 0.09, f"K cache MSE {k_mse:.4f} too high"
        # 3-bit PolarQuant for V: paper MSE bound = 0.03 (3× slack)
        assert v_mse < 0.09, f"V cache MSE {v_mse:.4f} too high"

    def test_attention_score_preservation(self):
        """Compressed KV cache should produce similar attention scores."""
        head_dim = 64
        seq_len = 16

        compressor = KVCacheCompressor(head_dim=head_dim, k_bits=3, v_bits=3)
        rng = np.random.default_rng(42)

        # Single layer, single head for simplicity
        q = rng.standard_normal((1, head_dim))  # query
        k = rng.standard_normal((seq_len, head_dim))
        v = rng.standard_normal((seq_len, head_dim))

        # Original attention
        scores_orig = q @ k.T / np.sqrt(head_dim)
        attn_orig = _softmax(scores_orig)
        out_orig = attn_orig @ v

        # Compressed
        k_cache = k[np.newaxis, np.newaxis, :, :]  # (1, 1, seq, dim)
        v_cache = v[np.newaxis, np.newaxis, :, :]

        compressed = compressor.compress(k_cache, v_cache)
        k_hat, v_hat = compressor.decompress(compressed)

        k_hat = k_hat[0, 0]  # back to (seq, dim)
        v_hat = v_hat[0, 0]

        scores_comp = q @ k_hat.T / np.sqrt(head_dim)
        attn_comp = _softmax(scores_comp)
        out_comp = attn_comp @ v_hat

        # Output should be similar
        cosine_sim = np.dot(out_orig.ravel(), out_comp.ravel()) / (
            np.linalg.norm(out_orig) * np.linalg.norm(out_comp)
        )
        # 3-bit quantization on both K and V with small d=64 and seq_len=16
        # introduces significant error. Cosine > 0.5 is reasonable here.
        # Higher d and higher bit-width would give much better similarity.
        assert cosine_sim > 0.5, f"Attention output cosine similarity {cosine_sim:.3f} too low"

    def test_memory_stats(self):
        """Memory stats should show compression."""
        compressor = KVCacheCompressor(head_dim=128, k_bits=3, v_bits=3)
        stats = compressor.memory_stats(seq_len=1024, num_layers=32, num_heads=32)

        # Combined K/V fp16 baseline: 32 bits/value pair.
        # For 3-bit K and V, actual stored metadata gives ~4.74x compression.
        assert stats["compression_ratio"] > 4.0
        assert stats["compressed_mb"] < stats["original_mb"]

    def test_memory_stats_exact_accounting(self):
        """Memory stats should match the actual K/V storage layout."""
        compressor = KVCacheCompressor(head_dim=128, k_bits=3, v_bits=3)
        stats = compressor.memory_stats(seq_len=1, num_layers=1, num_heads=1)

        # One K vector and one V vector at head_dim=128:
        # - Original fp16 K/V pair: 128 * 2 bytes * 2 tensors = 512 bytes
        # - K compressed: 128 * 3 bits + 64 bits of norms = 448 bits = 56 bytes
        # - V compressed: 128 * 3 bits + 32-bit norm = 416 bits = 52 bytes
        # - Total compressed = 108 bytes
        assert stats["original_mb"] == pytest.approx(512 / 1024 / 1024)
        assert stats["compressed_mb"] == pytest.approx(108 / 1024 / 1024)
        assert stats["compression_ratio"] == pytest.approx(512 / 108)

    def test_metadata_stored(self):
        """Compressed cache should store correct metadata."""
        compressor = KVCacheCompressor(head_dim=64, k_bits=3, v_bits=3)
        rng = np.random.default_rng(42)

        k = rng.standard_normal((2, 4, 8, 64))
        v = rng.standard_normal((2, 4, 8, 64))

        compressed = compressor.compress(k, v)

        assert compressed.num_layers == 2
        assert compressed.num_heads == 4
        assert compressed.seq_len == 8
        assert compressed.head_dim == 64
        assert compressed.k_bit_width == 3
        assert compressed.v_bit_width == 3


def _softmax(x):
    """Simple softmax for testing."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)
