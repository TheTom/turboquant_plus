"""Tests for KV cache integration layer."""

import tempfile
import numpy as np
import pytest

from turboquant.kv_cache import KVCacheCompressor, CompressedKVCache
from turboquant.turboquant import CompressedVector


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

        # K: 3 bits/val + 32-bit norm, V: 3 bits/val + 32-bit norm
        # Both K and V include per-vector norm (float32) for rescaling.
        # Ratio vs fp16 (16 bits/val): 16*128 / (128*3 + 32 + 128*3 + 32) / 2 ≈ 2.46x
        assert stats["compression_ratio"] > 2.0
        assert stats["compressed_mb"] < stats["original_mb"]

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


class TestCompressedVectorSerialization:
    """Tests for CompressedVector.to_bytes() / from_bytes()."""

    def test_round_trip_single_vector(self):
        """Serialize and deserialize a single-vector CompressedVector."""
        from turboquant.turboquant import TurboQuant

        d = 64
        tq = TurboQuant(d=d, bit_width=3, seed=42)
        rng = np.random.default_rng(1)
        x = rng.standard_normal(d)

        cv = tq.quantize(x)
        data = cv.to_bytes()
        cv2 = CompressedVector.from_bytes(data)

        assert cv2.bit_width == cv.bit_width
        np.testing.assert_array_equal(cv2.mse_indices, cv.mse_indices)
        np.testing.assert_allclose(cv2.vector_norms, cv.vector_norms)
        np.testing.assert_array_equal(cv2.qjl_signs, cv.qjl_signs)
        np.testing.assert_allclose(cv2.residual_norms, cv.residual_norms)

    def test_round_trip_batch(self):
        """Serialize and deserialize a batched CompressedVector."""
        from turboquant.turboquant import TurboQuant

        d = 64
        batch = 8
        tq = TurboQuant(d=d, bit_width=2, seed=7)
        rng = np.random.default_rng(2)
        X = rng.standard_normal((batch, d))

        cv = tq.quantize(X)
        data = cv.to_bytes()
        cv2 = CompressedVector.from_bytes(data)

        assert cv2.bit_width == cv.bit_width
        np.testing.assert_array_equal(cv2.mse_indices, cv.mse_indices)
        np.testing.assert_allclose(cv2.vector_norms, cv.vector_norms)
        np.testing.assert_array_equal(cv2.qjl_signs, cv.qjl_signs)
        np.testing.assert_allclose(cv2.residual_norms, cv.residual_norms)

    def test_invalid_magic_raises(self):
        """from_bytes() should raise ValueError on corrupt/wrong data."""
        bad_data = b"XXXX" + b"\x00" * 20
        with pytest.raises(ValueError, match="Invalid magic bytes"):
            CompressedVector.from_bytes(bad_data)


class TestCompressedKVCacheSaveLoad:
    """Tests for CompressedKVCache.save() / load()."""

    def test_save_load_round_trip(self):
        """Save and load should produce a cache that decompresses to the same result."""
        head_dim = 64
        num_layers, num_heads, seq_len = 2, 2, 8

        compressor = KVCacheCompressor(head_dim=head_dim, k_bits=3, v_bits=3, seed=42)
        rng = np.random.default_rng(99)
        k = rng.standard_normal((num_layers, num_heads, seq_len, head_dim))
        v = rng.standard_normal((num_layers, num_heads, seq_len, head_dim))

        original_cache = compressor.compress(k, v)
        k_orig, v_orig = compressor.decompress(original_cache)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name

        try:
            original_cache.save(path)
            loaded_cache = CompressedKVCache.load(path)
        finally:
            import os
            os.unlink(path)

        assert loaded_cache.num_layers == num_layers
        assert loaded_cache.num_heads == num_heads
        assert loaded_cache.seq_len == seq_len
        assert loaded_cache.head_dim == head_dim
        assert loaded_cache.k_bit_width == 3
        assert loaded_cache.v_bit_width == 3

        k_loaded, v_loaded = compressor.decompress(loaded_cache)
        np.testing.assert_allclose(k_loaded, k_orig, atol=1e-6,
                                   err_msg="K cache changed after save/load")
        np.testing.assert_allclose(v_loaded, v_orig, atol=1e-6,
                                   err_msg="V cache changed after save/load")


class TestStreamingAPI:
    """Tests for the compress_token() / get_compressed_cache() streaming API."""

    def test_streaming_produces_same_result_as_batch(self):
        """Token-by-token streaming should produce the same compressed output as batch compress.

        Both use the same quantizer objects (same rotation matrices and codebooks),
        so individual token compressions must match the batch-compressed result.
        """
        head_dim = 64
        num_layers, num_heads, seq_len = 2, 2, 8

        rng = np.random.default_rng(42)
        k_cache = rng.standard_normal((num_layers, num_heads, seq_len, head_dim))
        v_cache = rng.standard_normal((num_layers, num_heads, seq_len, head_dim))

        # Batch compress
        compressor_batch = KVCacheCompressor(head_dim=head_dim, k_bits=3, v_bits=3, seed=42)
        batch_compressed = compressor_batch.compress(k_cache, v_cache)

        # Stream token-by-token (same seed → same quantizer state)
        compressor_stream = KVCacheCompressor(head_dim=head_dim, k_bits=3, v_bits=3, seed=42)
        for t in range(seq_len):
            for layer in range(num_layers):
                for head in range(num_heads):
                    compressor_stream.compress_token(
                        k_cache[layer, head, t, :],
                        v_cache[layer, head, t, :],
                        layer=layer, head=head,
                    )

        stream_compressed = compressor_stream.get_compressed_cache()

        # Check metadata
        assert stream_compressed.num_layers == num_layers
        assert stream_compressed.num_heads == num_heads
        assert stream_compressed.seq_len == seq_len

        # Check that decompressed results match
        k_batch, v_batch = compressor_batch.decompress(batch_compressed)
        k_stream, v_stream = compressor_stream.decompress(stream_compressed)

        np.testing.assert_allclose(k_stream, k_batch, atol=1e-10,
                                   err_msg="Streaming K cache differs from batch K cache")
        np.testing.assert_allclose(v_stream, v_batch, atol=1e-10,
                                   err_msg="Streaming V cache differs from batch V cache")

    def test_get_compressed_cache_returns_valid_cache(self):
        """get_compressed_cache() returns a CompressedKVCache that decompresses without error."""
        from turboquant.kv_cache import CompressedKVCache

        head_dim = 64
        compressor = KVCacheCompressor(head_dim=head_dim, k_bits=3, v_bits=3, seed=7)
        rng = np.random.default_rng(55)

        num_layers, num_heads, seq_len = 1, 2, 4
        for t in range(seq_len):
            for layer in range(num_layers):
                for head in range(num_heads):
                    compressor.compress_token(
                        rng.standard_normal(head_dim),
                        rng.standard_normal(head_dim),
                        layer=layer, head=head,
                    )

        cache = compressor.get_compressed_cache()

        assert isinstance(cache, CompressedKVCache)
        assert cache.num_layers == num_layers
        assert cache.num_heads == num_heads
        assert cache.seq_len == seq_len
        assert cache.head_dim == head_dim
        assert cache.k_bit_width == 3
        assert cache.v_bit_width == 3

        # Should decompress without error
        k_hat, v_hat = compressor.decompress(cache)
        assert k_hat.shape == (num_layers, num_heads, seq_len, head_dim)
        assert v_hat.shape == (num_layers, num_heads, seq_len, head_dim)

    def test_get_compressed_cache_empty(self):
        """get_compressed_cache() on a fresh compressor returns an empty cache."""
        from turboquant.kv_cache import CompressedKVCache

        compressor = KVCacheCompressor(head_dim=64, k_bits=3, v_bits=3)
        cache = compressor.get_compressed_cache()

        assert isinstance(cache, CompressedKVCache)
        assert cache.num_layers == 0
        assert cache.num_heads == 0
        assert cache.seq_len == 0


def _softmax(x):
    """Simple softmax for testing."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)
