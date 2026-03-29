"""Tests for layer-adaptive KV cache compression."""

import numpy as np
import pytest

from turboquant.layer_adaptive import (
    LayerAdaptiveCompressor,
    CompressedLayerAdaptiveKVCache,
    default_40layer_config,
    make_layer_config,
)


class TestMakeLayerConfig:
    """Test configuration builders."""

    def test_default_40layer(self):
        """Default config: 32 layers at 3-bit, 8 layers at 8-bit."""
        config = default_40layer_config()
        assert len(config) == 40
        for i in range(32):
            assert config[i] == 3
        for i in range(32, 40):
            assert config[i] == 8

    def test_make_layer_config_basic(self):
        """make_layer_config should split at the right cutoff."""
        config = make_layer_config(total_layers=10, default_bits=3,
                                   high_bits=8, high_frac=0.2)
        assert len(config) == 10
        for i in range(8):
            assert config[i] == 3
        for i in range(8, 10):
            assert config[i] == 8

    def test_make_layer_config_all_high(self):
        """high_frac=1.0 should make all layers high-precision."""
        config = make_layer_config(total_layers=5, default_bits=2,
                                   high_bits=4, high_frac=1.0)
        for v in config.values():
            assert v == 4

    def test_make_layer_config_none_high(self):
        """high_frac=0.0 should make all layers default."""
        config = make_layer_config(total_layers=5, default_bits=3,
                                   high_bits=8, high_frac=0.0)
        for v in config.values():
            assert v == 3

    def test_make_layer_config_custom_split(self):
        """50% high-precision layers."""
        config = make_layer_config(total_layers=20, default_bits=3,
                                   high_bits=8, high_frac=0.5)
        low_count = sum(1 for v in config.values() if v == 3)
        high_count = sum(1 for v in config.values() if v == 8)
        assert low_count == 10
        assert high_count == 10


class TestLayerAdaptiveCompressor:
    """Test the LayerAdaptiveCompressor class."""

    def _make_compressor(self, num_layers=4, head_dim=64):
        config = make_layer_config(num_layers, default_bits=3,
                                   high_bits=4, high_frac=0.25)
        return LayerAdaptiveCompressor(head_dim=head_dim, layers_config=config)

    def test_round_trip_shape(self):
        """Output shape matches input shape."""
        num_layers, num_heads, seq_len, head_dim = 4, 2, 8, 64
        compressor = self._make_compressor(num_layers, head_dim)
        rng = np.random.default_rng(42)

        k = rng.standard_normal((num_layers, num_heads, seq_len, head_dim))
        v = rng.standard_normal((num_layers, num_heads, seq_len, head_dim))

        compressed = compressor.compress(k, v)
        k_hat, v_hat = compressor.decompress(compressed)

        assert k_hat.shape == k.shape
        assert v_hat.shape == v.shape

    def test_round_trip_quality(self):
        """Decompressed values have bounded error."""
        num_layers, num_heads, seq_len, head_dim = 4, 2, 16, 128
        compressor = self._make_compressor(num_layers, head_dim)
        rng = np.random.default_rng(42)

        k = rng.standard_normal((num_layers, num_heads, seq_len, head_dim))
        v = rng.standard_normal((num_layers, num_heads, seq_len, head_dim))

        compressed = compressor.compress(k, v)
        k_hat, v_hat = compressor.decompress(compressed)

        mse = np.mean((k - k_hat) ** 2)
        assert mse < 1.0, f"K MSE {mse:.4f} too high"

    def test_missing_layer_raises(self):
        """compress should raise if a layer index is not in config."""
        config = {0: 3, 1: 3}  # only 2 layers
        compressor = LayerAdaptiveCompressor(head_dim=64, layers_config=config)
        rng = np.random.default_rng(42)
        k = rng.standard_normal((4, 2, 8, 64))  # 4 layers
        v = rng.standard_normal((4, 2, 8, 64))

        with pytest.raises(ValueError, match="Layer 2 not in layers_config"):
            compressor.compress(k, v)

    def test_metadata_stored(self):
        """Compressed cache stores correct metadata."""
        num_layers, num_heads, seq_len, head_dim = 4, 2, 8, 64
        compressor = self._make_compressor(num_layers, head_dim)
        rng = np.random.default_rng(42)

        k = rng.standard_normal((num_layers, num_heads, seq_len, head_dim))
        v = rng.standard_normal((num_layers, num_heads, seq_len, head_dim))

        compressed = compressor.compress(k, v)
        assert compressed.num_layers == num_layers
        assert compressed.num_heads == num_heads
        assert compressed.seq_len == seq_len
        assert compressed.head_dim == head_dim
        assert len(compressed.layer_caches) == num_layers

    def test_per_layer_caches_have_correct_bit_widths(self):
        """Each per-layer cache should record its own bit-width."""
        config = {0: 3, 1: 3, 2: 4, 3: 4}
        compressor = LayerAdaptiveCompressor(head_dim=64, layers_config=config)
        rng = np.random.default_rng(42)

        k = rng.standard_normal((4, 2, 8, 64))
        v = rng.standard_normal((4, 2, 8, 64))

        compressed = compressor.compress(k, v)
        assert compressed.layer_caches[0].k_bit_width == 3
        assert compressed.layer_caches[1].k_bit_width == 3
        assert compressed.layer_caches[2].k_bit_width == 4
        assert compressed.layer_caches[3].k_bit_width == 4

    def test_v_bits_override(self):
        """V cache can use different bits than K cache per layer."""
        config = {0: 3, 1: 3}
        v_override = {0: 4, 1: 4}
        compressor = LayerAdaptiveCompressor(
            head_dim=64, layers_config=config, v_bits_override=v_override,
        )
        rng = np.random.default_rng(42)

        k = rng.standard_normal((2, 2, 8, 64))
        v = rng.standard_normal((2, 2, 8, 64))

        compressed = compressor.compress(k, v)
        assert compressed.layer_caches[0].k_bit_width == 3
        assert compressed.layer_caches[0].v_bit_width == 4


class TestEffectiveStats:
    """Test statistics methods."""

    def test_effective_bits_uniform(self):
        """All layers same bits -> effective bits equals that value."""
        config = {i: 3 for i in range(10)}
        comp = LayerAdaptiveCompressor(head_dim=64, layers_config=config)
        assert comp.effective_bits_per_value() == 3.0

    def test_effective_bits_mixed(self):
        """Mixed bit-widths should produce weighted average."""
        # 8 layers at 3-bit, 2 layers at 8-bit
        config = {}
        for i in range(8):
            config[i] = 3
        for i in range(8, 10):
            config[i] = 8
        comp = LayerAdaptiveCompressor(head_dim=64, layers_config=config)
        expected = (8 * 3 + 2 * 8) / 10.0  # 4.0
        assert comp.effective_bits_per_value() == pytest.approx(expected)

    def test_effective_compression_ratio(self):
        """Compression ratio should be original_bits / avg_bits."""
        config = {i: 4 for i in range(5)}
        comp = LayerAdaptiveCompressor(head_dim=64, layers_config=config)
        assert comp.effective_compression_ratio(16) == pytest.approx(4.0)

    def test_layer_summary(self):
        """layer_summary returns correct per-layer info."""
        config = {0: 3, 1: 8}
        comp = LayerAdaptiveCompressor(head_dim=64, layers_config=config)
        summary = comp.layer_summary()
        assert len(summary) == 2
        assert summary[0]["layer"] == 0
        assert summary[0]["k_bits"] == 3
        assert summary[1]["layer"] == 1
        assert summary[1]["k_bits"] == 8

    def test_empty_config(self):
        """Empty config -> 0 effective bits."""
        comp = LayerAdaptiveCompressor(head_dim=64, layers_config={})
        assert comp.effective_bits_per_value() == 0.0
        assert comp.effective_compression_ratio() == float("inf")
