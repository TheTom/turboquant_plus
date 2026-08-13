"""Tests for temporal decay configuration and compression."""

import numpy as np
import pytest

from turboquant.temporal_decay import TemporalDecayConfig, TemporalDecayCompressor


class TestTemporalDecayConfig:
    """Test the configuration dataclass."""

    def test_defaults(self):
        cfg = TemporalDecayConfig()
        assert cfg.recent_bits == 3
        assert cfg.old_bits == 2
        assert cfg.decay_threshold == 256
        assert cfg.layer_aware is True

    def test_custom(self):
        cfg = TemporalDecayConfig(recent_bits=4, old_bits=2,
                                  decay_threshold=512, layer_aware=False)
        assert cfg.recent_bits == 4
        assert cfg.decay_threshold == 512
        assert cfg.layer_aware is False


class TestGetBitsForToken:
    """Test bit-width selection logic."""

    def test_recent_token_gets_recent_bits(self):
        """Token younger than threshold -> recent_bits."""
        cfg = TemporalDecayConfig(recent_bits=3, old_bits=2,
                                  decay_threshold=256, layer_aware=False)
        tdc = TemporalDecayCompressor(head_dim=64, config=cfg)
        assert tdc.get_bits_for_token(age=0, layer=0, total_layers=40) == 3
        assert tdc.get_bits_for_token(age=255, layer=0, total_layers=40) == 3

    def test_old_token_gets_old_bits(self):
        """Token older than threshold -> old_bits."""
        cfg = TemporalDecayConfig(recent_bits=3, old_bits=2,
                                  decay_threshold=256, layer_aware=False)
        tdc = TemporalDecayCompressor(head_dim=64, config=cfg)
        assert tdc.get_bits_for_token(age=256, layer=0, total_layers=40) == 2
        assert tdc.get_bits_for_token(age=1000, layer=0, total_layers=40) == 2

    def test_layer_aware_late_layer_keeps_recent(self):
        """Late layers (last 20%) always use recent_bits."""
        cfg = TemporalDecayConfig(recent_bits=3, old_bits=2,
                                  decay_threshold=256, layer_aware=True)
        tdc = TemporalDecayCompressor(head_dim=64, config=cfg)
        # Layer 39 in a 40-layer model (last 20% = layers 32-39)
        assert tdc.get_bits_for_token(age=1000, layer=39, total_layers=40) == 3
        assert tdc.get_bits_for_token(age=1000, layer=35, total_layers=40) == 3
        assert tdc.get_bits_for_token(age=1000, layer=32, total_layers=40) == 3

    def test_layer_aware_early_layer_decays_faster(self):
        """Early layers decay faster (lower effective threshold)."""
        cfg = TemporalDecayConfig(recent_bits=3, old_bits=2,
                                  decay_threshold=256, layer_aware=True)
        tdc = TemporalDecayCompressor(head_dim=64, config=cfg)
        # Layer 0: effective_threshold = 256 * 0.5 = 128
        # Age 130 > 128 -> old_bits
        assert tdc.get_bits_for_token(age=130, layer=0, total_layers=40) == 2
        # But same age at a later (but still early) layer has higher threshold
        # Layer 16: scale = 0.5 + 0.5*(16/32) = 0.75, threshold = 192
        assert tdc.get_bits_for_token(age=130, layer=16, total_layers=40) == 3

    def test_layer_aware_boundary(self):
        """Token at exactly the effective threshold should get old_bits."""
        cfg = TemporalDecayConfig(recent_bits=3, old_bits=2,
                                  decay_threshold=200, layer_aware=True)
        tdc = TemporalDecayCompressor(head_dim=64, config=cfg)
        # Layer 0: eff_thresh = 200 * 0.5 = 100
        assert tdc.get_bits_for_token(age=99, layer=0, total_layers=40) == 3
        assert tdc.get_bits_for_token(age=100, layer=0, total_layers=40) == 2


class TestGetBitsMap:
    """Test vectorized bit-width mapping."""

    def test_shape(self):
        tdc = TemporalDecayCompressor(head_dim=64)
        ages = np.array([0, 100, 200, 300, 500])
        bits = tdc.get_bits_map(ages, layer_idx=0, total_layers=40)
        assert bits.shape == (5,)

    def test_values_match_scalar(self):
        """Vectorized result should match per-element calls."""
        tdc = TemporalDecayCompressor(head_dim=64)
        ages = np.array([0, 50, 128, 256, 512])
        bits_map = tdc.get_bits_map(ages, layer_idx=5, total_layers=40)
        for i, age in enumerate(ages):
            expected = tdc.get_bits_for_token(int(age), layer=5, total_layers=40)
            assert bits_map[i] == expected


class TestCompressWithDecay:
    """Test compression/decompression with temporal decay."""

    def _make_compressor(self, head_dim=64):
        cfg = TemporalDecayConfig(recent_bits=3, old_bits=2,
                                  decay_threshold=256, layer_aware=False)
        return TemporalDecayCompressor(head_dim=head_dim, config=cfg)

    def test_round_trip_shape(self):
        """Output shape matches input."""
        head_dim = 64
        seq_len = 32
        tdc = self._make_compressor(head_dim)
        rng = np.random.default_rng(42)

        keys = rng.standard_normal((seq_len, head_dim))
        values = rng.standard_normal((seq_len, head_dim))
        ages = np.arange(seq_len) * 16  # ages 0, 16, 32, ..., 496

        compressed = tdc.compress_with_decay(keys, values, ages,
                                             layer_idx=0, total_layers=40)
        k_hat, v_hat = tdc.decompress_with_decay(compressed)

        assert k_hat.shape == keys.shape
        assert v_hat.shape == values.shape

    def test_round_trip_quality(self):
        """Reconstruction error should be bounded."""
        head_dim = 128
        seq_len = 64
        tdc = self._make_compressor(head_dim)
        rng = np.random.default_rng(42)

        keys = rng.standard_normal((seq_len, head_dim))
        values = rng.standard_normal((seq_len, head_dim))
        ages = np.arange(seq_len) * 8

        compressed = tdc.compress_with_decay(keys, values, ages,
                                             layer_idx=0, total_layers=40)
        k_hat, v_hat = tdc.decompress_with_decay(compressed)

        k_mse = np.mean((keys - k_hat) ** 2)
        v_mse = np.mean((values - v_hat) ** 2)
        assert k_mse < 1.0, f"K MSE {k_mse:.4f} too high"
        assert v_mse < 1.0, f"V MSE {v_mse:.4f} too high"

    def test_groups_reflect_ages(self):
        """Tokens should be split into groups based on age threshold."""
        tdc = self._make_compressor(64)
        rng = np.random.default_rng(42)

        keys = rng.standard_normal((10, 64))
        values = rng.standard_normal((10, 64))
        # First 5 tokens recent, last 5 old
        ages = np.array([0, 10, 20, 30, 40, 300, 400, 500, 600, 700])

        compressed = tdc.compress_with_decay(keys, values, ages,
                                             layer_idx=0, total_layers=40)
        groups = compressed["groups"]

        # Should have 2 groups: recent (3-bit) and old (2-bit)
        bits_seen = {g["bits"] for g in groups}
        assert bits_seen == {2, 3}

    def test_all_recent(self):
        """All tokens below threshold -> single group at recent_bits."""
        tdc = self._make_compressor(64)
        rng = np.random.default_rng(42)

        keys = rng.standard_normal((5, 64))
        values = rng.standard_normal((5, 64))
        ages = np.array([0, 10, 20, 30, 40])

        compressed = tdc.compress_with_decay(keys, values, ages,
                                             layer_idx=0, total_layers=40)
        assert len(compressed["groups"]) == 1
        assert compressed["groups"][0]["bits"] == 3

    def test_all_old(self):
        """All tokens above threshold -> single group at old_bits."""
        tdc = self._make_compressor(64)
        rng = np.random.default_rng(42)

        keys = rng.standard_normal((5, 64))
        values = rng.standard_normal((5, 64))
        ages = np.array([300, 400, 500, 600, 700])

        compressed = tdc.compress_with_decay(keys, values, ages,
                                             layer_idx=0, total_layers=40)
        assert len(compressed["groups"]) == 1
        assert compressed["groups"][0]["bits"] == 2

    def test_bits_map_in_result(self):
        """Compressed result should contain the bits_map."""
        tdc = self._make_compressor(64)
        rng = np.random.default_rng(42)

        keys = rng.standard_normal((4, 64))
        values = rng.standard_normal((4, 64))
        ages = np.array([0, 100, 300, 500])

        compressed = tdc.compress_with_decay(keys, values, ages,
                                             layer_idx=0, total_layers=40)
        assert "bits_map" in compressed
        assert len(compressed["bits_map"]) == 4


class TestMemorySavingsEstimate:
    """Test memory savings calculation."""

    def test_all_same_bits(self):
        """When all tokens use same bits, ratio should be consistent."""
        cfg = TemporalDecayConfig(recent_bits=3, old_bits=2,
                                  decay_threshold=256, layer_aware=False)
        tdc = TemporalDecayCompressor(head_dim=128, config=cfg)

        ages = np.zeros(100)  # all recent
        result = tdc.memory_savings_estimate(ages, layer_idx=0, total_layers=40)
        assert result["ratio"] > 1.0
        assert result["avg_bits"] == 3.0

    def test_mixed_savings(self):
        """Mixed ages should give intermediate compression."""
        cfg = TemporalDecayConfig(recent_bits=4, old_bits=2,
                                  decay_threshold=100, layer_aware=False)
        tdc = TemporalDecayCompressor(head_dim=128, config=cfg)

        ages = np.concatenate([np.zeros(50), np.full(50, 200)])
        result = tdc.memory_savings_estimate(ages, layer_idx=0, total_layers=40)
        assert result["avg_bits"] == pytest.approx(3.0)
        assert result["ratio"] > 1.0

    def test_higher_old_bits_less_savings(self):
        """Higher old_bits should give less compression."""
        cfg_low = TemporalDecayConfig(recent_bits=4, old_bits=2,
                                      decay_threshold=100, layer_aware=False)
        cfg_high = TemporalDecayConfig(recent_bits=4, old_bits=3,
                                       decay_threshold=100, layer_aware=False)
        tdc_low = TemporalDecayCompressor(head_dim=128, config=cfg_low)
        tdc_high = TemporalDecayCompressor(head_dim=128, config=cfg_high)

        ages = np.full(100, 200)  # all old
        r_low = tdc_low.memory_savings_estimate(ages, 0, 40)
        r_high = tdc_high.memory_savings_estimate(ages, 0, 40)
        assert r_low["ratio"] > r_high["ratio"]


class TestDefaultConfig:
    """Test compressor with default config."""

    def test_default_config_compresses(self):
        """Default config should work out of the box."""
        tdc = TemporalDecayCompressor(head_dim=64)
        rng = np.random.default_rng(42)

        keys = rng.standard_normal((20, 64))
        values = rng.standard_normal((20, 64))
        ages = np.arange(20) * 20

        compressed = tdc.compress_with_decay(keys, values, ages,
                                             layer_idx=10, total_layers=40)
        k_hat, v_hat = tdc.decompress_with_decay(compressed)
        assert k_hat.shape == keys.shape
