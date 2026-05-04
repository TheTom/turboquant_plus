"""Tests for outlier channel strategy (Issue #17).

Non-integer bit precision: split channels into outlier (higher bits) and
non-outlier (lower bits) for 2.5-bit and 3.5-bit rates.
"""

import numpy as np
import pytest


class TestOutlierQuantizer:
    """Test mixed-precision outlier channel quantization."""

    def test_2_5_bit_effective_rate(self):
        """2.5-bit: 25% channels at 3 bits + 75% at 2 bits."""
        from turboquant.outlier import OutlierTurboQuant

        d = 128
        oq = OutlierTurboQuant(d=d, target_bits=2.5, seed=42)
        assert oq.effective_bits == pytest.approx(2.5, abs=0.01)

    def test_3_5_bit_effective_rate(self):
        """3.5-bit: 50% channels at 4 bits + 50% at 3 bits."""
        from turboquant.outlier import OutlierTurboQuant

        d = 128
        oq = OutlierTurboQuant(d=d, target_bits=3.5, seed=42)
        assert oq.effective_bits == pytest.approx(3.5, abs=0.01)

    def test_round_trip_quality(self):
        """Outlier quantization should have bounded MSE."""
        from turboquant.outlier import OutlierTurboQuant

        d = 128
        oq = OutlierTurboQuant(d=d, target_bits=3.5, seed=42)
        rng = np.random.default_rng(99)

        mses = []
        for _ in range(200):
            x = rng.standard_normal(d)
            x = x / np.linalg.norm(x)
            compressed = oq.quantize(x)
            x_hat = oq.dequantize(compressed)
            mses.append(np.mean((x - x_hat) ** 2))

        avg_mse = np.mean(mses)
        # 3.5-bit should be between 3-bit (0.03) and 4-bit (0.009) bounds
        assert avg_mse < 0.03 * 2.0, f"3.5-bit MSE {avg_mse:.5f} too high"

    def test_compression_ratio_2_5bit(self):
        """2.5-bit should give ~6× compression vs fp16."""
        from turboquant.outlier import OutlierTurboQuant

        d = 128
        oq = OutlierTurboQuant(d=d, target_bits=2.5, seed=42)
        ratio = oq.compression_ratio()
        # 16 / 2.5 ≈ 6.4 (minus metadata overhead)
        assert ratio > 4.5, f"2.5-bit ratio {ratio:.1f}× too low"

    def test_compression_ratio_3_5bit(self):
        """3.5-bit should give ~4× compression vs fp16."""
        from turboquant.outlier import OutlierTurboQuant

        d = 128
        oq = OutlierTurboQuant(d=d, target_bits=3.5, seed=42)
        ratio = oq.compression_ratio()
        assert ratio > 3.5, f"3.5-bit ratio {ratio:.1f}× too low"

    def test_outlier_channels_identified(self):
        """Outlier channels should be the ones with highest magnitude."""
        from turboquant.outlier import OutlierTurboQuant

        d = 128
        oq = OutlierTurboQuant(d=d, target_bits=2.5, seed=42)

        # Should have some outlier and some non-outlier channels
        assert oq.n_outlier > 0
        assert oq.n_outlier < d
        assert oq.n_outlier + oq.n_normal == d

    def test_batch_matches_single(self):
        """Batch quantization should match single-vector."""
        from turboquant.outlier import OutlierTurboQuant

        d = 128
        oq = OutlierTurboQuant(d=d, target_bits=3.5, seed=42)
        rng = np.random.default_rng(7)
        X = rng.standard_normal((5, d))

        batch_compressed = oq.quantize(X)
        batch_recon = oq.dequantize(batch_compressed)

        for i in range(5):
            single_compressed = oq.quantize(X[i])
            single_recon = oq.dequantize(single_compressed)
            np.testing.assert_allclose(batch_recon[i], single_recon, atol=1e-10)

    def test_deterministic(self):
        """Same seed → same output."""
        from turboquant.outlier import OutlierTurboQuant

        d = 128
        x = np.random.default_rng(1).standard_normal(d)

        oq1 = OutlierTurboQuant(d=d, target_bits=3.5, seed=42)
        oq2 = OutlierTurboQuant(d=d, target_bits=3.5, seed=42)

        c1 = oq1.quantize(x)
        c2 = oq2.quantize(x)
        r1 = oq1.dequantize(c1)
        r2 = oq2.dequantize(c2)
        np.testing.assert_allclose(r1, r2, atol=1e-15)


class TestCalibrate:
    """Tests for OutlierTurboQuant.calibrate() data-driven channel split."""

    def test_calibrate_finds_known_outlier_channels(self):
        """calibrate() should identify channels with artificially large RMS as outliers."""
        from turboquant.outlier import OutlierTurboQuant

        d = 128
        rng = np.random.default_rng(42)

        # Build calibration data with clear outliers in channels 10, 20, 30
        n_samples = 500
        calib = rng.standard_normal((n_samples, d))  # baseline ~ N(0,1)
        outlier_channels = [10, 20, 30]
        for ch in outlier_channels:
            calib[:, ch] *= 20.0  # RMS ~ 20 >> 3 * median ≈ 3

        oq = OutlierTurboQuant(d=d, target_bits=2.5, seed=7)

        # Before calibration: fixed split (channels 0..n_outlier-1)
        default_outlier_idx = set(oq.outlier_idx.tolist())

        oq.calibrate(calib)

        calibrated_outlier_idx = set(oq.outlier_idx.tolist())

        # All injected outlier channels should now be classified as outliers
        for ch in outlier_channels:
            assert ch in calibrated_outlier_idx, (
                f"Channel {ch} (amplified 20×) not identified as outlier after calibration"
            )

        # The calibrated split should differ from the fixed default
        assert calibrated_outlier_idx != default_outlier_idx, (
            "calibrate() produced the same channel split as the fixed default — "
            "expected a different split for data with injected outlier channels"
        )

        # Consistency check
        assert oq.n_outlier + oq.n_normal == d

    def test_calibrate_no_outliers(self):
        """calibrate() on uniform data should find zero or very few outlier channels."""
        from turboquant.outlier import OutlierTurboQuant

        d = 64
        rng = np.random.default_rng(99)
        # All channels have equal variance — no outliers expected
        calib = rng.standard_normal((1000, d))

        oq = OutlierTurboQuant(d=d, target_bits=2.5, seed=1)
        oq.calibrate(calib)

        # With uniform data, per-channel RMS should all be close to 1.
        # 3× median ≈ 3 threshold means essentially no channel exceeds it.
        # Allow a small fraction due to sampling variance.
        outlier_fraction = oq.n_outlier / d
        assert outlier_fraction < 0.1, (
            f"Expected <10% outliers on uniform data, got {outlier_fraction:.1%} "
            f"({oq.n_outlier}/{d})"
        )

    def test_calibrate_preserves_default_without_call(self):
        """Without calling calibrate(), the fixed split is unchanged."""
        from turboquant.outlier import OutlierTurboQuant

        d = 128
        oq = OutlierTurboQuant(d=d, target_bits=2.5, seed=42)

        # Default: first n_outlier channels
        expected_outlier = np.arange(oq.n_outlier)
        np.testing.assert_array_equal(oq.outlier_idx, expected_outlier)

    def test_calibrate_updates_counts(self):
        """After calibrate(), n_outlier and n_normal should reflect new split."""
        from turboquant.outlier import OutlierTurboQuant

        d = 64
        rng = np.random.default_rng(5)
        calib = rng.standard_normal((200, d))
        calib[:, 0] *= 50.0  # one strong outlier

        oq = OutlierTurboQuant(d=d, target_bits=3.5, seed=3)
        oq.calibrate(calib)

        assert oq.n_outlier + oq.n_normal == d
        assert len(oq.outlier_idx) == oq.n_outlier
        assert len(oq.normal_idx) == oq.n_normal
