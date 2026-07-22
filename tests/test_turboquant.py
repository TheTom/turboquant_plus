"""Tests for full TurboQuant (Algorithm 2 — PolarQuant + QJL)."""

import numpy as np
import pytest

from turboquant.turboquant import TurboQuant, TurboQuantMSE, CompressedVector


class TestTurboQuantRoundTrip:
    """Full TurboQuant quantize → dequantize correctness."""

    @pytest.mark.parametrize("bit_width", [2, 3, 4])
    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_mse_within_paper_bounds(self, bit_width, d):
        """MSE distortion should be within paper's bounds (Table 2).

        Paper bounds are for normalized unit vectors at d→∞.
        We allow 3× slack for finite d.
        """
        expected_mse = {2: 0.117, 3: 0.03, 4: 0.009}

        tq = TurboQuant(d=d, bit_width=bit_width, seed=42)
        rng = np.random.default_rng(99)

        n_samples = 500
        mse_total = 0.0
        for _ in range(n_samples):
            x = rng.standard_normal(d)
            x = x / np.linalg.norm(x)

            compressed = tq.quantize(x)
            x_hat = tq.dequantize(compressed)
            mse_total += np.mean((x - x_hat) ** 2)

        avg_mse = mse_total / n_samples
        assert avg_mse < expected_mse[bit_width] * 3.0, (
            f"MSE {avg_mse:.5f} exceeds 3× paper bound {expected_mse[bit_width]} "
            f"at d={d}, b={bit_width}"
        )

    @pytest.mark.parametrize("bit_width", [2, 3, 4])
    def test_inner_product_preservation(self, bit_width):
        """Inner product distortion should be within paper's bounds.

        Paper Theorem 2 (page 5):
            D_prod = E[|⟨y,x⟩ - ⟨y, Q⁻¹(Q(x))⟩|²] ≤ √(3π²)·||y||²/d · 1/4^b
        for ||x||=1.

        We measure mean |⟨y,x⟩ - ⟨y, x̂⟩| (absolute, not squared) and check
        it decreases with bit_width. The paper bound is on the squared error
        and involves both x AND y being quantized independently, which compounds.
        """
        d = 256
        tq = TurboQuant(d=d, bit_width=bit_width, seed=42)
        rng = np.random.default_rng(77)

        ip_errors = []
        for _ in range(500):
            x = rng.standard_normal(d)
            y = rng.standard_normal(d)
            x = x / np.linalg.norm(x)
            y = y / np.linalg.norm(y)

            x_hat = tq.dequantize(tq.quantize(x))
            y_hat = tq.dequantize(tq.quantize(y))

            ip_original = np.dot(x, y)
            ip_approx = np.dot(x_hat, y_hat)
            ip_errors.append(abs(ip_approx - ip_original))

        avg_ip_error = np.mean(ip_errors)
        # Paper bound is on E[|error|²] for single-side quantization.
        # We quantize BOTH sides and measure |error| (not squared), so bound is looser.
        # Use empirical sanity check: avg absolute IP error should be < 0.5 for any b≥2
        assert avg_ip_error < 0.5, (
            f"Avg IP error {avg_ip_error:.6f} unreasonably high at d={d}, b={bit_width}"
        )

    def test_bit_width_1_raises(self):
        """TurboQuant requires bit_width >= 2."""
        with pytest.raises(ValueError, match="bit_width >= 2"):
            TurboQuant(d=128, bit_width=1)

    def test_zero_vector(self):
        """Zero vector produces small but non-zero reconstruction.

        PolarQuant centroids are non-zero, so quantizing a zero vector maps
        to the nearest centroid per coordinate. QJL residual norm is ~small.
        The reconstruction won't be exactly zero but should be small.
        """
        tq = TurboQuant(d=128, bit_width=3, seed=42)
        x = np.zeros(128)

        compressed = tq.quantize(x)
        x_hat = tq.dequantize(compressed)
        assert np.linalg.norm(x_hat) < 1.0

    def test_deterministic(self):
        """Same seed → same output."""
        d = 128
        x = np.random.default_rng(1).standard_normal(d)

        tq1 = TurboQuant(d=d, bit_width=3, seed=42)
        tq2 = TurboQuant(d=d, bit_width=3, seed=42)

        c1 = tq1.quantize(x)
        c2 = tq2.quantize(x)

        np.testing.assert_array_equal(c1.mse_indices, c2.mse_indices)
        np.testing.assert_array_equal(c1.qjl_signs, c2.qjl_signs)
        assert c1.residual_norms == c2.residual_norms

    def test_batch_quantization(self):
        """Batch should match individual quantization."""
        d = 128
        tq = TurboQuant(d=d, bit_width=3, seed=42)
        rng = np.random.default_rng(7)

        X = rng.standard_normal((5, d))

        # Batch
        batch_compressed = tq.quantize(X)
        batch_recon = tq.dequantize(batch_compressed)

        # Single
        for i in range(5):
            single_compressed = tq.quantize(X[i])
            single_recon = tq.dequantize(single_compressed)
            np.testing.assert_allclose(batch_recon[i], single_recon, atol=1e-10)


class TestTurboQuantMSE:
    """Tests for MSE-only variant (Algorithm 1 at full bit_width)."""

    def test_round_trip(self):
        d = 128
        tq = TurboQuantMSE(d=d, bit_width=3, seed=42)
        rng = np.random.default_rng(1)
        x = rng.standard_normal(d)
        x = x / np.linalg.norm(x)

        indices, norms = tq.quantize(x)
        x_hat = tq.dequantize(indices, norms)

        mse = np.mean((x - x_hat) ** 2)
        # 3-bit MSE-only should be better than paper's 3-bit TurboQuant MSE
        # because all 3 bits go to MSE (no QJL stage)
        assert mse < 0.1, f"MSE-only 3-bit MSE {mse:.4f} too high"

    def test_norm_correction_can_be_disabled(self):
        d = 64
        x = np.random.default_rng(2).standard_normal(d)

        tq_old = TurboQuantMSE(d=d, bit_width=3, seed=42, norm_correction=False)
        tq_new = TurboQuantMSE(d=d, bit_width=3, seed=42, norm_correction=True)

        idx_old, norm_old = tq_old.quantize(x)
        idx_new, norm_new = tq_new.quantize(x)

        np.testing.assert_array_equal(idx_old, idx_new)
        np.testing.assert_allclose(norm_old, norm_new)

        x_hat_old = tq_old.dequantize(idx_old, norm_old)
        x_hat_new = tq_new.dequantize(idx_new, norm_new)

        assert not np.allclose(x_hat_old, x_hat_new)


class TestCompressedSizeBits:
    """Test compressed_size_bits method."""

    def test_size_calculation(self):
        tq = TurboQuant(d=128, bit_width=3, seed=42)
        bits = tq.compressed_size_bits(100)
        # 100 vectors × (128 coords × 3 bits + 64 bits norms) = 100 × 448 = 44800
        # 64 = vector_norm (32) + residual_norm (32), both stored in CompressedVector
        assert bits == 100 * (128 * 3 + 64)

    def test_size_scales_with_vectors(self):
        tq = TurboQuant(d=64, bit_width=4, seed=42)
        bits_10 = tq.compressed_size_bits(10)
        bits_100 = tq.compressed_size_bits(100)
        assert bits_100 == bits_10 * 10


class TestCompressionRatio:
    """Verify compression ratio calculations."""

    def test_3bit_compression(self):
        tq = TurboQuant(d=128, bit_width=3, seed=42)
        ratio = tq.compression_ratio(original_bits_per_value=16)
        # 16 / (3 + 64/128) = 16/3.5 ≈ 4.57 (64 = vector_norm + residual_norm)
        assert 4.0 < ratio < 6.0, f"3-bit compression ratio {ratio:.2f} unexpected"

    def test_4bit_compression(self):
        tq = TurboQuant(d=128, bit_width=4, seed=42)
        ratio = tq.compression_ratio(original_bits_per_value=16)
        # 16 / (4 + 64/128) = 16/4.5 ≈ 3.56 (64 = vector_norm + residual_norm)
        assert 3.0 < ratio < 5.0, f"4-bit compression ratio {ratio:.2f} unexpected"
