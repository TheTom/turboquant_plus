"""Tests for QJL (Quantized Johnson-Lindenstrauss)."""

import numpy as np
import pytest

from turboquant.qjl import QJL, QJL_CONST


class TestQJLProjection:
    """Structural invariants of the projection matrix S."""

    @pytest.mark.parametrize("d", [64, 128, 256, 512])
    def test_projection_matrix_is_orthogonal(self, d):
        """S must be orthogonal: ||S Sᵀ − I||_F < 1e-12.

        Required for the classical unbiased estimator E[⟨x̂, y⟩] = ⟨x, y⟩
        and the closed-form E[||x̂||²] = (π/2)·||x||².
        """
        qjl = QJL(d=d, seed=42)
        err = np.linalg.norm(qjl.S @ qjl.S.T - np.eye(d), "fro")
        assert err < 1e-12, f"||S Sᵀ − I||_F = {err:.2e} exceeds 1e-12"


class TestQJLRoundTrip:
    """QJL quantize → dequantize should approximately preserve vectors."""

    @pytest.mark.parametrize("d", [64, 128, 256, 512])
    def test_dequantized_has_correct_scale(self, d):
        """||x_hat|| = sqrt(pi/2) * ||x|| for every x, not only on average.

        S is orthogonal and every sign is +1 or -1, so ||S^T signs|| = sqrt(d).
        """
        qjl = QJL(d=d, seed=42)
        rng = np.random.default_rng(99)

        for _ in range(200):
            x = rng.standard_normal(d)
            signs, norm = qjl.quantize(x)
            x_hat = qjl.dequantize(signs, norm)
            ratio = np.linalg.norm(x_hat) / np.linalg.norm(x)
            assert np.isclose(ratio, QJL_CONST, rtol=1e-9), (
                f"Norm ratio {ratio:.6f} != sqrt(pi/2) = {QJL_CONST:.6f}"
            )

    @pytest.mark.parametrize("d", [64, 128, 256, 512])
    def test_shrinkage_mse_matches_closed_form(self, d):
        """Relative MSE is pi/2 - 1 at shrinkage 1.0 and 1 - 2/pi at 2/pi.

        Pins the derivation in QJL.dequantize. At finite d both values sit
        slightly below the limit, by less than 0.01 at d=64.
        """
        qjl = QJL(d=d, seed=42)
        rng = np.random.default_rng(5)

        rel_mse = {1.0: [], 2 / np.pi: []}
        for _ in range(200):
            x = rng.standard_normal(d)
            signs, norm = qjl.quantize(x)
            for shrinkage, errors in rel_mse.items():
                x_hat = qjl.dequantize(signs, norm, shrinkage=shrinkage)
                errors.append(np.sum((x_hat - x) ** 2) / np.sum(x**2))

        assert np.mean(rel_mse[1.0]) == pytest.approx(np.pi / 2 - 1, abs=0.02)
        assert np.mean(rel_mse[2 / np.pi]) == pytest.approx(1 - 2 / np.pi, abs=0.02)

    def test_inner_product_unbiased_single_side(self):
        """QJL is unbiased: E[⟨y, Q⁻¹(Q(x))⟩] = ⟨y, x⟩ (paper Theorem 2).

        Unbiasedness holds when only ONE side is quantized (y is exact).
        When BOTH sides are quantized, the estimator is no longer unbiased.
        """
        d = 256
        qjl = QJL(d=d, seed=42)
        rng = np.random.default_rng(77)

        errors = []
        for _ in range(500):
            x = rng.standard_normal(d)
            y = rng.standard_normal(d)

            signs_x, norm_x = qjl.quantize(x)
            x_hat = qjl.dequantize(signs_x, norm_x)

            ip_original = np.dot(x, y)
            ip_approx = np.dot(x_hat, y)  # y is NOT quantized

            errors.append(ip_approx - ip_original)

        # Mean error should be near zero (unbiased estimator)
        mean_error = np.mean(errors)
        # Standard error of mean: σ/√n. With 500 samples, allow 3 SE
        std_error = np.std(errors) / np.sqrt(len(errors))
        assert abs(mean_error) < 3 * std_error + 0.1, (
            f"Mean IP error {mean_error:.4f} ± {std_error:.4f} — QJL should be unbiased (single-side)"
        )

    def test_signs_are_binary(self):
        """All sign values should be exactly +1 or -1."""
        d = 128
        qjl = QJL(d=d, seed=42)
        x = np.random.default_rng(1).standard_normal(d)

        signs, _ = qjl.quantize(x)
        unique_vals = set(signs.tolist())
        assert unique_vals.issubset({1, -1}), f"Unexpected sign values: {unique_vals}"

    def test_zero_vector(self):
        """Zero vector should produce zero norm and near-zero reconstruction."""
        d = 128
        qjl = QJL(d=d, seed=42)
        x = np.zeros(d)

        signs, norm = qjl.quantize(x)
        assert norm == 0.0
        x_hat = qjl.dequantize(signs, norm)
        np.testing.assert_allclose(x_hat, 0.0, atol=1e-15)

    def test_batch_matches_single(self):
        """Batch quantization should match single-vector results."""
        d = 128
        qjl = QJL(d=d, seed=42)
        rng = np.random.default_rng(7)

        X = rng.standard_normal((10, d))

        signs_batch, norms_batch = qjl.quantize(X)

        for i in range(10):
            signs_single, norm_single = qjl.quantize(X[i])
            np.testing.assert_array_equal(signs_batch[i], signs_single)
            np.testing.assert_allclose(norms_batch[i], norm_single, atol=1e-12)

    def test_deterministic(self):
        """Same seed, same input → same output."""
        d = 128
        x = np.random.default_rng(1).standard_normal(d)

        qjl1 = QJL(d=d, seed=42)
        qjl2 = QJL(d=d, seed=42)

        signs1, norm1 = qjl1.quantize(x)
        signs2, norm2 = qjl2.quantize(x)

        np.testing.assert_array_equal(signs1, signs2)
        assert norm1 == norm2
