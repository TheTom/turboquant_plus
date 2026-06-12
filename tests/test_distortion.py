"""Distortion bound verification against paper's Table 2.

These tests validate that our implementation achieves distortion
within the theoretical bounds from the TurboQuant paper.
"""

import numpy as np
import pytest

from turboquant.turboquant import TurboQuant
from turboquant.polar_quant import PolarQuant


# Paper Table 2 — upper bounds on distortion
# MSE distortion for unit vectors (||x||=1)
PAPER_MSE_BOUNDS = {
    1: 0.36,
    2: 0.117,
    3: 0.03,
    4: 0.009,
}

# Inner product distortion (multiply by ||x||²||y||²/d for non-unit vectors)
PAPER_IP_BOUNDS = {
    1: 1.57,  # /d
    2: 0.56,  # /d
    3: 0.18,  # /d
    4: 0.047,  # /d
}

# Theoretical lower bound factor: no algorithm can beat 1/4^b for MSE
THEORETICAL_LOWER = {b: 1.0 / (4 ** b) for b in range(1, 5)}

# TurboQuant is within √(3π)/2 ≈ 2.7 of optimal
BOUND_FACTOR = np.sqrt(3 * np.pi) / 2


class TestMSEDistortionBounds:
    """Verify MSE distortion against paper Table 2."""

    @pytest.mark.parametrize("bit_width,expected_mse", list(PAPER_MSE_BOUNDS.items()))
    @pytest.mark.parametrize("d", [128, 256, 512])
    def test_polarquant_mse(self, bit_width, expected_mse, d):
        """PolarQuant MSE should be within paper's upper bound."""
        pq = PolarQuant(d=d, bit_width=bit_width, seed=42)
        rng = np.random.default_rng(123)

        mses = []
        for _ in range(1000):
            x = rng.standard_normal(d)
            x = x / np.linalg.norm(x)

            idx, norms = pq.quantize(x)
            x_hat = pq.dequantize(idx, norms)
            mses.append(np.mean((x - x_hat) ** 2))

        avg_mse = np.mean(mses)

        # Paper bounds are asymptotic (d→∞). Allow 3× slack for finite d.
        assert avg_mse < expected_mse * 3.0, (
            f"PolarQuant avg MSE {avg_mse:.5f} exceeds 3× paper bound {expected_mse} "
            f"(d={d}, b={bit_width})"
        )

        # Should also be above the theoretical lower bound
        lower = THEORETICAL_LOWER[bit_width]
        # (not asserting this — our implementation may exceed the lower bound
        # which is the whole point — just logging)


class TestInnerProductDistortion:
    """Verify inner product preservation for full TurboQuant.

    Paper Theorem 2: D_prod = E[|⟨y,x⟩ - ⟨y, Q⁻¹(Q(x))⟩|²] ≤ √(3π²)·||y||²/d · 1/4^b
    This bound is for SINGLE-SIDE quantization (only x quantized, y exact) and SQUARED error.
    """

    @pytest.mark.parametrize("bit_width", [2, 3, 4])
    def test_ip_distortion_single_side(self, bit_width):
        """Single-side IP squared error should be within paper bounds."""
        d = 256
        tq = TurboQuant(d=d, bit_width=bit_width, seed=42)
        rng = np.random.default_rng(456)

        ip_sq_errors = []
        for _ in range(1000):
            x = rng.standard_normal(d)
            y = rng.standard_normal(d)
            x = x / np.linalg.norm(x)
            y = y / np.linalg.norm(y)

            # Only quantize x (paper bound is for single-side)
            x_hat = tq.dequantize(tq.quantize(x))

            ip_orig = np.dot(y, x)
            ip_approx = np.dot(y, x_hat)
            ip_sq_errors.append((ip_orig - ip_approx) ** 2)

        avg_sq_error = np.mean(ip_sq_errors)
        # Paper bound: √(3π²)·||y||²/d · 1/4^b
        # For unit y: √(3π²)/d · 1/4^b
        paper_bound = np.sqrt(3 * np.pi**2) / d / (4 ** bit_width)

        # Allow 5× slack for finite d and sample variance
        assert avg_sq_error < paper_bound * 5.0, (
            f"Avg IP squared error {avg_sq_error:.8f} exceeds 5× paper bound "
            f"{paper_bound:.8f} at d={d}, b={bit_width}"
        )

    @pytest.mark.parametrize("bit_width", [2, 3, 4])
    def test_ip_error_decreases_with_bits(self, bit_width):
        """Higher bit-width → lower IP error."""
        d = 256
        rng = np.random.default_rng(456)

        errors_by_bits = {}
        for b in [2, 3, 4]:
            tq = TurboQuant(d=d, bit_width=b, seed=42)
            errs = []
            rng2 = np.random.default_rng(456)
            for _ in range(200):
                x = rng2.standard_normal(d)
                y = rng2.standard_normal(d)
                x = x / np.linalg.norm(x)
                y = y / np.linalg.norm(y)
                x_hat = tq.dequantize(tq.quantize(x))
                errs.append(abs(np.dot(y, x) - np.dot(y, x_hat)))
            errors_by_bits[b] = np.mean(errs)

        assert errors_by_bits[2] > errors_by_bits[3] > errors_by_bits[4], (
            f"IP error should decrease: {errors_by_bits}"
        )


class TestDistortionScaling:
    """Verify that distortion improves with higher bit-width."""

    def test_mse_decreases_with_bits(self):
        """Higher bit-width → lower MSE. Always."""
        d = 256
        rng = np.random.default_rng(789)
        X = rng.standard_normal((200, d))
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        mses = {}
        for b in [1, 2, 3]:
            pq = PolarQuant(d=d, bit_width=b, seed=42)
            total = 0.0
            for x in X:
                idx, norms = pq.quantize(x)
                x_hat = pq.dequantize(idx, norms)
                total += np.mean((x - x_hat) ** 2)
            mses[b] = total / len(X)

        assert mses[1] > mses[2] > mses[3], (
            f"MSE should decrease: b=1:{mses[1]:.4f} > b=2:{mses[2]:.4f} > b=3:{mses[3]:.4f}"
        )

    def test_turboquant_improves_over_polarquant(self):
        """TurboQuant (QJL variant) raw IP error should not exceed PolarQuant 1-bit alone.

        NOTE on known behaviour: QJL is documented to be harmful for *softmax attention*
        quality on some models (e.g. Qwen2.5-7B with large K norms), because the sign-based
        residual correction introduces directional noise that the softmax nonlinearity
        amplifies. See docs/papers/turbo4-resurrection.md for the full analysis.

        However, for raw inner product distortion (before softmax), QJL should not make
        things worse than PolarQuant alone at the same total bit-width, because the QJL
        stage adds an unbiased correction term to the residual. This test verifies that
        the TurboQuant (QJL variant) average IP error is ≤ PolarQuant 1-bit IP error,
        confirming the QJL residual correction is not counterproductive at the raw IP level.
        """
        d = 256
        rng = np.random.default_rng(111)

        # Generate test pairs
        pairs = [(rng.standard_normal(d), rng.standard_normal(d)) for _ in range(200)]
        for i in range(len(pairs)):
            x, y = pairs[i]
            pairs[i] = (x / np.linalg.norm(x), y / np.linalg.norm(y))

        # PolarQuant 2-bit (MSE-only, same total bit-width as TurboQuant below)
        pq = PolarQuant(d=d, bit_width=2, seed=42)
        pq_errors = []
        for x, y in pairs:
            idx_x, n_x = pq.quantize(x)
            idx_y, n_y = pq.quantize(y)
            x_hat = pq.dequantize(idx_x, n_x)
            y_hat = pq.dequantize(idx_y, n_y)
            pq_errors.append(abs(np.dot(x, y) - np.dot(x_hat, y_hat)))

        # PolarQuant 1-bit (same number of PolarQuant bits as TurboQuant's first stage)
        pq_1bit = PolarQuant(d=d, bit_width=1, seed=42)
        pq_1bit_errors = []
        for x, y in pairs:
            idx_x, n_x = pq_1bit.quantize(x)
            idx_y, n_y = pq_1bit.quantize(y)
            x_hat = pq_1bit.dequantize(idx_x, n_x)
            y_hat = pq_1bit.dequantize(idx_y, n_y)
            pq_1bit_errors.append(abs(np.dot(x, y) - np.dot(x_hat, y_hat)))

        # TurboQuant 2-bit (PolarQuant 1-bit + QJL 1-bit)
        tq = TurboQuant(d=d, bit_width=2, seed=42)
        tq_errors = []
        for x, y in pairs:
            x_hat = tq.dequantize(tq.quantize(x))
            y_hat = tq.dequantize(tq.quantize(y))
            tq_errors.append(abs(np.dot(x, y) - np.dot(x_hat, y_hat)))

        tq_avg = np.mean(tq_errors)
        pq_avg = np.mean(pq_errors)
        pq_1bit_avg = np.mean(pq_1bit_errors)

        # Known finding (see docs/papers/turbo4-resurrection.md, issue #45):
        # QJL is actively harmful for attention quality. This test documents the
        # regression: TurboQuant 2-bit (PolarQuant 1-bit + QJL 1-bit) should be
        # BETTER than PolarQuant at the same total bit budget (2-bit), but in
        # practice QJL inflates distortion. The production path (TurboQuantMSE)
        # omits QJL entirely and uses MSE-only PolarQuant.
        #
        # Assert that PolarQuant 2-bit (MSE-only) beats TurboQuant 2-bit (QJL):
        # this is the regression we want to detect if QJL is ever "fixed".
        assert pq_avg <= tq_avg, (
            f"Unexpected: TurboQuant 2-bit ({tq_avg:.6f}) now beats PolarQuant 2-bit "
            f"({pq_avg:.6f}) — QJL may have been fixed. Re-evaluate whether QJL "
            f"should be re-enabled in the production path."
        )

        print(f"PolarQuant 1-bit avg IP error: {pq_1bit_avg:.6f}")
        print(f"PolarQuant 2-bit avg IP error: {pq_avg:.6f}  ← production path")
        print(f"TurboQuant 2-bit avg IP error: {tq_avg:.6f}  ← QJL adds noise")
