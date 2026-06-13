"""Tests for random rotation matrix generation (Issue #2).

Tests are written BEFORE implementation per workflow rules.
All tests should FAIL until rotation.py is implemented.
"""

import numpy as np
import pytest


class TestDenseRotation:
    """Tests for Haar-distributed random rotation via QR decomposition."""

    def test_orthogonality(self):
        """Π @ Π.T should equal identity matrix."""
        from turboquant.rotation import random_rotation_dense

        d = 128
        rng = np.random.default_rng(42)
        Pi = random_rotation_dense(d, rng)

        identity = Pi @ Pi.T
        np.testing.assert_allclose(identity, np.eye(d), atol=1e-10)

    def test_transpose_is_inverse(self):
        """Π.T @ Π should also equal identity (orthogonal both ways)."""
        from turboquant.rotation import random_rotation_dense

        d = 64
        rng = np.random.default_rng(42)
        Pi = random_rotation_dense(d, rng)

        identity = Pi.T @ Pi
        np.testing.assert_allclose(identity, np.eye(d), atol=1e-10)

    def test_determinant_positive_one(self):
        """det(Π) should be +1 (proper rotation, not reflection).

        Uses slogdet for numerical stability — np.linalg.det overflows for large d.
        """
        from turboquant.rotation import random_rotation_dense

        for seed in [42, 99, 7, 123, 0]:
            d = 64
            rng = np.random.default_rng(seed)
            Pi = random_rotation_dense(d, rng)
            sign, logdet = np.linalg.slogdet(Pi)
            assert sign > 0, f"det sign = {sign} for seed={seed}, expected +1"
            np.testing.assert_allclose(logdet, 0.0, atol=1e-8,
                err_msg=f"log|det| = {logdet} for seed={seed}, expected 0.0")

    def test_deterministic_same_seed(self):
        """Same seed should produce identical rotation matrix."""
        from turboquant.rotation import random_rotation_dense

        d = 128
        Pi1 = random_rotation_dense(d, np.random.default_rng(42))
        Pi2 = random_rotation_dense(d, np.random.default_rng(42))
        np.testing.assert_array_equal(Pi1, Pi2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different matrices."""
        from turboquant.rotation import random_rotation_dense

        d = 64
        Pi1 = random_rotation_dense(d, np.random.default_rng(42))
        Pi2 = random_rotation_dense(d, np.random.default_rng(99))
        assert not np.allclose(Pi1, Pi2)

    def test_preserves_norms(self):
        """||Π @ x|| should equal ||x|| for any x (isometry)."""
        from turboquant.rotation import random_rotation_dense

        d = 128
        rng_rot = np.random.default_rng(42)
        Pi = random_rotation_dense(d, rng_rot)

        rng_vec = np.random.default_rng(99)
        for _ in range(100):
            x = rng_vec.standard_normal(d)
            y = Pi @ x
            np.testing.assert_allclose(
                np.linalg.norm(y), np.linalg.norm(x), rtol=1e-10
            )

    def test_preserves_inner_products(self):
        """⟨Π@x, Π@y⟩ should equal ⟨x, y⟩ for any x, y."""
        from turboquant.rotation import random_rotation_dense

        d = 128
        rng_rot = np.random.default_rng(42)
        Pi = random_rotation_dense(d, rng_rot)

        rng_vec = np.random.default_rng(77)
        for _ in range(100):
            x = rng_vec.standard_normal(d)
            y = rng_vec.standard_normal(d)
            ip_original = np.dot(x, y)
            ip_rotated = np.dot(Pi @ x, Pi @ y)
            np.testing.assert_allclose(ip_rotated, ip_original, rtol=1e-10)

    @pytest.mark.parametrize("d", [32, 64, 128, 256])
    def test_post_rotation_distribution(self, d):
        """After rotating a FIXED vector by many random Π, each coordinate
        should have mean ≈ 0 and variance ≈ ||x||²/d.

        Codex review: fixing Π and varying x tests input distribution, not
        rotation quality. Fix by sampling many Π with fixed x.
        """
        from turboquant.rotation import random_rotation_dense

        # Fixed unit vector (first basis vector)
        x = np.zeros(d)
        x[0] = 1.0

        n_samples = 2000
        rotated = np.zeros((n_samples, d))
        for i in range(n_samples):
            rng = np.random.default_rng(i)  # different rotation each time
            Pi = random_rotation_dense(d, rng)
            rotated[i] = Pi @ x

        # Each coordinate: mean ≈ 0 (tighter bound per Codex review)
        coord_means = rotated.mean(axis=0)
        # With 2000 samples and var ≈ 1/d, SE ≈ 1/(√(d*n)) ≈ small
        # Use 4σ bound: 4 * sqrt(1/d / 2000)
        mean_bound = 4 * np.sqrt(1.0 / d / n_samples)
        assert np.all(np.abs(coord_means) < max(mean_bound, 0.05)), (
            f"Max coordinate mean: {np.max(np.abs(coord_means)):.4f}, "
            f"expected < {max(mean_bound, 0.05):.4f}"
        )

        # Each coordinate: variance ≈ 1/d
        coord_vars = rotated.var(axis=0)
        expected_var = 1.0 / d
        # Tighter bounds: [0.5, 1.8] × expected (still statistical)
        assert np.all(coord_vars < expected_var * 1.8), (
            f"Max coordinate variance: {np.max(coord_vars):.6f}, "
            f"expected < {expected_var * 1.8:.6f}"
        )
        assert np.all(coord_vars > expected_var * 0.5), (
            f"Min coordinate variance: {np.min(coord_vars):.6f}, "
            f"expected > {expected_var * 0.5:.6f}"
        )

    def test_small_dimension(self):
        """Should work for small d like 1, 2, and 4."""
        from turboquant.rotation import random_rotation_dense

        for d in [1, 2, 4, 8]:
            rng = np.random.default_rng(42)
            Pi = random_rotation_dense(d, rng)
            assert Pi.shape == (d, d)
            np.testing.assert_allclose(Pi @ Pi.T, np.eye(d), atol=1e-10)

    def test_invalid_dimension_raises(self):
        """d < 1 should raise ValueError."""
        from turboquant.rotation import random_rotation_dense

        with pytest.raises(ValueError, match="d must be >= 1"):
            random_rotation_dense(0, np.random.default_rng(42))
        with pytest.raises(ValueError, match="d must be >= 1"):
            random_rotation_dense(-5, np.random.default_rng(42))

    def test_shape(self):
        """Output should be (d, d)."""
        from turboquant.rotation import random_rotation_dense

        d = 256
        Pi = random_rotation_dense(d, np.random.default_rng(42))
        assert Pi.shape == (d, d)
        assert Pi.dtype == np.float64


class TestFastWalshHadamard:
    """Tests for the fast Walsh-Hadamard transform."""

    def test_hadamard_non_pow2_raises(self):
        """Non-power-of-2 should raise ValueError."""
        from turboquant.rotation import hadamard_matrix

        with pytest.raises(ValueError, match="positive power of 2"):
            hadamard_matrix(3)
        with pytest.raises(ValueError, match="positive power of 2"):
            hadamard_matrix(0)
        with pytest.raises(ValueError, match="positive power of 2"):
            hadamard_matrix(-1)

    def test_fwht_non_pow2_raises(self):
        """Non-power-of-2 input should raise ValueError."""
        from turboquant.rotation import fast_walsh_hadamard_transform

        with pytest.raises(ValueError, match="positive power of 2"):
            fast_walsh_hadamard_transform(np.array([1.0, 2.0, 3.0]))

    def test_hadamard_matrix_matches_scipy(self):
        """Our hadamard_matrix should match scipy's."""
        from turboquant.rotation import hadamard_matrix
        from scipy.linalg import hadamard as scipy_hadamard

        for n in [2, 4, 8, 16]:
            ours = hadamard_matrix(n)
            theirs = scipy_hadamard(n).astype(np.float64)
            np.testing.assert_array_equal(ours, theirs)

    def test_matches_scipy_hadamard(self):
        """FWHT should match scipy's Hadamard matrix multiply (normalized).

        Codex review: use independent reference (scipy) not our own hadamard_matrix
        to avoid shared-source risk.
        """
        from turboquant.rotation import fast_walsh_hadamard_transform
        from scipy.linalg import hadamard as scipy_hadamard

        n = 16
        H = scipy_hadamard(n).astype(np.float64) / np.sqrt(n)  # normalized
        x = np.random.default_rng(42).standard_normal(n)

        fwht_result = fast_walsh_hadamard_transform(x)
        dense_result = H @ x

        np.testing.assert_allclose(fwht_result, dense_result, atol=1e-10)

    def test_involutory(self):
        """Applying FWHT twice should return the original vector."""
        from turboquant.rotation import fast_walsh_hadamard_transform

        n = 32
        x = np.random.default_rng(42).standard_normal(n)
        y = fast_walsh_hadamard_transform(x)
        x_back = fast_walsh_hadamard_transform(y)
        np.testing.assert_allclose(x_back, x, atol=1e-10)

    def test_preserves_norm(self):
        """Normalized Hadamard preserves vector norms."""
        from turboquant.rotation import fast_walsh_hadamard_transform

        n = 64
        x = np.random.default_rng(42).standard_normal(n)
        y = fast_walsh_hadamard_transform(x)
        np.testing.assert_allclose(np.linalg.norm(y), np.linalg.norm(x), rtol=1e-10)

    @pytest.mark.parametrize("n", [2, 4, 8, 16, 32, 64, 128])
    def test_various_sizes(self, n):
        """Should work for all powers of 2."""
        from turboquant.rotation import fast_walsh_hadamard_transform

        x = np.random.default_rng(42).standard_normal(n)
        y = fast_walsh_hadamard_transform(x)
        assert y.shape == (n,)
        # Norm preserved
        np.testing.assert_allclose(np.linalg.norm(y), np.linalg.norm(x), rtol=1e-10)


class TestFastRotation:
    """Tests for structured random rotation (Hadamard + random signs)."""

    def test_preserves_norms_pow2(self):
        """Fast rotation should preserve vector norms for power-of-2 dimensions."""
        from turboquant.rotation import random_rotation_fast, apply_fast_rotation

        d = 64  # power of 2 — no padding needed
        rng = np.random.default_rng(42)
        signs1, signs2, padded_d = random_rotation_fast(d, rng)

        rng_vec = np.random.default_rng(99)
        for _ in range(50):
            x = rng_vec.standard_normal(d)
            y = apply_fast_rotation(x, signs1, signs2, padded_d)
            np.testing.assert_allclose(
                np.linalg.norm(y), np.linalg.norm(x), rtol=1e-8
            )

    def test_non_pow2_returns_correct_length(self):
        """For non-power-of-2 d, output should still have length d.

        Codex review: pad+crop doesn't exactly preserve norms for non-pow2,
        so we only check shape, not exact norm preservation.
        """
        from turboquant.rotation import random_rotation_fast, apply_fast_rotation

        d = 100
        rng = np.random.default_rng(42)
        signs1, signs2, padded_d = random_rotation_fast(d, rng)
        assert padded_d == 128  # next power of 2

        x = np.random.default_rng(99).standard_normal(d)
        y = apply_fast_rotation(x, signs1, signs2, padded_d)
        assert y.shape == (d,)

    def test_transpose_inverts(self):
        """apply_fast_rotation_transpose(apply_fast_rotation(x)) ≈ x."""
        from turboquant.rotation import (
            random_rotation_fast, apply_fast_rotation, apply_fast_rotation_transpose
        )

        d = 64
        rng = np.random.default_rng(42)
        signs1, signs2, padded_d = random_rotation_fast(d, rng)

        x = np.random.default_rng(99).standard_normal(d)
        y = apply_fast_rotation(x, signs1, signs2, padded_d)
        x_back = apply_fast_rotation_transpose(y, signs1, signs2, padded_d)
        np.testing.assert_allclose(x_back, x, atol=1e-10)

    def test_deterministic(self):
        """Same seed → same rotation."""
        from turboquant.rotation import random_rotation_fast, apply_fast_rotation

        d = 64
        x = np.random.default_rng(1).standard_normal(d)

        s1a, s2a, pa = random_rotation_fast(d, np.random.default_rng(42))
        s1b, s2b, pb = random_rotation_fast(d, np.random.default_rng(42))

        ya = apply_fast_rotation(x, s1a, s2a, pa)
        yb = apply_fast_rotation(x, s1b, s2b, pb)
        np.testing.assert_array_equal(ya, yb)

    def test_batch_matches_single(self):
        """Batch rotation should match single-vector rotation."""
        from turboquant.rotation import (
            random_rotation_fast, apply_fast_rotation, apply_fast_rotation_batch
        )

        d = 64
        rng = np.random.default_rng(42)
        signs1, signs2, padded_d = random_rotation_fast(d, rng)

        rng_vec = np.random.default_rng(99)
        X = rng_vec.standard_normal((10, d))

        batch_result = apply_fast_rotation_batch(X, signs1, signs2, padded_d)

        for i in range(10):
            single_result = apply_fast_rotation(X[i], signs1, signs2, padded_d)
            np.testing.assert_allclose(batch_result[i], single_result, atol=1e-10)


class TestFastRotationExtended:
    """Additional tests for fast rotation: round-trip, batch, and Gaussianization."""

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_fast_rotation_round_trip(self, d):
        """apply_fast_rotation_transpose(apply_fast_rotation(x)) ≈ x for power-of-2 sizes."""
        from turboquant.rotation import (
            random_rotation_fast, apply_fast_rotation, apply_fast_rotation_transpose
        )

        rng = np.random.default_rng(42)
        signs1, signs2, padded_d = random_rotation_fast(d, rng)

        rng_vec = np.random.default_rng(7)
        for _ in range(20):
            x = rng_vec.standard_normal(d)
            y = apply_fast_rotation(x, signs1, signs2, padded_d)
            x_back = apply_fast_rotation_transpose(y, signs1, signs2, padded_d)
            np.testing.assert_allclose(
                x_back, x, atol=1e-10,
                err_msg=f"Round-trip failed for d={d}"
            )

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_fast_rotation_batch_matches_single(self, d):
        """apply_fast_rotation_batch applied to a batch matches apply_fast_rotation element-wise."""
        from turboquant.rotation import (
            random_rotation_fast, apply_fast_rotation, apply_fast_rotation_batch
        )

        rng = np.random.default_rng(99)
        signs1, signs2, padded_d = random_rotation_fast(d, rng)

        rng_vec = np.random.default_rng(11)
        X = rng_vec.standard_normal((8, d))

        batch_result = apply_fast_rotation_batch(X, signs1, signs2, padded_d)
        assert batch_result.shape == (8, d)

        for i in range(8):
            single_result = apply_fast_rotation(X[i], signs1, signs2, padded_d)
            np.testing.assert_allclose(
                batch_result[i], single_result, atol=1e-10,
                err_msg=f"Batch vs single mismatch at index {i}, d={d}"
            )

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_fast_rotation_distributes_energy(self, d):
        """Post-rotation coordinates should be approximately zero-mean with variance ≈ 1/d.

        This verifies the Gaussianize property: the structured rotation (D@H@D) spreads
        energy uniformly across dimensions. We apply the same rotation to many random
        vectors and check that each output coordinate has mean ≈ 0 and variance ≈ 1/d.
        """
        from turboquant.rotation import random_rotation_fast, apply_fast_rotation

        rng = np.random.default_rng(42)
        signs1, signs2, padded_d = random_rotation_fast(d, rng)

        n_samples = 2000
        rng_vec = np.random.default_rng(55)
        # Use unit vectors so norms don't dominate
        X = rng_vec.standard_normal((n_samples, d))
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        rotated = np.stack([
            apply_fast_rotation(X[i], signs1, signs2, padded_d)
            for i in range(n_samples)
        ])

        # Each coordinate should be approximately zero-mean
        coord_means = rotated.mean(axis=0)
        mean_bound = 4 * np.sqrt(1.0 / d / n_samples)
        assert np.all(np.abs(coord_means) < max(mean_bound, 0.05)), (
            f"Max coordinate mean {np.max(np.abs(coord_means)):.4f} exceeds bound "
            f"{max(mean_bound, 0.05):.4f} (d={d})"
        )

        # Each coordinate should have variance ≈ 1/d (energy spread uniformly)
        coord_vars = rotated.var(axis=0)
        expected_var = 1.0 / d
        assert np.all(coord_vars < expected_var * 2.0), (
            f"Max coordinate variance {np.max(coord_vars):.6f} exceeds 2× expected "
            f"{expected_var:.6f} (d={d})"
        )
        assert np.all(coord_vars > expected_var * 0.3), (
            f"Min coordinate variance {np.min(coord_vars):.6f} is below 0.3× expected "
            f"{expected_var:.6f} (d={d})"
        )
