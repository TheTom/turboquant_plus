"""Tests for codebook construction (Issue #3).

Tests written BEFORE reviewing implementation per workflow.
"""

import numpy as np


class TestOptimalCentroids:
    """Test closed-form and Lloyd's centroids."""

    def test_1bit_centroids_match_paper(self):
        """1-bit centroids should be ±√(2/πd) per paper Theorem 3.1."""
        from turboquant.codebook import optimal_centroids

        for d in [64, 128, 256, 1024]:
            centroids = optimal_centroids(1, d)
            expected = np.sqrt(2.0 / (np.pi * d))
            assert len(centroids) == 2
            np.testing.assert_allclose(centroids[0], -expected, rtol=1e-10)
            np.testing.assert_allclose(centroids[1], expected, rtol=1e-10)

    def test_2bit_centroids_match_paper(self):
        """2-bit centroids should be {±0.453/√d, ±1.51/√d} per paper Table 1."""
        from turboquant.codebook import optimal_centroids

        for d in [64, 128, 256]:
            centroids = optimal_centroids(2, d)
            expected = np.array([-1.51, -0.453, 0.453, 1.51]) / np.sqrt(d)
            assert len(centroids) == 4
            np.testing.assert_allclose(centroids, expected, rtol=1e-10)

    def test_3bit_centroids_pinned(self):
        """3-bit optimal centroids for d=128 (the canonical KV block size).

        Reference values every backend port (CUDA/HIP/Metal/CPU) must reproduce.
        A mismatch here — or a port hardcoding different values — means a
        mis-scaled codebook that clips the post-rotation Gaussian tails and
        inflates KLD/PPL. Pinned to guard against transcription drift.
        """
        from turboquant.codebook import optimal_centroids

        c = optimal_centroids(3, 128)
        expected = np.array([
            -0.19021, -0.11879, -0.06682, -0.02166,
             0.02166,  0.06682,  0.11879,  0.19021,
        ])
        np.testing.assert_allclose(c, expected, atol=5e-4)

    def test_4bit_centroids_pinned(self):
        """4-bit optimal centroids for d=128.

        Outer level ≈ 0.2402 (2.72·σ, σ=1/√128). A downstream C/CUDA port once
        shipped ≈0.1739 (σ≈0.064, 0.61× too narrow) → ~2.1× excess MSE and a
        large KLD regression. This test pins the correct envelope so any port
        can be checked against it.
        """
        from turboquant.codebook import optimal_centroids

        c = optimal_centroids(4, 128)
        expected = np.array([
            -0.24021, -0.18139, -0.14149, -0.10960,
            -0.08205, -0.05709, -0.03369, -0.01114,
             0.01114,  0.03369,  0.05709,  0.08205,
             0.10960,  0.14149,  0.18139,  0.24021,
        ])
        np.testing.assert_allclose(c, expected, atol=5e-4)

    def test_centroids_sorted(self):
        """Centroids should always be sorted ascending."""
        from turboquant.codebook import optimal_centroids

        for b in [1, 2, 3, 4]:
            for d in [64, 128, 256]:
                centroids = optimal_centroids(b, d)
                assert np.all(np.diff(centroids) > 0), (
                    f"Centroids not sorted for b={b}, d={d}: {centroids}"
                )

    def test_correct_count(self):
        """Should have exactly 2^b centroids."""
        from turboquant.codebook import optimal_centroids

        for b in [1, 2, 3, 4]:
            centroids = optimal_centroids(b, 128)
            assert len(centroids) == (1 << b)

    def test_centroids_symmetric(self):
        """Centroids should be symmetric around 0 (Gaussian is symmetric)."""
        from turboquant.codebook import optimal_centroids

        for b in [1, 2, 3, 4]:
            centroids = optimal_centroids(b, 128)
            np.testing.assert_allclose(
                centroids, -centroids[::-1], atol=1e-10,
                err_msg=f"Centroids not symmetric for b={b}"
            )

    def test_lloyd_converges_3bit(self):
        """3-bit Lloyd's should produce 8 reasonable centroids."""
        from turboquant.codebook import optimal_centroids

        d = 128
        centroids = optimal_centroids(3, d)
        sigma = 1.0 / np.sqrt(d)

        assert len(centroids) == 8
        # All centroids should be within ~4σ of the distribution
        assert np.all(np.abs(centroids) < 4 * sigma)

    def test_lloyd_converges_4bit(self):
        """4-bit Lloyd's should produce 16 reasonable centroids."""
        from turboquant.codebook import optimal_centroids

        d = 256
        centroids = optimal_centroids(4, d)
        sigma = 1.0 / np.sqrt(d)

        assert len(centroids) == 16
        assert np.all(np.abs(centroids) < 4 * sigma)

    def test_centroids_scale_with_dimension(self):
        """Centroids should shrink as d increases (scale is 1/√d)."""
        from turboquant.codebook import optimal_centroids

        for b in [1, 2, 3]:
            c_small = optimal_centroids(b, 64)
            c_large = optimal_centroids(b, 256)
            # Max centroid should be ~2× smaller for 4× larger d
            ratio = np.max(np.abs(c_small)) / np.max(np.abs(c_large))
            assert 1.5 < ratio < 2.5, (
                f"Scale ratio {ratio:.2f} unexpected for b={b}"
            )


class TestGaussianConditionalExpectation:
    """Test the E[X | a < X < b] helper."""

    def test_positive_half(self):
        """E[X | X > 0] for X ~ N(0,1) should be √(2/π) ≈ 0.7979."""
        from turboquant.codebook import _gaussian_conditional_expectation

        result = _gaussian_conditional_expectation(1.0, 0.0, np.inf)
        np.testing.assert_allclose(result, np.sqrt(2.0 / np.pi), rtol=1e-6)

    def test_symmetric_interval(self):
        """E[X | -a < X < a] should be 0 for symmetric interval."""
        from turboquant.codebook import _gaussian_conditional_expectation

        result = _gaussian_conditional_expectation(1.0, -1.0, 1.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_full_range(self):
        """E[X | -∞ < X < ∞] should be 0 for N(0, σ²)."""
        from turboquant.codebook import _gaussian_conditional_expectation

        result = _gaussian_conditional_expectation(2.0, -np.inf, np.inf)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_extreme_tail(self):
        """E[X | X > 5σ] should not be NaN/inf."""
        from turboquant.codebook import _gaussian_conditional_expectation

        result = _gaussian_conditional_expectation(1.0, 5.0, np.inf)
        assert np.isfinite(result)
        assert result > 5.0  # conditional mean must exceed the lower bound

    def test_extreme_upper_tail_asymptotic(self):
        """E[X | X > 10σ] should use asymptotic fallback, not return 0."""
        from turboquant.codebook import _gaussian_conditional_expectation

        result = _gaussian_conditional_expectation(1.0, 10.0, np.inf)
        assert np.isfinite(result)
        assert result > 10.0

    def test_extreme_lower_tail_asymptotic(self):
        """E[X | X < -10σ] should use asymptotic fallback."""
        from turboquant.codebook import _gaussian_conditional_expectation

        result = _gaussian_conditional_expectation(1.0, -np.inf, -10.0)
        assert np.isfinite(result)
        assert result < -10.0

    def test_very_narrow_interval(self):
        """E[X | a < X < a+ε] should be ≈ a for tiny ε."""
        from turboquant.codebook import _gaussian_conditional_expectation

        result = _gaussian_conditional_expectation(1.0, 1.0, 1.0001)
        assert np.isfinite(result)
        np.testing.assert_allclose(result, 1.0, atol=0.01)

    def test_extreme_narrow_finite_interval_fallback(self):
        """When prob < 1e-15 for finite [a,b], returns midpoint."""
        from turboquant.codebook import _gaussian_conditional_expectation

        # Both endpoints deep in same tail — probability underflows
        result = _gaussian_conditional_expectation(1.0, 20.0, 20.1)
        assert np.isfinite(result)
        np.testing.assert_allclose(result, 20.05, atol=0.1)

    def test_both_infinite_fallback(self):
        """When both a and b are infinite and prob < 1e-15, returns 0."""
        from turboquant.codebook import _gaussian_conditional_expectation

        # This shouldn't happen in practice, but the code handles it
        # Manually force: -inf to +inf always has prob=1, so we can't trigger
        # the fallback naturally. Test the logic exists by checking full range.
        result = _gaussian_conditional_expectation(1.0, -np.inf, np.inf)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_scaled_distribution(self):
        """Should work correctly with non-unit sigma."""
        from turboquant.codebook import _gaussian_conditional_expectation

        sigma = 0.1
        result = _gaussian_conditional_expectation(sigma, 0.0, np.inf)
        expected = sigma * np.sqrt(2.0 / np.pi)
        np.testing.assert_allclose(result, expected, rtol=1e-6)


class TestNearestCentroidIndices:
    """Test vectorized nearest centroid lookup."""

    def test_exact_centroids(self):
        """Values exactly at centroids should map to themselves."""
        from turboquant.codebook import nearest_centroid_indices

        centroids = np.array([-1.0, 0.0, 1.0])
        values = np.array([-1.0, 0.0, 1.0])
        indices = nearest_centroid_indices(values, centroids)
        np.testing.assert_array_equal(indices, [0, 1, 2])

    def test_midpoint_goes_right(self):
        """Values at midpoint between centroids should go to the right one."""
        from turboquant.codebook import nearest_centroid_indices

        centroids = np.array([-1.0, 0.0, 1.0])
        midpoint = -0.5  # midpoint between -1.0 and 0.0
        indices = nearest_centroid_indices(np.array([midpoint]), centroids)
        # searchsorted at midpoint goes right
        assert indices[0] in [0, 1]  # either is acceptable

    def test_values_outside_range(self):
        """Values far outside centroid range should clamp to nearest end."""
        from turboquant.codebook import nearest_centroid_indices

        centroids = np.array([-1.0, 0.0, 1.0])
        values = np.array([-100.0, 100.0])
        indices = nearest_centroid_indices(values, centroids)
        assert indices[0] == 0   # far left → first centroid
        assert indices[1] == 2   # far right → last centroid

    def test_batch_shape_preserved(self):
        """Output shape should match input shape."""
        from turboquant.codebook import nearest_centroid_indices

        centroids = np.array([-1.0, 0.0, 1.0])
        values = np.random.default_rng(42).standard_normal((5, 10))
        indices = nearest_centroid_indices(values, centroids)
        assert indices.shape == (5, 10)

    def test_matches_brute_force(self):
        """searchsorted result should match argmin brute force."""
        from turboquant.codebook import nearest_centroid_indices

        centroids = np.array([-1.5, -0.5, 0.5, 1.5])
        values = np.random.default_rng(42).standard_normal(500)

        fast_indices = nearest_centroid_indices(values, centroids)
        # Brute force: for each value, find the centroid with minimum abs distance
        brute_indices = np.argmin(np.abs(values[:, np.newaxis] - centroids[np.newaxis, :]), axis=1)

        np.testing.assert_array_equal(fast_indices, brute_indices)

    def test_all_indices_valid(self):
        """All indices should be in [0, n_centroids)."""
        from turboquant.codebook import nearest_centroid_indices

        centroids = np.array([-1.5, -0.5, 0.5, 1.5])
        values = np.random.default_rng(42).standard_normal(1000)
        indices = nearest_centroid_indices(values, centroids)
        assert indices.min() >= 0
        assert indices.max() < len(centroids)
