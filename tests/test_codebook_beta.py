"""Tests for Beta distribution codebook enhancement."""

import numpy as np
import pytest

from turboquant.codebook import (
    compute_centroids,
    optimal_centroids,
    _lloyds_beta,
    _beta_conditional_expectation,
)
from scipy import stats


class TestBetaConditionalExpectation:
    """Test the E[X | a < X < b] helper for Beta distributions."""

    def test_full_range_equals_mean(self):
        """E[X | 0 < X < 1] should equal the distribution mean."""
        rv = stats.beta(3.0, 3.0)
        result = _beta_conditional_expectation(rv, 0.0, 1.0)
        np.testing.assert_allclose(result, 0.5, rtol=1e-6)

    def test_upper_half(self):
        """E[X | 0.5 < X < 1.0] for symmetric Beta should be > 0.5."""
        rv = stats.beta(5.0, 5.0)
        result = _beta_conditional_expectation(rv, 0.5, 1.0)
        assert result > 0.5
        assert result < 1.0

    def test_lower_half(self):
        """E[X | 0 < X < 0.5] for symmetric Beta should be < 0.5."""
        rv = stats.beta(5.0, 5.0)
        result = _beta_conditional_expectation(rv, 0.0, 0.5)
        assert result < 0.5
        assert result > 0.0

    def test_symmetric_halves(self):
        """For symmetric Beta, E[X|X<0.5] + E[X|X>0.5] should equal 1.0."""
        rv = stats.beta(10.0, 10.0)
        low = _beta_conditional_expectation(rv, 0.0, 0.5)
        high = _beta_conditional_expectation(rv, 0.5, 1.0)
        np.testing.assert_allclose(low + high, 1.0, rtol=1e-6)

    def test_narrow_interval(self):
        """Narrow interval conditional mean should be near midpoint."""
        rv = stats.beta(5.0, 5.0)
        result = _beta_conditional_expectation(rv, 0.49, 0.51)
        np.testing.assert_allclose(result, 0.5, atol=0.02)

    def test_extreme_interval_fallback(self):
        """Extremely narrow interval far from mass should use fallback."""
        rv = stats.beta(50.0, 50.0)
        # Very far in the tail - probability underflows
        result = _beta_conditional_expectation(rv, 0.99, 1.0)
        assert np.isfinite(result)

    def test_asymmetric_beta(self):
        """Should work with asymmetric Beta as well."""
        rv = stats.beta(2.0, 5.0)
        result = _beta_conditional_expectation(rv, 0.0, 1.0)
        expected_mean = 2.0 / (2.0 + 5.0)
        np.testing.assert_allclose(result, expected_mean, rtol=1e-6)


class TestLloydsBeta:
    """Test Lloyd's algorithm with Beta distribution."""

    def test_correct_count(self):
        """Should produce 2^b centroids."""
        for b in [3, 4]:
            n = 1 << b
            centroids = _lloyds_beta(n, d=64)
            assert len(centroids) == n

    def test_centroids_sorted(self):
        """Centroids should be sorted ascending."""
        centroids = _lloyds_beta(8, d=64)
        assert np.all(np.diff(centroids) > 0)

    def test_centroids_centered(self):
        """Centroids should be roughly centered around 0."""
        centroids = _lloyds_beta(8, d=64)
        assert abs(np.mean(centroids)) < 0.01

    def test_centroids_symmetric(self):
        """For symmetric Beta(d/2,d/2), centroids should be symmetric around 0."""
        centroids = _lloyds_beta(8, d=128)
        np.testing.assert_allclose(centroids, -centroids[::-1], atol=1e-6)

    def test_centroids_within_range(self):
        """All centroids should be within [-1/sqrt(d), 1/sqrt(d)]."""
        d = 64
        centroids = _lloyds_beta(8, d=d)
        bound = 1.0 / np.sqrt(d)
        assert np.all(centroids >= -bound - 1e-10)
        assert np.all(centroids <= bound + 1e-10)

    def test_scale_with_dimension(self):
        """Centroid magnitude should decrease with increasing d."""
        c_small = _lloyds_beta(8, d=32)
        c_large = _lloyds_beta(8, d=128)
        assert np.max(np.abs(c_small)) > np.max(np.abs(c_large))

    def test_16_centroids(self):
        """4-bit beta codebook should produce 16 centroids."""
        centroids = _lloyds_beta(16, d=64)
        assert len(centroids) == 16
        assert np.all(np.diff(centroids) > 0)


class TestComputeCentroids:
    """Test the unified compute_centroids dispatcher."""

    def test_use_beta_false_matches_optimal(self):
        """use_beta=False should give same result as optimal_centroids."""
        for b in [1, 2, 3]:
            for d in [64, 128]:
                c1 = compute_centroids(b, d, use_beta=False)
                c2 = optimal_centroids(b, d)
                np.testing.assert_array_equal(c1, c2)

    def test_use_beta_true_small_d(self):
        """use_beta=True with small d should use Beta codebook."""
        # For b >= 3 and d < 256, should use Beta
        c_beta = compute_centroids(3, d=64, use_beta=True)
        c_gauss = compute_centroids(3, d=64, use_beta=False)
        # They should be different (Beta vs Gaussian optimization)
        assert not np.allclose(c_beta, c_gauss, atol=1e-8)

    def test_use_beta_true_large_d_falls_back(self):
        """use_beta=True with large d should fall back to Gaussian."""
        c_beta = compute_centroids(3, d=256, use_beta=True)
        c_gauss = compute_centroids(3, d=256, use_beta=False)
        np.testing.assert_array_equal(c_beta, c_gauss)

    def test_use_beta_true_low_bits_falls_back(self):
        """use_beta=True with bit_width < 3 should fall back to Gaussian."""
        for b in [1, 2]:
            c_beta = compute_centroids(b, d=64, use_beta=True)
            c_gauss = compute_centroids(b, d=64, use_beta=False)
            np.testing.assert_array_equal(c_beta, c_gauss)

    def test_beta_centroids_still_sorted(self):
        """Beta centroids should still be sorted."""
        c = compute_centroids(3, d=64, use_beta=True)
        assert np.all(np.diff(c) > 0)

    def test_beta_centroids_correct_count(self):
        """Beta centroids should have 2^b entries."""
        c = compute_centroids(4, d=64, use_beta=True)
        assert len(c) == 16

    def test_beta_centroids_symmetric(self):
        """Beta centroids should be symmetric for symmetric Beta."""
        c = compute_centroids(3, d=64, use_beta=True)
        np.testing.assert_allclose(c, -c[::-1], atol=1e-6)

    def test_default_use_beta_false(self):
        """Default use_beta=False should match optimal_centroids."""
        c1 = compute_centroids(3, 128)
        c2 = optimal_centroids(3, 128)
        np.testing.assert_array_equal(c1, c2)
