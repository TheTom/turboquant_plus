"""Codebook construction for PolarQuant.

After random rotation, each coordinate follows Beta(d/2, d/2) on [-1/sqrt(d), 1/sqrt(d)],
which converges to N(0, 1/d) for large d. We use optimal scalar quantizers for this
distribution.

Paper provides closed-form centroids for 1-bit and 2-bit. For higher bit-widths,
we use Lloyd's algorithm on the Gaussian approximation, or (for small d) the true
Beta distribution.

Enhancement: ``compute_centroids`` dispatches to Beta-based Lloyd's for d < 256
when ``use_beta=True``, giving tighter codebooks for low-dimensional heads.
"""

import numpy as np
from scipy import stats


def optimal_centroids(bit_width: int, d: int) -> np.ndarray:
    """Compute optimal MSE centroids for the post-rotation coordinate distribution.

    Args:
        bit_width: Number of bits per coordinate (1, 2, 3, 4, ...).
        d: Vector dimension (affects centroid scale).

    Returns:
        Sorted array of 2^bit_width centroids.
    """
    n_centroids = 1 << bit_width

    if bit_width == 1:
        c = np.sqrt(2.0 / (np.pi * d))
        return np.array([-c, c])

    if bit_width == 2:
        return np.array([-1.51, -0.453, 0.453, 1.51]) / np.sqrt(d)

    # For b >= 3, use Lloyd's algorithm on N(0, 1/d)
    return _lloyds_gaussian(n_centroids, sigma=1.0 / np.sqrt(d))


def compute_centroids(bit_width: int, d: int, use_beta: bool = False) -> np.ndarray:
    """Compute optimal centroids, optionally using the true Beta distribution.

    For d < 256 and ``use_beta=True``, uses Lloyd's algorithm on the Beta(d/2, d/2)
    distribution (centered on [-0.5, 0.5] then scaled to [-1/sqrt(d), 1/sqrt(d)]).
    For d >= 256 or ``use_beta=False``, falls back to the Gaussian approximation
    via ``optimal_centroids``.

    Args:
        bit_width: Number of bits per coordinate.
        d: Vector dimension.
        use_beta: If True AND d < 256, use Beta distribution for codebook.

    Returns:
        Sorted array of 2^bit_width centroids.
    """
    if use_beta and d < 256 and bit_width >= 3:
        n_centroids = 1 << bit_width
        return _lloyds_beta(n_centroids, d)
    return optimal_centroids(bit_width, d)


def _lloyds_gaussian(n_centroids: int, sigma: float, n_iter: int = 100) -> np.ndarray:
    """Lloyd's algorithm (iterative k-means) for optimal scalar quantization of N(0, sigma²).

    Args:
        n_centroids: Number of quantization levels (2^b).
        sigma: Standard deviation of the Gaussian.
        n_iter: Number of Lloyd iterations.

    Returns:
        Sorted array of optimal centroids.
    """
    # Initialize boundary positions from uniform quantiles
    boundaries = stats.norm.ppf(
        np.linspace(0, 1, n_centroids + 1)[1:-1], scale=sigma
    )
    centroids = np.zeros(n_centroids)

    # Initial centroids: conditional expectations within each region
    centroids[0] = _gaussian_conditional_expectation(sigma, -np.inf, boundaries[0])
    for i in range(1, n_centroids - 1):
        centroids[i] = _gaussian_conditional_expectation(sigma, boundaries[i - 1], boundaries[i])
    centroids[-1] = _gaussian_conditional_expectation(sigma, boundaries[-1], np.inf)

    for _ in range(n_iter):
        # Update boundaries (midpoints between consecutive centroids)
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # Update centroids (conditional expectations within each region)
        centroids[0] = _gaussian_conditional_expectation(sigma, -np.inf, boundaries[0])
        for i in range(1, n_centroids - 1):
            centroids[i] = _gaussian_conditional_expectation(sigma, boundaries[i - 1], boundaries[i])
        centroids[-1] = _gaussian_conditional_expectation(sigma, boundaries[-1], np.inf)

    return np.sort(centroids)


def _lloyds_beta(n_centroids: int, d: int, n_iter: int = 100) -> np.ndarray:
    """Lloyd's algorithm for optimal scalar quantization of Beta(d/2, d/2).

    After random rotation, coordinates of a unit vector in R^d follow
    Beta(d/2, d/2) supported on [0, 1]. We center to [-0.5, 0.5] (mean 0)
    and then scale to [-1/sqrt(d), 1/sqrt(d)] to match the coordinate scale.

    For d >= 256 the Beta is nearly Gaussian and this gives essentially the same
    result as ``_lloyds_gaussian``; the benefit is for small d (32-128) where the
    Beta has heavier tails relative to its support.

    Args:
        n_centroids: Number of quantization levels (2^b).
        d: Vector dimension.
        n_iter: Number of Lloyd iterations.

    Returns:
        Sorted array of optimal centroids on the [-1/sqrt(d), 1/sqrt(d)] scale.
    """
    alpha = d / 2.0
    beta_param = d / 2.0
    rv = stats.beta(alpha, beta_param)

    # Work in the native [0, 1] space, then shift+scale at the end.
    # Initialize boundaries from uniform quantiles of Beta(d/2, d/2)
    boundaries = rv.ppf(np.linspace(0, 1, n_centroids + 1)[1:-1])
    centroids = np.zeros(n_centroids)

    # Initial centroids: conditional expectations within each region
    centroids[0] = _beta_conditional_expectation(rv, 0.0, boundaries[0])
    for i in range(1, n_centroids - 1):
        centroids[i] = _beta_conditional_expectation(rv, boundaries[i - 1], boundaries[i])
    centroids[-1] = _beta_conditional_expectation(rv, boundaries[-1], 1.0)

    for _ in range(n_iter):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        centroids[0] = _beta_conditional_expectation(rv, 0.0, boundaries[0])
        for i in range(1, n_centroids - 1):
            centroids[i] = _beta_conditional_expectation(rv, boundaries[i - 1], boundaries[i])
        centroids[-1] = _beta_conditional_expectation(rv, boundaries[-1], 1.0)

    centroids = np.sort(centroids)

    # Transform from [0, 1] to centered [-1/sqrt(d), 1/sqrt(d)]
    # Shift: subtract mean (0.5), so range becomes [-0.5, 0.5]
    # Scale: multiply by 2/sqrt(d), so range becomes [-1/sqrt(d), 1/sqrt(d)]
    centroids = (centroids - 0.5) * (2.0 / np.sqrt(d))
    return centroids


def _beta_conditional_expectation(
    rv: stats.rv_continuous, a: float, b: float,
) -> float:
    """E[X | a < X < b] where X ~ Beta(alpha, beta) on [0, 1].

    Uses numerical integration: E[X | a<X<b] = integral(x * f(x), a, b) / P(a<X<b).

    Args:
        rv: A frozen scipy.stats Beta distribution.
        a: Lower bound of interval (clipped to [0, 1]).
        b: Upper bound of interval (clipped to [0, 1]).

    Returns:
        Conditional expectation.
    """
    a = max(a, 0.0)
    b = min(b, 1.0)

    prob = rv.cdf(b) - rv.cdf(a)
    if prob < 1e-15:
        return (a + b) / 2.0

    # E[X | a<X<b] = (1/prob) * integral(x * pdf(x), a, b)
    # For Beta(alpha, beta): integral(x * pdf(x), a, b) can be computed via
    # the incomplete beta function. E[X*I(a<X<b)] = alpha/(alpha+beta) * (F_{a+1,b}(b) - F_{a+1,b}(a))
    # where F_{a+1,b} is the CDF of Beta(alpha+1, beta).
    alpha = rv.args[0]
    beta_param = rv.args[1]
    mean = alpha / (alpha + beta_param)

    rv_shifted = stats.beta(alpha + 1, beta_param)
    integral = mean * (rv_shifted.cdf(b) - rv_shifted.cdf(a))

    return integral / prob


def _gaussian_conditional_expectation(sigma: float, a: float, b: float) -> float:
    """E[X | a < X < b] where X ~ N(0, sigma²).

    Uses the formula: E[X | a < X < b] = sigma² * (φ(a/σ) - φ(b/σ)) / (Φ(b/σ) - Φ(a/σ))
    where φ is the PDF and Φ is the CDF of standard normal.
    """
    a_std = a / sigma if np.isfinite(a) else a
    b_std = b / sigma if np.isfinite(b) else b

    # Use sf() for upper tail to avoid CDF cancellation at extreme values
    # prob = P(a < X/σ < b) using the more numerically stable formulation
    if not np.isfinite(a_std):
        prob = stats.norm.cdf(b_std)
    elif not np.isfinite(b_std):
        prob = stats.norm.sf(a_std)
    else:
        prob = stats.norm.cdf(b_std) - stats.norm.cdf(a_std)

    if prob < 1e-15:
        # For semi-infinite intervals, use asymptotic approximation
        if np.isfinite(a) and not np.isfinite(b):
            return a + sigma  # E[X | X > a] ≈ a + σ for extreme a
        elif not np.isfinite(a) and np.isfinite(b):
            return b - sigma
        elif np.isfinite(a) and np.isfinite(b):
            return (a + b) / 2.0
        else:  # pragma: no cover — both infinite always has prob=1
            return 0.0

    pdf_diff = stats.norm.pdf(a_std) - stats.norm.pdf(b_std)
    return sigma * pdf_diff / prob


def nearest_centroid_indices(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Find nearest centroid index for each value. Vectorized.

    Args:
        values: Array of values to quantize, shape (...).
        centroids: Sorted centroid array, shape (n_centroids,).

    Returns:
        Integer indices into centroids array, same shape as values.
    """
    # Use searchsorted for sorted centroids — O(n log k) instead of O(n * k)
    # Find the insertion point, then check left and right neighbors
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return np.searchsorted(boundaries, values.ravel()).reshape(values.shape)
