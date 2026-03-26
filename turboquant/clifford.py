"""Clifford algebra Cl(3,0) for RotorQuant.

Multivector basis: [1, e1, e2, e3, e12, e13, e23, e123]
All operations are NumPy-vectorized for batch processing.
"""

import numpy as np

MV_DIM = 8  # 2^3 components for Cl(3,0)


def geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full Cl(3,0) geometric product. a, b shape (..., 8) -> (..., 8)."""
    a0, a1, a2, a3, a12, a13, a23, a123 = [a[..., i] for i in range(8)]
    b0, b1, b2, b3, b12, b13, b23, b123 = [b[..., i] for i in range(8)]

    r = np.empty_like(a)
    r[..., 0] = a0*b0 + a1*b1 + a2*b2 + a3*b3 - a12*b12 - a13*b13 - a23*b23 - a123*b123
    r[..., 1] = a0*b1 + a1*b0 - a2*b12 + a12*b2 - a3*b13 + a13*b3 + a23*b123 + a123*b23
    r[..., 2] = a0*b2 + a2*b0 + a1*b12 - a12*b1 - a3*b23 + a23*b3 - a13*b123 - a123*b13
    r[..., 3] = a0*b3 + a3*b0 + a1*b13 - a13*b1 + a2*b23 - a23*b2 + a12*b123 + a123*b12
    r[..., 4] = a0*b12 + a12*b0 + a1*b2 - a2*b1 + a13*b23 - a23*b13 + a3*b123 - a123*b3
    r[..., 5] = a0*b13 + a13*b0 + a1*b3 - a3*b1 - a12*b23 + a23*b12 - a2*b123 + a123*b2
    r[..., 6] = a0*b23 + a23*b0 + a2*b3 - a3*b2 + a12*b13 - a13*b12 + a1*b123 - a123*b1
    r[..., 7] = a0*b123 + a123*b0 + a1*b23 - a23*b1 - a2*b13 + a13*b2 + a3*b12 - a12*b3
    return r


def gp_rotor_mv(s, p12, p13, p23, x):
    """Sparse geometric product: rotor * multivector. ~28 FMAs vs 64 for full GP.
    s, p12, p13, p23: rotor components, shape (...,)
    x: multivector, shape (..., 8)
    Returns: shape (..., 8)
    """
    r = np.empty_like(x)
    r[..., 0] = s*x[..., 0] - p12*x[..., 4] - p13*x[..., 5] - p23*x[..., 6]
    r[..., 1] = s*x[..., 1] + p12*x[..., 2] + p13*x[..., 3] + p23*x[..., 7]
    r[..., 2] = s*x[..., 2] - p12*x[..., 1] + p23*x[..., 3] - p13*x[..., 7]
    r[..., 3] = s*x[..., 3] - p13*x[..., 1] - p23*x[..., 2] + p12*x[..., 7]
    r[..., 4] = s*x[..., 4] + p12*x[..., 0]
    r[..., 5] = s*x[..., 5] + p13*x[..., 0]
    r[..., 6] = s*x[..., 6] + p23*x[..., 0]
    r[..., 7] = s*x[..., 7] - p23*x[..., 1] + p13*x[..., 2] - p12*x[..., 3]
    return r


def make_random_rotor(rng: np.random.Generator) -> np.ndarray:
    """Generate a random normalized rotor in Cl(3,0). Returns shape (8,)."""
    bv = rng.standard_normal(3)
    angle = rng.uniform(0, 2 * np.pi)
    bv_norm = np.linalg.norm(bv)
    if bv_norm < 1e-8:
        return np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    bv_hat = bv / bv_norm
    ha = angle / 2
    rotor = np.zeros(8)
    rotor[0] = np.cos(ha)
    rotor[4] = np.sin(ha) * bv_hat[0]  # e12
    rotor[5] = np.sin(ha) * bv_hat[1]  # e13
    rotor[6] = np.sin(ha) * bv_hat[2]  # e23
    # Normalize
    norm = np.sqrt(rotor[0]**2 + rotor[4]**2 + rotor[5]**2 + rotor[6]**2)
    return rotor / norm


def rotor_sandwich(s, p12, p13, p23, x):
    """Fused rotor sandwich: R x R_tilde. Two sparse GPs."""
    temp = gp_rotor_mv(s, p12, p13, p23, x)
    return gp_rotor_mv(s, -p12, -p13, -p23, temp)


def embed_vectors(v: np.ndarray) -> tuple[np.ndarray, int]:
    """Embed d-dim vectors as Cl(3,0) multivectors. v shape (..., d) -> (..., n_groups, 8)."""
    d = v.shape[-1]
    pad = (3 - d % 3) % 3
    if pad > 0:
        v = np.pad(v, [(0, 0)] * (v.ndim - 1) + [(0, pad)])
    n_groups = v.shape[-1] // 3
    v_grouped = v.reshape(*v.shape[:-1], n_groups, 3)
    mv = np.zeros((*v_grouped.shape[:-1], 8), dtype=v.dtype)
    mv[..., 1] = v_grouped[..., 0]
    mv[..., 2] = v_grouped[..., 1]
    mv[..., 3] = v_grouped[..., 2]
    return mv, d


def extract_vectors(mv: np.ndarray, orig_dim: int) -> np.ndarray:
    """Extract vectors from multivectors. mv shape (..., n_groups, 8) -> (..., d)."""
    v = np.stack([mv[..., 1], mv[..., 2], mv[..., 3]], axis=-1)
    v = v.reshape(*mv.shape[:-2], -1)
    return v[..., :orig_dim]
