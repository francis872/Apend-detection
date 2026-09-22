from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import stats


def gaussian_quadrature(func, a: float, b: float, n: int = 64) -> float:
    """Approximate integral_a^b func(x) dx with Gauss-Legendre quadrature."""
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("Gauss-Legendre requires finite integration bounds")
    if b < a:
        return -gaussian_quadrature(func, b, a, n=n)
    if a == b:
        return 0.0
    nodes, weights = leggauss(n)
    x = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    return float(0.5 * (b - a) * np.sum(weights * func(x)))


def _finite_support(dist, params: tuple[float, ...], eps: float = 1e-10) -> tuple[float, float]:
    """Return numerically finite bounds covering almost all fitted probability mass."""
    lo, hi = dist.support(*params)
    if not np.isfinite(lo):
        lo = float(dist.ppf(eps, *params))
    if not np.isfinite(hi):
        hi = float(dist.ppf(1.0 - eps, *params))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError("Could not derive finite quadrature support")
    return float(lo), float(hi)


def distribution_interval_probability(
    distribution: str,
    params: tuple[float, ...],
    a: float,
    b: float,
    n: int = 64,
) -> float:
    """Area under a fitted PDF between a and b using Gaussian quadrature."""
    dist = getattr(stats, distribution)
    return gaussian_quadrature(lambda z: dist.pdf(z, *params), float(a), float(b), n=n)


def quadrature_cdf(
    values: np.ndarray,
    distribution: str,
    params: tuple[float, ...],
    n: int = 48,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Numerical CDF evaluated with Gauss-Legendre quadrature.

    Infinite support is truncated using extreme fitted quantiles. Values outside the
    truncated support are assigned 0 or 1. Computation is chunked to avoid creating
    a very large nodes matrix for geospatial datasets.
    """
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    dist = getattr(stats, distribution)
    lo, hi = _finite_support(dist, params)
    nodes, weights = leggauss(n)

    finite_idx = np.flatnonzero(np.isfinite(x))
    for start in range(0, len(finite_idx), chunk_size):
        ids = finite_idx[start:start + chunk_size]
        v = x[ids]
        low = v <= lo
        high = v >= hi
        mid = ~(low | high)

        out[ids[low]] = 0.0
        out[ids[high]] = 1.0

        if np.any(mid):
            vm = v[mid]
            # For each value x_j transform Legendre nodes from [-1,1] to [lo,x_j].
            scale = 0.5 * (vm - lo)
            center = 0.5 * (vm + lo)
            z = scale[:, None] * nodes[None, :] + center[:, None]
            pdf = dist.pdf(z, *params)
            integ = scale * np.sum(pdf * weights[None, :], axis=1)
            out[ids[mid]] = np.clip(integ, 0.0, 1.0)

    return out


def quadrature_tail_score(
    values: np.ndarray,
    distribution: str,
    params: tuple[float, ...],
    n: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two-sided tail probability and -log10 anomaly score from quadrature."""
    cdf = quadrature_cdf(values, distribution, params, n=n)
    tail = 2.0 * np.minimum(cdf, 1.0 - cdf)
    tail = np.clip(tail, 1e-15, 1.0)
    score = -np.log10(tail)
    return tail, score


def confidence_interval_areas(
    distribution: str,
    params: tuple[float, ...],
    levels: tuple[float, ...] = (0.90, 0.95, 0.99),
    n: int = 96,
) -> dict[str, dict[str, float]]:
    """Integrate central probability intervals and report numerical area error."""
    dist = getattr(stats, distribution)
    result: dict[str, dict[str, float]] = {}
    for level in levels:
        alpha = 1.0 - level
        a = float(dist.ppf(alpha / 2.0, *params))
        b = float(dist.ppf(1.0 - alpha / 2.0, *params))
        area = distribution_interval_probability(distribution, params, a, b, n=n)
        pct = str(int(round(level * 100)))
        result[pct] = {
            "lower": a,
            "upper": b,
            "quadrature_area": float(area),
            "target_area": float(level),
            "absolute_error": float(abs(area - level)),
        }
    return result
