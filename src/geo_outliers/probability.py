from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats

DISTRIBUTIONS = [
    'norm','lognorm','gamma','weibull_min','weibull_max','expon','gumbel_r','gumbel_l',
    'logistic','laplace','cauchy','t','chi2','f','beta','pareto','genextreme','invgauss',
    'rayleigh','uniform'
]


def _hist_density(x: np.ndarray, bins: int = 80):
    hist, edges = np.histogram(x, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    p = np.clip(hist * widths, 1e-15, None)
    p /= p.sum()
    return centers, widths, p


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), 1e-15, None); p /= p.sum()
    q = np.clip(np.asarray(q, float), 1e-15, None); q /= q.sum()
    bc = np.sum(np.sqrt(p*q))
    return float(-np.log(np.clip(bc, 1e-15, 1.0)))


def fisher_rao_distance_discrete(p: np.ndarray, q: np.ndarray) -> float:
    """Fisher-Rao/Hellinger-sphere geodesic for discrete probability vectors."""
    p = np.clip(np.asarray(p, float), 1e-15, None); p /= p.sum()
    q = np.clip(np.asarray(q, float), 1e-15, None); q /= q.sum()
    inner = np.sum(np.sqrt(p*q))
    return float(2.0 * np.arccos(np.clip(inner, -1.0, 1.0)))


def fit_distributions(series: pd.Series, bins: int = 80) -> pd.DataFrame:
    x = pd.to_numeric(series, errors='coerce').dropna().to_numpy(float)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        raise ValueError('At least 30 finite values are required')
    centers, widths, p_emp = _hist_density(x, bins=bins)
    rows = []
    for name in DISTRIBUTIONS:
        dist = getattr(stats, name)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                params = dist.fit(x)
                logpdf = dist.logpdf(x, *params)
                finite = np.isfinite(logpdf)
                if finite.mean() < .95:
                    continue
                ll = float(logpdf[finite].sum())
                k = len(params)
                n = int(finite.sum())
                aic = 2*k - 2*ll
                bic = k*np.log(n) - 2*ll
                ks_stat, ks_p = stats.kstest(x, name, args=params)
                pdf = dist.pdf(centers, *params)
                q = np.clip(pdf*widths, 1e-15, None); q /= q.sum()
                bdist = bhattacharyya_distance(p_emp, q)
                fr = fisher_rao_distance_discrete(p_emp, q)
                rows.append({
                    'distribution': name, 'params': repr(tuple(float(v) for v in params)),
                    'log_likelihood': ll, 'aic': aic, 'bic': bic,
                    'ks_stat': float(ks_stat), 'ks_pvalue': float(ks_p),
                    'bhattacharyya': bdist, 'fisher_rao': fr,
                })
        except Exception:
            continue
    if not rows:
        raise RuntimeError('No distribution could be fit')
    out = pd.DataFrame(rows)
    out['rank_aic'] = out['aic'].rank(method='min')
    out['rank_bic'] = out['bic'].rank(method='min')
    out['rank_ks'] = out['ks_stat'].rank(method='min')
    out['rank_bhat'] = out['bhattacharyya'].rank(method='min')
    out['rank_fr'] = out['fisher_rao'].rank(method='min')
    out['rank_total'] = out[['rank_aic','rank_bic','rank_ks','rank_bhat','rank_fr']].mean(axis=1)
    return out.sort_values(['rank_total','bic','aic']).reset_index(drop=True)


def fitted_tail_score(series: pd.Series, distribution: str, params: tuple[float, ...]) -> np.ndarray:
    x = pd.to_numeric(series, errors='coerce').to_numpy(float)
    dist = getattr(stats, distribution)
    cdf = dist.cdf(x, *params)
    tail = 2*np.minimum(cdf, 1-cdf)
    score = -np.log10(np.clip(tail, 1e-15, 1.0))
    score[~np.isfinite(x)] = np.nan
    return score
