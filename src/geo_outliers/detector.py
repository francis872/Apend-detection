from __future__ import annotations

import ast
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import RobustScaler

from .probability import fit_distributions, fitted_tail_score
from .geometry import local_spd_scores, procrustes_neighborhood_scores

DEFAULT_FEATURES = ['BRIGHTNESS','BRIGHT_T31','FRP','SCAN','TRACK']


def _pct_rank(v: np.ndarray) -> np.ndarray:
    s = pd.Series(v)
    return s.rank(pct=True, method='average').to_numpy(float)


def detect_outliers(
    gdf: gpd.GeoDataFrame,
    features: list[str] | None = None,
    probability_feature: str = 'FRP',
    confidence_levels: tuple[float,...] = (.90,.95,.99),
    spd_k: int = 25,
    procrustes_k: int = 12,
    max_geometry_rows: int = 20000,
) -> tuple[gpd.GeoDataFrame, dict, pd.DataFrame]:
    x = gdf.copy()
    features = features or [c for c in DEFAULT_FEATURES if c in x.columns]
    if len(features) < 2:
        raise ValueError('At least two numeric features are required')

    num = x[features].apply(pd.to_numeric, errors='coerce')
    med = num.median()
    num = num.fillna(med)
    scaler = RobustScaler().fit(num)
    z = scaler.transform(num)

    try:
        mcd = MinCovDet(random_state=42, support_fraction=None).fit(z)
        mahal = np.sqrt(np.maximum(mcd.mahalanobis(z), 0))
    except Exception:
        cov = np.cov(z, rowvar=False) + np.eye(z.shape[1])*1e-6
        inv = np.linalg.pinv(cov)
        mahal = np.sqrt(np.einsum('ij,jk,ik->i', z, inv, z))

    fit_table = fit_distributions(x[probability_feature])
    best = fit_table.iloc[0]
    params = ast.literal_eval(best['params'])
    prob_tail = fitted_tail_score(x[probability_feature], best['distribution'], params)
    prob_tail = np.nan_to_num(prob_tail, nan=np.nanmedian(prob_tail[np.isfinite(prob_tail)]))

    n = len(x)
    if x.crs is None:
        raise ValueError('CRS is required')
    projected = x.to_crs(x.estimate_utm_crs())
    coords = np.c_[projected.geometry.x, projected.geometry.y]

    if n <= max_geometry_rows:
        spd = local_spd_scores(z, k=spd_k)
        proc = procrustes_neighborhood_scores(coords, k=procrustes_k)
    else:
        rng = np.random.default_rng(42)
        ids = np.sort(rng.choice(n, size=max_geometry_rows, replace=False))
        z_s, c_s = z[ids], coords[ids]
        spd_s = local_spd_scores(z_s, k=spd_k)
        proc_s = procrustes_neighborhood_scores(c_s, k=procrustes_k)
        from sklearn.neighbors import NearestNeighbors
        cscale = RobustScaler().fit_transform(coords)
        cs_s = cscale[ids]
        joint_s = np.c_[z_s, cs_s]
        joint = np.c_[z, cscale]
        kn = NearestNeighbors(n_neighbors=1).fit(joint_s)
        nearest = kn.kneighbors(joint, return_distance=False).ravel()
        spd = spd_s[nearest]
        proc = proc_s[nearest]

    components = pd.DataFrame({
        'score_mahal': _pct_rank(mahal),
        'score_probability': _pct_rank(prob_tail),
        'score_spd': _pct_rank(spd),
        'score_procrustes': _pct_rank(proc),
    })
    weights = np.array([.30,.30,.25,.15])
    ensemble = components.to_numpy() @ weights
    x = x.reset_index(drop=True)
    for c in components.columns:
        x[c] = components[c]
    x['outlier_score'] = ensemble

    thresholds = {}
    for level in confidence_levels:
        pct = int(round(level*100))
        q = float(np.quantile(ensemble, level))
        thresholds[pct] = q
        x[f'outlier_{pct}'] = ensemble >= q
    x['outlier_level'] = 'normal'
    if 90 in thresholds: x.loc[x['outlier_90'], 'outlier_level'] = '90'
    if 95 in thresholds: x.loc[x['outlier_95'], 'outlier_level'] = '95'
    if 99 in thresholds: x.loc[x['outlier_99'], 'outlier_level'] = '99'

    summary = {
        'rows': n,
        'features': features,
        'probability_feature': probability_feature,
        'best_distribution': best['distribution'],
        'best_distribution_metrics': {
            'aic': float(best['aic']), 'bic': float(best['bic']),
            'ks_stat': float(best['ks_stat']), 'ks_pvalue': float(best['ks_pvalue']),
            'bhattacharyya': float(best['bhattacharyya']),
            'fisher_rao': float(best['fisher_rao']),
        },
        'thresholds': thresholds,
        'counts': {str(p): int(x[f'outlier_{p}'].sum()) for p in thresholds},
        'note': '90/95/99 are empirical score quantiles: top 10%, 5%, 1% respectively.'
    }
    return x, summary, fit_table


def remove_noise(gdf: gpd.GeoDataFrame, level: int = 99) -> gpd.GeoDataFrame:
    col = f'outlier_{level}'
    if col not in gdf.columns:
        raise KeyError(f'{col} not found; run detect_outliers first')
    return gdf.loc[~gdf[col]].copy()
