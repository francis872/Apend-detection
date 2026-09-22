from __future__ import annotations

import ast
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import RobustScaler

from .probability import fit_distributions
from .quadrature import quadrature_tail_score, confidence_interval_areas
from .geometry import local_spd_scores, procrustes_neighborhood_scores
from .spatial import spatial_validation
from .temporal import temporal_validation
from .derivatives import observation_derivative_features

DEFAULT_FEATURES=['BRIGHTNESS','BRIGHT_T31','FRP','SCAN','TRACK']


def _pct_rank(v):
    return pd.Series(v).rank(pct=True,method='average').fillna(.5).to_numpy(float)


def detect_outliers(
    gdf:gpd.GeoDataFrame, features:list[str]|None=None, probability_feature:str='FRP',
    confidence_levels:tuple[float,...]=(.90,.95,.99), spd_k:int=25,
    procrustes_k:int=12, max_geometry_rows:int=20000, quadrature_order:int=48,
):
    x=gdf.copy()
    features=features or [c for c in DEFAULT_FEATURES if c in x.columns]
    if probability_feature not in x.columns:
        raise KeyError(f'{probability_feature} not found')
    if len(features)<2:
        raise ValueError('At least two numeric features are required')

    num=x[features].apply(pd.to_numeric,errors='coerce').fillna(
        x[features].apply(pd.to_numeric,errors='coerce').median()
    )
    z=RobustScaler().fit_transform(num)
    try:
        mcd=MinCovDet(random_state=42).fit(z)
        mahal=np.sqrt(np.maximum(mcd.mahalanobis(z),0))
    except Exception:
        cov=np.cov(z,rowvar=False)+np.eye(z.shape[1])*1e-6
        inv=np.linalg.pinv(cov)
        mahal=np.sqrt(np.einsum('ij,jk,ik->i',z,inv,z))

    fits=fit_distributions(x[probability_feature])
    best=fits.iloc[0]; params=ast.literal_eval(best['params'])
    raw=pd.to_numeric(x[probability_feature],errors='coerce').to_numpy(float)
    quad_tail,quad_score=quadrature_tail_score(raw,best['distribution'],params,n=quadrature_order)
    finite=quad_score[np.isfinite(quad_score)]
    quad_score=np.nan_to_num(quad_score,nan=float(np.median(finite)) if len(finite) else 0,posinf=15,neginf=0)

    if x.crs is None: raise ValueError('CRS is required')
    projected=x.to_crs(x.estimate_utm_crs())
    coords=np.c_[projected.geometry.x,projected.geometry.y]
    n=len(x)

    if n<=max_geometry_rows:
        spd=local_spd_scores(z,k=spd_k)
        proc=procrustes_neighborhood_scores(coords,k=procrustes_k)
    else:
        from sklearn.neighbors import NearestNeighbors
        rng=np.random.default_rng(42); ids=np.sort(rng.choice(n,max_geometry_rows,replace=False))
        spd_s=local_spd_scores(z[ids],k=spd_k)
        proc_s=procrustes_neighborhood_scores(coords[ids],k=procrustes_k)
        cscale=RobustScaler().fit_transform(coords)
        joint_s=np.c_[z[ids],cscale[ids]]; joint=np.c_[z,cscale]
        nearest=NearestNeighbors(n_neighbors=1).fit(joint_s).kneighbors(joint,return_distance=False).ravel()
        spd,proc=spd_s[nearest],proc_s[nearest]

    spatial,spatial_meta=spatial_validation(coords,raw,k=12)
    temporal,temporal_meta=temporal_validation(x,probability_feature)
    derivative_features,derivative_meta=observation_derivative_features(raw,best['distribution'],params)

    components=pd.DataFrame({
        'score_mahal':_pct_rank(mahal),
        'score_probability':_pct_rank(quad_score),
        'score_spd':_pct_rank(spd),
        'score_procrustes':_pct_rank(proc),
        'score_spatial':spatial,
        'score_temporal':temporal,
    })
    # Probability remains central while spatial and temporal context add validation.
    weights=np.array([.20,.25,.18,.10,.17,.10])
    ensemble=components.to_numpy()@weights

    x=x.reset_index(drop=True)
    for c in components: x[c]=components[c]
    x['probability_tail_area']=quad_tail
    x['probability_quadrature_score']=quad_score
    for name,values in derivative_features.items(): x[name]=values
    x['outlier_score']=ensemble

    thresholds={}
    for level in confidence_levels:
        pct=int(round(level*100)); q=float(np.quantile(ensemble,level))
        thresholds[pct]=q; x[f'outlier_{pct}']=ensemble>=q
    x['outlier_level']='normal'
    for p in (90,95,99):
        if p in thresholds: x.loc[x[f'outlier_{p}'],'outlier_level']=str(p)

    # Explainability: count independent high-score methods and record causes.
    component_cols=list(components.columns)
    high=components>=.95
    x['consensus_methods']=high.sum(axis=1).astype(int)
    def reasons(row):
        names=[c.replace('score_','') for c in component_cols if row[c]>=.95]
        return ','.join(names) if names else 'ensemble'
    x['anomaly_reason']=x.apply(reasons,axis=1)
    x['anomaly_candidate']=x.get('outlier_95',False)
    # Noise is intentionally stricter than anomaly: extreme ensemble + >=3 methods.
    x['noise_candidate']=x.get('outlier_99',False) & (x['consensus_methods']>=3)
    x['record_status']=np.where(x['noise_candidate'],'probable_noise',
                         np.where(x['anomaly_candidate'],'valid_anomaly_candidate','normal'))

    areas=confidence_interval_areas(best['distribution'],params,levels=confidence_levels,n=max(96,quadrature_order))
    summary={
        'rows':n,'features':features,'probability_feature':probability_feature,
        'best_distribution':best['distribution'],
        'best_distribution_params':[float(v) for v in params],
        'best_distribution_metrics':{
            'aic':float(best['aic']),'bic':float(best['bic']),
            'ks_stat':float(best['ks_stat']),'ks_pvalue':float(best['ks_pvalue']),
            'bhattacharyya':float(best['bhattacharyya']),'fisher_rao':float(best['fisher_rao'])
        },
        'gaussian_quadrature':{'method':'Gauss-Legendre','order':quadrature_order,
            'confidence_interval_areas':areas,'tail_score':'-log10(two-sided tail area)'},
        'spatial_validation':spatial_meta,'temporal_validation':temporal_meta,
        'derivative_analysis':derivative_meta,
        'ensemble_weights':dict(zip(component_cols,weights.tolist())),
        'thresholds':thresholds,
        'counts':{str(p):int(x[f'outlier_{p}'].sum()) for p in thresholds},
        'noise_candidates':int(x['noise_candidate'].sum()),
        'strong_consensus_count':int((x['consensus_methods']>=3).sum()),
        'note':'Anomaly != noise. Noise requires 99% ensemble level plus >=3 independent high-score methods.'
    }
    return x,summary,fits


def remove_noise(gdf:gpd.GeoDataFrame,level:int=99):
    if 'noise_candidate' in gdf.columns:
        return gdf.loc[~gdf['noise_candidate']].copy()
    col=f'outlier_{level}'
    if col not in gdf.columns: raise KeyError(f'{col} not found')
    return gdf.loc[~gdf[col]].copy()
