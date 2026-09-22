from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.preprocessing import RobustScaler


def spatial_validation(coords: np.ndarray, values: np.ndarray, k: int = 12) -> tuple[np.ndarray, dict]:
    """Spatial anomaly score using kNN Local Moran and spatial LOF."""
    xy=np.asarray(coords,float); y=np.asarray(values,float)
    finite=np.isfinite(y) & np.isfinite(xy).all(axis=1)
    score=np.zeros(len(y),float)
    if finite.sum() < max(20,k+2):
        return score, {"global_moran_i": None, "k": k, "valid_rows": int(finite.sum())}
    ids=np.flatnonzero(finite); x=xy[ids]; v=y[ids]
    k_eff=min(k, len(ids)-1)
    nn=NearestNeighbors(n_neighbors=k_eff+1).fit(x)
    neigh=nn.kneighbors(return_distance=False)[:,1:]
    z=(v-v.mean())/(v.std(ddof=1)+1e-12)
    local_i=z * np.mean(z[neigh],axis=1)
    global_i=float(np.mean(local_i))
    local_mean=np.mean(v[neigh],axis=1)
    resid=np.abs(v-local_mean)
    denom=max(len(resid)-1,1)
    resid_rank=np.argsort(np.argsort(resid,kind="mergesort"),kind="mergesort")/denom
    joint=np.c_[RobustScaler().fit_transform(x), RobustScaler().fit_transform(v.reshape(-1,1))]
    lof=LocalOutlierFactor(n_neighbors=min(k_eff+1,len(ids)-1), contamination="auto")
    lof.fit_predict(joint)
    lof_factor=-lof.negative_outlier_factor_
    lof_rank=np.argsort(np.argsort(lof_factor,kind="mergesort"),kind="mergesort")/denom
    local_moran_anomaly=np.abs(local_i-np.median(local_i))
    moran_rank=np.argsort(np.argsort(local_moran_anomaly,kind="mergesort"),kind="mergesort")/denom
    score[ids]=.45*resid_rank+.35*lof_rank+.20*moran_rank
    return score,{
        "global_moran_i": global_i, "k": int(k_eff), "valid_rows": int(len(ids)),
        "method": "kNN Local Moran + spatial LOF + neighbour residual"
    }
