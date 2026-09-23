from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.preprocessing import RobustScaler


def _rank01(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,float)
    if len(x)<=1:return np.zeros(len(x))
    return np.argsort(np.argsort(x,kind="mergesort"),kind="mergesort")/max(len(x)-1,1)


def spatial_validation(coords: np.ndarray, values: np.ndarray, k: int = 12) -> tuple[np.ndarray, dict]:
    """Spatial anomaly score with kNN Local Moran, LOF, neighbour residual and DBSCAN context."""
    xy=np.asarray(coords,float); y=np.asarray(values,float)
    finite=np.isfinite(y) & np.isfinite(xy).all(axis=1)
    score=np.zeros(len(y),float)
    if finite.sum() < max(20,k+2):
        return score, {"global_moran_i":None,"k":k,"valid_rows":int(finite.sum()),"hotspots":0,"coldspots":0,"clusters":0}

    ids=np.flatnonzero(finite); x=xy[ids]; v=y[ids]
    k_eff=min(k,len(ids)-1)
    nn=NearestNeighbors(n_neighbors=k_eff+1).fit(x)
    dist, neigh=nn.kneighbors(x)
    neigh=neigh[:,1:]; dist=dist[:,1:]

    z=(v-v.mean())/(v.std(ddof=1)+1e-12)
    neigh_z=np.mean(z[neigh],axis=1)
    local_i=z*neigh_z
    global_i=float(np.mean(local_i))
    hotspot=(z>0)&(neigh_z>0)
    coldspot=(z<0)&(neigh_z<0)

    local_mean=np.mean(v[neigh],axis=1)
    resid=np.abs(v-local_mean)
    resid_rank=_rank01(resid)

    joint=np.c_[RobustScaler().fit_transform(x),RobustScaler().fit_transform(v.reshape(-1,1))]
    lof=LocalOutlierFactor(n_neighbors=min(k_eff+1,len(ids)-1),contamination="auto")
    lof.fit_predict(joint)
    lof_rank=_rank01(-lof.negative_outlier_factor_)

    moran_rank=_rank01(np.abs(local_i-np.median(local_i)))
    score[ids]=.45*resid_rank+.35*lof_rank+.20*moran_rank

    median_nn=float(np.median(dist[:,0])) if dist.size else 1.0
    eps=max(median_nn*2.5,1e-9)
    labels=DBSCAN(eps=eps,min_samples=max(4,min(12,k_eff))).fit_predict(x)
    cluster_labels={int(c):int((labels==c).sum()) for c in np.unique(labels) if c>=0}
    cluster_count=len(cluster_labels)
    noise_points=int((labels<0).sum())

    return score,{
        "global_moran_i":global_i,
        "k":int(k_eff),
        "valid_rows":int(len(ids)),
        "method":"kNN Local Moran + spatial LOF + neighbour residual",
        "hotspots":int(hotspot.sum()),
        "coldspots":int(coldspot.sum()),
        "clusters":int(cluster_count),
        "cluster_sizes":cluster_labels,
        "dbscan_noise_points":noise_points,
        "median_nearest_neighbor_distance":median_nn,
        "dbscan_eps":float(eps),
        "local_moran_summary":{
            "median":float(np.median(local_i)),
            "p95":float(np.quantile(local_i,.95)),
            "p05":float(np.quantile(local_i,.05))
        }
    }
