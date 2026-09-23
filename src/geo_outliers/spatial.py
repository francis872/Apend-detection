from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.preprocessing import RobustScaler


def _rank01(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,float)
    if len(x)<=1:return np.zeros(len(x))
    return np.argsort(np.argsort(x,kind="mergesort"),kind="mergesort")/max(len(x)-1,1)


def _knn(xy:np.ndarray,k:int):
    k_eff=min(max(1,k),len(xy)-1)
    nn=NearestNeighbors(n_neighbors=k_eff+1).fit(xy)
    dist,neigh=nn.kneighbors(xy)
    return dist[:,1:],neigh[:,1:],k_eff


def local_moran_lisa(coords:np.ndarray,values:np.ndarray,k:int=12,permutations:int=199,random_state:int=42)->dict:
    xy=np.asarray(coords,float); y=np.asarray(values,float)
    finite=np.isfinite(y)&np.isfinite(xy).all(axis=1)
    ids=np.flatnonzero(finite)
    empty={"local_i":np.zeros(len(y)),"pvalue":np.ones(len(y)),"cluster":np.array(["NS"]*len(y),dtype=object),
           "significant":np.zeros(len(y),dtype=bool),"global_moran_i":None,"global_pvalue":None,"k":k,"permutations":permutations}
    if len(ids)<max(20,k+2):return empty
    x=xy[ids];v=y[ids];dist,neigh,k_eff=_knn(x,k)
    z=(v-v.mean())/(v.std(ddof=1)+1e-12)
    lag=np.mean(z[neigh],axis=1); local=z*lag
    observed_global=float(np.mean(local))
    rng=np.random.default_rng(random_state)
    exceed=np.zeros(len(ids),int); global_perm=[]
    for _ in range(max(0,permutations)):
        zp=rng.permutation(z); lp=np.mean(zp[neigh],axis=1); ip=zp*lp
        exceed+=np.abs(ip)>=np.abs(local)
        global_perm.append(float(np.mean(ip)))
    p=(exceed+1)/(max(0,permutations)+1)
    gp=None
    if global_perm:
        gp=float((sum(abs(vv)>=abs(observed_global) for vv in global_perm)+1)/(len(global_perm)+1))
    sig=p<=.05
    labels=np.full(len(ids),"NS",dtype=object)
    labels[sig&(z>0)&(lag>0)]="HH"
    labels[sig&(z<0)&(lag<0)]="LL"
    labels[sig&(z>0)&(lag<0)]="HL"
    labels[sig&(z<0)&(lag>0)]="LH"
    full_i=np.zeros(len(y));full_p=np.ones(len(y));full_sig=np.zeros(len(y),bool);full_lab=np.array(["NS"]*len(y),dtype=object)
    full_i[ids]=local;full_p[ids]=p;full_sig[ids]=sig;full_lab[ids]=labels
    return {"local_i":full_i,"pvalue":full_p,"cluster":full_lab,"significant":full_sig,
            "global_moran_i":observed_global,"global_pvalue":gp,"k":k_eff,"permutations":int(permutations)}


def spatial_density(coords:np.ndarray,k:int=12)->np.ndarray:
    xy=np.asarray(coords,float);finite=np.isfinite(xy).all(axis=1);out=np.zeros(len(xy))
    ids=np.flatnonzero(finite)
    if len(ids)<3:return out
    dist,_,k_eff=_knn(xy[ids],k)
    radius=np.maximum(dist[:,-1],1e-9)
    density=k_eff/(np.pi*radius**2)
    out[ids]=_rank01(density)
    return out


def spatial_validation(coords: np.ndarray, values: np.ndarray, k: int = 12) -> tuple[np.ndarray, dict]:
    """Spatial anomaly score with permutation LISA, kNN density, LOF, residual and DBSCAN context."""
    xy=np.asarray(coords,float); y=np.asarray(values,float)
    finite=np.isfinite(y) & np.isfinite(xy).all(axis=1)
    score=np.zeros(len(y),float)
    if finite.sum() < max(20,k+2):
        return score, {"global_moran_i":None,"global_moran_pvalue":None,"k":k,"valid_rows":int(finite.sum()),"hotspots":0,"coldspots":0,"clusters":0}

    ids=np.flatnonzero(finite); x=xy[ids]; v=y[ids]
    dist,neigh,k_eff=_knn(x,k)
    lisa=local_moran_lisa(x,v,k=k_eff,permutations=199,random_state=42)
    local_i=lisa["local_i"]; labels=lisa["cluster"]; significant=lisa["significant"]
    hotspot=labels=="HH"; coldspot=labels=="LL"

    local_mean=np.mean(v[neigh],axis=1)
    resid_rank=_rank01(np.abs(v-local_mean))
    joint=np.c_[RobustScaler().fit_transform(x),RobustScaler().fit_transform(v.reshape(-1,1))]
    lof=LocalOutlierFactor(n_neighbors=min(k_eff+1,len(ids)-1),contamination="auto")
    lof.fit_predict(joint);lof_rank=_rank01(-lof.negative_outlier_factor_)
    moran_rank=_rank01(np.abs(local_i))
    density_rank=spatial_density(x,k_eff)
    score[ids]=.35*resid_rank+.30*lof_rank+.20*moran_rank+.15*density_rank

    median_nn=float(np.median(dist[:,0])) if dist.size else 1.0
    eps=max(median_nn*2.5,1e-9)
    db=DBSCAN(eps=eps,min_samples=max(4,min(12,k_eff))).fit_predict(x)
    cluster_labels={int(c):int((db==c).sum()) for c in np.unique(db) if c>=0}

    return score,{
        "global_moran_i":lisa["global_moran_i"],"global_moran_pvalue":lisa["global_pvalue"],
        "k":int(k_eff),"valid_rows":int(len(ids)),
        "method":"Permutation LISA + spatial LOF + neighbour residual + kNN density",
        "lisa_permutations":int(lisa["permutations"]),"lisa_significant":int(significant.sum()),
        "hotspots":int(hotspot.sum()),"coldspots":int(coldspot.sum()),
        "spatial_outliers_hl":int((labels=="HL").sum()),"spatial_outliers_lh":int((labels=="LH").sum()),
        "clusters":len(cluster_labels),"cluster_sizes":cluster_labels,"dbscan_noise_points":int((db<0).sum()),
        "median_nearest_neighbor_distance":median_nn,"dbscan_eps":float(eps),
        "density_rank_p95":float(np.quantile(density_rank,.95)),
        "local_moran_summary":{"median":float(np.median(local_i)),"p95":float(np.quantile(local_i,.95)),"p05":float(np.quantile(local_i,.05))}
    }
