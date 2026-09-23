from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors


def _prepare(coords, *arrays):
    xy=np.asarray(coords,float)
    masks=[np.isfinite(xy).all(axis=1)]
    vals=[]
    for arr in arrays:
        a=np.asarray(arr,float)
        if len(a)!=len(xy):raise ValueError("All arrays must match coords length")
        masks.append(np.isfinite(a));vals.append(a)
    mask=np.logical_and.reduce(masks)
    ids=np.flatnonzero(mask)
    return xy[ids], [v[ids] for v in vals], ids, mask


def knn_weights(coords,k=8,include_self=False,row_standardized=True):
    xy=np.asarray(coords,float)
    if len(xy)<3:raise ValueError("At least 3 coordinates are required")
    k_eff=min(max(1,int(k)),len(xy)-1)
    n_neighbors=k_eff+(0 if include_self else 1)
    nn=NearestNeighbors(n_neighbors=n_neighbors).fit(xy)
    _,idx=nn.kneighbors(xy)
    if not include_self:idx=idx[:,1:]
    W=np.zeros((len(xy),len(xy)),float)
    for i,js in enumerate(idx):W[i,js]=1.0
    if include_self:
        np.fill_diagonal(W,1.0)
    if row_standardized:
        s=W.sum(axis=1,keepdims=True);W=np.divide(W,s,out=np.zeros_like(W),where=s>0)
    return W,k_eff


def getis_ord_gi_star(coords,values,k=8,permutations=199,random_state=42):
    xy,(y,),ids,_=_prepare(coords,values)
    if len(y)<max(8,k+2):raise ValueError("Insufficient valid observations for Gi*")
    W,k_eff=knn_weights(xy,k,include_self=True,row_standardized=False)
    n=len(y);mean=float(np.mean(y));sd=float(np.std(y,ddof=1))
    if sd<1e-12:raise ValueError("Gi* requires non-constant values")
    sumw=W.sum(axis=1);sumw2=(W**2).sum(axis=1)
    denom=sd*np.sqrt(np.maximum((n*sumw2-sumw**2)/(n-1),1e-12))
    z=(W@y-mean*sumw)/denom

    rng=np.random.default_rng(random_state);exceed=np.zeros(n,int)
    for _ in range(max(0,int(permutations))):
        yp=rng.permutation(y)
        zp=(W@yp-mean*sumw)/denom
        exceed+=np.abs(zp)>=np.abs(z)
    p=(exceed+1)/(max(0,int(permutations))+1)
    labels=np.full(n,"NS",dtype=object)
    labels[(p<=.05)&(z>0)]="HOT"
    labels[(p<=.05)&(z<0)]="COLD"
    return {
        "method":"Getis-Ord Gi*",
        "k":k_eff,
        "permutations":int(permutations),
        "n":n,
        "records":[{"index":int(ids[i]),"z_score":float(z[i]),"p_value":float(p[i]),"cluster":str(labels[i]),"significant":bool(p[i]<=.05)} for i in range(n)],
        "summary":{"hotspots":int((labels=="HOT").sum()),"coldspots":int((labels=="COLD").sum()),"significant":int((p<=.05).sum())},
        "methodology":{"weights":"binary kNN with self for Gi*","significance":"empirical two-sided permutation p-value","causal_claims":False},
    }


def global_moran(coords,values,k=8,permutations=199,random_state=42):
    xy,(y,),ids,_=_prepare(coords,values)
    if len(y)<max(8,k+2):raise ValueError("Insufficient valid observations for Moran's I")
    W,k_eff=knn_weights(xy,k,include_self=False,row_standardized=True)
    z=(y-y.mean())/(y.std(ddof=1)+1e-12)
    obs=float(np.mean(z*(W@z)))
    rng=np.random.default_rng(random_state)
    null=[float(np.mean(z*(W@rng.permutation(z)))) for _ in range(max(0,int(permutations)))]
    p=float((sum(abs(v)>=abs(obs) for v in null)+1)/(len(null)+1)) if null else None
    return {"method":"Global Moran's I","n":len(y),"i":obs,"p_value":p,"k":k_eff,"permutations":int(permutations),"causal_claim":False}


def bivariate_moran(coords,x,y,k=8,permutations=199,random_state=42):
    xy,(a,b),ids,_=_prepare(coords,x,y)
    if len(a)<max(8,k+2):raise ValueError("Insufficient valid observations for Bivariate Moran's I")
    W,k_eff=knn_weights(xy,k,include_self=False,row_standardized=True)
    zx=(a-a.mean())/(a.std(ddof=1)+1e-12);zy=(b-b.mean())/(b.std(ddof=1)+1e-12)
    lag_y=W@zy
    local=zx*lag_y
    global_i=float(np.mean(local))
    rng=np.random.default_rng(random_state);global_null=[];exceed=np.zeros(len(a),int)
    for _ in range(max(0,int(permutations))):
        zyp=rng.permutation(zy);lp=W@zyp;loc=zx*lp
        global_null.append(float(np.mean(loc)));exceed+=np.abs(loc)>=np.abs(local)
    gp=float((sum(abs(v)>=abs(global_i) for v in global_null)+1)/(len(global_null)+1)) if global_null else None
    p=(exceed+1)/(max(0,int(permutations))+1)
    labels=np.full(len(a),"NS",dtype=object)
    sig=p<=.05
    labels[sig&(zx>0)&(lag_y>0)]="HH"
    labels[sig&(zx<0)&(lag_y<0)]="LL"
    labels[sig&(zx>0)&(lag_y<0)]="HL"
    labels[sig&(zx<0)&(lag_y>0)]="LH"
    return {
        "method":"Bivariate Moran's I","n":len(a),"global_i":global_i,"global_p_value":gp,"k":k_eff,"permutations":int(permutations),
        "records":[{"index":int(ids[i]),"local_i":float(local[i]),"p_value":float(p[i]),"cluster":str(labels[i]),"significant":bool(sig[i])} for i in range(len(a))],
        "summary":{c:int((labels==c).sum()) for c in ("HH","LL","HL","LH","NS")},
        "methodology":{"x":"local standardized x","spatial_lag_y":"row-standardized kNN lag of y","significance":"permutation on y","causal_claims":False},
    }


def correlation_matrix(rows,variables,method="pearson"):
    if len(variables)<2:raise ValueError("At least two variables are required")
    matrix=[];data={v:np.asarray([r.get(v,np.nan) for r in rows],float) for v in variables}
    for i,a in enumerate(variables):
        for b in variables[i+1:]:
            x=data[a];y=data[b];mask=np.isfinite(x)&np.isfinite(y)
            if mask.sum()<3:res={"n":int(mask.sum()),"coefficient":None,"p_value":None}
            else:
                coef,p=(stats.pearsonr(x[mask],y[mask]) if method=="pearson" else stats.spearmanr(x[mask],y[mask]))
                res={"n":int(mask.sum()),"coefficient":float(coef),"p_value":float(p)}
            matrix.append({"x":a,"y":b,"method":method,**res,"causal_claim":False})
    return {"variables":variables,"method":method,"pairs":matrix}


def compare_periods(coords,values_t0,values_t1,k=8,permutations=199,method="gi_star"):
    if method=="gi_star":
        a=getis_ord_gi_star(coords,values_t0,k,permutations);b=getis_ord_gi_star(coords,values_t1,k,permutations)
        la={r["index"]:r["cluster"] for r in a["records"]};lb={r["index"]:r["cluster"] for r in b["records"]}
    elif method=="lisa":
        from .spatial import local_moran_lisa
        xy,(x0,x1),ids,_=_prepare(coords,values_t0,values_t1)
        r0=local_moran_lisa(xy,x0,k=k,permutations=permutations);r1=local_moran_lisa(xy,x1,k=k,permutations=permutations)
        la={int(ids[i]):str(r0["cluster"][i]) for i in range(len(ids))};lb={int(ids[i]):str(r1["cluster"][i]) for i in range(len(ids))}
        a={"method":"Local Moran's I"};b={"method":"Local Moran's I"}
    else:raise ValueError("method must be gi_star or lisa")
    transitions={}
    records=[]
    for idx in sorted(set(la)&set(lb)):
        tr=f"{la[idx]}->{lb[idx]}";transitions[tr]=transitions.get(tr,0)+1
        records.append({"index":idx,"t0":la[idx],"t1":lb[idx],"transition":tr})
    return {"method":method,"t0":a,"t1":b,"transitions":transitions,"records":records,"causal_claim":False}
