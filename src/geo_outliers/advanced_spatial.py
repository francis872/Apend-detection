from __future__ import annotations
import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors

def _prepare(points,value_keys):
    if len(points)<5:raise ValueError("At least 5 observations required")
    xy=np.asarray([[float(p["longitude"]),float(p["latitude"])] for p in points],float)
    vals={k:np.asarray([float(p[k]) for p in points],float) for k in value_keys}
    finite=np.isfinite(xy).all(axis=1)
    for v in vals.values():finite&=np.isfinite(v)
    if finite.sum()<5:raise ValueError("At least 5 finite observations required")
    return xy[finite],{k:v[finite] for k,v in vals.items()},np.flatnonzero(finite)

def _weights(xy,k):
    k_eff=min(max(1,int(k)),len(xy)-1)
    nn=NearestNeighbors(n_neighbors=k_eff+1).fit(xy)
    _,ind=nn.kneighbors(xy)
    W=np.zeros((len(xy),len(xy)),float)
    for i,row in enumerate(ind[:,1:]):W[i,row]=1
    W/=np.maximum(W.sum(axis=1,keepdims=True),1)
    return W,k_eff

def getis_ord_gi_star(points,value_key="value",k=8,permutations=199,random_state=42):
    xy,v,ids=_prepare(points,[value_key]);x=v[value_key];n=len(x);W,k_eff=_weights(xy,k)
    # Gi* includes self in the local neighbourhood.
    Ws=W.copy();np.fill_diagonal(Ws,1);Ws/=np.maximum(Ws.sum(axis=1,keepdims=True),1)
    mean=x.mean();sd=x.std(ddof=1)
    sumw=Ws.sum(axis=1);sumw2=(Ws**2).sum(axis=1)
    denom=sd*np.sqrt(np.maximum((n*sumw2-sumw**2)/max(n-1,1),1e-15))
    z=(Ws@x-mean*sumw)/denom
    rng=np.random.default_rng(random_state);exceed=np.zeros(n,int)
    for _ in range(max(0,int(permutations))):
        xp=rng.permutation(x);zp=(Ws@xp-xp.mean()*sumw)/(xp.std(ddof=1)*np.sqrt(np.maximum((n*sumw2-sumw**2)/max(n-1,1),1e-15)))
        exceed+=np.abs(zp)>=np.abs(z)
    p=(exceed+1)/(max(0,int(permutations))+1)
    records=[]
    for j,i in enumerate(ids):
        label="NS"
        if p[j]<=.05:label="HOTSPOT" if z[j]>0 else "COLDSPOT"
        records.append({"index":int(i),"gi_star_z":float(z[j]),"p_value":float(p[j]),"significant":bool(p[j]<=.05),"cluster":label})
    return {"method":"Getis-Ord Gi*","n":n,"k":k_eff,"permutations":int(permutations),"records":records,
            "significance_is_statistical":True,"multiple_testing_note":"Local p-values are unadjusted; consider FDR for many simultaneous tests.","causal_claim":False}

def bivariate_moran(points,x_key,y_key,k=8,permutations=199,random_state=42):
    xy,v,ids=_prepare(points,[x_key,y_key]);x=v[x_key];y=v[y_key];W,k_eff=_weights(xy,k)
    zx=(x-x.mean())/(x.std(ddof=1)+1e-12);zy=(y-y.mean())/(y.std(ddof=1)+1e-12);lag_y=W@zy
    local=zx*lag_y;observed=float(local.mean());rng=np.random.default_rng(random_state);null=[]
    for _ in range(max(0,int(permutations))):null.append(float((zx*(W@rng.permutation(zy))).mean()))
    p=float((sum(abs(q)>=abs(observed) for q in null)+1)/(len(null)+1)) if null else None
    records=[{"index":int(ids[i]),"local_i":float(local[i]),"x_z":float(zx[i]),"lag_y_z":float(lag_y[i])} for i in range(len(ids))]
    return {"method":"Bivariate Moran's I","n":len(ids),"x":x_key,"y":y_key,"coefficient":observed,"p_value":p,"k":k_eff,
            "permutations":int(permutations),"records":records,"interpretation":"Association between X at each unit and spatial lag of Y in neighbouring units.","causal_claim":False}

def spatial_association_matrix(points,variables,k=8,permutations=99):
    if len(variables)<2:raise ValueError("At least two variables required")
    matrix=[]
    for x in variables:
        row=[]
        for y in variables:
            if x==y:row.append({"coefficient":1.0,"p_value":0.0})
            else:
                r=bivariate_moran(points,x,y,k,permutations)
                row.append({"coefficient":r["coefficient"],"p_value":r["p_value"]})
        matrix.append(row)
    return {"variables":variables,"matrix":matrix,"method":"Bivariate Moran spatial association matrix","causal_claim":False}

def temporal_spatial_comparison(points_t0,points_t1,value_key="value",method="gi_star",k=8,permutations=199):
    if method=="gi_star":
        a=getis_ord_gi_star(points_t0,value_key,k,permutations);b=getis_ord_gi_star(points_t1,value_key,k,permutations)
        ca={r["index"]:r for r in a["records"]};cb={r["index"]:r for r in b["records"]};common=sorted(set(ca)&set(cb))
        changes=[{"index":i,"z_t0":ca[i]["gi_star_z"],"z_t1":cb[i]["gi_star_z"],"delta_z":cb[i]["gi_star_z"]-ca[i]["gi_star_z"],
                  "cluster_t0":ca[i]["cluster"],"cluster_t1":cb[i]["cluster"]} for i in common]
        return {"method":"Getis-Ord Gi* t0/t1","t0":a,"t1":b,"changes":changes,"causal_claim":False}
    raise ValueError("method must be gi_star")
