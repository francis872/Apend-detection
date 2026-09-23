from __future__ import annotations

import json, sqlite3, uuid
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.neighbors import KernelDensity, NearestNeighbors
from shapely.geometry import Point, shape, mapping
from shapely.ops import unary_union

from .data_quality import assess_data_quality
from .spatial import local_moran_lisa

DB_PATH=Path("runtime/meridian_territorial.sqlite3")
VERSION="1.0.0"


def _now(): return datetime.now(timezone.utc).isoformat()
def _connect():
    con=sqlite3.connect(DB_PATH);con.row_factory=sqlite3.Row;return con
def _json(v): return json.dumps(v,ensure_ascii=False,default=str)


def register_dataset(gdf:gpd.GeoDataFrame, metadata:dict)->dict:
    dataset_id=metadata.get("dataset_id") or "ds_"+uuid.uuid4().hex
    quality=assess_data_quality(gdf)
    if gdf.crs is None: raise ValueError("Dataset CRS is required")
    geometry_type="mixed"
    if len(gdf): geometry_type=",".join(sorted(set(gdf.geometry.geom_type.dropna().astype(str))))
    record={
      "dataset_id":dataset_id,"name":metadata.get("name") or dataset_id,"source":metadata.get("source"),
      "source_url":metadata.get("source_url"),"license":metadata.get("license"),"description":metadata.get("description"),
      "geometry_type":geometry_type,"crs":str(gdf.crs),"temporal_start":metadata.get("temporal_start"),
      "temporal_end":metadata.get("temporal_end"),"resolution":metadata.get("resolution"),
      "update_frequency":metadata.get("update_frequency"),"created_at":_now(),
      "metadata":{**(metadata.get("metadata") or {}),"quality":quality}
    }
    with _connect() as con:
        con.execute("""INSERT INTO geodata_datasets(dataset_id,name,source,source_url,license,description,geometry_type,crs,
        temporal_start,temporal_end,resolution,update_frequency,created_at,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (record["dataset_id"],record["name"],record["source"],record["source_url"],record["license"],record["description"],
         record["geometry_type"],record["crs"],record["temporal_start"],record["temporal_end"],record["resolution"],
         record["update_frequency"],record["created_at"],_json(record["metadata"])))
        con.execute("INSERT INTO data_lineage(dataset_id,stage,transformation,algorithm_version,created_at,metadata_json) VALUES(?,?,?,?,?,?)",
                    (dataset_id,"INGESTION","validation + registration",VERSION,_now(),_json({"rows":len(gdf),"crs":str(gdf.crs)})))
    return record


def list_datasets(limit:int=200)->list[dict]:
    with _connect() as con: rows=con.execute("SELECT * FROM geodata_datasets ORDER BY created_at DESC LIMIT ?",(limit,)).fetchall()
    return [{**dict(r),"metadata":json.loads(r["metadata_json"])} for r in rows]


def add_lineage(dataset_id:str,stage:str,transformation:str,metadata:dict|None=None,parent_dataset_id:str|None=None,job_id:str|None=None):
    with _connect() as con:
        con.execute("INSERT INTO data_lineage(dataset_id,stage,transformation,algorithm_version,parent_dataset_id,job_id,created_at,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
                    (dataset_id,stage,transformation,VERSION,parent_dataset_id,job_id,_now(),_json(metadata or {})))


def lineage(dataset_id:str)->list[dict]:
    with _connect() as con: rows=con.execute("SELECT * FROM data_lineage WHERE dataset_id=? ORDER BY id",(dataset_id,)).fetchall()
    return [{**dict(r),"metadata":json.loads(r["metadata_json"])} for r in rows]


def spatial_join(left:gpd.GeoDataFrame,right:gpd.GeoDataFrame,predicate:str="intersects",how:str="inner")->gpd.GeoDataFrame:
    if left.crs is None or right.crs is None:raise ValueError("Both layers require CRS")
    return gpd.sjoin(left,right.to_crs(left.crs),predicate=predicate,how=how)


def nearest_join(left:gpd.GeoDataFrame,right:gpd.GeoDataFrame,max_distance:float|None=None)->gpd.GeoDataFrame:
    if left.crs is None or right.crs is None:raise ValueError("Both layers require CRS")
    if left.crs.is_geographic:
        crs=left.estimate_utm_crs(); left=left.to_crs(crs); right=right.to_crs(crs)
    else:right=right.to_crs(left.crs)
    return gpd.sjoin_nearest(left,right,how="left",max_distance=max_distance,distance_col="distance")


def create_event(payload:dict)->dict:
    geom=payload.get("geometry")
    if not geom:
        if payload.get("latitude") is None or payload.get("longitude") is None:raise ValueError("geometry or latitude/longitude required")
        geom=mapping(Point(float(payload["longitude"]),float(payload["latitude"])))
    p=shape(geom)
    if p.geom_type!="Point":raise ValueError("SpatialEvent geometry must be Point")
    eid=payload.get("id") or "evt_"+uuid.uuid4().hex
    rec={"id":eid,"event_type":str(payload["event_type"]),"latitude":float(p.y),"longitude":float(p.x),"geometry":mapping(p),
         "date":payload.get("date"),"source":payload.get("source"),"source_id":payload.get("source_id"),
         "confidence":payload.get("confidence"),"administrative_unit":payload.get("administrative_unit"),
         "metadata":payload.get("metadata") or {},"dataset_id":payload.get("dataset_id"),"created_at":_now()}
    with _connect() as con:
        con.execute("""INSERT INTO spatial_events(id,event_type,latitude,longitude,geometry_json,date,source,source_id,confidence,
        administrative_unit,metadata_json,dataset_id,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (eid,rec["event_type"],rec["latitude"],rec["longitude"],_json(rec["geometry"]),rec["date"],rec["source"],rec["source_id"],
         rec["confidence"],rec["administrative_unit"],_json(rec["metadata"]),rec["dataset_id"],rec["created_at"]))
    return rec


def territorial_snapshot(payload:dict)->dict:
    required=("territory_id","timestamp")
    for k in required:
        if not payload.get(k):raise ValueError(f"{k} is required")
    groups={k:payload.get(k) or {} for k in ("social_features","economic_features","ecological_features","satellite_features","land_use_features")}
    with _connect() as con:
        cur=con.execute("""INSERT INTO territorial_snapshots(territory_id,timestamp,social_json,economic_json,ecological_json,
        satellite_json,land_use_json,provenance_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)""",
        (payload["territory_id"],payload["timestamp"],_json(groups["social_features"]),_json(groups["economic_features"]),
         _json(groups["ecological_features"]),_json(groups["satellite_features"]),_json(groups["land_use_features"]),
         _json(payload.get("provenance") or {}),_now()))
    return {"id":cur.lastrowid,"territory_id":payload["territory_id"],"timestamp":payload["timestamp"],**groups}


def snapshots(territory_id:str)->list[dict]:
    with _connect() as con: rows=con.execute("SELECT * FROM territorial_snapshots WHERE territory_id=? ORDER BY timestamp",(territory_id,)).fetchall()
    out=[]
    for r in rows:
        d=dict(r)
        for col,key in (("social_json","social_features"),("economic_json","economic_features"),("ecological_json","ecological_features"),("satellite_json","satellite_features"),("land_use_json","land_use_features"),("provenance_json","provenance")):
            d[key]=json.loads(d.pop(col))
        out.append(d)
    return out


def gini(values:list[float])->float|None:
    x=np.asarray([v for v in values if v is not None and np.isfinite(v)],float)
    if not len(x):return None
    if np.min(x)<0:x=x-np.min(x)
    if np.allclose(x,0):return 0.0
    x=np.sort(x);n=len(x)
    return float((2*np.sum((np.arange(1,n+1))*x)/(n*np.sum(x)))-(n+1)/n)


def territorial_indicators(values:dict)->dict:
    area=float(values.get("area",0) or 0);pop=float(values.get("population",0) or 0);events=float(values.get("event_count",0) or 0)
    out={"event_count":events,"event_density":events/area if area>0 else None,
         "population_normalized_rate":events/pop if pop>0 else None,"area_normalized_rate":events/area if area>0 else None}
    if values.get("land_values") is not None:out["gini"]=gini(values["land_values"])
    if values.get("land_areas") is not None:out["land_gini"]=gini(values["land_areas"])
    for k in ("land_use_change","vegetation_change","water_change","thermal_anomaly","infrastructure_accessibility"):
        if k in values:out[k]=values[k]
    return {"indicators":out,"interpretation_policy":{"gini_measures_concentration_not_conflict":True,"causal_claims":False}}


def correlation(x:list[float],y:list[float],method:str="pearson")->dict:
    a=np.asarray(x,float);b=np.asarray(y,float);valid=np.isfinite(a)&np.isfinite(b);a=a[valid];b=b[valid]
    if len(a)<3:raise ValueError("At least 3 paired observations required")
    if method=="pearson":coef,p=stats.pearsonr(a,b)
    elif method=="spearman":coef,p=stats.spearmanr(a,b)
    else:raise ValueError("method must be pearson or spearman")
    return {"method":method,"n":len(a),"coefficient":float(coef),"p_value":float(p),"causal_claim":False}


def distance_model(a:list[float],b:list[float],reference:list[list[float]]|None=None)->dict:
    x=np.asarray(a,float);y=np.asarray(b,float)
    if x.shape!=y.shape:raise ValueError("Vectors must have same shape")
    eu=float(np.linalg.norm(x-y))
    scale=np.ones_like(x)
    maha=None
    if reference is not None:
        ref=np.asarray(reference,float)
        if ref.ndim==2 and ref.shape[1]==len(x) and len(ref)>=3:
            sd=np.std(ref,axis=0,ddof=1);scale=np.where(sd>1e-12,sd,1)
            cov=np.cov(ref,rowvar=False);inv=np.linalg.pinv(np.atleast_2d(cov))
            d=x-y;maha=float(np.sqrt(d@inv@d))
    standardized=float(np.linalg.norm((x-y)/scale))
    cos=None if np.allclose(x,0) or np.allclose(y,0) else float(1-cosine(x,y))
    return {"euclidean":eu,"standardized_euclidean":standardized,"mahalanobis":maha,"cosine_similarity":cos,"distance_is_not_probability":True}


def hotspot(points:list[dict],value_key:str="value",method:str="lisa",bandwidth:float|None=None)->dict:
    if len(points)<5:raise ValueError("At least 5 observations required")
    xy=np.array([[float(p["longitude"]),float(p["latitude"])] for p in points])
    vals=np.array([float(p.get(value_key,1)) for p in points])
    if method=="lisa":
        result=local_moran_lisa(xy,vals,k=min(12,len(points)-2),permutations=199)
        return {"method":"Local Moran's I","global_moran_i":result["global_moran_i"],"global_pvalue":result["global_pvalue"],
                "records":[{"cluster":str(result["cluster"][i]),"local_i":float(result["local_i"][i]),"p_value":float(result["pvalue"][i])} for i in range(len(points))],
                "significance_is_statistical":True}
    if method=="kde":
        bw=float(bandwidth or max(np.std(xy),1e-3)/5)
        kde=KernelDensity(bandwidth=bw).fit(xy,sample_weight=np.maximum(vals,0))
        density=np.exp(kde.score_samples(xy))
        return {"method":"KDE","bandwidth":bw,"density":density.tolist(),"significance_is_statistical":False}
    raise ValueError("method must be lisa or kde")


def detect_corridors(points:list[dict],distance_threshold:float,period:dict|None=None,data_sources:list|None=None)->list[dict]:
    if len(points)<3:raise ValueError("At least 3 points required")
    xy=np.array([[float(p["longitude"]),float(p["latitude"])] for p in points])
    nn=NearestNeighbors(radius=float(distance_threshold)).fit(xy)
    neigh=nn.radius_neighbors(xy,return_distance=False)
    seen=set();groups=[]
    for i in range(len(points)):
        if i in seen:continue
        stack=[i];comp=set()
        while stack:
            j=stack.pop()
            if j in comp:continue
            comp.add(j);seen.add(j);stack.extend(int(k) for k in neigh[j] if int(k) not in comp)
        if len(comp)>=3:groups.append(sorted(comp))
    results=[]
    for group in groups:
        geoms=[Point(*xy[i]) for i in group]
        line=unary_union(geoms).convex_hull
        support=len(group);confidence=float(min(.99,.5+.5*support/len(points)))
        results.append({"corridor_id":"cor_"+uuid.uuid4().hex,"geometry":mapping(line),"variables":sorted(set(str(points[i].get("variable","event")) for i in group)),
                        "period":period or {},"supporting_events":[points[i].get("id",i) for i in group],"confidence":confidence,
                        "method":"proximity connected-components candidate","data_sources":data_sources or [],
                        "limitations":["Candidate corridor from proximity/connectivity only","Confidence is support score, not probability","No causal or conflict interpretation"]})
    return results
