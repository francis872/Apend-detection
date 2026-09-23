from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats
from shapely.geometry import shape, mapping, LineString

DB_PATH=Path("runtime/meridian_territorial.sqlite3")


def _connect():
    DB_PATH.parent.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(DB_PATH); con.row_factory=sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS layers(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      job_id TEXT,
      geometry_json TEXT NOT NULL,
      metadata_json TEXT,
      created_at TEXT NOT NULL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS territorial_objects(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      object_key TEXT NOT NULL,
      version INTEGER NOT NULL,
      name TEXT NOT NULL,
      object_type TEXT NOT NULL,
      job_id TEXT,
      geometry_json TEXT NOT NULL,
      metadata_json TEXT,
      status TEXT NOT NULL,
      created_at TEXT NOT NULL,
      UNIQUE(object_key,version)
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS territorial_alerts(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      object_key TEXT NOT NULL,
      job_id TEXT NOT NULL,
      severity TEXT NOT NULL,
      alert_type TEXT NOT NULL,
      title TEXT NOT NULL,
      evidence_json TEXT NOT NULL,
      created_at TEXT NOT NULL
    )""")
    return con


def load_analysis_layer(job_dir:Path)->gpd.GeoDataFrame:
    candidates=[job_dir/"results"/"analysis.gpkg",job_dir/"results"/"analysis.geojson"]
    for p in candidates:
        if p.exists():
            gdf=gpd.read_file(p)
            if gdf.crs is None: raise ValueError("Analysis layer has no CRS")
            return gdf.to_crs(4326)
    raise FileNotFoundError("Completed analysis geospatial layer is unavailable")


def select_polygon(gdf:gpd.GeoDataFrame, geometry:dict)->gpd.GeoDataFrame:
    geom=shape(geometry)
    if geom.is_empty or geom.geom_type not in {"Polygon","MultiPolygon"}:
        raise ValueError("Geometry must be a non-empty Polygon or MultiPolygon")
    if not geom.is_valid: geom=geom.buffer(0)
    mask=gdf.geometry.intersects(geom)
    return gdf.loc[mask].copy()


def summarize_region(gdf:gpd.GeoDataFrame)->dict:
    n=len(gdf)
    if n==0:
        return {"rows":0,"critical_99":0,"high_95":0,"elevated_90":0,"mean_score":None,"median_score":None,
                "mean_probability":None,"mean_spatial":None,"mean_temporal":None,"consensus_mean":None}
    def num(col):
        return pd.to_numeric(gdf[col],errors="coerce") if col in gdf.columns else pd.Series(dtype=float)
    score=num("outlier_score")
    return {
        "rows":int(n),
        "critical_99":int(gdf.get("outlier_99",pd.Series(False,index=gdf.index)).fillna(False).astype(bool).sum()),
        "high_95":int(gdf.get("outlier_95",pd.Series(False,index=gdf.index)).fillna(False).astype(bool).sum()),
        "elevated_90":int(gdf.get("outlier_90",pd.Series(False,index=gdf.index)).fillna(False).astype(bool).sum()),
        "mean_score":float(score.mean()) if score.notna().any() else None,
        "median_score":float(score.median()) if score.notna().any() else None,
        "mean_probability":float(num("score_probability").mean()) if "score_probability" in gdf else None,
        "mean_spatial":float(num("score_spatial").mean()) if "score_spatial" in gdf else None,
        "mean_temporal":float(num("score_temporal").mean()) if "score_temporal" in gdf else None,
        "consensus_mean":float(num("consensus_methods").mean()) if "consensus_methods" in gdf else None,
    }


def compare_regions(a:gpd.GeoDataFrame,b:gpd.GeoDataFrame)->dict:
    sa,sb=summarize_region(a),summarize_region(b)
    xa=pd.to_numeric(a.get("outlier_score"),errors="coerce").dropna().to_numpy(float) if len(a) else np.array([])
    xb=pd.to_numeric(b.get("outlier_score"),errors="coerce").dropna().to_numpy(float) if len(b) else np.array([])
    tests={"mann_whitney":None,"ks_2samp":None,"effect_size_rank_biserial":None}
    if len(xa)>=3 and len(xb)>=3:
        mw=stats.mannwhitneyu(xa,xb,alternative="two-sided")
        ks=stats.ks_2samp(xa,xb)
        u=float(mw.statistic); rbc=1.0-(2.0*u/(len(xa)*len(xb)))
        tests={"mann_whitney":{"statistic":u,"pvalue":float(mw.pvalue)},
               "ks_2samp":{"statistic":float(ks.statistic),"pvalue":float(ks.pvalue)},
               "effect_size_rank_biserial":float(rbc)}
    return {
        "region_a":sa,"region_b":sb,
        "delta":{"rows":sa["rows"]-sb["rows"],
                 "critical_99":sa["critical_99"]-sb["critical_99"],
                 "mean_score":None if sa["mean_score"] is None or sb["mean_score"] is None else sa["mean_score"]-sb["mean_score"]},
        "tests":tests,
        "interpretation_note":"P-values describe distributional differences in the selected samples; they do not establish causality."
    }


def save_layer(name:str,geometry:dict,job_id:str|None=None,metadata:dict|None=None)->dict:
    geom=shape(geometry)
    if geom.is_empty or geom.geom_type not in {"Polygon","MultiPolygon"}: raise ValueError("Only Polygon/MultiPolygon layers are supported")
    now=datetime.now(timezone.utc).isoformat()
    with _connect() as con:
        cur=con.execute("INSERT INTO layers(name,job_id,geometry_json,metadata_json,created_at) VALUES(?,?,?,?,?)",
                        (name,job_id,json.dumps(geometry),json.dumps(metadata or {}),now))
        lid=int(cur.lastrowid)
    return {"id":lid,"name":name,"job_id":job_id,"geometry":geometry,"metadata":metadata or {},"created_at":now}


def list_layers(limit:int=100)->list[dict]:
    with _connect() as con:
        rows=con.execute("SELECT * FROM layers ORDER BY id DESC LIMIT ?",(min(max(limit,1),500),)).fetchall()
    out=[]
    for r in rows:
        d=dict(r); d["geometry"]=json.loads(d.pop("geometry_json")); d["metadata"]=json.loads(d.pop("metadata_json") or "{}"); out.append(d)
    return out


def delete_layer(layer_id:int)->bool:
    with _connect() as con:
        cur=con.execute("DELETE FROM layers WHERE id=?",(layer_id,))
    return cur.rowcount>0


def _object_row(r):
    d=dict(r); d["geometry"]=json.loads(d.pop("geometry_json")); d["metadata"]=json.loads(d.pop("metadata_json") or "{}"); return d


def create_territorial_object(name:str,object_type:str,geometry:dict,job_id:str|None=None,metadata:dict|None=None,object_key:str|None=None)->dict:
    import uuid
    geom=shape(geometry)
    if geom.is_empty or geom.geom_type not in {"Polygon","MultiPolygon"}: raise ValueError("Territorial objects require Polygon/MultiPolygon geometry")
    key=object_key or uuid.uuid4().hex
    with _connect() as con:
        row=con.execute("SELECT MAX(version) v FROM territorial_objects WHERE object_key=?",(key,)).fetchone()
        version=int(row["v"] or 0)+1
        now=datetime.now(timezone.utc).isoformat()
        con.execute("UPDATE territorial_objects SET status='superseded' WHERE object_key=? AND status='active'",(key,))
        con.execute("INSERT INTO territorial_objects(object_key,version,name,object_type,job_id,geometry_json,metadata_json,status,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (key,version,name,object_type,job_id,json.dumps(geometry),json.dumps(metadata or {}),"active",now))
        row=con.execute("SELECT * FROM territorial_objects WHERE object_key=? AND version=?",(key,version)).fetchone()
    return _object_row(row)


def list_territorial_objects(active_only:bool=True,limit:int=200)->list[dict]:
    with _connect() as con:
        if active_only: rows=con.execute("SELECT * FROM territorial_objects WHERE status='active' ORDER BY id DESC LIMIT ?",(min(limit,500),)).fetchall()
        else: rows=con.execute("SELECT * FROM territorial_objects ORDER BY id DESC LIMIT ?",(min(limit,500),)).fetchall()
    return [_object_row(r) for r in rows]


def object_versions(object_key:str)->list[dict]:
    with _connect() as con: rows=con.execute("SELECT * FROM territorial_objects WHERE object_key=? ORDER BY version DESC",(object_key,)).fetchall()
    return [_object_row(r) for r in rows]


def buffer_geometry(geometry:dict,distance_m:float)->dict:
    if distance_m<=0: raise ValueError("distance_m must be positive")
    geom=shape(geometry)
    gs=gpd.GeoSeries([geom],crs=4326)
    local=gs.estimate_utm_crs()
    buffered=gs.to_crs(local).buffer(distance_m).to_crs(4326).iloc[0]
    return mapping(buffered)


def corridor_geometry(coordinates:list,distance_m:float)->dict:
    if len(coordinates)<2: raise ValueError("Corridor requires at least two coordinates")
    return buffer_geometry(mapping(LineString(coordinates)),distance_m)


def intersect_geometries(a:dict,b:dict)->dict:
    result=shape(a).intersection(shape(b))
    if result.is_empty:return {"type":"GeometryCollection","geometries":[]}
    return mapping(result)


def territorial_findings(summary:dict,name:str)->list[dict]:
    findings=[]
    if summary["critical_99"]>0:
        findings.append({"severity":"critical","type":"critical_concentration","title":f"Critical observations inside {name}","evidence":{"count":summary["critical_99"],"rows":summary["rows"]}})
    if summary.get("mean_spatial") is not None and summary["mean_spatial"]>=.75:
        findings.append({"severity":"high","type":"spatial_signal","title":f"Elevated spatial signal inside {name}","evidence":{"mean_spatial":summary["mean_spatial"]}})
    if summary.get("mean_temporal") is not None and summary["mean_temporal"]>=.75:
        findings.append({"severity":"high","type":"temporal_signal","title":f"Elevated temporal signal inside {name}","evidence":{"mean_temporal":summary["mean_temporal"]}})
    return findings


def evaluate_object(job_dir:Path,obj:dict)->dict:
    gdf=load_analysis_layer(job_dir); selected=select_polygon(gdf,obj["geometry"]); summary=summarize_region(selected)
    findings=territorial_findings(summary,obj["name"])
    alerts=[]
    now=datetime.now(timezone.utc).isoformat()
    with _connect() as con:
        for f in findings:
            con.execute("INSERT INTO territorial_alerts(object_key,job_id,severity,alert_type,title,evidence_json,created_at) VALUES(?,?,?,?,?,?,?)",
                        (obj["object_key"],obj.get("job_id") or job_dir.name,f["severity"],f["type"],f["title"],json.dumps(f["evidence"]),now))
            alerts.append({**f,"created_at":now})
    return {"object":obj,"summary":summary,"findings":findings,"alerts":alerts}


def list_territorial_alerts(object_key:str|None=None,limit:int=200)->list[dict]:
    with _connect() as con:
        rows=con.execute("SELECT * FROM territorial_alerts WHERE object_key=? ORDER BY id DESC LIMIT ?",(object_key,min(limit,500))).fetchall() if object_key else con.execute("SELECT * FROM territorial_alerts ORDER BY id DESC LIMIT ?",(min(limit,500),)).fetchall()
    out=[]
    for r in rows:
        d=dict(r); d["evidence"]=json.loads(d.pop("evidence_json")); out.append(d)
    return out
