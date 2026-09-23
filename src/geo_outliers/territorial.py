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
    con.execute("""CREATE TABLE IF NOT EXISTS territorial_monitoring(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      object_key TEXT NOT NULL,
      object_version INTEGER NOT NULL,
      job_id TEXT NOT NULL,
      summary_json TEXT NOT NULL,
      baseline_json TEXT,
      change_json TEXT,
      status TEXT NOT NULL,
      created_at TEXT NOT NULL,
      UNIQUE(object_key,job_id)
    )""")
    con.execute("""CREATE INDEX IF NOT EXISTS idx_monitoring_object ON territorial_monitoring(object_key,id)""")
    con.execute("""CREATE TABLE IF NOT EXISTS watchlists(
      id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, domain TEXT,
      enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS watchlist_objects(
      watchlist_id INTEGER NOT NULL, object_key TEXT NOT NULL, created_at TEXT NOT NULL,
      UNIQUE(watchlist_id,object_key))""")
    con.execute("""CREATE TABLE IF NOT EXISTS alert_rules(
      id INTEGER PRIMARY KEY AUTOINCREMENT, watchlist_id INTEGER NOT NULL, name TEXT NOT NULL,
      metric TEXT NOT NULL, operator TEXT NOT NULL, threshold REAL NOT NULL,
      severity TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS alert_incidents(
      id INTEGER PRIMARY KEY AUTOINCREMENT, fingerprint TEXT NOT NULL UNIQUE, watchlist_id INTEGER,
      object_key TEXT NOT NULL, rule_id INTEGER, first_job_id TEXT NOT NULL, last_job_id TEXT NOT NULL,
      severity TEXT NOT NULL, status TEXT NOT NULL, title TEXT NOT NULL, evidence_json TEXT NOT NULL,
      occurrences INTEGER NOT NULL DEFAULT 1, opened_at TEXT NOT NULL, updated_at TEXT NOT NULL,
      resolved_at TEXT)""")
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


def _monitor_row(r):
    d=dict(r)
    for key in ("summary_json","baseline_json","change_json"):
        raw=d.pop(key)
        d[key.replace("_json","")]=json.loads(raw) if raw else None
    return d


def monitoring_history(object_key:str,limit:int=100)->list[dict]:
    with _connect() as con:
        rows=con.execute("SELECT * FROM territorial_monitoring WHERE object_key=? ORDER BY id DESC LIMIT ?",(object_key,min(max(limit,1),500))).fetchall()
    return [_monitor_row(r) for r in rows]


def _baseline_from_history(history:list[dict],window:int=8)->dict|None:
    if len(history)<3:return None
    rows=history[:window]
    keys=("rows","critical_99","high_95","elevated_90","mean_score","mean_probability","mean_spatial","mean_temporal","consensus_mean")
    out={}
    for key in keys:
        vals=[]
        for r in rows:
            v=(r.get("summary") or {}).get(key)
            if isinstance(v,(int,float)) and np.isfinite(v): vals.append(float(v))
        if vals:
            out[key]={"median":float(np.median(vals)),"mad":float(np.median(np.abs(np.asarray(vals)-np.median(vals))))}
    return out or None


def _change_against_baseline(summary:dict,baseline:dict|None)->dict:
    if not baseline:return {"status":"baseline_building","signals":[],"score":0.0}
    signals=[]; scores=[]
    for key,stats_ in baseline.items():
        value=summary.get(key)
        if not isinstance(value,(int,float)) or not np.isfinite(value):continue
        median=float(stats_["median"]); mad=float(stats_.get("mad") or 0)
        scale=max(1.4826*mad,abs(median)*.10,0.05 if key.startswith("mean_") or key=="consensus_mean" else 1.0)
        deviation=abs(float(value)-median)/scale
        scores.append(min(deviation/5.0,1.0))
        if deviation>=3.5:
            signals.append({"metric":key,"current":float(value),"baseline":median,"robust_deviation":float(deviation),
                            "direction":"up" if float(value)>median else "down"})
    return {"status":"changed" if signals else "stable","signals":signals,"score":float(np.mean(scores)) if scores else 0.0}


def monitor_object(job_dir:Path,obj:dict,baseline_window:int=8)->dict:
    gdf=load_analysis_layer(job_dir)
    selected=select_polygon(gdf,obj["geometry"])
    summary=summarize_region(selected)
    history=monitoring_history(obj["object_key"],limit=baseline_window)
    baseline=_baseline_from_history(history,baseline_window)
    change=_change_against_baseline(summary,baseline)
    now=datetime.now(timezone.utc).isoformat()

    with _connect() as con:
        con.execute("""INSERT OR REPLACE INTO territorial_monitoring(
          object_key,object_version,job_id,summary_json,baseline_json,change_json,status,created_at
        ) VALUES(?,?,?,?,?,?,?,?)""",
        (obj["object_key"],int(obj["version"]),job_dir.name,json.dumps(summary),json.dumps(baseline) if baseline else None,
         json.dumps(change),change["status"],now))

        if change["status"]=="changed":
            for sig in change["signals"]:
                severity="critical" if sig["robust_deviation"]>=5 else "high"
                title=f"{obj['name']} changed: {sig['metric']} {sig['direction']}"
                evidence={"metric":sig["metric"],"current":sig["current"],"baseline":sig["baseline"],
                          "robust_deviation":sig["robust_deviation"],"object_version":obj["version"]}
                con.execute("INSERT INTO territorial_alerts(object_key,job_id,severity,alert_type,title,evidence_json,created_at) VALUES(?,?,?,?,?,?,?)",
                            (obj["object_key"],job_dir.name,severity,"territorial_change",title,json.dumps(evidence),now))

    return {"object_key":obj["object_key"],"object_version":obj["version"],"job_id":job_dir.name,
            "summary":summary,"baseline":baseline,"change":change,"status":change["status"],"created_at":now}


def monitor_all_objects(job_dir:Path)->dict:
    objects=list_territorial_objects(True,500)
    results=[]
    for obj in objects:
        try:results.append(monitor_object(job_dir,obj))
        except Exception as e:
            results.append({"object_key":obj["object_key"],"job_id":job_dir.name,"status":"error","error":str(e)})
    changed=sum(1 for x in results if x.get("status")=="changed")
    return {"job_id":job_dir.name,"objects":len(objects),"changed":changed,"results":results}


def territorial_monitoring_dashboard(object_key:str)->dict:
    objects=[x for x in list_territorial_objects(False,500) if x["object_key"]==object_key]
    if not objects: raise KeyError(object_key)
    latest=objects[0]
    history=monitoring_history(object_key,100)
    alerts=list_territorial_alerts(object_key,100)
    return {
        "object":latest,
        "history":history,
        "alerts":alerts,
        "baseline_state":"ready" if len(history)>=3 else "building",
        "monitoring_runs":len(history),
        "latest_change":history[0]["change"] if history else None,
    }


def create_watchlist(name:str,domain:str|None=None)->dict:
    now=datetime.now(timezone.utc).isoformat()
    with _connect() as con:
        cur=con.execute("INSERT INTO watchlists(name,domain,enabled,created_at) VALUES(?,?,1,?)",(name,domain,now))
        return {"id":int(cur.lastrowid),"name":name,"domain":domain,"enabled":True,"created_at":now}


def list_watchlists()->list[dict]:
    with _connect() as con:
        rows=con.execute("""SELECT w.*,COUNT(wo.object_key) object_count
          FROM watchlists w LEFT JOIN watchlist_objects wo ON wo.watchlist_id=w.id
          GROUP BY w.id ORDER BY w.id DESC""").fetchall()
    return [{**dict(r),"enabled":bool(r["enabled"])} for r in rows]


def subscribe_object(watchlist_id:int,object_key:str)->dict:
    if not any(x["object_key"]==object_key for x in list_territorial_objects(True,500)):raise KeyError(object_key)
    now=datetime.now(timezone.utc).isoformat()
    with _connect() as con:
        con.execute("INSERT OR IGNORE INTO watchlist_objects(watchlist_id,object_key,created_at) VALUES(?,?,?)",(watchlist_id,object_key,now))
    return {"watchlist_id":watchlist_id,"object_key":object_key,"subscribed":True}


def create_alert_rule(watchlist_id:int,name:str,metric:str,operator:str,threshold:float,severity:str)->dict:
    allowed_metrics={"change_score","critical_99","high_95","elevated_90","mean_score","mean_spatial","mean_temporal","consensus_mean"}
    if metric not in allowed_metrics:raise ValueError("Unsupported alert metric")
    if operator not in {">=",">","<=","<"}:raise ValueError("Unsupported operator")
    if severity not in {"info","medium","high","critical"}:raise ValueError("Unsupported severity")
    now=datetime.now(timezone.utc).isoformat()
    with _connect() as con:
        cur=con.execute("INSERT INTO alert_rules(watchlist_id,name,metric,operator,threshold,severity,enabled,created_at) VALUES(?,?,?,?,?,?,1,?)",
                        (watchlist_id,name,metric,operator,float(threshold),severity,now))
        rid=int(cur.lastrowid)
    return {"id":rid,"watchlist_id":watchlist_id,"name":name,"metric":metric,"operator":operator,"threshold":float(threshold),"severity":severity,"enabled":True}


def list_alert_rules(watchlist_id:int|None=None)->list[dict]:
    with _connect() as con:
        rows=con.execute("SELECT * FROM alert_rules WHERE watchlist_id=? ORDER BY id DESC",(watchlist_id,)).fetchall() if watchlist_id else con.execute("SELECT * FROM alert_rules ORDER BY id DESC").fetchall()
    return [{**dict(r),"enabled":bool(r["enabled"])} for r in rows]


def _rule_match(value:float,op:str,threshold:float)->bool:
    return {">=":value>=threshold,">":value>threshold,"<=":value<=threshold,"<":value<threshold}[op]


def _severity_rank(s:str)->int:return {"info":0,"medium":1,"high":2,"critical":3}.get(s,0)


def process_watchlist_alerts(job_id:str,monitoring_results:list[dict])->dict:
    by_object={x.get("object_key"):x for x in monitoring_results if x.get("status")!="error"}
    opened=updated=resolved=0
    now=datetime.now(timezone.utc).isoformat()
    with _connect() as con:
        subscriptions=con.execute("""SELECT wo.object_key,w.id watchlist_id,w.name watchlist_name
          FROM watchlist_objects wo JOIN watchlists w ON w.id=wo.watchlist_id WHERE w.enabled=1""").fetchall()
        active_fingerprints=set()
        for sub in subscriptions:
            result=by_object.get(sub["object_key"])
            if not result:continue
            rules=con.execute("SELECT * FROM alert_rules WHERE watchlist_id=? AND enabled=1",(sub["watchlist_id"],)).fetchall()
            for rule in rules:
                value=(result.get("change") or {}).get("score") if rule["metric"]=="change_score" else (result.get("summary") or {}).get(rule["metric"])
                if not isinstance(value,(int,float)) or not np.isfinite(value):continue
                fp=f'{sub["watchlist_id"]}:{sub["object_key"]}:{rule["id"]}'
                if _rule_match(float(value),rule["operator"],float(rule["threshold"])):
                    active_fingerprints.add(fp)
                    evidence={"metric":rule["metric"],"value":float(value),"operator":rule["operator"],"threshold":float(rule["threshold"])}
                    old=con.execute("SELECT * FROM alert_incidents WHERE fingerprint=?",(fp,)).fetchone()
                    title=f'{rule["name"]} · {sub["watchlist_name"]}'
                    if old and old["status"] in {"open","escalated"}:
                        severity=rule["severity"]
                        status="escalated" if _severity_rank(severity)>_severity_rank(old["severity"]) else old["status"]
                        con.execute("UPDATE alert_incidents SET last_job_id=?,severity=?,status=?,evidence_json=?,occurrences=occurrences+1,updated_at=? WHERE fingerprint=?",
                                    (job_id,severity,status,json.dumps(evidence),now,fp)); updated+=1
                    else:
                        con.execute("""INSERT OR REPLACE INTO alert_incidents(fingerprint,watchlist_id,object_key,rule_id,first_job_id,last_job_id,severity,status,title,evidence_json,occurrences,opened_at,updated_at,resolved_at)
                          VALUES(?,?,?,?,?,?,?,?,?,?,1,?,?,NULL)""",(fp,sub["watchlist_id"],sub["object_key"],rule["id"],job_id,job_id,rule["severity"],"open",title,json.dumps(evidence),now,now)); opened+=1
        existing=con.execute("SELECT fingerprint FROM alert_incidents WHERE status IN ('open','escalated')").fetchall()
        for row in existing:
            if row["fingerprint"] not in active_fingerprints:
                con.execute("UPDATE alert_incidents SET status='resolved',resolved_at=?,updated_at=? WHERE fingerprint=?",(now,now,row["fingerprint"])); resolved+=1
    return {"opened":opened,"updated":updated,"resolved":resolved,"active":len(active_fingerprints)}


def list_incidents(status:str|None=None,limit:int=200)->list[dict]:
    with _connect() as con:
        rows=con.execute("SELECT * FROM alert_incidents WHERE status=? ORDER BY updated_at DESC LIMIT ?",(status,min(limit,500))).fetchall() if status else con.execute("SELECT * FROM alert_incidents ORDER BY updated_at DESC LIMIT ?",(min(limit,500),)).fetchall()
    out=[]
    for r in rows:
        d=dict(r);d["evidence"]=json.loads(d.pop("evidence_json"));out.append(d)
    return out


def operations_center()->dict:
    incidents=list_incidents(None,500)
    active=[x for x in incidents if x["status"] in {"open","escalated"}]
    return {"watchlists":list_watchlists(),"rules":list_alert_rules(),"active_incidents":active,
            "counts":{"active":len(active),"critical":sum(x["severity"]=="critical" for x in active),
                      "high":sum(x["severity"]=="high" for x in active),"resolved":sum(x["status"]=="resolved" for x in incidents)}}
