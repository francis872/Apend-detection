from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .event_store import append_event

DB_PATH=Path("runtime/meridian_governance.sqlite3")
DEFAULT_STRATEGY={
    "name":"meridian_ensemble",
    "version":"1.0",
    "weights":{"score_mahal":.20,"score_probability":.25,"score_spd":.18,"score_procrustes":.10,"score_spatial":.17,"score_temporal":.10},
    "status":"champion",
}


def _connect():
    DB_PATH.parent.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(DB_PATH); con.row_factory=sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS strategies(
      id INTEGER PRIMARY KEY AUTOINCREMENT, domain TEXT NOT NULL, name TEXT NOT NULL,
      version TEXT NOT NULL, status TEXT NOT NULL, config_json TEXT NOT NULL,
      created_at TEXT NOT NULL, UNIQUE(domain,name,version))""")
    con.execute("""CREATE TABLE IF NOT EXISTS evaluations(
      id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, domain TEXT NOT NULL,
      champion_json TEXT NOT NULL, challenger_json TEXT, metrics_json TEXT NOT NULL,
      decision TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS drift_snapshots(
      id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, domain TEXT NOT NULL,
      signature_json TEXT NOT NULL, drift_json TEXT NOT NULL, created_at TEXT NOT NULL)""")
    return con


def ensure_champion(domain:str,weights:dict|None=None)->dict:
    with _connect() as con:
        row=con.execute("SELECT * FROM strategies WHERE domain=? AND status='champion' ORDER BY id DESC LIMIT 1",(domain,)).fetchone()
        if row:return _row(row)
        cfg=DEFAULT_STRATEGY.copy(); cfg["weights"]=weights or DEFAULT_STRATEGY["weights"]
        now=datetime.now(timezone.utc).isoformat()
        con.execute("INSERT INTO strategies(domain,name,version,status,config_json,created_at) VALUES(?,?,?,?,?,?)",
                    (domain,cfg["name"],cfg["version"],"champion",json.dumps(cfg),now))
    return get_champion(domain)


def _row(row):
    d=dict(row); d["config"]=json.loads(d.pop("config_json")); return d


def get_champion(domain:str)->dict|None:
    with _connect() as con:
        row=con.execute("SELECT * FROM strategies WHERE domain=? AND status='champion' ORDER BY id DESC LIMIT 1",(domain,)).fetchone()
    return _row(row) if row else None


def register_challenger(domain:str,name:str,version:str,config:dict)->dict:
    now=datetime.now(timezone.utc).isoformat()
    with _connect() as con:
        con.execute("INSERT OR REPLACE INTO strategies(domain,name,version,status,config_json,created_at) VALUES(?,?,?,?,?,?)",
                    (domain,name,version,"challenger",json.dumps(config),now))
        row=con.execute("SELECT * FROM strategies WHERE domain=? AND name=? AND version=?",(domain,name,version)).fetchone()
    return _row(row)


def list_strategies(domain:str|None=None)->list[dict]:
    with _connect() as con:
        rows=con.execute("SELECT * FROM strategies WHERE domain=? ORDER BY id DESC",(domain,)).fetchall() if domain else con.execute("SELECT * FROM strategies ORDER BY id DESC").fetchall()
    return [_row(r) for r in rows]


def evaluate_strategy(job_id:str,domain:str,summary:dict,elapsed_seconds:float)->dict:
    champion=ensure_champion(domain,summary.get("ensemble_weights"))
    with _connect() as con:
        row=con.execute("SELECT * FROM strategies WHERE domain=? AND status='challenger' ORDER BY id DESC LIMIT 1",(domain,)).fetchone()
    challenger=_row(row) if row else None
    metrics={
        "ks_pvalue":(summary.get("best_distribution_metrics") or {}).get("ks_pvalue"),
        "strong_consensus_count":summary.get("strong_consensus_count",0),
        "noise_candidates":summary.get("noise_candidates",0),
        "elapsed_seconds":float(elapsed_seconds),
        "rows":summary.get("rows",0),
    }
    # Safe first governance phase: observe challengers, never auto-promote from one run.
    decision="observe" if challenger else "champion_only"
    reason="Challenger requires repeated evaluation before promotion." if challenger else "No challenger registered for this domain."
    record={"champion":champion,"challenger":challenger,"metrics":metrics,"decision":decision,"reason":reason}
    now=datetime.now(timezone.utc).isoformat()
    with _connect() as con:
        con.execute("INSERT INTO evaluations(job_id,domain,champion_json,challenger_json,metrics_json,decision,reason,created_at) VALUES(?,?,?,?,?,?,?,?)",
                    (job_id,domain,json.dumps(champion),json.dumps(challenger) if challenger else None,json.dumps(metrics),decision,reason,now))
    append_event(job_id,"governance_evaluation",category="governance",stage="governance",payload={"domain":domain,"decision":decision,"reason":reason})
    return record


def promote(domain:str,strategy_id:int)->dict:
    with _connect() as con:
        target=con.execute("SELECT * FROM strategies WHERE id=? AND domain=?",(strategy_id,domain)).fetchone()
        if not target:raise KeyError(strategy_id)
        con.execute("UPDATE strategies SET status='retired' WHERE domain=? AND status='champion'",(domain,))
        con.execute("UPDATE strategies SET status='champion' WHERE id=?",(strategy_id,))
    append_event(None,"strategy_promoted",category="governance",stage="governance",payload={"domain":domain,"strategy_id":strategy_id})
    return get_champion(domain)


def rollback(domain:str)->dict:
    with _connect() as con:
        current=con.execute("SELECT * FROM strategies WHERE domain=? AND status='champion' ORDER BY id DESC LIMIT 1",(domain,)).fetchone()
        previous=con.execute("SELECT * FROM strategies WHERE domain=? AND status='retired' ORDER BY id DESC LIMIT 1",(domain,)).fetchone()
        if not previous:raise ValueError("No retired strategy available for rollback")
        con.execute("UPDATE strategies SET status='challenger' WHERE id=?",(current["id"],))
        con.execute("UPDATE strategies SET status='champion' WHERE id=?",(previous["id"],))
    append_event(None,"strategy_rollback",category="governance",stage="governance",severity="warning",payload={"domain":domain,"restored_strategy_id":previous["id"]})
    return get_champion(domain)


def experiment_history(domain:str|None=None,limit:int=100)->list[dict]:
    limit=min(max(int(limit),1),500)
    with _connect() as con:
        rows=con.execute("SELECT * FROM evaluations WHERE domain=? ORDER BY id DESC LIMIT ?",(domain,limit)).fetchall() if domain else con.execute("SELECT * FROM evaluations ORDER BY id DESC LIMIT ?",(limit,)).fetchall()
    out=[]
    for r in rows:
        d=dict(r)
        for key in ("champion_json","challenger_json","metrics_json"):
            raw=d.pop(key)
            d[key.replace("_json","")]=json.loads(raw) if raw else None
        out.append(d)
    return out


def _signature(summary:dict)->dict:
    rows=max(int(summary.get("rows") or 0),1)
    counts=summary.get("counts") or {}
    spatial=summary.get("spatial_validation") or {}
    temporal=summary.get("temporal_validation") or {}
    return {
        "critical_rate":float(counts.get("99",0))/rows,
        "high_rate":float(counts.get("95",0))/rows,
        "noise_rate":float(summary.get("noise_candidates",0))/rows,
        "consensus_rate":float(summary.get("strong_consensus_count",0))/rows,
        "hotspots":float(spatial.get("hotspots",0)),
        "change_points":float(temporal.get("change_point_count",0)),
        "ks_stat":float((summary.get("best_distribution_metrics") or {}).get("ks_stat") or 0),
    }


def monitor_drift(job_id:str,domain:str,summary:dict,window:int=10)->dict:
    current=_signature(summary)
    with _connect() as con:
        rows=con.execute("SELECT signature_json FROM drift_snapshots WHERE domain=? ORDER BY id DESC LIMIT ?",(domain,window)).fetchall()
    previous=[json.loads(r["signature_json"]) for r in rows]
    if len(previous)<3:
        result={"status":"baseline_building","history":len(previous),"score":0.0,"signals":[],"current":current}
    else:
        signals=[]; scores=[]
        for key,value in current.items():
            hist=[float(x.get(key,0)) for x in previous]
            mean=sum(hist)/len(hist)
            mad=sum(abs(x-mean) for x in hist)/len(hist)
            scale=max(mad,abs(mean)*.10,1e-6)
            z=abs(value-mean)/scale
            scores.append(min(z/5.0,1.0))
            if z>=3.5:signals.append({"metric":key,"current":value,"baseline":mean,"robust_deviation":z})
        score=sum(scores)/len(scores) if scores else 0.0
        result={"status":"drift_detected" if signals else "stable","history":len(previous),"score":round(score,4),"signals":signals,"current":current}
    now=datetime.now(timezone.utc).isoformat()
    with _connect() as con:
        con.execute("INSERT INTO drift_snapshots(job_id,domain,signature_json,drift_json,created_at) VALUES(?,?,?,?,?)",
                    (job_id,domain,json.dumps(current),json.dumps(result),now))
    append_event(job_id,"drift_evaluation",category="governance",stage="drift",severity="warning" if result["status"]=="drift_detected" else "info",payload=result)
    return result


def promotion_gate(domain:str,min_runs:int=5)->dict:
    history=experiment_history(domain,limit=50)
    observed=[x for x in history if x.get("challenger")]
    challenger=list_strategies(domain)
    challenger=next((x for x in challenger if x["status"]=="challenger"),None)
    if not challenger:return {"eligible":False,"reason":"No active challenger","evidence_runs":0}
    same=[x for x in observed if (x.get("challenger") or {}).get("id")==challenger["id"]]
    if len(same)<min_runs:return {"eligible":False,"reason":f"Requires at least {min_runs} repeated evaluations","evidence_runs":len(same),"challenger":challenger}
    failures=sum(1 for x in same if x["decision"] not in {"observe","eligible"})
    eligible=failures==0
    return {"eligible":eligible,"reason":"Repeated evaluation gate passed" if eligible else "One or more evaluations failed governance gates","evidence_runs":len(same),"challenger":challenger}


def governance_dashboard(domain:str)->dict:
    return {
        "domain":domain,
        "champion":get_champion(domain),
        "strategies":list_strategies(domain),
        "promotion_gate":promotion_gate(domain),
        "experiments":experiment_history(domain,20),
    }
