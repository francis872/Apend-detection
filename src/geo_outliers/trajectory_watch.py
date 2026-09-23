from __future__ import annotations
import json, sqlite3, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .spatial_dynamics import trajectory

DB=Path("runtime/meridian_territorial.sqlite3")
def _now():return datetime.now(timezone.utc).isoformat()
def _con():c=sqlite3.connect(DB);c.row_factory=sqlite3.Row;return c

def assess_trajectory(snapshots,baseline_window=4):
    r=trajectory(snapshots,baseline_window)
    dep=np.asarray(r["departure_from_baseline"],float);spd=np.asarray(r["speed"],float)
    prior_dep=dep[:-1];prior_spd=spd[:-1]
    def robust_level(current,prior):
        if len(prior)<2:return {"z":0.0,"level":"BASELINE_BUILDING"}
        med=float(np.median(prior));mad=float(np.median(np.abs(prior-med)));scale=max(1.4826*mad,abs(med)*.1,.25)
        z=float((current-med)/scale)
        return {"z":z,"level":"HIGH" if z>=3.5 else ("ELEVATED" if z>=2 else "NORMAL")}
    departure=robust_level(float(dep[-1]),prior_dep);speed=robust_level(float(spd[-1]),prior_spd)
    acceleration=float(r["acceleration"][-1]) if r["acceleration"] else 0.0
    evidence=[]
    if departure["level"]!="NORMAL" and departure["level"]!="BASELINE_BUILDING":evidence.append("baseline_departure")
    if speed["level"]!="NORMAL" and speed["level"]!="BASELINE_BUILDING":evidence.append("trajectory_speed")
    if r["change_points"]:evidence.append("robust_change_point")
    severity="HIGH" if (departure["level"]=="HIGH" and speed["level"]=="HIGH") else ("ELEVATED" if evidence else "NORMAL")
    return {"territory_id":r["territory_id"],"timestamp":r["timestamps"][-1],"status":severity,
      "signals":{"departure":departure,"speed":speed,"acceleration":acceleration,"change_points":r["change_points"]},
      "evidence":evidence,"trajectory":r,
      "methodology":{"type":"trajectory early-warning evidence","causal_claims":False,"forecast":False,"risk_probability":False,
      "note":"Signals indicate unusual territorial dynamics relative to observed history; they do not predict an event or establish cause."}}

def persist_assessment(result):
    aid=str(uuid.uuid4())
    with _con() as c:c.execute("""INSERT INTO trajectory_assessments(id,territory_id,timestamp,status,evidence_json,methodology_json,created_at)
      VALUES(?,?,?,?,?,?,?)""",(aid,result["territory_id"],result["timestamp"],result["status"],json.dumps(result["signals"]),json.dumps(result["methodology"]),_now()))
    return {**result,"assessment_id":aid}

def assessment_history(territory_id,limit=100):
    with _con() as c:rows=c.execute("SELECT * FROM trajectory_assessments WHERE territory_id=? ORDER BY created_at DESC LIMIT ?",(territory_id,int(limit))).fetchall()
    return [{"assessment_id":r["id"],"territory_id":r["territory_id"],"timestamp":r["timestamp"],"status":r["status"],
      "signals":json.loads(r["evidence_json"]),"methodology":json.loads(r["methodology_json"]),"created_at":r["created_at"]} for r in rows]

def watch_territory(snapshots,baseline_window=4,persist=True):
    out=assess_trajectory(snapshots,baseline_window)
    return persist_assessment(out) if persist else out
