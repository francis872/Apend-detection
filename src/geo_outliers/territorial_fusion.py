from __future__ import annotations

import json, math, sqlite3
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

DB=Path("runtime/meridian_territorial.sqlite3")
GROUPS=("social_features","economic_features","ecological_features","satellite_features","land_use_features","infrastructure_features","network_features","event_features")

def _now():return datetime.now(timezone.utc).isoformat()
def _con():c=sqlite3.connect(DB);c.row_factory=sqlite3.Row;return c
def _num(v):
    try:
        x=float(v);return x if math.isfinite(x) else None
    except (TypeError,ValueError):return None

def fuse_features(payload:dict)->dict:
    territory=str(payload.get("territory_id") or "").strip();ts=str(payload.get("timestamp") or "").strip()
    if not territory or not ts:raise ValueError("territory_id and timestamp are required")
    groups={g:dict(payload.get(g) or {}) for g in GROUPS}
    vector={};collisions=[]
    for group,features in groups.items():
        prefix=group.replace("_features","")
        for key,value in features.items():
            val=_num(value)
            if val is None:continue
            canonical=f"{prefix}.{key}"
            if canonical in vector:collisions.append(canonical)
            vector[canonical]=val
    refs=list(payload.get("source_refs") or [])
    lineage={"pipeline":["SOURCE","INGESTION","NORMALIZATION","SPATIAL_PROCESSING","FUSION","TERRITORIAL_VECTOR"],
             "algorithm_version":"territorial-fusion-1.0","causal_claims":False,"collisions":collisions}
    with _con() as c:
        c.execute("""INSERT INTO territorial_feature_vectors(territory_id,timestamp,vector_json,groups_json,source_refs_json,lineage_json,created_at)
        VALUES(?,?,?,?,?,?,?) ON CONFLICT(territory_id,timestamp) DO UPDATE SET vector_json=excluded.vector_json,groups_json=excluded.groups_json,
        source_refs_json=excluded.source_refs_json,lineage_json=excluded.lineage_json,created_at=excluded.created_at""",
        (territory,ts,json.dumps(vector),json.dumps(groups),json.dumps(refs),json.dumps(lineage),_now()))
    return {"territory_id":territory,"timestamp":ts,"vector":vector,"groups":groups,"source_refs":refs,"lineage":lineage}

def feature_history(territory_id:str)->list[dict]:
    with _con() as c:rows=c.execute("SELECT * FROM territorial_feature_vectors WHERE territory_id=? ORDER BY timestamp",(territory_id,)).fetchall()
    return [{"id":r["id"],"territory_id":r["territory_id"],"timestamp":r["timestamp"],"vector":json.loads(r["vector_json"]),
             "groups":json.loads(r["groups_json"]),"source_refs":json.loads(r["source_refs_json"]),"lineage":json.loads(r["lineage_json"])} for r in rows]

def temporal_profile(territory_id:str,window:int=8)->dict:
    hist=feature_history(territory_id)
    if not hist:return {"territory_id":territory_id,"snapshots":[],"baseline":None,"change":None}
    current=hist[-1];prior=hist[:-1][-max(1,int(window)):]
    keys=sorted(current["vector"])
    baseline={};mad={};delta={};robust_z={}
    for k in keys:
        vals=np.array([x["vector"].get(k,np.nan) for x in prior],float);vals=vals[np.isfinite(vals)]
        if not len(vals):continue
        med=float(np.median(vals));m=float(np.median(np.abs(vals-med)));baseline[k]=med;mad[k]=m
        d=current["vector"][k]-med;delta[k]=float(d);scale=max(1.4826*m,abs(med)*.10,1e-9);robust_z[k]=float(d/scale)
    changed={k:z for k,z in robust_z.items() if abs(z)>=3.5}
    return {"territory_id":territory_id,"snapshots":hist,"baseline":{"method":"rolling median/MAD","window":window,"values":baseline,"mad":mad} if prior else None,
            "change":{"timestamp":current["timestamp"],"delta":delta,"robust_z":robust_z,"signals":changed,
                      "status":"changed" if changed else ("stable" if prior else "baseline_building"),
                      "threshold":3.5,"score_is_not_probability":True}}

def _norm(v):
    if v is None:return None
    return " ".join(str(v).strip().lower().replace("_"," ").replace("-"," ").split())

def land_use_divergence(payload:dict,persist:bool=True)->dict:
    territory=str(payload.get("territory_id") or "").strip();ts=str(payload.get("timestamp") or "").strip()
    if not territory or not ts:raise ValueError("territory_id and timestamp are required")
    fields={k:_norm(payload.get(k)) for k in ("historical_use","observed_use","planned_use","satellite_classification")}
    present=[v for v in fields.values() if v]
    if len(present)<2:raise ValueError("At least two land-use observations are required")
    pairs=0;different=0
    for i in range(len(present)):
        for j in range(i+1,len(present)):
            pairs+=1;different+=int(present[i]!=present[j])
    score=float(different/pairs) if pairs else 0
    level="LOW DIVERGENCE" if score<.34 else ("MODERATE DIVERGENCE" if score<.67 else "HIGH DIVERGENCE")
    evidence={"comparisons":pairs,"different_pairs":different,"method":"categorical pairwise disagreement",
              "limitations":["Semantic category harmonization is required for heterogeneous taxonomies","Divergence is not conflict","No causal interpretation"]}
    out={"territory_id":territory,"timestamp":ts,**fields,"divergence_score":score,"divergence_level":level,
         "evidence":evidence,"sources":payload.get("sources") or []}
    if persist:
        with _con() as c:c.execute("""INSERT INTO land_use_divergence(territory_id,timestamp,historical_use,observed_use,planned_use,satellite_classification,
        divergence_score,divergence_level,evidence_json,sources_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (territory,ts,fields["historical_use"],fields["observed_use"],fields["planned_use"],fields["satellite_classification"],score,level,json.dumps(evidence),json.dumps(out["sources"]),_now()))
    return out
