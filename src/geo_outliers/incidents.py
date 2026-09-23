from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .territorial import DB_PATH, list_incidents, list_territorial_objects, monitoring_history


def _connect():
    DB_PATH.parent.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(DB_PATH); con.row_factory=sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS incident_management(
      incident_id INTEGER PRIMARY KEY,
      owner TEXT,
      priority TEXT NOT NULL DEFAULT 'normal',
      acknowledged INTEGER NOT NULL DEFAULT 0,
      acknowledged_at TEXT,
      investigation_status TEXT NOT NULL DEFAULT 'new',
      resolution_code TEXT,
      resolution_note TEXT,
      updated_at TEXT NOT NULL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS incident_notes(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      incident_id INTEGER NOT NULL,
      author TEXT,
      note TEXT NOT NULL,
      created_at TEXT NOT NULL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS incident_timeline(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      incident_id INTEGER NOT NULL,
      event_type TEXT NOT NULL,
      payload_json TEXT,
      created_at TEXT NOT NULL
    )""")
    return con


def _now(): return datetime.now(timezone.utc).isoformat()


def _incident(incident_id:int):
    matches=[x for x in list_incidents(None,1000) if int(x["id"])==int(incident_id)]
    if not matches: raise KeyError(incident_id)
    return matches[0]


def _timeline(incident_id:int,event_type:str,payload:dict|None=None):
    with _connect() as con:
        con.execute("INSERT INTO incident_timeline(incident_id,event_type,payload_json,created_at) VALUES(?,?,?,?)",
                    (incident_id,event_type,json.dumps(payload or {},ensure_ascii=False),_now()))


def ensure_management(incident_id:int)->dict:
    _incident(incident_id)
    now=_now()
    with _connect() as con:
        con.execute("""INSERT OR IGNORE INTO incident_management(
          incident_id,priority,acknowledged,investigation_status,updated_at
        ) VALUES(?, 'normal', 0, 'new', ?)""",(incident_id,now))
        row=con.execute("SELECT * FROM incident_management WHERE incident_id=?",(incident_id,)).fetchone()
    d=dict(row); d["acknowledged"]=bool(d["acknowledged"]); return d


def update_incident_management(incident_id:int,owner:str|None=None,priority:str|None=None,
                               acknowledged:bool|None=None,investigation_status:str|None=None,
                               resolution_code:str|None=None,resolution_note:str|None=None)->dict:
    current=ensure_management(incident_id)
    allowed_priority={"low","normal","high","urgent"}
    allowed_status={"new","acknowledged","investigating","monitoring","resolved","closed"}
    if priority is not None and priority not in allowed_priority: raise ValueError("Unsupported priority")
    if investigation_status is not None and investigation_status not in allowed_status: raise ValueError("Unsupported investigation status")
    values={
        "owner":current.get("owner") if owner is None else owner,
        "priority":current["priority"] if priority is None else priority,
        "acknowledged":current["acknowledged"] if acknowledged is None else bool(acknowledged),
        "acknowledged_at":current.get("acknowledged_at"),
        "investigation_status":current["investigation_status"] if investigation_status is None else investigation_status,
        "resolution_code":current.get("resolution_code") if resolution_code is None else resolution_code,
        "resolution_note":current.get("resolution_note") if resolution_note is None else resolution_note,
    }
    if values["acknowledged"] and not values["acknowledged_at"]: values["acknowledged_at"]=_now()
    now=_now()
    with _connect() as con:
        con.execute("""UPDATE incident_management SET owner=?,priority=?,acknowledged=?,acknowledged_at=?,
          investigation_status=?,resolution_code=?,resolution_note=?,updated_at=? WHERE incident_id=?""",
          (values["owner"],values["priority"],int(values["acknowledged"]),values["acknowledged_at"],
           values["investigation_status"],values["resolution_code"],values["resolution_note"],now,incident_id))
    _timeline(incident_id,"management_updated",values)
    return get_incident_case(incident_id)


def add_incident_note(incident_id:int,note:str,author:str|None=None)->dict:
    if not note.strip(): raise ValueError("Note cannot be empty")
    ensure_management(incident_id)
    now=_now()
    with _connect() as con:
        cur=con.execute("INSERT INTO incident_notes(incident_id,author,note,created_at) VALUES(?,?,?,?)",
                        (incident_id,author,note.strip(),now))
    _timeline(incident_id,"note_added",{"note_id":int(cur.lastrowid),"author":author})
    return {"id":int(cur.lastrowid),"incident_id":incident_id,"author":author,"note":note.strip(),"created_at":now}


def incident_notes(incident_id:int)->list[dict]:
    with _connect() as con:
        rows=con.execute("SELECT * FROM incident_notes WHERE incident_id=? ORDER BY id DESC",(incident_id,)).fetchall()
    return [dict(r) for r in rows]


def incident_timeline(incident_id:int)->list[dict]:
    with _connect() as con:
        rows=con.execute("SELECT * FROM incident_timeline WHERE incident_id=? ORDER BY id DESC",(incident_id,)).fetchall()
    out=[]
    for r in rows:
        d=dict(r)
        try:d["payload"]=json.loads(d.pop("payload_json") or "{}")
        except Exception:d["payload"]={}
        out.append(d)
    return out


def get_incident_case(incident_id:int)->dict:
    incident=_incident(incident_id)
    management=ensure_management(incident_id)
    obj=next((x for x in list_territorial_objects(False,500) if x["object_key"]==incident["object_key"]),None)
    history=monitoring_history(incident["object_key"],20)
    return {"incident":incident,"management":management,"object":obj,"monitoring_history":history,
            "notes":incident_notes(incident_id),"timeline":incident_timeline(incident_id)}


def build_intelligence_brief(incident_id:int)->dict:
    case=get_incident_case(incident_id)
    inc=case["incident"]; mg=case["management"]; obj=case["object"] or {}
    history=case["monitoring_history"]
    latest=history[0] if history else {}
    change=latest.get("change") or {}
    evidence=inc.get("evidence") or {}

    where=obj.get("name") or inc.get("object_key")
    when=inc.get("updated_at") or inc.get("opened_at")
    what=f"{evidence.get('metric','territorial rule')} reached {evidence.get('value','n/a')}"
    threshold=evidence.get("threshold")
    if threshold is not None: what+=f" against threshold {evidence.get('operator','>=')} {threshold}"
    signals=change.get("signals") or []
    why=[f"{s.get('metric')} changed {s.get('direction')} with robust deviation {s.get('robust_deviation'):.2f}" for s in signals if isinstance(s.get("robust_deviation"),(int,float))]
    if not why: why=[f"Operational rule '{inc.get('title')}' is active with {inc.get('occurrences',1)} occurrence(s)."]

    next_steps=[]
    if inc.get("severity") in {"critical","high"}: next_steps.append("Review the latest territorial evidence and source quality.")
    if signals: next_steps.append("Compare current values against the territory baseline and preceding monitoring runs.")
    if not mg.get("acknowledged"): next_steps.append("Acknowledge and assign an owner before operational follow-up.")
    next_steps.append("Do not infer causality from the alert alone; validate with domain-specific evidence.")

    return {
        "incident_id":incident_id,
        "title":inc.get("title"),
        "severity":inc.get("severity"),
        "status":inc.get("status"),
        "priority":mg.get("priority"),
        "owner":mg.get("owner"),
        "where":where,
        "when":when,
        "what_changed":what,
        "why_flagged":why,
        "evidence":{"rule":evidence,"territorial_change":change,"latest_summary":latest.get("summary")},
        "next_steps":next_steps,
        "methodology_note":"This brief summarizes Meridian evidence and operational rules. It does not establish causality or replace domain review.",
        "generated_at":_now(),
    }


def operations_cases(status:str|None=None,limit:int=200)->list[dict]:
    incidents=list_incidents(status,limit)
    out=[]
    for inc in incidents:
        try:mg=ensure_management(int(inc["id"]))
        except Exception:mg={}
        out.append({**inc,"management":mg})
    return out
