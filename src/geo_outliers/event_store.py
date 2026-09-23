from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH=Path("runtime/meridian_events.sqlite3")


def _connect():
    DB_PATH.parent.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(DB_PATH)
    con.row_factory=sqlite3.Row
    con.execute("""
    CREATE TABLE IF NOT EXISTS events(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      job_id TEXT,
      created_at TEXT NOT NULL,
      category TEXT NOT NULL,
      event_type TEXT NOT NULL,
      stage TEXT,
      severity TEXT NOT NULL,
      payload_json TEXT
    )""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_events_job ON events(job_id,id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type,created_at)")
    return con


def append_event(job_id:str|None,event_type:str,category:str="orchestrator",stage:str|None=None,
                 severity:str="info",payload:dict|None=None)->int:
    now=datetime.now(timezone.utc).isoformat()
    with _connect() as con:
        cur=con.execute(
            "INSERT INTO events(job_id,created_at,category,event_type,stage,severity,payload_json) VALUES(?,?,?,?,?,?,?)",
            (job_id,now,category,event_type,stage,severity,json.dumps(payload or {},ensure_ascii=False))
        )
        return int(cur.lastrowid)


def list_events(job_id:str|None=None,limit:int=200)->list[dict]:
    limit=min(max(int(limit),1),1000)
    with _connect() as con:
        if job_id:
            rows=con.execute("SELECT * FROM events WHERE job_id=? ORDER BY id DESC LIMIT ?",(job_id,limit)).fetchall()
        else:
            rows=con.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?",(limit,)).fetchall()
    out=[]
    for row in rows:
        d=dict(row)
        try:d["payload"]=json.loads(d.pop("payload_json") or "{}")
        except Exception:d["payload"]={}
        out.append(d)
    return out


def event_stats()->dict:
    with _connect() as con:
        total=con.execute("SELECT COUNT(*) n FROM events").fetchone()["n"]
        errors=con.execute("SELECT COUNT(*) n FROM events WHERE severity IN ('error','critical')").fetchone()["n"]
        jobs=con.execute("SELECT COUNT(DISTINCT job_id) n FROM events WHERE job_id IS NOT NULL").fetchone()["n"]
    return {"events":int(total),"errors":int(errors),"jobs":int(jobs)}
