from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH=Path("runtime/apend_detection.sqlite3")


def _connect(path: Path=DB_PATH):
    path.parent.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(path)
    con.row_factory=sqlite3.Row
    con.execute("""
    CREATE TABLE IF NOT EXISTS analyses(
      job_id TEXT PRIMARY KEY,
      created_at TEXT NOT NULL,
      status TEXT NOT NULL,
      input_hash TEXT,
      probability_feature TEXT,
      rows INTEGER,
      crs TEXT,
      params_json TEXT,
      summary_json TEXT,
      outputs_json TEXT,
      elapsed_seconds REAL,
      error TEXT
    )""")
    return con


def save_analysis(job_id:str, status:str, **fields):
    now=datetime.now(timezone.utc).isoformat()
    with _connect() as con:
        existing=con.execute("SELECT job_id FROM analyses WHERE job_id=?",(job_id,)).fetchone()
        payload={
            "input_hash":fields.get("input_hash"),
            "probability_feature":fields.get("probability_feature"),
            "rows":fields.get("rows"),
            "crs":fields.get("crs"),
            "params_json":json.dumps(fields.get("params"),ensure_ascii=False) if fields.get("params") is not None else None,
            "summary_json":json.dumps(fields.get("summary"),ensure_ascii=False) if fields.get("summary") is not None else None,
            "outputs_json":json.dumps(fields.get("outputs"),ensure_ascii=False) if fields.get("outputs") is not None else None,
            "elapsed_seconds":fields.get("elapsed_seconds"),
            "error":fields.get("error"),
        }
        if existing:
            con.execute("""UPDATE analyses SET status=?,input_hash=COALESCE(?,input_hash),
              probability_feature=COALESCE(?,probability_feature),rows=COALESCE(?,rows),
              crs=COALESCE(?,crs),params_json=COALESCE(?,params_json),
              summary_json=COALESCE(?,summary_json),outputs_json=COALESCE(?,outputs_json),
              elapsed_seconds=COALESCE(?,elapsed_seconds),error=? WHERE job_id=?""",
              (status,payload["input_hash"],payload["probability_feature"],payload["rows"],payload["crs"],
               payload["params_json"],payload["summary_json"],payload["outputs_json"],
               payload["elapsed_seconds"],payload["error"],job_id))
        else:
            con.execute("""INSERT INTO analyses(job_id,created_at,status,input_hash,probability_feature,rows,crs,
              params_json,summary_json,outputs_json,elapsed_seconds,error)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
              (job_id,now,status,payload["input_hash"],payload["probability_feature"],payload["rows"],payload["crs"],
               payload["params_json"],payload["summary_json"],payload["outputs_json"],payload["elapsed_seconds"],payload["error"]))


def list_analyses(limit:int=50):
    with _connect() as con:
        rows=con.execute("SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?",(limit,)).fetchall()
    return [dict(r) for r in rows]


def get_analysis(job_id:str):
    with _connect() as con:
        row=con.execute("SELECT * FROM analyses WHERE job_id=?",(job_id,)).fetchone()
    return dict(row) if row else None
