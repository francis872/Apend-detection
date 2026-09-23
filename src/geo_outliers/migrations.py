from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

CORE_DB=Path("runtime/apend_detection.sqlite3")
TERRITORIAL_DB=Path("runtime/meridian_territorial.sqlite3")
EVENT_DB=Path("runtime/meridian_events.sqlite3")
GOVERNANCE_DB=Path("runtime/meridian_governance.sqlite3")


def _apply(db_path:Path, migrations:list[tuple[int,str,str]]) -> dict:
    db_path.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS schema_migrations(
          version INTEGER PRIMARY KEY,
          name TEXT NOT NULL,
          applied_at TEXT NOT NULL
        )""")
        applied={int(r[0]) for r in con.execute("SELECT version FROM schema_migrations").fetchall()}
        ran=[]
        for version,name,sql in migrations:
            if version in applied: continue
            con.executescript(sql)
            con.execute("INSERT INTO schema_migrations(version,name,applied_at) VALUES(?,?,?)",
                        (version,name,datetime.now(timezone.utc).isoformat()))
            ran.append({"version":version,"name":name})
    return {"database":str(db_path),"applied":ran,"current_version":max([m[0] for m in migrations],default=0)}


CORE_MIGRATIONS=[
    (1,"analyses_base","""
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
    );
    """),
]

TERRITORIAL_MIGRATIONS=[
    (1,"territorial_base","""
    CREATE TABLE IF NOT EXISTS layers(
      id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, job_id TEXT,
      geometry_json TEXT NOT NULL, metadata_json TEXT, created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS territorial_objects(
      id INTEGER PRIMARY KEY AUTOINCREMENT, object_key TEXT NOT NULL, version INTEGER NOT NULL,
      name TEXT NOT NULL, object_type TEXT NOT NULL, job_id TEXT, geometry_json TEXT NOT NULL,
      metadata_json TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(object_key,version));
    CREATE TABLE IF NOT EXISTS territorial_alerts(
      id INTEGER PRIMARY KEY AUTOINCREMENT, object_key TEXT NOT NULL, job_id TEXT NOT NULL,
      severity TEXT NOT NULL, alert_type TEXT NOT NULL, title TEXT NOT NULL,
      evidence_json TEXT NOT NULL, created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS territorial_monitoring(
      id INTEGER PRIMARY KEY AUTOINCREMENT, object_key TEXT NOT NULL, object_version INTEGER NOT NULL,
      job_id TEXT NOT NULL, summary_json TEXT NOT NULL, baseline_json TEXT, change_json TEXT,
      status TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(object_key,job_id));
    CREATE INDEX IF NOT EXISTS idx_monitoring_object ON territorial_monitoring(object_key,id);
    """),
    (2,"watchlists_alerting","""
    CREATE TABLE IF NOT EXISTS watchlists(
      id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, domain TEXT,
      enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS watchlist_objects(
      watchlist_id INTEGER NOT NULL, object_key TEXT NOT NULL, created_at TEXT NOT NULL,
      UNIQUE(watchlist_id,object_key));
    CREATE TABLE IF NOT EXISTS alert_rules(
      id INTEGER PRIMARY KEY AUTOINCREMENT, watchlist_id INTEGER NOT NULL, name TEXT NOT NULL,
      metric TEXT NOT NULL, operator TEXT NOT NULL, threshold REAL NOT NULL,
      severity TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS alert_incidents(
      id INTEGER PRIMARY KEY AUTOINCREMENT, fingerprint TEXT NOT NULL UNIQUE, watchlist_id INTEGER,
      object_key TEXT NOT NULL, rule_id INTEGER, first_job_id TEXT NOT NULL, last_job_id TEXT NOT NULL,
      severity TEXT NOT NULL, status TEXT NOT NULL, title TEXT NOT NULL, evidence_json TEXT NOT NULL,
      occurrences INTEGER NOT NULL DEFAULT 1, opened_at TEXT NOT NULL, updated_at TEXT NOT NULL,
      resolved_at TEXT);
    """),
    (3,"incident_management","""
    CREATE TABLE IF NOT EXISTS incident_management(
      incident_id INTEGER PRIMARY KEY, owner TEXT, priority TEXT NOT NULL DEFAULT 'normal',
      acknowledged INTEGER NOT NULL DEFAULT 0, acknowledged_at TEXT,
      investigation_status TEXT NOT NULL DEFAULT 'new', resolution_code TEXT,
      resolution_note TEXT, updated_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS incident_notes(
      id INTEGER PRIMARY KEY AUTOINCREMENT, incident_id INTEGER NOT NULL, author TEXT,
      note TEXT NOT NULL, created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS incident_timeline(
      id INTEGER PRIMARY KEY AUTOINCREMENT, incident_id INTEGER NOT NULL,
      event_type TEXT NOT NULL, payload_json TEXT, created_at TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS idx_incident_notes_incident ON incident_notes(incident_id,id);
    CREATE INDEX IF NOT EXISTS idx_incident_timeline_incident ON incident_timeline(incident_id,id);
    CREATE INDEX IF NOT EXISTS idx_alert_incidents_status ON alert_incidents(status,updated_at);
    """),
    (4,"territorial_network_socio_ecological","""
    CREATE TABLE IF NOT EXISTS geodata_datasets(
      dataset_id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      source TEXT,
      source_url TEXT,
      license TEXT,
      description TEXT,
      geometry_type TEXT,
      crs TEXT,
      temporal_start TEXT,
      temporal_end TEXT,
      resolution TEXT,
      update_frequency TEXT,
      created_at TEXT NOT NULL,
      metadata_json TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS data_lineage(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      dataset_id TEXT,
      stage TEXT NOT NULL,
      transformation TEXT,
      algorithm_version TEXT,
      parent_dataset_id TEXT,
      job_id TEXT,
      created_at TEXT NOT NULL,
      metadata_json TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS spatial_events(
      id TEXT PRIMARY KEY,
      event_type TEXT NOT NULL,
      latitude REAL,
      longitude REAL,
      geometry_json TEXT NOT NULL,
      date TEXT,
      source TEXT,
      source_id TEXT,
      confidence REAL,
      administrative_unit TEXT,
      metadata_json TEXT NOT NULL,
      dataset_id TEXT,
      created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS territorial_snapshots(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      territory_id TEXT NOT NULL,
      timestamp TEXT NOT NULL,
      social_json TEXT NOT NULL,
      economic_json TEXT NOT NULL,
      ecological_json TEXT NOT NULL,
      satellite_json TEXT NOT NULL,
      land_use_json TEXT NOT NULL,
      provenance_json TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_snapshot_territory_time ON territorial_snapshots(territory_id,timestamp);
    CREATE TABLE IF NOT EXISTS land_use_snapshots(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      territory_id TEXT NOT NULL,
      observed_use TEXT,
      planned_use TEXT,
      historical_use TEXT,
      satellite_classification TEXT,
      timestamp TEXT NOT NULL,
      sources_json TEXT NOT NULL,
      divergence_level TEXT,
      divergence_score REAL,
      created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS corridor_results(
      corridor_id TEXT PRIMARY KEY,
      geometry_json TEXT NOT NULL,
      variables_json TEXT NOT NULL,
      period_json TEXT NOT NULL,
      supporting_events_json TEXT NOT NULL,
      confidence REAL NOT NULL,
      method TEXT NOT NULL,
      data_sources_json TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_lineage_dataset ON data_lineage(dataset_id,id);
    CREATE INDEX IF NOT EXISTS idx_spatial_events_date ON spatial_events(date);
    """),
]


def migrate_all() -> dict:
    results=[
        _apply(CORE_DB,CORE_MIGRATIONS),
        _apply(TERRITORIAL_DB,TERRITORIAL_MIGRATIONS),
    ]
    return {"status":"ok","databases":results}


def migration_status() -> dict:
    out={}
    for name,path in {"core":CORE_DB,"territorial":TERRITORIAL_DB}.items():
        if not path.exists():
            out[name]={"exists":False,"versions":[]}; continue
        with sqlite3.connect(path) as con:
            try: rows=con.execute("SELECT version,name,applied_at FROM schema_migrations ORDER BY version").fetchall()
            except sqlite3.OperationalError: rows=[]
        out[name]={"exists":True,"versions":[{"version":r[0],"name":r[1],"applied_at":r[2]} for r in rows]}
    return out
