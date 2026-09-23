import json
import sqlite3

import geo_outliers.territorial as territorial
import geo_outliers.incidents as incidents
import geo_outliers.migrations as migrations


def setup_db(tmp_path,monkeypatch):
    db=tmp_path/"territorial.sqlite3"
    monkeypatch.setattr(territorial,"DB_PATH",db)
    monkeypatch.setattr(incidents,"DB_PATH",db)
    monkeypatch.setattr(migrations,"TERRITORIAL_DB",db)
    migrations._apply(db,migrations.TERRITORIAL_MIGRATIONS)
    return db


def seed_incident(db):
    now="2026-01-01T00:00:00+00:00"
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO territorial_objects(object_key,version,name,object_type,geometry_json,metadata_json,status,created_at) VALUES(?,?,?,?,?,?,?,?)",
                    ("zone-1",1,"Zone One","risk_zone",json.dumps({"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}),"{}","active",now))
        cur=con.execute("""INSERT INTO alert_incidents(fingerprint,watchlist_id,object_key,rule_id,first_job_id,last_job_id,severity,status,title,evidence_json,occurrences,opened_at,updated_at)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
          ("1:zone-1:1",1,"zone-1",1,"job-1","job-1","high","open","Spatial escalation",
           json.dumps({"metric":"mean_spatial","value":0.91,"operator":">=","threshold":0.8}),1,now,now))
        return cur.lastrowid


def test_incident_full_management_cycle(tmp_path,monkeypatch):
    db=setup_db(tmp_path,monkeypatch)
    incident_id=seed_incident(db)

    case=incidents.get_incident_case(incident_id)
    assert case["management"]["investigation_status"]=="new"

    case=incidents.update_incident_management(incident_id,owner="Analyst",priority="high",acknowledged=True,investigation_status="investigating")
    assert case["management"]["owner"]=="Analyst"
    assert case["management"]["acknowledged"] is True
    assert case["management"]["investigation_status"]=="investigating"

    note=incidents.add_incident_note(incident_id,"Validated source and territorial baseline.","Analyst")
    assert note["incident_id"]==incident_id
    assert len(incidents.incident_notes(incident_id))==1

    brief=incidents.build_intelligence_brief(incident_id)
    assert brief["where"]=="Zone One"
    assert brief["evidence"]["rule"]["metric"]=="mean_spatial"
    assert brief["methodology_note"]

    resolved=incidents.update_incident_management(incident_id,investigation_status="resolved",resolution_code="reviewed",resolution_note="No further action.")
    assert resolved["management"]["resolution_code"]=="reviewed"
    assert len(resolved["timeline"])>=3


def test_migrations_are_idempotent(tmp_path,monkeypatch):
    db=tmp_path/"territorial.sqlite3"
    monkeypatch.setattr(migrations,"TERRITORIAL_DB",db)
    first=migrations._apply(db,migrations.TERRITORIAL_MIGRATIONS)
    second=migrations._apply(db,migrations.TERRITORIAL_MIGRATIONS)
    assert len(first["applied"])==3
    assert second["applied"]==[]
    assert second["current_version"]==3
