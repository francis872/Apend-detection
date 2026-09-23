import pytest
from geo_outliers import territorial_fusion as tf

def test_fusion_preserves_groups_and_namespaces(tmp_path,monkeypatch):
    monkeypatch.setattr(tf,"DB",tmp_path/"fusion.sqlite3")
    import sqlite3
    with sqlite3.connect(tf.DB) as c:
        c.executescript("""CREATE TABLE territorial_feature_vectors(id INTEGER PRIMARY KEY AUTOINCREMENT,territory_id TEXT,timestamp TEXT,vector_json TEXT,groups_json TEXT,source_refs_json TEXT,lineage_json TEXT,created_at TEXT,UNIQUE(territory_id,timestamp));""")
    out=tf.fuse_features({"territory_id":"T1","timestamp":"2026-01-01","social_features":{"population":100},"satellite_features":{"NDVI":.6},"source_refs":["ds1"]})
    assert out["vector"]["social.population"]==100
    assert out["vector"]["satellite.NDVI"]==pytest.approx(.6)
    assert out["lineage"]["causal_claims"] is False

def test_temporal_profile_detects_robust_change(tmp_path,monkeypatch):
    monkeypatch.setattr(tf,"DB",tmp_path/"fusion.sqlite3")
    import sqlite3
    with sqlite3.connect(tf.DB) as c:
        c.executescript("""CREATE TABLE territorial_feature_vectors(id INTEGER PRIMARY KEY AUTOINCREMENT,territory_id TEXT,timestamp TEXT,vector_json TEXT,groups_json TEXT,source_refs_json TEXT,lineage_json TEXT,created_at TEXT,UNIQUE(territory_id,timestamp));""")
    for i,v in enumerate([.50,.51,.49,.50,.90]):
        tf.fuse_features({"territory_id":"T1","timestamp":f"2026-01-0{i+1}","satellite_features":{"NDVI":v}})
    p=tf.temporal_profile("T1")
    assert p["change"]["status"]=="changed"
    assert "satellite.NDVI" in p["change"]["signals"]
    assert p["change"]["score_is_not_probability"] is True

def test_land_use_divergence_is_not_conflict(tmp_path,monkeypatch):
    monkeypatch.setattr(tf,"DB",tmp_path/"fusion.sqlite3")
    import sqlite3
    with sqlite3.connect(tf.DB) as c:
        c.executescript("""CREATE TABLE land_use_divergence(id INTEGER PRIMARY KEY AUTOINCREMENT,territory_id TEXT,timestamp TEXT,historical_use TEXT,observed_use TEXT,planned_use TEXT,satellite_classification TEXT,divergence_score REAL,divergence_level TEXT,evidence_json TEXT,sources_json TEXT,created_at TEXT);""")
    out=tf.land_use_divergence({"territory_id":"T1","timestamp":"2026-01-01","historical_use":"agriculture","observed_use":"mining","planned_use":"agriculture","satellite_classification":"mining"})
    assert out["divergence_level"] in {"MODERATE DIVERGENCE","HIGH DIVERGENCE"}
    assert "not conflict" in " ".join(out["evidence"]["limitations"]).lower()
