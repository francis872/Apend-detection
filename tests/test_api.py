import io
import zipfile

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from geo_outliers.api import app
from geo_outliers.validation import ablation_sensitivity


client=TestClient(app)


def test_health_endpoint():
    r=client.get("/health")
    assert r.status_code==200
    body=r.json()
    assert body["service"]=="Meridian"
    assert body["version"]=="3.2.0"
    assert "migrations" in body


def test_migration_status_endpoint():
    r=client.get("/v1/system/migrations")
    assert r.status_code==200
    body=r.json()
    assert "core" in body
    assert "territorial" in body
    assert body["territorial"]["exists"] is True


def test_operations_center_contract():
    r=client.get("/v1/operations")
    assert r.status_code==200
    body=r.json()
    assert set(["watchlists","rules","active_incidents","counts"]).issubset(body)
    assert set(["active","critical","high","resolved"]).issubset(body["counts"])


def test_missing_incident_is_404():
    r=client.get("/v1/incidents/999999999")
    assert r.status_code==404


def test_incident_validation_is_404_before_payload_processing():
    r=client.patch("/v1/incidents/999999999",json={"priority":"impossible"})
    assert r.status_code==404


def test_ablation_sensitivity():
    rng=np.random.default_rng(42)
    c=pd.DataFrame({"a":rng.random(200),"b":rng.random(200),"c":rng.random(200)})
    out=ablation_sensitivity(c,{"a":.4,"b":.3,"c":.3})
    assert set(out)=={"a","b","c"}
    assert all(0<=v["jaccard_vs_full_99"]<=1 for v in out.values())


def test_inspect_rejects_incomplete_shapefile_zip():
    mem=io.BytesIO()
    with zipfile.ZipFile(mem,"w") as z:
        z.writestr("sample.shp",b"not-a-real-shapefile")
    mem.seek(0)
    r=client.post("/inspect",files=[("files",("sample.zip",mem.getvalue(),"application/zip"))])
    assert r.status_code==422
    assert "missing" in r.json()["detail"].lower()
