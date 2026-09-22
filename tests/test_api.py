import io
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from fastapi.testclient import TestClient
from geo_outliers.api import app
from geo_outliers.validation import ablation_sensitivity


def test_health_endpoint():
    r=TestClient(app).get("/health")
    assert r.status_code==200
    assert r.json()["version"]=="1.1.0"


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
    r=TestClient(app).post("/inspect",files=[("files",("sample.zip",mem.getvalue(),"application/zip"))])
    assert r.status_code==422
    assert "missing" in r.json()["detail"].lower()
