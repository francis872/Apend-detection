from __future__ import annotations

import argparse
import sys
from pathlib import Path


def quick_test() -> None:
    from geo_outliers.api import app, health
    payload=health()
    assert payload["status"]=="ok"
    assert app.title=="Apend Detection API"
    ui=Path("ui/index.html")
    if not ui.exists():
        raise RuntimeError("ui/index.html is missing")
    route_paths={getattr(r,"path",None) for r in app.routes}
    for required in ("/health","/inspect","/analyze","/jobs/{job_id}/progress","/jobs/{job_id}/result"):
        if required not in route_paths:
            raise RuntimeError(f"Missing API route: {required}")
    print(f"[OK] API import: {app.title} {app.version}")
    print("[OK] Health function")
    print("[OK] UI present")
    print("[OK] Required routes present")


def full_test() -> None:
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point
    from geo_outliers.detector import detect_outliers

    rng=np.random.default_rng(42)
    n=100
    lat=6.2+rng.normal(0,.04,n)
    lon=-75.57+rng.normal(0,.04,n)
    gdf=gpd.GeoDataFrame({
        "BRIGHTNESS":rng.normal(330,10,n),
        "BRIGHT_T31":rng.normal(295,5,n),
        "FRP":rng.lognormal(2,.6,n),
        "SCAN":rng.uniform(.4,1.5,n),
        "TRACK":rng.uniform(.4,1.5,n),
        "ACQ_DATE":pd.date_range("2026-01-01",periods=n,freq="h").date,
        "ACQ_TIME":[f"{i%24:02d}00" for i in range(n)],
    },geometry=[Point(x,y) for x,y in zip(lon,lat)],crs="EPSG:4326")
    scored,summary,fits=detect_outliers(
        gdf,quadrature_order=16,spd_k=10,procrustes_k=8,max_geometry_rows=500
    )
    assert len(scored)==n
    assert len(fits)>0
    assert "outlier_score" in scored.columns
    assert "pdf_derivative_2" in scored.columns
    assert set(summary["counts"])=={"90","95","99"}
    print(f"[OK] Synthetic full analysis: {n} rows")
    print(f"[OK] Comparative fit: {summary['best_distribution']}")
    print(f"[OK] Outlier counts: {summary['counts']}")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--full",action="store_true")
    args=parser.parse_args()
    print("Apend Detection self-test")
    print("Python:",sys.version.split()[0])
    quick_test()
    if args.full:
        full_test()
    print("[OK] SELF-TEST PASSED")


if __name__=="__main__":
    main()
