import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from geo_outliers.data_quality import assess_data_quality, quality_gate


def test_clean_spatial_data_passes_quality_gate():
    gdf=gpd.GeoDataFrame({"value":[1,2,3]},geometry=[Point(-75,6),Point(-75.1,6.1),Point(-75.2,6.2)],crs="EPSG:4326")
    report=assess_data_quality(gdf)
    assert report["ready"] is True
    assert report["quality_score"]>=.95
    assert quality_gate(report)["ok"] is True


def test_duplicate_geometry_is_reported():
    p=Point(-75,6)
    gdf=gpd.GeoDataFrame({"value":[1,2,3]},geometry=[p,p,Point(-75.2,6.2)],crs="EPSG:4326")
    report=assess_data_quality(gdf)
    assert report["metrics"]["duplicate_geometry"]==1
    assert any(x["code"]=="duplicate_geometry" for x in report["issues"])


def test_invalid_geometry_is_reported():
    bow=Polygon([(0,0),(1,1),(1,0),(0,1),(0,0)])
    gdf=gpd.GeoDataFrame({"value":[1]},geometry=[bow],crs="EPSG:4326")
    report=assess_data_quality(gdf)
    assert report["metrics"]["invalid_geometry"]==1
    assert report["quality_score"]<1


def test_quality_score_is_not_probability():
    gdf=gpd.GeoDataFrame({"value":[1,2]},geometry=[Point(0,0),Point(1,1)],crs="EPSG:4326")
    report=assess_data_quality(gdf)
    assert report["policy"]["score_is_quality_index_not_probability"] is True
