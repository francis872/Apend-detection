import pytest
from geo_outliers.earth_observation import sensor_catalog, index_catalog, build_index_plan, _bbox_from_geometry


def test_sensor_catalog_has_core_missions():
    ids={x["id"] for x in sensor_catalog()}
    assert {"sentinel-2-l2a","landsat-c2-l2","modis-09A1-061"}.issubset(ids)


def test_sentinel_ndvi_plan_maps_required_assets():
    plan=build_index_plan("sentinel-2-l2a",["NDVI","NDMI"])
    ndvi=next(x for x in plan["indices"] if x["index"]=="NDVI")
    assert ndvi["asset_keys"]==["nir","red"]
    assert plan["policy"]["silent_scaling"] is False


def test_landsat_lst_requires_thermal_asset():
    plan=build_index_plan("landsat-c2-l2",["LST"])
    assert plan["indices"][0]["asset_keys"]==["lwir11"]


def test_invalid_index_is_rejected():
    with pytest.raises(ValueError):
        build_index_plan("sentinel-2-l2a",["MAGIC_INDEX"])


def test_geometry_bbox():
    geometry={"type":"Polygon","coordinates":[[[-76,5],[-75,5],[-75,6],[-76,6],[-76,5]]]}
    assert _bbox_from_geometry(geometry)==[-76.0,5.0,-75.0,6.0]


def test_index_catalog_marks_lst_as_provider_specific():
    assert "Provider-specific" in index_catalog()["formulas"]["LST"]
