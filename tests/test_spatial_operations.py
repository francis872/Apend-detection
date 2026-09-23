from geo_outliers.spatial_operations import spatial_operation

poly={"type":"Feature","id":"zone","properties":{"name":"A"},"geometry":{"type":"Polygon","coordinates":[[[-75,5],[-74,5],[-74,6],[-75,6],[-75,5]]]}}
inside={"type":"Feature","id":"p1","properties":{"value":2},"geometry":{"type":"Point","coordinates":[-74.5,5.5]}}
outside={"type":"Feature","id":"p2","properties":{"value":3},"geometry":{"type":"Point","coordinates":[-73,5.5]}}

def test_point_in_polygon_spatial_join():
    out=spatial_operation([inside,outside],[poly],"within")
    assert out["count"]==1
    assert out["features"][0]["properties"]["_feature_id_left"]=="p1"

def test_metric_buffer_returns_polygon():
    out=spatial_operation([inside],[],"buffer",distance_m=1000)
    assert out["count"]==1
    assert out["features"][0]["geometry"]["type"] in {"Polygon","MultiPolygon"}

def test_nearest_reports_metric_distance():
    out=spatial_operation([outside],[inside],"nearest")
    assert out["count"]==1
    assert out["features"][0]["properties"]["distance_m"]>0

def test_intersection_produces_traceable_feature():
    b={"type":"Feature","id":"b","properties":{},"geometry":{"type":"Polygon","coordinates":[[[-74.7,5.2],[-73.8,5.2],[-73.8,5.8],[-74.7,5.8],[-74.7,5.2]]]}}
    out=spatial_operation([poly],[b],"intersection")
    assert out["count"]==1
