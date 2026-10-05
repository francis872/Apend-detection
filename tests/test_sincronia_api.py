from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from geo_outliers.sincronia_api import analyze_property_context, compare_property_contexts, property_accessibility


def _analysis(tmp_path: Path):
    results = tmp_path / "results"
    results.mkdir()
    gdf = gpd.GeoDataFrame(
        {
            "outlier_score": [0.1, 0.8, 0.4],
            "outlier_90": [False, True, False],
            "outlier_95": [False, True, False],
            "outlier_99": [False, False, False],
            "score_probability": [0.2, 0.7, 0.3],
            "score_spatial": [0.1, 0.9, 0.4],
            "score_temporal": [0.2, 0.8, 0.4],
            "consensus_methods": [1, 4, 2],
        },
        geometry=[
            Point(-75.5812, 6.2440),
            Point(-75.5808, 6.2443),
            Point(-75.70, 6.30),
        ],
        crs="EPSG:4326",
    )
    gdf.to_file(results / "analysis.geojson", driver="GeoJSON")


def test_property_context_returns_only_evidence(tmp_path):
    _analysis(tmp_path)
    result = analyze_property_context(tmp_path, "prop-1", 6.2440, -75.5812, 500)
    assert result["external_id"] == "prop-1"
    assert result["territorial_summary"]["rows"] == 2
    assert result["methodology"]["property_valuation"] is False


def test_compare_requires_two_properties(tmp_path):
    _analysis(tmp_path)
    with pytest.raises(ValueError):
        compare_property_contexts(tmp_path, [{"external_id":"a","latitude":6.2,"longitude":-75.5}])


def test_accessibility_uses_existing_meridian_graph():
    graph = {
        "nodes": [{"id":"p"},{"id":"hospital"},{"id":"school"}],
        "edges": [
            {"source":"p","target":"hospital","weight":3},
            {"source":"p","target":"school","weight":7},
        ],
    }
    result = property_accessibility(graph, "p", [
        {"node_id":"school","category":"education","name":"School"},
        {"node_id":"hospital","category":"health","name":"Hospital"},
    ])
    assert result["nearest_reachable"]["node_id"] == "hospital"
