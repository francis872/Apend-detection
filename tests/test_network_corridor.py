import pytest

from geo_outliers.network_corridor import build_territorial_graph, shortest_path, detect_network_corridors


def sample_nodes():
    return [
      {"id":"a","longitude":-74.000,"latitude":4.600,"variable":"event","date":"2026-01-01"},
      {"id":"b","longitude":-74.005,"latitude":4.600,"variable":"event","date":"2026-01-05"},
      {"id":"c","longitude":-74.010,"latitude":4.600,"variable":"infrastructure","date":"2026-01-10"},
      {"id":"d","longitude":-74.015,"latitude":4.600,"variable":"event","date":"2026-01-15"},
    ]


def test_graph_builds_centrality_components_and_density():
    graph=build_territorial_graph(sample_nodes(),proximity_m=800)
    assert graph["metrics"]["node_count"]==4
    assert graph["metrics"]["edge_count"]>=3
    assert graph["metrics"]["connected_components"]==1
    assert 0<=graph["metrics"]["network_density"]<=1
    assert all("betweenness_centrality" in n for n in graph["nodes"])


def test_shortest_path_uses_weighted_graph():
    graph=build_territorial_graph(sample_nodes(),proximity_m=800)
    result=shortest_path(graph,"a","d")
    assert result["connected"] is True
    assert result["path"][0]=="a"
    assert result["path"][-1]=="d"
    assert result["distance"]>0


def test_corridor_combines_density_connectivity_continuity_and_time():
    result=detect_network_corridors(sample_nodes(),proximity_m=800,min_nodes=3,permutations=19)
    assert result["corridors"]
    corridor=result["corridors"][0]
    assert set(["density","connectivity","continuity","temporal_support"]).issubset(corridor["metrics"])
    assert 0<=corridor["confidence"]<=1
    assert corridor["validation_pvalue"] is not None
    assert result["methodology"]["causal_claims"] is False
    assert result["methodology"]["conflict_classification"] is False


def test_disconnected_graph_reports_no_path():
    graph=build_territorial_graph(sample_nodes(),proximity_m=10)
    result=shortest_path(graph,"a","d")
    assert result["connected"] is False
    assert result["path"]==[]
