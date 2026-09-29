import pytest

from geo_outliers.navigation import apply_traffic_updates, build_navigation_graph, route, route_alternatives


NODES = [
    {"id": "a", "latitude": 0.0, "longitude": 0.0},
    {"id": "b", "latitude": 0.0, "longitude": 0.01},
    {"id": "c", "latitude": 0.01, "longitude": 0.01},
    {"id": "d", "latitude": 0.01, "longitude": 0.0},
]
EDGES = [
    {"source": "a", "target": "b", "distance": 1000, "time": 10, "risk": 8, "cost": 1},
    {"source": "b", "target": "c", "distance": 1000, "time": 10, "risk": 8, "cost": 1},
    {"source": "a", "target": "d", "distance": 1400, "time": 16, "risk": 1, "cost": 2},
    {"source": "d", "target": "c", "distance": 1400, "time": 16, "risk": 1, "cost": 2},
]


def test_fastest_and_lower_risk_routes_can_differ():
    graph = build_navigation_graph(NODES, EDGES)
    fastest = route(graph, "a", "c", {"distance": 0, "time": 1})
    safer = route(graph, "a", "c", {"distance": 0, "time": .1, "risk": 1})
    assert fastest["path"] == ["a", "b", "c"]
    assert safer["path"] == ["a", "d", "c"]


def test_closed_edge_triggers_reroute():
    graph = build_navigation_graph(NODES, EDGES)
    updated = apply_traffic_updates(graph, [{"source": "b", "target": "c", "closed": True}])
    result = route(updated, "a", "c", {"distance": 0, "time": 1})
    assert result["path"] == ["a", "d", "c"]
    assert updated["traffic_updates_applied"] == 1


def test_alternatives_deduplicate_identical_paths():
    graph = build_navigation_graph(NODES, EDGES)
    result = route_alternatives(graph, "a", "c")
    paths = [tuple(x["path"]) for x in result["routes"]]
    assert len(paths) == len(set(paths))
    assert len(paths) >= 2


def test_unknown_algorithm_is_rejected():
    graph = build_navigation_graph(NODES, EDGES)
    with pytest.raises(ValueError):
        route(graph, "a", "c", algorithm="magic")
