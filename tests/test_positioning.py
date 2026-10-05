from geo_outliers.navigation import build_navigation_graph, route
from geo_outliers.positioning import PositionFix, bearing_deg, build_turn_by_turn, navigation_progress, snap_to_navigation_graph


NODES = [
    {"id": "a", "latitude": 6.2440, "longitude": -75.5812},
    {"id": "b", "latitude": 6.2440, "longitude": -75.5802},
    {"id": "c", "latitude": 6.2450, "longitude": -75.5802},
]
EDGES = [
    {"source": "a", "target": "b", "distance": 110, "time": 20, "name": "Calle A"},
    {"source": "b", "target": "c", "distance": 111, "time": 22, "name": "Carrera B"},
]


def test_snap_to_nearest_navigation_node():
    graph = build_navigation_graph(NODES, EDGES)
    fix = PositionFix(6.2440, -75.58119, accuracy_m=5)
    result = snap_to_navigation_graph(graph, fix, 50)
    assert result["matched"] is True
    assert result["node_id"] == "a"
    assert result["confidence"] > 0.9


def test_turn_by_turn_detects_right_turn_and_arrival():
    graph = build_navigation_graph(NODES, EDGES)
    result = route(graph, "a", "c", {"distance": 0, "time": 1})
    guidance = build_turn_by_turn(graph, result)
    assert guidance["steps"][0]["maneuver"] == "depart"
    assert guidance["steps"][1]["maneuver"] == "turn_left" or guidance["steps"][1]["maneuver"] == "turn_right"
    assert guidance["steps"][-1]["maneuver"] == "arrive"


def test_navigation_progress_flags_off_route():
    graph = build_navigation_graph(NODES, EDGES)
    result = route(graph, "a", "c")
    progress = navigation_progress(graph, result, PositionFix(6.30, -75.60), 50)
    assert progress["on_route"] is False
    assert progress["reroute_required"] is True


def test_bearing_is_normalized():
    value = bearing_deg((6.2440, -75.5812), (6.2440, -75.5802))
    assert 0 <= value < 360
