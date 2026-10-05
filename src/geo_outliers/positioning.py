from __future__ import annotations

import math
from dataclasses import dataclass


EARTH_RADIUS_M = 6371000.0


@dataclass(frozen=True)
class PositionFix:
    latitude: float
    longitude: float
    accuracy_m: float | None = None
    speed_mps: float | None = None
    heading_deg: float | None = None
    timestamp: str | None = None

    def validate(self) -> None:
        if not (-90 <= self.latitude <= 90):
            raise ValueError("latitude out of range")
        if not (-180 <= self.longitude <= 180):
            raise ValueError("longitude out of range")
        if self.accuracy_m is not None and self.accuracy_m < 0:
            raise ValueError("accuracy_m must be non-negative")


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return EARTH_RADIUS_M * 2 * math.asin(min(1.0, math.sqrt(h)))


def bearing_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def snap_to_navigation_graph(graph: dict, position: PositionFix, max_distance_m: float = 100.0) -> dict:
    position.validate()
    if max_distance_m <= 0:
        raise ValueError("max_distance_m must be positive")
    best_id = None
    best_distance = math.inf
    for node_id, node in graph["nodes"].items():
        if "latitude" not in node or "longitude" not in node:
            continue
        distance = haversine_m(
            (position.latitude, position.longitude),
            (float(node["latitude"]), float(node["longitude"])),
        )
        if distance < best_distance:
            best_distance = distance
            best_id = node_id
    matched = best_id is not None and best_distance <= max_distance_m
    return {
        "matched": matched,
        "node_id": best_id if matched else None,
        "distance_m": None if best_id is None else float(best_distance),
        "max_distance_m": float(max_distance_m),
        "position": {
            "latitude": position.latitude,
            "longitude": position.longitude,
            "accuracy_m": position.accuracy_m,
            "speed_mps": position.speed_mps,
            "heading_deg": position.heading_deg,
            "timestamp": position.timestamp,
        },
        "confidence": None if best_id is None else float(max(0.0, 1.0 - best_distance / max_distance_m)),
    }


def _turn_type(delta: float) -> str:
    normalized = (delta + 540.0) % 360.0 - 180.0
    magnitude = abs(normalized)
    if magnitude < 20:
        return "continue"
    if magnitude < 60:
        return "slight_right" if normalized > 0 else "slight_left"
    if magnitude < 135:
        return "turn_right" if normalized > 0 else "turn_left"
    return "u_turn"


def build_turn_by_turn(graph: dict, route_result: dict) -> dict:
    if not route_result.get("connected"):
        return {"connected": False, "steps": [], "total_distance_m": None}
    path = route_result.get("path", [])
    if len(path) < 2:
        return {"connected": True, "steps": [], "total_distance_m": 0.0}

    nodes = graph["nodes"]
    steps = []
    cumulative = 0.0
    previous_bearing = None

    for i in range(len(path) - 1):
        a_id, b_id = path[i], path[i + 1]
        a, b = nodes[a_id], nodes[b_id]
        a_xy = (float(a["latitude"]), float(a["longitude"]))
        b_xy = (float(b["latitude"]), float(b["longitude"]))
        distance = haversine_m(a_xy, b_xy)
        heading = bearing_deg(a_xy, b_xy)
        cumulative += distance

        if i == 0:
            maneuver = "depart"
        else:
            maneuver = _turn_type(heading - float(previous_bearing))

        edge = next(
            (
                e for e in route_result.get("edges", [])
                if str(e["source"]) == str(a_id) and str(e["target"]) == str(b_id)
            ),
            {},
        )
        road_name = edge.get("name") or edge.get("road_name") or edge.get("street")
        instruction = {
            "depart": "Start",
            "continue": "Continue straight",
            "slight_right": "Keep slightly right",
            "slight_left": "Keep slightly left",
            "turn_right": "Turn right",
            "turn_left": "Turn left",
            "u_turn": "Make a U-turn",
        }[maneuver]
        if road_name:
            instruction = f"{instruction} onto {road_name}"

        steps.append({
            "index": i,
            "from": a_id,
            "to": b_id,
            "maneuver": maneuver,
            "instruction": instruction,
            "distance_m": float(distance),
            "bearing_deg": float(heading),
            "cumulative_distance_m": float(cumulative),
            "road_name": road_name,
        })
        previous_bearing = heading

    steps.append({
        "index": len(steps),
        "from": path[-1],
        "to": None,
        "maneuver": "arrive",
        "instruction": "You have arrived at your destination",
        "distance_m": 0.0,
        "bearing_deg": None,
        "cumulative_distance_m": float(cumulative),
        "road_name": None,
    })
    return {
        "connected": True,
        "steps": steps,
        "total_distance_m": float(cumulative),
        "route_score": route_result.get("score"),
        "totals": route_result.get("totals", {}),
    }


def navigation_progress(
    graph: dict,
    route_result: dict,
    position: PositionFix,
    off_route_threshold_m: float = 80.0,
) -> dict:
    matched = snap_to_navigation_graph(graph, position, max_distance_m=max(off_route_threshold_m, 1.0))
    path = [str(x) for x in route_result.get("path", [])]
    current = matched.get("node_id")
    on_route = bool(matched["matched"] and current in path)
    remaining_path = []
    if on_route:
        idx = path.index(current)
        remaining_path = path[idx:]
    return {
        "on_route": on_route,
        "reroute_required": not on_route,
        "matched_node": current,
        "distance_to_graph_m": matched.get("distance_m"),
        "remaining_path": remaining_path,
        "position_confidence": matched.get("confidence"),
    }
