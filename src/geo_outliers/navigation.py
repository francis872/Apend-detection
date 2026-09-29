from __future__ import annotations

import heapq
import math
from typing import Any


DEFAULT_WEIGHTS = {
    "distance": 1.0,
    "time": 0.0,
    "cost": 0.0,
    "risk": 0.0,
    "slope": 0.0,
    "accessibility_penalty": 0.0,
}


def _edge_cost(edge: dict, weights: dict[str, float]) -> float:
    total = 0.0
    for key, weight in weights.items():
        value = float(edge.get(key, 0.0) or 0.0)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite edge metric: {key}")
        total += float(weight) * value
    return total


def build_navigation_graph(nodes: list[dict], edges: list[dict], directed: bool = True) -> dict:
    if not nodes:
        raise ValueError("nodes are required")
    node_map = {str(n["id"]): dict(n) for n in nodes}
    adjacency: dict[str, list[dict]] = {node_id: [] for node_id in node_map}
    normalized_edges = []
    for i, raw in enumerate(edges):
        source, target = str(raw["source"]), str(raw["target"])
        if source not in node_map or target not in node_map:
            raise ValueError(f"Edge {i} references an unknown node")
        edge = dict(raw)
        edge["source"], edge["target"] = source, target
        edge.setdefault("distance", float(edge.get("weight", 1.0)))
        edge.setdefault("time", edge["distance"])
        edge.setdefault("cost", 0.0)
        edge.setdefault("risk", 0.0)
        edge.setdefault("slope", 0.0)
        edge.setdefault("accessibility_penalty", 0.0)
        edge.setdefault("closed", False)
        adjacency[source].append(edge)
        normalized_edges.append(edge)
        if not directed and not edge.get("one_way", False):
            reverse = {**edge, "source": target, "target": source}
            adjacency[target].append(reverse)
    return {"nodes": node_map, "edges": normalized_edges, "adjacency": adjacency, "directed": directed}


def route(
    graph: dict,
    source: str,
    target: str,
    weights: dict[str, float] | None = None,
    constraints: dict[str, float] | None = None,
    algorithm: str = "dijkstra",
) -> dict:
    source, target = str(source), str(target)
    nodes = graph["nodes"]
    if source not in nodes or target not in nodes:
        raise ValueError("source and target must exist")
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    constraints = constraints or {}
    if any(float(v) < 0 for v in weights.values()):
        raise ValueError("routing weights must be non-negative")
    if algorithm not in {"dijkstra", "astar"}:
        raise ValueError("algorithm must be dijkstra or astar")

    def allowed(edge: dict) -> bool:
        if edge.get("closed"):
            return False
        for metric, maximum in constraints.items():
            if float(edge.get(metric, 0.0) or 0.0) > float(maximum):
                return False
        return True

    def heuristic(node_id: str) -> float:
        if algorithm != "astar":
            return 0.0
        a, b = nodes[node_id], nodes[target]
        if not all(k in a and k in b for k in ("latitude", "longitude")):
            return 0.0
        lat1, lon1, lat2, lon2 = map(math.radians, [float(a["latitude"]), float(a["longitude"]), float(b["latitude"]), float(b["longitude"])])
        h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
        meters = 6371000.0 * 2 * math.asin(min(1.0, math.sqrt(h)))
        # Admissible only for distance contribution; other non-negative objectives add cost.
        return weights.get("distance", 0.0) * meters

    frontier = [(heuristic(source), 0.0, source)]
    best = {source: 0.0}
    previous: dict[str, tuple[str, dict]] = {}
    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current == target:
            break
        if current_cost > best.get(current, math.inf):
            continue
        for edge in graph["adjacency"].get(current, []):
            if not allowed(edge):
                continue
            nxt = edge["target"]
            candidate = current_cost + _edge_cost(edge, weights)
            if candidate < best.get(nxt, math.inf):
                best[nxt] = candidate
                previous[nxt] = (current, edge)
                heapq.heappush(frontier, (candidate + heuristic(nxt), candidate, nxt))

    if target not in best:
        return {"connected": False, "path": [], "edges": [], "score": None}

    path = [target]
    selected_edges = []
    cursor = target
    while cursor != source:
        parent, edge = previous[cursor]
        selected_edges.append(edge)
        path.append(parent)
        cursor = parent
    path.reverse()
    selected_edges.reverse()
    totals = {metric: float(sum(float(e.get(metric, 0.0) or 0.0) for e in selected_edges)) for metric in DEFAULT_WEIGHTS}
    return {
        "connected": True,
        "algorithm": algorithm,
        "path": path,
        "edges": selected_edges,
        "score": float(best[target]),
        "totals": totals,
        "weights": weights,
        "constraints": constraints,
    }


def route_alternatives(graph: dict, source: str, target: str, profiles: dict[str, dict] | None = None) -> dict:
    profiles = profiles or {
        "fastest": {"time": 1.0, "distance": 0.0},
        "shortest": {"distance": 1.0},
        "lower_risk": {"risk": 1.0, "time": 0.15, "distance": 0.0},
        "lower_cost": {"cost": 1.0, "time": 0.10, "distance": 0.0},
        "accessible": {"accessibility_penalty": 1.0, "time": 0.10, "distance": 0.0},
    }
    results = []
    seen = set()
    for name, weights in profiles.items():
        result = route(graph, source, target, weights=weights)
        key = tuple(result.get("path", []))
        if result["connected"] and key not in seen:
            seen.add(key)
            result["profile"] = name
            results.append(result)
    return {
        "routes": results,
        "methodology": {
            "multiobjective": True,
            "human_decision": True,
            "dynamic_edge_metrics_supported": True,
        },
    }


def apply_traffic_updates(graph: dict, updates: list[dict]) -> dict:
    """Return a new graph with live traffic/closure values applied to matching edges."""
    nodes = list(graph["nodes"].values())
    edges = [dict(e) for e in graph["edges"]]
    lookup = {(str(e["source"]), str(e["target"])): e for e in edges}
    changed = 0
    for update in updates:
        key = (str(update["source"]), str(update["target"]))
        edge = lookup.get(key)
        if edge is None:
            continue
        for field in ("time", "risk", "cost", "closed"):
            if field in update:
                edge[field] = update[field]
        changed += 1
    refreshed = build_navigation_graph(nodes, edges, directed=bool(graph.get("directed", True)))
    refreshed["traffic_updates_applied"] = changed
    return refreshed
