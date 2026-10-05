from __future__ import annotations

from pathlib import Path

from shapely.geometry import Point, mapping

from .network_corridor import shortest_path
from .territorial import buffer_geometry, load_analysis_layer, select_polygon, summarize_region


def _point_geometry(latitude: float, longitude: float) -> dict:
    lat, lon = float(latitude), float(longitude)
    if not (-90 <= lat <= 90):
        raise ValueError("latitude out of range")
    if not (-180 <= lon <= 180):
        raise ValueError("longitude out of range")
    return mapping(Point(lon, lat))


def analyze_property_context(
    job_dir: Path,
    external_id: str,
    latitude: float,
    longitude: float,
    radius_m: float = 1000.0,
    metadata: dict | None = None,
) -> dict:
    """Return Meridian territorial evidence around a Sincronia property.

    The response exposes measured/derived Meridian evidence only. It does not invent
    valuation, safety or investment scores when the underlying analysis does not
    provide those variables.
    """
    if not external_id:
        raise ValueError("external_id is required")
    if radius_m <= 0 or radius_m > 50000:
        raise ValueError("radius_m must be between 0 and 50000")

    point = _point_geometry(latitude, longitude)
    area = buffer_geometry(point, float(radius_m))
    analysis = load_analysis_layer(job_dir)
    selected = select_polygon(analysis, area)
    summary = summarize_region(selected)

    signals = {
        "anomaly_intensity": summary.get("mean_score"),
        "probability_signal": summary.get("mean_probability"),
        "spatial_signal": summary.get("mean_spatial"),
        "temporal_signal": summary.get("mean_temporal"),
        "critical_observations": summary.get("critical_99", 0),
        "high_observations": summary.get("high_95", 0),
        "elevated_observations": summary.get("elevated_90", 0),
    }
    available = {k: v for k, v in signals.items() if v is not None}

    return {
        "consumer": "sincronia",
        "external_id": str(external_id),
        "location": {"latitude": float(latitude), "longitude": float(longitude)},
        "radius_m": float(radius_m),
        "geometry": area,
        "territorial_summary": summary,
        "signals": available,
        "metadata": metadata or {},
        "methodology": {
            "source": "Meridian completed analysis layer",
            "causal_claims": False,
            "investment_recommendation": False,
            "property_valuation": False,
            "missing_metrics_are_not_invented": True,
        },
    }


def compare_property_contexts(job_dir: Path, properties: list[dict], radius_m: float = 1000.0) -> dict:
    if len(properties) < 2:
        raise ValueError("At least two properties are required")
    if len(properties) > 50:
        raise ValueError("A maximum of 50 properties can be compared per request")

    analyses = [
        analyze_property_context(
            job_dir,
            str(p["external_id"]),
            float(p["latitude"]),
            float(p["longitude"]),
            float(p.get("radius_m", radius_m)),
            p.get("metadata"),
        )
        for p in properties
    ]

    comparable_metrics = (
        "anomaly_intensity",
        "probability_signal",
        "spatial_signal",
        "temporal_signal",
    )
    ranking = {}
    for metric in comparable_metrics:
        rows = [
            {"external_id": x["external_id"], "value": x["signals"].get(metric)}
            for x in analyses
            if x["signals"].get(metric) is not None
        ]
        ranking[metric] = sorted(rows, key=lambda x: x["value"])

    return {
        "consumer": "sincronia",
        "properties": analyses,
        "rankings": ranking,
        "ranking_note": "Rankings order measured Meridian signals from lower to higher; they are not purchase recommendations.",
    }


def property_accessibility(graph: dict, source_node: str, targets: list[dict]) -> dict:
    if not targets:
        raise ValueError("targets are required")
    results = []
    for target in targets:
        target_node = str(target["node_id"])
        result = shortest_path(graph, str(source_node), target_node)
        results.append({
            "category": target.get("category"),
            "name": target.get("name"),
            "node_id": target_node,
            **result,
        })
    reachable = [x for x in results if x["connected"]]
    reachable.sort(key=lambda x: x["distance"])
    return {
        "source_node": str(source_node),
        "destinations": results,
        "nearest_reachable": reachable[0] if reachable else None,
        "methodology": {
            "distance_is_graph_weight": True,
            "travel_time_requires_time-calibrated_edges": True,
        },
    }


def sincronia_capabilities() -> dict:
    return {
        "consumer": "sincronia",
        "version": "1.0",
        "capabilities": [
            "property_territorial_context",
            "property_comparison",
            "territorial_anomaly_context",
            "spatial_signal_context",
            "temporal_signal_context",
            "network_accessibility",
        ],
        "not_inferred_without_data": [
            "property_market_value",
            "crime_or_personal_safety",
            "investment_return",
            "legal_status",
        ],
    }
