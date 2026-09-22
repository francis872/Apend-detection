from __future__ import annotations

import json
from pathlib import Path


def build_report(cleaning: dict, summary: dict) -> str:
    q=summary.get("gaussian_quadrature",{}).get("confidence_interval_areas",{})
    lines=[
        "# Apend Detection - Statistical Analysis Report","",
        f"- Observations analysed: {summary.get('rows',0):,}",
        f"- Probability feature: {summary.get('probability_feature')}",
        f"- Best fitted distribution: {summary.get('best_distribution')}",
        f"- Spatial method: {summary.get('spatial_validation',{}).get('method','n/a')}",
        f"- Temporal method: {summary.get('temporal_validation',{}).get('method','n/a')}",
        "","## Gaussian quadrature"
    ]
    for p in ("90","95","99"):
        if p in q:
            a=q[p]
            lines.append(f"- {p}%: area={a['quadrature_area']:.8f}, target={a['target_area']:.2f}, error={a['absolute_error']:.3e}, interval=[{a['lower']:.6g}, {a['upper']:.6g}]")
    lines += ["","## Differential analysis",
        f"- Inflection points: {summary.get('derivative_analysis',{}).get('inflection_points',[])}",
        f"- Derivative method: {summary.get('derivative_analysis',{}).get('method','n/a')}",
        "","## Detection"]
    for p,c in summary.get("counts",{}).items():
        lines.append(f"- Outliers {p}%: {c:,}")
    lines += [
        f"- Noise candidates: {summary.get('noise_candidates',0):,}",
        f"- Strong consensus anomalies: {summary.get('strong_consensus_count',0):,}",
        "","## Interpretation",
        "An anomaly is statistically unusual; it is not automatically an erroneous record. noise_candidate is intentionally more conservative and requires strong multi-method disagreement.",
        "","## Cleaning",
        f"- Rows before: {cleaning.get('rows_before')}",
        f"- Rows after basic cleaning: {cleaning.get('rows_after_basic_clean')}",
        f"- Invalid geometries fixed: {cleaning.get('invalid_geometry_fixed')}",
    ]
    return "\n".join(lines)+"\n"


def write_report(out_dir: str | Path, cleaning: dict, summary: dict):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    json_path=out/"summary.json"; md_path=out/"REPORT.md"
    json_path.write_text(json.dumps({"cleaning":cleaning,"detector":summary},ensure_ascii=False,indent=2),encoding="utf-8")
    md_path.write_text(build_report(cleaning,summary),encoding="utf-8")
    return json_path,md_path
