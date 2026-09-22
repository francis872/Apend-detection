from __future__ import annotations

import html
import json
from pathlib import Path


def write_html_report(out_dir, cleaning:dict, summary:dict, plot:dict|None=None, map_points:list|None=None):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    q=summary.get("gaussian_quadrature",{}).get("confidence_interval_areas",{})
    rows="".join(
        f"<tr><td>{p}%</td><td>{a.get('quadrature_area',0):.8f}</td><td>{a.get('absolute_error',0):.3e}</td></tr>"
        for p,a in q.items()
    )
    inf=", ".join(f"{v:.6g}" for v in summary.get("derivative_analysis",{}).get("inflection_points",[])) or "None"
    doc=f"""<!doctype html><html><head><meta charset="utf-8"><title>Apend Detection Report</title>
<style>body{{font-family:Arial,sans-serif;margin:40px;color:#17211f}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd7d3;padding:8px}}h1,h2{{color:#0e5a46}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.card{{border:1px solid #ccd7d3;padding:14px;border-radius:10px}}pre{{white-space:pre-wrap}}</style></head><body>
<h1>Apend Detection - Analysis Report</h1>
<div class="grid"><div class="card"><b>Rows</b><br>{summary.get('rows',0):,}</div>
<div class="card"><b>Feature</b><br>{html.escape(str(summary.get('probability_feature')))}</div>
<div class="card"><b>Comparative best fit</b><br>{html.escape(str(summary.get('best_distribution')))}</div></div>
<h2>Gaussian quadrature</h2><table><tr><th>Central interval</th><th>Area</th><th>Error</th></tr>{rows}</table>
<h2>Derivatives</h2><p>Inflection points: {html.escape(inf)}</p>
<h2>Anomaly results</h2><pre>{html.escape(json.dumps(summary.get('counts',{}),indent=2))}</pre>
<p>Probable noise: {summary.get('noise_candidates',0):,} · Strong consensus: {summary.get('strong_consensus_count',0):,}</p>
<h2>Spatial / temporal</h2><pre>{html.escape(json.dumps({'spatial':summary.get('spatial_validation',{}),'temporal':summary.get('temporal_validation',{})},indent=2))}</pre>
<h2>Reproducibility</h2><pre>{html.escape(json.dumps(summary.get('reproducibility',{}),indent=2))}</pre>
</body></html>"""
    path=out/"REPORT.html"
    path.write_text(doc,encoding="utf-8")
    return path
