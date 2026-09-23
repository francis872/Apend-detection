from __future__ import annotations
import html, json
from pathlib import Path

def write_html_report(out_dir, cleaning:dict, summary:dict, plot:dict|None=None, map_points:list|None=None):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    plot=plot or {}; map_points=map_points or []
    q=summary.get("gaussian_quadrature",{}).get("confidence_interval_areas",{})
    qrows="".join(f"<tr><td>{p}%</td><td>{a.get('quadrature_area',0):.8f}</td><td>{a.get('absolute_error',0):.3e}</td></tr>" for p,a in q.items())
    inf=", ".join(f"{v:.6g}" for v in summary.get("derivative_analysis",{}).get("inflection_points",[])) or "None"
    counts=summary.get("counts",{})
    comparison=summary.get("variable_comparison",[])
    crows="".join(f"<tr><td>{html.escape(str(x.get('variable','')))}</td><td>{html.escape(str(x.get('distribution','—')))}</td><td>{x.get('n','—')}</td><td>{x.get('ks','—')}</td></tr>" for x in comparison)
    repro=html.escape(json.dumps(summary.get("reproducibility",{}),indent=2))
    spatial=html.escape(json.dumps(summary.get("spatial_validation",{}),indent=2))
    temporal=html.escape(json.dumps(summary.get("temporal_validation",{}),indent=2))
    doc=f"""<!doctype html><html><head><meta charset="utf-8"><title>Meridian Analysis Report</title>
<style>:root{{--ink:#17151a;--lav:#b9a5c4;--line:#ddd7e1}}body{{font-family:Inter,Arial,sans-serif;margin:0;color:var(--ink);background:#f7f5f8}}main{{max-width:1100px;margin:auto;padding:44px}}header{{background:#17171a;color:#fff;padding:32px;border-radius:18px}}.brand{{letter-spacing:.22em;color:#d2c1db}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}}.card{{background:#fff;border:1px solid var(--line);padding:16px;border-radius:13px}}table{{border-collapse:collapse;width:100%;background:#fff}}td,th{{border-bottom:1px solid var(--line);padding:9px;text-align:left}}h2{{margin-top:30px}}pre{{white-space:pre-wrap;background:#fff;border:1px solid var(--line);padding:14px;border-radius:12px}}.note{{color:#625b67;font-size:13px}}</style></head><body><main>
<header><div class="brand">MERIDIAN</div><h1>Geospatial Intelligence Report</h1><p>Explainable statistical and spatial anomaly analysis.</p></header>
<div class="grid"><div class="card"><small>ROWS</small><h2>{summary.get('rows',0):,}</h2></div><div class="card"><small>FEATURE</small><h2>{html.escape(str(summary.get('probability_feature')))}</h2></div><div class="card"><small>BEST FIT</small><h2>{html.escape(str(summary.get('best_distribution')))}</h2></div><div class="card"><small>CRITICAL 99%</small><h2>{counts.get('99',counts.get(99,0))}</h2></div></div>
<p class="note">90/95/99 are empirical anomaly-score quantiles, not posterior probabilities that an observation is anomalous.</p>
<h2>Gaussian quadrature</h2><table><tr><th>Central interval</th><th>Area</th><th>Absolute error</th></tr>{qrows}</table>
<h2>Derivative diagnostics</h2><p>Inflection points: {html.escape(inf)}</p>
<h2>Variable comparison</h2><table><tr><th>Variable</th><th>Comparative best fit</th><th>N</th><th>KS</th></tr>{crows}</table>
<h2>Anomaly results</h2><pre>{html.escape(json.dumps(counts,indent=2))}</pre>
<p>Possible data errors: {summary.get('noise_candidates',0):,} · Strong consensus: {summary.get('strong_consensus_count',0):,}</p>
<h2>Spatial validation</h2><pre>{spatial}</pre><h2>Temporal validation</h2><pre>{temporal}</pre>
<h2>Reproducibility</h2><pre>{repro}</pre>
</main></body></html>"""
    path=out/"REPORT.html"; path.write_text(doc,encoding="utf-8"); return path
