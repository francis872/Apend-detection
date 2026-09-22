from __future__ import annotations

from pathlib import Path


def export_analysis(scored, fits, cleaning: dict, summary: dict, out_dir: str | Path, noise_level: int = 99):
    from .report import write_report
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    scored.to_file(out/"analysis.gpkg",layer="analysis",driver="GPKG")
    for p in (90,95,99):
        c=f"outlier_{p}"
        if c in scored.columns:
            scored.loc[scored[c]].to_file(out/f"outliers_{p}.gpkg",layer=f"outliers_{p}",driver="GPKG")
    if "noise_candidate" in scored.columns:
        clean=scored.loc[~scored["noise_candidate"]].copy()
    else:
        c=f"outlier_{noise_level}"
        clean=scored.loc[~scored[c]].copy() if c in scored.columns else scored.copy()
    clean.to_file(out/"cleaned.gpkg",layer="cleaned",driver="GPKG")
    flat=scored.copy()
    flat["geometry_wkt"]=flat.geometry.to_wkt()
    flat.drop(columns="geometry").to_csv(out/"analysis.csv",index=False)
    fits.to_csv(out/"distribution_fits.csv",index=False)
    write_report(out,cleaning,summary)
    return {
        "analysis_gpkg":"analysis.gpkg","cleaned_gpkg":"cleaned.gpkg",
        "outliers_90":"outliers_90.gpkg","outliers_95":"outliers_95.gpkg","outliers_99":"outliers_99.gpkg",
        "analysis_csv":"analysis.csv","fits":"distribution_fits.csv","summary":"summary.json","report":"REPORT.md"
    }
