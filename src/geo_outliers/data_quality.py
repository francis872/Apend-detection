from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd

WGS84="EPSG:4326"

def assess_data_quality(gdf:gpd.GeoDataFrame)->dict:
    """Deterministic spatial/data quality assessment. It reports evidence; it does not silently repair source data."""
    n=max(len(gdf),1)
    issues=[]; metrics={}
    metrics["rows"]=int(len(gdf))
    metrics["crs"]=str(gdf.crs) if gdf.crs else None
    metrics["missing_geometry"]=int(gdf.geometry.isna().sum()) if "geometry" in gdf else len(gdf)
    metrics["empty_geometry"]=int(gdf.geometry.is_empty.sum()) if "geometry" in gdf else len(gdf)
    metrics["invalid_geometry"]=int((~gdf.geometry.is_valid).sum()) if "geometry" in gdf else len(gdf)
    metrics["duplicate_geometry"]=int(gdf.geometry.to_wkb().duplicated().sum()) if "geometry" in gdf else 0

    geo=gdf
    if gdf.crs:
        try: geo=gdf.to_crs(WGS84)
        except Exception: issues.append({"severity":"critical","code":"crs_transform_failed","message":"Geometry could not be transformed to WGS84."})
    else:
        issues.append({"severity":"critical","code":"missing_crs","message":"Source has no declared CRS."})

    if "geometry" in geo:
        points=geo.geometry[geo.geometry.notna() & ~geo.geometry.is_empty]
        if len(points) and bool((points.geom_type=="Point").all()):
            xs=points.x; ys=points.y
            metrics["coordinates_outside_wgs84"]=int((~xs.between(-180,180)|~ys.between(-90,90)).sum())
        else: metrics["coordinates_outside_wgs84"]=0

    column_quality=[]
    for col in gdf.columns:
        if col=="geometry":continue
        s=gdf[col]
        missing=float(s.isna().mean()) if len(s) else 0.0
        column_quality.append({"column":str(col),"missing_ratio":missing,"unique":int(s.nunique(dropna=True))})
        if missing>.5: issues.append({"severity":"high","code":"high_missingness","column":str(col),"value":missing})

    if metrics["invalid_geometry"]:issues.append({"severity":"high","code":"invalid_geometry","value":metrics["invalid_geometry"]})
    if metrics["duplicate_geometry"]:issues.append({"severity":"medium","code":"duplicate_geometry","value":metrics["duplicate_geometry"]})
    if metrics.get("coordinates_outside_wgs84",0):issues.append({"severity":"critical","code":"coordinate_range","value":metrics["coordinates_outside_wgs84"]})

    penalties=(
        min(metrics["missing_geometry"]/n,1)*.25+
        min(metrics["invalid_geometry"]/n,1)*.25+
        min(metrics["duplicate_geometry"]/n,1)*.15+
        min(metrics.get("coordinates_outside_wgs84",0)/n,1)*.25+
        min(sum(1 for x in column_quality if x["missing_ratio"]>.5)/max(len(column_quality),1),1)*.10
    )
    score=float(max(0,min(1,1-penalties)))
    grade="excellent" if score>=.95 else "good" if score>=.85 else "warning" if score>=.70 else "poor"
    blockers=[x for x in issues if x["severity"]=="critical"]
    return {"version":"1.0","quality_score":score,"grade":grade,"metrics":metrics,"columns":column_quality,
            "issues":issues,"blockers":blockers,"ready":not blockers,
            "policy":{"silent_repairs":False,"score_is_quality_index_not_probability":True}}

def quality_gate(report:dict,min_score:float=.70)->dict:
    if report.get("blockers"):
        return {"ok":False,"status":"blocked","reason":"Critical spatial data quality issue detected."}
    if float(report.get("quality_score",0))<min_score:
        return {"ok":True,"status":"warning","reason":f"Data quality score is below {min_score:.2f}; analysis may continue with explicit warning."}
    return {"ok":True,"status":"passed","reason":"Data quality gate passed."}
