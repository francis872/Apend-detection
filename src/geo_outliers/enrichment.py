from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
import pandas as pd

from .integrations import fetch_open_meteo, fetch_firms_area, integration_status


WEATHER_DOMAINS={"environment","agriculture","risk","urban","energy","earth_observation"}
FIRMS_DOMAINS={"environment","risk","earth_observation"}


def _date_column(gdf):
    for c in ("ACQ_DATE","date","fecha","timestamp","datetime"):
        if c in gdf.columns:
            s=pd.to_datetime(gdf[c],errors="coerce")
            if s.notna().any(): return c,s
    return None,pd.Series(pd.NaT,index=gdf.index)


def plan_auto_enrichment(gdf,domain:str,contract:dict|None=None)->dict:
    """Create an explicit enrichment plan. External context is never silently added to the anomaly score."""
    status=integration_status()
    geo=gdf.to_crs(4326)
    valid=geo[geo.geometry.notna() & ~geo.geometry.is_empty]
    if valid.empty:
        return {"enabled":False,"actions":[],"warnings":["No valid geometry for external enrichment"],"policy":"context_only"}

    minx,miny,maxx,maxy=[float(x) for x in valid.total_bounds]
    centroid=valid.geometry.unary_union.centroid
    date_col,dates=_date_column(valid)
    min_date=str(dates.min().date()) if dates.notna().any() else None
    max_date=str(dates.max().date()) if dates.notna().any() else None
    columns={str(c).lower() for c in gdf.columns}

    actions=[]; warnings=[]
    if domain in WEATHER_DOMAINS:
        missing_weather=not any(x in columns for x in ("precipitation","rainfall","temperature","surface_temperature","humidity","relative_humidity"))
        actions.append({
            "provider":"Open-Meteo",
            "kind":"weather_context",
            "recommended":bool(missing_weather),
            "configured":True,
            "reason":"weather variables are relevant to this domain"+(" and are missing from the source" if missing_weather else ""),
            "mode":"context_only",
            "location":{"lat":float(centroid.y),"lon":float(centroid.x)},
            "date_range":{"start":min_date,"end":max_date},
        })

    if domain in FIRMS_DOMAINS:
        has_fire=any(x in columns for x in ("frp","brightness","bright_ti4","bright_ti5"))
        configured=bool(status.get("NASA FIRMS",{}).get("configured"))
        actions.append({
            "provider":"NASA FIRMS",
            "kind":"fire_context",
            "recommended":not has_fire,
            "configured":configured,
            "reason":"thermal anomaly context is relevant to this domain"+(" and fire variables are missing" if not has_fire else ""),
            "mode":"context_only",
            "bbox":[minx,miny,maxx,maxy],
            "date":max_date,
        })
        if not configured:
            warnings.append("NASA FIRMS context planned but MAP_KEY is not configured")

    return {
        "enabled":bool(actions),
        "policy":"context_only",
        "score_mutation":False,
        "causal_claims":False,
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "domain":domain,
        "date_column":date_col,
        "bounds":[minx,miny,maxx,maxy],
        "actions":actions,
        "warnings":warnings,
    }


def execute_auto_enrichment(gdf,plan:dict,max_weather_points:int=8)->dict:
    """Execute recommended/configured enrichment actions and preserve provenance."""
    results=[]
    geo=gdf.to_crs(4326)
    for action in plan.get("actions",[]):
        provider=action.get("provider")
        if not action.get("recommended"):
            results.append({"provider":provider,"status":"skipped","reason":"source already contains related variables","mode":"context_only"})
            continue
        if not action.get("configured",False):
            results.append({"provider":provider,"status":"skipped","reason":"provider not configured","mode":"context_only"})
            continue
        try:
            if provider=="Open-Meteo":
                date_col,dates=_date_column(geo)
                valid=geo[geo.geometry.notna() & ~geo.geometry.is_empty].copy()
                if dates.notna().any():
                    valid["_meridian_date"]=pd.to_datetime(valid[date_col],errors="coerce")
                    valid=valid[valid["_meridian_date"].notna()]
                if valid.empty:
                    results.append({"provider":provider,"status":"skipped","reason":"no dated geometry","mode":"context_only"})
                    continue
                take=min(max_weather_points,len(valid))
                sample=valid.sample(take,random_state=42) if len(valid)>take else valid
                rows=[]
                for idx,row in sample.iterrows():
                    day=str((row.get("_meridian_date") if "_meridian_date" in row else pd.Timestamp.utcnow()).date())
                    weather=fetch_open_meteo(float(row.geometry.y),float(row.geometry.x),day)
                    weather["row_id"]=int(idx)
                    rows.append(weather)
                metrics={}
                for key in ("temperature_2m_mean","relative_humidity_2m_mean","precipitation_sum","wind_speed_10m_mean"):
                    vals=[r[key] for r in rows if key in r and np.isfinite(r[key])]
                    if vals: metrics[key]={"mean":float(np.mean(vals)),"min":float(np.min(vals)),"max":float(np.max(vals))}
                results.append({"provider":provider,"status":"ok","mode":"context_only","samples":rows,"summary":metrics,
                                "provenance":{"provider":"Open-Meteo","retrieved_utc":datetime.now(timezone.utc).isoformat()}})
            elif provider=="NASA FIRMS":
                west,south,east,north=action["bbox"]
                date=action.get("date")
                firms=fetch_firms_area(west,south,east,north,days=1,date=date or None)
                results.append({"provider":provider,"status":"ok","mode":"context_only","rows":int(len(firms)),
                                "summary":{"frp_mean":float(pd.to_numeric(firms.get("FRP"),errors="coerce").mean()) if "FRP" in firms else None},
                                "provenance":{"provider":"NASA FIRMS","product":"VIIRS_NOAA21_NRT","retrieved_utc":datetime.now(timezone.utc).isoformat()}})
        except Exception as e:
            results.append({"provider":provider,"status":"error","error":str(e),"mode":"context_only"})

    return {
        "policy":plan.get("policy","context_only"),
        "score_mutation":False,
        "causal_claims":False,
        "results":results,
        "provenance_note":"External data are contextual evidence only and are not automatically included in the anomaly score.",
    }
