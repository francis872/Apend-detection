from __future__ import annotations

import io
import os
from dataclasses import dataclass

import geopandas as gpd
import httpx
import pandas as pd
from shapely.geometry import Point


NASA_FIRMS_BASE="https://firms.modaps.eosdis.nasa.gov/api"
OPEN_METEO_ARCHIVE="https://archive-api.open-meteo.com/v1/archive"


class ExternalAPIError(RuntimeError):
    pass


def fetch_firms_area(
    west: float,
    south: float,
    east: float,
    north: float,
    days: int=1,
    source: str="VIIRS_NOAA21_NRT",
    date: str | None=None,
    map_key: str | None=None,
) -> gpd.GeoDataFrame:
    """Load real NASA FIRMS hotspots as an EPSG:4326 GeoDataFrame."""
    key=(map_key or os.getenv("NASA_FIRMS_MAP_KEY") or "").strip()
    if not key:
        raise ExternalAPIError("NASA_FIRMS_MAP_KEY is required. Request a free key from NASA FIRMS.")
    if not (1 <= int(days) <= 5):
        raise ValueError("FIRMS day range must be between 1 and 5")
    if west >= east or south >= north:
        raise ValueError("Invalid bounding box")
    area=f"{west},{south},{east},{north}"
    url=f"{NASA_FIRMS_BASE}/area/csv/{key}/{source}/{area}/{int(days)}"
    if date:
        url += f"/{date}"
    try:
        with httpx.Client(timeout=60,follow_redirects=True) as client:
            r=client.get(url)
            r.raise_for_status()
    except httpx.HTTPError as e:
        raise ExternalAPIError(f"NASA FIRMS request failed: {e}") from e
    text=r.text.strip()
    if not text or text.lower().startswith("error"):
        raise ExternalAPIError(text or "NASA FIRMS returned an empty response")
    try:
        df=pd.read_csv(io.StringIO(text))
    except Exception as e:
        raise ExternalAPIError("NASA FIRMS returned a response that could not be parsed as CSV") from e
    if df.empty:
        return gpd.GeoDataFrame(df,geometry=[],crs="EPSG:4326")
    lat_col=next((c for c in df.columns if c.lower()=="latitude"),None)
    lon_col=next((c for c in df.columns if c.lower()=="longitude"),None)
    if not lat_col or not lon_col:
        raise ExternalAPIError("NASA FIRMS response does not include latitude/longitude")
    df.columns=[c.upper() for c in df.columns]
    lat_col=lat_col.upper(); lon_col=lon_col.upper()
    if "BRIGHT_TI4" in df.columns and "BRIGHTNESS" not in df.columns:
        df["BRIGHTNESS"]=pd.to_numeric(df["BRIGHT_TI4"],errors="coerce")
    if "BRIGHT_TI5" in df.columns and "BRIGHT_T31" not in df.columns:
        df["BRIGHT_T31"]=pd.to_numeric(df["BRIGHT_TI5"],errors="coerce")
    gdf=gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(pd.to_numeric(df[lon_col],errors="coerce"),pd.to_numeric(df[lat_col],errors="coerce"))],
        crs="EPSG:4326",
    )
    gdf["_source_file"]="NASA_FIRMS_API"
    return gdf


def fetch_open_meteo_history(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch real historical weather from Open-Meteo for one coordinate."""
    params={
        "latitude":latitude,"longitude":longitude,
        "start_date":start_date,"end_date":end_date,
        "hourly":"temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone":"UTC",
    }
    try:
        with httpx.Client(timeout=45) as client:
            r=client.get(OPEN_METEO_ARCHIVE,params=params)
            r.raise_for_status()
            data=r.json()
    except httpx.HTTPError as e:
        raise ExternalAPIError(f"Open-Meteo request failed: {e}") from e
    hourly=data.get("hourly") or {}
    if not hourly.get("time"):
        raise ExternalAPIError("Open-Meteo returned no hourly data")
    return pd.DataFrame(hourly)


def sample_weather_enrichment(
    gdf: gpd.GeoDataFrame,
    max_points: int=25,
) -> dict:
    """Enrich a geospatial dataset with a bounded sample of real weather observations.

    This is deliberately bounded so a large geospatial analysis does not trigger
    thousands of external API calls.
    """
    if "ACQ_DATE" not in gdf.columns or gdf.empty:
        return {"status":"skipped","reason":"ACQ_DATE unavailable","samples":[]}
    geo=gdf.to_crs(4326)
    valid=geo[geo.geometry.notna() & ~geo.geometry.is_empty].copy()
    if valid.empty:
        return {"status":"skipped","reason":"no valid geometry","samples":[]}
    take=min(max_points,len(valid))
    sample=valid.sample(take,random_state=42) if len(valid)>take else valid
    samples=[]
    for idx,row in sample.iterrows():
        date=str(pd.to_datetime(row["ACQ_DATE"],errors="coerce").date())
        if date=="NaT":
            continue
        try:
            weather=fetch_open_meteo_history(float(row.geometry.y),float(row.geometry.x),date,date)
            samples.append({
                "row_id":int(idx),"latitude":float(row.geometry.y),"longitude":float(row.geometry.x),
                "date":date,
                "temperature_2m_mean":float(pd.to_numeric(weather["temperature_2m"],errors="coerce").mean()),
                "relative_humidity_2m_mean":float(pd.to_numeric(weather["relative_humidity_2m"],errors="coerce").mean()),
                "precipitation_sum":float(pd.to_numeric(weather["precipitation"],errors="coerce").sum()),
                "wind_speed_10m_mean":float(pd.to_numeric(weather["wind_speed_10m"],errors="coerce").mean()),
            })
        except Exception as e:
            samples.append({"row_id":int(idx),"date":date,"error":str(e)})
    return {"status":"ok","provider":"Open-Meteo","samples":samples}
