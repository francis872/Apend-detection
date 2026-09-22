from __future__ import annotations

import io
import os
from datetime import date as date_type

import geopandas as gpd
import httpx
import pandas as pd
from shapely.geometry import Point

NASA_FIRMS_BASE="https://firms.modaps.eosdis.nasa.gov/api"
OPEN_METEO_ARCHIVE="https://archive-api.open-meteo.com/v1/archive"


class ExternalAPIError(RuntimeError):
    pass


def integration_status() -> dict:
    """Report configured real-data integrations without exposing credentials."""
    return {
        "NASA FIRMS": {
            "configured": bool((os.getenv("NASA_FIRMS_MAP_KEY") or "").strip()),
            "requires_key": True,
            "purpose": "Incendios y anomalías térmicas satelitales VIIRS/MODIS",
        },
        "Open-Meteo": {
            "configured": True,
            "requires_key": False,
            "purpose": "Contexto meteorológico histórico para observaciones geográficas",
        },
    }


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
        raise ExternalAPIError(
            "NASA_FIRMS_MAP_KEY is required. Configure the free NASA FIRMS MAP_KEY "
            "as an environment variable before using the live source."
        )
    days=int(days)
    if not (1 <= days <= 5):
        raise ValueError("FIRMS day range must be between 1 and 5")
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Invalid FIRMS bounding box")

    allowed={
        "LANDSAT_NRT","MODIS_NRT","MODIS_SP","VIIRS_NOAA20_NRT",
        "VIIRS_NOAA20_SP","VIIRS_NOAA21_NRT","VIIRS_SNPP_NRT","VIIRS_SNPP_SP"
    }
    if source not in allowed:
        raise ValueError(f"Unsupported FIRMS source: {source}")

    area=f"{west},{south},{east},{north}"
    url=f"{NASA_FIRMS_BASE}/area/csv/{key}/{source}/{area}/{days}"
    if date:
        # Basic validation keeps malformed values away from the upstream API.
        pd.Timestamp(date)
        url += f"/{date}"

    try:
        with httpx.Client(timeout=90,follow_redirects=True) as client:
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
        raise ExternalAPIError("NASA FIRMS returned invalid CSV") from e

    if df.empty:
        return gpd.GeoDataFrame(df,geometry=gpd.GeoSeries([],crs="EPSG:4326"),crs="EPSG:4326")

    df.columns=[str(c).upper() for c in df.columns]
    if "LATITUDE" not in df.columns or "LONGITUDE" not in df.columns:
        raise ExternalAPIError("NASA FIRMS response does not include latitude/longitude")

    # Normalize VIIRS fields to the names already used by Apend Detection.
    if "BRIGHT_TI4" in df.columns and "BRIGHTNESS" not in df.columns:
        df["BRIGHTNESS"]=pd.to_numeric(df["BRIGHT_TI4"],errors="coerce")
    if "BRIGHT_TI5" in df.columns and "BRIGHT_T31" not in df.columns:
        df["BRIGHT_T31"]=pd.to_numeric(df["BRIGHT_TI5"],errors="coerce")
    for col in ("LATITUDE","LONGITUDE","FRP","SCAN","TRACK","BRIGHTNESS","BRIGHT_T31"):
        if col in df.columns:
            df[col]=pd.to_numeric(df[col],errors="coerce")

    valid=df["LATITUDE"].notna() & df["LONGITUDE"].notna()
    df=df.loc[valid].copy()
    gdf=gpd.GeoDataFrame(
        df,
        geometry=[Point(x,y) for x,y in zip(df["LONGITUDE"],df["LATITUDE"])],
        crs="EPSG:4326",
    )
    gdf["_source_file"]="NASA_FIRMS_API"
    gdf["_api_source"]=source
    return gdf.reset_index(drop=True)


def fetch_nasa_firms(
    west: float,
    south: float,
    east: float,
    north: float,
    days: int=1,
    source: str="VIIRS_NOAA21_NRT",
    date: str | None=None,
) -> list[dict]:
    """JSON-friendly preview used by the public API endpoint."""
    gdf=fetch_firms_area(west,south,east,north,days,source,date)
    cols=[c for c in ("LATITUDE","LONGITUDE","FRP","BRIGHTNESS","BRIGHT_T31","ACQ_DATE","ACQ_TIME","SATELLITE","CONFIDENCE") if c in gdf.columns]
    return gdf[cols].head(2000).where(pd.notna(gdf[cols]),None).to_dict(orient="records")


def fetch_open_meteo_history(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch real historical weather from Open-Meteo for one coordinate."""
    params={
        "latitude":float(latitude),"longitude":float(longitude),
        "start_date":start_date,"end_date":end_date,
        "hourly":"temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone":"UTC",
    }
    try:
        with httpx.Client(timeout=45,follow_redirects=True) as client:
            r=client.get(OPEN_METEO_ARCHIVE,params=params)
            r.raise_for_status()
            data=r.json()
    except httpx.HTTPError as e:
        raise ExternalAPIError(f"Open-Meteo request failed: {e}") from e
    hourly=data.get("hourly") or {}
    if not hourly.get("time"):
        raise ExternalAPIError("Open-Meteo returned no hourly data")
    return pd.DataFrame(hourly)


def fetch_open_meteo(latitude: float, longitude: float, day: str | None=None) -> dict:
    """JSON-friendly daily weather summary for UI/API use."""
    target=day or str(date_type.today())
    df=fetch_open_meteo_history(latitude,longitude,target,target)
    return {
        "latitude":float(latitude),"longitude":float(longitude),"date":target,
        "temperature_2m_mean":float(pd.to_numeric(df["temperature_2m"],errors="coerce").mean()),
        "relative_humidity_2m_mean":float(pd.to_numeric(df["relative_humidity_2m"],errors="coerce").mean()),
        "precipitation_sum":float(pd.to_numeric(df["precipitation"],errors="coerce").sum()),
        "wind_speed_10m_mean":float(pd.to_numeric(df["wind_speed_10m"],errors="coerce").mean()),
    }


def sample_weather_enrichment(gdf:gpd.GeoDataFrame,max_points:int=12) -> dict:
    """Bounded real-weather context for a completed geospatial analysis."""
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
        ts=pd.to_datetime(row["ACQ_DATE"],errors="coerce")
        if pd.isna(ts):
            continue
        day=str(ts.date())
        try:
            weather=fetch_open_meteo(float(row.geometry.y),float(row.geometry.x),day)
            weather["row_id"]=int(idx)
            samples.append(weather)
        except Exception as e:
            samples.append({"row_id":int(idx),"date":day,"error":str(e)})
    return {"status":"ok","provider":"Open-Meteo","samples":samples}
