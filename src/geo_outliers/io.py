from __future__ import annotations

from pathlib import Path
from typing import Iterable
import geopandas as gpd
import pandas as pd


SUPPORTED_EXTENSIONS={".shp",".gpkg",".geojson",".json",".csv",".xlsx",".xls",".parquet"}


def find_shapefiles(root: str | Path) -> list[Path]:
    return sorted(Path(root).rglob("*.shp"))


def find_data_files(root: str | Path) -> list[Path]:
    root=Path(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def _find_coord(columns):
    lookup={str(c).strip().lower():c for c in columns}
    lat=next((lookup[x] for x in ("latitude","lat","latitud","y") if x in lookup),None)
    lon=next((lookup[x] for x in ("longitude","lon","lng","long","longitud","x") if x in lookup),None)
    return lat,lon


def _tabular_to_geo(df:pd.DataFrame, source:str, target_crs:str)->gpd.GeoDataFrame:
    lat,lon=_find_coord(df.columns)
    if not lat or not lon:
        raise ValueError(f"{source}: tabular source needs latitude/longitude columns (aliases lat/lon, latitud/longitud or x/y)")
    y=pd.to_numeric(df[lat],errors="coerce"); x=pd.to_numeric(df[lon],errors="coerce")
    valid=x.between(-180,180)&y.between(-90,90)
    df=df.loc[valid].copy()
    if df.empty: raise ValueError(f"{source}: no valid WGS84 coordinates")
    return gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(pd.to_numeric(df[lon]),pd.to_numeric(df[lat])),crs="EPSG:4326").to_crs(target_crs)


def read_data_file(path:str|Path,target_crs:str="EPSG:4326")->gpd.GeoDataFrame:
    p=Path(path); ext=p.suffix.lower()
    if ext in {".shp",".gpkg",".geojson"}:
        gdf=gpd.read_file(p)
    elif ext==".json":
        try:
            gdf=gpd.read_file(p)
            if "geometry" not in gdf.columns: raise ValueError("not geospatial")
        except Exception:
            raw=pd.read_json(p)
            gdf=_tabular_to_geo(raw,p.name,target_crs)
    elif ext==".csv":
        gdf=_tabular_to_geo(pd.read_csv(p),p.name,target_crs)
    elif ext in {".xlsx",".xls"}:
        gdf=_tabular_to_geo(pd.read_excel(p),p.name,target_crs)
    elif ext==".parquet":
        try:gdf=gpd.read_parquet(p)
        except Exception:gdf=_tabular_to_geo(pd.read_parquet(p),p.name,target_crs)
    else:raise ValueError(f"Unsupported source format: {ext}")
    if gdf.crs is None: raise ValueError(f"{p.name} has no CRS")
    gdf=gdf.to_crs(target_crs); gdf["_source_file"]=p.name
    return gdf


def append_data_files(paths:Iterable[str|Path],target_crs:str="EPSG:4326")->gpd.GeoDataFrame:
    frames=[read_data_file(p,target_crs) for p in paths]
    if not frames:raise FileNotFoundError("No supported geospatial data files found")
    merged=pd.concat(frames,ignore_index=True,sort=False)
    return gpd.GeoDataFrame(merged,geometry="geometry",crs=target_crs)


def append_shapefiles(paths: Iterable[str | Path], target_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    return append_data_files(paths,target_crs)


def load_and_append(root: str | Path, target_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    return append_data_files(find_data_files(root),target_crs)


def basic_clean(gdf:gpd.GeoDataFrame)->tuple[gpd.GeoDataFrame,dict]:
    before=len(gdf); x=gdf.copy()
    x=x[x.geometry.notna() & ~x.geometry.is_empty].copy()
    invalid=~x.geometry.is_valid
    if invalid.any():x.loc[invalid,"geometry"]=x.loc[invalid,"geometry"].buffer(0)
    dedup_cols=[c for c in ["LATITUDE","LONGITUDE","latitude","longitude","ACQ_DATE","ACQ_TIME","SATELLITE","FRP","BRIGHTNESS"] if c in x.columns]
    if dedup_cols:x=x.drop_duplicates(subset=dedup_cols,keep="first")
    numeric=x.select_dtypes(include="number").columns
    x[numeric]=x[numeric].replace([float("inf"),float("-inf")],pd.NA)
    return x.reset_index(drop=True),{"rows_before":before,"rows_after_basic_clean":len(x),"rows_removed":before-len(x),"invalid_geometry_fixed":int(invalid.sum())}
