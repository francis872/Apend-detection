from __future__ import annotations

from pathlib import Path
from typing import Iterable
import geopandas as gpd
import pandas as pd


def find_shapefiles(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(root.rglob('*.shp'))


def append_shapefiles(paths: Iterable[str | Path], target_crs: str = 'EPSG:4326') -> gpd.GeoDataFrame:
    """Append many shapefiles into one GeoDataFrame.

    pandas.DataFrame.append was removed in pandas 2.x; pd.concat is the supported
    append operation and is faster for many files.
    """
    frames: list[gpd.GeoDataFrame] = []
    for p in paths:
        p = Path(p)
        gdf = gpd.read_file(p)
        if gdf.crs is None:
            raise ValueError(f'{p} has no CRS')
        gdf = gdf.to_crs(target_crs)
        gdf['_source_file'] = p.name
        frames.append(gdf)
    if not frames:
        raise FileNotFoundError('No shapefiles found')
    merged = pd.concat(frames, ignore_index=True, sort=False)
    return gpd.GeoDataFrame(merged, geometry='geometry', crs=target_crs)


def load_and_append(root: str | Path, target_crs: str = 'EPSG:4326') -> gpd.GeoDataFrame:
    return append_shapefiles(find_shapefiles(root), target_crs=target_crs)


def basic_clean(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    before = len(gdf)
    x = gdf.copy()
    x = x[x.geometry.notna() & ~x.geometry.is_empty].copy()
    invalid = ~x.geometry.is_valid
    if invalid.any():
        x.loc[invalid, 'geometry'] = x.loc[invalid, 'geometry'].buffer(0)
    dedup_cols = [c for c in ['LATITUDE','LONGITUDE','ACQ_DATE','ACQ_TIME','SATELLITE','FRP','BRIGHTNESS'] if c in x.columns]
    if dedup_cols:
        x = x.drop_duplicates(subset=dedup_cols, keep='first')
    numeric = x.select_dtypes(include='number').columns
    x[numeric] = x[numeric].replace([float('inf'), float('-inf')], pd.NA)
    report = {
        'rows_before': before,
        'rows_after_basic_clean': len(x),
        'rows_removed': before-len(x),
        'invalid_geometry_fixed': int(invalid.sum()),
    }
    return x.reset_index(drop=True), report
