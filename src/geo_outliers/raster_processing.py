from __future__ import annotations

from datetime import datetime, timezone
import math

import numpy as np
from shapely.geometry import shape, mapping

from .earth_observation import SENSORS
from .integrations import ExternalAPIError


def _safe_ratio(a:np.ndarray,b:np.ndarray)->np.ndarray:
    den=a+b
    out=np.full(a.shape,np.nan,dtype="float32")
    valid=np.isfinite(a)&np.isfinite(b)&(np.abs(den)>1e-8)
    out[valid]=(a[valid]-b[valid])/den[valid]
    return out


def compute_spectral_indices(bands:dict[str,np.ndarray], indices:list[str])->dict[str,np.ndarray]:
    """Compute EO indices from already scaled semantic bands."""
    out={}
    for name in [str(x).upper() for x in indices]:
        if name=="NDVI":out[name]=_safe_ratio(bands["nir"],bands["red"])
        elif name=="NDMI":out[name]=_safe_ratio(bands["nir"],bands["swir16"])
        elif name=="NBR":out[name]=_safe_ratio(bands["nir"],bands["swir22"])
        elif name=="NDWI":out[name]=_safe_ratio(bands["green"],bands["nir"])
        elif name=="LST":
            if "thermal" not in bands:raise ValueError("LST requires a thermal band")
            out[name]=bands["thermal"].astype("float32")
        else:raise ValueError(f"Unsupported index: {name}")
    return out


def scale_band(sensor_id:str,semantic_band:str,array:np.ndarray)->np.ndarray:
    """Apply documented product-level scale/offset rules used by Meridian."""
    a=array.astype("float32")
    if sensor_id=="sentinel-2-l2a":
        if semantic_band in {"red","green","blue","nir","swir16","swir22"}:return a*0.0001
    elif sensor_id=="landsat-c2-l2":
        if semantic_band in {"red","green","blue","nir","swir16","swir22"}:return a*0.0000275-0.2
        if semantic_band=="thermal":return a*0.00341802+149.0-273.15
    elif sensor_id=="modis-09A1-061":
        if semantic_band in {"red","green","blue","nir","swir16","swir22"}:return a*0.0001
    return a


def cloud_mask(sensor_id:str,qa:np.ndarray|None)->tuple[np.ndarray|None,dict]:
    if qa is None:
        return None,{"applied":False,"reason":"quality asset unavailable"}
    q=np.nan_to_num(qa,nan=0).astype("uint32")
    if sensor_id=="sentinel-2-l2a":
        invalid=np.isin(q,[0,1,3,8,9,10,11])
        return ~invalid,{"applied":True,"method":"Sentinel-2 SCL classes","invalid_classes":[0,1,3,8,9,10,11]}
    if sensor_id=="landsat-c2-l2":
        invalid=np.zeros(q.shape,dtype=bool)
        for bit in (0,1,2,3,4,5):invalid|=((q>>bit)&1).astype(bool)
        return ~invalid,{"applied":True,"method":"Landsat QA_PIXEL bits","invalid_bits":[0,1,2,3,4,5]}
    return None,{"applied":False,"reason":"cloud-mask semantics not configured for sensor"}


def zonal_statistics(array:np.ndarray,valid_mask:np.ndarray|None=None)->dict:
    a=np.asarray(array,float)
    valid=np.isfinite(a)
    if valid_mask is not None:valid&=np.asarray(valid_mask,bool)
    vals=a[valid]
    if not len(vals):
        return {"count":0,"mean":None,"median":None,"std":None,"min":None,"max":None,"p05":None,"p95":None}
    return {
        "count":int(len(vals)),"mean":float(np.mean(vals)),"median":float(np.median(vals)),
        "std":float(np.std(vals)),"min":float(np.min(vals)),"max":float(np.max(vals)),
        "p05":float(np.quantile(vals,.05)),"p95":float(np.quantile(vals,.95)),
    }


def _signed_href(scene:dict,href:str)->str:
    if scene.get("provider")!="Microsoft Planetary Computer":return href
    try:
        import httpx
        r=httpx.get("https://planetarycomputer.microsoft.com/api/sas/v1/sign",params={"href":href},timeout=30)
        r.raise_for_status()
        return r.json().get("href",href)
    except Exception as e:
        raise ExternalAPIError(f"Planetary Computer asset signing failed: {e}") from e


def process_scene(
    scene:dict,
    geometry:dict,
    indices:list[str],
    max_pixels:int=1_000_000,
)->dict:
    """Read only the territorial COG window, align required bands, mask clouds and compute zonal EO features."""
    try:
        import rasterio
        from affine import Affine
        from rasterio.features import geometry_mask
        from rasterio.vrt import WarpedVRT
        from rasterio.warp import transform_geom
        from rasterio.windows import from_bounds, Window
        from rasterio.enums import Resampling
    except ImportError as e:
        raise RuntimeError("Raster Processing requires rasterio") from e

    sensor_id=scene.get("sensor")
    if sensor_id not in SENSORS:raise ValueError("Unsupported sensor")
    cfg=SENSORS[sensor_id]
    requested=[str(x).upper() for x in indices]
    needed=set()
    for idx in requested:
        if idx not in cfg["indices"]:raise ValueError(f"{idx} is not supported by {sensor_id}")
        needed.update(cfg["indices"][idx])
    assets=scene.get("assets") or {}
    missing=[b for b in needed if cfg["bands"].get(b) not in assets or not (assets[cfg["bands"][b]] or {}).get("href")]
    if missing:raise ValueError("Scene is missing required semantic bands: "+", ".join(sorted(missing)))

    geom=shape(geometry)
    if geom.is_empty:raise ValueError("Geometry is empty")
    first_sem=next(iter(sorted(needed)))
    ref_key=cfg["bands"][first_sem]
    ref_href=_signed_href(scene,assets[ref_key]["href"])

    with rasterio.Env(GDAL_HTTP_MULTIRANGE="YES",GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(ref_href) as ref:
            if ref.crs is None:raise ValueError("Reference raster has no CRS")
            geom_ref=transform_geom("EPSG:4326",ref.crs.to_string(),mapping(geom),precision=6)
            bounds=shape(geom_ref).bounds
            win=from_bounds(*bounds,transform=ref.transform).round_offsets().round_lengths()
            full=Window(0,0,ref.width,ref.height)
            win=win.intersection(full)
            if win.width<=0 or win.height<=0:raise ValueError("Territory does not intersect raster scene")
            pixels=float(win.width*win.height)
            factor=max(1.0,math.sqrt(pixels/max(1,int(max_pixels))))
            height=max(1,int(math.ceil(win.height/factor)));width=max(1,int(math.ceil(win.width/factor)))
            transform=ref.window_transform(win)*Affine.scale(win.width/width,win.height/height)
            territory_mask=geometry_mask([geom_ref],out_shape=(height,width),transform=transform,invert=True)

            semantic={}
            for sem in sorted(needed):
                key=cfg["bands"][sem]; href=_signed_href(scene,assets[key]["href"])
                with rasterio.open(href) as src:
                    with WarpedVRT(src,crs=ref.crs,transform=transform,width=width,height=height,resampling=Resampling.bilinear) as vrt:
                        arr=vrt.read(1,masked=True).filled(np.nan)
                semantic[sem]=scale_band(sensor_id,sem,arr)

            qa=None; qa_meta={"applied":False,"reason":"quality asset unavailable"}
            qkey=cfg.get("quality_asset")
            if qkey and qkey in assets and (assets[qkey] or {}).get("href"):
                href=_signed_href(scene,assets[qkey]["href"])
                with rasterio.open(href) as src:
                    with WarpedVRT(src,crs=ref.crs,transform=transform,width=width,height=height,resampling=Resampling.nearest) as vrt:
                        qa=vrt.read(1,masked=True).filled(0)
                valid_cloud,qa_meta=cloud_mask(sensor_id,qa)
            else:
                valid_cloud=None

    valid=territory_mask.copy()
    if valid_cloud is not None:valid&=valid_cloud
    derived=compute_spectral_indices(semantic,requested)
    stats={name:zonal_statistics(arr,valid) for name,arr in derived.items()}
    feature_vector={name:vals["mean"] for name,vals in stats.items() if vals["mean"] is not None}
    return {
        "scene_id":scene.get("id"),"sensor":sensor_id,"datetime":scene.get("datetime"),
        "indices":stats,"feature_vector":feature_vector,
        "pixels":{"window":int(width*height),"valid":int(valid.sum()),"max_pixels":int(max_pixels),"downsample_factor":float(factor)},
        "cloud_mask":qa_meta,
        "provenance":{"processed_utc":datetime.now(timezone.utc).isoformat(),"source_assets":sorted([cfg["bands"][x] for x in needed]),
                      "spatial_scope":"territorial window","engine":"rasterio/COG"},
        "policy":{"remote_window_only":True,"distance_is_not_probability":True,"causal_claims":False},
    }
