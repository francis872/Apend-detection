from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from shapely.geometry import shape

from .integrations import ExternalAPIError

EARTH_SEARCH="https://earth-search.aws.element84.com/v1/search"
PLANETARY_COMPUTER="https://planetarycomputer.microsoft.com/api/stac/v1/search"

SENSORS={
    "sentinel-2-l2a":{
        "provider":"Earth Search",
        "endpoint":EARTH_SEARCH,
        "collection":"sentinel-2-l2a",
        "bands":{"red":"red","green":"green","blue":"blue","nir":"nir","swir16":"swir16","swir22":"swir22"},
        "indices":{"NDVI":["nir","red"],"NDMI":["nir","swir16"],"NBR":["nir","swir22"],"NDWI":["green","nir"]},
        "quality_asset":"scl",
    },
    "landsat-c2-l2":{
        "provider":"Microsoft Planetary Computer",
        "endpoint":PLANETARY_COMPUTER,
        "collection":"landsat-c2-l2",
        "bands":{"red":"red","green":"green","blue":"blue","nir":"nir08","swir16":"swir16","swir22":"swir22","thermal":"lwir11"},
        "indices":{"NDVI":["nir","red"],"NDMI":["nir","swir16"],"NBR":["nir","swir22"],"NDWI":["green","nir"],"LST":["thermal"]},
        "quality_asset":"qa_pixel",
    },
    "modis-09A1-061":{
        "provider":"Microsoft Planetary Computer",
        "endpoint":PLANETARY_COMPUTER,
        "collection":"modis-09A1-061",
        "bands":{"red":"sur_refl_b01","nir":"sur_refl_b02","blue":"sur_refl_b03","green":"sur_refl_b04","swir16":"sur_refl_b06","swir22":"sur_refl_b07"},
        "indices":{"NDVI":["nir","red"],"NDMI":["nir","swir16"],"NBR":["nir","swir22"],"NDWI":["green","nir"]},
    },
}

INDEX_FORMULAS={
    "NDVI":"(NIR - RED) / (NIR + RED)",
    "NDMI":"(NIR - SWIR16) / (NIR + SWIR16)",
    "NBR":"(NIR - SWIR22) / (NIR + SWIR22)",
    "NDWI":"(GREEN - NIR) / (GREEN + NIR)",
    "LST":"Provider-specific thermal surface temperature product/scaling; not synthesized from RGB reflectance.",
}


def sensor_catalog()->list[dict]:
    return [{"id":k,"provider":v["provider"],"collection":v["collection"],"indices":sorted(v["indices"])} for k,v in SENSORS.items()]


def index_catalog()->dict:
    return {"formulas":INDEX_FORMULAS,"note":"Formulas define derived-index semantics. Meridian does not silently compute an index unless required source bands and scaling metadata are available."}


def _bbox_from_geometry(geometry:dict)->list[float]:
    geom=shape(geometry)
    if geom.is_empty: raise ValueError("Geometry is empty")
    minx,miny,maxx,maxy=[float(v) for v in geom.bounds]
    if not (-180<=minx<maxx<=180 and -90<=miny<maxy<=90):
        raise ValueError("Geometry bounds are outside WGS84 limits")
    return [minx,miny,maxx,maxy]


def _normalize_item(item:dict,sensor_id:str)->dict:
    props=item.get("properties") or {}
    assets=item.get("assets") or {}
    cfg=SENSORS[sensor_id]
    wanted=set(cfg["bands"].values())
    if cfg.get("quality_asset"): wanted.add(cfg["quality_asset"])
    selected={k:{"href":v.get("href"),"type":v.get("type"),"roles":v.get("roles")} for k,v in assets.items() if k in wanted}
    cloud=props.get("eo:cloud_cover")
    return {
        "id":item.get("id"),
        "sensor":sensor_id,
        "collection":item.get("collection"),
        "datetime":props.get("datetime") or props.get("start_datetime"),
        "cloud_cover":float(cloud) if isinstance(cloud,(int,float)) else None,
        "bbox":item.get("bbox"),
        "assets":selected,
        "available_indices":[name for name,bands in cfg["indices"].items() if all(cfg["bands"].get(b) in assets for b in bands)],
        "stac_self":next((x.get("href") for x in item.get("links",[]) if x.get("rel")=="self"),None),
    }


def search_scenes(sensor_id:str,bbox:list[float],start_date:str,end_date:str,cloud_cover_max:float|None=30,limit:int=20)->dict:
    if sensor_id not in SENSORS: raise ValueError(f"Unsupported earth-observation sensor: {sensor_id}")
    if len(bbox)!=4: raise ValueError("bbox must contain west,south,east,north")
    west,south,east,north=[float(x) for x in bbox]
    if not (-180<=west<east<=180 and -90<=south<north<=90): raise ValueError("Invalid WGS84 bbox")
    if limit<1 or limit>100: raise ValueError("limit must be between 1 and 100")
    cfg=SENSORS[sensor_id]
    query={}
    if cloud_cover_max is not None: query["eo:cloud_cover"]={"lte":float(cloud_cover_max)}
    payload={"collections":[cfg["collection"]],"bbox":[west,south,east,north],
             "datetime":f"{start_date}T00:00:00Z/{end_date}T23:59:59Z","limit":int(limit)}
    if query: payload["query"]=query
    try:
        with httpx.Client(timeout=60,follow_redirects=True) as client:
            r=client.post(cfg["endpoint"],json=payload)
            r.raise_for_status()
            data=r.json()
    except httpx.HTTPError as e:
        raise ExternalAPIError(f"{cfg['provider']} STAC request failed: {e}") from e
    items=[_normalize_item(x,sensor_id) for x in data.get("features",[])]
    return {
        "sensor":sensor_id,"provider":cfg["provider"],"collection":cfg["collection"],
        "query":{"bbox":[west,south,east,north],"start_date":start_date,"end_date":end_date,
                 "cloud_cover_max":cloud_cover_max,"limit":limit},
        "scenes":items,"count":len(items),
        "provenance":{"retrieved_utc":datetime.now(timezone.utc).isoformat(),"endpoint":cfg["endpoint"],"protocol":"STAC"},
    }


def search_geometry(sensor_id:str,geometry:dict,start_date:str,end_date:str,cloud_cover_max:float|None=30,limit:int=20)->dict:
    return search_scenes(sensor_id,_bbox_from_geometry(geometry),start_date,end_date,cloud_cover_max,limit)


def build_index_plan(sensor_id:str,indices:list[str])->dict:
    if sensor_id not in SENSORS: raise ValueError(f"Unsupported earth-observation sensor: {sensor_id}")
    cfg=SENSORS[sensor_id]
    requested=[str(x).upper() for x in indices]
    unsupported=[x for x in requested if x not in cfg["indices"]]
    if unsupported: raise ValueError("Unsupported indices for sensor: "+", ".join(unsupported))
    plan=[]
    for idx in requested:
        semantic_bands=cfg["indices"][idx]
        plan.append({"index":idx,"formula":INDEX_FORMULAS[idx],"semantic_bands":semantic_bands,
                     "asset_keys":[cfg["bands"][b] for b in semantic_bands]})
    return {"sensor":sensor_id,"provider":cfg["provider"],"collection":cfg["collection"],"indices":plan,
            "policy":{"requires_source_assets":True,"silent_scaling":False,"provenance_required":True}}
