from __future__ import annotations
import geopandas as gpd
from shapely.geometry import shape, mapping

OPS={"intersects","within","contains","covers","covered_by","crosses","touches","overlaps"}

def _gdf(features,crs="EPSG:4326"):
    rows=[]
    for i,f in enumerate(features):
        geom=f.get("geometry")
        if not geom:raise ValueError(f"Feature {i} has no geometry")
        props=dict(f.get("properties") or {})
        props["_feature_id"]=str(f.get("id",i));props["geometry"]=shape(geom);rows.append(props)
    return gpd.GeoDataFrame(rows,geometry="geometry",crs=crs)

def spatial_operation(left_features,right_features,operation="intersects",left_crs="EPSG:4326",right_crs="EPSG:4326",distance_m=None):
    left=_gdf(left_features,left_crs);right=_gdf(right_features,right_crs)
    if operation=="nearest":
        if left.crs.is_geographic:
            crs=left.estimate_utm_crs();lp=left.to_crs(crs);rp=right.to_crs(crs)
        else:lp=left;rp=right.to_crs(left.crs)
        out=gpd.sjoin_nearest(lp,rp,how="left",max_distance=distance_m,distance_col="distance_m").to_crs(left.crs)
    elif operation=="buffer":
        if distance_m is None:raise ValueError("distance_m is required for buffer")
        crs=left.estimate_utm_crs() if left.crs.is_geographic else left.crs
        out=left.to_crs(crs).copy();out.geometry=out.geometry.buffer(float(distance_m));out=out.to_crs(left.crs)
    elif operation=="intersection":
        out=gpd.overlay(left,right.to_crs(left.crs),how="intersection",keep_geom_type=False)
    elif operation=="union":
        out=gpd.overlay(left,right.to_crs(left.crs),how="union",keep_geom_type=False)
    elif operation in OPS:
        out=gpd.sjoin(left,right.to_crs(left.crs),predicate=operation,how="inner")
    else:raise ValueError("Unsupported spatial operation")
    clean=[]
    for _,r in out.iterrows():
        props={k:(v.item() if hasattr(v,"item") else v) for k,v in r.items() if k!="geometry" and k!="index_right"}
        clean.append({"type":"Feature","geometry":mapping(r.geometry),"properties":props})
    return {"type":"FeatureCollection","features":clean,"count":len(clean),"operation":operation,"crs":str(left.crs),
            "methodology":{"causal_claims":False,"distance_unit":"meters" if operation in {"nearest","buffer"} else None}}
