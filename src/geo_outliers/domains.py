from __future__ import annotations
from dataclasses import dataclass, asdict

DEFAULT_WEIGHTS={"score_mahal":.20,"score_probability":.25,"score_spd":.18,"score_procrustes":.10,"score_spatial":.17,"score_temporal":.10}

@dataclass(frozen=True)
class DomainPack:
    id:str
    name:str
    description:str
    variables:tuple[str,...]
    sources:tuple[str,...]
    use_cases:tuple[str,...]
    governance:str="standard"
    aliases:tuple[tuple[str,...],...]=()
    weights:tuple[tuple[str,float],...]=()

PACKS={
 "general":DomainPack("general","General Geospatial","Domain-neutral geospatial anomaly intelligence",(),("files","api","database"),("anomaly_detection","regional_comparison"),aliases=(),weights=tuple(DEFAULT_WEIGHTS.items())),
 "environment":DomainPack("environment","Environment & Climate","Environmental change, fire, vegetation, drought, pollution and water intelligence",("FRP","NDVI","soil_moisture","surface_temperature","precipitation","air_quality","water_quality"),("satellite","weather","sensors"),("wildfire","deforestation","drought","pollution","ecosystem_change"),aliases=(("FRP","frp","fire_radiative_power"),("NDVI","ndvi"),("soil_moisture","soil moisture","sm"),("surface_temperature","lst","land_surface_temperature"),("precipitation","rainfall","rain")),weights=(("score_probability",.22),("score_spatial",.23),("score_temporal",.18),("score_mahal",.17),("score_spd",.12),("score_procrustes",.08))),
 "agriculture":DomainPack("agriculture","Precision Agriculture","Field-scale crop, soil and sensor intelligence",("NDVI","soil_moisture","surface_temperature","precipitation","yield"),("satellite","weather","iot"),("crop_stress","irrigation","yield_anomaly"),aliases=(("NDVI","ndvi"),("soil_moisture","soil moisture","moisture"),("surface_temperature","lst","temperature"),("precipitation","rainfall","rain"),("yield","productivity","production")),weights=(("score_probability",.20),("score_spatial",.25),("score_temporal",.20),("score_mahal",.15),("score_spd",.12),("score_procrustes",.08))),
 "risk":DomainPack("risk","Risk & Disaster","Spatial-temporal surveillance for emerging hazards",("FRP","precipitation","soil_moisture","water_level","slope"),("satellite","weather","terrain","sensors"),("wildfire","flood","landslide","drought"),aliases=(("FRP","frp"),("precipitation","rainfall","rain"),("soil_moisture","moisture"),("water_level","river_level"),("slope","terrain_slope")),weights=(("score_probability",.18),("score_spatial",.27),("score_temporal",.25),("score_mahal",.12),("score_spd",.10),("score_procrustes",.08))),
 "urban":DomainPack("urban","Smart Cities","Territorial anomaly intelligence for urban operations",("traffic","noise","air_quality","energy","waste","accidents"),("iot","mobility","municipal"),("traffic","pollution","mobility","urban_operations"),weights=(("score_probability",.18),("score_spatial",.24),("score_temporal",.22),("score_mahal",.16),("score_spd",.12),("score_procrustes",.08))),
 "real_estate":DomainPack("real_estate","Real Estate","Territorial market and property intelligence",("price_m2","rent","accessibility","noise","security","density"),("listings","maps","territorial"),("pricing_anomaly","neighbourhood_change","site_selection"),aliases=(("price_m2","price_per_m2","precio_m2"),("rent","rental","canon","arriendo"),("accessibility","access"),("density","densidad")),weights=(("score_probability",.24),("score_spatial",.25),("score_temporal",.12),("score_mahal",.18),("score_spd",.13),("score_procrustes",.08))),
 "infrastructure":DomainPack("infrastructure","Infrastructure","Condition and failure intelligence for distributed assets",("temperature","vibration","pressure","flow","faults"),("iot","scada","asset_registry"),("failure_concentration","condition_change","inspection_priority"),weights=(("score_probability",.18),("score_spatial",.20),("score_temporal",.25),("score_mahal",.17),("score_spd",.12),("score_procrustes",.08))),
 "logistics":DomainPack("logistics","Logistics & Transport","Fleet, route and cold-chain intelligence",("speed","travel_time","fuel","cargo_temperature","delay"),("telemetry","gps","iot"),("route_anomaly","fleet_anomaly","cold_chain"),weights=(("score_probability",.18),("score_spatial",.20),("score_temporal",.24),("score_mahal",.18),("score_spd",.12),("score_procrustes",.08))),
 "energy":DomainPack("energy","Energy","Generation, consumption and grid anomaly intelligence",("generation","consumption","voltage","frequency","weather"),("scada","smart_meter","weather"),("generation_anomaly","grid_anomaly","regional_comparison"),weights=(("score_probability",.22),("score_spatial",.18),("score_temporal",.24),("score_mahal",.18),("score_spd",.10),("score_procrustes",.08))),
 "public_health":DomainPack("public_health","Territorial Public Health","Privacy-aware aggregated spatial-temporal health intelligence",("incidence","service_access","hospital_demand"),("aggregated_health","territorial"),("epidemiology","access","capacity"),"restricted",weights=(("score_probability",.18),("score_spatial",.27),("score_temporal",.25),("score_mahal",.14),("score_spd",.10),("score_procrustes",.06))),
 "earth_observation":DomainPack("earth_observation","Earth Observation & Aerospace","Satellite, drone and ground-station observation intelligence",("reflectance","temperature","FRP","NDVI","elevation"),("satellite","drone","ground_station"),("change_detection","mission_observation","sensor_anomaly"),weights=(("score_probability",.20),("score_spatial",.24),("score_temporal",.18),("score_mahal",.16),("score_spd",.14),("score_procrustes",.08))),
}

def _public(pack:DomainPack):
    d=asdict(pack); d["weights"]=dict(pack.weights) if pack.weights else DEFAULT_WEIGHTS.copy(); d["aliases"]=[list(x) for x in pack.aliases]; return d

def list_domain_packs(): return [_public(x) for x in PACKS.values()]

def get_domain_pack(pack_id:str):
    if pack_id not in PACKS: raise KeyError(pack_id)
    return _public(PACKS[pack_id])

def configure_domain(pack_id:str, numeric_columns:list[str], requested_feature:str|None=None, requested_variables:list[str]|None=None, custom_weights:dict|None=None):
    pack=PACKS.get(pack_id)
    if pack is None: raise KeyError(pack_id)
    lookup={str(c).lower():c for c in numeric_columns}
    recommended=[]
    groups=pack.aliases or tuple((v,v.lower()) for v in pack.variables)
    for group in groups:
        hit=next((lookup[a.lower()] for a in group if a.lower() in lookup),None)
        if hit and hit not in recommended: recommended.append(hit)
    requested=[v for v in (requested_variables or []) if v in numeric_columns]
    variables=list(dict.fromkeys(requested+recommended))
    if len(variables)<2: variables=list(dict.fromkeys(variables+numeric_columns))[:8]
    feature=requested_feature if requested_feature in numeric_columns else (recommended[0] if recommended else (numeric_columns[0] if numeric_columns else None))
    weights=DEFAULT_WEIGHTS.copy(); weights.update(dict(pack.weights)); weights.update(custom_weights or {})
    return {"domain":pack_id,"pack":_public(pack),"probability_feature":feature,"variables":variables[:8],"weights":weights,"matched_variables":recommended,"governance":pack.governance}
