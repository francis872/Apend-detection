from __future__ import annotations
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class DomainPack:
    id:str
    name:str
    description:str
    variables:tuple[str,...]
    sources:tuple[str,...]
    use_cases:tuple[str,...]
    governance:str="standard"

PACKS={
 "environment":DomainPack("environment","Environment & Climate","Environmental change, fire, vegetation, drought, pollution and water intelligence",("FRP","NDVI","soil_moisture","surface_temperature","precipitation","air_quality","water_quality"),("satellite","weather","sensors"),("wildfire","deforestation","drought","pollution","ecosystem_change")),
 "agriculture":DomainPack("agriculture","Precision Agriculture","Field-scale crop, soil and sensor intelligence",("NDVI","soil_moisture","surface_temperature","precipitation","yield"),("satellite","weather","iot"),("crop_stress","irrigation","yield_anomaly")),
 "risk":DomainPack("risk","Risk & Disaster","Spatial-temporal surveillance for emerging hazards",("FRP","precipitation","soil_moisture","water_level","slope"),("satellite","weather","terrain","sensors"),("wildfire","flood","landslide","drought")),
 "urban":DomainPack("urban","Smart Cities","Territorial anomaly intelligence for urban operations",("traffic","noise","air_quality","energy","waste","accidents"),("iot","mobility","municipal"),("traffic","pollution","mobility","urban_operations")),
 "real_estate":DomainPack("real_estate","Real Estate","Territorial market and property intelligence",("price_m2","rent","accessibility","noise","security","density"),("listings","maps","territorial"),("pricing_anomaly","neighbourhood_change","site_selection")),
 "infrastructure":DomainPack("infrastructure","Infrastructure","Condition and failure intelligence for distributed assets",("temperature","vibration","pressure","flow","faults"),("iot","scada","asset_registry"),("failure_concentration","condition_change","inspection_priority")),
 "logistics":DomainPack("logistics","Logistics & Transport","Fleet, route and cold-chain intelligence",("speed","travel_time","fuel","cargo_temperature","delay"),("telemetry","gps","iot"),("route_anomaly","fleet_anomaly","cold_chain")),
 "energy":DomainPack("energy","Energy","Generation, consumption and grid anomaly intelligence",("generation","consumption","voltage","frequency","weather"),("scada","smart_meter","weather"),("generation_anomaly","grid_anomaly","regional_comparison")),
 "public_health":DomainPack("public_health","Territorial Public Health","Privacy-aware aggregated spatial-temporal health intelligence",("incidence","service_access","hospital_demand"),("aggregated_health","territorial"),("epidemiology","access","capacity"),"restricted"),
 "earth_observation":DomainPack("earth_observation","Earth Observation & Aerospace","Satellite, drone and ground-station observation intelligence",("reflectance","temperature","FRP","NDVI","elevation"),("satellite","drone","ground_station"),("change_detection","mission_observation","sensor_anomaly")),
}

def list_domain_packs():
    return [asdict(x) for x in PACKS.values()]

def get_domain_pack(pack_id:str):
    if pack_id not in PACKS: raise KeyError(pack_id)
    return asdict(PACKS[pack_id])
