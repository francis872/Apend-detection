from dataclasses import asdict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from meridian_core.spatial import SpatialEngine, SpatialPoint
from meridian_core.corridors import CorridorEngine
from meridian_core.baseline import BaselineEngine
from meridian_core.risk import RiskEngine
from meridian_core.forecasting import ForecastingEngine

app=FastAPI(title="Meridian API",version="1.0.0")
spatial=SpatialEngine(); corridors=CorridorEngine(); risk=RiskEngine(); forecast=ForecastingEngine()

class Node(BaseModel):
 id:str
 latitude:float=Field(ge=-90,le=90)
 longitude:float=Field(ge=-180,le=180)
 kind:str="property"
class Signal(BaseModel): value:float
class PropertyIntelligenceRequest(BaseModel):
 propertyId:str
 latitude:float=Field(ge=-90,le=90)
 longitude:float=Field(ge=-180,le=180)
 radiusKm:float=Field(default=5,gt=0,le=100)
 contextNodes:list[Node]=[]
 history:list[Signal]=[]

@app.get("/api/v1/health")
def health(): return {"status":"ok","service":"meridian-api","version":"1.0.0"}

@app.get("/api/v1/version")
def version(): return {"name":"Meridian Core","version":"1.0.0","engines":["spatial","baseline","corridors","risk","forecasting","gwo"]}

@app.post("/api/v1/properties/intelligence")
def property_intelligence(r:PropertyIntelligenceRequest):
 target=SpatialPoint(r.propertyId,r.latitude,r.longitude,"property")
 points=[target]
 seen={r.propertyId}
 for n in r.contextNodes:
  if n.id not in seen:
   seen.add(n.id);points.append(SpatialPoint(n.id,n.latitude,n.longitude,n.kind))
 graph=spatial.graph(points,r.radiusKm)
 outgoing=[e for e in graph.edges if e.source==r.propertyId]
 nearest=min((e.distance_km for e in outgoing),default=None)
 density=min(1.0,len(outgoing)/12.0)
 connectivity=min(1.0,sum(e.weight for e in outgoing)/6.0)
 ranked=corridors.rank(graph,20)
 risk_score=risk_band=None
 forecast_next=forecast_trend=forecast_confidence=None
 if r.history:
  values=[x.value for x in r.history]
  change=BaselineEngine.detect(values,values[-1])
  rr=risk.score(connectivity,density,change.relative_delta,0 if abs(change.baseline)<1e-12 else change.volatility/abs(change.baseline))
  ff=forecast.linear(values)
  risk_score,risk_band=rr.score,rr.band
  forecast_next,forecast_trend,forecast_confidence=ff.next_value,ff.trend,ff.confidence
 return {"nearestKm":nearest,"nearbyCount":len(outgoing),"densityScore":density,"connectivityScore":connectivity,"riskScore":risk_score,"riskBand":risk_band,"forecastNext":forecast_next,"forecastTrend":forecast_trend,"forecastConfidence":forecast_confidence,"corridors":[{"from":c.source,"to":c.target,"distanceKm":c.distance_km,"score":c.score} for c in ranked],"status":"available"}
