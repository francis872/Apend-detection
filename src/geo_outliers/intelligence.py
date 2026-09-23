from __future__ import annotations

def build_intelligence(summary:dict, domain:str="general")->list[dict]:
    spatial=summary.get("spatial_validation",{})
    temporal=summary.get("temporal_validation",{})
    counts=summary.get("counts",{})
    findings=[]
    critical=int(counts.get("99",counts.get(99,0)) or 0)
    if critical:
        findings.append({"severity":"critical","dimension":"statistical","title":"Critical deviation detected","evidence":{"observations":critical,"threshold":"empirical_99_quantile"},"statement":f"{critical} observations reached the empirical 99% anomaly-score quantile."})
    if spatial.get("hotspots",0):
        findings.append({"severity":"high","dimension":"spatial","title":"Spatial concentration detected","evidence":{"hotspots":spatial["hotspots"],"global_moran_like":spatial.get("global_moran_i")},"statement":f"{spatial['hotspots']} observations show high-high local spatial association."})
    if spatial.get("clusters",0):
        findings.append({"severity":"elevated","dimension":"spatial","title":"Proximity clusters identified","evidence":{"clusters":spatial["clusters"],"method":"DBSCAN"},"statement":f"{spatial['clusters']} proximity clusters were identified."})
    if temporal.get("change_point_count",0):
        findings.append({"severity":"elevated","dimension":"temporal","title":"Temporal change candidates","evidence":{"change_points":temporal["change_point_count"],"method":"robust first-difference MAD"},"statement":f"{temporal['change_point_count']} temporal change candidates were detected."})
    for f in findings:f["domain"]=domain
    return findings[:12]
