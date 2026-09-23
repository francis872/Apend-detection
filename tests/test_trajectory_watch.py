from geo_outliers.trajectory_watch import assess_trajectory

def snaps(values):
 return [{"territory_id":"T1","timestamp":f"2026-01-{i+1:02d}","vector":{"satellite.NDVI":v,"social.population":100+i}} for i,v in enumerate(values)]

def test_watch_contract_is_noncausal_nonforecast():
 r=assess_trajectory(snaps([.5,.51,.49,.5,.52,.9]),3)
 assert r["status"] in {"NORMAL","ELEVATED","HIGH"}
 assert r["methodology"]["causal_claims"] is False
 assert r["methodology"]["forecast"] is False
 assert r["methodology"]["risk_probability"] is False
 assert "trajectory" in r

def test_watch_baseline_building():
 r=assess_trajectory(snaps([.5,.51]),2)
 assert r["signals"]["departure"]["level"]=="BASELINE_BUILDING"
