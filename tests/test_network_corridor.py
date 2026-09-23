from geo_outliers.network_corridor import build_territorial_graph,shortest_path,detect_network_corridors
def nodes():
 return [{"id":"a","longitude":-74.0,"latitude":4.6,"date":"2026-01-01"},{"id":"b","longitude":-74.005,"latitude":4.6,"date":"2026-01-02"},{"id":"c","longitude":-74.01,"latitude":4.6,"date":"2026-01-03"},{"id":"d","longitude":-74.015,"latitude":4.6,"date":"2026-01-04"}]
def test_network_metrics_and_path():
 g=build_territorial_graph(nodes(),proximity_m=800)
 assert g["metrics"]["connected_components"]==1
 assert g["metrics"]["edge_count"]>=3
 p=shortest_path(g,"a","d");assert p["connected"] and p["distance"]>0
def test_corridor_evidence_is_noncausal():
 r=detect_network_corridors(nodes(),proximity_m=800,permutations=19)
 assert r["corridors"];c=r["corridors"][0]
 assert 0<=c["confidence"]<=1 and c["validation_pvalue"] is not None
 assert r["methodology"]["causal_claims"] is False
 assert r["methodology"]["conflict_classification"] is False
