import numpy as np
import pytest

from geo_outliers.socio_ecological import gini, territorial_indicators, correlation, distance_model, hotspot, detect_corridors


def test_gini_is_concentration_only():
    assert gini([1,1,1,1])==pytest.approx(0)
    assert gini([0,0,0,10])>.7
    result=territorial_indicators({"event_count":10,"area":2,"population":100,"land_areas":[1,1,1,10]})
    assert result["indicators"]["event_density"]==5
    assert result["interpretation_policy"]["gini_measures_concentration_not_conflict"] is True


def test_correlation_reports_n_pvalue_and_no_causality():
    out=correlation([1,2,3,4,5],[2,4,6,8,10],"pearson")
    assert out["n"]==5
    assert out["coefficient"]==pytest.approx(1)
    assert out["p_value"]<=.001
    assert out["causal_claim"] is False


def test_distance_model_supports_requested_metrics():
    ref=[[0,0],[1,1],[2,1],[1,2],[3,3]]
    out=distance_model([1,2],[2,4],ref)
    assert out["euclidean"]>0
    assert out["standardized_euclidean"]>0
    assert out["mahalanobis"] is not None
    assert -1<=out["cosine_similarity"]<=1
    assert out["distance_is_not_probability"] is True


def test_hotspot_lisa_keeps_statistical_significance_explicit():
    points=[{"longitude":float(i%5),"latitude":float(i//5),"value":float(i)} for i in range(25)]
    out=hotspot(points,method="lisa")
    assert out["method"]=="Local Moran's I"
    assert out["significance_is_statistical"] is True
    assert len(out["records"])==25


def test_corridor_candidates_do_not_claim_conflict_or_probability():
    pts=[{"id":str(i),"longitude":i*.01,"latitude":0.0,"variable":"events"} for i in range(5)]
    out=detect_corridors(pts,.02)
    assert out
    assert out[0]["method"]=="proximity connected-components candidate"
    assert "not probability" in " ".join(out[0]["limitations"]).lower()
    assert "conflict" in " ".join(out[0]["limitations"]).lower()
