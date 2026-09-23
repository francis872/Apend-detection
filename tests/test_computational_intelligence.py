import numpy as np
import pandas as pd
import pytest

from geo_outliers.computational_intelligence import euclidean_intelligence, compare_to_baseline


def test_center_is_closer_than_extreme_observation():
    df=pd.DataFrame({"ndvi":[.5,.51,.49,.50,.95],"ndmi":[.2,.21,.19,.20,-.8],"nbr":[.3,.31,.29,.30,-.7]})
    out,summary=euclidean_intelligence(df,["ndvi","ndmi","nbr"],feature_weight=1,spatial_weight=0)
    assert out["euclidean_feature_distance"].iloc[-1]>out["euclidean_feature_distance"].iloc[0]
    assert summary["policy"]["distance_is_not_probability"] is True


def test_robust_scaling_prevents_large_units_from_automatic_domination():
    df=pd.DataFrame({"ndvi":[.1,.2,.3,.4,.5],"temperature":[100,200,300,400,500]})
    out,_=euclidean_intelligence(df,["ndvi","temperature"])
    assert np.isfinite(out["euclidean_intelligence_score"]).all()


def test_spatial_context_can_be_combined():
    df=pd.DataFrame({"ndvi":[.1,.2,.3,.4,.9],"ndmi":[.2,.3,.4,.5,.9],"x":[0,1,2,3,50],"y":[0,1,2,3,50]})
    out,summary=euclidean_intelligence(df,["ndvi","ndmi"],("x","y"),.4,.6)
    assert summary["weights"]["spatial"]==pytest.approx(.4)
    assert out["euclidean_spatial_distance"].iloc[-1]>out["euclidean_spatial_distance"].iloc[0]


def test_baseline_distance_reports_deltas():
    result=compare_to_baseline({"ndvi":.3,"ndmi":.1,"nbr":.4},{"ndvi":.6,"ndmi":.2,"nbr":.5},["ndvi","ndmi","nbr"])
    assert result["distance"]>0
    assert result["per_feature_delta"]["ndvi"]==pytest.approx(-.3)


def test_baseline_requires_two_shared_features():
    with pytest.raises(ValueError):
        compare_to_baseline({"ndvi":.3},{"ndvi":.6},["ndvi"])
