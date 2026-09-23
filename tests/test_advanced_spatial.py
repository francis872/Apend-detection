import numpy as np
import pytest
from geo_outliers.advanced_spatial import getis_ord_gi_star,bivariate_moran,spatial_association_matrix,temporal_spatial_comparison

def grid():
    out=[]
    for y in range(5):
        for x in range(5):
            hot=10.0 if x>=3 and y>=3 else 1.0
            out.append({"longitude":float(x),"latitude":float(y),"a":hot,"b":hot*2+float(x),"value":hot})
    return out

def test_gistar_reports_permutation_significance_contract():
    r=getis_ord_gi_star(grid(),"value",k=4,permutations=39)
    assert r["method"]=="Getis-Ord Gi*"
    assert r["n"]==25 and len(r["records"])==25
    assert r["significance_is_statistical"] is True
    assert r["causal_claim"] is False
    assert all(0<=(x["p_value"])<=1 for x in r["records"])

def test_bivariate_moran_reports_n_coefficient_pvalue():
    r=bivariate_moran(grid(),"a","b",k=4,permutations=39)
    assert r["n"]==25
    assert -5<r["coefficient"]<5
    assert 0<=r["p_value"]<=1
    assert r["causal_claim"] is False

def test_spatial_association_matrix_shape():
    r=spatial_association_matrix(grid(),["a","b"],k=4,permutations=19)
    assert r["variables"]==["a","b"]
    assert len(r["matrix"])==2 and len(r["matrix"][0])==2

def test_temporal_gistar_comparison_exposes_delta():
    a=grid();b=[{**p,"value":p["value"]+(5 if p["longitude"]<2 else 0)} for p in a]
    r=temporal_spatial_comparison(a,b,"value","gi_star",4,19)
    assert len(r["changes"])==25
    assert all("delta_z" in x for x in r["changes"])
    assert r["causal_claim"] is False
