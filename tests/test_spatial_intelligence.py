import numpy as np
from geo_outliers.spatial import local_moran_lisa, spatial_density, spatial_validation


def grid(n=10):
    x,y=np.meshgrid(np.arange(n,dtype=float),np.arange(n,dtype=float))
    return np.c_[x.ravel(),y.ravel()]


def test_lisa_is_deterministic_and_bounded():
    coords=grid(10)
    values=np.r_[np.ones(50)*10,np.ones(50)]
    a=local_moran_lisa(coords,values,k=8,permutations=49,random_state=42)
    b=local_moran_lisa(coords,values,k=8,permutations=49,random_state=42)
    assert np.allclose(a["local_i"],b["local_i"])
    assert np.allclose(a["pvalue"],b["pvalue"])
    assert np.all((a["pvalue"]>0)&(a["pvalue"]<=1))
    assert set(np.unique(a["cluster"])).issubset({"NS","HH","LL","HL","LH"})


def test_density_rank_detects_dense_cluster():
    dense=np.random.default_rng(42).normal(0,.05,(80,2))
    sparse=np.c_[np.linspace(2,20,20),np.zeros(20)]
    d=spatial_density(np.vstack([dense,sparse]),k=8)
    assert len(d)==100
    assert np.all((d>=0)&(d<=1))
    assert np.median(d[:80])>np.median(d[80:])


def test_spatial_validation_reports_inferential_metadata():
    coords=grid(10)
    values=np.linspace(0,1,100)
    score,meta=spatial_validation(coords,values,k=8)
    assert len(score)==100
    assert np.all((score>=0)&(score<=1))
    assert meta["lisa_permutations"]==199
    assert "global_moran_pvalue" in meta
    assert "lisa_significant" in meta
    assert "density_rank_p95" in meta
