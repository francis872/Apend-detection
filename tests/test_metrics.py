import numpy as np
import pandas as pd
from scipy import stats
from geo_outliers.probability import bhattacharyya_distance, fisher_rao_distance_discrete
from geo_outliers.geometry import spd_affine_invariant_distance
from geo_outliers.quadrature import gaussian_quadrature, distribution_interval_probability, quadrature_cdf, confidence_interval_areas
from geo_outliers.spatial import spatial_validation
from geo_outliers.temporal import temporal_validation
from geo_outliers.derivatives import pdf_derivatives, find_inflection_points


def test_probability_distances_zero_on_identity():
    p=np.array([.2,.3,.5])
    assert abs(bhattacharyya_distance(p,p))<1e-10
    assert abs(fisher_rao_distance_discrete(p,p))<1e-10


def test_spd_distance_zero_on_identity():
    a=np.array([[2.,.2],[.2,1.]])
    assert abs(spd_affine_invariant_distance(a,a))<1e-9


def test_gauss_legendre_polynomial():
    assert abs(gaussian_quadrature(lambda x:x**4,-1.,1.,n=3)-.4)<1e-12


def test_normal_central_95_area():
    lo,hi=stats.norm.ppf([.025,.975])
    area=distribution_interval_probability('norm',(0.,1.),lo,hi,n=64)
    assert abs(area-.95)<1e-10


def test_quadrature_cdf_matches_normal_cdf():
    x=np.array([-2.,0.,1.,2.])
    assert np.allclose(quadrature_cdf(x,'norm',(0.,1.),n=64),stats.norm.cdf(x),atol=1e-8)


def test_confidence_areas():
    areas=confidence_interval_areas('norm',(0.,1.),n=64)
    assert all(areas[p]['absolute_error']<1e-8 for p in ('90','95','99'))


def test_spatial_validation_flags_local_difference():
    x=np.arange(60,dtype=float)
    coords=np.c_[x,np.zeros_like(x)]
    values=np.ones(60); values[30]=100.
    score,meta=spatial_validation(coords,values,k=6)
    assert score[30]>.8
    assert meta['valid_rows']==60


def test_temporal_validation_flags_spike():
    n=80
    df=pd.DataFrame({'ACQ_DATE':pd.date_range('2026-01-01',periods=n,freq='D'),'FRP':np.ones(n)})
    df.loc[40,'FRP']=100.
    score,meta=temporal_validation(df,'FRP',window=21)
    assert score[40]>.9
    assert meta['valid_rows']==n


def test_normal_inflection_points_are_near_one_sigma():
    result=find_inflection_points('norm',(0.,1.),lower=-4.,upper=4.,grid_size=4096)
    pts=np.asarray(result['inflection_points'])
    assert len(pts)==2
    assert np.allclose(pts,[-1.,1.],atol=.02)


def test_pdf_derivatives_normal_mode():
    x=np.linspace(-3,3,2001)
    pdf,d1,d2=pdf_derivatives('norm',(0.,1.),x)
    i=np.argmin(np.abs(x))
    assert abs(d1[i])<1e-4
    assert d2[i]<0
