import numpy as np
from scipy import stats
from geo_outliers.probability import bhattacharyya_distance, fisher_rao_distance_discrete
from geo_outliers.geometry import spd_affine_invariant_distance
from geo_outliers.quadrature import (
    gaussian_quadrature, distribution_interval_probability,
    quadrature_cdf, confidence_interval_areas,
)


def test_probability_distances_zero_on_identity():
    p=np.array([.2,.3,.5])
    assert abs(bhattacharyya_distance(p,p)) < 1e-10
    assert abs(fisher_rao_distance_discrete(p,p)) < 1e-10


def test_spd_distance_zero_on_identity():
    a=np.array([[2.,.2],[.2,1.]])
    assert abs(spd_affine_invariant_distance(a,a)) < 1e-9


def test_gauss_legendre_polynomial():
    # Integral from -1 to 1 of x^4 is exactly 2/5.
    area = gaussian_quadrature(lambda x: x**4, -1.0, 1.0, n=3)
    assert abs(area - 0.4) < 1e-12


def test_normal_central_95_area():
    params=(0.0,1.0)
    lo,hi=stats.norm.ppf([.025,.975])
    area=distribution_interval_probability('norm',params,lo,hi,n=64)
    assert abs(area-.95) < 1e-10


def test_quadrature_cdf_matches_normal_cdf():
    x=np.array([-2.,0.,1.,2.])
    got=quadrature_cdf(x,'norm',(0.,1.),n=64)
    assert np.allclose(got,stats.norm.cdf(x),atol=1e-8)


def test_confidence_areas():
    areas=confidence_interval_areas('norm',(0.,1.),n=64)
    for level in ('90','95','99'):
        assert areas[level]['absolute_error'] < 1e-8
