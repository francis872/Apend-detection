import numpy as np
from geo_outliers.probability import bhattacharyya_distance, fisher_rao_distance_discrete
from geo_outliers.geometry import spd_affine_invariant_distance


def test_probability_distances_zero_on_identity():
    p=np.array([.2,.3,.5])
    assert abs(bhattacharyya_distance(p,p)) < 1e-10
    assert abs(fisher_rao_distance_discrete(p,p)) < 1e-10


def test_spd_distance_zero_on_identity():
    a=np.array([[2.,.2],[.2,1.]])
    assert abs(spd_affine_invariant_distance(a,a)) < 1e-9
