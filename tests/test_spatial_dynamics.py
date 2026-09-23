import pytest
from geo_outliers.spatial_dynamics import trajectory,compare_trajectories,cluster_trajectories

def snaps(tid,boost=0):
    return [{"territory_id":tid,"timestamp":f"2026-01-0{i+1}","vector":{"satellite.NDVI":v+boost,"social.population":100+i*2}} for i,v in enumerate([.50,.51,.49,.52,.90])]

def test_trajectory_has_velocity_acceleration_and_departure():
    r=trajectory(snaps("T1"),3)
    assert r["territory_id"]=="T1"
    assert len(r["speed"])==4
    assert len(r["acceleration"])==3
    assert r["current_departure"]>=0
    assert r["methodology"]["causal_claims"] is False
    assert r["methodology"]["scores_are_not_probabilities"] is True

def test_compare_trajectory_distance_matrix():
    r=compare_trajectories([{"snapshots":snaps("A")},{"snapshots":snaps("B",.2)}])
    assert r["territories"]==["A","B"]
    assert len(r["speed_distance_matrix"])==2
    assert r["distance_is_not_probability"] is True

def test_cluster_trajectories_is_descriptive():
    r=cluster_trajectories([{"snapshots":snaps("A")},{"snapshots":snaps("B",.2)},{"snapshots":snaps("C",-.2)}],2)
    assert r["n_clusters"]==2
    assert len(r["clusters"])==3
    assert r["methodology"]["cluster_is_descriptive_not_causal"] is True
