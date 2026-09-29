import numpy as np
import pandas as pd
import pytest

from geo_outliers.grey_wolf import GWOConfig, grey_wolf_optimize
from geo_outliers.optimization import Constraint, Objective, optimize_candidates


def test_gwo_converges_near_zero_for_sphere():
    def sphere(points):
        return np.sum(points ** 2, axis=1)
    result = grey_wolf_optimize(sphere, np.array([-5.0, -5.0]), np.array([5.0, 5.0]), GWOConfig(30, 80, 7))
    assert result["best_fitness"] < 0.05
    assert len(result["leaders"]) == 3
    assert len(result["convergence"]) == 80


def test_territorial_optimizer_respects_min_and_max_objectives():
    df = pd.DataFrame({
        "risk": [0.1, 0.4, 0.8, 0.2, 0.5],
        "accessibility": [0.9, 0.8, 0.3, 0.7, 0.6],
        "cost": [0.2, 0.1, 0.9, 0.3, 0.5],
    })
    result = optimize_candidates(
        df,
        [Objective("risk", "min", .45), Objective("accessibility", "max", .35), Objective("cost", "min", .20)],
        config=GWOConfig(20, 40, 11),
    )
    assert result["leaders"][0]["candidate_index"] == 0
    assert result["policy"]["causal_claims"] is False


def test_constraints_remove_infeasible_candidates():
    df = pd.DataFrame({
        "risk": [.1, .2, .3, .4, .5],
        "accessibility": [.5, .6, .7, .8, .9],
        "slope": [5, 8, 10, 20, 30],
    })
    result = optimize_candidates(
        df,
        [Objective("risk", "min"), Objective("accessibility", "max")],
        [Constraint("slope", "<=", 12)],
        GWOConfig(12, 20, 3),
    )
    assert result["feasible_count"] == 3


def test_fewer_than_three_feasible_candidates_is_blocked():
    df = pd.DataFrame({"risk": [.1, .2, .3], "slope": [1, 20, 30]})
    with pytest.raises(ValueError):
        optimize_candidates(df, [Objective("risk")], [Constraint("slope", "<", 10)])
