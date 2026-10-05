from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .grey_wolf import GWOConfig, grey_wolf_optimize


@dataclass(frozen=True)
class Objective:
    field: str
    direction: str = "min"
    weight: float = 1.0

    def validate(self) -> None:
        if self.direction not in {"min", "max"}:
            raise ValueError(f"Unsupported direction for {self.field}: {self.direction}")
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError(f"Weight for {self.field} must be non-negative")


@dataclass(frozen=True)
class Constraint:
    field: str
    operator: str
    threshold: float


def _normalized_matrix(frame: pd.DataFrame, objectives: list[Objective]) -> tuple[np.ndarray, dict]:
    if not objectives:
        raise ValueError("At least one objective is required")
    for objective in objectives:
        objective.validate()
        if objective.field not in frame.columns:
            raise KeyError(f"Missing objective field: {objective.field}")
    weights = np.asarray([o.weight for o in objectives], dtype=float)
    if weights.sum() <= 0:
        raise ValueError("At least one objective weight must be positive")
    weights /= weights.sum()

    columns = []
    metadata = {}
    for objective in objectives:
        values = pd.to_numeric(frame[objective.field], errors="coerce").to_numpy(float)
        finite = np.isfinite(values)
        if not finite.any():
            raise ValueError(f"Objective {objective.field} has no finite values")
        median = float(np.nanmedian(values[finite]))
        values = np.where(finite, values, median)
        lo, hi = float(values.min()), float(values.max())
        normalized = np.zeros_like(values) if hi == lo else (values - lo) / (hi - lo)
        cost = normalized if objective.direction == "min" else 1.0 - normalized
        columns.append(cost)
        metadata[objective.field] = {"min": lo, "max": hi, "direction": objective.direction}
    return np.column_stack(columns), {"weights": weights, "fields": metadata}


def _constraint_mask(frame: pd.DataFrame, constraints: list[Constraint]) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    operators = {
        "<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
        "==": np.equal, "!=": np.not_equal,
    }
    for constraint in constraints:
        if constraint.field not in frame.columns:
            raise KeyError(f"Missing constraint field: {constraint.field}")
        if constraint.operator not in operators:
            raise ValueError(f"Unsupported constraint operator: {constraint.operator}")
        values = pd.to_numeric(frame[constraint.field], errors="coerce").to_numpy(float)
        mask &= np.isfinite(values) & operators[constraint.operator](values, float(constraint.threshold))
    return mask


def optimize_candidates(
    frame: pd.DataFrame,
    objectives: list[Objective],
    constraints: list[Constraint] | None = None,
    config: GWOConfig | None = None,
) -> dict:
    """Use GWO to search the continuous objective space, then map leaders to real candidates."""
    if len(frame) < 3:
        raise ValueError("At least three territorial candidates are required")
    constraints = constraints or []
    feasible_mask = _constraint_mask(frame, constraints)
    feasible = frame.loc[feasible_mask].copy()
    if len(feasible) < 3:
        raise ValueError("Constraints leave fewer than three feasible candidates")

    costs, meta = _normalized_matrix(feasible, objectives)
    weights = meta["weights"]
    candidate_fitness = costs @ weights
    lower = np.zeros(costs.shape[1], dtype=float)
    upper = np.ones(costs.shape[1], dtype=float)

    def fitness(points: np.ndarray) -> np.ndarray:
        return points @ weights

    gwo = grey_wolf_optimize(fitness, lower, upper, config)
    # GWO explores ideal objective space. Leaders are projected to the nearest observed candidate.
    leaders = []
    used: set[int] = set()
    for leader in gwo["leaders"]:
        distances = np.linalg.norm(costs - np.asarray(leader["position"]), axis=1)
        for local_idx in np.argsort(distances):
            local_idx = int(local_idx)
            if local_idx not in used:
                used.add(local_idx)
                original_idx = feasible.index[local_idx]
                row = feasible.loc[original_idx]
                contributions = {
                    objective.field: float(costs[local_idx, j] * weights[j])
                    for j, objective in enumerate(objectives)
                }
                leaders.append({
                    "role": leader["role"],
                    "candidate_index": int(original_idx) if isinstance(original_idx, (int, np.integer)) else str(original_idx),
                    "fitness": float(candidate_fitness[local_idx]),
                    "distance_to_gwo_leader": float(distances[local_idx]),
                    "objective_values": {
                        objective.field: float(pd.to_numeric(pd.Series([row[objective.field]]), errors="coerce").iloc[0])
                        for objective in objectives
                    },
                    "weighted_cost_contributions": contributions,
                })
                break

    ranking_order = np.argsort(candidate_fitness)
    ranking = [
        {
            "candidate_index": int(feasible.index[i]) if isinstance(feasible.index[i], (int, np.integer)) else str(feasible.index[i]),
            "fitness": float(candidate_fitness[i]),
        }
        for i in ranking_order[: min(25, len(feasible))]
    ]
    return {
        "algorithm": "Territorial Grey Wolf Optimization",
        "objective_policy": "Lower normalized weighted cost is better; max objectives are inverted before aggregation.",
        "objectives": [{"field": o.field, "direction": o.direction, "weight": o.weight} for o in objectives],
        "constraints": [c.__dict__ for c in constraints],
        "candidate_count": int(len(frame)),
        "feasible_count": int(len(feasible)),
        "leaders": leaders,
        "ranking": ranking,
        "convergence": gwo["convergence"],
        "normalization": meta["fields"],
        "policy": {
            "causal_claims": False,
            "optimization_is_decision_support": True,
            "observed_candidates_only": True,
        },
    }
