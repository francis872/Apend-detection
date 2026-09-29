from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class GWOConfig:
    population_size: int = 40
    iterations: int = 100
    seed: int | None = 42

    def validate(self) -> None:
        if self.population_size < 3:
            raise ValueError("population_size must be at least 3")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")


def grey_wolf_optimize(
    fitness: Callable[[np.ndarray], np.ndarray],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    config: GWOConfig | None = None,
) -> dict:
    """Minimize a vectorized fitness function using the Grey Wolf Optimizer."""
    cfg = config or GWOConfig()
    cfg.validate()
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    if lower.ndim != 1 or upper.shape != lower.shape:
        raise ValueError("Bounds must be one-dimensional arrays with matching shape")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(upper <= lower):
        raise ValueError("Each upper bound must be finite and greater than its lower bound")

    rng = np.random.default_rng(cfg.seed)
    wolves = rng.uniform(lower, upper, size=(cfg.population_size, len(lower)))
    history: list[dict] = []

    for iteration in range(cfg.iterations):
        scores = np.asarray(fitness(wolves), dtype=float)
        if scores.shape != (cfg.population_size,):
            raise ValueError("fitness must return one score per wolf")
        scores = np.where(np.isfinite(scores), scores, np.inf)
        order = np.argsort(scores)
        alpha, beta, delta = (wolves[order[i]].copy() for i in range(3))
        alpha_score = float(scores[order[0]])
        history.append({"iteration": iteration + 1, "alpha_fitness": alpha_score})

        a = 2.0 * (1.0 - iteration / max(cfg.iterations - 1, 1))
        proposals = []
        for leader in (alpha, beta, delta):
            r1 = rng.random(wolves.shape)
            r2 = rng.random(wolves.shape)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            D = np.abs(C * leader - wolves)
            proposals.append(leader - A * D)
        wolves = np.clip(np.mean(proposals, axis=0), lower, upper)

    final_scores = np.asarray(fitness(wolves), dtype=float)
    final_scores = np.where(np.isfinite(final_scores), final_scores, np.inf)
    order = np.argsort(final_scores)
    leaders = []
    for rank, name in enumerate(("alpha", "beta", "delta")):
        idx = int(order[rank])
        leaders.append({"role": name, "position": wolves[idx].tolist(), "fitness": float(final_scores[idx])})
    return {
        "algorithm": "Grey Wolf Optimizer",
        "leaders": leaders,
        "best_position": leaders[0]["position"],
        "best_fitness": leaders[0]["fitness"],
        "convergence": history,
        "iterations": cfg.iterations,
        "population_size": cfg.population_size,
        "seed": cfg.seed,
    }
