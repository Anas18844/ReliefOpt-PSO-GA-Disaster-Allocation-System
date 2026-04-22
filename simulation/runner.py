"""
High-level runner for experiments.

Provides:
  - run_single:   one (algorithm, scenario, seed) combination
  - run_experiments: batch of repeated runs across scenarios/algorithms

Each run returns a dict with: final fitness, history, feasibility flag,
coverage summary — everything experiments and the UI need.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import numpy as np

from problem import make_scenario, solution_report
from core.pso.pso_solver import PSOSolver, PSOConfig
from core.hybrid.hybrid_solver import HybridPSOGASolver, HybridConfig


ALGO_CHOICES = ("pso", "hybrid")


@dataclass
class RunResult:
    algorithm: str
    scenario: str
    seed: int
    fitness: float
    history: List[float]
    diversity_history: List[float]
    duration_s: float
    overall_coverage: float
    feasible: bool
    per_region_coverage: List[float]
    mean_coverage: float
    total_shortage: float
    best_position: np.ndarray = field(repr=False)

    @property
    def final_diversity(self) -> float:
        return self.diversity_history[-1] if self.diversity_history else 0.0

    def as_dict(self) -> Dict:
        """Plain-dict view for pandas / CSV export."""
        return {
            "algorithm": self.algorithm,
            "scenario": self.scenario,
            "seed": self.seed,
            "fitness": self.fitness,
            "duration_s": round(self.duration_s, 3),
            "overall_coverage": round(self.overall_coverage, 4),
            "mean_coverage": round(self.mean_coverage, 4),
            "total_shortage": round(self.total_shortage, 3),
            "final_diversity": round(self.final_diversity, 4),
            "feasible": self.feasible,
        }


def _build_solver(algorithm: str, scenario, rng,
                  iterations: int, pop_size: int,
                  pso_fraction: float = 0.6):
    if algorithm == "pso":
        return PSOSolver(
            scenario,
            PSOConfig(pop_size=pop_size, iterations=iterations),
            rng=rng,
        )
    if algorithm == "hybrid":
        return HybridPSOGASolver(
            scenario,
            HybridConfig(pop_size=pop_size, iterations=iterations,
                         pso_fraction=pso_fraction),
            rng=rng,
        )
    raise ValueError(f"Unknown algorithm '{algorithm}'. Choose: {ALGO_CHOICES}")


def run_single(algorithm: str, scenario_name: str,
               iterations: int = 100, pop_size: int = 30,
               seed: int = 0, pso_fraction: float = 0.6,
               verbose: bool = False,
               num_regions: Optional[int] = None,
               scenario_seed: Optional[int] = None) -> RunResult:
    """Run one algorithm on one scenario with one seed."""
    scenario = make_scenario(scenario_name, num_regions=num_regions,
                             seed=scenario_seed)
    rng = np.random.default_rng(seed)
    solver = _build_solver(algorithm, scenario, rng, iterations, pop_size,
                           pso_fraction)

    t0 = time.perf_counter()
    best_pos, best_fit, history, diversity_history = solver.run(verbose=verbose)
    duration = time.perf_counter() - t0

    report = solution_report(best_pos, scenario)

    return RunResult(
        algorithm=algorithm,
        scenario=scenario.name,
        seed=seed,
        fitness=float(best_fit),
        history=list(history),
        diversity_history=list(diversity_history),
        duration_s=duration,
        overall_coverage=report["overall_coverage"],
        feasible=report["constraints"]["feasible"],
        per_region_coverage=report["per_region_coverage"].tolist(),
        mean_coverage=report["mean_coverage"],
        total_shortage=report["total_shortage"],
        best_position=best_pos,
    )


def run_experiments(algorithms: List[str], scenarios: List[str],
                    seeds: List[int], iterations: int = 100,
                    pop_size: int = 30, pso_fraction: float = 0.6,
                    verbose: bool = False) -> List[RunResult]:
    """Full factorial over (algorithm x scenario x seed)."""
    results: List[RunResult] = []
    total = len(algorithms) * len(scenarios) * len(seeds)
    i = 0
    for algo in algorithms:
        for scen in scenarios:
            for seed in seeds:
                i += 1
                if verbose:
                    print(f"[{i:3d}/{total}] {algo} on {scen} seed={seed}")
                results.append(run_single(
                    algo, scen, iterations=iterations, pop_size=pop_size,
                    seed=seed, pso_fraction=pso_fraction, verbose=False,
                ))
    return results
