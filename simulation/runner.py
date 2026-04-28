from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Dict, List, Optional
import time
import numpy as np

from problem import make_scenario, solution_report
from core.pso.pso_solver import PSOSolver, PSOConfig
from core.hybrid.hybrid_solver import HybridPSOGASolver, HybridConfig
from core.ga.operators import (
    SELECTION_OPERATORS, CROSSOVER_OPERATORS, MUTATION_OPERATORS,
)


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
    # Operator metadata (None for pure PSO).
    selection: Optional[str] = None
    crossover: Optional[str] = None
    mutation: Optional[str]  = None

    @property
    def final_diversity(self) -> float:
        return self.diversity_history[-1] if self.diversity_history else 0.0

    def as_dict(self) -> Dict:
        return {
            "algorithm": self.algorithm,
            "scenario": self.scenario,
            "seed": self.seed,
            "selection": self.selection or "-",
            "crossover": self.crossover or "-",
            "mutation":  self.mutation or "-",
            "fitness": round(self.fitness, 6),
            "duration_s": round(self.duration_s, 3),
            "overall_coverage": round(self.overall_coverage, 4),
            "mean_coverage": round(self.mean_coverage, 4),
            "total_shortage": round(self.total_shortage, 3),
            "final_diversity": round(self.final_diversity, 4),
            "feasible": self.feasible,
        }


def _build_solver(algorithm: str, scenario, rng,
                  iterations: int, pop_size: int,
                  pso_fraction: float = 0.6,
                  selection: str = "tournament",
                  crossover: str = "whole",
                  mutation: str  = "non_uniform"):
    if algorithm == "pso":
        return PSOSolver(
            scenario,
            PSOConfig(pop_size=pop_size, iterations=iterations),
            rng=rng,
        )
    if algorithm == "hybrid":
        return HybridPSOGASolver(
            scenario,
            HybridConfig(
                pop_size=pop_size, iterations=iterations,
                pso_fraction=pso_fraction,
                selection=selection, crossover=crossover, mutation=mutation,
            ),
            rng=rng,
        )
    raise ValueError(f"Unknown algorithm '{algorithm}'. Choose: {ALGO_CHOICES}")


def run_single(algorithm: str, scenario_name: str,
               iterations: int = 100, pop_size: int = 30,
               seed: int = 0, pso_fraction: float = 0.6,
               selection: str = "tournament",
               crossover: str = "whole",
               mutation: str  = "non_uniform",
               verbose: bool = False,
               num_regions: Optional[int] = None,
               scenario_seed: Optional[int] = None) -> RunResult:
    """Run one algorithm on one scenario with one seed."""
    scenario = make_scenario(scenario_name, num_regions=num_regions,
                             seed=scenario_seed)
    rng = np.random.default_rng(seed)
    solver = _build_solver(algorithm, scenario, rng, iterations, pop_size,
                           pso_fraction, selection, crossover, mutation)

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
        selection=selection if algorithm == "hybrid" else None,
        crossover=crossover if algorithm == "hybrid" else None,
        mutation=mutation  if algorithm == "hybrid" else None,
    )


def run_experiments(algorithms: List[str], scenarios: List[str],
                    seeds: List[int], iterations: int = 100,
                    pop_size: int = 30, pso_fraction: float = 0.6,
                    selection: str = "tournament",
                    crossover: str = "whole",
                    mutation: str  = "non_uniform",
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
                    algorithm=algo, scenario_name=scen,
                    iterations=iterations, pop_size=pop_size,
                    seed=seed, pso_fraction=pso_fraction,
                    selection=selection, crossover=crossover, mutation=mutation,
                    verbose=False,
                ))
    return results


# ---------------------------------------------------------------------------
# Operator comparison
# ---------------------------------------------------------------------------

def run_operator_comparison(
    scenario_name: str,
    seeds: List[int],
    iterations: int = 100,
    pop_size: int = 30,
    pso_fraction: float = 0.6,
    selections: Optional[List[str]] = None,
    crossovers: Optional[List[str]] = None,
    mutations: Optional[List[str]] = None,
    num_regions: Optional[int] = None,
    scenario_seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    """Run a full factorial of GA operator combinations on the hybrid solver.

    For each (selection, crossover, mutation) triple, runs `len(seeds)` seeds
    and aggregates mean/best/std of fitness + coverage + duration.

    Returns a dict with:
        "runs":    list[RunResult]         — every individual run
        "summary": list[dict]              — one row per operator combo
    """
    selections = selections or list(SELECTION_OPERATORS.keys())
    crossovers = crossovers or list(CROSSOVER_OPERATORS.keys())
    mutations  = mutations  or list(MUTATION_OPERATORS.keys())

    all_runs: List[RunResult] = []
    buckets: Dict[tuple, List[RunResult]] = defaultdict(list)

    total = len(selections) * len(crossovers) * len(mutations) * len(seeds)
    i = 0
    for sel in selections:
        for cx in crossovers:
            for mut in mutations:
                for seed in seeds:
                    i += 1
                    if verbose:
                        print(f"[{i:3d}/{total}] {sel}/{cx}/{mut} seed={seed}")
                    r = run_single(
                        algorithm="hybrid",
                        scenario_name=scenario_name,
                        iterations=iterations, pop_size=pop_size,
                        seed=seed, pso_fraction=pso_fraction,
                        selection=sel, crossover=cx, mutation=mut,
                        num_regions=num_regions,
                        scenario_seed=scenario_seed,
                    )
                    all_runs.append(r)
                    buckets[(sel, cx, mut)].append(r)

    summary = []
    for (sel, cx, mut), runs in buckets.items():
        fits = [r.fitness for r in runs]
        covs = [r.overall_coverage for r in runs]
        durs = [r.duration_s for r in runs]
        summary.append({
            "selection": sel,
            "crossover": cx,
            "mutation":  mut,
            "runs":           len(runs),
            "mean_fitness":   round(mean(fits), 6),
            "best_fitness":   round(min(fits), 6),
            "worst_fitness":  round(max(fits), 6),
            "std_fitness":    round(stdev(fits) if len(fits) > 1 else 0.0, 6),
            "mean_coverage":  round(mean(covs), 4),
            "mean_duration_s": round(mean(durs), 3),
            "feas_rate":      round(sum(1 for r in runs if r.feasible) / len(runs), 3),
        })

    # Rank by mean fitness (lower is better).
    summary.sort(key=lambda row: row["mean_fitness"])
    return {"runs": all_runs, "summary": summary}
