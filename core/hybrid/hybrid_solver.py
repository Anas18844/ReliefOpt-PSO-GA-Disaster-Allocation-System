"""
Hybrid PSO → GA solver.

Idea:
    PSO is great at sweeping a large search space quickly, but it can stall
    around a near-optimum without refining it. GA (with tournament selection,
    arithmetic crossover, and non-uniform mutation) is good at local
    refinement through recombination and small targeted jumps.

    So we run PSO for the FIRST `pso_fraction` of the budget, take the
    evolved swarm as a seeded GA population, then run GA for the remaining
    iterations with ELITISM so we never lose the best-so-far.

Each iteration of the GA stage:
    1. Score the population (fitness + repair).
    2. Keep the top `elite_count` individuals verbatim.
    3. Fill the rest of the next generation by:
         - tournament-selecting two parents
         - arithmetic crossover with probability pc
         - non-uniform mutation with probability pm
    4. Repair every child (so the population stays feasible).

Contract: same public interface as PSOSolver.run() — returns
(best_position, best_fitness, history).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from problem.fitness import repair_constraints, fitness_function
from core.pso.pso_solver import PSOSolver, PSOConfig
from core.ga.operators import (
    tournament_selection,
    whole_arithmetic_crossover,
    non_uniform_mutation,
)
from utils.diversity import population_diversity


@dataclass
class HybridConfig:
    # Shared budget
    pop_size: int = 30
    iterations: int = 100
    pso_fraction: float = 0.6       # 60% of budget → PSO, 40% → GA refinement

    # PSO params (passed through to PSOSolver)
    w_max: float = 0.9
    w_min: float = 0.4
    c1: float = 1.5
    c2: float = 1.5

    # GA params
    pc: float = 0.8                 # crossover probability
    pm: float = 0.15                # mutation probability
    tournament_k: int = 3
    elite_count: int = 2            # elitism: keep top N individuals


class HybridPSOGASolver:
    def __init__(self, scenario, config: Optional[HybridConfig] = None,
                 rng: Optional[np.random.Generator] = None):
        self.scenario = scenario
        self.config = config or HybridConfig()
        self.rng = rng if rng is not None else np.random.default_rng()

        # Search bounds used by mutation.
        upper = scenario.demands.max() * 1.5
        self._pos_low = 0.0
        self._pos_high = float(upper)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, verbose: bool = False
            ) -> Tuple[np.ndarray, float, List[float], List[float]]:
        """Runs PSO stage, then GA stage. Returns:
            best_position, best_fitness, fitness_history, diversity_history
        diversity_history is continuous across both stages (same length as
        fitness_history), so plots line up 1-to-1.
        """
        cfg = self.config
        pso_iters = max(1, int(cfg.iterations * cfg.pso_fraction))
        ga_iters = max(0, cfg.iterations - pso_iters)

        # ---- Stage 1: PSO -------------------------------------------------
        pso_cfg = PSOConfig(
            pop_size=cfg.pop_size,
            iterations=pso_iters,
            w_max=cfg.w_max, w_min=cfg.w_min,
            c1=cfg.c1, c2=cfg.c2,
        )
        pso = PSOSolver(self.scenario, pso_cfg, rng=self.rng)
        pso_best_pos, pso_best_fit, history, diversity_history = pso.run(verbose=verbose)

        if verbose:
            print(f"  [HYBRID] PSO stage done. fitness={pso_best_fit:.6f}")

        # If no GA budget left, we're done.
        if ga_iters == 0:
            return pso_best_pos, pso_best_fit, history, diversity_history

        # ---- Stage 2: seed GA with PSO swarm ------------------------------
        # Each particle's pBest is used as the starting genome (it's a
        # stronger starting point than the particle's current position).
        population: List[np.ndarray] = [p.pbest_position.copy() for p in pso.swarm]
        fitness = np.array([
            fitness_function(repair_constraints(g, self.scenario), self.scenario)
            for g in population
        ])
        # Track best across the hybrid run.
        best_idx = int(np.argmin(fitness))
        best_pos = population[best_idx].copy()
        best_fit = float(fitness[best_idx])
        if best_fit > pso_best_fit:
            best_pos, best_fit = pso_best_pos.copy(), pso_best_fit

        # ---- Stage 3: GA refinement --------------------------------------
        for it in range(ga_iters):
            new_pop = self._next_generation(population, fitness, it, ga_iters)

            # Repair + evaluate the new generation.
            new_fitness = np.empty(len(new_pop))
            for i, g in enumerate(new_pop):
                new_pop[i] = repair_constraints(g, self.scenario)
                new_fitness[i] = fitness_function(new_pop[i], self.scenario)

            population = new_pop
            fitness = new_fitness

            gen_best_idx = int(np.argmin(fitness))
            if fitness[gen_best_idx] < best_fit:
                best_fit = float(fitness[gen_best_idx])
                best_pos = population[gen_best_idx].copy()

            history.append(best_fit)
            diversity_history.append(population_diversity(population))
            if verbose and (it % 10 == 0 or it == ga_iters - 1):
                print(f"  [GA]  iter {it:3d}  best = {best_fit:.6f}"
                      f"  div = {diversity_history[-1]:.4f}")

        return best_pos, best_fit, history, diversity_history

    # ------------------------------------------------------------------
    # GA helpers
    # ------------------------------------------------------------------

    def _next_generation(self, population: List[np.ndarray], fitness: np.ndarray,
                         current_iter: int, max_iter: int) -> List[np.ndarray]:
        """Build the next population: elitism + selection + crossover + mutation."""
        cfg = self.config
        n = len(population)

        # Elites — top-k individuals passed through untouched.
        elite_idx = np.argsort(fitness)[: cfg.elite_count]
        next_pop: List[np.ndarray] = [population[i].copy() for i in elite_idx]

        while len(next_pop) < n:
            parent1 = tournament_selection(population, fitness, self.rng, cfg.tournament_k)
            parent2 = tournament_selection(population, fitness, self.rng, cfg.tournament_k)

            # Crossover (or clone parents as-is).
            if self.rng.random() < cfg.pc:
                child1, child2 = whole_arithmetic_crossover(
                    parent1, parent2, self.rng, alpha=self.rng.uniform(0.3, 0.7)
                )
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation — non-uniform: strong early, gentle late.
            if self.rng.random() < cfg.pm:
                child1 = non_uniform_mutation(
                    child1, self._pos_low, self._pos_high, self.rng,
                    current_iter, max_iter, rate=0.2,
                )
            if self.rng.random() < cfg.pm:
                child2 = non_uniform_mutation(
                    child2, self._pos_low, self._pos_high, self.rng,
                    current_iter, max_iter, rate=0.2,
                )

            next_pop.append(child1)
            if len(next_pop) < n:
                next_pop.append(child2)

        return next_pop[:n]
