"""
Particle Swarm Optimisation solver.

Workflow per iteration:
  1. Repair every particle (forces feasibility).
  2. Score every particle with the fitness function.
  3. Update personal best (pBest) and global best (gBest).
  4. Update velocity:
         v = w*v + c1*r1*(pBest - x) + c2*r2*(gBest - x)
  5. Apply velocity clamp, then move: x = x + v.
  6. Keep a history of gBest fitness so we can plot convergence.

Improvements over the baseline repo:
  - Fixed Particle constructor (used to be called with wrong argument).
  - Linear inertia decay w: w_max → w_min, encourages exploration early,
    exploitation late.
  - Velocity clamp v_max so the swarm doesn't explode numerically.
  - Position bounds so particles stay in a sensible range; repair handles
    the feasibility constraints separately.
  - Reproducible randomness via a numpy Generator.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from problem.fitness import repair_constraints, fitness_function
from utils.diversity import population_diversity
from .particle import Particle


@dataclass
class PSOConfig:
    pop_size: int = 30
    iterations: int = 100
    w_max: float = 0.9
    w_min: float = 0.4
    c1: float = 1.5      # cognitive coefficient (pull towards pBest)
    c2: float = 1.5      # social coefficient    (pull towards gBest)
    v_clamp_frac: float = 0.2   # max velocity = v_clamp_frac * upper bound


class PSOSolver:
    def __init__(self, scenario, config: Optional[PSOConfig] = None,
                 rng: Optional[np.random.Generator] = None):
        self.scenario = scenario
        self.config = config or PSOConfig()
        self.rng = rng if rng is not None else np.random.default_rng()

        self.swarm: List[Particle] = [
            Particle(scenario, self.rng) for _ in range(self.config.pop_size)
        ]

        # Placeholders — filled by the first evaluation.
        self.gbest_position: Optional[np.ndarray] = None
        self.gbest_fitness: float = float("inf")

        # Precomputed bounds used for velocity clamp and position clip.
        upper = scenario.demands.max() * 1.5
        self._pos_low = 0.0
        self._pos_high = float(upper)
        self._v_max = self.config.v_clamp_frac * upper

    # ------------------------------------------------------------------
    # Core steps
    # ------------------------------------------------------------------

    def _evaluate_all(self) -> None:
        """Repair → score → update pBest and gBest for every particle."""
        for p in self.swarm:
            p.position = repair_constraints(p.position, self.scenario)
            p.fitness = fitness_function(p.position, self.scenario)
            p.update_pbest()
            if p.fitness < self.gbest_fitness:
                self.gbest_fitness = p.fitness
                self.gbest_position = p.position.copy()

    def _move_swarm(self, w: float) -> None:
        """Classic PSO velocity + position update."""
        d = self.scenario.dimension
        for p in self.swarm:
            r1 = self.rng.random(d)
            r2 = self.rng.random(d)

            # نحرك الجسيم ناحية أحسن تجربة ليه (pBest)
            # وكمان ناحية أحسن تجربة في السرب كله (gBest)
            cognitive = self.config.c1 * r1 * (p.pbest_position - p.position)
            social    = self.config.c2 * r2 * (self.gbest_position - p.position)

            p.velocity = w * p.velocity + cognitive + social
            np.clip(p.velocity, -self._v_max, self._v_max, out=p.velocity)

            p.position = p.position + p.velocity
            np.clip(p.position, self._pos_low, self._pos_high, out=p.position)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, verbose: bool = False
            ) -> Tuple[np.ndarray, float, List[float], List[float]]:
        """Run the swarm for `iterations` steps.

        Returns:
            best_position, best_fitness, convergence_history, diversity_history
            (diversity = mean std-dev of particle positions across dimensions)
        """
        history: List[float] = []
        diversity_history: List[float] = []

        # First pass — we need gBest defined before we can move anything.
        self._evaluate_all()

        for it in range(self.config.iterations):
            # Linear inertia decay — encourages exploration → exploitation.
            w = self.config.w_max - (self.config.w_max - self.config.w_min) * (
                it / max(1, self.config.iterations - 1)
            )

            self._move_swarm(w)
            self._evaluate_all()

            history.append(self.gbest_fitness)
            diversity_history.append(
                population_diversity([p.position for p in self.swarm])
            )
            if verbose and (it % 10 == 0 or it == self.config.iterations - 1):
                print(f"  [PSO] iter {it:3d}  gBest = {self.gbest_fitness:.6f}"
                      f"  div = {diversity_history[-1]:.4f}")

        return (self.gbest_position.copy(), self.gbest_fitness,
                history, diversity_history)
