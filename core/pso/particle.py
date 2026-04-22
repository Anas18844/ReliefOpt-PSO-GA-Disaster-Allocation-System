"""
A single particle in the swarm.

Think of a particle as one candidate allocation plan flying through
solution-space. It remembers the best plan it has ever personally found
(pBest) and its current direction of travel (velocity).
"""

from __future__ import annotations
import numpy as np


class Particle:
    """One candidate solution with memory + momentum."""

    def __init__(self, scenario, rng: np.random.Generator):
        self.scenario = scenario
        self.dimension = scenario.dimension
        self._rng = rng

        self.position = self._init_position()
        self.velocity = self._init_velocity()

        # pBest memory — starts as the initial position.
        self.pbest_position = self.position.copy()
        self.pbest_fitness = float("inf")
        self.fitness = float("inf")

    def _init_position(self) -> np.ndarray:
        """Start somewhere reasonable: uniformly between 0 and 1.5 x mean demand.

        We spread particles across a useful range of the search space so the
        swarm explores early. The repair operator will pull them into the
        feasible region on the first evaluation.
        """
        upper = self.scenario.demands.mean() * 1.5
        return self._rng.uniform(0.0, upper, size=self.dimension)

    def _init_velocity(self) -> np.ndarray:
        """Small initial velocities — PSO classic advice."""
        upper = self.scenario.demands.mean() * 0.1
        return self._rng.uniform(-upper, upper, size=self.dimension)

    def update_pbest(self) -> None:
        """If current fitness beats stored pBest, overwrite memory."""
        if self.fitness < self.pbest_fitness:
            self.pbest_fitness = self.fitness
            self.pbest_position = self.position.copy()
