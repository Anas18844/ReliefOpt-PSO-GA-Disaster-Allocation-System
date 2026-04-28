from __future__ import annotations
import numpy as np


class Particle:
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
        upper = self.scenario.demands.mean() * 1.5
        return self._rng.uniform(0.0, upper, size=self.dimension)

    def _init_velocity(self) -> np.ndarray:
        upper = self.scenario.demands.mean() * 0.1
        return self._rng.uniform(-upper, upper, size=self.dimension)

    def update_pbest(self) -> None:
        if self.fitness < self.pbest_fitness:
            self.pbest_fitness = self.fitness
            self.pbest_position = self.position.copy()
