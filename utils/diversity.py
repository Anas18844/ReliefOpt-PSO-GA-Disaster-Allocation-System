"""
Diversity metric for a population of candidate solutions.

Why diversity matters
---------------------
When a swarm or GA population collapses onto a single point, the search is
effectively over — velocities drop, crossover produces clones, and no new
region of the space is explored. Measuring diversity lets us *see* this
happening.

A healthy run typically shows:
  - diversity high at the start (random init)
  - diversity drops as particles converge toward gBest
  - if diversity goes to near-zero too early, the swarm has stagnated
    (premature convergence) — we're probably stuck in a local optimum

How we measure it
-----------------
For a population P of shape (N, D):
    1. for each dimension d, compute the std across the N solutions
    2. take the mean over all dimensions

That single number summarises "how spread out" the population is. Higher =
more diverse, zero = every solution is identical.
"""

from __future__ import annotations
from typing import Iterable
import numpy as np


def population_diversity(population: Iterable[np.ndarray]) -> float:
    """Mean standard deviation across dimensions of a population of genomes.

    Args:
        population: iterable of 1D numpy arrays of identical length D.

    Returns:
        A single non-negative float. Zero means every solution is identical.
    """
    matrix = np.asarray([np.asarray(g) for g in population], dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return 0.0
    # std per column (dimension), then average.
    return float(matrix.std(axis=0).mean())
