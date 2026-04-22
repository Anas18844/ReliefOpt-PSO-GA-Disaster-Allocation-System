"""
Genetic operators: selection, crossover, mutation.

Each operator is a plain function so it's easy to swap, test, and reuse in
the hybrid solver. An explicit `rng` (numpy Generator) is passed in for
reproducibility.

A "genome" here is a flat numpy array (the same vector shape PSO uses).
That consistency lets the hybrid stage pick up PSO particles and feed
them straight into GA operators.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Selection — picks one genome from a small pool using its fitness.
# Lower fitness = better (we minimise).
# ---------------------------------------------------------------------------

def tournament_selection(pool: List[np.ndarray], fitness: np.ndarray,
                         rng: np.random.Generator, k: int = 3) -> np.ndarray:
    """Pick `k` random candidates; return the best one (copy)."""
    k = max(2, min(k, len(pool)))
    idx = rng.choice(len(pool), size=k, replace=False)
    best = idx[np.argmin(fitness[idx])]
    return pool[best].copy()


def roulette_selection(pool: List[np.ndarray], fitness: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    """Inverse-fitness weighted roulette wheel.

    Since we minimise, we invert: weight = 1 / (fitness + eps).
    """
    weights = 1.0 / (np.asarray(fitness) + 1e-9)
    weights = weights / weights.sum()
    idx = rng.choice(len(pool), p=weights)
    return pool[idx].copy()


# ---------------------------------------------------------------------------
# Crossover — combine two parent genomes into two children.
# ---------------------------------------------------------------------------

def whole_arithmetic_crossover(p1: np.ndarray, p2: np.ndarray,
                               rng: np.random.Generator,
                               alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Blend the whole genomes:  c1 = a*p1 + (1-a)*p2,  c2 = (1-a)*p1 + a*p2.

    Produces two balanced children; a=0.5 gives two copies of the average.
    """
    c1 = alpha * p1 + (1.0 - alpha) * p2
    c2 = (1.0 - alpha) * p1 + alpha * p2
    return c1, c2


def simple_arithmetic_crossover(p1: np.ndarray, p2: np.ndarray,
                                rng: np.random.Generator,
                                alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Keep genes [0:k] from the parents, then arithmetically blend the tail."""
    c1 = p1.copy()
    c2 = p2.copy()
    k = int(rng.integers(1, len(p1)))   # cut-point in [1, n-1]
    c1[k:] = alpha * p1[k:] + (1.0 - alpha) * p2[k:]
    c2[k:] = (1.0 - alpha) * p1[k:] + alpha * p2[k:]
    return c1, c2


# ---------------------------------------------------------------------------
# Mutation — random perturbation to escape local optima.
# ---------------------------------------------------------------------------

def uniform_mutation(genome: np.ndarray, low: float, high: float,
                     rng: np.random.Generator, rate: float = 0.1) -> np.ndarray:
    """Each gene has `rate` probability to be replaced with U(low, high)."""
    mask = rng.random(genome.shape) < rate
    new_genes = rng.uniform(low, high, size=genome.shape)
    return np.where(mask, new_genes, genome)


def non_uniform_mutation(genome: np.ndarray, low: float, high: float,
                         rng: np.random.Generator,
                         current_iter: int, max_iter: int,
                         rate: float = 0.1, b: float = 2.0) -> np.ndarray:
    """Perturbation size shrinks as the run progresses (Michalewicz 1996).

    Early on we jump far (exploration); near the end we nudge (exploitation).
    """
    progress = current_iter / max(1, max_iter)
    mask = rng.random(genome.shape) < rate

    # Random direction (+ or -) for each perturbed gene.
    direction = rng.choice([-1.0, 1.0], size=genome.shape)
    # Magnitude: decays towards 0 as progress → 1.
    r = rng.random(genome.shape)
    delta = (high - low) * (1.0 - r ** ((1.0 - progress) ** b))

    perturbed = genome + direction * delta
    out = np.where(mask, perturbed, genome)
    return np.clip(out, low, high)
