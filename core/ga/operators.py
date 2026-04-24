"""
Genetic operators: selection, crossover, mutation.

Each operator is a plain function so it's easy to swap, test, and reuse in
the hybrid solver. An explicit `rng` (numpy Generator) is passed in for
reproducibility.

A "genome" here is a flat numpy array (the same vector shape PSO uses).
That consistency lets the hybrid stage pick up PSO particles and feed
them straight into GA operators.

Operator registries at the bottom of the file expose every operator by
short name so the hybrid solver / UI / CLI can pick one without the
caller importing the function directly.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Tuple
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
                       rng: np.random.Generator, k: int = 3) -> np.ndarray:
    """Inverse-fitness weighted roulette wheel.

    Since we minimise, we invert: weight = 1 / (fitness + eps).
    `k` is accepted for a uniform selection signature but unused here.
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
    """Blend the whole genomes:  c1 = a*p1 + (1-a)*p2,  c2 = (1-a)*p1 + a*p2."""
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
                     rng: np.random.Generator,
                     current_iter: int = 0, max_iter: int = 1,
                     rate: float = 0.1) -> np.ndarray:
    """Each gene has `rate` probability to be replaced with U(low, high).

    `current_iter` / `max_iter` accepted for a uniform mutation signature
    but unused here — they matter for non_uniform_mutation.
    """
    mask = rng.random(genome.shape) < rate
    new_genes = rng.uniform(low, high, size=genome.shape)
    return np.where(mask, new_genes, genome)


def non_uniform_mutation(genome: np.ndarray, low: float, high: float,
                         rng: np.random.Generator,
                         current_iter: int = 0, max_iter: int = 1,
                         rate: float = 0.1, b: float = 2.0) -> np.ndarray:
    """Perturbation size shrinks as the run progresses (Michalewicz 1996).

    Early on we jump far (exploration); near the end we nudge (exploitation).
    """
    progress = current_iter / max(1, max_iter)
    mask = rng.random(genome.shape) < rate

    direction = rng.choice([-1.0, 1.0], size=genome.shape)
    r = rng.random(genome.shape)
    delta = (high - low) * (1.0 - r ** ((1.0 - progress) ** b))

    perturbed = genome + direction * delta
    out = np.where(mask, perturbed, genome)
    return np.clip(out, low, high)


# ---------------------------------------------------------------------------
# Registries — short-name → function. Used by HybridPSOGASolver and the UI.
# Signatures are unified so the solver can call any entry the same way.
# ---------------------------------------------------------------------------

SelectionFn = Callable[[List[np.ndarray], np.ndarray, np.random.Generator, int], np.ndarray]
CrossoverFn = Callable[[np.ndarray, np.ndarray, np.random.Generator, float], Tuple[np.ndarray, np.ndarray]]
MutationFn = Callable[..., np.ndarray]

SELECTION_OPERATORS: Dict[str, SelectionFn] = {
    "tournament": tournament_selection,
    "roulette":   roulette_selection,
}

CROSSOVER_OPERATORS: Dict[str, CrossoverFn] = {
    "whole":  whole_arithmetic_crossover,
    "simple": simple_arithmetic_crossover,
}

MUTATION_OPERATORS: Dict[str, MutationFn] = {
    "uniform":     uniform_mutation,
    "non_uniform": non_uniform_mutation,
}


def get_selection(name: str) -> SelectionFn:
    if name not in SELECTION_OPERATORS:
        raise ValueError(f"Unknown selection '{name}'. Choose: {list(SELECTION_OPERATORS)}")
    return SELECTION_OPERATORS[name]


def get_crossover(name: str) -> CrossoverFn:
    if name not in CROSSOVER_OPERATORS:
        raise ValueError(f"Unknown crossover '{name}'. Choose: {list(CROSSOVER_OPERATORS)}")
    return CROSSOVER_OPERATORS[name]


def get_mutation(name: str) -> MutationFn:
    if name not in MUTATION_OPERATORS:
        raise ValueError(f"Unknown mutation '{name}'. Choose: {list(MUTATION_OPERATORS)}")
    return MUTATION_OPERATORS[name]
