"""Small helper to seed every randomness source the project uses."""

from __future__ import annotations
import random
import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed Python's `random`, NumPy's legacy RNG, and return a fresh Generator.

    We return a Generator so callers can pass it explicitly to solvers;
    this keeps behaviour deterministic even if some other code also imports
    numpy and draws from the global RNG.
    """
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
