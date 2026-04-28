from __future__ import annotations
from typing import Iterable
import numpy as np


def population_diversity(population: Iterable[np.ndarray]) -> float:
    matrix = np.asarray([np.asarray(g) for g in population], dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return 0.0
    # std per column (dimension), then average.
    return float(matrix.std(axis=0).mean())
