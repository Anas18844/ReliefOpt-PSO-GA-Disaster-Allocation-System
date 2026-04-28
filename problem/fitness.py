from __future__ import annotations
import numpy as np

DEFAULT_ALPHA = 0.75   # shortage weight  (people first)
DEFAULT_BETA  = 0.25   # transport cost weight


def repair_constraints(position: np.ndarray, scenario) -> np.ndarray:
    num_regions = scenario.num_regions
    num_resources = scenario.num_resources
    available = scenario.available_resources
    demands = scenario.demands

    alloc = np.asarray(position, dtype=float).reshape(num_regions, num_resources)

    # Step 1 — no negative shipments.
    np.maximum(alloc, 0.0, out=alloc)

    # Step 2 — respect supply limit per resource type.
    column_totals = alloc.sum(axis=0)
    over_budget = column_totals > available
    if np.any(over_budget):
        # Scale only the columns that exceed their budget.
        scale = np.where(over_budget, available / (column_totals + 1e-12), 1.0)
        alloc = alloc * scale  # broadcasting across rows

    # Step 3 — cap at demand (no point shipping more than needed).
    np.minimum(alloc, demands, out=alloc)

    return alloc.flatten()


def fitness_function(position: np.ndarray, scenario,
                     alpha: float = DEFAULT_ALPHA,
                     beta: float = DEFAULT_BETA) -> float:
    alloc = np.asarray(position, dtype=float).reshape(
        scenario.num_regions, scenario.num_resources
    )

    # --- 1. Unmet demand (the thing that actually hurts people) ---
    shortage = np.maximum(0.0, scenario.demands - alloc).sum()
    max_shortage = scenario.demands.sum() + 1e-12
    shortage_norm = shortage / max_shortage           # in [0, 1]

    # --- 2. Transport cost = how much volume * how far ---
    delivered_per_region = alloc.sum(axis=1)          # total units per region
    cost = np.sum(delivered_per_region * scenario.distances)
    # Worst case = ship everything we have to the farthest region.
    max_cost = scenario.distances.max() * scenario.available_resources.sum() + 1e-12
    cost_norm = cost / max_cost                       # roughly [0, 1]

    return float(alpha * shortage_norm + beta * cost_norm)
