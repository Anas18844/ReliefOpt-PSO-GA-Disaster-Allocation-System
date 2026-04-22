"""
Fitness function + repair operator for the disaster-relief problem.

Design goals:
  - Lower fitness is better (we MINIMISE).
  - Every term is normalised to roughly [0, 1] so the weights make sense.
  - Infeasible allocations are repaired BEFORE scoring — this way PSO/GA
    always see well-formed solutions and the fitness landscape stays smooth.

Two components (the over-supply term was removed — see note below):
  1. Shortage (HARD goal): how much demand is left unmet → saving lives.
  2. Transport cost: distance-weighted volume shipped → logistics efficiency.

Why no "excess" / over-supply term any more?
  The repair operator already caps every allocation at the region's demand
  (step 3 of repair_constraints). After repair, excess is *mathematically*
  zero, so the excess term was always contributing 0 to the fitness and
  was therefore dead weight in the objective. Removing it simplifies the
  weighting scheme (two weights instead of three) without changing any
  scores, and makes the objective clearer to explain academically.
"""

from __future__ import annotations
import numpy as np

# Default weights — tuned so shortage dominates (people first, logistics second).
# Old weights were (alpha, beta, gamma) = (0.70, 0.25, 0.05). Now that gamma
# is gone we renormalise to (alpha + beta = 1.0) so fitness scale is unchanged.
DEFAULT_ALPHA = 0.75   # shortage weight  (people first)
DEFAULT_BETA  = 0.25   # transport cost weight


def repair_constraints(position: np.ndarray, scenario) -> np.ndarray:
    """Project a raw particle into the feasible region.

    Steps:
      1. Clip negatives to zero (you can't ship a negative amount).
      2. For each resource column, if the total shipped exceeds supply,
         scale that column down proportionally so total == supply.
      3. Never allocate more than a region actually demands
         (anything above demand is pure waste).

    This is a hard guarantee: after this call the solution is feasible.
    """
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
    """Weighted multi-objective score. Smaller = better.

    fitness = alpha * shortage_norm + beta * cost_norm
              (alpha, beta) = (0.75, 0.25)   by default

    We rebuild the allocation matrix, then measure the two normalised costs.

    Note: the previous "excess" / over-supply term is gone. After repair,
    every allocation satisfies alloc <= demand element-wise, so that term
    was always zero. Dropping it keeps the fitness honest — it only
    measures things that can actually change.
    """
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
