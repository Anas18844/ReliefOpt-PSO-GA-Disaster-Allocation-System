from __future__ import annotations
from typing import Dict
import numpy as np


def check_constraints(solution: np.ndarray, scenario,
                      tol: float = 1e-6) -> Dict[str, bool]:
    """Check every hard constraint and return a dict of pass/fail flags."""
    alloc = np.asarray(solution, dtype=float).reshape(
        scenario.num_regions, scenario.num_resources
    )

    results = {
        "non_negative": bool(np.all(alloc >= -tol)),
        "within_supply": bool(
            np.all(alloc.sum(axis=0) <= scenario.available_resources + tol)
        ),
        "no_over_demand": bool(np.all(alloc <= scenario.demands + tol)),
    }
    results["feasible"] = all(results.values())
    return results


def solution_report(solution: np.ndarray, scenario) -> Dict:
    alloc = np.asarray(solution, dtype=float).reshape(
        scenario.num_regions, scenario.num_resources
    )
    demand = scenario.demands

    per_region_demand = demand.sum(axis=1)
    per_region_delivered = alloc.sum(axis=1)
    # Avoid divide-by-zero for any region that happened to have zero demand.
    coverage_per_region = np.where(
        per_region_demand > 0,
        per_region_delivered / per_region_demand,
        1.0,
    )

    total_demand = demand.sum()
    total_delivered = alloc.sum()
    total_shortage = float(np.maximum(0.0, demand - alloc).sum())

    return {
        "allocation": alloc,
        "per_region_coverage": coverage_per_region,          # 0..1 each
        "mean_coverage": float(coverage_per_region.mean()),
        "total_delivered": float(total_delivered),
        "total_demand": float(total_demand),
        "overall_coverage": float(total_delivered / (total_demand + 1e-12)),
        "total_shortage": total_shortage,
        "supply_usage": alloc.sum(axis=0) / (scenario.available_resources + 1e-12),
        "constraints": check_constraints(solution, scenario),
    }
