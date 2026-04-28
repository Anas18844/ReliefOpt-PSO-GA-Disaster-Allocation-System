from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


# Resource type indices (kept as a small constant — helps readability).
RESOURCE_NAMES = ("food", "water", "medicine")


@dataclass
class DisasterScenario:
    name: str
    num_regions: int
    available_resources: np.ndarray          # shape (num_resources,)
    demands: np.ndarray                      # shape (num_regions, num_resources)
    regions_coords: np.ndarray               # shape (num_regions, 2)
    warehouse_coord: np.ndarray = field(default_factory=lambda: np.array([50.0, 50.0]))

    @property
    def num_resources(self) -> int:
        return len(self.available_resources)

    @property
    def dimension(self) -> int:
        """Flat length of the decision vector."""
        return self.num_regions * self.num_resources

    @property
    def distances(self) -> np.ndarray:
        """Euclidean distance from warehouse to every region (cached per call)."""
        return np.linalg.norm(self.regions_coords - self.warehouse_coord, axis=1)

    def total_demand(self) -> np.ndarray:
        """Sum of demand per resource across all regions."""
        return self.demands.sum(axis=0)

    def supply_vs_demand_ratio(self) -> np.ndarray:
        """supply / demand per resource — useful for reading scenarios."""
        return self.available_resources / (self.total_demand() + 1e-9)


# --------------------------------------------------------------------------
# Scenario presets
# --------------------------------------------------------------------------
# We keep presets deterministic so experiments are reproducible.
# Each preset fixes: regions, demand pattern, supply pattern.
# Students can tune these to study different disaster "moods".

def _make_balanced(num_regions: int = 6, seed: int = 1) -> DisasterScenario:
    """Supply roughly matches demand — every region can be fully served."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, size=(num_regions, 2))
    demands = rng.integers(10, 30, size=(num_regions, 3)).astype(float)
    # Supply tuned to ~110% of demand (a little slack).
    supply = demands.sum(axis=0) * 1.10
    return DisasterScenario(
        name="Balanced",
        num_regions=num_regions,
        available_resources=supply,
        demands=demands,
        regions_coords=coords,
    )


def _make_high_demand(num_regions: int = 6, seed: int = 2) -> DisasterScenario:
    """Supply is ~50% of demand — algorithm must prioritise hardest-hit regions."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, size=(num_regions, 2))
    demands = rng.integers(20, 60, size=(num_regions, 3)).astype(float)
    supply = demands.sum(axis=0) * 0.50
    return DisasterScenario(
        name="High Demand",
        num_regions=num_regions,
        available_resources=supply,
        demands=demands,
        regions_coords=coords,
    )


def _make_resource_scarcity(num_regions: int = 6, seed: int = 3) -> DisasterScenario:
    """One resource (medicine) is very scarce — tests trade-off decisions."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, size=(num_regions, 2))
    demands = rng.integers(15, 40, size=(num_regions, 3)).astype(float)
    totals = demands.sum(axis=0)
    # Plenty of food & water, only 30% of medicine demand.
    supply = np.array([totals[0] * 1.1, totals[1] * 1.1, totals[2] * 0.30])
    return DisasterScenario(
        name="Resource Scarcity",
        num_regions=num_regions,
        available_resources=supply,
        demands=demands,
        regions_coords=coords,
    )


SCENARIO_PRESETS: Dict[str, callable] = {
    "Balanced": _make_balanced,
    "High Demand": _make_high_demand,
    "Resource Scarcity": _make_resource_scarcity,
}


def make_scenario(name: str, num_regions: Optional[int] = None,
                  seed: Optional[int] = None) -> DisasterScenario:
    if name not in SCENARIO_PRESETS:
        raise ValueError(
            f"Unknown scenario '{name}'. Choose from: {list(SCENARIO_PRESETS)}"
        )
    builder = SCENARIO_PRESETS[name]
    kwargs = {}
    if num_regions is not None:
        kwargs["num_regions"] = num_regions
    if seed is not None:
        kwargs["seed"] = seed
    return builder(**kwargs)
