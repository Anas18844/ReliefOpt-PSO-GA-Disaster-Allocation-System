"""
Matplotlib plotting helpers.

All functions return a Figure so the caller (CLI or Streamlit) can decide
whether to show/save/embed. No calls to plt.show() here.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt

from problem.scenario import RESOURCE_NAMES


def plot_convergence(histories: Dict[str, List[float]],
                     title: str = "Convergence",
                     ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Overlay one line per algorithm, gBest vs iteration."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    for label, hist in histories.items():
        ax.plot(hist, label=label, linewidth=1.8)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness (lower = better)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_allocation_heatmap(allocation: np.ndarray, demands: np.ndarray,
                            title: str = "Allocation vs Demand") -> plt.Figure:
    """Two side-by-side heatmaps (rows=regions, cols=resources)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    vmax = max(demands.max(), allocation.max())
    for ax, data, name in [
        (axes[0], demands, "Demand"),
        (axes[1], allocation, "Allocation"),
    ]:
        im = ax.imshow(data, aspect="auto", vmin=0, vmax=vmax, cmap="YlOrRd")
        ax.set_title(name)
        ax.set_xticks(range(len(RESOURCE_NAMES)))
        ax.set_xticklabels(RESOURCE_NAMES)
        ax.set_xlabel("Resource")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_ylabel("Region")
    axes[0].set_yticks(range(demands.shape[0]))
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_diversity(diversities: Dict[str, List[float]],
                   title: str = "Population Diversity",
                   ax: Optional[plt.Axes] = None) -> plt.Figure:
    """One line per algorithm, diversity vs iteration.

    Diversity = mean per-dimension std-dev of the population. High at start
    (random init), drops as the swarm converges. A flat-near-zero curve
    means the search has stopped exploring.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    for label, d in diversities.items():
        ax.plot(d, label=label, linewidth=1.8)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Population diversity (mean std)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_comparison(runs: Dict[str, List[float]],
                    title: str = "Algorithm Comparison") -> plt.Figure:
    """Boxplot/bar of final fitness across repeated runs."""
    labels = list(runs.keys())
    data = [runs[k] for k in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("Final Fitness (lower = better)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_map(scenario, allocation: Optional[np.ndarray] = None,
             title: str = "Disaster Map") -> plt.Figure:
    """Warehouse + regions, with bubble size ∝ total units delivered.

    Helpful for visually checking that far-away regions aren't over-served
    while close ones are starved.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(*scenario.warehouse_coord, s=250, marker="s",
               color="black", label="Warehouse", zorder=3)

    xs = scenario.regions_coords[:, 0]
    ys = scenario.regions_coords[:, 1]

    if allocation is not None:
        alloc = allocation.reshape(scenario.num_regions, scenario.num_resources)
        per_region = alloc.sum(axis=1)
        sizes = 40 + 4 * per_region
        sc = ax.scatter(xs, ys, s=sizes, c=per_region, cmap="viridis",
                        alpha=0.85, edgecolor="k", label="Region (size = delivered)")
        fig.colorbar(sc, ax=ax, label="Total delivered")
    else:
        ax.scatter(xs, ys, s=80, label="Region")

    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.annotate(f"R{i}", (x, y), textcoords="offset points", xytext=(6, 6))

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
