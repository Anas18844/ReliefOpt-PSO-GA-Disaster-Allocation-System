"""
Streamlit UI for the Disaster Relief EA project.

Run with:
    streamlit run ui/app.py
"""

from __future__ import annotations
import sys
from pathlib import Path

# Make the project root importable when launched via `streamlit run ui/app.py`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from problem import make_scenario, SCENARIO_PRESETS, solution_report
from simulation.runner import run_single
from utils.plotting import (
    plot_convergence, plot_allocation_heatmap, plot_comparison, plot_map,
    plot_diversity,
)


st.set_page_config(page_title="Disaster Relief EA", layout="wide")
st.title("Disaster Relief Resource Allocation")
st.caption("PSO and Hybrid PSO+GA — Evolutionary Algorithms project")


# ---------- Sidebar controls ----------
st.sidebar.header("Scenario")
scenario_name = st.sidebar.selectbox("Preset", list(SCENARIO_PRESETS.keys()))
num_regions = st.sidebar.slider("Number of regions", 3, 20, 6)
scenario_seed = st.sidebar.number_input("Scenario seed", value=1, step=1)

st.sidebar.header("Algorithm")
mode = st.sidebar.radio("Mode", ["PSO", "Hybrid (PSO → GA)"])
iterations = st.sidebar.slider("Iterations", 20, 500, 100, step=10)
pop_size = st.sidebar.slider("Population size", 10, 100, 30, step=5)
pso_fraction = st.sidebar.slider("PSO fraction (hybrid only)", 0.1, 1.0, 0.6, step=0.05)

st.sidebar.header("Reproducibility")
algo_seed = st.sidebar.number_input("Algorithm seed", value=0, step=1)

compare_mode = st.sidebar.checkbox("Compare PSO vs Hybrid side-by-side", value=False)
repeat_runs = st.sidebar.slider("Repeat runs (for variance plot)", 1, 15, 1)

run_button = st.sidebar.button("Run optimisation", type="primary")


# ---------- Scenario preview ----------
scenario = make_scenario(scenario_name, num_regions=num_regions,
                         seed=int(scenario_seed))

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Scenario overview")
    st.write(
        f"**{scenario.name}** — {scenario.num_regions} regions, "
        f"{scenario.num_resources} resources "
        f"(dim={scenario.dimension})"
    )
    demand_df = pd.DataFrame(
        scenario.demands, columns=["food", "water", "medicine"],
        index=[f"R{i}" for i in range(scenario.num_regions)],
    )
    supply_df = pd.DataFrame(
        [scenario.available_resources],
        columns=["food", "water", "medicine"],
        index=["Supply"],
    )
    ratio_df = pd.DataFrame(
        [scenario.supply_vs_demand_ratio()],
        columns=["food", "water", "medicine"],
        index=["Supply / Demand"],
    )
    st.dataframe(supply_df.style.format("{:.1f}"))
    st.dataframe(ratio_df.style.format("{:.2f}"))
    with st.expander("Per-region demand matrix"):
        st.dataframe(demand_df.style.format("{:.1f}"))

with col2:
    st.subheader("Map")
    st.pyplot(plot_map(scenario, title=f"{scenario.name} — regions & warehouse"))


# ---------- Run ----------
if run_button:
    algo_name = "pso" if mode == "PSO" else "hybrid"
    algos_to_run = ["pso", "hybrid"] if compare_mode else [algo_name]

    st.divider()
    st.subheader("Results")

    # Store all runs keyed by (algo, seed_offset).
    all_results = {a: [] for a in algos_to_run}

    with st.spinner("Running..."):
        for a in algos_to_run:
            for r in range(repeat_runs):
                res = run_single(
                    algorithm=a,
                    scenario_name=scenario_name,
                    iterations=iterations,
                    pop_size=pop_size,
                    seed=int(algo_seed) + r,
                    pso_fraction=pso_fraction,
                    num_regions=num_regions,
                    scenario_seed=int(scenario_seed),
                )
                all_results[a].append(res)

    # --- Headline metrics: use the first run per algo ---
    headline_cols = st.columns(len(algos_to_run))
    for i, a in enumerate(algos_to_run):
        r = all_results[a][0]
        with headline_cols[i]:
            st.markdown(f"### {a.upper()}")
            st.metric("Best fitness", f"{r.fitness:.4f}")
            st.metric("Overall coverage", f"{r.overall_coverage:.2%}")
            st.metric("Mean region coverage", f"{r.mean_coverage:.2%}")
            st.metric("Total shortage", f"{r.total_shortage:.1f}")
            st.metric("Final diversity", f"{r.final_diversity:.4f}")
            st.metric("Feasible", "Done" if r.feasible else "Failed")
            st.caption(f"Runtime: {r.duration_s:.2f}s")

    # --- Convergence curves ---
    st.subheader("Convergence")
    hist_dict = {a.upper(): all_results[a][0].history for a in algos_to_run}
    st.pyplot(plot_convergence(hist_dict, title=f"{scenario.name} — best fitness per iteration"))

    # --- Diversity curves ---
    st.subheader("Population diversity")
    st.caption(
        "Mean per-dimension std-dev of the population. Starts high (random init), "
        "drops as the algorithm converges. A curve that flattens near zero too early "
        "signals premature convergence."
    )
    div_dict = {a.upper(): all_results[a][0].diversity_history for a in algos_to_run}
    st.pyplot(plot_diversity(div_dict, title=f"{scenario.name} — diversity per iteration"))

    # --- Allocation heatmap ---
    st.subheader("Allocation vs demand")
    for a in algos_to_run:
        best = all_results[a][0].best_position
        alloc = best.reshape(scenario.num_regions, scenario.num_resources)
        st.markdown(f"**{a.upper()}**")
        st.pyplot(plot_allocation_heatmap(
            alloc, scenario.demands,
            title=f"{a.upper()} — allocation vs demand",
        ))

    # --- Delivery map ---
    st.subheader("Delivery map")
    map_cols = st.columns(len(algos_to_run))
    for i, a in enumerate(algos_to_run):
        with map_cols[i]:
            st.pyplot(plot_map(
                scenario,
                allocation=all_results[a][0].best_position,
                title=f"{a.upper()} delivery",
            ))

    # --- Variance plot over repeated runs ---
    if repeat_runs > 1:
        st.subheader(f"Variance across {repeat_runs} runs")
        comp = {a.upper(): [r.fitness for r in all_results[a]] for a in algos_to_run}
        st.pyplot(plot_comparison(comp, title="Final fitness distribution"))

    # --- Per-region coverage table ---
    st.subheader("Per-region coverage")
    rows = []
    for a in algos_to_run:
        r = all_results[a][0]
        for i, cov in enumerate(r.per_region_coverage):
            rows.append({"algorithm": a.upper(), "region": f"R{i}", "coverage": cov})
    cov_df = pd.DataFrame(rows)
    pivoted = cov_df.pivot(index="region", columns="algorithm", values="coverage")
    st.dataframe(pivoted.style.format("{:.1%}").background_gradient(cmap="RdYlGn"))

    # --- Aggregate table ---
    st.subheader("Run details")
    detail_rows = []
    for a in algos_to_run:
        for r in all_results[a]:
            detail_rows.append(r.as_dict())
    st.dataframe(pd.DataFrame(detail_rows))


st.caption(
    "Lower fitness = better. Coverage shows the fraction of total demand actually "
    "delivered. Hybrid runs PSO first, then refines with GA (tournament + "
    "arithmetic crossover + non-uniform mutation, with elitism)."
)
