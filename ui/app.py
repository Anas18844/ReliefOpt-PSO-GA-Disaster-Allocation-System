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
from simulation.runner import run_single, run_operator_comparison
from core.ga.operators import (
    SELECTION_OPERATORS, CROSSOVER_OPERATORS, MUTATION_OPERATORS,
)
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

st.sidebar.header("GA operators (hybrid only)")
selection_name = st.sidebar.selectbox("Selection", list(SELECTION_OPERATORS.keys()), index=0)
crossover_name = st.sidebar.selectbox("Crossover", list(CROSSOVER_OPERATORS.keys()), index=0)
mutation_name = st.sidebar.selectbox("Mutation", list(MUTATION_OPERATORS.keys()), index=1)

st.sidebar.header("Reproducibility")
algo_seed = st.sidebar.number_input("Algorithm seed", value=0, step=1)

compare_mode = st.sidebar.checkbox("Compare PSO vs Hybrid side-by-side", value=False)
repeat_runs = st.sidebar.slider("Repeat runs (for variance plot)", 1, 15, 1)


# ---------- Tabs ----------
tab_single, tab_ops = st.tabs(["Single run / Compare PSO vs Hybrid",
                               "GA operator comparison"])


# ======================================================================
# Tab 1 — single run or PSO vs Hybrid side-by-side
# ======================================================================
with tab_single:
    run_button = st.button("Run optimisation", type="primary", key="run_single")

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

    if run_button:
        algo_name = "pso" if mode == "PSO" else "hybrid"
        algos_to_run = ["pso", "hybrid"] if compare_mode else [algo_name]

        st.divider()
        st.subheader("Results")

        # Show active operator choice for any hybrid runs.
        if "hybrid" in algos_to_run:
            st.info(
                f"Hybrid GA operators: **selection = {selection_name}**, "
                f"**crossover = {crossover_name}**, **mutation = {mutation_name}**"
            )

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
                        selection=selection_name,
                        crossover=crossover_name,
                        mutation=mutation_name,
                        num_regions=num_regions,
                        scenario_seed=int(scenario_seed),
                    )
                    all_results[a].append(res)

        # --- Headline metrics ---
        headline_cols = st.columns(len(algos_to_run))
        for i, a in enumerate(algos_to_run):
            r = all_results[a][0]
            with headline_cols[i]:
                st.markdown(f"### {a.upper()}")
                if a == "hybrid":
                    st.caption(f"{r.selection} / {r.crossover} / {r.mutation}")
                st.metric("Best fitness", f"{r.fitness:.4f}")
                st.metric("Overall coverage", f"{r.overall_coverage:.2%}")
                st.metric("Mean region coverage", f"{r.mean_coverage:.2%}")
                st.metric("Total shortage", f"{r.total_shortage:.1f}")
                st.metric("Final diversity", f"{r.final_diversity:.4f}")
                st.metric("Feasible", "Done" if r.feasible else "Failed")
                st.caption(f"Runtime: {r.duration_s:.2f}s")

        # --- Convergence ---
        st.subheader("Convergence")
        hist_dict = {a.upper(): all_results[a][0].history for a in algos_to_run}
        st.pyplot(plot_convergence(hist_dict, title=f"{scenario.name} — best fitness per iteration"))

        # --- Diversity ---
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

        # --- Variance plot ---
        if repeat_runs > 1:
            st.subheader(f"Variance across {repeat_runs} runs")
            comp = {a.upper(): [r.fitness for r in all_results[a]] for a in algos_to_run}
            st.pyplot(plot_comparison(comp, title="Final fitness distribution"))

        # --- Per-region coverage ---
        st.subheader("Per-region coverage")
        rows = []
        for a in algos_to_run:
            r = all_results[a][0]
            for i, cov in enumerate(r.per_region_coverage):
                rows.append({"algorithm": a.upper(), "region": f"R{i}", "coverage": cov})
        cov_df = pd.DataFrame(rows)
        pivoted = cov_df.pivot(index="region", columns="algorithm", values="coverage")
        st.dataframe(pivoted.style.format("{:.1%}").background_gradient(cmap="RdYlGn"))

        # --- Run details ---
        st.subheader("Run details")
        detail_rows = []
        for a in algos_to_run:
            for r in all_results[a]:
                detail_rows.append(r.as_dict())
        detail_df = pd.DataFrame(detail_rows)
        st.dataframe(detail_df)
        st.download_button(
            "Download run details (CSV)",
            detail_df.to_csv(index=False).encode("utf-8"),
            file_name=f"runs_{scenario_name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )


# ======================================================================
# Tab 2 — GA operator comparison
# ======================================================================
with tab_ops:
    st.subheader("GA operator comparison")
    st.write(
        "Runs the Hybrid PSO→GA solver across a factorial of "
        "**selection × crossover × mutation** combinations, each for "
        "multiple seeds, and aggregates the results."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        sel_choice = st.multiselect(
            "Selections",
            list(SELECTION_OPERATORS.keys()),
            default=list(SELECTION_OPERATORS.keys()),
        )
    with c2:
        cx_choice = st.multiselect(
            "Crossovers",
            list(CROSSOVER_OPERATORS.keys()),
            default=list(CROSSOVER_OPERATORS.keys()),
        )
    with c3:
        mut_choice = st.multiselect(
            "Mutations",
            list(MUTATION_OPERATORS.keys()),
            default=list(MUTATION_OPERATORS.keys()),
        )

    ops_runs = st.slider("Seeds per combination", 1, 10, 3)

    n_combos = max(1, len(sel_choice) * len(cx_choice) * len(mut_choice))
    st.caption(
        f"This will run **{n_combos}** operator combinations × "
        f"**{ops_runs}** seeds = **{n_combos * ops_runs}** total runs "
        f"on scenario '{scenario_name}'."
    )

    run_ops_btn = st.button("Run comparison", type="primary", key="run_ops")

    if run_ops_btn:
        if not (sel_choice and cx_choice and mut_choice):
            st.error("Please pick at least one option in each operator category.")
        else:
            with st.spinner(f"Running {n_combos * ops_runs} configurations..."):
                out = run_operator_comparison(
                    scenario_name=scenario_name,
                    seeds=list(range(ops_runs)),
                    iterations=iterations,
                    pop_size=pop_size,
                    pso_fraction=pso_fraction,
                    selections=sel_choice,
                    crossovers=cx_choice,
                    mutations=mut_choice,
                    num_regions=num_regions,
                    scenario_seed=int(scenario_seed),
                )

            summary_df = pd.DataFrame(out["summary"])
            summary_df.insert(0, "rank", range(1, len(summary_df) + 1))
            summary_df.insert(1, "scenario", scenario_name)

            st.success(f"Done. Ranked {len(summary_df)} configurations by mean fitness "
                       f"(lower is better).")

            # Highlight best row.
            st.dataframe(
                summary_df.style
                .format({
                    "mean_fitness": "{:.5f}",
                    "best_fitness": "{:.5f}",
                    "worst_fitness": "{:.5f}",
                    "std_fitness": "{:.5f}",
                    "mean_coverage": "{:.3f}",
                    "mean_duration_s": "{:.2f}",
                    "feas_rate": "{:.2f}",
                })
                .background_gradient(subset=["mean_fitness"], cmap="RdYlGn_r")
            )

            # Per-run breakdown
            runs_df = pd.DataFrame([r.as_dict() for r in out["runs"]])
            runs_df.insert(0, "scenario", scenario_name)
            with st.expander(f"All {len(runs_df)} individual runs"):
                st.dataframe(runs_df)

            # Downloads
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "Download summary (CSV)",
                    summary_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"operator_comparison_{scenario_name.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                )
            with col_dl2:
                st.download_button(
                    "Download all runs (CSV)",
                    runs_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"operator_comparison_runs_{scenario_name.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                )


st.caption(
    "Lower fitness = better. Coverage shows the fraction of total demand actually "
    "delivered. Hybrid runs PSO first, then refines with GA using the "
    "selection / crossover / mutation operators chosen in the sidebar."
)
