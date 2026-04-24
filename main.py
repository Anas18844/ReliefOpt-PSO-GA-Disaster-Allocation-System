"""
Command-line entry point for the Disaster Relief EA project.

Examples:
    python main.py --algo pso --scenario Balanced --iters 150
    python main.py --algo hybrid --scenario "High Demand" --selection roulette --crossover simple --mutation uniform
    python main.py --algo both --scenario all --iters 100 --pop 30 --seed 7
    python main.py --benchmark                                 # full 30-run study
    python main.py --compare-ops --scenario Balanced           # GA operator factorial
    python main.py --compare-ops --scenario Balanced --export experiments/ops.csv
"""

from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

from simulation.runner import run_single, run_experiments, run_operator_comparison
from core.ga.operators import (
    SELECTION_OPERATORS, CROSSOVER_OPERATORS, MUTATION_OPERATORS,
)
from problem import SCENARIO_PRESETS


ALL_SCENARIOS = list(SCENARIO_PRESETS.keys())
ALGO_CHOICES = ("pso", "hybrid", "both")
SELECTION_CHOICES = list(SELECTION_OPERATORS.keys())
CROSSOVER_CHOICES = list(CROSSOVER_OPERATORS.keys())
MUTATION_CHOICES  = list(MUTATION_OPERATORS.keys())


def _print_run(result) -> None:
    c = result.as_dict()
    ops = ""
    if result.algorithm == "hybrid":
        ops = f" | {c['selection']}/{c['crossover']}/{c['mutation']}"
    print(
        f"  {c['algorithm']:6s} | {c['scenario']:18s} | seed={c['seed']:3d} "
        f"| fit={c['fitness']:.4f} | cov={c['overall_coverage']:.2%} "
        f"| feas={c['feasible']} | {c['duration_s']:.2f}s{ops}"
    )


def _export_csv(rows, path: Path) -> None:
    if not rows:
        print(f"Nothing to export to {path}.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def cmd_single(args: argparse.Namespace) -> None:
    scenarios = ALL_SCENARIOS if args.scenario == "all" else [args.scenario]
    algos = ["pso", "hybrid"] if args.algo == "both" else [args.algo]

    print(f"Running {len(algos)} x {len(scenarios)} combination(s)...")
    for algo in algos:
        for scen in scenarios:
            r = run_single(
                algorithm=algo, scenario_name=scen,
                iterations=args.iters, pop_size=args.pop,
                seed=args.seed, pso_fraction=args.pso_frac,
                selection=args.selection, crossover=args.crossover,
                mutation=args.mutation,
                verbose=args.verbose,
            )
            _print_run(r)


def cmd_benchmark(args: argparse.Namespace) -> None:
    seeds = list(range(args.runs))
    print(f"Benchmark: pso+hybrid x {len(ALL_SCENARIOS)} scenarios x {len(seeds)} seeds"
          f" = {2 * len(ALL_SCENARIOS) * len(seeds)} runs")

    results = run_experiments(
        algorithms=["pso", "hybrid"],
        scenarios=ALL_SCENARIOS,
        seeds=seeds,
        iterations=args.iters,
        pop_size=args.pop,
        pso_fraction=args.pso_frac,
        selection=args.selection, crossover=args.crossover,
        mutation=args.mutation,
        verbose=args.verbose,
    )

    print("\n--- Per-run results ---")
    for r in results:
        _print_run(r)

    print("\n--- Aggregate summary ---")
    import statistics
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        groups[(r.algorithm, r.scenario)].append(r)

    print(f"{'algo':6s} | {'scenario':18s} | n | mean_fit | std_fit | mean_cov | feas_rate")
    print("-" * 80)
    for (algo, scen), runs in sorted(groups.items()):
        fits = [x.fitness for x in runs]
        covs = [x.overall_coverage for x in runs]
        feas = sum(1 for x in runs if x.feasible) / len(runs)
        print(
            f"{algo:6s} | {scen:18s} | {len(runs):1d} | "
            f"{statistics.mean(fits):.4f} | {statistics.stdev(fits) if len(fits) > 1 else 0:.4f} | "
            f"{statistics.mean(covs):.2%} | {feas:.0%}"
        )

    print("\n--- Head-to-head (Hybrid vs PSO, same seed) ---")
    for scen in ALL_SCENARIOS:
        pso_runs = sorted([r for r in results if r.algorithm == "pso" and r.scenario == scen],
                          key=lambda r: r.seed)
        hy_runs = sorted([r for r in results if r.algorithm == "hybrid" and r.scenario == scen],
                         key=lambda r: r.seed)
        wins = sum(1 for p, h in zip(pso_runs, hy_runs) if h.fitness < p.fitness)
        ties = sum(1 for p, h in zip(pso_runs, hy_runs) if abs(h.fitness - p.fitness) < 1e-6)
        losses = len(pso_runs) - wins - ties
        print(f"  {scen:18s}: Hybrid wins {wins}, ties {ties}, loses {losses} / {len(pso_runs)}")

    if args.export:
        _export_csv([r.as_dict() for r in results], Path(args.export))


def cmd_compare_ops(args: argparse.Namespace) -> None:
    """Run a factorial over (selection x crossover x mutation) for hybrid."""
    scenarios = ALL_SCENARIOS if args.scenario == "all" else [args.scenario]
    seeds = list(range(args.runs))

    all_summary_rows = []
    all_run_rows = []

    for scen in scenarios:
        n_combos = len(SELECTION_CHOICES) * len(CROSSOVER_CHOICES) * len(MUTATION_CHOICES)
        print(f"\n=== Operator comparison on '{scen}' "
              f"({n_combos} combos x {len(seeds)} seeds = {n_combos * len(seeds)} runs) ===")

        out = run_operator_comparison(
            scenario_name=scen,
            seeds=seeds,
            iterations=args.iters,
            pop_size=args.pop,
            pso_fraction=args.pso_frac,
            verbose=args.verbose,
        )

        print(f"\n{'rank':4s} | {'selection':12s} | {'crossover':10s} | {'mutation':12s}"
              f" | {'mean_fit':10s} | {'best_fit':10s} | {'std_fit':10s}"
              f" | {'mean_cov':8s} | {'feas':5s}")
        print("-" * 105)
        for rank, row in enumerate(out["summary"], start=1):
            row_out = {"scenario": scen, "rank": rank, **row}
            all_summary_rows.append(row_out)
            print(f"{rank:4d} | {row['selection']:12s} | {row['crossover']:10s} "
                  f"| {row['mutation']:12s} | {row['mean_fitness']:10.5f} "
                  f"| {row['best_fitness']:10.5f} | {row['std_fitness']:10.5f} "
                  f"| {row['mean_coverage']:8.3f} | {row['feas_rate']:5.2f}")

        for r in out["runs"]:
            d = r.as_dict()
            d["scenario"] = scen
            all_run_rows.append(d)

    if args.export:
        summary_path = Path(args.export)
        _export_csv(all_summary_rows, summary_path)
        runs_path = summary_path.with_name(summary_path.stem + "_runs.csv")
        _export_csv(all_run_rows, runs_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Disaster Relief EA — PSO & Hybrid")

    parser.add_argument("--algo", choices=ALGO_CHOICES, default="pso")
    parser.add_argument("--scenario", default="Balanced",
                        choices=ALL_SCENARIOS + ["all"])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--pop", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pso-frac", dest="pso_frac", type=float, default=0.6,
                        help="Fraction of iterations used by PSO in hybrid mode.")

    # GA operator knobs (hybrid only).
    parser.add_argument("--selection", choices=SELECTION_CHOICES, default="tournament")
    parser.add_argument("--crossover", choices=CROSSOVER_CHOICES, default="whole")
    parser.add_argument("--mutation",  choices=MUTATION_CHOICES,  default="non_uniform")

    parser.add_argument("--benchmark", action="store_true",
                        help="Run full statistical benchmark across all scenarios and seeds.")
    parser.add_argument("--compare-ops", dest="compare_ops", action="store_true",
                        help="Run a factorial comparison of GA operator combinations.")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of seeds per cell in benchmark / compare-ops mode.")
    parser.add_argument("--export", default=None,
                        help="CSV path to export results to (benchmark / compare-ops).")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)

    if args.compare_ops:
        cmd_compare_ops(args)
    elif args.benchmark:
        cmd_benchmark(args)
    else:
        cmd_single(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
