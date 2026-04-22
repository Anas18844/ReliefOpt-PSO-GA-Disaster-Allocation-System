"""
Command-line entry point for the Disaster Relief EA project.

Examples:
    python main.py --algo pso --scenario Balanced --iters 150
    python main.py --algo hybrid --scenario "High Demand"
    python main.py --algo both --scenario all --iters 100 --pop 30 --seed 7
    python main.py --benchmark                                 # full 30-run study
"""

from __future__ import annotations
import argparse
import sys

from simulation.runner import run_single, run_experiments
from problem import SCENARIO_PRESETS


ALL_SCENARIOS = list(SCENARIO_PRESETS.keys())
ALGO_CHOICES = ("pso", "hybrid", "both")


def _print_run(result) -> None:
    c = result.as_dict()
    print(
        f"  {c['algorithm']:6s} | {c['scenario']:18s} | seed={c['seed']:3d} "
        f"| fit={c['fitness']:.4f} | cov={c['overall_coverage']:.2%} "
        f"| feas={c['feasible']} | {c['duration_s']:.2f}s"
    )


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
                verbose=args.verbose,
            )
            _print_run(r)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Full 30-run benchmark: PSO + Hybrid x 3 scenarios x 5 seeds = 30 runs."""
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
        verbose=args.verbose,
    )

    # Per-run detail
    print("\n--- Per-run results ---")
    for r in results:
        _print_run(r)

    # Aggregate summary
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

    # Head-to-head
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Disaster Relief EA — PSO & Hybrid")
    sub = parser.add_subparsers(dest="command")

    # Default command (no subcommand) = single run
    parser.add_argument("--algo", choices=ALGO_CHOICES, default="pso")
    parser.add_argument("--scenario", default="Balanced",
                        choices=ALL_SCENARIOS + ["all"])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--pop", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pso-frac", dest="pso_frac", type=float, default=0.6,
                        help="Fraction of iterations used by PSO in hybrid mode.")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run full statistical benchmark across all scenarios and seeds.")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of seeds per (algo, scenario) cell in benchmark mode.")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)

    if args.benchmark:
        cmd_benchmark(args)
    else:
        cmd_single(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
