# ReliefOpt — PSO, GA & Hybrid PSO-GA for Disaster Relief Allocation

An academic project for **AI420 — Evolutionary Algorithms (Spring 2026)**.

Distributes food, water, and medicine across disaster-affected regions using
Particle Swarm Optimisation, a standalone Genetic Algorithm, and a Hybrid
PSO → GA refinement pipeline, under supply, non-negativity, and demand-cap
constraints.

---

## Contributors

**Meriem Ashraf** — lead contributor on the core algorithm codebase. Her work
forms the foundation of this project: the original PSO solver, the disaster
scenario model, the fitness function, the repair operator, and the first
hybrid PSO/GA design all came from her. The rest of this repository is a
refactor, extension, and productionisation of that core. The project would
simply not exist without her contribution, and the credit for the
algorithmic heart of ReliefOpt belongs to her.

---

## Project at a glance

- **Problem**: allocate limited relief resources across multiple disaster
  regions to minimise unmet demand (primary) and transport cost (secondary).
- **Algorithms**:
  - **PSO** with linear inertia decay, velocity clamp, and a repair-based
    feasibility projection.
  - **GA** (standalone): tournament/roulette selection, whole/simple
    arithmetic crossover, uniform/non-uniform mutation, top-2 elitism,
    constraint repair every generation.
  - **Hybrid PSO → GA**: PSO explores and converges fast, then GA refines
    the best solutions found, using the same configurable operators.
- **Fitness (minimised)**:
  `fitness = 0.75 × shortage_norm + 0.25 × cost_norm`
- **Validation**: full benchmark across PSO + GA + Hybrid × 3 scenarios ×
  N seeds — 100% feasibility; hybrid achieves the lowest variance on
  non-bounded scenarios.

## Features

- Reproducible scenario presets: *Balanced*, *High Demand*, *Resource Scarcity*.
- Three fully-fledged solvers (PSO, GA, Hybrid) with a unified runner.
- Configurable GA operators (selection × crossover × mutation) for both
  the standalone GA and the Hybrid solver.
- Convergence **and** population-diversity tracking over iterations.
- Constraint validator + per-region coverage reports.
- CLI with single-run, full-benchmark, and operator-comparison modes.
- Interactive Streamlit UI with PSO vs GA vs Hybrid side-by-side comparison.
- Documented project report with time complexity and limitations.

## Project structure

```text
core/
  pso/        particle + PSO solver
  ga/         operators (pure functions) + standalone GA solver
  hybrid/     PSO -> GA pipeline with elitism
problem/      scenarios, fitness, repair, validator
simulation/   run_single / run_experiments / run_operator_comparison
utils/        plotting, diversity metric, seeding
ui/           Streamlit app
experiments/  benchmark logs
main.py       CLI entry point
```

## How to run

See [`HOW_TO_RUN.md`](HOW_TO_RUN.md) for a short, copy-pasteable guide.

In short:

```bash
pip install -r requirements.txt

# CLI — single run
python main.py --algo pso    --scenario Balanced --iters 150
python main.py --algo ga     --scenario Balanced --selection roulette --crossover simple --mutation uniform
python main.py --algo hybrid --scenario "High Demand"
python main.py --algo both   --scenario all --iters 100 --pop 30 --seed 7

# Full benchmark across PSO + GA + Hybrid
python main.py --benchmark --runs 10 --export experiments/benchmark.csv

# GA operator factorial (selection x crossover x mutation)
python main.py --compare-ops --scenario Balanced --runs 5 --export experiments/ops.csv

# Interactive UI
streamlit run ui/app.py
```

### CLI flags

| Flag | Choices / default | Notes |
| --- | --- | --- |
| `--algo` | `pso`, `ga`, `hybrid`, `both` (default `pso`) | `both` runs PSO + GA + Hybrid |
| `--scenario` | `Balanced`, `High Demand`, `Resource Scarcity`, `all` | scenario preset |
| `--iters` | int (default 100) | iterations |
| `--pop` | int (default 30) | population size |
| `--seed` | int (default 0) | RNG seed |
| `--pso-frac` | float (default 0.6) | hybrid only: PSO budget share |
| `--selection` | `tournament`, `roulette` | used by GA & Hybrid |
| `--crossover` | `whole`, `simple` | used by GA & Hybrid |
| `--mutation` | `uniform`, `non_uniform` | used by GA & Hybrid |
| `--benchmark` | flag | full PSO + GA + Hybrid statistical study |
| `--compare-ops` | flag | factorial GA operator comparison (Hybrid) |
| `--runs` | int (default 5) | seeds per cell in benchmark / compare-ops |
| `--export` | path | CSV export path |
| `--verbose` | flag | per-iteration logging |

### UI modes

- **Single run / Comparative Analysis** — run one algorithm, or tick
  *"Compare PSO vs GA vs Hybrid"* to run all three on the same scenario
  and view convergence, diversity, allocation heatmaps, delivery maps,
  per-region coverage, and a variance plot.
- **GA operator comparison** — factorial sweep of selection × crossover
  × mutation, ranked by mean fitness, with downloadable CSV summaries.

## Report

Full design, algorithm walkthroughs, experimental results, time complexity,
and limitations are in [`PROJECT_REPORT.md`](PROJECT_REPORT.md).

## Licence

Academic coursework project. Not for production use without review.
