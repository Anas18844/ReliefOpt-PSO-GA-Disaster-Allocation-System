# ReliefOpt — PSO & Hybrid PSO-GA for Disaster Relief Allocation

An academic project for **AI420 — Evolutionary Algorithms (Spring 2026)**.

Distributes food, water, and medicine across disaster-affected regions using
Particle Swarm Optimisation and a Hybrid PSO → GA refinement pipeline, under
supply, non-negativity, and demand-cap constraints.

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
  - **Hybrid PSO → GA**: PSO explores and converges fast, then GA
    (tournament selection, whole arithmetic crossover, non-uniform mutation,
    top-2 elitism) refines the best solutions found.
- **Fitness (minimised)**:
  `fitness = 0.75 × shortage_norm + 0.25 × cost_norm`
- **Validation**: 60-run benchmark across 3 scenarios × 10 seeds — 100%
  feasibility; hybrid achieves ~7× lower variance than PSO on the
  non-bounded scenarios.

## Features

- Reproducible scenario presets: *Balanced*, *High Demand*, *Resource Scarcity*.
- Convergence **and** population-diversity tracking over iterations.
- Constraint validator + per-region coverage reports.
- CLI with single-run and full-benchmark modes.
- Interactive Streamlit UI with side-by-side algorithm comparison.
- Documented project report with time complexity and limitations.

## Project structure

```text
core/
  pso/        particle + PSO solver
  ga/         selection, crossover, mutation (pure functions)
  hybrid/     PSO -> GA pipeline with elitism
problem/      scenarios, fitness, repair, validator
simulation/   run_single / run_experiments harness
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
python main.py --algo both --scenario all       # CLI
streamlit run ui/app.py                          # UI
```

## Report

Full design, algorithm walkthroughs, experimental results, time complexity,
and limitations are in [`PROJECT_REPORT.md`](PROJECT_REPORT.md).

## Licence

Academic coursework project. Not for production use without review.
