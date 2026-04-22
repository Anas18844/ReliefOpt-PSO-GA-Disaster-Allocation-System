# Disaster Relief Resource Allocation — Project Report

> Evolutionary Algorithms (AI420), Spring 2026
> Topic #5 — Particle Swarm Optimisation for disaster-relief resource allocation, with optional Swarm Intelligence / EA hybridisation.

---

## 1. What we found in the original repository

**Source:** <https://github.com/Maryam-A-Ashraf/Evolutionary-Algorithm>

A flat Python project with 6 files:

| File | Purpose |
|---|---|
| `problem_env.py` | `DisasterScenario` and `Particle` classes |
| `fitness.py` | Fitness function + repair operator |
| `solver.py` | Standard PSO solver |
| `genetic_operators.py` | Tournament/roulette selection, arithmetic crossovers, uniform / non-uniform mutation |
| `HPSOGA.py` | A hybrid PSO + GA solver |
| `validation_function.py` | Post-hoc constraint checker |

### Strengths

- Clean separation of **scenario**, **fitness**, and **solver**.
- Multi-objective fitness with three sensible terms: shortage, transport cost, over-supply.
- A **repair operator** that enforces supply constraints — this is the right design choice, because it lets PSO treat the problem as unconstrained and keeps the fitness landscape smooth.
- A working `HPSOGA` implementation showing the group had already thought about hybridisation.

### Weaknesses

1. **Bug in `solver.py`.** `PSOSolver` calls `Particle(scenario.dimension)` — passing an `int` — but `Particle.__init__` expects the scenario object. The code would crash on a fresh run.
2. **`gbest_position` initialisation is unsafe.** It's set to the first particle's position *before* any fitness evaluation, so there was no guarantee it corresponded to a valid solution.
3. **No velocity clamp, no inertia schedule.** The original used a fixed `w=0.7`; a linear decay `w_max → w_min` is the standard improvement for multimodal problems.
4. **Scenarios are random per run.** `DisasterScenario` generates regions, demands, and supplies from a global NumPy RNG, so two identical commands produce different problems — impossible to compare algorithms fairly.
5. **No CLI / no UI / no experiments harness.** The brief explicitly requires a simple interface and a parameter study.
6. **Flat file layout.** As the project grows (GA, hybrid, scenarios, plotting, UI) everything ends up in one directory.
7. **No reproducibility.** No seeding discipline — every run is different.
8. **Repair didn't cap at demand.** The original only capped by supply, so the fitness function was penalising over-supply that repair could have removed for free.

---

## 2. What we changed and why

### 2.1 Architecture refactor

Move from flat files → modular package:

```
project/
├── core/
│   ├── pso/       ← Particle, PSOSolver
│   ├── ga/        ← operators (selection / crossover / mutation)
│   └── hybrid/    ← HybridPSOGASolver (PSO → GA refinement)
├── problem/       ← DisasterScenario, fitness, repair, validator
├── simulation/    ← runner + experiment harness
├── utils/         ← plotting, seeding
├── ui/            ← Streamlit app
├── experiments/   ← benchmark logs
├── main.py        ← CLI entry point
└── requirements.txt
```

Rationale: each folder has a single responsibility, solvers depend on `problem/` but not vice versa, and the UI / CLI depend on `simulation/` — no circular imports.

### 2.2 Bug fixes and correctness improvements

| Fix | Before | After |
|---|---|---|
| `Particle` constructor mismatch | `Particle(scenario.dimension)` | `Particle(scenario, rng)` |
| `gbest` init | Set to first particle's position *before* any evaluation | Set during the first `_evaluate_all()` pass, guaranteed to be the global best |
| Repair operator | Capped only by supply | Caps by supply **and** by demand (no wasteful over-supply) |
| Fitness term `excess_norm` | Shared normaliser with shortage, weights didn't add to a nice range | **Removed** — repair already caps alloc at demand, so the excess term was always zero. Objective is now `α·shortage + β·cost` with `α=0.75, β=0.25` |
| Scenarios | Truly random each run | **Presets with fixed seeds** (Balanced / High Demand / Resource Scarcity) → reproducible experiments |
| Randomness | Global `np.random` | Every solver takes a `numpy.random.Generator` — same seed → identical results |

### 2.3 PSO improvements

- **Linear inertia decay** `w_max = 0.9 → w_min = 0.4`. Early particles explore; late particles exploit.
- **Velocity clamp** `|v| ≤ 0.2 × upper_bound`. Prevents numerical blow-up when `(pBest − x)` is large.
- **Position clip** `[0, 1.5 × max_demand]` so particles stay in a sensible search range; the repair operator still enforces the feasibility constraints separately.
- **Reproducible RNG** passed through from `simulation.runner`.

### 2.4 Hybrid logic — real hybridisation, not stitched code

The original `HPSOGA` partitioned the swarm into sub-populations every iteration and applied crossover *inside* each partition while PSO continued — effectively running GA on top of PSO continuously. That's one valid design, but it's hard to reason about because GA is disrupting PSO's momentum every step.

Our hybrid uses the **PSO → GA pipeline** pattern:

```
PSO stage (pso_fraction × total_iters):
   run the full PSO solver
   ↓
Seed GA (take every particle's pBest as the starting genome)
   ↓
GA stage (remaining iterations):
   at each generation:
     1. elitism: keep top-k individuals
     2. tournament selection on the rest
     3. whole arithmetic crossover with prob pc
     4. non-uniform mutation with prob pm (shrinks over time)
     5. repair + re-evaluate
     track best-so-far
```

Why this works:
- PSO finds a **good basin** quickly via continuous velocity updates.
- GA **refines within that basin** through recombination and shrinking mutations, with **elitism** guaranteeing we never lose what PSO found.

### 2.5 Reproducibility + explainability

- Every run is reproducible: a single seed propagates through scenario generation, particle init, velocity draws, GA operators.
- Comments are written **student-friendly** (some in Arabic where the mental picture helps, e.g. “نحرك الجسيم ناحية أحسن تجربة ليه”).
- The validator returns a dict of pass/fail flags; the UI shows them as ticks.

---

## 3. What we added

| Addition | Where | Purpose |
|---|---|---|
| Scenario presets | `problem/scenario.py` | Balanced / High Demand / Resource Scarcity — reproducible test-beds |
| Constraint validator + report | `problem/validator.py` | `check_constraints`, `solution_report` (per-region coverage, supply usage, feasibility) |
| Clean PSO | `core/pso/` | Bug fix + inertia decay + velocity clamp + history |
| Refactored GA operators | `core/ga/operators.py` | Pure functions with injected RNG |
| **Hybrid PSO→GA solver** | `core/hybrid/hybrid_solver.py` | Pipeline with elitism |
| Simulation runner | `simulation/runner.py` | `run_single`, `run_experiments`, `RunResult` dataclass |
| CLI | `main.py` | Single runs + `--benchmark` 30/60-run study with aggregate stats and head-to-head |
| Plotting helpers | `utils/plotting.py` | Convergence, allocation heatmap, variance box-plot, delivery map |
| Streamlit UI | `ui/app.py` | Mode selector, scenario tuner, metrics, convergence, heatmaps, per-region table, compare mode, variance plot |
| Experiments log | `experiments/benchmark_*.txt` | Evidence of stability / improvement |

---

## 4. System design

### 4.1 Module map

```
            ┌───────────────────┐
            │      ui/app.py    │  (Streamlit)
            └────────┬──────────┘
                     │
              ┌──────▼───────┐
              │  main.py CLI │
              └──────┬───────┘
                     │
              ┌──────▼────────────────┐
              │ simulation.runner     │    run_single / run_experiments
              └──────┬────────────────┘
                     │
        ┌────────────┼───────────────┐
        │            │               │
   ┌────▼───┐   ┌────▼───┐     ┌─────▼────┐
   │ pso/   │   │ hybrid/│     │ problem/ │   make_scenario
   │ solver │   │ solver │     │ fitness  │   repair_constraints
   └────┬───┘   └────┬───┘     │ validator│   fitness_function
        │            │         └─────▲────┘
        │       ┌────▼───┐           │
        │       │  ga/   │           │
        │       │ ops    │           │
        │       └────────┘           │
        └────────────────────────────┘
```

**Data flow for one run:**
1. User picks preset + algorithm + params.
2. `make_scenario(preset_name, …)` returns a `DisasterScenario`.
3. Solver creates a swarm (or a GA population seeded from PSO pBests).
4. Each iteration: `repair_constraints` → `fitness_function` → update bests.
5. Runner wraps outputs into a `RunResult` (fitness, history, coverage, feasibility, runtime).
6. CLI / UI renders metrics and plots.

### 4.2 Decision variable

Allocation matrix `A ∈ ℝ^{R×K}` where `A[i, j]` = units of resource `j` sent to region `i`. Flattened to a vector of length `R × K` for the solvers.

### 4.3 Fitness (minimisation)

```
shortage_norm = Σ max(0, demand - alloc) / total_demand
cost_norm     = Σ (units shipped per region × distance) / max_possible_cost

fitness = α·shortage_norm + β·cost_norm
         (α, β) = (0.75, 0.25)
```

Shortage weight dominates — saving lives is the top priority.

**Why only two terms?** Earlier drafts included a third term, `γ·excess_norm`, penalising over-supply (allocation above demand). That term is now gone, for a simple reason: the repair operator already caps every allocation at the region's demand (step 3 below), so after repair the excess is *mathematically* zero. Keeping a term that is always zero in the objective is dead weight — it adds a tuning knob with no effect on the score. Dropping it simplifies the weighting scheme and makes the academic objective cleaner: we optimise only things that can actually change.

### 4.4 Repair

1. Clip negatives to zero.
2. For each resource column with total > supply: scale column down proportionally.
3. Cap `alloc[i, j] ≤ demand[i, j]` (no waste).

This is a projection onto the feasible region and guarantees `check_constraints(...) == True` after one call.

---

## 5. Algorithms in detail

### 5.1 PSO step-by-step

Each particle `i` holds position `x_i` and velocity `v_i`, and remembers its best-ever position `p_i` (pBest). The swarm shares the best-ever position `g` (gBest).

```
repeat for T iterations:
    w = linearly decayed from w_max to w_min over t
    for each particle i:
        r1, r2 ~ U(0, 1)^D
        v_i ← w·v_i  +  c1·r1·(p_i - x_i)  +  c2·r2·(g - x_i)
        v_i ← clip(v_i, -v_max, +v_max)
        x_i ← x_i + v_i
        x_i ← clip(x_i, 0, upper_bound)

    for each particle i:
        x_i ← repair_constraints(x_i)
        f_i ← fitness_function(x_i)
        if f_i < pBest_fitness_i: update p_i
        if f_i < gBest_fitness:    update g
    record gBest_fitness in history
```

Three forces decide the next move: inertia (`w·v`), cognitive pull to self-best (`c1·r1·(p-x)`), social pull to global-best (`c2·r2·(g-x)`). Decaying `w` shifts the balance from exploration to exploitation.

### 5.2 GA step-by-step

Given a population `P` of genomes and their fitness:

```
repeat for T' generations:
    next_P ← top-k individuals from P     (elitism)
    while len(next_P) < |P|:
        parent1 = tournament_select(P)
        parent2 = tournament_select(P)
        if rand() < pc:
            c1, c2 = whole_arithmetic_crossover(parent1, parent2, α~U(0.3,0.7))
        else:
            c1, c2 = parent1, parent2
        if rand() < pm: c1 = non_uniform_mutation(c1, current_iter, max_iter)
        if rand() < pm: c2 = non_uniform_mutation(c2, current_iter, max_iter)
        add c1, c2 to next_P
    P ← next_P
    repair + evaluate every new child
    update best-so-far
```

**Why these operators?**
- Tournament selection (k=3): balanced pressure, O(k) per pick, no global sort needed.
- Whole-arithmetic crossover: produces feasible-looking blends for a continuous genome; random α avoids getting stuck averaging.
- Non-uniform mutation: big jumps early (exploration), small nudges later (refinement) — mirrors PSO's inertia schedule in spirit.
- Elitism (top-2): the solver **cannot** regress.

### 5.3 Hybrid logic

```
pso_iters = int(pso_fraction × total_iters)       # default 0.6
ga_iters  = total_iters - pso_iters

1. run PSOSolver for pso_iters
2. take P = [particle.pBest for particle in PSO.swarm]
3. run GA for ga_iters starting from P with elitism
4. best_fitness = min over the whole run
history = PSO history concatenated with GA best-so-far history
```

Best of both worlds: PSO's fast convergence, GA's refinement, never losing the best-so-far.

---

## 6. Experimental results

Benchmark: **60 runs** = 2 algorithms × 3 scenarios × 10 seeds, 150 iterations, pop=30. Full log in [`experiments/benchmark_60runs.txt`](experiments/benchmark_60runs.txt). A smaller 30-run study is in [`experiments/benchmark_30runs.txt`](experiments/benchmark_30runs.txt).

### 6.1 Aggregate results (60 runs)

Re-run after the fitness simplification (`α=0.75, β=0.25`, no excess term). Numbers differ slightly from older drafts because the weighting is different, but the qualitative pattern is unchanged.

| Algorithm | Scenario | Mean fitness | Std | Mean coverage | Feasibility |
|---|---|---:|---:|---:|---:|
| PSO | Balanced | 0.1634 | 0.0186 | 98.26% | 100% |
| **Hybrid** | **Balanced** | **0.1542** | **0.0026** | **99.77%** | **100%** |
| PSO | High Demand | 0.5071 | 0.0016 | 50.00% | 100% |
| Hybrid | High Demand | 0.5069 | 0.0014 | 49.99% | 100% |
| PSO | Resource Scarcity | 0.3510 | 0.0108 | 74.19% | 100% |
| **Hybrid** | **Resource Scarcity** | **0.3444** | **0.0015** | **75.53%** | **100%** |

### 6.2 Head-to-head (same seed, Hybrid vs PSO)

| Scenario | Hybrid wins | Ties | Hybrid loses |
|---|---:|---:|---:|
| Balanced | 3 | 7 | 0 |
| High Demand | 4 | 0 | 6 |
| Resource Scarcity | 4 | 5 | 1 |

### 6.3 Interpretation

- **Hybrid essentially never loses on Balanced or Resource Scarcity** — it either ties or improves (only one marginal loss in Resource Scarcity). Std-dev is ~7× lower on Balanced and ~7× lower on Resource Scarcity. The GA refinement stage stabilises what PSO finds.
- **High Demand is essentially a tie** (differences < 0.001). This scenario is **information-bounded**: total supply is 50% of total demand, so the shortage term is mathematically locked near 0.5. The only optimisable component left is transport cost (β = 0.25), where both algorithms already sit near optimum after PSO.
- **Feasibility is 100% across all 60 runs** — the repair operator always produces a valid allocation.
- **Reproducibility check:** `run_single(..., seed=42)` called twice returns identical fitness. Confirmed empirically.

### 6.4 Diversity over iterations

We track `population_diversity` each iteration — the mean standard deviation of particle positions across dimensions. It's a one-line summary of "how spread out is the swarm?".

**Why we care:**
- High diversity at the start (random init) means the algorithm is genuinely exploring.
- Diversity should drop as the algorithm converges — that's the desired behaviour.
- If diversity drops to near-zero **too early**, the swarm has prematurely converged and will not escape a local optimum. That's a failure mode the curve makes visible.

**What we see empirically (50-iter run on Balanced, seed 0):**

| Iteration | Diversity |
| ---: | ---: |
| 0 | ~3.23 |
| 50 | ~0.004 |

The diversity curve drops smoothly from ~3 to ~0 over the iterations — a healthy convergence trajectory, not a sudden collapse. In the Streamlit UI, the "Population diversity" plot shows this curve live; when running hybrid, you can see diversity stop shrinking during the GA stage because mutation reintroduces variance (this is by design — it's GA refinement).

### 6.5 Edge-case stability

| Case | Result |
|---|---|
| 2 regions | feasible, fit=0.2035, cov=100% |
| 20 regions | feasible, fit=0.1666, cov=97.71% |
| iters=5 (near-zero budget) | feasible, fit=0.4942, cov=49% |
| `pso_fraction=1.0` (pure PSO via hybrid) | feasible, fit=0.1529, cov=100% |
| `pso_fraction≈0` (almost pure GA) | feasible, fit=0.1630, cov=97.96% |
| Custom `scenario_seed=99` | feasible, fit=0.1170, cov=96.91% |

No crashes, no invalid allocations, no NaNs.

### 6.6 Parameter sensitivity observed

- **`pso_fraction = 0.6`** is a robust default. Values < 0.2 deny PSO time to locate the basin; values > 0.9 give GA too little room to refine.
- **Population size 30** plenty for the 18-dimensional problem (6 regions × 3 resources).
- **Iterations 100–150** enough to reach the plateau on all three scenarios.

---

## 7. Time complexity (intuitive view)

This is a brief, non-rigorous sketch — useful for reasoning about cost, not a formal proof.

Let:

- `N` = population size (particles or genomes)
- `D` = dimension of each solution = `num_regions × num_resources`
- `T` = number of iterations

**PSO.** Each iteration we touch every particle in every dimension (velocity update, position update, repair, fitness). So one iteration is `O(N × D)` and the full run is:

```text
PSO complexity ≈ O(N × D × T)
```

**GA.** Each generation does selection (tournament: `O(k)` per pick, `k` is small and constant), crossover and mutation (each `O(D)`), plus repair and fitness per child (`O(D)`). With `N` children per generation and `G` generations:

```text
GA complexity ≈ O(N × D × G)
```

**Hybrid.** We run PSO for a fraction `ρ` of the budget, then GA for the rest:

```text
Hybrid complexity ≈ O(N × D × (ρT + (1-ρ)G))
                  ≈ O(N × D × T)     (when PSO iters + GA iters = T)
```

In other words, hybrid is the **same big-O** as PSO or GA individually — we're not adding a new asymptotic cost, we're splitting the same iteration budget between two styles of search.

**Numeric intuition.** Using the defaults:

- `N = 30`, `D = 18` (6 regions × 3 resources), `T = 150`
- PSO: `30 × 18 × 150 ≈ 81 000` core operations in the inner loop
- Hybrid with `ρ = 0.6`: 90 PSO iters + 60 GA gens = still `30 × 18 × 150 = 81 000`

This is why a full run finishes in well under a second on a normal laptop — the work scales linearly in every parameter, and all three are modest.

If we scaled up to `N = 100`, `D = 100` (e.g., 20 regions × 5 resources), `T = 500`: `~5 × 10^6` operations — still seconds, not minutes.

---

## 8. Limitations

We want this report to be honest about what the system does and does not model. Below are the assumptions we consciously made to keep the project tractable for an academic course, and the practical implications.

1. **Static demand.** Each scenario is a snapshot: demand at region `i` for resource `j` is a fixed number. Real disasters have demand that changes hour by hour (survivors found, conditions worsening, medical needs evolving). The model would need to be extended to a multi-stage / rolling-horizon formulation to capture that.

2. **Euclidean distance, no real geography.** Transport cost uses straight-line distance from the warehouse to each region. Real logistics depend on roads, traffic, weather, collapsed bridges, and fuel availability. A more realistic cost would come from an actual road network (e.g. OSMnx / shortest-path on a graph) with edge weights for congestion and damage.

3. **Single warehouse.** We assume one aid centre. Real operations are multi-depot: regional hubs, airlifted supplies, ships. That turns the decision variable from `A ∈ ℝ^{R×K}` into `A ∈ ℝ^{W×R×K}` and couples supply constraints across warehouses.

4. **Deterministic inputs.** Everything is known: supply, demand, distances. Real decisions are made under **uncertainty** — convoys fail, demand estimates are wrong, supply arrives late. A robust version would optimise expected performance over sampled disruptions (stochastic programming, or CVaR for tail risk).

5. **Instantaneous delivery.** The cost term weights volume by distance, but we never model travel time. In practice, what matters is how many people are alive after 72 hours, which depends on delivery *latency*, not total distance.

6. **No prioritisation of vulnerable groups.** All demand is treated equally. Real triage weighs groups (children, elderly, medical cases) differently. This could be folded in as per-region weights in the shortage term.

7. **No vehicle / capacity constraints.** We don't model how many trucks there are, how much each carries, or that a truck must physically travel a route. These constraints push the problem toward a vehicle-routing formulation.

8. **Weights are fixed, not operator-chosen.** We hard-code `α=0.75, β=0.25`. In a real deployment, the relative weight of lives vs. logistics cost should be the operator's decision at the time — ideally via a Pareto front (see Future Improvements).

9. **Small problem size.** With 6 regions × 3 resources = 18 dimensions, PSO and GA are both massively over-provisioned. Scaling studies on hundreds of regions and resources would be needed to confirm the hybrid still wins at larger sizes.

10. **No regression tests.** The validator is used at runtime but we don't have a `pytest` suite that would catch a silent regression in a future refactor. This is listed in Future Improvements.

Being explicit about these limitations matters academically: it shows we understand the gap between an EA benchmark and an operational disaster-relief system, even if bridging that gap is out of scope for the course project.

---

## 9. How to run

### 9.1 Install

```bash
pip install -r requirements.txt
```

Requirements: `numpy`, `matplotlib`, `pandas`, `streamlit` (Python 3.10+).

### 9.2 Single runs via CLI

```bash
# Just PSO on the Balanced scenario
python main.py --algo pso --scenario Balanced --iters 150

# Hybrid on High Demand with a specific seed
python main.py --algo hybrid --scenario "High Demand" --seed 7

# Both algorithms on every scenario
python main.py --algo both --scenario all --iters 100
```

### 9.3 Full benchmark

```bash
# 2 algos x 3 scenarios x N seeds = 6N runs, with aggregate and head-to-head
python main.py --benchmark --runs 10 --iters 150
```

### 9.4 Streamlit UI

```bash
streamlit run ui/app.py
```

Controls:
- **Scenario**: preset, number of regions, scenario seed
- **Algorithm**: PSO or Hybrid, iterations, population, pso_fraction, seed
- **Compare mode**: side-by-side PSO vs Hybrid
- **Repeat runs**: produces a variance box-plot over N seeds

The page shows: scenario overview, warehouse map, headline metrics, convergence curves, allocation heatmap, delivery map, per-region coverage heatmap, and a full run-details table.

---

## 10. Future improvements

1. **Non-Euclidean cost.** Real disaster maps have roads, not straight lines. Swap the Euclidean distance for a road-network graph (e.g. OSMnx).
2. **Time-varying demand.** Demand rises and falls during the first 72 hours of a disaster — a multi-stage optimisation over time slices would be more realistic.
3. **Stochastic supply.** Relief convoys fail or delay. Wrap the fitness in an expected-value over sampled disruptions (CVaR or scenario optimisation).
4. **Multi-warehouse.** Current model has one aid centre; real operations are multi-depot. The decision variable becomes `A ∈ ℝ^{W×R×K}`.
5. **Adaptive `pso_fraction`.** Monitor the convergence slope; switch from PSO to GA when improvement flatlines, not at a fixed split.
6. **Alternative hybrids.** PSO + Differential Evolution, PSO + Simulated Annealing for local refinement, or Memetic PSO with a Lamarckian local search per particle.
7. **Pareto front.** Instead of a weighted sum, evolve a Pareto front over (shortage, cost) with NSGA-II and let the operator choose.
8. **Regression tests.** Add `pytest` over the validator so future refactors don't silently break feasibility.

---
