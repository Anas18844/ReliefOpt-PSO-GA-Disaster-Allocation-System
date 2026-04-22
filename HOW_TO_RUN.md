# How to Run

A short, copy-pasteable guide. For design details see [`PROJECT_REPORT.md`](PROJECT_REPORT.md).

## 1. Install

```bash
pip install -r requirements.txt
```

Needs Python 3.10+ and the packages in `requirements.txt` (numpy, matplotlib, pandas, streamlit).

## 2. Run from the command line

```bash
# Single PSO run on the Balanced scenario
python main.py --algo pso --scenario Balanced --iters 150

# Hybrid PSO -> GA on High Demand
python main.py --algo hybrid --scenario "High Demand" --seed 7

# Both algorithms on every scenario
python main.py --algo both --scenario all
```

## 3. Full benchmark (~60 runs)

```bash
python main.py --benchmark --runs 10 --iters 150
```

Prints per-run results, aggregate statistics (mean / std / coverage /
feasibility), and a head-to-head "Hybrid vs PSO" win/tie/loss table.
Logs are saved under `experiments/`.

## 4. Streamlit UI

```bash
streamlit run ui/app.py
```

Open the URL it prints (usually <http://localhost:8501>). Use the sidebar to:

- pick a scenario preset (Balanced / High Demand / Resource Scarcity)
- choose PSO or Hybrid, set iterations / population / seed
- toggle *Compare mode* to run both and view side by side
- set *Repeat runs* for a variance box-plot

The page shows metrics, convergence curves, population-diversity curves,
allocation heatmaps, and a delivery map.

## 5. Scenarios at a glance

| Preset | Supply vs total demand | Purpose |
| --- | --- | --- |
| Balanced | ~110% of demand | Everyone can be fully served |
| High Demand | ~50% of demand | Algorithm must triage under scarcity |
| Resource Scarcity | Food + water plentiful, medicine ~30% | Tests trade-offs between resource types |

## 6. Reproducibility

All runs are deterministic given the same seed. `python main.py --algo pso --seed 42`
produces the exact same fitness every time.
