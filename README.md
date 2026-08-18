# Fusion of Drone Tracking — LSTM & CMA-ES Approaches

Reproduction and extension of:

> Abu Zitar, R., Fares, S., El Fallah Seghrouchni, A., & Barbaresco, F. (2025).
> *Fusion of drone tracking using different LSTM approaches and a CMA-ES knowledge base approach.*
> Neural Computing and Applications, 37, 9991–10036. https://doi.org/10.1007/s00521-025-11060-5

Paper analysis and phased implementation plan: [`docs/paper_analysis_notes.txt`](docs/paper_analysis_notes.txt).

## Problem

Two radar sensors independently track the same target. Covariance Intersection (CI) fuses
their estimates using a mixing parameter ω. Instead of a fixed ω, this project learns a
time-varying ω via four approaches, benchmarked against the fixed-ω baseline:

1. **KL-divergence LSTM** — offline KL-optimal ω labels, LSTM forecasts future ω.
2. **Dynamic Fusing LSTM** — LSTM consumes raw sensor covariances, outputs ω(t) directly.
3. **IMM-LSTM** — per-sensor innovation-minimizing LSTMs feed a fusing LSTM.
4. **CMA-ES knowledge base** — offline evolutionary search builds a lookup table;
   nearest-neighbor retrieval at inference.

Built on [Stone Soup](https://github.com/dstl/Stone-Soup) (JPDA / GM-LCC trackers), evaluated
with OSPA distance and SIAP metrics.

## Layout

```
sim/       ground truth, sensors, scenarios, simulation driver
trackers/  JPDA and GM-LCC per-sensor trackers
fusion/    covariance intersection + the four ω-selection approaches
models/    PyTorch model definitions
eval/      OSPA/SIAP metrics, cross-approach comparison, reporting
docs/      paper analysis and blueprint
data/      generated datasets, trained models, results (gitignored)
```

## Setup

```bash
venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch is CPU-only (no GPU required).

## Running

```bash
venv\Scripts\python.exe sim\hello_world.py       # smoke test
venv\Scripts\python.exe -m sim.run_baseline       # fixed-ω baseline
venv\Scripts\python.exe -m fusion.approach1_forecast
venv\Scripts\python.exe -m fusion.approach2_train
venv\Scripts\python.exe -m fusion.approach3_train
venv\Scripts\python.exe -m fusion.approach4_evaluate
venv\Scripts\python.exe -m eval.compare_approaches   # full 5-method comparison
venv\Scripts\python.exe -m eval.report_results       # tables, charts, report
```

## Status

`fusion/covariance_intersection.py` (Covariance Intersection itself) is intentionally left
unimplemented. Everything else — trackers, sensors, scenarios, all four ω-selection
approaches, and the comparison/reporting pipeline — is built and will run once that function
is filled in.

Not yet done: folding the multi-target scenarios (`parallel`/`crossing`/`converging`/
`diverging`) into the comparison pipeline, which needs cross-sensor track-to-track
association; and Phase 7 optional extensions (GRU/TCN variants, real sensor data).
