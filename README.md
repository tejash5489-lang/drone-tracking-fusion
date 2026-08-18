# Fusion of Drone Tracking — LSTM & CMA-ES Approaches

Reproduction + extension project based on:

> Abu Zitar, R., Fares, S., El Fallah Seghrouchni, A., & Barbaresco, F. (2025).
> *Fusion of drone tracking using different LSTM approaches and a CMA-ES knowledge base approach.*
> Neural Computing and Applications, 37, 9991–10036. https://doi.org/10.1007/s00521-025-11060-5

Source analysis and full phased blueprint: [`docs/paper_analysis_notes.txt`](docs/paper_analysis_notes.txt)
(also available as the original [`docs/Drone_Tracking_Fusion_Paper_Analysis_and_Blueprint.docx`](docs/Drone_Tracking_Fusion_Paper_Analysis_and_Blueprint.docx)).

## Problem

Two radar sensors each track the same drone. Covariance Intersection (CI) fuses their
estimates using a mixing parameter ω. Instead of a fixed ω (the baseline, ω = 0.5), this
project learns a time-varying ω via four approaches:

1. **Adaptive LSTM** — offline KL-divergence-optimal ω labels → LSTM forecasts future ω.
2. **Dynamic Fusing LSTM** — LSTM consumes raw sensor covariances, outputs ω(t) directly,
   trained end-to-end against OSPA/SIAP.
3. **IMM-LSTM** — three LSTMs; two tune per-sensor JPDA gating via innovation minimization,
   a third learns ω minimizing RMSE to ground truth.
4. **CMA-ES knowledge base** — offline evolutionary search builds a (state → optimal ω)
   lookup table; nearest-neighbor retrieval at inference (no live optimization).

All are benchmarked against the fixed-ω baseline using OSPA distance and SIAP metrics inside
the [Stone Soup](https://github.com/dstl/Stone-Soup) tracking simulation framework.

## Project layout

```
sim/         ground-truth + sensor + clutter simulation (Stone Soup wiring)
trackers/    per-sensor trackers (JPDA, GM-LCC) and CI fusion
fusion/      the four ω-selection approaches (LSTM x3, CMA-ES)
models/      LSTM model definitions (PyTorch) and CMA-ES knowledge base
eval/        OSPA/SIAP metrics, Friedman test, comparison plots
notebooks/   exploratory analysis
data/        generated simulation data, trained models, knowledge base (gitignored)
docs/        paper analysis + blueprint
```

## Setup

```bash
# from D:\BTP_project
venv\Scripts\activate
pip install -r requirements.txt
```

Deep learning framework: **PyTorch** (CPU-only wheel — no GPU required, per blueprint §4.2).

Run the Phase 0 smoke test or Phase 1 baseline (as modules, from the project root, so the
`sim`/`trackers`/`fusion`/`eval` packages resolve):

```bash
venv\Scripts\python.exe sim\hello_world.py
venv\Scripts\python.exe -m sim.run_baseline
```

## Status

**Phase 0** (environment + scaffolding) — done.

**Phase 1** (baseline simulation + fixed-ω fusion) — scaffolded, one piece left:

- [`sim/scenario.py`](sim/scenario.py) — 3D NCV ground-truth generator
- [`sim/sensors.py`](sim/sensors.py) — airborne (moving) + ground (fixed) radar, with clutter
- [`trackers/jpda_tracker.py`](trackers/jpda_tracker.py) — JPDA tracker (airborne radar), verified
- [`trackers/gmlcc_tracker.py`](trackers/gmlcc_tracker.py) — GM-LCC tracker (ground radar), verified
  — Stone Soup ships `LCCUpdater` natively, so the blueprint's flagged risk here ("may need to
  implement GM-LCC from scratch") turned out to be a non-issue
- [`eval/metrics.py`](eval/metrics.py) — OSPA + SIAP wiring, verified
- [`fusion/covariance_intersection.py`](fusion/covariance_intersection.py) — **left for you to
  implement** — the CI fusion equations (paper's Eqs. 2–4). Every later approach (Phases 2–5)
  is just a different way of picking ω(t) before calling this same function, so it's worth
  writing by hand once.
- [`sim/run_baseline.py`](sim/run_baseline.py) — orchestrates all of the above end-to-end;
  will raise `NotImplementedError` at the fusion step until you fill in
  `covariance_intersection.py`, everything upstream of it is verified working.
- [`sim/two_radar_simulation.py`](sim/two_radar_simulation.py) — the shared "run both
  sensors/trackers, get a snapshot per step" driver factored out of `run_baseline.py` so
  Phase 2+ don't reimplement it.

**Phase 2** (Approach 1: KL-divergence labels + forecasting LSTM) — scaffolded and verified
end-to-end (with a throwaway local CI implementation, since `covariance_intersection.py` isn't
filled in yet — everything here calls the real one and will Just Work once it is):

- [`fusion/kl_objective.py`](fusion/kl_objective.py) — the KL-divergence ω objective (paper's
  Eq. 7) and its L-BFGS-B optimizer. **Read the module docstring — it now documents two dead
  ends, not one.** First attempt (`KL(sensor‖fused)`) always minimized at ω=0/1. The Phase 2 fix
  (`KL(fused‖sensor)`) looked correct on a handful of examples but turned out to be a
  mathematical identity — proven by testing 30 fully random (unrelated to any tracking
  scenario) covariance/mean quadruples, all landing within 1e-9 of exactly ω=0.5, confirmed not
  to be an optimizer-tolerance artifact. **This was only caught during Phase 6**, by testing a
  scenario with a genuine +150m sensor bias and finding even that didn't move it — see below.
  The version actually used now is the *symmetrized* KL
  (`0.5·[KL(A‖fused)+KL(fused‖A)] + 0.5·[KL(B‖fused)+KL(fused‖B)]`), which breaks both
  degeneracies: verified non-degenerate on the same 30 random quadruples (std 0.093, range
  [0.36, 0.65]) and on real tracking data.
- [`sim/generate_omega_dataset.py`](sim/generate_omega_dataset.py) — runs many short
  simulations and records the KL-optimal ω at each step where both sensors have a track. With
  the corrected objective, the original benign scenario now gives real, non-degenerate labels
  (mean 0.31, std 0.11, individual sequences show genuine smooth temporal structure — e.g.
  `[0.40, 0.35, 0.26, 0.19, 0.13, 0.09, ...]` — exactly what an LSTM should be able to learn to
  forecast) instead of a flat 0.5 everywhere. Re-ran Approach 1's full training on the corrected
  dataset: loss now drops meaningfully over 20 epochs (0.078→0.055) rather than collapsing to
  ~0 in one step, which is itself a symptom worth knowing — a loss that hits zero immediately
  usually means the label distribution was trivial, not that training went great.
- [`models/omega_lstm.py`](models/omega_lstm.py) — the paper's 2×LSTM(20) + Dense(10)
  architecture in PyTorch (see docstring for the Keras `TimeDistributed` → PyTorch translation).
- [`fusion/approach1_forecast.py`](fusion/approach1_forecast.py) — windowing, training loop
  (Adam, MSE, 20 epochs, train/val split), and autoregressive multi-step forecasting. Run via
  `venv\Scripts\python.exe -m fusion.approach1_forecast` once `covariance_intersection.py` is
  done — it'll generate the dataset, train, and save loss curves + the trained model to `data/`.

**Phase 3** (Approach 2: Dynamic Fusing LSTM) — scaffolded and verified end-to-end. Unlike
Phase 2, **training itself needed no CI stand-in and is already fully working for real** — it's
only the final real-metrics evaluation step that's still gated on your CI implementation:

- [`fusion/covariance_intersection_torch.py`](fusion/covariance_intersection_torch.py) — a
  second implementation of the *same* CI formula you're writing, but in PyTorch instead of
  numpy, so gradients can flow from a loss on the fused track back through omega into the LSTM.
  Verified to match the numpy formula to numerical precision. Your numpy version stays the one
  used everywhere else (it's what Stone Soup's Track objects need); this one exists purely to
  make Approach 2's end-to-end training differentiable.
- [`models/dynamic_fusing_lstm.py`](models/dynamic_fusing_lstm.py) — LSTM(50 hidden units) that
  ingests both sensors' flattened covariances (upper-triangle only, 21 values each — the matrix
  is symmetric, so the full 36-value flatten would just duplicate 15 of them) and outputs ω(t)
  directly, no label-generation step.
- [`sim/generate_approach2_dataset.py`](sim/generate_approach2_dataset.py) — collects raw
  (sensor state, covariance, ground truth) sequences; simpler than Approach 1's dataset since
  there's no KL optimization step here.
- [`fusion/approach2_train.py`](fusion/approach2_train.py) — **read the module docstring**: the
  paper trains against OSPA/SIAP discrepancy, but those metrics aren't differentiable (blueprint
  flags this explicitly as a risk). The training loss here is a differentiable surrogate — for
  our single-target scenario, OSPA/SIAP's positional and velocity accuracy terms reduce exactly
  to Euclidean position/velocity error, so that's what's backpropagated through. The *real*
  OSPA/SIAP (via `eval.metrics`) are computed separately at evaluation time,
  `evaluate_approach2()`, using your real numpy `covariance_intersection` — training and
  evaluation deliberately use different code paths, since only one of them can carry gradients.
  Ran a full 50-epoch training locally (loss curves + trained model already saved to `data/` —
  legitimate output, not a throwaway test); only `evaluate_approach2()` still hits your
  `NotImplementedError`, right where it should. Run via
  `venv\Scripts\python.exe -m fusion.approach2_train`.

**Phase 4** (Approach 3: IMM-LSTM / innovation minimization) — scaffolded and verified
end-to-end, including a full real training run. **This phase required the most interpretation**
— the blueprint's description of Approach 3 is high-level (no closed-form Eq. 12/13 in the
extracted notes), and "tunes the JPDA gating threshold" describes tuning a discrete,
non-differentiable hyperparameter, the same class of problem Approach 2 hit with OSPA/SIAP.
**Read the docstrings in the two files below before trusting the specifics** — once you've read
the paper's actual equations and Fig. 13, this is the piece most likely to need reshaping:

- [`models/innovation_lstm.py`](models/innovation_lstm.py) — LSTM-1/LSTM-2 architecture, output
  is a per-step trust weight in [0,1] rather than a literal gate threshold — see the docstring
  for why, and for the design's honest limitations.
- [`fusion/approach3_innovation.py`](fusion/approach3_innovation.py) — the trust weight blends
  each step's tracked state against a pure-dynamics prediction from the previous *refined*
  estimate (`x_refined(t) = trust(t)·x(t) + (1-trust(t))·F·x_refined(t-1)`), with a matching
  covariance blend. Minimizing the resulting innovation + trace(covariance) trains the network
  to do what a well-tuned gate would, without needing JPDA itself to be differentiable.
- [`fusion/approach3_train.py`](fusion/approach3_train.py) — orchestrates all three LSTMs:
  trains LSTM-1 (airborne) and LSTM-2 (ground) to convergence, freezes them, feeds their refined
  outputs to LSTM-3 (which reuses Approach 2's `DynamicFusingLSTM` architecture unchanged — same
  computational role, just fed refined instead of raw covariances), trained on plain RMSE against
  ground truth (the paper's actual, simpler loss for this LSTM — no surrogate needed here).
  Ran the full pipeline for real: all three loss curves decrease cleanly (saved to `data/`, not
  a throwaway test). Only `evaluate_approach3()` — final real OSPA/SIAP via your numpy CI — hits
  the expected `NotImplementedError`. Run via `venv\Scripts\python.exe -m fusion.approach3_train`.

**Phase 5** (Approach 4: CMA-ES offline knowledge base) — scaffolded and verified end-to-end
(with a throwaway CI stand-in — this phase needs your CI from the very first step, unlike
Phase 3, since CMA-ES's own objective calls it directly; no training involved at all, this
approach has no neural network):

- [`fusion/cma_es_objective.py`](fusion/cma_es_objective.py) — `log(det(Pcc))` objective (paper's
  Eq. 14) optimized via the `cma` package with the paper's specified hyperparameters
  (x0=0.5, σ0=0.5, 100 iterations). **Worth knowing**: pycma explicitly warns that 1-D
  optimization "is not supported and may bail or work poorly" — ω being a single scalar hits
  this directly (confirmed by a real crash during verification). Fixed with the standard
  workaround: pad the search vector to 2-D with an unused dummy coordinate.
- [`sim/generate_cma_es_knowledgebase.py`](sim/generate_cma_es_knowledgebase.py) — builds the
  offline (covariance-fingerprint → optimal ω) database, same "many shorter runs" scaling
  choice as Approach 1's dataset. **Contrast worth noting**: this objective only looks at
  covariances, never sensor *disagreement* — so unlike Approach 1's KL labels (which landed on
  ω≈0.5 almost everywhere in our benign scenario), this knowledge base came out genuinely varied
  (mean 0.95, std 0.20, spanning the full range) — CMA-ES and KL-divergence are picking up on
  different signals entirely, not just different optimizers for the same target.
- [`fusion/cma_es_lookup.py`](fusion/cma_es_lookup.py) — nearest-neighbor lookup via
  z-score-normalized Euclidean distance (paper's Eqs. 16-17) against the knowledge base — no
  optimization at inference, just a search.
- [`fusion/approach4_evaluate.py`](fusion/approach4_evaluate.py) — real OSPA/SIAP evaluation
  plus the blueprint's explicit latency benchmark (knowledge-base lookup vs. an LSTM forward
  pass). Verified end-to-end: lookup came out ~2.8x faster than a comparable LSTM call —
  matching the paper's own claimed advantage for this approach. Run via
  `venv\Scripts\python.exe -m fusion.approach4_evaluate` once CI is done.

That's all four approaches (Phases 2-5) scaffolded now, alongside the Phase 1 baseline. Every
piece that doesn't strictly need your CI has been run for real; everything that does stops at
the same `NotImplementedError`, ready to go the moment it's filled in.

**Phase 6 (in progress)** — scenario diversity, the first deliverable of Phase 6
(blueprint: "Design 7 (or more) distinct 3-D multi-target simulation scenarios"):

- [`sim/scenarios.py`](sim/scenarios.py) — 7 `ScenarioConfig` definitions: `parallel`,
  `crossing`, `converging`, `diverging` (multi-target, 2 targets each, geometry
  range-checked against both radars' max_range), plus `airborne_degraded`,
  `ground_degraded` (single target, one sensor's clutter rate raised well above normal),
  and `high_clutter_multitarget`. The paper analysis this project is built from has no
  Fig. 23 image, so these geometries are this project's own design, not a reproduction.
- [`sim/two_radar_simulation.py`](sim/two_radar_simulation.py) — added
  `run_scenario_simulation()`, a multi-target-capable driver, as a **new, separate**
  function rather than generalizing the existing `run_two_radar_simulation()` in place —
  every Phase 1-5 module was verified against that function's exact single-track-pair
  shape, and reshaping it risked silently breaking that work. Regression-tested
  `run_baseline.py` after adding this — unchanged. Both JPDA and GM-LCC were verified to
  correctly track 2 simultaneous targets (untested until now — every prior phase only
  ever exercised 1 target).
- [`sim/verify_scenarios.py`](sim/verify_scenarios.py) — fast structural smoke test across
  all 7 (no CI needed). All 7 ran without crashing, including under `ground_degraded`'s 6x
  clutter, which specifically stresses GM-LCC's known numerical fragility (Phase 2's fix
  held up).

**The degenerate-omega finding is now resolved** (was an open question as of the previous
session; chasing it further is what resolved it). Recap of the investigation, in order:

1. Built `_degraded` and `crossing` scenarios (clutter-based) specifically to fix Phase 2's
   "KL-omega always 0.5" finding. Tested across ~600 real steps — still always 0.5.
2. Added [`sim/sensors.py`](sim/sensors.py)'s `apply_measurement_bias()` and a `ground_biased`
   scenario (persistent +150m range bias, 10x the sensor's own noise std) to inject *genuine*
   mean-level disagreement, since clutter alone only affects track *confidence*, not *accuracy*.
   Still always 0.5, even with a 150m bias and even with a 1235m gap on an unconverged track.
3. That result was surprising enough to question the objective itself rather than the
   scenarios: tested the KL(fused‖sensor) direction against 30 fully random (xa, Pa, xb, Pb)
   quadruples with no connection to any tracking scenario — every single one landed within
   1e-9 of exactly 0.5. **This is a mathematical identity, not a property of any scenario** —
   the Phase 2 fix was never going to produce a usable label no matter how the scenarios were
   designed.
4. Fixed the objective itself: symmetrizing each KL term
   (`0.5·[KL(A‖fused)+KL(fused‖A)]`) breaks the identity. Verified non-degenerate on the same
   30 random quadruples, then on real data — the original benign scenario now gives real
   variation (mean 0.31, std 0.11) and the biased scenario shifts further (mean 0.61, std 0.28)
   in the sensible direction (favoring the unbiased sensor).

Full detail in [`fusion/kl_objective.py`](fusion/kl_objective.py)'s docstring. Net effect:
Approach 1's dataset generator and training loop needed no changes at all — only the objective
function did — and re-running both on the corrected objective confirms real, learnable
temporal structure in the labels now (individual sequences show smooth trends like
`[0.40, 0.35, 0.26, 0.19, 0.13, 0.09, ...]`), and training loss drops meaningfully over 20
epochs instead of collapsing to ~0 in one step the way a trivial constant target would.

**Phase 6 comparison pipeline — built and verified end-to-end** (with a throwaway CI
stand-in; a full real run is the very next thing to do once CI lands):

- Scope: the 4 *single-target* scenarios (`baseline`, `airborne_degraded`, `ground_degraded`,
  `ground_biased` — [`sim/scenarios.py`](sim/scenarios.py)'s `SINGLE_TARGET_SCENARIOS`). The 4
  multi-target geometric scenarios (`parallel`/`crossing`/`converging`/`diverging`) and
  `high_clutter_multitarget` are validated for tracking (Phase 6's first session) but not
  wired into this comparison — doing so needs cross-sensor track-to-track association
  (matching which of the airborne sensor's tracks corresponds to which of the ground sensor's,
  when there's more than one target), which is real, well-scoped future work, not attempted
  here. Documented explicitly in [`sim/two_radar_simulation.py`](sim/two_radar_simulation.py)'s
  `run_scenario_simulation` docstring rather than silently skipped.
- [`sim/two_radar_simulation.py`](sim/two_radar_simulation.py) — added
  `run_single_target_scenario()`, an adapter letting any single-target `ScenarioConfig` plug
  into the existing (already-verified) Approach 1-4 pipelines without changing their control
  flow. Verified byte-for-byte equivalent to the original function on matching metrics across
  3 seeds before building anything on top of it.
- [`sim/generate_omega_dataset.py`](sim/generate_omega_dataset.py),
  [`sim/generate_approach2_dataset.py`](sim/generate_approach2_dataset.py),
  [`sim/generate_cma_es_knowledgebase.py`](sim/generate_cma_es_knowledgebase.py) — each gained
  an optional `scenarios=` parameter (default `None` = exact original behaviour, zero
  regression risk) to pool training data across multiple scenarios instead of just the
  original benign one.
- [`fusion/approach1_forecast.py`](fusion/approach1_forecast.py) gained `evaluate_approach1()`
  (Approaches 2-4 already had one; Approach 1 didn't need one until there was something to
  compare it against) — uses a short KL-computed warmup history, then the model's own
  autoregressive forecast for the rest, matching the blueprint's actual inference description.
  `evaluate_approach2/3/4` and `sim/run_baseline.py`'s new `evaluate_fixed_baseline()` all
  gained a matching `scenario=` parameter.
- [`eval/compare_approaches.py`](eval/compare_approaches.py) — trains all 4 approaches once
  (pooled across the 4 scenarios), then evaluates all 5 methods (Fixed + Approaches 1-4) ×
  5 repetitions × scenario into a tidy DataFrame. A method that fails on a given draw (no
  confirmed track — tracking is stochastic) is skipped for that draw only, not treated as
  failing every other method's repetition.
- [`eval/report_results.py`](eval/report_results.py) — mean±std tables, `scipy.stats.friedmanchisquare`
  per metric (correctly handles a metric like Ambiguity being constant across a scenario —
  reports it as not-significant rather than crashing), bar chart of mean OSPA per
  method/scenario, box plots of OSPA/Completeness distribution per method, and a written
  Markdown report naming a winner per metric and an overall recommendation.

Ran the whole thing end-to-end (reduced scope — 3 scenarios, 3 training runs, 3 repetitions —
for speed) with a throwaway CI: 45 evaluation rows collected correctly, tables/charts/report
all generated without error. Run for real via
`venv\Scripts\python.exe -m eval.compare_approaches` then
`venv\Scripts\python.exe -m eval.report_results` once CI is done — full-scale defaults are
10 training runs × 4 scenarios and 5 repetitions × 4 scenarios × 5 methods (100 evaluation
runs); expect several minutes, mostly from `ground_degraded`'s 6x clutter making GM-LCC's
mixture math meaningfully slower per run (~7s vs ~0.6s for the other scenarios).

Phase 7 (optional extensions) and folding in the 3 multi-target-only scenarios remain, but
the core Phase 0-6 pipeline described in the blueprint is now fully scaffolded end to end.

See `docs/paper_analysis_notes.txt` section 3 for the full phase-by-phase plan (Phases 0–7).
