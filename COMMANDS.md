# Drone tracking fusion — command cheat-sheet

Save this as `COMMANDS.md` in the project root (`D:\BTP_project`).
All commands are for the Windows **cmd** prompt, run from the project root.

## 0. Every session starts here

```
cd D:\BTP_project
venv\Scripts\activate
```

The prompt must show `(venv) D:\BTP_project>`.
If it shows `>>>` you are inside the Python shell, not cmd. Type `exit()` and press Enter.

Always run modules with `-m` and dotted names (`eval.compare_approaches`), never as file paths.

---

## 1. Single-target pipeline (Phases 1–6)

| Step | Command | What it does / writes |
|---|---|---|
| Phase 1 baseline | `venv\Scripts\python.exe -m sim.run_baseline` | Fixed omega=0.5 fusion; prints OSPA/SIAP for airborne, ground and fused |
| Approach 1 | `venv\Scripts\python.exe -m fusion.approach1_forecast` | Trains KL-LSTM; `data/omega_lstm.pt`, `data/approach1_loss_curves.{npz,png}` |
| Approach 2 | `venv\Scripts\python.exe -m fusion.approach2_train` | Trains Dynamic Fusing LSTM; `data/dynamic_fusing_lstm.pt`, `data/approach2_loss_curves.{npz,png}` |
| Approach 3 | `venv\Scripts\python.exe -m fusion.approach3_train` | Trains the three IMM-LSTM models |
| Approach 4 | `venv\Scripts\python.exe -m fusion.approach4_evaluate` | CMA-ES knowledge base + nearest-neighbour omega (no epochs printed) |
| **Full comparison** (slow) | `venv\Scripts\python.exe -m eval.compare_approaches` | Trains all approaches, 5 methods x 4 scenarios x 5 runs; `data/phase6_comparison_results.csv` |
| **Report + graphs** | `venv\Scripts\python.exe -m eval.report_results` | `data/phase6_report.md`, `data/phase6_ospa_bar_chart.png`, `data/phase6_box_plots.png` |

Run `eval.compare_approaches` before `eval.report_results`. The report only reads the CSV.

---

## 2. Multi-target (cross-sensor track association)

| Step | Command | What it does / writes |
|---|---|---|
| Smoke test | `venv\Scripts\python.exe -m sim.multitarget_fusion` | Fixed-omega baseline on the 5 multi-target scenarios (one seed) |
| **Comparison** (slow) | `venv\Scripts\python.exe -m eval.compare_multitarget` | 5 methods x 5 scenarios x 5 runs; `data/multitarget_comparison_results.csv` |
| **Report + graphs** | `venv\Scripts\python.exe -m eval.report_multitarget` | `data/multitarget_report.md`, `data/multitarget_ospa_bar_chart.png`, `data/multitarget_box_plots.png` |
| Diagnostics | `venv\Scripts\python.exe -m eval.diagnose_multitarget` | Tracks held, pairs made, segment lengths, at native / 30 / 45 steps; `data/multitarget_diagnostics_by_length.csv` |

Run `eval.compare_multitarget` before `eval.report_multitarget`.

Reading the diagnostics: low `g_tracks` means the ground tracker is the bottleneck; `pairs` close to `g_tracks` means association is not dropping matches; low `post_warmup` means the LSTM approaches barely get to act; `in_range` well below 1.0 means targets left sensor coverage.

---

## 3. Phase 7 — GRU and TCN versions of Approach 2

| Step | Command | What it does / writes |
|---|---|---|
| Model self-test | `venv\Scripts\python.exe -m models.sequence_variants` | Checks shapes, omega in [0,1] and causality for LSTM/GRU/both TCNs. Run this first |
| Comparison | `venv\Scripts\python.exe -m eval.compare_sequence_models` | Trains LSTM, GRU, TCN (raw) and TCN (log input) from 3 inits each, evaluates on the Phase 6 protocol; `data/phase7_sequence_models.csv`, `data/phase7_sequence_models_report.md` (includes an omega-behaviour table: a std near 0 means a near-constant omega) |
| Quicker trial | `venv\Scripts\python.exe -m eval.compare_sequence_models --init-seeds 1 --reps 2` | Same, smaller, to check it runs before the full one |

---

## 4. Files added in this work (where each goes)

| File | Folder |
|---|---|
| `track_association.py` | `fusion/` |
| `multitarget_policies.py` | `fusion/` |
| `multitarget_fusion.py` | `sim/` |
| `compare_multitarget.py` | `eval/` |
| `diagnose_multitarget.py` | `eval/` |
| `report_multitarget.py` | `eval/` |
| `sequence_variants.py` | `models/` |
| `compare_sequence_models.py` | `eval/` |

Also required: in `sim/scenarios.py`, after the `SINGLE_TARGET_SCENARIOS` list, add

```python
MULTI_TARGET_SCENARIOS = [
    SCENARIOS_BY_NAME["parallel"],
    SCENARIOS_BY_NAME["crossing"],
    SCENARIOS_BY_NAME["converging"],
    SCENARIOS_BY_NAME["diverging"],
    SCENARIOS_BY_NAME["high_clutter_multitarget"],
]
```

---

## 5. Git (your branch is `V2`)

```
git status
git add fusion\track_association.py fusion\multitarget_policies.py sim\multitarget_fusion.py sim\scenarios.py
git add eval\compare_multitarget.py eval\diagnose_multitarget.py eval\report_multitarget.py
git add models\sequence_variants.py eval\compare_sequence_models.py COMMANDS.md
git commit -m "Add multi-target association, multi-target comparison, Phase 7 GRU/TCN"
git push
```

Check `.gitignore` before adding anything from `data/`; results files can be large.

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `SyntaxError: unexpected character after line continuation character` | You typed a command inside the Python shell (`>>>`). `exit()`, then retry |
| `ModuleNotFoundError: No module named 'fusion'` (or `sim`, `eval`) | Not in the project root, or ran a file path instead of `-m module.name` |
| `ModuleNotFoundError: ... MULTI_TARGET_SCENARIOS` | The list from section 4 is missing from `sim/scenarios.py` |
| `skipped: No fused tracks produced` | Association paired nothing in that run; try another seed, or the 500 m gate in `fusion/track_association.py` may be too tight |
| `IndexError: timestamp not found in states` | Old `multitarget_fusion.py` without gap-free segments; use the latest version |
| `RuntimeWarning: invalid value encountered in scalar divide` in a report | A metric had no variation (e.g. Ambiguity all 1.0), so its Friedman test cannot be computed. Harmless |
| Comparison seems stuck | It is slow (several LSTM trainings plus CMA-ES). Progress lines print per scenario and run |