# Runbook

Everything runs from the repository root against `Local_Env`. The thread count is explicit on every command because every reported timing depends on it:

```
julia --project=Local_Env -t 5 --heap-size-hint=20G <script>
```

`-t 5` throughout (five usable cores on the development machine). `--heap-size-hint=20G` matters for E2: the O(p) forward-sensitivity arms allocate heavily and, under Julia's default heap target, garbage collection dominates the timing. The heap hint, thread count and machine load are recorded on every timed row.

## Setup

```
julia setup.jl
```

Dev-installs the `NLMEReverse` package (`src/`) into `Local_Env`, adds the packages the scripts use directly, and precompiles. The committed `Local_Env/Manifest.toml` pins every version.

## Order

| # | command | writes | needs |
|---|---|---|---|
| 0 | `test/runtests.jl` | — | — |
| 0b | `test/verification/verify_reverse.jl`, `verify_foce.jl` | `test/verification/results/` | 0 |
| 1 | `warfarin_turnover/examples/realdata/fit_par_truth.jl` | `warfarin_turnover/data/truth_par.csv`, `examples/realdata/results/` | 0 |
| 2 | `warfarin_turnover/test/e1_gate.jl` (**hard gate**) | `warfarin_turnover/test/results/e1_gate.csv` | 0 |
| 3 | `warfarin_turnover/examples/cost/e2_cost_ladder.jl` | `examples/cost/results/e2_*.csv` | 1, 2 |
| 4 | `warfarin_turnover/examples/cohort/simulate_cohort_figure.jl` | `examples/cohort/results/cohort_*.csv` | 1 |
| 5 | `warfarin_turnover/examples/misspecification/e9b_penalised_curves.jl`, then `e9c_model_comparison.jl` | `examples/misspecification/results/e9b_*`, `e9c_*` | 1, 2 |
| 6 | `warfarin_turnover/examples/regularisation/e11_penalised_inference.jl` | `examples/regularisation/results/e11_*` | 1, 2 |
| 7 | `warfarin_turnover/examples/regularisation/e15_realdata_replicates.jl 0`, `… 1`, `… 10`, then `e15b_mpar_timing.jl`, `e15c_summary_table.jl`, `e15d_df_eff_blocks.jl` | `examples/regularisation/results/e15_*`, `e15b_*` | 2 |
| 8 | `warfarin_turnover/examples/cost/e14_single_run_budget.jl`, then `e14b_cost_table.jl`, `e14c_information_timing.jl`, `e14d_compile_penalised.jl` | `examples/cost/results/e14*.csv` | 3, 6 |
| 9 | `warfarin_turnover/figures/make_figures.jl` (`-t 1`) | `figures/*.png`, `*.svg` | 3–7 |

Notes:

- **Step 1 is a prerequisite.** The simulation truth is the `M_par` fit to the real data. The simulator refuses to run without `truth_par.csv`, and step 1 refuses to write it if its own fit did not converge. The committed `truth_par.csv` is the one the committed results were generated from.
- **Step 2 is a hard gate.** It exits non-zero on failure. Nothing downstream is meaningful until it passes.
- **Step 7** takes the ridge strength as its first argument (`0`, `1`, `10`), one process per strength. A second argument `smoke` runs a short check into `results_smoke/`. Steps 5 and 6 accept `smoke` as their first argument for the same purpose.
- **Step 9 reads committed CSVs only.** It fits, simulates and evaluates nothing, so the figures can be regenerated in about a minute from the results in this repository. `make_figures.jl` also accepts one of `cohort|e2|e11|e9|e15` to draw a single figure.

## Artefacts

Every result is a CSV with an explicit header. In addition:

- `*_information_*.bin` (+ `.dims.csv`) is a raw `p × p` observed-information matrix, little-endian `Float64` in column order. `e9c_model_comparison.jl` and `e15d_df_eff_blocks.jl` read these, which is why they are committed.
- `e15_curves_g*.csv` carries the fitted effect curve and its delta-method band on a fixed concentration grid; `e15_functionals_g*.csv` carries the five scalar functionals, their standard errors, and the identified share ρ.
- Timings are taken with `BLAS.set_num_threads(1)`, set in scripts only. `loadavg1` is recorded on every timed row.
