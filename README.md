# NeuralNLME — reverse-sensitivity Laplace estimation for neural nonlinear mixed-effects models

The population gradient of the Laplace approximation to the NLME marginal likelihood — including the implicit dependence of the empirical Bayes estimates and the derivative of the log-determinant — is written as the gradient of a single scalar per subject, so that one reverse-mode sweep yields the full gradient at a cost independent of the number of population parameters `p`. The accompanying study applies this to a joint PK–PD model of the classic 32-subject warfarin data with a neural network in place of the parametric concentration–effect function, and shows how objective function values, functional confidence intervals and effective parameter counts carry over to the neural-NLME model.

Every figure and summary table is produced by a script in this repository, and the CSV artefacts those scripts wrote are committed, so the figures can be regenerated without rerunning the fits.

## Layout

```
src/                     NLMEReverse — the problem-agnostic Julia package
                         (Laplace/FOCE objective, inner mode solve, four independent gradient
                         constructions, observed information, spectral-cutoff inverse)
test/                    the package's tests (tiers 0–7) and verification gates
Local_Env/               the run environment: Project.toml + Manifest.toml (Julia 1.11.4)
setup.jl                 one-time environment preparation
warfarin_turnover/       the study
  data/                  warfarin.csv (32 subjects, from nlmixr2data) and truth_par.csv
  src/                   turnover PK–PD model, neural effect function, simulator, fit driver
  test/e1_gate.jl        the gradient verification gate
  examples/
    realdata/            M_par fit to the real data: the simulation truth
    cohort/              the synthetic cohort
    cost/                E2 cost ladder, E14 runtime budget
    misspecification/    E9b/E9c recovery under misspecification
    regularisation/      E11 synthetic inference, E15 real-data replicates
  figures/               make_figures.jl and the figures it writes
```

## Running

Requires Julia 1.11 (the committed `Local_Env/Manifest.toml` pins Julia 1.11.4 and every package version the results were produced with). From the repository root:

```
julia setup.jl                                   # once: dev-installs src/ into Local_Env and instantiates
julia --project=Local_Env -t 5 test/runtests.jl  # package tests, tiers 0–7
```

Then see `RUNBOOK.md` for the study, step by step. The short version:

```
julia --project=Local_Env -t 5 warfarin_turnover/examples/realdata/fit_par_truth.jl    # simulation truth
julia --project=Local_Env -t 5 warfarin_turnover/test/e1_gate.jl                       # verification gate
julia --project=Local_Env -t 1 warfarin_turnover/figures/make_figures.jl                # figures from committed CSVs
```

Thread counts (`-t 5`) are part of every reported timing; see `RUNBOOK.md`.

## The package in brief

A model supplies one method, `negative_log_joint_density(model, η, θ, subject)`, and every derivative is obtained from it by automatic differentiation. Four constructions of the population gradient live side by side — the reverse sweep, a full-implicit adjoint, forward sensitivity in the Almquist et al. orientation, and finite differences — because agreement between routes that share no implementation is what verifies any of them (`warfarin_turnover/test/e1_gate.jl`). Reverse-mode engines are loaded at run time from `src/backends/`; Mooncake.jl is the engine used for all results.

## Data

The warfarin data are the `warfarin` dataset distributed with the R package `nlmixr2data`, originally from O'Reilly and colleagues. `warfarin_turnover/data/warfarin.csv` is that dataset in the long format the scripts read.
