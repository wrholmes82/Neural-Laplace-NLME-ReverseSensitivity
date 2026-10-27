# Cohort — the representative synthetic cohort drawn in the paper's Figure 1.
#
#   julia --project=Local_Env -t 5 warfarin_turnover/examples/cohort/simulate_cohort_figure.jl
#
# This is a simulation and nothing else: no fit, no gradient, no information matrix. It exists so
# that the cohort figure has a generator of its own. It previously borrowed `cohort_e3.csv`, an
# incidental by-product of the capability experiment, which meant the cheapest figure in the paper
# was reproducible only by rerunning the most expensive artefact in the study.
#
# The seed and the subject count are the ones that cohort was drawn at, so the dataset written
# here is identical to the one the published figure was made from.

include(joinpath(@__DIR__, "..", "..", "src", "warfarin.jl"))

banner("Cohort — the representative synthetic cohort for Figure 1")

const RESULTS = results_dir(@__DIR__, "results")
const SEED_COHORT = 20260813
const N_SUBJECTS = 100

# `simulate_cohort` draws from M_par at the truth fitted to the real 32-subject data, so this
# depends on `data/truth_par.csv` — the artefact `examples/realdata/fit_par_truth.jl` writes.
sim = simulate_cohort(N_SUBJECTS; seed = SEED_COHORT)
nobs = n_observations(sim.pop)

@printf("\ndata    N = %d subjects, %d observations (%d conc + %d PCA), %.1f per subject\n",
        N_SUBJECTS, nobs.total, nobs.n_pk, nobs.n_pd, nobs.total / N_SUBJECTS)
@printf("doses   %s mg\n", join(sort(unique(sim.dose)), ", "))
@printf("seed    %d\n", SEED_COHORT)

write_cohort(RESULTS, sim; tag = "fig")

println("\nWrote to ", RESULTS)
for f in ("cohort_fig.csv", "cohort_eta_true_fig.csv", "cohort_truth_fig.csv")
    println("  ", f)
end
