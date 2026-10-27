# E14c — how long an observed information takes at the penalised estimate.
#
#   julia --project=Local_Env -t 5 --heap-size-hint=20G warfarin_turnover/examples/cost/e14c_information_timing.jl
#
# The runtime table reports estimation at γ = 1, so the information row should be measured there
# too rather than carried over from γ = 0. In principle the two should be indistinguishable: the
# observed information is always formed from the UNPENALISED gradient, and its cost is 2p
# evaluations of the same closure whichever estimate it is evaluated at. In practice the estimate
# differs, and a conditional-mode solve can take a different number of inner Newton steps at a
# different θ̂, so the claim is worth measuring instead of assuming.
#
# E11 fitted fifteen 100-subject cohorts at γ = 1 and archived θ̂ for each, so nothing needs
# refitting: this reads those estimates, rebuilds each cohort from the same seed, and times the
# information alone. One warm-up is discarded first, so no cohort absorbs compilation — the same
# discipline every other timing in this project follows.

include(joinpath(@__DIR__, "..", "..", "src", "warfarin.jl"))

BLAS.set_num_threads(1)
load_backends!(verbose = false)
banner("E14c — observed information timing at the penalised estimate")

const E11 = normpath(joinpath(@__DIR__, "..", "regularisation", "results"))
const RESULTS = results_dir(@__DIR__, "results")
const WIDTH = SCIENTIFIC_WIDTH
const N_SUBJECTS = 100
const LAMBDA = 1.0

cfg = default_config(reg = 1e-8, inner_tol = 1e-8, inner_maxiter = 200, threaded = true)
rcfg = reverse_config()
m_par = WarfarinTurnover(EmaxEffect())
m_nn = WarfarinTurnover(MLPEffect(WIDTH))
tp = truth_theta()

cohort_pop(r) = begin
    sim = simulate_cohort(N_SUBJECTS; seed = SEED_COHORT + r, theta = tp.theta, model = m_par)
    PopulationData([make_subject(m_nn, s.id, s.input, s.covariates.pk_times,
                                 s.observations[1:s.covariates.n_pk], s.covariates.pd_times,
                                 s.observations[s.covariates.n_pk+1:end]) for s in sim.pop.subjects])
end

read_theta(path) = Float64.(readdlm(path, ','; header = true)[1][:, 2])

files = sort(filter(f -> startswith(f, "e11_theta_c") && endswith(f, "_lam1.csv"), readdir(E11)))
cohorts = [parse(Int, match(r"c(\d+)_lam1", f)[1]) for f in files]
@printf("\n%d cohorts at γ = %g, N = %d, p = %d, %d julia threads\n",
        length(cohorts), LAMBDA, N_SUBJECTS, m_nn.p, Threads.nthreads()); flush(stdout)

println("\nwarm-up on a small cohort (discarded) so no timing below contains compilation ...")
let w = simulate_cohort(8; seed = SEED_COHORT, theta = tp.theta, model = m_par)
    wp = PopulationData([make_subject(m_nn, s.id, s.input, s.covariates.pk_times,
                                      s.observations[1:s.covariates.n_pk], s.covariates.pd_times,
                                      s.observations[s.covariates.n_pk+1:end]) for s in w.pop.subjects])
    observed_information(curvature_gradient(m_nn, wp, cfg, rcfg, CC_FOCE), theta_start(m_nn);
                         step = 1e-5, central = true)
end

rows = Any[csvrow("lambda", "cohort", "p", "information_seconds", "lambda_min", "lambda_max")]
secs = Float64[]
for (f, r) in zip(files, cohorts)
    θ = read_theta(joinpath(E11, f))
    gcl = curvature_gradient(m_nn, cohort_pop(r), cfg, rcfg, CC_FOCE)
    el = @elapsed J = Matrix(observed_information(gcl, θ; step = 1e-5, central = true))
    ev = eigvals(Symmetric(J))
    push!(secs, el)
    push!(rows, csvrow(LAMBDA, r, m_nn.p, el, minimum(ev), maximum(ev)))
    @printf("  cohort %2d: %6.1f s   λ_min %9.3f  λ_max %.4g\n", r, el, minimum(ev), maximum(ev))
    flush(stdout)
    writedlm(joinpath(RESULTS, "e14c_information_timing.csv"), vcat(rows...), ',')
end

@printf("\nγ = %g: n = %d, median %.1f s, range %.1f-%.1f\n",
        LAMBDA, length(secs), median(secs), minimum(secs), maximum(secs))
println("""

The information is formed from the unpenalised gradient at both penalty strengths, so a difference
against the γ = 0 timing would come from the conditional-mode solves at a different estimate, not
from the penalty entering the differentiation.""")
println("=" ^ 78)
