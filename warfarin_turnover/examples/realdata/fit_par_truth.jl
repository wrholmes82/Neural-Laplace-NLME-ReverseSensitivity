# Fix the simulation truth by fitting `M_par` to the real warfarin data.
#
#   julia --project=Local_Env -t 5 warfarin_turnover/examples/realdata/fit_par_truth.jl
#
# §2.3: "Truth is obtained by fitting `M_par` to the real warfarin data first and using those
# estimates as the data-generating parameters. Self-consistent, defensible, and it removes any
# argument about where the truth came from."
#
# This is a **prerequisite**, not a result. It must run before E2, E3, E4 or E5, because
# `simulate_cohort` refuses to invent a truth and errors if `data/truth_par.csv` is absent.
# The E6 real-data anchor is a separate script and reports the `M_par` and `M_nn(10)` fits as
# results with their uncertainty; this one exists only to write the truth file.
#
# It also performs the PCA-versus-INR endpoint resolution (§2.4) by computation rather than by
# assertion, and records it.

include(joinpath(@__DIR__, "..", "..", "src", "warfarin.jl"))

BLAS.set_num_threads(1)
loaded = load_backends!(verbose = true)      # top level — see the note in the E1 gate
backend_available(:mooncake) ||
    error("the truth fit requires the Mooncake engine; loaded = $(loaded).")

banner("Simulation truth — M_par fitted to the real 32-subject warfarin data")

const RESULTS = results_dir(@__DIR__, "results")
const TRUTH_OUT = joinpath(@__DIR__, "..", "..", "data", "truth_par.csv")

# ---------------------------------------------------------------------------
# Data, and the endpoint resolution
# ---------------------------------------------------------------------------

model = WarfarinTurnover(EmaxEffect())
data = load_warfarin(model)
pop = data.pop
nobs = n_observations(pop)

ep = warfarin_endpoint_check(pop)
@printf("\nEndpoint check (§2.4):\n")
@printf("  median response  baseline(t<=6h) = %.1f   nadir(24-72h) = %.1f   late(t>=120h) = %.1f\n",
        ep.baseline, ep.nadir, ep.late)
@printf("  resolution: %s\n", ep.endpoint)
ep.suppressed || error(
    "the response RISES under drug in this data file. The turnover model as written has E >= 0 " *
    "suppressing R, which is correct for PCA and WRONG for INR. Resolve before fitting — see " *
    "docs/DECISIONS.md D1. Refusing to fit a model whose effect sign contradicts the data.")

@printf("\nData: %d subjects, %d observations (%d conc + %d PCA), %d BLQ concentration records dropped\n",
        data.n_subjects, nobs.total, nobs.n_pk, nobs.n_pd, data.n_blq)
@printf("  observations per subject: %.1f against q = %d random effects\n",
        nobs.total / data.n_subjects, model.q)

# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

cfg = default_config(reg = 1e-8, inner_tol = 1e-10, inner_maxiter = 200, threaded = true)
rcfg = reverse_config()
θ0 = theta_start(model)

println("\nfitting M_par (p = $(model.p)) by FOCE with the reverse gradient ...")
flush(stdout)
fit = fit_warfarin(model, pop; θ0 = θ0, cfg = cfg, rcfg = rcfg, ccfg = CC_FOCE,
                   maxiters = 500, g_tol = 1e-3, show_trace = true)
r = fit.result
println(r)
flush(stdout)

# ---------------------------------------------------------------------------
# Observed information and standard errors
#
# Cheap here — p = 14, so 2p = 28 gradients — and worth having: a non-positive-definite
# information matrix at θ̂ would say the truth we are about to simulate from is not identified,
# which is something to discover now rather than after 20 cohorts have been generated from it.
# ---------------------------------------------------------------------------

println("\nobserved information (2p = $(2 * model.p) gradients) ...")
flush(stdout)
t_info = @elapsed H = observed_information(fit.grad_closure, r.theta; step = 1e-5, central = true)
pd = issuccess(cholesky(H; check = false))
@printf("  %.2f s, positive definite = %s\n", t_info, pd)

write_fit(RESULTS, model, pop, fit; tag = "par_real", information = H,
          extra = (:endpoint => ep.endpoint, :n_blq_dropped => data.n_blq,
                   :information_seconds => t_info, :role => "simulation truth (§2.3)"))

pp = population_parameters(model, r.theta)
@printf("\nFitted M_par:\n")
@printf("  ka   = %.4f /h      CL   = %.4f L/h    V = %.3f L   (t_half = %.1f h)\n",
        pp.ka, pp.CL, pp.V, log(2) * pp.V / pp.CL)
@printf("  R0   = %.2f          kout = %.4f /h                 (t_half = %.1f h)\n",
        pp.R0, pp.kout, log(2) / pp.kout)
@printf("  Emax = %.4f          EC50 = %.4f mg/L\n", exp(r.theta[6]), exp(r.theta[7]))
@printf("  omega = %s\n", string(round.(pp.omega; digits = 4)))
@printf("  sigma_prop = %.4f    sigma_add = %.3f\n", pp.sigma_prop, pp.sigma_add)

# ---------------------------------------------------------------------------
# Write the truth file
#
# Guarded: a non-converged fit must not become the data-generating truth of the whole study.
# ---------------------------------------------------------------------------

if !r.converged
    println("\n" * "="^78)
    println("REFUSING to write the truth file: the M_par fit did not converge.")
    println("  return_code = ", r.return_code, "   ||grad||_inf = ", r.gradient_norm)
    println("  Nothing downstream may simulate from a non-converged truth. Fix the fit first.")
    println("="^78)
    exit(1)
end

rows = Any[["name" "value"]]
names = parameter_names(model)
for j in 1:model.p
    push!(rows, ["theta_$(j)" r.theta[j]])
end
for (k, v) in [("OFV", r.OFV), ("grad_inf_norm", r.gradient_norm), ("iterations", r.iterations),
               ("n_subjects", pop.n_subjects), ("n_obs", nobs.total), ("p", model.p),
               ("information_pd", pd ? 1 : 0), ("rk4_n_seg", model.n_seg),
               ("rk4_gamma", model.gamma), ("n_blq_dropped", data.n_blq)]
    push!(rows, [k v])
end
writedlm(TRUTH_OUT, vcat(rows...), ',')

# A human-readable companion, so the truth is legible without decoding θ indices.
writedlm(joinpath(RESULTS, "truth_par_named.csv"),
         vcat(["index" "name" "theta" "natural_scale"],
              [Any[j names[j] r.theta[j] (occursin("log_", names[j]) ? exp(r.theta[j]) : r.theta[j])]
               for j in 1:model.p]...), ',')

println("\n" * "="^78)
println("Wrote the simulation truth to ", normpath(TRUTH_OUT))
println("  OFV = ", r.OFV, "   ||grad||_inf = ", r.gradient_norm, "   iterations = ", r.iterations)
println("  E2, E3, E4 and E5 may now simulate. Full artefacts in ", RESULTS)
println("="^78)
