# E15b — the parametric model's own timing on the real data, in a process of its own.
#
#   julia --project=Local_Env -t 5 --heap-size-hint=20G .../e15b_mpar_timing.jl
#
# E15 fits M_par inside its γ = 0 run, after six network fits. The shared infrastructure is warm
# by then but `EmaxEffect`'s own specialisations are not, so that fit's 16.8 s contains an unknown
# amount of compilation and its compile time was never separated at all. A row of the timing table
# cannot rest on that.
#
# Same design as E14 and E15: a cold fit and an identical warm one in a fresh process, then a cold
# and a warm observed information. Compilation is the cold-minus-warm difference.

const T_START = time()
t_include = @elapsed include(joinpath(@__DIR__, "..", "..", "src", "warfarin.jl"))
using Dates

BLAS.set_num_threads(1)
t_backends = @elapsed load_backends!(verbose = false)
const RESULTS = results_dir(@__DIR__, "results")
const LOG_PATH = joinpath(RESULTS, "e15b_mpar_laplace.log")
const LOG_IO = open(LOG_PATH, "w")
redirect_stdout(LOG_IO)
redirect_stderr(LOG_IO)
banner("E15b — exact-Laplace M_par timing on the real warfarin data")
println("log = ", LOG_PATH)
println("started = ", now()); flush(stdout)

const MAXITERS = 400
const MAX_FIT_SECONDS = 1800.0
const MAX_INFORMATION_SECONDS = 1800.0
const TAU = :noise

cfg = default_config(reg = 1e-8, inner_tol = 1e-8, inner_maxiter = 200, threaded = true)
rcfg = reverse_config()
m_par = WarfarinTurnover(EmaxEffect())
data = load_warfarin(m_par)
@printf("\n%d subjects, p = %d, %d julia threads\n",
        data.pop.n_subjects, m_par.p, Threads.nthreads()); flush(stdout)

function fit_once()
    b = theta_bounds(m_par)
    θ0 = theta_start(m_par)
    starts = [collect(Float64, e) for e in population_objective(m_par, θ0, data.pop, cfg).etas]
    obj = foce_objective(m_par, data.pop, cfg, CC_LAPLACE; fixed_starts = starts)
    grd = curvature_gradient(m_par, data.pop, cfg, rcfg, CC_LAPLACE;
                             fixed_starts = starts)
    r = fit_nlme(m_par, data.pop, θ0; lower = b.lower, upper = b.upper, cfg = cfg,
                 objective = obj, grad = grd, maxiters = MAXITERS,
                 max_seconds = MAX_FIT_SECONDS, show_trace = true,
                 g_tol = 1e-3, f_tol = 0.001, optimizer = capability_optimizer(m_par.p))
    return (theta = r.theta, ofv = 2 * obj(r.theta), iters = r.iterations,
            starts = starts, code = String(r.return_code),
            objective_rescues = r.objective_rescues,
            gradient_rescues = r.gradient_rescues)
end

println("\nfit, cold (absorbs compilation) ..."); flush(stdout)
t_fit1 = @elapsed f1 = fit_once()
@printf("  %.1f s, %d iterations, OFV %.1f\n", t_fit1, f1.iters, f1.ofv); flush(stdout)

println("fit, warm (steady state) ..."); flush(stdout)
t_fit2 = @elapsed f2 = fit_once()
@printf("  %.1f s, %d iterations, OFV %.1f\n", t_fit2, f2.iters, f2.ofv); flush(stdout)

gcl = curvature_gradient(m_par, data.pop, cfg, rcfg, CC_LAPLACE;
                         fixed_starts = f2.starts)
println("information, cold ..."); flush(stdout)
t_inf1 = @elapsed J = Matrix(observed_information(
    gcl, f2.theta; step = 1e-5, central = true,
    max_seconds = MAX_INFORMATION_SECONDS, show_progress = true))
@printf("  %.1f s\n", t_inf1); flush(stdout)
println("information, warm ..."); flush(stdout)
t_inf2 = @elapsed observed_information(
    gcl, f2.theta; step = 1e-5, central = true,
    max_seconds = MAX_INFORMATION_SECONDS, show_progress = true)
@printf("  %.1f s\n", t_inf2); flush(stdout)

ev = eigvals(Symmetric(J))
τ = resolve_tau(TAU, ev)
t_compile = max(0.0, t_fit1 - t_fit2) + max(0.0, t_inf1 - t_inf2)

rows = Any[csvrow("quantity", "value", "note")]
push!(rows, csvrow("include_package", t_include, ""))
push!(rows, csvrow("load_backends", t_backends, ""))
push!(rows, csvrow("fit_cold", t_fit1, "first fit of the session"))
push!(rows, csvrow("fit_warm", t_fit2, "steady-state estimation"))
push!(rows, csvrow("information_cold", t_inf1, ""))
push!(rows, csvrow("information_warm", t_inf2, "steady state, 2p gradients"))
push!(rows, csvrow("compilation", t_compile, "(fit_cold - fit_warm) + (info_cold - info_warm)"))
push!(rows, csvrow("OFV", f2.ofv, "no penalty applies to M_par"))
push!(rows, csvrow("df_eff", count(>(τ), ev), "= rank(J_tau); no penalised block"))
push!(rows, csvrow("n_kept", count(>(τ), ev), ""))
push!(rows, csvrow("tau", τ, ""))
push!(rows, csvrow("p", m_par.p, ""))
push!(rows, csvrow("hessian_regularization", cfg.reg, ""))
push!(rows, csvrow("information_finite", all(isfinite, J), ""))
push!(rows, csvrow("objective_rescues", f2.objective_rescues, ""))
push!(rows, csvrow("gradient_rescues", f2.gradient_rescues, ""))
push!(rows, csvrow("session_total", time() - T_START, ""))
writedlm(joinpath(RESULTS, "e15b_mpar_timing.csv"), vcat(rows...), ',')

println("\n", "="^70)
@printf("%-22s %8.1f s\n", "compilation", t_compile)
@printf("%-22s %8.1f s\n", "estimation (warm)", t_fit2)
@printf("%-22s %8.1f s\n", "information (warm)", t_inf2)
@printf("%-22s %8.1f\n", "OFV", f2.ofv)
@printf("%-22s %8d of %d\n", "df_eff", count(>(τ), ev), m_par.p)
println("=" ^ 70)
println("finished = ", now())
