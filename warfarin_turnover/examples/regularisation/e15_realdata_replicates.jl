# E15 — the real warfarin data, five network fits at each penalty strength, fully instrumented.
#
#   julia --project=Local_Env -t 5 --heap-size-hint=20G .../e15_realdata_replicates.jl 0
#   julia --project=Local_Env -t 5 --heap-size-hint=20G .../e15_realdata_replicates.jl 1
#   julia --project=Local_Env -t 5 --heap-size-hint=20G .../e15_realdata_replicates.jl 3
#   julia --project=Local_Env -t 5 --heap-size-hint=20G .../e15_realdata_replicates.jl 10
#
# E13 fitted the real data once per penalty strength, which shows what the penalty does to the
# fitted curve but says nothing about how much of that curve is the data and how much is the
# starting value. Five fits per strength, differing only in the network initialisation seed,
# separate the two: spread across seeds at one γ is fit-to-fit variability, and the gap between
# γ levels is the penalty.
#
# One γ per process, deliberately. Compilation is paid once per session and shared by every fit
# in it, so measuring it per γ requires a cold process per γ. Each run therefore fits its first
# seed twice — once cold, once warm — and reports the difference, then fits the remaining seeds
# warm. Fit times below the first are steady-state.
#
# Everything expensive is archived: θ̂, the observed information, the gradient at θ̂, the fitted
# curve, and the delta-method standard error of E(C) at every grid point. The figure is then a
# pure read of CSVs and can be re-drawn, re-banded or re-ranged without refitting.
#
# The information is the UNPENALISED observed information at whatever θ̂ the penalised fit
# returned, as everywhere in this work: the penalty belongs to the optimisation criterion, not to
# the likelihood whose curvature sets a standard error. The reported OFV likewise excludes it.

const T_START = time()
t_include = @elapsed include(joinpath(@__DIR__, "..", "..", "src", "warfarin.jl"))
using Dates

BLAS.set_num_threads(1)
t_backends = @elapsed load_backends!(verbose = false)

const GAMMA = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1.0
const SMOKE = length(ARGS) >= 2 && ARGS[2] == "smoke"
const RESULTS = results_dir(@__DIR__, SMOKE ? "results_smoke" : "results")
const GAMMA_TAG = @sprintf("%g", GAMMA)
const LOG_PATH = joinpath(RESULTS, "e15_laplace_g$(GAMMA_TAG).log")
const LOG_IO = open(LOG_PATH, "w")
redirect_stdout(LOG_IO)
redirect_stderr(LOG_IO)

banner("E15 — exact-Laplace real warfarin data, five network fits at γ = $(GAMMA)")
println("log = ", LOG_PATH)
println("started = ", now()); flush(stdout)

const WIDTH = SCIENTIFIC_WIDTH
const MAXITERS = SMOKE ? 30 : 400
const MAX_FIT_SECONDS = SMOKE ? 180.0 : 1800.0
const MAX_INFORMATION_SECONDS = SMOKE ? 180.0 : 1800.0
const TAU = :noise
const N_SEEDS = SMOKE ? 1 : 5
const CMIN, CMAX = 0.60, 17.6          # observed concentration range of the 32-subject dataset
const CGRID = collect(range(CMIN, CMAX; length = 141))

const CFG_STANDARD = default_config(reg = 1e-8, inner_tol = 1e-8,
                                    inner_maxiter = 200, threaded = true)
const CFG_UNREGULARISED = default_config(reg = 1e-4, inner_tol = 1e-8,
                                         inner_maxiter = 200, threaded = true)
rcfg = reverse_config()
m_nn = WarfarinTurnover(MLPEffect(WIDTH))
m_par = WarfarinTurnover(EmaxEffect())
const IXE = collect(theta_index(m_nn).effect)
cfg_for(m::WarfarinTurnover, γ::Float64) =
    m === m_nn && γ == 0.0 ? CFG_UNREGULARISED : CFG_STANDARD

data = load_warfarin(m_nn)
data_par = load_warfarin(m_par)
const N_OBS = sum(length(s.observations) for s in data.pop.subjects)
@printf("\n%d subjects, %d observations, W = %d, p = %d, γ = %g, %d julia threads\n",
        data.pop.n_subjects, N_OBS, WIDTH, m_nn.p, GAMMA, Threads.nthreads())
@printf("Hessian regularization = %.1e; include %.1f s, backends %.1f s\n",
        cfg_for(m_nn, GAMMA).reg, t_include, t_backends); flush(stdout)

# ---------------------------------------------------------------------------
# Fit and diagnose
# ---------------------------------------------------------------------------

"""`fit_warfarin`'s objective plus `γ‖φ‖²` on the network block. The penalty gradient `2γφ` is
exact, so the penalised fit is as exactly differentiated as the unpenalised one."""
function fit_one(m::WarfarinTurnover, pop::PopulationData, γ::Float64, θ0::Vector{Float64})
    cfg_fit = cfg_for(m, γ)
    b = theta_bounds(m)
    ixe = m === m_nn ? IXE : Int[]
    starts = [collect(Float64, e) for e in population_objective(m, θ0, pop, cfg_fit).etas]
    obj_inner = foce_objective(m, pop, cfg_fit, CC_LAPLACE; fixed_starts = starts)
    grad_inner = curvature_gradient(m, pop, cfg_fit, rcfg, CC_LAPLACE;
                                    fixed_starts = starts)
    objective = θ -> obj_inner(θ) + γ * sum(abs2, view(θ, ixe))
    grad = function (G, θ)
        grad_inner(G, θ)
        isempty(ixe) || (@views G[ixe] .+= 2γ .* θ[ixe])
        return G
    end
    el = @elapsed r = fit_nlme(m, pop, θ0; lower = b.lower, upper = b.upper, cfg = cfg_fit,
                               objective = objective, grad = grad, maxiters = MAXITERS,
                               max_seconds = MAX_FIT_SECONDS, show_trace = true,
                               g_tol = 1e-3, f_tol = 0.001, optimizer = capability_optimizer(m.p))
    return (theta = r.theta, ofv = 2 * obj_inner(r.theta), iters = r.iterations,
            seconds = el, code = String(r.return_code), gradient_norm = r.gradient_norm,
            objective_rescues = r.objective_rescues, gradient_rescues = r.gradient_rescues,
            inner_nonconverged = r.inner_nonconverged, starts = starts, cfg = cfg_fit)
end

"""Exact conditional-Hessian diagnostics at the fitted parameters."""
function laplace_diagnostics(m::WarfarinTurnover, pop::PopulationData, fit)
    mineig = fill(NaN, pop.n_subjects)
    converged = fill(false, pop.n_subjects)
    Threads.@threads for i in 1:pop.n_subjects
        sh = subject_reverse(m, fit.theta, pop.subjects[i], fit.cfg, rcfg;
                             η_start = fit.starts[i])
        mineig[i] = minimum(eigvals(Symmetric(Float64.(sh.Htilde))))
        converged[i] = sh.converged
    end
    return (n_nonpd = count(<=(0.0), mineig), min_eig = minimum(mineig),
            n_nonconverged = count(!, converged))
end

"""E(C) on the grid with a delta-method standard error at every point, plus the spectral
bookkeeping. `Jf` is the 141 x p Jacobian of the curve in the FULL parameter vector, so the
standard error accounts for the truncation exactly as any other functional does."""
function band(m::WarfarinTurnover, pop::PopulationData, fit, γ::Float64, tag::String)
    θ = fit.theta
    ixe = theta_index(m).effect
    gcl = curvature_gradient(m, pop, fit.cfg, rcfg, CC_LAPLACE;
                             fixed_starts = fit.starts)
    t_inf = @elapsed J = Matrix(observed_information(
        gcl, θ; step = 1e-5, central = true,
        max_seconds = MAX_INFORMATION_SECONDS, show_progress = true))
    write_information_binary(joinpath(RESULTS, "e15_information_$(tag).bin"), J)
    information_finite = all(isfinite, J)
    G = zeros(m.p); gcl(G, θ)
    γ == 0 || m !== m_nn || (@views G[ixe] .+= 2γ .* θ[ixe])

    curve = t -> effect(m.effect, view(t, theta_index(m).effect), CGRID)
    # Five scalar functionals, on the same footing as the curve.
    proto = make_subject(m, 0, DOSE_LEVELS[2], pk_design_times(), zeros(N_PK),
                         pd_design_times(), zeros(N_PD))
    fun = t -> (φ = view(t, theta_index(m).effect);
                pd = structural(m, zeros(eltype(t), m.q), t, proto).pd;
                [effect(m.effect, φ, [2.0])[1], effect(m.effect, φ, [10.0])[1],
                 effect(m.effect, φ, [25.0])[1], pd[end-6], pd[end]])
    writedlm(joinpath(RESULTS, "e15_gradient_$(tag).csv"),
             vcat(csvrow("index", "gradient"), [csvrow(k, G[k]) for k in eachindex(G)]...), ',')
    writedlm(joinpath(RESULTS, "e15_theta_$(tag).csv"),
             vcat(csvrow("index", "theta"), [csvrow(k, θ[k]) for k in eachindex(θ)]...), ',')

    E = curve(θ)
    fv = fun(θ)
    if !information_finite
        println("  covariance failure: observed information contains non-finite entries")
        return (E = E, se = fill(NaN, length(E)), share = fill(NaN, length(E)),
                fun = fv, fun_se = fill(NaN, length(fv)), fun_share = fill(NaN, length(fv)),
                df_eff = NaN, tau = NaN, n_kept = 0, n_neg = -1, lmin = NaN,
                lmax = NaN, nd = NaN, t_inf = t_inf, information_finite = false)
    end

    dm = delta_method(J, ForwardDiff.jacobian(curve, θ); tau = TAU)
    dmf = delta_method(J, ForwardDiff.jacobian(fun, θ); tau = TAU)
    F = eigen(Symmetric(J))
    τ = resolve_tau(TAU, F.values)
    keep = findall(>(τ), F.values)
    df_eff = if isempty(keep)
        NaN
    elseif γ == 0 || m !== m_nn
        Float64(length(keep))
    else
        V = F.vectors[:, keep]
        JS = Diagonal(F.values[keep])
        pd = zeros(m.p); pd[ixe] .= 2γ
        tr((JS + transpose(V) * (Diagonal(pd) * V)) \ Matrix(JS))
    end
    return (E = E, se = dm.se, share = dm.identified_share,
            fun = fv, fun_se = dmf.se, fun_share = dmf.identified_share,
            df_eff = df_eff, tau = τ, n_kept = length(keep),
            n_neg = count(<=(0), F.values), lmin = minimum(F.values), lmax = maximum(F.values),
            nd = newton_decrement(J, G; tau = TAU).decrement, t_inf = t_inf,
            information_finite = true)
end

const FUNCS = ["E(C=2)", "E(C=10)", "E(C=25)", "PCA(24h)", "PCA(168h)"]
frows = Any[csvrow("model", "gamma", "seed", "p", "OFV", "phi_norm2", "iterations",
                   "fit_seconds", "information_seconds", "return_code", "df_eff", "tau",
                   "n_kept", "n_negative", "lambda_min", "lambda_max", "newton_decrement",
                   "hessian_regularization", "information_finite", "gradient_norm",
                   "objective_rescues", "gradient_rescues", "fit_inner_nonconverged",
                   "diagnostic_inner_nonconverged", "subjects_nonpd_hessian",
                   "min_subject_hessian_eigenvalue")]
crows = Any[csvrow("model", "gamma", "seed", "C", "E", "se", "lo95", "hi95", "identified_share")]
qrows = Any[csvrow("model", "gamma", "seed", "functional", "estimate", "se", "lo95", "hi95",
                   "identified_share")]
trows = Any[csvrow("quantity", "value", "note")]
prows = Any[csvrow("gamma", "seed", "model", "completed_at", "fit_seconds",
                   "information_seconds", "information_finite", "return_code")]

record! = (mname, γ, seed, fit, bd, hd) -> begin
    push!(frows, csvrow(mname, γ, seed, length(fit.theta), fit.ofv,
                        sum(abs2, view(fit.theta, theta_index(mname == "M_par" ? m_par : m_nn).effect)),
                        fit.iters, fit.seconds, bd.t_inf, fit.code, bd.df_eff, bd.tau,
                        bd.n_kept, bd.n_neg, bd.lmin, bd.lmax, bd.nd, fit.cfg.reg,
                        bd.information_finite, fit.gradient_norm, fit.objective_rescues,
                        fit.gradient_rescues, fit.inner_nonconverged, hd.n_nonconverged,
                        hd.n_nonpd, hd.min_eig))
    for (k, C) in enumerate(CGRID)
        push!(crows, csvrow(mname, γ, seed, C, bd.E[k], bd.se[k],
                            bd.E[k] - 1.96bd.se[k], bd.E[k] + 1.96bd.se[k], bd.share[k]))
    end
    for (k, nm) in enumerate(FUNCS)
        push!(qrows, csvrow(mname, γ, seed, nm, bd.fun[k], bd.fun_se[k],
                            bd.fun[k] - 1.96bd.fun_se[k], bd.fun[k] + 1.96bd.fun_se[k],
                            bd.fun_share[k]))
    end
end
flush_all! = () -> begin
    g = @sprintf("%g", GAMMA)
    writedlm(joinpath(RESULTS, "e15_fits_g$(g).csv"), vcat(frows...), ',')
    writedlm(joinpath(RESULTS, "e15_curves_g$(g).csv"), vcat(crows...), ',')
    writedlm(joinpath(RESULTS, "e15_functionals_g$(g).csv"), vcat(qrows...), ',')
    writedlm(joinpath(RESULTS, "e15_timing_g$(g).csv"), vcat(trows...), ',')
    writedlm(joinpath(RESULTS, "e15_progress_g$(g).csv"), vcat(prows...), ',')
end

function render_checkpoint!(completed_seeds::Int)
    SMOKE && return
    script = joinpath(REPO_ROOT, "warfarin_turnover", "figures", "make_figures.jl")
    project = dirname(Base.active_project())
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    figure_arg = GAMMA == 3.0 ? "e15b" : "e15"
    figure_stem = GAMMA == 3.0 ? "fig5b_e11_realdata_gamma3" : "fig5_e11_realdata"
    run(`$julia --project=$project -t 1 $script $figure_arg`)
    src = joinpath(REPO_ROOT, "warfarin_turnover", "figures",
                   "$(figure_stem).png")
    dst = joinpath(RESULTS, @sprintf("e15_figure_g%g_after_%d_seeds.png",
                                     GAMMA, completed_seeds))
    cp(src, dst; force = true)
    println("rendered = ", dst); flush(stdout)
end

# ---------------------------------------------------------------------------
# Compilation: the first seed fitted twice, cold then warm
# ---------------------------------------------------------------------------

θ0_of(k) = theta_start(m_nn; seed = SEED_START + 1000k)

println("\nfit at seed 1, cold (absorbs compilation) ..."); flush(stdout)
cold = fit_one(m_nn, data.pop, GAMMA, θ0_of(1))
@printf("  %.1f s, %d iterations, OFV %.1f\n", cold.seconds, cold.iters, cold.ofv); flush(stdout)

println("fit at seed 1, warm (steady state) ..."); flush(stdout)
warm = fit_one(m_nn, data.pop, GAMMA, θ0_of(1))
@printf("  %.1f s, %d iterations, OFV %.1f\n", warm.seconds, warm.iters, warm.ofv); flush(stdout)

t_compile = max(0.0, cold.seconds - warm.seconds)
push!(trows, csvrow("gamma", GAMMA, ""))
push!(trows, csvrow("include_package", t_include, "parse and load the package"))
push!(trows, csvrow("load_backends", t_backends, "AD backends and rules"))
push!(trows, csvrow("fit_cold", cold.seconds, "seed 1, first fit of the session"))
push!(trows, csvrow("fit_warm", warm.seconds, "seed 1 again, identical settings"))
push!(trows, csvrow("compilation", t_compile, "fit_cold - fit_warm"))

# ---------------------------------------------------------------------------
# The five seeds
# ---------------------------------------------------------------------------

t0 = time()
for k in 1:N_SEEDS
    fit = k == 1 ? warm : fit_one(m_nn, data.pop, GAMMA, θ0_of(k))
    tag = @sprintf("M_nn_g%g_s%d", GAMMA, k)
    hd = laplace_diagnostics(m_nn, data.pop, fit)
    bd = band(m_nn, data.pop, fit, GAMMA, tag)
    record!("M_nn($(WIDTH))", GAMMA, k, fit, bd, hd)
    push!(prows, csvrow(GAMMA, k, "M_nn($(WIDTH))", string(now()), fit.seconds,
                        bd.t_inf, bd.information_finite, fit.code))
    @printf("  seed %d: %5.1f s fit, %5.1f s information, OFV %9.1f, ‖φ‖² %8.2f, df_eff %6.2f, %d kept, finite=%s, subject non-PD %d (min eig %.3g), rescues %d/%d  (%.1f min)\n",
            k, fit.seconds, bd.t_inf, fit.ofv, sum(abs2, view(fit.theta, IXE)), bd.df_eff,
            bd.n_kept, string(bd.information_finite), hd.n_nonpd, hd.min_eig,
            fit.objective_rescues, fit.gradient_rescues, (time() - t0) / 60); flush(stdout)
    flush_all!()
    render_checkpoint!(k)
end

# One parametric fit, only in the γ = 0 run: M_par has no network block, so the penalty never
# touches it and refitting it once per γ would be three copies of one number.
if GAMMA == 0
    println("\nM_par (one fit; the penalty does not apply to it) ..."); flush(stdout)
    pf = fit_one(m_par, data_par.pop, 0.0, theta_start(m_par))
    phd = laplace_diagnostics(m_par, data_par.pop, pf)
    pb = band(m_par, data_par.pop, pf, 0.0, "M_par")
    record!("M_par", 0.0, 0, pf, pb, phd)
    push!(prows, csvrow(GAMMA, 0, "M_par", string(now()), pf.seconds,
                        pb.t_inf, pb.information_finite, pf.code))
    @printf("  %.1f s fit, %.1f s information, OFV %.1f, df_eff %.2f\n",
            pf.seconds, pb.t_inf, pf.ofv, pb.df_eff)
    flush_all!()
    render_checkpoint!(N_SEEDS)
end

push!(trows, csvrow("session_total", time() - T_START, "wall clock"))
push!(trows, csvrow("julia_threads", Threads.nthreads(), ""))
push!(trows, csvrow("blas_threads", BLAS.get_num_threads(), ""))
push!(trows, csvrow("n_subjects", data.pop.n_subjects, ""))
push!(trows, csvrow("n_observations", N_OBS, ""))
flush_all!()

fs = Float64[Float64(r[8]) for r in frows[2:end] if String(r[1]) != "M_par"]
is = Float64[Float64(r[9]) for r in frows[2:end] if String(r[1]) != "M_par"]
println("\n", "="^78)
@printf("γ = %g:  compilation %.1f s   estimation %.1f s (%.1f-%.1f)   information %.1f s (%.1f-%.1f)\n",
        GAMMA, t_compile, median(fs), minimum(fs), maximum(fs),
        median(is), minimum(is), maximum(is))
println("""
Everything needed to redraw the figure is on disk: the curve, its pointwise standard error and
interval, the functionals, the observed information and the gradient at each θ̂.""")
println("=" ^ 78)
println("finished = ", now())
