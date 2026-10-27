# E11 — does the ridge penalty that fixed the curve also improve inference?
#
#   julia --project=Local_Env -t 5 --heap-size-hint=20G warfarin_turnover/examples/regularisation/e11_penalised_inference.jl
#   julia --project=Local_Env -t 3 --heap-size-hint=8G  .../e11_penalised_inference.jl smoke
#
# E10 showed that `λ = 1` removes the oscillation from the fitted effect curve and moves it closer
# to the generating one. That is a statement about the point estimate. Whether it also improves
# *inference* is a separate question with no obvious answer, because a penalised fit is a MAP
# estimate rather than a maximum likelihood one and shrinkage trades variance for bias:
#
#   * variance falls, because the penalty removes the directions along which the unpenalised fit
#     was free to wander, so intervals should narrow;
#   * bias rises, because the estimator is pulled toward the shrinkage target — here a linear
#     effect model — so the interval is centred in the wrong place.
#
# Coverage is the resultant of those two and can move either way. This experiment measures it,
# repeating the coverage study at `λ = 0` and `λ = 1` under an identical protocol.
#
# ---------------------------------------------------------------------------------------
# Which curvature the diagnostics use, and why it is the UNPENALISED one
# ---------------------------------------------------------------------------------------
#
# The penalised objective has curvature `J_pen = J + 2λ·I` on the φ block, exactly, since the
# Hessian of `λ‖φ‖²` is `2λI` there and the penalty does not touch θ. It is tempting to run the
# spectral machinery on `J_pen`, and it is wrong to do so for the identifiability diagnostics.
# With `λ = 1` the penalty adds `2λ = 2` to 141 diagonal entries while the cut is `τ = 1`, so
# every φ direction is lifted across the threshold by arithmetic alone: measured on one cohort,
# `J` has 63 negative eigenvalues and retains 34 directions, while `J_pen` has ZERO negative
# eigenvalues and retains 143 of 153. None of that is information the data supplied. A prior does
# not make a parameter estimable; it makes the posterior proper, which is a different statement.
#
# So every quantity that describes what the DATA determine — the eigenvalue counts, the retained
# rank at `τ`, the identified share, and the standard error reported as primary — is computed from
# the unpenalised `J`, exactly as at `λ = 0`. That keeps `τ` meaning one thing throughout the study
# and keeps the two arms comparable.
#
# The estimator-specific variance is reported alongside, because θ̂_λ is a penalised estimator and
# its frequentist variance is the sandwich
#
#     Var(θ̂_λ) ≈ J_pen⁻¹ J J_pen⁻¹ ,
#
# which is smaller than `J⁻¹` and correctly does not credit the prior with having been data. What
# neither form corrects for is the shrinkage BIAS, which is why coverage rather than interval width
# is the quantity this experiment turns on.
#
# The comparison is run at a single fixed start for BOTH arms.

include(joinpath(@__DIR__, "..", "..", "src", "warfarin.jl"))
using Dates

BLAS.set_num_threads(1)
const SMOKE = length(ARGS) >= 1 && ARGS[1] == "smoke"
const RESULTS = results_dir(@__DIR__, SMOKE ? "results_smoke" : "results")
const LOG_PATH = joinpath(RESULTS, "e11_laplace.log")
const LOG_IO = open(LOG_PATH, "w")
redirect_stdout(LOG_IO)
redirect_stderr(LOG_IO)

loaded = load_backends!(verbose = true)
backend_available(:mooncake) || error("E11 requires the Mooncake engine; loaded = $(loaded).")

banner("E11 — exact-Laplace penalised inference on 20 synthetic cohorts")
println("log = ", LOG_PATH)
println("started = ", now()); flush(stdout)

const WIDTH = SCIENTIFIC_WIDTH
const N_SUBJECTS = SMOKE ? 20 : 100
const MAXITERS = SMOKE ? 30 : 400
const MAX_FIT_SECONDS = SMOKE ? 180.0 : 1800.0
const MAX_INFORMATION_SECONDS = SMOKE ? 180.0 : 1800.0
const LAMBDAS = (0.0, 1.0)
const TAU = :noise                   # |λ_min|; see spectral_split and e4d_cut_rules.jl
const CGRID = collect(range(0.0, 35.0; length = 351))

const REG_UNPENALISED = 1e-4
const REG_PENALISED = 1e-8
const CFG_UNPENALISED = default_config(reg = REG_UNPENALISED, inner_tol = 1e-8,
                                       inner_maxiter = 200, threaded = true)
const CFG_PENALISED = default_config(reg = REG_PENALISED, inner_tol = 1e-8,
                                     inner_maxiter = 200, threaded = true)
cfg_for(λ::Float64) = λ == 0.0 ? CFG_UNPENALISED :
                       λ == 1.0 ? CFG_PENALISED : error("no Hessian regularization specified for λ=$(λ)")
@printf("Hessian regularization: λ=0 -> %.1e; λ=1 -> %.1e\n",
        REG_UNPENALISED, REG_PENALISED); flush(stdout)
rcfg = reverse_config()
m_nn = WarfarinTurnover(MLPEffect(WIDTH))
m_par = WarfarinTurnover(EmaxEffect())
IXE = theta_index(m_nn).effect

tp = truth_theta()
const FUNCS = ["E(C=2)", "E(C=10)", "E(C=25)", "PCA(24h)", "PCA(168h)"]
proto_for(m) = make_subject(m, 0, DOSE_LEVELS[2], pk_design_times(), zeros(N_PK),
                            pd_design_times(), zeros(N_PD))
function functionals(m::WarfarinTurnover, θ::AbstractVector)
    φ = view(θ, theta_index(m).effect)
    pd = structural(m, zeros(eltype(θ), m.q), θ, proto_for(m)).pd
    return [effect(m.effect, φ, [2.0])[1], effect(m.effect, φ, [10.0])[1],
            effect(m.effect, φ, [25.0])[1], pd[end-6], pd[end]]
end
const TRUE_VALS = functionals(m_par, tp.theta)

"""`fit_warfarin` with `L + λ‖φ‖²`; the returned OFV excludes the penalty."""
function fit_penalised(pop::PopulationData, λ::Float64; maxiters::Int = MAXITERS)
    cfg_fit = cfg_for(λ)
    b = theta_bounds(m_nn)
    θ0 = theta_start(m_nn)
    starts = [collect(Float64, e) for e in population_objective(m_nn, θ0, pop, cfg_fit).etas]
    obj_inner = foce_objective(m_nn, pop, cfg_fit, CC_LAPLACE; fixed_starts = starts)
    grad_inner = curvature_gradient(m_nn, pop, cfg_fit, rcfg, CC_LAPLACE;
                                    fixed_starts = starts)
    objective = θ -> obj_inner(θ) + λ * sum(abs2, view(θ, IXE))
    grad = function (G, θ)
        grad_inner(G, θ)
        @views G[IXE] .+= 2λ .* θ[IXE]
        return G
    end
    el = @elapsed r = fit_nlme(m_nn, pop, θ0; lower = b.lower, upper = b.upper, cfg = cfg_fit,
                               objective = objective, grad = grad, maxiters = maxiters,
                               max_seconds = MAX_FIT_SECONDS,
                               g_tol = 1e-3, f_tol = 0.001, show_trace = true,
                               optimizer = capability_optimizer(m_nn.p))
    return (theta = r.theta, ofv = 2 * obj_inner(r.theta), iters = r.iterations,
            seconds = el, code = String(r.return_code), gradient_norm = r.gradient_norm,
            objective_rescues = r.objective_rescues, gradient_rescues = r.gradient_rescues,
            inner_nonconverged = r.inner_nonconverged, starts = starts, cfg = cfg_fit)
end

"""Observed information of the PENALISED objective. The penalty contributes exactly `2λ` to the
diagonal of the `φ` block and nothing elsewhere, so no extra differentiation is needed."""
function penalised_information(pop::PopulationData, θ::Vector{Float64}, λ::Float64,
                               cfg_fit::LaplaceConfig;
                               tag::String = "", fixed_starts = nothing)
    gcl = curvature_gradient(m_nn, pop, cfg_fit, rcfg, CC_LAPLACE;
                             fixed_starts = fixed_starts)
    seconds = @elapsed J = Matrix(observed_information(
        gcl, θ; step = 1e-5, central = true,
        max_seconds = MAX_INFORMATION_SECONDS, show_progress = true))
    if !isempty(tag)
        # Only the UNPENALISED information is archived: the penalised one is J + 2*lambda on the
        # phi block, reconstructible exactly from it and lambda, so storing it would be redundant.
        G = zeros(m_nn.p); gcl(G, θ)
        write_information_binary(joinpath(RESULTS, "e11_information_$(tag).bin"), J)
        writedlm(joinpath(RESULTS, "e11_gradient_$(tag).csv"),
                 vcat(csvrow("index", "gradient"), [csvrow(k, G[k]) for k in eachindex(G)]...), ',')
        writedlm(joinpath(RESULTS, "e11_theta_$(tag).csv"),
                 vcat(csvrow("index", "theta"), [csvrow(k, θ[k]) for k in eachindex(θ)]...), ',')
    end
    Jp = copy(J)
    for j in IXE
        Jp[j, j] += 2λ
    end
    return J, Jp, seconds
end

"""Definiteness of each subject's exact random-effect Hessian at the fitted parameters."""
function laplace_diagnostics(pop::PopulationData, θ::Vector{Float64},
                             cfg_fit::LaplaceConfig; fixed_starts = nothing)
    n = pop.n_subjects
    mineig = fill(NaN, n)
    converged = fill(false, n)
    Threads.@threads for i in 1:n
        st = fixed_starts === nothing ? nothing : fixed_starts[i]
        sh = subject_reverse(m_nn, θ, pop.subjects[i], cfg_fit, rcfg; η_start = st)
        mineig[i] = minimum(eigvals(Symmetric(Float64.(sh.Htilde))))
        converged[i] = sh.converged
    end
    return (n_nonpd = count(<=(0.0), mineig), min_eig = minimum(mineig),
            n_nonconverged = count(!, converged))
end

"""Two standard errors, answering two different questions.

`dm` uses the **unpenalised** information, so `τ`, the retained rank and the identified share mean
what they mean everywhere else in this study, and the `λ = 0` and `λ = 1` arms are comparable.
`se_sand` is the sandwich `J_pen⁻¹ J J_pen⁻¹`, the frequentist variance of the penalised estimator
itself. Neither corrects for shrinkage bias."""
function ses(J::Matrix{Float64}, Jp::Matrix{Float64}, Jf::Matrix{Float64})
    dm = delta_method(J, Jf; tau = TAU)              # LIKELIHOOD information: the primary report
    S = truncated_inverse(Jp; tau = TAU).Sigma
    sand = S * J * S                                  # estimator variance of the penalised fit
    se_sand = [sqrt(max(0.0, (view(Jf, k, :)' * sand * view(Jf, k, :)))) for k in axes(Jf, 1)]
    return dm, se_sand
end

rows = Any[csvrow("lambda", "cohort", "functional", "truth", "estimate", "se_likelihood",
                  "se_sandwich", "lo95", "hi95", "covered", "covered_sandwich",
                  "identified_share", "n_kept")]
frows = Any[csvrow("lambda", "hessian_regularization", "cohort", "OFV", "iterations", "seconds",
                   "information_seconds", "information_finite", "return_code", "gradient_norm",
                   "objective_rescues", "gradient_rescues", "inner_nonconverged",
                   "subjects_nonpd_hessian", "min_subject_hessian_eigenvalue",
                   "phi_norm2", "roughness", "curve_MAE",
                   "n_negative_unpenalised", "n_negative_penalised",
                   "n_kept_unpenalised", "n_kept_penalised")]
crows = Any[csvrow("source", "lambda", "cohort", "C", "E")]
progress = Any[csvrow("cohort", "lambda", "hessian_regularization", "completed_at", "fit_seconds",
                      "information_seconds", "information_finite", "return_code")]

const HSTEP = CGRID[2] - CGRID[1]
roughness(E) = (d2 = (E[3:end] .- 2 .* E[2:end-1] .+ E[1:end-2]) ./ HSTEP^2; sum(abs2, d2) * HSTEP)
const ETRUE = [effect(m_par.effect, view(tp.theta, theta_index(m_par).effect), [C])[1] for C in CGRID]
for C in CGRID
    push!(crows, csvrow("simulation_truth", -1.0, 0, C,
                        effect(m_par.effect, view(tp.theta, theta_index(m_par).effect), [C])[1]))
end

# Twenty fixed replicates, with no result-based selection.
COHORTS = SMOKE ? collect(1:2) : collect(1:20)
@printf("\n%d cohorts, λ ∈ %s, N = %d\n", length(COHORTS), string(LAMBDAS), N_SUBJECTS)

function coverage_table()
    summ = Any[csvrow("lambda", "functional", "n", "k_covered", "coverage",
                      "k_covered_sandwich", "coverage_sandwich", "truth",
                      "mean_estimate", "bias", "rmse", "mean_se", "mean_se_sandwich",
                      "sd_across_cohorts", "se_sd_ratio", "mean_width")]
    for λ in LAMBDAS, nm in FUNCS
        sel = [x for x in rows[2:end] if x[1] == λ && x[3] == nm]
        isempty(sel) && continue
        est = Float64[x[5] for x in sel]
        se = Float64[x[6] for x in sel]
        se_sand = Float64[x[7] for x in sel]
        k = count(x -> x[10] === true, sel)
        ks = count(x -> x[11] === true, sel)
        n = length(sel)
        tv = Float64(sel[1][4])
        sdv = n > 1 ? std(est) : NaN
        push!(summ, csvrow(λ, nm, n, k, k / n, ks, ks / n, tv, mean(est), mean(est) - tv,
                           sqrt(mean((est .- tv) .^ 2)), mean(se), mean(se_sand), sdv,
                           mean(se) / sdv, mean(2 * 1.96 .* se)))
    end
    return summ
end

function checkpoint!(completed_replicates::Int; render::Bool = false)
    writedlm(joinpath(RESULTS, "e11_functionals.csv"), vcat(rows...), ',')
    writedlm(joinpath(RESULTS, "e11_fits.csv"), vcat(frows...), ',')
    writedlm(joinpath(RESULTS, "e11_curves.csv"), vcat(crows...), ',')
    writedlm(joinpath(RESULTS, "e11_coverage.csv"), vcat(coverage_table()...), ',')
    writedlm(joinpath(RESULTS, "e11_progress.csv"), vcat(progress...), ',')
    @printf("checkpoint: %d complete replicate(s), %d complete fit arm(s) at %s\n",
            completed_replicates, length(progress) - 1, string(now())); flush(stdout)
    if render && !SMOKE
        script = joinpath(REPO_ROOT, "warfarin_turnover", "figures", "make_figures.jl")
        project = dirname(Base.active_project())
        julia = joinpath(Sys.BINDIR, Base.julia_exename())
        run(`$julia --project=$project -t 1 $script e11`)
        src = joinpath(REPO_ROOT, "warfarin_turnover", "figures",
                       "fig3_e11_inference.png")
        dst = joinpath(RESULTS, @sprintf("e11_figure_after_%d_replicates.png",
                                        completed_replicates))
        cp(src, dst; force = true)
        println("rendered = ", dst); flush(stdout)
    end
end

println("\nwarm-up (discarded) ...")
let w = simulate_cohort(8; seed = SEED_COHORT, theta = tp.theta, model = m_par)
    sub = [make_subject(m_nn, s.id, s.input, s.covariates.pk_times,
                        s.observations[1:s.covariates.n_pk], s.covariates.pd_times,
                        s.observations[s.covariates.n_pk+1:end]) for s in w.pop.subjects]
    fit_penalised(PopulationData(sub), 1.0; maxiters = 3)
end
println("  warm."); flush(stdout)

t0 = time()
for r in COHORTS
    sim = simulate_cohort(N_SUBJECTS; seed = SEED_COHORT + r, theta = tp.theta, model = m_par)
    subs = [make_subject(m_nn, s.id, s.input, s.covariates.pk_times,
                         s.observations[1:s.covariates.n_pk], s.covariates.pd_times,
                         s.observations[s.covariates.n_pk+1:end]) for s in sim.pop.subjects]
    pop = PopulationData(subs)
    for λ in LAMBDAS
        println("\narm started: cohort=", r, " lambda=", λ, " at ", now()); flush(stdout)
        f = fit_penalised(pop, λ)
        hd = laplace_diagnostics(pop, f.theta, f.cfg; fixed_starts = f.starts)
        J, Jp, info_seconds = penalised_information(
            pop, f.theta, λ, f.cfg; tag = @sprintf("c%d_lam%g", r, λ),
            fixed_starts = f.starts)
        Jf = ForwardDiff.jacobian(t -> functionals(m_nn, t), f.theta)
        est = functionals(m_nn, f.theta)
        φ = view(f.theta, IXE)
        E = [effect(m_nn.effect, φ, [C])[1] for C in CGRID]
        for (k, C) in enumerate(CGRID)
            push!(crows, csvrow("simulated", λ, r, C, E[k]))
        end
        information_finite = all(isfinite, J)
        negJ = negP = keptJ = keptP = -1
        if information_finite
            dm, se_sand = ses(J, Jp, Jf)
            for (j, nm) in enumerate(FUNCS)
                lo, hi = est[j] - 1.96dm.se[j], est[j] + 1.96dm.se[j]
                los, his = est[j] - 1.96se_sand[j], est[j] + 1.96se_sand[j]
                push!(rows, csvrow(λ, r, nm, TRUE_VALS[j], est[j], dm.se[j], se_sand[j], lo, hi,
                                   (lo <= TRUE_VALS[j] <= hi), (los <= TRUE_VALS[j] <= his),
                                   dm.identified_share[j], dm.n_kept))
            end
            ev, evp = eigvals(Symmetric(J)), eigvals(Symmetric(Jp))
            τJ, τP = resolve_tau(TAU, ev), resolve_tau(TAU, evp)
            negJ, negP = count(<=(0), ev), count(<=(0), evp)
            keptJ, keptP = count(>(τJ), ev), count(>(τP), evp)
        else
            println("  covariance failure: observed information contains non-finite entries; " *
                    "curve retained, intervals omitted"); flush(stdout)
        end
        push!(frows, csvrow(λ, f.cfg.reg, r, f.ofv, f.iters, f.seconds, info_seconds,
                            information_finite, f.code,
                            f.gradient_norm, f.objective_rescues, f.gradient_rescues,
                            hd.n_nonconverged, hd.n_nonpd, hd.min_eig, sum(abs2, φ),
                            roughness(E), sum(abs.(E .- ETRUE)) / length(E),
                            negJ, negP, keptJ, keptP))
        push!(progress, csvrow(r, λ, f.cfg.reg, string(now()), f.seconds, info_seconds,
                               information_finite, f.code))
        @printf("  λ=%-4g reg=%.1e cohort %2d  OFV %9.1f  fit %6.1fs info %6.1fs finite=%-5s  subject non-PD %d (min eig %.3g)  rescues %d/%d  info neg %3d→%3d  kept %3d→%3d  %5.1f min\n",
                λ, f.cfg.reg, r, f.ofv, f.seconds, info_seconds, string(information_finite),
                hd.n_nonpd, hd.min_eig, f.objective_rescues, f.gradient_rescues,
                negJ, negP, keptJ, keptP,
                (time() - t0) / 60); flush(stdout)
        checkpoint!(findfirst(==(r), COHORTS) - 1 + (λ == last(LAMBDAS) ? 1 : 0);
                    render = λ == last(LAMBDAS))
    end
end
@printf("\nsimulation arm complete in %.1f min\n", (time() - t0) / 60)
println("finished = ", now())
println("=" ^ 78); flush(stdout)
