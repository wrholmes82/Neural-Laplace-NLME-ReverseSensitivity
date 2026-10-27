# E9b — exact-Laplace recovery under increasing parametric misspecification.
#
#   julia --project=Local_Env -t 5 --heap-size-hint=20G \
#       warfarin_turnover/examples/misspecification/e9b_penalised_curves.jl
#
# Five paired M_par and ridge-regularised M_nn(10) fits are run for each of three generating
# shapes. Every fit and covariance step is checkpointed immediately; Figure 4 is regenerated
# after each cohort has completed all three shapes. The unpenalised OFV is retained so the two
# fitted models remain directly comparable.

include(joinpath(@__DIR__, "..", "..", "src", "warfarin.jl"))
using Dates

BLAS.set_num_threads(1)
const SMOKE = length(ARGS) >= 1 && ARGS[1] == "smoke"
const RESULTS = results_dir(@__DIR__, SMOKE ? "results_smoke" : "results")
const LOG_PATH = joinpath(RESULTS, "e9b_laplace.log")
const LOG_IO = open(LOG_PATH, "w")
redirect_stdout(LOG_IO)
redirect_stderr(LOG_IO)

loaded = load_backends!(verbose = true)
backend_available(:mooncake) || error("E9b requires the Mooncake engine; loaded = $(loaded).")
banner("E9b — exact-Laplace misspecification analysis")
println("log = ", LOG_PATH)
println("started = ", now()); flush(stdout)

const WIDTH = SCIENTIFIC_WIDTH
const N_SUBJECTS = SMOKE ? 20 : 100
const MAXITERS = SMOKE ? 30 : 400
const MAX_FIT_SECONDS = SMOKE ? 180.0 : 1800.0
const MAX_INFORMATION_SECONDS = SMOKE ? 180.0 : 1800.0
const N_COHORTS = SMOKE ? 2 : 5
const LAMBDA = 1.0
const HESSIAN_REGULARIZATION = 1e-8
const HILL_N = (1.5, 3.0)
const C_MATCH = 10.0
const CGRID = collect(range(0.0, 35.0; length = 141))

const CFG = default_config(reg = HESSIAN_REGULARIZATION, inner_tol = 1e-8,
                           inner_maxiter = 200, threaded = true)
rcfg = reverse_config()

m_par = WarfarinTurnover(EmaxEffect())
m_nn = WarfarinTurnover(MLPEffect(WIDTH))
m_hill = WarfarinTurnover(HillEffect())
const IXE = collect(theta_index(m_nn).effect)

tp = truth_theta()
ixp = theta_index(m_par)
EMAX = exp(tp.theta[ixp.effect][1])
EC50 = exp(tp.theta[ixp.effect][2])

"""Sigmoid truth matched to the Emax truth at `C_MATCH`."""
function hill_truth(n)
    Et = EMAX * C_MATCH / (EC50 + C_MATCH)
    ec = C_MATCH * ((EMAX - Et) / Et)^(1 / n)
    return vcat(tp.theta[1:5], [log(EMAX), log(ec), log(n)], tp.theta[8:14])
end

TRUTHS = vcat(
    [(name = "Emax", model = m_par, theta = tp.theta, seed0 = SEED_COHORT)],
    [(name = @sprintf("sigmoid_g%.1f", n), model = m_hill, theta = hill_truth(n),
      seed0 = SEED_MISSPEC + Int(round(1000n))) for n in HILL_N])
COHORTS = collect(1:N_COHORTS)

tagof(fitname) = replace(fitname, "(" => "", ")" => "")

"""Fit `L + lambda*||phi||^2` by exact Laplace; the returned OFV excludes the penalty."""
function fit_penalised(m::WarfarinTurnover, pop::PopulationData, λ::Float64;
                       maxiters::Int = MAXITERS)
    cfg_fit = CFG
    b = theta_bounds(m)
    θ0 = theta_start(m)
    starts = [collect(Float64, e) for e in population_objective(m, θ0, pop, cfg_fit).etas]
    obj_inner = foce_objective(m, pop, cfg_fit, CC_LAPLACE; fixed_starts = starts)
    grad_inner = curvature_gradient(m, pop, cfg_fit, rcfg, CC_LAPLACE;
                                    fixed_starts = starts)
    pen_ix = m === m_nn ? IXE : Int[]
    objective = θ -> obj_inner(θ) + λ * sum(abs2, view(θ, pen_ix))
    grad = function (G, θ)
        grad_inner(G, θ)
        isempty(pen_ix) || (@views G[pen_ix] .+= 2λ .* θ[pen_ix])
        return G
    end
    seconds = @elapsed r = fit_nlme(
        m, pop, θ0; lower = b.lower, upper = b.upper, cfg = cfg_fit,
        objective = objective, grad = grad, maxiters = maxiters,
        max_seconds = MAX_FIT_SECONDS, g_tol = 1e-3, f_tol = 0.001,
        show_trace = true, optimizer = capability_optimizer(m.p))
    return (theta = r.theta, ofv = 2 * obj_inner(r.theta), iters = r.iterations,
            seconds = seconds, code = String(r.return_code),
            gradient_norm = r.gradient_norm, objective_rescues = r.objective_rescues,
            gradient_rescues = r.gradient_rescues,
            inner_nonconverged = r.inner_nonconverged, starts = starts,
            cfg = cfg_fit,
            penalty = λ * sum(abs2, view(r.theta, pen_ix)))
end

"""Exact subject-Hessian diagnostics at the fitted parameters."""
function laplace_diagnostics(m::WarfarinTurnover, pop::PopulationData,
                             θ::Vector{Float64}, starts, cfg_fit::LaplaceConfig)
    mineig = fill(NaN, pop.n_subjects)
    converged = fill(false, pop.n_subjects)
    Threads.@threads for i in 1:pop.n_subjects
        sh = subject_reverse(m, θ, pop.subjects[i], cfg_fit, rcfg; η_start = starts[i])
        mineig[i] = minimum(eigvals(Symmetric(Float64.(sh.Htilde))))
        converged[i] = sh.converged
    end
    return (n_nonpd = count(<=(0.0), mineig), min_eig = minimum(mineig),
            n_nonconverged = count(!, converged))
end

"""Observed information with fixed starts, progress output, and a wall-time guard."""
function information_for(m::WarfarinTurnover, pop::PopulationData, f, tag::String)
    gcl = curvature_gradient(m, pop, f.cfg, rcfg, CC_LAPLACE; fixed_starts = f.starts)
    seconds = @elapsed J = Matrix(observed_information(
        gcl, f.theta; step = 1e-5, central = true,
        max_seconds = MAX_INFORMATION_SECONDS, show_progress = true))
    write_information_binary(joinpath(RESULTS, "e9b_information_$(tag).bin"), J)
    return J, seconds
end

crows = Any[csvrow("truth", "fit_model", "cohort", "lambda", "C", "E")]
mrows = Any[csvrow("truth", "fit_model", "cohort", "lambda", "hessian_regularization",
                   "p", "OFV_unpenalised", "penalty_on_L", "phi_norm2", "iterations",
                   "seconds", "information_seconds", "information_finite", "return_code",
                   "gradient_norm", "objective_rescues", "gradient_rescues",
                   "fit_inner_nonconverged", "diagnostic_inner_nonconverged",
                   "subjects_nonpd_hessian",
                   "min_subject_hessian_eigenvalue", "n_negative_information",
                   "n_kept_information")]
trows = Any[csvrow("truth", "fit_model", "cohort", "lambda", "theta")]
progress = Any[csvrow("cohort", "truth", "fit_model", "completed_at", "fit_seconds",
                      "information_seconds", "information_finite", "return_code")]

for t in TRUTHS, C in CGRID
    push!(crows, csvrow(t.name, "truth", 0, LAMBDA, C,
                        effect(t.model.effect, view(t.theta, theta_index(t.model).effect), [C])[1]))
end

function checkpoint!(completed_replicates::Int; render::Bool = false)
    writedlm(joinpath(RESULTS, "e9b_effect_curves.csv"), vcat(crows...), ',')
    writedlm(joinpath(RESULTS, "e9b_fits.csv"), vcat(mrows...), ',')
    writedlm(joinpath(RESULTS, "e9b_theta_hat.csv"), vcat(trows...), ',')
    writedlm(joinpath(RESULTS, "e9b_progress.csv"), vcat(progress...), ',')
    @printf("checkpoint: %d complete replicate(s), %d complete fit arm(s) at %s\n",
            completed_replicates, length(progress) - 1, string(now())); flush(stdout)
    if render && !SMOKE
        script = joinpath(REPO_ROOT, "warfarin_turnover", "figures", "make_figures.jl")
        project = dirname(Base.active_project())
        julia = joinpath(Sys.BINDIR, Base.julia_exename())
        run(`$julia --project=$project -t 1 $script e9`)
        src = joinpath(REPO_ROOT, "warfarin_turnover", "figures",
                       "fig4_e9_misspecification.png")
        dst = joinpath(RESULTS, @sprintf("e9b_figure_after_%d_replicates.png",
                                        completed_replicates))
        cp(src, dst; force = true)
        println("rendered = ", dst); flush(stdout)
    end
end

@printf("\ngamma = %.3g, Hessian regularization = %.1e, %d cohorts, %d subjects, %d truths\n",
        LAMBDA, HESSIAN_REGULARIZATION, length(COHORTS), N_SUBJECTS, length(TRUTHS))

println("\nwarm-up (discarded) ..."); flush(stdout)
let w = simulate_cohort(8; seed = SEED_COHORT, theta = tp.theta, model = m_par)
    for m in (m_par, m_nn)
        sub = [make_subject(m, s.id, s.input, s.covariates.pk_times,
                            s.observations[1:s.covariates.n_pk], s.covariates.pd_times,
                            s.observations[s.covariates.n_pk+1:end]) for s in w.pop.subjects]
        fit_penalised(m, PopulationData(sub), LAMBDA; maxiters = 3)
    end
end
println("  warm."); flush(stdout)

t0 = time()
for r in COHORTS
    for t in TRUTHS
        sim = simulate_cohort(N_SUBJECTS; seed = t.seed0 + r, theta = t.theta, model = t.model)
        for (fitname, m) in (("M_par", m_par), ("M_nn($(WIDTH))", m_nn))
            println("\narm started: cohort=", r, " truth=", t.name, " model=", fitname,
                    " at ", now()); flush(stdout)
            subs = [make_subject(m, s.id, s.input, s.covariates.pk_times,
                                 s.observations[1:s.covariates.n_pk], s.covariates.pd_times,
                                 s.observations[s.covariates.n_pk+1:end]) for s in sim.pop.subjects]
            pop = PopulationData(subs)
            f = fit_penalised(m, pop, LAMBDA)
            hd = laplace_diagnostics(m, pop, f.theta, f.starts, f.cfg)
            tag = "$(t.name)_$(tagof(fitname))_c$(r)"
            J, info_seconds = information_for(m, pop, f, tag)
            information_finite = all(isfinite, J)
            negJ = keptJ = -1
            if information_finite
                ev = eigvals(Symmetric(J))
                tau = resolve_tau(:noise, ev)
                negJ, keptJ = count(<=(0), ev), count(>(tau), ev)
            else
                println("  covariance failure: observed information contains non-finite entries")
            end

            phi = view(f.theta, theta_index(m).effect)
            for C in CGRID
                push!(crows, csvrow(t.name, fitname, r, LAMBDA, C,
                                    effect(m.effect, phi, [C])[1]))
            end
            push!(mrows, csvrow(t.name, fitname, r, LAMBDA, f.cfg.reg,
                                m.p, f.ofv, f.penalty, sum(abs2, phi), f.iters, f.seconds,
                                info_seconds, information_finite, f.code, f.gradient_norm,
                                f.objective_rescues, f.gradient_rescues,
                                f.inner_nonconverged, hd.n_nonconverged, hd.n_nonpd,
                                hd.min_eig, negJ, keptJ))
            push!(trows, csvrow(t.name, fitname, r, LAMBDA, join(f.theta, " ")))
            push!(progress, csvrow(r, t.name, fitname, string(now()), f.seconds,
                                   info_seconds, information_finite, f.code))
            @printf("  %-13s %-10s c%d OFV %9.1f fit %6.1fs info %6.1fs finite=%-5s subject non-PD %d (min eig %.3g) rescues %d/%d info neg %3d kept %3d (%5.1f min)\n",
                    t.name, fitname, r, f.ofv, f.seconds, info_seconds,
                    string(information_finite), hd.n_nonpd, hd.min_eig,
                    f.objective_rescues, f.gradient_rescues, negJ, keptJ,
                    (time() - t0) / 60); flush(stdout)
            complete = t.name == last(TRUTHS).name && m === m_nn
            checkpoint!(r - 1 + (complete ? 1 : 0); render = complete)
        end
    end
end

@printf("\nall fits and covariance steps in %.1f min\n", (time() - t0) / 60)
println("finished = ", now())
println("=" ^ 78); flush(stdout)
