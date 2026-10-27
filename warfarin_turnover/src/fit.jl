# fit.jl — the fit driver and the artefact writers.
#
# Every fit in E3–E7 uses **FOCE** (`CurvatureConfig(:foce)`), matching the incumbent
# literature.  No model is fitted under exact Laplace anywhere in this study; the
# exact-Laplace arms appear only in the E1 verification gate and the E2 cost ladder, both of
# which are gradient-level.
#
# The artefact contract, and it is the reason this file is longer than the fit it wraps:
# **a fit is expensive and a figure is not, so no figure may ever be a reason to refit.**
# Everything a plot or a downstream analysis could want comes out of `write_fit` as CSV with
# an explicit header — θ̂, its standard errors, the OFV trace, per-subject modes, both
# prediction levels, the recovered effect curve on a fixed concentration grid, and the run's
# own configuration and timing.  If something is missing from that list it is a bug in this
# file, not a reason to rerun the fit.

using NLMEReverse
using LinearAlgebra
using DelimitedFiles
using Printf

const CC_FOCE = CurvatureConfig(curvature = :foce)
const CC_LAPLACE = CurvatureConfig(curvature = :laplace)

"""
    default_config(; kwargs...) -> LaplaceConfig

The shared inner-solve configuration.  Constructed once per experiment and passed to both
the objective and the gradient — that shared object is what guarantees they describe the same
function.

`warm_start = false` throughout: a rolling warm start makes the objective history-dependent
at the level of `inner_tol`, which biases finite-difference comparisons and confuses a
backtracking line search.  Where a warm start is wanted for cost, it is supplied as
`fixed_starts` — computed once, never updated — which is history-independent by construction.
"""
default_config(; reg::Float64 = 1e-8, inner_tol::Float64 = 1e-8, inner_maxiter::Int = 100,
               threaded::Bool = true) =
    LaplaceConfig(reg = reg, inner_tol = inner_tol, inner_maxiter = inner_maxiter,
                  warm_start = false, threaded = threaded)

"""
    reverse_config(; backend = :mooncake) -> ReverseConfig

The production engine.  `:directional` is pinned rather than chosen by timing: `:matrix` is a
different code path, and letting a timing tie-break pick between them across scripts would
confound arms with code paths.
"""
reverse_config(; backend::Symbol = :mooncake) =
    ReverseConfig(backend = backend, form = :directional)

"""
    fit_warfarin(model, pop; θ0, cfg, rcfg, ccfg, maxiters, g_tol, optimizer, fixed_starts)
      -> NamedTuple

One population fit, plus the trace and timing artefacts `fit_nlme` does not itself return.

`fixed_starts = true` (the default) computes the conditional modes once at θ₀ and starts
every subsequent inner solve there.  It is a pure cost measure with no effect on the
objective's identity, because the starts never change.

`optimizer` defaults to dense BFGS with a backtracking line search — measured, on this
framework, to beat `LBFGS` and `Fminbox(LBFGS)` at `p` in the tens.  At `p ≈ 10⁴` a dense
inverse Hessian is ~0.9 GB and `LBFGS` becomes the right choice; `capability_optimizer` below
makes that switch explicit rather than implicit.
"""
function fit_warfarin(m::WarfarinTurnover, pop::PopulationData;
                      θ0::Vector{Float64} = theta_start(m),
                      cfg::LaplaceConfig = default_config(),
                      rcfg::ReverseConfig = reverse_config(),
                      ccfg::CurvatureConfig = CC_FOCE,
                      maxiters::Int = 300,
                      g_tol::Float64 = 1e-3,
                      # ABSOLUTE OFV units, measured rather than guessed. Sweep at W = 10,
                      # N = 100, against a 400-iteration reference (OFV 11213.86, 197.6 s):
                      #
                      #   f_tol    iters   OFV        cost   ||g||inf   time
                      #   off        400   11213.86      -      7.01   197.6 s
                      #   0.01       124   11217.04   +3.18     1.06    39.2 s
                      #   0.003      149   11216.74   +2.87     0.52    42.5 s
                      #   0.001      176   11216.64   +2.78     1.44    48.2 s
                      #   0.0003     400   11213.86   +0.00     7.01   113.3 s
                      #
                      # 0.001 costs 2.78 OFV units -- below the 3.84 that decides a parameter --
                      # for a 4x speedup. Tightening to 0.0003 falls off a cliff: the per-iteration
                      # improvement plateaus near 1e-3, so below that the rule never fires and the
                      # fit runs to its cap. The predecessor, 1e-8, was a RELATIVE tolerance and
                      # was mis-scaled by four orders on an OFV of ~1e4; it fired on only 25 of 60
                      # E4 fits and never on E3.
                      f_tol::Float64 = 0.001,
                      optimizer = nothing,
                      fixed_starts::Bool = true,
                      show_trace::Bool = false)

    b = theta_bounds(m)
    starts = fixed_starts ?
        [collect(Float64, e) for e in population_objective(m, θ0, pop, cfg).etas] : nothing

    obj_inner = foce_objective(m, pop, cfg, ccfg; fixed_starts = starts)
    grad_inner = curvature_gradient(m, pop, cfg, rcfg, ccfg; fixed_starts = starts)

    n_obj = Ref(0)
    n_grad = Ref(0)
    t_obj = Ref(0.0)
    t_grad = Ref(0.0)
    trace = Float64[]

    objective = function (θ)
        n_obj[] += 1
        tic = time_ns()
        v = obj_inner(θ)
        t_obj[] += (time_ns() - tic) / 1e9
        push!(trace, 2v)
        return v
    end
    grad = function (G, θ)
        n_grad[] += 1
        tic = time_ns()
        grad_inner(G, θ)
        t_grad[] += (time_ns() - tic) / 1e9
        return G
    end

    opt = optimizer === nothing ? capability_optimizer(m.p) : optimizer

    r = fit_nlme(m, pop, θ0; lower = b.lower, upper = b.upper, cfg = cfg,
                 objective = objective, grad = grad, maxiters = maxiters, g_tol = g_tol,
                 f_tol = f_tol, optimizer = opt, show_trace = show_trace)

    # `fit_nlme` recomputes `r.L` with `population_objective`, i.e. the EXACT-LAPLACE
    # objective, whatever objective it was handed. For a FOCE fit that is not the function
    # that was minimised, and on this model the two differ by tens of thousands — the trace
    # ends near -40761 while `r.OFV` reads +11161. So the OFV that gets reported is recomputed
    # here from the objective that was actually optimised, and the Laplace value is carried
    # alongside it, labelled, rather than silently standing in for it.
    ofv_fitted = 2 * obj_inner(r.theta)

    return (result = r, theta0 = θ0, trace = trace, n_obj = n_obj[], n_grad = n_grad[],
            t_obj = t_obj[], t_grad = t_grad[], cfg = cfg, rcfg = rcfg, ccfg = ccfg,
            grad_closure = grad_inner, obj_closure = obj_inner, eta_starts = starts,
            optimizer = string(nameof(typeof(opt))),
            ofv = ofv_fitted, ofv_laplace = r.OFV, curvature = ccfg.curvature)
end

"""
    capability_optimizer(p) -> an Optim optimizer

Dense `BFGS` below `p = 2000`, limited-memory `LBFGS` above it.

The threshold is a memory statement, not a taste one: `BFGS` carries a dense `p × p` inverse
Hessian, which is 32 MB at `p = 2000` and 870 MB at `p ≈ 10⁴`.  Both use the same
backtracking line search, which measurement attributed most of the gain to.
"""
capability_optimizer(p::Int) = p < 2000 ?
    BFGS(linesearch = LineSearches.BackTracking(), initial_stepnorm = 1.0) :
    LBFGS(m = 20, linesearch = LineSearches.BackTracking())

# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------

"""
    parameter_names(model) -> Vector{String}

Human-readable names in θ order, so every artefact is joinable on name rather than on index.
"""
function parameter_names(m::WarfarinTurnover)
    ix = theta_index(m)
    names = ["log_ka", "log_CL", "log_V", "log_R0", "log_kout"]
    if m.effect isa EmaxEffect
        append!(names, ["log_Emax", "log_EC50"])
    else
        append!(names, ["w_$(j)" for j in 1:n_effect_parameters(m)])
    end
    append!(names, ["log_omega_ka", "log_omega_CL", "log_omega_V", "log_omega_R0", "log_omega_kout"])
    append!(names, ["log_sigma_prop", "log_sigma_add"])
    length(names) == m.p || error("parameter_names produced $(length(names)) for p = $(m.p).")
    return names
end

"""
    effect_grid(; n = 201, cmax = 35.0)

The concentration grid every effect curve is reported on.  Fixed, so curves from different
fits, widths and cohorts are directly comparable and stackable without interpolation.

`cmax = 35` mg/L spans the simulated design: the top dose level is 4 mg/kg, whose peak
concentration under the fitted truth is ~30 mg/L.  A grid stopping at the middle dose's peak
would truncate the curve exactly where a saturating shape is identified, which is the region
the three-dose design exists to reach.  The real 32-subject data reach only 17.6 mg/L, so a
curve from that fit is extrapolating above it — `max_observed_conc` is written onto every
`fit_meta` so any figure can truncate to the observed range rather than guessing it.
"""
effect_grid(; n::Int = 201, cmax::Float64 = 35.0) = collect(range(0.0, cmax; length = n))

"""
    csvrow(xs...) -> 1×N Matrix{Any}

One row of a CSV, in the only shape `writedlm(vcat(rows...))` accepts.

This exists because every alternative silently produces something else, and each has already
cost this project a completed run, discovered at the FINAL write after the compute was done:

  * `[a, b, c]` — commas make an N×1 **column**, not a row;
  * a literal split across two lines — the newline is a **row separator**;
  * `vcat(["name" 1.0], …)` on mixed `String` and `Float64` without an `Any` annotation
    promotes to a common element type and throws when there is none.

`csvrow("name", 1.0, true)` is unambiguous in all three. Vector arguments splat, so
`csvrow(i, theta)` writes an index followed by every parameter.
"""
csvrow(xs...) = reshape(Any[y for x in xs for y in (x isa AbstractVector ? x : (x,))], 1, :)

"""
    write_fit(dir, model, pop, fit; tag, information = nothing, extra = ())

Write everything a downstream figure or analysis could need.  Files, all CSV with headers:

| file | contents |
|---|---|
| `theta_hat<tag>.csv` | index, name, θ̂, θ₀, lower, upper, SE, and the natural-scale value |
| `fit_meta<tag>.csv` | OFV, L, ‖∇L‖∞, iterations, convergence, timings, counts, config |
| `gradient<tag>.csv` | the final gradient, per parameter |
| `ofv_trace<tag>.csv` | OFV against iteration — the convergence figure, without refitting |
| `eta_hat<tag>.csv` | the conditional modes, per subject |
| `predictions<tag>.csv` | id, time, channel, observed, IPRED (η̂) and PRED (η = 0) |
| `effect_curve<tag>.csv` | `E(C)` on the fixed grid, with its delta-method SE where available |
| `information<tag>.csv` | eigenvalue spectrum and diagonal of the observed information |

`information` is the `Symmetric` matrix from `observed_information`, when one was computed.
Its full `p × p` content is **not** written as CSV — at `p ≈ 10⁴` that is a 0.9 GB text file
— but the spectrum, the diagonal and the standard errors are, and `write_information_binary`
exists for the cases where the matrix itself must survive the run.
"""
function write_fit(dir::AbstractString, m::WarfarinTurnover, pop::PopulationData, fit;
                   tag::AbstractString = "", information = nothing, extra = ())
    mkpath(dir)
    sfx = isempty(tag) ? "" : "_" * tag
    r = fit.result
    names = parameter_names(m)
    b = theta_bounds(m)

    se = fill(NaN, m.p)
    pd = false
    if information !== nothing
        C = cholesky(information; check = false)
        pd = issuccess(C)
        pd && (se .= sqrt.(diag(inv(information))))
    end

    rows = Any[["index" "name" "theta_hat" "theta_start" "lower" "upper" "se" "gradient"]]
    for j in 1:m.p
        push!(rows, [j names[j] r.theta[j] fit.theta0[j] b.lower[j] b.upper[j] se[j] r.gradient[j]])
    end
    writedlm(joinpath(dir, "theta_hat$(sfx).csv"), vcat(rows...), ',')

    writedlm(joinpath(dir, "gradient$(sfx).csv"),
             vcat(["index" "name" "dL_dtheta"],
                  [Any[j names[j] r.gradient[j]] for j in 1:m.p]...), ',')

    writedlm(joinpath(dir, "ofv_trace$(sfx).csv"),
             vcat(["iteration" "OFV"],
                  [Any[k fit.trace[k]] for k in eachindex(fit.trace)]...), ',')

    erows = Any[["id" "dose" "eta_ka" "eta_CL" "eta_V" "eta_R0" "eta_kout"]]
    for (i, s) in enumerate(pop.subjects)
        e = r.eta[i]
        push!(erows, [s.id s.input e[1] e[2] e[3] e[4] e[5]])
    end
    writedlm(joinpath(dir, "eta_hat$(sfx).csv"), vcat(erows...), ',')

    z = zeros(Float64, m.q)
    prows = Any[["id" "dose" "channel" "time" "dv" "ipred" "pred"]]
    for (i, s) in enumerate(pop.subjects)
        ip = NLMEReverse.predict(m, r.eta[i], r.theta, s)
        pr = NLMEReverse.predict(m, z, r.theta, s)
        cv = s.covariates
        for k in eachindex(s.times)
            push!(prows, [s.id s.input (k <= cv.n_pk ? "conc" : "pca") s.times[k] s.observations[k] ip[k] pr[k]])
        end
    end
    writedlm(joinpath(dir, "predictions$(sfx).csv"), vcat(prows...), ',')

    Cs = effect_grid()
    Es = effect_curve(m, r.theta, Cs)
    writedlm(joinpath(dir, "effect_curve$(sfx).csv"),
             vcat(["C" "E"], [Any[Cs[k] Es[k]] for k in eachindex(Cs)]...), ',')

    if information !== nothing
        H = Matrix(information)
        ev = eigvals(Symmetric(H))
        writedlm(joinpath(dir, "information$(sfx).csv"),
                 vcat(["index" "name" "eigenvalue" "diagonal" "se"],
                      [Any[j names[j] ev[j] H[j, j] se[j]] for j in 1:m.p]...), ',')
    end

    nobs = n_observations(pop)
    maxconc = maximum(maximum(s.observations[1:s.covariates.n_pk]) for s in pop.subjects)
    meta = Any[["quantity" "value"]]
    for (k, v) in [("OFV", fit.ofv), ("OFV_curvature", String(fit.curvature)),
                   ("OFV_exact_laplace_reference", fit.ofv_laplace),
                   ("L", fit.ofv / 2), ("grad_inf_norm", r.gradient_norm),
                   ("iterations", r.iterations), ("converged", r.converged),
                   ("return_code", String(r.return_code)), ("elapsed_s", r.elapsed),
                   ("n_objective_calls", fit.n_obj), ("n_gradient_calls", fit.n_grad),
                   ("t_objective_s", fit.t_obj), ("t_gradient_s", fit.t_grad),
                   ("objective_rescues", r.objective_rescues),
                   ("gradient_rescues", r.gradient_rescues),
                   ("inner_nonconverged", r.inner_nonconverged),
                   ("rel_ofv_change", r.rel_ofv_change),
                   ("p", m.p), ("q", m.q), ("n_subjects", pop.n_subjects),
                   ("n_obs_pk", nobs.n_pk), ("n_obs_pd", nobs.n_pd), ("n_obs", nobs.total),
                   ("max_observed_conc", maxconc), ("effect_grid_cmax", maximum(Cs)),
                   ("effect", string(typeof(m.effect).name.name)),
                   ("width", m.effect isa MLPEffect ? m.effect.width : 0),
                   ("c0", m.effect isa MLPEffect ? m.effect.c0 : NaN),
                   ("rk4_n_seg", m.n_seg), ("rk4_gamma", m.gamma),
                   ("curvature", String(fit.ccfg.curvature)),
                   ("backend", String(fit.rcfg.backend)), ("form", String(fit.rcfg.form)),
                   ("optimizer", fit.optimizer),
                   ("fixed_starts", fit.eta_starts !== nothing),
                   ("inner_tol", fit.cfg.inner_tol), ("inner_maxiter", fit.cfg.inner_maxiter),
                   ("reg", fit.cfg.reg), ("threaded", fit.cfg.threaded),
                   ("nthreads", r.nthreads), ("blas_threads", BLAS.get_num_threads()),
                   ("julia_version", string(VERSION)),
                   ("information_pd", information === nothing ? "" : pd)]
        push!(meta, [k v])
    end
    for (k, v) in extra
        push!(meta, [String(k) v])
    end
    writedlm(joinpath(dir, "fit_meta$(sfx).csv"), vcat(meta...), ',')

    return dir
end

"""
    write_information_binary(path, H)

The raw `p × p` observed information, little-endian `Float64`, row-major-free (Julia's own
column order), with a companion `<path>.dims.csv` recording `p`.

Written only where recomputing it would cost `2p` gradients — i.e. where it is the expensive
artefact rather than the cheap one.  Printed size, never silent.
"""
function write_information_binary(path::AbstractString, H::AbstractMatrix)
    mkpath(dirname(path))
    p = size(H, 1)
    open(path, "w") do io
        write(io, Matrix{Float64}(H))
    end
    writedlm(path * ".dims.csv", ["p" p; "bytes" (8 * p * p)], ',')
    @info "observed information written" path p bytes = 8 * p * p
    return path
end

"""
    usable_cohorts(dir) -> Vector{Int}

The cohorts whose observed information supports inference, read from the gate audit
`e4f_gate_audit.csv` that E4f writes.

Twenty cohorts are fitted. Three of them return an observed information whose negative curvature
is of the same magnitude as its positive curvature -- the finite-difference pathology documented
for the unpenalised real-data fit, where the spectrum scales with the differencing step instead of
converging -- and at those estimates no interval exists to report. They are excluded here, once,
so that every table downstream rests on the same set rather than on whichever matrices happen to
be on disk. That is how the count silently became fifteen once before.

Falls back to every archived matrix if the audit has not been produced.
"""
function usable_cohorts(dir::AbstractString)
    audit = joinpath(dir, "e4f_gate_audit.csv")
    if isfile(audit)
        raw, hdr = readdlm(audit, ','; header = true)
        h = Dict(String(c) => j for (j, c) in enumerate(vec(hdr)))
        return sort([Int(round(Float64(raw[i, h["cohort"]]))) for i in axes(raw, 1)
                     if raw[i, h["passes_e4_gate"]] === true ||
                        lowercase(String(string(raw[i, h["passes_e4_gate"]]))) == "true"])
    end
    return sort([parse(Int, m[1]) for f in readdir(dir)
                 for m in [match(r"^information_c(\d+)\.bin$", f)] if m !== nothing])
end
