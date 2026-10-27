# Outer (population-level) optimization and asymptotic standard errors.
#
# This is explicitly the part intended to be swapped and tuned later: one file,
# depending on nothing else here, taking the gradient as an argument.

"""
    observed_information(grad!, θ; step=1e-5, central=true) -> Symmetric

`∇²_θ L` by differencing the supplied **L**-gradient, symmetrized.

`central = true` (the default) uses `(g(θ+h) − g(θ−h)) / 2h`, at `2p` gradient
evaluations against `p + 1` for the one-sided form. The extra cost buys an error
of `O(h²)` rather than `O(h)`, and — more importantly here — insensitivity to the
step. Measured on the Michaelis-Menten problem at the fitted θ̂, relative to
central differencing at `h = 1e-5`:

| scheme | h = 1e-4 | h = 1e-5 | h = 1e-6 |
|---|---|---|---|
| one-sided | 3.7e-3 | 6.1e-4 | 2.8e-2 |
| central | 5.4e-4 | — | 3.3e-3 |

The one-sided scheme degrades by a factor of 45 between `h = 1e-5` and `1e-6`,
because the `O(h)` truncation term stops dominating and gradient noise divided by
`h` takes over. Central differencing keeps that transition an order of magnitude
further away, which matters because the right step is not known in advance.
"""
function observed_information(grad!::G, θ::AbstractVector{Float64};
                              step::Float64 = 1e-5, central::Bool = true,
                              max_seconds::Float64 = Inf,
                              show_progress::Bool = false) where {G}
    p = length(θ)
    H = zeros(p, p)
    gp = zeros(p)
    gm = zeros(p)
    t0 = time()
    central || grad!(gm, collect(Float64, θ))     # one-sided: gm is the base point
    for j in 1:p
        time() - t0 < max_seconds ||
            error("observed-information wall-clock limit of $(max_seconds) seconds reached at column $(j)/$(p)")
        s = step * max(abs(θ[j]), 1.0)
        θp = collect(Float64, θ); θp[j] += s
        grad!(gp, θp)
        if central
            θm = collect(Float64, θ); θm[j] -= s
            grad!(gm, θm)
            @views H[:, j] .= (gp .- gm) ./ (2s)
        else
            @views H[:, j] .= (gp .- gm) ./ s
        end
        show_progress && (println("  information column ", j, "/", p,
                                  "  elapsed_s = ", round(time() - t0; digits = 1));
                          flush(stdout))
    end
    return Symmetric((H + H') / 2)
end

"""
    standard_errors(grad!, θ; step=1e-5, central=true) -> NamedTuple

Asymptotic standard errors by differencing the supplied **L**-gradient to get
`∇²_θ L`, symmetrizing, and inverting.

`L = −log p(y | θ)` up to an additive constant, so `∇²_θ L` *is* the observed
Fisher information and `Cov(θ̂) = (∇²_θ L)⁻¹`. (On the OFV scale the same thing
reads `Cov = 2 (∇²_θ OFV)⁻¹`; passing an OFV gradient here would halve every
variance.)

Returns `(se, vcov, hessian, pd)`. `pd = false` — a non-positive-definite
information matrix — means the SEs are meaningless and they come back as NaN
rather than as a plausible-looking number. This is also a second consumer of the
gradient, which makes it a useful smoke test of it.

These are *model-based* errors: they are correct if the model is correctly
specified. See [`robust_errors`](@ref) for the sandwich form, which is not.
"""
function standard_errors(grad!::G, θ::AbstractVector{Float64};
                         step::Float64 = 1e-5, central::Bool = true) where {G}
    p = length(θ)
    Hs = observed_information(grad!, θ; step = step, central = central)
    C = cholesky(Hs; check = false)
    issuccess(C) || return (se = fill(NaN, p), vcov = fill(NaN, p, p),
                            hessian = Matrix(Hs), pd = false)
    Σ = inv(Hs)
    return (se = sqrt.(diag(Σ)), vcov = Matrix(Σ), hessian = Matrix(Hs), pd = true)
end

"""
    robust_errors(model, pop, θ, cfg; step=1e-5, central=true) -> NamedTuple

Sandwich (Huber-White) standard errors,

    Cov(θ̂) = H⁻¹ ( Σᵢ gᵢ gᵢᵀ ) H⁻¹,     H = ∇²_θ L,   gᵢ = ∇_θ Lᵢ,

where the per-subject scores `gᵢ` are the columns the adjoint gradient already
produces — `adjoint_gradient_parts(...).grads` — so the "meat" costs one extra
gradient evaluation rather than `N` of them.

The model-based errors in [`standard_errors`](@ref) equate the two matrices via
the information equality, which holds only when the model is correctly specified.
The sandwich form does not assume it, so the *ratio* of the two is a
misspecification diagnostic: on data simulated from the model itself the two
agree, and a large disagreement means the variance structure is wrong even where
the point estimates look reasonable.

Returns `(se, vcov, hessian, meat, pd, score_sum)`. `score_sum` is `Σᵢ gᵢ`, which
the theory assumes is zero at θ̂; it equals the population gradient, so a value
far from zero means the fit has not converged and the sandwich rests on a
premise that does not hold there.
"""
function robust_errors(model, pop::PopulationData, θ::AbstractVector{Float64},
                       cfg::LaplaceConfig; step::Float64 = 1e-5,
                       central::Bool = true)
    p = length(θ)
    g! = adjoint_gradient(model, pop, cfg)
    Hs = observed_information(g!, θ; step = step, central = central)

    parts = adjoint_gradient_parts(model, collect(Float64, θ), pop, cfg)
    G = parts.grads                       # p × N, one column per subject
    meat = Symmetric(G * transpose(G))

    C = cholesky(Hs; check = false)
    issuccess(C) || return (se = fill(NaN, p), vcov = fill(NaN, p, p),
                            hessian = Matrix(Hs), meat = Matrix(meat), pd = false,
                            score_sum = vec(sum(G, dims = 2)))
    Hi = inv(Hs)
    V = Hi * meat * Hi
    return (se = sqrt.(abs.(diag(V))), vcov = Matrix(V), hessian = Matrix(Hs),
            meat = Matrix(meat), pd = true, score_sum = vec(sum(G, dims = 2)))
end

"""
    fit_nlme(model, pop, θ0; lower, upper, kwargs...) -> NLMEResult

Population fit driven by an externally supplied gradient.

`lower` and `upper` are **required**: there is no sensible default, and defaulting
them assumes every parameter is on a log scale. They are always checked against
the returned θ̂, whether or not the optimizer enforced them (see below).

## Choice of optimizer

The default is unconstrained `BFGS` with a backtracking line search, which is
what Pumas documents as its own default
(`Optim.BFGS(linesearch = LineSearches.BackTracking(), initial_stepnorm = 1.0)`).
It replaced `Fminbox(LBFGS())` on measurement, not on taste. On the
two-compartment Michaelis-Menten problem at N = 200, from an identical start:

| optimizer | seconds | iterations | final ‖∇L‖∞ | outcome |
|---|---|---|---|---|
| `Fminbox(LBFGS())` | 105.1 | 90 | 1.56e-3 | stalled |
| `LBFGS()` unconstrained | 65.8 | 90 | 1.56e-3 | stalled |
| **`BFGS()` unconstrained** | **30.8** | **53** | **8.5e-4** | **converged** |
| `BFGS()`, Hager-Zhang line search | 179.2 | 300 | 8.5e-1 | hit the cap |

All reached the same OFV and the same θ̂ to within 1e-4, so this is a pure cost
and robustness difference. Two things it shows:

  * `Fminbox` was *why* the fit stalled just above `g_tol`. Its barrier
    reformulation perturbs the objective near the box even when the box is not
    active, and here the box — ±6 on a log scale — is never active. Dropping it
    alone removed 40% of the runtime; dropping it and switching to a dense
    quasi-Newton removed 70%.
  * The line search matters as much as the quasi-Newton update. `BFGS` with
    Hager-Zhang was the worst of the four, so the gain belongs to the
    `BFGS` + `BackTracking` pairing rather than to `BFGS` alone.

At `p` in the tens, a dense BFGS inverse-Hessian costs nothing and converges in
fewer iterations than the limited-memory form; `LBFGS` becomes the right choice
only when `p` is large enough for an `O(p²)` matrix to matter.

**Bounds are then advisory rather than enforced.** Since the default optimizer is
unconstrained, `fit_nlme` checks the returned θ̂ against `lower`/`upper` itself and
reports `return_code = :outside_bounds` with `converged = false` if it escaped —
a silent bound violation being exactly the failure this project refuses to
tolerate. Pass `optimizer = Fminbox(LBFGS(...))` to enforce the box genuinely,
at the cost above. That is the right choice when a bound is a hard physical
constraint rather than a sanity rail.

No AD backend is attached to the `OptimizationFunction`, so the outer optimizer
never differentiates anything itself.

Keyword arguments:

  * `cfg` — the `LaplaceConfig` shared by the objective and the gradient.
  * `gradient` — `:adjoint` (default), `:ad` or `:fd`.
  * `grad` / `objective` — supply your own callables instead, keeping the outer
    layer method-agnostic.

    **`NLMEResult.L` and `.OFV` are always the EXACT-LAPLACE values**, recomputed here by
    `population_objective`, whatever objective was supplied. That is deliberate — the result
    then carries a curvature-independent reference — but it is a trap: fit under
    `CurvatureConfig(:foce)` and the returned OFV is *not* the objective that was minimised,
    and on a neural model the two can differ by tens of thousands. A caller that reports an
    OFV must recompute it from its own objective at θ̂ and say which curvature it is.
  * `maxiters` — enforced in the callback, which fires once per iteration.
    Counting in the *gradient* closure instead would count gradient evaluations,
    which a backtracking line search inflates.
  * `max_seconds` — optional wall-clock limit for one optimization. The callback
    checks it after each completed iteration, so it cannot interrupt an individual
    objective or gradient evaluation.
  * `g_tol` — convergence is judged on a recomputed final gradient norm, not on
    the solver's return code.
  * `compute_se` — also return standard errors.
"""
function fit_nlme(model, pop::PopulationData, θ0::AbstractVector;
                  lower, upper,
                  cfg::LaplaceConfig = LaplaceConfig(),
                  gradient::Symbol = :adjoint,
                  objective = nothing,
                  grad = nothing,
                  fd_step::Float64 = 1e-4,
                  maxiters::Int = 200,
                  max_seconds::Float64 = Inf,
                  g_tol::Float64 = 1e-3,
                  f_tol::Float64 = 0.0,          # ABSOLUTE OFV change; see the callback below
                  f_tol_patience::Int = 5,
                  compute_se::Bool = false,
                  se_step::Float64 = 1e-5,
                  optimizer = BFGS(linesearch = LineSearches.BackTracking(),
                                   initial_stepnorm = 1.0),
                  show_trace::Bool = false)

    f = objective === nothing ? laplace_objective(model, pop, cfg) : objective
    g! = grad === nothing ?
         gradient_factory(gradient, model, pop, cfg; fd_step = fd_step) : grad

    # Non-finite values propagate as NaN everywhere upstream and are converted at
    # exactly this boundary, where the conversion is counted. The two counters are
    # kept apart because they mean different things — see `NLMEResult`. An
    # objective rescue is the line search discovering that θ is outside the region
    # where the Laplace expression is defined, and is benign. A gradient rescue
    # deletes information the optimizer cannot tell is missing, and must be zero.
    obj_rescues = Ref(0)
    grad_rescues = Ref(0)

    # The last value of the CALLER's objective, recorded where it is actually computed.
    #
    # It is not taken from the callback, and that is not fastidiousness. `OptimizationProblem`
    # is given `lb`/`ub` here, and Optimization.jl reformulates a bounded problem — so the value
    # the callback receives is the *reformulated* objective, not `f`. On a two-parameter toy the
    # gap is 118.0 against 117.871; on a 1773-parameter fit with ±50 weight bounds the same
    # mechanism turned a true OFV of +11137 into a reported −40761. Anything derived from the
    # callback's value — a printed trace, `rel_ofv_change` — would describe a function the
    # caller never asked for.
    last_obj = Ref(NaN)

    fwrap = function (θ, _p)
        v = f(θ)
        if !isfinite(v)
            obj_rescues[] += 1
            return 1e10
        end
        last_obj[] = v
        return v
    end

    gwrap = function (G, θ, _p)
        g!(G, θ)
        if !all(isfinite, G)
            grad_rescues[] += 1
            @inbounds for j in eachindex(G)
                isfinite(G[j]) || (G[j] = 0.0)
            end
        end
        return G
    end

    niter = Ref(0)
    history = Float64[]
    n_flat = Ref(0)
    stopped_on_f = Ref(false)
    stopped_on_time = Ref(false)
    fit_start = Ref(time())
    cb = function (state, args...)
        niter[] += 1
        v = last_obj[]
        isfinite(v) && push!(history, 2v)
        show_trace && (println("  iter ", niter[], "  OFV = ", 2v,
                               "  elapsed_s = ", round(time() - fit_start[]; digits = 1));
                       flush(stdout))
        if time() - fit_start[] >= max_seconds
            stopped_on_time[] = true
            show_trace && (println("  stopping: wall-clock limit of ", max_seconds,
                                   " seconds reached"); flush(stdout))
            return true
        end
        # Objective-based early stop, in ABSOLUTE OFV units.
        #
        # `g_tol` alone is not usable on a model whose curvature spans five orders of magnitude:
        # the gradient norm measures steepness, not distance from the optimum, and a finished
        # fit can still show ‖∇L‖∞ of order 1. Without a second rule every such fit runs to
        # `maxiters` regardless of having arrived.
        #
        # The test is absolute, not relative, and that is the whole point. A likelihood-ratio
        # decision turns on ΔOFV = 3.84 whether the OFV is 100 or 11137, so "improving by less
        # than X OFV units per iteration" has a meaning that "improving by less than a fraction
        # of the OFV" does not. Measured on a p = 1773 fit: a relative tolerance of 1e-8 demands
        # an absolute change below 1.1e-4 while the late iterations move by ~5e-3, so it never
        # fired and the fit ran to its cap; the same rule fired on only 25 of 60 fits elsewhere.
        # An absolute 0.01 — one four-hundredth of 3.84 — stops at the point where nothing that
        # could change an inference remains.
        #
        # `f_tol_patience` consecutive iterations are required, because the trace is not
        # monotone in step size: a late line search can take a bigger step than an earlier one.
        if f_tol > 0 && length(history) >= 2
            rel = abs(history[end] - history[end-1])
            n_flat[] = rel < f_tol ? n_flat[] + 1 : 0
            if n_flat[] >= f_tol_patience
                stopped_on_f[] = true
                show_trace && println("  stopping: relative OFV change < ", f_tol,
                                      " for ", f_tol_patience, " consecutive iterations")
                return true
            end
        end
        return niter[] >= maxiters
    end

    optf = OptimizationFunction(fwrap; grad = gwrap)

    # The box is passed to the PROBLEM only when the optimizer is itself a box method.
    #
    # `Optimization.jl` silently substitutes the optimizer otherwise: `SciMLBase.__init` wraps a
    # non-box optimizer in `Optim.Fminbox` whenever `prob.lb`/`ub` are set. So constructing the
    # problem with bounds and passing unconstrained `BFGS` does not run `BFGS` — it runs
    # `Fminbox(BFGS(...))`, a log-barrier interior-point method, which is precisely the optimizer
    # the table above says was dropped *on measurement* for stalling just above `g_tol` and for
    # costing 40% more runtime. Verified by printing `typeof(cache.opt)`: with bounds it is
    # `Fminbox{BFGS{...}}`, without them `BFGS{...}`.
    #
    # The barrier also displaces the objective by an amount that grows with `p` — measured at
    # 4.6e3 OFV units at p = 58 and 5.2e4 at p = 1773 — which is what made a fit whose true FOCE
    # OFV was 11137 report a trace running to −40761.
    #
    # Dropping the box here does not drop the *check*: θ̂ is still tested against `lower`/`upper`
    # below and reported as `:outside_bounds`. The bounds stay advisory, which is what the
    # docstring always claimed they were.
    wants_box = optimizer isa Optim.Fminbox || optimizer isa Optim.SAMIN
    prob = wants_box ?
        OptimizationProblem(optf, collect(Float64, θ0), nothing;
                            lb = collect(Float64, lower), ub = collect(Float64, upper)) :
        OptimizationProblem(optf, collect(Float64, θ0), nothing)

    t0 = time()
    fit_start[] = t0
    sol = solve(prob, optimizer; callback = cb, maxiters = maxiters)
    elapsed = time() - t0

    θ̂ = collect(Float64, sol.u)
    gfinal = zeros(length(θ̂))
    g!(gfinal, θ̂)
    gnorm = maximum(abs, gfinal)

    popres = population_objective(model, θ̂, pop, cfg)
    L = popres.L
    etas = [collect(Float64, e) for e in popres.etas]

    rel_change = length(history) >= 2 ?
        abs(history[end] - history[end - 1]) / max(abs(history[end]), 1.0) : NaN

    # Judged on the recomputed final gradient, not on the solver's return code and
    # not on what the line search probed on the way here.
    # The default optimizer is unconstrained, so the box is checked here rather
    # than assumed. Reported, never silently clipped: a θ̂ outside the box is a
    # statement about the fit, and clipping it would hide that.
    lo_v = collect(Float64, lower)
    hi_v = collect(Float64, upper)
    inside = all(lo_v .<= θ̂ .<= hi_v)

    converged = isfinite(gnorm) && gnorm < g_tol && grad_rescues[] == 0 && inside
    rc = if grad_rescues[] > 0
        :gradient_rescued
    elseif !inside
        :outside_bounds
    elseif converged
        :converged
    elseif stopped_on_time[]
        :time_limit
    elseif stopped_on_f[]
        :objective_flat
    elseif niter[] >= maxiters
        :maxiters
    else
        :terminated_without_gradient_convergence
    end

    se, vc = nothing, nothing
    if compute_se
        s = standard_errors(g!, θ̂; step = se_step)
        se, vc = s.se, s.vcov
    end

    return NLMEResult(θ̂, L, 2L, gfinal, gnorm, niter[], converged, rc, elapsed,
                      etas, obj_rescues[], grad_rescues[], popres.n_nonconverged,
                      rel_change, cfg, Threads.nthreads(), se, vc)
end

"""
    empirical_bayes(model, pop, θ, cfg) -> Vector{Vector{Float64}}

Per-subject conditional modes η̂ᵢ(θ) — the empirical Bayes estimates. A genuine
output, not a diagnostic afterthought.
"""
empirical_bayes(model, pop::PopulationData, θ::AbstractVector, cfg::LaplaceConfig) =
    [collect(Float64, e) for e in population_objective(model, θ, pop, cfg).etas]
