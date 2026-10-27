# The Laplace objective, per subject and over the population, plus the factories
# that hand a plain `θ -> value` closure to the outer optimizer.
#
#   L_i(θ) = h_i(η̂_i, θ) + ½ logdet H̃_i(θ) − (q/2) log 2π
#   L(θ)   = Σ_i L_i(θ),        OFV(θ) = 2 L(θ)
#
# The −(q/2)log2π constant is kept: it costs nothing and it is what makes the
# linear-Gaussian exactness test possible. It also means OFV values here are NOT
# comparable to OFVs reported by software that drops additive constants.

"""
    subject_objective(model, θ, subject, cfg; η_start=nothing) -> NamedTuple

One subject's Laplace contribution. Returns `(L, eta, Htilde, value, converged,
iters, gnorm)`, reusing the Hessian the inner solver already produced.
"""
function subject_objective(model, θ::AbstractVector, subject, cfg::LaplaceConfig;
                           η_start = nothing)
    q = n_random_effects(model)
    T = promote_type(eltype(θ), Float64)
    η_start = resolve_start(model, θ, subject, η_start, cfg)
    η0 = η_start === nothing ? zeros(T, q) : convert(Vector{T}, η_start)
    sol = newton_solve(model, θ, subject, η0, cfg)
    L = sol.value + logdet_pd(sol.Htilde) / 2 - (q / 2) * log(2π)
    return (L = L, eta = sol.η, Htilde = sol.Htilde, value = sol.value,
            converged = sol.converged, iters = sol.iters, gnorm = sol.gnorm)
end

"""
    population_objective(model, θ, pop, cfg; eta_starts=nothing) -> NamedTuple

`L(θ) = Σ_i L_i(θ)` with per-subject diagnostics.

This is the framework's only parallel region. Every per-subject result is written
to its own index in a buffer of `eltype(θ)` — never `Float64`, which would throw
a `MethodError` inside a `TaskFailedException` the moment AD-through runs — and
the reduction is a serial `sum` in index order, which is what makes the result
bitwise independent of the schedule and of the thread count. Convergence flags
and iteration counts are per-index vectors reduced afterwards, never shared
counters incremented in place.
"""
function population_objective(model, θ::AbstractVector, pop::PopulationData,
                              cfg::LaplaceConfig; eta_starts = nothing)
    n = pop.n_subjects
    T = promote_type(eltype(θ), Float64)
    Ls = zeros(T, n)
    etas = Vector{Vector{T}}(undef, n)
    conv = Vector{Bool}(undef, n)
    iters = Vector{Int}(undef, n)
    gnorms = zeros(Float64, n)
    subs = pop.subjects

    foreach_subject(n, cfg.threaded) do i
        st = eta_starts === nothing ? nothing : eta_starts[i]
        r = subject_objective(model, θ, subs[i], cfg; η_start = st)
        Ls[i] = r.L
        etas[i] = r.eta
        conv[i] = r.converged
        iters[i] = r.iters
        gnorms[i] = r.gnorm
    end

    return (L = sum(Ls), Ls = Ls, etas = etas, converged = conv, iters = iters,
            gnorms = gnorms, n_nonconverged = count(!, conv))
end

"""
    laplace_objective(model, pop, cfg) -> (θ -> L)

Factory for the population Laplace objective. Closing over `model`, `pop` and
`cfg` — the *same* `cfg` object the gradient factories receive — is the mechanism
that guarantees the objective and its gradient describe the same function.

When `cfg.warm_start` is set the previous modes are reused as starting points.
That is a documented perturbation, not a free lunch: it makes the objective
history-dependent at the level of `inner_tol`, which biases finite-difference
comparisons and can confuse a backtracking line search. The cache is written
serially, after the parallel region, never from inside it.
"""
function laplace_objective(model, pop::PopulationData, cfg::LaplaceConfig)
    cfg.warm_start || return θ -> population_objective(model, θ, pop, cfg).L
    cache = [zeros(Float64, n_random_effects(model)) for _ in 1:pop.n_subjects]
    return function (θ)
        r = population_objective(model, θ, pop, cfg; eta_starts = cache)
        for i in 1:pop.n_subjects
            cache[i] .= extract_value(r.etas[i])
        end
        return r.L
    end
end

"""
    laplace_ofv(model, pop, cfg) -> (θ -> 2L)

`OFV = 2L`, under a name distinct from [`laplace_objective`](@ref) so the factor
of two can never be ambiguous at a call site.
"""
function laplace_ofv(model, pop::PopulationData, cfg::LaplaceConfig)
    f = laplace_objective(model, pop, cfg)
    return θ -> 2 * f(θ)
end
