# reverse.jl — the reverse-mode population gradient (the O(1)-in-p construction).
#
# A genuinely reverse-mode population gradient for the exact-Laplace NLME objective the
# rest of this package implements.
#
# It adds nothing to the mathematics of `Mathematical_Approach.md` Eq. (4.4), which
# `gradient_adjoint.jl` already evaluates. It changes only **how the three terms of that
# equation are evaluated**, and that change is what makes the cost independent of `p`.
#
# Full derivation: `docs/reverse_gradient_derivation.md`.
# Code walkthrough: `docs/source_guide.md`.
#
# --------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------
#
# Eq. (4.4) is
#
#     ∇θ Lᵢ = ∂θ hᵢ(η̂ᵢ,θ) + ½ ∂θ logdet H̃ᵢ(η̂ᵢ,θ) − (∇²_ηθ hᵢ)ᵀ λᵢ .
#
# `gradient_adjoint.jl` avoids forming the q×p matrix in the third term via the VJP
# identity, but evaluates all three terms with `ForwardDiff.gradient`, i.e. `p` tangent
# directions: the adjoint *orientation* with the forward *cost model*.
#
# Term 2 is also a VJP. With `W ≡ ½H̃⁻¹` **frozen**,
#
#     d/dθ ½ logdet H̃(θ) = ½ tr( H̃⁻¹ ∂θH̃ ) = d/dθ ⟨ W , H̃(θ) ⟩ ,
#
# so all three terms are gradients of the **single** scalar
#
#     sᵢ(θ) = hᵢ(η̂ᵢ,θ) + ⟨Wᵢ, H̃ᵢ(η̂ᵢ,θ)⟩ − (∇_η hᵢ(η̂ᵢ,θ))ᵀ λᵢ ,
#     ∇θ Lᵢ = ∇θ sᵢ(θ)                                   [ONE reverse sweep, O(1) in p]
#
# --------------------------------------------------------------------------------
# Two forms of the same scalar
# --------------------------------------------------------------------------------
#
# `ReverseConfig(form = ...)` selects how `⟨W, H̃(θ)⟩` is written. Both are mathematically
# identical and both are O(1) in `p`; they differ only in what the reverse engine has to
# digest, which is the practical question when the engine is Enzyme or Mooncake.
#
# `:directional` (default, and the faster of the two)
#     With `H̃ᵢ = LLᵀ` — the Cholesky factor the objective already computed for its
#     log-determinant — and `uₘ` the m-th row of `L⁻¹`,
#
#         ⟨½H̃⁻¹, H̃(θ)⟩ = ½ tr(L⁻¹ H̃(θ) L⁻ᵀ) = ½ Σₘ uₘᵀ ∇²_ηη h uₘ
#                       = ½ Σₘ d²/dε² h(η̂ + ε uₘ, θ)|₀
#         (∇_η h)ᵀλ     =      d/dε  h(η̂ + ε λ , θ)|₀
#
#     so `sᵢ` is `q` second directional derivatives and one first directional derivative of
#     the user's `h` — scalar nested-`Dual` arithmetic and nothing else. The reverse sweep
#     never sees a factorisation, a log-determinant, or a `q×q` Hessian. Also cheaper
#     before the sweep: `q` evaluations carrying two partials each, against a Hessian's
#     evaluations carrying `q` partials each.
#
# `:matrix`
#     `⟨W, ∇²_ηη h(η̂,θ) + reg·I⟩` with `W = ½H̃⁻¹` formed explicitly and
#     `ForwardDiff.hessian` inside the sweep. Kept as a fallback: it asks the engine to
#     differentiate ForwardDiff's Hessian machinery rather than plain nested scalars, so it
#     is the more likely of the two to be rejected — but if an engine rejects one it may
#     accept the other, and `select_backend` tries both.
#
# Which curvature goes where is unchanged and remains load-bearing: `H̃` in the
# log-determinant, in `b` and in `W`; unshifted `H` as the coefficient of `Hλ = b`.
#
# --------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------
#
# Engines are added by including one file per engine (`src/backends/backends.jl` does this), each
# adding a method to `reverse_sweep`. This module depends on none of them, so a missing or
# non-precompiling engine cannot stop it loading. `:forwarddiff` is always present and is
# the **control arm**: the same scalar, `p` tangents. It exists to prove the reformulation
# alone buys nothing — the gain is reverse mode — and it is never a valid "reverse" result.

"""
    ReverseConfig(; backend = :forwarddiff, form = :directional)

`backend` selects the θ-gradient engine; `form` selects how the log-determinant term is
written (`:directional` or `:matrix` — see the module docstring).

Everything else about the estimator — the mode solve, `H`, `H̃`, `b`, `λ`, the objective
— comes from the `LaplaceConfig` passed alongside, so the two configurations together fix
exactly one estimator.

The default backend is the `:forwarddiff` control arm deliberately: a default that
silently required an optional package would fail at load rather than at the point of
choice. Use [`select_backend`](@ref) to pick a working reverse engine.
"""
Base.@kwdef struct ReverseConfig
    backend::Symbol = :forwarddiff
    form::Symbol = :directional
end

const CANDIDATE_BACKENDS = (:mooncake, :reversediff)
const FORMS = (:directional, :matrix)

# --------------------------------------------------------------------------
# Scalar directional derivatives of h in η, with θ live.
# --------------------------------------------------------------------------

"""
    dir1(f, x, v) -> vᵀ∇f(x)

First directional derivative, by one `Dual` carrying a single partial: one evaluation of
`f` at `Dual` depth one, against `length(x)` partials for a full gradient. This is what
keeps `(∇_η h)ᵀλ` a scalar the reverse sweep can handle without meeting an array-valued
intermediate.
"""
@inline function dir1(f::F, x::AbstractVector, v::AbstractVector) where {F}
    T = typeof(ForwardDiff.Tag(f, eltype(x)))
    xd = ForwardDiff.Dual{T}.(x, v)
    y = f(xd)
    y isa ForwardDiff.Dual{T} || error(
        "dir1: ForwardDiff tag nesting inverted (got $(typeof(y))). The ε-tag must sit " *
        "outside θ's tag; see docs/reverse_gradient_derivation.md §6.")
    return ForwardDiff.partials(y, 1)
end

"""
    dir2(f, x, u) -> uᵀ∇²f(x)u

Second directional derivative, by two nested `Dual` layers each carrying a single
partial. `f(x + ε₁u + ε₂u)` has `uᵀ∇²f(x)u` as its `ε₁ε₂` coefficient — the ε₁ε₂ cross
term is absent from the input by construction (the outer partial is `(u, 0)`), so the
`ε₁²=ε₂²=0` Taylor expansion leaves exactly that coefficient — and the quantity wanted is
the partial of the partial.

Tag ordering matters. ForwardDiff assigns tag precedence from a generated counter at first
COMPILATION, not at construction, so "T1 was built before T2" does not by itself guarantee
the ε-tags end up outside θ's tag once θ carries `Dual`s. If they inverted, this would
silently return **zero** — `partials` of a non-`Dual` is zero rather than an error — so the
result type is checked explicitly and an inversion raises. `verify.jl` V3 additionally
checks both directional identities against the matrix objects they replace.
"""
@inline function dir2(f::F, x::AbstractVector, u::AbstractVector) where {F}
    T1 = typeof(ForwardDiff.Tag(f, eltype(x)))
    inner = ForwardDiff.Dual{T1}.(x, u)
    slope = ForwardDiff.Dual{T1}.(u, zero(eltype(x)))
    T2 = typeof(ForwardDiff.Tag(dir2, eltype(inner)))
    xd = ForwardDiff.Dual{T2}.(inner, slope)
    y = f(xd)
    y isa ForwardDiff.Dual{T2} || error(
        "dir2: ForwardDiff tag nesting inverted (got $(typeof(y))). The ε-tags must sit " *
        "outside θ's tag; see docs/reverse_gradient_derivation.md §6.")
    return ForwardDiff.partials(ForwardDiff.partials(y, 1), 1)
end

# --------------------------------------------------------------------------
# The single scalar, in both forms
# --------------------------------------------------------------------------

"""
    directional_scalar(model, θ, subject, η̂, λ, U) -> Real

`sᵢ(θ)` in directional form. `U` holds the rows `uₘ` of `L⁻¹` where `H̃ᵢ = LLᵀ`.

`η̂`, `λ` and `U` are plain floats, frozen: that is what makes the result a *partial*
derivative in θ, with the implicit dependence of `η̂` on θ carried entirely by `λ`.

The `reg·I` part of `H̃` contributes `reg·Σₘ‖uₘ‖²`, a constant in θ, and is omitted
because it differentiates to zero.
"""
function directional_scalar(model, θ, subject, η̂::AbstractVector{Float64},
                            λ::AbstractVector{Float64}, U::AbstractMatrix{Float64})
    f = z -> negative_log_joint_density(model, z, θ, subject)
    s = f(η̂)
    @inbounds for m in axes(U, 1)
        s += 0.5 * dir2(f, η̂, view(U, m, :))
    end
    return s - dir1(f, η̂, λ)
end

"""
    matrix_scalar(model, θ, subject, η̂, λ, W) -> Real

`sᵢ(θ)` in matrix form: `h + ⟨W, ∇²_ηη h⟩ − (∇_η h)ᵀλ` with `W = ½H̃⁻¹` frozen.

Mathematically identical to [`directional_scalar`](@ref) and equally O(1) in `p`, but it
asks the reverse engine to differentiate `ForwardDiff.hessian` and `ForwardDiff.gradient`
rather than plain nested scalars. Kept as a fallback for engines that reject one form and
accept the other.
"""
function matrix_scalar(model, θ, subject, η̂::AbstractVector{Float64},
                       λ::AbstractVector{Float64}, W::AbstractMatrix{Float64})
    f = z -> negative_log_joint_density(model, z, θ, subject)
    Hv = ForwardDiff.hessian(f, η̂)
    gv = ForwardDiff.gradient(f, η̂)
    return f(η̂) + sum(W .* Hv) - dot(gv, λ)
end

"""
    reverse_scalar(form, model, θ, subject, η̂, λ, M) -> Real

Dispatch on `form`. `M` is `U` (rows of `L⁻¹`) for `:directional`, `W = ½H̃⁻¹` for
`:matrix`.
"""
@inline reverse_scalar(::Val{:directional}, model, θ, subject, η̂, λ, M) =
    directional_scalar(model, θ, subject, η̂, λ, M)
@inline reverse_scalar(::Val{:matrix}, model, θ, subject, η̂, λ, M) =
    matrix_scalar(model, θ, subject, η̂, λ, M)

# --------------------------------------------------------------------------
# The θ-sweep. One method per engine; extra engines are added by including
# their backend file, which is why none of them is a dependency here.
# --------------------------------------------------------------------------

"""
    reverse_sweep(::Val{backend}, s, θ) -> Vector{Float64}

`∇θ s(θ)` for a scalar-valued `s`. Add a backend by adding a method.
"""
function reverse_sweep end

# Control arm: the same scalar, p forward tangents. NOT a reverse method.
reverse_sweep(::Val{:forwarddiff}, s::F, θ::Vector{Float64}) where {F} =
    ForwardDiff.gradient(s, θ)

"""
    backend_available(b) -> Bool

Whether a `reverse_sweep` method exists for backend `b`. Loading the backend file is what
makes this true; see `src/backends/backends.jl`.
"""
backend_available(b::Symbol) =
    hasmethod(reverse_sweep, Tuple{Val{b},Function,Vector{Float64}})

function _sweep(backend::Symbol, s, θ)
    return reverse_sweep(Val(backend), s, θ)
end

# --------------------------------------------------------------------------
# Shared per-subject quantities
# --------------------------------------------------------------------------

"""
    subject_shared(model, θ, subject, cfg; η_start) -> NamedTuple

The mode solve and everything downstream of it that is common to all gradient
constructions: `η̂`, `H`, `H̃`, `b`, `λ`, `L`, and the frozen matrix each `form` needs.

Computed by the same formulas `subject_adjoint` uses, so no construction can gain or lose
accuracy here and any difference between them is attributable to the θ-derivative alone.
"""
function subject_shared(model, θ::Vector{Float64}, subject, cfg::LaplaceConfig,
                        form::Symbol; η_start = nothing)
    q = n_random_effects(model)
    η_start = resolve_start(model, θ, subject, η_start, cfg)
    η0 = η_start === nothing ? zeros(Float64, q) : collect(Float64, η_start)
    sol = newton_solve(model, θ, subject, η0, cfg)

    η̂ = extract_value(sol.η)
    H = sol.H                       # mode-equation Jacobian: coefficient of the solve
    H̃ = sol.Htilde                  # objective curvature: logdet, b, and W

    h_of_η(z, t) = negative_log_joint_density(model, z, t, subject)
    b = 0.5 .* ForwardDiff.gradient(
        ηv -> logdet_pd(htilde(ForwardDiff.hessian(z -> h_of_η(z, θ), ηv), cfg)), η̂)
    λ = solve_pd(H, b)

    # `check = false` keeps the framework's definiteness contract: a non-PD H̃ at the mode
    # means η̂ is a saddle, the Laplace expression is invalid there, and NaN is the correct
    # answer and must propagate rather than be papered over.
    C = cholesky(Symmetric(H̃); check = false)
    Iq = Matrix{Float64}(I, q, q)
    M = if !issuccess(C)
        fill(NaN, q, q)
    elseif form === :directional
        Matrix(C.L \ Iq)                       # rows are the uₘ
    else
        0.5 .* (C \ Iq)                        # W = ½H̃⁻¹
    end

    L = sol.value + logdet_pd(H̃) / 2 - (q / 2) * log(2π)
    return (eta = η̂, H = H, Htilde = H̃, b = b, lambda = λ, M = M, L = L,
            converged = sol.converged, iters = sol.iters, gnorm = sol.gnorm)
end

# --------------------------------------------------------------------------
# Subject and population gradients
# --------------------------------------------------------------------------

"""
    subject_reverse(model, θ, subject, cfg, rcfg; η_start=nothing) -> NamedTuple

One subject's Eq. (4.4) gradient by a single sweep of one scalar.

`θ` is a plain `Vector{Float64}`: this routine *is* the derivative, so nothing
differentiates through it.
"""
function subject_reverse(model, θ::Vector{Float64}, subject,
                         cfg::LaplaceConfig, rcfg::ReverseConfig; η_start = nothing)
    cfg.inner_deriv === :ad || error(
        "subject_reverse requires cfg.inner_deriv = :ad; got :$(cfg.inner_deriv). " *
        "Like the adjoint path its terms are AD-derived, so an FD inner Hessian in " *
        "the objective would make this the gradient of a different function.")

    sh = subject_shared(model, θ, subject, cfg, rcfg.form; η_start = η_start)
    fv = Val(rcfg.form)
    s = θv -> reverse_scalar(fv, model, θv, subject, sh.eta, sh.lambda, sh.M)
    g = _sweep(rcfg.backend, s, θ)

    return (grad = g, b = sh.b, lambda = sh.lambda, L = sh.L, eta = sh.eta,
            Htilde = sh.Htilde, converged = sh.converged, iters = sh.iters,
            gnorm = sh.gnorm)
end

"""
    reverse_gradient_parts(model, θ, pop, cfg, rcfg; eta_starts=nothing) -> NamedTuple

Population reverse gradient with its diagnostic parts. Mirror of
`adjoint_gradient_parts`; a debugging and benchmark entry point, not a separate
estimator.

The reduction is a serial `sum` in index order, so the result is bitwise independent of
the thread schedule and of the thread count, exactly as in the adjoint and forward paths.
"""
function reverse_gradient_parts(model, θ::Vector{Float64}, pop::PopulationData,
                                cfg::LaplaceConfig, rcfg::ReverseConfig;
                                eta_starts = nothing)
    # Both checks belong here, before the parallel region: raised inside it, the same
    # error would reach the caller wrapped in a TaskFailedException.
    cfg.inner_deriv === :ad || error(
        "the reverse gradient requires cfg.inner_deriv = :ad; see `subject_reverse`.")
    backend_available(rcfg.backend) || error(
        "no reverse_sweep method for backend :$(rcfg.backend). Run " *
        "`include(\"src/backends/backends.jl\"); load_backends!()` first, or choose another " *
        "backend. Available now: $(filter(backend_available, (:forwarddiff, CANDIDATE_BACKENDS...))).")
    rcfg.form in FORMS || error("unknown form :$(rcfg.form); expected one of $FORMS.")

    n = pop.n_subjects
    p = length(θ)
    G = zeros(Float64, p, n)
    B = Vector{Vector{Float64}}(undef, n)
    Λ = Vector{Vector{Float64}}(undef, n)
    etas = Vector{Vector{Float64}}(undef, n)
    Ls = zeros(Float64, n)
    conv = Vector{Bool}(undef, n)
    iters = Vector{Int}(undef, n)
    gnorms = zeros(Float64, n)
    subs = pop.subjects

    foreach_subject(n, cfg.threaded) do i
        st = eta_starts === nothing ? nothing : eta_starts[i]
        r = subject_reverse(model, θ, subs[i], cfg, rcfg; η_start = st)
        @views G[:, i] .= r.grad
        B[i] = r.b
        Λ[i] = r.lambda
        etas[i] = r.eta
        Ls[i] = r.L
        conv[i] = r.converged
        iters[i] = r.iters
        gnorms[i] = r.gnorm
    end

    grad = [sum(@view G[j, :]) for j in 1:p]
    return (grad = grad, b = B, lambda = Λ, L = sum(Ls), Ls = Ls, grads = G,
            etas = etas, converged = conv, iters = iters, gnorms = gnorms,
            n_nonconverged = count(!, conv), worst_gnorm = maximum(gnorms))
end

"""
    reverse_gradient(model, pop, cfg, rcfg) -> grad!(G, θ)

Factory in the same in-place `grad!(G, θ)` form as `adjoint_gradient` and
`forward_gradient`, so it drops into `fit_nlme` and the benchmark harnesses unchanged.
"""
function reverse_gradient(model, pop::PopulationData, cfg::LaplaceConfig,
                          rcfg::ReverseConfig = ReverseConfig())
    if cfg.warm_start
        cache = [zeros(Float64, n_random_effects(model)) for _ in 1:pop.n_subjects]
        return function (G, θ)
            r = reverse_gradient_parts(model, θ, pop, cfg, rcfg; eta_starts = cache)
            for i in 1:pop.n_subjects
                cache[i] .= r.etas[i]
            end
            G .= r.grad
            return G
        end
    end
    return function (G, θ)
        G .= reverse_gradient_parts(model, θ, pop, cfg, rcfg).grad
        return G
    end
end

# --------------------------------------------------------------------------
# Backend selection by measurement, not by assumption
# --------------------------------------------------------------------------

"""
    selftest_backend(model, θ, pop, cfg, backend, form; atol) -> NamedTuple

Run one population gradient with `(backend, form)` and compare it against
`adjoint_gradient_parts`, which is the framework's already-validated construction.

Returns `(ok, err, seconds, message)`. A backend that throws is reported, not rethrown:
the point of this routine is to survey which engines work on *this* model, on *this*
machine, at *these* versions, and an engine that cannot differentiate the model is a fact
to record rather than a reason to abort.

This is deliberately a run against a known-good gradient rather than a smoke test. An AD
engine that silently returns a wrong gradient — a real failure mode when nested `Dual`
tags are involved, because `partials` of a non-`Dual` is zero rather than an error — would
pass any test that only checked it did not throw.
"""
function selftest_backend(model, θ::Vector{Float64}, pop::PopulationData,
                          cfg::LaplaceConfig, backend::Symbol, form::Symbol;
                          atol::Float64 = 1e-6, ref = nothing)
    backend_available(backend) ||
        return (ok = false, err = NaN, seconds = NaN, message = "backend not loaded")
    ref === nothing && (ref = adjoint_gradient_parts(model, θ, pop, cfg).grad)
    scale = max(maximum(abs, ref), 1.0)
    rc = ReverseConfig(backend = backend, form = form)
    try
        # Warm-up first, THEN time.  A cold call includes JIT and, for Enzyme or Mooncake,
        # rule construction — which is exactly the cost these engines pay once and the
        # tape-based one never pays.  Selecting on a cold timing would systematically
        # pick the slowest engine.
        reverse_gradient_parts(model, θ, pop, cfg, rc)
        t0 = time_ns()
        g = reverse_gradient_parts(model, θ, pop, cfg, rc).grad
        el = (time_ns() - t0) / 1e9
        all(isfinite, g) ||
            return (ok = false, err = NaN, seconds = el, message = "non-finite gradient")
        err = maximum(abs, g .- ref) / scale
        return (ok = err < atol, err = err, seconds = el,
                message = err < atol ? "ok" : "disagrees with the adjoint gradient")
    catch e
        return (ok = false, err = NaN, seconds = NaN,
                # `first(s, n)` counts CHARACTERS.  Byte-slicing a showerror dump throws
                # StringIndexError the moment it lands mid-character, and these messages
                # routinely contain ∂, → and non-ASCII paths.
                message = first(sprint(showerror, e; context = :limit => true), 300))
    end
end

"""
    select_backend(model, θ, pop, cfg; backends, forms, atol, verbose) -> NamedTuple

Survey every `(backend, form)` pair, return the fastest one that reproduces the adjoint
gradient, and the full table.

Returns `(rcfg, table)` where `rcfg` is a `ReverseConfig` and `table` is a vector of
`(backend, form, ok, err, seconds, message)`. If nothing passes, `rcfg` falls back to the
`:forwarddiff` control arm and `ok_any` is false — in which case every arm of the study
collapses onto the same curve, which is a correct null result and must be reported as
one, not as a speed-up.
"""
function select_backend(model, θ::Vector{Float64}, pop::PopulationData,
                        cfg::LaplaceConfig;
                        backends = CANDIDATE_BACKENDS, forms = FORMS,
                        atol::Float64 = 1e-6, verbose::Bool = true)
    table = NamedTuple[]
    # One adjoint gradient for the whole survey, not one per pair: at N=128, p=1321 that
    # would be six full population gradients spent before anything is printed.
    ref = adjoint_gradient_parts(model, θ, pop, cfg).grad
    for b in backends, f in forms
        r = selftest_backend(model, θ, pop, cfg, b, f; atol = atol, ref = ref)
        push!(table, (backend = b, form = f, ok = r.ok, err = r.err,
                      seconds = r.seconds, message = r.message))
        verbose && println(rpad("  $b / $f", 30),
                           r.ok ? "OK   " : "FAIL ",
                           "rel.err=", r.err, "  ", r.ok ? "" : r.message)
    end
    passing = filter(t -> t.ok, table)
    if isempty(passing)
        verbose && @warn "no reverse engine reproduced the adjoint gradient on this " *
                         "model; falling back to the :forwarddiff CONTROL ARM. Every " *
                         "arm will now have the same cost model and the study cannot " *
                         "show a reverse effect. Report this as a null result."
        return (rcfg = ReverseConfig(backend = :forwarddiff, form = :directional),
                table = table, ok_any = false)
    end
    best = passing[argmin([t.seconds for t in passing])]
    verbose && println("  selected: $(best.backend) / $(best.form)")
    return (rcfg = ReverseConfig(backend = best.backend, form = best.form),
            table = table, ok_any = true)
end
