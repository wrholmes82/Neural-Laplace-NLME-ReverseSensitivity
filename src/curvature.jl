# curvature.jl — the FOCE curvature arm (Phase 0).
#
# `reverse.jl` computes the exact-Laplace population gradient as one reverse sweep of a single
# scalar sᵢ(θ).  This file adds the Gauss-Newton (FOCE) curvature as a drop-in alternative to
# the exact Hessian in the curvature slots, keeping the exact Hessian everywhere the mode
# equation lives.  The :laplace path is delegated to `reverse.jl` verbatim, so it stays
# bit-identical to the already-validated construction; only :foce is new here.
#
# Self-consistency (PLAN_adjoint_foce.md §3, the load-bearing table):
#
#   curvature slot            matrix     why
#   objective log-determinant   G̃        it IS the FOCE curvature term
#   adjoint RHS  b = ½∇_η logdet G̃        η-gradient of that same log-determinant
#   frozen weight W = ½G̃⁻¹      G̃        same log-determinant, θ-derivative
#   λ-solve coefficient         H (exact) Jacobian of the mode equation ∇_η h = 0 — FOCE
#                                          does NOT change the mode, so this stays exact
#   correction (∇²_ηθ h)ᵀλ      h (exact) mode sensitivity is −H⁻¹∇²_ηθ h, from the exact mode

# --------------------------------------------------------------------------
# Interface extension: the conditional moments FOCE needs
# --------------------------------------------------------------------------

"""
    conditional_moments(model, η, θ, subject) -> (f, R, Ω)

The decomposition the Gauss-Newton curvature requires and that the single scalar `h` hides:

  * `f` — the structural predictions, one per observation (length `n_obs`);
  * `R` — the residual variances, one per observation (the diagonal of the residual
    covariance).  Phase 1 uses additive-on-log error, so `R = σ²·ones` is **independent of
    `η`** and the FOCE arm is plain FOCE (no interaction).  A model whose `R` depends on `η`
    (proportional/combined error) makes this FOCE-I, handled by the same code with no change.
  * `Ω` — the random-effect variances (length `q`, the diagonal of `Ω`).

Only models that implement this method can be fitted with `CurvatureConfig(:foce)`; models
implementing only `negative_log_joint_density` are Laplace-only, exactly as before.

The argument order matches `negative_log_joint_density`: `(model, η, θ, subject)`.
Must be non-mutating and ForwardDiff-differentiable in both `η` (for `J = ∂f/∂η`) and `θ`.
"""
function conditional_moments end

"""
    CurvatureConfig(; curvature = :laplace, freeze_J = false)

  * `curvature ∈ {:laplace, :foce}` — which matrix plays the curvature role.  `:laplace`
    delegates to `NLMEReverse` unchanged.  `:foce` uses `G = JᵀR⁻¹J + Ω⁻¹`.
  * `freeze_J` — FOCE only.  `false` (default, and the mathematically consistent choice)
    differentiates `G` through `η` when forming `b = ½∇_η logdet G̃`, i.e. carries `∂J/∂η`.
    `true` freezes `J` at the mode when forming `b` — a *different estimator*.  §3.3 of the
    plan requires implementing the consistent version as default and reporting which was used.
    (Note: with η-independent `R`,`Ω`, `freeze_J = true` makes `G̃` η-independent, so `b ≡ 0`
    and `λ ≡ 0` — a legitimate but distinct estimator.)
  * `drop_implicit` — the **dropped-implicit-term comparator**, and it is a comparator, never
    a fallback. `true` zeroes the adjoint direction `λ` after the mode solve, so the gradient
    omits `−(∇²_ηθ h)ᵀλ` entirely: the published "FOCE" gradient for deep compartment models,
    which is the gradient of no objective. Gated here rather than reimplemented per study, so
    there is exactly one of it.

    It is **not** the same thing as `freeze_J`, and conflating them is a live trap. `freeze_J`
    makes `λ ≡ 0` only when `R` and `Ω` are η-independent; a model with proportional residual
    error has `R` depending on `η` through the prediction, so `G̃` still varies with `η`, `b ≠ 0`
    and `λ ≠ 0`. On such a model `freeze_J` is a *third*, different estimator. Use
    `drop_implicit` when the intent is to drop the implicit term.
"""
Base.@kwdef struct CurvatureConfig
    curvature::Symbol = :laplace
    freeze_J::Bool = false
    drop_implicit::Bool = false
end

# --------------------------------------------------------------------------
# dir1_vec — first directional derivative of a vector-valued map
# --------------------------------------------------------------------------

"""
    dir1_vec(f, x, v) -> J·v          where J = ∂f/∂x, f vector-valued

The vector analogue of `NLMEReverse.dir1`: one `Dual` layer carrying the single partial `v`,
so `f(x + εv)` has `J·v` as its `ε`-partial — one evaluation at Dual depth one, against
`length(x)` for a full Jacobian.  This is what makes `Juₘ` in the FOCE directional scalar a
plain vector the reverse sweep can carry, with no Jacobian materialised inside the sweep.

Same tag-ordering discipline as `dir1`: the ε-tag must sit outside θ's tag when θ carries
`Dual`s (or a reverse engine's tracked type).  It does, because the tag is created here,
inside the sweep's already-θ-typed closure, so its global-counter id is larger (= outer).
An inversion would make `eltype(y)` not a `Dual{T}` and is caught rather than returning zero.
"""
@inline function dir1_vec(f::F, x::AbstractVector, v::AbstractVector) where {F}
    T = typeof(ForwardDiff.Tag(f, eltype(x)))
    xd = ForwardDiff.Dual{T}.(x, v)
    y = f(xd)
    eltype(y) <: ForwardDiff.Dual{T} || error(
        "dir1_vec: ForwardDiff tag nesting inverted (got eltype $(eltype(y))). The ε-tag " *
        "must sit outside θ's tag; see NLMEReverse.dir1 and docs/01.")
    return ForwardDiff.partials.(y, 1)
end

# --------------------------------------------------------------------------
# FOCE curvature assembly
# --------------------------------------------------------------------------

"""
    foce_curvature(model, η, θ, subject) -> G

`G(η,θ) = J(η,θ)ᵀ diag(1/R) J(η,θ) + diag(1/Ω)`, the Gauss-Newton / expected-information
curvature, positive definite by construction (a sum of PSD outer products plus `Ω⁻¹ ≻ 0`).
`J = ∂f/∂η` is one ForwardDiff Jacobian pass.  Used to assemble `G̃`, its Cholesky factor,
and the objective's log-determinant.  Not called inside the reverse sweep (the directional
scalar builds the same quantity from `dir1_vec` instead).
"""
function foce_curvature(model, η::AbstractVector, θ, subject)
    J = ForwardDiff.jacobian(z -> conditional_moments(model, z, θ, subject).f, η)
    cm = conditional_moments(model, η, θ, subject)
    return transpose(J) * (Diagonal(1 ./ cm.R) * J) + Diagonal(1 ./ cm.Ω)
end

# G̃(η) = G(η) + reg·I as a function of η, honouring freeze_J.  Used only to form `b` (an
# η-gradient of its log-determinant) and its value at the mode — always at plain-float θ,
# so the nested ForwardDiff.jacobian-inside-gradient here is depth-2 and never meets a
# reverse engine.
function _foce_gtilde_of_η(model, θ, subject, reg::Float64, freeze_J::Bool, η̂)
    if !freeze_J
        return ηv -> foce_curvature(model, ηv, θ, subject) + reg * I
    end
    J0 = ForwardDiff.jacobian(z -> conditional_moments(model, z, θ, subject).f, η̂)
    return function (ηv)
        cm = conditional_moments(model, ηv, θ, subject)
        return transpose(J0) * (Diagonal(1 ./ cm.R) * J0) + Diagonal(1 ./ cm.Ω) + reg * I
    end
end

# --------------------------------------------------------------------------
# The FOCE single scalar, in both forms
# --------------------------------------------------------------------------

"""
    foce_directional_scalar(model, θ, subject, η̂, λ, U) -> Real

`sᵢ(θ)` for the FOCE curvature, matrix-free.  With `G̃ = LLᵀ` and `uₘ` the m-th row of `L⁻¹`,

    ⟨½G̃⁻¹, G̃(θ)⟩ = ½ Σₘ uₘᵀ G(η̂,θ) uₘ
                  = ½ Σₘ [ (J(η̂,θ)uₘ)ᵀ R⁻¹ (J(η̂,θ)uₘ) + uₘᵀ Ω⁻¹ uₘ ] ,

and `J(η̂,θ)uₘ = dir1_vec(f(·,θ), η̂, uₘ)` is a first directional derivative — one JVP each,
against the second directional derivatives the Laplace form needs.  `η̂`, `λ`, `U` are frozen
floats.  The `reg·I` part contributes `reg·Σ‖uₘ‖²`, constant in θ, and is omitted.
"""
function foce_directional_scalar(model, θ, subject, η̂::AbstractVector{Float64},
                                 λ::AbstractVector{Float64}, U::AbstractMatrix{Float64})
    h = negative_log_joint_density(model, η̂, θ, subject)
    cm = conditional_moments(model, η̂, θ, subject)     # R, Ω at (η̂, θ); θ live, η̂ frozen
    invR = 1 ./ cm.R
    invΩ = 1 ./ cm.Ω
    ffun = z -> conditional_moments(model, z, θ, subject).f
    s = h
    @inbounds for m in axes(U, 1)
        u = view(U, m, :)
        Ju = dir1_vec(ffun, η̂, u)
        s += 0.5 * (sum(invR .* Ju .* Ju) + sum(invΩ .* u .* u))
    end
    hfun = z -> negative_log_joint_density(model, z, θ, subject)
    return s - dir1(hfun, η̂, λ)
end

"""
    foce_matrix_scalar(model, θ, subject, η̂, λ, W) -> Real

`sᵢ(θ) = h + ⟨W, G(η̂,θ)⟩ − (∇_η h)ᵀλ` with `W = ½G̃⁻¹` frozen.  Assembles `G` explicitly and
uses `ForwardDiff` for `J` and `∇_η h`, so a reverse sweep over it nests over ForwardDiff —
kept as the independent check (F4) and a fallback, mirroring `NLMEReverse.matrix_scalar`.
"""
function foce_matrix_scalar(model, θ, subject, η̂::AbstractVector{Float64},
                            λ::AbstractVector{Float64}, W::AbstractMatrix{Float64})
    h = negative_log_joint_density(model, η̂, θ, subject)
    G = foce_curvature(model, η̂, θ, subject)
    gh = ForwardDiff.gradient(z -> negative_log_joint_density(model, z, θ, subject), η̂)
    return h + sum(W .* G) - dot(gh, λ)
end

@inline foce_scalar(::Val{:directional}, model, θ, subject, η̂, λ, M) =
    foce_directional_scalar(model, θ, subject, η̂, λ, M)
@inline foce_scalar(::Val{:matrix}, model, θ, subject, η̂, λ, M) =
    foce_matrix_scalar(model, θ, subject, η̂, λ, M)

# PD-safe implicit solve for the FOCE adjoint direction.  H is the EXACT mode-equation Jacobian
# (the self-consistency table keeps it exact, not the FOCE curvature).  At a genuine mode H ≻ 0,
# so this is the exact solve and introduces NO bias.  When the outer line search transiently
# probes θ where a subject's η̂ is a saddle (H indefinite), `solve_pd` would return NaN and poison
# the whole population gradient (→ a gradient rescue, which stalls the outer optimiser).  Instead
# we lift H to positive definite (a Levenberg shift to a bounded condition number, exactly as the
# inner search does) and return a FINITE implicit correction there.  This fires ONLY at non-mode
# points; the bias is O(shift) and vanishes wherever the fit actually settles.
function _safe_lambda_solve(H::AbstractMatrix, b::AbstractVector, cfg::LaplaceConfig)
    C = cholesky(Symmetric(H); check = false)
    issuccess(C) && return C \ b
    Hv = Float64.(H)
    ev = eigvals(Symmetric(Hv))
    β = cfg.inner_pd_floor * max(ev[end], 1.0)
    return (Symmetric(Hv) + (β - ev[1]) * I) \ b
end

# --------------------------------------------------------------------------
# Shared per-subject quantities (FOCE)
# --------------------------------------------------------------------------

"""
    subject_curvature(model, θ, subject, cfg, rcfg, ccfg; η_start) -> NamedTuple

Mode solve and the frozen quantities the chosen curvature needs.  For `:laplace` this defers
to `subject_shared`; for `:foce` it assembles `G`, `G̃`, `b`, `λ`, the frozen
`M`, and the FOCE objective value.  `H` is always the exact mode-equation Jacobian.
"""
function subject_curvature(model, θ::Vector{Float64}, subject, cfg::LaplaceConfig,
                           form::Symbol, ccfg::CurvatureConfig; η_start = nothing)
    if ccfg.curvature === :laplace
        return subject_shared(model, θ, subject, cfg, form; η_start = η_start)
    end
    q = n_random_effects(model)
    η_start = resolve_start(model, θ, subject, η_start, cfg)
    η0 = η_start === nothing ? zeros(Float64, q) : collect(Float64, η_start)
    sol = newton_solve(model, θ, subject, η0, cfg)

    η̂ = extract_value(sol.η)
    H = sol.H                                    # exact Hessian: the λ-solve coefficient

    gt_of_η = _foce_gtilde_of_η(model, θ, subject, cfg.reg, ccfg.freeze_J, η̂)
    G̃ = gt_of_η(η̂)                               # FOCE curvature at the mode (+ reg·I)
    b = 0.5 .* ForwardDiff.gradient(ηv -> logdet_pd(gt_of_η(ηv)), η̂)
    # The comparator: drop the implicit term by zeroing the adjoint direction. Done AFTER the
    # solve, not by skipping it, so `b` and `H` are still reported and the two arms differ in
    # exactly one place.
    λ = ccfg.drop_implicit ? zeros(Float64, q) : _safe_lambda_solve(H, b, cfg)

    C = cholesky(Symmetric(G̃); check = false)    # NaN propagates if G̃ not PD — same contract
    Iq = Matrix{Float64}(I, q, q)
    M = if !issuccess(C)
        fill(NaN, q, q)
    elseif form === :directional
        Matrix(C.L \ Iq)                         # rows are the uₘ
    else
        0.5 .* (C \ Iq)                          # W = ½G̃⁻¹
    end

    L = sol.value + logdet_pd(G̃) / 2 - (q / 2) * log(2π)
    return (eta = η̂, H = H, Htilde = G̃, b = b, lambda = λ, M = M, L = L,
            converged = sol.converged, iters = sol.iters, gnorm = sol.gnorm)
end

# --------------------------------------------------------------------------
# Subject and population gradients (curvature-aware)
# --------------------------------------------------------------------------

"""
    subject_curvature_gradient(model, θ, subject, cfg, rcfg, ccfg; η_start) -> NamedTuple

One subject's gradient by a single sweep of the curvature-appropriate scalar.
"""
function subject_curvature_gradient(model, θ::Vector{Float64}, subject, cfg::LaplaceConfig,
                                    rcfg::ReverseConfig, ccfg::CurvatureConfig; η_start = nothing)
    cfg.inner_deriv === :ad || error("curvature gradient requires cfg.inner_deriv = :ad.")

    if ccfg.curvature === :laplace
        return subject_reverse(model, θ, subject, cfg, rcfg; η_start = η_start)
    end

    sh = subject_curvature(model, θ, subject, cfg, rcfg.form, ccfg; η_start = η_start)
    fv = Val(rcfg.form)
    s = θv -> foce_scalar(fv, model, θv, subject, sh.eta, sh.lambda, sh.M)
    g = reverse_sweep(Val(rcfg.backend), s, θ)
    return (grad = g, b = sh.b, lambda = sh.lambda, L = sh.L, eta = sh.eta,
            Htilde = sh.Htilde, converged = sh.converged, iters = sh.iters, gnorm = sh.gnorm)
end

"""
    curvature_gradient_parts(model, θ, pop, cfg, rcfg, ccfg; eta_starts) -> NamedTuple

Population gradient with diagnostics, for the chosen curvature.  Mirror of
`reverse_gradient_parts`: per-index buffers, serial `sum` in index order, so the
result is bitwise independent of the thread schedule.
"""
function curvature_gradient_parts(model, θ::Vector{Float64}, pop::PopulationData,
                                  cfg::LaplaceConfig, rcfg::ReverseConfig,
                                  ccfg::CurvatureConfig; eta_starts = nothing)
    if ccfg.curvature === :laplace
        return reverse_gradient_parts(model, θ, pop, cfg, rcfg;
                                                  eta_starts = eta_starts)
    end
    cfg.inner_deriv === :ad || error("curvature gradient requires cfg.inner_deriv = :ad.")
    backend_available(rcfg.backend) ||
        error("no reverse_sweep method for backend :$(rcfg.backend); load backends first.")
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
        r = subject_curvature_gradient(model, θ, subs[i], cfg, rcfg, ccfg; η_start = st)
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
    curvature_gradient(model, pop, cfg, rcfg, ccfg) -> grad!(G, θ)

In-place `grad!(G, θ)` factory, same signature as `NLMEReverse.reverse_gradient`, so it drops
into `fit_nlme(...; grad = curvature_gradient(...))`.
"""
function curvature_gradient(model, pop::PopulationData, cfg::LaplaceConfig,
                            rcfg::ReverseConfig = ReverseConfig(),
                            ccfg::CurvatureConfig = CurvatureConfig();
                            fixed_starts = nothing)
    if fixed_starts !== nothing
        # Every inner solve begins at these fixed modes (computed once, at the start θ): a good
        # basin so η̂ never wanders to overflow (no NaN gradient), with NO per-call cache update,
        # so the objective is history-independent and the outer line search is not confused —
        # the failure mode a rolling warm start hits.  See fit_joint.
        return function (G, θ)
            G .= curvature_gradient_parts(model, θ, pop, cfg, rcfg, ccfg;
                                          eta_starts = fixed_starts).grad
            return G
        end
    end
    if cfg.warm_start
        cache = [zeros(Float64, n_random_effects(model)) for _ in 1:pop.n_subjects]
        return function (G, θ)
            r = curvature_gradient_parts(model, θ, pop, cfg, rcfg, ccfg; eta_starts = cache)
            for i in 1:pop.n_subjects
                cache[i] .= r.etas[i]
            end
            G .= r.grad
            return G
        end
    end
    return function (G, θ)
        G .= curvature_gradient_parts(model, θ, pop, cfg, rcfg, ccfg).grad
        return G
    end
end

# --------------------------------------------------------------------------
# The FOCE objective — for the finite-difference cross-check (F2) and for fitting
# --------------------------------------------------------------------------

"""
    foce_subject_objective(model, θ, subject, cfg, ccfg; η_start) -> NamedTuple

One subject's FOCE (or Laplace) Laplace-type contribution
`Lᵢ = h(η̂,θ) + ½ logdet M̃(η̂,θ) − (q/2)log2π`, with `M = G` under `:foce` and `M = ∇²_ηη h`
under `:laplace`.  Type-generic in `θ` so it can be finite-differenced (Float64) or, if
wanted, AD-differentiated.
"""
function foce_subject_objective(model, θ::AbstractVector, subject, cfg::LaplaceConfig,
                                ccfg::CurvatureConfig; η_start = nothing)
    if ccfg.curvature === :laplace
        return subject_objective(model, θ, subject, cfg; η_start = η_start)
    end
    q = n_random_effects(model)
    T = promote_type(eltype(θ), Float64)
    η_start = resolve_start(model, θ, subject, η_start, cfg)
    η0 = η_start === nothing ? zeros(T, q) : convert(Vector{T}, η_start)
    sol = newton_solve(model, θ, subject, η0, cfg)
    G̃ = foce_curvature(model, sol.η, θ, subject) + cfg.reg * I
    L = sol.value + logdet_pd(G̃) / 2 - (q / 2) * log(2π)
    return (L = L, eta = sol.η, Htilde = G̃, value = sol.value,
            converged = sol.converged, iters = sol.iters, gnorm = sol.gnorm)
end

"""
    foce_objective(model, pop, cfg, ccfg) -> (θ -> L)

Population FOCE (or Laplace) objective `L(θ) = Σᵢ Lᵢ(θ)`.  Serial reduction in index order.
"""
function foce_objective(model, pop::PopulationData, cfg::LaplaceConfig, ccfg::CurvatureConfig;
                        fixed_starts = nothing)
    if ccfg.curvature === :laplace
        fixed_starts === nothing && return laplace_objective(model, pop, cfg)
        return θ -> population_objective(model, θ, pop, cfg;
                                         eta_starts = fixed_starts).L
    end
    if fixed_starts !== nothing
        # History-independent fixed warm start, matching curvature_gradient(fixed_starts=...).
        return function (θ)
            n = pop.n_subjects
            T = promote_type(eltype(θ), Float64)
            Ls = zeros(T, n)
            subs = pop.subjects
            foreach_subject(n, cfg.threaded) do i
                Ls[i] = foce_subject_objective(model, θ, subs[i], cfg, ccfg;
                                               η_start = fixed_starts[i]).L
            end
            return sum(Ls)
        end
    end
    if !cfg.warm_start
        return function (θ)
            n = pop.n_subjects
            T = promote_type(eltype(θ), Float64)
            Ls = zeros(T, n)
            subs = pop.subjects
            foreach_subject(n, cfg.threaded) do i
                Ls[i] = foce_subject_objective(model, θ, subs[i], cfg, ccfg).L
            end
            return sum(Ls)
        end
    end
    # Warm start: each inner solve begins at the previous iterate's mode, so η̂ tracks θ smoothly
    # and never wanders to absurd values where the structural solve overflows (the NaN-gradient
    # source in an un-warm-started fit).  Mirrors `laplace_objective`; the gradient factory keeps
    # its own matching cache, so both describe the same warm-started objective.
    cache = [zeros(Float64, n_random_effects(model)) for _ in 1:pop.n_subjects]
    return function (θ)
        n = pop.n_subjects
        T = promote_type(eltype(θ), Float64)
        Ls = zeros(T, n)
        etas = Vector{Vector{T}}(undef, n)
        subs = pop.subjects
        foreach_subject(n, cfg.threaded) do i
            r = foce_subject_objective(model, θ, subs[i], cfg, ccfg; η_start = cache[i])
            Ls[i] = r.L
            etas[i] = r.eta
        end
        for i in 1:n
            cache[i] .= extract_value(etas[i])
        end
        return sum(Ls)
    end
end
