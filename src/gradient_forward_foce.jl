# foce_forward.jl — FORWARD-SENSITIVITY gradient of the **FOCE** objective.
#
# The incumbent's construction (Almquist et al. 2015; NONMEM FAST; Pumas FOCE) applied to the
# FOCE curvature.  It exists because the repository had no forward-mode FOCE arm at all: every
# O(p) comparator in `src/` (`forward_gradient`, `adjoint_gradient`) differentiates the
# **exact-Laplace** objective, while the O(1) headline arm (`curvature_gradient` with
# `CurvatureConfig(:foce)`) differentiates the **FOCE** objective.  Comparing those two is not
# like-for-like.  This file supplies the missing like-for-like O(p) arm.
#
# Per subject, at the converged mode η̂ detached to plain floats:
#
#   G̃   = J(η̂,θ)ᵀ R⁻¹ J(η̂,θ) + Ω⁻¹ + reg·I                the FOCE curvature (`foce_curvature`)
#   b   = ½ ∇_η logdet G̃(η,θ) |_{η̂}                        same b the reverse FOCE arm uses
#   gφ  = ∂θ [ h(η̂,θ) + ½ logdet G̃(η̂,θ) ]   (η̂ frozen)     explicit part, p tangents
#   B   = ∂θ [ ∇_η h(η̂,θ) ] = ∇²_ηθ h                       q×p — the object that costs O(p)
#   S   = dη̂/dθ = −H⁻¹ B                                    q×p, ONE factorisation, p RHS
#   ∇θL = gφ + Sᵀ b
#
# **The self-consistency table (`curvature.jl` header, docs) is
# obeyed exactly**, and getting it wrong here is the FOCE analogue of the H-vs-H̃ bug:
#
#   * G̃ (the FOCE curvature) in the objective's log-determinant, in `b`, and in `gφ`;
#   * **H = ∇²_ηη h, the EXACT Hessian, as the coefficient of the sensitivity solve** — it is
#     the Jacobian of the mode equation ∇_η h = 0, and FOCE does not change the mode.  Using
#     G̃ there would give the gradient of neither objective.
#   * B built from the exact h, for the same reason.
#
# Relation to the adjoint orientation: Sᵀb = −(H⁻¹B)ᵀb = −Bᵀ(H⁻¹b) = −Bᵀλ, so this is
# algebraically the same gradient the reverse FOCE arm computes.  The only difference is the
# orientation — p right-hand sides through one q×q factor, and a q×p matrix materialised —
# which is exactly the cost axis under study.
#
# Part of the module: requires a model implementing `conditional_moments`.

# Multi-right-hand-side PD-safe solve.  Mirrors ``_safe_lambda_solve`` (the FOCE
# λ-solve) exactly — including its Levenberg lift at non-mode points where the exact H is
# indefinite — but factors ONCE for the whole q×p system rather than p times, so the forward
# arm is not unfairly charged p redundant factorisations.  Matching the lift matters for the
# validation: at a genuine mode H ≻ 0 and both are the exact solve, but on the line-search
# excursions the two arms must lift identically or they are different estimators.
function _foce_solve_multi(H::AbstractMatrix, Bm::AbstractMatrix, cfg::LaplaceConfig)
    C = cholesky(Symmetric(H); check = false)
    issuccess(C) && return C \ Bm
    Hv = Float64.(H)
    ev = eigvals(Symmetric(Hv))
    β = cfg.inner_pd_floor * max(ev[end], 1.0)
    return (Symmetric(Hv) + (β - ev[1]) * I) \ Bm
end

"""
    subject_foce_forward(model, θ, subject, cfg; η_start=nothing) -> NamedTuple

One subject's FOCE gradient in the forward-sensitivity (Almquist) orientation, plus its parts
`(grad, implicit, b, S, B, L, eta, Gtilde, converged, iters, gnorm)`.

`b`, `gφ`, the mode `η̂` and the curvature `G̃` are computed by the same formulas
``subject_curvature`` uses under `CurvatureConfig(:foce)`, so any gradient
difference is attributable to the direct-vs-adjoint contraction alone.
"""
function subject_foce_forward(model, θ::AbstractVector{T}, subject, cfg::LaplaceConfig;
                              η_start = nothing) where {T<:AbstractFloat}
    cfg.inner_deriv === :ad || error(
        "subject_foce_forward requires cfg.inner_deriv = :ad; got :$(cfg.inner_deriv).")

    q = n_random_effects(model)
    η0 = η_start === nothing ? zeros(T, q) : collect(T, η_start)
    sol = newton_solve(model, θ, subject, η0, cfg)

    # Detach the mode: the ∂θ terms are partial derivatives; the implicit dependence of η̂ on
    # θ enters only through S.
    η̂ = extract_value(sol.η)
    H = sol.H                                    # exact mode-equation Jacobian — the solve's coefficient
    reg = cfg.reg

    # G̃(η) at frozen θ, and b = ½∇_η logdet G̃.  `freeze_J = false` (the consistent choice and
    # the default of CurvatureConfig), so ∂J/∂η is carried.
    gt_of_η = ηv -> foce_curvature(model, ηv, θ, subject) + reg * I
    G̃ = gt_of_η(η̂)
    b = 0.5 .* ForwardDiff.gradient(ηv -> logdet_pd(gt_of_η(ηv)), η̂)

    h_of(z, t) = negative_log_joint_density(model, z, t, subject)

    # gφ = ∂θ[ h(η̂,θ) + ½ logdet G̃(η̂,θ) ], η̂ frozen.  p tangents; the inner J-Jacobian is
    # a second, later-created ForwardDiff tag, hence outer, as in `foce_matrix_scalar`.
    gφ = ForwardDiff.gradient(θv -> begin
            hv = h_of(η̂, θv)
            Gv = foce_curvature(model, η̂, θv, subject) + reg * I
            hv + 0.5 * logdet_pd(Gv)
        end, θ)

    # B = ∇²_ηθ h — the full q×p mixed-derivative matrix.  The reverse arm never forms it.
    B = ForwardDiff.jacobian(θv -> ForwardDiff.gradient(z -> h_of(z, θv), η̂), θ)

    S = _foce_solve_multi(H, -B, cfg)            # q×p mode sensitivity, one factorisation
    implicit = transpose(S) * b                  # = −Bᵀλ, up to numerical error

    L = sol.value + logdet_pd(G̃) / 2 - (q / 2) * log(2π)
    return (grad = gφ .+ implicit, implicit = implicit, b = b, S = S, B = B, L = L,
            eta = η̂, Gtilde = G̃, converged = sol.converged, iters = sol.iters,
            gnorm = sol.gnorm)
end

"""
    foce_forward_gradient_parts(model, θ, pop, cfg; eta_starts=nothing) -> NamedTuple

Population FOCE forward-sensitivity gradient with diagnostics.  Per-index buffers and a serial
`sum` in index order, so the result is bitwise independent of the thread schedule — the same
determinism contract as every other gradient here.
"""
function foce_forward_gradient_parts(model, θ::AbstractVector{T}, pop::PopulationData,
                                     cfg::LaplaceConfig; eta_starts = nothing) where {T<:AbstractFloat}
    cfg.inner_deriv === :ad || error(
        "the FOCE forward gradient requires cfg.inner_deriv = :ad; see `subject_foce_forward`.")

    n = pop.n_subjects
    p = length(θ)
    G = zeros(T, p, n)
    Impl = zeros(T, p, n)
    bs = Vector{Vector{T}}(undef, n)
    etas = Vector{Vector{T}}(undef, n)
    Ls = zeros(T, n)
    conv = Vector{Bool}(undef, n)
    iters = Vector{Int}(undef, n)
    gnorms = zeros(T, n)
    subs = pop.subjects

    foreach_subject(n, cfg.threaded) do i
        st = eta_starts === nothing ? nothing : eta_starts[i]
        r = subject_foce_forward(model, θ, subs[i], cfg; η_start = st)
        @views G[:, i] .= r.grad
        @views Impl[:, i] .= r.implicit
        bs[i] = r.b
        etas[i] = r.eta
        Ls[i] = r.L
        conv[i] = r.converged
        iters[i] = r.iters
        gnorms[i] = r.gnorm
    end

    grad = [sum(@view G[j, :]) for j in 1:p]
    implicit = [sum(@view Impl[j, :]) for j in 1:p]
    return (grad = grad, implicit = implicit, b = bs, L = sum(Ls), Ls = Ls, grads = G,
            etas = etas, converged = conv, iters = iters, gnorms = gnorms,
            n_nonconverged = count(!, conv), worst_gnorm = maximum(gnorms))
end

"""
    foce_forward_gradient(model, pop, cfg; fixed_starts=nothing) -> grad!(G, θ)

In-place `grad!(G, θ)` factory in the same form as ``curvature_gradient`` and
``forward_gradient``, so it drops into `fit_nlme` and the benchmark harnesses
unchanged.  `fixed_starts` mirrors `curvature_gradient`'s history-independent fixed warm start,
which is what the joint fit uses.
"""
function foce_forward_gradient(model, pop::PopulationData, cfg::LaplaceConfig;
                               fixed_starts = nothing)
    if fixed_starts !== nothing
        return function (G, θ)
            G .= foce_forward_gradient_parts(model, θ, pop, cfg; eta_starts = fixed_starts).grad
            return G
        end
    end
    if cfg.warm_start
        cache = [zeros(Float64, n_random_effects(model)) for _ in 1:pop.n_subjects]
        return function (G, θ)
            r = foce_forward_gradient_parts(model, θ, pop, cfg; eta_starts = cache)
            for i in 1:pop.n_subjects
                cache[i] .= r.etas[i]
            end
            G .= r.grad
            return G
        end
    end
    return function (G, θ)
        G .= foce_forward_gradient_parts(model, θ, pop, cfg).grad
        return G
    end
end
