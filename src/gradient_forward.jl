# FORWARD-sensitivity (direct implicit) gradient — a comparator for the adjoint.
# Spec: forward_sensitivity_comparator.md. Mathematical_Approach.md §4.2–4.4.
#
# This differs from `gradient_adjoint.jl` in exactly one place: how the implicit
# dependence of the mode η̂(θ) enters the population gradient. Everything else —
# the mode solve, the exact Hessian H and its regularization H̃ = H + reg·I, the
# objective, the log-determinant term b, and the explicit ∂θφ term gφ — is
# computed by the identical formulas the adjoint uses, so any gradient difference
# is attributable to the direct-vs-adjoint contraction alone.
#
# Per subject, at the converged mode η̂ detached to plain floats:
#
#   b   = ½ ∇_η logdet H̃(η, θ) |_{η=η̂}                         (same as adjoint)
#   gφ  = ∂θ [ h(η̂,θ) + ½ logdet H̃(η̂,θ) ]   (η̂ frozen)          (same as adjoint)
#   B   = ∂θ [ ∇_η h(η̂, θ) ] = ∇²_ηθ h                          q×p, the new object
#   S   = dη̂/dθ = −H⁻¹ B                                        q×p, the new object
#   ∇θL = gφ + Sᵀ b
#
# where the adjoint instead solves Hλ = b and forms ∇θL = gφ − Bᵀλ. The two are
# algebraically the same gradient:
#
#   Sᵀ b = −(H⁻¹B)ᵀ b = −Bᵀ H⁻¹ b = −Bᵀ λ = −c,
#
# so `implicit = Sᵀb` is exactly the adjoint's `−c`, and the validation identity
# `Sᵀb + Bᵀλ = 0` holds to numerical error. The orientation is the only
# difference: the adjoint does one q×q solve with a single right-hand side then a
# VJP; the forward method forms the full q×p mode-sensitivity S from p right-hand
# sides. This is the apples-to-apples cost comparison the comparator exists for —
# expected to favour forward at small p and adjoint as p grows.
#
# As in the adjoint path, **H (not H̃) is the coefficient of the sensitivity
# solve**: it is the Jacobian of the mode equation ∇_η h = 0, while H̃ is the
# objective's curvature and appears only in b and gφ. Using H̃ here would bias the
# gradient by O(reg) exactly as it would in the adjoint solve.
#
# B is obtained by forward-mode ForwardDiff through the same numerical model/ODE
# solve the rest of the framework uses — the derivative backend is held fixed, so
# this isolates the EBE-layer orientation as the sole variable (spec §"ODE
# derivative backend", required first implementation).

# Multi-right-hand-side positive-definite solve, sharing `solve_pd`'s definiteness
# contract (Cholesky with check=false → all-NaN on failure) but factoring once for
# the whole q×p system rather than p times — otherwise the forward method would be
# unfairly charged p redundant factorizations against the adjoint's one.
function _solve_pd_multi(A::AbstractMatrix, Bmat::AbstractMatrix)
    C = cholesky(Symmetric(A); check = false)
    T = promote_type(eltype(A), eltype(Bmat))
    issuccess(C) || return fill(nan_like(zero(T)), size(Bmat))
    return C \ Bmat
end

"""
    subject_forward(model, θ, subject, cfg; η_start=nothing) -> NamedTuple

One subject's forward/direct-sensitivity gradient and its parts: `(grad, implicit,
b, S, B, L, eta, Htilde, converged, iters, gnorm)`.

The `b`, the explicit `gφ` term, the mode `η̂`, and the curvature `H`/`H̃` are
computed by the same formulas as [`subject_adjoint`](@ref); the forward-specific
objects are the mixed-derivative matrix `B = ∇²_ηθ h` (q×p) and the mode
sensitivity `S = −H⁻¹B` (q×p). The implicit correction is `implicit = Sᵀb`, which
equals the adjoint's `−c` up to numerical error.

Same contract on `θ` and `cfg.inner_deriv` as [`subject_adjoint`](@ref): `θ` is a
plain float (`AbstractFloat`, admitting `BigFloat`, excluding `Dual` — this
routine *is* the derivative), and `:ad` inner derivatives are required so the
objective's `logdet H̃` and every term here come from the same ForwardDiff Hessian.
"""
function subject_forward(model, θ::AbstractVector{T}, subject,
                         cfg::LaplaceConfig; η_start = nothing) where {T<:AbstractFloat}
    cfg.inner_deriv === :ad ||
        error("subject_forward requires cfg.inner_deriv = :ad; got :$(cfg.inner_deriv). \
               Like the adjoint, its terms are ForwardDiff-derived, so an FD inner \
               Hessian in the objective would make this the gradient of a different \
               function. Use `gradient = :fd` for the fully finite-difference baseline.")

    q = n_random_effects(model)
    η_start = resolve_start(model, θ, subject, η_start, cfg)
    η0 = η_start === nothing ? zeros(T, q) : collect(T, η_start)
    sol = newton_solve(model, θ, subject, η0, cfg)

    # Detach the mode: the ∂θ terms below are partial derivatives, and the implicit
    # dependence of η̂ on θ enters only through the sensitivity S.
    η̂ = extract_value(sol.η)
    H = sol.H            # constraint Jacobian: coefficient of the sensitivity solve
    H̃ = sol.Htilde       # objective curvature: what logdet, b and gφ refer to

    h_of_η(z, t) = negative_log_joint_density(model, z, t, subject)

    # b = ½ ∇_η logdet H̃(η, θ), θ frozen — identical to the adjoint's b.
    b = 0.5 .* ForwardDiff.gradient(
        ηv -> logdet_pd(htilde(ForwardDiff.hessian(z -> h_of_η(z, θ), ηv), cfg)),
        η̂)

    # gφ = ∂θ [ h(η̂,θ) + ½ logdet H̃(η̂,θ) ] with η̂ frozen — identical to adjoint.
    gφ = ForwardDiff.gradient(θv -> begin
            hv = h_of_η(η̂, θv)
            Hv = ForwardDiff.hessian(z -> h_of_η(z, θv), η̂)
            hv + 0.5 * logdet_pd(htilde(Hv, cfg))
        end, θ)

    # B = ∇²_ηθ h, the full q×p mixed-derivative matrix: forward-mode jacobian in θ
    # of the η-gradient of h. The adjoint never forms this — it contracts it into a
    # single VJP — which is precisely the cost axis being compared.
    B = ForwardDiff.jacobian(θv -> ForwardDiff.gradient(z -> h_of_η(z, θv), η̂), θ)

    # S = dη̂/dθ = −H⁻¹B via one factorization and p right-hand sides. Coefficient
    # is H, the mode-equation Jacobian — not H̃ (see header).
    S = _solve_pd_multi(H, -B)
    implicit = S' * b                                    # = −c, up to numerical error

    L = sol.value + logdet_pd(H̃) / 2 - (q / 2) * log(2π)
    return (grad = gφ .+ implicit, implicit = implicit, b = b, S = S, B = B,
            L = L, eta = η̂, Htilde = H̃, converged = sol.converged,
            iters = sol.iters, gnorm = sol.gnorm)
end

"""
    forward_gradient_parts(model, θ, pop, cfg; eta_starts=nothing) -> NamedTuple

Population forward gradient with its diagnostic parts: the summed gradient, the
summed implicit correction, and the per-subject `b`, `S` and `B`. Mirror of
[`adjoint_gradient_parts`](@ref) for the direct-sensitivity orientation; a
debugging/benchmark entry point, not a separate estimator.
"""
function forward_gradient_parts(model, θ::AbstractVector{T},
                                pop::PopulationData, cfg::LaplaceConfig;
                                eta_starts = nothing) where {T<:AbstractFloat}
    cfg.inner_deriv === :ad ||
        error("the forward gradient requires cfg.inner_deriv = :ad; got :$(cfg.inner_deriv). \
               See `subject_forward`.")

    n = pop.n_subjects
    p = length(θ)
    G = zeros(T, p, n)
    Impl = zeros(T, p, n)
    Bs = Vector{Matrix{T}}(undef, n)
    Ss = Vector{Matrix{T}}(undef, n)
    bs = Vector{Vector{T}}(undef, n)
    etas = Vector{Vector{T}}(undef, n)
    Ls = zeros(T, n)
    conv = Vector{Bool}(undef, n)
    iters = Vector{Int}(undef, n)
    subs = pop.subjects

    foreach_subject(n, cfg.threaded) do i
        st = eta_starts === nothing ? nothing : eta_starts[i]
        r = subject_forward(model, θ, subs[i], cfg; η_start = st)
        @views G[:, i] .= r.grad
        @views Impl[:, i] .= r.implicit
        Bs[i] = r.B
        Ss[i] = r.S
        bs[i] = r.b
        etas[i] = r.eta
        Ls[i] = r.L
        conv[i] = r.converged
        iters[i] = r.iters
    end

    # Serial reduction in index order — schedule-independent, bitwise reproducible.
    grad = [sum(@view G[j, :]) for j in 1:p]
    implicit = [sum(@view Impl[j, :]) for j in 1:p]
    return (grad = grad, implicit = implicit, b = bs, S = Ss, B = Bs,
            L = sum(Ls), Ls = Ls, grads = G, etas = etas, converged = conv,
            iters = iters, n_nonconverged = count(!, conv))
end

"""
    forward_gradient(model, pop, cfg) -> grad!(G, θ)

Factory for the forward/direct-sensitivity gradient, in the same in-place
`grad!(G, θ)` form as [`adjoint_gradient`](@ref) so it drops into [`fit_nlme`](@ref)
and the example harnesses without any special handling.
"""
function forward_gradient(model, pop::PopulationData, cfg::LaplaceConfig)
    if cfg.warm_start
        cache = [zeros(Float64, n_random_effects(model)) for _ in 1:pop.n_subjects]
        return function (G, θ)
            r = forward_gradient_parts(model, θ, pop, cfg; eta_starts = cache)
            for i in 1:pop.n_subjects
                cache[i] .= r.etas[i]
            end
            G .= r.grad
            return G
        end
    end
    return function (G, θ)
        G .= forward_gradient_parts(model, θ, pop, cfg).grad
        return G
    end
end
