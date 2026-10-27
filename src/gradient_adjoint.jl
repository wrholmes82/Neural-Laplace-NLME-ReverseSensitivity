# FULL-implicit (adjoint) gradient — Mathematical_Approach.md §4.3–4.4,
# Eqs. (4.3) and (4.4).
#
# Per subject, at the converged mode η̂ with η̂ detached to plain floats:
#
#   b   = ½ ∇_η logdet H̃(η, θ) |_{η=η̂}                          (Eq. 4.3 RHS)
#   H λ = b                                                      (Eq. 4.3)
#   c   = (∇²_ηθ h)ᵀ λ = ∇_θ [ (∇_η h(η̂, θ))ᵀ λ ]                (§4.4 step 4)
#   ∇θ L = ∂θ h|_η̂ + ½ ∂θ logdet H̃|_η̂ − c                       (Eq. 4.4)
#
# Which matrix goes where is not a free choice, and the two roles are different:
#
#   * `H̃` is the curvature the OBJECTIVE uses (`½ logdet H̃`), so it is also what
#     `b` differentiates — b is ∇_η of that same log-determinant.
#   * `H` — the raw ∇²_ηη h — is the coefficient of the adjoint SOLVE, because it
#     is the Jacobian of the constraint that defines the mode. The inner solver
#     terminates on ‖∇_η h‖ < inner_tol, so η̂ solves ∇_η h = 0 (the `reg` shift
#     enters the Newton *direction*, never the fixed point), and the implicit
#     function theorem applied to that constraint gives dη̂/dθ = −H⁻¹ ∇²_ηθ h with
#     H, not H̃. Solving against H̃ instead biases every gradient by O(reg):
#     measured on the two-compartment problem against AD-through at
#     inner_tol = 1e-12, max|adjoint − AD| was 6.5e-8 at reg = 1e-6 and 6.5e-4 at
#     reg = 1e-2, i.e. exactly linear in reg, while solving against H held the
#     difference at the 1e-12 inner-tolerance floor across eight orders of
#     magnitude of reg. Invisible at the 1e-10 default; not invisible for anyone
#     who raises reg, which is precisely the ill-conditioned case where it
#     amplifies as reg·‖H⁻²‖.
#
# With that split, this is the exact gradient of the objective `objective.jl`
# actually evaluates, not of a nearby one.
#
# `c` is the **adjoint correction term**: the only part carrying the implicit
# dependence of η̂ on θ, the hardest part to get right, and the part whose silent
# omission would still leave a plausible-looking gradient behind. It is returned
# as a diagnostic alongside the gradient — it is already computed, so this is
# free — which is what makes the correction-term tests possible without exporting
# a second, approximate gradient.

"""
    subject_adjoint(model, θ, subject, cfg; η_start=nothing) -> NamedTuple

One subject's adjoint gradient and its parts: `(grad, c, b, lambda, L, eta,
Htilde, converged, iters, gnorm)`.

`θ` must be a plain float — this routine *is* the derivative, so nothing
differentiates through it. `AbstractFloat` rather than `Float64` deliberately:
`ForwardDiff.Dual` is a `Real` but not an `AbstractFloat`, so the signature still
excludes it and still enforces "nothing differentiates through this", while
admitting `BigFloat` so an extended-precision reference gradient can be computed
for work-precision studies.

Requires `cfg.inner_deriv === :ad`. Under `:fd` the objective's `logdet H̃` is
built from a finite-difference Hessian while every term here is built from a
ForwardDiff one, so the two would be a gradient and a function of *different*
problems — the exact defect the shared-`cfg` discipline exists to prevent. The
`:fd` inner mode is for the FD baseline (`fd_gradient`), where both layers are
finite differences and the pairing is consistent.
"""
function subject_adjoint(model, θ::AbstractVector{T}, subject,
                         cfg::LaplaceConfig; η_start = nothing) where {T<:AbstractFloat}
    cfg.inner_deriv === :ad ||
        error("subject_adjoint requires cfg.inner_deriv = :ad; got :$(cfg.inner_deriv). \
               The adjoint terms are ForwardDiff-derived, so an FD inner Hessian in the \
               objective would make this the gradient of a different function. Use \
               `gradient = :fd` for the fully finite-difference baseline.")

    q = n_random_effects(model)
    η_start = resolve_start(model, θ, subject, η_start, cfg)
    η0 = η_start === nothing ? zeros(T, q) : collect(T, η_start)
    sol = newton_solve(model, θ, subject, η0, cfg)

    # Detach the mode. This is what makes the ∂θ terms below *partial*
    # derivatives; the implicit dependence of η̂ on θ enters only through λ.
    η̂ = extract_value(sol.η)
    H = sol.H            # constraint Jacobian: coefficient of the adjoint solve
    H̃ = sol.Htilde       # objective curvature: what logdet and b refer to

    h_of_η(z, t) = negative_log_joint_density(model, z, t, subject)

    # b = ½ ∇_η logdet H̃(η, θ), θ frozen.  Nested AD: a Hessian in η inside a
    # gradient in η.
    b = 0.5 .* ForwardDiff.gradient(
        ηv -> logdet_pd(htilde(ForwardDiff.hessian(z -> h_of_η(z, θ), ηv), cfg)),
        η̂)

    # Coefficient is H, the Jacobian of the mode equation ∇_η h = 0 — see the
    # header. A non-PD H at the mode means η̂ is a saddle and the Laplace
    # expression is invalid there, so solve_pd's NaN is the correct answer.
    λ = solve_pd(H, b)

    # ∂θ [ h(η̂,θ) + ½ logdet H̃(η̂,θ) ] with η̂ held fixed — Eq. (4.4) terms 1 and 2,
    # taken in a single pass since both need the same Dual-θ sweep.
    gφ = ForwardDiff.gradient(θv -> begin
            hv = h_of_η(η̂, θv)
            Hv = ForwardDiff.hessian(z -> h_of_η(z, θv), η̂)
            hv + 0.5 * logdet_pd(htilde(Hv, cfg))
        end, θ)

    # c = (∇²_ηθ h)ᵀ λ via the VJP identity. The q×p mixed-derivative matrix is
    # never formed: this is one gradient of a scalar with respect to θ.
    c = ForwardDiff.gradient(
        θv -> dot(ForwardDiff.gradient(z -> h_of_η(z, θv), η̂), λ), θ)

    L = sol.value + logdet_pd(H̃) / 2 - (q / 2) * log(2π)
    return (grad = gφ .- c, c = c, b = b, lambda = λ, L = L, eta = η̂,
            Htilde = H̃, converged = sol.converged, iters = sol.iters,
            gnorm = sol.gnorm)
end

"""
    adjoint_gradient_parts(model, θ, pop, cfg; eta_starts=nothing) -> NamedTuple

Population adjoint gradient together with its diagnostic parts: the summed
gradient, the summed correction term `c`, and the per-subject `b`, `λ` and `c`.

This is a debugging and testing entry point, **not** a second gradient method.
There is one gradient; `c` is one of its terms. A routine that returned a
gradient with `c` omitted would be a second method with its own semantics and its
own opportunity to drift.
"""
function adjoint_gradient_parts(model, θ::AbstractVector{T},
                                pop::PopulationData, cfg::LaplaceConfig;
                                eta_starts = nothing) where {T<:AbstractFloat}
    # Checked here, before the parallel region: raised inside it, the same error
    # would reach the caller wrapped in a TaskFailedException.
    cfg.inner_deriv === :ad ||
        error("the adjoint gradient requires cfg.inner_deriv = :ad; got :$(cfg.inner_deriv). \
               See `subject_adjoint`.")

    n = pop.n_subjects
    p = length(θ)
    G = zeros(T, p, n)
    C = zeros(T, p, n)
    B = Vector{Vector{T}}(undef, n)
    Λ = Vector{Vector{T}}(undef, n)
    etas = Vector{Vector{T}}(undef, n)
    Ls = zeros(T, n)
    conv = Vector{Bool}(undef, n)
    iters = Vector{Int}(undef, n)
    subs = pop.subjects

    foreach_subject(n, cfg.threaded) do i
        st = eta_starts === nothing ? nothing : eta_starts[i]
        r = subject_adjoint(model, θ, subs[i], cfg; η_start = st)
        @views G[:, i] .= r.grad
        @views C[:, i] .= r.c
        B[i] = r.b
        Λ[i] = r.lambda
        etas[i] = r.eta
        Ls[i] = r.L
        conv[i] = r.converged
        iters[i] = r.iters
    end

    # Serial reduction in index order — bitwise independent of the schedule.
    grad = [sum(@view G[j, :]) for j in 1:p]
    c = [sum(@view C[j, :]) for j in 1:p]
    return (grad = grad, c = c, b = B, lambda = Λ, L = sum(Ls), Ls = Ls,
            grads = G, cs = C, etas = etas, converged = conv, iters = iters,
            n_nonconverged = count(!, conv))
end

"""
    adjoint_gradient(model, pop, cfg) -> grad!(G, θ)

Factory for the FULL-implicit gradient, in the in-place `grad!(G, θ)` form all
three gradient factories share so that [`fit_nlme`](@ref) is method-agnostic.

`grad!` is a plain gradient with no extra outputs — `fit_nlme` sees nothing
unusual. Use [`adjoint_gradient_parts`](@ref) when you want the correction term.
"""
function adjoint_gradient(model, pop::PopulationData, cfg::LaplaceConfig)
    if cfg.warm_start
        cache = [zeros(Float64, n_random_effects(model)) for _ in 1:pop.n_subjects]  # Float64: warm starts are a Float64 cache
        return function (G, θ)
            r = adjoint_gradient_parts(model, θ, pop, cfg; eta_starts = cache)
            for i in 1:pop.n_subjects
                cache[i] .= r.etas[i]
            end
            G .= r.grad
            return G
        end
    end
    return function (G, θ)
        G .= adjoint_gradient_parts(model, θ, pop, cfg).grad
        return G
    end
end
