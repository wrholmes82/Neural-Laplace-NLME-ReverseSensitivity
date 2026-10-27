# The inner conditional-mode solve: damped Newton with Armijo backtracking,
# generic over ForwardDiff.Dual at arbitrary nesting depth.
#
# This is the deliberate departure from SciML for the inner level. The AD-through
# reference gradient differentiates *through* this solve, so it must accept and
# return Duals; Optimization.jl's solve path cannot. Using one solver for the
# adjoint path and a different one for the AD path would mean the two gradients
# are no longer gradients of the same computed object, which would destroy the
# cross-validation the project exists to perform.

"""
    inner_gradient(f, η, cfg)

`∇_η f` by ForwardDiff (`cfg.inner_deriv === :ad`) or by central differences
(`:fd`). The `:fd` branch reproduces the reference manuscript's FD baseline,
where the inner derivatives inherit step-size noise too.
"""
function inner_gradient(f::F, η::AbstractVector, cfg::LaplaceConfig) where {F}
    cfg.inner_deriv === :ad && return ForwardDiff.gradient(f, η)
    g = similar(η)
    for k in eachindex(η)
        s = cfg.inner_fd_step * max(abs(extract_value(η[k])), 1.0)
        ep = copy(η); ep[k] += s
        em = copy(η); em[k] -= s
        g[k] = (f(ep) - f(em)) / (2s)
    end
    return g
end

"""
    inner_hessian(f, η, cfg)

`∇²_ηη f` by ForwardDiff (`:ad`) or by central second differences (`:fd`),
symmetrized.
"""
function inner_hessian(f::F, η::AbstractVector, cfg::LaplaceConfig) where {F}
    cfg.inner_deriv === :ad && return ForwardDiff.hessian(f, η)
    q = length(η)
    H = similar(η, q, q)
    s = [cfg.inner_fd_step * max(abs(extract_value(η[k])), 1.0) for k in 1:q]
    f0 = f(η)
    for j in 1:q
        ep = copy(η); ep[j] += s[j]
        em = copy(η); em[j] -= s[j]
        # `2 * f0`, never `2f0`: Julia lexes `2f0` as the Float32 literal 2.0f0,
        # not as two-times-f0, so the centre term would silently become the
        # constant 2.0 and corrupt every diagonal entry.
        H[j, j] = (f(ep) - 2 * f0 + f(em)) / s[j]^2
        for k in (j + 1):q
            pp = copy(η); pp[j] += s[j]; pp[k] += s[k]
            pm = copy(η); pm[j] += s[j]; pm[k] -= s[k]
            mp = copy(η); mp[j] -= s[j]; mp[k] += s[k]
            mm = copy(η); mm[j] -= s[j]; mm[k] -= s[k]
            v = (f(pp) - f(pm) - f(mp) + f(mm)) / (4 * s[j] * s[k])
            H[j, k] = v
            H[k, j] = v
        end
    end
    return H
end

_gnorm(g) = sqrt(sum(abs2, extract_value(g)))

"""
    pd_search_hessian(H, cfg) -> AbstractMatrix

A positive-definite curvature matrix for the inner Newton **direction only**.
Full rationale and safety argument: `PLAN_pd_inner_search.md`.

⚠ SEARCH-ONLY — must never reach the objective or the gradient. Those use the
exact `H̃ = H + reg·I` (the objective's curvature, in `logdet` and the adjoint RHS)
and the exact `H` (the mode-equation Jacobian, the adjoint solve's coefficient),
both returned by `newton_solve`. Substituting this modified matrix into either
would change the estimator — it would be the FOCE/Gauss-Newton objective
(`PLAN_adjoint_foce.md`), not this one. The mode is defined by `∇_η h = 0`
whatever matrix drives the iteration, so this only changes the search *path*.

Far from the mode `∇²_ηη h` is routinely indefinite and `−H⁻¹ g` is not a descent
direction. Rather than the old normalized-steepest-descent fallback we add a
diagonal shift restoring positive definiteness (modified-Newton / Levenberg),
letting the trust radius and line search supply the step length. The shift targets
a **bounded condition number**, not a minimal perturbation, and this is
load-bearing: `λ_min(M) = inner_pd_floor · max(λ_max, 1)` gives
`cond(M) ≤ 1/inner_pd_floor` (default 10). A minimal, barely-PD shift would leave a
near-zero eigenvalue in the previously-negative direction and take an enormous step
there whose unrolled derivative fails to converge, breaking adjoint/AD-through
agreement to ~1e-3 (measured); a well-conditioned shift keeps it at ~1e-10.

The shift is decided on **primal values** (`extract_value(H)`), carrying no
partials, so like the step lengths it is a constant of the realized trajectory —
what keeps AD-through differentiating the same procedure the objective evaluated.
It fires **only where `H` is genuinely indefinite** (`λ_min ≤ 0`); a positive-
definite `H` is left as the exact Newton step `H̃`, bit-identical to no
modification.
"""
function pd_search_hessian(H::AbstractMatrix, cfg::LaplaceConfig)
    Hval = extract_value(H)
    # Non-finite exact Hessian (e.g. structural-model overflow at absurd η): do
    # not attempt an eigensolve; hand back H̃ and let solve_pd/​the steepest-descent
    # fallback below handle it exactly as before.
    all(isfinite, Hval) || return htilde(H, cfg)
    # The shift only steers the search path, never the mode or the returned exact
    # Hessian, so its magnitude may be decided in Float64 — this also keeps the
    # eigensolve on the LAPACK path when H is BigFloat (high-precision reference),
    # where `eigvals` on a Symmetric{BigFloat} is unsupported.
    ev = eigvals(Symmetric(Float64.(Hval)))           # ascending
    λmin, λmax = ev[1], ev[end]
    λmin > 0 && return htilde(H, cfg)                 # positive definite: exact Newton, unchanged
    β = cfg.inner_pd_floor * max(λmax, 1.0)           # target smallest eigenvalue after the shift
    return H + (β - λmin) * I                         # PD by construction: λ_min(M) = β
end

"""
    newton_solve(model, θ, subject, η0, cfg) -> NamedTuple

Damped Newton with Armijo backtracking on `h(η) = negative_log_joint_density(model, η, θ, subject)`.

Returns `(η, H, Htilde, value, gnorm, converged, iters)`. `H` is the raw
η-Hessian at the returned `η`; `Htilde = H + reg·I` is the matrix every downstream
equation uses. Both are returned so no caller ever recomputes a Hessian.

The Newton direction solves a **positive-definite** system: `H̃ = H + reg·I` on the
common path, and the modified-Newton matrix of [`pd_search_hessian`](@ref) where
the exact `H` is indefinite, so the step is always a descent direction without
steepest descent. This changes the *path* only — the returned `H`/`Htilde` are the
exact Hessian the objective and adjoint use. A genuinely non-finite `H` still falls
back to normalized steepest descent. Every branch is taken on `extract_value`d
primal quantities, so the realized trajectory is a function of θ's value alone —
which is exactly what AD-through must differentiate.

**Non-convergence is recorded, not swallowed.** Hitting `inner_maxiter` (or a
failed line search) returns `converged = false` along with the objective value at
whatever η was reached, because that is the value AD-through is differentiating
and the two must agree.
"""
function newton_solve(model, θ::AbstractVector, subject, η0::AbstractVector,
                      cfg::LaplaceConfig)
    T = promote_type(eltype(θ), eltype(η0), Float64)
    η = convert(Vector{T}, η0)
    f = ηv -> negative_log_joint_density(model, ηv, θ, subject)

    fval = f(η)
    g = inner_gradient(f, η, cfg)
    H = inner_hessian(f, η, cfg)
    converged = false
    iters = 0

    for it in 1:cfg.inner_maxiter
        gn = _gnorm(g)
        if isfinite(gn) && gn < cfg.inner_tol
            converged = true
            break
        end
        isfinite(gn) || break                       # NaN/Inf: stop and propagate

        # `pd_search_hessian` gives a positive-definite curvature (H̃ on the common
        # path, a modified-Newton shift where the exact H is indefinite), so the
        # solve succeeds and the direction is a descent direction. SEARCH-ONLY:
        # this matrix never leaves the loop — the objective and adjoint use the
        # exact H/H̃ returned below.
        d = -solve_pd(pd_search_hessian(H, cfg), g)
        dv = extract_value(d)
        if !all(isfinite, dv) || extract_value(dot(g, d)) >= 0
            # Last resort for a genuinely non-finite H (structural-model overflow
            # at absurd η), which the PD shift cannot rescue. Normalized steepest
            # descent — `-g` has no natural scale, so on a standardized η scale a
            # raw step of length tens reaches degenerate parameters; measured on
            # the two-compartment problem, a ‖g‖ ≈ 36 step drove Q → 0 and the
            # model threw before the line search could backtrack. The line search,
            # not the direction, then supplies the scale.
            d = -g ./ max(gn, 1.0)
        end

        # Trust radius. Backtracking alone cannot rescue an arbitrarily long
        # direction — `linesearch_maxiter` backtracks at `damping` shrink a step
        # by only damping^maxiter (3.8e-3 at the defaults), so a Newton direction
        # of length 1e14, which a nearly singular H̃ produces, leaves the line
        # search exhausted having never evaluated a sane point and the solve
        # takes ZERO steps. Measured: one subject stuck at η = 0 with
        # ‖∇_η h‖ = 29 and min eig(H̃) = 5e-14, whose ½ logdet H̃ then contributed
        # −7.2 to L and manufactured a spurious "better" optimum 24 OFV units
        # below the true one. Scaling preserves the direction and lets the line
        # search supply the length.
        dn = sqrt(sum(abs2, extract_value(d)))
        if isfinite(dn) && dn > cfg.step_max
            d = d .* (cfg.step_max / dn)
        end

        # α stays Float64: the step length is chosen by primal comparisons, so it
        # is a constant of the realized trajectory and must carry no partials.
        α = 1.0
        slope = extract_value(dot(g, d))
        f0 = extract_value(fval)
        # Roundoff allowance on the Armijo test. Near the mode the predicted
        # decrease c·α·slope falls below eps(f0) and h can no longer express it:
        # measured on the two-compartment problem, a subject at ‖∇_η h‖ ≈ 1.3e-7
        # has df = 0 exactly, so a strict test rejects every step and the solve
        # stalls three orders short of a tight inner tolerance — which then shows
        # up as spurious adjoint/AD-through disagreement. The allowance is pure
        # floating-point noise (~7e-15 at h ≈ 3) and cannot mask a real increase.
        τ = 8 * eps(Float64) * max(abs(f0), 1.0)
        accepted = false
        ηnew = η
        fnew = fval
        for _ in 1:cfg.linesearch_maxiter
            ηnew = η .+ α .* d
            fnew = f(ηnew)
            fv = extract_value(fnew)
            if isfinite(fv) && fv <= f0 + cfg.armijo_c * α * slope + τ
                accepted = true
                break
            end
            α *= cfg.damping
        end
        accepted || break                           # line search exhausted

        η = ηnew
        fval = fnew
        g = inner_gradient(f, η, cfg)
        H = inner_hessian(f, η, cfg)
        iters = it
    end

    gn = _gnorm(g)
    converged = converged || (isfinite(gn) && gn < cfg.inner_tol)
    return (η = η, H = H, Htilde = htilde(H, cfg), value = fval,
            gnorm = gn, converged = converged, iters = iters)
end

"""
    resolve_start(model, θ, subject, η_start, cfg) -> start vector or `nothing`

Resolve a subject's inner start. `η_start` may be `nothing`, a single start vector, or a **list** of
candidate start vectors. Given a list, each candidate is run to its mode and the one with the
smallest `h(η̂)` is returned. Selection is on `value`, not on the Laplace contribution `L` (which
adds `½logdet`), because the conditional mode is `argmin h` — selecting on `L` would return a point
that is not the mode.

Why this exists. With a single start the Laplace expansion is taken at whichever basin that start
falls into, and the basin containing a *fixed* start moves as θ moves. On this problem 60–90% of
subjects have two conditional modes separated by 10–100 units of objective, so when the basin flips
the population objective jumps by O(10) and a backtracking line search cannot cross it: the fit stops
dead at a point that is not stationary. `η = 0` finds the global mode for all but a handful of
subjects, but *which* handful depends on θ, so the alternative must be carried and re-checked rather
than precomputed.

Selection is made on `extract_value`d primals, so the choice is a function of θ's value alone — the
invariant every other branch in [`newton_solve`](@ref) obeys, and the one AD-through requires.
"""
resolve_start(model, θ, subject, η_start::Nothing, cfg::LaplaceConfig) = nothing
resolve_start(model, θ, subject, η_start::AbstractVector{<:Number}, cfg::LaplaceConfig) = η_start
function resolve_start(model, θ, subject, η_starts, cfg::LaplaceConfig)
    θv = extract_value(θ)
    best = nothing
    bestv = Inf
    for s in η_starts
        sol = newton_solve(model, θv, subject, collect(Float64, extract_value(s)), cfg)
        v = extract_value(sol.value)
        if sol.converged && isfinite(v) && v < bestv - 1e-10
            bestv = v
            best = extract_value(sol.η)
        end
    end
    return best === nothing ? first(η_starts) : best
end
