# information.jl — reading a singular observed information honestly.
#
# Two facts about this model class drive everything here, and both are measured rather than
# assumed (see `docs/DECISIONS.md`):
#
#   1. The observed information at θ̂ is **not** positive definite and will not become so. At
#      the scientific width 61 of 153 eigenvalues are negative; at the scaling width, 876 of
#      1773. That is a property of an over-parameterised neural component, not a bug, and
#      `cholesky` failing on it is the correct behaviour of `cholesky`.
#   2. `‖∇L‖∞` says nothing useful about whether a fit is finished. The spectrum spans to
#      ~5e5, so a gradient of 5 in a stiff direction is a parameter displacement of 1e-5.
#      Measured: 98.5–99.98% of the residual gradient lies in directions with `λ > 1`, and
#      the OFV still recoverable there is 0.013 at the scientific width against a scale where
#      ΔOFV = 3.84 decides a parameter.
#
# So the package needs a stopping rule that measures *remaining objective*, and an inverse
# that exists. Both are one eigendecomposition, which every fit that computes the observed
# information has already paid for.

"""Resolve a `tau` specification against a spectrum: a number passes through, `:noise` becomes
`|λ_min|`."""
resolve_tau(tau::Real, values::AbstractVector) = Float64(tau)
function resolve_tau(tau::Symbol, values::AbstractVector)
    tau === :noise || error("unknown tau rule :$(tau); expected a number or :noise.")
    # The floor is estimated FROM THE NEGATIVE EIGENVALUES. If there are none the matrix is
    # positive definite, there is no evidence of a floor, and the correct cut is zero — every
    # direction is determined. Using `abs(λ_min)` there would discard the least-determined
    # direction of a perfectly well-behaved matrix for no reason.
    return max(0.0, -minimum(values))
end

"""
    spectral_split(H; tau = :noise) -> (values, vectors, kept, tau)

Eigendecomposition of a symmetric observed information, split at `tau`.

`tau` is either an absolute eigenvalue cut or the symbol `:noise`, which resolves to
`|λ_min|` — the modulus of the most negative eigenvalue of `H` itself.

`:noise` is the default and is the rule this package recommends. A finite-differenced Hessian
of an indefinite objective carries negative eigenvalues that are numerical noise, since genuine
information about a direction produces POSITIVE curvature. Their magnitude therefore estimates
the noise floor of the construction, and `|λ_min|` keeps exactly the directions that no
negative eigenvalue competes with. It needs no constant, and it adapts automatically to sample
size, parameterisation and differencing step, all of which move the whole spectrum.

Measured against the alternatives on fifteen cohorts (`e4d_cut_rules.jl`), it matches the
former default `tau = 1.0` on coverage exactly (73/75, and identically per functional) with a
slightly better ratio of standard error to across-cohort spread (1.02 against 1.05) and a more
stable retained rank (21–28 against 16–66 across an absolute ladder).

A RELATIVE cut, `c · λ_max`, is the obvious alternative and fails: `λ_max` is set by whichever
direction is best determined, which is unrelated to the noise floor. At `c = 0.01` coverage
falls to 57/75 and the discarded directions carry a third of the functional gradient.

Two limits are worth stating. The rule is defined by finite-difference noise, so a different
differencing step gives a different floor. And for a genuinely positive-definite `H` there is no
negative eigenvalue to read: `:noise` then falls back to `λ_min`, which keeps every direction
but the least determined — harmless, but the justification above no longer applies, and an
explicit numeric `tau` is the better choice for such a model.
"""
function spectral_split(H::AbstractMatrix; tau = :noise)
    E = eigen(Symmetric(Matrix(H)))
    t = resolve_tau(tau, E.values)
    kept = E.values .> t
    return (values = E.values, vectors = E.vectors, kept = kept, tau = t)
end

"""
    newton_decrement(H, g; tau = :noise) -> NamedTuple

`Σ_{λ>tau} (vᵀg)² / λ` — the objective still recoverable in the well-determined subspace, and
the convergence criterion this package recommends in place of a gradient norm.

It is the second-order predicted decrease `L(θ̂) − L(θ̂ + Δ)` for the Newton step `Δ = H⁺g`
restricted to that subspace, so it is **in OFV units** and comparable against the number that
actually decides something: `ΔOFV = 3.84` for one parameter at the 5% level. A fit is finished
when there is nothing left to gain, not when a gradient happens to be small — and on a
log-parameterised model with curvature spanning five orders of magnitude those are very
different statements.

Returns `(decrement, n_kept, stiff_share, grad_inf_norm)`. `stiff_share` is the fraction of
`‖g‖²` lying in the retained subspace: it is the diagnostic that says whether the criterion is
*applicable*. Near 1, the residual gradient is in well-determined directions and the decrement
is the whole story. Well below 1, a lot of gradient sits in near-null directions, the truncation
is discarding something real, and neither this nor a gradient norm should be trusted.
"""
function newton_decrement(H::AbstractMatrix, g::AbstractVector; tau = :noise)
    s = spectral_split(H; tau = tau)
    c = transpose(s.vectors) * collect(Float64, g)
    dec = sum(abs2.(c[s.kept]) ./ s.values[s.kept]; init = 0.0)
    tot = sum(abs2, c)
    return (decrement = dec, n_kept = count(s.kept),
            stiff_share = tot > 0 ? sum(abs2, c[s.kept]) / tot : NaN,
            grad_inf_norm = maximum(abs, g))
end

"""
    truncated_inverse(H; tau = :noise) -> (Sigma, n_kept, ...)

`Σ = Σ_{λ>tau} v vᵀ / λ`, the covariance restricted to the identified subspace.

This is what `inv` cannot supply and `cholesky` correctly refuses. The justification for using
it is not that it makes a number appear — it is that the **functionals** the study reports load
almost entirely on the retained subspace. Measured on the fitted effect curve: `∂E(C)/∂θ` lies
100% in directions with `λ > 1e-4`, and the resulting standard errors are stable to ~10% across
four orders of magnitude of `tau`. `delta_method` below returns that share for every functional
so the claim is checked per quantity rather than asserted once.
"""
function truncated_inverse(H::AbstractMatrix; tau = :noise)
    s = spectral_split(H; tau = tau)
    V = s.vectors[:, s.kept]
    λ = s.values[s.kept]
    return (Sigma = V * Diagonal(1 ./ λ) * transpose(V), n_kept = count(s.kept),
            lambda_min_kept = isempty(λ) ? NaN : minimum(λ),
            lambda_max = maximum(s.values), n_negative = count(<=(0), s.values))
end

"""
    delta_method(H, J; tau = :noise) -> (se, identified_share, n_kept)

Standard errors for functionals whose Jacobian rows are the rows of `J`, by
`Var[f] = J Σ Jᵀ` with `Σ` the truncated inverse.

`identified_share[k]` is the fraction of `‖J[k,:]‖²` inside the retained subspace. **A
functional with a share near 1 is identified even when θ is not**, which is the design
principle the whole study rests on: individual network weights are not estimable at these
widths and no claim is made about them, while `E(C)` and the quantities derived from it are.
A share well below 1 means that particular functional is *not* rescued by the truncation and
its interval must not be reported.
"""
function delta_method(H::AbstractMatrix, J::AbstractMatrix; tau = :noise)
    s = spectral_split(H; tau = tau)
    A = J * s.vectors                      # rows of J in the eigenbasis
    Ak = A[:, s.kept]
    λ = s.values[s.kept]
    var = vec(sum(abs2.(Ak) ./ transpose(λ), dims = 2))
    tot = vec(sum(abs2, A, dims = 2))
    share = [t > 0 ? sum(abs2, view(Ak, k, :)) / t : NaN for (k, t) in enumerate(tot)]
    return (se = sqrt.(max.(var, 0.0)), identified_share = share, n_kept = count(s.kept))
end

"""
    fit_quality(; inner_nonconverged, gradient_rescues, lambda_min, lambda_max) -> (ok, reason)

The mechanical accept/reject rule for a fit entering an aggregate.

Two conditions, both measured rather than judged. A non-zero inner-failure count means some
subject's conditional mode was never found, so the objective at θ̂ is not the objective at all —
observed at `N = 25`, where 18% of gradient calls had a failing subject, the outer solve
terminated early, and the resulting information spanned ±1e7. A spectrum with
`|λ_min| > λ_max/1e3` is the same pathology showing up in the curvature. Either one disqualifies
a fit from a coverage or surrogate table; neither is a reason to alter the fit.
"""
function fit_quality(; inner_nonconverged::Int, gradient_rescues::Int,
                     lambda_min::Float64, lambda_max::Float64)
    gradient_rescues > 0 && return (ok = false, reason = "gradient rescues = $(gradient_rescues)")
    inner_nonconverged > 0 &&
        return (ok = false, reason = "inner solves non-converged = $(inner_nonconverged)")
    abs(lambda_min) > lambda_max / 1e3 &&
        return (ok = false, reason = "indefinite information: |lambda_min| = " *
                "$(round(abs(lambda_min); sigdigits=3)) exceeds lambda_max/1e3 = " *
                "$(round(lambda_max/1e3; sigdigits=3))")
    return (ok = true, reason = "")
end
