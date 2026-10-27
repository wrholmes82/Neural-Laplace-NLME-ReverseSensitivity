# Two-compartment linear system with an impulse (bolus) input into the central
# compartment; closed-form bi-exponential solution.
#
# Adapted from OLD/DEMCMC_NLME1_011226/Linear_2comp_IV, with three non-cosmetic
# conversions, because that source module is Bayesian and self-contained:
#   1. Sign flip — its `log_likelihood_*` are positive log densities; `h` here is
#      the *negative* log joint.
#   2. The prior on θ is dropped. This project is frequentist.
#   3. The `-Inf` cliffs are removed (it returned `-Inf` for C ≤ 0 and after a
#      failed eigenvalue check). A hard discontinuity in the objective is exactly
#      what the H̃ convention and the definiteness check in `logdet_pd` replace.
#
# Proportional residual error is deliberate: it makes H_i genuinely η-dependent,
# so the adjoint correction term c is nonzero and actually gets exercised. The
# additive variant is kept as a near-degenerate contrast.

using LinearAlgebra
using ForwardDiff
using NLMEReverse
import NLMEReverse: negative_log_joint_density, predict, n_random_effects

"""
    TwoComp(; error_model=:proportional, re_scale=1.0)

q = 4 random effects (log-normal on CL, V1, Q, V2), diagonal Ω. CL is the
elimination conductance, V1 and V2 the central and peripheral compartment
volumes, Q the inter-compartmental transfer conductance.

p = 9: `θ = (log CL, log V1, log Q, log V2, log ω_CL, log ω_V1, log ω_Q, log ω_V2, log σ)`,
all on log scale.

`re_scale = ε` multiplies every ω, giving `Ω = ε² Ω₀` — the scaled random-effect
variant used for the `O(ε⁴)` correction-term test. It costs nothing: it is a
parameter of the existing model, not a third model.
"""
struct TwoComp <: NLMEModel
    q::Int
    p::Int
    error_model::Symbol
    re_scale::Float64
end

TwoComp(; error_model::Symbol = :proportional, re_scale::Real = 1.0) =
    TwoComp(4, 9, error_model, Float64(re_scale))

"""
    twocomp_conc(t, CL, V1, Q, V2, input)

`C(t) = (input/V1) (A e^{−αt} + B e^{−βt})` from the micro-constants
`k10 = CL/V1`, `k12 = Q/V1`, `k21 = Q/V2`, with α, β the roots of
`s² − (k10+k12+k21)s + k10 k21`.

Written to be cancellation-free, which is not cosmetic. With
`s = k10+k12−k21` and `w = 4 k12 k21 ≥ 0`, the discriminant is `√(s² + w)` — never
the equivalent `√((k10+k12+k21)² − 4 k10 k21)` — and the coefficients are

    A = (disc + s) / 2disc,    B = (disc − s) / 2disc

with whichever of `disc ± s` is the *subtractive* one evaluated through its
conjugate, `disc − s = w/(disc + s)` for `s > 0` and `disc + s = w/(disc − s)` for
`s < 0`. Both `A` and `B` are then manifestly nonnegative, so `C(t) ≥ 0`
identically and the proportional error model never sees a nonpositive prediction.

This matters when `Q → 0`, where `w ≪ s²` and `disc → |s|`. Evaluated directly,
`disc − s` is a catastrophic subtraction that comes out slightly *negative*; `B`
then goes negative, and at late times — where the `A e^{−αt}` term has already
underflowed — `C(t)` is a tiny negative concentration which the proportional
error model hands straight to `log`. Observed twice as a `DomainError`, at
`C ≈ −1.4e−20` during an inner line search and at `C ≈ −3.1e−22` during an outer
fit that had wandered toward the lower box bound on `log Q`.

No guard is needed and none is present — the identity is exact, not clamped.
"""
function twocomp_conc(t, CL, V1, Q, V2, input)
    k10 = CL / V1
    k12 = Q / V1
    k21 = Q / V2
    u = k10 + k12 + k21
    s = k10 + k12 - k21
    w = 4 * k12 * k21
    disc = sqrt(s * s + w)
    dps, dms = s >= 0 ? (disc + s, w / (disc + s)) : (w / (disc - s), disc - s)
    α = (u + disc) / 2
    β = (u - disc) / 2
    A = dps / (2 * disc)
    B = dms / (2 * disc)
    return (input / V1) * (A * exp(-α * t) + B * exp(-β * t))
end

"""
    twocomp_params(m, η, θ) -> (CL, V1, Q, V2, ω, σ)

Unpack θ and η into natural-scale quantities. `ω = re_scale .* exp.(θ[5:8])`.
"""
@inline function twocomp_params(m::TwoComp, η, θ)
    CL = exp(θ[1] + η[1])
    V1 = exp(θ[2] + η[2])
    Q  = exp(θ[3] + η[3])
    V2 = exp(θ[4] + η[4])
    ω  = ntuple(k -> m.re_scale * exp(θ[4 + k]), 4)
    σ  = exp(θ[9])
    return CL, V1, Q, V2, ω, σ
end

predict(m::TwoComp, η, θ, subj::SubjectData) =
    let (CL, V1, Q, V2, _, _) = twocomp_params(m, η, θ)
        [twocomp_conc(t, CL, V1, Q, V2, subj.input) for t in subj.times]
    end

# ---------------------------------------------------------------------------
# Machine-independent work accounting (off by default)
# ---------------------------------------------------------------------------
# The mirror of `count_ode` in the Michaelis-Menten example. There the currency
# was ODE right-hand-side evaluations; here the structural model is closed form,
# so the currency is evaluations of `h` itself. Both are weighted by the width of
# the number type, because an evaluation carrying `Dual{10,4,4}` costs far more
# than one carrying `Float64` and counting them equally would rank the most
# expensive method cheapest.
#
# Gated on a plain `Bool` that is false unless a study switches it on, so the
# test suite and every timing run pay a single predictable branch and nothing
# else, and the derivative propagation never sees it.

const H_EVALS        = Threads.Atomic{Int}(0)
const H_EVALS_SCALAR = Threads.Atomic{Int}(0)
const COUNT_H        = Ref(false)

h_width(::Type{<:Real}) = 1
h_width(::Type{ForwardDiff.Dual{T,V,N}}) where {T,V,N} = (N + 1) * h_width(V)

"Reset the counters and count `h` evaluations for the duration of `f`."
function count_h(f)
    Threads.atomic_xchg!(H_EVALS, 0); Threads.atomic_xchg!(H_EVALS_SCALAR, 0)
    COUNT_H[] = true
    try
        f()
    finally
        COUNT_H[] = false
    end
    return (n = H_EVALS[], n_scalar = H_EVALS_SCALAR[])
end

function negative_log_joint_density(m::TwoComp, η, θ, subj::SubjectData)
    CL, V1, Q, V2, ω, σ = twocomp_params(m, η, θ)
    T = promote_type(eltype(η), eltype(θ))
    if COUNT_H[]
        Threads.atomic_add!(H_EVALS, 1)
        Threads.atomic_add!(H_EVALS_SCALAR, h_width(T))
    end
    h = zero(T)
    half_log2π = log(2π) / 2

    @inbounds for j in eachindex(subj.times)
        C = twocomp_conc(subj.times[j], CL, V1, Q, V2, subj.input)
        sd = m.error_model === :proportional ? σ * C : σ
        r = subj.observations[j] - C
        h += half_log2π + log(sd) + r * r / (2 * sd * sd)
    end

    @inbounds for k in 1:4
        h += half_log2π + log(ω[k]) + η[k] * η[k] / (2 * ω[k] * ω[k])
    end
    return h
end

"Default generating parameters: CL 2, V1 30, Q 5, V2 80, all ω 0.2, σ 0.2."
const TWOCOMP_THETA_TRUE =
    [log(2.0), log(30.0), log(5.0), log(80.0),
     log(0.2), log(0.2), log(0.2), log(0.2), log(0.2)]

const TWOCOMP_TIMES = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 30.0, 40.0, 50.0, 60.0]
const TWOCOMP_INPUT = 400.0

"Wide but finite boxes on the log scale, for the box-constrained outer optimizer."
twocomp_bounds() = (fill(-6.0, 9), fill(6.0, 9))
