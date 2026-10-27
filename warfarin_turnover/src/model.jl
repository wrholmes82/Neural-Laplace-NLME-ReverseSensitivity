# model.jl — the warfarin turnover model as an `NLMEReverse.NLMEModel`.
#
# Vehicle: Huth et al. (2026), NoLimits.jl, arXiv:2606.24427, §6.2 — warfarin turnover PD
# with a neural concentration–effect function, adopted essentially verbatim and then scaled.
#
#   PK   C_i(t) = Bateman(D_i, ka_i, CL_i, V_i, t)                    closed form, no solve
#   PD   dR_i/dt = k_out,i ( R_0,i − R_i(t)(1 + E(C_i(t))) ),  R_i(0) = R_0,i
#
# The endpoint is PCA (prothrombin complex activity), not INR — resolved against the data
# file, see `docs/DECISIONS.md`.  PCA falls when the drug acts, so `E > 0` suppressing `R` is
# the correct sign as written above.
#
# Two required methods, both in the fixed argument order `(model, η, θ, subject)`:
#   `negative_log_joint_density` — the one method the whole framework differentiates
#   `conditional_moments`        — the decomposition the FOCE curvature needs
# Both are total, non-mutating in `(η, θ)`, free of `try`/`catch`, and free of any branch on
# a parameter value.
#
# --------------------------------------------------------------------------------
# θ layout — fixed, and the only place it is written down
# --------------------------------------------------------------------------------
#
#   index                       quantity                       scale
#   1                           ka         (absorption)        log
#   2                           CL         (clearance)         log
#   3                           V          (volume)            log
#   4                           R_0        (PD baseline)       log
#   5                           k_out      (turnover rate)     log
#   6 : 5+ne                    effect parameters              log (Emax) / unconstrained (MLP)
#   6+ne : 10+ne                ω₁..ω₅     (random-effect SDs)  log
#   11+ne                       σ_prop     (PK, proportional)  log
#   12+ne                       σ_add      (PD, additive)      log
#
#   ne = 2 for `EmaxEffect`, ne = W² + 4W + 1 for `MLPEffect(W)`;  p = 12 + ne.
#
# `q = 5`: log-normal random effects on ka, CL, V, R_0, k_out, with a **diagonal Ω**, so `p`
# is inflated only by the network and every cost result is attributable to one cause.

using NLMEReverse
using LinearAlgebra
using ForwardDiff

"""
    WarfarinTurnover(effect; n_seg = 48, gamma = 1.8)

The model.  `effect` is `EmaxEffect()` (the reference form `M_par`, and the
data-generating truth) or `MLPEffect(W)` (the neural form `M_nn(W)`).

`n_seg` and `gamma` describe the RK4 grid and belong to the model rather than to the data
because they are part of the *objective's definition*: two models with different grids
define different (both legitimate) objectives, and every arm of every comparison must share
one.  They are recorded on every artefact for that reason.

The defaults `n_seg = 32, gamma = 2.2` were chosen by measurement, not by taste, and the
measurement is at the **fitted truth** and the **top dose level** — where the effect is
strongest and the ODE works hardest — not at the starting values. Worst error in the predicted
response over 15 random η at all three doses, against an `n_seg = 1536` reference:

| `n_seg` | stage evaluations | max rel. err | max abs. err | as a fraction of σ_add = 7.03 |
|---:|---:|---:|---:|---:|
| 24 | 69 | 3.4e-4 | 1.4e-2 | 2.0e-3 |
| **32** | **85** | **1.6e-4** | **5.6e-3** | **7.9e-4** |
| 48 | 117 | 2.3e-5 | 1.3e-3 | 1.8e-4 |
| 64 | 149 | 8.8e-6 | 4.1e-4 | 5.9e-5 |
| 96 | 213 | 1.7e-6 | 8.4e-5 | 1.2e-5 |

Observed convergence order 4, as RK4 requires. `gamma = 2.2` beats 1.8 by ~5× at equal node
count because the concentration peaks near 10 h and decays with a 40 h half-life, so the mesh
should be fine over the first day and coarse after it; uniform grading is far worse than either.

`n_seg = 32` moves the response by **1/1300th of the residual SD** and costs 85 stage
evaluations against 149 at `n_seg = 64` — a 1.75× saving on the objective, on every gradient
arm and on every fit in the study. The corresponding OFV shift is 4.4e-8 relative, against a
scale on which a meaningful ΔOFV is 3.84. The E1 gate measures that OFV shift directly rather
than relying on this table.
"""
struct WarfarinTurnover{E} <: NLMEModel
    effect::E
    q::Int
    p::Int
    n_seg::Int
    gamma::Float64
end

function WarfarinTurnover(effect; n_seg::Int = 32, gamma::Float64 = 2.2)
    ne = n_effect_parameters(effect)
    return WarfarinTurnover(effect, 5, 12 + ne, n_seg, Float64(gamma))
end

n_parameters(m::WarfarinTurnover) = m.p
n_effect_parameters(m::WarfarinTurnover) = n_effect_parameters(m.effect)

"""Index ranges into θ.  Written once; every consumer asks here."""
struct ThetaIndex
    fixed::UnitRange{Int}     # 1:5   — log ka, CL, V, R_0, k_out
    effect::UnitRange{Int}    # the effect parameters
    omega::UnitRange{Int}     # 5 log ω
    sigma_prop::Int
    sigma_add::Int
end

function theta_index(m::WarfarinTurnover)
    ne = n_effect_parameters(m)
    return ThetaIndex(1:5, 6:(5+ne), (6+ne):(10+ne), 11 + ne, 12 + ne)
end

# ---------------------------------------------------------------------------
# The structural prediction
# ---------------------------------------------------------------------------

"""
    structural(model, η, θ, subject) -> (pk, pd)

Concentrations at the PK sampling times and response at the PD sampling times, for one
subject.

The whole reason the network is evaluated in one batch lives here: `a(t)` is assembled at
every RK4 stage time *before* the march starts, because the turnover right-hand side depends
on time only through `E(C(t))` and never on the state.
"""
function structural(m::WarfarinTurnover, η::AbstractVector, θ::AbstractVector, subject)
    ix = theta_index(m)
    cv = subject.covariates

    ka = exp(θ[1] + η[1])
    CL = exp(θ[2] + η[2])
    V = exp(θ[3] + η[3])
    R0 = exp(θ[4] + η[4])
    kout = exp(θ[5] + η[5])

    dose = subject.input
    φ = @view θ[ix.effect]

    pk = bateman(dose, ka, CL, V, cv.pk_times)

    Cstage = bateman(dose, ka, CL, V, cv.stage_times)
    a = kout .* (1 .+ effect(m.effect, φ, Cstage))
    pd = turnover_solve(R0, kout, a, cv.grid, cv.obs_idx, length(cv.pd_times))

    return (pk = pk, pd = pd)
end

# ---------------------------------------------------------------------------
# The two required interface methods
# ---------------------------------------------------------------------------

"""
    conditional_moments(model, η, θ, subject) -> (f, R, Ω)

`f` — the structural predictions at every observation, PK first then PD;
`R` — the residual variances at those observations, **proportional on `C`** (`(σ_p·f)²`) and
      **additive on the response** (`σ_a²`), which is the `nlmixr2` warfarin convention;
`Ω` — the diagonal of the random-effect covariance.

`R` depends on `η` through the PK predictions, so the Gauss-Newton curvature assembled from
this is FOCE *with interaction*.  The package handles that with no change: `foce_curvature`
evaluates `R` at the mode and differentiates only `f`, which is the standard FOCE-I curvature
and — this is the point — is the exact curvature of the objective the package then
*defines*.  The gradient E1 verifies is the exact gradient of that objective, whatever one
thinks of the approximation itself.
"""
function NLMEReverse.conditional_moments(m::WarfarinTurnover, η::AbstractVector,
                                         θ::AbstractVector, subject)
    ix = theta_index(m)
    s = structural(m, η, θ, subject)
    σp = exp(θ[ix.sigma_prop])
    σa = exp(θ[ix.sigma_add])

    f = vcat(s.pk, s.pd)
    R = vcat((σp .* s.pk) .^ 2, fill(σa^2, length(s.pd)))
    Ω = exp.(2 .* @view θ[ix.omega])
    return (f = f, R = R, Ω = Ω)
end

"""
    negative_log_joint_density(model, η, θ, subject)

`h_i(η,θ) = −log p(yᵢ|η,θ) − log p(η|θ)`, **including** every normalising constant, since the
objective is reported as an OFV on the standard scale and a dropped `log 2π` would move it.
"""
function NLMEReverse.negative_log_joint_density(m::WarfarinTurnover, η::AbstractVector,
                                                θ::AbstractVector, subject)
    cm = NLMEReverse.conditional_moments(m, η, θ, subject)
    y = subject.observations
    r = y .- cm.f
    h = sum(@. 0.5 * (log(2π * cm.R) + r * r / cm.R))
    h += sum(@. 0.5 * (log(2π * cm.Ω) + η * η / cm.Ω))
    return h
end

"""
    predict(model, η, θ, subject) -> Vector

Structural prediction at every observation, in the same order as `subject.observations`.
Diagnostics only; nothing in the objective or any gradient calls it.
"""
function NLMEReverse.predict(m::WarfarinTurnover, η::AbstractVector, θ::AbstractVector, subject)
    s = structural(m, η, θ, subject)
    return vcat(s.pk, s.pd)
end

# ---------------------------------------------------------------------------
# Reading the fitted model back out
# ---------------------------------------------------------------------------

"""
    effect_curve(model, θ, Cs) -> Vector

`E(C)` at the given concentrations, for the population (η = 0) parameters in θ.  This is the
object E4's functionals and E6's figure are built from, and it is deliberately the *only*
way the network is ever read: individual weights are not identifiable at these widths and no
claim is made about them.
"""
effect_curve(m::WarfarinTurnover, θ::AbstractVector, Cs::AbstractVector) =
    effect(m.effect, view(θ, theta_index(m).effect), Cs)

"""
    population_parameters(model, θ) -> NamedTuple

The five typical values, the five random-effect SDs and the two residual SDs, on their
natural scales.
"""
function population_parameters(m::WarfarinTurnover, θ::AbstractVector)
    ix = theta_index(m)
    return (ka = exp(θ[1]), CL = exp(θ[2]), V = exp(θ[3]), R0 = exp(θ[4]), kout = exp(θ[5]),
            omega = exp.(θ[ix.omega]), sigma_prop = exp(θ[ix.sigma_prop]),
            sigma_add = exp(θ[ix.sigma_add]))
end
