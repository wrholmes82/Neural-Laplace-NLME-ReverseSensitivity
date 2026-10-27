# Linear-Gaussian analytic oracle:  y_i = X_i β + Z_i η_i + ε_i.
#
# Not for realism. Because h_i is exactly quadratic in η,
# three things hold *exactly*, which is what makes this the highest-value test
# problem in the suite:
#
#   1. The Laplace approximation is exact — the marginal is N(Xβ, ZΩZᵀ + σ²I) in
#      closed form, giving a ground-truth objective value.
#   2. Differentiating that closed form gives a ground-truth gradient.
#   3. By Prop. 1 (reference manuscript §S1b), H is η-independent, so
#      ∇_η logdet H ≡ 0 and hence
#      b = 0, λ = 0 and the adjoint correction term c = 0 identically.

using LinearAlgebra
using ForwardDiff
using NLMEReverse
import NLMEReverse: negative_log_joint_density, predict, n_random_effects

"""
    LinearGaussian(nβ, q)

`θ = (β₁..β_nβ, log ω₁..log ω_q, log σ)`, so `p = nβ + q + 1`. Ω is diagonal.

Each subject carries its design matrices in `covariates.X` (n×nβ) and
`covariates.Z` (n×q).
"""
struct LinearGaussian <: NLMEModel
    q::Int
    p::Int
    nβ::Int
end

LinearGaussian(nβ::Int, q::Int) = LinearGaussian(q, nβ + q + 1, nβ)

@inline function lg_params(m::LinearGaussian, θ)
    β = @view θ[1:m.nβ]
    ω = ntuple(k -> exp(θ[m.nβ + k]), m.q)
    σ = exp(θ[m.nβ + m.q + 1])
    return β, ω, σ
end

predict(m::LinearGaussian, η, θ, subj::SubjectData) =
    let (β, _, _) = lg_params(m, θ)
        subj.covariates.X * β .+ subj.covariates.Z * η
    end

function negative_log_joint_density(m::LinearGaussian, η, θ, subj::SubjectData)
    β, ω, σ = lg_params(m, θ)
    X = subj.covariates.X
    Z = subj.covariates.Z
    y = subj.observations
    n = length(y)
    T = promote_type(eltype(η), eltype(θ))

    h = zero(T)
    @inbounds for j in 1:n
        μ = zero(T)
        for l in 1:m.nβ
            μ += X[j, l] * β[l]
        end
        for k in 1:m.q
            μ += Z[j, k] * η[k]
        end
        r = y[j] - μ
        h += r * r / (2 * σ * σ)
    end
    h += n * (log(2π) / 2 + log(σ))

    @inbounds for k in 1:m.q
        h += log(2π) / 2 + log(ω[k]) + η[k] * η[k] / (2 * ω[k] * ω[k])
    end
    return h
end

"""
    lg_marginal_nll(m, θ, subj)

`−log p(yᵢ | θ)` in closed form: `y ~ N(Xβ, ZΩZᵀ + σ²I)`.

This is the ground truth for the objective. Because `h` is exactly quadratic in η,
`∫ e^{−h} dη = e^{−h(η̂)} (2π)^{q/2} / √det H`, so the implemented Laplace
expression `h(η̂) + ½ logdet H̃ − (q/2) log 2π` equals this to machine precision
(up to the `reg` shift in `H̃`).
"""
function lg_marginal_nll(m::LinearGaussian, θ, subj::SubjectData)
    β, ω, σ = lg_params(m, θ)
    X = subj.covariates.X
    Z = subj.covariates.Z
    y = subj.observations
    n = length(y)
    T = promote_type(eltype(θ), Float64)

    Ω = Diagonal(collect(T, ω) .^ 2)
    V = Z * Ω * transpose(Z) + (σ * σ) * I
    r = y .- X * β
    Vs = Symmetric(Matrix(V))
    return (n * log(2π) + logdet(Vs) + dot(r, Vs \ r)) / 2
end

"Population closed-form `−log p(y | θ) = Σᵢ −log p(yᵢ | θ)`."
lg_marginal_nll(m::LinearGaussian, θ, pop::PopulationData) =
    sum(lg_marginal_nll(m, θ, s) for s in pop.subjects)

"""
    lg_marginal_gradient(m, θ, pop)

Ground-truth gradient: ForwardDiff applied to the **closed-form marginal**.

This shares no code path with the objective, the inner solve or the adjoint
machinery — it never forms `h`, never solves for a mode and never touches `H̃` —
so agreement with the adjoint gradient is genuinely independent evidence rather
than a restatement.
"""
lg_marginal_gradient(m::LinearGaussian, θ, pop::PopulationData) =
    ForwardDiff.gradient(t -> lg_marginal_nll(m, t, pop), collect(Float64, θ))

lg_bounds(m::LinearGaussian) = (fill(-10.0, m.p), fill(10.0, m.p))
