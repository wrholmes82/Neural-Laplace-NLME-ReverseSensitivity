# Seeded data generation. Shared by test/ and examples/ so neither duplicates it
# nor reaches sideways into the other.

using Random
using LinearAlgebra
using NLMEReverse

"""
    simulate_twocomp(model=TwoComp(); θ, N, times, input, seed) -> (pop, θ, η_true)

Two-compartment data at the generating parameters. `η_true` is returned so
tests can compare recovered empirical Bayes estimates against what generated the
data.
"""
function simulate_twocomp(model::TwoComp = TwoComp();
                          θ::AbstractVector = TWOCOMP_THETA_TRUE,
                          N::Int = 50,
                          times::AbstractVector = TWOCOMP_TIMES,
                          input::Real = TWOCOMP_INPUT,
                          seed::Int = 12345)
    rng = MersenneTwister(seed)
    ω = [model.re_scale * exp(θ[4 + k]) for k in 1:4]
    σ = exp(θ[9])
    subjects = SubjectData[]
    ηs = Vector{Vector{Float64}}(undef, N)

    for i in 1:N
        η = ω .* randn(rng, 4)
        ηs[i] = η
        CL = exp(θ[1] + η[1]); V1 = exp(θ[2] + η[2])
        Q  = exp(θ[3] + η[3]); V2 = exp(θ[4] + η[4])
        C = [twocomp_conc(t, CL, V1, Q, V2, input) for t in times]
        y = model.error_model === :proportional ?
            C .+ σ .* C .* randn(rng, length(C)) :
            C .+ σ .* randn(rng, length(C))
        push!(subjects, SubjectData(i, times, y, input))
    end
    return PopulationData(subjects), collect(Float64, θ), ηs
end

"""
    simulate_linear_gaussian(; nβ, q, N, n_obs, θ, seed) -> (model, pop, θ, η_true)

Design matrices are drawn once per subject from the seeded RNG; the first column
of `X` is an intercept so the fit is identifiable.
"""
function simulate_linear_gaussian(; nβ::Int = 2, q::Int = 2, N::Int = 12,
                                  n_obs::Int = 6,
                                  θ::Union{Nothing,AbstractVector} = nothing,
                                  seed::Int = 4321)
    rng = MersenneTwister(seed)
    m = LinearGaussian(nβ, q)
    θtrue = θ === nothing ?
        vcat(collect(range(1.0, 2.0; length = nβ)),
             fill(log(0.5), q), [log(0.3)]) : collect(Float64, θ)

    β = θtrue[1:nβ]
    ω = exp.(θtrue[nβ+1:nβ+q])
    σ = exp(θtrue[end])

    subjects = SubjectData[]
    ηs = Vector{Vector{Float64}}(undef, N)
    for i in 1:N
        X = hcat(ones(n_obs), randn(rng, n_obs, nβ - 1))
        Z = randn(rng, n_obs, q)
        η = ω .* randn(rng, q)
        ηs[i] = η
        y = X * β .+ Z * η .+ σ .* randn(rng, n_obs)
        push!(subjects, SubjectData(i, collect(1.0:n_obs), y, 0.0;
                                    covariates = (X = X, Z = Z)))
    end
    return m, PopulationData(subjects), θtrue, ηs
end

"""
    perturb(θ, scale; seed) -> θ′

Seeded multiplicative perturbation on the log scale, for recovery tests and for
matched multi-start sets.
"""
function perturb(θ::AbstractVector, scale::Real; seed::Int = 999)
    rng = MersenneTwister(seed)
    return collect(Float64, θ) .+ scale .* randn(rng, length(θ))
end

"""
    matched_starts(θ, M, scale; seed) -> Vector{Vector{Float64}}

M initial points in the transformed (log) space, identical across gradient
methods — the reference manuscript's matched-start protocol.
"""
function matched_starts(θ::AbstractVector, M::Int, scale::Real; seed::Int = 2024)
    rng = MersenneTwister(seed)
    return [collect(Float64, θ) .+ scale .* randn(rng, length(θ)) for _ in 1:M]
end
