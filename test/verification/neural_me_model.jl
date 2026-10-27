# Neural mixed-effects model — the test problem for the reverse-mode study.
#
# Domain: `Alternative_Domains.md` §5, neural / latent-variable mixed effects, the
# extreme large-p regime.  A neural network maps a subject's covariates to that
# subject's TYPICAL structural parameters; the random effects are inter-individual
# variability on those parameters:
#
#     φᵢ = g(xᵢ ; W) + ηᵢ ,        ηᵢ ~ N(0, Ω),  Ω diagonal
#     yᵢⱼ = m(tᵢⱼ ; φᵢ) + εᵢⱼ ,    ε ~ N(0, σ²)
#
#     m(t; φ) = Σ_{k=1..K} exp(a_k) · exp( −exp(r_k)·t ) ,   φ = (a₁,r₁,…,a_K,r_K)
#
# so q = 2K.  This is the published neural-NLME architecture: a network producing
# individual structural parameters inside a marginal-likelihood mixed-effects fit
# (Janssen et al. 2024, mixed-effect deep compartment model, J. Pharmacokinet.
# Pharmacodyn. 51(6):797–808; DeepNLME / DeepPumas, Martensen et al. 2024).
#
# Why this and not the pharmacometric examples already in the repo:
#
#   * p is the network's weight count, so it sweeps from 10² to 10⁴ while q stays small.
#     `Alternative_Domains.md` §1: the benefit grows with p, not with q.
#   * ∇²_ηθ h is FULLY DENSE — every weight influences every subject's h — so a forward
#     method cannot exploit sparsity to dodge its p right-hand sides.  This is what
#     separates the domain from IRT, where B is block-sparse and a structure-aware
#     forward method recovers most of the gap.
#   * one evaluation of h is itself O(p), so forward sensitivity is O(p²) against the
#     reverse construction's O(p).
#   * h is genuinely nonlinear in η — η enters through exp() and through the exponential
#     decay — so ∇²_ηη h depends on η and the Laplace step is not trivial.  Same
#     situation as the two-compartment Michaelis-Menten example, unlike a linear
#     Gaussian mixed model where Laplace is exact but degenerate.
#
# Parameter layout, one flat vector so that p == length(θ) literally:
#
#     θ = [ vec(W₁), b₁, …, vec(W_L), b_L , log σ , log ω₁ … log ω_q ]
#
# Ω is diagonal deliberately: it keeps p attributable to the network alone, so the
# p-sweep varies exactly one thing.
#
# The one-method contract of `NLMEReverse` is respected exactly: this file defines
# `negative_log_joint_density(model, η, θ, subject)` and nothing else is required.  The
# function is non-mutating and free of `try`/`catch` so that it is differentiable both
# by ForwardDiff at arbitrary Dual nesting depth and by a reverse-mode engine.

using LinearAlgebra
using Random
using NLMEReverse

"""
    NeuralMEModel(; q, d_x, width, depth)

Neural mixed-effects model.  `q` must be even (`q = 2K` exponential terms).
"""
struct NeuralMEModel <: NLMEModel
    q::Int
    d_x::Int
    width::Int
    depth::Int
    sizes::Vector{Int}     # layer widths, input to output
    nw::Int                # number of network weights and biases
end

function NeuralMEModel(; q::Int = 4, d_x::Int = 3, width::Int = 16, depth::Int = 1)
    iseven(q) || throw(ArgumentError("q = 2K exponential terms; q must be even, got $q"))
    depth >= 1 || throw(ArgumentError("depth must be at least 1, got $depth"))
    sizes = [d_x; fill(width, depth); q]
    nw = sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in 1:(length(sizes) - 1))
    return NeuralMEModel(q, d_x, width, depth, sizes, nw)
end

NLMEReverse.n_random_effects(m::NeuralMEModel) = m.q

# `sizes` and `nw` are STORED, not recomputed.  Both are needed on every call to
# `negative_log_joint_density`, which is the innermost loop of the whole study: computing
# them there would allocate a fresh tuple or vector per evaluation, and a runtime-length
# `ntuple` would leave the layer count un-inferable — the kind of thing that both slows
# the hot loop and gives a reverse-mode engine a reason to reject the function.
"""Layer widths, input to output."""
layer_sizes(m::NeuralMEModel) = m.sizes

"""Number of network weights and biases."""
n_weights(m::NeuralMEModel) = m.nw

"""Total population-parameter count `p = n_weights + 1 (log σ) + q (log ω)`."""
n_parameters(m::NeuralMEModel) = m.nw + 1 + m.q

# --------------------------------------------------------------------------
# Structural model
# --------------------------------------------------------------------------

"""
    network(m, θ, x) -> φ_typical

tanh MLP from covariates to the subject's typical structural parameters.

Written with `reshape`d views of θ and no scratch buffers, so it is generic over any
element type θ carries — `Float64`, `ForwardDiff.Dual` at any nesting depth, or a
reverse-mode engine's tracked type.
"""
function network(m::NeuralMEModel, θ, x)
    s = layer_sizes(m)
    z = x
    k = 0
    @inbounds for i in 1:(length(s) - 1)
        n_in, n_out = s[i], s[i + 1]
        W = reshape(view(θ, (k + 1):(k + n_in * n_out)), n_in, n_out)
        k += n_in * n_out
        b = view(θ, (k + 1):(k + n_out))
        k += n_out
        z = transpose(W) * z .+ b
        if i < length(s) - 1
            z = tanh.(z)
        end
    end
    return z
end

"""
    structural(t, φ, K) -> m(t; φ)

`Σ_k exp(a_k) exp(−exp(r_k) t)` with amplitudes and rates on the log scale, so the
prediction is positive and monotone for every real φ — no branch, no domain error,
total for every η the inner line search can probe.

The exponent clamps exist only so that an absurd η probed during backtracking returns a
finite number instead of `Inf`; they never bind at any η the mode solver returns.
"""
function structural(t, φ, K::Int)
    acc = zero(eltype(φ)) * zero(eltype(t))
    @inbounds for k in 1:K
        a = clamp(φ[2k - 1], -30, 30)
        r = clamp(φ[2k], -30, 30)
        acc += exp(a) * exp(-clamp(exp(r) * t, 0, 700))
    end
    return acc
end

"""Structural prediction at the subject's observation times — diagnostics only."""
function NLMEReverse.predict(m::NeuralMEModel, η, θ, subject)
    φ = network(m, θ, subject.covariates.x) .+ η
    K = m.q ÷ 2
    return [structural(t, φ, K) for t in subject.times]
end

# --------------------------------------------------------------------------
# The one required method
# --------------------------------------------------------------------------

"""
    negative_log_joint_density(model, η, θ, subject)

`hᵢ(η,θ) = −log p(yᵢ|η,θ) − log p(η|θ)`, including every normalising constant, so that
`Lᵢ = hᵢ + ½ logdet H̃ᵢ − (q/2)log 2π` is directly comparable with a log marginal
likelihood.
"""
function NLMEReverse.negative_log_joint_density(m::NeuralMEModel, η, θ, subject)
    nw = n_weights(m)
    log_σ = θ[nw + 1]
    σ = exp(log_σ)

    φ = network(m, θ, subject.covariates.x) .+ η
    K = m.q ÷ 2

    # `eltype(φ)` rather than `promote_type(eltype(η), eltype(θ))`: φ is already the
    # product of that promotion, so using it avoids a second, separate promotion call.
    # That matters under a tape-based reverse engine, where `promote_type(Dual, TrackedReal)`
    # is an ambiguity site — one fewer place for the nesting to go wrong.
    T = eltype(φ)
    ss = zero(T)
    @inbounds for j in eachindex(subject.times)
        r = (subject.observations[j] - structural(subject.times[j], φ, K)) / σ
        ss += r * r
    end
    n_obs = length(subject.times)
    nll_obs = 0.5 * ss + n_obs * (log_σ + 0.5 * log(2π))

    nll_η = zero(T)
    @inbounds for k in 1:m.q
        lω = θ[nw + 1 + k]
        e = η[k] * exp(-lω)
        nll_η += 0.5 * e * e + lω
    end
    nll_η += 0.5 * m.q * log(2π)

    return nll_obs + nll_η
end

# --------------------------------------------------------------------------
# Ground truth and synthetic data
# --------------------------------------------------------------------------

"""
    truth_theta(model; seed, sigma, omega) -> Vector{Float64}

A fixed ground-truth parameter vector.

Weights are Glorot-scaled so the network output is O(1) and the tanh units are not
saturated.  The OUTPUT BIASES are then set to fixed, separated values rather than random
ones:

    amplitude a_k → log(1/K)        rate r_k → log(0.5 · 4^(k−1))

A sum of exponentials with randomly coincident rates is numerically degenerate — two
equal rates make the model rank-deficient in φ — which would show up as an
ill-conditioned `H` and contaminate the timing study with conditioning effects that have
nothing to do with gradient orientation.  Separating the rates by a factor of four per
term keeps `H` well conditioned at the truth while leaving the model genuinely nonlinear
in η.
"""
function truth_theta(m::NeuralMEModel; seed::Int = 20260729,
                     sigma::Float64 = 0.05, omega::Float64 = 0.30)
    rng = MersenneTwister(seed)
    s = layer_sizes(m)
    θ = Float64[]
    K = m.q ÷ 2
    for i in 1:(length(s) - 1)
        n_in, n_out = s[i], s[i + 1]
        scale = sqrt(2.0 / n_in) * (i == length(s) - 1 ? 0.5 : 1.0)
        append!(θ, scale .* randn(rng, n_in * n_out))
        if i == length(s) - 1
            b = Vector{Float64}(undef, n_out)
            for k in 1:K
                b[2k - 1] = log(1.0 / K)
                b[2k]     = log(0.5 * 4.0^(k - 1))
            end
            append!(θ, b)
        else
            append!(θ, 0.1 .* randn(rng, n_out))
        end
    end
    push!(θ, log(sigma))
    append!(θ, fill(log(omega), m.q))
    return θ
end

"""
    simulate(model, θ_true, n_subjects, n_obs; seed) -> (PopulationData, etas)

Synthetic data at the ground truth.  Observation times are geometric on [0.05, 6] so the
fast and slow exponential terms are both informed.
"""
function simulate(m::NeuralMEModel, θ_true::Vector{Float64}, n_subjects::Int,
                  n_obs::Int; seed::Int = 20260730)
    rng = MersenneTwister(seed)
    nw = n_weights(m)
    σ = exp(θ_true[nw + 1])
    ω = exp.(θ_true[(nw + 2):(nw + 1 + m.q)])
    K = m.q ÷ 2

    times = collect(exp.(range(log(0.05), log(6.0); length = n_obs)))
    # Built by comprehension so the vector's element type is concrete: a
    # `Vector{SubjectData}` of the abstract parametric type would send every call to
    # `negative_log_joint_density` in the hot loop through dynamic dispatch, adding a
    # constant to every timing in the study.
    etas = Vector{Vector{Float64}}(undef, n_subjects)
    raw = Vector{Any}(undef, n_subjects)

    for i in 1:n_subjects
        x = randn(rng, m.d_x)
        η = ω .* randn(rng, m.q)
        φ = network(m, θ_true, x) .+ η
        y = [structural(t, φ, K) + σ * randn(rng) for t in times]
        raw[i] = SubjectData(i, times, y, 0.0; covariates = (x = x,))
        etas[i] = η
    end
    subjects = [r for r in raw]      # concrete eltype
    return PopulationData(subjects), etas
end

"""Start point away from the truth, matching the repo's `perturb` convention."""
function perturb(θ::Vector{Float64}, rel::Float64; seed::Int = 77)
    rng = MersenneTwister(seed)
    return θ .* (1 .+ rel .* randn(rng, length(θ)))
end
