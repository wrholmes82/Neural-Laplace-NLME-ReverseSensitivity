# starts.jl — starting values and bounds.
#
# Two rules govern everything here.
#
#   1. **The start is neutral.**  The network is initialised from a seed and a target effect
#      size, never from the truth and never from the data.  A study whose neural start was
#      pretrained against the generating curve would report a recovery that the
#      initialisation had already supplied.
#   2. **The start is reproducible.**  One named seed constant, one function, no state.

using Random
using LinearAlgebra

"""
    fixed_effect_start() -> Vector{Float64}

The five structural log-parameters, from published warfarin values rather than from this
dataset: `ka = 0.5 /h`, `CL = 0.14 L/h`, `V = 8 L` (so `t½ ≈ 40 h`), PCA baseline
`R₀ = 100`, `k_out = 0.05 /h` (response half-life ≈ 14 h).
"""
fixed_effect_start() = [log(0.5), log(0.14), log(8.0), log(100.0), log(0.05)]

"""
    variance_start() -> Vector{Float64}

Five `log ω` at 0.3, then `log σ_prop` at 0.1 (proportional, on `C`) and `log σ_add` at 5
(additive, in PCA units against a baseline of 100).
"""
variance_start() = vcat(fill(log(0.3), 5), log(0.1), log(5.0))

"""
    effect_start(e::EmaxEffect) -> Vector{Float64}

`E_max = 1`, `EC50 = 5 mg/L` — a half-maximal effect near the middle of the concentration
range the three dose levels span.
"""
effect_start(::EmaxEffect; kwargs...) = [log(1.0), log(5.0)]

# Present so `theta_start` works on a Hill model; E9 builds its truth vector explicitly instead.
effect_start(::HillEffect; kwargs...) = [log(1.0), log(5.0), log(1.0)]

"""
    effect_start(e::MLPEffect; seed, e_target, c_ref) -> Vector{Float64}

A neutral network initialisation.

Layer scales are the usual `tanh` ones (`1/√fan_in`), with two deliberate departures:

  * **First-layer biases are spread across the input range** rather than zeroed.  With a
    single scalar input, zero biases put every hidden unit's kink at `C = 0`, so the whole
    layer sees the same feature and the network starts on a near-degenerate manifold.
    Spreading them means the units tile the range the doses actually cover.
  * **The output bias is set from a target effect size**, `b₃ = log(exp(E_target/C_ref) − 1)`,
    so that `E(C_ref) ≈ E_target` at initialisation regardless of `W`.  Without this the
    softplus starts near `log 2 ≈ 0.69` and `E(C) ≈ 0.69·C`, which at therapeutic
    concentrations suppresses the response by an order of magnitude and starts the fit in a
    region the data can only argue it out of slowly.

`E_target = 0.5` at `C_ref = 10 mg/L` is a modest, plainly-not-the-truth effect: the same
initialisation is used at every width, so nothing about the `p` ladder depends on the start.
"""
function effect_start(e::MLPEffect; seed::Int = SEED_START, e_target::Float64 = 0.5,
                      c_ref::Float64 = 10.0)
    rng = MersenneTwister(seed)
    W = e.width
    xmax = 2.0                                   # C/c0 at the top of the design range

    w1 = randn(rng, W)
    b1 = -w1 .* range(0.0, xmax; length = W)     # each unit's kink at a distinct input value
    w2 = randn(rng, W * W) ./ sqrt(W)
    b2 = zeros(W)
    w3 = 0.5 .* randn(rng, W) ./ sqrt(W)
    b3 = log(expm1(e_target / c_ref))

    return vcat(w1, b1, w2, b2, w3, b3)
end

"""
    theta_start(model; kwargs...) -> Vector{Float64}

The full starting θ in the layout `model.jl` documents.
"""
function theta_start(m::WarfarinTurnover; kwargs...)
    θ = vcat(fixed_effect_start(), effect_start(m.effect; kwargs...), variance_start())
    length(θ) == m.p || error("theta_start produced $(length(θ)) entries; p = $(m.p).")
    return θ
end

"""
    theta_bounds(model) -> (lower, upper)

Sanity rails, not physical constraints.  The default optimiser is unconstrained, so
`fit_nlme` checks θ̂ against these and reports `:outside_bounds` rather than clipping — a θ̂
outside the box is a statement about the fit and hiding it would be the failure this project
refuses to tolerate.

Network weights get `±50`, wide enough that the box is never active and narrow enough that a
runaway weight is caught.
"""
function theta_bounds(m::WarfarinTurnover)
    ix = theta_index(m)
    ne = n_effect_parameters(m)
    lo = Vector{Float64}(undef, m.p)
    hi = Vector{Float64}(undef, m.p)
    lo[ix.fixed] .= -12.0
    hi[ix.fixed] .= 12.0
    lo[ix.effect] .= m.effect isa MLPEffect ? -50.0 : -12.0
    hi[ix.effect] .= m.effect isa MLPEffect ? 50.0 : 12.0
    lo[ix.omega] .= -6.0
    hi[ix.omega] .= 3.0
    lo[ix.sigma_prop] = -8.0
    hi[ix.sigma_prop] = 3.0
    lo[ix.sigma_add] = -8.0
    hi[ix.sigma_add] = 6.0
    return (lower = lo, upper = hi)
end
