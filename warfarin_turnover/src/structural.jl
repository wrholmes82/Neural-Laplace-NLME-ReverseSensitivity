# structural.jl — the deterministic parts of the warfarin turnover model.
#
# Three pieces, in dependency order:
#
#   1. Bateman PK, closed form.  No ODE solve for the PK.  This is the single most
#      important cost decision in the study: it removes the only multi-state solve and
#      leaves one scalar ODE state behind.
#   2. The effect function E(C), in its two competing forms.
#   3. The turnover ODE, marched by fixed-step RK4 on a grid fixed at data-construction
#      time.
#
# Everything here is total, non-mutating in its inputs, free of `try`/`catch`, and free of
# any branch on the *value* of a parameter or a random effect.  Two branches exist and both
# are on quantities whose two sides agree to machine precision (`_phi`, `softplus`); they
# are numerical-evaluation choices, not model choices, and the function they define is
# smooth.  Loop bounds and array indices branch on integers only.

using LinearAlgebra

# ---------------------------------------------------------------------------
# 1. Bateman PK
# ---------------------------------------------------------------------------

"""
    _phi(u) = (1 − exp(−u)) / u,  continued analytically to u = 0 where it equals 1.

The removable singularity in the Bateman equation, isolated.  Writing the absorption
profile as `C ∝ t·exp(−ke·t)·_phi((ka−ke)·t)` keeps it finite and smooth as `ka → ke`,
which the textbook form `(exp(−ke t) − exp(−ka t))/(ka − ke)` does not.

Warfarin has `ka/ke ≈ 50`, so the singular region is never visited in practice — but the
random effects on `ka`, `CL` and `V` are independent and the outer line search probes θ
freely, so "never in practice" is not a guarantee the objective can be built on.

The series branch is taken for `|u| < 1e-4`, where the first omitted term is `u⁴/120 ≈
8e-19` relative — below the double-precision floor.  Both branches therefore define the
same function to machine precision, in value and in every derivative, so the switch
introduces no kink for the optimiser or for AD to trip over.
"""
@inline function _phi(u::T) where {T}
    if abs(u) < 1e-4
        return one(T) - u / 2 + u * u / 6 - u * u * u / 24
    end
    return -expm1(-u) / u
end

"""
    bateman(dose, ka, CL, V, t) -> C(t)

Single oral dose, first-order absorption, one compartment (Huth §6.1):

    C(t) = (D·ka) / (V·(ka − ke)) · (exp(−ke·t) − exp(−ka·t)),   ke = CL/V

evaluated as `C(t) = (D·ka/V) · t · exp(−k_min·t) · _phi((k_max − k_min)·t)`, where `k_min`
and `k_max` are the smaller and larger of `ka` and `ke`.  The identity is exact:

    (exp(−k_min·t) − exp(−k_max·t)) / (k_max − k_min) = t·exp(−k_min·t)·_phi((k_max−k_min)·t)

and the bracketed expression is symmetric in `ka` and `ke`, so ordering them changes nothing
about the function.

Ordering them changes a great deal about the arithmetic. Using `k_max` in the prefactor makes
the remaining factor `exp(+(k_max−k_min)·t)`, which at `ka = 2 /h` and `t = 168 h` is
`exp(333)` — an overflow to `Inf`, multiplied by an underflow to `0`, giving `NaN` at a
perfectly ordinary parameter value. With `k_min` in the prefactor every exponential argument
is `≤ 0` and the expression is bounded everywhere.

`C(0) = 0` exactly, which is what makes `E(0) = 0` the baseline condition of the turnover
model rather than an approximation.
"""
@inline function bateman(dose, ka, CL, V, t)
    ke = CL / V
    kmin = min(ka, ke)
    kmax = max(ka, ke)
    return (dose * ka / V) * t * exp(-kmin * t) * _phi((kmax - kmin) * t)
end

"""
    bateman(dose, ka, CL, V, times::AbstractVector) -> Vector

Vectorised over times.  Allocating and non-mutating: the reverse engine tapes it whole.
"""
bateman(dose, ka, CL, V, times::AbstractVector) =
    [bateman(dose, ka, CL, V, t) for t in times]

# ---------------------------------------------------------------------------
# 2. The effect function E(C)
# ---------------------------------------------------------------------------

"""
    softplus(x) = log(1 + exp(x)), evaluated stably.

`log1p(exp(−|x|)) + max(x, 0)` is the same function as `log1p(exp(x))` — algebraically, not
approximately — and does not overflow for large `x`.  The `max` is therefore not a branch
on a modelling quantity: both sides give the same value and the same derivative at `x = 0`.
"""
@inline softplus(x) = log1p(exp(-abs(x))) + max(x, zero(x))

"""
    EmaxEffect()

`E(C) = E_max · C / (EC50 + C)`.  The reference form and the data-generating truth
(`M_par` in the plan).  Two parameters, both carried on the log scale.
"""
struct EmaxEffect end

n_effect_parameters(::EmaxEffect) = 2

"""
    HillEffect()

`E(C) = E_max · C^γ / (EC50^γ + C^γ)`, carried as `(log E_max, log EC50, log γ)`.

A generating truth that `EmaxEffect` cannot represent, used for **simulation only** and never
fitted. It is the minimal such truth: `EmaxEffect` is the special case `γ = 1`, so simulating
from it tests one extra shape parameter rather than an adversarially chosen curve.

Simulation-only matters numerically. Differentiating `C^γ` with respect to `γ` introduces
`C^γ log C`, which at `C = 0` is `0 · (-Inf)` and evaluates to `NaN` rather than to its limit of
zero. Forward evaluation is clean, since `0^γ = 0` for `γ > 0`, so this is safe as a truth; were
it ever fitted, that branch would need guarding.
"""
struct HillEffect end

n_effect_parameters(::HillEffect) = 3

"""
    MLPEffect(width; c0)

`E(C) = C · ζ( MLP_W(C / c₀) )`, with `ζ = softplus` and `MLP_W : 1 → W → W → 1` with
`tanh` activations.  Huth's scaffolding, kept verbatim, and the three pieces are all
load-bearing:

  * multiplying by `C` enforces `E(0) = 0`, so the turnover baseline is stationary by
    construction and no boundary constraint is needed;
  * `softplus` enforces `E ≥ 0`, so the drug cannot spuriously *raise* prothrombin complex
    activity;
  * the input is normalised by a **fixed** constant `c₀`, which is what keeps the network's
    input on an O(1) scale without making the normalisation itself a fitted quantity.

Dropping any of them reintroduces the dead-unit problem that had to be hand-patched
elsewhere in this literature.  They are not simplifiable.

Parameter count `nw = W² + 4W + 1`: `W` first-layer weights + `W` biases, `W²` + `W` for the
hidden layer, `W` + `1` for the output.  W = 4 → 33, W = 10 → 141, W = 32 → 1153,
W = 100 → 10401.
"""
struct MLPEffect
    width::Int
    c0::Float64
end

MLPEffect(width::Int; c0::Float64 = 5.0) = MLPEffect(width, Float64(c0))

n_effect_parameters(e::MLPEffect) = e.width^2 + 4 * e.width + 1

"""
    mlp_batch(w, x, width) -> Vector

`MLP_W` applied to a whole vector of inputs in one pass: two `width × n` broadcasts and one
`width × width` by `width × n` product.

Batching is not cosmetic here.  The turnover right-hand side depends on time only through
`C(t)`, never on the state `R`, so every network evaluation an RK4 march will ever need is
known before the march starts.  Evaluating them one at a time turns the hidden layer into
`n` separate matrix–vector products; evaluating them together makes it one matrix–matrix
product, which at `W = 100` is the difference between a feasible fit and an infeasible one.

The hidden-layer weights are reshaped through a `view`, not a copy. A copy would be the right
call if it bought a BLAS `gemm`, but it does not: `H1` carries `Dual`s on every derivative path
(η-Jacobians, the inner Hessian) and Mooncake's tracked type on the reverse sweep, so the
product goes through the generic fallback either way. The copy would only add `width²` doubles
per call and one more node for the reverse engine to tape.
"""
function mlp_batch(w::AbstractVector, x::AbstractVector, width::Int)
    W1 = @view w[1:width]
    b1 = @view w[width+1:2*width]
    o = 2 * width
    W2 = reshape(view(w, o+1:o+width*width), width, width)
    b2 = @view w[o+width*width+1:o+width*width+width]
    o2 = o + width * width + width
    W3 = @view w[o2+1:o2+width]
    b3 = w[o2+width+1]

    H1 = tanh.(W1 .* transpose(x) .+ b1)          # width × n
    H2 = tanh.(W2 * H1 .+ b2)                     # width × n
    return vec(transpose(H2) * W3) .+ b3          # n
end

"""
    effect(e, φ, Cs) -> Vector

`E(C)` at every concentration in `Cs`, for effect parameters `φ`.
"""
effect(::EmaxEffect, φ::AbstractVector, Cs::AbstractVector) =
    (Emax = exp(φ[1]); EC50 = exp(φ[2]); Emax .* Cs ./ (EC50 .+ Cs))

effect(::HillEffect, φ::AbstractVector, Cs::AbstractVector) =
    (Emax = exp(φ[1]); EC50 = exp(φ[2]); γ = exp(φ[3]);
     Emax .* Cs .^ γ ./ (EC50^γ .+ Cs .^ γ))

# ReverseDiff cannot differentiate this model at all — its broadcast rules fail once a
# ForwardDiff `Dual` is nested inside a `TrackedReal`, which the inner η-derivatives put there
# — so Mooncake is the only working reverse engine here. The E1 gate does not lean on a second
# reverse engine as a result; it cross-checks against the `:forwarddiff` control on the
# identical scalar, the two scalar constructions (`:directional` and `:matrix`), the
# forward-sensitivity gradient in the opposite orientation, and central differences of an
# independently written objective. See `docs/DECISIONS.md`.
effect(e::MLPEffect, φ::AbstractVector, Cs::AbstractVector) =
    Cs .* softplus.(mlp_batch(φ, Cs ./ e.c0, e.width))

# ---------------------------------------------------------------------------
# 3. The turnover ODE
# ---------------------------------------------------------------------------

"""
    turnover_solve(R0, kout, a, grid, obs_idx, n_pd) -> R at the PD observation times

    dR/dt = k_out · ( R₀ − R(t)·(1 + E(C(t))) ),   R(0) = R₀

marched by classical RK4 on `grid`, a **fixed** node vector built once per subject at data
construction time and stored with the subject.  Fixed steps rather than adaptive ones is a
deliberate choice and the central one in this file: an adaptive controller makes the number
of steps a function of the parameters, which makes the objective piecewise constant in
places and non-differentiable at every step-count change.  A discontinuous objective is
exactly what stalls a line search, and it is not worth risking for a solve this cheap.

Written in the linear form `dR/dt = b − a(t)·R` with `b = k_out·R₀` and
`a(t) = k_out·(1 + E(C(t)))`.  Because the right-hand side depends on `t` only through
`a(t)` and never on `R`, every stage value the march needs is precomputed:

    a[2j−1] = a(grid[j]),   a[2j] = a(midpoint of segment j),   a[2M+1] = a(grid[M+1])

so `a` has length `2M+1` for `M = length(grid) − 1` segments, and the network is evaluated
once, in one batch, for the whole trajectory.

`obs_idx[k]` is the index in `grid` of the k-th PD observation time, sorted ascending; the
march writes `out[k]` as it passes.  Every branch below is on an integer index.
"""
function turnover_solve(R0, kout, a::AbstractVector, grid::Vector{Float64},
                        obs_idx::Vector{Int}, n_pd::Int)
    T = promote_type(typeof(R0), typeof(kout), eltype(a))
    M = length(grid) - 1
    out = Vector{T}(undef, n_pd)

    R = convert(T, R0)
    b = kout * R0
    k = 1
    @inbounds while k <= n_pd && obs_idx[k] == 1
        out[k] = R
        k += 1
    end

    @inbounds for j in 1:M
        h = grid[j+1] - grid[j]
        a1 = a[2j-1]
        am = a[2j]
        a2 = a[2j+1]
        k1 = b - a1 * R
        k2 = b - am * (R + (h / 2) * k1)
        k3 = b - am * (R + (h / 2) * k2)
        k4 = b - a2 * (R + h * k3)
        R = R + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        while k <= n_pd && obs_idx[k] == j + 1
            out[k] = R
            k += 1
        end
    end
    return out
end

"""
    build_grid(pd_times; tmax, n_seg, gamma) -> (grid, obs_idx, stage_times)

The fixed RK4 grid for one subject, and the stage times the right-hand side must be
evaluated at.

Nodes are graded, `t_j = tmax · (j/n_seg)^gamma` with `gamma > 1`, so the mesh is fine while
the concentration is rising and coarse once it has decayed — the concentration moves on the
absorption timescale (`1/ka`, about an hour) while the response moves on the turnover
timescale (`1/k_out`, about a day), and a uniform mesh would have to resolve the former
everywhere to be safe.

Every PD observation time is inserted as a node, so its value is read off the march exactly
rather than interpolated.  Graded nodes falling within `1e-6` of an observation time are
dropped, so inserting an observation cannot create a degenerate segment.

`stage_times` has length `2M+1`: nodes and segment midpoints interleaved, in the order
`turnover_solve` consumes them.
"""
function build_grid(pd_times::AbstractVector{Float64}; tmax::Float64,
                    n_seg::Int = 32, gamma::Float64 = 2.2)
    pd = sort(collect(Float64, pd_times))
    base = [tmax * (j / n_seg)^gamma for j in 0:n_seg]
    keep = [b for b in base if b > 0.0 && all(t -> abs(b - t) > 1e-6, pd)]
    grid = sort(unique(vcat(0.0, keep, filter(>(0.0), pd))))

    obs_idx = Vector{Int}(undef, length(pd))
    for (k, t) in enumerate(pd)
        j = searchsortedfirst(grid, t)
        (j <= length(grid) && grid[j] == t) ||
            error("build_grid: PD time $t is not a grid node (nearest index $j).")
        obs_idx[k] = j
    end

    M = length(grid) - 1
    stage = Vector{Float64}(undef, 2M + 1)
    for j in 1:M
        stage[2j-1] = grid[j]
        stage[2j] = (grid[j] + grid[j+1]) / 2
    end
    stage[2M+1] = grid[M+1]

    return (grid = grid, obs_idx = obs_idx, stage_times = stage)
end
