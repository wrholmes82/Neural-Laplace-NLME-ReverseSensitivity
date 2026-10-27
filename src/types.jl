# Containers, the model interface, and the single configuration object.

"""
    SubjectData(id, times, observations, input; covariates=NamedTuple())

One subject's data:

  * `id` — the subject's integer label.
  * `times` — the observation times.
  * `observations` — the observed responses, one per entry of `times`.
  * `input` — the scalar known input driving that subject's structural model.
  * `covariates` — an optional `NamedTuple` of anything else the model needs.

Always `Float64` — parameter element types (possibly `Dual`) never enter here.
"""
struct SubjectData{C}
    id::Int
    times::Vector{Float64}
    observations::Vector{Float64}
    input::Float64
    covariates::C
end

SubjectData(id, times, observations, input; covariates = NamedTuple()) =
    SubjectData(Int(id), collect(Float64, times), collect(Float64, observations),
                Float64(input), covariates)

struct PopulationData{S}
    subjects::Vector{S}
    n_subjects::Int
end

PopulationData(subjects::AbstractVector) = PopulationData(collect(subjects), length(subjects))

Base.length(pop::PopulationData) = pop.n_subjects
Base.getindex(pop::PopulationData, i) = pop.subjects[i]
Base.iterate(pop::PopulationData, s = 1) = s > pop.n_subjects ? nothing : (pop.subjects[s], s + 1)

"""
    NLMEModel

Supertype for concrete models. A model must

  * declare its random-effect dimension, either as a field `q` or by adding a
    method to [`n_random_effects`](@ref); and
  * implement [`negative_log_joint_density`](@ref).

Nothing else is required. Every derivative used by this framework is obtained
from that one method by ForwardDiff.
"""
abstract type NLMEModel end

"""
    negative_log_joint_density(model, η, θ, subject)

`h_i(η, θ) = −log p(yᵢ | η, θ) − log p(η | θ)` for one subject, **including** all
normalising constants (the `log 2π` and `log σ` terms).

The argument order `(model, η, θ, subject)` is fixed and used without exception
throughout the framework. The function must be twice ForwardDiff-differentiable
in `η` and once in `θ`, at arbitrary Dual nesting depth.
"""
function negative_log_joint_density end

"""
    n_random_effects(model) -> Int

Random-effect dimension `q`. Defaults to the field `model.q`.
"""
n_random_effects(model::NLMEModel) = model.q

"""
    predict(model, η, θ, subject)

Optional, diagnostics only: the structural prediction at the subject's observation
times. Not used by the objective or any gradient. Takes the same `(model, η, θ, …)`
argument order as [`negative_log_joint_density`](@ref) — a swapped pair of
same-length vectors is a bug with no symptom, so the order never varies.
"""
function predict end

"""
    LaplaceConfig

The single knob container. Constructed once per fit and passed to both the
objective and the gradient — that shared object is what guarantees they describe
the same function.

  * `reg` — the shift in `H̃ = ∇²_ηη h + reg·I`. `H̃` is the curvature the
    *objective* carries, so it appears in the log-determinant and in the adjoint
    right-hand side `b`, which is the η-gradient of that log-determinant. The
    adjoint *solve* takes the unshifted `H` instead: its coefficient is the
    Jacobian of the mode equation `∇_η h = 0`, and `reg` enters the Newton
    direction but never the fixed point. Keeping those two roles distinct is what
    makes the adjoint gradient the exact gradient of the implemented objective
    rather than of one displaced by O(reg). Default `1e-10`,
    deliberately far below a problem-specific jitter: with the definiteness check
    in `logdet_pd` we would rather be told `H` is indefinite than have it papered
    over. Raise it consciously, per problem.
  * `step_max` — trust radius on the inner Newton step, in η units. Backtracking
    alone cannot recover from an arbitrarily long direction: 25 backtracks at
    `damping = 0.8` shrink a step by only 3.8e-3, so a Newton direction of length
    1e14 — which a nearly singular `H̃` produces — leaves the line search
    exhausted having never evaluated a sane point, and the solve takes *zero*
    steps. Since η is on a standardized random-effect scale, `10` is already far
    outside anything meaningful.
  * `inner_pd_floor` — conditioning target for the **search-only** positive-definite
    curvature the inner solver uses to pick its Newton direction where the exact
    `∇²_ηη h` is indefinite (see [`pd_search_hessian`](@ref)): the diagonal shift
    lifts the smallest eigenvalue to `inner_pd_floor · max(λ_max, 1)`, bounding the
    modified matrix's condition number at `1/inner_pd_floor`. Default `0.1`
    (cond ≤ 10). It never touches the objective's `H̃` or the adjoint gradient,
    which keep the exact `H`. The well-conditioned target is load-bearing: a
    barely-PD shift takes a huge near-null-direction step whose unrolled derivative
    breaks adjoint/AD-through agreement to ~1e-3; cond ≤ 10 keeps it at ~1e-10.
  * `inner_deriv` — `:ad` (ForwardDiff) or `:fd` (finite differences) for the
    inner η-gradient and η-Hessian. `:fd` exists to reproduce the reference
    manuscript's FD baseline, in which *both* layers inherit step-size noise.
  * `warm_start` — off by default. Warm-starting makes the objective
    history-dependent at the level of `inner_tol`, which biases finite-difference
    comparisons and can confuse a backtracking line search.
  * `threaded` — on by default. The serial path exists for clean stack traces and
    as the control arm of the thread-count-independence test.
"""
Base.@kwdef struct LaplaceConfig
    reg::Float64 = 1e-10
    inner_tol::Float64 = 1e-6
    inner_maxiter::Int = 50
    linesearch_maxiter::Int = 25
    armijo_c::Float64 = 1e-4
    damping::Float64 = 0.8
    step_max::Float64 = 10.0
    inner_pd_floor::Float64 = 0.1
    inner_deriv::Symbol = :ad
    inner_fd_step::Float64 = 1e-5
    warm_start::Bool = false
    threaded::Bool = true
end

"""
    NLMEResult

Everything a fit produced, including the diagnostics that say whether to believe
it.

The two rescue counters are **not** equally alarming, and separating them matters:

  * `objective_rescues` — a non-finite `L` replaced by a large finite barrier
    value. This is normal and healthy. The line search probes θ where some
    subject's `H̃` is genuinely not positive definite, so η̂ is not a mode and the
    Laplace expression is undefined there; NaN propagates rather than being
    papered over, the barrier makes the optimizer retreat, and the fit is
    unaffected. A clean 50-subject fit typically shows a handful.
  * `gradient_rescues` — a non-finite gradient *component* zeroed. This is the
    alarming one, and it should be zero. A zeroed component is indistinguishable
    from a converged one, so a quasi-Newton method will report success on a
    broken gradient. This is the `OLD/` failure mode the no-silent-rescue rule
    exists to prevent.

`inner_nonconverged` counts subject solves that hit the inner iteration cap at
the final θ. Convergence is judged on the recomputed final gradient norm, never
on the optimizer's return code.
"""
struct NLMEResult
    theta::Vector{Float64}
    L::Float64
    OFV::Float64
    gradient::Vector{Float64}
    gradient_norm::Float64
    iterations::Int
    converged::Bool
    return_code::Symbol
    elapsed::Float64
    eta::Vector{Vector{Float64}}
    objective_rescues::Int
    gradient_rescues::Int
    inner_nonconverged::Int
    rel_ofv_change::Float64
    config::LaplaceConfig
    nthreads::Int
    standard_errors::Union{Nothing,Vector{Float64}}
    vcov::Union{Nothing,Matrix{Float64}}
end

function Base.show(io::IO, r::NLMEResult)
    println(io, "NLMEResult")
    println(io, "  OFV              = ", r.OFV)
    println(io, "  L                = ", r.L)
    println(io, "  θ                = ", r.theta)
    println(io, "  ‖∇L‖∞            = ", r.gradient_norm)
    println(io, "  iterations       = ", r.iterations, "  converged = ", r.converged,
                "  (", r.return_code, ")")
    println(io, "  rel. OFV change  = ", r.rel_ofv_change)
    println(io, "  rescues          = ", r.objective_rescues, " objective (benign), ",
                r.gradient_rescues, " gradient (must be 0)")
    println(io, "  inner non-conv   = ", r.inner_nonconverged)
    println(io, "  elapsed          = ", r.elapsed, " s on ", r.nthreads, " thread(s)")
    if r.standard_errors !== nothing
        println(io, "  standard errors  = ", r.standard_errors)
    end
end
