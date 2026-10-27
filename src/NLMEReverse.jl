"""
    NLMEReverse

Nonlinear mixed-effects estimation with **reverse sensitivities**: a population gradient
for the exact-Laplace (and FOCE) objective whose cost is independent of the number of
parameters `p`.

Problem-agnostic. One required model method —
`negative_log_joint_density(model, η, θ, subject)` — from which the η-gradient, η-Hessian,
log-determinant, mixed derivatives and every adjoint term are derived by AD. Nothing here
knows about pharmacokinetics, neural networks, or any particular application: this package
references nothing outside its own directory, and a copy of it is a working NLME framework
anywhere.

--------------------------------------------------------------------------------
The three layers
--------------------------------------------------------------------------------

**1. The Laplace objective and the adjoint gradient** (`types`, `numerics`, `inner`,
`objective`, `gradient_*`, `outer`). The marginal likelihood by Laplace approximation
about the conditional mode `η̂`, with

    ∇θ Lᵢ = ∂θ hᵢ(η̂ᵢ,θ) + ½ ∂θ logdet H̃ᵢ(η̂ᵢ,θ) − (∇²_ηθ hᵢ)ᵀ λᵢ        (Eq. 4.4)

evaluated three independent ways — FULL-implicit adjoint, AD-through, finite differences —
over one shared objective, plus a forward-sensitivity comparator in the Almquist (2015)
orientation. They compute the same quantity by different routes, so disagreement between
them means a bug, which is how every gradient here is verified.

**2. The reverse construction** (`reverse.jl`). Eq. (4.4) has the adjoint *orientation* but
`ForwardDiff.gradient` gives it the forward *cost model*: `p` tangents. Term 2 is also a
VJP — with `W ≡ ½H̃⁻¹` **frozen**, `d/dθ ½ logdet H̃ = ⟨W, ∂θH̃⟩` — so all three terms are
gradients of the **single scalar**

    sᵢ(θ) = hᵢ(η̂ᵢ,θ) + ⟨Wᵢ, H̃ᵢ(η̂ᵢ,θ)⟩ − (∇_η hᵢ(η̂ᵢ,θ))ᵀλᵢ ,   ∇θ Lᵢ = ∇θ sᵢ(θ)

one reverse sweep per subject, O(1) in `p`. The engines (Mooncake, ReverseDiff)
are **not** dependencies: each is loaded at run time from `src/backends/`, so an engine
that will not precompile cannot stop this module loading. `:forwarddiff` is always present
as the **control arm** — the same scalar, `p` tangents — and is never a valid "reverse"
result.

**3. The curvature choice** (`curvature.jl`). The frozen-weight identity does not require
`M` to be the exact Hessian, so `CurvatureConfig` selects

    :laplace →  M = ∇²_ηη h            exact; may be indefinite
    :foce    →  M = JᵀR⁻¹J + Ω⁻¹       Gauss-Newton; PD by construction, one AD order cheaper

with the **exact** `H = ∇²_ηη h` kept in the λ-solve under both (it is the Jacobian of the
mode equation, which FOCE does not change).

**4. The like-for-like O(p) comparator** (`gradient_forward_foce.jl`). `forward_gradient` and
`adjoint_gradient` differentiate the *exact-Laplace* objective; the headline reverse arm
differentiates the *FOCE* objective, and comparing those two is not like-for-like. This is the
forward-sensitivity (Almquist orientation) gradient of the **FOCE** objective — same objective,
same inner solve, `p` right-hand sides through one `q×q` factorisation instead of one reverse
sweep. It is the incumbent arm every cost ratio is quoted against.

Derivations, the cost accounting, and what would break the construction: `docs/`.
"""
module NLMEReverse

using LinearAlgebra
using ForwardDiff
using Optimization
using OptimizationOptimJL
using Optim
using LineSearches

# Plain relative includes, so both load modes work:
#   include(".../src/NLMEReverse.jl"); using .NLMEReverse      (drop-in)
#   Pkg.develop(path="."); using NLMEReverse                   (precompiled)
include("types.jl")
include("numerics.jl")
include("inner.jl")
include("objective.jl")
include("gradient_adjoint.jl")
include("gradient_forward.jl")
include("gradient_reference.jl")
include("outer.jl")
include("reverse.jl")
include("curvature.jl")
include("gradient_forward_foce.jl")
include("information.jl")

# types and interface
export SubjectData, PopulationData, NLMEModel, LaplaceConfig, NLMEResult
export negative_log_joint_density, n_random_effects, predict

# numerics
export extract_value, logdet_pd, solve_pd, htilde

# inner solve
export newton_solve, inner_gradient, inner_hessian

# objective
export subject_objective, population_objective, laplace_objective, laplace_ofv

# gradients
export adjoint_gradient, adjoint_gradient_parts, subject_adjoint
export forward_gradient, forward_gradient_parts, subject_forward
export ad_gradient, fd_gradient, gradient_factory

# outer
export fit_nlme, standard_errors, robust_errors, observed_information, empirical_bayes

# reading a singular observed information: the convergence criterion and the delta method
export spectral_split, resolve_tau, newton_decrement, truncated_inverse, delta_method, fit_quality

# reverse construction
export ReverseConfig, subject_reverse, reverse_gradient_parts, reverse_gradient
export directional_scalar, reverse_scalar, dir1, dir2, reverse_sweep
export backend_available, selftest_backend, select_backend, CANDIDATE_BACKENDS, FORMS

# curvature choice (:laplace | :foce)
export CurvatureConfig, conditional_moments, dir1_vec
export foce_curvature, foce_directional_scalar, foce_matrix_scalar
export subject_curvature, curvature_gradient_parts, curvature_gradient
export foce_objective, foce_subject_objective

# forward-sensitivity FOCE comparator (the like-for-like O(p) arm)
export subject_foce_forward, foce_forward_gradient_parts, foce_forward_gradient

end # module
