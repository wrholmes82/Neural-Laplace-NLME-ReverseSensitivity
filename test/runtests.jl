# Run with:  julia --project=Local_Env -t auto test/runtests.jl
#
# Governing rules for everything under test/:
#   * A crash is a failure. No try/catch wrapping an assertion into @test true.
#   * Tolerances are derived, never relaxed to pass. If a tolerance has to loosen,
#     the finding is a bug report, not a test edit.
#   * Elementwise, not norm-based. Norms hide sign errors and single-component
#     failures.
#   * Every test_*.jl in this directory is included below. Tier 7 checks that
#     mechanically rather than trusting discipline.

using Test
using LinearAlgebra
using Random
using Statistics
using ForwardDiff
using NLMEReverse

BLAS.set_num_threads(1)   # timing reproducibility; measured a non-issue at q=4

const TEST_DIR = @__DIR__
const ROOT_DIR = dirname(TEST_DIR)
const PROBLEMS = joinpath(TEST_DIR, "problems")

include(joinpath(PROBLEMS, "twocomp.jl"))
include(joinpath(PROBLEMS, "linear_gaussian.jl"))
include(joinpath(PROBLEMS, "simulate.jl"))

println("="^72)
println("NLMEReverse test suite")
println("  Julia            ", VERSION)
println("  Threads.nthreads ", Threads.nthreads())
if Threads.nthreads() == 1
    println("  NOTE: running single-threaded. A green run at -t 1 is not evidence")
    println("        of thread safety. Re-run with -t auto.")
end
println("="^72)

# Included at top level, not inside an enclosing @testset: `@testset` wraps its
# body in a local scope, and tier files need to define `struct`s (toy models),
# which is only legal at top level. Each tier file opens its own @testset.
include("test_tier0_oracle.jl")
include("test_tier1_numerics.jl")
include("test_tier2_inner.jl")
include("test_tier3_model.jl")
include("test_tier4_objective.jl")
include("test_tier5_gradients.jl")
include("test_tier6_outer.jl")
include("test_tier7_structure.jl")
