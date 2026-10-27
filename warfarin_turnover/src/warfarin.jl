# warfarin.jl — the single entry point for the warfarin turnover study.
#
#   include(joinpath(REPO_ROOT, "warfarin_turnover", "src", "warfarin.jl"))
#
# Loads `NLMEReverse`, the reverse-mode engine loaders, and the study's own model, design,
# simulator, starting values and fit driver.  Every script under `warfarin_turnover/` starts
# with this line and nothing else.
#
# `BLAS.set_num_threads(1)` is deliberately NOT set here.  It belongs in scripts, never in a
# module or a shared loader: a script that sets it announces the thread configuration its
# timings were taken under, and a loader that sets it takes that decision away from every
# caller silently.

using NLMEReverse
using LinearAlgebra
using ForwardDiff
using Optim
using LineSearches
using Random
using Statistics
using Printf
using DelimitedFiles

const WARFARIN_SRC = @__DIR__
const REPO_ROOT = normpath(joinpath(WARFARIN_SRC, "..", ".."))

include(joinpath(REPO_ROOT, "src", "backends", "backends.jl"))

include(joinpath(WARFARIN_SRC, "structural.jl"))
include(joinpath(WARFARIN_SRC, "model.jl"))
include(joinpath(WARFARIN_SRC, "design.jl"))
include(joinpath(WARFARIN_SRC, "starts.jl"))
include(joinpath(WARFARIN_SRC, "simulate.jl"))
include(joinpath(WARFARIN_SRC, "realdata.jl"))
include(joinpath(WARFARIN_SRC, "binomial.jl"))
include(joinpath(WARFARIN_SRC, "fit.jl"))

"""
    the_ladder() -> Vector{NamedTuple}

The parameter ladder E2 sweeps and E3 tops out.

| `W` | `p` | role |
|---:|---:|---|
| 5 | 58 | at the incumbent's published scale (four to five hidden units) |
| 10 | 153 | the **scientific** width — every inference result is reported here |
| 20 | 493 | mid-ladder |
| 40 | 1773 | the **scaling** width, and the top of every experiment here |

`p` counts the whole θ (`12 + W² + 4W + 1`), not the network alone.

Widths 20 and 40 are **deliberately oversized to locate the cost curve, not proposed as
practical models** — the same framing the 15,410-parameter published comparator uses for
itself.

The ladder stops at `W = 40` rather than reaching `p ≈ 10⁴` directly. What that costs, stated
plainly rather than buried: the study no longer *measures* a completed fit at the scale where
forward-mode FOCE was publicly abandoned, and must instead **project** to it from the measured
exponents. The projection is sound — the arms' exponents in `p` are estimated over a 30-fold
range of `p` and the reverse arm's cost is a fixed multiple of one objective evaluation — but a
projection is not a measurement and E3 labels it as one.
"""
function the_ladder()
    return [(width = w, model = WarfarinTurnover(MLPEffect(w)),
             p = WarfarinTurnover(MLPEffect(w)).p) for w in (5, 10, 20, 40)]
end

const SCIENTIFIC_WIDTH = 10
const SCALING_WIDTH = 40

"""
    banner(title)

Every script prints one, and every timing in this study is quoted with the thread count and
the machine state it was taken under.
"""
function banner(title::AbstractString)
    println("=" ^ 78)
    println(title)
    @printf("  julia %s   julia threads %d   BLAS threads %d   loadavg1 %.2f\n",
            VERSION, Threads.nthreads(), BLAS.get_num_threads(), Sys.loadavg()[1])
    println("=" ^ 78)
end

"""
    results_dir(parts...) -> String

Result directories live beside the script that writes them, under `results/`.  **No result is
reported from a directory that is not in the layout** — this function is the only place a
result path is constructed.
"""
function results_dir(parts...)
    d = joinpath(parts...)
    mkpath(d)
    return d
end
