# One-time environment preparation.
#
#   julia setup.jl
#   julia setup.jl /path/to/env
#
# Dev-installs the `NLMEReverse` package (this repository's `src/`) into the environment and
# ensures the packages the scripts `using` directly are present.  Nothing outside the chosen
# environment is modified.

using Pkg

const ROOT = @__DIR__
const ENVDIR = length(ARGS) >= 1 ? abspath(ARGS[1]) : joinpath(ROOT, "Local_Env")

isdir(ENVDIR) || mkpath(ENVDIR)
@info "preparing environment" ENVDIR julia = VERSION
Pkg.activate(ENVDIR)

Pkg.develop(path = ROOT)

# Scripts `using` these directly; stdlibs must be declared even if only indirect deps.
for pkg in ("ForwardDiff", "Optim", "Printf", "DelimitedFiles", "Statistics", "Random",
            "LinearAlgebra", "Test", "CairoMakie")
    try
        Pkg.add(pkg)
    catch err
        @warn "could not add $pkg" exception = (err,)
    end
end

# Reverse engines.  They are deliberately NOT dependencies of NLMEReverse — they are loaded
# per-run from src/backends/ — so at least one must install here, and an engine that fails
# to precompile (Enzyme, routinely) is not a blocker.
engines = String[]
for pkg in ("Enzyme", "Mooncake", "ReverseDiff")
    try
        Pkg.add(pkg); push!(engines, pkg)
    catch err
        @warn "could not add $pkg" exception = (err,)
    end
end

Pkg.instantiate()
Pkg.precompile()

println("""

Setup complete.  Engines: $(isempty(engines) ? "NONE" : join(engines, ", "))

Next, from the repository root (note the thread count):

    julia --project=$(ENVDIR) -t 5 test/runtests.jl                        # tiers 0-7
    julia --project=$(ENVDIR) -t 5 test/verification/verify_reverse.jl     # reverse gradient GATE
    julia --project=$(ENVDIR) -t 5 test/verification/verify_foce.jl        # FOCE curvature GATE

Then the study, in order (see RUNBOOK.md):

    julia --project=$(ENVDIR) -t 5 warfarin_turnover/examples/realdata/fit_par_truth.jl
    julia --project=$(ENVDIR) -t 5 warfarin_turnover/test/e1_gate.jl        # E1 -- HARD GATE
    julia --project=$(ENVDIR) -t 5 warfarin_turnover/examples/cost/e2_cost_ladder.jl
    julia --project=$(ENVDIR) -t 1 warfarin_turnover/figures/make_figures.jl
""")
