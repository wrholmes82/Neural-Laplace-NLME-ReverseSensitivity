# simulate.jl — the data-generating process.
#
# Truth is the `M_par` fit to the real 32-subject warfarin data (§2.3): self-consistent,
# defensible, and it removes any argument about where the truth came from.  `truth_theta()`
# reads that fit back from `data/truth_par.csv`, which
# `examples/realdata/fit_par_truth.jl` writes.
#
# Every cohort is independent and identified by a single seed derived from one named
# constant, so raising the replicate count from 20 to 100 is a change to one number and a
# rerun — never a redesign.  Nothing here aggregates across cohorts.

using NLMEReverse
using Random
using DelimitedFiles
using Printf

const TRUTH_FILE = joinpath(@__DIR__, "..", "data", "truth_par.csv")

"""
    truth_theta() -> (theta, model, meta)

The data-generating parameters: the `M_par` estimate from the real warfarin data, together
with the `EmaxEffect` model they belong to and the provenance recorded alongside them.

Errors — rather than falling back to invented constants — if the truth file is absent.  A
simulation study whose truth silently defaulted would be worse than one that stopped.
"""
function truth_theta()
    isfile(TRUTH_FILE) || error(
        "no simulation truth at $(TRUTH_FILE).\n" *
        "Run  julia --project=Local_Env -t 5 warfarin_turnover/examples/realdata/fit_par_truth.jl  first; " *
        "§2.3 fixes the truth by fitting M_par to the real data, and there is no default.")
    raw, hdr = readdlm(TRUTH_FILE, ','; header = true)
    names = String.(vec(raw[:, 1]))
    vals = Float64.(vec(raw[:, 2]))
    θ = vals[startswith.(names, "theta_")]
    meta = Dict(names[i] => vals[i] for i in eachindex(names) if !startswith(names[i], "theta_"))
    model = WarfarinTurnover(EmaxEffect())
    length(θ) == model.p || error(
        "truth file holds $(length(θ)) θ entries; M_par has p = $(model.p).")
    return (theta = θ, model = model, meta = meta)
end

"""
    simulate_cohort(n; seed, theta = nothing, model = nothing) -> NamedTuple

One simulated cohort of `n` subjects from the `M_par` truth.

Returns `(pop, etas, theta, model, dose)` — the population, the **true** random effects
(needed to check that the inner solve recovers them, and to report diagnostics that compare
against the generating values rather than against the conditional modes),
the truth θ used, and each subject's dose.  All of it is written to CSV by
`write_cohort`, so nothing downstream ever has to re-simulate to plot.

Residual error is proportional on `C` and additive on `R`, the `nlmixr2` warfarin
convention.  Simulated concentrations are **not** truncated at zero: truncation would make
the data-generating process differ from the likelihood being fitted, which is a silent
misspecification, and at these σ values a negative draw is vanishingly rare.  The count of
any that occur is returned and written, never suppressed.
"""
function simulate_cohort(n::Int; seed::Int,
                         theta::Union{Nothing,Vector{Float64}} = nothing,
                         model = nothing)
    t = theta === nothing || model === nothing ? truth_theta() : (theta = theta, model = model)
    θ = t.theta
    m = t.model
    ix = theta_index(m)

    rng = MersenneTwister(seed)
    ω = exp.(θ[ix.omega])
    σp = exp(θ[ix.sigma_prop])
    σa = exp(θ[ix.sigma_add])

    pkt = pk_design_times()
    pdt = pd_design_times()

    subs = Vector{SubjectData}(undef, n)
    etas = Vector{Vector{Float64}}(undef, n)
    doses = Vector{Float64}(undef, n)
    n_negative = 0

    for i in 1:n
        dose = DOSE_LEVELS[mod1(i, length(DOSE_LEVELS))]     # balanced by construction
        η = ω .* randn(rng, m.q)
        # A placeholder subject carries the design; the structural call needs the grid, and
        # the grid depends only on the times.
        proto = make_subject(m, i, dose, pkt, zeros(length(pkt)), pdt, zeros(length(pdt)))
        s = structural(m, η, θ, proto)
        pky = s.pk .* (1 .+ σp .* randn(rng, length(pkt)))
        pdy = s.pd .+ σa .* randn(rng, length(pdt))
        n_negative += count(<(0.0), pky)
        subs[i] = make_subject(m, i, dose, pkt, pky, pdt, pdy)
        etas[i] = η
        doses[i] = dose
    end

    n_negative > 0 && @info "simulate_cohort: negative simulated concentrations retained (not truncated)" seed n_negative

    return (pop = PopulationData(subs), etas = etas, theta = θ, model = m,
            dose = doses, seed = seed, n_negative = n_negative)
end

"""
    write_cohort(dir, sim; tag = "")

Write a simulated cohort in full: the long-format dataset, the true random effects, and the
truth θ.  Three files, explicit headers, one row per record.

The point of writing all three is that no downstream analysis or figure ever needs to
re-simulate.  A cohort is expensive relative to a plot; a plot must never be the reason a
cohort is regenerated.
"""
function write_cohort(dir::AbstractString, sim; tag::AbstractString = "")
    mkpath(dir)
    sfx = isempty(tag) ? "" : "_" * tag

    rows = Any[["id" "dose" "dvid" "kind" "time" "dv"]]
    for s in sim.pop.subjects
        cv = s.covariates
        for k in 1:cv.n_pk
            push!(rows, [s.id cv.dose 1 "conc" cv.pk_times[k] s.observations[k]])
        end
        for k in 1:cv.n_pd
            push!(rows, [s.id cv.dose 2 "pca" cv.pd_times[k] s.observations[cv.n_pk+k]])
        end
    end
    writedlm(joinpath(dir, "cohort$(sfx).csv"), vcat(rows...), ',')

    erows = Any[["id" "dose" "eta_ka" "eta_CL" "eta_V" "eta_R0" "eta_kout"]]
    for (i, s) in enumerate(sim.pop.subjects)
        push!(erows, [s.id sim.dose[i] sim.etas[i][1] sim.etas[i][2] sim.etas[i][3] sim.etas[i][4] sim.etas[i][5]])
    end
    writedlm(joinpath(dir, "cohort_eta_true$(sfx).csv"), vcat(erows...), ',')

    trows = Any[["name" "value"]]
    for (j, v) in enumerate(sim.theta)
        push!(trows, ["theta_$(j)" v])
    end
    push!(trows, ["seed" sim.seed])
    push!(trows, ["n_subjects" sim.pop.n_subjects])
    push!(trows, ["n_negative_conc" sim.n_negative])
    writedlm(joinpath(dir, "cohort_truth$(sfx).csv"), vcat(trows...), ',')

    return dir
end
