# realdata.jl — the real 32-subject warfarin dataset.
#
# Provenance: the `nlmixr2data::warfarin` / Monolix demo dataset (Funaki, Holford & Fujita,
# PAGJA 2018; original study O'Reilly et al. 1963, 1968).  32 subjects, single oral dose of
# 1.5 mg/kg, concentrations and prothrombin complex activity to 144 h.
#
# **Endpoint resolution.**  Huth et al. describe the PD endpoint as INR; this file carries
# **PCA**.  The two are inversely related and the sign of the drug effect flips between them.
# The check is direct and it is in `warfarin_endpoint_check` below: PCA starts at 100 and
# falls to a nadir before recovering, which is suppression, so `E > 0` suppressing `R` in the
# turnover equation is correct as written.  Recorded in `docs/DECISIONS.md`.

using NLMEReverse
using DelimitedFiles

const WARFARIN_FILE = joinpath(@__DIR__, "..", "data", "warfarin.csv")

"""
    load_warfarin(model) -> NamedTuple

`(pop, n_blq, n_subjects)` — the real data as a `PopulationData`, built through the same
`make_subject` the simulator uses.

Records with `dv <= 0` on the concentration channel (4 of them, all below the assay limit)
are **dropped**, and the count is returned and printed rather than absorbed.  A proportional
residual model has no likelihood for a zero observation; the honest alternatives are an M3
likelihood or dropping, and dropping four records out of 283 does not move a 32-subject
anchor.  Nothing here is silently capped.
"""
function load_warfarin(m::WarfarinTurnover; file::AbstractString = WARFARIN_FILE)
    raw, hdr = readdlm(file, ','; header = true)
    cols = Dict(String(h) => j for (j, h) in enumerate(vec(hdr)))
    id = Int.(raw[:, cols["id"]])
    time = Float64.(raw[:, cols["time"]])
    amt = Float64.(raw[:, cols["amt"]])
    dv = Float64.(raw[:, cols["dv"]])
    dvid = Int.(raw[:, cols["dvid"]])
    evid = Int.(raw[:, cols["evid"]])

    ids = sort(unique(id))
    subs = Vector{SubjectData}(undef, length(ids))
    n_blq = 0

    for (k, sid) in enumerate(ids)
        rows = findall(==(sid), id)
        drow = findfirst(r -> evid[r] == 1, rows)
        drow === nothing && error("subject $sid has no dosing record (evid == 1).")
        dose = amt[rows[drow]]

        pkr = [r for r in rows if evid[r] == 0 && dvid[r] == 1 && time[r] > 0]
        blq = [r for r in pkr if dv[r] <= 0]
        n_blq += length(blq)
        pkr = [r for r in pkr if dv[r] > 0]
        pdr = [r for r in rows if evid[r] == 0 && dvid[r] == 2]

        subs[k] = make_subject(m, sid, dose, time[pkr], dv[pkr], time[pdr], dv[pdr])
    end

    n_blq > 0 && @info "load_warfarin: concentration records at or below zero dropped" n_blq

    return (pop = PopulationData(subs), n_blq = n_blq, n_subjects = length(ids))
end

"""
    warfarin_endpoint_check(pop) -> NamedTuple

The PCA-versus-INR resolution, computed rather than asserted: the population median response
at baseline, at its nadir, and at the last sampling time.

PCA falls under drug (baseline ≈ 100, nadir well below it, partial recovery); INR would rise.
Any future dataset swap that flipped the endpoint would flip `suppressed` and stop the study
rather than silently invert the sign of every fitted effect.
"""
function warfarin_endpoint_check(pop::PopulationData)
    ts = Float64[]
    ys = Float64[]
    for s in pop.subjects
        cv = s.covariates
        append!(ts, cv.pd_times)
        append!(ys, s.observations[cv.n_pk+1:cv.n_pk+cv.n_pd])
    end
    early = ys[ts .<= 6.0]
    mid = ys[(ts .> 24.0) .& (ts .<= 72.0)]
    late = ys[ts .>= 120.0]
    med(v) = isempty(v) ? NaN : sort(v)[cld(length(v), 2)]
    b, n, l = med(early), med(mid), med(late)
    return (baseline = b, nadir = n, late = l, suppressed = n < b,
            endpoint = n < b ? "PCA (drug suppresses response)" : "INR-like (drug raises response)")
end
