# E14b — the single-run cost budget as one table, ready to drop into the paper.
#
#   julia --project=Local_Env warfarin_turnover/examples/cost/e14b_cost_table.jl
#
# E14 times one exact-Laplace M_nn(10) run end to end in a cold process. E11 ran five penalised
# (gamma = 1) fits and observed-information calculations at the same cohort size and network
# width. This script joins them: E14 supplies the session costs, which only a cold process can
# measure, and E11 supplies the distribution of the per-fit costs.
#
# Nothing is typed in by hand. Every number is read from an artefact, so re-running any of the
# upstream experiments and re-running this refreshes the table.

include(joinpath(@__DIR__, "..", "..", "src", "warfarin.jl"))

banner("E14b — single-run cost budget")

const E14 = joinpath(@__DIR__, "results")
const E11 = normpath(joinpath(@__DIR__, "..", "regularisation", "results"))
const RESULTS = results_dir(@__DIR__, "results")

"""Column of a two-column `quantity,value,...` table, as a lookup."""
function keyed(path::AbstractString, keycol::String, valcol::String)
    raw, hdr = readdlm(path, ','; header = true)
    h = Dict(String(c) => j for (j, c) in enumerate(vec(hdr)))
    return Dict(String(raw[i, h[keycol]]) => Float64(raw[i, h[valcol]]) for i in axes(raw, 1))
end

b = keyed(joinpath(E14, "e14_single_run_budget.csv"), "quantity", "seconds")

function numcol(path, name; lambda = nothing, require_finite_information = false)
    raw, hdr = readdlm(path, ','; header = true)
    h = Dict(String(c) => j for (j, c) in enumerate(vec(hdr)))
    v = Float64[]
    for i in axes(raw, 1)
        if lambda !== nothing
            lam = raw[i, h["lambda"]]
            lam = lam isa Number ? Float64(lam) : parse(Float64, String(lam))
            lam == lambda || continue
        end
        if require_finite_information
            lowercase(string(raw[i, h["information_finite"]])) == "true" || continue
        end
        x = raw[i, h[name]]
        y = x isa Number ? Float64(x) : tryparse(Float64, String(x))
        y !== nothing && isfinite(y) && push!(v, y)
    end
    v
end

e11_fits = joinpath(E11, "e11_fits.csv")
fits = numcol(e11_fits, "seconds"; lambda = 1.0, require_finite_information = true)
info = numcol(e11_fits, "information_seconds"; lambda = 1.0,
              require_finite_information = true)
length(fits) == 5 || error("Expected five finite gamma=1 E11 fits; found $(length(fits)).")
length(info) == 5 || error("Expected five finite gamma=1 E11 information timings; found $(length(info)).")
med(v) = isempty(v) ? NaN : median(v)
lo(v) = isempty(v) ? NaN : minimum(v)
hi(v) = isempty(v) ? NaN : maximum(v)

rows = Any[csvrow("category", "quantity", "value", "units", "scope", "reference_n",
                  "reference_median", "reference_min", "reference_max", "source", "note")]

push!(rows, csvrow("timing", "package load", b["include_package"] + b["load_backends"], "s",
                   "per session", 1, NaN, NaN, NaN, "E14",
                   "parsing and loading the package, its dependencies and the AD backends"))
push!(rows, csvrow("timing", "compilation", b["compilation"], "s", "per session", 1,
                   NaN, NaN, NaN, "E14",
                   "cold minus warm on the fit and on the information; paid once because the " *
                   "network width is a runtime field, so all fits share compiled code"))
push!(rows, csvrow("timing", "estimation", b["fit_warm"], "s", "per fit", length(fits),
                   med(fits), lo(fits), hi(fits), "E14 value, E11 gamma=1 reference",
                   "quasi-Newton outer solve with a damped Newton conditional mode per subject"))
push!(rows, csvrow("timing", "observed information", b["information_warm"], "s", "per fit",
                   length(info), med(info), lo(info), hi(info),
                   "E14 value, E11 gamma=1 reference",
                   "central differences of the analytic gradient, 2p gradient evaluations"))

push!(rows, csvrow("configuration", "julia threads", b["julia_threads"], "threads", "",
                   NaN, NaN, NaN, NaN, "E14", "parallelism is over subjects"))
push!(rows, csvrow("configuration", "BLAS threads", b["blas_threads"], "threads", "",
                   NaN, NaN, NaN, NaN, "E14", "pinned to 1 so it cannot contend with the above"))
push!(rows, csvrow("configuration", "CPU threads reported", b["cpu_threads"], "threads", "",
                   NaN, NaN, NaN, NaN, "E14",
                   "$(Sys.CPU_NAME); Julia reports $(Int(b["cpu_threads"])) hardware threads " *
                   "in this process"))
push!(rows, csvrow("configuration", "subjects", b["n_subjects"], "subjects", "",
                   NaN, NaN, NaN, NaN, "E14", "one synthetic cohort"))
push!(rows, csvrow("configuration", "population parameters", b["p"], "parameters", "",
                   NaN, NaN, NaN, NaN, "E14", "M_nn(10): 12 interpretable and 141 network"))

writedlm(joinpath(RESULTS, "e14_cost_table.csv"), vcat(rows...), ',')

@printf("\n%-22s %10s %6s %-12s %8s %10s %10s\n",
        "quantity", "value", "units", "scope", "ref n", "ref median", "ref range")
for r in rows[2:end]
    rng = isnan(Float64(r[8])) ? "" : @sprintf("%.0f-%.0f", r[8], r[9])
    @printf("%-22s %10.1f %6s %-12s %8s %10s %10s\n", r[2], Float64(r[3]), r[4], r[5],
            isnan(Float64(r[6])) ? "" : string(Int(r[6])),
            isnan(Float64(r[7])) ? "" : @sprintf("%.1f", r[7]), rng)
end

println("""

The per-fit rows carry both E14's own draw and E11's distribution over five exact-Laplace,
gamma=1 fits and information matrices at the same cohort size and width. The reference columns
are the ones quoted in the manuscript.""")
println("=" ^ 78)
