# E15c — the real-data cost and fit table, assembled from the artefacts.
#
#   julia --project=Local_Env warfarin_turnover/examples/regularisation/e15c_summary_table.jl
#
# One row per model: the parametric fit and the network at each ridge strength. Compilation comes
# from the cold/warm pair each run measured in its own process; estimation and observed
# information are medians and ranges over the five initialisation seeds; the objective range and
# the effective dimension are over the same five.
#
# Nothing is typed in. Re-running E15 or E15b and re-running this refreshes the table.

include(joinpath(@__DIR__, "..", "..", "src", "warfarin.jl"))
BLAS.set_num_threads(1)

banner("E15c — real-data cost and fit summary")

const R = results_dir(@__DIR__, "results")
const WIDTH = SCIENTIFIC_WIDTH
const GAMMAS = (0.0, 1.0, 10.0)

keyed(path) = begin
    raw, hdr = readdlm(path, ','; header = true)
    h = Dict(String(c) => j for (j, c) in enumerate(vec(hdr)))
    Dict(String(raw[i, h["quantity"]]) => Float64(raw[i, h["value"]]) for i in axes(raw, 1))
end
tabl(path) = begin
    raw, hdr = readdlm(path, ','; header = true)
    h = Dict(String(c) => j for (j, c) in enumerate(vec(hdr)))
    [Dict(k => raw[i, j] for (k, j) in h) for i in axes(raw, 1)]
end
num(x) = x isa Number ? Float64(x) : something(tryparse(Float64, String(x)), NaN)
truthy(x) = x isa Bool ? x : lowercase(string(x)) == "true"
med(v) = isempty(v) ? NaN : median(v)
rng(v) = isempty(v) ? "—" : @sprintf("%.1f (%.1f–%.1f)", median(v), minimum(v), maximum(v))

rows = Any[csvrow("model", "gamma", "p", "n_fits", "compile_s", "estimation_s_median",
                  "estimation_s_min", "estimation_s_max", "information_s_median",
                  "information_s_min", "information_s_max", "OFV_min", "OFV_max",
                  "df_eff_min", "df_eff_max", "n_df_eff_defined",
                  "hessian_regularization", "n_information_finite", "n_zero_retained")]

# --- the parametric model, timed in its own process ---------------------------------------
mp = keyed(joinpath(R, "e15b_mpar_timing.csv"))
push!(rows, csvrow("M_par", NaN, Int(mp["p"]), 1, mp["compilation"],
                   mp["fit_warm"], mp["fit_warm"], mp["fit_warm"],
                   mp["information_warm"], mp["information_warm"], mp["information_warm"],
                   mp["OFV"], mp["OFV"], mp["df_eff"], mp["df_eff"], 1,
                   mp["hessian_regularization"], Int(mp["information_finite"]), 0))

# --- the network at each ridge strength ----------------------------------------------------
for γ in GAMMAS
    g = @sprintf("%g", γ)
    t = keyed(joinpath(R, "e15_timing_g$(g).csv"))
    f = [r for r in tabl(joinpath(R, "e15_fits_g$(g).csv")) if String(r["model"]) != "M_par"]
    fs = num.(getindex.(f, "fit_seconds"))
    is = num.(getindex.(f, "information_seconds"))
    of = num.(getindex.(f, "OFV"))
    df = filter(isfinite, num.(getindex.(f, "df_eff")))
    regs = num.(getindex.(f, "hessian_regularization"))
    nfinite = count(truthy, getindex.(f, "information_finite"))
    nzero = count(==(0.0), num.(getindex.(f, "n_kept")))
    push!(rows, csvrow("M_nn($(WIDTH))", γ, Int(num(f[1]["p"])), length(f), t["compilation"],
                       med(fs), minimum(fs), maximum(fs), med(is), minimum(is), maximum(is),
                       minimum(of), maximum(of),
                       isempty(df) ? NaN : minimum(df), isempty(df) ? NaN : maximum(df),
                       length(df), only(unique(regs)), nfinite, nzero))
end
writedlm(joinpath(R, "e15_summary_table.csv"), vcat(rows...), ',')

println("\n", "="^116)
@printf("%-16s %5s %6s %10s %22s %22s %18s %18s\n", "model", "p", "fits", "compile s",
        "estimation s", "information s", "OFV range", "df_eff")
println("-"^116)
for r in rows[2:end]
    γ = Float64(r[2])
    name = isnan(γ) ? String(r[1]) : @sprintf("%s γ=%g", r[1], γ)
    ofv = Float64(r[12]) == Float64(r[13]) ? @sprintf("%.1f", r[12]) :
          @sprintf("%.1f–%.1f", r[12], r[13])
    nd = Int(r[16]); nf = Int(r[4])
    dfe = isnan(Float64(r[14])) ? "none" :
          (Float64(r[14]) == Float64(r[15]) ? @sprintf("%.1f", r[14]) :
           @sprintf("%.1f–%.1f", r[14], r[15]))
    nd == nf || (dfe *= @sprintf(" [%d/%d]", nd, nf))
    @printf("%-16s %5d %6d %10.1f %22s %22s %18s %18s\n", name, Int(r[3]), nf, Float64(r[5]),
            @sprintf("%.1f (%.1f–%.1f)", r[6], r[7], r[8]),
            @sprintf("%.1f (%.1f–%.1f)", r[9], r[10], r[11]), ofv, dfe)
end
println("="^116)
println("""
Compilation is per session and measured cold-minus-warm in a process of its own for each row.
Estimation and observed information are per fit, median and range over the initialisation seeds.
A bracketed count marks rows where the effective dimension is undefined for some seeds because
the observed information retained no direction above the noise floor.""")
