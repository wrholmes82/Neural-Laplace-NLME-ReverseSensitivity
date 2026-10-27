# E9c — exact-Laplace OFV and effective dimension for the Figure 4 fits.
#
#   julia --project=Local_Env -t 5 \
#       warfarin_turnover/examples/misspecification/e9c_model_comparison.jl
#
# E9b has already fitted both models and archived every observed-information matrix. This script
# performs no fitting and no covariance calculation: it rescales the archived results into the
# paired model-comparison summary used by Table 3.

include(joinpath(@__DIR__, "..", "..", "src", "warfarin.jl"))

BLAS.set_num_threads(1)
load_backends!(verbose = false)
banner("E9c — exact-Laplace model comparison across misspecification")

const RESULTS = results_dir(@__DIR__, "results")
const WIDTH = SCIENTIFIC_WIDTH
const N_SUBJECTS = 100
const LAMBDA = 1.0
const HESSIAN_REGULARIZATION = 1e-8
const C_MATCH = 10.0
const HILL_N = (1.5, 3.0)

const CFG = default_config(reg = HESSIAN_REGULARIZATION, inner_tol = 1e-8,
                           inner_maxiter = 200, threaded = true)

m_par = WarfarinTurnover(EmaxEffect())
m_nn = WarfarinTurnover(MLPEffect(WIDTH))
m_hill = WarfarinTurnover(HillEffect())
model_of = Dict("M_par" => m_par, "M_nn($(WIDTH))" => m_nn)

tp = truth_theta()
ixp = theta_index(m_par)
EMAX = exp(tp.theta[ixp.effect][1])
EC50 = exp(tp.theta[ixp.effect][2])
function hill_truth(n)
    Et = EMAX * C_MATCH / (EC50 + C_MATCH)
    ec = C_MATCH * ((EMAX - Et) / Et)^(1 / n)
    return vcat(tp.theta[1:5], [log(EMAX), log(ec), log(n)], tp.theta[8:14])
end
TRUTH = Dict{String,Any}("Emax" => (model = m_par, theta = tp.theta, seed0 = SEED_COHORT))
for n in HILL_N
    TRUTH[@sprintf("sigmoid_g%.1f", n)] =
        (model = m_hill, theta = hill_truth(n), seed0 = SEED_MISSPEC + Int(round(1000n)))
end

tagof(fitname) = replace(fitname, "(" => "", ")" => "")
read_info(path) = (p = Int(readdlm(path * ".dims.csv", ',')[1, 2]);
                   H = Matrix{Float64}(undef, p, p); read!(path, H); H)

function read_theta_table(path::String)
    raw, hdr = readdlm(path, ','; header = true)
    h = Dict(String(c) => j for (j, c) in enumerate(vec(hdr)))
    return [(truth = String(raw[i, h["truth"]]), fit = String(raw[i, h["fit_model"]]),
             cohort = Int(round(Float64(raw[i, h["cohort"]]))),
             gamma = Float64(raw[i, h["lambda"]]),
             theta = parse.(Float64, split(strip(String(raw[i, h["theta"]])))))
            for i in axes(raw, 1)]
end

function fit_metadata(path::String)
    raw, hdr = readdlm(path, ','; header = true)
    h = Dict(String(c) => j for (j, c) in enumerate(vec(hdr)))
    out = Dict{Tuple{String,String,Int},NamedTuple}()
    for i in axes(raw, 1)
        key = (String(raw[i, h["truth"]]), String(raw[i, h["fit_model"]]),
               Int(round(Float64(raw[i, h["cohort"]]))))
        out[key] = (information_seconds = Float64(raw[i, h["information_seconds"]]),
                    information_finite =
                        lowercase(string(raw[i, h["information_finite"]])) == "true")
    end
    return out
end

"""Effective ridge dimension within the likelihood-supported eigenspace."""
function effective_dimension(J::AbstractMatrix, gamma::Float64, phi_ix::Vector{Int})
    all(isfinite, J) || return (NaN, 0, NaN, NaN, -1)
    F = eigen(Symmetric(Matrix(J)))
    tau = resolve_tau(:noise, F.values)
    keep = findall(>(tau), F.values)
    isempty(keep) && return (NaN, 0, tau, minimum(F.values), count(<=(0), F.values))
    V = F.vectors[:, keep]
    Jt = Symmetric(V' * J * V)
    if isempty(phi_ix) || gamma == 0
        return (Float64(length(keep)), length(keep), tau, minimum(F.values),
                count(<=(0), F.values))
    end
    K = zeros(size(J))
    for i in phi_ix
        K[i, i] = 2gamma
    end
    Kt = Symmetric(V' * K * V)
    return (tr((Jt + Kt) \ Matrix(Jt)), length(keep), tau, minimum(F.values),
            count(<=(0), F.values))
end

fits = read_theta_table(joinpath(RESULTS, "e9b_theta_hat.csv"))
meta = fit_metadata(joinpath(RESULTS, "e9b_fits.csv"))
length(fits) == 30 || error("Expected 30 exact-Laplace Figure 4 fits; found $(length(fits)).")

rows = Any[csvrow("gamma", "hessian_regularization", "truth", "fit_model", "cohort", "p",
                  "OFV", "df_eff", "n_kept", "tau", "lambda_min",
                  "n_negative_eigenvalues", "phi_norm2", "information_finite",
                  "information_seconds", "information_source")]

for f in fits
    m = model_of[f.fit]
    cfg_fit = CFG
    tr_ = TRUTH[f.truth]
    sim = simulate_cohort(N_SUBJECTS; seed = tr_.seed0 + f.cohort, theta = tr_.theta,
                          model = tr_.model)
    subs = [make_subject(m, s.id, s.input, s.covariates.pk_times,
                         s.observations[1:s.covariates.n_pk], s.covariates.pd_times,
                         s.observations[s.covariates.n_pk+1:end]) for s in sim.pop.subjects]
    pop = PopulationData(subs)
    starts = [collect(Float64, e) for e in population_objective(
        m, theta_start(m), pop, cfg_fit).etas]
    ofv = 2 * foce_objective(m, pop, cfg_fit, CC_LAPLACE; fixed_starts = starts)(f.theta)

    tag = "$(f.truth)_$(tagof(f.fit))_c$(f.cohort)"
    info_path = joinpath(RESULTS, "e9b_information_$(tag).bin")
    isfile(info_path) || error("Missing information artifact: $(info_path)")
    J = read_info(info_path)
    md = meta[(f.truth, f.fit, f.cohort)]
    phi_ix = startswith(f.fit, "M_nn") ? collect(theta_index(m).effect) : Int[]
    gamma_eff = isempty(phi_ix) ? 0.0 : f.gamma
    df, nk, tau, lmin, nneg = effective_dimension(J, gamma_eff, phi_ix)
    push!(rows, csvrow(f.gamma, cfg_fit.reg, f.truth, f.fit, f.cohort, m.p,
                       ofv, df, nk, tau, lmin, nneg,
                       isempty(phi_ix) ? 0.0 : sum(abs2, view(f.theta, phi_ix)),
                       md.information_finite, md.information_seconds,
                       "E9b exact-Laplace archive"))
    @printf("  %-13s %-10s c%d OFV %9.1f df_eff %6.2f kept %3d finite=%s\n",
            f.truth, f.fit, f.cohort, ofv, df, nk, string(md.information_finite))
end
writedlm(joinpath(RESULTS, "e9c_model_comparison.csv"), vcat(rows...), ',')

med(v) = isempty(v) ? NaN : median(v)
NN = "M_nn($(WIDTH))"
get_(t, m, c, k) = (s = [r for r in rows[2:end] if r[3] == t && r[4] == m && r[5] == c];
                     isempty(s) ? NaN : Float64(s[1][k]))

summ = Any[csvrow("gamma", "hessian_regularization",
                  "truth", "n_cohorts", "n_df_eff_nn",
                  "p_par", "p_nn", "median_OFV_par", "median_OFV_nn",
                  "median_dOFV_paired", "median_df_eff_par", "median_df_eff_nn",
                  "median_ddf_eff_paired", "n_covariance_failures", "n_degenerate_nn")]

println("\npaired comparison: negative delta OFV favours the network")
for t in ["Emax", [@sprintf("sigmoid_g%.1f", n) for n in HILL_N]...]
    cohorts = sort(unique(Int(r[5]) for r in rows[2:end] if r[3] == t))
    op = [get_(t, "M_par", c, 7) for c in cohorts]
    on = [get_(t, NN, c, 7) for c in cohorts]
    dp = [get_(t, "M_par", c, 8) for c in cohorts]
    dn = [get_(t, NN, c, 8) for c in cohorts]
    ddf = [dn[i] - dp[i] for i in eachindex(cohorts) if isfinite(dn[i]) && isfinite(dp[i])]
    failures = count(c -> !isfinite(get_(t, "M_par", c, 8)) ||
                          !isfinite(get_(t, NN, c, 8)), cohorts)
    ndeg = count(c -> get_(t, NN, c, 9) < 2, cohorts)
    push!(summ, csvrow(LAMBDA, HESSIAN_REGULARIZATION, t, length(cohorts),
                       count(isfinite, dn), 14, m_nn.p, med(op), med(on), med(on .- op),
                       med(filter(isfinite, dp)), med(filter(isfinite, dn)), med(ddf),
                       failures, ndeg))
    @printf("  %-13s n=%d OFV %9.1f vs %9.1f delta=%8.1f df %5.1f vs %5.1f failures=%d\n",
            t, length(cohorts), med(op), med(on), med(on .- op),
            med(filter(isfinite, dp)), med(filter(isfinite, dn)), failures)
end
writedlm(joinpath(RESULTS, "e9c_summary.csv"), vcat(summ...), ',')
println("=" ^ 92)
