# E15d — how much of the effective dimension is the network, and how much is θ?
#
#   julia --project=Local_Env warfarin_turnover/examples/regularisation/e15d_df_eff_blocks.jl
#
# `df_eff = tr[(J_τ + K_{γ,τ})⁻¹ J_τ]` is a total over the whole population parameter vector. The
# penalty enters only through `K_γ`, which is `2γ` on the network block and zero elsewhere, but
# the trace still runs over every retained direction, so the reported number counts the twelve
# interpretable parameters together with whatever the network contributes. That is the right
# quantity for comparing models, and the wrong one for answering "how many parameters is the
# network actually using".
#
# The trace splits exactly, because trace is linear and the coordinate blocks partition ψ:
#
#     df_eff = Σ_B tr(E_Bᵀ V_τ (J_τ + K_τ)⁻¹ J_τ V_τᵀ E_B),   B ∈ {θ, φ}
#
# with `E_B` the selector for block `B`. This is E8's block-identifiability measure with the
# shrinkage included, and it reduces to E8's `d_B(τ) = tr(E_Bᵀ P_τ E_B)` when γ = 0.
#
# Reads the information matrices E15 archived. Refits nothing.

include(joinpath(@__DIR__, "..", "..", "src", "warfarin.jl"))
BLAS.set_num_threads(1)

banner("E15d — effective dimension split between the interpretable and network blocks")

const R = results_dir(@__DIR__, "results")
const WIDTH = SCIENTIFIC_WIDTH
const TAU = :noise
const GAMMAS = (0.0, 1.0, 10.0)

m_nn = WarfarinTurnover(MLPEffect(WIDTH))
m_par = WarfarinTurnover(EmaxEffect())
blocks(m) = (phi = collect(theta_index(m).effect),
             theta = setdiff(1:m.p, collect(theta_index(m).effect)))

read_info(path) = (p = Int(readdlm(path * ".dims.csv", ',')[1, 2]);
                   H = Matrix{Float64}(undef, p, p); read!(path, H); H)

"""Total effective dimension and its per-block parts, all from one eigendecomposition."""
function split_df(J::AbstractMatrix, γ::Float64, m)
    all(isfinite, J) || return (total = NaN, dθ = NaN, dφ = NaN,
                                τ = NaN, n_kept = 0)
    b = blocks(m)
    F = eigen(Symmetric(Matrix(J)))
    τ = resolve_tau(TAU, F.values)
    keep = findall(>(τ), F.values)
    isempty(keep) && return (total = NaN, dθ = NaN, dφ = NaN, τ = τ, n_kept = 0)
    V = F.vectors[:, keep]
    Jτ = Diagonal(F.values[keep])
    pd = zeros(size(J, 1)); pd[b.phi] .= 2γ
    Kτ = transpose(V) * (Diagonal(pd) * V)
    M = (Jτ + Kτ) \ Matrix(Jτ)              # the "hat" operator in retained coordinates
    A = V * M * transpose(V)                # back to parameter coordinates
    return (total = tr(M), dθ = sum(A[i, i] for i in b.theta),
            dφ = sum(A[i, i] for i in b.phi), τ = τ, n_kept = length(keep))
end

rows = Any[csvrow("model", "gamma", "seed", "p", "p_theta", "p_phi", "df_eff_total",
                  "df_eff_theta", "df_eff_phi", "frac_theta", "tau", "n_kept")]

println("\n  model            γ   seed   df_eff total    θ block    φ block   of p_φ")
for γ in GAMMAS, s in 1:5
    path = joinpath(R, @sprintf("e15_information_M_nn_g%g_s%d.bin", γ, s))
    isfile(path * ".dims.csv") || continue
    d = split_df(read_info(path), γ, m_nn)
    b = blocks(m_nn)
    push!(rows, csvrow("M_nn($(WIDTH))", γ, s, m_nn.p, length(b.theta), length(b.phi),
                       d.total, d.dθ, d.dφ, d.dθ / d.total, d.τ, d.n_kept))
    @printf("  M_nn(%d)  %6g %6d %13s %10s %10s %8d\n", WIDTH, γ, s,
            isnan(d.total) ? "—" : @sprintf("%.2f", d.total),
            isnan(d.dθ) ? "—" : @sprintf("%.2f", d.dθ),
            isnan(d.dφ) ? "—" : @sprintf("%.2f", d.dφ), length(b.phi))
end
let path = joinpath(R, "e15_information_M_par.bin")
    if isfile(path * ".dims.csv")
        d = split_df(read_info(path), 0.0, m_par)
        b = blocks(m_par)
        push!(rows, csvrow("M_par", 0.0, 0, m_par.p, length(b.theta), length(b.phi),
                           d.total, d.dθ, d.dφ, d.dθ / d.total, d.τ, d.n_kept))
        @printf("  M_par    %6s %6d %13.2f %10.2f %10.2f %8d\n", "—", 0, d.total, d.dθ, d.dφ,
                length(b.phi))
    end
end
writedlm(joinpath(R, "e15_df_eff_blocks.csv"), vcat(rows...), ',')

med(v) = isempty(v) ? NaN : median(v)
println("\n", "="^78)
@printf("%-16s %14s %14s %14s\n", "model", "df_eff total", "θ block", "φ block")
for γ in GAMMAS
    sel = [r for r in rows[2:end] if Float64(r[2]) == γ && String(r[1]) != "M_par" &&
           isfinite(Float64(r[7]))]
    isempty(sel) && (@printf("%-16s %14s %14s %14s\n", "M_nn γ=$(γ)", "none", "none", "none");
                     continue)
    @printf("%-16s %14.2f %14.2f %14.2f   (median over %d of 5 seeds)\n", "M_nn γ=$(γ)",
            med(Float64[r[7] for r in sel]), med(Float64[r[8] for r in sel]),
            med(Float64[r[9] for r in sel]), length(sel))
end
let sel = [r for r in rows[2:end] if String(r[1]) == "M_par"]
    isempty(sel) || @printf("%-16s %14.2f %14.2f %14.2f\n", "M_par",
                            Float64(sel[1][7]), Float64(sel[1][8]), Float64(sel[1][9]))
end
println("""

The θ block is 12 parameters for the network model and 12 for the parametric one; the remainder
of p is the effect block, 141 weights against 2. A θ contribution close to 12 means the
interpretable parameters are fully determined whatever the penalty is doing to the weights.""")
println("=" ^ 78)
