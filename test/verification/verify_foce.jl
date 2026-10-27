# Phase 0 GATE — the FOCE curvature arm, verified on the already-validated neural mixed-
# effects model of reverse_mode_study (whose Laplace reverse gradient passed verify.jl).
# This is pure method work: it establishes that the Gauss-Newton curvature is assembled
# correctly, that its reverse gradient is the exact gradient of the FOCE objective, and that
# the directional (matrix-free) form equals the matrix form — all before any application model
# exists.  Mirrors test/verification/verify_reverse.jl in structure and thresholds.
#
#   julia --project=Local_Env -t 6 test/verification/verify_foce.jl
#
# Verification tier (PLAN_joint_covariate_discovery.md §3.4):
#   F0  Laplace-via-curvature-module reproduces NLMEReverse (delegation sanity)
#   F1  G_FOCE is positive definite at every subject's mode; report min eig, cond
#   F2  FOCE reverse gradient vs central finite differences of the FOCE objective
#   F3  FOCE reverse (Mooncake) vs the forwarddiff control arm and the matrix form
#   F4  directional curvature term == matrix curvature term (the F4 identity)
#   F5a H − G equals the residual-weighted second-derivative term, exactly (the estimator gap, made exact)
#   F5b report the FOCE-vs-Laplace gradient gap (the approximation introduced deliberately)
#   F6  freeze_J = true gives b ≡ 0 here (η-independent R,Ω) — the §3.3 estimator distinction

using LinearAlgebra, Printf, DelimitedFiles, Statistics, Random
using ForwardDiff
using NLMEReverse

const HERE = @__DIR__
const ROOT = normpath(joinpath(HERE, "..", ".."))
include(joinpath(ROOT, "src", "backends", "backends.jl"))
include(joinpath(HERE, "neural_me_model.jl"))

const RESULTS = joinpath(HERE, "results")
mkpath(RESULTS)
res(name) = joinpath(RESULTS, name)

BLAS.set_num_threads(1)

const SEED = 20260729
const N_SUBJECTS = 24
const N_OBS = 10

@printf("julia %s | %d thread(s) | BLAS %d thread(s)\n",
        VERSION, Threads.nthreads(), BLAS.get_num_threads())

# --------------------------------------------------------------------------
# The FOCE interface for the neural ME model: additive Gaussian error (R = σ², η-independent
# ⇒ plain FOCE) and diagonal Ω.  This is the only model-specific code the FOCE arm needs.
# --------------------------------------------------------------------------
function NLMEReverse.conditional_moments(m::NeuralMEModel, η, θ, subject)
    nw = n_weights(m)
    σ2 = exp(2 * θ[nw + 1])
    φ = network(m, θ, subject.covariates.x) .+ η
    K = m.q ÷ 2
    f = [structural(t, φ, K) for t in subject.times]
    R = fill(σ2, length(subject.times))
    Ω = [exp(2 * θ[nw + 1 + k]) for k in 1:m.q]
    return (f = f, R = R, Ω = Ω)
end

# --------------------------------------------------------------------------
model = NeuralMEModel(q = 4, d_x = 3, width = 12, depth = 1)
θ_true = truth_theta(model; seed = SEED)
pop, eta_true = simulate(model, θ_true, N_SUBJECTS, N_OBS; seed = SEED + 1)
θ = perturb(θ_true, 0.10; seed = 77)
p = length(θ)
@assert p == n_parameters(model)
@printf("model: q=%d width=%d depth=%d  p=%d  N=%d  n_obs=%d\n",
        model.q, model.width, model.depth, p, N_SUBJECTS, N_OBS)

cfg = LaplaceConfig(reg = 1e-10, inner_tol = 1e-12, inner_maxiter = 200, threaded = true)
ccfg_lap = CurvatureConfig(curvature = :laplace)
ccfg_foce = CurvatureConfig(curvature = :foce, freeze_J = false)

load_backends!()
# Which reverse engines reproduce the *FOCE* control arm (forwarddiff of the same scalar,
# itself validated against finite differences by F2).  There is no adjoint-FOCE reference in
# NLMEReverse, so the control arm is the reference here.
rc_ctrl = ReverseConfig(backend = :forwarddiff, form = :directional)
g_ctrl = curvature_gradient_parts(model, θ, pop, cfg, rc_ctrl, ccfg_foce)
scale = max(maximum(abs, g_ctrl.grad), 1.0)

rows = Any["check" "quantity" "value" "threshold" "pass"]
record!(check, name, value, threshold) = begin
    ok = isfinite(value) && value <= threshold
    global rows = vcat(rows, [check name value threshold ok])
    ok
end
verdicts = Bool[]

# --------------------------------------------------------------------------
println("\n", "="^78, "\nF0  Laplace via the curvature module reproduces NLMEReverse\n", "="^78)
sel = select_backend(model, θ, pop, cfg; verbose = false)
rcfg = sel.rcfg
@printf("    selected reverse engine for the study: %s / %s\n", rcfg.backend, rcfg.form)
g_lap_mod = curvature_gradient_parts(model, θ, pop, cfg, rcfg, ccfg_lap).grad
g_lap_ref = NLMEReverse.reverse_gradient_parts(model, θ, pop, cfg, rcfg).grad
d0 = maximum(abs, g_lap_mod .- g_lap_ref) / max(maximum(abs, g_lap_ref), 1.0)
@printf("F0  rel|laplace(curvature module) − NLMEReverse| = %.3e\n", d0)
push!(verdicts, record!("F0", "laplace_delegation", d0, 1e-14))

# --------------------------------------------------------------------------
println("\n", "="^78, "\nF1  G_FOCE positive definite at every subject's mode\n", "="^78)
min_eig = Inf
max_cond = 0.0
n_indef = 0
for i in 1:N_SUBJECTS
    sub = pop[i]
    sol = newton_solve(model, θ, sub, zeros(model.q), cfg)
    η̂ = extract_value(sol.η)
    G = foce_curvature(model, η̂, θ, sub)
    ev = eigvals(Symmetric(Matrix(G)))
    global min_eig = min(min_eig, minimum(ev))
    global max_cond = max(max_cond, maximum(ev) / minimum(ev))
    minimum(ev) <= 0 && (global n_indef += 1)
end
@printf("F1  min eig(G) over subjects = %.4e   max cond(G) = %.3e   indefinite: %d\n",
        min_eig, max_cond, n_indef)
push!(verdicts, record!("F1", "n_indefinite_G", n_indef, 0))
rows = vcat(rows, ["F1" "min_eig_G" min_eig NaN true], ["F1" "max_cond_G" max_cond NaN true])

# --------------------------------------------------------------------------
println("\n", "="^78, "\nF2  FOCE reverse gradient vs central FD of the FOCE objective\n", "="^78)
objf = foce_objective(model, pop, cfg, ccfg_foce)
g_foce = curvature_gradient_parts(model, θ, pop, cfg, rcfg, ccfg_foce)
@printf("    inner: nonconverged=%d  worst‖∇_η h‖=%.2e  L_FOCE=%.10f\n",
        g_foce.n_nonconverged, g_foce.worst_gnorm, g_foce.L)
push!(verdicts, record!("F0", "inner_nonconverged", g_foce.n_nonconverged, 0))
best = Inf
for step in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)
    gfd = zeros(p)
    for k in 1:p
        s = step * max(abs(θ[k]), 1.0)
        θp = copy(θ); θp[k] += s; θm = copy(θ); θm[k] -= s
        gfd[k] = (objf(θp) - objf(θm)) / (2s)
    end
    d = maximum(abs, gfd .- g_foce.grad)
    global best = min(best, d)
    @printf("    step=%8.1e  max|FD − FOCE reverse| = %.3e  (rel %.3e)\n", step, d, d / scale)
    global rows = vcat(rows, ["F2" "fd_step_$(step)" d NaN true])
end
@printf("F2  best over steps = %.3e  (rel %.3e)\n", best, best / scale)
push!(verdicts, record!("F2", "rel_fd_best", best / scale, 1e-6))

# --------------------------------------------------------------------------
println("\n", "="^78, "\nF3  FOCE reverse vs control arm and matrix form\n", "="^78)
d_rev = maximum(abs, g_foce.grad .- g_ctrl.grad) / scale
@printf("    %-22s rel diff = %.3e   (reverse mode: %s/%s)\n",
        "reverse − control", d_rev, rcfg.backend, rcfg.form)
push!(verdicts, record!("F3", "reverse_vs_control", d_rev, 1e-8))
for t in sel.table
    t.ok || continue
    gb = curvature_gradient_parts(model, θ, pop, cfg,
                                  ReverseConfig(backend = t.backend, form = t.form), ccfg_foce).grad
    d = maximum(abs, gb .- g_ctrl.grad) / scale
    @printf("    %-12s %-12s rel diff = %.3e\n", t.backend, t.form, d)
    global rows = vcat(rows, ["F3" "$(t.backend)_$(t.form)" d 1e-8 d <= 1e-8])
    push!(verdicts, record!("F3", "$(t.backend)_$(t.form)", d, 1e-8))
end
g_mat_ctrl = curvature_gradient_parts(model, θ, pop, cfg,
                                      ReverseConfig(backend = :forwarddiff, form = :matrix), ccfg_foce).grad
d_mat = maximum(abs, g_mat_ctrl .- g_ctrl.grad) / scale
@printf("    %-22s rel diff = %.3e   (reformulation, not reverse mode)\n", "matrix − directional", d_mat)
push!(verdicts, record!("F3", "matrix_vs_directional", d_mat, 1e-8))

# --------------------------------------------------------------------------
println("\n", "="^78, "\nF4  directional curvature term == matrix curvature term\n", "="^78)
# ½ Σₘ [(Juₘ)ᵀR⁻¹(Juₘ) + uₘᵀΩ⁻¹uₘ]  ==  ½ tr(L⁻¹ G L⁻ᵀ),  at subject 1's mode.
sub = pop[1]
sol = newton_solve(model, θ, sub, zeros(model.q), cfg)
η̂ = extract_value(sol.η)
G1 = foce_curvature(model, η̂, θ, sub)
G̃1 = G1 + cfg.reg * I
C = cholesky(Symmetric(Matrix(G̃1)); check = false)
@assert issuccess(C)
U = Matrix(C.L \ Matrix{Float64}(I, model.q, model.q))
cm = NLMEReverse.conditional_moments(model, η̂, θ, sub)
invR = 1 ./ cm.R; invΩ = 1 ./ cm.Ω
ffun = z -> NLMEReverse.conditional_moments(model, z, θ, sub).f
dir_term = 0.0
for m in 1:model.q
    u = U[m, :]
    Ju = dir1_vec(ffun, η̂, u)
    global dir_term += 0.5 * (sum(invR .* Ju .* Ju) + sum(invΩ .* u .* u))
end
mat_term = 0.5 * tr(U * Matrix(G1) * transpose(U))
@printf("F4  |directional − matrix curvature term| = %.3e\n", abs(dir_term - mat_term))
push!(verdicts, record!("F4", "dir_vs_matrix_curv", abs(dir_term - mat_term), 1e-8 * max(abs(mat_term), 1)))

# --------------------------------------------------------------------------
println("\n", "="^78, "\nF5  H − G is exactly the residual-weighted second-derivative term\n", "="^78)
# Laplace uses H = ∇²_ηη h; FOCE uses G.  Their difference is precisely the Gauss-Newton
# dropped term −Σⱼ rⱼ ∇²_η fⱼ / σ² (rⱼ = yⱼ − fⱼ).  Verifying this to machine precision both
# validates the G assembly against the exact Hessian AND makes exact the statement "FOCE → Laplace as residuals → 0".
function f5_residual_identity(model, θ, pop, cfg, N)
    worst = 0.0
    for i in 1:N
        subi = pop[i]
        soli = newton_solve(model, θ, subi, zeros(model.q), cfg)
        ηi = extract_value(soli.η)
        H = ForwardDiff.hessian(z -> negative_log_joint_density(model, z, θ, subi), ηi)
        G = Matrix(foce_curvature(model, ηi, θ, subi))
        cmv = NLMEReverse.conditional_moments(model, ηi, θ, subi)
        σ2 = cmv.R[1]
        r = subi.observations .- cmv.f
        T = zeros(model.q, model.q)
        for j in eachindex(subi.times)
            Hfj = ForwardDiff.hessian(z -> NLMEReverse.conditional_moments(model, z, θ, subi).f[j], ηi)
            T .-= (r[j] / σ2) .* Hfj
        end
        worst = max(worst, maximum(abs, (H .- G) .- T))
    end
    return worst
end
worst_id = f5_residual_identity(model, θ, pop, cfg, N_SUBJECTS)
@printf("F5a |(H − G) − residual term| over subjects = %.3e\n", worst_id)
push!(verdicts, record!("F5", "H_minus_G_identity", worst_id, 1e-8))

g_lap = curvature_gradient_parts(model, θ, pop, cfg, rcfg, ccfg_lap).grad
gap = maximum(abs, g_foce.grad .- g_lap) / scale
@printf("F5b FOCE-vs-Laplace population gradient gap = %.3e  (rel)  — the deliberate estimator difference\n", gap)
rows = vcat(rows, ["F5" "foce_laplace_gap_rel" gap NaN true])

# --------------------------------------------------------------------------
println("\n", "="^78, "\nF6  freeze_J = true ⇒ b ≡ 0 here (η-independent R,Ω): the §3.3 distinction\n", "="^78)
ccfg_frozen = CurvatureConfig(curvature = :foce, freeze_J = true)
sh_frozen = subject_curvature(model, θ, pop[1], cfg, :directional, ccfg_frozen)
sh_consist = subject_curvature(model, θ, pop[1], cfg, :directional, ccfg_foce)
@printf("F6  ‖b‖∞ frozen-J = %.3e   consistent = %.3e   (consistent ≠ 0 ⇒ ∂J/∂η matters)\n",
        maximum(abs, sh_frozen.b), maximum(abs, sh_consist.b))
push!(verdicts, record!("F6", "frozenJ_b_is_zero", maximum(abs, sh_frozen.b), 1e-12))
rows = vcat(rows, ["F6" "consistent_b_norm" maximum(abs, sh_consist.b) NaN true])

# --------------------------------------------------------------------------
writedlm(res("foce_verification.csv"), rows, ',')
println("\nwrote ", res("foce_verification.csv"))

println("\n", "="^78)
if all(verdicts)
    println("VERDICT: PASS — the FOCE curvature arm is verified.")
    println("  Phase 0 complete. curvature=:foce is the exact gradient of the FOCE objective,")
    println("  matrix-free, and reproduced by the reverse engine to the numerical floor.")
else
    println("VERDICT: FAIL — ", count(!, verdicts), " of ", length(verdicts), " checks outside threshold.")
end
println("="^78)
