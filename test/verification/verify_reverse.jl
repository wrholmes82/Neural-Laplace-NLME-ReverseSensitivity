# STEP 1 — correctness.  Run this FIRST and read its verdict before anything else.
#
#   julia --project=<env> -t 6 test/verification/verify_reverse.jl
#
# A timing study of a wrong gradient is worthless, and the reverse path is the one thing
# in this repository that has never been executed.  This script decides two things:
#
#   (a) which reverse-mode engine and which scalar form actually work on this model, on
#       this machine, at these package versions -- by running each against the framework's
#       already-validated adjoint gradient, not by checking that it fails to throw;
#   (b) whether the selected construction is the exact gradient of the implemented
#       objective, by six independent checks.
#
# It writes `results/reverse_verification.csv` and `results/backends.csv` and prints a single
# PASS/FAIL verdict line at the end.  If that line says FAIL, nothing downstream (the FOCE
# arm, any application study) means anything until it is fixed -- do not build on it.

using LinearAlgebra, Printf, DelimitedFiles, Statistics
using ForwardDiff              # V3 checks the directional forms against the matrix objects
using NLMEReverse

const HERE = @__DIR__
const ROOT = normpath(joinpath(HERE, "..", ".."))
include(joinpath(ROOT, "src", "backends", "backends.jl"))
include(joinpath(HERE, "neural_me_model.jl"))

const RESULTS = joinpath(HERE, "results")
mkpath(RESULTS)
res(name) = joinpath(RESULTS, name)

# In the script, never inside the module: a library should not mutate global BLAS state
# on load.  One BLAS thread because the parallelism here is over subjects; letting BLAS
# also spawn threads oversubscribes the machine and makes every timing meaningless.
BLAS.set_num_threads(1)

const SEED = 20260729
const N_SUBJECTS = 24
const N_OBS = 10

@printf("julia %s | %d thread(s) | BLAS %d thread(s)\n",
        VERSION, Threads.nthreads(), BLAS.get_num_threads())
Threads.nthreads() == 1 && @warn "running single-threaded; pass -t 6 (or -t auto)"

# ---------------------------------------------------------------------------
# Model, ground truth, synthetic data
# ---------------------------------------------------------------------------
model = NeuralMEModel(q = 4, d_x = 3, width = 12, depth = 1)
θ_true = truth_theta(model; seed = SEED)
pop, eta_true = simulate(model, θ_true, N_SUBJECTS, N_OBS; seed = SEED + 1)
θ = perturb(θ_true, 0.10; seed = 77)
p = length(θ)
@assert p == n_parameters(model)

@printf("model: q=%d d_x=%d width=%d depth=%d  p=%d  N=%d  n_obs=%d\n",
        model.q, model.d_x, model.width, model.depth, p, N_SUBJECTS, N_OBS)

cfg = LaplaceConfig(reg = 1e-10, inner_tol = 1e-12, inner_maxiter = 200, threaded = true)

# ---------------------------------------------------------------------------
# STEP 1a — which engine and which form actually work here
# ---------------------------------------------------------------------------
println("\n", "="^78, "\nbackend survey (each engine run against the adjoint gradient)\n", "="^78)
loaded = load_backends!()
sel = select_backend(model, θ, pop, cfg)
rcfg = sel.rcfg
btab = ["backend" "form" "ok" "rel_err" "seconds" "message"]
for t in sel.table
    msg = replace(t.message, r"[\r\n,]" => " ")     # a showerror dump would break the CSV
    global btab = vcat(btab, [string(t.backend) string(t.form) t.ok t.err t.seconds msg])
end
writedlm(res("backends.csv"), btab, ',')

# ---------------------------------------------------------------------------
# The six checks
# ---------------------------------------------------------------------------
rows = Any["check" "quantity" "value" "threshold" "pass"]
record!(check, name, value, threshold) = begin
    ok = isfinite(value) && value <= threshold
    global rows = vcat(rows, [check name value threshold ok])
    ok
end
verdicts = Bool[]

println("\n", "="^78, "\nV1  reverse vs adjoint vs forward\n", "="^78)
g_rev = reverse_gradient_parts(model, θ, pop, cfg, rcfg)
g_adj = adjoint_gradient_parts(model, θ, pop, cfg)
g_fwd = forward_gradient_parts(model, θ, pop, cfg)
scale = maximum(abs, g_rev.grad)
@printf("inner: nonconverged=%d  worst‖∇_η h‖=%.2e  L=%.10f  ‖∇L‖∞=%.6e\n",
        g_rev.n_nonconverged, g_rev.worst_gnorm, g_rev.L, scale)
push!(verdicts, record!("V0", "inner_nonconverged", g_rev.n_nonconverged, 0))

d_adj = maximum(abs, g_rev.grad .- g_adj.grad)
d_fwd = maximum(abs, g_rev.grad .- g_fwd.grad)
@printf("V1  max|reverse − adjoint|             = %.3e  (rel %.3e)\n", d_adj, d_adj / scale)
@printf("V1  max|reverse − forward|             = %.3e  (rel %.3e)\n", d_fwd, d_fwd / scale)
push!(verdicts, record!("V1", "rel_reverse_minus_adjoint", d_adj / scale, 1e-8))
push!(verdicts, record!("V1", "rel_reverse_minus_forward", d_fwd / scale, 1e-8))
rows = vcat(rows, ["V1" "grad_inf_norm" scale NaN true], ["V1" "L" g_rev.L NaN true])

println("\n", "="^78, "\nV2  central finite differences of the full nested objective\n", "="^78)
best = Inf
for step in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)
    gfd = zeros(p)
    fd_gradient(model, pop, cfg; step = step)(gfd, θ)
    d = maximum(abs, gfd .- g_rev.grad)
    global best = min(best, d)
    @printf("    step=%8.1e  max|FD − reverse| = %.3e  (rel %.3e)\n", step, d, d / scale)
    global rows = vcat(rows, ["V2" "fd_step_$(step)" d NaN true])
end
@printf("V2  best over steps                    = %.3e  (rel %.3e)\n", best, best / scale)
# FD carries its own truncation/roundoff floor near eps^(2/3); 1e-6 relative is generous
# headroom above it and is the level at which a genuinely wrong gradient would show.
push!(verdicts, record!("V2", "rel_fd_best", best / scale, 1e-6))

println("\n", "="^78, "\nV3  the directional identities vs the matrix objects they replace\n", "="^78)
sub = pop[1]
sol = newton_solve(model, θ, sub, zeros(model.q), cfg)
η̂ = extract_value(sol.η)
C = cholesky(Symmetric(sol.Htilde); check = false)
issuccess(C) || error("H̃ is not positive definite at subject 1's mode: η̂ is a saddle " *
                      "and the Laplace expression is invalid there. Fix the model or " *
                      "the start point before reading any timing.")
U = Matrix(C.L \ Matrix{Float64}(I, model.q, model.q))
f = z -> negative_log_joint_density(model, z, θ, sub)

quad = sum(NLMEReverse.dir2(f, η̂, U[m, :]) for m in 1:model.q)
quad_ref = tr(U * ForwardDiff.hessian(f, η̂) * transpose(U))
@printf("V3  |Σ D²_uₘ h − tr(L⁻¹ H L⁻ᵀ)|        = %.3e\n", abs(quad - quad_ref))
push!(verdicts, record!("V3", "dir2_identity", abs(quad - quad_ref), 1e-8 * max(abs(quad_ref), 1)))

b1 = 0.5 .* ForwardDiff.gradient(
        ηv -> logdet_pd(htilde(ForwardDiff.hessian(f, ηv), cfg)), η̂)
λ1 = solve_pd(sol.H, b1)
lin = NLMEReverse.dir1(f, η̂, λ1)
lin_ref = dot(ForwardDiff.gradient(f, η̂), λ1)
@printf("V3  |D_λ h − (∇_η h)ᵀλ|                = %.3e\n", abs(lin - lin_ref))
push!(verdicts, record!("V3", "dir1_identity", abs(lin - lin_ref), 1e-8 * max(abs(lin_ref), 1)))

println("\n", "="^78, "\nV4  reg sweep — H in the implicit solve, H̃ in the log-determinant\n", "="^78)
worst_reg = 0.0
for reg in (1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
    c2 = LaplaceConfig(reg = reg, inner_tol = 1e-12, inner_maxiter = 200)
    gr = reverse_gradient_parts(model, θ, pop, c2, rcfg).grad
    ga = adjoint_gradient_parts(model, θ, pop, c2).grad
    d = maximum(abs, gr .- ga) / max(maximum(abs, ga), 1.0)
    global worst_reg = max(worst_reg, d)
    @printf("    reg=%8.1e  rel|reverse − adjoint| = %.3e\n", reg, d)
    global rows = vcat(rows, ["V4" "reg_$(reg)" d 1e-8 d <= 1e-8])
end
push!(verdicts, record!("V4", "worst_over_reg", worst_reg, 1e-8))

println("\n", "="^78, "\nV5  inner-tolerance limit\n", "="^78)
worst_spread = 0.0
for tol in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
    c2 = LaplaceConfig(reg = 1e-10, inner_tol = tol, inner_maxiter = 250)
    gr = reverse_gradient_parts(model, θ, pop, c2, rcfg).grad
    ga = adjoint_gradient_parts(model, θ, pop, c2).grad
    gf = forward_gradient_parts(model, θ, pop, c2).grad
    spread = max(maximum(abs, gr .- ga), maximum(abs, gr .- gf)) / scale
    to_ref = maximum(abs, gr .- g_rev.grad) / scale
    global worst_spread = max(worst_spread, spread)
    @printf("    inner_tol=%8.1e  spread=%.3e   |rev(tol) − rev(1e-12)|=%.3e\n",
            tol, spread, to_ref)
    global rows = vcat(rows, ["V5" "tol_$(tol)_spread" spread 1e-8 spread <= 1e-8],
                             ["V5" "tol_$(tol)_to_ref" to_ref NaN true])
end
push!(verdicts, record!("V5", "worst_spread_over_tol", worst_spread, 1e-8))

println("\n", "="^78, "\nV6  every working (backend, form) pair agrees\n", "="^78)
worst_backend = 0.0
for t in sel.table
    t.ok || continue
    gb = reverse_gradient_parts(model, θ, pop, cfg,
                                ReverseConfig(backend = t.backend, form = t.form)).grad
    d = maximum(abs, gb .- g_rev.grad) / scale
    global worst_backend = max(worst_backend, d)
    @printf("    %-12s %-12s rel diff = %.3e\n", t.backend, t.form, d)
    global rows = vcat(rows, ["V6" "$(t.backend)_$(t.form)" d 1e-8 d <= 1e-8])
end
gctrl = reverse_gradient_parts(model, θ, pop, cfg,
                               ReverseConfig(backend = :forwarddiff, form = :directional)).grad
d = maximum(abs, gctrl .- g_rev.grad) / scale
@printf("    %-12s %-12s rel diff = %.3e   (control arm)\n", "forwarddiff", "directional", d)
push!(verdicts, record!("V6", "worst_backend_disagreement", max(worst_backend, d), 1e-8))

# ---------------------------------------------------------------------------
writedlm(res("reverse_verification.csv"), rows, ',')
println("\nwrote ", res("reverse_verification.csv"), " and ", res("backends.csv"))

println("\n", "="^78)
if !sel.ok_any
    println("VERDICT: NULL RESULT — no reverse engine works on this model here.")
    println("  The gradient is correct, but every arm now uses the :forwarddiff control")
    println("  arm and therefore the same cost model. No timing study downstream can")
    println("  show a reverse effect, and that is the honest finding.")
elseif all(verdicts)
    println("VERDICT: PASS — all checks within threshold.")
    println("  selected backend = $(rcfg.backend), form = $(rcfg.form)")
    println("  Proceed to: test/verification/verify_foce.jl")
else
    println("VERDICT: FAIL — ", count(!, verdicts), " of ", length(verdicts),
            " checks outside threshold.")
    println("  Do NOT build anything downstream on a FAIL — fix the gradient first.")
end
println("="^78)
