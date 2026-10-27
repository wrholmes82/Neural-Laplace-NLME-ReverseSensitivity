# E1 — the verification gate.
#
#   julia --project=Local_Env -t 5 warfarin_turnover/test/e1_gate.jl
#
# **Fail → stop.** No timing and no statistical result in this study is meaningful until this
# passes. That is the whole point of running it first and of it having no dependencies on
# anything downstream.
#
# What is verified, and why each check is independent evidence rather than a restatement of
# the last one:
#
# | check | compares | independent because |
# |---|---|---|
# | V1 | reverse sweep vs `:forwarddiff` on the **identical scalar** | same scalar, opposite AD mode — isolates the *engine* |
# | V2 | `:directional` vs `:matrix` scalar | two algebraic constructions of `⟨W, ∂H⟩` |
# | V3 | reverse vs forward sensitivity | opposite orientations: `−Bᵀλ` against `Sᵀb` |
# | V4 | reverse vs the FULL-implicit adjoint | a third construction of the same gradient |
# | V5 | threaded vs serial | bitwise, threshold exactly `0.0` |
# | V6 | package objective vs an **independent** Laplace objective | own mode solve and log-determinant |
# | V7 | reverse gradient vs central differences of the objective | numerical, not algebraic |
#
# ReverseDiff is not among the arms: it cannot differentiate this model at all (its broadcast
# rules fail on a ForwardDiff `Dual` nested inside a `TrackedReal`, which the inner
# η-derivatives put there). That is recorded in `docs/DECISIONS.md`; the six independent
# routes above are what stands in for a second reverse engine.
#
# No threshold below is ever relaxed to make a check pass. They are set from what each
# comparison is limited by — machine precision for same-η̂ algebraic identities, the
# finite-difference floor for V7 — and a failure is a bug to find, not a number to raise.

include(joinpath(@__DIR__, "..", "src", "warfarin.jl"))

using Test
BLAS.set_num_threads(1)

# `load_backends!` must be called at top level: it `include`s the engine into `Main`, and a
# `hasmethod` check inside the same call cannot see a method added during it. Called from
# inside a function, it silently returns "nothing loaded" and demotes the whole run to the
# ForwardDiff control arm.
loaded = load_backends!(verbose = true)
backend_available(:mooncake) ||
    error("E1 requires the Mooncake engine; loaded = $(loaded). Without it there is no " *
          "reverse arm to verify and the gate would be checking ForwardDiff against itself.")

banner("E1 — verification gate: warfarin turnover")

const RESULTS = results_dir(@__DIR__, "results")
const N_GATE = 20            # a correctness gate, not an experiment
const SEED_GATE = 20260811
const PERTURB = 0.05

const RC_MC_DIR = ReverseConfig(backend = :mooncake, form = :directional)
const RC_MC_MAT = ReverseConfig(backend = :mooncake, form = :matrix)
const RC_FD_DIR = ReverseConfig(backend = :forwarddiff, form = :directional)

# The measured floating-point floor of ‖∇_η h‖ on this model, and why the gate does not chase
# below it. Trajectory of the package's own solver (`newton_solve` called with
# `inner_maxiter = 1, 2, 3, …`, so this is the real trajectory, not a re-implementation):
#
#   4.9e+02  1.1e+02  1.1e+01  1.7e-01  3.9e-05  2.1e-12  8.6e-13  4.5e-13  2.9e-13  ...
#
# Textbook quadratic convergence in 6-8 iterations, then a hard floor around 5e-13 that it
# bounces around indefinitely. A perturbed θ does not change this.
#
# So an `inner_tol` at or below the floor does not buy accuracy — it buys a handful of subjects
# that never satisfy the test and run to `inner_maxiter`. At `inner_tol = 1e-12` this gate
# reported `mean inner iterations = 66.1` with 3 of 20 non-converged, which is exactly
# (17 × 8 + 3 × 400) / 20 = 66.8: seventeen subjects converging normally and three pinned at the
# cap. Not a slow solver — three subjects on the wrong side of the noise floor.
#
# `1e-10` sits safely above the worst observed floor (1.4e-12) and still leaves V1--V4
# comparing contractions built from a mode located to ten significant figures, which is far
# tighter than those checks need — they compare contractions from the *same* η̂ and are limited
# by linear algebra, not by the mode.
const INNER_FLOOR = 1e-10

# `inner_tol` is tight on purpose. V1--V4 compare contractions built from the *same* η̂ and
# are limited by linear algebra, not by the mode; V6 and V7 are limited by the mode, so a
# loose inner tolerance would show up there and nowhere else, which is exactly the confusion
# a gate is supposed to prevent.
const CFG = LaplaceConfig(reg = 1e-8, inner_tol = 1e-10, inner_maxiter = 400,
                          warm_start = false, threaded = true)
const CFG_SERIAL = LaplaceConfig(reg = CFG.reg, inner_tol = CFG.inner_tol,
                                 inner_maxiter = CFG.inner_maxiter, warm_start = false,
                                 threaded = false)

rows = Any[["model" "p" "check" "quantity" "value" "threshold" "pass"]]
allpass = Ref(true)

function chk!(model, p, name, what, value, thresh)
    ok = isfinite(value) && value <= thresh
    ok || (allpass[] = false)
    push!(rows, [model p name what value thresh ok])
    @printf("  %-6s %-38s %12.4e  <= %8.1e   %s\n", name, what, value, thresh, ok ? "PASS" : "FAIL")
    flush(stdout)
    return ok
end

function info!(model, p, name, what, value)
    push!(rows, [model p name what value NaN ""])
    @printf("  %-6s %-38s %12.4e  %s\n", name, what, value, "(info)")
    flush(stdout)
end

# ===========================================================================
# An INDEPENDENT Laplace objective.
#
# Nothing below calls `newton_solve`, `subject_objective`, `htilde`, `logdet_pd` or any other
# inner machinery from `src/`. The conditional mode is
# found by `Optim`'s own Newton method over a `TwiceDifferentiable` built by ForwardDiff; the
# curvature is assembled here; the log-determinant is taken from a Cholesky factor obtained
# here. If this and the package agree, they agree because the mathematics is right, not
# because they share an implementation.
# ===========================================================================

"""
A hand-written modified-Newton solve for the conditional mode.

Deliberately *not* `Optim` and deliberately not `newton_solve`. The package lifts an
indefinite Hessian by a Cholesky-driven Levenberg shift and backtracks by a damping factor
with an Armijo test on a trust-radius-capped step; this clamps the **eigenvalues** and
backtracks by halving with no trust radius. Two different algorithms converging to the same
stationary point of the same `h` is the evidence; sharing an algorithm would not be.
"""
function independent_mode(m, θ, subject; q = n_random_effects(m), maxit = 100, tol = INNER_FLOOR)
    f = z -> NLMEReverse.negative_log_joint_density(m, z, θ, subject)
    η = zeros(q)
    iters = 0
    for _ in 1:maxit
        iters += 1
        g = ForwardDiff.gradient(f, η)
        maximum(abs, g) < tol && break
        E = eigen(Symmetric(ForwardDiff.hessian(f, η)))
        λ = max.(E.values, 1e-8 * max(maximum(E.values), 1.0))     # clamp, do not shift
        d = -(E.vectors * ((transpose(E.vectors) * g) ./ λ))
        f0 = f(η)
        slope = dot(g, d)
        α = 1.0
        # Bail out of the backtrack as soon as the step is numerically meaningless. Without
        # this the Armijo test can never be satisfied at the mode — `slope` is ~0 there and the
        # comparison is decided by rounding — so every iteration burns the full halving budget.
        while α > 1e-10 && !(f(η .+ α .* d) <= f0 + 1e-4 * α * slope)
            α *= 0.5
        end
        α <= 1e-10 && break
        η = η .+ α .* d
    end
    return (eta = η, value = f(η), iters = iters,
            gnorm = maximum(abs, ForwardDiff.gradient(f, η)))
end

function independent_objective(m, θ, pop; reg = CFG.reg)
    q = n_random_effects(m)
    n = pop.n_subjects
    Ls = zeros(Float64, n)
    gs = zeros(Float64, n)
    its = zeros(Int, n)
    Threads.@threads for i in 1:n
        s = pop.subjects[i]
        sol = independent_mode(m, θ, s; q = q)
        gs[i] = sol.gnorm
        its[i] = sol.iters
        M = ForwardDiff.hessian(
            z -> NLMEReverse.negative_log_joint_density(m, z, θ, s), sol.eta)
        C = cholesky(Symmetric(M + reg * I); check = false)
        issuccess(C) || error("independent objective: curvature not PD for subject $(s.id).")
        Ls[i] = sol.value + sum(log, diag(C.U)) - (q / 2) * log(2π)
    end
    # Serial reduction in index order, so this reference value is itself independent of the
    # thread schedule — the same determinism contract every gradient in `src/` obeys.
    return (L = sum(Ls), worst_inner_gnorm = maximum(gs), max_iters = maximum(its))
end

# ===========================================================================
# The gate, run over both effect forms and two widths.
# ===========================================================================

function gate!(label::AbstractString, m::WarfarinTurnover, pop::PopulationData, θ::Vector{Float64})
    p = m.p
    println("\n" * "-"^78)
    println("$label   p = $p   N = $(pop.n_subjects)   n_obs = $(n_observations(pop).total)")
    println("-"^78)
    flush(stdout)

    # ---- the arms ---------------------------------------------------------
    rev_dir = reverse_gradient_parts(m, θ, pop, CFG, RC_MC_DIR)
    rev_mat = reverse_gradient_parts(m, θ, pop, CFG, RC_MC_MAT)
    ctl_dir = reverse_gradient_parts(m, θ, pop, CFG, RC_FD_DIR)
    fwd = forward_gradient_parts(m, θ, pop, CFG)
    adj = adjoint_gradient_parts(m, θ, pop, CFG)

    g = rev_mat.grad
    scale = max(maximum(abs, g), 1.0)
    rel(a, b, s) = maximum(abs, a .- b) / s

    # ---- V1  engine: reverse sweep vs ForwardDiff on the identical scalar ---
    chk!(label, p, "V1", "reverse vs forward-mode scalar", rel(rev_dir.grad, ctl_dir.grad, scale), 1e-11)

    # ---- V2  construction: directional vs matrix scalar --------------------
    chk!(label, p, "V2", "directional vs matrix scalar", rel(rev_dir.grad, g, scale), 1e-11)

    # ---- V3/V4  independent analytic constructions -------------------------
    chk!(label, p, "V3", "reverse vs forward sensitivity", rel(g, fwd.grad, scale), 1e-9)
    chk!(label, p, "V4", "reverse vs FULL-implicit adjoint", rel(g, adj.grad, scale), 1e-9)

    # ---- V5  thread independence, bitwise ---------------------------------
    ser = reverse_gradient_parts(m, θ, pop, CFG_SERIAL, RC_MC_MAT)
    chk!(label, p, "V5", "threaded vs serial (bitwise)", maximum(abs, g .- ser.grad), 0.0)

    # ---- V6  objective value vs an independent implementation --------------
    indep = independent_objective(m, θ, pop)
    pkg = laplace_objective(m, pop, CFG)(θ)
    chk!(label, p, "V6", "OFV vs independent implementation",
         abs(2pkg - 2indep.L) / max(abs(2indep.L), 1.0), 1e-8)
    info!(label, p, "V6a", "Laplace OFV", 2pkg)
    info!(label, p, "V6b", "independent inner worst |grad_eta|", indep.worst_inner_gnorm)
    info!(label, p, "V6c", "independent inner max iterations", indep.max_iters)

    # ---- V7  central differences of the objective --------------------------
    # A numerical check, so it catches an algebra error that both analytic routes share.
    # Swept over steps and judged at the best one: the truncation error falls as h² and the
    # noise floor rises as 1/h, so a single step proves nothing either way.
    obj = laplace_objective(m, pop, CFG)
    js = unique(round.(Int, range(1, p; length = min(p, 12))))
    best_err = Inf
    best_step = NaN
    for step in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
        err = 0.0
        for j in js
            hj = step * max(abs(θ[j]), 1.0)
            θp = copy(θ); θp[j] += hj
            θm = copy(θ); θm[j] -= hj
            fd = (obj(θp) - obj(θm)) / (2hj)
            err = max(err, abs(fd - g[j]) / scale)
        end
        info!(label, p, "V7s", "FD rel err at step $(step)", err)
        err < best_err && (best_err = err; best_step = step)
    end
    chk!(label, p, "V7", "reverse vs central FD (best step)", best_err, 1e-5)
    info!(label, p, "V7a", "FD best step", best_step)

    # ---- diagnostics --------------------------------------------------------
    info!(label, p, "D1", "inner non-converged", rev_mat.n_nonconverged)
    info!(label, p, "D2", "worst inner |grad_eta|", rev_mat.worst_gnorm)
    info!(label, p, "D3", "mean inner Newton iterations", mean(rev_mat.iters))
    info!(label, p, "D4", "max |b|", maximum(v -> maximum(abs, v), rev_mat.b))
    info!(label, p, "D5", "max |lambda|", maximum(v -> maximum(abs, v), rev_mat.lambda))
    info!(label, p, "D6", "||grad||_inf", maximum(abs, g))

    # Positive-definiteness of the exact-Laplace curvature, per E2's diagnostic requirement.
    npd = 0
    for s in pop.subjects
        sh = subject_reverse(m, θ, s, CFG, RC_MC_MAT)
        issuccess(cholesky(Symmetric(Float64.(sh.Htilde)); check = false)) || (npd += 1)
    end
    info!(label, p, "D7", "subjects with non-PD Laplace Htilde", npd)

    return nothing
end

# ===========================================================================
# Discretisation diagnostic — how much of the OFV is the RK4 grid?
#
# Reported, not asserted at a tight threshold. Every arm of every comparison shares one grid,
# so this cannot affect V1–V7; it matters because the study reports an OFV on the standard
# scale, and an OFV is only on the standard scale if the discretisation is far below the
# resolution anyone reads it at.
# ===========================================================================

function discretisation_probe(effect_ctor, θ_of, seed)
    println("\n" * "-"^78)
    println("Discretisation: OFV against the RK4 node count")
    println("-"^78)
    base = nothing
    # The gate is deliberately self-contained: it simulates from `theta_start`, never from
    # `truth_par.csv`. E1 must be runnable before the truth fit exists — it is the gate that
    # says whether the machinery the truth fit will use is correct.
    ref = WarfarinTurnover(EmaxEffect())
    for ns in (32, 64, 128, 256)
        m = WarfarinTurnover(effect_ctor(); n_seg = ns)
        sim = simulate_cohort(N_GATE; seed = seed, theta = theta_start(ref), model = ref)
        # rebuild the cohort's subjects on this grid, keeping the same observations
        subs = [make_subject(m, s.id, s.input, s.covariates.pk_times,
                             s.observations[1:s.covariates.n_pk], s.covariates.pd_times,
                             s.observations[s.covariates.n_pk+1:end]) for s in sim.pop.subjects]
        pop = PopulationData(subs)
        θ = θ_of(m)
        ofv = 2 * laplace_objective(m, pop, CFG)(θ)
        base === nothing && (base = ofv)
        d = abs(ofv - base)
        @printf("  n_seg=%4d  nodes=%4d  OFV=%18.10f   |OFV − OFV(n_seg=32)| = %.3e\n",
                ns, length(subs[1].covariates.grid), ofv, d)
        push!(rows, ["discretisation" m.p "D9" "OFV at n_seg=$(ns)" ofv NaN ""])
    end
    return nothing
end

# ===========================================================================
# Run
# ===========================================================================

perturb(θ, seed) = θ .+ PERTURB .* randn(MersenneTwister(seed), length(θ))

# M_par first — the reference form, and the one whose gradient is small enough to read.
let m = WarfarinTurnover(EmaxEffect())
    sim = simulate_cohort(N_GATE; seed = SEED_GATE, theta = theta_start(m), model = m)
    gate!("M_par", m, sim.pop, perturb(theta_start(m), SEED_GATE + 1))
end

# M_nn at two widths — the model the study is actually about, at the scientific width and at
# a width small enough that the finite-difference sweep is affordable over every parameter.
for w in (5, SCIENTIFIC_WIDTH)
    m = WarfarinTurnover(MLPEffect(w))
    sim = simulate_cohort(N_GATE; seed = SEED_GATE, theta = theta_start(WarfarinTurnover(EmaxEffect())),
                          model = WarfarinTurnover(EmaxEffect()))
    subs = [make_subject(m, s.id, s.input, s.covariates.pk_times,
                         s.observations[1:s.covariates.n_pk], s.covariates.pd_times,
                         s.observations[s.covariates.n_pk+1:end]) for s in sim.pop.subjects]
    gate!("M_nn(W=$w)", m, PopulationData(subs), perturb(theta_start(m), SEED_GATE + 2))
end

discretisation_probe(EmaxEffect, theta_start, SEED_GATE)

# ---------------------------------------------------------------------------
push!(rows, ["meta" 0 "nthreads" "Threads.nthreads()" Threads.nthreads() NaN ""])
push!(rows, ["meta" 0 "blas_threads" "BLAS.get_num_threads()" BLAS.get_num_threads() NaN ""])
push!(rows, ["meta" 0 "n_subjects" "N" N_GATE NaN ""])
push!(rows, ["meta" 0 "inner_tol" "LaplaceConfig.inner_tol" CFG.inner_tol NaN ""])
push!(rows, ["meta" 0 "reg" "LaplaceConfig.reg" CFG.reg NaN ""])
push!(rows, ["meta" 0 "seed" "SEED_GATE" SEED_GATE NaN ""])

writedlm(joinpath(RESULTS, "e1_gate.csv"), vcat(rows...), ',')
println("\nwrote ", joinpath(RESULTS, "e1_gate.csv"))

println("\n" * "="^78)
if allpass[]
    println("VERDICT: PASS — the exact-Laplace reverse gradient is verified on this model.")
    println("  Proceed to E2 (cost ladder).")
else
    println("VERDICT: FAIL — STOP. Nothing downstream of E1 is meaningful until this passes.")
end
println("="^78)
allpass[] || exit(1)
