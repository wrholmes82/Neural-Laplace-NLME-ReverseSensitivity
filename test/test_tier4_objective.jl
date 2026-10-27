# Tier 4 -- the Laplace objective.
#
# The middle group of tests is the parallelism contract of the plan's §3.3, and
# they are the only tests in the suite that can catch a race. Determinism,
# thread-count independence and serial-path equivalence are asserted BITWISE:
# every per-subject task writes to its own index and the reduction is a serial
# sum in index order, so the schedule cannot influence the result. Anything
# looser would let a shared ForwardDiff config or a shared counter through as an
# intermittently flaky test rather than a failing one.

using Optim

"Central finite-difference Hessian, written here so it shares no code with the framework."
function t4_fd_hessian(f, η, h)
    q = length(η)
    H = zeros(q, q)
    f0 = f(η)
    for j in 1:q
        ep = copy(η); ep[j] += h
        em = copy(η); em[j] -= h
        H[j, j] = (f(ep) - 2 * f0 + f(em)) / h^2
        for k in (j + 1):q
            pp = copy(η); pp[j] += h; pp[k] += h
            pm = copy(η); pm[j] += h; pm[k] -= h
            mp = copy(η); mp[j] -= h; mp[k] += h
            mm = copy(η); mm[j] -= h; mm[k] -= h
            v = (f(pp) - f(pm) - f(mp) + f(mm)) / (4h^2)
            H[j, k] = v; H[k, j] = v
        end
    end
    return H
end

"A population whose expensive subjects sit ADJACENT at the front of the index range."
function t4_frontloaded_population(m, θ; n_light = 60, n_heavy = 4, seed = 777)
    rng = MersenneTwister(seed)
    ω = [exp(θ[4 + k]) for k in 1:4]
    σ = exp(θ[9])
    mk = function (id, times)
        η = ω .* randn(rng, 4)
        CL = exp(θ[1] + η[1]); V1 = exp(θ[2] + η[2])
        Q = exp(θ[3] + η[3]); V2 = exp(θ[4] + η[4])
        C = [twocomp_conc(t, CL, V1, Q, V2, TWOCOMP_INPUT) for t in times]
        SubjectData(id, times, C .+ σ .* C .* randn(rng, length(C)), TWOCOMP_INPUT)
    end
    heavy = [mk(i, collect(range(0.0, 60.0; length = 600))) for i in 1:n_heavy]
    light = [mk(n_heavy + i, TWOCOMP_TIMES) for i in 1:n_light]
    return PopulationData(vcat(heavy, light))
end

@testset "Tier 4 -- Laplace objective" begin
    m = TwoComp()
    pop, θtrue, _ = simulate_twocomp(m; N = 16, seed = 12345)
    θ = θtrue .+ 0.05
    cfg = LaplaceConfig(inner_tol = 1e-10, inner_maxiter = 100)
    q = n_random_effects(m)

    @testset "population objective is the sum of subject objectives" begin
        r = population_objective(m, θ, pop, cfg)
        direct = sum(subject_objective(m, θ, pop[i], cfg).L for i in 1:pop.n_subjects)
        @test r.L === direct                        # bitwise: serial sum, index order
        @test r.n_nonconverged == 0
        for i in 1:pop.n_subjects
            @test r.Ls[i] === subject_objective(m, θ, pop[i], cfg).L
        end
    end

    @testset "factories are transparent wrappers" begin
        L = population_objective(m, θ, pop, cfg).L
        @test laplace_objective(m, pop, cfg)(θ) === L
        @test laplace_ofv(m, pop, cfg)(θ) === 2L
    end

    @testset "first-order optimality at the returned mode" begin
        # Every line of the adjoint derivation rests on ∇_η h(η̂) = 0. Recompute
        # the gradient rather than trusting the reported gnorm.
        r = population_objective(m, θ, pop, cfg)
        for i in 1:pop.n_subjects
            g = ForwardDiff.gradient(e -> negative_log_joint_density(m, e, θ, pop[i]), r.etas[i])
            @test norm(g) < cfg.inner_tol
        end
    end

    @testset "determinism -- repeated evaluation is bitwise identical" begin
        f = laplace_objective(m, pop, cfg)
        v = f(θ)
        for _ in 1:20
            @test f(θ) === v
        end
        g! = adjoint_gradient(m, pop, cfg)
        G0 = zeros(9); g!(G0, θ)
        G1 = zeros(9)
        for _ in 1:5
            g!(G1, θ)
            @test all(G1 .=== G0)
        end
    end

    @testset "serial path equivalence -- threaded=false is bitwise identical" begin
        cser = LaplaceConfig(inner_tol = 1e-10, inner_maxiter = 100, threaded = false)
        @test population_objective(m, θ, pop, cser).L ===
              population_objective(m, θ, pop, cfg).L
        Gt = zeros(9); Gs = zeros(9)
        adjoint_gradient(m, pop, cfg)(Gt, θ)
        adjoint_gradient(m, pop, cser)(Gs, θ)
        @test all(Gt .=== Gs)
    end

    @testset "thread-count independence -- subprocess at -t 1 is bitwise identical" begin
        # The test that catches a shared ForwardDiff config, a shared counter, or
        # any other race DETERMINISTICALLY. Tier 5's cross-validation would catch
        # the same faults only intermittently.
        script = joinpath(mktempdir(), "t4_threads.jl")
        open(script, "w") do io
            print(io, """
            using LinearAlgebra
            BLAS.set_num_threads(1)
            using NLMEReverse
            include(raw"$(joinpath(PROBLEMS, "twocomp.jl"))")
            include(raw"$(joinpath(PROBLEMS, "simulate.jl"))")
            m = TwoComp()
            pop, th, _ = simulate_twocomp(m; N = 16, seed = 12345)
            th = th .+ 0.05
            cfg = LaplaceConfig(inner_tol = 1e-10, inner_maxiter = 100)
            L = population_objective(m, th, pop, cfg).L
            G = zeros(9); adjoint_gradient(m, pop, cfg)(G, th)
            println(repr(L))
            println(repr(G))
            println(Threads.nthreads())
            """)
        end
        cmd = `$(Base.julia_cmd()[1]) --project=$(joinpath(ROOT_DIR, "Local_Env")) -t 1 $script`
        out = read(cmd, String)
        lines = split(strip(out), '\n')
        @test length(lines) >= 3
        Lsub = eval(Meta.parse(lines[end-2]))
        Gsub = eval(Meta.parse(lines[end-1]))
        @test parse(Int, lines[end]) == 1          # the subprocess really ran single-threaded

        Lhere = population_objective(m, θ, pop, cfg).L
        Ghere = zeros(9); adjoint_gradient(m, pop, cfg)(Ghere, θ)
        @test Lsub === Lhere
        @test all(Gsub .=== Ghere)
    end

    @testset "load balance -- :greedy survives adjacent expensive subjects" begin
        if Threads.nthreads() == 1
            @test_skip "needs more than one thread"
        else
            popfl = t4_frontloaded_population(m, θtrue)
            cthr = LaplaceConfig(inner_tol = 1e-8, inner_maxiter = 100)
            cser = LaplaceConfig(inner_tol = 1e-8, inner_maxiter = 100, threaded = false)
            ft = laplace_objective(m, popfl, cthr)
            fs = laplace_objective(m, popfl, cser)
            ft(θtrue); fs(θtrue)                    # warm up
            tt = minimum(@elapsed(ft(θtrue)) for _ in 1:3)
            ts = minimum(@elapsed(fs(θtrue)) for _ in 1:3)
            @info "Tier4: load balance" serial_s=ts threaded_s=tt speedup=ts/tt nthreads=Threads.nthreads()
            # A floor, not a tight bound. Plain Threads.@threads chunks
            # contiguously and would give ~1.0x here, since the four expensive
            # subjects are adjacent at the front of the index range.
            @test tt < 0.7 * ts
            @test ft(θtrue) === fs(θtrue)           # and still bitwise equal
        end
    end

    @testset "warm-start neutrality" begin
        cold = LaplaceConfig(inner_tol = 1e-10, inner_maxiter = 100)
        warm = LaplaceConfig(inner_tol = 1e-10, inner_maxiter = 100, warm_start = true)
        fcold = laplace_objective(m, pop, cold)
        fwarm = laplace_objective(m, pop, warm)
        Lref = fcold(θ)
        fwarm(θ .+ 0.3)                              # poison the cache with another θ
        Lw = fwarm(θ)
        @info "Tier4: warm-start" cold=Lref warm=Lw absdiff=abs(Lw - Lref)
        # The mode is recomputed to inner_tol either way, so the objective differs
        # only at second order in the mode error: ~½ δηᵀ H δη with ‖δη‖ ≲ tol/λmin.
        @test isapprox(Lw, Lref; atol = 1e-8)
        # Cold evaluation remains history-independent, i.e. bitwise repeatable.
        @test fcold(θ) === Lref
    end

    @testset "independent objective check on the nonlinear model" begin
        # Different optimizer for the mode, finite differences for the Hessian,
        # Laplace expression assembled by hand with LinearAlgebra's logdet.
        # Shares no code path with the framework beyond the model itself.
        s = pop[3]
        f = e -> negative_log_joint_density(m, e, θ, s)
        gf! = (G, e) -> (G .= ForwardDiff.gradient(f, e))
        hf! = (H, e) -> (H .= ForwardDiff.hessian(f, e))
        # Optim's Newton: a different implementation with a different (Hager-Zhang)
        # line search from the framework's damped Newton with Armijo backtracking.
        opt = Optim.optimize(f, gf!, hf!, zeros(q), Optim.Newton(),
                             Optim.Options(g_abstol = 1e-12))
        ηopt = Optim.minimizer(opt)
        @test norm(ForwardDiff.gradient(f, ηopt)) < 1e-9

        r = subject_objective(m, θ, s, cfg)
        @test isapprox(ηopt, r.eta; atol = 1e-7)

        h = 1e-4
        Hfd = t4_fd_hessian(f, ηopt, h)
        Lmanual = f(ηopt) + logdet(Symmetric(Hfd + cfg.reg * I)) / 2 - (q / 2) * log(2π)
        @info "Tier4: independent objective" framework=r.L manual=Lmanual absdiff=abs(r.L - Lmanual)
        # Error budget: an FD Hessian at h=1e-4 carries relative error ~1e-7
        # (truncation h²·f⁗/12 against roundoff eps·|f|/h²), and logdet inherits
        # it through tr(H⁻¹δH), so ~1e-6 absolute on L is the honest bound.
        # (Measured 8.8e-9 -- the bound is the method's capability, not the
        # observation, so it is not tightened to the lucky number.)
        @test isapprox(r.L, Lmanual; atol = 1e-6)
    end

    @testset "diagnostics are recorded honestly" begin
        # A deliberately starved inner solve must report non-convergence rather
        # than silently succeeding, and must still return a finite value (that is
        # what AD-through differentiates, so the two must match).
        tight = LaplaceConfig(inner_tol = 1e-14, inner_maxiter = 1)
        r = population_objective(m, θ, pop, tight)
        @test r.n_nonconverged > 0
        @test isfinite(r.L)
        @test count(!, r.converged) == r.n_nonconverged
        @test length(r.etas) == pop.n_subjects
        @test all(length(e) == q for e in r.etas)
    end

    @testset "empirical_bayes returns the per-subject modes" begin
        eb = empirical_bayes(m, pop, θ, cfg)
        r = population_objective(m, θ, pop, cfg)
        @test length(eb) == pop.n_subjects
        for i in 1:pop.n_subjects
            @test eb[i] ≈ r.etas[i]
        end
    end
end
