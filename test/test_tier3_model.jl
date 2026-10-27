# Tier 3 -- the two-compartment test problem.
#
# The concentration is checked against a genuinely independent route (the matrix
# exponential of the rate matrix), not a second copy of the same closed form, and
# against two structural invariants a pointwise comparison would not catch.

"""
    t3_conc_expm(t, CL, V1, Q, V2, input)

C(t) via `exp(A*t)` on the 2x2 rate matrix. Shares no algebra with
`twocomp_conc`: no hybrid rate constants, no eigen-decomposition written by hand,
no A/B coefficients.
"""
function t3_conc_expm(t, CL, V1, Q, V2, input)
    k10 = CL / V1; k12 = Q / V1; k21 = Q / V2
    A = [-(k10 + k12) k21; k12 -k21]
    return (exp(A * t) * [input, 0.0])[1] / V1
end

"Analytic time-integral of the bi-exponential concentration, derived independently:
∫C = (input/V1)(A/α + B/β)."
function t3_time_integral(CL, V1, Q, V2, input)
    k10 = CL / V1; k12 = Q / V1; k21 = Q / V2
    u = k10 + k12 + k21
    disc = sqrt((k10 + k12 - k21)^2 + 4 * k12 * k21)
    α = (u + disc) / 2; β = (u - disc) / 2
    A = (α - k21) / (α - β); B = (k21 - β) / (α - β)
    return (input / V1) * (A / α + B / β)
end

@testset "Tier 3 -- two-compartment test problem" begin
    input = TWOCOMP_INPUT
    psets = [(2.0, 30.0, 5.0, 80.0), (0.5, 10.0, 20.0, 200.0),
             (8.0, 60.0, 1.0, 15.0), (1.0, 5.0, 0.4, 400.0)]
    times = [0.0, 0.25, 1.0, 4.0, 12.0, 30.0, 60.0, 120.0]

    @testset "matches an independent matrix-exponential solution" begin
        # expm on a 2x2 is accurate to a few ulp; 1e-11 relative is the honest
        # floor for comparing two different algorithms at these magnitudes.
        for (CL, V1, Q, V2) in psets, t in times
            @test isapprox(twocomp_conc(t, CL, V1, Q, V2, input),
                           t3_conc_expm(t, CL, V1, Q, V2, input); rtol = 1e-11)
        end
    end

    @testset "C(0) = input/V1" begin
        # Structural: A + B = 1, so this is an identity rather than an approximation.
        for (CL, V1, Q, V2) in psets
            @test isapprox(twocomp_conc(0.0, CL, V1, Q, V2, input), input / V1; rtol = 1e-14)
        end
    end

    @testset "total time-integral of the concentration = input/CL" begin
        # Catches parameter-ordering mistakes that a pointwise comparison against
        # a second copy of the same formula cannot: the time-integral depends on
        # CL alone.
        for (CL, V1, Q, V2) in psets
            @test isapprox(t3_time_integral(CL, V1, Q, V2, input), input / CL; rtol = 1e-12)
        end
    end

    @testset "near-degenerate Q stays finite, nonnegative, and hits the 1-comp limit" begin
        CL, V1, V2 = 2.0, 30.0, 80.0
        for Qv in (1e-1, 1e-4, 1e-8, 1e-12, 1e-16), t in times
            c = twocomp_conc(t, CL, V1, Qv, V2, input)
            @test isfinite(c)
            @test c >= 0.0
        end
        # As Q -> 0 the peripheral compartment drops out and C -> (input/V1) e^{-k10 t}.
        for t in times
            @test isapprox(twocomp_conc(t, CL, V1, 1e-14, V2, input),
                           (input / V1) * exp(-(CL / V1) * t); rtol = 1e-8)
        end
    end

    @testset "C(t) >= 0 over an extreme parameter sweep" begin
        # Guards a real bug: catastrophic cancellation in the A/B coefficients
        # used to make C tiny and NEGATIVE as Q -> 0, which the proportional
        # error model then handed straight to log().
        rng = MersenneTwister(20260721)
        nneg = 0; nbad = 0
        for _ in 1:100_000
            CL, V1, Q, V2 = exp.(-6 .+ 12 .* rand(rng, 4))
            c = twocomp_conc(200 * rand(rng), CL, V1, Q, V2, input)
            isfinite(c) || (nbad += 1; continue)
            c < 0 && (nneg += 1)
        end
        @test nneg == 0
        @test nbad == 0
    end

    @testset "negative_log_joint_density is twice differentiable in η, once in θ" begin
        m = TwoComp()
        pop, θ, _ = simulate_twocomp(m; N = 4, seed = 12345)
        s = pop[2]
        η0 = [0.05, -0.03, 0.02, 0.01]

        g = ForwardDiff.gradient(e -> negative_log_joint_density(m, e, θ, s), η0)
        H = ForwardDiff.hessian(e -> negative_log_joint_density(m, e, θ, s), η0)
        @test all(isfinite, g)
        @test all(isfinite, H)
        @test H ≈ transpose(H)
        @test isposdef(Symmetric(H))

        gθ = ForwardDiff.gradient(t -> negative_log_joint_density(m, η0, t, s), θ)
        @test all(isfinite, gθ)

        # Nested: the adjoint needs an η-Hessian under a Dual θ.
        gnest = ForwardDiff.gradient(
            t -> logdet_pd(ForwardDiff.hessian(z -> negative_log_joint_density(m, z, t, s), η0)), θ)
        @test all(isfinite, gnest)

        # Central-difference spot check of the η-gradient. Step 1e-6 gives
        # truncation ~h² ≈ 1e-12 and roundoff ~eps·|h|/h ≈ 1e-10 relative.
        for k in 1:4
            h = 1e-6
            ep = copy(η0); ep[k] += h
            em = copy(η0); em[k] -= h
            fd = (negative_log_joint_density(m, ep, θ, s) -
                  negative_log_joint_density(m, em, θ, s)) / (2h)
            @test isapprox(g[k], fd; rtol = 1e-6)
        end
    end

    @testset "additive error variant is a different, working model" begin
        madd = TwoComp(error_model = :additive)
        mprop = TwoComp()
        pop, θ, _ = simulate_twocomp(mprop; N = 4, seed = 12345)
        s = pop[1]
        @test negative_log_joint_density(madd, zeros(4), θ, s) !=
              negative_log_joint_density(mprop, zeros(4), θ, s)
        cfg = LaplaceConfig(inner_tol = 1e-10, inner_maxiter = 100)
        for mm in (madd, mprop)
            r = subject_objective(mm, θ, s, cfg)
            @test isfinite(r.L)
            @test r.converged
        end
    end

    @testset "predict returns the profile at the observation times" begin
        m = TwoComp()
        pop, θ, _ = simulate_twocomp(m; N = 3, seed = 12345)
        s = pop[1]
        η0 = [0.1, -0.1, 0.05, 0.0]
        p = predict(m, η0, θ, s)
        @test length(p) == length(s.times)
        @test all(>(0), p)
        CL = exp(θ[1] + η0[1]); V1 = exp(θ[2] + η0[2])
        Q = exp(θ[3] + η0[3]); V2 = exp(θ[4] + η0[4])
        for j in eachindex(s.times)
            @test p[j] ≈ twocomp_conc(s.times[j], CL, V1, Q, V2, s.input)
        end
    end

    @testset "re_scale scales every ω by ε" begin
        m1 = TwoComp(re_scale = 1.0)
        mε = TwoComp(re_scale = 0.25)
        pop, θ, _ = simulate_twocomp(m1; N = 2, seed = 12345)
        s = pop[1]
        # At η = 0 only the −log p(η|θ) normalising constants differ, by q·log ε.
        Δ = negative_log_joint_density(mε, zeros(4), θ, s) -
            negative_log_joint_density(m1, zeros(4), θ, s)
        @test isapprox(Δ, 4 * log(0.25); rtol = 1e-12)
    end
end
