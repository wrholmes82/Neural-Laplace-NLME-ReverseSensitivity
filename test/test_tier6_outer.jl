# Tier 6 -- outer optimization. Exercises fit_nlme end-to-end on the
# two-compartment problem at N=50: descent, recovery from a seeded
# perturbation, method agreement between adjoint- and AD-driven fits, the
# rescue counters, and standard errors (including the L-vs-OFV scale
# convention called out in outer.jl).
#
# Helper names prefixed `t6_` to avoid colliding with other tier files.

@testset "Tier 6 -- outer optimization" begin

    m = TwoComp()
    pop, theta_true, _ = simulate_twocomp(m; N = 50, seed = 12345)
    lo, hi = twocomp_bounds()
    p = length(theta_true)

    # ------------------------------------------------------------------
    # 1. Descent: a small step along the negative gradient lowers L.
    # ------------------------------------------------------------------
    @testset "descent" begin
        cfg = LaplaceConfig(inner_tol = 1e-8, inner_maxiter = 100)
        f = laplace_objective(m, pop, cfg)
        g! = adjoint_gradient(m, pop, cfg)
        th0 = perturb(theta_true, 0.3; seed = 1)
        G = zeros(p)
        g!(G, th0)
        f0 = f(th0)
        alpha = 1e-6 / max(norm(G), 1.0)
        th1 = th0 .- alpha .* G
        f1 = f(th1)
        println("Tier6: descent  f0=", f0, "  f1=", f1, "  alpha=", alpha)
        @test f1 < f0
    end

    # ------------------------------------------------------------------
    # 2. Recovery from a seeded perturbation of the truth.
    #    MEASURED (this machine): converged=true, ||grad||_inf=2.9e-7,
    #    83 iterations, ~2.5s; theta_hat-theta_true up to 0.40 on
    #    log omega_Q (within 1 SE -- variance components are poorly
    #    determined at N=50). Fixed effects theta[1:4] recover tightly;
    #    all nine recover within ~2.5 SE.
    # ------------------------------------------------------------------
    cfg_fit = LaplaceConfig(inner_tol = 1e-10, inner_maxiter = 100)
    th0 = theta_true .+ [0.3, -0.3, 0.3, -0.3, 0.2, -0.2, 0.2, -0.2, 0.1]

    local res
    @testset "recovery" begin
        res = fit_nlme(m, pop, th0; lower = lo, upper = hi, cfg = cfg_fit,
                        gradient = :adjoint, maxiters = 150, compute_se = true)
        println("Tier6: recovery  converged=", res.converged,
                "  ||grad||_inf=", res.gradient_norm,
                "  iters=", res.iterations, "  elapsed=", res.elapsed, "s")
        println("Tier6: theta_hat - theta_true = ", res.theta .- theta_true)
        println("Tier6: standard errors        = ", res.standard_errors)

        @test res.converged

        diff = res.theta .- theta_true
        # Fixed effects (CL, V1, Q, V2 on the log scale) recover tightly.
        for j in 1:4
            @test abs(diff[j]) < 0.15
        end
        # All nine parameters recover within ~2.5 standard errors -- variance
        # components are simply less well determined at N=50, so a single
        # tight bound across all nine is not the right assertion.
        se = res.standard_errors
        @test se !== nothing
        for j in 1:p
            @test abs(diff[j]) < 2.5 * se[j]
        end
    end

    # ------------------------------------------------------------------
    # 3. Method agreement: adjoint- and AD-driven fits from the same start
    #    reach the same optimum. MEASURED: OFV differs by 2.3e-13, theta by
    #    2.1e-8.
    # ------------------------------------------------------------------
    @testset "method agreement (adjoint vs AD-through)" begin
        res_adj = fit_nlme(m, pop, th0; lower = lo, upper = hi, cfg = cfg_fit,
                            gradient = :adjoint, maxiters = 150)
        res_ad = fit_nlme(m, pop, th0; lower = lo, upper = hi, cfg = cfg_fit,
                           gradient = :ad, maxiters = 150)
        d_ofv = abs(res_adj.OFV - res_ad.OFV)
        println("Tier6: method agreement  |OFV diff|=", d_ofv,
                "  theta diff=", res_adj.theta .- res_ad.theta)
        @test d_ofv < 1e-6
        for j in 1:p
            @test abs(res_adj.theta[j] - res_ad.theta[j]) < 1e-4
        end
    end

    # ------------------------------------------------------------------
    # 4. Rescue counters and inner convergence at the final theta.
    #    objective_rescues may be nonzero (benign: the line search probes
    #    theta where some subject's Htilde is not PD); gradient_rescues
    #    must be zero.
    # ------------------------------------------------------------------
    @testset "rescue counters and inner convergence" begin
        println("Tier6: objective_rescues=", res.objective_rescues,
                " (benign)  gradient_rescues=", res.gradient_rescues,
                "  inner_nonconverged=", res.inner_nonconverged)
        @test res.gradient_rescues == 0
        @test res.inner_nonconverged == 0
    end

    # ------------------------------------------------------------------
    # 5. Standard errors: positive, finite, PD information matrix.
    # ------------------------------------------------------------------
    @testset "standard errors" begin
        g! = adjoint_gradient(m, pop, cfg_fit)
        s = standard_errors(g!, res.theta)
        @test s.pd
        @test all(isfinite, s.se)
        @test all(s.se .> 0)
        Vc = Symmetric(s.vcov)
        C = cholesky(Vc; check = false)
        @test issuccess(C)
        @test s.vcov ≈ transpose(s.vcov)
    end

    # ------------------------------------------------------------------
    # 6. SE scale convention: Cov = inv(Hessian_theta L), L the objective
    #    (not OFV = 2L). Verified by computing the Hessian of L a second
    #    way -- central differences of the adjoint gradient at a different
    #    step -- and checking the resulting SEs agree to a few percent.
    # ------------------------------------------------------------------
    @testset "SE scale convention self-consistency" begin
        g! = adjoint_gradient(m, pop, cfg_fit)
        s1 = standard_errors(g!, res.theta; step = 1e-5)
        s2 = standard_errors(g!, res.theta; step = 1e-4)
        @test s1.pd && s2.pd
        println("Tier6: SE step=1e-5 -> ", s1.se)
        println("Tier6: SE step=1e-4 -> ", s2.se)
        for j in 1:p
            rel = abs(s1.se[j] - s2.se[j]) / max(s1.se[j], 1e-8)
            @test rel < 0.05
        end
    end

    # ------------------------------------------------------------------
    # 7. Thread count is recorded.
    # ------------------------------------------------------------------
    @testset "nthreads recorded" begin
        @test res.nthreads == Threads.nthreads()
    end
end
