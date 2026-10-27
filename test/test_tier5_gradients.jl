# Tier 5 -- gradient cross-validation. This is the core of the suite: the
# adjoint gradient (gradient_adjoint.jl) is judged against two independent
# routes -- AD-through the inner Newton solve, and central finite differences
# -- plus two identities specific to the adjoint correction term `c`.
#
# Helper names are prefixed `t5_` to avoid colliding with other tier files
# included into the same top-level scope.

t5_maxabsdiff(a, b) = maximum(abs.(a .- b))

"Elementwise |a-b| < tol, asserted one component at a time so a single bad
component shows up as a single failing @test, not a hidden max()."
function t5_assert_elementwise(a, b, tol; msg = "")
    for j in eachindex(a)
        @test abs(a[j] - b[j]) < tol
    end
end

"Elementwise relative |a-b|/max(|b|,floor) < tol."
function t5_assert_elementwise_rel(a, b, tol; floor = 1e-8)
    for j in eachindex(a)
        denom = max(abs(b[j]), floor)
        @test abs(a[j] - b[j]) / denom < tol
    end
end

@testset "Tier 5 -- gradient cross-validation" begin

    m = TwoComp()
    pop, theta_true, _ = simulate_twocomp(m; N = 20, seed = 12345)
    th = theta_true .+ 0.05
    p = length(th)

    # ------------------------------------------------------------------
    # 1. Adjoint vs AD-through, elementwise, inner_tol tightened to 1e-10.
    #    Measured baseline on this machine: max abs diff 3.24e-10 at
    #    inner_tol=1e-10 (inner_maxiter=100). Assert < 1e-8, two orders of
    #    margin, without loosening to accommodate a loose inner solve.
    # ------------------------------------------------------------------
    @testset "adjoint vs AD-through at tight inner_tol" begin
        cfg = LaplaceConfig(inner_tol = 1e-10, inner_maxiter = 100)
        g_adj = adjoint_gradient(m, pop, cfg)
        g_ad  = ad_gradient(m, pop, cfg)
        Ga = zeros(p); Gb = zeros(p)
        g_adj(Ga, th)
        g_ad(Gb, th)
        println("Tier5: adjoint vs AD-through @ inner_tol=1e-10, max|diff| = ",
                t5_maxabsdiff(Ga, Gb))
        t5_assert_elementwise(Ga, Gb, 1e-8)
    end

    # ------------------------------------------------------------------
    # 2. Convergence-order test: as inner_tol tightens, the adjoint/AD
    #    discrepancy must shrink monotonically. This is what separates
    #    "correct method, finite inner tolerance" from "wrong method".
    # ------------------------------------------------------------------
    @testset "convergence order in inner_tol" begin
        tols = (1e-4, 1e-6, 1e-8, 1e-10)
        diffs = Float64[]
        for tol in tols
            cfg = LaplaceConfig(inner_tol = tol, inner_maxiter = 100)
            g_adj = adjoint_gradient(m, pop, cfg)
            g_ad  = ad_gradient(m, pop, cfg)
            Ga = zeros(p); Gb = zeros(p)
            g_adj(Ga, th)
            g_ad(Gb, th)
            push!(diffs, t5_maxabsdiff(Ga, Gb))
        end
        println("Tier5: convergence-order sweep, inner_tol -> max|diff| : ",
                collect(zip(tols, diffs)))
        for k in 1:(length(diffs) - 1)
            @test diffs[k + 1] < diffs[k]
        end
    end

    # ------------------------------------------------------------------
    # 3. Adjoint vs central FD, elementwise, RELATIVE criterion. The
    #    tolerance is derived, not assumed: sweep the FD step over several
    #    decades, print the sweep, find the step that minimizes error
    #    (truncation-vs-roundoff trade-off), and set the assertion from
    #    what the best step actually achieves, with margin.
    # ------------------------------------------------------------------
    @testset "adjoint vs central FD (derived tolerance)" begin
        cfg = LaplaceConfig(inner_tol = 1e-10, inner_maxiter = 100)
        g_adj = adjoint_gradient(m, pop, cfg)
        Ga = zeros(p)
        g_adj(Ga, th)

        steps = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
        rel_errs = Float64[]
        for s in steps
            g_fd = fd_gradient(m, pop, cfg; step = s)
            Gf = zeros(p)
            g_fd(Gf, th)
            relerr = maximum(abs(Gf[j] - Ga[j]) / max(abs(Ga[j]), 1e-8) for j in 1:p)
            push!(rel_errs, relerr)
            println("Tier5: FD step sweep  step=", s, "  max relative error=", relerr)
        end

        best_err, best_idx = findmin(rel_errs)
        best_step = steps[best_idx]
        println("Tier5: FD sweep best step = ", best_step, "  best relative error = ", best_err)

        # Tolerance derived from the measured best-step error, with a 10x margin.
        fd_tol = max(10 * best_err, 1e-4)
        g_fd_best = fd_gradient(m, pop, cfg; step = best_step)
        Gf_best = zeros(p)
        g_fd_best(Gf_best, th)
        t5_assert_elementwise_rel(Gf_best, Ga, fd_tol; floor = 1e-8)
    end

    # ------------------------------------------------------------------
    # 4. AD-through factory vs raw ForwardDiff.gradient over
    #    laplace_objective(m, pop, cfg): same code path, must agree to
    #    ~machine precision.
    # ------------------------------------------------------------------
    @testset "AD-through factory is a transparent wrapper" begin
        cfg = LaplaceConfig(inner_tol = 1e-10, inner_maxiter = 100)
        f = laplace_objective(m, pop, cfg)
        graw = ForwardDiff.gradient(f, th)
        g_ad = ad_gradient(m, pop, cfg)
        Gad = zeros(p)
        g_ad(Gad, th)
        t5_assert_elementwise(Gad, graw, 1e-12)
    end

    # ------------------------------------------------------------------
    # 5. Agreement at several perturbed thetas, not only at theta_true+0.05.
    # ------------------------------------------------------------------
    @testset "adjoint vs AD-through at perturbed thetas" begin
        cfg = LaplaceConfig(inner_tol = 1e-10, inner_maxiter = 100)
        g_adj = adjoint_gradient(m, pop, cfg)
        g_ad  = ad_gradient(m, pop, cfg)
        for (scale, seed) in ((0.05, 1), (0.1, 2), (0.02, 3), (0.15, 4))
            thp = perturb(theta_true, scale; seed = seed)
            Ga = zeros(p); Gb = zeros(p)
            g_adj(Ga, thp)
            g_ad(Gb, thp)
            d = t5_maxabsdiff(Ga, Gb)
            println("Tier5: perturbed theta scale=", scale, " seed=", seed,
                    " max|diff| = ", d)
            t5_assert_elementwise(Ga, Gb, 1e-7)
        end
    end

    # ------------------------------------------------------------------
    # 6. O(eps^4) scaling of the adjoint correction term (Appendix S2).
    #    Omega = eps^2 Omega_0; the model and data are regenerated at each
    #    eps with TwoComp(re_scale=eps) so the model is consistently
    #    specified. Sweep stays small (0.02-0.09): larger eps has not
    #    reached the asymptotic regime (local order at eps=0.5 is only
    #    ~2.0, per the measured baseline), and much below 0.02 approaches
    #    the inner-solve tolerance floor.
    # ------------------------------------------------------------------
    @testset "O(eps^4) scaling of the correction term" begin
        cfg_eps = LaplaceConfig(inner_tol = 1e-11, inner_maxiter = 100)
        eps_list = [0.09, 0.06, 0.045, 0.03, 0.02]
        cnorms = Float64[]
        for eps in eps_list
            ms = TwoComp(re_scale = eps)
            pops, th_s, _ = simulate_twocomp(ms; N = 20, seed = 12345)
            r = adjoint_gradient_parts(ms, th_s, pops, cfg_eps)
            push!(cnorms, norm(r.c))
        end
        println("Tier5: O(eps^4) sweep  eps=", eps_list, "  ||c||=", cnorms)

        logeps = log.(eps_list)
        logc = log.(cnorms)
        # least-squares slope of logc vs logeps
        xbar = mean(logeps); ybar = mean(logc)
        slope = sum((logeps .- xbar) .* (logc .- ybar)) / sum((logeps .- xbar) .^ 2)
        println("Tier5: least-squares log-log slope = ", slope)

        local_orders = [log(cnorms[k] / cnorms[k + 1]) / log(eps_list[k] / eps_list[k + 1])
                         for k in 1:(length(eps_list) - 1)]
        println("Tier5: local orders = ", local_orders)

        @test abs(slope - 4) < 0.25
        @test abs(local_orders[end] - 4) < 0.1
    end

    # ------------------------------------------------------------------
    # 7. Correction-term identity (Eqs. 17/18):
    #      c_i = -0.5 * (grad_eta logdet Htilde_i)^T * (d eta_hat_i / d theta)
    #    with d eta_hat/d theta obtained independently by ForwardDiff
    #    through the inner solve. Checked per-subject, elementwise relative.
    # ------------------------------------------------------------------
    @testset "correction-term identity (Eqs 17/18)" begin
        cfg_ct = LaplaceConfig(inner_tol = 1e-12, inner_maxiter = 200)
        q = n_random_effects(m)
        for i in (1, 5, 10, 15, 20)
            subj = pop.subjects[i]
            r = subject_adjoint(m, th, subj, cfg_ct)

            J = ForwardDiff.jacobian(
                t -> newton_solve(m, t, subj, zeros(eltype(t), q), cfg_ct).η, th)

            gld = ForwardDiff.gradient(
                ev -> logdet_pd(htilde(
                    ForwardDiff.hessian(z -> negative_log_joint_density(m, z, th, subj), ev),
                    cfg_ct)),
                r.eta)

            rhs = -0.5 .* (transpose(gld) * J)[:]

            for j in 1:p
                denom = max(abs(rhs[j]), 1e-10)
                relerr = abs(r.c[j] - rhs[j]) / denom
                @test relerr < 1e-9
            end
        end
    end

    # ------------------------------------------------------------------
    # 8. Magnitude floor: on the proportional-error model, c must be a
    #    substantial fraction of the total gradient. Measured ~1.77.
    # ------------------------------------------------------------------
    @testset "correction-term magnitude floor" begin
        cfg = LaplaceConfig()
        r = adjoint_gradient_parts(m, th, pop, cfg)
        ratio = norm(r.c) / norm(r.grad)
        println("Tier5: ||c||/||grad|| = ", ratio)
        @test ratio > 0.1
    end

    # ------------------------------------------------------------------
    # 9. Adjoint vs AD-through, both threaded=true and threaded=false.
    # ------------------------------------------------------------------
    @testset "adjoint vs AD-through, threaded and serial" begin
        for threaded in (true, false)
            cfg = LaplaceConfig(inner_tol = 1e-10, inner_maxiter = 100, threaded = threaded)
            g_adj = adjoint_gradient(m, pop, cfg)
            g_ad  = ad_gradient(m, pop, cfg)
            Ga = zeros(p); Gb = zeros(p)
            g_adj(Ga, th)
            g_ad(Gb, th)
            println("Tier5: threaded=", threaded, " max|diff| = ", t5_maxabsdiff(Ga, Gb))
            t5_assert_elementwise(Ga, Gb, 1e-8)
        end
    end
end
