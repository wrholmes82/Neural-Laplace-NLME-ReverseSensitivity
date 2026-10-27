# Tier 0 -- analytic oracles (linear-Gaussian).
#
# Because h_i is exactly quadratic in η for the linear-Gaussian problem
# (`problems/linear_gaussian.jl`), three things hold *exactly*, not just
# approximately:
#
#   (a) population_objective(...).L == lg_marginal_nll(...)     [closed-form marginal]
#   (b) the adjoint gradient == ForwardDiff gradient of the closed form, elementwise
#   (c) b == 0, λ == 0, c == 0 identically (Proposition 1: H is η-independent)
#
# cfg uses reg=0.0 (no regularization shift, so H̃ == H exactly) and a very tight
# inner tolerance, so the inner Newton solve is converged far beyond what the
# exactness claims above need -- any residual gap is genuinely about the outer
# machinery, not about an under-converged inner solve.

function _tier0_elemclose(a, b, tol)
    @assert length(a) == length(b) "length mismatch: $(length(a)) vs $(length(b))"
    for i in eachindex(a)
        @test abs(a[i] - b[i]) < tol
    end
end

@testset "Tier 0 -- analytic oracles (linear-Gaussian)" begin
    model, pop, theta_true, eta_true = simulate_linear_gaussian(seed = 4321)
    cfg = LaplaceConfig(reg = 0.0, inner_tol = 1e-12, inner_maxiter = 100)

    # Truth plus two seeded perturbations -- implementations have historically
    # passed only at the generating point and failed elsewhere.
    thetas = [
        ("truth", theta_true),
        ("perturbed #1", perturb(theta_true, 0.2; seed = 101)),
        ("perturbed #2", perturb(theta_true, 0.2; seed = 202)),
    ]

    for (label, theta) in thetas
        @testset "$label" begin
            # (a) objective equals the closed-form log marginal.
            # MEASURED on this machine at truth: |diff| = 0.0. 1e-9 gives ~4
            # orders of headroom over anything float64 accumulation could produce
            # for a sum over N=12 subjects of O(1)-scale terms.
            po = population_objective(model, theta, pop, cfg)
            lgm = lg_marginal_nll(model, theta, pop)
            @test abs(po.L - lgm) < 1e-9

            # Ground-truth gradient: autodiff of the closed-form marginal. Shares
            # no code path with the objective, the inner solve or the adjoint
            # machinery.
            gm = lg_marginal_gradient(model, theta, pop)

            # (b) adjoint gradient equals the closed-form gradient, elementwise.
            # MEASURED at truth: max abs diff 2.5e-14.
            parts = adjoint_gradient_parts(model, theta, pop, cfg)
            _tier0_elemclose(parts.grad, gm, 1e-9)

            # (c) b, λ and c are identically zero -- not approximately zero.
            # H = ∇²_η h is, for this model, a constant matrix (Z'Z/σ² plus a
            # diagonal Ω⁻¹ term) with no η anywhere in its expression. So
            # `ForwardDiff.gradient(η -> logdet_pd(H̃(η)), η̂)` differentiates a
            # function that is a *constant* of η: every partial ForwardDiff
            # propagates is seeded from a zero derivative of H̃ w.r.t. η, so b
            # comes out as a structural 0.0 in every component, not a small
            # residual. λ = solve_pd(H, b) of an exact zero RHS is then an exact
            # zero (Cholesky forward/back-substitution of an all-zero vector
            # introduces no rounding), and c is the gradient of
            # `dot(∇_η h, λ)` with λ captured as an exact-zero constant vector,
            # i.e. the gradient of the identically-zero function, again exact.
            for bi in parts.b
                @test all(iszero, bi)
            end
            for li in parts.lambda
                @test all(iszero, li)
            end
            @test all(iszero, parts.c)

            # AD-through vs the same closed-form gradient.
            f = laplace_objective(model, pop, cfg)
            g_ad = ForwardDiff.gradient(f, theta)
            _tier0_elemclose(g_ad, gm, 1e-8)

            # Central FD vs the same closed-form gradient -- looser tolerance,
            # derived from the central-difference truncation formula
            #   D_h f(θ) = f'(θ) + (h²/6) f'''(θ) + O(h⁴)
            # fd_gradient's default step is `1e-4 * max(|θⱼ|,1)`, so h ~ 1e-4 and
            # h²/6 ~ 1.7e-9; for this smooth log-likelihood |f'''| = O(1)-O(10),
            # so the predicted error band is ~1e-9..1e-7. Measured on this
            # machine below 5e-6 at every θ tested; 1e-5 keeps headroom without
            # being loosened to paper over anything.
            g_fd = similar(theta)
            fd_gradient(model, pop, cfg)(g_fd, theta)
            _tier0_elemclose(g_fd, gm, 1e-5)
        end
    end
end
