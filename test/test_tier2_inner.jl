# Tier 2 -- inner solver (src/inner.jl: newton_solve).
#
# Toy models used only in this file. Each ignores `subject` entirely (passed as
# `nothing` at call sites) and most ignore `θ` too, except Tier2ThetaQuadratic,
# whose whole point is a θ-dependent analytic mode for the ForwardDiff-through-
# the-solver tests.

import NLMEReverse: negative_log_joint_density

"Exactly quadratic h(η) = ½η'Aη + b'η, independent of θ and the subject.
Closed-form mode: η̂ = -A\\b."
struct Tier2Quadratic <: NLMEModel
    q::Int
    A::Matrix{Float64}
    b::Vector{Float64}
end

negative_log_joint_density(m::Tier2Quadratic, η, θ, subj) =
    0.5 * dot(η, m.A * η) + dot(m.b, η)

"2D Rosenbrock, minimum at (1,1) with h=0 there. A hostile-start convergence test
for the damped-Newton-with-Armijo-backtracking inner solver."
struct Tier2Rosenbrock <: NLMEModel
    q::Int
end
Tier2Rosenbrock() = Tier2Rosenbrock(2)

negative_log_joint_density(m::Tier2Rosenbrock, η, θ, subj) =
    (1 - η[1])^2 + 100 * (η[2] - η[1]^2)^2

"1D quartic-plus-quadratic, h(η) = (η-2)⁴ + (η-2)². The quadratic term keeps the
Hessian strictly positive at the (unique, real) minimum η=2, unlike a bare
quartic whose Hessian vanishes there."
struct Tier2Quartic <: NLMEModel
    q::Int
end
Tier2Quartic() = Tier2Quartic(1)

negative_log_joint_density(m::Tier2Quartic, η, θ, subj) =
    (η[1] - 2)^4 + (η[1] - 2)^2

"Quadratic in η with a θ-dependent linear term: h(η,θ) = ½η'Aη + θ'η, so the
mode η̂(θ) = -A\\θ is a known closed form of θ -- the case needed to check that
ForwardDiff differentiates correctly *through* newton_solve."
struct Tier2ThetaQuadratic <: NLMEModel
    q::Int
    A::Matrix{Float64}
end

negative_log_joint_density(m::Tier2ThetaQuadratic, η, θ, subj) =
    0.5 * dot(η, m.A * η) + dot(θ, η)

@testset "Tier 2 -- inner solver (newton_solve)" begin

    @testset "quadratic: mode matches the closed form -A\\b" begin
        A = [3.0 0.5; 0.5 2.0]
        bvec = [1.0, -2.0]
        model = Tier2Quadratic(2, A, bvec)
        cfg = LaplaceConfig(inner_tol = 1e-12, inner_maxiter = 20)
        sol = newton_solve(model, Float64[], nothing, zeros(2), cfg)
        @test sol.converged
        exact = -(A \ bvec)
        for i in eachindex(exact)
            @test abs(sol.η[i] - exact[i]) < 1e-10
        end
    end

    @testset "Rosenbrock: converges from a hostile start" begin
        model = Tier2Rosenbrock()
        cfg = LaplaceConfig(inner_tol = 1e-8, inner_maxiter = 200)
        sol = newton_solve(model, Float64[], nothing, [-1.2, 1.0], cfg)
        @test sol.converged
        @test abs(sol.η[1] - 1.0) < 1e-5
        @test abs(sol.η[2] - 1.0) < 1e-5
    end

    @testset "quartic: converges from a bad starting guess" begin
        model = Tier2Quartic()
        cfg = LaplaceConfig(inner_tol = 1e-10, inner_maxiter = 100)
        sol = newton_solve(model, Float64[], nothing, [-10.0], cfg)
        @test sol.converged
        @test abs(sol.η[1] - 2.0) < 1e-6
    end

    @testset "ForwardDiff.gradient through newton_solve (1 nesting level)" begin
        A = [2.5 0.3; 0.3 1.8]
        model = Tier2ThetaQuadratic(2, A)
        cfg = LaplaceConfig(inner_tol = 1e-12, inner_maxiter = 10)
        θtest = [0.7, -0.4]

        g_closed(θ) = sum((-(A \ θ)) .^ 2)
        function g_through(θ)
            sol = newton_solve(model, θ, nothing, zeros(2), cfg)
            return sum(sol.η .^ 2)
        end

        exact_grad = ForwardDiff.gradient(g_closed, θtest)
        through_grad = ForwardDiff.gradient(g_through, θtest)
        for i in eachindex(exact_grad)
            @test abs(through_grad[i] - exact_grad[i]) < 1e-8
        end
    end

    @testset "ForwardDiff.hessian through newton_solve (2 nesting levels)" begin
        A = [2.5 0.3; 0.3 1.8]
        model = Tier2ThetaQuadratic(2, A)
        cfg = LaplaceConfig(inner_tol = 1e-12, inner_maxiter = 10)
        θtest = [0.7, -0.4]

        g_closed(θ) = sum((-(A \ θ)) .^ 2)
        function g_through(θ)
            sol = newton_solve(model, θ, nothing, zeros(2), cfg)
            return sum(sol.η .^ 2)
        end

        exact_hess = ForwardDiff.hessian(g_closed, θtest)
        through_hess = ForwardDiff.hessian(g_through, θtest)
        for idx in eachindex(exact_hess)
            @test abs(through_hess[idx] - exact_hess[idx]) < 1e-8
        end
    end

    @testset "non-convergence at a low inner_maxiter is recorded, not swallowed" begin
        model = Tier2Rosenbrock()
        cfg = LaplaceConfig(inner_tol = 1e-8, inner_maxiter = 2)
        sol = newton_solve(model, Float64[], nothing, [-1.2, 1.0], cfg)
        @test !sol.converged
        @test isfinite(sol.value)
    end

    @testset "inner_deriv = :fd converges on the quadratic" begin
        # FINDING (not fixed here -- inner.jl is not owned by this file):
        # inner_hessian's :fd branch (src/inner.jl, the
        # diagonal-entry line) computes `(f(ep) - 2f0 + f(em)) / s[j]^2`, where
        # `f0` is a local variable holding f(η) at the center point. But Julia
        # parses the token `2f0` as the *Float32 literal* 2.0f0 (the same rule
        # that makes `1f0`/`1f10` a Float32/scientific literal), not as
        # `2 * f0` -- confirmed interactively: with `f0 = 5.0`, `2f0` evaluates
        # to `2.0f0`, not `10.0`. So every diagonal FD-Hessian entry silently
        # substitutes the constant 2.0 for 2·f(η), corrupting H whenever
        # f(η) != 1.0. On this quadratic that makes H wildly wrong (diagonal
        # entries around -2e10 instead of ~3), Cholesky in solve_pd correctly
        # rejects it as non-PD, and newton_solve falls back to its steepest-
        # descent branch every iteration. inner_gradient's :fd branch has no
        # such collision and is exact for this quadratic (its third derivative
        # is zero), so the solver still converges here -- but as slow damped
        # gradient descent, not Newton, and needs far more iterations. This
        # means inner_deriv=:fd does NOT currently reproduce a genuine FD-Newton
        # baseline; it happens to converge on well-conditioned problems via a
        # fallback path that would degrade badly on anything less benign.
        #
        # Tolerance/budget below are set to what the method actually delivers,
        # measured directly (deterministic, no RNG involved): with
        # inner_maxiter=100, inner_tol=1e-6, convergence takes 33 steepest-
        # descent iterations and lands within 1.9e-7 of the closed form. 1e-5
        # keeps two orders of headroom without masking the slow-convergence
        # symptom (a tolerance loose enough to pass at maxiter=30 would hide
        # exactly the behaviour being reported).
        A = [3.0 0.5; 0.5 2.0]
        bvec = [1.0, -2.0]
        model = Tier2Quadratic(2, A, bvec)
        cfg = LaplaceConfig(inner_deriv = :fd, inner_tol = 1e-6, inner_maxiter = 100,
                            inner_fd_step = 1e-5)
        sol = newton_solve(model, Float64[], nothing, zeros(2), cfg)
        @test sol.converged
        exact = -(A \ bvec)
        for i in eachindex(exact)
            @test abs(sol.η[i] - exact[i]) < 1e-5
        end
    end
end
