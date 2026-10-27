# Tier 1 -- numerical primitives (src/numerics.jl).

function _tier1_elemclose(a, b, tol)
    @assert length(a) == length(b) "length mismatch: $(length(a)) vs $(length(b))"
    for i in eachindex(a)
        @test abs(a[i] - b[i]) < tol
    end
end

@testset "Tier 1 -- numerical primitives" begin

    @testset "logdet_pd agrees with logdet on a well-conditioned SPD matrix" begin
        A = [4.0 1.0; 1.0 3.0]
        @test issymmetric(A)
        @test isposdef(A)
        # Both go through a Cholesky-adjacent path; agreement should be near
        # machine precision, not merely "close".
        @test abs(logdet_pd(A) - logdet(A)) < 1e-12
    end

    @testset "logdet_pd flags indefinite matrices (odd AND even negative eigenvalues)" begin
        # Odd number of negative eigenvalues: det < 0, so even a naive
        # sign-of-determinant check would catch this one.
        A_odd = [1.0 0.0; 0.0 -2.0]
        @test det(A_odd) < 0
        @test isnan(logdet_pd(A_odd))

        # Even number of negative eigenvalues: det > 0. This is the case a
        # sign-of-determinant (or logabsdet) check would wrongly accept --
        # exactly the hazard documented in numerics.jl. Cholesky must still
        # reject it because it is not positive definite.
        A_even = Matrix(Diagonal([-1.0, -2.0]))
        @test det(A_even) > 0
        @test isnan(logdet_pd(A_even))
    end

    @testset "solve_pd matches `\\` on SPD, all-NaN on non-SPD" begin
        A = [4.0 1.0; 1.0 3.0]
        b = [1.0, 2.0]
        sol = solve_pd(A, b)
        _tier1_elemclose(sol, A \ b, 1e-12)

        A_bad = [1.0 0.0; 0.0 -2.0]
        bad = solve_pd(A_bad, b)
        @test length(bad) == length(b)
        @test all(isnan, bad)
    end

    @testset "extract_value strips nested Duals and maps over arrays" begin
        inner = ForwardDiff.Dual(2.5, 1.0)
        outer = ForwardDiff.Dual(inner, inner)   # Dual{...,Dual{...,Float64}}
        @test extract_value(outer) === 2.5
        @test extract_value(3.5) === 3.5

        arr = [outer, ForwardDiff.Dual(3.5, 0.0), ForwardDiff.Dual(inner, inner)]
        @test extract_value(arr) == [2.5, 3.5, 2.5]
    end

    @testset "nan_like propagates NaN into the partials too" begin
        # Not exported by NLMEReverse (only extract_value/logdet_pd/solve_pd/
        # htilde are), so qualify it.
        d = ForwardDiff.Dual(3.0, 2.0)
        nd = NLMEReverse.nan_like(d)
        @test isnan(ForwardDiff.value(nd))
        @test all(isnan, ForwardDiff.partials(nd))

        @test isnan(NLMEReverse.nan_like(1.0))
    end

    @testset "logdet_pd / solve_pd differentiate through Dual-typed matrices" begin
        base = [2.0 0.5; 0.5 1.5]
        θd = ForwardDiff.Dual(1.0, 1.0)
        Adual = base + θd * I               # SPD for every real value of θd
        Aplain = base + 1.0 * Matrix(I, 2, 2)  # the value Adual carries

        ld_dual = logdet_pd(Adual)
        @test ld_dual isa ForwardDiff.Dual
        @test abs(ForwardDiff.value(ld_dual) - logdet_pd(Aplain)) < 1e-12

        b = [1.0, -1.0]
        sol_dual = solve_pd(Adual, b)
        @test all(x -> x isa ForwardDiff.Dual, sol_dual)
        _tier1_elemclose(extract_value(sol_dual), solve_pd(Aplain, b), 1e-12)
    end

    @testset "htilde == H + reg*I" begin
        cfg = LaplaceConfig(reg = 0.01)
        H = [1.0 0.2; 0.2 1.0]
        Ht = htilde(H, cfg)
        _tier1_elemclose(vec(Ht), vec(H + 0.01 * I), 1e-14)
    end

    @testset "regularizer's effect on logdet is O(reg)" begin
        # d/dreg logdet(H + reg I) |_{reg=0} = tr(H⁻¹) (standard matrix-calculus
        # identity), so to first order logdet_pd(H̃) - logdet_pd(H) ≈ reg*tr(H⁻¹).
        # The Taylor remainder is
        #   reg*tr(H⁻¹) - (reg²/2)*tr(H⁻²) + O(reg³),
        # so |actual - reg*tr(H⁻¹)| should be bounded by reg²*tr(H⁻²) with
        # comfortable headroom over the reg²/2 leading term.
        H = [3.0 0.4 0.1; 0.4 2.0 0.2; 0.1 0.2 1.5]
        reg = 1e-6
        cfg = LaplaceConfig(reg = reg)
        actual = logdet_pd(htilde(H, cfg)) - logdet_pd(H)
        first_order = reg * tr(inv(H))
        remainder_bound = reg^2 * tr(inv(H)^2)
        @test abs(actual - first_order) < remainder_bound
    end
end
