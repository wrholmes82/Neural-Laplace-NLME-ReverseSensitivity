# E14d — compilation cost on the PENALISED fit path.
#
#   julia --project=Local_Env -t 5 --heap-size-hint=20G warfarin_turnover/examples/cost/e14d_compile_penalised.jl
#
# E14 measured compilation at γ = 0, where the fit goes through `fit_warfarin`. The penalised fit
# does not: it builds its own objective and gradient closures and hands them to `fit_nlme` with
# `capability_optimizer`, which is a different call path and could compile differently. The
# runtime table reports γ = 1 in its other two rows, so the claim that compilation is unchanged
# should be measured on the path the table describes rather than argued from the fact that the
# penalty adds only `2γφ` to the gradient.
#
# Same design as E14: in a cold process, a cold fit and an identical warm one, then a cold and a
# warm observed information. Compilation is the cold-minus-warm difference, which is clean here
# because the network width is a runtime field, so both fits share compiled code.

const T_START = time()
t_include = @elapsed include(joinpath(@__DIR__, "..", "..", "src", "warfarin.jl"))

BLAS.set_num_threads(1)
t_backends = @elapsed load_backends!(verbose = false)
banner("E14d — compilation on the penalised fit path")

const RESULTS = results_dir(@__DIR__, "results")
const WIDTH = SCIENTIFIC_WIDTH
const N_SUBJECTS = 100
const MAXITERS = 400
const LAMBDA = 1.0

cfg = default_config(reg = 1e-8, inner_tol = 1e-8, inner_maxiter = 200, threaded = true)
rcfg = reverse_config()
m_par = WarfarinTurnover(EmaxEffect())
m_nn = WarfarinTurnover(MLPEffect(WIDTH))
const IXE = collect(theta_index(m_nn).effect)
tp = truth_theta()

sim = simulate_cohort(N_SUBJECTS; seed = SEED_COHORT + 1, theta = tp.theta, model = m_par)
pop = PopulationData([make_subject(m_nn, s.id, s.input, s.covariates.pk_times,
                                   s.observations[1:s.covariates.n_pk], s.covariates.pd_times,
                                   s.observations[s.covariates.n_pk+1:end]) for s in sim.pop.subjects])

@printf("\n%d subjects, W = %d, p = %d, γ = %g, %d julia threads, %d BLAS threads\n",
        N_SUBJECTS, WIDTH, m_nn.p, LAMBDA, Threads.nthreads(), BLAS.get_num_threads())
@printf("include %.1f s, backends %.1f s\n", t_include, t_backends); flush(stdout)

"""E11's penalised fit, reproduced exactly so this times the path the table describes."""
function fit_penalised()
    b = theta_bounds(m_nn)
    θ0 = theta_start(m_nn)
    starts = [collect(Float64, e) for e in population_objective(m_nn, θ0, pop, cfg).etas]
    obj_inner = foce_objective(m_nn, pop, cfg, CC_FOCE; fixed_starts = starts)
    grad_inner = curvature_gradient(m_nn, pop, cfg, rcfg, CC_FOCE; fixed_starts = starts)
    objective = θ -> obj_inner(θ) + LAMBDA * sum(abs2, view(θ, IXE))
    grad = function (G, θ)
        grad_inner(G, θ)
        @views G[IXE] .+= 2LAMBDA .* θ[IXE]
        return G
    end
    r = fit_nlme(m_nn, pop, θ0; lower = b.lower, upper = b.upper, cfg = cfg,
                 objective = objective, grad = grad, maxiters = MAXITERS,
                 g_tol = 1e-3, f_tol = 0.001, optimizer = capability_optimizer(m_nn.p))
    return (theta = r.theta, ofv = 2 * obj_inner(r.theta), iters = r.iterations)
end

println("\nfit 1 (cold: absorbs compilation) ..."); flush(stdout)
t_fit1 = @elapsed f1 = fit_penalised()
@printf("  %.1f s, %d iterations, OFV %.1f\n", t_fit1, f1.iters, f1.ofv); flush(stdout)

println("fit 2 (warm: steady-state estimation) ..."); flush(stdout)
t_fit2 = @elapsed f2 = fit_penalised()
@printf("  %.1f s, %d iterations, OFV %.1f\n", t_fit2, f2.iters, f2.ofv); flush(stdout)

# The information is formed from the UNPENALISED gradient, as everywhere in this work.
gcl = curvature_gradient(m_nn, pop, cfg, rcfg, CC_FOCE)
println("information 1 (cold) ..."); flush(stdout)
t_info1 = @elapsed observed_information(gcl, f2.theta; step = 1e-5, central = true)
@printf("  %.1f s\n", t_info1); flush(stdout)
println("information 2 (warm) ..."); flush(stdout)
t_info2 = @elapsed observed_information(gcl, f2.theta; step = 1e-5, central = true)
@printf("  %.1f s\n", t_info2); flush(stdout)

t_compile = max(0.0, t_fit1 - t_fit2) + max(0.0, t_info1 - t_info2)

rows = Any[csvrow("quantity", "seconds", "note")]
push!(rows, csvrow("lambda", LAMBDA, "ridge strength on the network block"))
push!(rows, csvrow("include_package", t_include, "parse and load the package and dependencies"))
push!(rows, csvrow("load_backends", t_backends, "AD backends and rules"))
push!(rows, csvrow("fit_cold", t_fit1, "first penalised fit; includes JIT"))
push!(rows, csvrow("fit_warm", t_fit2, "steady-state penalised estimation"))
push!(rows, csvrow("information_cold", t_info1, "first observed information; includes JIT"))
push!(rows, csvrow("information_warm", t_info2, "steady-state observed information"))
push!(rows, csvrow("compilation", t_compile, "(fit_cold - fit_warm) + (info_cold - info_warm)"))
push!(rows, csvrow("session_total", time() - T_START, "wall clock for this script"))
push!(rows, csvrow("julia_threads", Threads.nthreads(), ""))
push!(rows, csvrow("blas_threads", BLAS.get_num_threads(), ""))
push!(rows, csvrow("n_subjects", N_SUBJECTS, ""))
push!(rows, csvrow("p", m_nn.p, ""))
writedlm(joinpath(RESULTS, "e14d_compile_penalised.csv"), vcat(rows...), ',')

println("\n", "="^78)
@printf("%-24s %10.1f s\n", "compilation (γ = 1)", t_compile)
@printf("%-24s %10.1f s\n", "estimation (warm)", t_fit2)
@printf("%-24s %10.1f s\n", "information (warm)", t_info2)
println("""

Compare the compilation figure with E14's 77.9 s on the unpenalised path. A materially different
number would mean the penalised path compiles code the unpenalised one does not, and the table's
compilation row would need to say which arm it describes.""")
println("=" ^ 78)
