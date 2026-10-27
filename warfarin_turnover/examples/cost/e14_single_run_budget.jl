# E14 — what one exact-Laplace M_nn(10) run on a 100-subject cohort costs.
#
#   julia --project=Local_Env -t 5 --heap-size-hint=20G warfarin_turnover/examples/cost/e14_single_run_budget.jl
#
# Estimation and information timings are recorded by E11 across five penalised fits. Compilation
# is not: every experiment in this project
# deliberately warms up and discards a fit before timing anything, precisely so that no reported
# number contains compilation, which means no reported number measures it either. The only record
# of it is a comment in `e4_coverage.jl` putting it at "~90 s", inferred from a smoke run at ten
# subjects where the first fit took 86.7 s against a second fit's 2.7 s. That is an observation,
# not a measurement, and it is at the wrong cohort size.
#
# This script measures it, by doing the one thing the others avoid: timing the first of
# everything as well as the second. In a cold process, on one 100-subject cohort,
#
#   t_include   parsing and loading the package and its dependencies
#   t_backends  bringing up the AD backends and their rules
#   t_fit1      the first fit, which absorbs JIT for the whole objective/gradient path
#   t_fit2      an identical second fit on the same data — the steady-state estimation cost
#   t_info1     the first observed information, which absorbs JIT for the differencing path
#   t_info2     an identical second one — the steady-state information cost
#
# Compilation is then (t_fit1 - t_fit2) + (t_info1 - t_info2), with load time reported separately
# because it is a different thing: a user pays load once per session and compilation once per
# session, but pays estimation once per fit.
#
# `width` is a runtime field of `MLPEffect`, not a type parameter, so the two fits share compiled
# code — which is exactly what makes the difference between them a compilation measurement.

const T_START = time()
t_include = @elapsed include(joinpath(@__DIR__, "..", "..", "src", "warfarin.jl"))

BLAS.set_num_threads(1)
t_backends = @elapsed load_backends!(verbose = false)
banner("E14 — compile, estimate and information cost of one 100-subject M_nn(10) run")

const RESULTS = results_dir(@__DIR__, "results")
const WIDTH = SCIENTIFIC_WIDTH
const N_SUBJECTS = 100
const MAXITERS = 400

cfg = default_config(reg = 1e-8, inner_tol = 1e-8, inner_maxiter = 200, threaded = true)
rcfg = reverse_config()
model = WarfarinTurnover(MLPEffect(WIDTH))

sim = simulate_cohort(N_SUBJECTS; seed = SEED_COHORT + 1)
subs = [make_subject(model, s.id, s.input, s.covariates.pk_times,
                     s.observations[1:s.covariates.n_pk], s.covariates.pd_times,
                     s.observations[s.covariates.n_pk+1:end]) for s in sim.pop.subjects]
pop = PopulationData(subs)

@printf("\n%d subjects, W = %d, p = %d, %d julia threads, %d BLAS threads\n",
        N_SUBJECTS, WIDTH, model.p, Threads.nthreads(), BLAS.get_num_threads())
@printf("include %.1f s, backends %.1f s\n", t_include, t_backends); flush(stdout)

fit_once() = fit_warfarin(model, pop; cfg = cfg, rcfg = rcfg, ccfg = CC_LAPLACE,
                          maxiters = MAXITERS)

println("\nfit 1 (cold: absorbs compilation) ..."); flush(stdout)
t_fit1 = @elapsed f1 = fit_once()
@printf("  %.1f s, %d iterations, OFV %.1f\n", t_fit1, f1.result.iterations, f1.ofv); flush(stdout)

println("fit 2 (warm: steady-state estimation) ..."); flush(stdout)
t_fit2 = @elapsed f2 = fit_once()
@printf("  %.1f s, %d iterations, OFV %.1f\n", t_fit2, f2.result.iterations, f2.ofv); flush(stdout)

println("information 1 (cold) ..."); flush(stdout)
t_info1 = @elapsed H = Matrix(observed_information(f2.grad_closure, f2.result.theta;
                                                   step = 1e-5, central = true))
@printf("  %.1f s\n", t_info1); flush(stdout)

println("information 2 (warm: steady-state) ..."); flush(stdout)
t_info2 = @elapsed observed_information(f2.grad_closure, f2.result.theta; step = 1e-5, central = true)
@printf("  %.1f s\n", t_info2); flush(stdout)

t_compile = max(0.0, t_fit1 - t_fit2) + max(0.0, t_info1 - t_info2)
t_total = time() - T_START

rows = Any[csvrow("quantity", "seconds", "note")]
push!(rows, csvrow("include_package", t_include, "parse and load the package and dependencies"))
push!(rows, csvrow("load_backends", t_backends, "AD backends and rules"))
push!(rows, csvrow("fit_cold", t_fit1, "first fit; includes JIT of the objective/gradient path"))
push!(rows, csvrow("fit_warm", t_fit2, "steady-state estimation, identical data and settings"))
push!(rows, csvrow("information_cold", t_info1, "first observed information; includes JIT"))
push!(rows, csvrow("information_warm", t_info2, "steady-state observed information, 2p gradients"))
push!(rows, csvrow("compilation", t_compile, "(fit_cold - fit_warm) + (info_cold - info_warm)"))
push!(rows, csvrow("session_total", t_total, "wall clock for this whole script"))
push!(rows, csvrow("julia_threads", Threads.nthreads(), "-t on the command line"))
push!(rows, csvrow("blas_threads", BLAS.get_num_threads(), "set to 1; parallelism is over subjects"))
push!(rows, csvrow("cpu_threads", Sys.CPU_THREADS, "hardware threads visible to the process"))
push!(rows, csvrow("n_subjects", N_SUBJECTS, ""))
push!(rows, csvrow("p", model.p, "population parameters at W = $(WIDTH)"))
writedlm(joinpath(RESULTS, "e14_single_run_budget.csv"), vcat(rows...), ',')

println("\n", "="^78)
@printf("%-22s %10s\n", "quantity", "seconds")
@printf("%-22s %10.1f\n", "include package", t_include)
@printf("%-22s %10.1f\n", "load backends", t_backends)
@printf("%-22s %10.1f\n", "compilation", t_compile)
@printf("%-22s %10.1f\n", "estimation (warm)", t_fit2)
@printf("%-22s %10.1f\n", "information (warm)", t_info2)
@printf("%-22s %10d\n", "julia threads", Threads.nthreads())
@printf("%-22s %10d\n", "hardware threads", Sys.CPU_THREADS)
println("""

Compilation is paid once per session, not once per fit: the second fit shares the first's compiled
code because the network width is a runtime field rather than a type parameter. Estimation and
information are paid per fit.""")
println("=" ^ 78)
