# E2 — cost, and what it buys.
#
#   julia --project=Local_Env -t 5 warfarin_turnover/examples/cost/e2_cost_ladder.jl
#
# **This is a gradient-benchmark experiment, not a fitting experiment.** It measures the cost of
# *one gradient* at a fixed θ across the `p` ladder. Nothing here fits anything.
#
# Every arm is the **exact Laplace** objective. There is one curvature in this study, so no row
# below carries a curvature column and no ratio needs qualifying by one.
#
# | arm | what it computes | θ-derivative | expected |
# |---|---|---|---|
# | `objective` | `L(θ)` itself | none | the normalising denominator |
# | `reverse_laplace` | `∇L(θ)` | one reverse sweep of the single scalar | a fixed multiple of one objective |
# | `forward_laplace` | `∇L(θ)` | forward sensitivity (Almquist orientation) | `p` × one objective |
#
# ---------------------------------------------------------------------------------------
# The headline is the NORMALISED exponent, and the reason is not presentational
# ---------------------------------------------------------------------------------------
#
# In raw wall clock the forward arm here is `O(p²)`, not `O(p)`. ForwardDiff needs `p/12` chunked
# passes, each costing ~13× one `h` evaluation — that is the `O(p)` factor everyone quotes. But
# in *this* model `h` itself costs `O(W²) = O(p)`, because `p` **is** the network. The two
# multiply.
#
# The same argument applies to the reverse arm: one sweep at a fixed multiple of `h`, and `h`
# grows like `p`, so raw reverse wall clock is `O(p)`, not flat. Quoting "flat in p" against a
# measured `p¹` would be wrong, and quoting a `p²`-vs-`p¹` gap would be crediting the method with
# the model's own cost growth.
#
# So the quantity reported as the claim is **`gradient_time / objective_time`**, whose exponent in
# `p` is 0 for the reverse arm and 1 for the `O(p)` arm whatever the model costs. Raw exponents
# are written to the same CSV beside it, never instead of it.
#
# ---------------------------------------------------------------------------------------
# The forward arm is the incumbent's ACTUAL construction, not a strawman
# ---------------------------------------------------------------------------------------
#
# Nobody differentiates through the inner optimizer, and benchmarking against something nobody
# does would be worthless. `subject_forward` implements what the incumbent implements:
#
#   * the **envelope theorem** for the leading term — `∂θ[h(η̂,θ) + ½ logdet H̃(η̂,θ)]` with `η̂`
#     detached to plain floats, so the inner solve is never traversed;
#   * the **implicit function theorem** for the mode sensitivity — `S = dη̂/dθ = −H⁻¹ B` with
#     `B = ∇²_ηθ h`, `H` the exact mode-equation Jacobian, obtained by **one** Cholesky
#     factorisation with `p` right-hand sides, not `p` factorisations;
#   * every partial from **forward-mode AD**.
#
# `B` is the `q × p` Jacobian the incumbent calls `∂g/∂θ`, and it **is** the `O(p)` term — `p`
# forward tangents. It is exactly the object the reverse construction contracts from the left
# (`Sᵀb = −Bᵀλ`) and never forms. That identity is checked in the E1 gate, so the two arms are
# verified to compute the same gradient before either is timed.
#
# ---------------------------------------------------------------------------------------
# Why the observed-information curve is a citation, not an assumption
# ---------------------------------------------------------------------------------------
#
# `e2_information_cost.csv` costs the observed information as `2p` gradient evaluations. That is
# not a hypothetical worst case for the forward arm — it is the incumbent's implementation: the
# Laplace observed information is built by **finite differences of gradients**, precisely because
# the objective embeds an inner solve. So the forward arm pays `2p` outer evaluations at `O(p)`
# each, i.e. `O(p²)`, and the reverse arm pays `2p` at `O(1)` each. That curve, not the raw
# gradient ratio, is the cost result.
#
# ---------------------------------------------------------------------------------------
# Like-for-like discipline
# ---------------------------------------------------------------------------------------
#
#   * The **exponent in `p`** is the headline; the ratio is illustration; **allocations** are the
#     load-independent axis. Every timing carries its thread count and the machine load.
#   * Every arm pays cold inner solves — no `fixed_starts` anywhere in the ladder — so no arm is
#     charged for work another arm was spared. The confound section below then measures what
#     that costs, by giving *every* arm the same fixed starts.
#
# ---------------------------------------------------------------------------------------
# Warm-up — why it is a separate sweep and not a call before each timing
# ---------------------------------------------------------------------------------------
#
# `width` is a runtime field of `MLPEffect`, not a type parameter, so **every rung of the ladder
# shares the same compiled code and the same Mooncake rules**. Whichever rung is visited first
# absorbs all compilation. Warming one arm at one width and then timing the ladder in place is
# exactly how a negative "scaling exponent" gets manufactured — it happened once already in this
# project's history and it is the reason this comment exists.
#
# ForwardDiff's chunk size is a *type* parameter, but it saturates at 12 for any `p > 12`, so it
# is `Chunk{12}` at every rung here and does not reintroduce per-width compilation.
#
# **Every arm is warmed at every rung**, and every result of that sweep is discarded. Each row
# records `warm_at_width` so that stays checkable rather than assumed.

include(joinpath(@__DIR__, "..", "..", "src", "warfarin.jl"))

BLAS.set_num_threads(1)
loaded = load_backends!(verbose = true)     # top level: see the note in the E1 gate
backend_available(:mooncake) ||
    error("E2 requires the Mooncake engine; loaded = $(loaded). Without it every arm has the " *
          "same cost model and the ladder cannot show a reverse effect.")

# Garbage collection is a first-class part of this measurement, not an implementation detail.
# The O(p) forward-sensitivity arm allocates heavily — ForwardDiff intermediates for `p` tangents,
# across threads — and under Julia's default heap target a 5-second stack sample of an earlier run
# came back 142/164 in `jl_gc_collect`: the ladder was measuring collection, not gradients. Run
# with `--heap-size-hint` set well below physical RAM and the collector stops dominating. The
# hint, the GC thread count and the per-arm GC seconds are all recorded on every row, because a
# timing taken under a different heap target is a different timing.
const HEAP = (hint_gb = Base.JLOptions().heap_size_hint == typemax(UInt64) ? NaN :
                        Base.JLOptions().heap_size_hint / 2^30,
              gcthreads = Threads.ngcthreads())

banner("E2 — gradient cost across the p ladder (exact Laplace)")
@printf("  heap-size hint %s   GC threads %d   physical RAM %.0f GB\n",
        isnan(HEAP.hint_gb) ? "(default)" : @sprintf("%.0f GB", HEAP.hint_gb),
        HEAP.gcthreads, Sys.total_memory() / 2^30)

const RESULTS = results_dir(@__DIR__, "results")

# `derive` rebuilds every derived table (summary, ratios, observed-information cost, the
# projection) from the committed `e2_ladder.csv` — no model built, nothing simulated, nothing
# timed. The artefact rule that a figure must never be a reason to refit applies to derived
# TABLES too: a units change or a new ratio should cost a second, not an evening. There is only
# one implementation of the arithmetic; this switch chooses where its inputs come from.
const DERIVE = length(ARGS) >= 1 && ARGS[1] == "derive"
const SEED_E2 = 20260812
# The ladder measures an **exponent in `p`**, and `N` does not enter it: every arm's cost is
# per-subject work summed over subjects, so changing `N` rescales every arm identically and
# leaves the exponent alone. `N = 20` therefore buys the same headline at a fifth of the cost.
# `N`-linearity is not assumed on that argument — it is measured separately below, at the
# scientific width, and written to its own CSV.
const N_SUBJECTS = 20
const N_LADDER = (10, 20, 50, 100)      # the separate N sweep, at the scientific width
const PERTURB = 0.05

# Wall-clock guards. Both are printed on the row they apply to; nothing is ever cut silently.
const BUDGET_S = 2400.0      # an O(p) arm projected above this at the next rung is cut
const CHEAP_S = 120.0        # an O(p) arm below this gets 2 reps, above it gets 1

# `:matrix` is the scalar form the reverse arm is timed in, matching the form E1 gates first.
# The two forms agree to round-off (E1) and differ only in construction, so one of them is what
# a cost figure needs.
const RC_REVERSE = ReverseConfig(backend = :mooncake, form = :matrix)

const ARM_ORDER = ["objective", "reverse_laplace", "forward_laplace"]
const ONP = Set(["forward_laplace"])
const BACKEND = Dict("objective" => "-", "reverse_laplace" => "mooncake",
                     "forward_laplace" => "forwarddiff")
const FORM = Dict("objective" => "-", "reverse_laplace" => "matrix",
                  "forward_laplace" => "sensitivity")

# ---------------------------------------------------------------------------
# One rung of the ladder
# ---------------------------------------------------------------------------

function build_case(width::Int)
    m = WarfarinTurnover(MLPEffect(width))
    sim = simulate_cohort(N_SUBJECTS; seed = SEED_E2)
    subs = [make_subject(m, s.id, s.input, s.covariates.pk_times,
                         s.observations[1:s.covariates.n_pk], s.covariates.pd_times,
                         s.observations[s.covariates.n_pk+1:end]) for s in sim.pop.subjects]
    pop = PopulationData(subs)
    cfg = default_config(reg = 1e-8, inner_tol = 1e-8, inner_maxiter = 200, threaded = true)
    θ = theta_start(m) .+ PERTURB .* randn(MersenneTwister(SEED_E2 + width), m.p)

    arms = Dict{String,Function}(
        "objective"       => () -> laplace_objective(m, pop, cfg)(θ),
        "reverse_laplace" => () -> reverse_gradient_parts(m, θ, pop, cfg, RC_REVERSE).grad,
        "forward_laplace" => () -> forward_gradient_parts(m, θ, pop, cfg).grad,
    )
    return (width = width, m = m, p = m.p, pop = pop, cfg = cfg, theta = θ, arms = arms)
end

DERIVE || println("\nbuilding the ladder ...")
cases = DERIVE ? NamedTuple[] : [build_case(w) for w in (5, 10, 20, 40)]
if DERIVE
    raw, hdr = readdlm(joinpath(RESULTS, "e2_ladder.csv"), ','; header = true)
    h = Dict(String(x) => j for (j, x) in enumerate(vec(hdr)))
    global timing = Dict{Tuple{String,Int},Float64}()
    global alloc = Dict{Tuple{String,Int},Float64}()
    global widths = Int[]
    global ps = Float64[]
    for r in eachrow(raw)
        Bool(r[h["ran"]]) || continue
        a = String(r[h["method"]]); w = Int(r[h["width"]])
        timing[(a, w)] = Float64(r[h["seconds_min"]])
        alloc[(a, w)] = Float64(r[h["allocated_bytes"]])
        w in widths || (push!(widths, w); push!(ps, Float64(r[h["p"]])))
    end
    o = sortperm(ps); widths = widths[o]; ps = ps[o]
    @printf("derive mode: %d arms x %d rungs read from e2_ladder.csv (p = %s)\n",
            length(unique(k[1] for k in keys(timing))), length(widths), string(Int.(ps)))
end
nobs = DERIVE ? (total = -1, n_pk = -1, n_pd = -1) : n_observations(cases[1].pop)
if !DERIVE
    @printf("N = %d subjects, %d observations (%d PK + %d PD), q = 5\n",
            N_SUBJECTS, nobs.total, nobs.n_pk, nobs.n_pd)
    for c in cases
        @printf("  W = %3d   p = %6d\n", c.width, c.p)
    end
end

# ---------------------------------------------------------------------------
# Warm-up sweep — everything, before anything is timed
# ---------------------------------------------------------------------------

DERIVE || println("\nwarm-up sweep (every result discarded):")
warm_at = Dict{String,Int}()
# The `O(1)` arms are warmed at every rung. The `O(p)` arm is warmed only at the two cheap rungs:
# compilation is keyed on the TYPE, which is width-independent, so a warm call at W = 5 has
# already paid for every rung. Warming `forward_laplace` again at W = 40 buys nothing and costs
# minutes. `warm_at_width` on every row records which rung each arm was actually warmed at.
for c in cases, a in ARM_ORDER
    (a in ONP && c.width > 10) && continue
    t = @elapsed c.arms[a]()
    haskey(warm_at, a) || (warm_at[a] = c.width)
    warm_at[a] = c.width
    @printf("  W=%3d  %-20s %9.3f s (cold, discarded)\n", c.width, a, t)
    flush(stdout)
end

# ---------------------------------------------------------------------------
# Timing — rep-major, arm order rotated per rung, GC before each call
# ---------------------------------------------------------------------------

rows = Any[["p" "width" "n_subjects" "n_obs" "method" "backend" "form" "seconds_min" "seconds_median" "gc_seconds_min" "allocated_bytes" "rel_err_vs_reverse" "grad_inf_norm" "inner_nonconverged" "worst_inner_gnorm" "mean_inner_iters" "reps" "warm_at_width" "nthreads" "blas_threads" "gc_threads" "heap_size_hint_gb" "loadavg1" "rk4_n_seg" "rk4_gamma" "ran" "cut_reason"]]

lap_rows = Any[["p" "width" "n_subjects" "subjects_with_nonpd_htilde" "min_eig_htilde" "objective_nonfinite" "gradient_nonfinite_components"]]

if !DERIVE
    global timing = Dict{Tuple{String,Int},Float64}()
    global alloc = Dict{Tuple{String,Int},Float64}()
end
repsused = Dict{Tuple{String,Int},Int}()
cut = Dict(a => false for a in ARM_ORDER)

DERIVE || println("\ntimings:")
@printf("%6s %5s  %-20s %10s %10s %14s %8s %6s\n",
        "p", "W", "arm", "min_s", "median_s", "alloc_MB", "×obj", "reps")

for (ci, c) in enumerate(cases)
    # the reference gradient, for the correctness column
    ref = reverse_gradient_parts(c.m, c.theta, c.pop, c.cfg, RC_REVERSE)
    scale = max(maximum(abs, ref.grad), 1.0)

    # The objective is timed first, unrotated, because every other arm's headline number is
    # normalised by it. Rotating the remaining arms keeps drifting background load from landing
    # preferentially on any one of them.
    rest = filter(!=(1), collect(eachindex(ARM_ORDER)))
    order = vcat(1, circshift(rest, ci - 1))
    for k in order
        a = ARM_ORDER[k]
        onp = a in ONP

        # reps and the budget cut, both explicit
        prev = get(timing, (a, ci > 1 ? cases[ci-1].width : c.width), NaN)
        projected = onp && isfinite(prev) ? prev * (c.p / cases[max(ci - 1, 1)].p) : NaN
        reason = ""
        if cut[a]
            reason = "cut at an earlier rung"
        elseif onp && isfinite(projected) && projected > BUDGET_S
            cut[a] = true
            reason = @sprintf("projected %.0f s from the previous rung exceeds the %.0f s budget",
                              projected, BUDGET_S)
        end
        if !isempty(reason)
            @printf("%6d %5d  %-20s %10s %10s %14s %8s %6s   CUT: %s\n",
                    c.p, c.width, a, "-", "-", "-", "-", "0", reason)
            flush(stdout)
            push!(rows, [c.p c.width N_SUBJECTS nobs.total a BACKEND[a] FORM[a] NaN NaN NaN NaN NaN NaN -1 NaN NaN 0 get(warm_at, a, -1) Threads.nthreads() BLAS.get_num_threads() HEAP.gcthreads HEAP.hint_gb Sys.loadavg()[1] c.m.n_seg c.m.gamma false reason])
            continue
        end

        reps = !onp ? 3 : (isfinite(prev) && prev < CHEAP_S ? 2 : (c.width <= 10 ? 3 : 1))
        ts = Float64[]
        # `@timed` yields the time, the allocations AND the returned gradient from ONE call, so a
        # rep is a rep. The earlier shape — `reps` timed calls, then a separate `@allocated` call,
        # then another call to refill the gradient for the correctness check — charged a
        # 2-extra-call surcharge on every arm, which on an O(p) arm whose single call takes
        # minutes was most of the ladder's wall clock and bought nothing.
        ab = 0
        gout = nothing
        gct = Float64[]
        for _ in 1:reps
            GC.gc()
            r = @timed c.arms[a]()
            push!(ts, r.time)
            push!(gct, r.gctime)
            ab = r.bytes
            gout = r.value
        end

        tmin = minimum(ts)
        timing[(a, c.width)] = tmin
        repsused[(a, c.width)] = reps
        alloc[(a, c.width)] = Float64(ab)

        err, ng, nnc, wg, mi = NaN, NaN, -1, NaN, NaN
        if a != "objective"
            g = gout                                    # reuse the last timed rep's result
            err = maximum(abs, g .- ref.grad) / scale
            ng = maximum(abs, g)
            nnc, wg, mi = ref.n_nonconverged, ref.worst_gnorm, mean(ref.iters)
        end

        tobj = get(timing, ("objective", c.width), NaN)
        @printf("%6d %5d  %-20s %10.3f %10.3f %14.1f %8.2f %6d\n",
                c.p, c.width, a, tmin, median(ts), ab / 1e6, tmin / tobj, reps)
        flush(stdout)
        push!(rows, [c.p c.width N_SUBJECTS nobs.total a BACKEND[a] FORM[a] tmin median(ts) minimum(gct) ab err ng nnc wg mi reps get(warm_at, a, -1) Threads.nthreads() BLAS.get_num_threads() HEAP.gcthreads HEAP.hint_gb Sys.loadavg()[1] c.m.n_seg c.m.gamma true ""])
    end

    # ---- numerical diagnostics on the curvature, at the benchmark point ----
    # The exact-Laplace objective was previously observed on other models to need dozens to
    # hundreds of rescues per fit and to go non-finite where H̃ was indefinite. At q = 5 with ~20
    # observations per subject it should be far better behaved, and a well-behaved result here is
    # itself worth reporting.
    npd, mineig = 0, Inf
    for s in c.pop.subjects
        sh = subject_reverse(c.m, c.theta, s, c.cfg, RC_REVERSE)
        H = Symmetric(Float64.(sh.Htilde))
        e1 = minimum(eigvals(H))
        mineig = min(mineig, e1)
        issuccess(cholesky(H; check = false)) || (npd += 1)
    end
    Lval = laplace_objective(c.m, c.pop, c.cfg)(c.theta)
    push!(lap_rows, [c.p c.width N_SUBJECTS npd mineig (isfinite(Lval) ? 0 : 1) count(!isfinite, ref.grad)])
    @printf("       curvature diagnostics: non-PD H̃ subjects = %d/%d, min eig = %.3e, non-finite grad components = %d\n",
            npd, c.pop.n_subjects, mineig, count(!isfinite, ref.grad))
    flush(stdout)

    # Rewritten after EVERY rung, not once at the end. A ladder is a long run, and a killed one
    # must leave behind the rungs it did finish — that is the difference between losing a rung and
    # losing an evening. The final write below is then a no-op safety net.
    writedlm(joinpath(RESULTS, "e2_ladder.csv"), vcat(rows...), ',')
    writedlm(joinpath(RESULTS, "e2_laplace_diagnostics.csv"), vcat(lap_rows...), ',')
end

if !DERIVE
    writedlm(joinpath(RESULTS, "e2_ladder.csv"), vcat(rows...), ',')
    writedlm(joinpath(RESULTS, "e2_laplace_diagnostics.csv"), vcat(lap_rows...), ',')
end

# ---------------------------------------------------------------------------
# The shared-inner-solve confound, measured rather than argued about
#
# Every arm above pays a cold inner solve, so no arm is charged for work another was spared. That
# is fair, but it also DILUTES the ratio under study: the mode solve is a prerequisite both arms
# pay identically, and the claim is about the θ-derivative that follows it. Giving every arm the
# same history-independent `fixed_starts` is equally fair and isolates the θ-derivative.
#
# Rather than choose, measure both and report both — total, and net of the shared inner solve.
# The confound is only ever a problem when one arm gets fixed starts and the other does not.
# ---------------------------------------------------------------------------

DERIVE || println("\ncold vs fixed-start (the shared-inner-solve confound, both arms treated alike):")
crows = Any[["p" "width" "method" "fixed_starts" "seconds_min" "reps" "nthreads" "loadavg1"]]
for c in cases
    eta0 = [collect(Float64, e) for e in population_objective(c.m, c.theta, c.pop, c.cfg).etas]
    warm = Dict{String,Function}(
        "objective"       => () -> population_objective(c.m, c.theta, c.pop, c.cfg; eta_starts = eta0).L,
        "reverse_laplace" => () -> reverse_gradient_parts(c.m, c.theta, c.pop, c.cfg, RC_REVERSE; eta_starts = eta0).grad,
        "forward_laplace" => () -> forward_gradient_parts(c.m, c.theta, c.pop, c.cfg; eta_starts = eta0).grad)
    for a in ARM_ORDER
        warm[a]()                                   # warm this closure too
        r = min(3, a == "forward_laplace" && c.width > 20 ? 1 : 3)
        tw = minimum((GC.gc(); @elapsed warm[a]()) for _ in 1:r)
        tc = get(timing, (a, c.width), NaN)
        push!(crows, [c.p c.width a true tw r Threads.nthreads() Sys.loadavg()[1]])
        push!(crows, [c.p c.width a false tc get(repsused, (a, c.width), 0) Threads.nthreads() Sys.loadavg()[1]])
        @printf("  p=%5d %-16s cold %8.3f s   fixed-start %8.3f s   (%.2fx)\n", c.p, a, tc, tw, tc / tw)
        flush(stdout)
    end
end
DERIVE || writedlm(joinpath(RESULTS, "e2_confound_fixed_starts.csv"), vcat(crows...), ',')

# ---------------------------------------------------------------------------
# N-linearity at the scientific width — measured, not assumed
# ---------------------------------------------------------------------------

DERIVE || println("\nN-linearity at the scientific width (W = $(SCIENTIFIC_WIDTH)):")
nrows = Any[["p" "width" "n_subjects" "n_obs" "method" "seconds_min" "reps" "nthreads" "loadavg1"]]
DERIVE || let m = WarfarinTurnover(MLPEffect(SCIENTIFIC_WIDTH))
    θn = theta_start(m) .+ PERTURB .* randn(MersenneTwister(SEED_E2 + SCIENTIFIC_WIDTH), m.p)
    cfgn = default_config(reg = 1e-8, inner_tol = 1e-8, inner_maxiter = 200, threaded = true)
    for N in N_LADDER
        sim = simulate_cohort(N; seed = SEED_E2)
        subs = [make_subject(m, s.id, s.input, s.covariates.pk_times,
                             s.observations[1:s.covariates.n_pk], s.covariates.pd_times,
                             s.observations[s.covariates.n_pk+1:end]) for s in sim.pop.subjects]
        popn = PopulationData(subs)
        nb = n_observations(popn)
        arms_n = Dict{String,Function}(
            "objective"       => () -> laplace_objective(m, popn, cfgn)(θn),
            "reverse_laplace" => () -> reverse_gradient_parts(m, θn, popn, cfgn, RC_REVERSE).grad,
            "forward_laplace" => () -> forward_gradient_parts(m, θn, popn, cfgn).grad)
        for a in ARM_ORDER
            arms_n[a]()
            t = minimum((GC.gc(); @elapsed arms_n[a]()) for _ in 1:3)
            push!(nrows, [m.p SCIENTIFIC_WIDTH N nb.total a t 3 Threads.nthreads() Sys.loadavg()[1]])
            @printf("  N=%4d %-16s %8.3f s\n", N, a, t); flush(stdout)
        end
    end
end
DERIVE || writedlm(joinpath(RESULTS, "e2_N_linearity.csv"), vcat(nrows...), ',')

# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

"""Least-squares slope of `log y` on `log p`, and the intercept."""
function powfit(ps, ys)
    ok = (ps .> 0) .& (ys .> 0) .& isfinite.(ys)
    n = sum(ok)
    n < 2 && return (NaN, NaN, n)
    c = [ones(n) log.(ps[ok])] \ log.(ys[ok])
    return (c[1], c[2], n)
end

if !DERIVE
    global widths = [c.width for c in cases]
    global ps = [Float64(c.p) for c in cases]
end

# `normalised_exponent_in_p` is THE headline: the exponent of gradient_time / objective_time. It
# is 0 for the reverse arm and 1 for the O(p) arm however the model's own cost grows with p.
# `time_exponent_in_p` is the raw wall-clock exponent, written beside it and never instead.
tobjs = [get(timing, ("objective", w), NaN) for w in widths]
srows = Any[["method" "normalised_exponent_in_p" "time_exponent_in_p" "alloc_exponent_in_p" "n_points" "max_p_measured" "seconds_at_max_p" "seconds_at_p_min" "x_objective_at_p_min" "x_objective_at_max_p"]]
for a in ARM_ORDER
    ts = [get(timing, (a, w), NaN) for w in widths]
    as = [get(alloc, (a, w), NaN) for w in widths]
    _, tb, n = powfit(ps, ts)
    _, ab_, _ = powfit(ps, as)
    _, nb, _ = powfit(ps, ts ./ tobjs)
    idx = findall(isfinite, ts)
    isempty(idx) && continue
    push!(srows, [a nb tb ab_ n ps[idx[end]] ts[idx[end]] ts[idx[1]] ts[idx[1]]/tobjs[idx[1]] ts[idx[end]]/tobjs[idx[end]]])
end
length(srows) >= 2 || error(
    "derived summary is empty — refusing to overwrite e2_summary.csv. This means `timing` was " *
    "not populated (in derive mode, that it was clobbered after being read from the ladder CSV). " *
    "A derive pass must never replace a measured artefact with a header.")
writedlm(joinpath(RESULTS, "e2_summary.csv"), vcat(srows...), ',')

# The headline ratio. With one curvature there is exactly one of them, and no cross-curvature
# comparison to qualify or mislabel.
rrows = Any[["p" "width" "numerator" "denominator" "time_ratio" "alloc_ratio" "what"]]
for (i, w) in enumerate(widths)
    tn = get(timing, ("forward_laplace", w), NaN); td = get(timing, ("reverse_laplace", w), NaN)
    an = get(alloc, ("forward_laplace", w), NaN); ad = get(alloc, ("reverse_laplace", w), NaN)
    push!(rrows, reshape(Any[ps[i], w, "forward_laplace", "reverse_laplace",
                               tn / td, an / ad,
                               "INCUMBENT vs reverse, exact Laplace — the headline ratio"], 1, :))
end
writedlm(joinpath(RESULTS, "e2_ratios.csv"), vcat(rrows...), ',')

# ---------------------------------------------------------------------------
# The derived quantity that makes the figure matter: the cost of the observed
# information (2p gradients) against p, per arm.  Reverse is O(p); forward is O(p²).
# This curve, not the raw gradient ratio, is the paper's cost result.
# ---------------------------------------------------------------------------

irows = Any[["p" "width" "method" "seconds_per_gradient" "seconds_2p_gradients" "hours_2p_gradients" "basis"]]
for (i, w) in enumerate(widths), a in ARM_ORDER
    a == "objective" && continue
    t = get(timing, (a, w), NaN)
    basis = isfinite(t) ? "measured gradient x 2p" : "not measured at this rung (arm cut)"
    push!(irows, [ps[i] w a t 2 * ps[i] * t 2 * ps[i] * t / 3600 basis])
end
# Projection beyond the measured ladder. The ladder tops out at W = 40; the published neural
# NLME stress-test scale is p ~ 1.5e4. Extrapolating there is legitimate and is
# labelled PROJECTED on every row it touches - a projection is not a measurement.
for a in ARM_ORDER
    a == "objective" && continue
    ts = [get(timing, (a, w), NaN) for w in widths]
    ta, tb, n = powfit(ps, ts)
    n < 2 && continue
    for ptgt in (1.0e4, 1.5e4)
        tproj = exp(ta + tb * log(ptgt))
        basis = "PROJECTED from a p^$(round(tb; digits = 3)) fit over p in [$(Int(ps[1])), $(Int(ps[end]))]"
        push!(irows, [ptgt -1 a tproj (2 * ptgt * tproj) (2 * ptgt * tproj / 3600) basis])
    end
end
writedlm(joinpath(RESULTS, "e2_information_cost.csv"), vcat(irows...), ',')

println("\n" * "="^78)
println("E2 complete. Wrote to ", RESULTS)
for f in ("e2_ladder.csv", "e2_summary.csv", "e2_ratios.csv",
          "e2_information_cost.csv", "e2_laplace_diagnostics.csv")
    println("  ", f)
end
println("\nExponents in p.  NORMALISED (grad/objective) is the claim; raw is beside it.")
println("  the reverse arm should give normalised ~0; the O(p) arm should give normalised ~1.")
for r in srows[2:end]
    @printf("  %-20s  normalised p^%-7.3f  raw p^%-7.3f  alloc p^%-7.3f  (%d pts, max p=%.0f, %.1f×obj there)\n",
            r[1], r[2], r[3], r[4], r[5], r[6], r[10])
end
println("="^78)
