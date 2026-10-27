# design.jl — the sampling design and the subject constructor.
#
# One place builds a `SubjectData`, whether the observations came from the real file or from
# the simulator, so the RK4 grid attached to a subject is constructed by the same rule in
# both cases and a real-data fit and a simulated fit are the same objective.

using NLMEReverse
using Random

# ---------------------------------------------------------------------------
# Seeds — named constants, per the house rules.  Increasing a replicate count must be a
# single constant, so cohort r uses SEED_COHORT + r and nothing aggregates across r.
# ---------------------------------------------------------------------------
const SEED_COHORT = 20260807      # simulated cohorts: cohort r uses SEED_COHORT + r
const SEED_DESIGN = 20260808      # dose assignment / jitter, held fixed across cohorts
const SEED_START = 20260809       # network weight initialisation
const SEED_BOOTSTRAP = 20260810   # E4's nonparametric bootstrap resamples
const SEED_MISSPEC   = 20260812   # E9's cohorts simulated from the sigmoid-Emax truth

# ---------------------------------------------------------------------------
# The simulated design (§2.3 of the plan)
# ---------------------------------------------------------------------------

const TMAX = 168.0                             # hours
const N_PK = 8
const N_PD = 12

"""
Three dose levels, approximately 0.5, 1.5 and 4 mg/kg at a 70 kg reference, balanced across
the cohort.

This departs from the real study's single dose level and the departure is deliberate: **a
single dose level cannot identify the shape of a saturating effect curve.**  It is a
plausible partial explanation for why the published comparison of parametric, neural and
soft-decision-tree effect functions found them indistinguishable on held-out MSE, and it is
worth one sentence in the Discussion.
"""
const DOSE_LEVELS = (35.0, 105.0, 280.0)       # mg, = 0.5 / 1.5 / 4 mg/kg × 70 kg

"""Eight PK sampling times, log-spaced from 0.5 h to 120 h."""
pk_design_times() = exp.(range(log(0.5), log(120.0); length = N_PK))

"""
Twelve PD sampling times: the baseline at `t = 0`, then eleven log-spaced from 6 h to 168 h.

Keeping `t = 0` matters — `R(0) = R₀` exactly, so the baseline observation identifies `R₀`
almost directly and stops it trading off against `k_out`.
"""
pd_design_times() = vcat(0.0, exp.(range(log(6.0), log(TMAX); length = N_PD - 1)))

# ---------------------------------------------------------------------------
# Subject construction
# ---------------------------------------------------------------------------

"""
    make_subject(model, id, dose, pk_times, pk_obs, pd_times, pd_obs) -> SubjectData

The RK4 grid and its stage times are built here, once, in `Float64`, and travel with the
subject.  That is what makes the march a fixed-step one: nothing about the discretisation
depends on `θ` or `η`, so nothing about the objective's smoothness does either.

`times` and `observations` are the concatenation PK-then-PD, matching `conditional_moments`.
"""
function make_subject(m::WarfarinTurnover, id::Int, dose::Float64,
                      pk_times::AbstractVector{Float64}, pk_obs::AbstractVector{Float64},
                      pd_times::AbstractVector{Float64}, pd_obs::AbstractVector{Float64})
    length(pk_times) == length(pk_obs) ||
        error("subject $id: $(length(pk_times)) PK times against $(length(pk_obs)) PK observations.")
    length(pd_times) == length(pd_obs) ||
        error("subject $id: $(length(pd_times)) PD times against $(length(pd_obs)) PD observations.")

    ordpk = sortperm(collect(Float64, pk_times))
    ordpd = sortperm(collect(Float64, pd_times))
    pkt = collect(Float64, pk_times)[ordpk]
    pky = collect(Float64, pk_obs)[ordpk]
    pdt = collect(Float64, pd_times)[ordpd]
    pdy = collect(Float64, pd_obs)[ordpd]

    tmax = max(maximum(pdt), maximum(pkt))
    g = build_grid(pdt; tmax = tmax, n_seg = m.n_seg, gamma = m.gamma)

    cov = (pk_times = pkt, pd_times = pdt, n_pk = length(pkt), n_pd = length(pdt),
           stage_times = g.stage_times, grid = g.grid, obs_idx = g.obs_idx, dose = dose)

    return SubjectData(id, vcat(pkt, pdt), vcat(pky, pdy), dose; covariates = cov)
end

"""
    n_observations(pop) -> (n_pk, n_pd, total)

Counted, printed, and written onto every artefact.  ~20 observations per subject against
`q = 5` is a design decision, not an accident, and it is what makes the Gauss-Newton
curvature close enough to the exact Hessian that the FOCE approximation is not load-bearing
for any result here.
"""
function n_observations(pop::PopulationData)
    npk = sum(s.covariates.n_pk for s in pop.subjects)
    npd = sum(s.covariates.n_pd for s in pop.subjects)
    return (n_pk = npk, n_pd = npd, total = npk + npd)
end
