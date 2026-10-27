# Figures for the manuscript.
#
#   julia --project=Local_Env -t 1 warfarin_turnover/figures/make_figures.jl
#   julia --project=Local_Env -t 1 warfarin_turnover/figures/make_figures.jl e2
#
# Reads **only** committed CSVs. Nothing here fits, simulates, or evaluates a model — that is
# the point of the artefact contract, and this script is its test: if a panel below cannot be
# drawn from a CSV, the experiment that produced it did not write enough.
#
# One function per figure in `methods_results_codex.tex`, in the order the paper prints them:
#
#   figure_cohort          Figure 1  fig12_simulated_cohort   the synthetic cohort
#   figure_e2              Figure 2  fig2_e2_cost             gradient cost against p
#   figure_e11_inference   Figure 3  fig3_e11_inference       effect-function inference
#   figure_e9              Figure 4  fig4_e9_misspecification recovery under misspecification
#   figure_e11             Figure 5  fig5_e11_realdata        the 32-subject warfarin fit

using CairoMakie, DelimitedFiles, Printf, Statistics, LinearAlgebra

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUT = @__DIR__
mkpath(OUT)

E2 = joinpath(ROOT, "warfarin_turnover", "examples", "cost", "results")
E9 = joinpath(ROOT, "warfarin_turnover", "examples", "misspecification", "results")
E11 = joinpath(ROOT, "warfarin_turnover", "examples", "regularisation", "results")
COH = joinpath(ROOT, "warfarin_turnover", "examples", "cohort", "results")


# Okabe–Ito: colourblind-safe, and the two reverse arms share a hue family against the
# forward arms so the reader sees two groups before they read a legend.
const COL = Dict(
    "reverse_foce"             => RGBf(0.00, 0.45, 0.70),
    "reverse_foce_directional" => RGBf(0.35, 0.70, 0.90),
    "reverse_laplace"          => RGBf(0.00, 0.62, 0.45),
    "forward_foce"             => RGBf(0.84, 0.37, 0.00),
    "forward_laplace"          => RGBf(0.80, 0.47, 0.65),
    "foce_fd"                  => RGBf(0.90, 0.62, 0.00),
    "objective"                => RGBf(0.40, 0.40, 0.40))

read_csv(p) = isfile(p) ? readdlm(p, ','; header = true) : nothing

"""Numeric value of a CSV cell. `readdlm` returns a heterogeneous matrix, so a column can arrive
as numbers, as strings, or as a mixture; anything that is not a number becomes `NaN` rather than
throwing, which is what the report wants for absent or non-applicable entries."""
# The PD functionals were originally keyed "R(...)" after the turnover state; they are the
# predicted prothrombin complex activity, so they display as "PCA(...)". Archived CSVs still
# carry the old key, so display goes through this rather than a rename of the stored data.
disp(f) = replace(String(f), "R(24h)" => "PCA(24h)", "R(168h)" => "PCA(168h)")

_num(x) = x isa Number ? Float64(x) :
          (v = tryparse(Float64, strip(String(string(x)))); v === nothing ? NaN : v)

col(raw, hdr, name) = raw[:, findfirst(==(name), String.(vec(hdr)))]

set_theme!(Theme(fontsize = 13, Axis = (xgridvisible = false, ygridvisible = true,
                                        ygridcolor = (:black, 0.08), rightspinevisible = false,
                                        topspinevisible = false)))

# ===========================================================================
# Figure 12 — one simulated cohort, as data
# ===========================================================================

function figure_cohort()
    d = read_csv(joinpath(COH, "cohort_fig.csv"))
    d === nothing && return
    raw, hdr = d
    c = n -> col(raw, hdr, n)
    id = Int.(_num.(c("id"))); dose = _num.(c("dose"))
    kind = String.(vec(c("kind"))); t = _num.(c("time")); dv = _num.(c("dv"))
    doses = sort(unique(dose))
    # Dose is ordered, so move from cool to warm using Okabe--Ito colours not used for the
    # green/purple method comparison in Figures 2--4.
    pal = [COL["reverse_foce_directional"], COL["foce_fd"], COL["forward_foce"]]

    W_PT = 6.2 * 72
    fig = Figure(size = (W_PT, 0.46 * W_PT), fontsize = 11, figure_padding = 4)
    panels = (("conc", "(a) warfarin concentration", "concentration (mg/L)"),
              ("pca",  "(b) prothrombin complex activity", "PCA (%)"))
    for (j, (k, ttl, ylab)) in enumerate(panels)
        ax = Axis(fig[1, j]; title = ttl, titlesize = 11,
                  xlabel = "time (h)", ylabel = ylab,
                  xlabelsize = 11, ylabelsize = 11,
                  xticklabelsize = 10, yticklabelsize = 10)
        for (di, D) in enumerate(doses)
            sel = findall(n -> kind[n] == k && dose[n] == D, eachindex(kind))
            for sid in unique(id[sel])
                idx = [n for n in sel if id[n] == sid]
                o = sortperm(t[idx]); idx = idx[o]
                lines!(ax, t[idx], dv[idx]; color = (pal[di], 0.35), linewidth = 0.6)
                scatter!(ax, t[idx], dv[idx]; color = (pal[di], 0.75), markersize = 2.6)
            end
        end
        if j == 1
            elems = [MarkerElement(color = pal[di], marker = :circle, markersize = 7)
                     for di in eachindex(doses)]
            axislegend(ax, elems, [@sprintf("%g mg", D) for D in doses], "dose";
                       position = :rt, framevisible = false, labelsize = 10,
                       titlesize = 10, rowgap = -2, patchsize = (10, 8))
        end
    end
    colgap!(fig.layout, 16)
    save(joinpath(OUT, "fig12_simulated_cohort.png"), fig; px_per_unit = 4)
    save(joinpath(OUT, "fig12_simulated_cohort.svg"), fig)
    return fig
end

# ===========================================================================
# Figure 1 — E2: what a gradient costs, and what the observed information costs
# ===========================================================================

function figure_e2()
    lad = read_csv(joinpath(E2, "e2_ladder.csv"))
    sum_ = read_csv(joinpath(E2, "e2_summary.csv"))
    lad === nothing && return @warn "E2 ladder CSV missing; skipping figure 1"

    raw, hdr = lad
    ran = Bool.(col(raw, hdr, "ran"))
    p = Float64.(col(raw, hdr, "p"))[ran]
    meth = String.(col(raw, hdr, "method"))[ran]
    secs = Float64.(col(raw, hdr, "seconds_min"))[ran]
    arms = ["objective", "reverse_laplace", "forward_laplace"]
    missing_arms = setdiff(arms, unique(meth))
    isempty(missing_arms) || error("E2 CSV is missing requested arms: $(join(missing_arms, ", "))")
    # Read the conditions off the artefact rather than restating them in the title. A caption
    # that has drifted from the data it labels is worse than no caption.
    nsub = Int(col(raw, hdr, "n_subjects")[1])
    nthr = Int(col(raw, hdr, "nthreads")[1])
    expo = Dict{String,Float64}()
    if sum_ !== nothing
        sr, sh = sum_
        for i in axes(sr, 1)
            expo[String(col(sr, sh, "method")[i])] = Float64(col(sr, sh, "normalised_exponent_in_p")[i])
        end
    end

    tobj = Dict(p[i] => secs[i] for i in eachindex(p) if meth[i] == "objective")

    LAB = Dict("objective" => "OFV cost",
               "reverse_laplace" => "Reverse Laplace",
               "forward_laplace" => "Forward Laplace")

    W_PT = 6.2 * 72
    fig = Figure(size = (W_PT, 0.537 * W_PT), fontsize = 12, figure_padding = 5)

    ax1 = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "p (population parameters)",
               ylabel = "seconds per evaluation", title = "(a) wall clock", titlesize = 12,
               xlabelsize = 12, ylabelsize = 12, xticklabelsize = 12, yticklabelsize = 12)
    ax2 = Axis(fig[1, 2]; xscale = log10, yscale = log10, xlabel = "p (population parameters)",
               ylabel = "gradient / one OFV cost", title = "(b) normalised", titlesize = 12,
               xlabelsize = 12, ylabelsize = 12, xticklabelsize = 12, yticklabelsize = 12)
    # There is deliberately no observed-information panel. Building ∇²_θ L costs 2p gradients,
    # so its curve is 2p × panel (a) — a constant factor that shifts every arm identically on a
    # log axis and shows nothing panel (a) does not. The numbers live in
    # `e2_information_cost.csv`, where they can be read exactly rather than eyeballed off a
    # redundant third panel.

    for a in arms
        idx = findall(==(a), meth)
        isempty(idx) && continue
        o = sortperm(p[idx]); idx = idx[o]
        c = get(COL, a, :black)
        lines!(ax1, p[idx], secs[idx]; color = c, linewidth = 2)
        scatter!(ax1, p[idx], secs[idx]; color = c, markersize = 9)
        a == "objective" && continue
        norm = [secs[i] / tobj[p[i]] for i in idx]
        lines!(ax2, p[idx], norm; color = c, linewidth = 2, label = get(LAB, a, a))
        scatter!(ax2, p[idx], norm; color = c, markersize = 9)
    end

    # A ∝ p reference slope, anchored ON the forward arm's first point so the eye compares
    # SLOPE rather than intercept, and a flat reference at the reverse arm's first point.
    px = sort(unique(p))
    ffirst = get(tobj, px[1], 1.0)
    y0f = secs[findfirst(i -> meth[i] == "forward_laplace" && p[i] == px[1], eachindex(p))] / ffirst
    lines!(ax2, px, y0f .* px ./ px[1]; color = (:black, 0.30), linestyle = :dash)
    text!(ax2, px[2], 1.6 * y0f * px[2] / px[1]; text = "slope = 1", align = (:center, :bottom),
          color = (:black, 0.6), fontsize = 11)

    # The exponent IS the claim, so put it on the figure next to each arm.
    # Label each gradient arm with its exponent at its right-hand end.
    for a in arms
        a == "objective" && continue
        haskey(expo, a) || continue
        idx = findall(i -> meth[i] == a, eachindex(meth))
        j = idx[argmax(p[idx])]
        text!(ax2, p[j], secs[j] / tobj[p[j]];
              text = @sprintf("  p^%.2f", expo[a]), align = (:left, :center),
              color = get(COL, a, :black), fontsize = 10)
    end

    for ax in (ax1, ax2)
        ax.xticks = (px, string.(Int.(px)))
    end
    # Explicit ticks on the claim panel: the auto-chooser lands on half-decades here, which is
    # harder to read than the multiples anyone would actually quote.
    yt = [1, 3, 10, 30, 100, 300, 1000]
    ax2.yticks = (Float64.(yt), string.(yt))
    xlims!(ax2, px[1] / 1.30, px[end] * 5.0)

    lelems = [LineElement(color = get(COL, a, :black), linewidth = 2) for a in arms]
    Legend(fig[2, 1:2], lelems, [LAB[a] for a in arms];
           framevisible = false, labelsize = 11, nbanks = 3,
           tellheight = true, tellwidth = false, patchsize = (14, 10), colgap = 8,
           rowgap = -1, padding = (0, 0, 0, 0))
    Label(fig[0, :], "Exact-Laplace computation cost"; fontsize = 12, font = :bold, halign = :center)
    rowsize!(fig.layout, 1, Relative(0.663))   # panels; a further 15% shorter
    rowgap!(fig.layout, 3)
    save(joinpath(OUT, "fig2_e2_cost.png"), fig; px_per_unit = 4)
    save(joinpath(OUT, "fig2_e2_cost.svg"), fig)
    return fig
end

# ===========================================================================
# Figure 3 — E11: exact-Laplace inference, penalised and unpenalised
# ===========================================================================

function figure_e11_inference()
    fu = read_csv(joinpath(E11, "e11_functionals.csv"))
    cv = read_csv(joinpath(E11, "e11_coverage.csv"))
    cu = read_csv(joinpath(E11, "e11_curves.csv"))
    fu === nothing && return
    raw, hdr = fu; c = n -> col(raw, hdr, n)
    lams = sort(unique(_num.(c("lambda"))))
    funcs = unique(String.(vec(c("functional"))))
    CL = Dict(0.0 => COL["forward_laplace"], 1.0 => COL["reverse_laplace"])

    cov_of = (f, λ) -> begin
        cv === nothing && return ""
        vraw, vhdr = cv
        k = findfirst(z -> String(col(vraw, vhdr, "functional")[z]) == f &&
                           _num(col(vraw, vhdr, "lambda")[z]) == λ, axes(vraw, 1))
        k === nothing ? "" : @sprintf("%d/%d", Int(_num(col(vraw, vhdr, "k_covered")[k])),
                                      Int(_num(col(vraw, vhdr, "n")[k])))
    end

    # 6.2 inches wide at 72 points per inch; px_per_unit = 4 puts the PNG near 290 dpi.
    W_PT = 6.2 * 72
    fig = Figure(size = (W_PT, 0.44 * W_PT), fontsize = 12, figure_padding = 5)

    # (a) the fitted curves, as in the misspecification figure
    ax2 = Axis(fig[1, 1]; title = "(a) recovered effect curve", titlesize = 12,
               xlabel = "concentration C (mg/L)", ylabel = "E(C)",
               xlabelsize = 12, ylabelsize = 12, xticklabelsize = 12, yticklabelsize = 12)
    if cu !== nothing
        uraw, uhdr = cu; u = n -> col(uraw, uhdr, n)
        for λ in lams
            rows = [k for k in axes(uraw, 1) if String(u("source")[k]) == "simulated" &&
                    _num(u("lambda")[k]) == λ]
            for r in unique(Int.(_num.(u("cohort")[rows])))
                idx = [k for k in rows if Int(_num(u("cohort")[k])) == r]
                lines!(ax2, _num.(u("C")[idx]), _num.(u("E")[idx]);
                       color = (CL[λ], 0.5), linewidth = 0.9)
            end
        end
        ti = [k for k in axes(uraw, 1) if String(u("source")[k]) == "simulation_truth"]
        lines!(ax2, _num.(u("C")[ti]), _num.(u("E")[ti]); color = :black, linewidth = 2.0,
               linestyle = :dash)
        # Lower right of panel (a) is empty — the curves run to the upper right — so the key
        # sits there rather than costing a row beneath both panels.
        lelems = vcat([LineElement(color = CL[λ], linewidth = 2) for λ in lams],
                      [LineElement(color = :black, linewidth = 2, linestyle = :dash)])
        axislegend(ax2, lelems, vcat([@sprintf("λ = %g", λ) for λ in lams], ["truth"]);
                   position = :rb, framevisible = false, labelsize = 12,
                   patchsize = (16, 10), rowgap = -2)
    end

    # (b) every cohort's interval, the two arms in separate non-overlapping bands
    ARM, BAND = 0.18, 0.24
    ax1 = Axis(fig[1, 2]; title = "(b) estimates and 95% CI", titlesize = 12,
               xlabel = "estimate / truth", xlabelsize = 11,
               xticklabelsize = 12, yticklabelsize = 12,
               yticks = (1:length(funcs), disp.(reverse(funcs))))
    vlines!(ax1, [1.0]; color = (:black, 0.45), linestyle = :dash)
    xhi = 1.0
    for (fi, f) in enumerate(reverse(funcs))
        for (li, λ) in enumerate(lams)
            idx = [k for k in axes(raw, 1) if String(c("functional")[k]) == f &&
                   _num(c("lambda")[k]) == λ]
            isempty(idx) && continue
            tv = _num(c("truth")[idx[1]])
            off = (li == 1 ? -ARM : ARM)
            n = length(idx)
            for (j, k) in enumerate(idx)
                y = fi + off + BAND * ((j - 0.5) / n - 0.5)
                lo, hi = _num(c("lo95")[k]) / tv, _num(c("hi95")[k]) / tv
                xhi = max(xhi, hi)
                lines!(ax1, [lo, hi], [y, y]; color = (CL[λ], 0.6), linewidth = 0.7)
                scatter!(ax1, [_num(c("estimate")[k]) / tv], [y]; color = CL[λ], markersize = 2.2)
            end
        end
    end
    # Coverage beside the band it belongs to, in that band's colour.
    for (fi, f) in enumerate(reverse(funcs)), (li, λ) in enumerate(lams)
        txt = cov_of(f, λ)
        isempty(txt) && continue
        text!(ax1, xhi * 1.02, fi + (li == 1 ? -ARM : ARM); text = txt, fontsize = 10,
              align = (:left, :center), color = CL[λ])
    end
    xlims!(ax1, nothing, xhi * 1.20)

    colgap!(fig.layout, 14)
    save(joinpath(OUT, "fig3_e11_inference.png"), fig; px_per_unit = 4)
    save(joinpath(OUT, "fig3_e11_inference.svg"), fig)
    return fig
end

# ===========================================================================
# Tables
# ===========================================================================

# ===========================================================================
# Figure 4 — E9: what flexibility buys under exact-Laplace misspecification
# ===========================================================================

function figure_e9()
    # The misspecification results use the ridge strength established by Figure 3. The legend
    # takes gamma from the artifact so it cannot drift from the fitted models.
    cur = read_csv(joinpath(E9, "e9b_effect_curves.csv"))
    cur === nothing && return
    craw, chdr = cur; cc = n -> col(craw, chdr, n)

    all_t = unique(String.(vec(cc("truth"))))
    truths = vcat(["Emax"], sort([t for t in all_t if startswith(t, "sigmoid")]))
    # The truth is a Hill curve in every panel; only its exponent n differs, so the title
    # carries n alone and the caption carries the rest. γ is the ridge strength, not this.
    trim = x -> (v = parse(Float64, x); v == round(v) ? string(Int(v)) : string(v))
    label = Dict("Emax" => "n = 1 (Emax)")
    for t in truths
        startswith(t, "sigmoid") && (label[t] = "n = " * trim(split(t, "_g")[end]))
    end
    # γ is the ridge strength the fits were run at; carried in the artefact when the run
    # recorded it, so the legend cannot drift from the fits it describes.
    lam = "lambda" in String.(vec(chdr)) ? _num(cc("lambda")[1]) : 0.0
    nnlab = m -> m * ", γ = " * (lam == round(lam) ? string(Int(round(lam))) : string(lam))
    nnname = first(m for m in unique(String.(vec(cc("fit_model")))) if startswith(m, "M_nn"))
    MCOL = Dict("M_par" => COL["forward_laplace"], nnname => COL["reverse_laplace"])

    W_PT = 6.2 * 72
    fig = Figure(size = (W_PT, 0.42 * W_PT), fontsize = 12, figure_padding = 5)

    for (j, t) in enumerate(truths)
        ax = Axis(fig[1, j], title = "($(('a':'z')[j])) " * label[t], titlesize = 12,
                  xlabel = "C (mg/L)", ylabel = j == 1 ? "E(C)" : "",
                  xlabelsize = 12, ylabelsize = 12,
                  xticklabelsize = 12, yticklabelsize = 12)
        j > 1 && hideydecorations!(ax, grid = false)
        for m in (nnname, "M_par")
            rows = [k for k in axes(craw, 1) if String(cc("truth")[k]) == t &&
                    String(cc("fit_model")[k]) == m]
            for r in unique(Int.(_num.(cc("cohort")[rows])))
                idx = [k for k in rows if Int(_num(cc("cohort")[k])) == r]
                lines!(ax, _num.(cc("C")[idx]), _num.(cc("E")[idx]);
                       color = (MCOL[m], 0.30), linewidth = 1.1)
            end
        end
        ti = [k for k in axes(craw, 1) if String(cc("truth")[k]) == t &&
              String(cc("fit_model")[k]) == "truth"]
        lines!(ax, _num.(cc("C")[ti]), _num.(cc("E")[ti]);
               color = :black, linewidth = 2.6, linestyle = :dash)
        ylims!(ax, -0.4, 13.5)
        if j == 2
            # Coloured labels rather than swatch-plus-text: an axislegend keeps its patches in a
            # fixed left column, so right-aligning the text strands them over the curves. Here
            # every entry ends flush at the panel edge and the colour is carried by the word.
            for (k, (txt, cl)) in enumerate((("truth", :black), ("M_par", MCOL["M_par"]),
                                             (nnlab(nnname), MCOL[nnname])))
                text!(ax, 0.975, 0.30 - 0.085 * (k - 1); text = txt, color = cl,
                      fontsize = 10.2, align = (:right, :center), space = :relative)
            end
        end
    end

    colgap!(fig.layout, 8)
    save(joinpath(OUT, "fig4_e9_misspecification.png"), fig; px_per_unit = 4)
    save(joinpath(OUT, "fig4_e9_misspecification.svg"), fig)
    return fig
end

# ===========================================================================
# Figure 5 — E15: the real 32-subject data refitted across ridge strengths
# ===========================================================================

function figure_e11(; gamma_tags = ("0", "1", "10"),
                    output_stem = "fig5_e11_realdata", band_gamma = 1.0)
    # E15: five network fits per penalty strength on the real data, each from a different
    # initialisation seed, with a delta-method standard error for E(C) at every grid point.
    parts = [read_csv(joinpath(E11, "e15_curves_g$(g).csv")) for g in gamma_tags]
    fitp = [read_csv(joinpath(E11, "e15_fits_g$(g).csv")) for g in gamma_tags]
    all(x -> x !== nothing, parts) ||
        error("figure_e11: missing e15_curves_g{$(join(gamma_tags, ','))}.csv in $(E11); run e15_realdata_replicates.jl.")

    # One long table, so every lookup below is a filter rather than three special cases.
    rows = NamedTuple[]
    for pr in parts
        raw, hdr = pr; c = n -> col(raw, hdr, n)
        for k in axes(raw, 1)
            push!(rows, (model = String(c("model")[k]), gamma = _num(c("gamma")[k]),
                         seed = Int(_num(c("seed")[k])), C = _num(c("C")[k]),
                         E = _num(c("E")[k]), lo = _num(c("lo95")[k]), hi = _num(c("hi95")[k])))
        end
    end
    ofv_of = (g, sd) -> begin
        for fp in fitp
            fp === nothing && continue
            raw, hdr = fp; c = n -> col(raw, hdr, n)
            for k in axes(raw, 1)
                _num(c("gamma")[k]) == g && Int(_num(c("seed")[k])) == sd &&
                    startswith(String(c("model")[k]), "M_nn") && return _num(c("OFV")[k])
            end
        end
        NaN
    end

    gammas = sort(unique(r.gamma for r in rows if startswith(r.model, "M_nn")))
    # Keep the Okabe--Ito visual system without repeating Figures 2--4's green/purple pair:
    # warm marks the unregularized fit, while the regularized fits share a blue hue family.
    pal = Dict(0.0 => COL["forward_foce"], 1.0 => COL["reverse_foce"],
               3.0 => COL["foce_fd"], 10.0 => COL["reverse_foce_directional"])
    all(haskey(pal, g) for g in gammas) || error("figure_e11: no colour assigned to one of $(gammas)")
    nnname = first(r.model for r in rows if startswith(r.model, "M_nn"))
    BAND_SEED = 1                                   # the fit panel (b) bands

    W_PT = 6.2 * 72
    fig = Figure(size = (W_PT, 0.44 * W_PT), fontsize = 12, figure_padding = 5)

    axa = Axis(fig[1, 1], title = "(a) fit-to-fit variability", titlesize = 12,
               xlabel = "concentration C (mg/L)", ylabel = "E(C)",
               xlabelsize = 12, ylabelsize = 12, xticklabelsize = 12, yticklabelsize = 12)
    for g in gammas, sd in sort(unique(r.seed for r in rows if r.gamma == g && startswith(r.model, "M_nn")))
        sel = [r for r in rows if r.gamma == g && r.seed == sd && startswith(r.model, "M_nn")]
        isempty(sel) && continue
        lines!(axa, [r.C for r in sel], [r.E for r in sel];
               color = (pal[g], 0.75), linewidth = 1.3)
    end
    par = [r for r in rows if r.model == "M_par"]
    isempty(par) || lines!(axa, [r.C for r in par], [r.E for r in par];
                           color = :black, linewidth = 2.0, linestyle = :dash)
    elems = vcat([LineElement(color = pal[g], linewidth = 2) for g in gammas],
                 [LineElement(color = :black, linewidth = 2, linestyle = :dash)])
    labs = vcat([@sprintf("γ = %g", g) for g in gammas], ["M_par"])
    axislegend(axa, elems, labs; position = :lt, framevisible = false, labelsize = 10,
               rowgap = -3, patchsize = (14, 8), patchlabelgap = 4)

    # Panel b: one penalised fit, with the interval the delta method actually gives for E(C).
    gb = band_gamma in gammas ? band_gamma : gammas[1]
    band_seeds = sort(unique(r.seed for r in rows if r.gamma == gb && startswith(r.model, "M_nn")))
    finite_band_seeds = [sd for sd in band_seeds if any(r.seed == sd && isfinite(r.lo) && isfinite(r.hi)
                                                        for r in rows if r.gamma == gb && startswith(r.model, "M_nn"))]
    selected_seed = isempty(finite_band_seeds) ? BAND_SEED : first(finite_band_seeds)
    sel = [r for r in rows if r.gamma == gb && r.seed == selected_seed && startswith(r.model, "M_nn")]
    has_band = !isempty(sel) && all(r -> isfinite(r.lo) && isfinite(r.hi), sel)
    panel_title = has_band ? @sprintf("(b) one γ = %g fit, 95%% interval", gb) :
                             @sprintf("(b) one γ = %g fit", gb)
    axb = Axis(fig[1, 2], title = panel_title, titlesize = 12,
               xlabel = "concentration C (mg/L)", ylabel = "",
               xlabelsize = 12, ylabelsize = 12, xticklabelsize = 12, yticklabelsize = 12)
    if !isempty(sel)
        Cs = [r.C for r in sel]
        has_band && band!(axb, Cs, [r.lo for r in sel], [r.hi for r in sel]; color = (pal[gb], 0.22))
        lines!(axb, Cs, [r.E for r in sel]; color = pal[gb], linewidth = 2.0)
        isempty(par) || lines!(axb, [r.C for r in par], [r.E for r in par];
                               color = :black, linewidth = 2.0, linestyle = :dash)
        eb = has_band ? [LineElement(color = pal[gb], linewidth = 2),
                         PolyElement(color = (pal[gb], 0.22)),
                         LineElement(color = :black, linewidth = 2, linestyle = :dash)] :
                        [LineElement(color = pal[gb], linewidth = 2),
                         LineElement(color = :black, linewidth = 2, linestyle = :dash)]
        labels = has_band ? [@sprintf("γ = %g fit", gb), "95% interval", "M_par"] :
                            [@sprintf("γ = %g fit", gb), "M_par"]
        axislegend(axb, eb, labels;
                   position = :lt, framevisible = false, labelsize = 10, rowgap = -3,
                   patchsize = (14, 8), patchlabelgap = 4)
        has_band || text!(axb, 0.97, 0.04; text = "95% interval unavailable",
                          space = :relative, align = (:right, :bottom),
                          color = :gray40, fontsize = 10)
    end
    linkyaxes!(axa, axb)
    hideydecorations!(axb, grid = false)
    colgap!(fig.layout, 8)
    save(joinpath(OUT, "$(output_stem).png"), fig; px_per_unit = 4)
    save(joinpath(OUT, "$(output_stem).svg"), fig)
    return fig
end

figure_e11_gamma3() = figure_e11(gamma_tags = ("0", "1", "3", "10"),
                                 output_stem = "fig5b_e11_realdata_gamma3",
                                 band_gamma = 3.0)

# ===========================================================================

if isempty(ARGS)
    figure_cohort()
    figure_e2()
    figure_e11_inference()
    figure_e9()
    figure_e11()
elseif ARGS == ["e2"]
    figure_e2()
elseif ARGS == ["e11"]
    figure_e11_inference()
elseif ARGS == ["e9"]
    figure_e9()
elseif ARGS == ["e15"]
    figure_e11()
elseif ARGS == ["e15b"]
    figure_e11_gamma3()
elseif ARGS == ["cohort"]
    figure_cohort()
else
    error("usage: make_figures.jl [cohort|e2|e11|e9|e15|e15b]")
end

println("figures  -> ", OUT)
foreach(f -> println("  ", f), sort(readdir(OUT)))
