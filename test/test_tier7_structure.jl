# Tier 7 -- structure and portability.
#
# These checks are mechanical: enumerate the filesystem and parse source text,
# never a hard-coded list of filenames. The point is to catch a file added
# later that never gets wired in -- exactly the failure mode that leaves dead
# code and untrue documentation behind a refactor.
#
# What is under test is the claim the repository makes about itself: `Project.toml`
# plus `src/` is a self-contained, transportable NLME package that references
# nothing else in this repository -- not `test/`, not `joint_covariate_study/`,
# not `Local_Env`. Everything problem-specific lives outside it.
#
# A crash here is a failure, same as everywhere else in the suite: nothing
# below wraps an assertion in try/catch to force it green.

using TOML

const TIER7_SRC_DIR = joinpath(ROOT_DIR, "src")
const TIER7_PROJECT = joinpath(ROOT_DIR, "Project.toml")

# The reverse-mode engine loaders under src/backends/ are deliberately outside every
# check below. They are not part of the module's include closure -- they are `include`d
# into `Main` at run time by whoever wants an engine -- and they `using` packages that
# are deliberately NOT declared dependencies. That is the design: an engine which will
# not precompile on a given machine must not be able to stop the package loading.
const TIER7_BACKEND_DIR = joinpath(TIER7_SRC_DIR, "backends")

"""`.jl` files that belong to the module proper, i.e. excluding the engine loaders."""
tier7_module_files(dir) = Set(f for f in tier7_all_jl_files(dir)
                              if !startswith(f, TIER7_BACKEND_DIR))

"""The transportable package -- `Project.toml` + `src/`, and nothing else in the repository."""
function tier7_copy_package(dest::AbstractString)
    mkpath(dest)
    cp(TIER7_SRC_DIR, joinpath(dest, "src"))
    cp(TIER7_PROJECT, joinpath(dest, "Project.toml"))
    return dest
end

# ---------------------------------------------------------------------------
# Shared text-parsing helpers. All operate on comment-stripped text so that a
# documentation example inside a comment (e.g. NLMEReverse.jl's own docstring
# showing `include(".../src/NLMEReverse.jl")`) can never masquerade as a real
# include or a real outside reference.
# ---------------------------------------------------------------------------

"""Return `text` with everything from the first unescaped `#` on each line removed."""
function tier7_strip_comments(text::AbstractString)
    return join((first(split(line, "#"; limit = 2)) for line in split(text, '\n')), '\n')
end

"""Parse literal `include("....jl")` targets out of (comment-stripped) source text."""
function tier7_parse_include_targets(text::AbstractString)
    stripped = tier7_strip_comments(text)
    return String[String(m.captures[1]) for m in eachmatch(r"include\(\s*\"([^\"]+)\"\s*\)", stripped)]
end

"""
Transitive closure of `include(...)` targets reachable from `entry_path`,
following relative includes (which resolve relative to the including file,
exactly as Julia resolves them -- this is what makes Mode A work at all).
Returns a Set of normalized absolute paths, including `entry_path` itself.
"""
function tier7_reachable_includes(entry_path::AbstractString)
    visited = Set{String}()
    queue = [normpath(entry_path)]
    while !isempty(queue)
        f = pop!(queue)
        f in visited && continue
        push!(visited, f)
        isfile(f) || continue
        text = read(f, String)
        for inc in tier7_parse_include_targets(text)
            target = normpath(joinpath(dirname(f), inc))
            target in visited || push!(queue, target)
        end
    end
    return visited
end

"""All `.jl` files under `dir`, recursively, as normalized absolute paths."""
function tier7_all_jl_files(dir::AbstractString)
    files = Set{String}()
    for (root, _, fnames) in walkdir(dir)
        for f in fnames
            endswith(f, ".jl") && push!(files, normpath(joinpath(root, f)))
        end
    end
    return files
end

"""
Parse module names introduced by `using`/`import` statements in (comment-stripped)
source text. Handles `using A, B`, `import A: x, y`, `import A.B: x`, and
`using A: x, y` -- only the module name(s) before any `:` are kept; symbols
named after a colon are not modules.
"""
function tier7_parse_used_modules(text::AbstractString)
    mods = Set{String}()
    for line in split(tier7_strip_comments(text), '\n')
        s = strip(line)
        m = match(r"^(using|import)\s+(.+)$", s)
        m === nothing && continue
        modpart = first(split(m.captures[2], ':'))
        for item in split(modpart, ',')
            name = strip(item)
            isempty(name) && continue
            name = strip(first(split(name, r"\s+as\s+")))
            name = strip(first(split(name, '.')))
            isempty(name) || push!(mods, name)
        end
    end
    return mods
end

"""Count non-blank, non-comment lines in one `.jl` file."""
function tier7_count_code_lines(path::AbstractString)
    n = 0
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        startswith(s, "#") && continue
        n += 1
    end
    return n
end

"""Run `cmd` to completion, capturing stdout/stderr, without throwing on a nonzero exit."""
function tier7_run_process(cmd::Cmd)
    out, err = IOBuffer(), IOBuffer()
    proc = run(pipeline(cmd; stdout = out, stderr = err); wait = false)
    wait(proc)
    return (success = (proc.exitcode == 0), stdout = String(take!(out)), stderr = String(take!(err)))
end

"""Read the three-line result file a portability harness script writes."""
function tier7_read_portability_result(path::AbstractString)
    lines = readlines(path)
    L = parse(Float64, lines[1])
    G = parse.(Float64, split(lines[2], ","))
    fd = parse.(Float64, split(lines[3], ","))
    return (L = L, G = G, fd = fd)
end

# ---------------------------------------------------------------------------
# Portability harness generation. The toy model is deliberately exactly
# quadratic in eta (linear-Gaussian regime, Prop. 1 of the plan): it references
# nothing from problems/, so it proves the *copy* works rather than smuggling
# in a dependency on the rest of the repo.
# ---------------------------------------------------------------------------

const TIER7_TOY_MODEL_BODY = """
struct Tier7ToyModel <: NLMEModel
    q::Int
end

function negative_log_joint_density(m::Tier7ToyModel, eta, theta, subj)
    T = promote_type(eltype(eta), eltype(theta))
    beta = theta[1]
    sigma = exp(theta[2])
    h = zero(T)
    for j in eachindex(subj.times)
        pred = beta + eta[1]
        r = subj.observations[j] - pred
        h += log(2*pi)/2 + log(sigma) + r*r/(2*sigma*sigma)
    end
    h += log(2*pi)/2 + eta[1]*eta[1]/2
    return h
end

times = [1.0, 2.0, 3.0, 4.0]
subj1 = SubjectData(1, times, [1.05, 0.95, 1.10, 1.02], 0.0)
subj2 = SubjectData(2, times, [0.90, 1.15, 1.00, 0.85], 0.0)
pop   = PopulationData([subj1, subj2])
model = Tier7ToyModel(1)
cfg   = LaplaceConfig()

theta0 = [1.0, log(0.2)]

Lfun = laplace_objective(model, pop, cfg)
Lval = Lfun(theta0)

g! = adjoint_gradient(model, pop, cfg)
G  = zeros(length(theta0))
g!(G, theta0)

fdgrad = NLMEReverse.ForwardDiff.gradient(Lfun, theta0)
"""

function tier7_portability_script(mode_load_code::AbstractString, out_file::AbstractString)
    return """
    $mode_load_code

    $TIER7_TOY_MODEL_BODY

    open($(repr(out_file)), "w") do io
        println(io, Lval)
        println(io, join(G, ","))
        println(io, join(fdgrad, ","))
    end
    println("TIER7_PORTABILITY_DONE")
    """
end

tier7_mode_a_load(tmp_src::AbstractString) = """
    include($(repr(joinpath(tmp_src, "src", "NLMEReverse.jl"))))
    using .NLMEReverse
    # Dotted form: NLMEReverse here is the *local submodule* just include()'d,
    # not a package. A bare (undotted) `import NLMEReverse: ...` would instead
    # trigger package resolution against the active project -- which, launched
    # under --project=Local_Env for dependency access, would silently reach the
    # *original* (uncopied) package and defeat the entire portability check.
    import .NLMEReverse: negative_log_joint_density
    """

tier7_mode_b_load(tmp_src::AbstractString) = """
    using Pkg
    Pkg.activate(mktempdir())
    Pkg.develop(path=$(repr(tmp_src)))
    Pkg.instantiate()
    using NLMEReverse
    import NLMEReverse: negative_log_joint_density
    """

# ===========================================================================

@testset "Tier 7 -- structure and portability" begin

    # -----------------------------------------------------------------
    # 1. Portability: copy src/ to a fresh temp dir, load it
    #    both ways described in the README, each in its own fresh Julia
    #    process, evaluate one objective and one adjoint gradient, and
    #    check the adjoint gradient against a plain ForwardDiff.gradient
    #    of the objective -- proving the copy works, not merely loads.
    # -----------------------------------------------------------------
    @testset "portability -- both load modes, fresh processes" begin
        tmp_root = mktempdir()
        tmp_src  = tier7_copy_package(joinpath(tmp_root, "NLMEReverse_copy"))

        out_a, out_b = joinpath(tmp_root, "result_A.txt"), joinpath(tmp_root, "result_B.txt")
        script_a, script_b = joinpath(tmp_root, "harness_A.jl"), joinpath(tmp_root, "harness_B.jl")

        write(script_a, tier7_portability_script(tier7_mode_a_load(tmp_src), out_a))
        write(script_b, tier7_portability_script(tier7_mode_b_load(tmp_src), out_b))

        julia_bin  = Base.julia_cmd()
        local_env  = joinpath(ROOT_DIR, "Local_Env")

        # Mode A: a plain `include`, so it only needs the six deps to be
        # resolvable from *some* active project -- exactly what a user
        # dropping the folder into an existing project would have.
        cmd_a = `$julia_bin --startup-file=no --project=$local_env $script_a`
        # Mode B: brings its own throwaway environment via Pkg.develop, so it
        # is launched with no project of ours at all.
        cmd_b = `$julia_bin --startup-file=no $script_b`

        res_a = tier7_run_process(cmd_a)
        if !res_a.success
            println("Tier 7 portability Mode A subprocess FAILED.")
            println("--- stdout ---\n", res_a.stdout)
            println("--- stderr ---\n", res_a.stderr)
        end
        @test res_a.success

        res_b = tier7_run_process(cmd_b)
        if !res_b.success
            println("Tier 7 portability Mode B subprocess FAILED.")
            println("--- stdout ---\n", res_b.stdout)
            println("--- stderr ---\n", res_b.stderr)
        end
        @test res_b.success

        if res_a.success && res_b.success
            ra = tier7_read_portability_result(out_a)
            rb = tier7_read_portability_result(out_b)

            @test isfinite(ra.L)
            @test isfinite(rb.L)
            @test all(isfinite, ra.G)
            @test all(isfinite, rb.G)
            @test all(isfinite, ra.fd)
            @test all(isfinite, rb.fd)

            # Mode A and Mode B must compute the identical objective and gradient.
            @test ra.L ≈ rb.L atol=1e-8 rtol=1e-8
            @test length(ra.G) == length(rb.G)
            for k in eachindex(ra.G)
                @test ra.G[k] ≈ rb.G[k] atol=1e-8 rtol=1e-8
            end

            # Adjoint gradient vs plain ForwardDiff.gradient of the objective,
            # elementwise, in each process independently -- this is the check
            # that the copy actually computes something correct, not just that
            # it loads. The toy model is exactly quadratic in eta (Prop. 1:
            # degenerate case), so the adjoint correction term is identically
            # zero and the two methods should agree tightly.
            for k in eachindex(ra.G)
                @test ra.G[k] ≈ ra.fd[k] atol=1e-6 rtol=1e-6
            end
            for k in eachindex(rb.G)
                @test rb.G[k] ≈ rb.fd[k] atol=1e-6 rtol=1e-6
            end
        end
    end

    # -----------------------------------------------------------------
    # 2. Reachability: every .jl file under src/ is
    #    reachable from NLMEReverse.jl by following include(...) transitively.
    # -----------------------------------------------------------------
    @testset "reachability -- every src file included transitively" begin
        entry = joinpath(TIER7_SRC_DIR, "NLMEReverse.jl")
        @test isfile(entry)

        reachable = tier7_reachable_includes(entry)
        all_files = tier7_module_files(TIER7_SRC_DIR)

        unreachable = setdiff(all_files, reachable)
        if !isempty(unreachable)
            println("Tier 7: unreachable file(s) under src/: ", unreachable)
        end
        @test isempty(unreachable)
    end

    # -----------------------------------------------------------------
    # 3. Test-runner coverage: every test_*.jl in test/ is included by
    #    runtests.jl, and every include target in runtests.jl exists.
    # -----------------------------------------------------------------
    @testset "runner coverage -- every test_*.jl wired into runtests.jl" begin
        runtests_text = read(joinpath(TEST_DIR, "runtests.jl"), String)
        included_targets = Set(tier7_parse_include_targets(runtests_text))

        test_files_on_disk = Set(f for f in readdir(TEST_DIR) if occursin(r"^test_.*\.jl$", f))

        not_included = setdiff(test_files_on_disk, included_targets)
        if !isempty(not_included)
            println("Tier 7: test file(s) present on disk but not included by runtests.jl: ", not_included)
        end
        @test isempty(not_included)

        pointing_at_missing = [t for t in included_targets if !isfile(joinpath(TEST_DIR, t))]
        if !isempty(pointing_at_missing)
            println("Tier 7: runtests.jl include(s) pointing at missing file(s): ", pointing_at_missing)
        end
        @test isempty(pointing_at_missing)
    end

    # -----------------------------------------------------------------
    # 4. Declared dependencies: using/import'd modules under src/ <=> deps
    #    declared in Project.toml, both directions.
    # -----------------------------------------------------------------
    @testset "declared dependencies -- used <=> declared" begin
        project = TOML.parsefile(TIER7_PROJECT)
        declared_deps = Set(keys(project["deps"]))

        used_modules = Set{String}()
        for f in tier7_module_files(TIER7_SRC_DIR)
            union!(used_modules, tier7_parse_used_modules(read(f, String)))
        end

        undeclared = setdiff(used_modules, declared_deps)
        if !isempty(undeclared)
            println("Tier 7: module(s) used in src/ but not declared in Project.toml: ", undeclared)
        end
        @test isempty(undeclared)

        unused = setdiff(declared_deps, used_modules)
        if !isempty(unused)
            println("Tier 7: dependency(ies) declared in Project.toml but never used in src/: ", unused)
        end
        @test isempty(unused)
    end

    # -----------------------------------------------------------------
    # 5. Size tripwire: ~1800 lines is the signal, under 2000 is the assertion.
    # Raised from 900 for the modified-Newton PD search direction
    # (`pd_search_hessian`), then from 1000 for the forward-sensitivity
    # comparator (`gradient_forward.jl`), a deliberate second gradient method,
    # then from 1150 when the reverse construction (`reverse.jl`) and the FOCE
    # curvature arm (`curvature.jl`) were merged in from what used to be three
    # packages, and the like-for-like forward FOCE comparator
    # (`gradient_forward_foce.jl`) was folded in from the study directory.
    #
    # Raised from 2000 to 2100 for `information.jl` (105 lines): the spectral split,
    # the Newton decrement, the truncated inverse, the delta method and the fit-quality
    # gate. These are the `observed_information.jl` / `functionals.jl` additions the
    # plan schedules, and they exist because the observed information on this model
    # class is measurably NOT positive definite -- 61 of 153 eigenvalues negative at
    # the scientific width -- so `inv` and `cholesky` are the wrong tools and a
    # gradient norm is the wrong stopping rule.
    #
    # The threshold is a size SIGNAL, not a correctness bound: it exists to force this
    # justification to be written down, which is why each raise above names what it
    # bought. It is sized to what was added, deliberately without headroom, so that
    # `elbo.jl` trips it again and has to argue its own case.
    # Counted over the module proper; the engine loaders are excluded.
    # -----------------------------------------------------------------
    @testset "size tripwire" begin
        loc = sum(tier7_count_code_lines, tier7_module_files(TIER7_SRC_DIR))
        println("Tier 7: src/ non-blank, non-comment line count = ", loc)
        @test loc < 2100
    end

    # -----------------------------------------------------------------
    # 6. No reference to anything outside src/.
    # -----------------------------------------------------------------
    @testset "no references outside src/" begin
        forbidden_substrings = ["problems/", "examples/", "test/", "Local_Env",
                                "warfarin_turnover"]
        violations = Tuple{String,String}[]

        for f in tier7_module_files(TIER7_SRC_DIR)
            raw = read(f, String)
            stripped = tier7_strip_comments(raw)

            for inc in tier7_parse_include_targets(raw)
                occursin("..", inc) && push!(violations, (f, "include(\"$inc\") escapes its directory"))
            end
            for s in forbidden_substrings
                occursin(s, stripped) && push!(violations, (f, "mentions \"$s\""))
            end
        end

        if !isempty(violations)
            println("Tier 7: outside-reference violation(s): ", violations)
        end
        @test isempty(violations)
    end
end
