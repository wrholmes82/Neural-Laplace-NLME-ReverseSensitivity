# Reverse-mode engine loader.
#
#   include(joinpath(REPO_ROOT, "src", "backends", "backends.jl"))
#   loaded = load_backends!()
#
# One file per engine, each adding a method to `NLMEReverse.reverse_sweep`. None of them
# is a dependency of NLMEReverse, so an engine that is not installed — or that fails to
# precompile, which happens to Enzyme on new Julia releases more often than anyone would
# like — cannot stop the module loading or the study running.
#
# A failure here is REPORTED, not swallowed and not fatal. Which engines work on a given
# machine at given package versions is a fact about the environment, and the study's job
# is to record it, pick a working one, and say which it picked.
#
# Loading an engine is not the same as it working. `NLMEReverse.select_backend` runs each
# loaded engine against the already-validated adjoint gradient and keeps only those that
# reproduce it; that is the gate, not this file.

using NLMEReverse

const _BACKEND_DIR = @__DIR__

"""
    load_backends!(; which = NLMEReverse.CANDIDATE_BACKENDS, verbose = true)

Try to load each reverse-mode engine. Returns the vector of engines that are now
callable. Never throws.
"""
function load_backends!(; which = NLMEReverse.CANDIDATE_BACKENDS, verbose::Bool = true)
    for b in which
        NLMEReverse.backend_available(b) && continue
        file = joinpath(_BACKEND_DIR, "backend_$(b).jl")
        if !isfile(file)
            verbose && @warn "backend file missing" backend = b file
            continue
        end
        try
            Base.include(Main, file)
            verbose && @info "reverse backend loaded" backend = b
        catch err
            verbose && @warn("reverse backend NOT loaded — it is excluded from the " *
                             "study and this is not fatal",
                             backend = b,
                             reason = first(sprint(showerror, err; context = :limit => true), 400))
        end
    end
    loaded = [b for b in which if NLMEReverse.backend_available(b)]
    if isempty(loaded)
        verbose && @warn "no reverse-mode engine is available. Everything still runs " *
                         "with the :forwarddiff CONTROL ARM, but every arm then has the " *
                         "same cost model and the study cannot show a reverse effect."
    end
    return loaded
end
