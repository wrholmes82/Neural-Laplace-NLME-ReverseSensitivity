# ReverseDiff backend.  Loaded by `backends.jl`, never a dependency of NLMEReverse.
#
# Tape-based and the most permissive of the three on ordinary code, but it is the one
# expected to fail HERE, and the reason is worth writing down.  This construction puts
# `ForwardDiff.Dual` OUTSIDE `ReverseDiff.TrackedReal` (θ is tracked; η carries the
# Duals), which is the opposite of ReverseDiff's usual nesting.  `+(::Dual, ::TrackedReal)`
# matches methods from both packages and both define a `promote_rule`, so this is a
# classic ambiguity site.  If it errors or disagrees, that is expected; `select_backend`
# drops it and the study proceeds on another engine.
#
# The tape is NOT compiled or cached.  The scalar closes over a different η̂, λ and U for
# every subject, so a cached tape recorded at one subject's frozen quantities and reused
# at another's would be a wrong gradient with no symptom.  Taping per call is the correct,
# if slower, choice.

using ReverseDiff
import NLMEReverse: reverse_sweep

reverse_sweep(::Val{:reversediff}, s::F, θ::Vector{Float64}) where {F} =
    ReverseDiff.gradient(s, θ)
