# Mooncake backend.  Loaded by `backends.jl`, never a dependency of NLMEReverse.
#
# `build_rrule` is the expensive part, and it depends only on the TYPES involved, not on
# the captured values — every subject's closure has the same type — so the rule is built
# once and reused.  It is cached PER THREAD: `cfg.threaded` is on by default, so
# `reverse_sweep` runs concurrently from many tasks and `value_and_gradient!!` mutates the
# rule's internal storage.  One rule shared across threads is a data race that produces
# wrong gradients rather than an error, which is the worst kind.

using Mooncake
import NLMEReverse: reverse_sweep

const _MOONCAKE_RULES = Dict{Any,Vector{Any}}()
const _MOONCAKE_LOCK = ReentrantLock()

# A POOL, not a per-thread slot.  `lock` is a yield point and `@threads :greedy` tasks
# migrate across it, so keying the cache by `threadid()` does not guarantee exclusive
# use — two tasks could share one rule and race its mutable tape, giving wrong gradients
# with no error.  Check a rule out for the duration of the sweep and check it back in.
function reverse_sweep(::Val{:mooncake}, s::F, θ::Vector{Float64}) where {F}
    key = (F, typeof(θ))
    rule = lock(_MOONCAKE_LOCK) do
        pool = get!(() -> Any[], _MOONCAKE_RULES, key)
        isempty(pool) ? Mooncake.build_rrule(s, θ) : pop!(pool)
    end
    _, grads = Mooncake.value_and_gradient!!(rule, s, θ)
    lock(_MOONCAKE_LOCK) do
        push!(_MOONCAKE_RULES[key], rule)
    end
    return collect(Float64, grads[2])
end
