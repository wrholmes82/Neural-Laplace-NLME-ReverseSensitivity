# Two validation gradients. Neither is a production method: they exist so the
# adjoint gradient has something to be judged against.

"""
    ad_gradient(model, pop, cfg) -> grad!(G, θ)

**AD-through**: `ForwardDiff.gradient` over the whole population objective, inner
Newton loop included.

This is the exact gradient of the *implemented truncated procedure*, not of the
ideal Laplace objective, and it converges to the adjoint gradient as `inner_tol`
tightens. That distinction is the subject of the convergence-order test.

Semantically this is the unrolled gradient of the reference manuscript. It is not
commensurable with that unrolled gradient in *cost*: the manuscript's version uses
reverse mode, with memory scaling in the inner iteration cap; this is forward
mode, with cost scaling in `p` and no memory growth. Accuracy comparisons carry
over; timings do not.
"""
function ad_gradient(model, pop::PopulationData, cfg::LaplaceConfig)
    f = laplace_objective(model, pop, cfg)
    return function (G, θ)
        G .= ForwardDiff.gradient(f, θ)
        return G
    end
end

"""
    fd_gradient(model, pop, cfg; step=1e-4) -> grad!(G, θ)

**Central finite differences** on the outer parameters, every perturbed
evaluation cold-starting every inner solve. Shares no code path with either AD
method, so agreement with it is genuinely independent evidence.

Set `cfg.inner_deriv = :fd` to reproduce the reference manuscript's FD baseline,
in which the inner Newton solver's derivatives are finite differences too and
*both* layers inherit step-size noise. With `:ad` inner derivatives this is a
different — and unfairly favourable — method.

The `2p` perturbation loop is deliberately serial, each perturbation calling the
threaded population objective in turn. Threading it as well would nest parallel
regions and oversubscribe.
"""
function fd_gradient(model, pop::PopulationData, cfg::LaplaceConfig; step::Float64 = 1e-4)
    f = laplace_objective(model, pop, cfg)
    return function (G, θ)
        for j in eachindex(θ)
            s = step * max(abs(θ[j]), 1.0)
            θp = collect(Float64, θ); θp[j] += s
            θm = collect(Float64, θ); θm[j] -= s
            G[j] = (f(θp) - f(θm)) / (2s)
        end
        return G
    end
end

"""
    gradient_factory(method, model, pop, cfg; kwargs...)

Dispatch on `:adjoint`, `:ad` or `:fd`. All three return the same
`grad!(G, θ)` closure signature, which is what lets [`fit_nlme`](@ref) be
method-agnostic.
"""
function gradient_factory(method::Symbol, model, pop::PopulationData,
                          cfg::LaplaceConfig; fd_step::Float64 = 1e-4)
    method === :adjoint && return adjoint_gradient(model, pop, cfg)
    method === :ad      && return ad_gradient(model, pop, cfg)
    method === :fd      && return fd_gradient(model, pop, cfg; step = fd_step)
    error("unknown gradient method $method; expected :adjoint, :ad or :fd")
end
