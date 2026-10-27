# binomial.jl — an exact binomial interval, shared by every script that reports coverage.
#
# Kept here rather than in one experiment so that a recomputation of the same table cannot end
# up using a second, subtly different implementation.

"""Exact (Clopper-Pearson) binomial interval. Coverage at n = 20 is reported as `k/20` with
this interval and never as a bare percentage — the SE on a coverage near 0.95 is ~5 points, so
a percentage implies a precision that is not there."""
function clopper_pearson(k::Int, n::Int; α = 0.05)
    lo = k == 0 ? 0.0 : 1 - quantile_beta(1 - α / 2, n - k + 1, k)
    hi = k == n ? 1.0 : 1 - quantile_beta(α / 2, n - k, k + 1)
    return (lo, hi)
end
# Beta quantile by bisection on the regularised incomplete beta, computed here so the package
# gains no distribution dependency for one interval.
function quantile_beta(p, a, b)
    lo, hi = 0.0, 1.0
    for _ in 1:200
        mid = (lo + hi) / 2
        (betainc_reg(mid, a, b) < p) ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

function betainc_reg(x, a, b)
    x <= 0 && return 0.0
    x >= 1 && return 1.0
    n = 4000
    h = x / n
    s = 0.0
    for i in 0:n
        t = clamp(i * h, 1e-300, 1 - 1e-16)
        w = (i == 0 || i == n) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
        s += w * t^(a - 1) * (1 - t)^(b - 1)
    end
    s *= h / 3
    lB = lgamma_(a) + lgamma_(b) - lgamma_(a + b)
    return clamp(s / exp(lB), 0.0, 1.0)
end

lgamma_(z) = (v = 0.0; while z < 8; v -= log(z); z += 1; end;
              v + (z - 0.5) * log(z) - z + 0.5 * log(2π) + 1 / (12z) - 1 / (360z^3))
