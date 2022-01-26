## Product

# Tests with `Kolmogorov` seem to fail otherwise?!
ZygoteRules.@adjoint function Distributions._logpdf(d::Product, x::AbstractVector{<:Real})
    return ZygoteRules.pullback(d, x) do d, x
        sum(map(logpdf, d.v, x))
    end
end
ZygoteRules.@adjoint function Distributions._logpdf(
    d::FillVectorOfUnivariate,
    x::AbstractVector{<:Real},
)
    return ZygoteRules.pullback(d, x) do d, x
        _flat_logpdf(d.v.value, x)
    end
end

# Loglikelihood of multi- and matrixvariate distributions: multiple samples
# workaround for Zygote issues discussed in
# https://github.com/TuringLang/DistributionsAD.jl/pull/198
ZygoteRules.@adjoint function Distributions.loglikelihood(
    d::MultivariateDistribution, x::AbstractMatrix{<:Real}
)
    return ZygoteRules.pullback(d, x) do d, x
        return sum(xi -> Distributions._logpdf(d, xi), eachcol(x))
    end
end
ZygoteRules.@adjoint function Distributions.loglikelihood(
    d::MatrixDistribution, x::AbstractArray{<:Real,3}
)
    return ZygoteRules.pullback(d, x) do d, x
        return sum(xi -> Distributions._logpdf(d, xi), eachslice(x; dims=3))
    end
end

## Wishart ##

ZygoteRules.@adjoint function Distributions.Wishart(df::Real, S::AbstractMatrix{<:Real})
    return ZygoteRules.pullback(TuringWishart, df, S)
end
ZygoteRules.@adjoint function Distributions.InverseWishart(
    df::Real,
    S::AbstractMatrix{<:Real}
)
    return ZygoteRules.pullback(TuringInverseWishart, df, S)
end

## General definitions of `logpdf` for arrays

ZygoteRules.@adjoint function Distributions.logpdf(
    dist::MultivariateDistribution,
    X::AbstractMatrix{<:Real},
)
    size(X, 1) == length(dist) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    return ZygoteRules.pullback(dist, X) do dist, X
        return map(xi -> Distributions._logpdf(dist, xi), eachcol(X))
    end
end

ZygoteRules.@adjoint function Distributions.logpdf(
    dist::MatrixDistribution,
    X::AbstractArray{<:Real,3},
)
    (size(X, 1), size(X, 2)) == size(dist) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    return ZygoteRules.pullback(dist, X) do dist, X
        return map(xi -> Distributions._logpdf(dist, xi), eachslice(X; dims=3))
    end
end

ZygoteRules.@adjoint function Distributions.logpdf(
    dist::MatrixDistribution,
    X::AbstractArray{<:AbstractMatrix{<:Real}},
)
    return ZygoteRules.pullback(dist, X) do dist, X
        return map(x -> logpdf(dist, x), X)
    end
end
