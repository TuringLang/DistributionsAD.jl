# Zygote fill has issues with non-numbers
ZygoteRules.@adjoint function fill(x::T, dims...) where {T}
    return ZygoteRules.pullback(x, dims...) do x, dims...
        return reshape([x for i in 1:prod(dims)], dims)
    end
end


## Uniform ##

ZygoteRules.@adjoint function Distributions.Uniform(args...)
    return ZygoteRules.pullback(TuringUniform, args...)
end

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

## Wishart ##

# Custom adjoint since Zygote can't differentiate through `@warn`
# TODO: Remove when fixed upstream in Distributions
ZygoteRules.@adjoint function Wishart(df::T, S::AbstractPDMat{T}, warn::Bool = true) where T<:Real
    function _Wishart(df::T, S::AbstractPDMat{T}, warn::Bool = true) where T
        df > 0 || throw(ArgumentError("df must be positive. got $(df)."))
        p = dim(S)
        rnk = p
        singular = df <= p - 1
        if singular
            isinteger(df) || throw(ArgumentError("singular df must be an integer. got $(df)."))
            rnk = convert(Integer, df)
            warn && _warn("got df <= dim - 1; returning a singular Wishart")
        end
        logc0 = Distributions.wishart_logc0(df, S, rnk)
        R = Base.promote_eltype(T, logc0)
        prom_S = convert(AbstractArray{T}, S)
        Wishart{R, typeof(prom_S), typeof(rnk)}(R(df), prom_S, R(logc0), rnk, singular)
    end
    return ZygoteRules.pullback(_Wishart, df, S, warn)
end

_warn(msg) = @warn(msg)
@non_differentiable _warn(msg)

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
        return map(i -> Distributions._logpdf(dist, view(X, :, i)), axes(X, 2))
    end
end

ZygoteRules.@adjoint function Distributions.logpdf(
    dist::MatrixDistribution,
    X::AbstractArray{<:Real,3},
)
    (size(X, 1), size(X, 2)) == size(dist) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    return ZygoteRules.pullback(dist, X) do dist, X
        return map(i -> Distributions._logpdf(dist, view(X, :, :, i)), axes(X, 3))
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
