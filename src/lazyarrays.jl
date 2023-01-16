using .LazyArrays: BroadcastArray, BroadcastVector, LazyArray

const LazyVectorOfUnivariate{
    S<:ValueSupport,
    T<:UnivariateDistribution{S},
    Tdists<:BroadcastVector{T},
} = VectorOfUnivariate{S,T,Tdists}

_inner_constructor(::Type{<:BroadcastVector{<:Any,Type{D}}}) where {D} = D

function Distributions._logpdf(
    dist::LazyVectorOfUnivariate,
    x::AbstractVector{<:Real},
)
    # TODO: Implement chain rule for `LazyArray` constructor to support Zygote.
    f = make_closure(logpdf, _inner_constructor(typeof(dist.v)))
    # TODO: Make use of `sum(Broadcast.instantiate(Broadcast.broadcasted(f, x, args...)))` once
    # we've addressed performance issues in ReverseDiff.jl.
    return sum(f.(x, dist.v.args...))
end

function Distributions.logpdf(
    dist::LazyVectorOfUnivariate,
    x::AbstractMatrix{<:Real},
    )
    size(x, 1) == length(dist) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    f = make_closure(logpdf, _inner_constructor(typeof(dist.v)))
    return vec(sum(f.(x, dist.v.args...), dims = 1))
end

const LazyMatrixOfUnivariate{
    S<:ValueSupport,
    T<:UnivariateDistribution{S},
    Tdists<:BroadcastArray{T,2},
} = MatrixOfUnivariate{S,T,Tdists}

function Distributions._logpdf(
    dist::LazyMatrixOfUnivariate,
    x::AbstractMatrix{<:Real},
)
    f = make_closure(logpdf, _inner_constructor(typeof(dist.v)))
    
    return sum(f.(x, dist.v.args))
end

lazyarray(f, x...) = BroadcastArray(f, x...)
export lazyarray
