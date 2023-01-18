using .LazyArrays: BroadcastArray, BroadcastVector, LazyArray

# Necessary to make `BroadcastArray` work nicely with Zygote.
function ChainRulesCore.rrule(config::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode}, ::Type{BroadcastArray}, f, args...)
    return ChainRulesCore.rrule_via_ad(config, Broadcast.broadcasted, f, args...)
end

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
    # TODO: Make use of `sum(Broadcast.instantiate(Broadcast.broadcasted(f, x, args...)))` once
    # we've addressed performance issues in ReverseDiff.jl.
    constructor = _inner_constructor(typeof(dist.v))
    return sum(Closure(logpdf, constructor).(x, dist.v.args...))
end

function Distributions.logpdf(
    dist::LazyVectorOfUnivariate,
    x::AbstractMatrix{<:Real},
)
    size(x, 1) == length(dist) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    constructor = _inner_constructor(typeof(dist.v))
    return vec(sum(Closure(logpdf, constructor).(x, dist.v.args...), dims = 1))
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

    constructor = _inner_constructor(typeof(dist.v))
    return sum(Closure(logpdf, constructor).(x, dist.v.args))
end

lazyarray(f, x...) = BroadcastArray(f, x...)
export lazyarray
