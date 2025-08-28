module DistributionsADLazyArraysExt

if isdefined(Base, :get_extension)
    using DistributionsAD
    using LazyArrays
    using DistributionsAD: Distributions, ValueSupport, UnivariateDistribution, VectorOfUnivariate
    using LazyArrays: BroadcastArray, BroadcastVector, LazyArray
else
    using ..DistributionsAD
    using ..LazyArrays
    using ..DistributionsAD: Distributions, ValueSupport, UnivariateDistribution, VectorOfUnivariate
    using ..LazyArrays: BroadcastArray, BroadcastVector, LazyArray
end

const LazyVectorOfUnivariate{
    S<:ValueSupport,
    T<:UnivariateDistribution{S},
    Tdists<:BroadcastVector{T},
} = VectorOfUnivariate{S,T,Tdists}

function Distributions._logpdf(
    dist::LazyVectorOfUnivariate,
    x::AbstractVector{<:Real},
)
    return sum(copy(Distributions.logpdf.(dist.v, x)))
end

function Distributions.logpdf(
    dist::LazyVectorOfUnivariate,
    x::AbstractMatrix{<:Real},
)
    size(x, 1) == length(dist) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    return vec(sum(copy(Distributions.logpdf.(dists, x)), dims = 1))
end

DistributionsAD.lazyarray(f, x...) = LazyArray(Base.broadcasted(f, x...))

end # module
