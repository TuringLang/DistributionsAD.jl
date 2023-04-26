module DistributionsADLazyArraysExt

if isdefined(Base, :get_extension)
    using DistributionsAD
    using LazyArrays
else
    using ..DistributionsAD
    using ..LazyArrays
end
using DistributionsAD: Distributions
using LazyArrays: BroadcastArray, BroadcastVector, LazyArray

const LazyVectorOfUnivariate{
    S<:ValueSupport,
    T<:UnivariateDistribution{S},
    Tdists<:BroadcastVector{T},
} = VectorOfUnivariate{S,T,Tdists}

function Distributions._logpdf(
    dist::LazyVectorOfUnivariate,
    x::AbstractVector{<:Real},
)
    return sum(copy(logpdf.(dist.v, x)))
end

function Distributions.logpdf(
    dist::LazyVectorOfUnivariate,
    x::AbstractMatrix{<:Real},
)
    size(x, 1) == length(dist) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    return vec(sum(copy(logpdf.(dists, x)), dims = 1))
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
    return sum(copy(logpdf.(dist.dists, x)))
end

DistributionsAD.lazyarray(f, x...) = LazyArray(Base.broadcasted(f, x...))

end # module