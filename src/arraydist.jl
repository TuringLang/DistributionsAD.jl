"""
    arraydist(dists)

Create a distribution from an array of distributions.
"""
arraydist(dists::AbstractArray{<:Distribution}) = product_distribution(dists)

# Univariate

const VectorOfUnivariate = Distributions.Product

function arraydist(dists::AbstractVector{<:UnivariateDistribution})
    V = typeof(dists)
    T = eltype(dists)
    S = Distributions.value_support(T)
    return Product{S,T,V}(dists)
end

struct MatrixOfUnivariate{
    S <: ValueSupport,
    Tdist <: UnivariateDistribution{S},
    Tdists <: AbstractMatrix{Tdist},
} <: MatrixDistribution{S}
    dists::Tdists
end
Base.size(dist::MatrixOfUnivariate) = size(dist.dists)
function arraydist(dists::AbstractMatrix{<:UnivariateDistribution})
    return MatrixOfUnivariate(dists)
end
function Distributions._logpdf(dist::MatrixOfUnivariate, x::AbstractMatrix{<:Real})
    # Lazy broadcast to avoid allocations and use pairwise summation
    return sum(Broadcast.instantiate(Broadcast.broadcasted(logpdf, dist.dists, x)))
end
function Distributions.logpdf(dist::MatrixOfUnivariate, x::AbstractArray{<:AbstractMatrix{<:Real}})
    return map(Base.Fix1(logpdf, dist), x)
end
function Distributions.logpdf(dist::MatrixOfUnivariate, x::AbstractArray{<:Matrix{<:Real}})
    return map(Base.Fix1(logpdf, dist), x)
end

function Distributions.rand(rng::Random.AbstractRNG, dist::MatrixOfUnivariate)
    return rand.(Ref(rng), dist.dists)
end

# Multivariate

struct VectorOfMultivariate{
    S <: ValueSupport,
    Tdist <: MultivariateDistribution{S},
    Tdists <: AbstractVector{Tdist},
} <: MatrixDistribution{S}
    dists::Tdists
end
Base.size(dist::VectorOfMultivariate) = (length(dist.dists[1]), length(dist))
Base.length(dist::VectorOfMultivariate) = length(dist.dists)
function arraydist(dists::AbstractVector{<:MultivariateDistribution})
    return VectorOfMultivariate(dists)
end

function Distributions._logpdf(dist::VectorOfMultivariate, x::AbstractMatrix{<:Real})
    return sum(Broadcast.instantiate(Broadcast.broadcasted(logpdf, dist.dists, eachcol(x))))
end
function Distributions.logpdf(dist::VectorOfMultivariate, x::AbstractArray{<:AbstractMatrix{<:Real}})
    return map(Base.Fix1(logpdf, dist), x)
end
function Distributions.logpdf(dist::VectorOfMultivariate, x::AbstractArray{<:Matrix{<:Real}})
    return map(Base.Fix1(logpdf, dist), x)
end

function Distributions.rand(rng::Random.AbstractRNG, dist::VectorOfMultivariate)
    init = reshape(rand(rng, dist.dists[1]), :, 1)
    return mapreduce(Base.Fix1(rand, rng), hcat, view(dist.dists, 2:length(dist)); init = init)
end
