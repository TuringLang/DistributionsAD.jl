# Multivariate continuous

struct ProductVectorContinuousMultivariate{
    Tdists <: AbstractVector{<:ContinuousMultivariateDistribution},
} <: ContinuousMatrixDistribution
    dists::Tdists
end
Base.size(dist::ProductVectorContinuousMultivariate) = (length(dist.dists[1]), length(dist))
Base.length(dist::ProductVectorContinuousMultivariate) = length(dist.dists)
function ArrayDist(dists::AbstractVector{<:ContinuousMultivariateDistribution})
    return ProductVectorContinuousMultivariate(dists)
end
function Distributions.logpdf(
    dist::ProductVectorContinuousMultivariate,
    x::AbstractMatrix{<:Real},
)
    return sum(logpdf(dist.dists[i], x[:,i]) for i in 1:length(dist))
end
function Distributions.logpdf(
    dist::ProductVectorContinuousMultivariate,
    x::AbstractVector{<:AbstractVector{<:Real}},
)
    return sum(logpdf(dist.dists[i], x[i]) for i in 1:length(dist))
end
function Distributions.rand(
    rng::Random.AbstractRNG,
    dist::ProductVectorContinuousMultivariate,
)
    return mapreduce(i -> rand(rng, dist.dists[i]), hcat, 1:length(dist))
end

# Multivariate discrete

struct ProductVectorDiscreteMultivariate{
    Tdists <: AbstractVector{<:DiscreteMultivariateDistribution},
} <: DiscreteMatrixDistribution
    dists::Tdists
end
Base.size(dist::ProductVectorDiscreteMultivariate) = (length(dist.dists[1]), length(dist))
Base.length(dist::ProductVectorDiscreteMultivariate) = length(dist.dists)
function ArrayDist(dists::AbstractVector{<:DiscreteMultivariateDistribution})
    return ProductVectorDiscreteMultivariate(dists)
end
function Distributions.logpdf(
    dist::ProductVectorDiscreteMultivariate,
    x::AbstractMatrix{<:Integer},
)
    return sum(logpdf(dist.dists[i], x[:,i]) for i in 1:length(dist))
end
function Distributions.logpdf(
    dist::ProductVectorDiscreteMultivariate,
    x::AbstractVector{<:AbstractVector{<:Integer}},
)
    return sum(logpdf(dist.dists[i], x[i]) for i in 1:length(dist))
end
function Distributions.rand(
    rng::Random.AbstractRNG,
    dist::ProductVectorDiscreteMultivariate,
)
    return mapreduce(i -> rand(rng, dist.dists[i]), hcat, 1:length(dist))
end

# Univariate continuous

struct ProductVectorContinuousUnivariate{
    Tdists <: AbstractVector{<:ContinuousUnivariateDistribution},
} <: ContinuousMultivariateDistribution
    dists::Tdists
end
Base.length(dist::ProductVectorContinuousUnivariate) = length(dist.dists)
Base.size(dist::ProductVectorContinuousUnivariate) = (length(dist),)
function ArrayDist(dists::AbstractVector{<:ContinuousUnivariateDistribution})
    return ProductVectorContinuousUnivariate(dists)
end
function Distributions.logpdf(
    dist::ProductVectorContinuousUnivariate,
    x::AbstractVector{<:Real},
)
    return sum(logpdf.(dist.dists, x))
end
function Distributions.rand(
    rng::Random.AbstractRNG,
    dist::ProductVectorContinuousUnivariate,
)
    return rand.(Ref(rng), dist.dists)
end

struct ProductMatrixContinuousUnivariate{
    Tdists <: AbstractMatrix{<:ContinuousUnivariateDistribution},
} <: ContinuousMatrixDistribution
    dists::Tdists
end
Base.size(dist::ProductMatrixContinuousUnivariate) = size(dist.dists)
function ArrayDist(dists::AbstractMatrix{<:ContinuousUnivariateDistribution})
    return ProductMatrixContinuousUnivariate(dists)
end
function Distributions.logpdf(
    dist::ProductMatrixContinuousUnivariate,
    x::AbstractMatrix{<:Real},
)
    return sum(logpdf.(dist.dists, x))
end
function Distributions.rand(
    rng::Random.AbstractRNG,
    dist::ProductMatrixContinuousUnivariate,
)
    return rand.(Ref(rng), dist.dists)
end

# Univariate discrete

struct ProductVectorDiscreteUnivariate{
    Tdists <: AbstractVector{<:DiscreteUnivariateDistribution},
} <: ContinuousMultivariateDistribution
    dists::Tdists
end
Base.length(dist::ProductVectorDiscreteUnivariate) = length(dist.dists)
Base.size(dist::ProductVectorDiscreteUnivariate) = (length(dist.dists[1]), length(dist))
function ArrayDist(dists::AbstractVector{<:DiscreteUnivariateDistribution})
    ProductVectorDiscreteUnivariate(dists)
end
function Distributions.logpdf(
    dist::ProductVectorDiscreteUnivariate,
    x::AbstractVector{<:Integer},
)
    return sum(logpdf.(dist.dists, x))
end
function Distributions.rand(
    rng::Random.AbstractRNG,
    dist::ProductVectorDiscreteUnivariate,
)
    return rand.(Ref(rng), dist.dists)
end

struct ProductMatrixDiscreteUnivariate{
    Tdists <: AbstractMatrix{<:DiscreteUnivariateDistribution},
} <: DiscreteMatrixDistribution
    dists::Tdists
end
Base.size(dists::ProductMatrixDiscreteUnivariate) = size(dist.dists)
function ArrayDist(dists::AbstractMatrix{<:DiscreteUnivariateDistribution})
    return ProductMatrixDiscreteUnivariate(dists)
end
function Distributions.logpdf(
    dist::ProductMatrixDiscreteUnivariate,
    x::AbstractMatrix{<:Real},
)
    return sum(logpdf.(dist.dists, x))
end
function Distributions.rand(rng::Random.AbstractRNG, dist::ProductMatrixDiscreteUnivariate)
    return rand.(Ref(rng), dist.dists)
end
