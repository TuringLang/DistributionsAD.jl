# Univariate

const VectorOfUnivariate = Distributions.Product

function arraydist(dists::AbstractVector{<:UnivariateDistribution})
    return product_distribution(dists)
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
    # return sum(((d, xi),) -> logpdf(d, xi), zip(dist.dists, x))
    # Broadcasting here breaks Tracker for some reason
    return sum(map(logpdf, dist.dists, x))
end
function Distributions.logpdf(dist::MatrixOfUnivariate, x::AbstractArray{<:AbstractMatrix{<:Real}})
    return map(x -> logpdf(dist, x), x)
end
function Distributions.logpdf(dist::MatrixOfUnivariate, x::AbstractArray{<:Matrix{<:Real}})
    return map(x -> logpdf(dist, x), x)
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
    return sum(((di, xi),) -> logpdf(di, xi), zip(dist.dists, eachcol(x)))
end
function Distributions.logpdf(dist::VectorOfMultivariate, x::AbstractArray{<:AbstractMatrix{<:Real}})
    return map(x -> logpdf(dist, x), x)
end
function Distributions.logpdf(dist::VectorOfMultivariate, x::AbstractArray{<:Matrix{<:Real}})
    return map(x -> logpdf(dist, x), x)
end

function Distributions.rand(rng::Random.AbstractRNG, dist::VectorOfMultivariate)
    init = reshape(rand(rng, dist.dists[1]), :, 1)
    return mapreduce(i -> rand(rng, dist.dists[i]), hcat, 2:length(dist); init = init)
end
