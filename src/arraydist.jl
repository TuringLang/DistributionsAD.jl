# Utils

function summaporbroadcast(f, dists::AbstractArray, x::AbstractArray)
    # Broadcasting here breaks Tracker for some reason
    return sum(map(f, dists, x))
end
function summaporbroadcast(f, dists::AbstractVector, x::AbstractMatrix)
    # `eachcol` breaks Zygote, so we use `view` directly
    return map(i -> summaporbroadcast(f, dists, view(x, :, i)), axes(x, 2))
end
@init @require LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02" begin
    function summaporbroadcast(f, dists::LazyArrays.BroadcastArray, x::AbstractArray)
        return sum(copy(f.(dists, x)))
    end
    function summaporbroadcast(f, dists::LazyArrays.BroadcastVector, x::AbstractMatrix)
        return vec(sum(copy(f.(dists, x)), dims = 1))
    end
    lazyarray(f, x...) = LazyArrays.LazyArray(Base.broadcasted(f, x...))
    export lazyarray
end

# Univariate

const VectorOfUnivariate = Distributions.Product

function arraydist(dists::AbstractVector{<:UnivariateDistribution})
    return Product(dists)
end

function Distributions.logpdf(dist::VectorOfUnivariate, x::AbstractVector{<:Real})
    return summaporbroadcast(logpdf, dist.v, x)
end
function Distributions.logpdf(dist::VectorOfUnivariate, x::AbstractMatrix{<:Real})
    # eachcol breaks Zygote, so we need an adjoint
    return summaporbroadcast(logpdf, dist.v, x)
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
function Distributions.logpdf(dist::MatrixOfUnivariate, x::AbstractMatrix{<:Real})
    return summaporbroadcast(logpdf, dist.dists, x)
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

function Distributions.logpdf(dist::VectorOfMultivariate, x::AbstractMatrix{<:Real})
    # `eachcol` breaks Zygote, so we use `view` directly
    return sum(map(logpdf, dist.dists, axes(x, 2)))
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
