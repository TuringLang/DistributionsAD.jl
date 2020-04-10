# Utils

function maporbroadcastlogpdf(dists, x::AbstractVector)
    # Broadcasting here breaks Tracker for some reason
    return sum(map(dists, x) do dist, x
        logpdf(dist, x)
    end)
end
function maporbroadcastlogpdf(dists, x::AbstractMatrix)
    return map(x -> maporbroadcastlogpdf(dists, x), eachcol(x))
end
@require LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02" begin
    function maporbroadcastlogpdf(dists::LazyArrays.BroadcastArray, x::AbstractVector)
        return sum(copy(logpdf.(dists, x)))
    end
    function maporbroadcastlogpdf(dists::LazyArrays.BroadcastArray, x::AbstractMatrix)
        return vec(sum(copy(logpdf.(dists, x)), dims = 1))
    end
end

# Univariate

const VectorOfUnivariate = Distributions.Product

function arraydist(dists::AbstractVector{<:UnivariateDistribution})
    return Product(dists)
end

function Distributions.logpdf(dist::VectorOfUnivariate, x::AbstractVector{<:Real})
    return maporbroadcastlogpdf(dist.v, x)
end
function Distributions.logpdf(dist::VectorOfUnivariate, x::AbstractMatrix{<:Real})
    # eachcol breaks Zygote, so we need an adjoint
    return maporbroadcastlogpdf(dist.v, x)
end
@adjoint function Distributions.logpdf(dist::VectorOfUnivariate, x::AbstractMatrix{<:Real})
    # Any other more efficient implementation breaks Zygote
    f(dist, x) = [sum(logpdf.(dist.v, view(x, :, i))) for i in 1:size(x, 2)]
    return pullback(f, dist, x)
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
    return maporbroadcastlogpdf(dist.dists, x)
end
function Distributions.logpdf(dist::MatrixOfUnivariate, x::AbstractArray{<:AbstractMatrix{<:Real}})
    return mapvcat(x -> logpdf(dist, x), x)
end
function Distributions.logpdf(dist::MatrixOfUnivariate, x::AbstractArray{<:Matrix{<:Real}})
    return mapvcat(x -> logpdf(dist, x), x)
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
    # eachcol breaks Zygote, so we define an adjoint
    return sum(map(logpdf, dist.dists, eachcol(x)))
end
function Distributions.logpdf(dist::VectorOfMultivariate, x::AbstractArray{<:AbstractMatrix{<:Real}})
    return mapvcat(x -> logpdf(dist, x), x)
end
function Distributions.logpdf(dist::VectorOfMultivariate, x::AbstractArray{<:Matrix{<:Real}})
    return mapvcat(x -> logpdf(dist, x), x)
end
@adjoint function Distributions.logpdf(dist::VectorOfMultivariate, x::AbstractMatrix{<:Real})
    f(dist, x) = sum(mapvcat(i -> logpdf(dist.dists[i], view(x, :, i)), 1:size(x, 2)))
    return pullback(f, dist, x)
end
function Distributions.rand(rng::Random.AbstractRNG, dist::VectorOfMultivariate)
    init = reshape(rand(rng, dist.dists[1]), :, 1)
    return mapreduce(i -> rand(rng, dist.dists[i]), hcat, 2:length(dist); init = init)
end
