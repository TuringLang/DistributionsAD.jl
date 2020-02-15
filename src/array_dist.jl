# Univariate

const VectorOfUnivariate{
    S <: ValueSupport,
    Tdist <: UnivariateDistribution{S},
    Tdists <: AbstractVector{Tdist},
} = Distributions.Product{S, Tdist, Tdists}

function ArrayDist(dists::AbstractVector{<:Normal{T}}) where {T}
    if T <: TrackedReal
        init_m = dists[1].μ
        means = mapreduce(vcat, drop(dists, 1); init = init_m) do d
            d.μ
        end
        init_v = dists[1].σ^2
        vars = mapreduce(vcat, drop(dists, 1); init = init_v) do d
            d.σ^2
        end
    else
        means = [d.μ for d in dists]
        vars = [d.σ^2 for d in dists]
    end

    return MvNormal(means, vars)
end
function ArrayDist(dists::AbstractVector{<:UnivariateDistribution})
    return Distributions.Product(dists)
end
function Distributions.logpdf(dist::VectorOfUnivariate, x::AbstractVector{<:Real})
    return sum(logpdf.(dist.v, x))
end
function Distributions.logpdf(dist::VectorOfUnivariate, x::AbstractMatrix{<:Real})
    # Any other more efficient implementation breaks Zygote
    return [logpdf(dist, x[:,i]) for i in 1:size(x, 2)]
end
function Distributions.logpdf(
    dist::VectorOfUnivariate,
    x::AbstractVector{<:AbstractMatrix{<:Real}},
)
    return logpdf.(Ref(dist), x)
end

struct MatrixOfUnivariate{
    S <: ValueSupport,
    Tdist <: UnivariateDistribution{S},
    Tdists <: AbstractMatrix{Tdist},
} <: MatrixDistribution{S}
    dists::Tdists
end
Base.size(dist::MatrixOfUnivariate) = size(dist.dists)
function ArrayDist(dists::AbstractMatrix{<:UnivariateDistribution})
    return MatrixOfUnivariate(dists)
end
function Distributions.logpdf(dist::MatrixOfUnivariate, x::AbstractMatrix{<:Real})
    # Broadcasting here breaks Tracker for some reason
    return sum(zip(dist.dists, x)) do (dist, x)
        logpdf(dist, x)
    end
end
function Distributions.rand(rng::Random.AbstractRNG, dist::MatrixOfUnivariate)
    return rand.(Ref(rng), dist.dists)
end

# Multivariate continuous

struct VectorOfMultivariate{
    S <: ValueSupport,
    Tdist <: MultivariateDistribution{S},
    Tdists <: AbstractVector{Tdist},
} <: MatrixDistribution{S}
    dists::Tdists
end
Base.size(dist::VectorOfMultivariate) = (length(dist.dists[1]), length(dist))
Base.length(dist::VectorOfMultivariate) = length(dist.dists)
function ArrayDist(dists::AbstractVector{<:MultivariateDistribution})
    return VectorOfMultivariate(dists)
end
function Distributions.logpdf(dist::VectorOfMultivariate, x::AbstractMatrix{<:Real})
    return sum(logpdf(dist.dists[i], x[:,i]) for i in 1:length(dist))
end
function Distributions.logpdf(
    dist::VectorOfMultivariate,
    x::AbstractVector{<:AbstractVector{<:Real}},
)
    return sum(logpdf(dist.dists[i], x[i]) for i in 1:length(dist))
end
function Distributions.rand(rng::Random.AbstractRNG, dist::VectorOfMultivariate)
    init = reshape(rand(rng, dist.dists[1]), :, 1)
    return mapreduce(i -> rand(rng, dist.dists[i]), hcat, 2:length(dist); init = init)
end
