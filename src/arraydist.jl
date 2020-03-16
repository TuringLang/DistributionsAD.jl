# Univariate

const VectorOfUnivariate = Distributions.Product

function arraydist(dists::AbstractVector{<:UnivariateDistribution})
    return product_distribution(dists)
end
function arraydist(dists::AbstractVector{<:Normal})
    m = mapvcat(mean, dists)
    s = mapvcat(std, dists)
    return TuringMvNormal(m, s)
end

function Distributions.logpdf(dist::VectorOfUnivariate, x::AbstractVector{<:Real})
    return sum(map((d, x) -> logpdf(d, x), dist.v, x))
end
function Distributions.logpdf(dist::VectorOfUnivariate, x::AbstractMatrix{<:Real})
    # eachcol breaks Zygote, so we need an adjoint
    return mapvcat(dist.v, eachcol(x)) do dist, c
        sum(map(c) do x
            logpdf(dist, x)
        end)
    end
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
    # Broadcasting here breaks Tracker for some reason
    # A Zygote adjoint is defined for mapvcat to use broadcasting
    return sum(map(dist.dists, x) do dist, x
        logpdf(dist, x)
    end)
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
    return sum(logpdf.(dist.dists, eachcol(x)))
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
