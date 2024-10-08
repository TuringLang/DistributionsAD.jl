"""
    arraydist(dists::AbstractArray{<:Distribution})

Create a product distribution from an array of sub-distributions. Each element
of `dists` should have the same size. If the size of each element is `(d1, d2,
...)`, and `size(dists)` is `(n1, n2, ...)`, then the resulting distribution
will have size `(d1, d2, ..., n1, n2, ...)`.

The default behaviour is to directly use
[`Distributions.product_distribution`](https://juliastats.org/Distributions.jl/stable/multivariate/#Distributions.product_distribution),
although this can sometimes be specialised.

# Examples

```jldoctest; setup=:(using Distributions, Random)
julia> d1 = arraydist([Normal(0, 1), Normal(10, 1)])
Product{Continuous, Normal{Float64}, Vector{Normal{Float64}}}(v=Normal{Float64}[Normal{Float64}(μ=0.0, σ=1.0), Normal{Float64}(μ=10.0, σ=1.0)])

julia> size(d1)
(2,)

julia> Random.seed!(42); rand(d1)
2-element Vector{Float64}:
 0.7883556016042917
 9.1201414040456

julia> d2 = arraydist([Normal(0, 1) Normal(5, 1); Normal(10, 1) Normal(15, 1)])
DistributionsAD.MatrixOfUnivariate{Continuous, Normal{Float64}, Matrix{Normal{Float64}}}(
dists: Normal{Float64}[Normal{Float64}(μ=0.0, σ=1.0) Normal{Float64}(μ=5.0, σ=1.0); Normal{Float64}(μ=10.0, σ=1.0) Normal{Float64}(μ=15.0, σ=1.0)]
)

julia> size(d2)
(2, 2)

julia> Random.seed!(42); rand(d2)
2×2 Matrix{Float64}:
 0.788356   4.12621
 9.12014   14.2667
```
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
