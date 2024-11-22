"""
    filldist(d::Distribution, ns...)

Create a product distribution from a single distribution and a list of
dimension sizes. If `size(d)` is `(d1, d2, ...)` and `ns` is `(n1, n2, ...)`,
then the resulting distribution will have size `(d1, d2, ..., n1, n2, ...)`.

The default behaviour is to use
[`Distributions.product_distribution`](https://juliastats.org/Distributions.jl/stable/multivariate/#Distributions.product_distribution),
with `FillArrays.Fill` supplied as the array argument. However, this behaviour
is specialised in some instances, such as the one shown below.

When sampling from the resulting distribution, the output will be an array where
each element is sampled from the original distribution `d`.

# Examples

```jldoctest; setup=:(using Distributions, Random)
julia> d = filldist(Normal(0, 1), 4, 5);

julia> size(d)
(4, 5)

julia> rand(d) isa Matrix{Float64}
true
```
"""
filldist(d::Distribution, n1::Int, ns::Int...) = product_distribution(Fill(d, n1, ns...))

# Univariate

# TODO: Do we even need these? Probably should benchmark to be sure.
const FillVectorOfUnivariate{
    S <: ValueSupport,
    T <: UnivariateDistribution{S},
    Tdists <: Fill{T, 1},
} = VectorOfUnivariate{S, T, Tdists}

function filldist(dist::UnivariateDistribution, N::Int)
    return product_distribution(Fill(dist, N))
end
filldist(d::Normal, N::Int) = TuringMvNormal(fill(d.μ, N), d.σ)

function Distributions._logpdf(
    dist::FillVectorOfUnivariate,
    x::AbstractVector{<:Real},
)
    return _flat_logpdf(dist.v.value, x)
end

function Distributions.logpdf(
    dist::FillVectorOfUnivariate,
    x::AbstractMatrix{<:Real},
)
    size(x, 1) == length(dist) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    return _flat_logpdf_mat(dist.v.value, x)
end

function _flat_logpdf(dist, x)
    if toflatten(dist)
        f, args = flatten(dist)
        # Lazy broadcast to avoid allocations and use pairwise summation
        return sum(Broadcast.instantiate(Broadcast.broadcasted(xi -> f(args..., xi), x)))
    else
        return sum(Broadcast.instantiate(Broadcast.broadcasted(Base.Fix1(logpdf, dist), x)))
    end
end

function _flat_logpdf_mat(dist, x)
    if toflatten(dist)
        f, args = flatten(dist)
        return vec(mapreduce(xi -> f(args..., xi), +, x, dims = 1))
    else
        return vec(mapreduce(Base.Fix1(logpdf, dist), +, x; dims = 1))
    end
end

function Distributions.rand(rng::Random.AbstractRNG, d::FillVectorOfUnivariate)
    return rand(rng, d.v.value, length(d))
end
function Distributions.rand(rng::Random.AbstractRNG, d::FillVectorOfUnivariate, n::Int)
    return rand(rng, d.v.value, length(d), n)
end

const FillMatrixOfUnivariate{
    S <: ValueSupport,
    T <: UnivariateDistribution{S},
    Tdists <: Fill{T, 2},
} = MatrixOfUnivariate{S, T, Tdists}

function filldist(dist::UnivariateDistribution, N1::Int, N2::Int)
    return MatrixOfUnivariate(Fill(dist, N1, N2))
end
function Distributions._logpdf(dist::FillMatrixOfUnivariate, x::AbstractMatrix{<:Real})
    # return loglikelihood(dist.dists.value, x)
    return _flat_logpdf(dist.dists.value, x)
end
function Distributions.rand(rng::Random.AbstractRNG, dist::FillMatrixOfUnivariate)
    return rand(rng, dist.dists.value, length.(dist.dists.axes)...,)
end

# Multivariate

const FillVectorOfMultivariate{
    S <: ValueSupport,
    T <: MultivariateDistribution{S},
    Tdists <: Fill{T, 1},
} = VectorOfMultivariate{S, T, Tdists}

function filldist(dist::MultivariateDistribution, N::Int)
    return VectorOfMultivariate(Fill(dist, N))
end
function Distributions._logpdf(
    dist::FillVectorOfMultivariate,
    x::AbstractMatrix{<:Real},
)
    return loglikelihood(dist.dists.value, x)
end

function Distributions.rand(rng::Random.AbstractRNG, dist::FillVectorOfMultivariate)
    return rand(rng, dist.dists.value, length.(dist.dists.axes)...,)
end
