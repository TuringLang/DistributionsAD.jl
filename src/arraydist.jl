"""
    arraydist(dists::AbstractArray{<:Distribution})

!!! warning "Deprecated"
    `arraydist` is deprecated. Use `Distributions.product_distribution(dists)` instead.

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
function arraydist(dists::AbstractArray{<:Distribution})
    Base.depwarn("arraydist is deprecated. Use `Distributions.product_distribution(dists)` instead.", :arraydist)
    return product_distribution(dists)
end

# Univariate

const VectorOfUnivariate = Distributions.Product

function arraydist(dists::AbstractVector{<:UnivariateDistribution})
    Base.depwarn("arraydist is deprecated. Use `Distributions.product_distribution(dists)` instead.", :arraydist)
    return product_distribution(dists)
end

function arraydist(dists::AbstractMatrix{<:UnivariateDistribution})
    Base.depwarn("arraydist is deprecated. Use `Distributions.product_distribution(dists)` instead.", :arraydist)
    return product_distribution(dists)
end

# Multivariate

function arraydist(dists::AbstractVector{<:MultivariateDistribution})
    Base.depwarn("arraydist is deprecated. Use `Distributions.product_distribution(dists)` instead.", :arraydist)
    return product_distribution(dists)
end
