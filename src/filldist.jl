"""
    filldist(d::Distribution, ns...)

!!! warning "Deprecated"
    `filldist` is deprecated. Use `Distributions.product_distribution(Fill(d, ns...))` instead, 
    where `Fill` is from the `FillArrays` package.

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
function filldist(d::Distribution, n1::Int, ns::Int...)
    Base.depwarn("filldist is deprecated. Use `Distributions.product_distribution(Fill(d, n1, ns...))` instead, where `Fill` is from the `FillArrays` package.", :filldist)
    return product_distribution(Fill(d, n1, ns...))
end

# Univariate

function filldist(dist::UnivariateDistribution, N::Int)
    Base.depwarn("filldist is deprecated. Use `Distributions.product_distribution(Fill(dist, N))` instead, where `Fill` is from the `FillArrays` package.", :filldist)
    return product_distribution(Fill(dist, N))
end
function filldist(d::Normal, N::Int)
    Base.depwarn("filldist is deprecated. Use `Distributions.product_distribution(Fill(d, N))` instead, where `Fill` is from the `FillArrays` package.", :filldist)
    return product_distribution(Fill(d, N))
end

function filldist(dist::UnivariateDistribution, N1::Int, N2::Int)
    Base.depwarn("filldist is deprecated. Use `Distributions.product_distribution(Fill(dist, N1, N2))` instead, where `Fill` is from the `FillArrays` package.", :filldist)
    return product_distribution(Fill(dist, N1, N2))
end

# Multivariate

function filldist(dist::MultivariateDistribution, N::Int)
    Base.depwarn("filldist is deprecated. Use `Distributions.product_distribution(Fill(dist, N))` instead, where `Fill` is from the `FillArrays` package.", :filldist)
    return product_distribution(Fill(dist, N))
end
