using Distributions: @check_args

"""
    MvDiscreteNonParametric(xs, ps)

A *multivariate Discrete nonparametric distribution* explicitly defines an arbitrary
probability mass function in terms of a list of real support values and their
corresponding probabilities

```julia
d = MvDiscreteNonParametric(xs, ps)

params(d)  # Get the parameters, i.e. (xs, ps)
support(d) # Get a sorted AbstractVector describing the support (xs) of the distribution
probs(d)   # Get a Matrix of the probabilities (ps) associated with the support
```

External links

* [Probability mass function on Wikipedia](http://en.wikipedia.org/wiki/Probability_mass_function)
"""
struct MvDiscreteNonParametric{T<:Real,P<:Real,Ts<:AbstractVector{T},Ps<:AbstractMatrix{P}} <: DiscreteMultivariateDistribution
    support::Ts
    p::Ps

    function MvDiscreteNonParametric{T,P,Ts,Ps}(vs::Ts, ps::Ps; check_args=true) where {
            T<:Real,P<:Real,Ts<:AbstractVector{T},Ps<:AbstractMatrix{P}}
        check_args || return new{T,P,Ts,Ps}(vs, ps)
        @check_args(MvDiscreteNonParametric, length(vs) == size(ps,1))
        @check_args(MvDiscreteNonParametric, all(isprobvec, eachcol(ps)))
        @check_args(MvDiscreteNonParametric, allunique(vs))
        sort_order = sortperm(vs)
        new{T,P,Ts,Ps}(vs[sort_order], ps[sort_order,:])
    end
end

MvDiscreteNonParametric(vs::Ts, ps::Ps; check_args=true) where {
        T<:Real,P<:Real,Ts<:AbstractVector{T},Ps<:AbstractMatrix{P}} =
    MvDiscreteNonParametric{T,P,Ts,Ps}(vs, ps, check_args=check_args)

Base.eltype(::Type{<:MvDiscreteNonParametric{T}}) where T = T

# Conversion
Base.convert(::Type{MvDiscreteNonParametric{T,P,Ts,Ps}}, d::MvDiscreteNonParametric) where {T,P,Ts,Ps} =
    MvDiscreteNonParametric{T,P,Ts,Ps}(Ts(support(d)), Ps(probs(d)), check_args=false)

# Accessors
Distributions.params(d::MvDiscreteNonParametric) = (d.support, d.p)

"""
    support(d::MvDiscreteNonParametric)

Get a sorted AbstractVector defining the support of `d`.
"""
Distributions.support(d::MvDiscreteNonParametric) = d.support

"""
    probs(d::MvDiscreteNonParametric)

Get the vector of probabilities associated with the support of `d`.
"""
Distributions.probs(d::MvDiscreteNonParametric)  = d.p

import Base: ==
==(c1::D, c2::D) where D<:MvDiscreteNonParametric =
    (support(c1) == support(c2) || all(support(c1) .== support(c2))) &&
    (probs(c1) == probs(c2) || all(probs(c1) .== probs(c2)))

Base.isapprox(c1::D, c2::D) where D<:MvDiscreteNonParametric =
    (support(c1) ≈ support(c2) || all(support(c1) .≈ support(c2))) &&
    (probs(c1) ≈ probs(c2) || all(probs(c1) .≈ probs(c2)))

# Sampling

function Base.rand(rng::AbstractRNG, d::MvDiscreteNonParametric{T,P}) where {T,P}
    x = support(d)
    p = probs(d)
    n, k = size(p)
    map(1:k) do j
        draw = rand(rng, (P === Real ? Float64 : P))
        cp = zero(P)
        i = 0
        while cp < draw && i < n
            cp += p[i +=1, j]
        end
        x[max(i,1)]
    end
end

Base.rand(d::MvDiscreteNonParametric) = rand(Random.GLOBAL_RNG, d)

# Override the method in testutils.jl since it assumes
# an evenly-spaced integer support
Distributions.get_evalsamples(d::MvDiscreteNonParametric, ::Float64) = support(d)

# Evaluation

Distributions.pdf(d::MvDiscreteNonParametric) = copy(probs(d))

# Helper functions for pdf and cdf required to fix ambiguous method
# error involving [pc]df(::DisceteUnivariateDistribution, ::Int)
function _logpdf(d::MvDiscreteNonParametric{T,P}, x::AbstractVector{T}) where {T,P}
    s = zero(P)
    for col in 1:length(x)
        idx_range = searchsorted(support(d), x[col])
        if length(idx_range) > 0
            s += log(probs(d)[first(idx_range),col])
        end
    end
    return s
end
Distributions.logpdf(d::MvDiscreteNonParametric{T}, x::AbstractVector{<:Integer}) where T  = _logpdf(d, convert(AbstractVector{T}, x))
Distributions.logpdf(d::MvDiscreteNonParametric{T}, x::AbstractVector{<:Real}) where T = _logpdf(d, convert(AbstractVector{T}, x))
Distributions.pdf(d::MvDiscreteNonParametric, x::AbstractVector{<:Real}) = exp(logpdf(d, x))

Base.minimum(d::MvDiscreteNonParametric) = first(support(d))
Base.maximum(d::MvDiscreteNonParametric) = last(support(d))
Distributions.insupport(d::MvDiscreteNonParametric, x::AbstractVector{<:Real}) =
    all(x -> length(searchsorted(support(d), x)) > 0, x)

Distributions.mean(d::MvDiscreteNonParametric) = probs(d)' * support(d)

function Distributions.cov(d::MvDiscreteNonParametric)
    m = mean(d)
    x = support(d)
    p = probs(d)
    k = size(p,1)
    n = size(p,2)
    σ² = zero(m)
    for j in 1:n
        for i in 1:k
            @inbounds σ²[j] += abs2(x[i,j] - m[j]) * p[i,j]
        end
    end
    return Diagonal(σ²)
end

const MvCategorical{P,Ps} = MvDiscreteNonParametric{Int,P,Base.OneTo{Int},Ps}

MvCategorical(p::Ps; check_args=true) where {P<:Real, Ps<:AbstractMatrix{P}} =
    MvCategorical{P,Ps}(p, check_args=check_args)

function MvCategorical{P,Ps}(p::Ps; check_args=true) where {P<:Real, Ps<:AbstractMatrix{P}}
    check_args && @check_args(MvCategorical, all(isprobvec, eachcol(p)))
    return MvCategorical{P,Ps}(Base.OneTo(size(p, 1)), p, check_args=check_args)
end

Distributions.ncategories(d::MvCategorical) = support(d).stop
Distributions.params(d::MvCategorical{P,Ps}) where {P<:Real, Ps<:AbstractVector{P}} = (probs(d),)
Distributions.partype(::MvCategorical{T}) where {T<:Real} = T
function Distributions.logpdf(d::MvCategorical{T}, x::AbstractVector{<:Integer}) where {T<:Real}
    ps = probs(d)
    if insupport(d, x)
        _mv_categorical_logpdf(ps, x)
    else
        return zero(eltype(ps))
    end
end
_mv_categorical_logpdf(ps, x) = sum(log, view(ps, x, :))
_mv_categorical_logpdf(ps::Tracker.TrackedMatrix, x) = Tracker.track(_mv_categorical_logpdf, ps, x)
Tracker.@grad function _mv_categorical_logpdf(ps, x)
    ps_data = Tracker.data(ps)
    probs = view(ps_data, x, :)
    ps_grad = zero(ps_data)
    sum(log, probs), Δ -> begin
        ps_grad .= 0
        ps_grad[x,:] .= Δ ./ probs
        return (ps_grad, nothing)
    end
end
