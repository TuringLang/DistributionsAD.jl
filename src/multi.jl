# Multivariate continuous

struct MultipleContinuousMultivariate{
    Tdist <: ContinuousMultivariateDistribution
} <: ContinuousMatrixDistribution
    dist::Tdist
    N::Int
end
Base.size(dist::MultipleContinuousMultivariate) = (length(dist.dist), dist.N)
function Multi(dist::ContinuousMultivariateDistribution, N::Int)
    return MultipleContinuousMultivariate(dist, N)
end
function Distributions.logpdf(
    dist::MultipleContinuousMultivariate,
    x::AbstractMatrix{<:Real}
)
    return sum(logpdf(dist.dist, x))
end
function Distributions.rand(rng::Random.AbstractRNG, dist::MultipleContinuousMultivariate)
    return rand(rng, dist.dist, dist.N)
end
Distributions.MvNormal(m, s, N::Int) = MultipleContinuousMultivariate(MvNormal(m, s), N)


# Multivariate discrete

struct MultipleDiscreteMultivariate{
    Tdist <: DiscreteMultivariateDistribution
} <: DiscreteMatrixDistribution
    dist::Tdist
    N::Int
end
Base.size(dist::MultipleDiscreteMultivariate) = (length(dist.dist), dist.N)
function Multi(dist::DiscreteMultivariateDistribution, N::Int)
    return MultipleDiscreteMultivariate(dist, N)
end
function Distributions.logpdf(
    dist::MultipleDiscreteMultivariate,
    x::AbstractMatrix{<:Integer}
)
    return sum(logpdf(dist.dist, x))
end
function Distributions.rand(rng::Random.AbstractRNG, dist::MultipleDiscreteMultivariate)
    return rand(rng, dist.dist, dist.N)
end

# Univariate continuous

struct MultipleContinuousUnivariate{
    Tdist <: ContinuousUnivariateDistribution,
} <: ContinuousMultivariateDistribution
    dist::Tdist
    N::Int
end
Base.length(dist::MultipleContinuousUnivariate) = dist.N
Base.size(dist::MultipleContinuousUnivariate) = (dist.N,)
function Multi(dist::ContinuousUnivariateDistribution, N::Int)
    return MultipleContinuousUnivariate(dist, N)
end
function Distributions.logpdf(
    dist::MultipleContinuousUnivariate,
    x::AbstractVector{<:Real}
)
    return sum(logpdf.(dist.dist, x))
end
function Distributions.rand(rng::Random.AbstractRNG, dist::MultipleContinuousUnivariate)
    return rand(rng, dist.dist, dist.N)
end

struct MatrixContinuousUnivariate{
    Tdist <: ContinuousUnivariateDistribution,
    Tsize <: NTuple{2, Integer},
} <: ContinuousMatrixDistribution
    dist::Tdist
    S::Tsize
end
Base.size(dist::MatrixContinuousUnivariate) = dist.S
function Multi(dist::ContinuousUnivariateDistribution, N1::Integer, N2::Integer)
    return MatrixContinuousUnivariate(dist, (N1, N2))
end
function Distributions.logpdf(
    dist::MatrixContinuousUnivariate,
    x::AbstractMatrix{<:Real}
)
    return sum(logpdf.(dist.dist, x))
end
function Distributions.rand(rng::Random.AbstractRNG, dist::MatrixContinuousUnivariate)
    return rand(rng, dist.dist, dist.S)
end

# Univariate discrete

struct MultipleDiscreteUnivariate{
    Tdist <: DiscreteUnivariateDistribution,
} <: ContinuousMultivariateDistribution
    dist::Tdist
    N::Int
end
Base.length(dist::MultipleDiscreteUnivariate) = dist.N
Base.size(dist::MultipleDiscreteUnivariate) = (dist.N,)
function Multi(dist::DiscreteUnivariateDistribution, N::Int)
    MultipleDiscreteUnivariate(dist, N)
end
function Distributions.logpdf(
    dist::MultipleDiscreteUnivariate,
    x::AbstractVector{<:Integer}
)
    return sum(logpdf.(dist.dist, x))
end
function Distributions.rand(rng::Random.AbstractRNG, dist::MultipleDiscreteUnivariate)
    return rand(rng, dist.dist, dist.N)
end

struct MatrixDiscreteUnivariate{
    Tdist <: DiscreteUnivariateDistribution,
    Tsize <: NTuple{2, Integer},
} <: DiscreteMatrixDistribution
    dist::Tdist
    S::Tsize
end
Base.size(dist::MatrixDiscreteUnivariate) = dist.S
function Multi(dist::DiscreteUnivariateDistribution, N1::Integer, N2::Integer)
    return MatrixDiscreteUnivariate(dist, (N1, N2))
end
function Distributions.logpdf(
    dist::MatrixDiscreteUnivariate,
    x::AbstractMatrix{<:Real}
)
    return sum(logpdf.(dist.dist, x))
end
function Distributions.rand(rng::Random.AbstractRNG, dist::MatrixDiscreteUnivariate)
    return rand(rng, dist.dist, dist.S)
end
