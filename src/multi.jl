# Univariate

const FillVectorOfUnivariate{
    S <: ValueSupport,
    T <: UnivariateDistribution{S},
    Tdists <: Fill{T, 1},
} = VectorOfUnivariate{S, T, Tdists}

function FillDist(dist::UnivariateDistribution, N::Int)
    return Product(Fill(dist, N))
end
FillDist(d::Normal, N::Int) = MvNormal(fill(d.μ, N), d.σ)
function Distributions.logpdf(
    dist::FillVectorOfUnivariate,
    x::AbstractVector{<:Real},
)
    return _flat_logpdf(dist.v.value, x)
end
function Distributions.logpdf(
    dist::FillVectorOfUnivariate,
    x::AbstractMatrix{<:Real},
)
    return _flat_logpdf_mat(dist.v.value, x)
end
function _flat_logpdf(dist, x)
    if toflatten(dist)
        f, args = flatten(dist)
        return sum(f.(args..., x))
    else
        return sum(logpdf.(dist, x))
    end
end
function _flat_logpdf_mat(dist, x)
    if toflatten(dist)
        f, args = flatten(dist)
        return vec(sum(f.(args..., x), dims = 1))
    else
        return vec(sum(logpdf.(dist, x), dims = 1))
    end
end

const FillMatrixOfUnivariate{
    S <: ValueSupport,
    T <: UnivariateDistribution{S},
    Tdists <: Fill{T, 2},
} = MatrixOfUnivariate{S, T, Tdists}

function FillDist(dist::UnivariateDistribution, N1::Integer, N2::Integer)
    return MatrixOfUnivariate(Fill(dist, N1, N2))
end
function Distributions.logpdf(dist::FillMatrixOfUnivariate, x::AbstractMatrix{<:Real})
    return _flat_logpdf(dist.dists.value, x)
end
function Distributions.rand(rng::Random.AbstractRNG, dist::FillMatrixOfUnivariate)
    return rand(rng, dist.dists.value, length.(dist.dists.axes))
end

# Multivariate

const FillVectorOfMultivariate{
    S <: ValueSupport,
    T <: MultivariateDistribution{S},
    Tdists <: Fill{T, 1},
} = VectorOfMultivariate{S, T, Tdists}

function FillDist(dist::MultivariateDistribution, N::Int)
    return VectorOfMultivariate(Fill(dist, N))
end
function Distributions.logpdf(
    dist::FillVectorOfMultivariate,
    x::AbstractMatrix{<:Real},
)
    return sum(logpdf(dist.dists.value, x))
end
function Distributions.rand(rng::Random.AbstractRNG, dist::FillVectorOfMultivariate)
    return rand(rng, dist.dists.value, length.(dist.dists.axes))
end
