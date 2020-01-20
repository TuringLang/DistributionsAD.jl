# Adapted from Distributions.jl

## Wishart

using StatsFuns: logtwo, logmvgamma

struct TuringWishart{T<:Real, ST} <: ContinuousMatrixDistribution
    df::T     # degree of freedom
    chol::ST  # the Cholesky of scale matrix
    c0::T     # the logarithm of normalizing constant in pdf
end

#### Constructors

function TuringWishart(df::T, S::AbstractMatrix{T}) where {T <: Real}
    p = size(S, 1)
    df > p - 1 || error("dpf should be greater than dim - 1.")
    C = cholesky(S)
    return TuringWishart(df, C)
end
function TuringWishart(df::T, C::Cholesky{T}) where {T <: Real}
    c0 = _wishart_c0(df, C)
    R = Base.promote_eltype(T, c0)
    return TuringWishart(R(df), C, R(c0))
end

function TuringWishart(df::Real, S::AbstractMatrix)
    T = Base.promote_eltype(df, S)
    TuringWishart(T(df), convert(AbstractArray{T}, S))
end

function _wishart_c0(df::Real, C::Cholesky)
    h_df = df / 2
    p = size(C, 1)
    h_df * (logdet(C) + p * typeof(df)(logtwo)) + logmvgamma(p, h_df)
end

#### Properties

Distributions.insupport(::Type{TuringWishart}, X::Matrix) = isposdef(X)
Distributions.insupport(d::TuringWishart, X::Matrix) = size(X) == size(d) && isposdef(X)

dim(d::TuringWishart) = size(d.chol, 1)
Base.size(d::TuringWishart) = (p = dim(d); (p, p))
Base.size(d::TuringWishart, i) = size(d)[i]
LinearAlgebra.rank(d::TuringWishart) = dim(d)

#### Statistics

Distributions.mean(d::TuringWishart) = d.df * Matrix(d.chol)

function Distributions.mode(d::TuringWishart)
    r = d.df - dim(d) - 1.0
    if r > 0.0
        return Matrix(d.chol) * r
    else
        error("mode is only defined when df > p + 1")
    end
end

function Distributions.meanlogdet(d::TuringWishart)
    p = dim(d)
    df = d.df
    v = logdet(d.chol) + p * logtwo
    for i = 1:p
        v += digamma(0.5 * (df - (i - 1)))
    end
    return v
end

function Distributions.entropy(d::TuringWishart)
    p = dim(d)
    df = d.df
    d.c0 - 0.5 * (df - p - 1) * meanlogdet(d) + 0.5 * df * p
end

#  Gupta/Nagar (1999) Theorem 3.3.15.i
function Distributions.cov(d::TuringWishart, i::Integer, j::Integer, k::Integer, l::Integer)
    S = Matrix(d.chol)
    d.df * (S[i, k] * S[j, l] + S[i, l] * S[j, k])
end

function Distributions.var(d::TuringWishart, i::Integer, j::Integer)
    S = Matrix(d.chol)
    d.df * (S[i, i] * S[j, j] + S[i, j] ^ 2)
end

#### Evaluation

function Distributions.logpdf(d::TuringWishart, X::AbstractMatrix{<:Real})
    df = d.df
    p = dim(d)
    Xcf = cholesky(X)
    return 0.5 * ((df - (p + 1)) * logdet(Xcf) - tr(d.chol \ X)) - d.c0
end

#### Sampling
function Distributions._rand!(rng::AbstractRNG, d::TuringWishart, A::AbstractMatrix)
    _wishart_genA!(rng, dim(d), d.df, A)
    unwhiten!(d.chol, A)
    A .= A * A'
end

function _wishart_genA!(rng::AbstractRNG, p::Int, df::Real, A::AbstractMatrix)
    # Generate the matrix A in the Bartlett decomposition
    #
    #   A is a lower triangular matrix, with
    #
    #       A(i, j) ~ sqrt of Chisq(df - i + 1) when i == j
    #               ~ Normal()                  when i > j
    #
    A .= zero(eltype(A))
    for i = 1:p
        @inbounds A[i,i] = rand(rng, Chi(df - i + 1.0))
    end
    for j in 1:p-1, i in j+1:p
        @inbounds A[i,j] = randn(rng)
    end
end

function unwhiten!(C::Cholesky, x::StridedVecOrMat)
    cf = C.U
    lmul!(transpose(cf), x)
end

## InverseWishart

struct TuringInverseWishart{T<:Real, ST<:AbstractMatrix{T}} <: ContinuousMatrixDistribution
    df::T     # degree of freedom
    S::ST     # Scale matrix
    c0::T     # log of normalizing constant
end

#### Constructors

function TuringInverseWishart(df::T, Ψ::AbstractMatrix{T}) where T<:Real
    p = size(Ψ, 1)
    df > p - 1 || error("df should be greater than dim - 1.")
    C = cholesky(Ψ)
    c0 = _invwishart_c0(df, C)
    R = Base.promote_eltype(T, c0)
    return TuringInverseWishart(R(df), Ψ, R(c0))
end
function TuringInverseWishart(df::Real, Ψ::AbstractMatrix{<:Real})
    T = Base.promote_eltype(df, Ψ)
    return TuringInverseWishart(T(df), convert(AbstractArray{T}, Ψ))
end
function _invwishart_c0(df::Real, C::Cholesky)
    h_df = df / 2
    p = size(C, 1)
    h_df * (p * typeof(df)(logtwo) - logdet(C)) + logmvgamma(p, h_df)
end

#### Properties

Distributions.insupport(::Type{TuringInverseWishart}, X::Matrix) = isposdef(X)
Distributions.insupport(d::TuringInverseWishart, X::Matrix) = size(X) == size(d) && isposdef(X)

dim(d::TuringInverseWishart) = size(d.S, 1)
Base.size(d::TuringInverseWishart) = (p = dim(d); (p, p))
Base.size(d::TuringInverseWishart, i) = size(d)[i]
LinearAlgebra.rank(d::TuringInverseWishart) = dim(d)

#### Statistics

function Distributions.mean(d::TuringInverseWishart)
    df = d.df
    p = dim(d)
    r = df - (p + 1)
    if r > 0.0
        return d.S * (1.0 / r)
    else
        error("mean only defined for df > p + 1")
    end
end

Distributions.mode(d::TuringInverseWishart) = d.S * inv(d.df + dim(d) + 1.0)

#  https://en.wikipedia.org/wiki/Inverse-Wishart_distribution#Moments
function Distributions.cov(d::TuringInverseWishart, i::Integer, j::Integer, k::Integer, l::Integer)
    p, ν, Ψ = (dim(d), d.df, d.S)
    ν > p + 3 || throw(ArgumentError("cov only defined for df > dim + 3"))
    inv((ν - p)*(ν - p - 3)*(ν - p - 1)^2)*(2Ψ[i,j]*Ψ[k,l] + (ν-p-1)*(Ψ[i,k]*Ψ[j,l] + Ψ[i,l]*Ψ[k,j]))
end

function Distributions.var(d::TuringInverseWishart, i::Integer, j::Integer)
    p, ν, Ψ = (dim(d), d.df, d.S)
    ν > p + 3 || throw(ArgumentError("var only defined for df > dim + 3"))
    inv((ν - p)*(ν - p - 3)*(ν - p - 1)^2)*(ν - p + 1)*Ψ[i,j]^2 + (ν - p - 1)*Ψ[i,i]*Ψ[j,j]
end

#### Evaluation

function Distributions.logpdf(d::TuringInverseWishart, X::AbstractMatrix{<:Real})
    p = dim(d)
    df = d.df
    Xcf = cholesky(X)
    # we use the fact: tr(Ψ * inv(X)) = tr(inv(X) * Ψ) = tr(X \ Ψ)
    Ψ = d.S
    -0.5 * ((df + p + 1) * logdet(Xcf) + tr(Xcf \ Ψ)) - d.c0
end


#### Sampling

Distributions._rand!(rng::AbstractRNG, d::TuringInverseWishart, A::AbstractMatrix) =
    (A .= inv(cholesky!(_rand!(rng, TuringWishart(d.df, inv(cholesky(d.S))), A))))

## Adjoints

ZygoteRules.@adjoint function Distributions.Wishart(df::Real, S::AbstractMatrix{<:Real})
    return ZygoteRules.pullback(TuringWishart, df, S)
end
ZygoteRules.@adjoint function Distributions.InverseWishart(df::Real, S::AbstractMatrix{<:Real})
    return ZygoteRules.pullback(TuringInverseWishart, df, S)
end
