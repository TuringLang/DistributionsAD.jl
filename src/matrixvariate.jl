# Adapted from Distributions.jl

## Wishart

using StatsFuns: logtwo, logmvgamma

struct TuringWishart{T<:Real, ST <: Cholesky} <: ContinuousMatrixDistribution
    df::T     # degree of freedom
    chol::ST  # the Cholesky of scale matrix
    logc0::T  # the logarithm of normalizing constant in pdf
end

#### Constructors

function TuringWishart(d::Wishart)
    return TuringWishart(d.df, getchol(d.S), d.logc0)
end
getchol(p::AbstractMatrix) = cholesky(p)
getchol(p::PDMat) = p.chol
getchol(p::PDiagMat) = Cholesky(Diagonal(sqrt.(p.diag)), 'U', 0)
getchol(p::ScalMat) = Cholesky(Diagonal(fill(sqrt(p.value), p.dim)), 'U', 0) 

function TuringWishart(df::Real, S::AbstractMatrix)
    p = size(S, 1)
    df > p - 1 || error("dpf should be greater than dim - 1.")
    C = getchol(S)
    return TuringWishart(df, C)
end

function TuringWishart(df::T, C::Cholesky) where {T <: Real}
    p = size(C, 1)
    df > p - 1 || error("dpf should be greater than dim - 1.")
    logc0 = _wishart_logc0(df, C)
    R = Base.promote_eltype(T, logc0)
    return TuringWishart(R(df), C, R(logc0))
end

function _wishart_logc0(df::Real, C::Cholesky)
    h_df = df / 2
    p = size(C, 1)
    -h_df * (logdet(C) + p * float(typeof(df))(logtwo)) - logmvgamma(p, h_df)
end

#### Properties

Distributions.insupport(::Type{TuringWishart}, X::Matrix) = isposdef(X)
Distributions.insupport(d::TuringWishart, X::Matrix) = size(X) == size(d) && isposdef(X)

Distributions.dim(d::TuringWishart) = size(d.chol, 1)
Base.size(d::TuringWishart) = (p = Distributions.dim(d); (p, p))
Base.size(d::TuringWishart, i) = size(d)[i]
LinearAlgebra.rank(d::TuringWishart) = Distributions.dim(d)

#### Statistics

Distributions.mean(d::TuringWishart) = d.df * Matrix(d.chol)

function Distributions.mode(d::TuringWishart)
    r = d.df - Distributions.dim(d) - 1.0
    if r > 0.0
        return Matrix(d.chol) * r
    else
        error("mode is only defined when df > p + 1")
    end
end

function Distributions.meanlogdet(d::TuringWishart)
    p = Distributions.dim(d)
    df = d.df
    v = logdet(d.chol) + p * logtwo
    for i = 1:p
        v += digamma(0.5 * (df - (i - 1)))
    end
    return v
end

function Distributions.entropy(d::TuringWishart)
    p = Distributions.dim(d)
    df = d.df
    return -d.logc0 - 0.5 * (df - p - 1) * Distributions.meanlogdet(d) + 0.5 * df * p
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

function Distributions._logpdf(d::TuringWishart, X::AbstractMatrix{<:Real})
    df = d.df
    p = Distributions.dim(d)
    Xcf = cholesky(X)
    return ((df - (p + 1)) * logdet(Xcf) - tr(d.chol \ X)) / 2 + d.logc0
end

function Distributions.logpdf(d::TuringWishart, X::AbstractArray{<:AbstractMatrix{<:Real}})
    return map(x -> logpdf(d, x), X)
end
function Distributions.logpdf(d::TuringWishart, X::AbstractArray{<:Matrix{<:Real}})
    return map(x -> logpdf(d, x), X)
end

#### Sampling
function Distributions._rand!(rng::AbstractRNG, d::TuringWishart, A::AbstractMatrix)
    Distributions._wishart_genA!(rng, Distributions.dim(d), d.df, A)
    unwhiten!(d.chol, A)
    A .= A * A'
end

function unwhiten!(C::Cholesky, x::StridedVecOrMat)
    cf = C.U
    lmul!(transpose(cf), x)
end

## InverseWishart

struct TuringInverseWishart{T<:Real, ST<:AbstractMatrix} <: ContinuousMatrixDistribution
    df::T     # degree of freedom
    S::ST     # Scale matrix
    logc0::T  # log of normalizing constant
end

#### Constructors

function TuringInverseWishart(d::InverseWishart)
    d = TuringInverseWishart(d.df, getmatrix(d.Ψ), d.logc0)
end
getmatrix(p::AbstractMatrix) = p
getmatrix(p::PDMat) = p.mat
getmatrix(p::PDiagMat) = Diagonal(p.diag)
getmatrix(p::ScalMat) = Diagonal(fill(p.value, p.dim))

function TuringInverseWishart(df::T, x::AbstractMatrix) where T<:Real
    Ψ = getmatrix(x)
    p = size(Ψ, 1)
    df > p - 1 || error("df should be greater than dim - 1.")
    C = getchol(x)
    logc0 = _invwishart_logc0(df, C)
    R = Base.promote_eltype(T, logc0)
    return TuringInverseWishart(R(df), Ψ, R(logc0))
end
function _invwishart_logc0(df::Real, C::Cholesky)
    h_df = df / 2
    p = size(C, 1)
    -h_df * (p * float(typeof(df))(logtwo) - logdet(C)) - logmvgamma(p, h_df)
end

#### Properties

Distributions.insupport(::Type{TuringInverseWishart}, X::Matrix) = isposdef(X)
Distributions.insupport(d::TuringInverseWishart, X::Matrix) = size(X) == size(d) && isposdef(X)

Distributions.dim(d::TuringInverseWishart) = size(d.S, 1)
Base.size(d::TuringInverseWishart) = (p = Distributions.dim(d); (p, p))
Base.size(d::TuringInverseWishart, i) = size(d)[i]
LinearAlgebra.rank(d::TuringInverseWishart) = Distributions.dim(d)

#### Statistics

function Distributions.mean(d::TuringInverseWishart)
    df = d.df
    p = Distributions.dim(d)
    r = df - (p + 1)
    if r > 0.0
        return d.S * (1.0 / r)
    else
        error("mean only defined for df > p + 1")
    end
end

Distributions.mode(d::TuringInverseWishart) = d.S * inv(d.df + Distributions.dim(d) + 1.0)

#  https://en.wikipedia.org/wiki/Inverse-Wishart_distribution#Moments
function Distributions.cov(d::TuringInverseWishart, i::Integer, j::Integer, k::Integer, l::Integer)
    p, ν, Ψ = (Distributions.dim(d), d.df, d.S)
    ν > p + 3 || throw(ArgumentError("cov only defined for df > dim + 3"))
    inv((ν - p)*(ν - p - 3)*(ν - p - 1)^2)*(2Ψ[i,j]*Ψ[k,l] + (ν-p-1)*(Ψ[i,k]*Ψ[j,l] + Ψ[i,l]*Ψ[k,j]))
end

function Distributions.var(d::TuringInverseWishart, i::Integer, j::Integer)
    p, ν, Ψ = (Distributions.dim(d), d.df, d.S)
    ν > p + 3 || throw(ArgumentError("var only defined for df > dim + 3"))
    inv((ν - p)*(ν - p - 3)*(ν - p - 1)^2)*((ν - p + 1)*Ψ[i,j]^2 + (ν - p - 1)*Ψ[i,i]*Ψ[j,j])
end

#### Evaluation

function Distributions._logpdf(d::TuringInverseWishart, X::AbstractMatrix{<:Real})
    p = Distributions.dim(d)
    df = d.df
    Xcf = cholesky(X)
    # we use the fact: tr(Ψ * inv(X)) = tr(inv(X) * Ψ) = tr(X \ Ψ)
    Ψ = d.S
    return -((df + p + 1) * logdet(Xcf) + tr(Xcf \ Ψ)) / 2 + d.logc0
end

function Distributions.logpdf(d::TuringInverseWishart, X::AbstractArray{<:AbstractMatrix{<:Real}})
    return map(x -> logpdf(d, x), X)
end
function Distributions.logpdf(d::TuringInverseWishart, X::AbstractArray{<:Matrix{<:Real}})
    return map(x -> logpdf(d, x), X)
end

#### Sampling

function Distributions._rand!(rng::AbstractRNG, d::TuringInverseWishart, A::AbstractMatrix)
    X = Distributions._rand!(rng, TuringWishart(d.df, inv(cholesky(d.S))), A)
    A .= inv(cholesky!(X))
end

# TODO: Remove when available in Distributions
for T in (:MatrixBeta, :MatrixNormal, :Wishart, :InverseWishart,
          :TuringWishart, :TuringInverseWishart,
          :VectorOfMultivariate, :MatrixOfUnivariate)
    @eval begin
        Distributions.loglikelihood(d::$T, X::AbstractMatrix{<:Real}) = logpdf(d, X)
        function Distributions.loglikelihood(d::$T, X::AbstractArray{<:Real,3})
            (size(X, 1), size(X, 2)) == size(d) || throw(DimensionMismatch("Inconsistent array dimensions."))
            return sum(i -> _logpdf(d, view(X, :, :, i)), axes(X, 3))
        end
        function Distributions.loglikelihood(
            d::$T,
            X::AbstractArray{<:AbstractMatrix{<:Real}},
        )
            return sum(x -> logpdf(d, x), X)
        end
    end
end
