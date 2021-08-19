module ReverseDiffX

export NotTracked

using LinearAlgebra
using ..ReverseDiff
using StaticArrays
using Distributions
using PDMats
using ..ForwardDiff

using Base.Broadcast: BroadcastStyle, ArrayStyle, Broadcasted, broadcasted
using ..ForwardDiff: Dual
using ..ReverseDiff: SpecialInstruction, value, value!, deriv, track, record!,
                     tape, unseed!, @grad, TrackedReal, TrackedVector,
                     TrackedMatrix, TrackedArray
using ..DistributionsAD: DistributionsAD


import SpecialFunctions, NaNMath
import ..DistributionsAD: turing_chol, symm_turing_chol, _mv_categorical_logpdf, adapt_randn,
    simplex_logpdf
import Base.Broadcast: materialize
import StatsFuns: logsumexp

const TrackedVecOrMat{V,D} = Union{TrackedVector{V,D},TrackedMatrix{V,D}}
const RDBroadcasted{F, T} = Broadcasted{<:Any, <:Any, F, T}

import Random

import Distributions: logpdf,
                      _logpdf,
                      loglikelihood,
                      Gamma,
                      MvNormal,
                      MvLogNormal,
                      Dirichlet,
                      Wishart,
                      InverseWishart,
                      PoissonBinomial,
                      isprobvec

using ..DistributionsAD: TuringPoissonBinomial,
                         TuringMvNormal,
                         TuringMvLogNormal,
                         TuringWishart,
                         TuringInverseWishart,
                         TuringDirichlet,
                         TuringScalMvNormal,
                         TuringDiagMvNormal,
                         TuringDenseMvNormal,
                         VectorOfMultivariate,
                         FillVectorOfMultivariate

###############
## logsumexp ##
###############

logsumexp(x::TrackedArray; dims=:) = track(logsumexp, x, dims = dims)
@grad function logsumexp(x::AbstractArray; dims)
    x_value = value(x)
    lse = logsumexp(x_value; dims=dims)
    return lse, Δ -> (Δ .* exp.(x_value .- lse),)
end

############
## linalg ##
############

function LinearAlgebra.cholesky(A::Symmetric{<:Any, <:TrackedMatrix}; check=true)
    uplo = A.uplo == 'U' ? (:U) : (:L)
    factors, info = symm_turing_chol(parent(A), check, uplo)
    return Cholesky{eltype(factors), typeof(factors)}(factors, 'U', info)
end
function LinearAlgebra.cholesky(A::TrackedMatrix; check=true)
    factors, info = turing_chol(A, check)
    return Cholesky{eltype(factors), typeof(factors)}(factors, 'U', info)
end

function symm_turing_chol(x::TrackedArray{V,D}, check, uplo) where {V,D}
    tp = tape(x)
    x_value = value(x)
    (factors,info), back = DistributionsAD.symm_turing_chol_back(x_value, check, uplo)
    C = Cholesky{eltype(factors), typeof(factors)}(factors, 'U', info)
    out = track(C.factors, D, tp)
    record!(tp, SpecialInstruction, symm_turing_chol, (x, check, uplo), out, (back, issuccess(C)))
    return out, C.info
end
function turing_chol(x::TrackedArray{V,D}, check) where {V,D}
    tp = tape(x)
    x_value = value(x)
    (factors,info), back = DistributionsAD.turing_chol_back(x_value, check)
    C = Cholesky{eltype(factors), typeof(factors)}(factors, 'U', info)
    out = track(C.factors, D, tp)
    record!(tp, SpecialInstruction, turing_chol, (x, check), out, (back, issuccess(C)))
    return out, C.info
end

for f in (:turing_chol, :symm_turing_chol)
    @eval begin
        @noinline function ReverseDiff.special_reverse_exec!(
            instruction::SpecialInstruction{typeof($f)},
        )
            output = instruction.output
            instruction.cache[2] || throw(PosDefException(C.info))
            input = instruction.input
            input_deriv = deriv(input[1])
            P = instruction.cache[1]
            input_deriv .+= P((factors = deriv(output),))[1]
            unseed!(output)
            return nothing
        end
    end
end

@noinline function ReverseDiff.special_forward_exec!(
    instruction::SpecialInstruction{typeof(turing_chol)},
)
    output, input = instruction.output, instruction.input
    factors = turing_chol(value.(input)...)[1]
    value!(output, factors)
    return nothing
end

@noinline function ReverseDiff.special_forward_exec!(
    instruction::SpecialInstruction{typeof(symm_turing_chol)},
)
    output, input = instruction.output, instruction.input
    factors = symm_turing_chol(value.(input)...)[1]
    value!(output, factors)
    return nothing
end

adapt_randn(rng::Random.AbstractRNG, x::TrackedArray, dims...) = adapt_randn(rng, value(x), dims...)

function PoissonBinomial(p::TrackedArray{<:Real}; check_args=true)
    return TuringPoissonBinomial(p; check_args = check_args)
end

Gamma(α::TrackedReal, θ::Real; check_args=true) = pgamma(α, θ, check_args = check_args)
Gamma(α::Real, θ::TrackedReal; check_args=true) = pgamma(α, θ, check_args = check_args)
Gamma(α::TrackedReal, θ::TrackedReal; check_args=true) = pgamma(α, θ, check_args = check_args)
pgamma(α, θ; check_args=true) = Gamma(promote(α, θ)..., check_args = check_args)
Gamma(α::T; check_args=true) where {T <: TrackedReal} = Gamma(α, one(T), check_args = check_args)
function Gamma(α::T, θ::T; check_args=true) where {T <: TrackedReal}
    check_args && Distributions.@check_args(Gamma, α > zero(α) && θ > zero(θ))
    return Gamma{T}(α, θ)
end

# Work around to stop TrackedReal of Inf and -Inf from producing NaN in the derivative
function Base.minimum(d::LocationScale{T}) where {T <: TrackedReal}
    if isfinite(minimum(d.ρ))
        return d.μ + d.σ * minimum(d.ρ)
    else
        return convert(T, ReverseDiff.@skip(minimum)(d.ρ))
    end
end
function Base.maximum(d::LocationScale{T}) where {T <: TrackedReal}
    if isfinite(minimum(d.ρ))
        return d.μ + d.σ * maximum(d.ρ)
    else
        return convert(T, ReverseDiff.@skip(maximum)(d.ρ))
    end
end

## MvNormal

for (f, T) in (
    (:_logpdf, :TrackedVector),
    (:logpdf, :TrackedMatrix),
    (:loglikelihood, :TrackedMatrix),
)
    @eval begin
        function $f(d::MvNormal{<:Real,<:PDMats.ScalMat}, x::$T{<:Real})
            return $f(TuringScalMvNormal(d.μ, sqrt(d.Σ.value)), x)
        end
        function $f(d::MvNormal{<:Real,<:PDMats.PDiagMat}, x::$T{<:Real})
            return $f(TuringDiagMvNormal(d.μ, sqrt.(d.Σ.diag)), x)
        end
        function $f(d::MvNormal{<:Real,<:PDMats.PDMat}, x::$T{<:Real})
            return $f(TuringDenseMvNormal(d.μ, d.Σ.chol), x)
        end

        function $f(d::MvLogNormal{<:Real,<:PDMats.ScalMat}, x::$T{<:Real})
            return $f(
                TuringMvLogNormal(TuringScalMvNormal(d.normal.μ, sqrt(d.normal.Σ.value))),
                x,
            )
        end
        function $f(d::MvLogNormal{<:Real,<:PDMats.PDiagMat}, x::$T{<:Real})
            return $f(
                TuringMvLogNormal(TuringDiagMvNormal(d.normal.μ, sqrt.(d.normal.Σ.diag))),
                x,
            )
        end
        function $f(d::MvLogNormal{<:Real,<:PDMats.PDMat}, x::$T{<:Real})
            return $f(
                TuringMvLogNormal(TuringDenseMvNormal(d.normal.μ, d.normal.Σ.chol)),
                x,
            )
        end
    end
end

# zero mean, dense covariance
MvNormal(A::TrackedMatrix) = TuringMvNormal(A)

# zero mean, diagonal covariance
MvNormal(σ::TrackedVector) = TuringMvNormal(σ)

# dense mean, dense covariance
MvNormal(m::AbstractVector{<:Real}, A::TrackedMatrix{<:Real}) = TuringMvNormal(m, A)
MvNormal(m::TrackedVector{<:Real}, A::Matrix{<:Real}) = TuringMvNormal(m, A)
MvNormal(m::TrackedVector{<:Real}, A::TrackedMatrix{<:Real}) = TuringMvNormal(m, A)

# dense mean, diagonal covariance
function MvNormal(
    m::TrackedVector{<:Real},
    D::Diagonal{<:TrackedReal, <:TrackedVector{<:Real}},
)
    return TuringMvNormal(m, D)
end
function MvNormal(
    m::AbstractVector{<:Real},
    D::Diagonal{<:TrackedReal, <:TrackedVector{<:Real}},
)
    return TuringMvNormal(m, D)
end
function MvNormal(
    m::TrackedVector{<:Real},
    D::Diagonal{T, <:AbstractVector{T}} where {T<:Real},
)
    return TuringMvNormal(m, D)
end

# dense mean, diagonal covariance
MvNormal(m::TrackedVector{<:Real}, σ::TrackedVector{<:Real}) = TuringMvNormal(m, σ)
MvNormal(m::TrackedVector{<:Real}, σ::AbstractVector{<:Real}) = TuringMvNormal(m, σ)
MvNormal(m::TrackedVector{<:Real}, σ::Vector{<:Real}) = TuringMvNormal(m, σ)
MvNormal(m::AbstractVector{<:Real}, σ::TrackedVector{<:Real}) = TuringMvNormal(m, σ)

# dense mean, constant variance
MvNormal(m::TrackedVector{<:Real}, σ::TrackedReal) = TuringMvNormal(m, σ)
MvNormal(m::TrackedVector{<:Real}, σ::Real) = TuringMvNormal(m, σ)
MvNormal(m::AbstractVector{<:Real}, σ::TrackedReal) = TuringMvNormal(m, σ)

# dense mean, constant variance
function MvNormal(m::TrackedVector{<:Real}, A::UniformScaling{<:TrackedReal})
    return TuringMvNormal(m, A)
end
function MvNormal(m::AbstractVector{<:Real}, A::UniformScaling{<:TrackedReal})
    return TuringMvNormal(m, A)
end
function MvNormal(m::TrackedVector{<:Real}, A::UniformScaling{<:Real})
    return TuringMvNormal(m, A)
end

# zero mean,, constant variance
MvNormal(d::Int, σ::TrackedReal) = TuringMvNormal(d, σ)

# zero mean, dense covariance
MvLogNormal(A::TrackedMatrix) = TuringMvLogNormal(TuringMvNormal(A))

# zero mean, diagonal covariance
MvLogNormal(σ::TrackedVector) = TuringMvLogNormal(TuringMvNormal(σ))

# dense mean, dense covariance
MvLogNormal(m::TrackedVector{<:Real}, A::TrackedMatrix{<:Real}) = TuringMvLogNormal(TuringMvNormal(m, A))
MvLogNormal(m::TrackedVector{<:Real}, A::Matrix{<:Real}) = TuringMvLogNormal(TuringMvNormal(m, A))
MvLogNormal(m::AbstractVector{<:Real}, A::TrackedMatrix{<:Real}) = TuringMvLogNormal(TuringMvNormal(m, A))

# dense mean, diagonal covariance
function MvLogNormal(
    m::TrackedVector{<:Real},
    D::Diagonal{<:TrackedReal, <:TrackedVector{<:Real}},
)
    return TuringMvLogNormal(TuringMvNormal(m, D))
end
function MvLogNormal(
    m::AbstractVector{<:Real},
    D::Diagonal{<:TrackedReal, <:TrackedVector{<:Real}},
)
    return TuringMvLogNormal(TuringMvNormal(m, D))
end
function MvLogNormal(
    m::TrackedVector{<:Real},
    D::Diagonal{T, <:AbstractVector{T}} where {T<:Real},
)
    return TuringMvLogNormal(TuringMvNormal(m, D))
end

# dense mean, diagonal covariance
MvLogNormal(m::TrackedVector{<:Real}, σ::TrackedVector{<:Real}) = TuringMvLogNormal(TuringMvNormal(m, σ))
MvLogNormal(m::TrackedVector{<:Real}, σ::AbstractVector{<:Real}) = TuringMvLogNormal(TuringMvNormal(m, σ))
MvLogNormal(m::TrackedVector{<:Real}, σ::Vector{<:Real}) = TuringMvLogNormal(TuringMvNormal(m, σ))
MvLogNormal(m::AbstractVector{<:Real}, σ::TrackedVector{<:Real}) = TuringMvLogNormal(TuringMvNormal(m, σ))

# dense mean, constant variance
function MvLogNormal(m::TrackedVector{<:Real}, σ::TrackedReal)
    return TuringMvLogNormal(TuringMvNormal(m, σ))
end
function MvLogNormal(m::TrackedVector{<:Real}, σ::Real)
    return TuringMvLogNormal(TuringMvNormal(m, σ))
end
function MvLogNormal(m::AbstractVector{<:Real}, σ::TrackedReal)
    return TuringMvLogNormal(TuringMvNormal(m, σ))
end

# dense mean, constant variance
function MvLogNormal(m::TrackedVector{<:Real}, A::UniformScaling{<:TrackedReal})
    return TuringMvLogNormal(TuringMvNormal(m, A))
end
function MvLogNormal(m::AbstractVector{<:Real}, A::UniformScaling{<:TrackedReal})
    return TuringMvLogNormal(TuringMvNormal(m, A))
end
function MvLogNormal(m::TrackedVector{<:Real}, A::UniformScaling{<:Real})
    return TuringMvLogNormal(TuringMvNormal(m, A))
end

# zero mean,, constant variance
MvLogNormal(d::Int, σ::TrackedReal) = TuringMvLogNormal(TuringMvNormal(d, σ))

# Dirichlet

Dirichlet(alpha::AbstractVector{<:TrackedReal}) = TuringDirichlet(alpha)
Dirichlet(d::Integer, alpha::TrackedReal) = TuringDirichlet(d, alpha)

function _logpdf(d::Dirichlet, x::AbstractVector{<:TrackedReal})
    return _logpdf(TuringDirichlet(d), x)
end
function logpdf(d::Dirichlet, x::AbstractMatrix{<:TrackedReal})
    return logpdf(TuringDirichlet(d), x)
end
function loglikelihood(d::Dirichlet, x::AbstractMatrix{<:TrackedReal})
    return loglikelihood(TuringDirichlet(d), x)
end

for func_header in [
    :(simplex_logpdf(alpha::AbstractVector{<:TrackedReal}, lmnB::Real, x::AbstractVector)),
    :(simplex_logpdf(alpha::AbstractVector, lmnB::TrackedReal, x::AbstractVector)),
    :(simplex_logpdf(alpha::AbstractVector, lmnB::Real, x::AbstractVector{<:TrackedReal})),
    :(simplex_logpdf(alpha::AbstractVector{<:TrackedReal}, lmnB::TrackedReal, x::AbstractVector)),
    :(simplex_logpdf(alpha::AbstractVector, lmnB::TrackedReal, x::AbstractVector{<:TrackedReal})),
    :(simplex_logpdf(alpha::AbstractVector{<:TrackedReal}, lmnB::Real, x::AbstractVector{<:TrackedReal})),
    :(simplex_logpdf(alpha::AbstractVector{<:TrackedReal}, lmnB::TrackedReal, x::AbstractVector{<:TrackedReal})),

    :(simplex_logpdf(alpha::AbstractVector{<:TrackedReal}, lmnB::Real, x::AbstractMatrix)),
    :(simplex_logpdf(alpha::AbstractVector, lmnB::TrackedReal, x::AbstractMatrix)),
    :(simplex_logpdf(alpha::AbstractVector, lmnB::Real, x::AbstractMatrix{<:TrackedReal})),
    :(simplex_logpdf(alpha::AbstractVector{<:TrackedReal}, lmnB::TrackedReal, x::AbstractMatrix)),
    :(simplex_logpdf(alpha::AbstractVector, lmnB::TrackedReal, x::AbstractMatrix{<:TrackedReal})),
    :(simplex_logpdf(alpha::AbstractVector{<:TrackedReal}, lmnB::Real, x::AbstractMatrix{<:TrackedReal})),
    :(simplex_logpdf(alpha::AbstractVector{<:TrackedReal}, lmnB::TrackedReal, x::AbstractMatrix{<:TrackedReal})),
]
    @eval $func_header = track(simplex_logpdf, alpha, lmnB, x)
end
@grad function simplex_logpdf(alpha, lmnB, x::AbstractVector)
    _alpha = value(alpha)
    _lmnB = value(lmnB)
    _x = value(x)
    simplex_logpdf(_alpha, _lmnB, _x), Δ -> begin
        (Δ .* log.(_x), -Δ, Δ .* (_alpha .- 1) ./ _x)
    end
end
@grad function simplex_logpdf(alpha, lmnB, x::AbstractMatrix)
    _alpha = value(alpha)
    _lmnB = value(lmnB)
    _x = value(x)
    simplex_logpdf(_alpha, _lmnB, _x), Δ -> begin
        (log.(_x) * Δ, -sum(Δ), ((_alpha .- 1) ./ _x) * Diagonal(Δ))
    end
end

Distributions.Wishart(df::TrackedReal, S::Matrix{<:Real}) = TuringWishart(df, S)
Distributions.Wishart(df::TrackedReal, S::AbstractMatrix{<:Real}) = TuringWishart(df, S)
Distributions.Wishart(df::Real, S::AbstractMatrix{<:TrackedReal}) = TuringWishart(df, S)
Distributions.Wishart(df::TrackedReal, S::AbstractMatrix{<:TrackedReal}) = TuringWishart(df, S)
Distributions.Wishart(df::Real, S::TrackedMatrix) = TuringWishart(df, S)
Distributions.Wishart(df::TrackedReal, S::TrackedMatrix) = TuringWishart(df, S)
Distributions.Wishart(df::Real, S::AbstractPDMat{<:TrackedReal}) = TuringWishart(df, S)
Distributions.Wishart(df::TrackedReal, S::AbstractPDMat{<:TrackedReal}) = TuringWishart(df, S)

Distributions.InverseWishart(df::TrackedReal, S::Matrix{<:Real}) = TuringInverseWishart(df, S)
Distributions.InverseWishart(df::TrackedReal, S::AbstractMatrix{<:Real}) = TuringInverseWishart(df, S)
Distributions.InverseWishart(df::Real, S::AbstractMatrix{<:TrackedReal}) = TuringInverseWishart(df, S)
Distributions.InverseWishart(df::TrackedReal, S::AbstractMatrix{<:TrackedReal}) = TuringInverseWishart(df, S)
Distributions.InverseWishart(df::Real, S::TrackedMatrix) = TuringInverseWishart(df, S)
Distributions.InverseWishart(df::TrackedReal, S::TrackedMatrix) = TuringInverseWishart(df, S)
Distributions.InverseWishart(df::Real, S::AbstractPDMat{<:TrackedReal}) = TuringInverseWishart(df, S)
Distributions.InverseWishart(df::TrackedReal, S::AbstractPDMat{<:TrackedReal}) = TuringInverseWishart(df, S)

function _logpdf(d::Wishart, X::TrackedMatrix)
    return _logpdf(TuringWishart(d), X)
end
function logpdf(d::Wishart, X::AbstractArray{<:TrackedMatrix})
    return logpdf(TuringWishart(d), X)
end
function loglikelihood(d::Wishart, X::AbstractArray{<:TrackedMatrix})
    return loglikelihood(TuringWishart(d), X)
end

function _logpdf(d::InverseWishart, X::TrackedMatrix)
    return _logpdf(TuringInverseWishart(d), X)
end
function logpdf(d::InverseWishart, X::AbstractArray{<:TrackedMatrix})
    return logpdf(TuringInverseWishart(d), X)
end
function loglikelihood(d::InverseWishart, X::AbstractArray{<:TrackedMatrix})
    return loglikelihood(TuringInverseWishart(d), X)
end

# isprobvec

function isprobvec(p::TrackedArray{<:Real})
    pdata = value(p)
    all(x -> x ≥ zero(x), pdata) && isapprox(sum(pdata), one(eltype(pdata)), atol = 1e-6)
end
function isprobvec(p::SubArray{<:TrackedReal, 1, <:TrackedArray{<:Real}})
    pdata = value(p)
    all(x -> x ≥ zero(x), pdata) && isapprox(sum(pdata), one(eltype(pdata)), atol = 1e-6)
end
end
