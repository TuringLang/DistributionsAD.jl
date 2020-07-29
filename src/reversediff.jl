module ReverseDiffX

export NotTracked

using LinearAlgebra
using ..ReverseDiff
using StaticArrays
using Distributions
using PDMats
using ForwardDiff

using Base.Broadcast: BroadcastStyle, ArrayStyle, Broadcasted, broadcasted
using ForwardDiff: Dual
using ..ReverseDiff: SpecialInstruction, value, value!, deriv, track, record!,
                     tape, unseed!, @grad, TrackedReal, TrackedVector,
                     TrackedMatrix, TrackedArray
using ..DistributionsAD: DistributionsAD


import SpecialFunctions, NaNMath
import ..DistributionsAD: turing_chol, symm_turing_chol, _mv_categorical_logpdf
import Base.Broadcast: materialize
import StatsFuns: logsumexp

const TrackedVecOrMat{V,D} = Union{TrackedVector{V,D},TrackedMatrix{V,D}}
const RDBroadcasted{F, T} = Broadcasted{<:Any, <:Any, F, T}

import Distributions: logpdf,
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
                         TuringDenseMvNormal

include("reversediffx.jl")

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

for T in (:TrackedVector, :TrackedMatrix)
    @eval begin
        function logpdf(d::MvNormal{<:Any, <:PDMats.ScalMat}, x::$T)
            logpdf(TuringScalMvNormal(d.μ, sqrt(d.Σ.value)), x)
        end
        function logpdf(d::MvNormal{<:Any, <:PDMats.PDiagMat}, x::$T)
            logpdf(TuringDiagMvNormal(d.μ, sqrt.(d.Σ.diag)), x)
        end
        function logpdf(d::MvNormal{<:Any, <:PDMats.PDMat}, x::$T)
            logpdf(TuringDenseMvNormal(d.μ, d.Σ.chol), x)
        end
        
        function logpdf(d::MvLogNormal{<:Any, <:PDMats.ScalMat}, x::$T)
            logpdf(TuringMvLogNormal(TuringScalMvNormal(d.normal.μ, sqrt(d.normal.Σ.value))), x)
        end
        function logpdf(d::MvLogNormal{<:Any, <:PDMats.PDiagMat}, x::$T)
            logpdf(TuringMvLogNormal(TuringDiagMvNormal(d.normal.μ, sqrt.(d.normal.Σ.diag))), x)
        end
        function logpdf(d::MvLogNormal{<:Any, <:PDMats.PDMat}, x::$T)
            logpdf(TuringMvLogNormal(TuringDenseMvNormal(d.normal.μ, d.normal.Σ.chol)), x)
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

Dirichlet(alpha::TrackedVector) = TuringDirichlet(alpha)
Dirichlet(d::Integer, alpha::TrackedReal) = TuringDirichlet(d, alpha)
for func_header in [
    :(simplex_logpdf(alpha::TrackedVector, lmnB::Real, x::AbstractVector)),
    :(simplex_logpdf(alpha::AbstractVector, lmnB::TrackedReal, x::AbstractVector)),
    :(simplex_logpdf(alpha::AbstractVector, lmnB::Real, x::TrackedVector)),
    :(simplex_logpdf(alpha::TrackedVector, lmnB::TrackedReal, x::AbstractVector)),
    :(simplex_logpdf(alpha::AbstractVector, lmnB::TrackedReal, x::TrackedVector)),
    :(simplex_logpdf(alpha::TrackedVector, lmnB::Real, x::TrackedVector)),
    :(simplex_logpdf(alpha::TrackedVector, lmnB::TrackedReal, x::TrackedVector)),

    :(simplex_logpdf(alpha::TrackedVector, lmnB::Real, x::AbstractMatrix)),
    :(simplex_logpdf(alpha::AbstractVector, lmnB::TrackedReal, x::AbstractMatrix)),
    :(simplex_logpdf(alpha::AbstractVector, lmnB::Real, x::TrackedMatrix)),
    :(simplex_logpdf(alpha::TrackedVector, lmnB::TrackedReal, x::AbstractMatrix)),
    :(simplex_logpdf(alpha::AbstractVector, lmnB::TrackedReal, x::TrackedMatrix)),
    :(simplex_logpdf(alpha::TrackedVector, lmnB::Real, x::TrackedMatrix)),
    :(simplex_logpdf(alpha::TrackedVector, lmnB::TrackedReal, x::TrackedMatrix)),
]
    @eval $func_header = track(simplex_logpdf, alpha, lmnB, x)
end
@grad function simplex_logpdf(alpha, lmnB, x::AbstractVector)
    simplex_logpdf(value(alpha), value(lmnB), value(x)), Δ -> begin
        (Δ .* log.(value(x)), -Δ, Δ .* (value(alpha) .- 1))
    end
end
@grad function simplex_logpdf(alpha, lmnB, x::AbstractMatrix)
    simplex_logpdf(value(alpha), value(lmnB), value(x)), Δ -> begin
        (log.(value(x)) * Δ, -sum(Δ), repeat(value(alpha) .- 1, 1, size(x, 2)) * Diagonal(Δ))
    end
end

function logpdf(d::MatrixBeta, X::AbstractArray{<:TrackedMatrix{<:Real}})
    return map(x -> logpdf(d, x), X)
end

Wishart(df::TrackedReal, S::Matrix{<:Real}) = TuringWishart(df, S)
Wishart(df::TrackedReal, S::AbstractMatrix{<:Real}) = TuringWishart(df, S)
Wishart(df::Real, S::TrackedMatrix) = TuringWishart(df, S)
Wishart(df::TrackedReal, S::TrackedMatrix) = TuringWishart(df, S)

InverseWishart(df::TrackedReal, S::Matrix{<:Real}) = TuringInverseWishart(df, S)
InverseWishart(df::TrackedReal, S::AbstractMatrix{<:Real}) = TuringInverseWishart(df, S)
InverseWishart(df::Real, S::TrackedMatrix) = TuringInverseWishart(df, S)
InverseWishart(df::TrackedReal, S::TrackedMatrix) = TuringInverseWishart(df, S)

function logpdf(d::Wishart, X::TrackedMatrix)
    return logpdf(TuringWishart(d), X)
end
function logpdf(d::Wishart, X::AbstractArray{<:TrackedMatrix})
    return logpdf(TuringWishart(d), X)
end

function logpdf(d::InverseWishart, X::TrackedMatrix)
    return logpdf(TuringInverseWishart(d), X)
end
function logpdf(d::InverseWishart, X::AbstractArray{<:TrackedMatrix})
    return logpdf(TuringInverseWishart(d), X)
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

_mv_categorical_logpdf(ps::TrackedMatrix, x) = track(_mv_categorical_logpdf, ps, x)
@grad function _mv_categorical_logpdf(ps, x)
    ps_data = value(ps)
    probs = view(ps_data, x, :)
    ps_grad = zero(ps_data)
    sum(log, probs), Δ -> begin
        ps_grad .= 0
        ps_grad[x,:] .= Δ ./ probs
        return (ps_grad, nothing)
    end
end

end
