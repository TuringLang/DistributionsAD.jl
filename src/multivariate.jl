## Dirichlet ##

struct TuringDirichlet{T, TV <: AbstractVector} <: ContinuousMultivariateDistribution
    alpha::TV
    alpha0::T
    lmnB::T
end
function check(alpha)
    all(ai -> ai > 0, alpha) || 
        throw(ArgumentError("Dirichlet: alpha must be a positive vector."))
end
Zygote.@nograd DistributionsAD.check

function TuringDirichlet(alpha::AbstractVector)
    check(alpha)
    alpha0 = sum(alpha)
    lmnB = sum(loggamma, alpha) - loggamma(alpha0)
    T = promote_type(typeof(alpha0), typeof(lmnB))
    TV = typeof(alpha)
    TuringDirichlet{T, TV}(alpha, alpha0, lmnB)
end

function TuringDirichlet(d::Integer, alpha::Real)
    alpha0 = alpha * d
    _alpha = fill(alpha, d)
    lmnB = loggamma(alpha) * d - loggamma(alpha0)
    T = promote_type(typeof(alpha0), typeof(lmnB))
    TV = typeof(_alpha)
    TuringDirichlet{T, TV}(_alpha, alpha0, lmnB)
end
function TuringDirichlet(alpha::AbstractVector{T}) where {T <: Integer}
    Tf = float(T)
    TuringDirichlet(convert(AbstractVector{Tf}, alpha))
end
TuringDirichlet(d::Integer, alpha::Integer) = TuringDirichlet(d, Float64(alpha))

Distributions.Dirichlet(alpha::TrackedVector) = TuringDirichlet(alpha)
Distributions.Dirichlet(d::Integer, alpha::TrackedReal) = TuringDirichlet(d, alpha)

function Distributions.logpdf(d::TuringDirichlet, x::AbstractVector)
    simplex_logpdf(d.alpha, d.lmnB, x)
end
function Distributions.logpdf(d::TuringDirichlet, x::AbstractMatrix)
    simplex_logpdf(d.alpha, d.lmnB, x)
end
function Distributions.logpdf(d::Dirichlet{T}, x::TrackedVecOrMat) where {T}
    TV = typeof(d.alpha)
    logpdf(TuringDirichlet{T, TV}(d.alpha, d.alpha0, d.lmnB), x)
end

@adjoint function Distributions.Dirichlet(alpha)
    return pullback(TuringDirichlet, alpha)
end
@adjoint function Distributions.Dirichlet(d, alpha)
    return pullback(TuringDirichlet, d, alpha)
end

function simplex_logpdf(alpha, lmnB, x::AbstractVector)
    sum((alpha .- 1) .* log.(x)) - lmnB
end
function simplex_logpdf(alpha, lmnB, x::AbstractMatrix)
    init = vcat(sum((alpha .- 1) .* log.(view(x, :, 1))))
    mapreduce(vcat, drop(eachcol(x), 1); init = init) do c
        sum((alpha .- 1) .* log.(c)) - lmnB
    end
end

@grad function simplex_logpdf(alpha, lmnB, x::AbstractVector)
    simplex_logpdf(data(alpha), data(lmnB), data(x)), Δ -> begin
        (Δ .* log.(data(x)), -Δ, Δ .* (data(alpha) .- 1))
    end
end
@grad function simplex_logpdf(alpha, lmnB, x::AbstractMatrix)
    simplex_logpdf(data(alpha), data(lmnB), data(x)), Δ -> begin
        (log.(data(x)) * Δ, -sum(Δ), repeat(data(alpha) .- 1, 1, size(x, 2)) * Diagonal(Δ))
    end
end

@adjoint function simplex_logpdf(alpha, lmnB, x::AbstractVector)
    return simplex_logpdf(alpha, lmnB, x), Δ -> (Δ .* log.(x), -Δ, Δ .* (alpha .- 1))
end

@adjoint function simplex_logpdf(alpha, lmnB, x::AbstractMatrix)
    return simplex_logpdf(alpha, lmnB, x), Δ -> begin
        (log.(x) * Δ, -sum(Δ), repeat(alpha .- 1, 1, size(x, 2)) * Diagonal(Δ))
    end
end

## MvNormal ##

"""
    TuringDenseMvNormal{Tm<:AbstractVector, TC<:Cholesky} <: ContinuousMultivariateDistribution

A multivariate Normal distribution whose covariance is dense. Compatible with Tracker.
"""
struct TuringDenseMvNormal{Tm<:AbstractVector, TC<:Cholesky} <: ContinuousMultivariateDistribution
    m::Tm
    C::TC
end
function TuringDenseMvNormal(m::AbstractVector, A::AbstractMatrix)
    return TuringDenseMvNormal(m, cholesky(A))
end
Base.length(d::TuringDenseMvNormal) = length(d.m)
Distributions.rand(d::TuringDenseMvNormal, n::Int...) = rand(Random.GLOBAL_RNG, d, n...)
function Distributions.rand(rng::Random.AbstractRNG, d::TuringDenseMvNormal, n::Int...)
    return d.m .+ d.C.U' * randn(rng, length(d), n...)
end

"""
    TuringDiagMvNormal{Tm<:AbstractVector, Tσ<:AbstractVector} <: ContinuousMultivariateDistribution

A multivariate normal distribution whose covariance is diagonal. Compatible with Tracker.
"""
struct TuringDiagMvNormal{Tm<:AbstractVector, Tσ<:AbstractVector} <: ContinuousMultivariateDistribution
    m::Tm
    σ::Tσ
end

Distributions.params(d::TuringDiagMvNormal) = (d.m, d.σ)
Distributions.dim(d::TuringDiagMvNormal) = length(d.m)
Base.length(d::TuringDiagMvNormal) = length(d.m)
Base.size(d::TuringDiagMvNormal) = (length(d), length(d))
Distributions.rand(d::TuringDiagMvNormal, n::Int...) = rand(Random.GLOBAL_RNG, d, n...)
function Distributions.rand(rng::Random.AbstractRNG, d::TuringDiagMvNormal, n::Int...)
    return d.m .+ d.σ .* randn(rng, length(d), n...)
end

struct TuringScalMvNormal{Tm<:AbstractVector, Tσ<:Real} <: ContinuousMultivariateDistribution
    m::Tm
    σ::Tσ
end

Base.length(d::TuringScalMvNormal) = length(d.m)
Base.size(d::TuringScalMvNormal) = (length(d), length(d))
Distributions.rand(d::TuringScalMvNormal, n::Int...) = rand(Random.GLOBAL_RNG, d, n...)
function Distributions.rand(rng::Random.AbstractRNG, d::TuringScalMvNormal, n::Int...)
    return d.m .+ d.σ .* randn(rng, length(d), n...)
end

for T in (:AbstractVector, :AbstractMatrix)
    @eval Distributions.logpdf(d::TuringScalMvNormal, x::$T) = _logpdf(d, x)
    @eval Distributions.logpdf(d::TuringDiagMvNormal, x::$T) = _logpdf(d, x)
    @eval Distributions.logpdf(d::TuringDenseMvNormal, x::$T) = _logpdf(d, x)
end

function _logpdf(d::TuringScalMvNormal, x::AbstractVector)
    return -(length(x) * log(2π * abs2(d.σ)) + sum(abs2.((x .- d.m) ./ d.σ))) / 2
end
function _logpdf(d::TuringScalMvNormal, x::AbstractMatrix)
    return -(size(x, 1) * log(2π * abs2(d.σ)) .+ vec(sum(abs2.((x .- d.m) ./ d.σ), dims=1))) ./ 2
end

function _logpdf(d::TuringDiagMvNormal, x::AbstractVector)
    return -(length(x) * log(2π) + 2 * sum(log.(d.σ)) + sum(abs2.((x .- d.m) ./ d.σ))) / 2
end
function _logpdf(d::TuringDiagMvNormal, x::AbstractMatrix)
    return -((size(x, 1) * log(2π) + 2 * sum(log.(d.σ))) .+ vec(sum(abs2.((x .- d.m) ./ d.σ), dims=1))) ./ 2
end
function _logpdf(d::TuringDenseMvNormal, x::AbstractVector)
    return -(length(x) * log(2π) + logdet(d.C) + sum(abs2.(zygote_ldiv(d.C.U', x .- d.m)))) / 2
end
function _logpdf(d::TuringDenseMvNormal, x::AbstractMatrix)
    return -((size(x, 1) * log(2π) + logdet(d.C)) .+ vec(sum(abs2.(zygote_ldiv(d.C.U', x .- d.m)), dims=1))) ./ 2
end

import StatsBase: entropy
function entropy(d::TuringDiagMvNormal)
    T = eltype(d.σ)
    return (length(d) * (T(log2π) + one(T)) / 2 + sum(log.(d.σ)))
end

# zero mean, dense covariance
MvNormal(A::TrackedMatrix) = TuringMvNormal(A)

# zero mean, diagonal covariance
MvNormal(σ::TrackedVector) = TuringMvNormal(σ)

# dense mean, dense covariance
MvNormal(m::TrackedVector{<:Real}, A::TrackedMatrix{<:Real}) = TuringMvNormal(m, A)
MvNormal(m::TrackedVector{<:Real}, A::Matrix{<:Real}) = TuringMvNormal(m, A)
MvNormal(m::AbstractVector{<:Real}, A::TrackedMatrix{<:Real}) = TuringMvNormal(m, A)

# dense mean, diagonal covariance
function MvNormal(
    m::TrackedVector{<:Real},
    D::Diagonal{T, <:TrackedVector{T}} where {T<:Real},
)
    return TuringMvNormal(m, D)
end
function MvNormal(
    m::AbstractVector{<:Real},
    D::Diagonal{T, <:TrackedVector{T}} where {T<:Real},
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
MvNormal(d::Int, σ::TrackedReal{<:Real}) = TuringMvNormal(d, σ)

TuringMvNormal(d::Int, σ::Real) = TuringMvNormal(zeros(d), σ)
TuringMvNormal(m::AbstractVector{<:Real}, σ::Real) = TuringScalMvNormal(m, σ)
TuringMvNormal(σ::AbstractVector) = TuringMvNormal(zeros(length(σ)), σ)
TuringMvNormal(A::AbstractMatrix) = TuringMvNormal(zeros(size(A, 1)), A)
function TuringMvNormal(m::AbstractVector{<:Real}, σ::AbstractVector{<:Real})
    return TuringDiagMvNormal(m, σ)
end
function TuringMvNormal(
    m::AbstractVector{<:Real},
    D::Diagonal{T, <:AbstractVector{T}},
) where {T <: Real}
    return TuringMvNormal(m, sqrt.(D.diag))
end
function TuringMvNormal(m::AbstractVector{<:Real}, A::AbstractMatrix{<:Real})
    return TuringDenseMvNormal(m, A)
end
function TuringMvNormal(m::AbstractVector{<:Real}, A::UniformScaling{<:Real})
    return TuringMvNormal(m, sqrt(A.λ))
end

## MvLogNormal ##

struct TuringMvLogNormal{TD} <: AbstractMvLogNormal
    normal::TD
end
MvLogNormal(d::TuringDenseMvNormal) = TuringMvLogNormal(d)
MvLogNormal(d::TuringDiagMvNormal) = TuringMvLogNormal(d)
MvLogNormal(d::TuringScalMvNormal) = TuringMvLogNormal(d)
Distributions.length(d::TuringMvLogNormal) = length(d.normal)
function Distributions.rand(rng::Random.AbstractRNG, d::TuringMvLogNormal)
    return Distributions.exp!(rand(rng, d.normal))
end
function Distributions.rand(rng::Random.AbstractRNG, d::TuringMvLogNormal, n::Int)
    return Distributions.exp!(rand(rng, d.normal, n))
end
for T in (:AbstractVector, :AbstractMatrix)
    @eval Distributions.logpdf(d::TuringMvLogNormal, x::$T) = _logpdf(d, x)
end
for T in (:TrackedVector, :TrackedMatrix)
    @eval Distributions.logpdf(d::TuringMvLogNormal, x::$T) = _logpdf(d, x)
end
function _logpdf(d::TuringMvLogNormal, x::AbstractVector{T}) where {T<:Real}
    if insupport(d, x)
        logx = log.(x)        
        return _logpdf(d.normal, logx) - sum(logx)
    else
        return -T(Inf)
    end
end
function _logpdf(d::TuringMvLogNormal, x::AbstractMatrix{<:Real})
    if all(i -> DistributionsAD.insupport(d, view(x, :, i)), axes(x, 2))
        logx = log.(x)
        return DistributionsAD._logpdf(d.normal, logx) - vec(sum(logx; dims = 1))
    else
        return [DistributionsAD._logpdf(d, view(x, :, i)) for i in axes(x, 2)]
    end
end

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
    D::Diagonal{T, <:TrackedVector{T}} where {T<:Real},
)
    return TuringMvLogNormal(TuringMvNormal(m, D))
end
function MvLogNormal(
    m::AbstractVector{<:Real},
    D::Diagonal{T, <:TrackedVector{T}} where {T<:Real},
)
    return TuringMvLogNormal(TuringMvNormal(m, D))
end
function MvLogNormal(
    m::TrackedVector{<:Real},
    D::Diagonal{T, <:AbstractVector{T}} where {T<:Real},
)
    return TuringMvLogNormal(TuringMvNormal(m, D))
end
function MvLogNormal(
    m::AbstractVector{<:Real},
    D::Diagonal{T, <:AbstractVector{T}} where {T<:Real},
)
    return MvLogNormal(MvNormal(m, D))
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
MvLogNormal(d::Int, σ::TrackedReal{<:Real}) = TuringMvLogNormal(TuringMvNormal(d, σ))

## Zygote adjoint

@adjoint function Distributions.MvNormal(
    A::Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}},
)
    return pullback(TuringMvNormal, A)
end
@adjoint function Distributions.MvNormal(
    m::AbstractVector{<:Real},
    A::Union{Real, UniformScaling, AbstractVecOrMat{<:Real}},
)
    return pullback(TuringMvNormal, m, A)
end
@adjoint function Distributions.MvNormal(
    d::Int,
    A::Real,
)
    value, back = pullback(A -> TuringMvNormal(d, A), A)
    return value, x -> (nothing, back(x)[1])
end
