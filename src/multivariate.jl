## MvNormal ##

"""
    TuringMvNormal{Tm<:AbstractVector, TC<:Cholesky} <: ContinuousMultivariateDistribution

A multivariate Normal distribution whose covariance is dense. Compatible with Tracker.
"""
struct TuringMvNormal{Tm<:AbstractVector, TC<:Cholesky} <: ContinuousMultivariateDistribution
    m::Tm
    C::TC
end

TuringMvNormal(m::AbstractVector, A::AbstractMatrix) = TuringMvNormal(m, cholesky(A))

Distributions.dim(d::TuringMvNormal) = length(d.m)
function Distributions.rand(rng::Random.AbstractRNG, d::TuringMvNormal)
    return d.m .+ d.C.U' * randn(rng, dim(d))
end

"""
    TuringDiagNormal{Tm<:AbstractVector, Tσ<:AbstractVector} <: ContinuousMultivariateDistribution

A multivariate normal distribution whose covariance is diagonal. Compatible with Tracker.
"""
struct TuringDiagNormal{Tm<:AbstractVector, Tσ<:AbstractVector} <: ContinuousMultivariateDistribution
    m::Tm
    σ::Tσ
end

Distributions.dim(d::TuringDiagNormal) = length(d.m)
function Distributions.rand(rng::Random.AbstractRNG, d::TuringDiagNormal)
    return d.m .+ d.σ .* randn(rng, dim(d))
end
for T in (:AbstractVector, :AbstractMatrix)
    @eval Distributions.logpdf(d::TuringDiagNormal, x::$T) = _logpdf(d, x)
    @eval Distributions.logpdf(d::TuringMvNormal, x::$T) = _logpdf(d, x)
end
for T in (:(Tracker.TrackedVector), :(Tracker.TrackedMatrix))
    @eval Distributions.logpdf(d::MvNormal, x::$T) = _logpdf(d, x)
end

function _logpdf(d::TuringDiagNormal, x::AbstractVector)
    return -(dim(d) * log(2π) + 2 * sum(log.(d.σ)) + sum(abs2, (x .- d.m) ./ d.σ)) / 2
end
function _logpdf(d::TuringDiagNormal, x::AbstractMatrix)
    return -(dim(d) * log(2π) .+ 2 * sum(log.(d.σ)) .+ sum(abs2, (x .- d.m) ./ d.σ, dims=1)') ./ 2
end
function _logpdf(d::TuringMvNormal, x::AbstractVector)
    return -(dim(d) * log(2π) + logdet(d.C) + sum(abs2, zygote_ldiv(d.C.U', x .- d.m))) / 2
end
function _logpdf(d::TuringMvNormal, x::AbstractMatrix)
    return -(dim(d) * log(2π) .+ logdet(d.C) .+ sum(abs2, zygote_ldiv(d.C.U', x .- d.m), dims=1)') ./ 2
end
function _logpdf(d::MvNormal, x::Union{Tracker.TrackedVector, Tracker.TrackedMatrix})
    _logpdf(TuringMvNormal(d.μ, getchol(d.Σ)), x)
end

# zero mean, dense covariance
MvNormal(A::TrackedMatrix) = MvNormal(zeros(size(A, 1)), A)

# zero mean, diagonal covariance
MvNormal(σ::TrackedVector) = MvNormal(zeros(length(σ)), σ)

# dense mean, dense covariance
MvNormal(m::TrackedVector{<:Real}, A::TrackedMatrix{<:Real}) = TuringMvNormal(m, A)
MvNormal(m::TrackedVector{<:Real}, A::Matrix{<:Real}) = TuringMvNormal(m, A)
MvNormal(m::AbstractVector{<:Real}, A::TrackedMatrix{<:Real}) = TuringMvNormal(m, A)

# dense mean, diagonal covariance
function MvNormal(
    m::TrackedVector{<:Real},
    D::Diagonal{T, <:TrackedVector{T}} where {T<:Real},
)
    return MvNormal(m, sqrt.(D.diag))
end
function MvNormal(
    m::AbstractVector{<:Real},
    D::Diagonal{T, <:TrackedVector{T}} where {T<:Real},
)
    return MvNormal(m, sqrt.(D.diag))
end

# dense mean, diagonal covariance
MvNormal(m::TrackedVector{<:Real}, σ::TrackedVector{<:Real}) = TuringDiagNormal(m, σ)
MvNormal(m::TrackedVector{<:Real}, σ::AbstractVector{<:Real}) = TuringDiagNormal(m, σ)
MvNormal(m::TrackedVector{<:Real}, σ::Vector{<:Real}) = TuringDiagNormal(m, σ)
MvNormal(m::AbstractVector{<:Real}, σ::TrackedVector{<:Real}) = TuringDiagNormal(m, σ)

# dense mean, constant variance
MvNormal(m::TrackedVector{<:Real}, σ::TrackedReal) = MvNormal(m, fill(σ, length(m)))
MvNormal(m::TrackedVector{<:Real}, σ::Real) = MvNormal(m, fill(σ, length(m)))
MvNormal(m::AbstractVector{<:Real}, σ::TrackedReal) = MvNormal(m, fill(σ, length(m)))

# dense mean, constant variance
MvNormal(m::TrackedVector{<:Real}, A::UniformScaling{<:TrackedReal}) = MvNormal(m, A.λ)
MvNormal(m::AbstractVector{<:Real}, A::UniformScaling{<:TrackedReal}) = MvNormal(m, A.λ)
MvNormal(m::TrackedVector{<:Real}, A::UniformScaling{<:Real}) = MvNormal(m, A.λ)

# zero mean,, constant variance
MvNormal(d::Int, σ::TrackedReal{<:Real}) = MvNormal(zeros(d), σ)


## MvLogNormal ##

struct TuringMvLogNormal{TD} <: AbstractMvLogNormal
    normal::TD
end
MvLogNormal(d::TuringMvNormal) = TuringMvLogNormal(d)
MvLogNormal(d::TuringDiagNormal) = TuringMvLogNormal(d)
Distributions.dim(d::TuringMvLogNormal) = length(d.normal)
function Distributions.rand(rng::Random.AbstractRNG, d::TuringMvLogNormal)
    return exp!(rand(rng, d.normal))
end
for T in (:AbstractVector, :AbstractMatrix)
    @eval Distributions.logpdf(d::TuringMvLogNormal, x::$T) = _logpdf(d, x)
end
for T in (:(Tracker.TrackedVector), :(Tracker.TrackedMatrix))
    @eval Distributions.logpdf(d::TuringMvLogNormal, x::$T) = _logpdf(d, x)
end
function _logpdf(d::TuringMvLogNormal, x::AbstractVecOrMat{T}) where {T<:Real}
    return insupport(d, x) ? (_logpdf(d.normal, log.(x)) - sum(log.(x))) : -Inf
end
function _logpdf(d::MvLogNormal, x::Union{Tracker.TrackedVector, Tracker.TrackedMatrix})
    _logpdf(TuringMvLogNormal(TuringMvNormal(d.normal.μ, getchol(d.normal.Σ))), x)
end

# zero mean, dense covariance
MvLogNormal(A::TrackedMatrix) = MvLogNormal(zeros(size(A, 1)), A)

# zero mean, diagonal covariance
MvLogNormal(σ::TrackedVector) = MvLogNormal(zeros(length(σ)), σ)

# dense mean, dense covariance
MvLogNormal(m::TrackedVector{<:Real}, A::TrackedMatrix{<:Real}) = TuringMvLogNormal(TuringMvNormal(m, A))
MvLogNormal(m::TrackedVector{<:Real}, A::Matrix{<:Real}) = TuringMvLogNormal(TuringMvNormal(m, A))
MvLogNormal(m::AbstractVector{<:Real}, A::TrackedMatrix{<:Real}) = TuringMvLogNormal(TuringMvNormal(m, A))

# dense mean, diagonal covariance
function MvLogNormal(
    m::TrackedVector{<:Real},
    D::Diagonal{T, <:TrackedVector{T}} where {T<:Real},
)
    return MvLogNormal(m, sqrt.(D.diag))
end
function MvLogNormal(
    m::AbstractVector{<:Real},
    D::Diagonal{T, <:TrackedVector{T}} where {T<:Real},
)
    return MvLogNormal(m, sqrt.(D.diag))
end

# dense mean, diagonal covariance
MvLogNormal(m::TrackedVector{<:Real}, σ::TrackedVector{<:Real}) = TuringMvLogNormal(TuringDiagNormal(m, σ))
MvLogNormal(m::TrackedVector{<:Real}, σ::AbstractVector{<:Real}) = TuringMvLogNormal(TuringDiagNormal(m, σ))
MvLogNormal(m::TrackedVector{<:Real}, σ::Vector{<:Real}) = TuringMvLogNormal(TuringDiagNormal(m, σ))
MvLogNormal(m::AbstractVector{<:Real}, σ::TrackedVector{<:Real}) = TuringMvLogNormal(TuringDiagNormal(m, σ))

# dense mean, constant variance
MvLogNormal(m::TrackedVector{<:Real}, σ::TrackedReal) = MvLogNormal(m, fill(σ, length(m)))
MvLogNormal(m::TrackedVector{<:Real}, σ::Real) = MvLogNormal(m, fill(σ, length(m)))
MvLogNormal(m::AbstractVector{<:Real}, σ::TrackedReal) = MvLogNormal(m, fill(σ, length(m)))

# dense mean, constant variance
MvLogNormal(m::TrackedVector{<:Real}, A::UniformScaling{<:TrackedReal}) = MvLogNormal(m, A.λ)
MvLogNormal(m::AbstractVector{<:Real}, A::UniformScaling{<:TrackedReal}) = MvLogNormal(m, A.λ)
MvLogNormal(m::TrackedVector{<:Real}, A::UniformScaling{<:Real}) = MvLogNormal(m, A.λ)

# zero mean,, constant variance
MvLogNormal(d::Int, σ::TrackedReal{<:Real}) = MvLogNormal(zeros(d), σ)
