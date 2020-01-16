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
function Distributions.rand(rng::Random.AbstractRNG, d::TuringDenseMvNormal)
    return d.m .+ d.C.U' * randn(rng, dim(d))
end
function Distributions.rand(rng::Random.AbstractRNG, d::TuringDenseMvNormal, n::Int)
    return d.m .+ d.C.U' * randn(rng, dim(d), n)
end

"""
    TuringDiagMvNormal{Tm<:AbstractVector, Tσ<:AbstractVector} <: ContinuousMultivariateDistribution

A multivariate normal distribution whose covariance is diagonal. Compatible with Tracker.
"""
struct TuringDiagMvNormal{Tm<:AbstractVector, Tσ<:AbstractVector} <: ContinuousMultivariateDistribution
    m::Tm
    σ::Tσ
end

Base.length(d::TuringDiagMvNormal) = length(d.m)
Base.size(d::TuringDiagMvNormal) = (length(d), length(d))
function Distributions.rand(rng::Random.AbstractRNG, d::TuringDiagMvNormal)
    return d.m .+ d.σ .* randn(rng, length(d))
end
function Distributions.rand(rng::Random.AbstractRNG, d::TuringDiagMvNormal, n::Int)
    return d.m .+ d.σ .* randn(rng, length(d), n)
end

struct TuringScalMvNormal{Tm<:AbstractVector, Tσ<:Real} <: ContinuousMultivariateDistribution
    m::Tm
    σ::Tσ
end

Base.length(d::TuringScalMvNormal) = length(d.m)
Base.size(d::TuringScalMvNormal) = (length(d), length(d))
function Distributions.rand(rng::Random.AbstractRNG, d::TuringScalMvNormal)
    return d.m .+ d.σ .* randn(rng, length(d))
end
function Distributions.rand(rng::Random.AbstractRNG, d::TuringScalMvNormal, n::Int)
    return d.m .+ d.σ .* randn(rng, length(d), n)
end

for T in (:AbstractVector, :AbstractMatrix)
    @eval Distributions.logpdf(d::TuringScalMvNormal, x::$T) = _logpdf(d, x)
    @eval Distributions.logpdf(d::TuringDiagMvNormal, x::$T) = _logpdf(d, x)
    @eval Distributions.logpdf(d::TuringDenseMvNormal, x::$T) = _logpdf(d, x)
end

function _logpdf(d::TuringScalMvNormal, x::AbstractVector)
    return -(length(x) * log(2π * abs2(d.σ)) + sum(abs2, (x .- d.m) ./ d.σ)) / 2
end
function _logpdf(d::TuringScalMvNormal, x::AbstractMatrix)
    return -(size(x, 2) * log(2π) .+ 2 * sum(log(d.σ)) .+ sum(abs2, (x .- d.m) ./ d.σ, dims=1)') ./ 2
end

function _logpdf(d::TuringDiagMvNormal, x::AbstractVector)
    return -(length(x) * log(2π) + 2 * sum(log.(d.σ)) + sum(abs2, (x .- d.m) ./ d.σ)) / 2
end
function _logpdf(d::TuringDiagMvNormal, x::AbstractMatrix)
    return -(size(x, 2) * log(2π) .+ 2 * sum(log.(d.σ)) .+ sum(abs2, (x .- d.m) ./ d.σ, dims=1)') ./ 2
end
function _logpdf(d::TuringDenseMvNormal, x::AbstractVector)
    return -(length(x) * log(2π) + logdet(d.C) + sum(abs2, zygote_ldiv(d.C.U', x .- d.m))) / 2
end
function _logpdf(d::TuringDenseMvNormal, x::AbstractMatrix)
    return -(size(x, 2) * log(2π) .+ logdet(d.C) .+ sum(abs2, zygote_ldiv(d.C.U', x .- d.m), dims=1)') ./ 2
end

# zero mean, dense covariance
MvNormal(A::TrackedMatrix) = MvNormal(zeros(size(A, 1)), A)

# zero mean, diagonal covariance
MvNormal(σ::TrackedVector) = MvNormal(zeros(length(σ)), σ)

# dense mean, dense covariance
MvNormal(m::TrackedVector{<:Real}, A::TrackedMatrix{<:Real}) = TuringDenseMvNormal(m, A)
MvNormal(m::TrackedVector{<:Real}, A::Matrix{<:Real}) = TuringDenseMvNormal(m, A)
MvNormal(m::AbstractVector{<:Real}, A::TrackedMatrix{<:Real}) = TuringDenseMvNormal(m, A)

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
MvNormal(m::TrackedVector{<:Real}, σ::TrackedVector{<:Real}) = TuringDiagMvNormal(m, σ)
MvNormal(m::TrackedVector{<:Real}, σ::AbstractVector{<:Real}) = TuringDiagMvNormal(m, σ)
MvNormal(m::TrackedVector{<:Real}, σ::Vector{<:Real}) = TuringDiagMvNormal(m, σ)
MvNormal(m::AbstractVector{<:Real}, σ::TrackedVector{<:Real}) = TuringDiagMvNormal(m, σ)

# dense mean, constant variance
MvNormal(m::TrackedVector{<:Real}, σ::TrackedReal) = TuringScalMvNormal(m, σ)
MvNormal(m::TrackedVector{<:Real}, σ::Real) = TuringScalMvNormal(m, σ)
MvNormal(m::AbstractVector{<:Real}, σ::TrackedReal) = TuringScalMvNormal(m, σ)

# dense mean, constant variance
function MvNormal(m::TrackedVector{<:Real}, A::UniformScaling{<:TrackedReal})
    return MvNormal(m, sqrt(A.λ))
end
function MvNormal(m::AbstractVector{<:Real}, A::UniformScaling{<:TrackedReal})
    return MvNormal(m, sqrt(A.λ))
end
function MvNormal(m::TrackedVector{<:Real}, A::UniformScaling{<:Real})
    return MvNormal(m, sqrt(A.λ))
end

# zero mean,, constant variance
MvNormal(d::Int, σ::TrackedReal{<:Real}) = MvNormal(zeros(d), σ)


## MvLogNormal ##

struct TuringMvLogNormal{TD} <: AbstractMvLogNormal
    normal::TD
end
MvLogNormal(d::TuringDenseMvNormal) = TuringMvLogNormal(d)
MvLogNormal(d::TuringDiagMvNormal) = TuringMvLogNormal(d)
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

# zero mean, dense covariance
MvLogNormal(A::TrackedMatrix) = MvLogNormal(zeros(size(A, 1)), A)

# zero mean, diagonal covariance
MvLogNormal(σ::TrackedVector) = MvLogNormal(zeros(length(σ)), σ)

# dense mean, dense covariance
MvLogNormal(m::TrackedVector{<:Real}, A::TrackedMatrix{<:Real}) = TuringMvLogNormal(TuringDenseMvNormal(m, A))
MvLogNormal(m::TrackedVector{<:Real}, A::Matrix{<:Real}) = TuringMvLogNormal(TuringDenseMvNormal(m, A))
MvLogNormal(m::AbstractVector{<:Real}, A::TrackedMatrix{<:Real}) = TuringMvLogNormal(TuringDenseMvNormal(m, A))

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
MvLogNormal(m::TrackedVector{<:Real}, σ::TrackedVector{<:Real}) = TuringMvLogNormal(TuringDiagMvNormal(m, σ))
MvLogNormal(m::TrackedVector{<:Real}, σ::AbstractVector{<:Real}) = TuringMvLogNormal(TuringDiagMvNormal(m, σ))
MvLogNormal(m::TrackedVector{<:Real}, σ::Vector{<:Real}) = TuringMvLogNormal(TuringDiagMvNormal(m, σ))
MvLogNormal(m::AbstractVector{<:Real}, σ::TrackedVector{<:Real}) = TuringMvLogNormal(TuringDiagMvNormal(m, σ))

# dense mean, constant variance
function MvLogNormal(m::TrackedVector{<:Real}, σ::TrackedReal)
    return TuringMvLogNormal(MvNormal(m, σ))
end
function MvLogNormal(m::TrackedVector{<:Real}, σ::Real)
    return TuringMvLogNormal(MvNormal(m, σ))
end
function MvLogNormal(m::AbstractVector{<:Real}, σ::TrackedReal)
    return TuringMvLogNormal(MvNormal(m, σ))
end

# dense mean, constant variance
function MvLogNormal(m::TrackedVector{<:Real}, A::UniformScaling{<:TrackedReal})
    return TuringMvLogNormal(MvNormal(m, A))
end
function MvLogNormal(m::AbstractVector{<:Real}, A::UniformScaling{<:TrackedReal})
    return TuringMvLogNormal(MvNormal(m, A))
end
function MvLogNormal(m::TrackedVector{<:Real}, A::UniformScaling{<:Real})
    return TuringMvLogNormal(MvNormal(m, A))
end

# zero mean,, constant variance
MvLogNormal(d::Int, σ::TrackedReal{<:Real}) = MvLogNormal(zeros(d), σ)
