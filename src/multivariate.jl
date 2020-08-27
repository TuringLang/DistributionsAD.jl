## Dirichlet ##

struct TuringDirichlet{T, TV <: AbstractVector} <: ContinuousMultivariateDistribution
    alpha::TV
    alpha0::T
    lmnB::T
end
Base.length(d::TuringDirichlet) = length(d.alpha)
function check(alpha)
    all(ai -> ai > 0, alpha) || 
        throw(ArgumentError("Dirichlet: alpha must be a positive vector."))
end

function Distributions._rand!(rng::Random.AbstractRNG,
                d::TuringDirichlet,
                x::AbstractVector{<:Real})
    s = 0.0
    n = length(x)
    α = d.alpha
    for i in 1:n
        @inbounds s += (x[i] = rand(rng, Gamma(α[i])))
    end
    Distributions.multiply!(x, inv(s)) # this returns x
end

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
    TuringDirichlet(float.(alpha))
end
TuringDirichlet(d::Integer, alpha::Integer) = TuringDirichlet(d, Float64(alpha))

Distributions.Dirichlet(alpha::AbstractVector) = TuringDirichlet(alpha)

function Distributions._logpdf(d::TuringDirichlet, x::AbstractVector)
    return simplex_logpdf(d.alpha, d.lmnB, x)
end

ZygoteRules.@adjoint function Distributions.Dirichlet(alpha)
    return ZygoteRules.pullback(TuringDirichlet, alpha)
end
ZygoteRules.@adjoint function Distributions.Dirichlet(d, alpha)
    return ZygoteRules.pullback(TuringDirichlet, d, alpha)
end

function simplex_logpdf(alpha, lmnB, x::AbstractVector)
    sum((alpha .- 1) .* log.(x)) - lmnB
end

ZygoteRules.@adjoint function simplex_logpdf(alpha, lmnB, x::AbstractVector)
    simplex_logpdf(alpha, lmnB, x), Δ -> (Δ .* log.(x), -Δ, Δ .* (alpha .- 1) ./ x)
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
Base.length(d::TuringDiagMvNormal) = length(d.m)
Base.size(d::TuringDiagMvNormal) = (length(d),)
Distributions.rand(d::TuringDiagMvNormal, n::Int...) = rand(Random.GLOBAL_RNG, d, n...)
function Distributions.rand(rng::Random.AbstractRNG, d::TuringDiagMvNormal, n::Int...)
    return d.m .+ d.σ .* randn(rng, length(d), n...)
end

struct TuringScalMvNormal{Tm<:AbstractVector, Tσ<:Real} <: ContinuousMultivariateDistribution
    m::Tm
    σ::Tσ
end

Distributions.params(d::TuringScalMvNormal) = (d.m, d.σ)
Base.length(d::TuringScalMvNormal) = length(d.m)
Base.size(d::TuringScalMvNormal) = (length(d),)
Distributions.rand(d::TuringScalMvNormal, n::Int...) = rand(Random.GLOBAL_RNG, d, n...)
function Distributions.rand(rng::Random.AbstractRNG, d::TuringScalMvNormal, n::Int...)
    return d.m .+ d.σ .* randn(rng, length(d), n...)
end

function Distributions._logpdf(d::TuringScalMvNormal, x::AbstractVector)
    σ2 = abs2(d.σ)
    return -(length(x) * log(2π * σ2) + sum(abs2.(x .- d.m)) / σ2) / 2
end
function Distributions.loglikelihood(d::TuringScalMvNormal, x::AbstractMatrix{<:Real})
    σ2 = abs2(d.σ)
    return -(length(x) * log(2π * σ2) + sum(abs2.(x .- d.m)) / σ2) / 2
end

function Distributions._logpdf(d::TuringDiagMvNormal, x::AbstractVector)
    return -(length(x) * log(2π) + 2 * sum(log.(d.σ)) + sum(abs2.((x .- d.m) ./ d.σ))) / 2
end
function Distributions.loglikelihood(d::TuringDiagMvNormal, x::AbstractMatrix{<:Real})
    return -(length(x) * log(2π) + 2 * size(x, 2) * sum(log.(d.σ)) + sum(abs2.((x .- d.m) ./ d.σ))) / 2
end

function Distributions._logpdf(d::TuringDenseMvNormal, x::AbstractVector)
    return -(length(x) * log(2π) + logdet(d.C) + sum(abs2.(zygote_ldiv(d.C.U', x .- d.m)))) / 2
end
function Distributions.loglikelihood(d::TuringDenseMvNormal, x::AbstractMatrix{<:Real})
    return -(length(x) * log(2π) + size(x, 2) * logdet(d.C) + sum(abs2.(zygote_ldiv(d.C.U', x .- d.m)))) / 2
end

function StatsBase.entropy(d::TuringScalMvNormal)
    s = log(d.σ)
    return length(d) * ((1 + oftype(s, log2π)) / 2 + s)
end
function StatsBase.entropy(d::TuringDiagMvNormal)
    s = sum(log, d.σ)
    return length(d) * (1 + oftype(s, log2π)) / 2 + s
end
function StatsBase.entropy(d::TuringDenseMvNormal)
    s = logdet(d.C)
    return (length(d) * (1 + oftype(s, log2π)) + s) / 2
end

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

function Distributions._logpdf(d::TuringMvLogNormal, x::AbstractVector{T}) where {T<:Real}
    if insupport(d, x)
        logx = log.(x)
        return Distributions._logpdf(d.normal, logx) - sum(logx)
    else
        return -T(Inf)
    end
end

function MvLogNormal(
    m::AbstractVector{<:Real},
    D::Diagonal{T, <:AbstractVector{T}} where {T<:Real},
)
    return MvLogNormal(MvNormal(m, D))
end

## Zygote adjoint

ZygoteRules.@adjoint function Distributions.MvNormal(
    A::Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}},
)
    return ZygoteRules.pullback(TuringMvNormal, A)
end
ZygoteRules.@adjoint function Distributions.MvNormal(
    m::AbstractVector{<:Real},
    A::Union{Real, UniformScaling, AbstractVecOrMat{<:Real}},
)
    return ZygoteRules.pullback(TuringMvNormal, m, A)
end
ZygoteRules.@adjoint function Distributions.MvNormal(
    d::Int,
    A::Real,
)
    value, back = ZygoteRules.pullback(A -> TuringMvNormal(d, A), A)
    return value, x -> (nothing, back(x)[1])
end
for (f, T) in ((:_logpdf, :AbstractVector), (:loglikelihood, :AbstractMatrix))
    @eval begin
        ZygoteRules.@adjoint function Distributions.$f(
            d::MvNormal{<:Any, <:PDMats.ScalMat},
            x::$T
        )
            return ZygoteRules.pullback(d, x) do d, x
                Distributions.$f(TuringScalMvNormal(d.μ, sqrt(d.Σ.value)), x)
            end
        end
        ZygoteRules.@adjoint function Distributions.$f(
            d::MvNormal{<:Any, <:PDMats.PDiagMat},
            x::$T
        )
            return ZygoteRules.pullback(d, x) do d, x
                Distributions.$f(TuringDiagMvNormal(d.μ, sqrt.(d.Σ.diag)), x)
            end
        end
        ZygoteRules.@adjoint function Distributions.$f(
            d::MvNormal{<:Any, <:PDMats.PDMat},
            x::$T
        )
            return ZygoteRules.pullback(d, x) do d, x
                Distributions.$f(TuringDenseMvNormal(d.μ, d.Σ.chol), x)
            end
        end

        ZygoteRules.@adjoint function Distributions.$f(
            d::MvLogNormal{<:Any, <:PDMats.ScalMat},
            x::$T
        )
            return ZygoteRules.pullback(d, x) do d, x
                dist = TuringMvLogNormal(
                    TuringScalMvNormal(d.normal.μ, sqrt(d.normal.Σ.value)),
                )
                Distributions.$f(dist, x)
            end
        end
        ZygoteRules.@adjoint function Distributions.$f(
            d::MvLogNormal{<:Any, <:PDMats.PDiagMat},
            x::$T
        )
            return ZygoteRules.pullback(d, x) do d, x
                dist = TuringMvLogNormal(
                    TuringDiagMvNormal(d.normal.μ, sqrt.(d.normal.Σ.diag)),
                )
                Distributions.$f(dist, x)
            end
        end
        ZygoteRules.@adjoint function Distributions.$f(
            d::MvLogNormal{<:Any, <:PDMats.PDMat},
            x::$T
        )
            return ZygoteRules.pullback(d, x) do d, x
                dist = TuringMvLogNormal(TuringDenseMvNormal(d.normal.μ, d.normal.Σ.chol))
                Distributions.$f(dist, x)
            end
        end
    end
end
