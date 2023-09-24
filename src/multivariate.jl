## Dirichlet ##

struct TuringDirichlet{T<:Real,TV<:AbstractVector,S<:Real} <: Distributions.ContinuousMultivariateDistribution
    alpha::TV
    alpha0::T
    lmnB::S
end

function TuringDirichlet(alpha::AbstractVector)
    all(ai -> ai > 0, alpha) ||
        throw(ArgumentError("Dirichlet: alpha must be a positive vector."))

    alpha0 = sum(alpha)
    lmnB = sum(loggamma, alpha) - loggamma(alpha0)

    return TuringDirichlet(alpha, alpha0, lmnB)
end
TuringDirichlet(d::Integer, alpha::Real) = TuringDirichlet(Fill(alpha, d))

# TODO: remove?
TuringDirichlet(alpha::AbstractVector{<:Integer}) = TuringDirichlet(float.(alpha))
TuringDirichlet(d::Integer, alpha::Integer) = TuringDirichlet(d, float(alpha))

# TODO: remove and use `Dirichlet` only for `Tracker.TrackedVector`
Distributions.Dirichlet(alpha::AbstractVector) = TuringDirichlet(alpha)

TuringDirichlet(d::Dirichlet) = TuringDirichlet(d.alpha, d.alpha0, d.lmnB)

Base.length(d::TuringDirichlet) = length(d.alpha)

function Distributions.insupport(d::TuringDirichlet, x::AbstractVector{<:Real})
    return dirichlet_insupport(x, length(d))
end
function dirichlet_insupport(x::AbstractVector{<:Real}, d::Int)
    return d == length(x) && all(x -> zero(x) <= x <= one(x), x) && sum(x) ≈ 1
end

# copied from Distributions
# TODO: remove and use `Dirichlet`?
function Distributions._rand!(
    rng::Random.AbstractRNG,
    d::TuringDirichlet,
    x::AbstractVector{<:Real},
)
    @inbounds for (i, αi) in zip(eachindex(x), d.alpha)
        x[i] = rand(rng, Gamma(αi))
    end
    lmul!(inv(sum(x)), x) # this returns x
end
function Distributions._rand!(
    rng::AbstractRNG,
    d::TuringDirichlet{<:Real,<:FillArrays.AbstractFill},
    x::AbstractVector{<:Real}
)
    rand!(rng, Gamma(FillArrays.getindex_value(d.alpha)), x)
    lmul!(inv(sum(x)), x) # this returns x
end

function Distributions._logpdf(d::TuringDirichlet, x::AbstractVector{<:Real})
    return simplex_logpdf(d.alpha, d.lmnB, x)
end
function Distributions.logpdf(d::TuringDirichlet, x::AbstractMatrix{<:Real})
    size(x, 1) == length(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    return simplex_logpdf(d.alpha, d.lmnB, x)
end

ZygoteRules.@adjoint function Distributions.Dirichlet(alpha)
    return ZygoteRules.pullback(TuringDirichlet, alpha)
end
ZygoteRules.@adjoint function Distributions.Dirichlet(d, alpha)
    return ZygoteRules.pullback(TuringDirichlet, d, alpha)
end

function xlogy_or_neginf(x, y)
    z = zero(y)
    return y >= z ? xlogy(x, y) : xlogy(one(x), z)
end
function identity_or_neginf(x::Real, insupport::Bool)
    return insupport ? float(x) : log(zero(x))
end

function simplex_logpdf(alpha, lmnB, x::AbstractVector)
    logp = sum(xlogy_or_neginf.(alpha .- 1, x)) - lmnB
    return identity_or_neginf(logp, dirichlet_insupport(x, length(alpha)))
end
function simplex_logpdf(alpha, lmnB, x::AbstractMatrix)
    return identity_or_neginf.(
        vec(sum(xlogy_or_neginf.(alpha .- 1, x); dims=1)) .- lmnB,
        dirichlet_insupport.(eachcol(x), length(alpha)),
    )
end

ZygoteRules.@adjoint function simplex_logpdf(alpha, lmnB, x::AbstractVector)
    simplex_logpdf(alpha, lmnB, x), Δ -> (Δ .* log.(x), -Δ, Δ .* (alpha .- 1) ./ x)
end

ZygoteRules.@adjoint function simplex_logpdf(alpha, lmnB, x::AbstractMatrix)
    simplex_logpdf(alpha, lmnB, x), Δ -> begin
        (log.(x) * Δ, -sum(Δ), ((alpha .- 1) ./ x) * Diagonal(Δ))
    end
end

## MvNormal ##

"""
    TuringDenseMvNormal{Tm<:AbstractVector, TC<:Cholesky} <: Distributions.ContinuousMultivariateDistribution

A multivariate Normal distribution whose covariance is dense. Compatible with Tracker.
"""
struct TuringDenseMvNormal{Tm<:AbstractVector,TC<:Cholesky} <: Distributions.ContinuousMultivariateDistribution
    m::Tm
    C::TC
end
function TuringDenseMvNormal(m::AbstractVector, A::AbstractMatrix)
    return TuringDenseMvNormal(m, cholesky(A))
end
Base.length(d::TuringDenseMvNormal) = length(d.m)
Distributions.rand(d::TuringDenseMvNormal, n::Int...) = rand(Random.GLOBAL_RNG, d, n...)
function Distributions.rand(rng::Random.AbstractRNG, d::TuringDenseMvNormal, n::Int...)
    return d.m .+ d.C.U' * adapt_randn(rng, d.m, length(d), n...)
end

"""
    TuringDiagMvNormal{Tm<:AbstractVector, Tσ<:AbstractVector} <: Distributions.ContinuousMultivariateDistribution

A multivariate normal distribution whose covariance is diagonal. Compatible with Tracker.
"""
struct TuringDiagMvNormal{Tm<:AbstractVector,Tσ<:AbstractVector} <: Distributions.ContinuousMultivariateDistribution
    m::Tm
    σ::Tσ
end

Distributions.params(d::TuringDiagMvNormal) = (d.m, d.σ)
Base.length(d::TuringDiagMvNormal) = length(d.m)
Base.size(d::TuringDiagMvNormal) = (length(d),)
Distributions.rand(d::TuringDiagMvNormal, n::Int...) = rand(Random.GLOBAL_RNG, d, n...)
function Distributions.rand(rng::Random.AbstractRNG, d::TuringDiagMvNormal, n::Int...)
    return d.m .+ d.σ .* adapt_randn(rng, d.m, length(d), n...)
end

struct TuringScalMvNormal{Tm<:AbstractVector,Tσ<:Real} <: Distributions.ContinuousMultivariateDistribution
    m::Tm
    σ::Tσ
end

Distributions.params(d::TuringScalMvNormal) = (d.m, d.σ)
Base.length(d::TuringScalMvNormal) = length(d.m)
Base.size(d::TuringScalMvNormal) = (length(d),)
Distributions.rand(d::TuringScalMvNormal, n::Int...) = rand(Random.GLOBAL_RNG, d, n...)
function Distributions.rand(rng::Random.AbstractRNG, d::TuringScalMvNormal, n::Int...)
    return d.m .+ d.σ .* adapt_randn(rng, d.m, length(d), n...)
end

function Distributions._logpdf(d::TuringScalMvNormal, x::AbstractVector)
    σ2 = abs2(d.σ)
    return -(length(x) * log(twoπ * σ2) + sum(abs2.(x .- d.m)) / σ2) / 2
end
function Distributions.logpdf(d::TuringScalMvNormal, x::AbstractMatrix{<:Real})
    size(x, 1) == length(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    return -(size(x, 1) * log(twoπ * abs2(d.σ)) .+ vec(sum(abs2.((x .- d.m) ./ d.σ), dims=1))) ./ 2
end
function Distributions.loglikelihood(d::TuringScalMvNormal, x::AbstractMatrix{<:Real})
    σ2 = abs2(d.σ)
    return -(length(x) * log(twoπ * σ2) + sum(abs2.(x .- d.m)) / σ2) / 2
end

function Distributions._logpdf(d::TuringDiagMvNormal, x::AbstractVector)
    s = sum(log, d.σ)
    return -(length(x) * log(oftype(s, twoπ)) + 2 * s + sum(abs2.((x .- d.m) ./ d.σ))) / 2
end
function Distributions.logpdf(d::TuringDiagMvNormal, x::AbstractMatrix{<:Real})
    size(x, 1) == length(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    s = sum(log, d.σ)
    return -((size(x, 1) * log(oftype(s, twoπ)) + 2 * s) .+ vec(sum(abs2.((x .- d.m) ./ d.σ), dims=1))) ./ 2
end
function Distributions.loglikelihood(d::TuringDiagMvNormal, x::AbstractMatrix{<:Real})
    s = sum(log, d.σ)
    return -(length(x) * log(oftype(s, twoπ)) + 2 * size(x, 2) * s + sum(abs2.((x .- d.m) ./ d.σ))) / 2
end

function Distributions._logpdf(d::TuringDenseMvNormal, x::AbstractVector)
    z = logdet(d.C)
    return -(length(x) * log(oftype(z, twoπ)) + z + sum(abs2.(zygote_ldiv(d.C.U', x .- d.m)))) / 2
end
function Distributions.logpdf(d::TuringDenseMvNormal, x::AbstractMatrix{<:Real})
    size(x, 1) == length(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    z = logdet(d.C)
    return -((size(x, 1) * log(oftype(z, twoπ)) + z) .+ vec(sum(abs2.(zygote_ldiv(d.C.U', x .- d.m)), dims=1))) ./ 2
end
function Distributions.loglikelihood(d::TuringDenseMvNormal, x::AbstractMatrix{<:Real})
    z = logdet(d.C)
    return -(length(x) * log(oftype(z, twoπ)) + size(x, 2) * z + sum(abs2.(zygote_ldiv(d.C.U', x .- d.m)))) / 2
end

function Distributions.entropy(d::TuringScalMvNormal)
    s = log(d.σ)
    return length(d) * ((1 + log(oftype(s, twoπ))) / 2 + s)
end
function Distributions.entropy(d::TuringDiagMvNormal)
    s = sum(log, d.σ)
    return length(d) * (1 + log(oftype(s, twoπ))) / 2 + s
end
function Distributions.entropy(d::TuringDenseMvNormal)
    s = logdet(d.C)
    return (length(d) * (1 + log(oftype(s, twoπ))) + s) / 2
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
    D::Diagonal{T,<:AbstractVector{T}},
) where {T<:Real}
    return TuringMvNormal(m, sqrt.(D.diag))
end
function TuringMvNormal(m::AbstractVector{<:Real}, A::AbstractMatrix{<:Real})
    return TuringDenseMvNormal(m, A)
end
function TuringMvNormal(m::AbstractVector{<:Real}, A::UniformScaling{<:Real})
    return TuringMvNormal(m, sqrt(A.λ))
end

# Mean and covariance
Distributions.mean(d::TuringDiagMvNormal) = d.m
Distributions.var(d::TuringDiagMvNormal) = abs2.(d.σ)
Distributions.cov(d::TuringDiagMvNormal) = Diagonal(var(d))

Distributions.mean(d::TuringScalMvNormal) = d.m
Distributions.var(d::TuringScalMvNormal) = Fill(abs2(d.σ), length(d.m))
Distributions.cov(d::TuringScalMvNormal) = Diagonal(var(d))

Distributions.mean(d::TuringDenseMvNormal) = d.m
Distributions.var(d::TuringDenseMvNormal) = diag(cov(d))
Distributions.cov(d::TuringDenseMvNormal) = Matrix(d.C) # turns cholesky to matrix

## MvLogNormal ##

struct TuringMvLogNormal{TD} <: Distributions.AbstractMvLogNormal
    normal::TD
end
Distributions.MvLogNormal(d::TuringDenseMvNormal) = TuringMvLogNormal(d)
Distributions.MvLogNormal(d::TuringDiagMvNormal) = TuringMvLogNormal(d)
Distributions.MvLogNormal(d::TuringScalMvNormal) = TuringMvLogNormal(d)
Base.length(d::TuringMvLogNormal) = length(d.normal)
function Distributions.rand(rng::Random.AbstractRNG, d::TuringMvLogNormal)
    x = rand(rng, d.normal)
    map!(exp, x, x)
    return x
end
function Distributions.rand(rng::Random.AbstractRNG, d::TuringMvLogNormal, n::Int)
    x = rand(rng, d.normal, n)
    map!(exp, x, x)
    return x
end

function Distributions._logpdf(d::TuringMvLogNormal, x::AbstractVector{T}) where {T<:Real}
    if insupport(d, x)
        logx = log.(x)
        return Distributions._logpdf(d.normal, logx) - sum(logx)
    else
        return -T(Inf)
    end
end
function Distributions.logpdf(d::TuringMvLogNormal, x::AbstractMatrix{<:Real})
    size(x, 1) == length(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    if all(i -> DistributionsAD.insupport(d, view(x, :, i)), axes(x, 2))
        logx = log.(x)
        return Distributions.logpdf(d.normal, logx) .- vec(sum(logx; dims=1))
    else
        return [Distributions._logpdf(d, view(x, :, i)) for i in axes(x, 2)]
    end
end
function Distributions.loglikelihood(d::TuringMvLogNormal, x::AbstractMatrix{<:Real})
    if all(i -> DistributionsAD.insupport(d, view(x, :, i)), axes(x, 2))
        logx = log.(x)
        return loglikelihood(d.normal, logx) - sum(logx)
    else
        r = Distributions._logpdf(d.normal, view(x, :, 1))
        return oftype(r, -Inf)
    end
end

function Distributions.MvLogNormal(
    m::AbstractVector{<:Real},
    D::Diagonal{T,<:AbstractVector{T}} where {T<:Real},
)
    return MvLogNormal(MvNormal(m, D))
end

## Zygote adjoint

ZygoteRules.@adjoint function Distributions.MvNormal(
    A::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}},
)
    return ZygoteRules.pullback(TuringMvNormal, A)
end
ZygoteRules.@adjoint function Distributions.MvNormal(
    m::AbstractVector{<:Real},
    A::Union{Real,UniformScaling,AbstractVecOrMat{<:Real}},
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
for (f, T) in (
    (:_logpdf, :AbstractVector),
    (:logpdf, :AbstractMatrix),
    (:loglikelihood, :AbstractMatrix),
)
    @eval begin
        ZygoteRules.@adjoint function Distributions.$f(
            d::MvNormal{<:Real,<:PDMats.ScalMat},
            x::$T{<:Real},
        )
            return ZygoteRules.pullback(d, x) do d, x
                Distributions.$f(TuringScalMvNormal(d.μ, sqrt(d.Σ.value)), x)
            end
        end
        ZygoteRules.@adjoint function Distributions.$f(
            d::MvNormal{<:Real,<:PDMats.PDiagMat},
            x::$T{<:Real},
        )
            return ZygoteRules.pullback(d, x) do d, x
                Distributions.$f(TuringDiagMvNormal(d.μ, sqrt.(d.Σ.diag)), x)
            end
        end
        ZygoteRules.@adjoint function Distributions.$f(
            d::MvNormal{<:Real,<:PDMats.PDMat},
            x::$T{<:Real},
        )
            return ZygoteRules.pullback(d, x) do d, x
                Distributions.$f(TuringDenseMvNormal(d.μ, d.Σ.chol), x)
            end
        end

        ZygoteRules.@adjoint function Distributions.$f(
            d::MvLogNormal{<:Real,<:PDMats.ScalMat},
            x::$T{<:Real},
        )
            return ZygoteRules.pullback(d, x) do d, x
                dist = TuringMvLogNormal(
                    TuringScalMvNormal(d.normal.μ, sqrt(d.normal.Σ.value)),
                )
                Distributions.$f(dist, x)
            end
        end
        ZygoteRules.@adjoint function Distributions.$f(
            d::MvLogNormal{<:Real,<:PDMats.PDiagMat},
            x::$T{<:Real},
        )
            return ZygoteRules.pullback(d, x) do d, x
                dist = TuringMvLogNormal(
                    TuringDiagMvNormal(d.normal.μ, sqrt.(d.normal.Σ.diag)),
                )
                Distributions.$f(dist, x)
            end
        end
        ZygoteRules.@adjoint function Distributions.$f(
            d::MvLogNormal{<:Real,<:PDMats.PDMat},
            x::$T{<:Real},
        )
            return ZygoteRules.pullback(d, x) do d, x
                dist = TuringMvLogNormal(TuringDenseMvNormal(d.normal.μ, d.normal.Σ.chol))
                Distributions.$f(dist, x)
            end
        end
    end
end
