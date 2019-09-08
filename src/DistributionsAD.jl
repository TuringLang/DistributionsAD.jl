module DistributionsAD

using PDMats, ForwardDiff, Zygote, Tracker, LinearAlgebra, Distributions, Random
import StatsFuns: logsumexp, binomlogpdf, nbinomlogpdf, poislogpdf
using Tracker: TrackedReal, TrackedVector, TrackedMatrix
using LinearAlgebra: copytri!

import Distributions: MvNormal, MvLogNormal, poissonbinomial_pdf_fft, logpdf, quantile, PoissonBinomial
using Distributions: AbstractMvLogNormal, ContinuousMultivariateDistribution

export  TuringDiagNormal,
        TuringMvNormal,
        TuringMvLogNormal,
        TuringPoissonBinomial

logsumexp(x::Tracker.TrackedArray) = Tracker.track(logsumexp, x)
Tracker.@grad function logsumexp(x::Tracker.TrackedArray)
    lse = logsumexp(Tracker.data(x))
    return lse,
          Δ->(Δ .* exp.(x .- lse),)
end

binomlogpdf(n::Int, p::Tracker.TrackedReal, x::Int) = Tracker.track(binomlogpdf, n, p, x)
Tracker.@grad function binomlogpdf(n::Int, p::Tracker.TrackedReal, x::Int)
    return binomlogpdf(n, Tracker.data(p), x),
        Δ->(nothing, Δ * (x / p - (n - x) / (1 - p)), nothing)
end

# Note the definition of NegativeBinomial in Julia is not the same as Wikipedia's.
# Check the docstring of NegativeBinomial, r is the number of successes and
# k is the number of failures
_nbinomlogpdf_grad_1(r, p, k) = k == 0 ? log(p) : sum(1 / (k + r - i) for i in 1:k) + log(p)
_nbinomlogpdf_grad_2(r, p, k) = -k / (1 - p) + r / p

nbinomlogpdf(n::Tracker.TrackedReal, p::Tracker.TrackedReal, x::Int) = Tracker.track(nbinomlogpdf, n, p, x)
nbinomlogpdf(n::Real, p::Tracker.TrackedReal, x::Int) = Tracker.track(nbinomlogpdf, n, p, x)
nbinomlogpdf(n::Tracker.TrackedReal, p::Real, x::Int) = Tracker.track(nbinomlogpdf, n, p, x)
Tracker.@grad function nbinomlogpdf(r::Tracker.TrackedReal, p::Tracker.TrackedReal, k::Int)
    return nbinomlogpdf(Tracker.data(r), Tracker.data(p), k),
        Δ->(Δ * _nbinomlogpdf_grad_1(r, p, k), Δ * _nbinomlogpdf_grad_2(r, p, k), nothing)
end
Tracker.@grad function nbinomlogpdf(r::Real, p::Tracker.TrackedReal, k::Int)
    return nbinomlogpdf(Tracker.data(r), Tracker.data(p), k),
        Δ->(Tracker._zero(r), Δ * _nbinomlogpdf_grad_2(r, p, k), nothing)
end
Tracker.@grad function nbinomlogpdf(r::Tracker.TrackedReal, p::Real, k::Int)
    return nbinomlogpdf(Tracker.data(r), Tracker.data(p), k),
        Δ->(Δ * _nbinomlogpdf_grad_1(r, p, k), Tracker._zero(p), nothing)
end

poislogpdf(v::Tracker.TrackedReal, x::Int) = Tracker.track(poislogpdf, v, x)
Tracker.@grad function poislogpdf(v::Tracker.TrackedReal, x::Int)
      return poislogpdf(Tracker.data(v), x),
          Δ->(Δ * (x/v - 1), nothing)
end

function binomlogpdf(n::Int, p::ForwardDiff.Dual{T}, x::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val = ForwardDiff.value(p)
    Δ = ForwardDiff.partials(p)
    return FD(binomlogpdf(n, val, x),  Δ * (x / val - (n - x) / (1 - val)))
end

function nbinomlogpdf(r::ForwardDiff.Dual{T}, p::ForwardDiff.Dual{T}, k::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val_p = ForwardDiff.value(p)
    val_r = ForwardDiff.value(r)

    Δ_r = ForwardDiff.partials(r) * _nbinomlogpdf_grad_1(val_r, val_p, k)
    Δ_p = ForwardDiff.partials(p) * _nbinomlogpdf_grad_2(val_r, val_p, k)
    Δ = Δ_p + Δ_r
    return FD(nbinomlogpdf(val_r, val_p, k),  Δ)
end
function nbinomlogpdf(r::Real, p::ForwardDiff.Dual{T}, k::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val_p = ForwardDiff.value(p)
    Δ_p = ForwardDiff.partials(p) * _nbinomlogpdf_grad_2(r, val_p, k)
    return FD(nbinomlogpdf(r, val_p, k),  Δ_p)
end
function nbinomlogpdf(r::ForwardDiff.Dual{T}, p::Real, k::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val_r = ForwardDiff.value(r)
    Δ_r = ForwardDiff.partials(r) * _nbinomlogpdf_grad_1(val_r, p, k)
    return FD(nbinomlogpdf(val_r, p, k),  Δ_r)
end

function poislogpdf(v::ForwardDiff.Dual{T}, x::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val = ForwardDiff.value(v)
    Δ = ForwardDiff.partials(v)
    return FD(poislogpdf(val, x), Δ * (x/val - 1))
end


#
# Make Tracker work with MvNormal and unsafe Cholesky. This is quite nasty.
#

LinearAlgebra.UpperTriangular(A::Tracker.TrackedMatrix) = Tracker.track(UpperTriangular, A)
Tracker.@grad function LinearAlgebra.UpperTriangular(A::AbstractMatrix)
    return UpperTriangular(Tracker.data(A)), Δ->(UpperTriangular(Δ),)
end

function LinearAlgebra.cholesky(A::Tracker.TrackedMatrix; check=true)
    factors_info = turing_chol(A, check)
    factors = factors_info[1]
    info = Tracker.data(factors_info[2])
    return Cholesky{eltype(factors), typeof(factors)}(factors, 'U', info)
end
function turing_chol(A::AbstractMatrix, check)
    chol = cholesky(A, check=check)
    (chol.factors, chol.info)
end
turing_chol(A::Tracker.TrackedMatrix, check) = Tracker.track(turing_chol, A, check)
Tracker.@grad function turing_chol(A::AbstractMatrix, check)
    C, back = Zygote.forward(unsafe_cholesky, Tracker.data(A), Tracker.data(check))
    return (C.factors, C.info), Δ->back((factors=Tracker.data(Δ[1]),))
end

unsafe_cholesky(x, check) = cholesky(x, check=check)
Zygote.@adjoint function unsafe_cholesky(Σ::Real, check)
    C = cholesky(Σ; check=check)
    return C, function(Δ::NamedTuple)
        issuccess(C) || return (zero(Σ), nothing)
        (Δ.factors[1, 1] / (2 * C.U[1, 1]), nothing)
    end
end
Zygote.@adjoint function unsafe_cholesky(Σ::Diagonal, check)
    C = cholesky(Σ; check=check)
    return C, function(Δ::NamedTuple)
        issuccess(C) || (Diagonal(zero(diag(Δ.factors))), nothing)
        (Diagonal(diag(Δ.factors) .* inv.(2 .* C.factors.diag)), nothing)
    end
end
Zygote.@adjoint function unsafe_cholesky(Σ::Union{StridedMatrix, Symmetric{<:Real, <:StridedMatrix}}, check)
    C = cholesky(Σ; check=check)
    return C, function(Δ::NamedTuple)
        issuccess(C) || return (zero(Δ.factors), nothing)
        U, Ū = C.U, Δ.factors
        Σ̄ = Ū * U'
        Σ̄ = copytri!(Σ̄, 'U')
        Σ̄ = ldiv!(U, Σ̄)
        BLAS.trsm!('R', 'U', 'T', 'N', one(eltype(Σ)), U.data, Σ̄)
        @inbounds for n in diagind(Σ̄)
            Σ̄[n] /= 2
        end
        return (UpperTriangular(Σ̄), nothing)
    end
end
  
# Specialised logdet for cholesky to target the triangle directly.
logdet_chol_tri(U::AbstractMatrix) = 2 * sum(log, U[diagind(U)])
logdet_chol_tri(U::Tracker.TrackedMatrix) = Tracker.track(logdet_chol_tri, U)
Tracker.@grad function logdet_chol_tri(U::AbstractMatrix)
    U_data = Tracker.data(U)
    return logdet_chol_tri(U_data), Δ->(Matrix(Diagonal(2 .* Δ ./ diag(U_data))),)
end

function LinearAlgebra.logdet(C::Cholesky{<:Tracker.TrackedReal, <:Tracker.TrackedMatrix})
    return logdet_chol_tri(C.U)
end

# Tracker's implementation of ldiv isn't good. We'll use Zygote's instead.
const TrackedVecOrMat = Union{Tracker.TrackedVector, Tracker.TrackedMatrix}
function zygote_ldiv(A::AbstractMatrix, B::AbstractVecOrMat)
    T = typeof((zero(eltype(A))*zero(eltype(B)) + zero(eltype(A))*zero(eltype(B)))/one(eltype(A)))
    BB = similar(B, T)
    copyto!(BB, B)
    ldiv!(A, BB)
end
function zygote_ldiv(A::Tracker.TrackedMatrix, B::TrackedVecOrMat)
    return Tracker.track(zygote_ldiv, A, B)
end
function zygote_ldiv(A::Tracker.TrackedMatrix, B::AbstractVecOrMat)
    return Tracker.track(zygote_ldiv, A, B)
end
zygote_ldiv(A::AbstractMatrix, B::TrackedVecOrMat) =  Tracker.track(zygote_ldiv, A, B)
Tracker.@grad function zygote_ldiv(A, B)
    Y, back = Zygote.forward(\, Tracker.data(A), Tracker.data(B))
    return Y, Δ->back(Tracker.data(Δ))
end

function Base.fill(
    value::Tracker.TrackedReal,
    dims::Vararg{Union{Integer, AbstractUnitRange}},
)
    return Tracker.track(fill, value, dims...)
end
Tracker.@grad function Base.fill(value::Real, dims...)
    return fill(Tracker.data(value), dims...), function(Δ)
        size(Δ) ≢ dims && error("Dimension mismatch")
        return (sum(Δ), map(_->nothing, dims)...)
    end
end

PDMats.invquad(Σ::PDiagMat, x::Tracker.TrackedVector) = sum(abs2.(x) ./ Σ.diag)


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

getchol(m::PDMats.AbstractPDMat) = m.chol
getchol(m::PDMats.PDiagMat) = cholesky(Diagonal(m.diag))
getchol(m::PDMats.ScalMat) = cholesky(Diagonal(fill(m.value, m.dim)))

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


#
# Intercepts to construct appropriate TuringMvNormal types. Methods line-separated. Imports
# used do avoid excessive code duplication. This is mildly annoying to maintain, but it
# should do the job reasonably well for now.
#

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

# For InverseWishart
function Base.:\(a::Cholesky{<:Tracker.TrackedReal, <:Tracker.TrackedArray}, b::AbstractVecOrMat)
    return (a.U \ (a.U' \ b))
end

struct TuringPoissonBinomial{T<:Real, TV<:AbstractVector{T}} <: DiscreteUnivariateDistribution
    p::TV
    pmf::TV
end
function TuringPoissonBinomial(p::AbstractArray{<:Real})
    pb = Distributions.poissonbinomial_pdf_fft(p)
    @assert Distributions.isprobvec(pb)
    TuringPoissonBinomial(p, pb)
end
function logpdf(d::TuringPoissonBinomial{T}, k::Int) where T<:Real
    insupport(d, k) ? log(d.pmf[k + 1]) : -T(Inf)
end
quantile(d::TuringPoissonBinomial, x::Float64) = quantile(Categorical(d.pmf), x) - 1
PoissonBinomial(p::Tracker.TrackedArray) = TuringPoissonBinomial(p)
Base.minimum(d::TuringPoissonBinomial) = 0
Base.maximum(d::TuringPoissonBinomial) = length(d.p)

poissonbinomial_pdf_fft(x::Tracker.TrackedArray) = Tracker.track(poissonbinomial_pdf_fft, x)
Tracker.@grad function poissonbinomial_pdf_fft(x::Tracker.TrackedArray)
    x_data = Tracker.data(x)
    T = eltype(x_data)
    fft = poissonbinomial_pdf_fft(x_data)
    return  fft, Δ -> begin
        ((ForwardDiff.jacobian(x -> poissonbinomial_pdf_fft(x), x_data)::Matrix{T})' * Δ,)
    end
end

end
