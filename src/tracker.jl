## Generic ##

Tracker.dual(x::Bool, p) = x
Tracker.dual(x::Int, p) = x

Base.prevfloat(r::TrackedReal) = track(prevfloat, r)
@grad function prevfloat(r::Real)
    prevfloat(data(r)), Δ -> Δ
end
Base.nextfloat(r::TrackedReal) = track(nextfloat, r)
@grad function nextfloat(r::Real)
    nextfloat(data(r)), Δ -> Δ
end

for f = [:hcat, :vcat]
    for c = [
        [:TrackedReal],
        [:AbstractVecOrMat, :TrackedReal],
        [:TrackedVecOrMat, :TrackedReal],
    ]
        cnames = map(_ -> gensym(), c)
        @eval begin
            function Base.$f(
                $([:($x::$c) for (x, c) in zip(cnames, c)]...),
                x::Union{TrackedArray,TrackedReal},
                xs::Union{AbstractArray,Number}...,
            )
                return track($f, $(cnames...), x, xs...)
            end
        end
    end
    @eval begin
        @grad function $f(x::Real)
            $f(data(x)), (Δ) -> (Δ[1],)
        end
        @grad function $f(x1::Real, x2::Real)
            $f(data(x1), data(x2)), (Δ) -> (Δ[1], Δ[2])
        end
        @grad function $f(x1::AbstractVector, x2::Real)
            $f(data(x1), data(x2)), (Δ) -> (Δ[1:length(x1)], Δ[length(x1)+1])
        end
    end
end

function Base.copy(
    A::TrackedArray{T, 2, <:Adjoint{T, <:AbstractTriangular{T, <:AbstractMatrix{T}}}},
) where {T <: Real}
    return track(copy, A)
end
@grad function Base.copy(
    A::TrackedArray{T, 2, <:Adjoint{T, <:AbstractTriangular{T, <:AbstractMatrix{T}}}},
) where {T <: Real}
    return copy(data(A)), ∇ -> (copy(∇),)
end

Base.:*(A::TrackedMatrix, B::AbstractTriangular) = track(*, A, B)
Base.:*(A::AbstractTriangular{T}, B::TrackedVector) where {T} = track(*, A, B)
Base.:*(A::AbstractTriangular{T}, B::TrackedMatrix) where {T} = track(*, A, B)
Base.:*(A::Adjoint{T, <:AbstractTriangular{T}}, B::TrackedMatrix) where {T} = track(*, A, B)
Base.:*(A::Adjoint{T, <:AbstractTriangular{T}}, B::TrackedVector) where {T} = track(*, A, B)

function Base.fill(
    value::TrackedReal,
    dims::Vararg{Union{Integer, AbstractUnitRange}},
)
    return track(fill, value, dims...)
end
@grad function Base.fill(value::Real, dims...)
    return fill(data(value), dims...), function(Δ)
        size(Δ) ≢  dims && error("Dimension mismatch")
        return (sum(Δ), map(_->nothing, dims)...)
    end
end


## StatsFuns ##

logsumexp(x::TrackedArray; dims=:) = _logsumexp(x, dims)
_logsumexp(x::TrackedArray, dims=:) = track(_logsumexp, x, dims)
@grad function _logsumexp(x::TrackedArray, dims)
    lse = logsumexp(data(x), dims = dims)
    return lse, Δ -> (Δ .* exp.(x .- lse), nothing)
end


## Linear algebra ##

# Work around https://github.com/FluxML/Tracker.jl/pull/9#issuecomment-480051767
upper(A::AbstractMatrix) = UpperTriangular(A)
lower(A::AbstractMatrix) = LowerTriangular(A)
function upper(C::Cholesky)
    if C.uplo == 'U'
        return upper(C.factors)
    else
        return copy(lower(C.factors)')
    end
end
function lower(C::Cholesky)
    if C.uplo == 'U'
        return copy(upper(C.factors)')
    else
        return lower(C.factors)
    end
end

LinearAlgebra.LowerTriangular(A::TrackedMatrix) = lower(A)
lower(A::TrackedMatrix) = track(lower, A)
@grad lower(A) = lower(Tracker.data(A)), ∇ -> (lower(∇),)

LinearAlgebra.UpperTriangular(A::TrackedMatrix) = upper(A)
upper(A::TrackedMatrix) = track(upper, A)
@grad upper(A) = upper(Tracker.data(A)), ∇ -> (upper(∇),)

function LinearAlgebra.cholesky(A::TrackedMatrix; check=true)
    factors_info = turing_chol(A, check)
    factors = factors_info[1]
    info = data(factors_info[2])
    return Cholesky{eltype(factors), typeof(factors)}(factors, 'U', info)
end
function LinearAlgebra.cholesky(A::Symmetric{<:Any, <:TrackedMatrix}; check=true)
    uplo = A.uplo == 'U' ? (:U) : (:L)
    factors_info = symm_turing_chol(parent(A), check, uplo)
    factors = factors_info[1]
    info = data(factors_info[2])
    return Cholesky{eltype(factors), typeof(factors)}(factors, 'U', info)
end

turing_chol(A::TrackedMatrix, check) = track(turing_chol, A, check)
@grad function turing_chol(A::AbstractMatrix, check)
    Y, back = DistributionsAD.turing_chol_back(data(A),check)
    Y, Δ->back(data.(Δ))
end

symm_turing_chol(A::TrackedMatrix, check, uplo) = track(symm_turing_chol, A, check, uplo)
@grad function symm_turing_chol(A::AbstractMatrix, check, uplo)
    Y, back = DistributionsAD.symm_turing_chol_back(data(A),check,uplo)
    Y, Δ->back(data.(Δ))
end

# Specialised logdet for cholesky to target the triangle directly.
logdet_chol_tri(U::AbstractMatrix) = 2 * sum(log, U[diagind(U)])
logdet_chol_tri(U::TrackedMatrix) = track(logdet_chol_tri, U)
@grad function logdet_chol_tri(U::AbstractMatrix)
    U_data = data(U)
    return logdet_chol_tri(U_data), Δ->(Matrix(Diagonal(2 .* Δ ./ diag(U_data))),)
end

function LinearAlgebra.logdet(C::Cholesky{<:TrackedReal, <:TrackedMatrix})
    return logdet_chol_tri(C.U)
end

function zygote_ldiv(A::TrackedMatrix, B::TrackedVecOrMat)
    return track(zygote_ldiv, A, B)
end
function zygote_ldiv(A::TrackedMatrix, B::AbstractVecOrMat)
    return track(zygote_ldiv, A, B)
end
zygote_ldiv(A::AbstractMatrix, B::TrackedVecOrMat) =  track(zygote_ldiv, A, B)
@grad function zygote_ldiv(A, B)
    Y, dY_pullback = rrule(\, data(A), data(B))
    function back(Δ)
        _, dA, dB = dY_pullback(data(Δ))
        (unthunk(dA), unthunk(dB))
    end
    Y, back
end

function Base.:\(a::Cholesky{<:TrackedReal, <:TrackedArray}, b::AbstractVecOrMat)
    return (a.U \ (a.U' \ b))
end

## SpecialFunctions ##

SpecialFunctions.logabsgamma(x::TrackedReal) = track(logabsgamma, x)
@grad function SpecialFunctions.logabsgamma(x::Real)
    return logabsgamma(data(x)), Δ -> (digamma(data(x)) * Δ[1],)
end

# isprobvec
function Distributions.isprobvec(p::TrackedArray{<:Real})
    pdata = Tracker.data(p)
    all(x -> x ≥ zero(x), pdata) && isapprox(sum(pdata), one(eltype(pdata)), atol = 1e-6)
end

# Some array functions - workaround https://github.com/FluxML/Tracker.jl/issues/4
import Base: +, -, *, /, \
import LinearAlgebra: dot

for f in (:+, :-, :*, :/, :\, :dot), (T1, T2) in [
    (:TrackedArray, :AbstractArray),
    (:TrackedMatrix, :AbstractMatrix),
    (:TrackedMatrix, :AbstractVector),
    (:TrackedVector, :AbstractMatrix),
]
    @eval begin
        function $f(a::$T1{T}, b::$T2{<:TrackedReal}) where {T <: Real}
            return $f(convert(AbstractArray{TrackedReal{T}}, a), b)
        end
        function $f(a::$T2{<:TrackedReal}, b::$T1{T}) where {T <: Real}
            return $f(a, convert(AbstractArray{TrackedReal{T}}, b))
        end
    end
end


## Uniform ##

Distributions.Uniform(a::TrackedReal, b::Real) = TuringUniform{TrackedReal}(a, b)
Distributions.Uniform(a::Real, b::TrackedReal) = TuringUniform{TrackedReal}(a, b)
Distributions.Uniform(a::TrackedReal, b::TrackedReal) = TuringUniform{TrackedReal}(a, b)
Distributions.logpdf(d::Uniform, x::TrackedReal) = uniformlogpdf(d.a, d.b, x)

uniformlogpdf(a::Real, b::Real, x::TrackedReal) = track(uniformlogpdf, a, b, x)
uniformlogpdf(a::TrackedReal, b::TrackedReal, x::Real) = track(uniformlogpdf, a, b, x)
uniformlogpdf(a::TrackedReal, b::TrackedReal, x::TrackedReal) = track(uniformlogpdf, a, b, x)
@grad function uniformlogpdf(a, b, x)
    # compute log pdf
    diff = data(b) - data(a)
    insupport = a <= data(x) <= b
    lp = insupport ? -log(diff) : log(zero(diff))

    function pullback(Δ)
        z = zero(x) * Δ
        if insupport
            c = Δ / diff
            return c, -c, z
        else
            c = Δ / one(diff)
            cNaN = oftype(c, NaN)
            return cNaN, cNaN, oftype(z, NaN)
        end
    end

    return lp, pullback
end


## Binomial ##

binomlogpdf(n::Int, p::TrackedReal, x::Int) = track(binomlogpdf, n, p, x)
@grad function binomlogpdf(n::Int, p::TrackedReal, x::Int)
    return binomlogpdf(n, data(p), x),
        Δ->(nothing, Δ * (x / p - (n - x) / (1 - p)), nothing)
end


## Poisson ##

poislogpdf(v::TrackedReal, x::Int) = track(poislogpdf, v, x)
@grad function poislogpdf(v::TrackedReal, x::Int)
      return poislogpdf(data(v), x),
          Δ->(Δ * (x/v - 1), nothing)
end


## PoissonBinomial ##

PoissonBinomial(p::TrackedArray{<:Real}; check_args=true) =
    TuringPoissonBinomial(p; check_args = check_args)
poissonbinomial_pdf_fft(x::TrackedArray) = track(poissonbinomial_pdf_fft, x)
@grad function poissonbinomial_pdf_fft(x::TrackedArray)
    x_data = data(x)
    T = eltype(x_data)
    fft = poissonbinomial_pdf_fft(x_data)
    return  fft, Δ -> begin
        ((ForwardDiff.jacobian(x -> poissonbinomial_pdf_fft(x), x_data)::Matrix{T})' * Δ,)
    end
end


## Semicircle ##

function semicircle_dldr(r, x)
    diffsq = r^2 - x^2
    return -2 / r + r / diffsq
end
function semicircle_dldx(r, x)
    diffsq = r^2 - x^2
    return -x / diffsq
end

logpdf(d::Semicircle{<:Real}, x::TrackedReal) = semicirclelogpdf(d.r, x)
logpdf(d::Semicircle{<:TrackedReal}, x::Real) = semicirclelogpdf(d.r, x)
logpdf(d::Semicircle{<:TrackedReal}, x::TrackedReal) = semicirclelogpdf(d.r, x)

semicirclelogpdf(r, x) = logpdf(Semicircle(r), x)
M, f, arity = DiffRules.@define_diffrule DistributionsAD.semicirclelogpdf(r, x) =
    :(semicircle_dldr($r, $x)), :(semicircle_dldx($r, $x))
da, db = DiffRules.diffrule(M, f, :a, :b)
f = :($M.$f)
@eval begin
    @grad $f(a::TrackedReal, b::TrackedReal) = $f(data(a), data(b)), Δ -> (Δ * $da, Δ * $db)
    @grad $f(a::TrackedReal, b::Real) = $f(data(a), b), Δ -> (Δ * $da, Tracker._zero(b))
    @grad $f(a::Real, b::TrackedReal) = $f(a, data(b)), Δ -> (Tracker._zero(a), Δ * $db)
    $f(a::TrackedReal, b::TrackedReal)  = track($f, a, b)
    $f(a::TrackedReal, b::Real) = track($f, a, b)
    $f(a::Real, b::TrackedReal) = track($f, a, b)
end


## Negative binomial ##

# Note the definition of NegativeBinomial in Julia is not the same as Wikipedia's.
# Check the docstring of NegativeBinomial, r is the number of successes and
# k is the number of failures
_nbinomlogpdf_grad_1(r, p, k) = k == 0 ? log(p) : sum(1 / (k + r - i) for i in 1:k) + log(p)
_nbinomlogpdf_grad_2(r, p, k) = -k / (1 - p) + r / p

nbinomlogpdf(n::TrackedReal, p::TrackedReal, x::Int) = track(nbinomlogpdf, n, p, x)
nbinomlogpdf(n::Real, p::TrackedReal, x::Int) = track(nbinomlogpdf, n, p, x)
nbinomlogpdf(n::TrackedReal, p::Real, x::Int) = track(nbinomlogpdf, n, p, x)
@grad function nbinomlogpdf(r::TrackedReal, p::TrackedReal, k::Int)
    return nbinomlogpdf(data(r), data(p), k),
        Δ->(Δ * _nbinomlogpdf_grad_1(r, p, k), Δ * _nbinomlogpdf_grad_2(r, p, k), nothing)
end
@grad function nbinomlogpdf(r::Real, p::TrackedReal, k::Int)
    return nbinomlogpdf(data(r), data(p), k),
        Δ->(Tracker._zero(r), Δ * _nbinomlogpdf_grad_2(r, p, k), nothing)
end
@grad function nbinomlogpdf(r::TrackedReal, p::Real, k::Int)
    return nbinomlogpdf(data(r), data(p), k),
        Δ->(Δ * _nbinomlogpdf_grad_1(r, p, k), Tracker._zero(p), nothing)
end

## Multinomial

function Distributions.logpdf(
    dist::Multinomial{<:Real,<:TrackedVector},
    X::AbstractMatrix{<:Real}
)
    size(X, 1) == length(dist) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))

    return map(axes(X, 2)) do i
        Distributions._logpdf(dist, view(X, :, i))
    end
end

## Categorical ##

function Distributions.DiscreteNonParametric{T,P,Ts,Ps}(
    vs::Ts,
    ps::Ps;
    check_args=true,
) where {T<:Real,P<:Real,Ts<:AbstractVector{T},Ps<:TrackedArray{P, 1, <:SubArray{P, 1}}}
    cps = ps[:]
    return DiscreteNonParametric{T,P,Ts,typeof(cps)}(vs, cps; check_args = check_args)
end


## Dirichlet ##

Distributions.Dirichlet(alpha::TrackedVector) = TuringDirichlet(alpha)
Distributions.Dirichlet(d::Integer, alpha::TrackedReal) = TuringDirichlet(d, alpha)

function Distributions._logpdf(d::Dirichlet, x::TrackedVector{<:Real})
    return Distributions._logpdf(TuringDirichlet(d.alpha, d.alpha0, d.lmnB), x)
end
function Distributions.logpdf(d::Dirichlet, x::TrackedMatrix{<:Real})
    return logpdf(TuringDirichlet(d.alpha, d.alpha0, d.lmnB), x)
end
function Distributions.loglikelihood(d::Dirichlet, x::TrackedMatrix{<:Real})
    return loglikelihood(TuringDirichlet(d.alpha, d.alpha0, d.lmnB), x)
end

# Fix ambiguities
function Distributions.logpdf(d::TuringDirichlet, x::TrackedMatrix{<:Real})
    size(x, 1) == length(d) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    return simplex_logpdf(d.alpha, d.lmnB, x)
end

## Product

# TODO: Remove when modified upstream
function Distributions.loglikelihood(dist::Product, x::TrackedVector{<:Real})
    return Distributions.logpdf(dist, x)
end

## MvNormal

for (f, T) in (
    (:_logpdf, :TrackedVector),
    (:logpdf, :TrackedMatrix),
    (:loglikelihood, :TrackedMatrix),
)
    @eval begin
        function Distributions.$f(d::MvNormal{<:Real, <:PDMats.ScalMat}, x::$T{<:Real})
            return Distributions.$f(TuringScalMvNormal(d.μ, sqrt(d.Σ.value)), x)
        end
        function Distributions.$f(d::MvNormal{<:Real, <:PDMats.PDiagMat}, x::$T{<:Real})
            return Distributions.$f(TuringDiagMvNormal(d.μ, sqrt.(d.Σ.diag)), x)
        end
        function Distributions.$f(d::MvNormal{<:Real, <:PDMats.PDMat}, x::$T{<:Real})
            return Distributions.$f(TuringDenseMvNormal(d.μ, d.Σ.chol), x)
        end

        function Distributions.$f(
            d::MvLogNormal{<:Real, <:PDMats.ScalMat},
            x::$T{<:Real},
        )
            return Distributions.$f(
                TuringMvLogNormal(TuringScalMvNormal(d.normal.μ, sqrt(d.normal.Σ.value))),
                x,
            )
        end
        function Distributions.$f(
            d::MvLogNormal{<:Real, <:PDMats.PDiagMat},
            x::$T{<:Real},
        )
            return Distributions.$f(
                TuringMvLogNormal(TuringDiagMvNormal(d.normal.μ, sqrt.(d.normal.Σ.diag))),
                x,
            )
        end
        function Distributions.$f(
            d::MvLogNormal{<:Real, <:PDMats.PDMat},
            x::$T{<:Real},
        )
            return Distributions.$f(
                TuringMvLogNormal(TuringDenseMvNormal(d.normal.μ, d.normal.Σ.chol)),
                x,
            )
        end
    end
end

# Fix method ambiguities
Distributions.logpdf(d::TuringScalMvNormal, x::TrackedMatrix{<:Real}) = turing_logpdf(d, x)
Distributions.logpdf(d::TuringDiagMvNormal, x::TrackedMatrix{<:Real}) = turing_logpdf(d, x)
Distributions.logpdf(d::TuringDenseMvNormal, x::TrackedMatrix{<:Real}) = turing_logpdf(d, x)
Distributions.logpdf(d::TuringMvLogNormal, x::TrackedMatrix{<:Real}) = turing_logpdf(d, x)

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

# zero mean, dense covariance
MvLogNormal(A::TrackedMatrix) = TuringMvLogNormal(TuringMvNormal(A))

# zero mean, diagonal covariance
MvLogNormal(σ::TrackedVector) = TuringMvLogNormal(TuringMvNormal(σ))

# dense mean, dense covariance
MvLogNormal(m::TrackedVector{<:Real}, A::TrackedMatrix{<:Real}) = TuringMvLogNormal(TuringMvNormal(m, A))
MvLogNormal(m::TrackedVector{<:Real}, A::Matrix{<:Real}) = TuringMvLogNormal(TuringMvNormal(m, A))
MvLogNormal(m::AbstractVector{<:Real}, A::TrackedMatrix{<:Real}) = TuringMvLogNormal(TuringMvNormal(m, A))


## MvLogNormal ##

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


## MvCategorical ##

_mv_categorical_logpdf(ps::Tracker.TrackedMatrix, x) = Tracker.track(_mv_categorical_logpdf, ps, x)
Tracker.@grad function _mv_categorical_logpdf(ps, x)
    ps_data = Tracker.data(ps)
    probs = view(ps_data, x, :)
    ps_grad = zero(ps_data)
    sum(log, probs), Δ -> begin
        ps_grad .= 0
        ps_grad[x,:] .= Δ ./ probs
        return (ps_grad, nothing)
    end
end


## Wishart ##

function Distributions._logpdf(d::Wishart, X::TrackedMatrix)
    return Distributions._logpdf(TuringWishart(d), X)
end

function Distributions.loglikelihood(d::Wishart, X::AbstractArray{<:TrackedMatrix})
    return loglikelihood(TuringWishart(d), X)
end

Distributions.Wishart(df::TrackedReal, S::Matrix{<:Real}) = TuringWishart(df, S)
Distributions.Wishart(df::TrackedReal, S::AbstractMatrix{<:Real}) = TuringWishart(df, S)
Distributions.Wishart(df::Real, S::TrackedMatrix) = TuringWishart(df, S)
Distributions.Wishart(df::TrackedReal, S::TrackedMatrix) = TuringWishart(df, S)
Distributions.Wishart(df::TrackedReal, S::AbstractPDMat{<:TrackedReal}) = TuringWishart(df, S)


## Inverse Wishart ##

function Distributions._logpdf(d::InverseWishart, X::TrackedMatrix)
    return Distributions._logpdf(TuringInverseWishart(d), X)
end

function Distributions.loglikelihood(d::InverseWishart, X::AbstractArray{<:TrackedMatrix})
    return loglikelihood(TuringInverseWishart(d), X)
end

Distributions.InverseWishart(df::TrackedReal, S::Matrix{<:Real}) = TuringInverseWishart(df, S)
Distributions.InverseWishart(df::TrackedReal, S::AbstractMatrix{<:Real}) = TuringInverseWishart(df, S)
Distributions.InverseWishart(df::Real, S::TrackedMatrix) = TuringInverseWishart(df, S)
Distributions.InverseWishart(df::TrackedReal, S::TrackedMatrix) = TuringInverseWishart(df, S)
Distributions.InverseWishart(df::TrackedReal, S::AbstractPDMat{<:TrackedReal}) = TuringInverseWishart(df, S)

## General definitions of `logpdf` for arrays

function Distributions.logpdf(
    dist::MultivariateDistribution,
    X::TrackedMatrix{<:Real},
)
    size(X, 1) == length(dist) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    return map(axes(X, 2)) do i
        Distributions._logpdf(dist, view(X, :, i))
    end
end

function Distributions.logpdf(dist::MatrixDistribution, X::TrackedArray{<:Real,3})
    (size(X, 1), size(X, 2)) == size(dist) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    return map(axes(X, 3)) do i
        Distributions._logpdf(dist, view(X, :, :, i))
    end
end

function Distributions.logpdf(
    dist::MatrixDistribution,
    X::AbstractArray{<:TrackedMatrix{<:Real}},
)
    return map(x -> logpdf(dist, x), X)
end
