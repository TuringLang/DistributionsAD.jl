## Generic ##

Tracker.dual(x::Bool, p) = x
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

function turing_chol(A::AbstractMatrix, check)
    chol = cholesky(A, check=check)
    (chol.factors, chol.info)
end
function symm_turing_chol(A::AbstractMatrix, check, uplo)
    chol = cholesky(Symmetric(A, uplo), check=check)
    (chol.factors, chol.info)
end

turing_chol(A::TrackedMatrix, check) = track(turing_chol, A, check)
@grad function turing_chol(A::AbstractMatrix, check)
    C, back = ZygoteRules.pullback(data(A), check) do A, check
        cholesky(A, check=check)
    end
    return (C.factors, C.info), Δ->back((factors=data(Δ[1]),))
end

symm_turing_chol(A::TrackedMatrix, check, uplo) = track(symm_turing_chol, A, check, uplo)
@grad function symm_turing_chol(A::AbstractMatrix, check, uplo)
    C, back = ZygoteRules.pullback(data(A), check, uplo) do A, check, uplo
        cholesky(Symmetric(A, uplo), check=check)
    end
    return (C.factors, C.info), Δ->back((factors=data(Δ[1]),))
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

# Tracker's implementation of ldiv isn't good. We'll use Zygote's instead.

zygote_ldiv(A::AbstractMatrix, B::AbstractVecOrMat) = A \ B
function zygote_ldiv(A::TrackedMatrix, B::TrackedVecOrMat)
    return track(zygote_ldiv, A, B)
end
function zygote_ldiv(A::TrackedMatrix, B::AbstractVecOrMat)
    return track(zygote_ldiv, A, B)
end
zygote_ldiv(A::AbstractMatrix, B::TrackedVecOrMat) =  track(zygote_ldiv, A, B)
@grad function zygote_ldiv(A, B)
    Y, back = ZygoteRules.pullback(\, data(A), data(B))
    return Y, Δ->back(data(Δ))
end

function Base.:\(a::Cholesky{<:TrackedReal, <:TrackedArray}, b::AbstractVecOrMat)
    return (a.U \ (a.U' \ b))
end

# SpecialFunctions

SpecialFunctions.logabsgamma(x::TrackedReal) = track(logabsgamma, x)
@grad function SpecialFunctions.logabsgamma(x::Real)
    return logabsgamma(data(x)), Δ -> (digamma(data(x)) * Δ[1],)
end
ZygoteRules.@adjoint function SpecialFunctions.logabsgamma(x::Real)
    return logabsgamma(x), Δ -> (digamma(x) * Δ[1],)
end

# Zygote fill has issues with non-numbers

ZygoteRules.@adjoint function fill(x::T, dims...) where {T}
    return ZygoteRules.pullback(x, dims...) do x, dims...
        return reshape([x for i in 1:prod(dims)], dims)
    end
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
