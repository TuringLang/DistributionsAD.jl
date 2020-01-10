## Generic ##

function Base.fill(
    value::TrackedReal,
    dims::Vararg{Union{Integer, AbstractUnitRange}},
)
    return track(fill, value, dims...)
end
Tracker.@grad function Base.fill(value::Real, dims...)
    return fill(data(value), dims...), function(Δ)
        size(Δ) ≢  dims && error("Dimension mismatch")
        return (sum(Δ), map(_->nothing, dims)...)
    end
end

## StatsFuns ##

logsumexp(x::TrackedArray) = track(logsumexp, x)
Tracker.@grad function logsumexp(x::TrackedArray)
    lse = logsumexp(data(x))
    return lse, Δ -> (Δ .* exp.(x .- lse),)
end

## Linear algebra ##

LinearAlgebra.UpperTriangular(A::TrackedMatrix) = track(UpperTriangular, A)
Tracker.@grad function LinearAlgebra.UpperTriangular(A::AbstractMatrix)
    return UpperTriangular(data(A)), Δ->(UpperTriangular(Δ),)
end

function LinearAlgebra.cholesky(A::TrackedMatrix; check=true)
    factors_info = turing_chol(A, check)
    factors = factors_info[1]
    info = data(factors_info[2])
    return Cholesky{eltype(factors), typeof(factors)}(factors, 'U', info)
end
function turing_chol(A::AbstractMatrix, check)
    chol = cholesky(A, check=check)
    (chol.factors, chol.info)
end
turing_chol(A::TrackedMatrix, check) = track(turing_chol, A, check)
Tracker.@grad function turing_chol(A::AbstractMatrix, check)
    C, back = pullback(unsafe_cholesky, data(A), data(check))
    return (C.factors, C.info), Δ->back((factors=data(Δ[1]),))
end

unsafe_cholesky(x, check) = cholesky(x, check=check)
ZygoteRules.@adjoint function unsafe_cholesky(Σ::Real, check)
    C = cholesky(Σ; check=check)
    return C, function(Δ::NamedTuple)
        issuccess(C) || return (zero(Σ), nothing)
        (Δ.factors[1, 1] / (2 * C.U[1, 1]), nothing)
    end
end
ZygoteRules.@adjoint function unsafe_cholesky(Σ::Diagonal, check)
    C = cholesky(Σ; check=check)
    return C, function(Δ::NamedTuple)
        issuccess(C) || (Diagonal(zero(diag(Δ.factors))), nothing)
        (Diagonal(diag(Δ.factors) .* inv.(2 .* C.factors.diag)), nothing)
    end
end
ZygoteRules.@adjoint function unsafe_cholesky(Σ::Union{StridedMatrix, Symmetric{<:Real, <:StridedMatrix}}, check)
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
logdet_chol_tri(U::TrackedMatrix) = track(logdet_chol_tri, U)
Tracker.@grad function logdet_chol_tri(U::AbstractMatrix)
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
Tracker.@grad function zygote_ldiv(A, B)
    Y, back = pullback(\, data(A), data(B))
    return Y, Δ->back(data(Δ))
end

function Base.:\(a::Cholesky{<:TrackedReal, <:TrackedArray}, b::AbstractVecOrMat)
    return (a.U \ (a.U' \ b))
end

# SpecialFunctions

function SpecialFunctions.logabsgamma(x::TrackedReal)
    v = loggamma(x)
    return v, sign(data(v))
end
