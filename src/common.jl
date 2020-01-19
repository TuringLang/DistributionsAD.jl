## Generic ##

function Base.fill(
    value::Tracker.TrackedReal,
    dims::Vararg{Union{Integer, AbstractUnitRange}},
)
    return Tracker.track(fill, value, dims...)
end
Tracker.@grad function Base.fill(value::Real, dims...)
    return fill(Tracker.data(value), dims...), function(Δ)
        size(Δ) ≢  dims && error("Dimension mismatch")
        return (sum(Δ), map(_->nothing, dims)...)
    end
end

## StatsFuns ##

logsumexp(x::Tracker.TrackedArray) = Tracker.track(logsumexp, x)
Tracker.@grad function logsumexp(x::Tracker.TrackedArray)
    lse = logsumexp(Tracker.data(x))
    return lse,
          Δ->(Δ .* exp.(x .- lse),)
end

## Linear algebra ##

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
    C, back = ZygoteRules.pullback(unsafe_cholesky, Tracker.data(A), Tracker.data(check))
    return (C.factors, C.info), Δ->back((factors=Tracker.data(Δ[1]),))
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
logdet_chol_tri(U::Tracker.TrackedMatrix) = Tracker.track(logdet_chol_tri, U)
Tracker.@grad function logdet_chol_tri(U::AbstractMatrix)
    U_data = Tracker.data(U)
    return logdet_chol_tri(U_data), Δ->(Matrix(Diagonal(2 .* Δ ./ diag(U_data))),)
end

function LinearAlgebra.logdet(C::Cholesky{<:Tracker.TrackedReal, <:Tracker.TrackedMatrix})
    return logdet_chol_tri(C.U)
end

# Tracker's implementation of ldiv isn't good. We'll use Zygote's instead.
zygote_ldiv(A::AbstractMatrix, B::AbstractVecOrMat) = A \ B
function zygote_ldiv(A::Tracker.TrackedMatrix, B::Tracker.TrackedVecOrMat)
    return Tracker.track(zygote_ldiv, A, B)
end
function zygote_ldiv(A::Tracker.TrackedMatrix, B::AbstractVecOrMat)
    return Tracker.track(zygote_ldiv, A, B)
end
zygote_ldiv(A::AbstractMatrix, B::Tracker.TrackedVecOrMat) =  Tracker.track(zygote_ldiv, A, B)
Tracker.@grad function zygote_ldiv(A, B)
    Y, back = ZygoteRules.pullback(\, Tracker.data(A), Tracker.data(B))
    return Y, Δ->back(Tracker.data(Δ))
end

function Base.:\(a::Cholesky{<:Tracker.TrackedReal, <:Tracker.TrackedArray}, b::AbstractVecOrMat)
    return (a.U \ (a.U' \ b))
end

## PDMats ##

PDMats.invquad(Σ::PDiagMat, x::Tracker.TrackedVector) = sum(abs2.(x) ./ Σ.diag)
PDMats.invquad(Σ::PDMat, x::Tracker.TrackedVector) = sum(abs2, zygote_ldiv(Σ.chol.L, x))
