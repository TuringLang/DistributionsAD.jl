## Linear Algebra ##

function turing_chol(A::AbstractMatrix, check)
    chol = cholesky(A, check=check)
    (chol.factors, chol.info)
end
function turing_chol_back(A::AbstractMatrix, check)
    C, chol_pullback = rrule(cholesky, A, Val(false), check=check)
    function back(Δ)
        Ȳ = Composite{typeof(C)}((U=Δ[1]))
        ∂C = chol_pullback(Ȳ)[2]
        (∂C, nothing)
    end
    (C.factors,C.info), back
end

function symm_turing_chol(A::AbstractMatrix, check, uplo)
    chol = cholesky(Symmetric(A, uplo), check=check)
    (chol.factors, chol.info)
end
function symm_turing_chol_back(A::AbstractMatrix, check, uplo)
    C, chol_pullback = rrule(cholesky, Symmetric(A,uplo), Val(false), check=check)
    function back(Δ)
        Ȳ = Composite{typeof(C)}((U=Δ[1]))
        ∂C = chol_pullback(Ȳ)[2]
        (∂C, nothing, nothing)
    end
    (C.factors, C.info), back
end


# Tracker's implementation of ldiv isn't good. We'll use Zygote's instead.
zygote_ldiv(A::AbstractMatrix, B::AbstractVecOrMat) = A \ B

# fixes `randn` on GPU (https://github.com/TuringLang/DistributionsAD.jl/pull/108)
function adapt_randn(rng::AbstractRNG, x::AbstractArray, dims...)
    return adapt_randn(rng, eltype(x), x, dims...)
end
function adapt_randn(rng::AbstractRNG, ::Type{T}, x::AbstractArray, dims...) where {T}
    return adapt(parameterless_type(x), randn(rng, T, dims...))
end

# required by Adapt >= 3.3.0: https://github.com/SciML/OrdinaryDiffEq.jl/issues/1369
Base.@pure __parameterless_type(T) = Base.typename(T).wrapper
parameterless_type(x) = parameterless_type(typeof(x))
parameterless_type(x::Type) = __parameterless_type(x)

ChainRulesCore.@non_differentiable adapt_randn(::Any...)
