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

function adapt_randn(rng::AbstractRNG, x::AbstractArray, dims...)
    adapt(typeof(x), randn(rng, eltype(x), dims...))
end

# TODO: should be replaced by @non_differentiable when
# https://github.com/JuliaDiff/ChainRulesCore.jl/issues/212 is fixed
function ChainRules.rrule(::typeof(adapt_randn), rng::AbstractRNG, x::AbstractArray, dims...)
    function adapt_randn_pullback(ΔQ)
        return (NO_FIELDS, Zero(), Zero(), map(_ -> Zero(), dims)...)
    end
    adapt_randn(rng, x, dims...), adapt_randn_pullback
end
