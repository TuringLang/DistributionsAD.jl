## Linear Algebra ##

function turing_chol(A::AbstractMatrix, check)
    chol = cholesky(A, check=check)
    (chol.factors, chol.info)
end
function ChainRules.rrule(::typeof(turing_chol), A::AbstractMatrix, check)
    factors, info = turing_chol(A, check)
    function turing_chol_pullback(Ȳ)
        f̄ = Ȳ[1]
        ∂A = ChainRules.chol_blocked_rev(f̄, factors, 25, true)
        return (ChainRules.NO_FIELDS, ∂A, ChainRules.DoesNotExist())
    end
    (factors,info), turing_chol_pullback
end
function turing_chol_back(A::AbstractMatrix, check)
    C, dC_pullback = rrule(turing_chol, A, check)
    function back(Δ)
        _, dC = dC_pullback(Δ)
        (dC, nothing)
    end
    C, back
end

function symm_turing_chol(A::AbstractMatrix, check, uplo)
    chol = cholesky(Symmetric(A, uplo), check=check)
    (chol.factors, chol.info)
end
function ChainRules.rrule(::typeof(symm_turing_chol), A::AbstractMatrix, check, uplo)
    factors, info = symm_turing_chol(A, check, uplo)
    function symm_turing_chol_pullback(Ȳ)
        f̄ = Ȳ[1]
        ∂A = ChainRules.chol_blocked_rev(f̄, factors, 25, true)
        return (ChainRules.NO_FIELDS, ∂A, ChainRules.DoesNotExist(), ChainRules.DoesNotExist())
    end
    return (factors,info), symm_turing_chol_pullback
end
function symm_turing_chol_back(A::AbstractMatrix, check, uplo)
    C, dC_pullback = rrule(symm_turing_chol, A, check, uplo)
    function back(Δ)
        _, dC = dC_pullback(Δ)
        (dC, nothing, nothing)
    end
    C, back
end


# Tracker's implementation of ldiv isn't good. We'll use Zygote's instead.
zygote_ldiv(A::AbstractMatrix, B::AbstractVecOrMat) = A \ B
