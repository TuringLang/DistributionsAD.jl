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
