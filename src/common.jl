## Linear Algebra ##

function turing_chol(A::AbstractMatrix, check)
    chol = cholesky(A, check=check)
    (chol.factors, chol.info)
end
function symm_turing_chol(A::AbstractMatrix, check, uplo)
    chol = cholesky(Symmetric(A, uplo), check=check)
    (chol.factors, chol.info)
end

# Tracker's implementation of ldiv isn't good. We'll use Zygote's instead.
zygote_ldiv(A::AbstractMatrix, B::AbstractVecOrMat) = A \ B
