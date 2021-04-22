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

@non_differentiable adapt_randn(::Any...)

# PoissonBinomial

# compute matrix of partial derivatives [∂P(X=j-1)/∂pᵢ]_{i=1,…,n; j=1,…,n+1}
#
# This uses the same dynamic programming "trick" as for the computation of the primals
# in Distributions
#
# Reference (for the primal):
#
#      Marlin A. Thomas & Audrey E. Taub (1982)
#      Calculating binomial probabilities when the trial probabilities are unequal,
#      Journal of Statistical Computation and Simulation, 14:2, 125-131, DOI: 10.1080/00949658208810534
function poissonbinomial_partialderivatives(p)
    n = length(p)
    A = zeros(eltype(p), n, n + 1)
    @inbounds for j in 1:n
        A[j, end] = 1
    end
    @inbounds for (i, pi) in enumerate(p)
        qi = 1 - pi
        for k in (n - i + 1):n
            kp1 = k + 1
            for j in 1:(i - 1)
                A[j, k] = pi * A[j, k] + qi * A[j, kp1]
            end
            for j in (i+1):n
                A[j, k] = pi * A[j, k] + qi * A[j, kp1]
            end
        end
        for j in 1:(i-1)
            A[j, end] *= pi
        end
        for j in (i+1):n
            A[j, end] *= pi
        end
    end
    @inbounds for j in 1:n, i in 1:n
        A[i, j] -= A[i, j+1]
    end
    return A
end
