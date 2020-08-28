module DistributionsAD

using PDMats, 
      LinearAlgebra, 
      Distributions, 
      Random, 
      SpecialFunctions,
      StatsFuns,
      Compat,
      Requires,
      ZygoteRules,
      ChainRules,  # needed for `ChainRules.chol_blocked_rev`
      ChainRulesCore,
      FillArrays

using SpecialFunctions: logabsgamma, digamma
using LinearAlgebra: copytri!, AbstractTriangular
using Distributions: AbstractMvLogNormal, 
                     ContinuousMultivariateDistribution
using Base.Iterators: drop

import StatsBase
import StatsFuns: logsumexp, 
                  binomlogpdf, 
                  nbinomlogpdf, 
                  poislogpdf, 
                  nbetalogpdf
import Distributions: MvNormal, 
                      MvLogNormal, 
                      poissonbinomial_pdf_fft, 
                      logpdf, 
                      quantile, 
                      PoissonBinomial,
                      Binomial,
                      BetaBinomial,
                      Erlang

export TuringScalMvNormal,
       TuringDiagMvNormal,
       TuringDenseMvNormal,
       TuringMvLogNormal,
       TuringPoissonBinomial,
       TuringWishart,
       TuringInverseWishart,
       arraydist,
       filldist

include("common.jl")
include("arraydist.jl")
include("filldist.jl")
include("univariate.jl")
include("multivariate.jl")
include("mvcategorical.jl")
include("matrixvariate.jl")
include("flatten.jl")

include("chainrules.jl")
include("zygote.jl")

@init begin
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" begin
        using .ForwardDiff: @define_binary_dual_op # Needed for `eval`ing diffrules here
        include("forwarddiff.jl")

        # loads adjoint for `poissonbinomial_pdf_fft`
        include("zygote_forwarddiff.jl")
    end

    @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
        include("reversediff.jl")
    end

    @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
        using DiffRules
        using SpecialFunctions
        using LinearAlgebra: AbstractTriangular
        using .Tracker: Tracker, TrackedReal, TrackedVector, TrackedMatrix,
                        TrackedArray, TrackedVecOrMat, track, @grad, data
        include("tracker.jl")
    end

    @require LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02" begin
        using .LazyArrays: BroadcastArray, BroadcastVector, LazyArray

        const LazyVectorOfUnivariate{
            S<:ValueSupport,
            T<:UnivariateDistribution{S},
            Tdists<:BroadcastVector{T},
        } = VectorOfUnivariate{S,T,Tdists}

        function Distributions._logpdf(
            dist::LazyVectorOfUnivariate,
            x::AbstractVector{<:Real},
        )
            return sum(copy(logpdf.(dist.v, x)))
        end

        function Distributions.logpdf(
            dist::LazyVectorOfUnivariate,
            x::AbstractMatrix{<:Real},
        )
            size(x, 1) == length(dist) ||
                throw(DimensionMismatch("Inconsistent array dimensions."))
            return vec(sum(copy(logpdf.(dists, x)), dims = 1))
        end

        const LazyMatrixOfUnivariate{
            S<:ValueSupport,
            T<:UnivariateDistribution{S},
            Tdists<:BroadcastArray{T,2},
        } = MatrixOfUnivariate{S,T,Tdists}

        function Distributions._logpdf(
            dist::LazyMatrixOfUnivariate,
            x::AbstractMatrix{<:Real},
        )
            return sum(copy(logpdf.(dist.dists, x)))
        end

        lazyarray(f, x...) = LazyArray(Base.broadcasted(f, x...))
        export lazyarray
    end
end

end
