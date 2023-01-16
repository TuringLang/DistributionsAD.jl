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
      FillArrays,
      Adapt

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
include("matrixvariate.jl")
include("flatten.jl")

include("zygote.jl")

@init begin
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" begin
        using .ForwardDiff
        using .ForwardDiff: @define_binary_dual_op # Needed for `eval`ing diffrules here
        include("forwarddiff.jl")
    end

    @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
        # ensures that we can load ForwardDiff without depending on it
        # (it is a dependency of ReverseDiff and therefore always available)
        @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" begin
            include("reversediff.jl")
        end
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
        include("lazyarrays.jl")
    end
end

end
