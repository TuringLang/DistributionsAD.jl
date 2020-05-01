module DistributionsAD

using PDMats, 
      ForwardDiff, 
      LinearAlgebra, 
      Distributions, 
      Random, 
      Combinatorics,
      SpecialFunctions,
      StatsFuns,
      Compat,
      Requires

using Tracker: Tracker, TrackedReal, TrackedVector, TrackedMatrix, TrackedArray,
                TrackedVecOrMat, track, @grad, data
using SpecialFunctions: logabsgamma, digamma
using LinearAlgebra: copytri!, AbstractTriangular
using Distributions: AbstractMvLogNormal, 
                     ContinuousMultivariateDistribution
using DiffRules, SpecialFunctions, FillArrays
using ForwardDiff: @define_binary_dual_op # Needed for `eval`ing diffrules here
using Base.Iterators: drop

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
import ZygoteRules

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
include("univariate.jl")
include("multivariate.jl")
include("mvcategorical.jl")
include("matrixvariate.jl")
include("flatten.jl")
include("arraydist.jl")
include("filldist.jl")
@init @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
    include("reversediff.jl")
end

end
