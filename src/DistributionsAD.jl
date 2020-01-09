module DistributionsAD

using PDMats, 
      ForwardDiff, 
      Zygote, 
      LinearAlgebra, 
      Distributions, 
      Random, 
      Combinatorics,
      SpecialFunctions,
      StatsFuns

using Tracker: Tracker, TrackedReal, TrackedVector, TrackedMatrix, TrackedArray,
                TrackedVecOrMat, track, data
using ZygoteRules: ZygoteRules, pullback
using LinearAlgebra: copytri!
using Distributions: AbstractMvLogNormal, 
                     ContinuousMultivariateDistribution

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
                      PoissonBinomial

export TuringScalMvNormal,
       TuringDiagMvNormal,
       TuringDenseMvNormal,
       TuringMvLogNormal,
       TuringPoissonBinomial,
       TuringWishart,
       TuringInverseWishart,
       Multi,
       ArrayDist

include("common.jl")
include("univariate.jl")
include("multivariate.jl")
include("matrixvariate.jl")
include("multi.jl")
include("array_dist.jl")

end
