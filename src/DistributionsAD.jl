module DistributionsAD

using PDMats, 
      ForwardDiff, 
      Zygote, 
      ZygoteRules,
      Tracker, 
      LinearAlgebra, 
      Distributions, 
      Random, 
      Combinatorics,
      SpecialFunctions

using Tracker: TrackedReal
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
       TuringInverseWishart

include("common.jl")
include("univariate.jl")
include("multivariate.jl")
include("matrixvariate.jl")

end
