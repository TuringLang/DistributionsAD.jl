module DistributionsAD

using PDMats, 
      ForwardDiff, 
      Zygote, 
      Tracker, 
      LinearAlgebra, 
      Distributions, 
      Random, 
      Combinatorics

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

export TuringDiagNormal,
       TuringMvNormal,
       TuringMvLogNormal,
       TuringPoissonBinomial

include("common.jl")
include("univariate.jl")
include("multivariate.jl")

end
