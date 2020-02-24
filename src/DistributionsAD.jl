module DistributionsAD

using PDMats, 
      ForwardDiff, 
      Zygote, 
      LinearAlgebra, 
      Distributions, 
      Random, 
      Combinatorics,
      SpecialFunctions,
      StatsFuns,
      Compat

using Tracker: Tracker, TrackedReal, TrackedVector, TrackedMatrix, TrackedArray,
                TrackedVecOrMat, track, @grad, data
using SpecialFunctions: logabsgamma, digamma
using ZygoteRules: ZygoteRules, @adjoint, pullback
using LinearAlgebra: copytri!
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
                      PoissonBinomial

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
include("matrixvariate.jl")
include("flatten.jl")
include("arraydist.jl")
include("filldist.jl")

end
