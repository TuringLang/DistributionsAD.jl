module DistributionsAD

using PDMats,
      LinearAlgebra,
      Distributions,
      Random,
      SpecialFunctions,
      StatsFuns,
      Compat,
      ZygoteRules,
      ChainRules,  # needed for `ChainRules.chol_blocked_rev`
      ChainRulesCore,
      FillArrays,
      Adapt

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

# Empty definition, function requires the LazyArrays extension
function lazyarray end
export lazyarray

if !isdefined(Base, :get_extension)
    using Requires
end
function __init__()
    # Better error message if users forget to load LazyArrays
    Base.Experimental.register_error_hint(MethodError) do io, exc, arg_types, kwargs
        if exc.f === lazyarray
            print(io, "\\nDid you forget to load LazyArrays?")
        end
    end
    @static if !isdefined(Base, :get_extension)
        @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" include("../ext/DistributionsADForwardDiffExt.jl")
        @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" include("../ext/DistributionsADReverseDiffExt.jl")
        @require Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" include("../ext/DistributionsADTrackerExt.jl")
        @require LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02" include("../ext/DistributionsADLazyArraysExt.jl")
    end
end

end
