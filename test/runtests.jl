using DistributionsAD

using Combinatorics
using Distributions
using PDMats

using Random, LinearAlgebra, Test

using Distributions: meanlogdet
using DistributionsAD: TuringMvNormal, TuringMvLogNormal,
                       TuringPoissonBinomial, TuringDirichlet
using StatsBase: entropy
using StatsFuns: StatsFuns, logsumexp, logistic

@static if VERSION >= v"1.8"
  using Pkg; Pkg.status(outdated=true) # show reasons why packages are held back
end

Random.seed!(1) # Set seed that all testsets should reset to.

const GROUP = get(ENV, "GROUP", "All")

include("test_utils.jl")

if GROUP == "All" || GROUP == "Others"
    include("others.jl")
end

if GROUP == "All" || GROUP in ("ForwardDiff", "Zygote", "ReverseDiff", "Tracker")
    include("ad/utils.jl")
    include("ad/others.jl")
    include("ad/distributions.jl")
end
