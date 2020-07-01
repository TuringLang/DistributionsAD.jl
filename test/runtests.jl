using DistributionsAD

using Combinatorics
using Distributions
using FiniteDifferences
using PDMats

# Figure out which AD backend to test
const AD = get(ENV, "AD", "All")
if AD == "All"
    @eval using ForwardDiff
    @eval using ReverseDiff
    @eval using Tracker
    @eval using Zygote
elseif AD == "Zygote"
    @eval using Zygote
elseif AD == "ReverseDiff"
    @eval using ReverseDiff
elseif AD == "ForwardDiff_Tracker"
    @eval using ForwardDiff
    @eval using Tracker
else
    error("Unknown AD backend: $AD")
end

using Random, LinearAlgebra, Test

using Distributions: meanlogdet
using DistributionsAD: TuringUniform, #TuringMvNormal, TuringMvLogNormal,
                       TuringPoissonBinomial
using StatsBase: entropy
using StatsFuns: binomlogpdf, logsumexp

const FDM = FiniteDifferences
const GROUP = get(ENV, "GROUP", "All")

if GROUP == "All" || GROUP == "Others"
    include("others.jl")
end

if GROUP == "All" || GROUP == "AD"
    include("ad/utils.jl")
    include("ad/distributions.jl")
end
