using DistributionsAD

using Combinatorics
using Distributions
using FiniteDifferences
using PDMats

# Figure out which AD backend to test
const AD = get(ENV, "AD", "All")
if AD == "All" || AD == "ForwardDiff"
    @eval using ForwardDiff
end
if AD == "All" || AD == "Zygote"
    @eval using Zygote
end
if AD == "All" || AD == "ReverseDiff"
    @eval using ReverseDiff
end
if AD == "All" || AD == "Tracker"
    @eval using Tracker
end

using Random, LinearAlgebra, Test

using Distributions: meanlogdet
using DistributionsAD: TuringUniform, TuringMvNormal, TuringMvLogNormal,
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
