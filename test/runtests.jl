using DistributionsAD

using Combinatorics
using Distributions
using FiniteDifferences
using PDMats

# Figure out which AD backend to test
const AD = get(ENV, "AD", "All")
if AD == "All" || AD == "ForwardDiff_Tracker"
    @eval using ForwardDiff
    @eval using Tracker
end
if AD == "All" || AD == "Zygote"
    @eval using Zygote
end
if AD == "All" || AD == "ReverseDiff"
    @eval using ReverseDiff
end

using Random, LinearAlgebra, Test

using Distributions: meanlogdet
using DistributionsAD: TuringUniform, TuringMvNormal, TuringMvLogNormal,
                       TuringPoissonBinomial
using StatsBase: entropy
using StatsFuns: binomlogpdf, logsumexp

const FDM = FiniteDifferences
const GROUP = get(ENV, "GROUP", "All")

include("PosDef.jl")

if GROUP == "All" || GROUP == "Others"
    include("others.jl")
end

if GROUP == "All" || GROUP == "AD"
    include("ad/utils.jl")
    include("ad/distributions.jl")
end
