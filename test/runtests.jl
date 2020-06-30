using DistributionsAD

using Combinatorics
using Distributions
using FiniteDifferences
using ForwardDiff
using PDMats
using ReverseDiff
using Tracker
using Zygote

using Random, LinearAlgebra, Test

using Distributions: meanlogdet
using DistributionsAD: TuringUniform, #TuringMvNormal, TuringMvLogNormal,
                       TuringPoissonBinomial
using ForwardDiff: Dual
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
