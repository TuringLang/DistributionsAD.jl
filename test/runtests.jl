using DistributionsAD

using ChainRulesTestUtils
using Combinatorics
using Distributions
using FiniteDifferences
using PDMats
using Requires

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
using StatsFuns: binomlogpdf, logsumexp, logistic

Random.seed!(1) # Set seed that all testsets should reset to.

const FDM = FiniteDifferences
const GROUP = get(ENV, "GROUP", "All")

# Create positive definite matrix
to_posdef(A::AbstractMatrix) = A * A' + I
to_posdef_diagonal(a::AbstractVector) = Diagonal(a.^2 .+ 1)

@require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
    # Define adjoints for Tracker
    to_posdef(A::TrackedMatrix) = Tracker.track(to_posdef, A)
    Tracker.@grad function to_posdef(A::TrackedMatrix)
        data_A = Tracker.data(A)
        S = data_A * data_A' + I
        function pullback(∇)
            return ((∇ + ∇') * data_A,)
        end
        return S, pullback
    end
end

if GROUP == "All" || GROUP == "Others"
    include("others.jl")
end

if GROUP == "All" || GROUP == "AD"
    include("ad/utils.jl")
    include("ad/chainrules.jl")
    include("ad/distributions.jl")
end
