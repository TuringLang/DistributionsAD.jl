using DistributionsAD

using ChainRulesTestUtils
using Combinatorics
using Distributions
using FiniteDifferences
using PDMats

using Random, LinearAlgebra, Test

using Distributions: meanlogdet
using DistributionsAD: TuringUniform, TuringMvNormal, TuringMvLogNormal,
                       TuringPoissonBinomial, TuringDirichlet
using StatsBase: entropy
using StatsFuns: binomlogpdf, logsumexp, logistic

import NNlib

Random.seed!(1) # Set seed that all testsets should reset to.

const FDM = FiniteDifferences
const GROUP = get(ENV, "GROUP", "All")

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

if GROUP == "All" || GROUP == "Others"
    include("others.jl")
end

if GROUP == "All" || GROUP == "AD"
    # Create positive definite matrix
    to_posdef(A::AbstractMatrix) = A * A' + I
    to_posdef_diagonal(a::AbstractVector) = Diagonal(a.^2 .+ 1)

    # Create vectors in probability simplex.
    to_simplex(x::AbstractArray) = NNlib.softmax(x; dims=1)
    to_simplex(x::AbstractArray{<:AbstractArray}) = to_simplex.(x)

    if AD == "All" || AD == "ReverseDiff"
        @eval begin
            # Define adjoint for ReverseDiff
            function to_simplex(x::AbstractArray{<:ReverseDiff.TrackedReal})
                return ReverseDiff.track(to_simplex, x)
            end
            ReverseDiff.@grad function to_simplex(x)
                _x = ReverseDiff.value(x)
                y = to_simplex(_x)
                function pullback(∇)
                    return (NNlib.∇softmax(∇, _x, y; dims=1),)
                end
                return y, pullback
            end
        end
    end

    if AD == "All" || AD == "Tracker"
        @eval begin
            # Define adjoints for Tracker
            to_posdef(A::Tracker.TrackedMatrix) = Tracker.track(to_posdef, A)
            Tracker.@grad function to_posdef(A::Tracker.TrackedMatrix)
                data_A = Tracker.data(A)
                S = data_A * data_A' + I
                function pullback(∇)
                    return ((∇ + ∇') * data_A,)
                end
                return S, pullback
            end
        end
    end

    include("ad/utils.jl")
    include("ad/chainrules.jl")
    include("ad/distributions.jl")
end
