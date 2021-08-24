using DistributionsAD

using ChainRulesCore
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
using StatsFuns: StatsFuns, logsumexp, logistic

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
    function to_simplex(x::AbstractArray)
        max = maximum(x; dims=1)
        y .= exp.(x .- max)
        y ./= sum(y; dims=1)
        return y
    end
    to_simplex(x::AbstractArray{<:AbstractArray}) = to_simplex.(x)
    function to_simplex_pullback(ȳ::AbstractArray, y::AbstractArray)
        x̄ = ȳ .* y
        x̄ .= x̄ .- y .* sum(x̄; dims=1)
        return x̄
    end

    if AD == "All" || AD == "ReverseDiff"
        @eval begin
            # Define adjoint for ReverseDiff
            function to_simplex(x::AbstractArray{<:ReverseDiff.TrackedReal})
                return ReverseDiff.track(to_simplex, x)
            end
            ReverseDiff.@grad function to_simplex(x)
                _x = ReverseDiff.value(x)
                y = to_simplex(_x)
                pullback(ȳ) = (to_simplex_pullback(ȳ, y),)
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

            to_simplex(x::Tracker.TrackedArray) = Tracker.track(to_simplex, x)
            Tracker.@grad function to_simplex(x::Tracker.TrackedArray)
                data_x = Tracker.data(x)
                y = to_simplex(data_x)
                pullback(ȳ) = (to_simplex_pullback(ȳ, y),)
                return y, pullback
            end
        end
    end

    include("ad/utils.jl")
    include("ad/chainrules.jl")
    include("ad/distributions.jl")
end
