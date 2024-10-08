using DistributionsAD

using Combinatorics
using Distributions
using Documenter
using PDMats
import LazyArrays

using Random, LinearAlgebra, Test

using Distributions: meanlogdet
using DistributionsAD: TuringMvNormal, TuringMvLogNormal,
                       TuringPoissonBinomial, TuringDirichlet
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

# Run doctests (but not on older versions as rng seed behaves differently)
@static if VERSION >= v"1.10"
    @testset "doctests" begin
        DocMeta.setdocmeta!(
            DistributionsAD,
            :DocTestSetup,
            :(using DistributionsAD);
            recursive=true,
        )
        doctest(DistributionsAD; manual=false)
    end
end
