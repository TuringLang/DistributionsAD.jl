var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/#Functions","page":"API","title":"Functions","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"filldist\narraydist","category":"page"},{"location":"api/#DistributionsAD.filldist","page":"API","title":"DistributionsAD.filldist","text":"filldist(d::Distribution, ns...)\n\nCreate a product distribution from a single distribution and a list of dimension sizes. If size(d) is (d1, d2, ...) and ns is (n1, n2, ...), then the resulting distribution will have size (d1, d2, ..., n1, n2, ...).\n\nThe default behaviour is to use Distributions.product_distribution, with FillArrays.Fill supplied as the array argument. However, this behaviour is specialised in some instances, such as the one shown below.\n\nWhen sampling from the resulting distribution, the output will be an array where each element is sampled from the original distribution d.\n\nExamples\n\njulia> d = filldist(Normal(0, 1), 4, 5)  # size(Normal(0, 1)) == ()\nDistributionsAD.MatrixOfUnivariate{Continuous, Normal{Float64}, FillArrays.Fill{Normal{Float64}, 2, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}}(\ndists: Fill(Normal{Float64}(μ=0.0, σ=1.0), 4, 5)\n)\n\njulia> size(d)\n(4, 5)\n\njulia> Random.seed!(42); rand(d)\n4×5 Matrix{Float64}:\n -0.363357   0.816307  -2.11433     0.433886  -0.206613\n  0.251737   0.476738   0.0437817  -0.39544   -0.310744\n -0.314988  -0.859555  -0.825335    0.517131  -0.0404734\n -0.311252  -1.46929    0.840289    1.44722    0.104771\n\n\n\n\n\n","category":"function"},{"location":"api/#DistributionsAD.arraydist","page":"API","title":"DistributionsAD.arraydist","text":"arraydist(dists::AbstractArray{<:Distribution})\n\nCreate a product distribution from an array of sub-distributions. Each element of dists should have the same size. If the size of each element is (d1, d2, ...), and size(dists) is (n1, n2, ...), then the resulting distribution will have size (d1, d2, ..., n1, n2, ...).\n\nThe default behaviour is to directly use Distributions.product_distribution, although this can sometimes be specialised.\n\nExamples\n\njulia> d1 = arraydist([Normal(0, 1), Normal(10, 1)])\nProduct{Continuous, Normal{Float64}, Vector{Normal{Float64}}}(v=Normal{Float64}[Normal{Float64}(μ=0.0, σ=1.0), Normal{Float64}(μ=10.0, σ=1.0)])\n\njulia> size(d1)\n(2,)\n\njulia> Random.seed!(42); rand(d1)\n2-element Vector{Float64}:\n 0.7883556016042917\n 9.1201414040456\n\njulia> d2 = arraydist([Normal(0, 1) Normal(5, 1); Normal(10, 1) Normal(15, 1)])\nDistributionsAD.MatrixOfUnivariate{Continuous, Normal{Float64}, Matrix{Normal{Float64}}}(\ndists: Normal{Float64}[Normal{Float64}(μ=0.0, σ=1.0) Normal{Float64}(μ=5.0, σ=1.0); Normal{Float64}(μ=10.0, σ=1.0) Normal{Float64}(μ=15.0, σ=1.0)]\n)\n\njulia> size(d2)\n(2, 2)\n\njulia> Random.seed!(42); rand(d2)\n2×2 Matrix{Float64}:\n 0.788356   4.12621\n 9.12014   14.2667\n\n\n\n\n\n","category":"function"},{"location":"#DistributionsAD.jl","page":"Home","title":"DistributionsAD.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package defines the necessary functions to enable automatic differentiation (AD) of the logpdf function from Distributions.jl using the packages Tracker.jl, Zygote.jl, ForwardDiff.jl and ReverseDiff.jl. The goal of this package is to make the output of logpdf differentiable wrt all continuous parameters of a distribution as well as the random variable in the case of continuous distributions.","category":"page"}]
}
