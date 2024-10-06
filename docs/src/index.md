# DistributionsAD.jl

This package defines the necessary functions to enable automatic differentiation (AD) of the `logpdf` function from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) using the packages [Tracker.jl](https://github.com/FluxML/Tracker.jl), [Zygote.jl](https://github.com/FluxML/Zygote.jl), [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) and [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl).
The goal of this package is to make the output of `logpdf` differentiable wrt all continuous parameters of a distribution as well as the random variable in the case of continuous distributions.
