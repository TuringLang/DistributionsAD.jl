# DistributionsAD.jl

![Zygote tests](https://github.com/TuringLang/DistributionsAD.jl/workflows/Zygote%20tests/badge.svg)

![ForwardDiff tests](https://github.com/TuringLang/DistributionsAD.jl/workflows/ForwardDiff%20tests/badge.svg)

[![Zygote tests](https://github.com/TuringLang/DistributionsAD.jl/workflows/Zygote%20tests/badge.svg?branch=master)](https://github.com/TuringLang/DistributionsAD.jl/actions?query=workflow%3A%22Zygote+tests%22)

[![ReverseDiff tests](https://github.com/TuringLang/DistributionsAD.jl/workflows/ReverseDiff%20tests/badge.svg)](https://github.com/TuringLang/DistributionsAD.jl/actions?query=workflow%3A%22ReverseDiff+tests%22)


This package defines the necessary functions to enable automatic differentiation (AD) of the `logpdf` function from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) using the packages [Tracker.jl](https://github.com/FluxML/Tracker.jl), [Zygote.jl](https://github.com/FluxML/Zygote.jl), [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) and [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl). The goal of this package is to make the output of `logpdf` differentiable wrt all continuous parameters of a distribution as well as the random variable in the case of continuous distributions.

AD of `logpdf` is fully supported and tested for the following distributions wrt all combinations of continuous variables (distribution parameters and/or the random variable) and using all defined distribution constructors:
- Univariate discrete
    - `Bernoulli`
    - `BetaBinomial`
    - `Binomial`
    - `Categorical`
    - `Geometric`
    - `NegativeBinomial`
    - `Poisson`
    - `PoissonBinomial`
    - `Skellam`
- Univariate continuous
    - `Arcsine`
    - `Beta`
    - `BetaPrime`
    - `Biweight`
    - `Cauchy`
    - `Chi`
    - `Chisq`
    - `Cosine`
    - `Epanechnikov`
    - `Erlang`
    - `Exponential`
    - `FDist`
    - `Frechet`
    - `Gamma`
    - `GeneralizedExtremeValue`
    - `GeneralizedPareto`
    - `Gumbel`
    - `InverseGamma`
    - `InverseGaussian`
    - `Kolmogorov`
    - `Laplace`
    - `Levy`
    - `LocationScale`
    - `Logistic`
    - `LogitNormal`
    - `LogNormal`
    - `Normal`
    - `NormalCanon`
    - `NormalInverseGaussian`
    - `Pareto`
    - `PGeneralizedGaussian`
    - `Rayleigh`
    - `Semicircle`
    - `SymTriangularDist`
    - `TDist`
    - `TriangularDist`
    - `Triweight`
    - `Uniform`
    - `Weibull`
- Multivariate continuous
    - `MvLogNormal`
    - `MvNormal`
- Matrix-variate continuous
    - `MatrixBeta`
    - `Wishart`
    - `InverseWishart`

# Get Involved

A number of distributions are still either broken or not fully supported for various reasons. See [this issue](https://github.com/TuringLang/DistributionsAD.jl/issues/2). If you can fix any of the broken ones, a PR is welcome!
