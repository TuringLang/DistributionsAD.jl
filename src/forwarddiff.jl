function adapt_randn(rng::AbstractRNG, x::AbstractArray{<:ForwardDiff.Dual}, dims...)
    return adapt_randn(rng, ForwardDiff.valtype(eltype(x)), x, dims...)
end

## Binomial ##

function binomlogpdf(n::Int, p::ForwardDiff.Dual{T}, x::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val = ForwardDiff.value(p)
    Δ = ForwardDiff.partials(p)
    return FD(binomlogpdf(n, val, x),  Δ * (x / val - (n - x) / (1 - val)))
end


## Integer dual ##

function BetaBinomial(n::ForwardDiff.Dual{<:Any, <:Integer}, α::Real, β::Real; check_args = true)
    return BetaBinomial(ForwardDiff.value(n), α, β; check_args = check_args)
end
Binomial(n::ForwardDiff.Dual{<:Any, <:Integer}, p::Real) = Binomial(ForwardDiff.value(n), p)
function Erlang(α::ForwardDiff.Dual{<:Any, <:Integer}, θ::Real; check_args = true)
    return Erlang(ForwardDiff.value(α), θ, check_args = check_args)
end


## Negative binomial ##

function nbinomlogpdf(r::ForwardDiff.Dual{T}, p::ForwardDiff.Dual{T}, k::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val_p = ForwardDiff.value(p)
    val_r = ForwardDiff.value(r)
    Δ_r = ForwardDiff.partials(r)
    dr = _nbinomlogpdf_grad_1(val_r, val_p, k)
    Δ_p = ForwardDiff.partials(p)
    dp = _nbinomlogpdf_grad_2(val_r, val_p, k)
    Δ = ForwardDiff._mul_partials(Δ_r, Δ_p, dr, dp)
    return FD(nbinomlogpdf(val_r, val_p, k),  Δ)
end
function nbinomlogpdf(r::Real, p::ForwardDiff.Dual{T}, k::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val_p = ForwardDiff.value(p)
    Δ_p = ForwardDiff.partials(p) * _nbinomlogpdf_grad_2(r, val_p, k)
    return FD(nbinomlogpdf(r, val_p, k),  Δ_p)
end
function nbinomlogpdf(r::ForwardDiff.Dual{T}, p::Real, k::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val_r = ForwardDiff.value(r)
    Δ_r = ForwardDiff.partials(r) * _nbinomlogpdf_grad_1(val_r, p, k)
    return FD(nbinomlogpdf(val_r, p, k),  Δ_r)
end

## ForwardDiff broadcasting support ##
# If we use Distributions >= 0.24, then `DISTRIBUTIONS_HAS_GENERIC_UNIVARIATE_PDF` is `true`.
# In Distributions 0.24 `logpdf` is defined for inputs of type `Real` which are then
# converted to the support of the distributions (such as integers) in their concrete implementations.
# Thus it is no needed to have a special function for dual numbers that performs the conversion
# (and actually this method leads to method ambiguity errors since even discrete distributions now
# define logpdf(::MyDistribution, ::Real), see, e.g.,
# JuliaStats/Distributions.jl@ae2d6c5/src/univariate/discrete/binomial.jl#L119).
if !DISTRIBUTIONS_HAS_GENERIC_UNIVARIATE_PDF
    @eval begin
        function Distributions.logpdf(d::DiscreteUnivariateDistribution, k::ForwardDiff.Dual)
            return logpdf(d, convert(Integer, ForwardDiff.value(k)))
        end
    end
end
