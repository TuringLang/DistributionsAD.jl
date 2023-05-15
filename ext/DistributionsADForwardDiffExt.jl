module DistributionsADForwardDiffExt

if isdefined(Base, :get_extension)
    using DistributionsAD
    using DistributionsAD: Distributions, Random, StatsFuns
    using ForwardDiff
else
    using ..DistributionsAD
    using ..DistributionsAD: Distributions, Random, StatsFuns
    using ..ForwardDiff
end

function DistributionsAD.adapt_randn(rng::Random.AbstractRNG, x::AbstractArray{<:ForwardDiff.Dual}, dims...)
    return DistributionsAD.adapt_randn(rng, ForwardDiff.valtype(eltype(x)), x, dims...)
end

## Binomial ##

function StatsFuns.binomlogpdf(n::Int, p::ForwardDiff.Dual{T}, x::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val = ForwardDiff.value(p)
    Δ = ForwardDiff.partials(p)
    return FD(StatsFuns.binomlogpdf(n, val, x),  Δ * (x / val - (n - x) / (1 - val)))
end


## Integer dual ##

function Distributions.BetaBinomial(n::ForwardDiff.Dual{<:Any, <:Integer}, α::Real, β::Real; check_args = true)
    return Distributions.BetaBinomial(ForwardDiff.value(n), α, β; check_args = check_args)
end
Distributions.Binomial(n::ForwardDiff.Dual{<:Any, <:Integer}, p::Real) = Distributions.Binomial(ForwardDiff.value(n), p)
function Distributions.Erlang(α::ForwardDiff.Dual{<:Any, <:Integer}, θ::Real; check_args = true)
    return Distributions.Erlang(ForwardDiff.value(α), θ, check_args = check_args)
end


## Negative binomial ##

# Note the definition of NegativeBinomial in Julia is not the same as Wikipedia's.
# Check the docstring of NegativeBinomial, r is the number of successes and
# k is the number of failures
_nbinomlogpdf_grad_1(r, p, k) = k == 0 ? log(p) : sum(1 / (k + r - i) for i in 1:k) + log(p)
_nbinomlogpdf_grad_2(r, p, k) = -k / (1 - p) + r / p

function StatsFuns.nbinomlogpdf(r::ForwardDiff.Dual{T}, p::ForwardDiff.Dual{T}, k::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val_p = ForwardDiff.value(p)
    val_r = ForwardDiff.value(r)
    Δ_r = ForwardDiff.partials(r)
    dr = _nbinomlogpdf_grad_1(val_r, val_p, k)
    Δ_p = ForwardDiff.partials(p)
    dp = _nbinomlogpdf_grad_2(val_r, val_p, k)
    Δ = ForwardDiff._mul_partials(Δ_r, Δ_p, dr, dp)
    return FD(StatsFuns.nbinomlogpdf(val_r, val_p, k),  Δ)
end
function StatsFuns.nbinomlogpdf(r::Real, p::ForwardDiff.Dual{T}, k::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val_p = ForwardDiff.value(p)
    Δ_p = ForwardDiff.partials(p) * _nbinomlogpdf_grad_2(r, val_p, k)
    return FD(StatsFuns.nbinomlogpdf(r, val_p, k),  Δ_p)
end
function StatsFuns.nbinomlogpdf(r::ForwardDiff.Dual{T}, p::Real, k::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val_r = ForwardDiff.value(r)
    Δ_r = ForwardDiff.partials(r) * _nbinomlogpdf_grad_1(val_r, p, k)
    return FD(StatsFuns.nbinomlogpdf(val_r, p, k),  Δ_r)
end

end # module
