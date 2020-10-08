function adapt_randn(rng::AbstractRNG, x::AbstractArray{<:ForwardDiff.Dual}, dims...)
    adapt(typeof(x), randn(rng, ForwardDiff.valtype(eltype(x)), dims...))
end

## Binomial ##

function binomlogpdf(n::Int, p::ForwardDiff.Dual{T}, x::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val = ForwardDiff.value(p)
    Δ = ForwardDiff.partials(p)
    return FD(binomlogpdf(n, val, x),  Δ * (x / val - (n - x) / (1 - val)))
end


## Poisson

function poislogpdf(v::ForwardDiff.Dual{T}, x::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val = ForwardDiff.value(v)
    Δ = ForwardDiff.partials(v)
    return FD(poislogpdf(val, x), Δ * (x/val - 1))
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
if !DISTRIBUTIONS_HAS_GENERIC_UNIVARIATE_PDF
    @eval begin
        function Distributions.logpdf(d::DiscreteUnivariateDistribution, k::ForwardDiff.Dual)
            return logpdf(d, convert(Integer, ForwardDiff.value(k)))
        end
    end
end
