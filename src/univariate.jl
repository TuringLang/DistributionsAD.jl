## PoissonBinomial ##

struct TuringPoissonBinomial{T<:Real, TV1<:AbstractVector{T}, TV2<:AbstractVector} <: DiscreteUnivariateDistribution
    p::TV1
    pmf::TV2
end

# use the faster `poissonbinomial_pdf`
function TuringPoissonBinomial(p::AbstractArray{<:Real}; check_args = true)
    pb = Distributions.poissonbinomial_pdf(p)
    ϵ = eps(eltype(pb))
    check_args && @assert all(x -> x >= -ϵ, pb) && isapprox(sum(pb), 1; atol=ϵ)
    return TuringPoissonBinomial(p, pb)
end

function logpdf(d::TuringPoissonBinomial{T}, k::Int) where T<:Real
    insupport(d, k) ? log(d.pmf[k + 1]) : -T(Inf)
end
quantile(d::TuringPoissonBinomial, x::Float64) = quantile(Categorical(d.pmf), x) - 1
Base.minimum(d::TuringPoissonBinomial) = 0
Base.maximum(d::TuringPoissonBinomial) = length(d.p)
