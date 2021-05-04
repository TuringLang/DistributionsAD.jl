## Uniform ##

struct TuringUniform{T} <: ContinuousUnivariateDistribution
    a::T
    b::T
end
TuringUniform() = TuringUniform(0.0, 1.0)
function TuringUniform(a::Int, b::Int)
    return TuringUniform{Float64}(Float64(a), Float64(b))
end
function TuringUniform(a::Real, b::Real)
    T = promote_type(typeof(a), typeof(b))
    return TuringUniform{T}(T(a), T(b))
end
Distributions.logpdf(d::TuringUniform, x::Real) = uniformlogpdf(d.a, d.b, x)

Base.minimum(d::TuringUniform) = d.a
Base.maximum(d::TuringUniform) = d.b

function uniformlogpdf(a, b, x)
    diff = b - a
    if a <= x <= b
        return -log(diff)
    else
        return log(zero(diff))
    end
end

if VERSION < v"1.2"
    Base.inv(::Irrational{:π}) = 1/π
end


## PoissonBinomial ##

struct TuringPoissonBinomial{T<:Real, TV1<:AbstractVector{T}, TV2<:AbstractVector} <: DiscreteUnivariateDistribution
    p::TV1
    pmf::TV2
end

# if available use the faster `poissonbinomial_pdf`
@eval begin
    function TuringPoissonBinomial(p::AbstractArray{<:Real}; check_args = true)
        pb = $(isdefined(Distributions, :poissonbinomial_pdf) ? Distributions.poissonbinomial_pdf : Distributions.poissonbinomial_pdf_fft)(p)
        ϵ = eps(eltype(pb))
        check_args && @assert all(x -> x >= -ϵ, pb) && isapprox(sum(pb), 1; atol=ϵ)
        return TuringPoissonBinomial(p, pb)
    end
end

function logpdf(d::TuringPoissonBinomial{T}, k::Int) where T<:Real
    insupport(d, k) ? log(d.pmf[k + 1]) : -T(Inf)
end
quantile(d::TuringPoissonBinomial, x::Float64) = quantile(Categorical(d.pmf), x) - 1
Base.minimum(d::TuringPoissonBinomial) = 0
Base.maximum(d::TuringPoissonBinomial) = length(d.p)
