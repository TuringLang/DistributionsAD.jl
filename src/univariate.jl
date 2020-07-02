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
Distributions.logpdf(d::TuringUniform, x::AbstractArray) = uniformlogpdf.(d.a, d.b, x)
Base.minimum(d::TuringUniform) = d.a
Base.maximum(d::TuringUniform) = d.b

function uniformlogpdf(a, b, x)
    c = -log(b - a)
    if a <= x <= b
        return c
    else
        return oftype(c, -Inf)
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
function TuringPoissonBinomial(p::AbstractArray{<:Real}; check_args = true)
    pb = Distributions.poissonbinomial_pdf_fft(p)
    ϵ = eps(eltype(pb))
    check_args && @assert all(x -> x >= -ϵ, pb) && isapprox(sum(pb), 1, atol = ϵ)
    return TuringPoissonBinomial(p, pb)
end
function logpdf(d::TuringPoissonBinomial{T}, k::Int) where T<:Real
    insupport(d, k) ? log(d.pmf[k + 1]) : -T(Inf)
end
quantile(d::TuringPoissonBinomial, x::Float64) = quantile(Categorical(d.pmf), x) - 1
Base.minimum(d::TuringPoissonBinomial) = 0
Base.maximum(d::TuringPoissonBinomial) = length(d.p)


## Categorical ##

function Base.convert(
    ::Type{Distributions.DiscreteNonParametric{T,P,Ts,Ps}},
    d::Distributions.DiscreteNonParametric{T,P,Ts,Ps},
) where {T<:Real,P<:Real,Ts<:AbstractVector{T},Ps<:AbstractVector{P}}
    DiscreteNonParametric{T,P,Ts,Ps}(support(d), probs(d), check_args=false)
end

# Fix SubArray support
function Distributions.DiscreteNonParametric{T,P,Ts,Ps}(
    vs::Ts,
    ps::Ps;
    check_args=true,
) where {T<:Real, P<:Real, Ts<:AbstractVector{T}, Ps<:SubArray{P, 1}}
    cps = ps[:]
    return DiscreteNonParametric{T,P,Ts,typeof(cps)}(vs, cps; check_args = check_args)
end

# ## ForwardDiff broadcasting support ##
# 
# function Distributions.logpdf(d::DiscreteUnivariateDistribution, k::ForwardDiff.Dual)
#     return logpdf(d, convert(Integer, ForwardDiff.value(k)))
# end
