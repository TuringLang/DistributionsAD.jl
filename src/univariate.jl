## Uniform ##

logpdf(d::Uniform, x::TrackedReal) = uniformlogpdf(d.a, d.b, x)
uniformlogpdf(a, b, x) = Tracker.track(uniformlogpdf, a, b, x)
Tracker.@grad function uniformlogpdf(a, b, x)
    xd = Tracker.data(x)
    T = typeof(xd)
    l = logpdf(Uniform(a, b), Tracker.data(x))
    f = isfinite(l)
    da = 1/(b - a)
    n = T(NaN)
    return l, Δ->(f ? da : n, f ? -da : n, f ? zero(T) : n)
end

## Semicircle ##

logpdf(d::Semicircle{<:Real}, x::TrackedReal) = semicirclelogpdf(d.r, x)
logpdf(d::Semicircle{<:TrackedReal}, x::Real) = semicirclelogpdf(d.r, x)
logpdf(d::Semicircle{<:TrackedReal}, x::TrackedReal) = semicirclelogpdf(d.r, x)
semicirclelogpdf(r::TrackedReal, x::Real) = Tracker.track(semicirclelogpdf, r, x)
semicirclelogpdf(r::Real, x::TrackedReal) = Tracker.track(semicirclelogpdf, r, x)
semicirclelogpdf(r::TrackedReal, x::TrackedReal) = Tracker.track(semicirclelogpdf, r, x)
Tracker.@grad function semicirclelogpdf(r, x)
    rd = Tracker.data(r)
    xd = Tracker.data(x)
    xx, rr = promote(xd, float(rd))
    d = Semicircle(rr)
    T = typeof(xx)
    l = logpdf(d, xx)
    f = isfinite(l)
    n = T(NaN)
    return l, function (Δ) 
        diffsq = rr^2 - xx^2
        (f ? Δ*(-2/rr + rr/diffsq) : n, f ? Δ*(-xx/diffsq) : n)
    end
end

## Binomial ##

binomlogpdf(n::Int, p::Tracker.TrackedReal, x::Int) = Tracker.track(binomlogpdf, n, p, x)
Tracker.@grad function binomlogpdf(n::Int, p::Tracker.TrackedReal, x::Int)
    return binomlogpdf(n, Tracker.data(p), x),
        Δ->(nothing, Δ * (x / p - (n - x) / (1 - p)), nothing)
end

function binomlogpdf(n::Int, p::ForwardDiff.Dual{T}, x::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val = ForwardDiff.value(p)
    Δ = ForwardDiff.partials(p)
    return FD(binomlogpdf(n, val, x),  Δ * (x / val - (n - x) / (1 - val)))
end

## Negative binomial ##

# Note the definition of NegativeBinomial in Julia is not the same as Wikipedia's.
# Check the docstring of NegativeBinomial, r is the number of successes and
# k is the number of failures
_nbinomlogpdf_grad_1(r, p, k) = k == 0 ? log(p) : sum(1 / (k + r - i) for i in 1:k) + log(p)
_nbinomlogpdf_grad_2(r, p, k) = -k / (1 - p) + r / p

nbinomlogpdf(n::Tracker.TrackedReal, p::Tracker.TrackedReal, x::Int) = Tracker.track(nbinomlogpdf, n, p, x)
nbinomlogpdf(n::Real, p::Tracker.TrackedReal, x::Int) = Tracker.track(nbinomlogpdf, n, p, x)
nbinomlogpdf(n::Tracker.TrackedReal, p::Real, x::Int) = Tracker.track(nbinomlogpdf, n, p, x)
Tracker.@grad function nbinomlogpdf(r::Tracker.TrackedReal, p::Tracker.TrackedReal, k::Int)
    return nbinomlogpdf(Tracker.data(r), Tracker.data(p), k),
        Δ->(Δ * _nbinomlogpdf_grad_1(r, p, k), Δ * _nbinomlogpdf_grad_2(r, p, k), nothing)
end
Tracker.@grad function nbinomlogpdf(r::Real, p::Tracker.TrackedReal, k::Int)
    return nbinomlogpdf(Tracker.data(r), Tracker.data(p), k),
        Δ->(Tracker._zero(r), Δ * _nbinomlogpdf_grad_2(r, p, k), nothing)
end
Tracker.@grad function nbinomlogpdf(r::Tracker.TrackedReal, p::Real, k::Int)
    return nbinomlogpdf(Tracker.data(r), Tracker.data(p), k),
        Δ->(Δ * _nbinomlogpdf_grad_1(r, p, k), Tracker._zero(p), nothing)
end

function nbinomlogpdf(r::ForwardDiff.Dual{T}, p::ForwardDiff.Dual{T}, k::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val_p = ForwardDiff.value(p)
    val_r = ForwardDiff.value(r)

    Δ_r = ForwardDiff.partials(r) * _nbinomlogpdf_grad_1(val_r, val_p, k)
    Δ_p = ForwardDiff.partials(p) * _nbinomlogpdf_grad_2(val_r, val_p, k)
    Δ = Δ_p + Δ_r
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

## Poisson ##

poislogpdf(v::Tracker.TrackedReal, x::Int) = Tracker.track(poislogpdf, v, x)
Tracker.@grad function poislogpdf(v::Tracker.TrackedReal, x::Int)
      return poislogpdf(Tracker.data(v), x),
          Δ->(Δ * (x/v - 1), nothing)
end

function poislogpdf(v::ForwardDiff.Dual{T}, x::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val = ForwardDiff.value(v)
    Δ = ForwardDiff.partials(v)
    return FD(poislogpdf(val, x), Δ * (x/val - 1))
end

## PoissonBinomial ##

struct TuringPoissonBinomial{T<:Real, TV<:AbstractVector{T}} <: DiscreteUnivariateDistribution
    p::TV
    pmf::TV
end
function TuringPoissonBinomial(p::AbstractArray{<:Real})
    pb = Distributions.poissonbinomial_pdf_fft(p)
    @assert Distributions.isprobvec(pb)
    TuringPoissonBinomial(p, pb)
end
function logpdf(d::TuringPoissonBinomial{T}, k::Int) where T<:Real
    insupport(d, k) ? log(d.pmf[k + 1]) : -T(Inf)
end
quantile(d::TuringPoissonBinomial, x::Float64) = quantile(Categorical(d.pmf), x) - 1
PoissonBinomial(p::Tracker.TrackedArray) = TuringPoissonBinomial(p)
Base.minimum(d::TuringPoissonBinomial) = 0
Base.maximum(d::TuringPoissonBinomial) = length(d.p)

poissonbinomial_pdf_fft(x::Tracker.TrackedArray) = Tracker.track(poissonbinomial_pdf_fft, x)
Tracker.@grad function poissonbinomial_pdf_fft(x::Tracker.TrackedArray)
    x_data = Tracker.data(x)
    T = eltype(x_data)
    fft = poissonbinomial_pdf_fft(x_data)
    return  fft, Δ -> begin
        ((ForwardDiff.jacobian(x -> poissonbinomial_pdf_fft(x), x_data)::Matrix{T})' * Δ,)
    end
end
