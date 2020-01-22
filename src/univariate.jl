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

Distributions.Uniform(a::TrackedReal, b::Real) = TuringUniform{TrackedReal}(a, b)
Distributions.Uniform(a::Real, b::TrackedReal) = TuringUniform{TrackedReal}(a, b)
Distributions.Uniform(a::TrackedReal, b::TrackedReal) = TuringUniform{TrackedReal}(a, b)
Distributions.logpdf(d::Uniform, x::TrackedReal) = uniformlogpdf(d.a, d.b, x)

function uniformlogpdf(a, b, x)
    c = -log(b - a)
    if a <= x <= b
        return c
    else
        return oftype(c, -Inf)
    end
end
uniformlogpdf(a::Real, b::Real, x::TrackedReal) = track(uniformlogpdf, a, b, x)
uniformlogpdf(a::TrackedReal, b::TrackedReal, x::Real) = track(uniformlogpdf, a, b, x)
uniformlogpdf(a::TrackedReal, b::TrackedReal, x::TrackedReal) = track(uniformlogpdf, a, b, x)
Tracker.@grad function uniformlogpdf(a, b, x)
    diff = data(b) - data(a)
    T = typeof(diff)
    if a <= data(x) <= b && a < b
        l = -log(diff)
        da = 1/diff^2
        return l, Δ -> (da * Δ, -da * Δ, zero(T) * Δ)
    else
        n = T(NaN)
        return l, Δ -> (n, n, n)
    end
end
ZygoteRules.@adjoint function uniformlogpdf(a, b, x)
    diff = b - a
    T = typeof(diff)
    if a <= x <= b && a < b
        l = -log(diff)
        da = 1/diff^2
        return l, Δ -> (da * Δ, -da * Δ, zero(T) * Δ)
    else
        n = T(NaN)
        return l, Δ -> (n, n, n)
    end
end
ZygoteRules.@adjoint function Distributions.Uniform(args...)
    return pullback(TuringUniform, args...)
end

## Beta ##

function _betalogpdfgrad(α, β, x)
    di = digamma(α + β)
    dα = log(x) - digamma(α) + di
    dβ = log(1 - x) - digamma(β) + di
    dx = (α - 1)/x + (1 - β)/(1 - x)
    return (dα, dβ, dx)
end
ZygoteRules.@adjoint function betalogpdf(α::Real, β::Real, x::Number)
    return betalogpdf(α, β, x), Δ -> (Δ .* _betalogpdfgrad(α, β, x))
end    

## Gamma ##

function _gammalogpdfgrad(k, θ, x)
    dk = -digamma(k) - log(θ) + log(x)
    dθ = -k/θ + x/θ^2
    dx = (k - 1)/x - 1/θ
    return (dk, dθ, dx)
end
ZygoteRules.@adjoint function gammalogpdf(k::Real, θ::Real, x::Number)
    return gammalogpdf(k, θ, x), Δ -> (Δ .* _gammalogpdfgrad(k, θ, x))
end    

## Chisq ##

function _chisqlogpdfgrad(k, x)
    hk = k/2
    d = digamma(hk)
    dk = (-log(oftype(hk, 2)) - d + log(x))/2
    dx = (hk - 1)/x - one(hk)/2
    return (dk, dx)
end
ZygoteRules.@adjoint function chisqlogpdf(k::Real, x::Number)
    return chisqlogpdf(k, x), Δ -> (Δ .* _chisqlogpdfgrad(k, x))
end    

## FDist ##

function _fdistlogpdfgrad(v1, v2, x)
    temp1 = v1 * x + v2
    temp2 = log(temp1)
    vsum = v1 + v2
    temp3 = vsum / temp1
    temp4 = digamma(vsum / 2)
    dv1 = (log(v1 * x) + 1 - temp2 - x * temp3 - digamma(v1 / 2) + temp4) / 2
    dv2 = (log(v2) + 1 - temp2 - temp3 - digamma(v2 / 2) + temp4) / 2
    dx = v1 / 2 * (1 / x - temp3) - 1 / x
    return (dv1, dv2, dx)
end
ZygoteRules.@adjoint function fdistlogpdf(v1::Real, v2::Real, x::Number)
    return fdistlogpdf(v1, v2, x), Δ -> (Δ .* _fdistlogpdfgrad(v1, v2, x))
end

## TDist ##

function _tdistlogpdfgrad(v, x)
    dv = (digamma((v + 1) / 2) - 1 / v - digamma(v / 2) - log(1 + x^2 / v) + x^2 * (v + 1) / v^2 / (1 + x^2 / v)) / 2
    dx = -x * (v + 1) / (v + x^2)
    return (dv, dx)
end
ZygoteRules.@adjoint function tdistlogpdf(v::Real, x::Number)
    return tdistlogpdf(v, x), Δ -> (Δ .* _tdistlogpdfgrad(v, x))
end

## Semicircle ##

function semicircle_dldr(r, x)
    diffsq = r^2 - x^2
    return -2 / r + r / diffsq
end
function semicircle_dldx(r, x)
    diffsq = r^2 - x^2
    return -x / diffsq
end

logpdf(d::Semicircle{<:Real}, x::TrackedReal) = semicirclelogpdf(d.r, x)
logpdf(d::Semicircle{<:TrackedReal}, x::Real) = semicirclelogpdf(d.r, x)
logpdf(d::Semicircle{<:TrackedReal}, x::TrackedReal) = semicirclelogpdf(d.r, x)

semicirclelogpdf(r, x) = logpdf(Semicircle(r), x)
M, f, arity = DiffRules.@define_diffrule DistributionsAD.semicirclelogpdf(r, x) =
    :(semicircle_dldr($r, $x)), :(semicircle_dldx($r, $x))
da, db = DiffRules.diffrule(M, f, :a, :b)
f = :($M.$f)
@eval begin
    Tracker.@grad $f(a::TrackedReal, b::TrackedReal) = $f(data(a), data(b)), Δ -> (Δ * $da, Δ * $db)
    Tracker.@grad $f(a::TrackedReal, b::Real) = $f(data(a), b), Δ -> (Δ * $da, Tracker._zero(b))
    Tracker.@grad $f(a::Real, b::TrackedReal) = $f(a, data(b)), Δ -> (Tracker._zero(a), Δ * $db)
    $f(a::TrackedReal, b::TrackedReal)  = track($f, a, b)
    $f(a::TrackedReal, b::Real) = track($f, a, b)
    $f(a::Real, b::TrackedReal) = track($f, a, b)
end
if VERSION < v"1.2"
    Base.inv(::Irrational{:π}) = 1/π
end

## Binomial ##

binomlogpdf(n::Int, p::TrackedReal, x::Int) = track(binomlogpdf, n, p, x)
Tracker.@grad function binomlogpdf(n::Int, p::TrackedReal, x::Int)
    return binomlogpdf(n, data(p), x),
        Δ->(nothing, Δ * (x / p - (n - x) / (1 - p)), nothing)
end
ZygoteRules.@adjoint function binomlogpdf(n::Int, p::Real, x::Int)
    return binomlogpdf(n, p, x),
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

nbinomlogpdf(n::TrackedReal, p::TrackedReal, x::Int) = track(nbinomlogpdf, n, p, x)
nbinomlogpdf(n::Real, p::TrackedReal, x::Int) = track(nbinomlogpdf, n, p, x)
nbinomlogpdf(n::TrackedReal, p::Real, x::Int) = track(nbinomlogpdf, n, p, x)
Tracker.@grad function nbinomlogpdf(r::TrackedReal, p::TrackedReal, k::Int)
    return nbinomlogpdf(data(r), data(p), k),
        Δ->(Δ * _nbinomlogpdf_grad_1(r, p, k), Δ * _nbinomlogpdf_grad_2(r, p, k), nothing)
end
Tracker.@grad function nbinomlogpdf(r::Real, p::TrackedReal, k::Int)
    return nbinomlogpdf(data(r), data(p), k),
        Δ->(Tracker._zero(r), Δ * _nbinomlogpdf_grad_2(r, p, k), nothing)
end
Tracker.@grad function nbinomlogpdf(r::TrackedReal, p::Real, k::Int)
    return nbinomlogpdf(data(r), data(p), k),
        Δ->(Δ * _nbinomlogpdf_grad_1(r, p, k), Tracker._zero(p), nothing)
end

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

## Poisson ##

poislogpdf(v::TrackedReal, x::Int) = track(poislogpdf, v, x)
Tracker.@grad function poislogpdf(v::TrackedReal, x::Int)
      return poislogpdf(data(v), x),
          Δ->(Δ * (x/v - 1), nothing)
end
ZygoteRules.@adjoint function poislogpdf(v::Real, x::Int)
    return poislogpdf(v, x),
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
PoissonBinomial(p::TrackedArray) = TuringPoissonBinomial(p)
Base.minimum(d::TuringPoissonBinomial) = 0
Base.maximum(d::TuringPoissonBinomial) = length(d.p)

poissonbinomial_pdf_fft(x::TrackedArray) = track(poissonbinomial_pdf_fft, x)
Tracker.@grad function poissonbinomial_pdf_fft(x::TrackedArray)
    x_data = data(x)
    T = eltype(x_data)
    fft = poissonbinomial_pdf_fft(x_data)
    return  fft, Δ -> begin
        ((ForwardDiff.jacobian(x -> poissonbinomial_pdf_fft(x), x_data)::Matrix{T})' * Δ,)
    end
end
# FIXME: This is inefficient, replace with the commented code below once Zygote supports it.
ZygoteRules.@adjoint function poissonbinomial_pdf_fft(x::AbstractArray)
    T = eltype(x)
    fft = poissonbinomial_pdf_fft(x)
    return  fft, Δ -> begin
        ((ForwardDiff.jacobian(x -> poissonbinomial_pdf_fft(x), x)::Matrix{T})' * Δ,)
    end
end

# The code below doesn't work because of bugs in Zygote. The above is inefficient.
#=
ZygoteRules.@adjoint function poissonbinomial_pdf_fft(x::AbstractArray{<:Real})
    return pullback(poissonbinomial_pdf_fft_zygote, x)
end
function poissonbinomial_pdf_fft_zygote(p::AbstractArray{T}) where {T <: Real}
    n = length(p)
    ω = 2 * one(T) / (n + 1)

    lmax = ceil(Int, n/2)
    x1 = [one(T)/(n + 1)]
    x_lmaxp1 = map(1:lmax) do l
        logz = zero(T)
        argz = zero(T)
        for j=1:n
            zjl = 1 - p[j] + p[j] * cospi(ω*l) + im * p[j] * sinpi(ω * l)
            logz += log(abs(zjl))
            argz += atan(imag(zjl), real(zjl))
        end
        dl = exp(logz)
        return dl * cos(argz) / (n + 1) + dl * sin(argz) * im / (n + 1)
    end
    x_lmaxp2_end = [conj(x[l + 1]) for l in lmax:-1:1 if n + 1 - l > l]
    x = vcat(x1; x_lmaxp1, x_lmaxp2_end)
    y = [sum(x[j] * cis(-π * float(T)(2 * mod(j * k, n)) / n) for j in 1:n) for k in 1:n]
    return max.(0, real.(y))
end
function poissonbinomial_pdf_fft_zygote2(p::AbstractArray{T}) where {T <: Real}
    n = length(p)
    ω = 2 * one(T) / (n + 1)

    x = Vector{Complex{T}}(undef, n+1)
    lmax = ceil(Int, n/2)
    x[1] = one(T)/(n + 1)
    for l=1:lmax
        logz = zero(T)
        argz = zero(T)
        for j=1:n
            zjl = 1 - p[j] + p[j] * cospi(ω*l) + im * p[j] * sinpi(ω * l)
            logz += log(abs(zjl))
            argz += atan(imag(zjl), real(zjl))
        end
        dl = exp(logz)
        x[l + 1] = dl * cos(argz) / (n + 1) + dl * sin(argz) * im / (n + 1)
        if n + 1 - l > l
            x[n + 1 - l + 1] = conj(x[l + 1])
        end
    end
    max.(0, real.(_dft_zygote(copy(x))))
end
function _dft_zygote(x::Vector{T}) where T
    n = length(x)
    y = Zygote.Buffer(zeros(complex(float(T)), n))
    @inbounds for j = 0:n-1, k = 0:n-1
        y[k+1] += x[j+1] * cis(-π * float(T)(2 * mod(j * k, n)) / n)
    end
    return copy(y)
end
=#