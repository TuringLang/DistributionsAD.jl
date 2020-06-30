# Zygote fill has issues with non-numbers
@adjoint function fill(x::T, dims...) where {T}
    return pullback(x, dims...) do x, dims...
        return reshape([x for i in 1:prod(dims)], dims)
    end
end


## Uniform ##

@adjoint function uniformlogpdf(a, b, x)
    diff = b - a
    T = typeof(diff)
    if a <= x <= b && a < b
        l = -log(diff)
        da = 1/diff^2
        return l, Δ -> (da * Δ, -da * Δ, zero(T) * Δ)
    else
        n = T(NaN)
        return n, Δ -> (n, n, n)
    end
end

@adjoint function Distributions.Uniform(args...)
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
@adjoint function betalogpdf(α::Real, β::Real, x::Number)
    return betalogpdf(α, β, x), Δ -> (Δ .* _betalogpdfgrad(α, β, x))
end


## Gamma ##

function _gammalogpdfgrad(k, θ, x)
    dk = -digamma(k) - log(θ) + log(x)
    dθ = -k/θ + x/θ^2
    dx = (k - 1)/x - 1/θ
    return (dk, dθ, dx)
end
@adjoint function gammalogpdf(k::Real, θ::Real, x::Number)
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
@adjoint function chisqlogpdf(k::Real, x::Number)
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
@adjoint function fdistlogpdf(v1::Real, v2::Real, x::Number)
    return fdistlogpdf(v1, v2, x), Δ -> (Δ .* _fdistlogpdfgrad(v1, v2, x))
end

## TDist ##

function _tdistlogpdfgrad(v, x)
    dv = (digamma((v + 1) / 2) - 1 / v - digamma(v / 2) - log(1 + x^2 / v) + x^2 * (v + 1) / v^2 / (1 + x^2 / v)) / 2
    dx = -x * (v + 1) / (v + x^2)
    return (dv, dx)
end
@adjoint function tdistlogpdf(v::Real, x::Number)
    return tdistlogpdf(v, x), Δ -> (Δ .* _tdistlogpdfgrad(v, x))
end


## Binomial ##

@adjoint function binomlogpdf(n::Int, p::Real, x::Int)
    return binomlogpdf(n, p, x),
        Δ->(nothing, Δ * (x / p - (n - x) / (1 - p)), nothing)
end

## Poisson ##

@adjoint function poislogpdf(v::Real, x::Int)
    return poislogpdf(v, x),
        Δ->(Δ * (x/v - 1), nothing)
end


## PoissonBinomial ##

# FIXME: This is inefficient, replace with the commented code below once Zygote supports it.
@adjoint function poissonbinomial_pdf_fft(x::AbstractArray)
    T = eltype(x)
    fft = poissonbinomial_pdf_fft(x)
    return  fft, Δ -> begin
        ((ForwardDiff.jacobian(x -> poissonbinomial_pdf_fft(x), x)::Matrix{T})' * Δ,)
    end
end

# The code below doesn't work because of bugs in Zygote. The above is inefficient.
#=
@adjoint function poissonbinomial_pdf_fft(x::AbstractArray{<:Real})
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
    x = vcat(x1, x_lmaxp1, x_lmaxp2_end)
    #y = [sum(x[j] * cis(-π * float(T)(2 * mod(j * k, n)) / n) for j in 1:n) for k in 1:n]
    return max.(0, real.(x))
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
