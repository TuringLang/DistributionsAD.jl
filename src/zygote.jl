# Zygote fill has issues with non-numbers
ZygoteRules.@adjoint function fill(x::T, dims...) where {T}
    return ZygoteRules.pullback(x, dims...) do x, dims...
        return reshape([x for i in 1:prod(dims)], dims)
    end
end


## Uniform ##

ZygoteRules.@adjoint function uniformlogpdf(a, b, x)
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

ZygoteRules.@adjoint function Distributions.Uniform(args...)
    return ZygoteRules.pullback(TuringUniform, args...)
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


## Binomial ##

ZygoteRules.@adjoint function binomlogpdf(n::Int, p::Real, x::Int)
    return binomlogpdf(n, p, x),
        Δ->(nothing, Δ * (x / p - (n - x) / (1 - p)), nothing)
end

## Poisson ##

ZygoteRules.@adjoint function poislogpdf(v::Real, x::Int)
    return poislogpdf(v, x),
        Δ->(Δ * (x/v - 1), nothing)
end


## PoissonBinomial ##

# FIXME: This is inefficient, replace with the commented code below once Zygote supports it.
ZygoteRules.@adjoint function poissonbinomial_pdf_fft(x::AbstractArray{T}) where T<:Real
    error("The adjoint of poissonbinomial_pdf_fft needs ForwardDiff. `using ForwardDiff` should fix this error.")
end
