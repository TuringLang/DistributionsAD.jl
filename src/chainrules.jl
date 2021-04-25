## Uniform ##

@scalar_rule(
    uniformlogpdf(a, b, x),
    @setup(
        insupport = a <= x <= b,
        diff = b - a,
        c = insupport ? inv(diff) : inv(one(diff)),
        z = insupport ? zero(x) : oftype(x, NaN),
    ),
    (c, -c, z),
)

# StatsFuns: https://github.com/JuliaStats/StatsFuns.jl/pull/106

## Beta ##

@scalar_rule(
    betalogpdf(α::Real, β::Real, x::Number),
    @setup(z = digamma(α + β)),
    (
        log(x) + z - digamma(α),
        log1p(-x) + z - digamma(β),
        (α - 1) / x + (1 - β) / (1 - x),
    ),
)

## Gamma ##

@scalar_rule(
    gammalogpdf(k::Real, θ::Real, x::Number),
    @setup(
        invθ = inv(θ),
        xoθ = invθ * x,
        z = xoθ - k,
    ),
    (
        log(xoθ) - digamma(k),
        invθ * z,
        - (1 + z) / x,
    ),
)

## Chisq ##

@scalar_rule(
    chisqlogpdf(k::Real, x::Number),
    @setup(hk = k / 2),
    (
        (log(x) - logtwo - digamma(hk)) / 2,
        (hk - 1) / x - one(hk) / 2,
    ),
)

## FDist ##

@scalar_rule(
    fdistlogpdf(ν1::Real, ν2::Real, x::Number),
    @setup(
        xν1 = x * ν1,
        temp1 = xν1 + ν2,
        a = (x - 1) / temp1,
        νsum = ν1 + ν2,
        di = digamma(νsum / 2),
    ),
    (
        (-log1p(ν2 / xν1) - ν2 * a + di - digamma(ν1 / 2)) / 2,
        (-log1p(xν1 / ν2) + ν1 * a + di - digamma(ν2 / 2)) / 2,
        ((ν1 - 2) / x - ν1 * νsum / temp1) / 2,
    ),
)

## TDist ##

@scalar_rule(
    tdistlogpdf(ν::Real, x::Number),
    @setup(
        νp1 = ν + 1,
        xsq = x^2,
        invν = inv(ν),
        a = xsq * invν,
        b = νp1 / (ν + xsq),
    ),
    (
        (digamma(νp1 / 2) - digamma(ν / 2) + a * b - log1p(a) - invν) / 2,
        - x * b,
    ),
)

## Binomial ##

@scalar_rule(
    binomlogpdf(n::Real, p::Real, k::Real),
    @setup(z = digamma(n - k + 1)),
    (
        digamma(n + 2) - z + log1p(-p) - 1 / (1 + n),
        (k / p - n) / (1 - p),
        z - digamma(k + 1) + logit(p),
    ),
)

## Poisson ##

@scalar_rule(
    poislogpdf(λ::Number, x::Number),
    ((iszero(x) && iszero(λ) ? zero(x / λ) : x / λ) - 1, log(λ) - digamma(x + 1)),
)

## PoissonBinomial

function ChainRulesCore.rrule(
    ::typeof(Distributions.poissonbinomial_pdf_fft), p::AbstractVector{<:Real}
)
    y = Distributions.poissonbinomial_pdf_fft(p)
    A = poissonbinomial_partialderivatives(p)
    function poissonbinomial_pdf_fft_pullback(Δy)
        p̄ = InplaceableThunk(
            @thunk(A * Δy),
            Δ -> LinearAlgebra.mul!(Δ, A, Δy, true, true),
        )
        return (NO_FIELDS, p̄)
    end
    return y, poissonbinomial_pdf_fft_pullback
end

if isdefined(Distributions, :poissonbinomial_pdf)
    function ChainRulesCore.rrule(
        ::typeof(Distributions.poissonbinomial_pdf), p::AbstractVector{<:Real}
    )
        y = Distributions.poissonbinomial_pdf(p)
        A = poissonbinomial_partialderivatives(p)
        function poissonbinomial_pdf_pullback(Δy)
            p̄ = InplaceableThunk(
                @thunk(A * Δy),
                Δ -> LinearAlgebra.mul!(Δ, A, Δy, true, true),
            )
            return (NO_FIELDS, p̄)
        end
        return y, poissonbinomial_pdf_pullback
    end
end
