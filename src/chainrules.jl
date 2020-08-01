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

## Beta ##

@scalar_rule(
    betalogpdf(α::Real, β::Real, x::Number),
    @setup(di = digamma(α + β)),
    (
        @thunk(log(x) - digamma(α) + di),
        @thunk(log(1 - x) - digamma(β) + di),
        @thunk((α - 1)/x + (1 - β)/(1 - x)),
    ),
)

## Gamma ##

@scalar_rule(
    gammalogpdf(k::Real, θ::Real, x::Number),
    (
        @thunk(-digamma(k) - log(θ) + log(x)),
        @thunk(-k/θ + x/θ^2),
        @thunk((k - 1)/x - 1/θ),
    ),
)

## Chisq ##

@scalar_rule(
    chisqlogpdf(k::Real, x::Number),
    @setup(ko2 = k / 2),
    (@thunk((-logtwo - digamma(ko2) + log(x)) / 2), @thunk((ko2 - 1)/x - one(ko2) / 2)),
)

## FDist ##

@scalar_rule(
    fdistlogpdf(v1::Real, v2::Real, x::Number),
    @setup(
        temp1 = v1 * x + v2,
        temp2 = log(temp1),
        vsum = v1 + v2,
        temp3 = vsum / temp1,
        temp4 = digamma(vsum / 2),
    ),
    (
        @thunk((log(v1 * x) + 1 - temp2 - x * temp3 - digamma(v1 / 2) + temp4) / 2),
        @thunk((log(v2) + 1 - temp2 - temp3 - digamma(v2 / 2) + temp4) / 2),
        @thunk(v1 / 2 * (1 / x - temp3) - 1 / x),
    ),
)

## TDist ##

@scalar_rule(
    tdistlogpdf(v::Real, x::Number),
    (
        @thunk((digamma((v + 1) / 2) - 1 / v - digamma(v / 2) - log(1 + x^2 / v) + x^2 * (v + 1) / v^2 / (1 + x^2 / v)) / 2),
        @thunk(-x * (v + 1) / (v + x^2)),
    )
)

## Binomial ##

@scalar_rule(
    binomlogpdf(n::Int, p::Real, x::Int),
    (DoesNotExist(), x / p - (n - x) / (1 - p), DoesNotExist()),
)

## Poisson ##

@scalar_rule(
    poislogpdf(v::Real, x::Int),
    (x / v - 1, DoesNotExist()),
)
