function _mixlogpdf1(d::AbstractMixtureModel, x)
    # using the formula below for numerical stability
    #
    # logpdf(d, x) = log(sum_i pri[i] * pdf(cs[i], x))
    #              = log(sum_i pri[i] * exp(logpdf(cs[i], x)))
    #              = log(sum_i exp(logpri[i] + logpdf(cs[i], x)))
    #              = m + log(sum_i exp(logpri[i] + logpdf(cs[i], x) - m))
    #
    #  m is chosen to be the maximum of logpri[i] + logpdf(cs[i], x)
    #  such that the argument of exp is in a reasonable range
    #

    K = ncomponents(d)
    p = probs(d)
    # use Buffer to avoid mutating arrays.
    # lp = Vector{eltype(p)}(undef, K)
    lp = Zygote.Buffer(p, K)
    m = -Inf   # m <- the maximum of log(p(cs[i], x)) + log(pri[i])
    @inbounds for i in eachindex(p)
        pi = p[i]
        if pi > 0.0
            # lp[i] <- log(p(cs[i], x)) + log(pri[i])
            lp_i = logpdf(component(d, i), x) + log(pi)
            # zygote seems to have trouble here.
            # Mutating arrays is not supported
            lp[i] = lp_i
            if lp_i > m
                m = lp_i
            end
        end
    end
    v = 0.0
    @inbounds for i = 1:K
        if p[i] > 0.0
            v += exp(lp[i] - m)
        end
    end
    return m + log(v)
end



Distributions.logpdf(d::UnivariateMixture{Continuous}, x::Real) = _mixlogpdf1(d, x)
Distributions.logpdf(d::UnivariateMixture{Discrete}, x::Int) = _mixlogpdf1(d, x)

Distributions._logpdf(d::MultivariateMixture, x::AbstractVector) = _mixlogpdf1(d, x)