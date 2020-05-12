function _mixlogpdf1(d::AbstractMixtureModel, x)
    # using the formula below for numerical stability
    #
    # logpdf(d, x) = log(sum_i pri[i] * pdf(cs[i], x))
    #              = log(sum_i pri[i] * exp(logpdf(cs[i], x)))
    #              = log(sum_i exp(logpri[i] + logpdf(cs[i], x)))

    pri = probs(d)
    indices = findall(!iszero, pri)
    lp = map(indices) do i
            return logpdf(component(d, i), x) + log(pri[i])
        end

    return logsumexp(lp)
end



Distributions.logpdf(d::UnivariateMixture{Continuous}, x::Real) = _mixlogpdf1(d, x)
Distributions.logpdf(d::UnivariateMixture{Discrete}, x::Int) = _mixlogpdf1(d, x)
Distributions._logpdf(d::MultivariateMixture, x::AbstractVector) = _mixlogpdf1(d, x)