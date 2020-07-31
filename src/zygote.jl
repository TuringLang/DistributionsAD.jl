# Zygote fill has issues with non-numbers
ZygoteRules.@adjoint function fill(x::T, dims...) where {T}
    return ZygoteRules.pullback(x, dims...) do x, dims...
        return reshape([x for i in 1:prod(dims)], dims)
    end
end


## Uniform ##

ZygoteRules.@adjoint function Distributions.Uniform(args...)
    return ZygoteRules.pullback(TuringUniform, args...)
end

## PoissonBinomial ##

# Zygote loads ForwardDiff, so this dummy adjoint should never be needed.
# The adjoint that is used for `poissonbinomial_pdf_fft` is defined in `src/zygote_forwarddiff.jl`
# ZygoteRules.@adjoint function poissonbinomial_pdf_fft(x::AbstractArray{T}) where T<:Real
#     error("This needs ForwardDiff. `using ForwardDiff` should fix this error.")
# end


## MatrixBeta ##

ZygoteRules.@adjoint function Distributions.logpdf(
    d::MatrixBeta,
    X::AbstractArray{<:Matrix{<:Real}}
)
    return ZygoteRules.pullback(d, X) do d, X
        map(x -> logpdf(d, x), X)
    end
end

## Wishart ##

# Custom adjoint since Zygote can't differentiate through `@warn`
# TODO: Remove when fixed upstream in Distributions
ZygoteRules.@adjoint function Wishart(df::T, S::AbstractPDMat{T}, warn::Bool = true) where T<:Real
    function _Wishart(df::T, S::AbstractPDMat{T}, warn::Bool = true) where T
        df > 0 || throw(ArgumentError("df must be positive. got $(df)."))
        p = dim(S)
        rnk = p
        singular = df <= p - 1
        if singular
            isinteger(df) || throw(ArgumentError("singular df must be an integer. got $(df)."))
            rnk = convert(Integer, df)
            warn && _warn("got df <= dim - 1; returning a singular Wishart")
        end
        logc0 = Distributions.wishart_logc0(df, S, rnk)
        R = Base.promote_eltype(T, logc0)
        prom_S = convert(AbstractArray{T}, S)
        Wishart{R, typeof(prom_S), typeof(rnk)}(R(df), prom_S, R(logc0), rnk, singular)
    end
    return ZygoteRules.pullback(_Wishart, df, S, warn)
end

_warn(msg) = @warn(msg)
ZygoteRules.@adjoint _warn(msg) = _warn(msg), _ -> nothing

ZygoteRules.@adjoint function Distributions.Wishart(df::Real, S::AbstractMatrix{<:Real})
    return ZygoteRules.pullback(TuringWishart, df, S)
end
ZygoteRules.@adjoint function Distributions.InverseWishart(
    df::Real,
    S::AbstractMatrix{<:Real}
)
    return ZygoteRules.pullback(TuringInverseWishart, df, S)
end
