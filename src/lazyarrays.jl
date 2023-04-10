using .LazyArrays: BroadcastArray, BroadcastVector, LazyArray

const LazyVectorOfUnivariate{
    S<:ValueSupport,
    T<:UnivariateDistribution{S},
    Tdists<:BroadcastVector{T},
} = VectorOfUnivariate{S,T,Tdists}

_inner_constructor(::Type{<:BroadcastVector{<:Any,Type{D}}}) where {D} = D

function Distributions._logpdf(
    dist::LazyVectorOfUnivariate,
    x::AbstractVector{<:Real},
)
    # TODO: Make use of `sum(Broadcast.instantiate(Broadcast.broadcasted(f, x, args...)))` once
    # we've addressed performance issues in ReverseDiff.jl.
    constructor = _inner_constructor(typeof(dist.v))
    return sum(Closure(logpdf, constructor).(x, dist.v.args...))
end

function Distributions.logpdf(
    dist::LazyVectorOfUnivariate,
    x::AbstractMatrix{<:Real},
)
    size(x, 1) == length(dist) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    constructor = _inner_constructor(typeof(dist.v))
    return vec(sum(Closure(logpdf, constructor).(x, dist.v.args...), dims = 1))
end

const LazyMatrixOfUnivariate{
    S<:ValueSupport,
    T<:UnivariateDistribution{S},
    Tdists<:BroadcastArray{T,2},
} = MatrixOfUnivariate{S,T,Tdists}

function Distributions._logpdf(
    dist::LazyMatrixOfUnivariate,
    x::AbstractMatrix{<:Real},
)

    constructor = _inner_constructor(typeof(dist.v))
    return sum(Closure(logpdf, constructor).(x, dist.v.args))
end

lazyarray(f, x...) = BroadcastArray(f, x...)
export lazyarray

# HACK: All of the below probably shouldn't be here.
function ChainRulesCore.rrule(::Type{BroadcastArray}, f, args...)
    function BroadcastArray_pullback(Δ::ChainRulesCore.Tangent)
        return (ChainRulesCore.NoTangent(), Δ.f, Δ.args...)
    end
    return BroadcastArray(f, args...), BroadcastArray_pullback
end

ChainRulesCore.ProjectTo(ba::BroadcastArray) = ProjectTo{typeof(ba)}((f=ba.f,))
function (p::ChainRulesCore.ProjectTo{BA})(args...) where {BA<:BroadcastArray}
    return ChainRulesCore.Tangent{BA}(f=p.f, args=args)
end

function ChainRulesCore.rrule(
    config::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode},
    ::typeof(logpdf),
    dist::LazyVectorOfUnivariate,
    x::AbstractVector{<:Real}
)
    # Extract the constructor used in the `BroadcastArray`.
    constructor = DistributionsAD._inner_constructor(typeof(dist.v))

    # If it's not safe to ignore the `constructor` in the pullback, then we fall back
    # to the default implementation.
    is_diff_safe(constructor) || return ChainRulesCore.rrule_via_ad(config, (d,x) -> sum(logpdf.(d.v, x)), dist, x)

    # Otherwise, we use `Closure`.
    cl = DistributionsAD.Closure(logpdf, constructor)

    # Construct pullbacks manually to avoid the constructor of `BroadcastArray`.
    y, dy = ChainRulesCore.rrule_via_ad(config, broadcast, cl, x, dist.v.args...)
    z, dz = ChainRulesCore.rrule_via_ad(config, sum, y)

    project_broadcastarray = ChainRulesCore.ProjectTo(dist.v)
    function logpdf_adjoint(Δ...)
        # 1st argument is `sum` -> nothing.
        (_, sum_Δ...) = dz(Δ...)
        # 1st argument is `broadcast` -> nothing.
        # 2nd argument is `cl` -> `nothing`.
        # 3rd argument is `x` -> something.
        # Rest is `dist` arguments -> something
        (_, _, x_Δ, args_Δ...) = dy(sum_Δ...)
        # Construct the structural tangents.
        ba_tangent = project_broadcastarray(args_Δ...)
        dist_tangent = ChainRulesCore.Tangent{typeof(dist)}(v=ba_tangent)

        return (ChainRulesCore.NoTangent(), dist_tangent, x_Δ)
    end

    return z, logpdf_adjoint
end
