# Univariate

const VectorOfUnivariate = Distributions.Product

function arraydist(dists::AbstractVector{<:UnivariateDistribution})
    V = typeof(dists)
    T = eltype(dists)
    S = Distributions.value_support(T)
    return Product{S,T,V}(dists)
end

struct MatrixOfUnivariate{
    S <: ValueSupport,
    Tdist <: UnivariateDistribution{S},
    Tdists <: AbstractMatrix{Tdist},
} <: MatrixDistribution{S}
    dists::Tdists
end
Base.size(dist::MatrixOfUnivariate) = size(dist.dists)
function arraydist(dists::AbstractMatrix{<:UnivariateDistribution})
    return MatrixOfUnivariate(dists)
end
function Distributions._logpdf(dist::MatrixOfUnivariate, x::AbstractMatrix{<:Real})
    # Lazy broadcast to avoid allocations and use pairwise summation
    return sum(Broadcast.instantiate(Broadcast.broadcasted(logpdf, dist.dists, x)))
end
function Distributions.logpdf(dist::MatrixOfUnivariate, x::AbstractArray{<:AbstractMatrix{<:Real}})
    return map(Base.Fix1(logpdf, dist), x)
end
function Distributions.logpdf(dist::MatrixOfUnivariate, x::AbstractArray{<:Matrix{<:Real}})
    return map(Base.Fix1(logpdf, dist), x)
end

function Distributions.rand(rng::Random.AbstractRNG, dist::MatrixOfUnivariate)
    return rand.(Ref(rng), dist.dists)
end

# Multivariate

struct VectorOfMultivariate{
    S <: ValueSupport,
    Tdist <: MultivariateDistribution{S},
    Tdists <: AbstractVector{Tdist},
} <: MatrixDistribution{S}
    dists::Tdists
end
Base.size(dist::VectorOfMultivariate) = (length(dist.dists[1]), length(dist))
Base.length(dist::VectorOfMultivariate) = length(dist.dists)
function arraydist(dists::AbstractVector{<:MultivariateDistribution})
    return VectorOfMultivariate(dists)
end

function Distributions._logpdf(dist::VectorOfMultivariate, x::AbstractMatrix{<:Real})
    return sum(Broadcast.instantiate(Broadcast.broadcasted(logpdf, dist.dists, eachcol(x))))
end
function Distributions.logpdf(dist::VectorOfMultivariate, x::AbstractArray{<:AbstractMatrix{<:Real}})
    return map(Base.Fix1(logpdf, dist), x)
end
function Distributions.logpdf(dist::VectorOfMultivariate, x::AbstractArray{<:Matrix{<:Real}})
    return map(Base.Fix1(logpdf, dist), x)
end

function Distributions.rand(rng::Random.AbstractRNG, dist::VectorOfMultivariate)
    init = reshape(rand(rng, dist.dists[1]), :, 1)
    return mapreduce(Base.Fix1(rand, rng), hcat, view(dist.dists, 2:length(dist)); init = init)
end

# Lazy array dist
# HACK: Constructor which doesn't enforce the schema.
"""
    StructArrayNoSchema(::Type{T}, cols::C) where {T, C<:StructArrays.Tup}

Construct a `StructArray` without enforcing the schema of `T`.

This is useful in scenarios where there's a mismatch between the constructor of `T`
and the `fieldnames(T)`.

# Examples
```jldoctest
julia> using StructArrays, Distributions

julia> # `Normal` has two fields `μ` and `σ`, but here we only provide `μ`.
       StructArrayNoSchema(Normal, (zeros(2),))
2-element StructArray(::Vector{Float64}) with eltype Normal:
 Normal{Float64}(μ=0.0, σ=1.0)
 Normal{Float64}(μ=0.0, σ=1.0)

julia> # This is not allowed by `StructArray`:
       StructArray{Normal}((zeros(2),))
ERROR: NamedTuple names and field types must have matching lengths
[...]
```
"""
function StructArrayNoSchema(::Type{T}, cols::C) where {T, C<:StructArrays.Tup}
    N = isempty(cols) ? 1 : ndims(cols[1])
    StructArrays.StructArray{T, N, typeof(cols)}(cols)
end


arraydist(D::Type, args...) = arraydist(D, args)
arraydist(D::Type, args::Tuple) = arraydist(StructArrayNoSchema(D, args))

make_logpdf_closure(::Type{D}) where {D} = (x, args...) -> logpdf(D(args...), x)

function Distributions.logpdf(dist::Product{<:Any,D,<:StructArrays.StructArray}, x::AbstractVector{<:Real}) where {D}
    f = make_logpdf_closure(D)
    return sum(f.(x, StructArrays.components(dist.v)...))
end
