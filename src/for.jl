using Distributions
import Distributions.logpdf
using Base.Cartesian


export logpdf
export rand

export For
struct For{F,T,D,X} 
    f :: F  
    θ :: T
end

#########################################################
# T <: NTuple{N,J} where {J <: Integer}
#########################################################

For(f, θ::J...) where {J <: Integer} = For(f,θ)

function For(f::F, θ::T) where {F, N, J <: Integer, T <: NTuple{N,J}}
    d = f.(ones(Int, N)...)
    D = typeof(d)
    X = eltype(d)
    For{F, NTuple{N,J}, D, X}(f,θ)
end

@inline function logpdf(d::For{F,T,D,X1},xs::AbstractArray{X2,N}) where {F, N, J <: Integer, T <: NTuple{N,J}, D,  X1,  X2 <: X1}
    s = 0.0
    @inbounds @simd for θ in CartesianIndices(d.θ)
        s += logpdf(d.f(Tuple(θ)...), xs[θ])
    end
    s
end

function Base.rand(dist::For) 
    map(CartesianIndices(dist.θ)) do I
        (rand ∘ dist.f)(Tuple(I)...)
    end
end

#########################################################
# T <: NTuple{N,J} where {J <: AbstractUnitRange}
#########################################################

For(f, θ::J...) where {J <: AbstractUnitRange} = For(f,θ)

function For(f::F, θ::T) where {F, N, J <: AbstractRange, T <: NTuple{N,J}}
    d = f.(ones(Int, N)...)
    D = typeof(d)
    X = eltype(d)
    For{F, NTuple{N,J}, D, X}(f,θ)
end


@inline function logpdf(d::For{F,T,D,X1},xs::AbstractArray{X2,N}) where {F, N, J <: AbstractRange,  T <: NTuple{N,J}, D, X1, X2 <: X1}
    s = 0.0
    @inbounds @simd for θ in CartesianIndices(d.θ)
        s += logpdf(d.f(Tuple(θ)...), xs[θ])
    end
    s
end


function Base.rand(dist::For{F,T}) where {F,  N, J <: AbstractRange, T <: NTuple{N,J}}
    map(CartesianIndices(dist.θ)) do I
        (rand ∘ dist.f)(Tuple(I)...)
    end
end

#########################################################
# T <: Base.Generator
#########################################################

function For(f::F, θ::T) where {F, T <: Base.Generator}
    d = f(θ.f(θ.iter[1]))
    D = typeof(d)
    X = eltype(d)
    For{F, T, D, X}(f,θ)
end


@inline function logpdf(d :: For{F,T}, x) where {F,T <: Base.Generator}
    s = 0.0
    for (θj, xj) in zip(d.θ, x)
        s += logpdf(d.f(θj), xj)
    end
    s
end

@inline function rand(d :: For{F,T,D,X}) where {F,T <: Base.Generator, D, X}
    rand.(Base.Generator(d.θ.f, d.θ.iter))
end

#########################################################


