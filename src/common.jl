## Linear Algebra ##

const CHOLESKY_NoPivot = VERSION >= v"1.8.0-rc1" ? LinearAlgebra.NoPivot() : Val(false)

function turing_chol(A::AbstractMatrix, check)
    chol = cholesky(A, check=check)
    (chol.factors, chol.info)
end
function turing_chol_back(A::AbstractMatrix, check)
    C, chol_pullback = rrule(cholesky, A, CHOLESKY_NoPivot; check=check)
    function back(Δ)
        Ȳ = Tangent{typeof(C)}(; factors=Δ[1])
        ∂C = chol_pullback(Ȳ)[2]
        (∂C, nothing)
    end
    (C.factors,C.info), back
end

function symm_turing_chol(A::AbstractMatrix, check, uplo)
    chol = cholesky(Symmetric(A, uplo), check=check)
    (chol.factors, chol.info)
end
function symm_turing_chol_back(A::AbstractMatrix, check, uplo)
    C, chol_pullback = rrule(cholesky, Symmetric(A,uplo), CHOLESKY_NoPivot; check=check)
    function back(Δ)
        Ȳ = Tangent{typeof(C)}(; factors=Δ[1])
        ∂C = chol_pullback(Ȳ)[2]
        (∂C, nothing, nothing)
    end
    (C.factors, C.info), back
end


# Tracker's implementation of ldiv isn't good. We'll use Zygote's instead.
zygote_ldiv(A::AbstractMatrix, B::AbstractVecOrMat) = A \ B

# fixes `randn` on GPU (https://github.com/TuringLang/DistributionsAD.jl/pull/108)
function adapt_randn(rng::AbstractRNG, x::AbstractArray, dims...)
    return adapt_randn(rng, eltype(x), x, dims...)
end
function adapt_randn(rng::AbstractRNG, ::Type{T}, x::AbstractArray, dims...) where {T}
    return adapt(parameterless_type(x), randn(rng, T, dims...))
end

# required by Adapt >= 3.3.0: https://github.com/SciML/OrdinaryDiffEq.jl/issues/1369
Base.@pure __parameterless_type(T) = Base.typename(T).wrapper
parameterless_type(x) = parameterless_type(typeof(x))
parameterless_type(x::Type) = __parameterless_type(x)

@non_differentiable adapt_randn(::Any...)


"""
    Closure{F,G}

A callable of the form `(x, args...) -> F(G(args...), x)`.

# Examples

This is particularly useful when one wants to avoid broadcasting over constructors
which can sometimes cause issues with type-inference, in particular when combined
with reverse-mode AD frameworks.

```juliarepl
julia> using DistributionsAD, Distributions, ReverseDiff, BenchmarkTools

julia> const data = randn(1000);

julia> x = randn(length(data));

julia> f(x) = sum(logpdf.(Normal.(x), data))
f (generic function with 2 methods)

julia> @btime ReverseDiff.gradient(\$f, \$x);
  848.759 μs (14605 allocations: 521.84 KiB)

julia> # Much faster with ReverseDiff.jl.
       g(x) = sum(DistributionsAD.Closure(logpdf, Normal).(data, x))
g (generic function with 1 method)

julia> @btime ReverseDiff.gradient(\$g, \$x);
  17.460 μs (17 allocations: 71.52 KiB)
```

See https://github.com/TuringLang/Turing.jl/issues/1934 more further discussion.
"""
struct Closure{F,G} end

Closure(::F, ::G) where {F,G} = Closure{F,G}()
Closure(::F, ::Type{G}) where {F,G} = Closure{F,G}()
Closure(::Type{F}, ::G) where {F,G} = Closure{F,G}()
Closure(::Type{F}, ::Type{G}) where {F,G} = Closure{F,G}()

"""
    is_diff_safe(f)

Return `true` if it's safe to ignore gradients wrt. `f` when computing `f`.

Useful for checking it's okay to take faster paths in pullbacks for certain AD backends.

# Examples

```jldoctest
julia> using Distributions

julia> using DistributionsAD: is_diff_safe, Closure

julia> is_diff_safe(typeof(logpdf))
true

julia> is_diff_safe(typeof(x -> 2x))
true

julia> # But it fails if we make a closure over a variable, which we might want to compute
       # the gradient with respect to.
       makef(x) = y -> x + y
makef (generic function with 1 method)

julia> is_diff_safe(typeof(makef([1.0])))
false

julia> # Also works on `Closure`s from `DistributionsAD`.
       is_diff_safe(typeof(Closure(logpdf, Normal)))
true

julia> is_diff_safe(typeof(Closure(logpdf, makef([1.0]))))
false
"""
@inline is_diff_safe(_) = false
@inline is_diff_safe(::Type) = true
@inline is_diff_safe(::Type{F}) where {F<:Function} = Base.issingletontype(F)
@inline is_diff_safe(::Type{Closure{F,G}}) where {F,G} = is_diff_safe(F) && is_diff_safe(G)

@generated function (closure::Closure{F,G})(x, args...) where {F,G}
    f = Base.issingletontype(F) ? F.instance : F
    g = Base.issingletontype(G) ? G.instance : G
    return :($f($g(args...), x))
end


