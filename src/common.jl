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
    make_closure(f, g)

Return a closure of the form `(x, args...) -> f(g(args...), x)`.

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
       g(x) = let g_inner = DistributionsAD.make_closure(logpdf, Normal)
           sum(g_inner.(data, x))
       end
g (generic function with 1 method)

julia> @btime ReverseDiff.gradient(\$g, \$x);
  17.460 μs (17 allocations: 71.52 KiB)
```

See https://github.com/TuringLang/Turing.jl/issues/1934 more further discussion.

# Notes
To really go "vrooom!\" one needs to specialize on the arguments, e.g. if one
has a function `myfunc` then we need to define

```julia
make_closure(::typeof(myfunc), ::Type{D}) where {D} = myfunc(D(args...), x)
```

This can also be done using `DistributionsAD.@specialize_make_closure`:

```julia
julia> mylogpdf(d, x) = logpdf(d, x)
mylogpdf (generic function with 1 method)

julia> h(x) = let inner = DistributionsAD.make_closure(mylogpdf, Normal)
           sum(inner.(data, x))
       end
h (generic function with 1 method)

julia> @btime ReverseDiff.gradient(\$h, \$x);
  1.220 ms (37011 allocations: 1.42 MiB)

julia> DistributionsAD.@specialize_make_closure mylogpdf

julia> @btime ReverseDiff.gradient(\$h, \$x);
  17.038 μs (17 allocations: 71.52 KiB)
```
"""
make_closure(f, g) = (x, args...) -> f(g(args...), x)
make_closure(f, ::Type{D}) where {D} = (x, args...) -> f(D(args...), x)


"""
    has_specialized_make_closure(f, g)

Return `true` if there exists a specialized `make_closure(f, g)` implementation.
"""
has_specialized_make_closure(f, g) = false

# To go vroooom we need to specialize on the first argument, thus ensuring that
# a different closure is constructed for each method.
"""
    @specialize_make_closure(f)

Define `make_closure` and `has_specialized_make_closure` for first first argument being `f` 
and second argument being a type.
"""
macro specialize_make_closure(f)
    return quote
        $(DistributionsAD).make_closure(::typeof($(esc(f))), ::Type{D}) where {D} = (x, args...) -> $(esc(f))(D(args...), x)
        $(DistributionsAD).has_specialized_make_closure(::typeof($(esc(f))), ::Type{D}) where {D} = true
    end
end

"""
    @specialize_make_closure(f, g)

Define `make_closure` and `has_specialized_make_closure` for first first argument being `f` 
and second argument being `g`.
"""
macro specialize_make_closure(f, g)
    return quote
        $(DistributionsAD).make_closure(::typeof($(esc(f))), ::typeof($(esc(g)))) = (x, args...) -> $(esc(f))($(esc(g))(args...), x)
        $(DistributionsAD).has_specialized_make_closure(::typeof($(esc(f))), ::typeof{$(esc(g))}) = true
    end
end

@specialize_make_closure Distributions.pdf
@specialize_make_closure Distributions.logpdf
@specialize_make_closure Distributions.loglikelihood
@specialize_make_closure Distributions.cdf
@specialize_make_closure Distributions.logcdf
